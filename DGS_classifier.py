import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import RandomOverSampler
from torchvision import transforms
import random

from utils.opt import Options
from bvh import Bvh

from annotations_dat import annotation_dictionary as ad
# from annontations_jesse_leader import annotation_dictionary as ad

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class BvhDatasets(Dataset):
    def __init__(self, opt, transform=None):
        self.history_size = opt.history_size
        self.answers = []
        self.animation_data = {}
        self.transform = transform

        self.steps = ["SR", "SL", "BR", "BL", "CR", "CL", "FR", "FL"]
        for k in ad:
            with open(os.path.join('correct_annotation_jesse_leader', k)) as f:
                mocap = Bvh(f.read())
                self.animation_data[k] = torch.tensor([[[mocap.frame_joint_channel(frame, j, c) for c in
                                                         ['Zrotation', 'Xrotation', 'Yrotation']] for j in
                                                        mocap.get_joints_names()] for frame in range(mocap.nframes)])

            for e in ad[k]:
                # for start_f in range(e[1][0], e[1][1], self.history_size)[:-1]:
                for start_f in range(e[1][0], e[1][1] - self.history_size):
                    progression = (self.history_size + start_f - e[1][0]) / (e[1][1] - e[1][0])  ###

                    self.answers.append([k, start_f, e[0], progression])

    def __getitem__(self, item):
        filename, start_frame, anno, progression = self.answers[item]
        fs = range(start_frame, start_frame + self.history_size)
        # print(fs)
        # print(item)
        # print(filename)
        # print(self.animation_data[filename].shape)
        animation_dict = self.animation_data[filename][fs]

        image_data = animation_dict.view(animation_dict.size(0), -1).float()
        # print(image_data.shape)
        # exit(0)

        if self.transform:
            image_data = self.transform(image_data.unsqueeze(0)).squeeze(0)

        label = self.steps.index(anno)

        return image_data, torch.tensor(label, dtype=torch.long), torch.tensor(progression, dtype=torch.double)

    def __len__(self):
        return len(self.answers)


class EnhancedCRNN(nn.Module):
    def __init__(self, input_height, input_width, num_classes, model_prediction=0):
        super(EnhancedCRNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(0.5)

        self.gru = nn.GRU(input_size=128, hidden_size=256, num_layers=3, batch_first=True, dropout=0.5)
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, num_classes + model_prediction)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), x.size(1), -1)
        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def apply_random_oversampling(data, labels, progress):
    ros = RandomOverSampler(random_state=42)
    data_res, labels_res = ros.fit_resample(data, labels)

    # Get the indices of the resampled data
    indices = ros.sample_indices_

    # Use the indices to resample the progress values
    progress_res = progress[indices]

    return data_res, labels_res, progress_res


def main():
    set_seed(42)  # Set seed for reproducibility
    option = Options().parse()
    transform = transforms.Compose([
        transforms.RandomApply([transforms.Lambda(lambda x: x + 5 * torch.randn_like(x))], p=5),
    ])
    dataset = BvhDatasets(option, transform=transform)

    num_classes = len(dataset.steps)
    class_names = dataset.steps

    # Count samples before oversampling
    class_counts_before = {step: 0 for step in dataset.steps}
    for _, label, _ in dataset:
        class_counts_before[dataset.steps[label.item()]] += 1

    print("Class distribution before oversampling:")
    for step, count in class_counts_before.items():
        print(f"{step}: {count}")

    data = []
    progress = []
    labels = []
    for i in range(len(dataset)):
        img, label, progression = dataset[i]
        data.append(img.numpy().flatten())
        labels.append(label.numpy())
        progress.append(progression.numpy())
    data = np.array(data)
    labels = np.array(labels)
    progress = np.array(progress)

    data_res, labels_res, progress_res = apply_random_oversampling(data, labels, progress)
    data_res = torch.tensor(data_res).view(-1, option.history_size, 54).float()
    labels_res = torch.tensor(labels_res).long()
    progress_res = torch.tensor(progress_res)

    # Count samples after oversampling
    class_counts_after = {step: 0 for step in dataset.steps}
    for label in labels_res:
        class_counts_after[dataset.steps[label.item()]] += 1

    print("Class distribution after oversampling:")
    for step, count in class_counts_after.items():
        print(f"{step}: {count}")

    balanced_dataset = TensorDataset(data_res, labels_res, progress_res)

    train_size = int(0.7 * len(balanced_dataset))
    val_size = int(0.15 * len(balanced_dataset))
    test_size = len(balanced_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(balanced_dataset, [train_size, val_size, test_size])

    print(f"Training dataset length: {len(train_dataset)}")
    print(f"Validation dataset length: {len(val_dataset)}")
    print(f"Test dataset length: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    input_height = option.history_size
    input_width = 54

    ## take a single animation chuck from test set , feed in history size from frame 0, 1, 2 all to the last frame and save the output of the model to the csv i.e prediction and propression

    model = EnhancedCRNN(input_height, input_width, num_classes, model_prediction=1)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # Increased initial learning rate
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    epoch_list = []
    train_loss_list = []
    val_loss_list = []
    val_accuracy_list = []

    best_val_accuracy = 0.0

    for epoch in range(100):  # Increased number of epochs
        model.train()
        running_loss = 0.0
        for i, (inputs, labels, progression) in enumerate(train_loader):
            inputs = inputs.unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            classifier_loss = criterion(outputs[:, :-1], labels)
            #print("GT progress: ", progression.float())
            #print("GT pred:", outputs[:, -1])

            progress_loss = torch.norm(outputs[:, -1] - progression.float())
            loss = classifier_loss + option.progress_weight * progress_loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        class_loss_total = 0.0
        progress_loss_total = 0.0

        correct = 0
        total = 0
        with torch.no_grad():
            for i, (inputs, labels, progression) in enumerate(val_loader):
                inputs = inputs.unsqueeze(1)
                outputs = model(inputs)
                classifier_loss = criterion(outputs[:, :-1], labels)
                progress_loss = torch.norm(outputs[:, -1] - progression.float())
                loss = classifier_loss + option.progress_weight * progress_loss
                class_loss_total += classifier_loss.item()
                progress_loss_total += progress_loss.item()
                val_loss += loss.item()
                _, predicted = torch.max(outputs[:, :-1], 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total

        class_loss_total /= len(val_loader)
        progress_loss_total /= len(val_loader)

        print(f'Epoch {epoch + 1}, Training Loss: {train_loss:.4f}, Classifier Loss: {class_loss_total:.4f}, Progress Loss: {progress_loss_total:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

        scheduler.step(val_loss)

        epoch_list.append(epoch + 1)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        val_accuracy_list.append(val_accuracy)

        # Save the model checkpoint if the validation accuracy is the best we've seen so far
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy
            }
            torch.save(checkpoint, 'best_model_checkpoint.pth')

    # Evaluate the model on the test set
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    sample_results = []
    with torch.no_grad():
        for i, (inputs, labels, progression) in enumerate(test_loader):
            inputs = inputs.unsqueeze(1)
            outputs = model(inputs)

            classifier_loss = criterion(outputs[:, :-1], labels)
            progress_loss = torch.norm(outputs[:, -1] - progression.float())
            loss = classifier_loss + option.progress_weight * progress_loss
            test_loss += loss.item()
            _, predicted = torch.max(outputs[:, :-1], 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            probabilities = F.softmax(outputs[:, :-1], dim=1).cpu().numpy()
            for j in range(inputs.size(0)):
                sample_result = [i * inputs.size(0) + j + 1, class_names[labels[j]], class_names[predicted[j]]] + list(probabilities[j] * 100)
                sample_results.append(sample_result)

    test_loss /= len(test_loader)
    test_accuracy = 100 * correct / total
    print(f"Test Loss: {test_loss}, Accuracy: {test_accuracy}%")

    results = np.column_stack((epoch_list, train_loss_list, val_loss_list, val_accuracy_list))
    header = "Epoch,TrainingLoss,ValidationLoss,ValidationAccuracy"
    np.savetxt('training_log.csv', results, delimiter=',', header=header, comments='', fmt='%f')

    with open('training_log.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Test', test_loss, test_accuracy])

    class_names = ["SR", "SL", "BR", "BL", "CR", "CL", "FR", "FL"]
    labels = list(range(len(class_names)))
    conf_matrix = confusion_matrix(all_labels, all_preds, labels=labels)
    class_report = classification_report(all_labels, all_preds, target_names=class_names, labels=labels, zero_division=0)

    print("Confusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)

    np.savetxt('confusion_matrix.csv', conf_matrix, delimiter=',', fmt='%d')
    with open('classification_report.txt', 'w') as f:
        f.write(class_report)

    ground_truth = [class_names[label] for label in all_labels]
    predictions = [class_names[pred] for pred in all_preds]
    print("\nGround Truth: ", " ".join(ground_truth))
    print("Predicted:   ", " ".join(predictions))

    class_counts = {i: 0 for i in range(len(class_names))}
    correct_pred_counts = {i: 0 for i in range(len(class_names))}

    for true_label, pred_label in zip(all_labels, all_preds):
        class_counts[true_label] += 1
        if true_label == pred_label:
            correct_pred_counts[true_label] += 1

    for i, class_name in enumerate(class_names):
        class_count = class_counts[i]
        correct_pred_count = correct_pred_counts[i]
        percentage = (correct_pred_count / class_count) * 100 if class_count > 0 else 0
        print(f"Class {class_name}: {percentage:.2f}% of predictions match the ground truth.")

    samples_header = ["Sample", "GT", "Prediction"] + class_names
    with open('sample_testing_log.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(samples_header)
        writer.writerows(sample_results)

    # Select a single animation chunk from the test set
    single_animation_chunk, _, _ = test_dataset[0]


    # Prepare data for different starting frames
    single_animation_results = []
    for start_frame in range(0, single_animation_chunk.size(0) - option.history_size + 1):
        input_chunk = single_animation_chunk[start_frame:start_frame + option.history_size].unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            output = model(input_chunk)
        predicted_progression = torch.argmax(output[:, :-1], dim=1).item()
        #print(predicted_progression)
        progression_value = output[:, -1].item()
        #print(progression_value)
        #print(start_frame)
        #exit(0)
        single_animation_results.append([start_frame, predicted_progression, progression_value])

    # Save the single animation results to a CSV file
    with open('single_animation_results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['StartFrame', 'PredictedLabel', 'ProgressionValue'])
        writer.writerows(single_animation_results)

    print("Single animation results saved to 'single_animation_results.csv'")

    # Exit the code

    ## given file name and startt frame how can i get the data and the two annotations from it ?

    #exit(0)


if __name__ == '__main__':
    main()
