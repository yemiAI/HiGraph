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
from collections import defaultdict
from utils.opt import Options
from bvh import Bvh

# from annotations_dat import annotation_dictionary as ad
from annontations_jesse_leader import annotation_dictionary as ad


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
                for start_f in range(e[1][0], e[1][1], self.history_size)[:-1]:
                    self.answers.append([k, start_f, e[0]])

    def __getitem__(self, item):
        filename, start_frame, anno = self.answers[item]
        fs = range(start_frame, start_frame + self.history_size)
        animation_dict = self.animation_data[filename][fs]

        image_data = animation_dict.view(animation_dict.size(0), -1).float()
        print(image_data.shape)
        # exit(0)

        if self.transform:
            image_data = self.transform(image_data.unsqueeze(0)).squeeze(0)

        label = self.steps.index(anno)

        return image_data, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.answers)


class EnhancedCRNN(nn.Module):
    def __init__(self, input_height, input_width, num_classes):
        super(EnhancedCRNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(0.5)

        self.gru = nn.GRU(input_size=128, hidden_size=256, num_layers=3, batch_first=True, dropout=0.5)
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, num_classes)

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


def apply_random_oversampling(data, labels):
    ros = RandomOverSampler(random_state=42)
    data_res, labels_res = ros.fit_resample(data, labels)
    return data_res, labels_res


class StateMachine:
    def __init__(self, model, steps):
        self.state = None  # No initial state
        self.model = model
        self.steps = steps
        self.prev_state = None
        self.state_log = []  # To log state transitions
        self.transition_log = []  # To log transition actions
        self.transition_counts = defaultdict(lambda: defaultdict(int))  # To count transitions

    def predict_step(self, input_data):
        with torch.no_grad():
            output = self.model(input_data)
            _, predicted = torch.max(output, 1)
            return [self.steps[pred.item()] for pred in predicted]

    def enter_state(self):
        if self.state:
            print(f"Entering state: {self.state}")

    def exit_state(self):
        if self.prev_state:
            print(f"Exiting state: {self.prev_state}")

    def transition(self, input_data):
        predicted_steps = self.predict_step(input_data)
        for step in predicted_steps:
            if step in self.steps:
                if self.state != step:
                    self.prev_state = self.state
                    self.exit_state()
                    self.transition_log.append(('exit', self.state))
                    self.state = step
                    self.enter_state()
                    self.transition_log.append(('enter', self.state))
                    self.transition_counts[self.prev_state][self.state] += 1
                self.perform_action()
                self.transition_log.append(('action', self.state))
            else:
                print("Unknown step prediction:", step)

    def perform_action(self):
        if self.state:
            print(f"Performing action for {self.state}")

def print_state_transition_probabilities(steps, transition_counts):
    print("State Transition Probabilities:")
    for from_step, to_steps in transition_counts.items():
        total_transitions = sum(to_steps.values())
        for to_step, count in to_steps.items():
            probability = count / total_transitions
            print(f"From {from_step} to {to_step}: {probability:.2f}")

def main():
    set_seed(42)  # Set seed for reproducibility
    option = Options().parse()
    transform = transforms.Compose([
        transforms.RandomApply([transforms.Lambda(lambda x: x + 5 * torch.randn_like(x))], p=0.5),
    ])
    dataset = BvhDatasets(option, transform=transform)

    num_classes = len(dataset.steps)
    class_names = dataset.steps

    # Count samples before oversampling
    class_counts_before = {step: 0 for step in dataset.steps}
    for _, label in dataset:
        class_counts_before[dataset.steps[label.item()]] += 1

    print("Class distribution before oversampling:")
    for step, count in class_counts_before.items():
        print(f"{step}: {count}")

    data = []
    labels = []
    for i in range(len(dataset)):
        img, label = dataset[i]
        data.append(img.numpy().flatten())
        labels.append(label.numpy())
    data = np.array(data)
    labels = np.array(labels)

    data_res, labels_res = apply_random_oversampling(data, labels)
    data_res = torch.tensor(data_res).view(-1, option.history_size, 54).float()
    labels_res = torch.tensor(labels_res).long()

    # Count samples after oversampling
    class_counts_after = {step: 0 for step in dataset.steps}
    for label in labels_res:
        class_counts_after[dataset.steps[label.item()]] += 1

    print("Class distribution after oversampling:")
    for step, count in class_counts_after.items():
        print(f"{step}: {count}")

    balanced_dataset = TensorDataset(data_res, labels_res)

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
    model = EnhancedCRNN(input_height, input_width, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # Increased initial learning rate
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    epoch_list = []
    train_loss_list = []
    val_loss_list = []
    val_accuracy_list = []

    for epoch in range(200):  # Increased number of epochs
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(val_loader):
                inputs = inputs.unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total

        print(
            f'Epoch {epoch + 1}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

        scheduler.step(val_loss)

        epoch_list.append(epoch + 1)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        val_accuracy_list.append(val_accuracy)

    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    sample_results = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Save probabilities and predictions for each sample
            probabilities = F.softmax(outputs, dim=1).cpu().numpy()
            for j in range(inputs.size(0)):
                sample_result = [i * inputs.size(0) + j + 1, class_names[labels[j]], class_names[predicted[j]]] + list(
                    probabilities[j] * 100)
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
    class_report = classification_report(all_labels, all_preds, target_names=class_names, labels=labels,
                                         zero_division=0)

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

    # Save detailed sample testing results
    samples_header = ["Sample", "GT", "Prediction"] + class_names
    with open('sample_testing_log.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(samples_header)
        writer.writerows(sample_results)

    # Integrate state machine
    sm = StateMachine(model, class_names)  # No need to exclude initial

    # Real-time simulation: process each sample individually
    for inputs, _ in test_loader:
        for input_data in inputs:
            input_data = input_data.unsqueeze(0).unsqueeze(0)
            sm.transition(input_data)

    # Print state transitions
    print_state_transition_probabilities(sm.steps, sm.transition_counts)

if __name__ == '__main__':
    main()
