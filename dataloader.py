import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from sklearn.metrics import confusion_matrix, classification_report
from torchvision import transforms
import random
from sklearn.model_selection import train_test_split

# Assuming you have the Options class and annotation_dictionary already defined
from utils.opt import Options
from bvh import Bvh
from annotations_dat import annotation_dictionary as ad


class BvhDatasets(Dataset):
    def __init__(self, opt, actions=None, split=0, transform=None):
        self.history_size = opt.history_size
        self.answers = []
        self.animation_data = {}
        self.transform = transform

        self.steps = ["SR", "SL", "BR", "BL", "CR", "CL", "FR", "FL"]
        for k in ad:
            with open(os.path.join('bvh_follower_jesse', k)) as f:
                mocap = Bvh(f.read())
                self.animation_data[k] = torch.tensor([[[mocap.frame_joint_channel(frame, j, c) for c in
                                                         ['Zrotation', 'Xrotation', 'Yrotation']] for j in
                                                        mocap.get_joints_names()] for frame in range(mocap.nframes)])

            for e in ad[k]:
                for start_f in range(e[1][0], e[1][1], self.history_size)[:-1]:
                    self.answers.append([k, start_f, e[0]])

    def one_hot_vector(self, annotation):
        one_hot = np.zeros([len(self.steps)])
        one_hot[self.steps.index(annotation)] = 1
        return one_hot

    def __getitem__(self, item):
        filename, start_frame, anno = self.answers[item]
        fs = range(start_frame, start_frame + self.history_size)
        animation_dict = self.animation_data[filename][fs]

        # Flatten the tensor to 2D [channels, height * width]
        image_data = animation_dict.view(animation_dict.size(0), -1).float()  # [history_size, 3 * number_of_joints]

        if self.transform:
            image_data = self.transform(image_data)

        label = self.steps.index(anno)  # Convert annotation to class index

        return image_data, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.answers)


# Define an enhanced CNN model
class EnhancedCNN(nn.Module):
    def __init__(self, input_height, input_width):
        super(EnhancedCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(0.5)

        # Calculate the size of the flattened tensor after convolutions and pooling
        conv_output_height = input_height // 8
        conv_output_width = input_width // 8
        flattened_size = 128 * conv_output_height * conv_output_width

        self.fc1 = nn.Linear(flattened_size, 256)  # Update fc1 with the correct size
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, len(ad))

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor while keeping the batch size
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main():
    option = Options().parse()
    transform = transforms.Compose([
        transforms.RandomApply([transforms.Lambda(lambda x: x + 0.05 * torch.randn_like(x))], p=0.5),  # Gaussian noise
        transforms.RandomApply([transforms.Lambda(lambda x: x * (0.8 + 0.4 * torch.rand_like(x)))], p=0.5)  # Random scale
    ])
    dataset = BvhDatasets(option, transform=transform)

    # Calculate class weights to handle class imbalance
    class_counts = np.bincount([dataset.steps.index(anno) for _, _, anno in dataset.answers])
    class_weights = 1. / class_counts
    sample_weights = [class_weights[dataset.steps.index(anno)] for _, _, anno in dataset.answers]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    # Split dataset into train, validation, and test sets
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Determine the correct input size for the fully connected layer
    input_height = option.history_size
    input_width = 54  # This is fixed from your data format
    model = EnhancedCNN(input_height, input_width)

    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Initialize lists for logging
    epoch_list = []
    train_loss_list = []
    val_loss_list = []
    val_accuracy_list = []

    # Training the network
    for epoch in range(20):  # Increase the number of epochs
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.unsqueeze(1)  # Add a channel dimension for the CNN
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()

        train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Loss: {train_loss}")

        # Validation step
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.unsqueeze(1)  # Add a channel dimension for the CNN
                outputs = model(inputs)

                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total
        print(f"Validation Loss: {val_loss}, Accuracy: {val_accuracy}%")

        # Append results to lists
        epoch_list.append(epoch + 1)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        val_accuracy_list.append(val_accuracy)

    # Testing the network
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.unsqueeze(1)  # Add a channel dimension for the CNN
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss /= len(test_loader)
    test_accuracy = 100 * correct / total
    print(f"Test Loss: {test_loss}, Accuracy: {test_accuracy}%")

    # Save the results to a CSV file
    results = np.column_stack((epoch_list, train_loss_list, val_loss_list, val_accuracy_list))
    header = "Epoch,TrainingLoss,ValidationLoss,ValidationAccuracy"
    np.savetxt('training_log.csv', results, delimiter=',', header=header, comments='', fmt='%f')

    # Append test results to the CSV file
    with open('training_log.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Test', test_loss, test_accuracy])

    # Calculate and print the confusion matrix and classification report
    class_names = ["SR", "SL", "BR", "BL", "CR", "CL", "FR", "FL"]
    labels = list(range(len(class_names)))
    conf_matrix = confusion_matrix(all_labels, all_preds, labels=labels)
    class_report = classification_report(all_labels, all_preds, target_names=class_names, labels=labels)

    print("Confusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)

    # Save confusion matrix and classification report to CSV
    np.savetxt('confusion_matrix.csv', conf_matrix, delimiter=',', fmt='%d')
    with open('classification_report.txt', 'w') as f:
        f.write(class_report)


if __name__ == '__main__':
    main()
