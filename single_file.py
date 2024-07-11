## libraries go in here
import csv
import numpy as np
import os
import torch
import argparse

from bvh import Bvh
from annotations_dat import annotation_dictionary as ad
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F


class BvhDataset(Dataset):
    def __init__(self, filename, history_size, transform=None):
        self.history_size = history_size
        self.answers = []
        self.animation_data = {}
        self.transform = transform

        self.steps = ["SR", "SL", "BR", "BL", "CR", "CL", "FR", "FL"]

        # Load the specified file
        with open(filename) as f:
            mocap = Bvh(f.read())
            self.animation_data[filename] = torch.tensor([[[mocap.frame_joint_channel(frame, j, c) for c in
                                                            ['Zrotation', 'Xrotation', 'Yrotation']] for j in
                                                           mocap.get_joints_names()] for frame in range(mocap.nframes)])

        # Process annotations
        basename = os.path.basename(filename)
        if basename in ad:
            for e in ad[basename]:
                for start_f in range(e[1][0], e[1][1] - self.history_size):
                    progression = (self.history_size + start_f - e[1][0]) / (e[1][1] - e[1][0])

                    print("SF: %d, hz: %d, step_start: %d, step_end: %d, progression: %f"%
                          (start_f, self.history_size, e[1][0], e[1][1], progression))

                    self.answers.append([filename, start_f, e[0], progression])
        else:
            print(f"No annotations found for {basename}")

    def __getitem__(self, item):
        filename, start_frame, anno, progression = self.answers[item]
        fs = range(start_frame, start_frame + self.history_size)
        animation_dict = self.animation_data[filename][fs]

        image_data = animation_dict.view(animation_dict.size(0), -1).float()

        if self.transform:
            image_data = self.transform(image_data.unsqueeze(0)).squeeze(0)

        label = self.steps.index(anno)

        return image_data, torch.tensor(label, dtype=torch.long), torch.tensor(progression, dtype=torch.double), start_frame, filename

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


def single_test(opt):
    # Load the dataset
    dataset = BvhDataset(opt.filename, opt.history_size)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Initialize and load the model
    input_height = opt.history_size
    input_width = 54  # Assuming the width based on the 18 joints * 3 channels (Z, X, Y)
    num_classes = len(dataset.steps)
    model = EnhancedCRNN(input_height, input_width, num_classes, model_prediction=1)

    checkpoint = torch.load(opt.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    results = []

    with torch.no_grad():
        for data in loader:
            inputs, labels, progression, start_frame, filename = data
            inputs = inputs.unsqueeze(1)  # Add a channel dimension
            outputs = model(inputs)

            predicted_label = torch.argmax(outputs[:, :-1], dim=1).item()
            progression_value = outputs[:, -1].item()
            gt_label = labels.item()
            gt_progression = progression.item()
            results.append([start_frame.item(), predicted_label, progression_value, gt_label, gt_progression])

    # Save results to CSV
    result_filename = f"{os.path.splitext(os.path.basename(opt.filename))[0]}_results.csv"
    with open(result_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['StartFrame', 'PredictedLabel', 'PredictedProgression', 'GTLabel', 'GTProgression'])
        writer.writerows(results)

    print(f"Results saved to {result_filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process BVH file.')
    parser.add_argument('--filename', type=str, required=True, help='Path to the BVH file')
    parser.add_argument('--history_size', type=int, default=30, help='History size for the data')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint file')

    opt = parser.parse_args()
    single_test(opt)
