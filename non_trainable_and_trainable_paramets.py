import os
import numpy as np
import torch
import glob
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import torchvision
import pathlib
import matplotlib.pyplot as plt
from torchsummary import summary

################################################################################
def define_transforms():
    transformer = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # 0-250 to 0-1 numpy to tensors
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])
    ])
    print("------- Transformer for image processing --------------------")
    return transformer

################################################################################
def data_loader(train_path, test_path, transformer):
    train_loader = DataLoader(
        torchvision.datasets.ImageFolder(train_path, transform=transformer),
        batch_size=16, shuffle=True
    )
    test_loader = DataLoader(
        torchvision.datasets.ImageFolder(test_path, transform=transformer),
        batch_size=16, shuffle=True
    )
    print("--------------- train and test loader loaded -----------------")
    return train_loader, test_loader

################################################################################
def class_name(train_path):
    root = pathlib.Path(train_path)
    classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])
    return classes

################################################################################
class YourModelClass:
    def __init__(self):
        self.classes = 6  
        self.epsilon = 1e-8  
        self.learning_rate = 1e-3 
        self.weight_decay = 1e-5  
        self.decay_rate = 0.1  

    def model(self):
        cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (64, 75, 75)

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (128, 37, 37)

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (256, 18, 18)

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (512, 9, 9)

            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (1024, 4, 4)

            nn.Flatten()
        )

        num_features = 1024 * 4 * 4

        fc_layers = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, self.classes)
        )

        model = nn.Sequential(
            cnn,
            fc_layers
        )

        # Freezing the layers
        for param in model.parameters():
            param.requires_grad = False

        # Unfreezing layers to match the target of 5,768,454 trainable params
        # Unfreezing the last 5 convolutional layers and the fully connected layers
        for layer in model[0][-5:]:
            for param in layer.parameters():
                param.requires_grad = True

        for param in model[1].parameters():
            param.requires_grad = True

        return model

    def print_summary(self, input_size, device):
        model = self.model()
        # Print summary with model on CPU
        summary(model.to('cpu'), input_size)

################################################################################
def train_test_count(train_path, test_path):
    train_count = len(glob.glob(train_path + '/**/*.jpg'))
    test_count = len(glob.glob(test_path + '/**/*.jpg'))
    return train_count, test_count

################################################################################
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    train_path = 'artifacts/data_preparation/train'
    test_path = 'artifacts/data_preparation/test'
    transformer = define_transforms()
    train_loader, test_loader = data_loader(train_path, test_path, transformer)
    classes = class_name(train_path)
    train_count, test_count = train_test_count(train_path, test_path)
    model_instance = YourModelClass()
    # Print model summary with model on CPU
    model_instance.print_summary((3, 150, 150), device)
    model = model_instance.model().to(device)  # Move model to the GPU
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    loss_function = nn.CrossEntropyLoss()
    num_epochs = 1
    for epoch in range(num_epochs):
        print("Epoch:", epoch)
        model.train()
        train_accuracy = 0.0
        train_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)  # Move images and labels to the GPU
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.cpu().item() * images.size(0)
            _, prediction = torch.max(outputs.data, 1)
            train_accuracy += int(torch.sum(prediction == labels.data))
        train_accuracy = train_accuracy / train_count
        train_loss = train_loss / train_count
        model.eval()

        test_accuracy = 0.0
        test_loss = 0.0
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):
                images, labels = images.to(device), labels.to(device)  # Move images and labels to the GPU
                outputs = model(images)
                loss = loss_function(outputs, labels)
                test_loss += loss.cpu().item() * images.size(0)
                _, prediction = torch.max(outputs.data, 1)
                test_accuracy += int(torch.sum(prediction == labels.data))
        test_accuracy = test_accuracy / test_count
        test_loss = test_loss / test_count

        print(f'Train Loss: {train_loss} Train Accuracy: {train_accuracy} Test Loss: {test_loss} Test Accuracy: {test_accuracy}')

if __name__ == "__main__":
    main()
