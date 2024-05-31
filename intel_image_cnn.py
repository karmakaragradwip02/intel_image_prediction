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
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt

################################################################################
def define_transforms():
    transformer = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # 0-250 to 0-1 numpy to tensors
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])
    ])
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
    return train_loader, test_loader

################################################################################
def class_name(train_path):
    root = pathlib.Path(train_path)
    classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])
    return classes

#################################################################################
class ConvNet(nn.Module):
    def __init__(self, num_classes=6):
        super(ConvNet, self).__init__()

        # Output size after convolution filter
        # ((w-f+2P)/s) +1

        # Input shape= (256,3,150,150)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        # Shape= (256,12,150,150)
        self.bn1 = nn.BatchNorm2d(num_features=12)
        # Shape= (256,12,150,150)
        self.relu1 = nn.ReLU()
        # Shape= (256,12,150,150)

        self.pool = nn.MaxPool2d(kernel_size=2)
        # Reduce the image size by factor 2
        # Shape= (256,12,75,75)

        self.conv2 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1)
        # Shape= (256,20,75,75)
        self.relu2 = nn.ReLU()
        # Shape= (256,20,75,75)

        self.conv3 = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1)
        # Shape= (256,32,75,75)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        # Shape= (256,32,75,75)
        self.relu3 = nn.ReLU()
        # Shape= (256,32,75,75)

        self.fc = nn.Linear(in_features=75 * 75 * 32, out_features=num_classes)

    # Feed forward function
    def forward(self, input):
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu1(output)

        output = self.pool(output)

        output = self.conv2(output)
        output = self.relu2(output)

        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu3(output)

        # Above output will be in matrix form, with shape (256,32,75,75)

        output = output.view(-1, 32 * 75 * 75)

        output = self.fc(output)

        return output

###########################################################################
###########################################################################
def train_test_count(train_path, test_path):
    train_count = len(glob.glob(train_path + '/**/*.jpg'))
    test_count = len(glob.glob(test_path + '/**/*.jpg'))
    return train_count, test_count

###########################################################################
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    train_path = 'E:\\Deep Learning\\pytorch\\intel_image_prediction\\Dataset\\seg_train'
    test_path = 'E:\\Deep Learning\\pytorch\\intel_image_prediction\\Dataset\\seg_test'
    
    transformer = define_transforms()

    train_loader, test_loader = data_loader(train_path, test_path, transformer)

    classes = class_name(train_path)
    print(classes)

    train_count, test_count = train_test_count(train_path, test_path)

    model = ConvNet(num_classes=6).to(device)
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    loss_function = nn.CrossEntropyLoss()
    num_epochs = 1

    with mlflow.start_run() as run:
        try:
            print("mlflow starting")
            mlflow.log_param("learning_rate", 0.001)
            mlflow.log_param("weight_decay", 0.0001)
            mlflow.log_param("batch_size", 16)
            mlflow.log_param("epochs", 1)

            for epoch in range(num_epochs):
                print("Epoch:", epoch)
                model.train()
                train_accuracy = 0.0
                train_loss = 0.0

                for i, (images, labels) in enumerate(train_loader):
                    if torch.cuda.is_available():
                        images = Variable(images.cuda())
                        labels = Variable(labels.cuda())
                        
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
                
                mlflow.pytorch.log_model(model)

                model.eval()

                test_accuracy = 0.0
                for i, (images, labels) in enumerate(test_loader):
                    if torch.cuda.is_available():
                        images = Variable(images.cuda())
                        labels = Variable(labels.cuda())
                    
                    outputs = model(images)
                    _, prediction = torch.max(outputs.data, 1)
                    test_accuracy += int(torch.sum(prediction == labels.data))
                    
                test_accuracy = test_accuracy / test_count
                
                mlflow.log_metric("train_loss", train_loss)
                mlflow.log_metric("train_accuracy", train_accuracy)
                mlflow.log_metric("test_accuracy", test_accuracy)

                print('Epoch: ' + str(epoch) + ' Train Loss: ' + str(train_loss) + ' Train Accuracy: ' + str(train_accuracy) + ' Test Accuracy: ' + str(test_accuracy))
        except Exception as e:
            print(f"Exception during training: {e}")
        finally:
            mlflow.end_run()

if __name__ == "__main__":
    main()
