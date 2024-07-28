import torch
from torchsummary import summary
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import json
from Intel_image_prediction import logger
from Intel_image_prediction.entity.config_entity import ModelPreparationTrainingConfig
import torch
from torchsummary import summary
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import json

class ModelPreparation:
    def __init__(self, config):
        self.config = config

    def model(self):
        cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, self.config.classes)
        )
        return cnn

    def image_processing(self):
        resize_size = self.config.input_image_size[-2:]

        transformer = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        train_loader = DataLoader(
            torchvision.datasets.ImageFolder(self.config.train_dir, transform=transformer),
            batch_size=self.config.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            torchvision.datasets.ImageFolder(self.config.val_dir, transform=transformer),
            batch_size=self.config.batch_size, shuffle=True
        )

        train_count = len(train_loader.dataset)
        val_count = len(val_loader.dataset)

        return train_loader, val_loader, train_count, val_count

    def model_compilation(self, model):
        epsilon = self.config.epsilon
        learning_rate = self.config.learning_rate
        #optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=self.config.momentum, weight_decay=self.config.weight_decay)
        #scheduler = ExponentialLR(optimizer, gamma=self.config.decay_rate)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=self.config.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=self.config.decay_rate)
        criterion = nn.CrossEntropyLoss()
        return model, optimizer, scheduler, criterion

    def train_model(self, model, optimizer, scheduler, criterion, train_loader, val_loader, train_count, val_count):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        logger.info(f"------------- Training Started on {device} device ----------------")

        metrics = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": []
        }

        for epoch in range(self.config.epochs):
            print(f"Epoch {epoch+1}/{self.config.epochs}")
            model.train()
            train_loss, train_accuracy = 0, 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)
                _, prediction = torch.max(outputs.data, 1)
                train_accuracy += int(torch.sum(prediction == labels.data))

            train_accuracy = train_accuracy / train_count
            train_loss = train_loss / train_count

            # Scheduler step
            scheduler.step()

            # Validation phase
            model.eval()
            val_loss, val_accuracy = 0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    _, prediction = torch.max(outputs.data, 1)
                    val_accuracy += int(torch.sum(prediction == labels.data))

            val_accuracy = val_accuracy / val_count
            val_loss = val_loss / val_count

            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

            metrics["train_loss"].append(train_loss)
            metrics["train_accuracy"].append(train_accuracy)
            metrics["val_loss"].append(val_loss)
            metrics["val_accuracy"].append(val_accuracy)

        with open(self.config.history_dir, 'w') as f:
            json.dump(metrics, f, indent=4)

        logger.info("------------------Training And Evaluation Ended -------------------")
        return model

    def print_model_summary(self, model, input_size):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        summary(model, input_size, device=str(device))
        
    def save_model(self, model):
        model_path = self.config.model_dir
        torch.save(model.state_dict(), model_path)
        print(f'Model saved to {model_path}')