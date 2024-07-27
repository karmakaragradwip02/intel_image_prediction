import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from pathlib import Path
import json
import numpy as np
import mlflow
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, cohen_kappa_score
from urllib.parse import urlparse
from Intel_image_prediction import logger

MLFLOW_TRACKING_URI = "https://dagshub.com/karmakaragradwip02/intel_image_prediction.mlflow"
os.environ['MLFLOW_TRACKING_URI'] = MLFLOW_TRACKING_URI
os.environ['MLFLOW_TRACKING_USERNAME'] = 'karmakaragradwip02'
os.environ['MLFLOW_TRACKING_PASSWORD'] = '9ccb0f28354fcca6469017b32544fa0704b9c343'

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

class ModelEvaluation:
    def __init__(self, config):
        self.config = config

    def model(self):
        cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(64 * 37 * 37, 512),
            nn.ReLU(),
            nn.Linear(512, self.config.classes)
        )
        return cnn
    
    def define_transforms(self):
        transformer = transforms.Compose([
            transforms.Resize((150, 150)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), 
            transforms.Normalize([0.5, 0.5, 0.5],
                                [0.5, 0.5, 0.5])
        ])
        return transformer
    
    def data_loader(self, transformer):
        test_loader = DataLoader(
            torchvision.datasets.ImageFolder(self.config.test_dir, transform=transformer),
            batch_size=16, shuffle=True
        )
        return test_loader

    def load_model(self, model):
        model_dir = Path(self.config.trained_model_dir)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(model_dir, map_location=device))
        return model
    
    def evaluate_model(self, model, test_loader):
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
        
        return np.array(y_true), np.array(y_pred)
    
    def log_into_mlflow(self, model, test_loader):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        # Read and parse the history data
        history_path = Path(self.config.history_dir)
        if history_path.is_file():
            with history_path.open('r') as f:
                history_data = json.load(f)
                
            # Log each epoch's metrics individually
            with mlflow.start_run():
                mlflow.log_params(self.config.all_params)
                y_true, y_pred = self.evaluate_model(model, test_loader)

                # Calculate precision and recall
                precision = precision_score(y_true, y_pred, average='macro', zero_division=1)
                recall = recall_score(y_true, y_pred, average='macro')
                m_accuracy = accuracy_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred, average='macro')
                kappa = cohen_kappa_score(y_true, y_pred)
                
                # Log metrics
                mlflow.log_metric('Model Accuracy', m_accuracy)
                mlflow.log_metric('Model Precision', precision)
                mlflow.log_metric('Model Recall', recall)
                mlflow.log_metric('Model F1 Score', f1)
                mlflow.log_metric('Model Kappa', kappa)
                
                logger.info("--------------------- metrics logged ------------------")

                for epoch in range(len(history_data.get("loss", []))):
                    mlflow.log_metric("train_loss", history_data["train_loss"][epoch], step=epoch)
                    mlflow.log_metric("train_accuracy", history_data["train_accuracy"][epoch], step=epoch)
                    mlflow.log_metric("val_loss", history_data["val_loss"][epoch], step=epoch)
                    mlflow.log_metric("val_accuracy", history_data["val_accuracy"][epoch], step=epoch)
                if tracking_url_type_store != "file":
                    mlflow.pytorch.log_model(model, "model", registered_model_name="custom_model")
                else:
                    mlflow.pytorch.log_model(model, "model")
                logger.info("------------------- model logged ---------------------")