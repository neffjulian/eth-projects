import torch
from torch import nn
import torch.nn.functional as F

import torchmetrics

import pytorch_lightning as pl

from sklearn.metrics import accuracy_score
import shap

class BaselineClf(pl.LightningModule):
    def __init__(self, in_channels=3):
        super().__init__()
        self.in_channels = in_channels
        self.model = nn.Sequential(
            nn.Conv2d(self.in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False), # 64x64
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False), # 32x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False), # 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False), # 8x8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False), # 4x4
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False), # 2x2
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), 
            nn.Linear(512, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=-1),
        )
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch 
        # Get predictions
        out = self(images)         
        # Get loss        
        loss = F.cross_entropy(out, labels) 
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch 
        # Get predictions
        out = self(images)                    
        # Get loss
        loss = F.cross_entropy(out, labels)   
        # Get accuracy
        _, preds = torch.max(out, dim=1)
        acc = accuracy_score(labels.cpu(), preds.cpu())
        print("Accuracy: ", acc)          
        return {'val_loss': loss, 'val_acc': acc}

    def test_step(self, batch, batch_idx):
        print("test")
        images, labels = batch 
        # Get predictions
        out = self(images)                    
        # Get loss
        loss = F.cross_entropy(out, labels)   
        # Get accuracy
        _, preds = torch.max(out, dim=1)
        acc = accuracy_score(labels.cpu(), preds.cpu())          
        return {'val_loss': loss, 'val_acc': acc}


class SIFTClf(pl.LightningModule):
    def __init__(self, in_channels=3):
        super().__init__()
        self.accuracy = torchmetrics.Accuracy()
        self.in_channels = in_channels
        self.model = nn.Sequential(
            nn.Conv2d(self.in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False), # 64x64
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False), # 32x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False), # 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False), # 8x8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False), # 4x4
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False), # 2x2
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), 
            nn.Linear(512, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=-1),
        )
        
    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        images, labels = batch 
        labels = torch.squeeze(labels)
        out = self(images)         
        loss = F.cross_entropy(out, labels) 
        _, preds = torch.max(out, dim=1)
        self.accuracy(preds, labels)
        self.log('Training Accuracy: ', self.accuracy)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch 
        labels = torch.squeeze(labels)
        out = self(images)                    
        loss = F.cross_entropy(out, labels)   
        _, preds = torch.max(out, dim=1)
        self.accuracy(preds, labels)
        self.log('Validation Accuracy: ', self.accuracy)

    def test_step(self, batch, batch_idx):
        images, labels = batch 
        labels = torch.squeeze(labels)
        out = self(images)
        loss = F.cross_entropy(out, labels)   
        _, preds = torch.max(out, dim=1)
        self.accuracy(preds, labels)
        self.log('Test Accuracy: ', self.accuracy)      
