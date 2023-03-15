
from argparse import ArgumentParser
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torchmetrics import AUC, AUROC, Accuracy, PrecisionRecallCurve

OPTIMIZER = {
    "adam": Adam,
    "sgd": SGD
}

CRITERION = {
    "cross_entropy": nn.CrossEntropyLoss,
    "binary_cross_entropy": nn.BCEWithLogitsLoss
}


class Word2VecModel(pl.LightningModule):
    def __init__(self, num_features=1000, lr=0.01):
        super(Word2VecModel, self).__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.num_features = num_features
        self.optimizer = OPTIMIZER["adam"]
        self.criterion = CRITERION["cross_entropy"]

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        self.test_AUROC = AUROC(pos_label=1)
        self.test_PRC = PrecisionRecallCurve(pos_label=1)
        self.test_AUC = AUC()

        self.model = nn.Sequential(
            nn.Linear(self.num_features, 5)
        )

    def forward(self, x):
        print(x)
        logits = self.model(x)
        return logits

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        y_pred = torch.softmax(y_hat, dim=1)
        acc = self.train_acc(y_pred, y)
        self.log("train_loss", loss)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        y_pred = torch.softmax(y_hat, dim=1)
        acc = self.val_acc(y_pred, y)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        y_pred = torch.softmax(y_hat, dim=1)
        acc = self.test_acc(y_pred, y)
        self.log("test_acc", acc)
        return loss

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = parent_parser.add_argument_group("Model")
        parser.add_argument("--num_features", help="The number of input features")
        return parent_parser
