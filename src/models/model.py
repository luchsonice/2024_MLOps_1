import torch.nn as nn
from torch import optim
from pytorch_lightning import LightningModule
import torchmetrics
import timm

class ResNetModel(LightningModule):

    def __init__(self, model_name='resnet18', lr=1e-3):
        super().__init__()

        # Load pretrained model from TIMM
        self.net = timm.create_model(model_name=model_name, pretrained=True, num_classes=2)

        # Loss function
        self.lossfunc = nn.CrossEntropyLoss()
        # Learning rate
        self.lr = lr

        # Metrics
        self.train_acc = torchmetrics.classification.Accuracy(task="binary")
        self.val_acc = torchmetrics.classification.Accuracy(task="binary")
        self.test_acc = torchmetrics.classification.Accuracy(task="binary")
        
        self.train_precision = torchmetrics.classification.Precision(task="binary")
        self.val_precision = torchmetrics.classification.Precision(task="binary")
        self.test_precision = torchmetrics.classification.Precision(task="binary")

        self.train_recall = torchmetrics.classification.Recall(task="binary")
        self.val_recall = torchmetrics.classification.Recall(task="binary")
        self.test_recall = torchmetrics.classification.Recall(task="binary")

        self.train_f1 = torchmetrics.classification.F1Score(task="binary")
        self.val_f1 = torchmetrics.classification.F1Score(task="binary")
        self.test_f1 = torchmetrics.classification.F1Score(task="binary")

        # Note that confusion matrix cannot be logged as it is not a scalar
        #self.train_confusion_matrix = torchmetrics.classification.ConfusionMatrix(task="binary")
        #self.val_confusion_matrix = torchmetrics.classification.ConfusionMatrix(task="binary")
        #self.test_confusion_matrix = torchmetrics.classification.ConfusionMatrix(task="binary")

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.lossfunc(preds, y)
        y_hat = preds.argmax(dim=-1)

        self.train_acc(y_hat, y)
        self.train_precision(y_hat, y)
        self.train_recall(y_hat, y)
        self.train_f1(y_hat, y)
        #self.train_confusion_matrix(y_hat, y)

        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)
        self.log('train_precision', self.train_precision, on_step=True, on_epoch=True)
        self.log('train_recall', self.train_recall, on_step=True, on_epoch=True)
        self.log('train_f1', self.train_f1, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.lossfunc(preds, y)
        y_hat = preds.argmax(dim=-1)

        self.val_acc(y_hat, y)
        self.val_precision(y_hat, y)
        self.val_recall(y_hat, y)
        self.val_f1(y_hat, y)
        #self.val_confusion_matrix(y_hat, y)

        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True)
        self.log('val_precision', self.val_precision, on_step=False, on_epoch=True)
        self.log('val_recall', self.val_recall, on_step=False, on_epoch=True)
        self.log('val_f1', self.val_f1, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.lossfunc(preds, y)
        y_hat = preds.argmax(dim=-1)

        self.test_acc(y_hat, y)
        self.test_precision(y_hat, y)
        self.test_recall(y_hat, y)
        self.test_f1(y_hat, y)

        self.log('test_acc', self.test_acc)
        self.log('test_precision', self.test_precision)
        self.log('test_recall', self.test_recall)
        self.log('test_f1', self.test_f1)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch[0])

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.lr)
