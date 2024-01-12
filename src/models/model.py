import torch.nn as nn
from torch import optim
from pytorch_lightning import LightningModule
import timm

class ResNetModel(LightningModule):

    def __init__(self, model_name='resnet18', lr=1e-3):
        super().__init__()
        self.net = timm.create_model(model_name=model_name, pretrained=True, num_classes=2)
        self.lossfunc = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        loss = self.lossfunc(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log('train_loss', loss)
        self.log('train_acc', acc)

        return loss
    
    def validation_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        loss = self.lossfunc(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log('val_loss', loss)
        self.log('val_acc', acc)

    
    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.lr)
    
