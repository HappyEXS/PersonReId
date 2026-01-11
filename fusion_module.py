import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam

from fusion import Fusion

class FusionModule(pl.LightningModule):
    def __init__(self, appearance_net, face_net, num_classes=256, lr=0.00035, fusion_weight=0.9):
        super().__init__()
        self.save_hyperparameters(ignore=['appearance_net', 'face_net'])
        
        self.model = Fusion(
            appearance_net=appearance_net,
            face_net=face_net,
            num_classes=num_classes,
            fusion_weight=fusion_weight
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, body_img, face_img):
        return self.model(body_img, face_img)

    def training_step(self, batch, batch_idx):
        # img, pid, camid, clothes_id, face
        body_img, pids, _, _, face_img = batch
        
        # Forward pass przez model fuzyjny
        logits = self.model(body_img, face_img)
        
        loss = self.criterion(logits, pids)
        
        # Logowanie
        preds = torch.argmax(logits, dim=1)
        acc = (preds == pids).float().mean()
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        body_img, pids, _, _, face_img = batch
        
        # W walidacji model.training = False, więc forward zwróci embeddingi.
        # Aby policzyć loss/acc musimy wymusić zwrot logitów lub zmienić logikę w forward.
        # Najprościej: wywołajmy classifier ręcznie w walidacji.
        
        embedding = self.model(body_img, face_img)
        logits = self.model.classifier(embedding)
        
        loss = self.criterion(logits, pids)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == pids).float().mean()
        
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        body_img, pids, _, _, face_img = batch
        
        embedding = self.model(body_img, face_img)
        logits = self.model.classifier(embedding)
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == pids).float().mean()
        
        self.log("test_acc", acc)

    def configure_optimizers(self):
        # Optymalizujemy parametry obu sieci (chyba że są zamrożone) oraz nowego klasyfikatora
        # return Adam(self.model.parameters(), lr=self.lr)
    
        optimizer = Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr,
        )
        return optimizer