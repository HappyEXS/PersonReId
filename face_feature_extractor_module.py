# model_lightning.py
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import Adam
import torchmetrics

from face_feature_extractor import FaceFeatureExtractor


class FaceReIDModule(pl.LightningModule):
    def __init__(self, num_classes=256, learning_rate=0.00035, embedding_dim=512):
        super().__init__()
        self.lr = learning_rate

        self.model = FaceFeatureExtractor(
            num_classes=num_classes, embedding_dim=embedding_dim
        )

        self.criterion = nn.CrossEntropyLoss()
        # self.accuracy = torchmetrics.Accuracy(
        #     task="multiclass", num_classes=num_classes
        # )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        _, pids, _, _, faces = batch

        # Filtorwanie
        # valid_mask = faces.sum(dim=(1, 2, 3)) > 0

        # if valid_mask.sum() == 0:
        #     return None  # Pomijamy krok, jeśli w całym batchu nie ma twarzy

        # valid_faces = faces[valid_mask]
        # valid_pids = pids[valid_mask]

        valid_faces = faces
        valid_pids = pids

        logits = self.model(valid_faces)
        loss = self.criterion(logits, valid_pids)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == valid_pids).float().mean()
        # acc = self.accuracy(logits, valid_pids)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        _, pids, _, _, faces = batch

        # valid_mask = faces.sum(dim=(1, 2, 3)) > 0
        # if valid_mask.sum() == 0:
        #     return

        # valid_faces = faces[valid_mask]
        # valid_pids = pids[valid_mask]

        valid_faces = faces
        valid_pids = pids

        logits = self.model(valid_faces)
        loss = self.criterion(logits, valid_pids)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == valid_pids).float().mean()

        # acc = self.accuracy(logits, valid_pids)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        _, pids, _, _, faces = batch

        # valid_mask = faces.sum(dim=(1, 2, 3)) > 0
        # if valid_mask.sum() == 0:
        #     return

        # valid_faces = faces[valid_mask]
        # valid_pids = pids[valid_mask]

        valid_faces = faces
        valid_pids = pids

        logits = self.model(valid_faces)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == valid_pids).float().mean()

        # acc = self.accuracy(logits, valid_pids)

        self.log("test_acc", acc)

    def configure_optimizers(self):
        optimizer = Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr,
        )
        return optimizer
