import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam
import numpy as np

from models.fusion import Fusion


class FusionModule(pl.LightningModule):
    def __init__(
        self, appearance_net, face_net, num_classes=256, lr=0.00035, fusion_weight=0.9
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["appearance_net", "face_net"])

        self.model = Fusion(
            appearance_net=appearance_net,
            face_net=face_net,
            num_classes=num_classes,
            fusion_weight=fusion_weight,
        )

        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr

        self.test_outputs = []

    def forward(self, body_img, face_img):
        return self.model(body_img, face_img)

    def training_step(self, batch, batch_idx):
        # img, pid, camid, clothes_id, face
        body_img, pids, _, _, face_img = batch

        embedding = self.model(body_img, face_img)

        # Dla treningu embedingi przepuszczamy przez warstę klasyfikującą
        logits = self.model.classifier(embedding)
        loss = self.criterion(logits, pids)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == pids).float().mean()

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        body_img, pids, _, _, face_img = batch

        embedding = self.model(body_img, face_img)

        # Dla walidacji (również dane z train) embedingi przepuszczamy przez warstę klasyfikującą
        logits = self.model.classifier(embedding)
        loss = self.criterion(logits, pids)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == pids).float().mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        body_img, pids, camids, _, face_img = batch

        embeddings = self.model(body_img, face_img)

        self.test_outputs.append(
            {
                "embeddings": embeddings.cpu(),
                "pids": pids.cpu(),
                "camids": camids.cpu(),
                "d_idx": dataloader_idx,
            }
        )

    def on_test_epoch_end(self):
        # 1. Podział zebranych danych na Query i Gallery
        q_feats = torch.cat(
            [x["embeddings"] for x in self.test_outputs if x["d_idx"] == 0]
        )
        q_pids = torch.cat([x["pids"] for x in self.test_outputs if x["d_idx"] == 0])
        q_camids = torch.cat(
            [x["camids"] for x in self.test_outputs if x["d_idx"] == 0]
        )

        g_feats = torch.cat(
            [x["embeddings"] for x in self.test_outputs if x["d_idx"] == 1]
        )
        g_pids = torch.cat([x["pids"] for x in self.test_outputs if x["d_idx"] == 1])
        g_camids = torch.cat(
            [x["camids"] for x in self.test_outputs if x["d_idx"] == 1]
        )

        # 2. Obliczanie macierzy odległości (Cosinusowa)
        # Dist = 1 - Similarity
        distmat = 1 - torch.mm(q_feats, g_feats.t())
        distmat = distmat.numpy()

        # 3. Wywołanie funkcji liczącej metryki
        mAP, r1, r5, r10 = self.calculate_metrics(
            distmat, q_pids.numpy(), g_pids.numpy(), q_camids.numpy(), g_camids.numpy()
        )

        # 4. Logowanie wyników
        self.log("test_mAP", mAP)
        self.log("test_Rank1", r1)
        self.log("test_Rank5", r5)
        self.log("test_Rank10", r10)

        # Czyścimy pamięć po epoce
        self.test_outputs.clear()

    def calculate_metrics(
        self, distmat, q_pids, g_pids, q_camids, g_camids, max_rank=10
    ):
        """
        Oblicza Mean Average Precision (mAP) oraz Rank-N.

        Args:
            distmat: macierz odległości numpy [num_query x num_gallery]
            q_pids, g_pids: ID osób (numpy array)
            q_camids, g_camids: ID kamer (numpy array)
        """
        num_q, num_g = distmat.shape

        # Sortowanie indeksów galerii dla każdego zapytania (od najmniejszego dystansu)
        indices = np.argsort(distmat, axis=1)

        # Macierz trafień: 1 jeśli PID się zgadza, 0 w p.p.
        matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

        all_cmc = []
        all_ap = []

        for q_idx in range(num_q):
            # Pobieramy dane dla konkretnego zapytania
            q_pid = q_pids[q_idx]
            q_cam = q_camids[q_idx]

            # --- FILTROWANIE ---
            # Usuwamy hity z tej samej kamery dla tej samej osoby
            order = indices[q_idx]
            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_cam)
            keep = ~remove

            # Filtrujemy wektor trafień dla tego zapytania
            raw_cmc = matches[q_idx][keep]

            if not np.any(raw_cmc):
                # Jeśli po odfiltrowaniu tej samej kamery nie ma już tej osoby w galerii - pomijamy
                continue

            # --- Obliczanie AP (Average Precision) ---
            # Liczymy skumulowaną sumę trafień: [0, 1, 1, 2, ...]
            cmc_sum = raw_cmc.cumsum()
            # Precyzja w każdym punkcie: trafienia / pozycja
            precision = cmc_sum / (np.arange(len(raw_cmc)) + 1.0)
            # AP to średnia precyzja tylko w miejscach, gdzie faktycznie był trafiony PID
            ap = np.sum(precision * raw_cmc) / np.sum(raw_cmc)
            all_ap.append(ap)

            # --- Obliczanie CMC (Rank-N) ---
            # Skumulowane trafienie: [0, 1, 1, 1...] (wystarczy jedno trafienie na danej pozycji)
            cmc = cmc_sum.clip(max=1)
            all_cmc.append(cmc[:max_rank])

        mAP = np.mean(all_ap)
        rankn_results = np.mean(all_cmc, axis=0)

        return (
            mAP,
            rankn_results[0],  # Rank-1
            rankn_results[4],  # Rank-5
            rankn_results[9],  # Rank-10
        )

    def configure_optimizers(self):
        return Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr,
        )
