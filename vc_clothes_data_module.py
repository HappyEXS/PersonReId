import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
import torch

from vc_clothes_dataset_w_faces import VCClothesDatasetFaces, build_transforms


class VCClothesDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size=32, num_workers=0, val_split: float = 0.1):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split

    def setup(self, stage=None):
        train_dataset = VCClothesDatasetFaces(
            root_dir=self.root_dir, mode="train", transform=None
        )

        total_size = len(train_dataset)
        val_size = int(total_size * self.val_split)
        train_size = total_size - val_size

        self.train_ds, self.val_ds = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )

        self.query_ds = VCClothesDatasetFaces(
            root_dir=self.root_dir, mode="query", transform=None
        )

        self.gallery_ds = VCClothesDatasetFaces(
            root_dir=self.root_dir, mode="gallery", transform=None
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return [
            DataLoader(
                self.query_ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            ),
            DataLoader(
                self.gallery_ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            ),
        ]
