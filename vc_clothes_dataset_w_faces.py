import os
import glob
import re
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from face_detector import FaceDetector

mapping = {
    "train": "train_faces",
    "gallery": "gallery_faces",
    "query": "query_faces"
}


class VCClothesDatasetFaces(Dataset):
    def __init__(self, root_dir, mode="train", transform=None, verbose=True, save_faces=False):

        self.root_dir = root_dir
        self.mode = mode
        self.transform_img = (
            build_transforms(if_normalize=True) if transform is None else transform
        )
        self.transform_face = build_transforms(if_normalize=True, for_faces=True)
        
        self.save_faces = save_faces

        if mode == "train":
            self.data_dir = root_dir + "/train"
        elif mode == "query":
            self.data_dir = root_dir + "/query"
        elif mode == "gallery":
            self.data_dir = root_dir + "/gallery"
        else:
            raise ValueError("Mode musi być jednym z: 'train', 'query', 'gallery'")

        if not os.path.exists(self.data_dir):
            raise RuntimeError(f"Katalog {self.data_dir} nie istnieje.")

        self.dataset = self._process_dir(self.data_dir)

        self.img_paths = [x[0] for x in self.dataset]
        self.pids = [x[1] for x in self.dataset]
        self.camids = [x[2] for x in self.dataset]
        self.clothes_ids = [x[3] for x in self.dataset]

        # Zmiana mapowania PID dla treningu (muszą być ciągłe od 0 do N-1)
        unique_pids = sorted(list(set(self.pids)))
        self.pid_map = {pid: i for i, pid in enumerate(unique_pids)}
        self.pids = [self.pid_map[pid] for pid in self.pids]

        if verbose:
            print(f"Załadowano zbiór VC-Clothes ({mode}):")
            print(f"  Liczba obrazów: {len(self.dataset)}")
            print(f"  Liczba unikalnych ID: {len(set(self.pids))}")
            print(f"  Liczba kamer: {len(set(self.camids))}")
            print(f"  Liczba unikalnych ubrań: {len(set(self.clothes_ids))}")

    def _process_dir(self, dir_path):
        img_paths = glob.glob(os.path.join(dir_path, "*.jpg")) + glob.glob(
            os.path.join(dir_path, "*.png")
        )

        pattern = re.compile(
            r"([-\d]+)-(\d+)-(\d+)-(\d+)"
        )  # Format: PID-CAM-CLOTH_xxx.jpg

        dataset = []
        for img_path in img_paths:
            filename = os.path.basename(img_path)

            match = pattern.search(filename)
            if not match:
                continue

            pid, camid, clothes_id, _ = map(int, match.groups())

            if pid == -1:
                continue  # Pomijanie obrazów "junk"

            dataset.append((img_path, pid, camid, clothes_id))

        return dataset

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        path = self.img_paths[index]
        pid = self.pids[index]
        camid = self.camids[index]
        clothes_id = self.clothes_ids[index]

        img = Image.open(path).convert("RGB")

        if self.transform_img is not None:
            img = self.transform_img(img)

        pid = torch.tensor(pid, dtype=torch.long)
        camid = torch.tensor(camid, dtype=torch.long)
        clothes_id = torch.tensor(clothes_id, dtype=torch.long)

        if self.save_faces:
            face = torch.zeros((3, 50, 50), dtype=torch.float)
            return img, pid, camid, clothes_id, face, path
        else:
            face_path = path
            for old, new in mapping.items():
                face_path = face_path.replace(old, new)
            face = Image.open(face_path).convert("RGB")
            face = self.transform_face(face)
            return img, pid, camid, clothes_id, face


def build_transforms(if_normalize=False, for_faces=False):
    height=256
    width=128
    if for_faces: 
        height=50
        width=50

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    if if_normalize:
        transform = transforms.Compose(
            [
                transforms.Resize((height, width)),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize((height, width)),
                transforms.ToTensor(),
            ]
        )

    return transform
