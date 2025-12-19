import os
import glob
from torch.utils.data import Dataset
from PIL import Image


class FaceDataset(Dataset):
    """
    Dataset wczytujący gotowe, wycięte twarze z dysku.
    """

    def __init__(self, root_dir, mode="train", transform=None):
        if mode == "train":
            self.data_dir = root_dir + "/train_faces"
        elif mode == "query":
            self.data_dir = root_dir + "/query_faces"
        elif mode == "gallery":
            self.data_dir = root_dir + "/gallery_faces"
        else:
            raise ValueError("Mode musi być jednym z: 'train', 'query', 'gallery'")

        self.transform = transform

        # Pobranie wszystkich plików jpg
        # Zakładamy strukturę: root/mode/PID/image.jpg
        self.img_paths = glob.glob(os.path.join(self.data_dir, "*", "*.jpg"))
        print(f"Znaleziono {len(self.img_paths)} obrazów w {self.data_dir}")

        # Tworzenie mapowania klas (PIDów) na int
        # PID wyciągamy z nazwy folderu
        self.classes = sorted(
            list(set([os.path.basename(os.path.dirname(p)) for p in self.img_paths]))
        )
        print(f"Znaleziono {len(self.classes)} unikalnych klas (PIDów).")
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        print(
            f"Załadowano FaceDataset ({mode}): {len(self.img_paths)} obrazów, {len(self.classes)} klas."
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]

        # Ładowanie obrazu
        img = Image.open(path).convert("RGB")

        # Pobranie etykiety (PID)
        pid_str = os.path.basename(os.path.dirname(path))
        label = self.class_to_idx[pid_str]

        if self.transform:
            img = self.transform(img)

        return img, label
