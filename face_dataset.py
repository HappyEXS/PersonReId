import os
import glob
from torch.utils.data import Dataset
from PIL import Image


class FaceDataset(Dataset):
    """
    Dataset wczytujący gotowe, wycięte twarze z dysku.
    """

    def __init__(self, root_dir, mode="train", transform=None):
        self.root = os.path.join(root_dir, mode)
        self.transform = transform

        # Pobranie wszystkich plików jpg
        # Zakładamy strukturę: root/mode/PID/image.jpg
        self.img_paths = glob.glob(os.path.join(self.root, "*", "*.jpg"))

        # Tworzenie mapowania klas (PIDów) na int
        # PID wyciągamy z nazwy folderu
        self.classes = sorted(
            list(set([os.path.basename(os.path.dirname(p)) for p in self.img_paths]))
        )
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        print(
            f"Załadowano FaceCropDataset ({mode}): {len(self.img_paths)} obrazów, {len(self.classes)} klas."
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
