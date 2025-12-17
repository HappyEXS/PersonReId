import os
import re
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class VCClothesDataset(Dataset):
    """
    Klasa Dataset dla zbioru VC-Clothes (Virtually Changing-Clothes).

    Dataset zawiera:
    - 512 tożsamości
    - 4 kamery (sceny)
    - 2-3 zestawy ubrań na osobę
    - Podział: 256 ID trening, 256 ID test
    """

    def __init__(self, root_dir, mode="train", transform=None, verbose=True):
        """
        Args:
            root_dir (str): Ścieżka do głównego katalogu datasetu.
            mode (str): 'train', 'query', lub 'gallery'.
            transform (callable, optional): Transformacje obrazu (np. z torchvision).
            verbose (bool): Czy wypisywać statystyki ładowania.
        """
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform

        # Określenie podkatalogu na podstawie trybu
        # Zgodnie z artykułem zbiór jest dzielony na train, query i gallery [cite: 194]
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

        # Parsowanie danych
        self.dataset = self._process_dir(self.data_dir)

        # Przygotowanie list do __getitem__
        self.img_paths = [x[0] for x in self.dataset]
        self.pids = [x[1] for x in self.dataset]
        self.camids = [x[2] for x in self.dataset]
        self.clothes_ids = [x[3] for x in self.dataset]

        # Mapowanie PID dla treningu (muszą być ciągłe od 0 do N-1 dla CrossEntropyLoss)
        if mode == "train":
            unique_pids = sorted(list(set(self.pids)))
            self.pid_map = {pid: i for i, pid in enumerate(unique_pids)}
            # Przemapowanie PID-ów w liście
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
        )  # Oczekuje formatu: PID-CAM-CLOTH_xxx.jpg

        dataset = []
        for img_path in img_paths:
            filename = os.path.basename(img_path)

            # Próba wyciągnięcia metadanych z nazwy pliku
            # Zakładamy format zgodny z konwencją ReID, uwzględniający clothes_id (t)
            match = pattern.search(filename)
            if not match:
                # Alternatywny parser jeśli nazwy są inne, np. proste splitowanie
                # Dostosuj tę sekcję w zależności od fizycznej nazwy plików w pobranym zbiorze
                continue

            pid, camid, clothes_id, _ = map(int, match.groups())

            # ReID zazwyczaj wymaga, aby camid zaczynało się od 0 lub 1
            # W VC-Clothes mamy 4 kamery [cite: 170]
            if pid == -1:
                continue  # Pomijanie obrazów "junk" (jeśli istnieją)

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

        if self.transform is not None:
            img = self.transform(img)

        # Zwracamy również clothes_id, co jest ważne dla badań nad "clothes inconsistency"
        return img, pid, camid, clothes_id


def build_transforms(is_train=True, height=256, width=128):
    """
    Standardowe transformacje dla ReID.
    Wymiary 256x128 są standardem w benchmarkach ReID[cite: 634].
    """
    # normalize = transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    # )

    if is_train:
        transform = transforms.Compose(
            [
                transforms.Resize((height, width)),
                transforms.ToTensor(),
                # normalize,
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize((height, width)),
                transforms.ToTensor(),
                # normalize,
            ]
        )

    return transform
