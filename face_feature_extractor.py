import torch
import torch.nn as nn
from torchvision import models, transforms
from facenet_pytorch import MTCNN
import numpy as np


class FaceReIDModel(nn.Module):
    """
    Kompletny moduł detekcji i ekstrakcji cech twarzy dla ReID.
    Implementuje logikę opisaną w artykule 'When Person Re-identification Meets Changing Clothes'.
    """

    def __init__(self, device="cpu", embedding_dim=512):
        super(FaceReIDModel, self).__init__()
        self.device = device

        # 1. DETEKTOR TWARZY
        # Artykuł wykorzystuje Pyramidbox.
        # Tutaj używamy MTCNN jako nowoczesnego, łatwo dostępnego zamiennika.
        # keep_all=True pozwala wykryć wszystkie twarze, potem wybieramy najlepszą.
        self.detector = MTCNN(keep_all=True, device=device)

        # 2. EKSTRAKTOR CECH (Face Feature Extractor)
        # Artykuł wykorzystuje ResNet50 wytrenowany na zbiorze MS-1M.
        # Ładujemy pre-trenowany ResNet50 (domyślnie wagi ImageNet).
        # W środowisku produkcyjnym należy załadować tu wagi MS-Celeb-1M.
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Modyfikacja ostatniej warstwy, aby uzyskać wektor o wymiarze 512.
        # Wg rysunku 7 w artykule, cechy są mapowane do 512-wymiarowego wektora[cite: 277].
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, embedding_dim)

        # Przeniesienie modelu na urządzenie (CPU/GPU) i tryb ewaluacji
        self.to(device)
        self.eval()

        # 3. PREPROCESSING
        # Normalizacja standardowa dla ResNet
        self.normalizer = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        # Parametry z artykułu
        self.expansion_pixels = 15  # Rozszerzenie ramki
        self.confidence_threshold = 0.8  # Próg pewności
        self.target_size = (50, 50)  # Rozmiar docelowy

    def preprocess_face(self, image, box):
        """
        Przetwarza wycinek twarzy zgodnie z wytycznymi z artykułu.
        """
        # Konwersja boxa do int
        x1, y1, x2, y2 = [int(b) for b in box]
        w_img, h_img = image.size

        # "expand the detected bounding boxes of faces by 15 pixels in four directions"
        x1 = max(0, x1 - self.expansion_pixels)
        y1 = max(0, y1 - self.expansion_pixels)
        x2 = min(w_img, x2 + self.expansion_pixels)
        y2 = min(h_img, y2 + self.expansion_pixels)

        # Wycięcie twarzy
        face_crop = image.crop((x1, y1, x2, y2))

        # "All face images are then resized to the same size of 50x50 pixels"
        preprocess = transforms.Compose(
            [
                transforms.Resize(self.target_size),
                transforms.ToTensor(),
                self.normalizer,
            ]
        )

        return preprocess(face_crop).unsqueeze(0).to(self.device)

    def forward(self, image):
        """
        Główna metoda przetwarzająca obraz.
        Zwraca wektor cech (1, 512) lub tensor zerowy, jeśli twarzy nie wykryto.
        """
        # Detekcja twarzy
        # boxes: współrzędne ramek, probs: prawdopodobieństwa
        boxes, probs = self.detector.detect(image)

        best_face_tensor = None

        if boxes is not None:
            # Filtrowanie po pewności detekcji
            # "The confidence of the face bounding boxes is set to 0.8"
            valid_indices = [
                i for i, p in enumerate(probs) if p > self.confidence_threshold
            ]

            if valid_indices:
                # Jeśli wykryto wiele twarzy spełniających warunek, bierzemy tę o najwyższym p
                best_idx = valid_indices[np.argmax(probs[valid_indices])]
                best_box = boxes[best_idx]

                # Preprocessing wycinka
                best_face_tensor = self.preprocess_face(image, best_box)

        # Ekstrakcja cech
        if best_face_tensor is not None:
            with torch.no_grad():
                # f2(I) = F(G(I)) - ekstrakcja cech przez ResNet
                features = self.backbone(best_face_tensor)
                # Normalizacja L2 (częsta praktyka w ReID, choć nie wskazana explicite w tym akapicie)
                features = torch.nn.functional.normalize(features, p=2, dim=1)
                return features
        else:
            # Jeśli twarz jest niewykrywalna, zwracamy zera (system opiera się wtedy na holistic features)
            # "As for the images with undetectable faces, we use only the feature from a holistic feature extractor"
            return torch.zeros(1, 512).to(self.device)
