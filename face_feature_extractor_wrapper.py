import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Zakładam, że klasy FaceReIDModel i VCClothesDataset są zaimportowane lub zdefiniowane powyżej
# from face_model import FaceReIDModel
# from dataset import VCClothesDataset, build_transforms


class FaceReIDTrainWrapper(nn.Module):
    """
    Wrapper na FaceReIDModel dodający głowicę klasyfikacyjną do treningu.
    Pozwala trenować backbone (ResNet) przy użyciu CrossEntropyLoss.
    """

    def __init__(self, original_model, num_classes):
        super(FaceReIDTrainWrapper, self).__init__()
        self.device = original_model.device

        # Przejmujemy backbone i preprocessing z oryginalnego modelu
        self.backbone = original_model.backbone
        self.detector = original_model.detector
        self.preprocess_face = original_model.preprocess_face
        self.confidence_threshold = original_model.confidence_threshold

        # Głowica klasyfikacyjna (Classifier Head)
        # Wejście: 512 (embedding), Wyjście: num_classes (liczba tożsamości w zbiorze treningowym)
        # Zwykle w ReID stosuje się tutaj BatchNorm, ale dla uproszczenia dajemy Linear
        self.classifier = nn.Linear(512, num_classes)

        # Przeniesienie na urządzenie
        self.to(self.device)

    def forward(self, images):
        """
        Przetwarza batch obrazów PIL, wycina twarze i zwraca logity do klasyfikacji.
        """
        face_tensors = []
        valid_mask = []  # Maska wskazująca, czy udało się wykryć twarz

        # KROK 1: Detekcja i Preprocessing (na liście obrazów)
        # Niestety MTCNN i preprocessing PIL działają na CPU i pojedynczych obrazach
        # W pętli treningowej "online" musimy to obsłużyć iteracyjnie.

        for img in images:
            # Używamy logiki detekcji z oryginalnego modelu
            # Uwaga: w treningu zależy nam na czasie, można tu użyć detekcji na batchu jeśli MTCNN na to pozwala
            boxes, probs = self.detector.detect(img)

            best_face = None
            if boxes is not None:
                valid_indices = [
                    i for i, p in enumerate(probs) if p > self.confidence_threshold
                ]
                if valid_indices:
                    best_idx = valid_indices[np.argmax(probs[valid_indices])]
                    best_box = boxes[best_idx]
                    best_face = self.preprocess_face(
                        img, best_box
                    )  # Zwraca tensor (1, 3, 50, 50) na GPU

            if best_face is not None:
                face_tensors.append(best_face)
                valid_mask.append(True)
            else:
                # Jeśli nie wykryto twarzy, dodajemy dummy tensor (żeby zachować ciągłość batcha w listach)
                # Ale oznaczymy go jako invalid
                face_tensors.append(torch.zeros(1, 3, 50, 50).to(self.device))
                valid_mask.append(False)

        # Złączenie w jeden batch tensorów
        if not face_tensors:
            return None, None

        batch_input = torch.cat(face_tensors, dim=0)  # (B, 3, 50, 50)

        # KROK 2: Ekstrakcja cech (Backbone) - TUTAJ MAMY GRADIENTY
        embeddings = self.backbone(batch_input)  # (B, 512)

        # Normalizacja embeddingów (częsta praktyka w ReID)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        # KROK 3: Klasyfikacja
        logits = self.classifier(embeddings)  # (B, num_classes)

        return logits, torch.tensor(valid_mask, device=self.device)
