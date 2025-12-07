import torch
from facenet_pytorch import MTCNN
from torchvision import transforms
from PIL import Image
import numpy as np


class FaceDetector:
    """
    Moduł odpowiedzialny wyłącznie za detekcję, wycinanie i preprocessing twarzy.
    Realizuje logikę z artykułu: padding 15px, resize do 50x50.
    """

    def __init__(self, device="cpu", confidence_threshold=0.8):
        self.device = device
        self.confidence_threshold = confidence_threshold

        # Detektor MTCNN
        self.mtcnn = MTCNN(keep_all=True, device=device)

        # Parametry z artykułu
        self.expansion_pixels = 15
        self.target_size = (50, 50)

        # Transformacje końcowe (norma dla ResNet)
        self.transform = transforms.Compose(
            [
                transforms.Resize(self.target_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def get_face_tensor(self, image):
        """
        Pobiera obraz PIL, wykrywa twarz i zwraca tensor (1, 3, 50, 50) na GPU.
        Zwraca None, jeśli nie znaleziono twarzy o wystarczającej pewności.
        """
        # Detekcja
        boxes, probs = self.mtcnn.detect(image)

        if boxes is None:
            return None

        # Filtrowanie po progu pewności
        valid_indices = [
            i for i, p in enumerate(probs) if p > self.confidence_threshold
        ]

        if not valid_indices:
            return None

        # Wybór najlepszej twarzy
        best_idx = valid_indices[np.argmax(probs[valid_indices])]
        box = boxes[best_idx]

        # Logika wycinania z paddingiem
        x1, y1, x2, y2 = [int(b) for b in box]
        w_img, h_img = image.size

        x1 = max(0, x1 - self.expansion_pixels)
        y1 = max(0, y1 - self.expansion_pixels)
        x2 = min(w_img, x2 + self.expansion_pixels)
        y2 = min(h_img, y2 + self.expansion_pixels)

        # Crop i Transformacja
        face_crop = image.crop((x1, y1, x2, y2))
        tensor = self.transform(face_crop)

        return tensor.unsqueeze(0).to(
            self.device
        )  # Dodanie wymiaru batcha (1, C, H, W)
