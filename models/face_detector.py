from facenet_pytorch import MTCNN
from torchvision import transforms
from PIL import Image
import numpy as np


class FaceDetector:
    def __init__(self, device="cpu", confidence_threshold=0.8):
        self.device = device
        self.confidence_threshold = confidence_threshold

        self.mtcnn = MTCNN(keep_all=False, device=device)

        self.expansion_pixels = 15
        self.target_size = (50, 50)

        self.transform = transforms.Compose(
            [
                transforms.Resize(self.target_size),
                transforms.ToTensor(),
            ]
        )

    def get_face_tensor(self, image: Image):
        boxes, probs = self.mtcnn.detect(image)

        if boxes is None:
            return None

        # Filtrowanie po progu
        valid_indices = [
            i for i, p in enumerate(probs) if p > self.confidence_threshold
        ]

        if not valid_indices:
            return None

        # Wybór najlepszej twarzy
        best_idx = valid_indices[np.argmax(probs[valid_indices])]
        box = boxes[best_idx]

        # Wycinanie twarzy
        x1, y1, x2, y2 = [int(b) for b in box]
        w_img, h_img = image.size

        x1 = max(0, x1 - self.expansion_pixels)
        y1 = max(0, y1 - self.expansion_pixels)
        x2 = min(w_img, x2 + self.expansion_pixels)
        y2 = min(h_img, y2 + self.expansion_pixels)

        face_crop = image.crop((x1, y1, x2, y2))
        tensor = self.transform(face_crop)

        return tensor.to(self.device)
