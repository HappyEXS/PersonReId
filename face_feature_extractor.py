import torch
import torch.nn as nn
from torchvision import models


class FaceFeatureExtractor(nn.Module):
    """
    Czysty model sieci neuronowej.
    Wejście: Tensor (Batch, 3, 50, 50)
    Wyjście: Tensor (Batch, 512)
    """

    def __init__(self, num_classes=256, embedding_dim=512):
        super(FaceFeatureExtractor, self).__init__()

        # Załadowanie backbone
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Modyfikacja głowicy (Bottleneck)
        self.in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Warstwa redukująca wymiar (Embedding Layer)
        self.bottleneck = nn.Sequential(
            nn.Linear(self.in_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),  # Opcjonalnie, zależy czy chcemy embeddingi po aktywacji
        )

        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        self.backbone.eval()

        features = self.backbone(x)

        embedding = self.bottleneck(features)

        embedding_norm = torch.nn.functional.normalize(embedding, p=2, dim=1)

        return self.classifier(embedding_norm)
