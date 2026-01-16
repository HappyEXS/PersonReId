import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class FaceFeatureExtractor(nn.Module):
    """
    Face Feature Extractor z architekturą ResNet.
    """

    def __init__(self, embedding_dim=512):
        super(FaceFeatureExtractor, self).__init__()

        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.custom_head = nn.Sequential(
            nn.Linear(self.in_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        self.backbone.eval()

        # Backbone
        features = self.backbone(x)

        # Embedding
        embedding = self.custom_head(features)

        # Normalizacja
        embedding_norm = F.normalize(embedding, p=2, dim=1)

        return embedding_norm
