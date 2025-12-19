import torch
import torch.nn as nn
from torchvision import models


class FaceFeatureExtractor(nn.Module):
    def __init__(self, num_classes=256, embedding_dim=512):
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

        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        self.backbone.eval()

        features = self.backbone(x)

        embedding = self.custom_head(features)

        embedding_norm = torch.nn.functional.normalize(embedding, p=2, dim=1)

        return self.classifier(embedding_norm)
