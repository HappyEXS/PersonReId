import torch.nn as nn
import torch.nn.functional as F


class Fusion(nn.Module):
    def __init__(
        self,
        appearance_net,
        face_net,
        num_classes=256,
        embedding_dim=512,
        fusion_weight=0.9,
    ):
        """
        Model łączący Appearance Branch (Inception) i Face Branch (ResNet).

        Args:
            appearance_net (nn.Module)  : Model AppearanceBranch.
            face_net (nn.Module)        : Model FaceFeatureExtractor.
            num_classes (int)           : Liczba klas.
            embedding_dim (int)         : Wymiar wyjściowy obu sieci.
            fusion_weight (float)       : Stosunek wagi appearance_net do face_net.
        """
        super(Fusion, self).__init__()

        self.appearance_branch = appearance_net
        self.face_branch = face_net
        self.fusion_weight = fusion_weight

        # Wspólny klasyfikator dla połączonego wektora
        self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)

    def forward(self, body_img, face_img):
        # 1. Ekstrakcja cech z sylwetki (Appearance Branch)
        feat_app = self.appearance_branch(body_img)

        # 2. Ekstrakcja cech z twarzy (Face Branch)
        feat_face = self.face_branch(face_img)

        # 3. Fuzja (Weighted Sum)
        # f = w * f1 + (1-w) * f2
        fused_embedding = (self.fusion_weight * feat_app) + (
            (1 - self.fusion_weight) * feat_face
        )

        # Ponowna normalizacja po sumowaniu
        fused_embedding = F.normalize(fused_embedding, p=2, dim=1)

        # 4. Klasyfikacja (tylko w treningu)
        if self.training:
            return self.classifier(fused_embedding)
        else:
            return fused_embedding
