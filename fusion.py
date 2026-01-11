import torch
import torch.nn as nn
import torch.nn.functional as F

class Fusion(nn.Module):
    def __init__(self, appearance_net, face_net, num_classes=256, embedding_dim=512, fusion_weight=0.9):
        """
        Model łączący Appearance Branch (Inception) i Face Branch (ResNet).
        
        Args:
            appearance_net (nn.Module): Twój model AppearanceBranchInception.
            face_net (nn.Module): Twój model FaceFeatureExtractor (z notebooka).
            num_classes (int): Liczba klas (tożsamości).
            embedding_dim (int): Wymiar wyjściowy obu sieci (musi być taki sam, np. 512).
            fusion_weight (float): Parametr 'w' z równania (2) w artykule. 
                                   Wartość np. 0.9 lub 0.95 oznacza dominację sylwetki.
        """
        super(Fusion, self).__init__()
        
        self.appearance_branch = appearance_net
        self.face_branch = face_net
        self.fusion_weight = fusion_weight
        
        # Upewniamy się, że sub-modele nie mają własnych klasyfikatorów aktywnych w forward()
        # (zakładamy, że zwracają embeddingi w trybie eval lub zmodyfikowaliśmy ich forward)
        
        # Wspólny klasyfikator dla połączonego wektora
        self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)
        
        # Inicjalizacja klasyfikatora
        nn.init.normal_(self.classifier.weight, std=0.001)

    def forward(self, body_img, face_img):
        # 1. Ekstrakcja cech z sylwetki (Appearance Branch)
        # Oczekujemy znormalizowanego wektora (batch, 512)
        # Wymuszamy tryb eval dla branchy, żeby zwracały embeddingi, a nie logity
        # (chyba że zmodyfikowałeś forward w klasach bazowych)
        feat_app = self.appearance_branch(body_img)
        
        # 2. Ekstrakcja cech z twarzy (Face Branch)
        # Uwaga: FaceFeatureExtractor z Twojego notebooka zwraca 'classifier(embedding)',
        # więc musimy ominąć klasyfikator i pobrać 'custom_head'.
        # Poniższe obejście zadziała dla Twojego kodu z notebooka:
        # backbone_out = self.face_branch.backbone(face_img)
        # face_emb = self.face_branch.custom_head(backbone_out)
        # feat_face = face_emb # F.normalize(face_emb, p=2, dim=1)
        feat_face = self.face_branch(face_img)
        
        # 3. Fuzja (Weighted Sum) zgodnie z Eq. 2 w artykule [cite: 305]
        # f = w * f1 + (1-w) * f2
        fused_embedding = (self.fusion_weight * feat_app) + ((1 - self.fusion_weight) * feat_face)
        
        # Ponowna normalizacja po sumowaniu (częsta praktyka w ReID dla stabilności)
        fused_embedding = F.normalize(fused_embedding, p=2, dim=1)
        
        # 4. Klasyfikacja (tylko w treningu)
        if self.training:
            logits = self.classifier(fused_embedding)
            return logits
        else:
            return fused_embedding