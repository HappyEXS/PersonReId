import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy


class BasicConv2d(nn.Module):
    """
    Podstawowy blok: Konwolucja + BatchNorm + ReLU.
    Używany wielokrotnie w modułach Inception.
    """

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class InceptionModule(nn.Module):
    """
    Klasyczny moduł Inception z 4 gałęziami:
    1. 1x1 conv
    2. 1x1 conv -> 3x3 conv
    3. 1x1 conv -> 5x5 conv
    4. 3x3 pool -> 1x1 conv
    """

    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(InceptionModule, self).__init__()

        # Gałąź 1: 1x1 conv
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        # Gałąź 2: 1x1 conv -> 3x3 conv
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1),
        )

        # Gałąź 3: 1x1 conv -> 5x5 conv
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2),
        )

        # Gałąź 4: 3x3 pool -> 1x1 conv
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1),
        )

    def forward(self, x):
        outputs = [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)]
        # Konkatenacja wyników wszystkich gałęzi wzdłuż wymiaru kanałów
        return torch.cat(outputs, 1)


class AppearanceBranch(nn.Module):
    def __init__(self, num_classes=256, embedding_dim=512):
        """
        Appearance Branch z architekturą Inception.
        Dostosowana do VC-Clothes (256 klas, embedding 512).
        """
        super(AppearanceBranch, self).__init__()

        # --- Część wstępna (Stem) ---
        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        # --- Bloki Inception (3a, 3b) ---
        self.inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        # --- Bloki Inception (4a - 4e) ---
        self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # --- Bloki Inception (5a, 5b) ---
        self.inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)

        # --- Warstwa globalnego poolingu ---
        # Wyjście z Inception 5b ma 1024 kanały
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # --- Embedding Head (Redukcja do 512, zgodnie z Fig. 7 artykułu) ---
        # 1024 (wyjście z Inception) -> 512
        self.bottleneck = nn.Sequential(
            nn.Linear(1024, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),  # Opcjonalnie, w zależności od tego, czy embedding ma być przed czy po aktywacji
        )

        # --- Klasyfikator ---
        self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)

        # Inicjalizacja wag
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats

                stddev = m.stddev if hasattr(m, "stddev") else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Stem
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)

        # Inception 3x
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        # Inception 4x
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        # Inception 5x
        x = self.inception5a(x)
        x = self.inception5b(x)

        # Global Avg Pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Bottleneck (Embedding)
        embedding = self.bottleneck(x)

        # Normalizacja (kluczowa dla ReID)
        embedding_norm = F.normalize(embedding, p=2, dim=1)

        return embedding_norm
