import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchvision.models as models
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


class ImprovedDenseNet(nn.Module):
    """
    Improved DenseNet-121 Architecture for Digital Image Forgery Detection.
    Replaces default classification head with:
    Dense(40) -> ReLU -> Dense(20) -> ReLU -> Dense(2) -> Softmax
    """
    def __init__(self, pretrained=True):
        super(ImprovedDenseNet, self).__init__()
        
        if TORCHVISION_AVAILABLE:
            try:
                weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
                densenet = models.densenet121(weights=weights)
            except Exception:
                densenet = models.densenet121(pretrained=pretrained)
            
            self.features = densenet.features
            in_features = densenet.classifier.in_features
        else:
            self.features = self._build_custom_features()
            in_features = 1024

        self.bn = nn.BatchNorm2d(in_features)
        self.relu = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.dense1 = nn.Linear(in_features, 40)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.dense2 = nn.Linear(40, 20)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.dense_out = nn.Linear(20, 2)
        
    def _build_custom_features(self):
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Conv2d(256, 1024, kernel_size=1)
        )

    def forward(self, x):
        out = self.features(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.global_pool(out)
        
        out = torch.flatten(out, 1)
        
        out = self.dense1(out)
        out = self.relu1(out)
        
        out = self.dense2(out)
        out = self.relu2(out)
        
        logits = self.dense_out(out)
        probabilities = F.softmax(logits, dim=1)
        
        return logits, probabilities
