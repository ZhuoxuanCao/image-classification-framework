import torch.nn as nn
from torchvision import models

class ResNet50(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super(ResNet50, self).__init__()
        self.model = models.resnet50(pretrained=pretrained)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
