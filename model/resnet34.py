import torch.nn as nn
from torchvision import models

class ResNet34(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super(ResNet34, self).__init__()
        self.model = models.resnet34(pretrained=pretrained)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


