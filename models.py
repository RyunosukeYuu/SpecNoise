import timm
import torch
from torch import nn
import torch.nn.functional as F

import leaf_audio_pytorch.frontend as torch_frontend
import torchvision.models as models
from torchlibrosa import SpecAugmentation
import warnings


class ResNet18FeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNet18FeatureExtractor, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18 = nn.Sequential(*list(self.resnet18.children())[:-1])
        for param in self.resnet18.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.resnet18(x)
        features = features.view(features.size(0), -1)
        return features


class Singlebranchspectralclassifier(nn.Module):
    def __init__(self, num_classes):
        super(Singlebranchspectralclassifier, self).__init__()
        self.resnet18_feature_extractor = ResNet18FeatureExtractor()
        self.mlp = FCHeadNet(num_features=512, num_classes=num_classes)

    def forward(self, x):
        features = self.resnet18_feature_extractor(x)
        output = self.mlp(features)
        return output


class FCHeadNet(nn.Module):
    def __init__(self, num_features, num_classes):
        super(FCHeadNet, self).__init__()

        self.fc1 = nn.Linear(num_features, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)

        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        return x
