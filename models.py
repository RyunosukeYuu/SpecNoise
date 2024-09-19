import timm
import torch
from torch import nn
import torch.nn.functional as F

import leaf_audio_pytorch.frontend as torch_frontend
import torchvision.models as models
from torchlibrosa import SpecAugmentation
import warnings


class AudioClassifier(nn.Module):
    def __init__(self, frontend: nn.Module, resnet18: nn.Module):
        super(AudioClassifier, self).__init__()
        self.frontend = frontend
        self.resnet18 = resnet18

    def _preprocess_features(self, x):
        """
        Preprocesses the features for the encoder.

        Args:
            x (torch.Tensor): Input features tensor.

        Returns:
            torch.Tensor: Preprocessed features tensor.
        """
        x = x.unsqueeze(1)  # Add channel dimension
        x = x.repeat(1, 3, 1, 1)  # Repeat channels to match encoder input
        x = F.interpolate(x, size=(224, 224), mode='bilinear',
                          align_corners=False)  # Resize to match encoder input size
        x = x / 255.0  # Normalize the features
        return x

    def forward(self, x):
        x = self.frontend(x)
        x = self._preprocess_features(x)
        x = self.resnet18(x)
        return x


class FeatureFusionModule(nn.Module):
    def __init__(self, fusion_type='fixed'):
        super(FeatureFusionModule, self).__init__()
        self.fusion_type = fusion_type
        self.linear_spe = nn.Linear(512, 512)
        self.linear_sem = nn.Linear(512, 512)
        self.sigmoid = nn.Sigmoid()

    def forward(self, f_spe, f_sem):
        if self.fusion_type == 'fixed':
            # Fixed fusion
            h = torch.cat((f_spe, f_sem), dim=1)
        elif self.fusion_type == 'shared':
            # Shared fusion
            W_sem = self.sigmoid(self.linear_spe(f_spe))
            W_spe = self.sigmoid(self.linear_sem(f_sem))
            h_spe = f_spe * W_spe
            h_sem = f_sem * W_sem
            h = torch.cat((h_spe, h_sem), dim=1)
        elif self.fusion_type == 'sampling':
            # Sampling fusion
            f_prime_sem = f_sem * F.relu(self.linear_spe(f_spe))
            f_prime_spe = f_spe * F.relu(self.linear_sem(f_sem))
            s_sem = F.gumbel_softmax(f_prime_sem, tau=1, hard=True)
            s_spe = F.gumbel_softmax(f_prime_spe, tau=1, hard=True)
            h = torch.cat((f_prime_spe * s_spe, f_prime_sem * s_sem), dim=1)
        else:
            raise ValueError("Invalid fusion type")
        return h


class SSLNet(nn.Module):
    def __init__(self, branch1: nn.Module, branch2: nn.Module, fusion_type='fixed'):
        """
        Initializes the SSLNet model.

        Parameters:
        - branch1: The first branch of the model, expected to process one type of audio feature.
        - branch2: The second branch of the model, expected to process another type of audio feature.
        - fusion_type: Type of fusion to be used in the FeatureFusionModule (default is 'fixed').
        """
        super(SSLNet, self).__init__()
        self.branch1 = branch1
        self.branch2 = branch2
        self.resnet18_spe = fusion_resnet18()  # Pre-trained ResNet18 for spectral features
        self.resnet18_sem = fusion_resnet18()  # Pre-trained ResNet18 for semantic features
        self.fusion = FeatureFusionModule(fusion_type)  # Fusion module to combine features from both branches
        self.classifier = FCHeadNet(512 * 2 * 7 * 7, 20)  # Classification head with input size 1024 and output size 20

    def _process_branch(self, branch, x):
        """
        Process input x through the specified branch and the appropriate ResNet18 model.

        Parameters:
        - branch: The branch to process the input.
        - x: The input data.

        Returns:
        - Processed features.
        """
        if branch.frontend_type == 'spectral':
            f = branch(x)
            return self.resnet18_sem(f)
        elif branch.frontend_type == 'leaf':
            f = branch(x.unsqueeze(1))
            return self.resnet18_spe(f)
        else:
            f = branch(x)
            return self.resnet18_sem(f)

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Parameters:
        - x: Input audio data.

        Returns:
        - out: Output after classification.
        """
        # Process input through both branches
        f1 = self._process_branch(self.branch1, x)
        f2 = self._process_branch(self.branch2, x)

        # Fuse features from both branches
        fused_features = self.fusion(f1, f2)
        # Classify the fused features
        out = self.classifier(fused_features)
        return out


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


def fusion_resnet18():
    resnet18 = timm.create_model('resnet18.tv_in1k', pretrained=True, num_classes=0)
    model = nn.Sequential(*list(resnet18.children())[:-2])
    return model


def LEAF():
    front = torch_frontend.Leaf(sample_rate=16000, n_filters=64)
    for name, param in front.named_parameters():
        param.requires_grad = False
    return front


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