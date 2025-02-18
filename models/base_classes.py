import torch
import torch.nn as nn
from torchvision import models
import math

class VGG(nn.Module):
    def __init__(self, input_channels=3):
        super(VGG, self).__init__()

        self.features = nn.Sequential(
            # Conv1
            nn.Conv2d(input_channels, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv2
            nn.Conv2d(16, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),

            # Conv3
            nn.Conv2d(32, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv4
            nn.Conv2d(32, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv5
            nn.Conv2d(32, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),

            # Conv6
            nn.Conv2d(32, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3)
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 6 * 6, 8)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class BaseModel(nn.Module):
    def __init__(self, params=None):
        super(BaseModel, self).__init__()
        if params is None:
            params = {}
        self.CHANNELS = params.get('channels', 1)
        self.NUM_CLASSES = params.get('num_classes', 1)
        self.DROP_PROB = params.get('drop_prob', 0.25)
        self.INDUCING_POINTS = params.get('inducing_points', 32)
        self.MODEL_TYPE = params.get('model_type', 'resnet18')
        self.ATTENTION_HIDDEN_DIM = params.get('attention_hidden_dim', 512)

        # Choose feature extractor based on model_type
        if self.MODEL_TYPE == 'resnet18':
            self.features_extractor = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            # self.features_extractor = models.resnet18(weights=None)
            self.features_extractor.fc = nn.Identity()
            self.features_extractor.conv1 = nn.Conv2d(self.CHANNELS, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.feature_dim = 512  # ResNet18 feature dimension

        elif self.MODEL_TYPE == 'vgg16':
            self.features_extractor = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            self.features_extractor.classifier = nn.Identity()  # Remove the final classification layer
            self.features_extractor.features[0] = nn.Conv2d(self.CHANNELS, 64, kernel_size=3, stride=1, padding=1)
            self.feature_dim = 512 * 7 * 7  # VGG16 feature dimension (after flattening)

        elif self.MODEL_TYPE == 'vgg':
            self.features_extractor = VGG(input_channels=self.CHANNELS)
            self.feature_dim = 8 # VGG feature dimension (after flattening)

        elif self.MODEL_TYPE.startswith('resnext'):
            if self.MODEL_TYPE == 'resnext50':
                self.features_extractor = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT)
            elif self.MODEL_TYPE == 'resnext101':
                self.features_extractor = models.resnext101_32x8d(weights=models.ResNeXt101_32X8D_Weights.DEFAULT)
            else:
                raise ValueError(f"Unsupported ResNeXt variant: {self.MODEL_TYPE}")
            
            self.features_extractor.conv1 = nn.Conv2d(self.CHANNELS, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.features_extractor.fc = nn.Identity()
            self.feature_dim = 2048  # Feature dimension for ResNeXt

        else:
            raise ValueError(f"Unsupported model_type: {self.MODEL_TYPE}")

    def forward(self, bags):
        pass
