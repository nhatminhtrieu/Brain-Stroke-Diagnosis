import torch
import torch.nn as nn
from torchvision import models
import math

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
            self.features_extractor.fc = nn.Identity()
            self.features_extractor.conv1 = nn.Conv2d(self.CHANNELS, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.feature_dim = 512  # ResNet18 feature dimension

        elif self.MODEL_TYPE == 'vgg16':
            self.features_extractor = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            self.features_extractor.classifier = nn.Identity()  # Remove the final classification layer
            self.features_extractor.features[0] = nn.Conv2d(self.CHANNELS, 64, kernel_size=3, stride=1, padding=1)
            self.feature_dim = 512 * 7 * 7  # VGG16 feature dimension (after flattening)

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


class SequenceAwareModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            bidirectional=True, batch_first=True)
        self.position_encoder = PositionalEncoding(hidden_dim * 2)

    def forward(self, x):
        seq_features, _ = self.lstm(x)
        return self.position_encoder(seq_features)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class MultiHeadFeatureRefiner(nn.Module):
    def __init__(self, in_dim=512, num_heads=8, expansion=2):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=in_dim,
            num_heads=num_heads,
            dropout=0.25,
            batch_first=True
        )
        self.conv_branch = nn.Sequential(
            nn.Conv1d(in_dim, in_dim*expansion, 3, padding=1),
            nn.GELU(),
            nn.InstanceNorm1d(in_dim*expansion),
            nn.Conv1d(in_dim*expansion, in_dim, 1)
        )
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        conv_out = self.conv_branch(x.permute(0,2,1)).permute(0,2,1)
        return self.norm(attn_out + conv_out + x)
