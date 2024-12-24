import torch
import torch.nn as nn
from torchvision import models
from utils import attention as AttentionLayer, gaussian_process as GPModel

class MILResNet18(nn.Module):
    def __init__(self, params=None):
        super(MILResNet18, self).__init__()

        # Default parameters if none are provided
        if params is None:
            params = {}
        self.CHANNELS = params.get('channels', 1)

        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet.conv1 = nn.Conv2d(in_channels=self.CHANNELS, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Identity()

        self.attention = AttentionLayer.AttentionLayer(input_dim=512, hidden_dim=512)
        self.classifier = nn.Linear(512 + 1, 1)
        self.attention_classifier = nn.Linear(512, 1)
        self.dropout = nn.Dropout(0.4)

        inducing_points = torch.randn(32, 512)
        self.gp_layer = GPModel.GPModel(inducing_points=inducing_points)

        # Store parameters from the dictionary
        self.projection_location = params.get('projection_location', 'after_resnet')
        self.projection_hidden_dim = params.get('projection_hidden_dim', 256)
        self.projection_output_dim = params.get('projection_output_dim', 128)
        self.projection_input_dim = 513 if self.projection_location == 'after_gp' else 512
        self.projection_head = nn.Sequential(
            nn.Linear(self.projection_input_dim, self.projection_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.projection_hidden_dim, self.projection_output_dim)
        )

    def forward(self, bags):
        if self.CHANNELS == 1:
            batch_size, num_instances, c, h, w = bags.size()
        else:
            batch_size, num_instances, h, w, c = bags.size()

        bags_flattened = bags.view(batch_size * num_instances, c, h, w)

        features = self.resnet(bags_flattened)
        features = self.dropout(features)
        features = features.view(batch_size, num_instances, -1)

        attended_features, attended_weights = self.attention(features)
        attended_features_reshaped = attended_features.view(batch_size, -1)

        gp_output = self.gp_layer(attended_features_reshaped)
        gp_mean = gp_output.mean.view(batch_size, -1)
        combine_features = torch.cat((attended_features_reshaped, gp_mean), dim=1)

        # Combine features based on the specified projection location
        if self.projection_location == 'after_resnet':
            projection_output = self.projection_head(features.view(batch_size * num_instances, -1))

        elif self.projection_location == 'after_attention':
            projection_output = self.projection_head(attended_features_reshaped)

        elif self.projection_location == 'after_gp':
            projection_output = self.projection_head(combine_features)

        else:
            raise ValueError("Invalid projection location. Choose from 'after_resnet', 'after_attention', or 'after_gp'.")

        combine_features = self.dropout(combine_features)

        outputs = torch.sigmoid(self.classifier(combine_features))
        att_outputs = torch.sigmoid(self.attention_classifier(attended_features_reshaped))

        return outputs, att_outputs, attended_weights, gp_output, projection_output