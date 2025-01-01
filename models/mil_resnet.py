import torch
import torch.nn as nn
from torchvision import models
from layers import attention as AttentionLayer, gaussian_process as GPModel
import gpytorch

class BaseModel(nn.Module):
    def __init__(self, params=None):
        super(BaseModel, self).__init__()
        if params is None:
            params = {}
        self.CHANNELS = params.get('channels', 1)
        self.NUM_CLASSES = params.get('num_classes', 1)
        self.DROP_PROB = params.get('drop_prob', 0.25)
        self.INDUCING_POINTS = params.get('inducing_points', 32)

        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet.conv1 = nn.Conv2d(in_channels=self.CHANNELS, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Identity()

    def forward(self, bags):
        pass

class CNN_ATT_GP(BaseModel):
    def __init__(self, params=None):
        super(CNN_ATT_GP, self).__init__(params)
        self.attention_hidden_dim = params.get('attention_hidden_dim', 512)
        self.attention = AttentionLayer.AttentionLayer(input_dim=self.attention_hidden_dim, hidden_dim=self.attention_hidden_dim)
        self.classifier = nn.Linear(self.attention_hidden_dim, self.NUM_CLASSES)
        self.dropout = nn.Dropout(self.DROP_PROB)
        self.linear_to_8 = nn.Linear(512, self.attention_hidden_dim)

        inducing_points = torch.randn(self.INDUCING_POINTS, 1, dtype=torch.float32)
        self.gp_layer = GPModel.GPModel(inducing_points=inducing_points)

        # Store parameters from the dictionary
        self.projection_location = params.get('projection_location', 'after_resnet')
        self.projection_hidden_dim = params.get('projection_hidden_dim', 256)
        self.projection_output_dim = params.get('projection_output_dim', 128)
        self.projection_input_dim =  self.attention_hidden_dim

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
        features = self.linear_to_8(features)

        attended_features, attended_weights = self.attention(features)
        attended_features_reshaped = attended_features.view(batch_size, -1)

        attended_features_reshaped = self.classifier(attended_features_reshaped)
        gp_output = self.gp_layer(attended_features_reshaped)

        # Combine features based on the specified projection location
        if self.projection_location == 'after_resnet':
            projection_output = self.projection_head(features.view(batch_size * num_instances, -1))

        elif self.projection_location == 'after_attention':
            projection_output = self.projection_head(attended_features_reshaped)

        else:
            raise ValueError("Invalid projection location. Choose from 'after_resnet', 'after_attention', or 'after_gp'.")

        return gp_output, attended_weights, None, projection_output

class CNN_GP_ATT(BaseModel):
    def __init__(self, params=None):
        super(CNN_GP_ATT, self).__init__(params)
        self.attention_hidden_dim = params.get('attention_hidden_dim', 512)
        self.attention = AttentionLayer.AttentionLayer(input_dim=self.attention_hidden_dim + 1, hidden_dim=self.attention_hidden_dim + 1)
        self.classifier = nn.Linear(self.attention_hidden_dim + 1, self.NUM_CLASSES)
        self.dropout = nn.Dropout(self.DROP_PROB)

        inducing_points = torch.randn(self.INDUCING_POINTS, self.attention_hidden_dim)
        self.gp_layer = GPModel.GPModel(inducing_points=inducing_points)

        # Store parameters from the dictionary
        self.projection_location = params.get('projection_location', 'after_resnet')
        self.projection_hidden_dim = params.get('projection_hidden_dim', 256)
        self.projection_output_dim = params.get('projection_output_dim', 128)
        self.projection_input_dim = self.attention_hidden_dim + 1 if self.projection_location == 'after_gp' else self.attention_hidden_dim

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

        gp_output = self.gp_layer(features.view(batch_size, num_instances, -1))
        gp_mean = gp_output.mean.view(batch_size, num_instances,-1)
        combine_features = torch.cat((features, gp_mean), dim=2)
        # combine_features = self.dropout(combine_features)

        attended_features, attended_weights = self.attention(combine_features)
        attended_features_reshaped = attended_features.view(batch_size, -1)
        outputs = self.classifier(self.dropout(attended_features_reshaped))

        if self.projection_location == 'after_resnet':
            projection_output = self.projection_head(features.view(batch_size * num_instances, -1))
        elif self.projection_location == 'after_attention':
            projection_output = self.projection_head(outputs)
        elif self.projection_location == 'after_gp':
            projection_output = self.projection_head(combine_features.view(batch_size * num_instances, -1))
        else:
            raise ValueError("Invalid projection location. Choose from 'after_resnet', 'after_attention', or 'after_gp'.")

        return outputs, attended_weights, gp_output, projection_output

class SupConResnet(BaseModel):
    def __init__(self, params=None):
        super(SupConResnet, self).__init__(params)
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet.conv1 = nn.Conv2d(in_channels=self.CHANNELS, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Identity()

        # Store parameters from the dictionary
        self.projection_location = params.get('projection_location', 'after_resnet')
        self.projection_hidden_dim = params.get('projection_hidden_dim', 256)
        self.projection_output_dim = params.get('projection_output_dim', 128)
        self.projection_input_dim = 512 if self.projection_location == 'after_resnet' else 513

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
        features = features.view(batch_size, num_instances, -1)

        if self.projection_location != 'after_resnet':
            raise ValueError("Invalid projection location. Choose from 'after_resnet'.")

        projection_output = self.projection_head(features.view(batch_size * num_instances, -1))

        return projection_output, features

class LinearClassifier(BaseModel):
    def __init__(self, params=None):
        super(LinearClassifier, self).__init__(params)
        self.attention_hidden_dim = params.get('attention_hidden_dim', 512)
        self.classifier = nn.Linear(self.attention_hidden_dim + 1, self.NUM_CLASSES)

        self.attention = AttentionLayer.AttentionLayer(input_dim=self.attention_hidden_dim, hidden_dim=self.attention_hidden_dim)
        self.dropout = nn.Dropout(self.DROP_PROB)

        inducing_points = torch.randn(self.INDUCING_POINTS, self.attention_hidden_dim)
        self.gp_layer = GPModel.GPModel(inducing_points=inducing_points)

    def forward(self, features):
        attended_features, attended_weights = self.attention(features)
        attended_features_reshaped = attended_features.view(features.size(0), -1)

        gp_output = self.gp_layer(attended_features_reshaped)
        gp_mean = gp_output.mean.view(features.size(0), -1)
        combine_features = torch.cat((attended_features_reshaped, gp_mean), dim=1)

        return self.classifier(combine_features)


class DKLModel(gpytorch.Module):
    def __init__(self, grid_bounds=(-10., 10.), params=None):
        super(DKLModel, self).__init__()

        self.CHANNELS = params.get('channels', 1)
        self.NUM_CLASSES = params.get('num_classes', 1)
        self.DROP_PROB = params.get('drop_prob', 0.25)
        self.INDUCING_POINTS = params.get('inducing_points', 32)
        self.ATTENTION_HIDDEN_DIM = params.get('attention_hidden_dim', 512)

        self.feature_extractor = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.feature_extractor.conv1 = nn.Conv2d(self.CHANNELS, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.feature_extractor.fc = nn.Identity()
        self.gp_layer = GPModel.GaussianProcessLayer(num_dim=self.ATTENTION_HIDDEN_DIM, grid_bounds=grid_bounds)
        self.grid_bounds = grid_bounds
        self.num_dim = self.ATTENTION_HIDDEN_DIM

        # This module will scale the NN features so that they're nice values
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(self.grid_bounds[0], self.grid_bounds[1])

        self.attention = AttentionLayer.AttentionLayer(input_dim=self.ATTENTION_HIDDEN_DIM, hidden_dim=self.ATTENTION_HIDDEN_DIM)
        # self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        if self.CHANNELS == 1:
            batch_size, num_instances, c, h, w = x.size()
        else:
            batch_size, num_instances, h, w, c = x.size()

        x = x.view(batch_size * num_instances, c, h, w)
        features = self.feature_extractor(x)
        # features = self.dropout(features)

        features = features.view(batch_size, num_instances, -1)
        features, _ = self.attention(features)
        features = self.scale_to_bounds(features)

        assert features.size(-1) == self.num_dim, f"Features should be {self.num_dim} dimensions, got {features.shape}"
        features = features.transpose(-1, -2).unsqueeze(-1)

        assert features.size(-1) == 1, f"Features should be 1 dimension after unsqueeze, got {features.shape}"
        res = self.gp_layer(features)

        return res

