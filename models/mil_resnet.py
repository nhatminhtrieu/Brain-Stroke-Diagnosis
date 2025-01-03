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

class CNN_ATT_GP(nn.Module):
    def __init__(self, params=None):
        super(CNN_ATT_GP, self).__init__()
        self.params = params or {}
        self.CHANNELS = self.params.get('channels', 1)
        self.NUM_CLASSES = self.params.get('num_classes', 1)
        self.DROP_PROB = self.params.get('drop_prob', 0.25)
        self.INDUCING_POINTS = self.params.get('inducing_points', 32)
        self.attention_hidden_dim = self.params.get('attention_hidden_dim', 512)
        self.gp_type = self.params.get('gp_model', 'multi_task')  # Default to multi-task GP

        # ResNet backbone
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet.conv1 = nn.Conv2d(self.CHANNELS, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Identity()

        # Attention layer
        self.attention = AttentionLayer.AttentionLayer(input_dim=self.attention_hidden_dim, hidden_dim=self.attention_hidden_dim)

        # Linear layer to map ResNet features to attention input dimension
        self.linear_to_dim = nn.Linear(512, self.attention_hidden_dim)
        self.fc = nn.Linear(self.attention_hidden_dim, self.NUM_CLASSES)

        self.dropout = nn.Dropout(self.DROP_PROB)

        # Initialize GP model based on gp_type
        if self.gp_type == 'multi_task':
            self.num_latents = self.params.get('num_latents', 3)  # Number of latent functions for multi-task GP
            self.num_tasks = self.params.get('num_tasks', 4)  # Number of tasks for multi-task GP
            self.gp_layer = GPModel.MultitaskGPModel(num_latents=self.num_latents, num_tasks=self.num_tasks)
        elif self.gp_type == 'single_task':
            self.gp_layer = GPModel.SingletaskGPModel(inducing_points=torch.randn(self.INDUCING_POINTS, 1))
        else:
            raise ValueError("Invalid gp_model. Choose from 'single_task' or 'multi_task'.")

        # Projection head (optional)
        self.projection_location = self.params.get('projection_location', 'after_resnet')
        self.projection_hidden_dim = self.params.get('projection_hidden_dim', 256)
        self.projection_output_dim = self.params.get('projection_output_dim', 128)
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

        features = self.dropout(self.resnet(bags_flattened)).view(batch_size, num_instances, -1)
        features = self.linear_to_dim(features)

        attended_features, attended_weights = self.attention(features)
        attended_features_reshaped = attended_features.view(batch_size, -1)

        if self.gp_type == 'single_task':
            gp_output = self.gp_layer(self.fc(attended_features_reshaped))
        else:
            gp_output = self.gp_layer(attended_features_reshaped)

        if self.projection_location == 'after_resnet':
            projection_output = self.projection_head(features.view(batch_size * num_instances, -1))
        elif self.projection_location == 'after_attention':
            projection_output = self.projection_head(attended_features_reshaped)
        else:
            combined_features = torch.cat((attended_features_reshaped, gp_output.mean.unsqueeze(-1)), dim=1)
            projection_output = self.projection_head(combined_features)

        return gp_output, attended_weights, None, projection_output

class CNN_GP_ATT(BaseModel):
    def __init__(self, params=None):
        super(CNN_GP_ATT, self).__init__(params)
        self.attention_hidden_dim = self.params.get('attention_hidden_dim', 512)
        self.attention = AttentionLayer.AttentionLayer(input_dim=self.attention_hidden_dim + 1, hidden_dim=self.attention_hidden_dim + 1)
        self.classifier = nn.Linear(self.attention_hidden_dim + 1, self.NUM_CLASSES)
        self.dropout = nn.Dropout(self.DROP_PROB)

        self.gp_layer = GPModel.GPModel(inducing_points=torch.randn(self.INDUCING_POINTS, self.attention_hidden_dim))

        self.projection_location = self.params.get('projection_location', 'after_resnet')
        self.projection_hidden_dim = self.params.get('projection_hidden_dim', 256)
        self.projection_output_dim = self.params.get('projection_output_dim', 128)
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

        features = self.dropout(self.resnet(bags_flattened)).view(batch_size, num_instances, -1)
        gp_output = self.gp_layer(features.view(batch_size, num_instances, -1))
        gp_mean = gp_output.mean.view(batch_size, num_instances, -1)
        combine_features = torch.cat((features, gp_mean), dim=2)

        attended_features, attended_weights = self.attention(combine_features)
        attended_features_reshaped = attended_features.view(batch_size, -1)
        outputs = self.classifier(self.dropout(attended_features_reshaped))

        projection_output = self.projection_head(
            features.view(batch_size * num_instances, -1)) if self.projection_location == 'after_resnet' else \
            self.projection_head(outputs) if self.projection_location == 'after_attention' else \
                self.projection_head(combine_features.view(batch_size * num_instances, -1))

        return outputs, attended_weights, gp_output, projection_output

class SupConResnet(BaseModel):
    def __init__(self, params=None):
        super(SupConResnet, self).__init__(params)
        self.projection_location = self.params.get('projection_location', 'after_resnet')
        self.projection_hidden_dim = self.params.get('projection_hidden_dim', 256)
        self.projection_output_dim = self.params.get('projection_output_dim', 128)
        self.projection_input_dim = 512 if self.projection_location == 'after_resnet' else 513

        self.projection_head = nn.Sequential(
            nn.Linear(self.projection_input_dim, self.projection_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.projection_hidden_dim, self.projection_output_dim)
        )

    def forward(self, bags):
        batch_size, num_instances, c, h, w = bags.size()
        bags_flattened = bags.view(batch_size * num_instances, c, h, w)

        features = self.resnet(bags_flattened).view(batch_size, num_instances, -1)
        if self.projection_location != 'after_resnet':
            raise ValueError("Invalid projection location. Choose from 'after_resnet'.")

        return self.projection_head(features.view(batch_size * num_instances, -1)), features

class LinearClassifier(BaseModel):
    def __init__(self, params=None):
        super(LinearClassifier, self).__init__(params)
        self.attention_hidden_dim = self.params.get('attention_hidden_dim', 512)
        self.classifier = nn.Linear(self.attention_hidden_dim + 1, self.NUM_CLASSES)

        self.attention = AttentionLayer.AttentionLayer(input_dim=self.attention_hidden_dim, hidden_dim=self.attention_hidden_dim)
        self.dropout = nn.Dropout(self.DROP_PROB)

        self.gp_layer = GPModel.GPModel(inducing_points=torch.randn(self.INDUCING_POINTS, self.attention_hidden_dim))

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

        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(self.grid_bounds[0], self.grid_bounds[1])
        self.attention = AttentionLayer.AttentionLayer(input_dim=self.ATTENTION_HIDDEN_DIM, hidden_dim=self.ATTENTION_HIDDEN_DIM)

    def forward(self, x):
        if self.CHANNELS == 1:
            batch_size, num_instances, c, h, w = x.size()
        else:
            batch_size, num_instances, h, w, c = x.size()

        features = self.feature_extractor(x.view(batch_size * num_instances, c, h, w)).view(batch_size, num_instances,
                                                                                            -1)
        features, _ = self.attention(features)
        features = self.scale_to_bounds(features).transpose(-1, -2).unsqueeze(-1)

        assert features.size(-1) == 1, f"Features should be 1 dimension after unsqueeze, got {features.shape}"
        return self.gp_layer(features)

