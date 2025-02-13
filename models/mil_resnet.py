import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from models.base_classes import BaseModel, SequenceAwareModule, PositionalEncoding, MultiHeadFeatureRefiner
from layers import attention as AttentionLayer, gaussian_process as GPModel
import gpytorch

class CNN_ATT_GP(BaseModel):
    def __init__(self, params=None):
        super().__init__(params)
        self.attention = AttentionLayer.Attention(self.feature_dim)
        self.gp_layer = GPModel.ExactGPModel(
            inducing_points=self.INDUCING_POINTS,
            input_dim=self.feature_dim
        )
        self.classifier = nn.Linear(self.feature_dim, self.NUM_CLASSES)
        self.dropout = nn.Dropout(self.DROP_PROB)

        # Initialize GP model based on gp_type
        if self.gp_type == 'multi_task':
            self.num_latents = self.params.get('num_latents', 3)  # Number of latent functions for multi-task GP
            self.num_tasks = self.params.get('num_tasks', 4)  # Number of tasks for multi-task GP
            self.gp_layer = GPModel.MultitaskGPModel(num_latents=self.num_latents, num_tasks=self.num_tasks)
        elif self.gp_type == 'single_task':
            self.gp_layer = GPModel.SingletaskGPModel(inducing_points=torch.randn(self.INDUCING_POINTS, 1),
                                                      kernel_type=self.kernel_type)
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

        features = self.dropout(self.features_extractor(bags_flattened)).view(batch_size, num_instances, -1)
        features = self.linear_to_dim(features)

        attended_features, attended_weights = self.attention(features)
        attended_features_reshaped = attended_features.view(batch_size, -1)
        attended_features_reshaped = self.dropout(attended_features_reshaped)

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
        super().__init__(params)
        self.gp_layer = GPModel.ExactGPModel(
            inducing_points=self.INDUCING_POINTS,
            input_dim=self.feature_dim
        )
        self.attention = AttentionLayer.Attention(self.feature_dim)
        self.classifier = nn.Linear(self.feature_dim, self.NUM_CLASSES)
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

        features = self.dropout(self.features_extractor(bags_flattened)).view(batch_size, num_instances, -1)
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
        super().__init__(params)
        self.projection_head = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, 128)
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

        self.attention = AttentionLayer.AttentionLayer(input_dim=self.attention_hidden_dim,
                                                       hidden_dim=self.attention_hidden_dim)
        self.dropout = nn.Dropout(self.DROP_PROB)

        self.gp_layer = GPModel.GPModel(inducing_points=torch.randn(self.INDUCING_POINTS, self.attention_hidden_dim))

    def forward(self, features):
        attended_features, attended_weights = self.attention(features)
        attended_features_reshaped = attended_features.view(features.size(0), -1)

        gp_output = self.gp_layer(attended_features_reshaped)
        gp_mean = gp_output.mean.view(features.size(0), -1)
        combine_features = torch.cat((attended_features_reshaped, gp_mean), dim=1)

        return self.classifier(combine_features)

class CNN_ATT_GP_Multilabel(BaseModel):
    def __init__(self, params=None):
        super(CNN_ATT_GP_Multilabel, self).__init__(params=params)

        self.fc_8 = nn.Linear(self.feature_dim, self.ATTENTION_HIDDEN_DIM)  # Output from feature extractor
    
        if self.NUM_CLASSES != 1:
            self.attention_layers = nn.ModuleList(
                [AttentionLayer.AttentionLayer(self.ATTENTION_HIDDEN_DIM, self.ATTENTION_HIDDEN_DIM) for _ in
                 range(self.NUM_CLASSES)])  # Create multiple attention layers

            self.gp_layers = nn.ModuleList([GPModel.SingletaskGPModel(torch.randn(self.INDUCING_POINTS, 1)) for _ in
                                            range(self.NUM_CLASSES)])  # Create multiple GP layers
        else:
            self.attention_layers = AttentionLayer.AttentionLayer(self.ATTENTION_HIDDEN_DIM, self.ATTENTION_HIDDEN_DIM)
            self.gp_layers = GPModel.SingletaskGPModel(torch.randn(self.INDUCING_POINTS, 1))

        self.drop_out = nn.Dropout(self.DROP_PROB)

        self.fc = nn.Linear(self.ATTENTION_HIDDEN_DIM, 1)
        self.fc_for_combine = nn.Linear(self.ATTENTION_HIDDEN_DIM + 1, 1)

    def forward(self, bag):
        if self.CHANNELS == 1:
            batch_size, num_instances, c, h, w = bag.size()
        else:
            batch_size, num_instances, h, w, c = bag.size()
        bag = bag.view(batch_size * num_instances, c, h, w)

        x = self.features_extractor(bag)  # Extract features
        x = self.drop_out(x)

        x = x.view(batch_size, num_instances, -1)  # Reshape for attention
        x = self.fc_8(x)  # Pass through linear layer
        max_pooling = torch.max(x, dim=1)[0]

        if self.NUM_CLASSES == 1:
            att_outputs, _ = self.attention_layers(x)
            gp_outputs = self.gp_layers(self.fc(att_outputs))
            # Element-wise multiplication of attention outputs and Max pooling
            combined_features = att_outputs + max_pooling
            combined_features = torch.cat([combined_features, gp_outputs.mean.unsqueeze(-1)], dim=-1)
            combined_features = self.drop_out(combined_features)
            combined_features = self.fc_for_combine(combined_features)

        else:
            # Collect outputs from all attention layers
            att_outputs = []
            gp_outputs = []
            # Pass attention outputs through GP layer
            for i in range(len(self.attention_layers)):
                att_out, _ = self.attention_layers[i](x)
                gp_out = self.gp_layers[i](self.fc(att_out))
                att_outputs.append(att_out)
                gp_outputs.append(gp_out)

            combined_features = []
            for i in range(len(att_outputs)):
                combine_feature = torch.cat([att_outputs[i], gp_outputs[i].mean.unsqueeze(-1)], dim=-1)
                combined_features.append(self.fc_for_combine(combine_feature))
            combined_features = torch.cat(combined_features, dim=-1)

        return combined_features, gp_outputs, att_outputs

class CNN_ATT_GP_MIML(BaseModel):
    def __init__(self, params=None):
        super(CNN_ATT_GP_MIML, self).__init__(params=params)

        self.fc_8 = nn.Linear(self.feature_dim, self.ATTENTION_HIDDEN_DIM)  # Output from feature extractor

        self.attention_layers = AttentionLayer.AttentionLayer(self.ATTENTION_HIDDEN_DIM, self.ATTENTION_HIDDEN_DIM)
        self.gp_layers = GPModel.MultitaskGPModel(num_latents=self.NUM_CLASSES, num_tasks=self.NUM_CLASSES, hidden_dim=self.ATTENTION_HIDDEN_DIM)
        self.drop_out = nn.Dropout(self.DROP_PROB)
        self.classifier = nn.Linear(self.ATTENTION_HIDDEN_DIM, self.NUM_CLASSES)
        self.fc_for_gp = nn.Linear(self.ATTENTION_HIDDEN_DIM, 1)

    def forward(self, bag):
        if self.CHANNELS == 1:
            batch_size, num_instances, c, h, w = bag.size()
        else:
            batch_size, num_instances, h, w, c = bag.size()
        bag = bag.view(batch_size * num_instances, c, h, w)

        x = self.features_extractor(bag)  # Extract features
        x = self.drop_out(x)

        x = x.view(batch_size, num_instances, -1)  # Reshape for attention
        x = self.fc_8(x)  # Pass through linear layer

        # x = self.sequence_encoder(x)

        att_out, att_weights = self.attention_layers(x)
        gp_output = self.gp_layers(self.fc_for_gp(att_out))

        output = self.classifier(att_out)
        return output, gp_output, att_weights