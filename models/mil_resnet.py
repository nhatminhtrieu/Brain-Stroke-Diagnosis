import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from models.base_classes import BaseModel
from layers import attention as AttentionLayer, gaussian_process as GPModel
import gpytorch

from utils.data_process import NUM_CLASSES


class CNN_Attention(BaseModel):
    def __init__(self, params=None):
        super(CNN_Attention, self).__init__(params=params)
        hidden_dim = 8
        self.ATTENTION_HIDDEN_DIM = self.feature_dim
        # self.attention_layer = AttentionLayer.AttentionLayer(hidden_dim, self.ATTENTION_HIDDEN_DIM)
        self.attention_layer = AttentionLayer.GatedAttention(hidden_dim, self.ATTENTION_HIDDEN_DIM)
        self.att_layer = AttentionLayer.MILAttentionLayer(hidden_dim, self.ATTENTION_HIDDEN_DIM)
        self.drop_out = nn.Dropout(self.DROP_PROB)
        self.fc_to_8 = nn.Linear(self.feature_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, self.NUM_CLASSES)
        # self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)

    def forward(self, bags):
        batch_size, num_instances, h, w, c = bags.size()
        bags = bags.view(batch_size * num_instances, c, h, w)
        x = self.features_extractor(bags)
        x = self.drop_out(x)
        x = x.view(batch_size, num_instances, -1)
        cnn_features = self.fc_to_8(x)
        # att_out, att_weights = self.attention_layer(cnn_features)
        # cnn_features, _ = self.gru(cnn_features)
        att_out = self.att_layer(cnn_features)
        att_weights = []
        x = self.fc(att_out).squeeze(-1)
        return x, cnn_features, att_weights, att_out

class CNN_ATT_GP_Multilabel(BaseModel):
    def __init__(self, params=None):
        super(CNN_ATT_GP_Multilabel, self).__init__(params=params)
        self.ATTENTION_HIDDEN_DIM = 8
        self.fc_8 = nn.Linear(self.feature_dim, self.ATTENTION_HIDDEN_DIM)  # Output from feature extractor
    
        if self.NUM_CLASSES != 1:
            self.attention_layers = nn.ModuleList(
                [AttentionLayer.AttentionLayer(self.ATTENTION_HIDDEN_DIM, self.ATTENTION_HIDDEN_DIM) for _ in
                 range(self.NUM_CLASSES)])  # Create multiple attention layers

            self.gp_layers = nn.ModuleList([GPModel.SVGP_Model(torch.randn(self.INDUCING_POINTS, 1)) for _ in
                                            range(self.NUM_CLASSES)])  # Create multiple GP layers
        else:
            self.attention_layers = AttentionLayer.AttentionLayer(self.ATTENTION_HIDDEN_DIM, self.ATTENTION_HIDDEN_DIM)
            self.gp_layers = GPModel.SVGP_Model(torch.randn(self.INDUCING_POINTS, 1))

        self.drop_out = nn.Dropout(self.DROP_PROB)

        self.fc = nn.Linear(self.ATTENTION_HIDDEN_DIM, 1)
        self.fc_for_combine = nn.Linear(9 * 6, self.NUM_CLASSES)

    def forward(self, bag):
        # if self.CHANNELS == 1:
        #     batch_size, num_instances, c, h, w = bag.size()
        # else:
        batch_size, num_instances, h, w, c = bag.size()
        bag = bag.view(batch_size * num_instances, c, h, w)

        x = self.features_extractor(bag)  # Extract features
        x = self.drop_out(x)

        x = x.view(batch_size, num_instances, -1)  # Reshape for attention
        x = self.fc_8(x)  # Pass through linear layer

        if self.NUM_CLASSES == 1:
            att_outputs, _ = self.attention_layers(x)
            gp_outputs = self.gp_layers(self.fc(att_outputs))

            combined_features = torch.cat([att_outputs, gp_outputs.mean.unsqueeze(-1)], dim=-1)
            combined_features = self.fc_for_combine(combined_features)

        else:
            # Collect outputs from all attention layers
            att_weights = 0
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
                combined_feature = att_outputs[i]
                combined_feature = torch.cat([combined_feature, gp_outputs[i].mean.unsqueeze(-1)], dim=-1)
                # combined_features.append(self.fc_for_combine(combined_feature))
                combined_features.append(combined_feature)
            # print(f'shape of combined_features: {combined_features.shape}')
            combined_features = torch.cat(combined_features, dim=-1)
            combined_features = self.fc_for_combine(combined_features)
        return combined_features, gp_outputs, att_weights, att_outputs