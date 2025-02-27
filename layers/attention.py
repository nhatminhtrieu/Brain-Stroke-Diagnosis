import torch.nn as nn
import torch.nn.functional as F
import torch

class AttentionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, num_instances, feature_dim)
        attention_weights = self.attention(x)
        weights = F.softmax(attention_weights, dim=1)

        return (x * weights).sum(dim=1), weights.squeeze(-1)


class GatedAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GatedAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, num_instances, input_dim)
        attention_weights = self.attention(x)
        gate_weights = torch.sigmoid(self.gate(x))

        weights = attention_weights * gate_weights
        weights = F.softmax(weights, dim=1)

        return (x * weights).sum(dim=1), weights.squeeze(-1)

class MILAttentionLayer(nn.Module):
    """Implementation of the attention-based Deep MIL layer."""

    def __init__(
        self,
        input_dim,
        weight_params_dim,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        use_gated=False,
    ):
        super().__init__()

        self.weight_params_dim = weight_params_dim
        self.use_gated = use_gated

        # Initialize weights
        self.v_weight_params = nn.Parameter(torch.Tensor(input_dim, weight_params_dim))
        self.w_weight_params = nn.Parameter(torch.Tensor(weight_params_dim, 1))

        if self.use_gated:
            self.u_weight_params = nn.Parameter(torch.Tensor(input_dim, weight_params_dim))

        # Initialize weights using the specified initializer
        if kernel_initializer == "glorot_uniform":
            nn.init.xavier_uniform_(self.v_weight_params)
            nn.init.xavier_uniform_(self.w_weight_params)
            if self.use_gated:
                nn.init.xavier_uniform_(self.u_weight_params)

        # Add regularization if specified
        self.kernel_regularizer = kernel_regularizer

    def compute_attention_scores(self, instance):
        original_instance = instance
        instance = torch.tanh(torch.matmul(instance, self.v_weight_params))

        if self.use_gated:
            instance = instance * torch.sigmoid(torch.matmul(original_instance, self.u_weight_params))

        return torch.matmul(instance, self.w_weight_params)

    def forward(self, x):
        attention_scores = self.compute_attention_scores(x)
        attention_weights = torch.softmax(attention_scores, dim=1)
        return torch.sum(x * attention_weights, dim=1), attention_weights

    def regularization_loss(self):
        reg_loss = 0
        if self.kernel_regularizer:
            reg_loss += self.kernel_regularizer(self.v_weight_params)
            reg_loss += self.kernel_regularizer(self.w_weight_params)
            if self.use_gated:
                reg_loss += self.kernel_regularizer(self.u_weight_params)
        return reg_loss