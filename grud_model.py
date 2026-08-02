"""GRU-D architecture, shared between the training notebook and the demo app.

`torch.save(model.state_dict())` only stores tensors, so the class definition has to
live somewhere both sides can import. Keeping it here (instead of copy-pasting the
class into app code) means a change to the architecture can't silently break loading
an old checkpoint -- you get a key/shape mismatch at load time instead.

GRU-D.ipynb imports this class for its export cell, and model_utils.py imports it to
rebuild the model at prediction time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUDModel(nn.Module):
    """GRU-D style: decay + mask-aware input, per-timestep logits."""

    def __init__(self, input_size, hidden_size=128, output_size=1, dropout=0.2):
        super().__init__()
        num_layers = 2
        self.gamma_x = nn.Parameter(torch.zeros(input_size))
        self.gamma_h = nn.Parameter(torch.zeros(1))
        self.input_proj = nn.Linear(input_size * 2, hidden_size)

        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, X, M, D, mean_values):
        mean_values = mean_values.view(1, 1, -1)
        gamma_x = torch.exp(-F.softplus(self.gamma_x) * D)
        X_hat = M * X + (1.0 - M) * (gamma_x * X + (1.0 - gamma_x) * mean_values)
        h = torch.tanh(self.input_proj(torch.cat([X_hat, M], dim=-1)))
        out, _ = self.gru(h)
        gamma_h = torch.exp(-F.softplus(self.gamma_h) * D.mean(dim=-1, keepdim=True))
        out = self.dropout(gamma_h * out)
        return self.output_layer(out)
