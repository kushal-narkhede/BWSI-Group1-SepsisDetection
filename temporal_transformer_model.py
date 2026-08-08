"""Causal Temporal Transformer architecture, shared between training and the demo app.

Same reasoning as grud_model.py: `torch.save(state_dict)` stores only tensors, so the
class definition has to live somewhere both the training script and model_utils.py can
import. temptransformersepsis-2.py imports these classes rather than redefining them.

Note the causal mask -- each hour attends only to itself and earlier hours, so the model
cannot look into a patient's future when scoring the current hour.
"""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class InputProjectionHead(nn.Module):
    def __init__(self, in_features, d_model, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, x):
        return self.net(x)


class CausalTemporalTransformer(nn.Module):
    def __init__(self, num_features, d_model=128, nhead=8, num_layers=4,
                 dim_feedforward=256, dropout=0.3, max_len=48):
        super().__init__()
        self.max_len = max_len
        self.projection = InputProjectionHead(num_features, d_model, dropout)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x, src_key_padding_mask=None):
        seq_len = x.size(1)

        x_proj = self.projection(x)
        x_pos = self.pos_encoder(x_proj)

        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=x.device),
            diagonal=1
        )

        padding_mask_float = None
        if src_key_padding_mask is not None:
            padding_mask_float = torch.zeros_like(src_key_padding_mask, dtype=x.dtype)
            padding_mask_float = padding_mask_float.masked_fill(src_key_padding_mask, float('-inf'))

        out = self.transformer(
            x_pos,
            mask=causal_mask,
            src_key_padding_mask=padding_mask_float
        )

        logits = self.classifier(out).squeeze(-1)
        return logits
