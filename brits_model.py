"""BRITS architecture, reconstructed from models/brits_best.pt.

The parameter structure here is CONFIRMED: this module loads brits_best.pt with
strict=True and no missing or unexpected keys. Shapes pin down input_size=33,
rnn_hid_size=64, two RITS directions, and a 64 -> 64 -> 1 classifier head.

What the checkpoint does NOT record, and what therefore still has to be confirmed
against the original training code before this model can be served:

  1. WHICH 33 features, in what order. The app has no way to build an input row
     without the exact list. (For reference, the 34 PhysioNet physiological
     variables minus EtCO2 would be 33, but that is a guess, not a fact.)
  2. Whether inputs were standardized, and with what mean/std.
  3. How the forward and backward hidden states combine before the classifier.
     The head takes 64 features, not 128, so it is not a concatenation -- it is
     a mean, a sum, or forward-only. `combine` below is a placeholder.
  4. Whether the label is per-timestep or per-stay.

Until 1-3 are answered, model_utils keeps BRITS marked coming_soon. Serving it on
guessed preprocessing would produce confident, meaningless probabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalDecay(nn.Module):
    def __init__(self, input_size, output_size, diag=False):
        super().__init__()
        self.diag = diag
        self.linear = nn.Linear(input_size, output_size)
        if diag:
            # Not in the checkpoint, so it must be non-persistent to load strictly.
            self.register_buffer("m", torch.eye(input_size), persistent=False)

    def forward(self, d):
        if self.diag:
            gamma = F.relu(F.linear(d, self.linear.weight * self.m, self.linear.bias))
        else:
            gamma = F.relu(self.linear(d))
        return torch.exp(-gamma)


class FeatureRegression(nn.Module):
    """Regresses each variable on the others; the diagonal is masked out."""

    def __init__(self, input_size):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(input_size, input_size))
        self.bias = nn.Parameter(torch.zeros(input_size))
        self.register_buffer("mask", torch.ones(input_size, input_size) - torch.eye(input_size))

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)


class RITS(nn.Module):
    """One direction of BRITS: decay the hidden state, impute, then step the GRU cell."""

    def __init__(self, input_size, rnn_hid_size):
        super().__init__()
        self.input_size = input_size
        self.rnn_hid_size = rnn_hid_size
        self.temp_decay_h = TemporalDecay(input_size, rnn_hid_size, diag=False)
        self.temp_decay_x = TemporalDecay(input_size, input_size, diag=True)
        self.hist_reg = nn.Linear(rnn_hid_size, input_size)
        self.feat_reg = FeatureRegression(input_size)
        self.weight_combine = nn.Linear(input_size * 2, input_size)
        self.gru = nn.GRUCell(input_size * 2, rnn_hid_size)

    def forward(self, X, M, D):
        B, T, _ = X.shape
        h = torch.zeros(B, self.rnn_hid_size, device=X.device, dtype=X.dtype)
        states = []
        for t in range(T):
            x, m, d = X[:, t], M[:, t], D[:, t]

            h = h * self.temp_decay_h(d)
            x_h = self.hist_reg(h)                       # history-based estimate
            x_c = m * x + (1.0 - m) * x_h                # fill gaps with it
            z_h = self.feat_reg(x_c)                     # feature-based estimate
            alpha = torch.sigmoid(self.weight_combine(torch.cat([self.temp_decay_x(d), m], dim=1)))
            c_h = alpha * z_h + (1.0 - alpha) * x_h      # blended imputation
            c_c = m * x + (1.0 - m) * c_h

            h = self.gru(torch.cat([c_c, m], dim=1), h)
            states.append(h)
        return torch.stack(states, dim=1)                # (B, T, hidden)


class BRITS(nn.Module):
    def __init__(self, input_size=33, rnn_hid_size=64, dropout=0.3, combine="mean"):
        super().__init__()
        self.forward_rits = RITS(input_size, rnn_hid_size)
        self.backward_rits = RITS(input_size, rnn_hid_size)
        self.classifier = nn.Sequential(
            nn.Linear(rnn_hid_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        # UNVERIFIED -- see module docstring, item 3.
        self.combine = combine

    def forward(self, X, M, D):
        h_f = self.forward_rits(X, M, D)
        h_b = self.backward_rits(X.flip(1), M.flip(1), D.flip(1)).flip(1)

        if self.combine == "mean":
            h = (h_f + h_b) / 2.0
        elif self.combine == "sum":
            h = h_f + h_b
        elif self.combine == "forward":
            h = h_f
        else:
            raise ValueError(f"unknown combine mode: {self.combine}")

        return self.classifier(h)                        # (B, T, 1) per-hour logits
