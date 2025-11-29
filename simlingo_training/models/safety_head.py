import torch
import torch.nn as nn

class SafetyHead(nn.Module):
    """
    Predicts whether a Dreamer instruction is safe to execute.
    Input:  per-sample action representation (B, D)
    Output: logits (B, 2) for [unsafe, safe]
    """
    def __init__(self, d_model: int, hidden_dim: int = 256, num_classes: int = 2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D)
        return self.mlp(x)