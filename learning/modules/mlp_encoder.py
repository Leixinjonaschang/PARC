import torch
import torch.nn as nn

class MLPEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 128], activation=nn.ELU):
        super().__init__()
        layers = []
        curr_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(curr_dim, h))
            layers.append(activation())
            curr_dim = h
        layers.append(nn.Linear(curr_dim, output_dim))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        x = self.fc(x)
        return torch.nn.functional.normalize(x, p=2, dim=-1)

