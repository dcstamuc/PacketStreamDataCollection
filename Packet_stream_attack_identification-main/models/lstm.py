import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, encoder, device, hid_dim, n_classes):
        super().__init__()

        self.hidden_dim = hid_dim
        self.encoder = encoder
        self.fc1 = nn.Linear(hid_dim, n_classes)
        self.device = device

    def forward(self, src):
        # Hidden [n layers, batch, hid dim] [2, 20, 20]
        hidden, cell = self.encoder(src)

        last_hidden = hidden[-1]

        last_hidden = last_hidden.contiguous().view(-1, self.hidden_dim)

        output = self.fc1(last_hidden)

        return output
