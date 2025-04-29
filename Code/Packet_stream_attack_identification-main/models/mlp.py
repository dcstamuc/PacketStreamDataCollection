import torch.nn as nn
import torch.nn.functional as F


class ANN(nn.Module):
    def __init__(self, device, hid_dim, input_dim, n_classes):
        super().__init__()

        self.hidden_dim = hid_dim
        self.fc1 = nn.Linear(input_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.fc3 = nn.Linear(hid_dim, hid_dim)
        self.fc4 = nn.Linear(hid_dim, n_classes)
        self.device = device

    def forward(self, src):

        x = F.relu(self.fc1(src))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        output = self.fc4(x)

        return output
