import torch
from torch.utils.data import Dataset


class NetworkDataset(Dataset):
    def __init__(self, X, y):
        self.len = y.shape[0]
        self.x_data = torch.from_numpy(X).type(torch.float32)
        self.y_data = torch.from_numpy(y).type(torch.LongTensor)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len