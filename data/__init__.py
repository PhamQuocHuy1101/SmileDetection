from torch.utils.data import Dataset

import utils
import config

class WrapDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    def __len__(self):
        return len(self.Y)
    def __getitem__(self, index):
        return (self.X[index], self.Y[index])
