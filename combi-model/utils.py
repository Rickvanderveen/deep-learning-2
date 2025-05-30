import pickle
import torch
from torch.utils.data import Dataset

class EmbeddingDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        embedding = item['embedding']
        label = item['label']

        # If label is a scalar tensor, convert to long for classification
        label = label.long() if isinstance(label, torch.Tensor) else torch.tensor(label, dtype=torch.long)

        return embedding, label

    def get_path(self, idx):
        item = self.data[idx]
        return item['path']