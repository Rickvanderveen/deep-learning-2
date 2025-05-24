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
        path = item['path']  # optional, useful for debug or visualizing

        # If label is a scalar tensor, convert to long for classification
        label = label.long() if isinstance(label, torch.Tensor) else torch.tensor(label, dtype=torch.long)

        return embedding, label, path  # You can drop path if unused

if __name__ == "__main__":
    with open("embeddings.pkl", "rb") as f:
        data = pickle.load(f)
    
    dataset = EmbeddingDataset(data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    for embeddings, labels, paths in loader:
        # embeddings: [B, 2152]
        # labels: [B]
        print(embeddings.shape)
        print(labels)
        print(paths)