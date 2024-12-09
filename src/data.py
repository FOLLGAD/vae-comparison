import torch
from torch.utils.data import Dataset
import numpy as np
from datasets import load_dataset

ds = load_dataset("uoft-cs/cifar10")


class CIFAR10Dataset(Dataset):
    def __init__(self, type: str = "train"):
        self.ds = ds[type]  # type: ignore

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        image = np.array(item["img"], dtype=np.float32)
        image = image / 255.0
        image = torch.tensor(image)
        label = item["label"]
        return (image, label)
