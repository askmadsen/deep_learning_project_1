import torch
from torch.utils.data import Dataset
import os
from PIL import Image


class CatDogDataset(Dataset):
    def __init__(self, dogs_dir: str, cats_dir: str, transform=None) -> None:

        self.dogs_dir = dogs_dir
        self.cats_dir = cats_dir
        self.transform = transform

        self.dog_files = [os.path.join(self.dogs_dir, f) for f in os.listdir(self.dogs_dir) if os.path.isfile(os.path.join(self.dogs_dir, f))]
        self.cat_files = [os.path.join(self.cats_dir, f) for f in os.listdir(self.cats_dir) if os.path.isfile(os.path.join(self.cats_dir, f))]

        self.length = len(self.dog_files) + len(self.cat_files)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        if idx < len(self.dog_files):
            img_path = self.dog_files[idx]
            label = 0  # Dog
        else:
            img_path = self.cat_files[idx - len(self.dog_files)]
            label = 1  # Cat
        
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
