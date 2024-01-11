import os
from pathlib import Path
from typing import Union
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import datasets
from PIL import Image
import numpy as np
import random

# load data from data/iphone/training_data/iphone
train_original_dir = Path('data/iphone/training_data/iphone')
train_dslr_dir = Path('data/iphone/training_data/canon')

# load data from data/iphone/test_data/iphone
test_original_dir = Path('data/iphone/test_data/patches/iphone')
test_dslr_dir = Path('data/iphone/test_data/patches/canon')

IMAGE_SIZE = 100 * 100 * 3

class RandomSampler(Sampler):
    def __init__(self, data_source, train_size):
        self.data_source = data_source
        self.train_size = train_size

    def __iter__(self):
        return iter(random.sample(range(len(self.data_source)), self.train_size))

    def __len__(self):
        return self.train_size

def get_dataloaders() -> Union[DataLoader, DataLoader]:
    train_indices = np.arange(0, len(os.listdir(train_original_dir)))
    test_indices = np.arange(0, len(os.listdir(test_original_dir)))

    train_dataset = CustomImageDataset(train_original_dir, train_dslr_dir, train_indices, IMAGE_SIZE)
    test_dataset = CustomImageDataset(test_original_dir, test_dslr_dir, test_indices, IMAGE_SIZE)

    train_dataloader = DataLoader(train_dataset, batch_size=100, sampler=RandomSampler(train_dataset, 5000))
    test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=True, num_workers=2)

    return train_dataloader, test_dataloader


class CustomImageDataset(Dataset):
    def __init__(self, phone_dir, dslr_dir, indices, image_size):
        self.phone_dir = phone_dir
        self.dslr_dir = dslr_dir
        self.indices = indices
        self.image_size = image_size

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img_index = self.indices[idx]

        # Load and transform phone image
        phone_img = Image.open(Path(self.phone_dir, f"{img_index}.jpg"))
        phone_np = np.asarray(phone_img).copy()
        phone_np = phone_np.transpose((2, 0, 1))  # C, H, W
        phone_tensor = torch.from_numpy(phone_np).float() / 255.0  # Normalize

        # Load and transform DSLR image
        dslr_img = Image.open(Path(self.dslr_dir, f"{img_index}.jpg"))
        dslr_np = np.asarray(dslr_img).copy()
        dslr_np = dslr_np.transpose((2, 0, 1))  # C, H, W
        dslr_tensor = torch.from_numpy(dslr_np).float() / 255.0  # Normalize

        return phone_tensor, dslr_tensor