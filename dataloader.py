import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from PIL import Image
import numpy as np

# load data from data/iphone/training_data/iphone
train_original_dir = Path('data/iphone/training_data/iphone')
train_dslr_dir = Path('data/iphone/training_data/canon')

# load data from data/iphone/test_data/iphone
test_original_dir = Path('data/iphone/test_data/patches/iphone')
test_dslr_dir = Path('data/iphone/test_data/patches/canon')


def get_dataloaders():
    num_train_images = len([name for name in os.listdir(train_original_dir) if os.path.isfile(os.path.join(train_original_dir, name))])
    num_test_images = len([name for name in os.listdir(test_original_dir) if os.path.isfile(os.path.join(test_original_dir, name))])
    image_size = 100 * 100 * 3

    train_indices = np.arange(0, num_train_images)
    test_indices = np.arange(0, num_test_images)

    train_dataset = CustomImageDataset(train_original_dir, train_dslr_dir, train_indices, image_size)
    test_dataset = CustomImageDataset(test_original_dir, test_dslr_dir, test_indices, image_size)


    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

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

        phone_img = Image.open(Path(self.phone_dir, f"{img_index}.jpg"))
        phone_img = torch.from_numpy(np.asarray(phone_img)).float().view(1, self.image_size) / 255.0

        dslr_img = Image.open(Path(self.dslr_dir, f"{img_index}.jpg"))
        dslr_img = torch.from_numpy(np.asarray(dslr_img)).float().view(1, self.image_size) / 255.0

        return phone_img, dslr_img