from collections import Counter
from torch.utils.data import Dataset, DataLoader
import torch
from pathlib import Path
import scipy.io as sio
from PIL import Image
from torchvision import transforms
from typing import Optional, Tuple
import numpy as np
from tqdm import tqdm

def compute_mean_std(batch_size=128, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Fixed size for both dimensions
        transforms.ToTensor()
    ])

    dataset = ILSVRC2010Dataset(
        root_dir='data/ILSVRC2010',
        split='train',
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    mean = torch.zeros(3)
    squared_mean = torch.zeros(3)
    total_images = 0

    print('Computing mean...')
    for batch in tqdm(loader):
        images = batch[0]
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        total_images += batch_samples
        
    mean /= total_images

    print('Computing std...')
    for batch in tqdm(loader):
        images = batch[0]
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        squared_mean += (images.pow(2).mean(2)).sum(0)

    squared_mean /= total_images
    std = (squared_mean - mean.pow(2)).sqrt()

    return mean, std



class ILSVRC2010Dataset(Dataset):
    def __init__(self, root_dir, transform=None, split='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        
        # Load meta data
        meta = sio.loadmat(Path(root_dir, 'devkit', 'data', 'meta.mat'))
        self.synsets = meta['synsets']
        
        # Create WNID to label mapping
        self.wnid_to_label = {}
        for i in range(1000):
            synset = self.synsets[i]
            wnid = str(synset['WNID'][0][0])
            ilsvrc_id = int(synset['ILSVRC2010_ID'][0][0]) - 1
            self.wnid_to_label[wnid] = ilsvrc_id
            
        if split == 'train':
            self.cache_dir = Path(root_dir) / 'train_cache'
            self.images = []
            self.image_labels = []
            
            for wnid in self.wnid_to_label:
                cache_subdir = self.cache_dir / wnid
                if cache_subdir.exists():
                    synset_images = list(cache_subdir.glob('*.JPEG'))
                    if synset_images:
                        self.images.extend(synset_images)
                        label = self.wnid_to_label[wnid]
                        self.image_labels.extend([label] * len(synset_images))
                        
        elif split == 'val':
            self.image_dir = Path(root_dir) / 'val'
            gt_path = Path(root_dir) / 'devkit' / 'data' / 'ILSVRC2010_validation_ground_truth.txt'
            with open(gt_path, 'r') as f:
                self.val_labels = [int(line.strip()) - 1 for line in f]
            self.images = sorted(list(self.image_dir.glob('*.JPEG')))
            self.image_labels = self.val_labels

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        try:
            with Image.open(self.images[idx]) as img:
                image = img.convert('RGB')
            label = self.image_labels[idx]
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
        except Exception as e:
            print(f"Error loading image {self.images[idx]}: {e}")
            new_idx = (idx + 1) % len(self)
            return self.__getitem__(new_idx)


class PCAColorAugmentation:
    """
    PCA Color augmentation as described in AlexNet paper.
    Pre-computes PCA on a subset of training data for efficiency.
    """
    def __init__(self, dataloader: Optional[DataLoader] = None, alphastd: float = 0.1):
        self.alphastd = alphastd
        self.eigval = None
        self.eigvec = None
        
        if dataloader is not None:
            self.compute_pca(dataloader)

    def compute_pca(self, dataloader: DataLoader):
        """Compute PCA from a sample of training images."""
        print("Computing PCA for color augmentation...")
        pixels = []
        max_samples = 10000  # Limit number of images to process
        samples_processed = 0
        
        for images, _ in tqdm(dataloader):
            if samples_processed >= max_samples:
                break
            # Reshape to pixels x channels
            pixels.append(images.reshape(-1, 3))
            samples_processed += images.size(0)
            
        pixels = torch.cat(pixels, 0)
        
        # Center the pixel values
        mean = pixels.mean(dim=0, keepdim=True)
        pixels_centered = pixels - mean
        
        # Compute covariance matrix
        cov = torch.mm(pixels_centered.t(), pixels_centered) / (pixels.size(0) - 1)
        
        # Compute eigenvectors and eigenvalues
        self.eigval, self.eigvec = torch.linalg.eigh(cov)
        
        print("PCA computation completed.")

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply PCA color augmentation to an image.
        Args:
            img: Normalized tensor image [3,H,W]
        Returns:
            Augmented tensor image [3,H,W]
        """
        if self.eigval is None or self.eigvec is None:
            return img
            
        # Generate random weights
        alpha = torch.randn(3) * self.alphastd
        
        # Calculate the color perturbation
        perturbation = torch.mm(self.eigvec, (self.eigval.sqrt() * alpha).unsqueeze(1))
        perturbation = perturbation.view(3, 1, 1)
        
        # Apply perturbation
        return img + perturbation
    
class TenCropWrapper:
    """
    Wrapper for validation that performs 10-crop evaluation as in the AlexNet paper.
    """
    def __init__(self, mean, std):
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.TenCrop(224),
            transforms.Lambda(lambda crops: torch.stack([
                transforms.Normalize(mean=mean, std=std)(transforms.ToTensor()(crop))
                for crop in crops
            ]))
        ])
    
    def __call__(self, img):
        return self.transform(img)

def get_transforms(color_augmentation: Optional[PCAColorAugmentation] = None, 
                  is_training: bool = True,
                  mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                  std: Tuple[float, float, float] = (0.229, 0.224, 0.225)) -> transforms.Compose:
    """
    Get transforms for training or validation.
    """
    if is_training:
        transform_list = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
        if color_augmentation is not None:
            transform_list.append(color_augmentation)
        return transforms.Compose(transform_list)
    else:
        return TenCropWrapper(mean=mean, std=std)

def get_dataloaders(root_dir='data/ILSVRC2010', batch_size=128, num_workers=8):
    """
    Create and return training and validation dataloaders for ILSVRC2010.
    Simplified and optimized version.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Create datasets
    train_dataset = ILSVRC2010Dataset(
        root_dir=root_dir,
        split='train',
        transform=train_transform
    )

    val_dataset = ILSVRC2010Dataset(
        root_dir=root_dir,
        split='val',
        transform=val_transform
    )

    # Create dataloaders with basic optimizations
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader

if __name__ == '__main__':
    train_loader, val_loader = get_dataloaders()