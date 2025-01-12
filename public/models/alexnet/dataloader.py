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
    PCA Color augmentation as described in the AlexNet paper.
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
            batch_size = images.size(0)
            images = images.permute(0, 2, 3, 1).reshape(-1, 3)
            pixels.append(images)
            samples_processed += batch_size
            
        pixels = torch.cat(pixels, dim=0)
        
        # Center the pixel values
        mean = pixels.mean(dim=0, keepdim=True)
        pixels_centered = pixels - mean
        
        # Compute covariance matrix
        cov = torch.mm(pixels_centered.t(), pixels_centered) / (pixels_centered.size(0) - 1)
        
        # Compute eigenvectors and eigenvalues
        eigval, eigvec = torch.linalg.eigh(cov)
        
        # Reverse to descending order
        eigval = eigval.flip(0)
        eigvec = eigvec.flip(1)
        
        self.eigval = eigval
        self.eigvec = eigvec
        
        print("PCA computation completed.")

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply PCA color augmentation to an image.
        Args:
            img: Tensor image [C,H,W]
        Returns:
            Augmented tensor image [C,H,W]
        """
        if self.eigval is None or self.eigvec is None:
            return img
        
        # Generate random weights
        alpha = torch.randn(3, device=img.device) * self.alphastd
        
        # Calculate the color perturbation
        rgb = (self.eigvec * alpha.unsqueeze(0)) @ self.eigval.unsqueeze(1)
        perturbation = rgb.view(3, 1, 1)
        
        # Apply perturbation
        img_aug = img + perturbation.float()
        
        return img_aug
    
class TenCropWrapper:
    """
    Wrapper for validation that performs 10-crop evaluation as in the AlexNet paper.
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def _normalize_tensor(self, tensor):
        """Helper method to normalize a tensor using mean and std"""
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor
    
    def __call__(self, img):
        # First resize the image to 256x256
        resize_transform = transforms.Resize(256)
        img = resize_transform(img)
        
        # Get all 10 crops
        crops = transforms.TenCrop(224)(img)
        
        # Convert to tensors and normalize
        result = []
        for crop in crops:
            tensor = transforms.ToTensor()(crop)
            tensor = self._normalize_tensor(tensor.clone())
            result.append(tensor)
            
        return torch.stack(result)

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
            transforms.ToTensor()
        ]
        if color_augmentation is not None:
            transform_list.append(color_augmentation)
        # Normalization should be the last step
        transform_list.append(transforms.Normalize(mean=mean, std=std))
        return transforms.Compose(transform_list)
    else:
        return TenCropWrapper(mean=mean, std=std)

def get_dataloaders(root_dir='data/ILSVRC2010', batch_size=128, num_workers=8):
    """
    Create and return training and validation dataloaders for ILSVRC2010.
    Now includes PCA color augmentation and 10-crop validation.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Create a simple transform to compute PCA
    initial_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    # Create initial dataset for PCA computation
    initial_dataset = ILSVRC2010Dataset(
        root_dir=root_dir,
        split='train',
        transform=initial_transform
    )
    
    initial_loader = DataLoader(
        initial_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    # Create and compute PCA color augmentation
    color_augmentation = PCAColorAugmentation(initial_loader)

    # Training transforms with PCA color augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        color_augmentation,
        transforms.Normalize(mean=mean, std=std)
    ])

    # Validation transforms with 10-crop
    val_transform = TenCropWrapper(mean=mean, std=std)

    # Create datasets with final transforms
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
        batch_size=batch_size // 10,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader

if __name__ == '__main__':
    train_loader, val_loader = get_dataloaders()