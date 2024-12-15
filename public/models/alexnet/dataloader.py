from torch.utils.data import Dataset, DataLoader
import tarfile
from pathlib import Path
import scipy.io as sio
from tqdm import tqdm
import glob
from PIL import Image
from torchvision import transforms
import torch
import random

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
        for i in range(1000):  # Only first 1000 are competition classes
            synset = self.synsets[i]
            wnid = str(synset['WNID'][0][0])
            ilsvrc_id = int(synset['ILSVRC2010_ID'][0][0]) - 1  # Convert to 0-based index
            self.wnid_to_label[wnid] = ilsvrc_id
            
        print(f"Created mapping for {len(self.wnid_to_label)} WNIDs")
        
        if split == 'train':
            self.image_dir = Path(root_dir, 'train')
            self.cache_dir = Path(root_dir, 'train_cache')
            
            self.images = []
            self.image_labels = []
            
            # Get all JPEG files from cache
            for wnid in self.wnid_to_label.keys():
                cache_subdir = self.cache_dir / wnid
                if cache_subdir.exists():
                    synset_images = list(cache_subdir.glob('*.JPEG'))
                    if synset_images:
                        self.images.extend(synset_images)
                        label = self.wnid_to_label[wnid]
                        self.image_labels.extend([label] * len(synset_images))
                    
            print(f"Found {len(self.images)} training images")
            print(f"Label range: min={min(self.image_labels)}, max={max(self.image_labels)}")
            print(f"Number of unique labels: {len(set(self.image_labels))}")
            
            # Print a few samples
            import random
            for i in range(5):
                idx = random.randint(0, len(self.images)-1)
                img_path = self.images[idx]
                label = self.image_labels[idx]
                wnid = img_path.parent.name
                print(f"Sample {i}: WNID={wnid}, Label={label}")
                
        elif split == 'val':
            self.image_dir = Path(root_dir, 'val')
            
            # Load validation ground truth
            gt_path = Path(root_dir, 'devkit', 'data', 'ILSVRC2010_validation_ground_truth.txt')
            with open(gt_path, 'r') as f:
                # Val labels are 1-based in file, convert to 0-based
                self.val_labels = [int(line.strip()) - 1 for line in f]
            
            self.images = sorted(list(self.image_dir.glob('*.JPEG')))
            self.image_labels = self.val_labels
            
            print(f"Found {len(self.images)} validation images")
            print(f"Val label range: min={min(self.image_labels)}, max={max(self.image_labels)}")
            
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        try:
            # Open image while ignoring EXIF data
            with Image.open(self.images[idx]) as img:
                # Force RGB without EXIF
                image = Image.new('RGB', img.size)
                image.paste(img)
            label = self.image_labels[idx]
            
            if self.transform:
                image = self.transform(image)
            
            return image, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"Error loading image {self.images[idx]}: {e}")
            new_idx = (idx + 1) % len(self)
            return self.__getitem__(new_idx)

class ColorAugmentation:
    def __init__(self, alphastd=0.1):
        self.alphastd = alphastd

    def __call__(self, img):
        if not isinstance(img, torch.Tensor):
            img = transforms.ToTensor()(img)
        
        # Convert to float32 for numerical stability
        img = img.float()
        
        # Reshape image to get pixels as rows
        pixels = img.reshape(-1, 3)
        
        # Calculate covariance matrix and eigenvectors
        cov = torch.mm(pixels.T, pixels) / pixels.shape[0]
        eigvals, eigvecs = torch.linalg.eigh(cov)
        
        # Generate random weights for eigenvectors
        alpha = torch.randn(3) * self.alphastd
        
        # Calculate the color perturbation
        perturbation = torch.mm(eigvecs, (eigvals.sqrt() * alpha).unsqueeze(1))
        perturbation = perturbation.view(3, 1, 1)
        
        # Apply the perturbation
        img = img + perturbation
        
        # Clamp values to valid range
        return img.clamp_(0, 1)

def get_dataloaders(root_dir='data/ILSVRC2010', batch_size=512, num_workers=8):
    """
    Create and return training and validation dataloaders for ILSVRC2010.
    
    Args:
        root_dir (str): Root directory containing the dataset
        batch_size (int): Batch size for both loaders
        num_workers (int): Number of worker processes for data loading
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Compute dataset statistics
    mean = [0.49095413088798523, 0.4655124545097351, 0.4093688428401947]
    std = [0.2894245386123657, 0.281748503446579, 0.3004192113876343]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
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

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True
    )

    return train_loader, val_loader


if __name__ == '__main__':
    train_loader, val_loader = get_dataloaders()