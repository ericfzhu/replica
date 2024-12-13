from torch.utils.data import Dataset, DataLoader
import tarfile
from pathlib import Path
import scipy.io as sio
from tqdm import tqdm
import glob
from PIL import Image
from torchvision import transforms
import torch

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

        meta = sio.loadmat(Path(root_dir, 'devkit', 'data', 'meta.mat'))
        self.synsets = meta['synsets']

        if split == 'train':
            self.image_dir = Path(root_dir, 'train')

            self.tar_files = []
            self.labels = []
            for i in range(1000):  # 1000 classes
                synset = self.synsets[i]
                wnid = str(synset['WNID'][0][0])
                tar_path = Path(self.image_dir, f"{wnid}.tar")
                if tar_path.exists():
                    self.tar_files.append(tar_path)
                    self.labels.append(i)

            self.cache_dir = Path(root_dir, 'train_cache')
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            self.images = []
            self.image_labels = []  # Changed from self.labels to self.image_labels for consistency

            print(f'Found {len(self.tar_files)} tar files')
            print('Extracting training images...')
            for tar_file, label in tqdm(zip(self.tar_files, self.labels), total=len(self.tar_files)):
                synset_id = tar_file.stem
                cache_dir = Path(self.cache_dir, synset_id)

                if not cache_dir.exists():
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    with tarfile.open(tar_file, 'r') as tar:
                        tar.extractall(path=cache_dir)
                
                synset_images = glob.glob(str(cache_dir / '*.JPEG'))
                if synset_images:  # Only add if we found images
                    self.images.extend(synset_images)
                    self.image_labels.extend([label] * len(synset_images))
                else:
                    print(f"Warning: No images found in {cache_dir}")

        elif split == 'val':
            self.image_dir = Path(root_dir, 'val')

            val_dir = Path(root_dir, 'val')
            val_images_dir = Path(val_dir, 'val')
            if val_images_dir.exists():
                for img in val_images_dir.rglob('*.JPEG'):
                    img.rename(val_dir / img.name)
                val_images_dir.rmdir()

            with open(Path(root_dir, 'devkit', 'data', 'ILSVRC2010_validation_ground_truth.txt'), 'r') as f:
                self.val_labels = [int(line.strip()) - 1 for line in f]
            
            self.images = sorted(glob.glob(str(val_dir / '*.JPEG')))
            self.image_labels = self.val_labels
            
        # Add debug information
        print(f"Dataset initialized with {len(self.images)} images for {split} split")

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f'Error loading image {image_path}: {e}')
            return self.__getitem__(idx + 1)
        
        label = self.image_labels[idx]

        if self.transform:
            image = self.transform(image)

        # Convert label to torch tensor
        label = torch.tensor(label, dtype=torch.long)

        return image, label


def get_dataloaders(root_dir='data/ILSVRC2010', batch_size=128, num_workers=4):
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
    stats_file = 'dataset_stats.txt'
    if Path(stats_file).exists():
        with open(stats_file, 'r') as f:
            lines = f.readlines()
            mean = torch.tensor(eval(lines[0].split(': ')[1]))
            std = torch.tensor(eval(lines[1].split(': ')[1]))
    else:
        with open(stats_file, 'w') as f:
            mean, std = compute_mean_std(batch_size, num_workers)
            f.write(f'Mean: {mean.tolist()}\n')
            f.write(f'Std: {std.tolist()}\n')

    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Fixed size for both dimensions
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean.tolist(), std=std.tolist())
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Fixed size for both dimensions
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean.tolist(), std=std.tolist())
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
        pin_memory=True
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