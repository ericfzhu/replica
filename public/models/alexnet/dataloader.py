from torch.utils.data import Dataset, DataLoader
import tarfile
from pathlib import Path
from scipy import sio
from tqdm import tqdm
import glob
from PIL import Image
from torchvision import transforms
import torch

def compute_mean_std(batch_size=128, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize(256),
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
                wnid = str(synset['WNID'][0, 0][0])
                tar_path = Path(self.image_dir, f"{wnid}.tar")
                if tar_path.exists():
                    self.tar_files.append(tar_path)
                    self.labels.append(i)

            self.cache_dir = Path(root_dir, 'train_cache')
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            self.images = []
            self.labels = []

            print('Extracting training images...')
            for tar_file, label in tqdm(zip(self.tar_files, self.labels), total=len(self.tar_files)):
                synset_id = tar_file.stem
                cache_dir = Path(self.cache_dir, synset_id)

                if not cache_dir.exists():
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    with tarfile.open(tar_file, 'r') as tar:
                        tar.extractall(path=cache_dir)
                
                synset_images = glob.glob(str(cache_dir / '*.JPEG'))
                self.images.extend(synset_images)
                self.labels.extend([label] * len(synset_images))

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

    def __len__(self):
        return len(self.images)
    
    def __getitem(self, idx):
        image_path = self.images[idx]
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f'Error loading image {image_path}: {e}')
            return self[idx + 1], self.labels[idx + 1]
        
        label = self.image_labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
    

mean, std = compute_mean_std()

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean.tolist(), std=std.tolist())
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean.tolist(), std=std.tolist())
])

train_dataset = ILSVRC2010Dataset(
    root_dir='data/ILSVRC2010',
    split='train',
    transform=train_transform
)

val_dataset = ILSVRC2010Dataset(
    root_dir='data/ILSVRC2010',
    split='val',
    transform=val_transform
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)