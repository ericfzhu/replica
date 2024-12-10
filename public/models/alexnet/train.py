from torch.utils.data import Dataset, DataLoader
import tarfile
from pathlib import Path

class ILSVRC2010Dataset(Dataset):
    def __init__(self, path, transform=None, is_train=True):
        self.path = path
        self.transform = transform
        self.image_list = []

        extract_dir = Path('train_images' if is_train else 'val_images')
        if not extract_dir.exists():
            extract_dir.mkdir(parents=True, exist_ok=True)

            with tarfile.open(self.path, 'r') as tar:
                tar.extractall(path=extract_dir)

        for image_path in extract_dir.rglob('*.JPEG'):
            class_name = image_path.parent.name
            self.image_list.append((image_path, class_name))