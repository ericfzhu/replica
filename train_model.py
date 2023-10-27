from model import ResNeXt, Discriminator
from dataloader import get_dataloaders
from skimage.metrics import mean_squared_error as mse, structural_similarity as ssim
import random
import torch

# Seed
manual_seed = 0
random.seed(manual_seed)
torch.manual_seed(manual_seed)
torch.use_deterministic_algorithms(True)

# Epochs
num_epochs = 20000

# Initialize models
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
generator = ResNeXt().to(device)
discriminator = Discriminator().to(device)

# Dataloaders
train_dataloader, test_dataloader = get_dataloaders()

# Training loop
for i in range(num_epochs):
    for phone_imgs, dslr_imgs in train_dataloader:
        discriminator.zero_grad()
        phone_imgs, dslr_imgs = phone_imgs.to(device), dslr_imgs.to(device)

        gen_imgs = generator(phone_imgs)

