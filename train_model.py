from model import ResNeXt, Discriminator
from skimage.metrics import mean_squared_error as mse, structural_similarity as ssim
import random
import torch

manual_seed = 0
random.seed(manual_seed)
torch.manual_seed(manual_seed)
torch.use_deterministic_algorithms(True)
