import torch
import torch.nn.functional as F

def gaussian_window(size, sigma):
    coords = torch.arange(size, dtype=torch.float32)
    coords -= size // 2

    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()

    return g.view(1, -1) * g.view(-1, 1)

def ssim(img1, img2, max_val=255, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03):
    device = img1.device
    _, channels, _, _ = img1.size()
    
    window = gaussian_window(filter_size, filter_sigma).to(device)
    window = window.repeat(channels, 1, 1, 1)

    mu1 = F.conv2d(img1, window, padding=filter_size//2, groups=channels)
    mu2 = F.conv2d(img2, window, padding=filter_size//2, groups=channels)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=filter_size//2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=filter_size//2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=filter_size//2, groups=channels) - mu1_mu2

    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2

    v1 = 2.0 * sigma12 + c2
    v2 = sigma1_sq + sigma2_sq + c2

    ssim = ((2.0 * mu1_mu2 + c1) * v1) / ((mu1_sq + mu2_sq + c1) * v2)
    
    return ssim.mean()