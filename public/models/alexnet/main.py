import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from pathlib import Path
from tqdm import tqdm

from model import AlexNet
from dataloader import get_dataloaders
from torch.amp import autocast, GradScaler

def train_one_epoch(model, criterion, optimizer, train_loader, device, epoch):
    model.train()

    batch_size = train_loader.batch_size
    num_batches = len(train_loader)
    running_loss = torch.zeros(1, device=device)
    correct = torch.zeros(1, device=device)
    total = 0
    
    scaler = GradScaler(
        init_scale=2**10,
        growth_factor=2,
        backoff_factor=0.5,
        growth_interval=100
    )

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for i, (images, labels) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(device.type):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            if torch.isfinite(loss):
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                scaler.step(optimizer)
                scaler.update()
                
                with torch.no_grad():
                    running_loss += loss
                    correct += (outputs.argmax(1) == labels).sum()
                    total += labels.size(0)
            
            if i % 50 == 0:  # Update less frequently
                pbar.set_postfix({
                    'loss': (running_loss.item()/(i+1)),
                    'acc': 100.*correct.item()/total,
                    'lr': f'{optimizer.param_groups[0]["lr"]:.6f}',
                    'gpu_mem': f'{torch.cuda.memory_allocated()/1024**3:.1f}GB'
                })
                
    avg_loss = running_loss.item() / num_batches
    accuracy = 100. * correct.item() / total
    
    return avg_loss, accuracy

def validate(model, criterion, val_loader, device):
    model.eval()
    running_loss = torch.zeros(1, device=device)
    correct = torch.zeros(1, device=device)
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validation'):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss
            correct += (outputs.argmax(1) == labels).sum()
            total += labels.size(0)
    
    # Convert tensor values to floats before returning
    avg_loss = running_loss.item() / len(val_loader)
    accuracy = 100. * correct.item() / total
    
    return avg_loss, accuracy

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create model directory
    save_dir = Path('checkpoints')
    save_dir.mkdir(exist_ok=True)
    
    # Initialize model
    model = AlexNet(num_classes=1000)
    model = model.to(device)

    torch.cuda.empty_cache()

    # Training parameters
    num_epochs = 90
    best_acc = 0.0
    batch_size = 256
    base_lr = 0.01

    # Get dataloaders
    train_loader, val_loader = get_dataloaders(batch_size=batch_size, num_workers=12)
    
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=base_lr * (batch_size/128),
        momentum=0.9,
        weight_decay=0.0005
    )
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    
    # Training loop
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, criterion, optimizer, train_loader, device, epoch)
        
        val_loss, val_acc = validate(model, criterion, val_loader, device)
        
        scheduler.step()
        
        # Print statistics
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        print(f'GPU Memory: {torch.cuda.memory_allocated()/1024**3:.1f}GB')
        
        # Save checkpoint
        if val_acc > best_acc:
            print(f'Saving checkpoint... (accuracy improved from {best_acc:.2f}% to {val_acc:.2f}%)')
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
            }, save_dir / 'best_model.pth')

        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()