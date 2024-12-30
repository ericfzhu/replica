import os
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm

from model import AlexNet
from dataloader import get_dataloaders, PCAColorAugmentation, get_transforms
from torch.amp import autocast, GradScaler

def train_one_epoch(model, criterion, optimizer, train_loader, device, epoch, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for i, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast(device_type=device.type):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        scaler.step(optimizer)
        scaler.update()
        
        # Update metrics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if i % 50 == 0:
            pbar.set_postfix({
                'loss': f'{running_loss/(i+1):.3f}',
                'acc': f'{100.*correct/total:.2f}%',
                'gpu': f'{torch.cuda.memory_allocated()/1024**3:.1f}GB'
            })
    
    return running_loss / len(train_loader), 100. * correct / total

def validate(model, criterion, val_loader, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validation'):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(val_loader), 100. * correct / total

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name()}')
    
    # Create model directory
    save_dir = Path('checkpoints/alexnet')
    save_dir.mkdir(exist_ok=True, parents=True)

    # Initialize model
    model = AlexNet(num_classes=1000)
    model = model.to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    torch.cuda.empty_cache()
    
    # Training parameters
    num_epochs = 90
    batch_size = 128 * max(2, torch.cuda.device_count())
    base_lr = 0.01
    
    # Initialize gradient scaler
    scaler = GradScaler()

    train_loader, _ = get_dataloaders(
        root_dir='data/ILSVRC2010',
        batch_size=128,
        num_workers=12
    )

    # Compute PCA using a subset of the training data
    color_augmentation = PCAColorAugmentation(dataloader=train_loader)

    # Now get the transforms including PCAColorAugmentation
    train_transform = get_transforms(
        color_augmentation=color_augmentation, 
        is_training=True,
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
    
    # Get dataloaders
    train_loader, val_loader = get_dataloaders(
        batch_size=batch_size,
        num_workers=12
    )

    train_loader.dataset.transform = train_transform
    
    print(f"\nDataset sizes:")
    print(f"Training: {len(train_loader.dataset)} images")
    print(f"Validation: {len(val_loader.dataset)} images")
    print(f"Batch size: {batch_size}")
    print(f"Steps per epoch: {len(train_loader)}")
    
    criterion = nn.CrossEntropyLoss().to(device)
    
    # Optimizer settings from paper
    optimizer = optim.SGD(
        model.parameters(),
        lr=base_lr,
        momentum=0.9,
        weight_decay=0.0005
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.1,
        patience=5,
        verbose=True
    )
    
    # Training loop
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 20)
        
        # Training phase
        train_loss, train_acc = train_one_epoch(
            model, criterion, optimizer, train_loader, device, epoch, scaler
        )
        
        # Validation phase
        val_loss, val_acc = validate(model, criterion, val_loader, device)
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"Val - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        print(f'Learning rate: {optimizer.param_groups[0]["lr"]:.6f}')
        print(f'GPU Memory: {torch.cuda.memory_allocated()/1024**3:.1f}GB')
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            print(f"\nNew best accuracy: {val_acc:.2f}%")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'scaler_state_dict': scaler.state_dict(),
            }, save_dir / 'best_model.pth')
        
        # Save regular checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'scaler_state_dict': scaler.state_dict(),
            }, save_dir / f'checkpoint_epoch_{epoch+1}.pth')
        
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()