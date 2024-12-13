import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from pathlib import Path
from tqdm import tqdm

from model import AlexNet
from dataloader import get_dataloaders

def train_one_epoch(model, criterion, optimizer, train_loader, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    torch.cuda.empty_cache()
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for i, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': running_loss/(i+1),
            'acc': 100.*correct/total,
            'gpu_mem': f'{torch.cuda.memory_allocated()/1024**3:.1f}GB'
        })
    
    return running_loss/len(train_loader), 100.*correct/total

def validate(model, criterion, val_loader, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss/len(val_loader), 100.*correct/total

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

    # Get dataloaders
    train_loader, val_loader = get_dataloaders()
    
    # Initialize criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    
    # Training parameters
    num_epochs = 90
    best_acc = 0.0
    
    # Training loop
    for epoch in range(num_epochs):
        # Train for one epoch
        train_loss, train_acc = train_one_epoch(
            model, criterion, optimizer, train_loader, device, epoch)
        
        # Validate
        val_loss, val_acc = validate(model, criterion, val_loader, device)
        
        # Update learning rate
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