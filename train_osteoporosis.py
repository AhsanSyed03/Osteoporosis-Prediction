import os
import time
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as transforms
from torchvision import models
from tqdm import tqdm
from utils.dataset import OsteoporosisDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import GradScaler, autocast
import torch.nn.functional as F

if __name__ == '__main__':
    # Set device and print only once in the main process
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup logging (moved inside the main block)
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = f"logs/train_osteoporosis_{timestamp}.log"
    log_file = open(log_path, "w", encoding="utf-8")

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")

    # Dataset paths
    dataset_dirs = [
        "C:/Users/ahsan/Desktop/Osteo_IV_prj/datasets/KNEE_XRAY",
        "C:/Users/ahsan/Desktop/Osteo_IV_prj/datasets/SPINE_DEXA",
        "C:/Users/ahsan/Desktop/Osteo_IV_prj/datasets/HIP_DEXA"
    ]

    # Transformations with more augmentation
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load datasets and split into train and validation
    all_datasets = [OsteoporosisDataset(root_dir=dir, transform=transform_train) for dir in dataset_dirs]
    combined_dataset = ConcatDataset(all_datasets)

    train_size = int(0.85 * len(combined_dataset))  # Split as 85% for training
    val_size = len(combined_dataset) - train_size      # Remaining 15% for validation
    train_dataset, val_dataset = torch.utils.data.random_split(combined_dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    total_train_samples = len(train_dataset)
    total_val_samples = len(val_dataset)
    log(f"Total training samples: {total_train_samples}")
    log(f"Total validation samples: {total_val_samples}")

    if total_train_samples == 0:
        raise ValueError("Training dataset is empty. Check the folder paths and contents.")
    if total_val_samples == 0:
        log("Warning: Validation dataset is empty. Consider adjusting the split.")

    # Load ResNet-50 model with dropout
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 3)
    model = nn.Sequential(model, nn.Dropout(0.3))
    model = model.to(device)

    # Loss and optimizer with weight decay
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3) # Removed verbose argument

    # Mixed precision training
    scaler = torch.amp.GradScaler('cuda', enabled=True if device.type == 'cuda' else False)

    # Training and Validation
    EPOCHS = 33
    total_start_time = time.time()

    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        train_running_loss = 0.0
        train_correct = 0
        train_total = 0
        epoch_train_start_time = time.time()
        train_progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch+1}/{EPOCHS}] (Train)")

        for i, (images, labels) in enumerate(train_progress_bar):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=True if device.type == 'cuda' else False):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            train_progress_bar.set_postfix({
                'Loss': f"{train_running_loss/(i+1)/images.size(0):.4f}",
                'Acc': f"{100.*train_correct/train_total:.2f}%",
                'LR': f"{optimizer.param_groups[0]['lr']:.6f}"
            })

        epoch_train_loss = train_running_loss / total_train_samples
        epoch_train_acc = 100. * train_correct / total_train_samples
        epoch_train_duration = time.time() - epoch_train_start_time
        log(f"Epoch [{epoch+1}/{EPOCHS}] (Train) - Time: {epoch_train_duration:.2f}s - Loss: {epoch_train_loss:.4f} - Accuracy: {epoch_train_acc:.2f}% - LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        epoch_val_start_time = time.time()
        with torch.no_grad():
            val_progress_bar = tqdm(val_dataloader, desc=f"Epoch [{epoch+1}/{EPOCHS}] (Val)")
            for images, labels in val_progress_bar:
                images, labels = images.to(device), labels.to(device)
                with torch.amp.autocast('cuda', enabled=True if device.type == 'cuda' else False):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

                val_progress_bar.set_postfix({
                    'Loss': f"{val_running_loss/(len(val_dataloader)):.4f}",
                    'Acc': f"{100.*val_correct/val_total:.2f}%"
                })

        epoch_val_loss = val_running_loss / total_val_samples
        epoch_val_acc = 100. * val_correct / total_val_samples
        epoch_val_duration = time.time() - epoch_val_start_time
        log(f"Epoch [{epoch+1}/{EPOCHS}] (Val) - Time: {epoch_val_duration:.2f}s - Loss: {epoch_val_loss:.4f} - Accuracy: {epoch_val_acc:.2f}%")

        scheduler.step(epoch_val_loss)

    # Total time
    total_duration = time.time() - total_start_time
    log(f"\n✅ Training completed in {total_duration:.2f} seconds.")

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/osteoporosis_model_resnet50.pth")
    log("Model saved as models/osteoporosis_model_resnet50.pth")

    # Save final validation accuracy
    with open("logs/osteoporosis_accuracy_resnet50.txt", "w", encoding="utf-8") as f:
        f.write(f"{epoch_val_acc:.2f}")

    log(f"Final Validation Accuracy written to logs/osteoporosis_accuracy_resnet50.txt: {epoch_val_acc:.2f}%")
    log_file.close()