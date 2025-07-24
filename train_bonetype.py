import os
import time
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import ssl
import certifi
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau

#SSL certificate issue fix
ssl_context = ssl.create_default_context(cafile=certifi.where())
# Dataset class
class BoneTypeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform

        for label, bone_type in enumerate(['KNEE_XRAY', 'SPINE_DEXA', 'HIP_DEXA']):
            bone_path = os.path.join(root_dir, bone_type)
            for class_folder in os.listdir(bone_path):
                class_dir = os.path.join(bone_path, class_folder)
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Logging
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join("logs", f"train_bonetype_{timestamp}.log")
    log_file = open(log_path, "w", encoding="utf-8")

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")

    dataset_root = "C:/Users/ahsan/Desktop/Osteo_IV_prj/datasets"
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = BoneTypeDataset(root_dir=dataset_root, transform=transform_train)

    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    log(f"Total training samples: {len(train_dataset)}")
    log(f"Total validation samples: {len(val_dataset)}")

    # Load VGG19 model
    model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, 3)  # 3 bone types
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    scaler = GradScaler(enabled=(device.type == 'cuda'))

    EPOCHS = 2
    total_start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        iter_start = time.time()

        loop = tqdm(train_dataloader, desc=f"Epoch [{epoch+1}/{EPOCHS}] (Train)")
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            with autocast(device_type=device.type):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

            iter_time = time.time() - iter_start
            iter_start = time.time()

            loop.set_postfix(
                Loss=f"{running_loss/total:.4f}",
                Acc=f"{100. * correct / total:.2f}%",
                IterTime=f"{iter_time:.3f}s",
                LR=f"{optimizer.param_groups[0]['lr']:.6f}"
            )

        epoch_duration = time.time() - iter_start  # Time since last iter_start reset
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        log(f"Epoch [{epoch+1}/{EPOCHS}] (Train) - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.2f}% - Duration: {epoch_duration:.2f}s")

        # Validation
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            val_loop = tqdm(val_dataloader, desc=f"Epoch [{epoch+1}/{EPOCHS}] (Val)")
            for images, labels in val_loop:
                images, labels = images.to(device), labels.to(device)
                with autocast(device_type=device.type):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

                val_loop.set_postfix(
                    Loss=f"{val_running_loss/val_total:.4f}",
                    Acc=f"{100. * val_correct / val_total:.2f}%"
                )

        epoch_val_loss = val_running_loss / val_total
        epoch_val_acc = 100. * val_correct / val_total
        log(f"Epoch [{epoch+1}/{EPOCHS}] (Val) - Loss: {epoch_val_loss:.4f} - Accuracy: {epoch_val_acc:.2f}%")

        scheduler.step(epoch_val_loss)

    total_duration = time.time() - total_start_time
    log(f"\n✅ Training completed in {total_duration:.2f} seconds.")

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/bone_type_model_vgg19.pth")
    log("Model saved as models/bone_type_model_vgg19.pth")

    with open("logs/bonetype_accuracy_vgg19.txt", "w") as f:
        f.write(f"{epoch_acc:.2f}")

    with open("logs/bonetype_val_accuracy_vgg19.txt", "w") as f:
        f.write(f"{epoch_val_acc:.2f}")

    log(f"Final Training Accuracy written to logs/bonetype_accuracy_vgg19.txt: {epoch_acc:.2f}%")
    log(f"Final Validation Accuracy written to logs/bonetype_val_accuracy_vgg19.txt: {epoch_val_acc:.2f}%")

    log_file.close()
