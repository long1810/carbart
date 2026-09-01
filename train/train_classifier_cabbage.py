import os
import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
# ==========================================
# Tạo model MobileNetV2
# ==========================================
def get_model(num_classes=2):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


# ==========================================
# Train 1 epoch
# ==========================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        running_correct += torch.sum(preds == labels.data)
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_correct.double().item() / total
    return epoch_loss, epoch_acc


# ==========================================
# Đánh giá
# ==========================================
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        running_correct += torch.sum(preds == labels.data)
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_correct.double().item() / total
    return epoch_loss, epoch_acc


def main():
     # Giới hạn số luồng CPU PyTorch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)

    # dataset đã chia sẵn
    data_dir = "dataset_cabbage_split"

    # nơi lưu model
    output_dir = os.path.join("runs", "cabbage")

    img_size = 224
    batch_size = 8
    epochs = 15
    lr = 1e-4

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    print("Device:", device)

    # augmentation cho train
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    # transform cho val/test
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    # đọc dataset
    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "train"),
        transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "val"),
        transform=val_transform
    )
    test_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "test"),
        transform=val_transform
    )

    print("Class mapping:", train_dataset.class_to_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # model
    model = get_model(num_classes=len(train_dataset.classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    start_time = time.time()

    # train
    for epoch in range(epochs):
        print(f"\n===== Epoch {epoch + 1}/{epochs} =====")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

            torch.save({
                "model_state_dict": best_model_wts,
                "class_names": train_dataset.classes,
                "model_name": "mobilenetv2",
                "img_size": img_size
            }, os.path.join(output_dir, "best_model.pth"))

            print("Đã lưu best_model.pth")

    # test
    model.load_state_dict(best_model_wts)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print(f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    print(f"Best Val Acc: {best_acc:.4f}")
    print(f"Thời gian train: {(time.time() - start_time)/60:.2f} phút")

    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": train_dataset.classes,
        "model_name": "mobilenetv2",
        "img_size": img_size
    }, os.path.join(output_dir, "last_model.pth"))

    print("Đã lưu last_model.pth")


if __name__ == "__main__":
    main()