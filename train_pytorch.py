import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os

# ── CONFIG ────────────────────────────────────────────────────────────────────
EPOCHS     = 30
BATCH_SIZE = 64
LR         = 0.001
MODEL_PATH = "model/cnn_cifar10.pt"
os.makedirs("model", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── DATA ──────────────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

print("Downloading CIFAR-10...")
full_train = torchvision.datasets.CIFAR10(root="./data", train=True,
                                           download=True, transform=train_transform)
test_set   = torchvision.datasets.CIFAR10(root="./data", train=False,
                                           download=True, transform=test_transform)

val_size   = 5000
train_size = len(full_train) - val_size
train_set, val_set = random_split(full_train, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

# ── MODEL ─────────────────────────────────────────────────────────────────────
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout(0.25),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout(0.25),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model = CNN().to(device)

# ── TRAINING ──────────────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5,
                                                   patience=4, verbose=True)

best_val_acc = 0.0

print("\nStarting training...")
for epoch in range(EPOCHS):
    # Train
    model.train()
    train_loss, train_correct, total = 0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss    += loss.item()
        train_correct += (outputs.argmax(1) == labels).sum().item()
        total         += labels.size(0)

    train_acc = train_correct / total

    # Validate
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs     = model(images)
            val_correct += (outputs.argmax(1) == labels).sum().item()
            val_total   += labels.size(0)

    val_acc = val_correct / val_total
    scheduler.step(1 - val_acc)

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
          f"Train Acc: {train_acc*100:.2f}% | "
          f"Val Acc: {val_acc*100:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"  ✅ Best model saved! Val Acc: {val_acc*100:.2f}%")

# ── TEST ──────────────────────────────────────────────────────────────────────
print("\nEvaluating on test set...")
best_model = CNN().to(device)
best_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
best_model.eval()

test_correct, test_total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels  = images.to(device), labels.to(device)
        outputs         = best_model(images)
        test_correct   += (outputs.argmax(1) == labels).sum().item()
        test_total     += labels.size(0)

print(f"\n✅ Test Accuracy: {test_correct/test_total*100:.2f}%")
print(f"Model saved to {MODEL_PATH}")