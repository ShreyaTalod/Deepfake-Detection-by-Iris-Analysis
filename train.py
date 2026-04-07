import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import os
import cv2
from tqdm import tqdm

DATA_ROOT = "dataset_split/train"
VAL_ROOT = "dataset_split/val"

class IrisDataset(Dataset):
    def __init__(self, root):
        self.paths = []
        self.labels = []

        for label, cls in enumerate(["real", "fake"]):
            folder = os.path.join(root, cls)
            for f in os.listdir(folder):
                self.paths.append(os.path.join(folder, f))
                self.labels.append(label)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx], cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = self.transform(img)
        return img, self.labels[idx]


train_set = IrisDataset(DATA_ROOT)
val_set = IrisDataset(VAL_ROOT)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", device)

model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
model.classifier[1] = nn.Linear(1280, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

best_acc = 0

for epoch in range(10):
    model.train()
    total_loss = 0

    for imgs, labels in tqdm(train_loader):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            _, preds = torch.max(out, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total * 100
    print(f"Epoch {epoch+1}/10 — Loss: {total_loss:.4f}, Val Acc: {acc:.2f}%")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "efficientnet_b0_best.pth")
        print("✔ Best model updated!")
