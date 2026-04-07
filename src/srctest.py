import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import cv2
from sklearn.metrics import classification_report, confusion_matrix

TEST_ROOT = "dataset_split/test"

class IrisDataset:
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
        img = cv2.imread(self.paths[idx], 0)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = self.transform(img)
        return img, self.labels[idx]


dataset = IrisDataset(TEST_ROOT)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
model.classifier[1] = nn.Linear(1280, 2)
model.load_state_dict(torch.load("efficientnet_b0_best.pth", map_location=device))
model = model.to(device)
model.eval()

y_true, y_pred = [], []

with torch.no_grad():
    for imgs, labels in loader:
        imgs = imgs.to(device)
        out = model(imgs)
        _, preds = torch.max(out, 1)
        y_true.extend(labels.tolist())
        y_pred.extend(preds.cpu().tolist())

print(classification_report(y_true, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
