import torch
import cv2
import argparse
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
model.classifier[1] = torch.nn.Linear(1280, 2)
model.load_state_dict(torch.load("efficientnet_b0_best.pth", map_location=device))
model.to(device)
model.eval()

img = cv2.imread(args.image, 0)
img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

img = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    out = model(img)
    prob = torch.softmax(out, dim=1)
    pred = prob.argmax().item()

label = "REAL" if pred == 0 else "FAKE"
print(f"Prediction: {label}  | Confidence: {prob[0][pred].item():.4f}")
