
# Deepfake Iris Detection – Project README

## Project Overview
This project focuses on detecting **real vs fake iris images** using a deep learning model trained on a custom-preprocessed balanced dataset. The final model uses **EfficientNet-B0**, achieving **99.97% test accuracy**.

This README explains:
- Folder structure
- How to run training, testing, and inference
- Requirements
- Input files needed
- Model execution workflow
- Team details placeholder



## Environment Requirements
Install the following in Google Colab / local environment:

```bash
pip install torch torchvision opencv-python numpy tqdm
```

---

## How to Run the Project


### **Train the Model**
Run:
```bash
python train.py
```

This will:
- Load dataset from `dataset_split/`
- Train EfficientNet‑B0 for 10 epochs
- Save best model to:
```
efficientnet_b0_best.pth
```

### Test the Model
Run:
```bash
python test.py
```

Produces:
- Accuracy
- Precision / Recall
- F1‑score

Expected test accuracy:
```
Test Accuracy: 99.97%
```

---

##  Inference (Single Image Prediction)

Run:
```bash
python inference.py --image your_image.jpg
```

Output example:
```
Prediction: REAL (99.2% confidence)
```

You can upload any image using Colab:
```python
from google.colab import files
uploaded = files.upload()
```

---


- The model can be tested immediately using `test.py`.
- Inference script works with **any iris image**, not only dataset images.
- All required files are included in the ZIP submission.

---

## 👥 Team Details
**Name:** Shreya Talod  
**Roll No:** (Add your roll number here)




