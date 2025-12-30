# Indian Bird Species Image Classification (Custom CNN vs Transfer Learning)

**Dataset (Kaggle):** https://www.kaggle.com/datasets/arjunbasandrai/25-indian-bird-species-with-226k-images

This project tackles **fine-grained image classification** across **25 Indian bird species** using two approaches:
1. A **custom deep CNN trained from scratch** (no transfer learning / no recurrent layers).
2. A **transfer learning model** using a pretrained **ResNet50** backbone.

The goal is to compare performance and practical tradeoffs on a real-world style dataset with high variation in lighting, pose, background clutter, and visually similar classes. :contentReference[oaicite:0]{index=0}

The .ipynb file is available, but viewing and running through Google Colab is preferred:
[https://colab.research.google.com/drive/1XzRK75a2yZccgEI2BU2qc16uEt7bvaul?usp=sharing]

---

## Highlights

- **Input formatting:** Images resized to **128×128** and normalized to **[0, 1]**. :contentReference[oaicite:1]{index=1}  
- **Regularization used:** Batch Normalization + Dropout; plus data augmentation (rotation/zoom/horizontal flip in the notebook).
- **Metrics tracked (train + validation):** **Accuracy** and **Loss** for both models. :contentReference[oaicite:2]{index=2}

---

## Model 1 — Custom Deep CNN (trained from scratch)

### Architecture (meets “5 conv + ≥3 pooling” requirement)
- **5× Conv2D layers** with increasing filters (up to 256)
- **BatchNormalization after each conv block**
- **4× MaxPooling2D layers** (≥3 pooling requirement satisfied)
- **Dense(256)** → **Dropout(0.5)** → **Softmax(25 classes)** :contentReference[oaicite:3]{index=3}

### Training setup
- Optimizer: **Adam**
- Loss: **Sparse Categorical Cross-Entropy** (multi-class, integer labels)
- Trained for **20 epochs** :contentReference[oaicite:4]{index=4}

### Results
- Validation (used as test): **Accuracy = 71.62%**, **Loss = 1.1559** :contentReference[oaicite:5]{index=5}  
- Training accuracy reached **>75%** by the end, with validation tracking closely (regularization helped avoid major overfitting). :contentReference[oaicite:6]{index=6}

---

## Model 2 — Transfer Learning (ResNet50)

### Approach
- Pretrained **ResNet50** (ImageNet) used as a **frozen feature extractor**
- Custom head:
  - Global Average Pooling
  - Dense(256, ReLU)
  - Softmax(25 classes)
- No fine-tuning performed (pretrained layers frozen). :contentReference[oaicite:7]{index=7}

### Results
- Final accuracy: **18.83%** after **10 epochs**, only slightly above random guessing (1/25 = 4%). :contentReference[oaicite:8]{index=8}  
- Report discussion notes this setup likely underperformed because ResNet50 typically expects **~224×224 input** and freezing the backbone prevented specialization to subtle inter-species differences. :contentReference[oaicite:9]{index=9}

---

## Error patterns (custom CNN)

Most misclassifications occurred when:
- Species had very similar colors/silhouettes
- Birds were small in-frame or partially occluded
- Background foliage was dense or visually noisy :contentReference[oaicite:10]{index=10}

---

## Repository contents

- `MLFA25_HW3_RPuriCPSC393.ipynb` — End-to-end notebook:
  - Data loading + augmentation
  - Custom CNN training + curves (loss/accuracy)
  - Transfer learning model training + curves
  - Evaluation + example misclassifications
- `MFA25 - Assignment3.pdf` — Technical report with methodology, learning curves, and results :contentReference[oaicite:11]{index=11}

---

## Run in Google Colab

After you upload this repo to GitHub, open the notebook here:
https://colab.research.google.com/drive/1XzRK75a2yZccgEI2BU2qc16uEt7bvaul?usp=sharing


### Dataset setup (Colab)
This dataset is large. Typical workflow:
1. Download via Kaggle (API) or manually
2. Ensure directory structure matches `flow_from_directory` format (one folder per class)
3. Update the notebook paths to point to your extracted dataset directory

---

## Tools & Libraries

- Python (Google Colab)
- TensorFlow / Keras
- NumPy, Matplotlib
