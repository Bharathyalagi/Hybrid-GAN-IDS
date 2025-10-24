# GAN(WCGAN + XGBoost) - Synthetic Data Generation & Classification for Intrusion Detection

This project implements a WCGAN-based approach for generating synthetic data to handle class imbalance in the NSL-KDD dataset, followed by classification using XGBoost.

## 🔧 Folder Structure
```
GAN(WCGAN + XG)/
│
├── data/
│   └── NSL-KDD/
│       ├── train_processed.csv
│       └── test_processed.csv
│
├── models/
│   ├── wcgan.py                # WCGAN model (generator + discriminator + trainer)
│   └── xgboost_classifier.py   # XGBoost classifier functions
│
├── outputs/
│   ├── synthetic_data.csv      # GAN-generated attack samples
│   ├── feature_columns.pkl     # Saved features for reloading
│   ├── xgb_model.json          # Saved XGBoost model
│   ├── wcgan_model.pth         # Saved WCGAN model
│   └── plots/
│       └── confusion_matrix_test.png
│       └── confusion_matrix.png
│
├── utils/
│   ├── preprocess.py
│   ├── metrics.py (optional)
│   └── data_loader.py (optional)
│
├── train_gan.py               # Train GAN on attacks
├── train_classifier.py        # Train classifier on real + synthetic
├── test_classifier.py         # Test the classifier
├── generate_normal.py         # Train GAN on normal samples (optional)
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Clone and Set Up Environment
```bash
cd GAN(WCGAN + XG)
python -m venv venv
venv\Scripts\activate        # On Windows
pip install -r requirements.txt
```

### 2. Training Models

#### Train GAN on Attack Samples
```bash
python train_gan.py
```
This will generate synthetic attack samples and save the GAN model.

#### Train Classifier on Real + Synthetic
```bash
python train_classifier.py
```
This will train an XGBoost model and save it to `outputs/xgb_model.json`.

### 3. Testing
```bash
python test_classifier.py
```
Evaluates on test data and saves the confusion matrix to `outputs/plots/`.

---

## 💾 Reloading Saved Models (NO Need to Retrain)

### XGBoost:
```python
from xgboost import XGBClassifier
model = XGBClassifier()
model.load_model("outputs/xgb_model.json")
```

### WCGAN Generator:
```python
from models.wcgan import Generator
import torch

generator = Generator()
generator.load_state_dict(torch.load("outputs/wcgan_model.pth"))
generator.eval()
```

### Load Saved Features:
```python
import pickle
with open("outputs/feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)
```

---

## 🔐 Back Up Your Project

To avoid retraining in the future:
1. **Zip the full folder**:
```bash
zip -r GAN_WCGAN_XG_full.zip "GAN(WCGAN + XG)/"
```
2. **Upload it to Google Drive / External HDD**

---

## 🧠 Summary of Capabilities
- Handles class imbalance using GAN-generated attack samples.
- Boosts classification accuracy significantly (up to 79%+).
- Can regenerate synthetic data on demand.
- Fully reloadable with saved models.

---

## 📈 Results vs Base Paper
| Metric            | Base Paper | Your Model (Best) |
|-------------------|------------|-------------------|
| Classifier Acc.   | 78.21%     | **79.20%**        |
| Attack F1-Score   | ~78%       | **80.07%**        |
| Normal F1-Score   | ~76%       | **78.24%**        |
| Total Workflow    | Limited    | **GAN + XGBoost** |

---

For any issues or future extensions, feel free to revisit this README.

to run again the project 
1. cd D:\GAN(WCGAN + XG)
2. .\venv\Scripts\activate
3. python train_classifier.py
4. python test_classifier.py
