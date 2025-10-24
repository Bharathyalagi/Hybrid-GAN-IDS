# 🧠 Hybrid GAN-Based Synthetic Data Generation for Intrusion Detection Systems

This repository contains my M.Tech research implementation on **GAN-based synthetic data generation** for **Intrusion Detection Systems (IDS)** using two benchmark datasets:
- **NSL-KDD**
- **UNSW-NB15**

The project explores the use of **ACGAN** and **WCGAN** models combined with multiple classifiers (Decision Tree, XGBoost, and CNN) to address **class imbalance** and improve attack detection performance.

---

## 📂 Repository Structure
hybrid-gan-ids

│
├── CNN

│ ├── cnn.ipynb

│ └── README.md

│
├── nsl-kdd

│ ├── acgan+decision-tree

│ ├── wcgan+decision-tree

│ ├── wcgan+xgboost

│ └── README.md

│
├── unsw-nb15

│ ├── acgan+decision-tree

│ ├── wcgan+decision-tree

│ ├── wcgan+xgboost

│ └── README.md

│
└── README.md
