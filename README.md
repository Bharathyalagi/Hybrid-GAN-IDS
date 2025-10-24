# Hybrid GAN IDS

This repository contains implementations of GAN-based synthetic data generation and classifiers for intrusion detection using NSL-KDD and UNSW-NB15 datasets.

## Folder Structure

- `CNN/` : CNN experiments for both datasets.
- `nsl-kdd/` : GAN + classifiers experiments for NSL-KDD dataset.
- `unsw-nb15/` : GAN + classifiers experiments for UNSW-NB15 dataset.

Refer to individual folder `README.md` files for dataset-specific instructions.

## Dataset Links

- NSL-KDD: [https://www.kaggle.com/datasets/defcom17/nslkdd](https://www.kaggle.com/datasets/defcom17/nslkdd)
- UNSW-NB15: [https://www.kaggle.com/datasets/amirrezaeian/unsw-nb15](https://www.kaggle.com/datasets/amirrezaeian/unsw-nb15)

---

## 📂 Repository Structure
```text
hybrid-gan-ids/
├── CNN/
│   ├── cnn.ipynb
│   └── README.md
│
├── nsl-kdd/
│   ├── acgan+decision-tree/
│   ├── wcgan+decision-tree/
│   ├── wcgan+xgboost/
│   └── README.md
│
├── unsw-nb15/
│   ├── acgan+decision-tree/
│   ├── wcgan+decision-tree/
│   ├── wcgan+xgboost/
│   └── README.md
│
└── README.md
```
---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

-----

### Note

This is a large project, so it might be confusing at first. Please check the `README.md` files in each folder carefully to understand the workflow and structure.

-----
# Thank you


