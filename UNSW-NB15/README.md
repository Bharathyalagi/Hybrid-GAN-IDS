# UNSW-NB15 GAN Experiments

## Folder Structure

- `acgan+decision-tree/`
- `wcgan+decision-tree/`
- `wcgan+xgboost/`

## Instructions (example for any folder)

1. Navigate to folder:
    ```bash
    cd unsw-nb15/acgan+decision-tree
    ```
2. Activate virtual environment:
    ```bash
    python -m venv venv
    .\venv\Scripts\activate       # Windows
    source venv/bin/activate      # Linux/Mac
    ```
3. Run GAN model:
    ```bash
    python acgan.py
    ```
4. Preprocess data:
    ```bash
    python utils/featurecleaner.py
    python utils/preprocess.py
    ```
5. Generate synthetic data:
    ```bash
    python generate_data_normal.py
    ```
6. Train classifier:
    ```bash
    python train_classifier.py
    ```
7. Test classifier:
    ```bash
    python test_classifier.py
    ```

## Dataset Link
[UNSW-NB15](https://www.kaggle.com/datasets/amirrezaeian/unsw-nb15)

## note: Create a folder named data, in that paste the dataset folder and files
