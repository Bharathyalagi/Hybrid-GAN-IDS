import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

def train_decision_tree(X, y, params=None):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    if params is None:
        params = {
            "max_depth": 8,
            "min_samples_split": 4,
            "min_samples_leaf": 2,
        }

    model = DecisionTreeClassifier(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    print("✅ Classification Report:\n")
    print(classification_report(y_val, y_pred))

    return model, y_val, y_pred

def plot_confusion_matrix(y_true, y_pred, output_path="outputs/plots/dt_confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Decision Tree – Confusion Matrix")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"📊 Confusion matrix saved to: {output_path}")
