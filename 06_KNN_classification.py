
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

# 1. Load Dataset
df = pd.read_csv(r"C:\Users\prajw\OneDrive\Desktop\AIMLL\Breast cancer dataset.csv")

print("First 5 Records:\n", df.head())
print("\nDataset Shape:", df.shape)

# 2. Data Preprocessing
df = df.dropna(axis=1, how="all")
df = df.drop(columns=["id"], errors="ignore")
df = df.dropna()

df["diagnosis"] = df["diagnosis"].map({"B": 0, "M": 1})

X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Classification Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    # Evaluation Metrics
    results[name] = [
        accuracy_score(y_test, pred),
        precision_score(y_test, pred, zero_division=0),
        recall_score(y_test, pred, zero_division=0),
        f1_score(y_test, pred, zero_division=0)
    ]

    print("\n", name)
    print("Accuracy:", round(results[name][0], 4))
    print("Precision:", round(results[name][1], 4))
    print("Recall:", round(results[name][2], 4))
    print("F1 Score:", round(results[name][3], 4))

    # Confusion Matrix
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Benign", "Malignant"],
                yticklabels=["Benign", "Malignant"])
    plt.title(name + " Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# 6. Model Comparison
comparison = pd.DataFrame(
    results, index=["Accuracy", "Precision", "Recall", "F1 Score"]
)

print("\nMODEL COMPARISON:")
print(comparison)

# 7. Performance Graph
comparison.plot(kind="bar", figsize=(9, 5))
plt.title("Logistic Regression vs KNN")
plt.ylabel("Score")
plt.ylim(0, 1.1)
plt.xticks(rotation=0)
plt.grid(axis="y")
plt.tight_layout()
plt.show()

print("\nPROGRAM EXECUTED SUCCESSFULLY!")
