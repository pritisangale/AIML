
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv(r"C:\Users\prajw\OneDrive\Desktop\AIMLL\Wine.csv")
df.columns = df.columns.str.strip()
df = df[df["class"].astype(str).str.strip() != "class"]
df = df.apply(pd.to_numeric, errors="coerce").dropna()

# Separate features and target
X = df.drop("class", axis=1)
y = df["class"]

print("Features before PCA:", X.shape[1])
print("\nDataset Shape:", df.shape)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Classification without PCA
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)
pred1 = model.predict(X_test)
acc1 = accuracy_score(y_test, pred1)

# Apply PCA (95% variance)
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print("\nFeatures after PCA:", X_train_pca.shape[1])
print("Variance retained:", round(pca.explained_variance_ratio_.sum(), 4))

# Classification with PCA
model.fit(X_train_pca, y_train)
pred2 = model.predict(X_test_pca)
acc2 = accuracy_score(y_test, pred2)

print("\nClassification Report:")
print(classification_report(y_test, pred2))

# Graph 1: PCA Visualization
pca2 = PCA(n_components=2)
X_2d = pca2.fit_transform(X_train)
y_plot = y_train.reset_index(drop=True)

plt.figure(figsize=(7, 5))

for c in sorted(y_plot.unique()):
    plt.scatter(X_2d[y_plot == c, 0],
                X_2d[y_plot == c, 1], label=f"Class {c}")

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Visualization")
plt.legend()
plt.grid()
plt.show()

# Graph 2: Explained Variance
pca_all = PCA().fit(X_train)
variance = pca_all.explained_variance_ratio_.cumsum()

plt.figure(figsize=(7, 5))
plt.plot(range(1, len(variance)+1), variance, marker="o")
plt.axhline(0.95, color="red", linestyle="--", label="95% Variance")

plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")
plt.legend()
plt.grid()
plt.show()

# Final Comparison
print("\n========== FINAL COMPARISON ==========")
print("Features before PCA:", X.shape[1])
print("Features after PCA:", X_train_pca.shape[1])
print("Accuracy without PCA:", round(acc1, 4))
print("Accuracy with PCA:", round(acc2, 4))
