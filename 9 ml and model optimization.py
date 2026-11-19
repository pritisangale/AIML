
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv(r"C:\Users\prajw\OneDrive\Desktop\AIMLL\Breast cancer dataset.csv")
df.columns = df.columns.str.strip().str.lower()
df = df.drop(columns=["id"], errors="ignore")

# Data preprocessing
df["diagnosis"] = df["diagnosis"].astype(str).str.strip().str.upper()
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

features = ["radius_mean", "texture_mean", "perimeter_mean",
            "area_mean", "smoothness_mean"]

X = df[features].apply(pd.to_numeric, errors="coerce")
y = df["diagnosis"]

data = pd.concat([X, y], axis=1).dropna()
X, y = data[features], data["diagnosis"].astype(int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define models
models = {
    "SVM": Pipeline([("scaler", StandardScaler()), ("classifier", SVC())]),
    "Random Forest": RandomForestClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42)
}

parameters = {
    "SVM": {
        "classifier__C": [0.1, 1, 10],
        "classifier__kernel": ["linear", "rbf"]
    },
    "Random Forest": {
        "n_estimators": [50, 100],
        "max_depth": [None, 5, 10]
    },
    "AdaBoost": {
        "n_estimators": [50, 100],
        "learning_rate": [0.1, 1.0]
    }
}

baseline = {}
optimized = {}

# Baseline performance and cross validation
print("\nBASELINE PERFORMANCE")

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    baseline[name] = accuracy_score(y_test, pred)

    scores = cross_val_score(model, X_train, y_train, cv=5)

    print("\n", name)
    print("Accuracy:", round(baseline[name], 4))
    print("5-Fold CV Accuracy:", round(scores.mean(), 4))

# Grid Search optimization
print("\nOPTIMIZED PERFORMANCE")

for name, model in models.items():
    grid = GridSearchCV(model, parameters[name], cv=5,
                        scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)

    pred = grid.predict(X_test)
    optimized[name] = accuracy_score(y_test, pred)

    print("\n", name)
    print("Best Parameters:", grid.best_params_)
    print("Best CV Accuracy:", round(grid.best_score_, 4))
    print("Optimized Test Accuracy:", round(optimized[name], 4))

    print("\nClassification Report:")
    print(classification_report(
        y_test, pred, target_names=["Benign", "Malignant"],
        zero_division=0
    ))

# Final comparison
print("\n========== FINAL COMPARISON ==========")
print("Algorithm\tBaseline\tOptimized")

for name in models:
    print(f"{name}\t{baseline[name]:.4f}\t\t{optimized[name]:.4f}")

print("\nPROGRAM EXECUTED SUCCESSFULLY")
