
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Load Dataset
df = pd.read_csv(r"C:\Users\prajw\OneDrive\Desktop\AIMLL\california_housing.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# 2. Split Features and Target
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Define Models
models = {
    "Linear Regression": LinearRegression(),
    "Polynomial Regression": Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])
}

results = {}

# 4. Train and Evaluate Models
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    results[name] = {
        "MSE": mean_squared_error(y_test, pred),
        "MAE": mean_absolute_error(y_test, pred),
        "R2 Score": r2_score(y_test, pred)
    }

    print("\n", name)
    print("MSE:", round(results[name]["MSE"], 4))
    print("MAE:", round(results[name]["MAE"], 4))
    print("R2 Score:", round(results[name]["R2 Score"], 4))

    # Actual vs Predicted Graph
    plt.figure()
    plt.scatter(y_test, pred, alpha=0.4)
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], 'r')
    plt.xlabel("Actual House Value")
    plt.ylabel("Predicted House Value")
    plt.title(name)
    plt.show()

# 5. Model Comparison
comparison = pd.DataFrame(results).T
print("\nMODEL COMPARISON:")
print(comparison)

comparison.to_csv("regression_results.csv")

# 6. Performance Comparison Graph
comparison.plot(kind="bar", figsize=(9, 5))
plt.title("Linear vs Polynomial Regression")
plt.ylabel("Scores")
plt.xticks(rotation=0)
plt.grid(axis="y")
plt.tight_layout()
plt.show()

print("\nPROGRAM EXECUTED SUCCESSFULLY!")
