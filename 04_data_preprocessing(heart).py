
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 1. Load Dataset
df = pd.read_csv(r"C:\Users\prajw\OneDrive\Desktop\AIMLL\heart.csv")

print("\nFirst Five Records:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nDataset Shape:", df.shape)
print("\nData Types:\n", df.dtypes)

# 2. Data Cleaning
print("\nMissing Values:\n", df.isnull().sum())
print("\nDuplicate Records:", df.duplicated().sum())

df.drop_duplicates(inplace=True)

# Handle Missing Values
for col in df.select_dtypes(include=np.number).columns:
    df[col] = df[col].fillna(df[col].median())

for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing Values After Cleaning:\n", df.isnull().sum())

# 3. Statistical Analysis
print("\nStatistical Description:\n", df.describe())
print("\nMean:\n", df.mean(numeric_only=True))
print("\nMedian:\n", df.median(numeric_only=True))
print("\nStandard Deviation:\n", df.std(numeric_only=True))
print("\nCorrelation:\n", df.corr(numeric_only=True))

# 4. Encoding
for col in df.select_dtypes(include='object').columns:
    df[col] = LabelEncoder().fit_transform(df[col])

print("\nDataset After Encoding:\n", df.head())

# 5. Separate Features and Target
X = df.drop('target', axis=1)
y = df['target']

# 6. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=56
)

# 7. Feature Scaling
scaler = StandardScaler()
X_train = pd.DataFrame(
    scaler.fit_transform(X_train), columns=X.columns
)
X_test = pd.DataFrame(
    scaler.transform(X_test), columns=X.columns
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# 8. Target Distribution
print("\nTarget Counts:\n", y.value_counts())

# 9. Histograms
for col in ['age', 'chol']:
    plt.figure()
    plt.hist(df[col], bins=10, edgecolor='black')
    plt.title(col + " Distribution")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()

# 10. Boxplots
for col in ['age', 'chol']:
    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(col + " Boxplot")
    plt.show()

# 11. Correlation Heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

print("\nPROGRAM EXECUTED SUCCESSFULLY!")
