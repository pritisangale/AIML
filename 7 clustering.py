
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# 1. Load Dataset
data = pd.read_csv(r"C:\Users\prajw\OneDrive\Desktop\AIMLL\Mall_Customers.csv")

print("First 5 Records:\n", data.head())

# 2. Select Features
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# 3. Elbow Method
wcss = []

for k in range(1, 11):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X)
    wcss.append(model.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

# 4. K-Means Clustering
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
data['KMeans_Cluster'] = kmeans.fit_predict(X)

# 5. Hierarchical Clustering and Dendrogram
linked = linkage(X, method='ward')

plt.figure(figsize=(10, 5))
dendrogram(linked)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Distance")
plt.show()

hierarchical = AgglomerativeClustering(n_clusters=5, linkage='ward')
data['Hierarchical_Cluster'] = hierarchical.fit_predict(X)

# 6. Cluster Visualization
for col, title in [
    ('KMeans_Cluster', 'K-Means Clustering'),
    ('Hierarchical_Cluster', 'Hierarchical Clustering')
]:
    plt.figure(figsize=(8, 5))
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=data[col], s=50)
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.title(title)
    plt.show()

# Display K-Means Centroids
plt.figure(figsize=(8, 5))
plt.scatter(X.iloc[:, 0], X.iloc[:, 1],
            c=data['KMeans_Cluster'], s=50)
plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            marker='X', s=200, label='Centroids')
plt.title("K-Means Centroids")
plt.legend()
plt.show()

# 7. Cluster Analysis
print("\nCluster Results:")
print(data[['CustomerID', 'KMeans_Cluster', 'Hierarchical_Cluster']].head(20))

print("\nK-Means Cluster Counts:")
print(data['KMeans_Cluster'].value_counts())

print("\nHierarchical Cluster Counts:")
print(data['Hierarchical_Cluster'].value_counts())

print("\nPROGRAM EXECUTED SUCCESSFULLY!")
