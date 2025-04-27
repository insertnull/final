import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import LocalOutlierFactor

# Load datasets
movies = pd.read_csv('ml-latest-small/movies.csv')
ratings = pd.read_csv('ml-latest-small/ratings.csv')

# Define genres
genres = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
          'Documentary', 'Drama', 'Fantasy', 'Horror', 'Mystery', 'Romance',
          'Sci-Fi', 'Thriller', 'War', 'Western']

# Assign binary genre flags
for genre in genres:
    movies[genre] = movies['genres'].str.contains(genre, regex=False).astype(int)

# Merge ratings with genre information
df = ratings.merge(movies, on='movieId')

# Compute user average rating per genre
user_genre_ratings = df.groupby('userId')[genres].mean()

# Standardize data
scaler = StandardScaler()
user_genre_ratings_scaled = scaler.fit_transform(user_genre_ratings)

### --- IDENTIFY OUTLIERS USING LOF --- ###
lof = LocalOutlierFactor(n_neighbors=50, metric="euclidean", contamination=0.05)
outlier_labels = lof.fit_predict(user_genre_ratings_scaled)
outlier_scores = -lof.negative_outlier_factor_
outliers = outlier_labels == -1
user_genre_ratings["Outlier"] = outliers

print(f"\nOutliers identified: {np.sum(outliers)}")
print(f"Inliers: {np.sum(~outliers)}")

### --- ELBOW METHOD TO FIND OPTIMAL k --- ###
inertia_values = []
k_range = range(2, 16)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(user_genre_ratings_scaled)
    inertia_values.append(kmeans.inertia_)

# Plot elbow curve
plt.figure(figsize=(8, 6))
plt.plot(k_range, inertia_values, marker='o')
plt.title('Elbow Method for Optimal k (Original K-Means)')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.grid(True)
plt.tight_layout()
plt.show()

### --- MANUAL CLUSTERING BASED ON CHOSEN k --- ###
manual_k = int(input("Enter the desired number of clusters based on the elbow method: "))
kmeans_manual = KMeans(n_clusters=manual_k, random_state=42, n_init=10)
clusters_manual = kmeans_manual.fit_predict(user_genre_ratings_scaled)
user_genre_ratings["Cluster_Original"] = clusters_manual

### --- PCA VISUALIZATION WITH OUTLIERS AND CENTROIDS --- ###
def pca_visualization_with_outliers(data_scaled, labels, outliers, centers_scaled, title, feature_names):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(data_scaled)
    reduced_centers = pca.transform(centers_scaled)  # transform centroids too

    components = pd.DataFrame(pca.components_, columns=feature_names, index=['PC1', 'PC2'])
    top_pc1 = components.loc['PC1'].abs().sort_values(ascending=False).head(5).index.tolist()
    top_pc2 = components.loc['PC2'].abs().sort_values(ascending=False).head(5).index.tolist()

    xlabel = f"{', '.join(top_pc1)}"
    ylabel = f"{', '.join(top_pc2)}"

    plt.figure(figsize=(8, 6))
    for cluster in np.unique(labels):
        mask = (labels == cluster) & (~outliers)
        plt.scatter(reduced[mask, 0], reduced[mask, 1], label=f'Cluster {cluster}', alpha=0.7)

    # Plot outliers
    plt.scatter(reduced[outliers, 0], reduced[outliers, 1], color='red', marker='x', label='Outliers', s=70)

    # Plot centroids as black dots
    plt.scatter(reduced_centers[:, 0], reduced_centers[:, 1],
                color='black', marker='o', s=100, label='Centroids')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\nGenre Contributions to PCA Components:")
    print(components.T.sort_values(by="PC1", ascending=False))

# Plot with centroid markers
centers_manual = kmeans_manual.cluster_centers_
pca_visualization_with_outliers(user_genre_ratings_scaled, clusters_manual, outliers,
                                 centers_manual, f"K-Means (k={manual_k})", genres)

# Silhouette score for evaluation
sil_score = silhouette_score(user_genre_ratings_scaled, clusters_manual)
print(f"\nSilhouette Score (k={manual_k}): {sil_score:.4f}")

# Summary
print("\nUser Clustering Summary:")
print("  Total users:", user_genre_ratings.shape[0])
print("  Manual k chosen:", manual_k)
print("  Outliers identified (LOF):", np.sum(outliers))
