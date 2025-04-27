import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.decomposition import PCA

# Load datasets
movies = pd.read_csv('ml-latest-small/movies.csv')
ratings = pd.read_csv('ml-latest-small/ratings.csv')

# Extract genres
genres = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 
          'Documentary', 'Drama', 'Fantasy', 'Horror', 'Mystery', 'Romance', 
          'Sci-Fi', 'Thriller', 'War', 'Western']

# Assign binary genre flags
for genre in genres:
    movies[genre] = movies['genres'].str.contains(genre, regex=False).astype(int)

# Merge ratings with movies
df = ratings.merge(movies, on='movieId')

# Compute user average ratings per genre
user_genre_ratings = df.groupby(['userId'])[genres].mean()

# Standardize data for clustering
scaler = StandardScaler()
user_genre_ratings_scaled = scaler.fit_transform(user_genre_ratings)

### ELBOW METHOD FOR ORIGINAL K-MEANS
inertias = []
k_values = range(2, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(user_genre_ratings_scaled)
    inertias.append(kmeans.inertia_)

# Plot elbow graph
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertias, marker='o')
plt.title('Elbow Method for Original K-Means')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

### ORIGINAL K-MEANS (EUCLIDEAN DISTANCE)
kmeans_original = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters_original = kmeans_original.fit_predict(user_genre_ratings_scaled)
user_genre_ratings["Cluster_Original"] = clusters_original

### CALINSKI-HARABASZ INDEX FOR ENHANCED K-MEANS
ch_scores = []
k_range = range(2, 11)
best_k = None
best_score = -np.inf

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(user_genre_ratings_scaled)
    score = calinski_harabasz_score(user_genre_ratings_scaled, labels)
    ch_scores.append(score)
    
    if score > best_score:
        best_k = k
        best_score = score

# Plot CH Index graph
plt.figure(figsize=(8, 5))
plt.plot(k_range, ch_scores, marker='o')
plt.title('Calinski-Harabasz Index for Enhanced K-Means')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('CH Index Score')
plt.grid(True)
plt.show()

# Apply Enhanced K-Means with best k
kmeans_enhanced = KMeans(n_clusters=best_k, random_state=42, n_init=10)
clusters_enhanced = kmeans_enhanced.fit_predict(user_genre_ratings_scaled)
user_genre_ratings["Cluster_Enhanced"] = clusters_enhanced

### --- PCA FOR VISUALIZATION ---
pca = PCA(n_components=2)
pca_transformed = pca.fit_transform(user_genre_ratings_scaled)
user_genre_ratings["PCA1"] = pca_transformed[:, 0]
user_genre_ratings["PCA2"] = pca_transformed[:, 1]

# Get top contributing genres per component
pca_components = pd.DataFrame(np.abs(pca.components_), columns=genres)
top5_pca1 = pca_components.loc[0].sort_values(ascending=False).head(5).index.tolist()
top5_pca2 = pca_components.loc[1].sort_values(ascending=False).head(5).index.tolist()

### --- PLOT SIDE-BY-SIDE SCATTERPLOTS ---
def plot_side_by_side():
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Original K-Means Scatterplot
    ax = axes[0]
    for cluster in sorted(user_genre_ratings["Cluster_Original"].unique()):
        subset = user_genre_ratings[user_genre_ratings["Cluster_Original"] == cluster]
        ax.scatter(subset["PCA1"], subset["PCA2"], label=f'Cluster {cluster}', alpha=0.7)
    ax.set_xlabel(f'PCA 1: {", ".join(top5_pca1)}')
    ax.set_ylabel(f'PCA 2: {", ".join(top5_pca2)}')
    ax.set_title('Original K-Means - PCA')
    ax.legend()

    # Enhanced K-Means Scatterplot
    ax = axes[1]
    for cluster in sorted(user_genre_ratings["Cluster_Enhanced"].unique()):
        subset = user_genre_ratings[user_genre_ratings["Cluster_Enhanced"] == cluster]
        ax.scatter(subset["PCA1"], subset["PCA2"], label=f'Cluster {cluster}', alpha=0.7)
    ax.set_xlabel(f'PCA 1: {", ".join(top5_pca1)}')
    ax.set_ylabel(f'PCA 2: {", ".join(top5_pca2)}')
    ax.set_title(f'Enhanced K-Means - PCA (k={best_k})')
    ax.legend()

    plt.tight_layout()
    plt.show()

plot_side_by_side()

### --- SILHOUETTE SCORES ---
silhouette_original = silhouette_score(user_genre_ratings_scaled, user_genre_ratings["Cluster_Original"])
silhouette_enhanced = silhouette_score(user_genre_ratings_scaled, user_genre_ratings["Cluster_Enhanced"])

print(f"Silhouette Score for Original K-Means: {silhouette_original:.4f}")
print(f"Silhouette Score for Enhanced K-Means (Optimal k={best_k}): {silhouette_enhanced:.4f}")
