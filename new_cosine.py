import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
movies = pd.read_csv('ml-latest-small/movies.csv')
ratings = pd.read_csv('ml-latest-small/ratings.csv')

# Define genres and assign binary flags
genres = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 
          'Documentary', 'Drama', 'Fantasy', 'Horror', 'Mystery', 'Romance', 
          'Sci-Fi', 'Thriller', 'War', 'Western']

for genre in genres:
    movies[genre] = movies['genres'].str.contains(genre, regex=False).astype(int)

# Merge ratings with movies
df = ratings.merge(movies, on='movieId')

# Compute user average ratings per genre
user_genre_ratings = df.groupby(['userId'])[genres].mean()

# Standardize data for clustering
scaler = StandardScaler()
user_genre_ratings_scaled = scaler.fit_transform(user_genre_ratings)

### --- ORIGINAL K-MEANS (EUCLIDEAN DISTANCE) ---
kmeans_original = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters_original = kmeans_original.fit_predict(user_genre_ratings_scaled)
user_genre_ratings["Cluster_Original"] = clusters_original

### --- CUSTOM K-MEANS USING COSINE SIMILARITY ---
def custom_kmeans_cosine_similarity(X, n_clusters, max_iter=300, random_state=42):
    np.random.seed(random_state)
    initial_indices = np.random.choice(X.shape[0], n_clusters, replace=False)
    centers = X[initial_indices]

    for _ in range(max_iter):
        similarities = cosine_similarity(X, centers)  # shape: (n_samples, n_clusters)
        labels = np.argmax(similarities, axis=1)
        new_centers = np.array([X[labels == i].mean(axis=0) for i in range(n_clusters)])
        if np.allclose(centers, new_centers, atol=1e-4):
            break
        centers = new_centers

    return labels, centers

# Apply Cosine Similarity K-Means
chosen_k = 5
clusters_cosine, _ = custom_kmeans_cosine_similarity(user_genre_ratings_scaled, n_clusters=chosen_k)
user_genre_ratings["Cluster_Cosine"] = clusters_cosine

### --- SILHOUETTE SCORE CALCULATION ---
silhouette_original = silhouette_score(user_genre_ratings_scaled, clusters_original, metric='euclidean')
silhouette_cosine = silhouette_score(user_genre_ratings_scaled, clusters_cosine, metric='cosine')

print(f"Silhouette Score (Original K-Means - Euclidean): {silhouette_original:.4f}")
print(f"Silhouette Score (K-Means - Cosine Similarity): {silhouette_cosine:.4f}")

### --- OUTPUT USERS GROUPED BY CLUSTER ---
print("\n--- User Groupings by Cluster ---")

# For Original K-Means
print("\nOriginal K-Means (Euclidean):")
for cluster in sorted(user_genre_ratings["Cluster_Original"].unique()):
    users_in_cluster = user_genre_ratings[user_genre_ratings["Cluster_Original"] == cluster].index.tolist()
    print(f"Cluster {cluster}: Users {users_in_cluster}\n")

# For Custom K-Means (Cosine Similarity)
print("\nCustom K-Means (Cosine Similarity):")
for cluster in sorted(user_genre_ratings["Cluster_Cosine"].unique()):
    users_in_cluster = user_genre_ratings[user_genre_ratings["Cluster_Cosine"] == cluster].index.tolist()
    print(f"Cluster {cluster}: Users {users_in_cluster}\n")

### --- PCA FOR VISUALIZATION ---
pca = PCA(n_components=2)
pca_result = pca.fit_transform(user_genre_ratings_scaled)
user_genre_ratings["PCA1"] = pca_result[:, 0]
user_genre_ratings["PCA2"] = pca_result[:, 1]

### --- PLOT PCA SCATTERPLOTS WITH GENRE AXIS CONTRIBUTIONS ---
def plot_pca_clusters(data, pca_model, feature_names):
    # Get component loadings
    components = pd.DataFrame(pca_model.components_, columns=feature_names, index=["PC1", "PC2"])

    # Get top 5 contributing genres per PC
    top_pc1 = components.loc["PC1"].abs().sort_values(ascending=False).head(5).index.tolist()
    top_pc2 = components.loc["PC2"].abs().sort_values(ascending=False).head(5).index.tolist()

    # Create axis labels
    xlabel = f"{', '.join(top_pc1)}"
    ylabel = f"{', '.join(top_pc2)}"

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for cluster in sorted(data["Cluster_Original"].unique()):
        subset = data[data["Cluster_Original"] == cluster]
        axes[0].scatter(subset["PCA1"], subset["PCA2"], label=f'Cluster {cluster}', alpha=0.7)
    axes[0].set_title("Original K-Means (Euclidean)")
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    axes[0].legend()

    for cluster in sorted(data["Cluster_Cosine"].unique()):
        subset = data[data["Cluster_Cosine"] == cluster]
        axes[1].scatter(subset["PCA1"], subset["PCA2"], label=f'Cluster {cluster}', alpha=0.7)
    axes[1].set_title("K-Means (Cosine Similarity)")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel)
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    # Print full genre contributions
    print("\nFull Genre Contributions to PCA Components:")
    print(components.T.sort_values(by="PC1", ascending=False))

# Plot with genre contributions in axis labels
plot_pca_clusters(user_genre_ratings, pca, genres)
