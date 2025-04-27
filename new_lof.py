import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import silhouette_score
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

# Make a copy before modifying
user_genre_ratings_original = user_genre_ratings.copy()

# Standardize data for clustering
scaler = StandardScaler()
user_genre_ratings_scaled = scaler.fit_transform(user_genre_ratings)

### --- ORIGINAL K-MEANS (EUCLIDEAN DISTANCE) ---
kmeans_original = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters_original = kmeans_original.fit_predict(user_genre_ratings_scaled)
user_genre_ratings_original["Cluster_Original"] = clusters_original

### --- LOF OUTLIER REMOVAL ---
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
outliers_lof = lof.fit_predict(user_genre_ratings_scaled)

# Keep only inliers
user_genre_ratings_filtered = user_genre_ratings[outliers_lof == 1]
user_genre_ratings_filtered_scaled = scaler.fit_transform(user_genre_ratings_filtered)

### --- ENHANCED K-MEANS (ONLY LOF FOR ENHANCEMENT) ---
kmeans_enhanced = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters_enhanced = kmeans_enhanced.fit_predict(user_genre_ratings_filtered_scaled)
user_genre_ratings_filtered["Cluster_Enhanced"] = clusters_enhanced

### --- PCA FOR VISUALIZATION ---
pca = PCA(n_components=2)

# Apply PCA on the original data (before LOF removal)
pca_original_model = pca.fit(user_genre_ratings_scaled)
pca_original = pca_original_model.transform(user_genre_ratings_scaled)
user_genre_ratings_original["PCA1"] = pca_original[:, 0]
user_genre_ratings_original["PCA2"] = pca_original[:, 1]

# Apply PCA on the filtered data (after LOF removal)
pca_enhanced_model = pca.fit(user_genre_ratings_filtered_scaled)
pca_enhanced = pca_enhanced_model.transform(user_genre_ratings_filtered_scaled)
user_genre_ratings_filtered["PCA1"] = pca_enhanced[:, 0]
user_genre_ratings_filtered["PCA2"] = pca_enhanced[:, 1]

### --- FUNCTION TO EXTRACT TOP GENRES CONTRIBUTING TO EACH PCA AXIS ---
def get_top_genres(pca_model, feature_names, axis=0, top_n=5):
    component = pca_model.components_[axis]
    abs_component = np.abs(component)
    top_indices = np.argsort(abs_component)[::-1][:top_n]
    top_genres = [feature_names[i] for i in top_indices]
    return top_genres

### --- PLOT SIDE-BY-SIDE SCATTERPLOTS WITH GENRE AXES ---
def plot_side_by_side(pca_model_orig, pca_model_enh, feature_names):
    top_pc1_orig = get_top_genres(pca_model_orig, feature_names, axis=0)
    top_pc2_orig = get_top_genres(pca_model_orig, feature_names, axis=1)
    top_pc1_enh = get_top_genres(pca_model_enh, feature_names, axis=0)
    top_pc2_enh = get_top_genres(pca_model_enh, feature_names, axis=1)

    xlabel_orig = f"{', '.join(top_pc1_orig)}"
    ylabel_orig = f"{', '.join(top_pc2_orig)}"
    xlabel_enh = f"{', '.join(top_pc1_enh)}"
    ylabel_enh = f"{', '.join(top_pc2_enh)}"

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Original K-Means Scatterplot
    ax = axes[0]
    for cluster in sorted(user_genre_ratings_original["Cluster_Original"].unique()):
        subset = user_genre_ratings_original[user_genre_ratings_original["Cluster_Original"] == cluster]
        ax.scatter(subset["PCA1"], subset["PCA2"], label=f'Cluster {cluster}', alpha=0.7)
    
    # Plot centroids for original clusters (black circles)
    centroids_orig = kmeans_original.cluster_centers_
    pca_centroids_orig = pca_original_model.transform(centroids_orig)
    ax.scatter(pca_centroids_orig[:, 0], pca_centroids_orig[:, 1], color='black', s=100, marker='o', label='Centroids')
    
    ax.set_xlabel(xlabel_orig)
    ax.set_ylabel(ylabel_orig)
    ax.set_title('Original K-Means')
    ax.legend()

    # Enhanced K-Means (with LOF) Scatterplot
    ax = axes[1]
    for cluster in sorted(user_genre_ratings_filtered["Cluster_Enhanced"].unique()):
        subset = user_genre_ratings_filtered[user_genre_ratings_filtered["Cluster_Enhanced"] == cluster]
        ax.scatter(subset["PCA1"], subset["PCA2"], label=f'Cluster {cluster}', alpha=0.7)

    # Plot centroids for enhanced clusters (black circles)
    centroids_enhanced = kmeans_enhanced.cluster_centers_
    pca_centroids_enhanced = pca_enhanced_model.transform(centroids_enhanced)
    ax.scatter(pca_centroids_enhanced[:, 0], pca_centroids_enhanced[:, 1], color='black', s=100, marker='o', label='Centroids')

    ax.set_xlabel(xlabel_enh)
    ax.set_ylabel(ylabel_enh)
    ax.set_title('Enhanced K-Means (LOF)')
    ax.legend()

    plt.tight_layout()
    plt.show()

# Plot PCA with genre axis contributions
plot_side_by_side(pca_original_model, pca_enhanced_model, genres)

### --- PRINT CENTROIDS (after PCA) ---
# Transform centroids
centroids_orig = kmeans_original.cluster_centers_
pca_centroids_orig = pca_original_model.transform(centroids_orig)

centroids_enhanced = kmeans_enhanced.cluster_centers_
pca_centroids_enhanced = pca_enhanced_model.transform(centroids_enhanced)

print("\nCentroids for Original K-Means:")
for idx, (x, y) in enumerate(pca_centroids_orig):
    print(f"  Cluster {idx}: PCA1 = {x:.4f}, PCA2 = {y:.4f}")

print("\nCentroids for Enhanced K-Means with LOF:")
for idx, (x, y) in enumerate(pca_centroids_enhanced):
    print(f"  Cluster {idx}: PCA1 = {x:.4f}, PCA2 = {y:.4f}")

### --- SILHOUETTE SCORES ---
silhouette_original = silhouette_score(user_genre_ratings_scaled, user_genre_ratings_original["Cluster_Original"])
silhouette_enhanced = silhouette_score(user_genre_ratings_filtered_scaled, user_genre_ratings_filtered["Cluster_Enhanced"])

print(f"\nSilhouette Score for Original K-Means (Euclidean): {silhouette_original:.4f}")
print(f"Silhouette Score for Enhanced K-Means (LOF): {silhouette_enhanced:.4f}")

### --- USER CLUSTERING SUMMARY ---
outlier_user_ids = user_genre_ratings[outliers_lof == -1].index.tolist()

print("\nUser Clustering Summary:")
print("  Total users:", user_genre_ratings.shape[0])
print("  Inliers after LOF filtering:", user_genre_ratings_filtered.shape[0])
print("  Outliers removed:", user_genre_ratings.shape[0] - user_genre_ratings_filtered.shape[0])
print("  Outlier userIds (LOF detected):", outlier_user_ids)
