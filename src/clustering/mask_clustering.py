import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

def cluster_bit_matrices(bit_matrices, cluster_sizes):
    """
    Cluster bit matrices into specified cluster sizes
    
    Args:
        bit_matrices: List of flattened bit matrices
        cluster_sizes: List of cluster sizes to create
        
    Returns:
        Dictionary mapping cluster sizes to (labels, centers, closest_indices)
    """
    # Convert to numpy
    X = np.array(bit_matrices)
    
    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    results = {}
    
    # Process each cluster size
    for k in cluster_sizes:
        print(f"\nClustering into {k} clusters...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(X_scaled)
        centers = kmeans.cluster_centers_
        
        # Find closest real sample to each cluster center
        dists = cdist(centers, X_scaled)  # shape (k, n_samples)
        closest_indices = np.argmin(dists, axis=1)
        
        # Store results
        results[k] = {
            'labels': labels,
            'centers': centers,
            'scaler': scaler,
            'closest_indices': closest_indices
        }
        
    return results

def cluster_bit_matrices_pca(bit_matrices, cluster_sizes):
    """
    Use PCA for dimensionality reduction and select representative samples
    instead of clustering via KMeans.
    """
    X = np.array(bit_matrices)

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = {}

    for n_components in cluster_sizes:
        print(f"\nSelecting {n_components} principal components as representatives...")

        # Apply PCA
        pca = PCA(n_components=n_components, random_state=42)
        X_pca = pca.fit_transform(X_scaled)

        # Each PCA component vector is of shape (n_samples,)
        # Select sample with the highest absolute projection value for each component
        components = pca.components_  # shape: (n_components, original_features)
        projections = X_scaled @ components.T  # shape: (n_samples, n_components)
        closest_indices = np.argmax(np.abs(projections), axis=0)

        results[n_components] = {
            'labels': None,
            'centers': None,
            'scaler': scaler,
            'pca': pca,
            'closest_indices': closest_indices
        }

    return results