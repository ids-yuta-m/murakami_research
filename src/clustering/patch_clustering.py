import os
import numpy as np
from scipy.io import loadmat, savemat
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


class PatchClusterer:
    """
    Class for clustering scene patches
    """
    def __init__(self, config):
        """
        Initialization function
        
        Args:
            config: Clustering configuration
        """
        self.input_dir = config['input_dir']
        self.output_dir = config['output_dir']
        self.n_clusters = config['n_clusters']
        self.patch_size = config['patch_size']
        self.random_state = config.get('random_state', 42)
        self.min_patches_per_cluster = config.get('min_patches_per_cluster', 5)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # For metrics storage
        self.metrics = {}
    
    def get_valid_key(self, mat_dict):
        """Get valid key from mat file"""
        return next((k for k in mat_dict if not k.startswith("__")), None)
        
    def collect_patches(self):
        """Collect patches from input directory"""
        print("Collecting patches...")
        self.all_patches = []
        self.file_indices = []  # Record original file numbers
        self.patch_positions = []  # Record patch positions
        
        for file_idx, filename in enumerate(sorted(os.listdir(self.input_dir))):
            if filename.endswith(".mat"):
                filepath = os.path.join(self.input_dir, filename)
                try:
                    mat = loadmat(filepath)
                    key = self.get_valid_key(mat)
                    if key is None:
                        continue
                    
                    tensor = mat[key]
                    expected_shape = (16, 256, 256)
                    if tensor.shape != expected_shape:
                        print(f"Warning: Invalid shape in {filename}: {tensor.shape}, expected: {expected_shape}")
                        continue
                    
                    # Split based on patch size
                    stride = self.patch_size
                    n_patches = 256 // stride
                    
                    for i in range(n_patches):
                        for j in range(n_patches):
                            patch = tensor[:, i*stride:(i+1)*stride, j*stride:(j+1)*stride]
                            self.all_patches.append(patch)
                            self.file_indices.append(file_idx)
                            self.patch_positions.append((i, j))
                except Exception as e:
                    print(f"Error reading file {filename}: {e}")
        
        # Convert to NumPy array
        self.patch_array = np.array(self.all_patches)
        self.n_patches = len(self.patch_array)
        print(f"Total patches extracted: {self.n_patches}")
        
        return self.n_patches
    
    def perform_clustering(self):
        """Perform clustering on patches"""
        print("Performing clustering...")
        
        # Flatten patches for clustering
        flat_patches = self.patch_array.reshape((self.n_patches, -1))
        
        # Run KMeans clustering
        self.kmeans = KMeans(
            n_clusters=self.n_clusters, 
            random_state=self.random_state, 
            n_init='auto'
        )
        self.labels = self.kmeans.fit_predict(flat_patches)
        
        # Analyze clustering results
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        
        # Save metrics
        self.metrics['n_clusters'] = len(unique_labels)
        self.metrics['avg_patches_per_cluster'] = np.mean(counts)
        self.metrics['min_patches_per_cluster'] = np.min(counts)
        self.metrics['max_patches_per_cluster'] = np.max(counts)
        
        print(f"Number of clusters: {self.metrics['n_clusters']}")
        print(f"Average patches per cluster: {self.metrics['avg_patches_per_cluster']:.2f}")
        print(f"Minimum patches per cluster: {self.metrics['min_patches_per_cluster']}")
        print(f"Maximum patches per cluster: {self.metrics['max_patches_per_cluster']}")
        
        return self.labels
    
    def save_results(self):
        """Save clustering results"""
        print("Saving patches for each cluster...")
        
        # Get number of patches per cluster
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        
        # Save histogram (before filtering)
        plt.figure(figsize=(10, 6))
        plt.hist(counts, bins=30)
        plt.title('Cluster Size Distribution (Before Filtering)')
        plt.xlabel('Number of Patches per Cluster')
        plt.ylabel('Number of Clusters')
        plt.savefig(os.path.join(self.output_dir, 'cluster_distribution_before_filtering.png'))
        plt.close()
        
        # Identify clusters with few patches
        small_clusters = [label for label, count in zip(unique_labels, counts) if count <= self.min_patches_per_cluster]
        valid_clusters = [label for label, count in zip(unique_labels, counts) if count > self.min_patches_per_cluster]
        
        print(f"Clusters with <= {self.min_patches_per_cluster} patches: {len(small_clusters)} / {len(unique_labels)}")
        print(f"Valid clusters: {len(valid_clusters)} / {len(unique_labels)}")
        
        # Remap labels
        # Assign new indices to valid clusters
        label_map = {old_label: new_label for new_label, old_label in enumerate(valid_clusters)}
        
        # Create new labels
        new_labels = np.full_like(self.labels, -1)  # Initialize with -1
        for old_label, new_label in label_map.items():
            new_labels[self.labels == old_label] = new_label
        
        # Patches from small clusters remain -1
        removed_patch_count = np.sum(new_labels == -1)
        if removed_patch_count > 0:
            print(f"Patches removed from small clusters: {removed_patch_count}")
        
        # Save new label distribution
        valid_counts = [counts[unique_labels == label][0] for label in valid_clusters]
        
        # Save histogram (after filtering)
        plt.figure(figsize=(10, 6))
        plt.hist(valid_counts, bins=30)
        plt.title('Cluster Size Distribution (After Filtering)')
        plt.xlabel('Number of Patches per Cluster')
        plt.ylabel('Number of Clusters')
        plt.savefig(os.path.join(self.output_dir, 'cluster_distribution_after_filtering.png'))
        plt.close()
        
        # Save only valid clusters
        for new_cluster_id, old_cluster_id in enumerate(valid_clusters):
            # Get indices of patches in this cluster
            cluster_indices = np.where(self.labels == old_cluster_id)[0]
            
            # Create cluster directory
            cluster_path = os.path.join(self.output_dir, f"cluster_{new_cluster_id:04d}")
            os.makedirs(cluster_path, exist_ok=True)
            
            # Save cluster info
            with open(os.path.join(cluster_path, "info.txt"), "w", encoding="utf-8") as f:
                f.write(f"Cluster ID: {new_cluster_id}\n")
                f.write(f"Original Cluster ID: {old_cluster_id}\n")
                f.write(f"Number of Patches: {len(cluster_indices)}\n")
            
            # Save each patch
            for i, idx in enumerate(cluster_indices):
                patch = self.patch_array[idx]
                file_idx = self.file_indices[idx]
                pos_i, pos_j = self.patch_positions[idx]
                
                # Save with filename containing patch info
                file_path = os.path.join(
                    cluster_path, 
                    f"patch_{i:04d}_file{file_idx}_pos{pos_i}_{pos_j}.mat"
                )
                savemat(file_path, {"data": patch})
        
        # Save filtering info
        filter_info_path = os.path.join(self.output_dir, "filtering_info.txt")
        with open(filter_info_path, "w", encoding="utf-8") as f:
            f.write(f"Total clusters before filtering: {len(unique_labels)}\n")
            f.write(f"Clusters with <= {self.min_patches_per_cluster} patches: {len(small_clusters)}\n")
            f.write(f"Clusters after filtering: {len(valid_clusters)}\n")
            f.write(f"Removed patches: {removed_patch_count}\n")
            f.write("\nRemoved clusters:\n")
            for cluster_id in small_clusters:
                count = counts[unique_labels == cluster_id][0]
                f.write(f"  Cluster {cluster_id}: {count} patches\n")
        
        # Save clustering summary
        summary_path = os.path.join(self.output_dir, "clustering_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"Total patches: {self.n_patches}\n")
            f.write(f"Clusters before filtering: {len(unique_labels)}\n")
            f.write(f"Clusters after filtering: {len(valid_clusters)}\n")
            
            # Post-filtering statistics
            if valid_counts:
                avg_count = np.mean(valid_counts)
                min_count = np.min(valid_counts)
                max_count = np.max(valid_counts)
                f.write(f"Average patches per cluster after filtering: {avg_count:.2f}\n")
                f.write(f"Minimum patches per cluster after filtering: {min_count}\n")
                f.write(f"Maximum patches per cluster after filtering: {max_count}\n")
            
            # Record patch count for each cluster
            f.write("\nPatch count per cluster (after filtering):\n")
            for new_id, old_id in enumerate(valid_clusters):
                count = counts[unique_labels == old_id][0]
                f.write(f"Cluster {new_id} (original ID: {old_id}): {count} patches\n")
        
        # Update metrics for return
        self.metrics['original_n_clusters'] = len(unique_labels)
        self.metrics['filtered_n_clusters'] = len(valid_clusters)
        self.metrics['removed_clusters'] = len(small_clusters)
        self.metrics['removed_patches'] = removed_patch_count
        
        if valid_counts:
            self.metrics['filtered_avg_patches'] = np.mean(valid_counts)
            self.metrics['filtered_min_patches'] = np.min(valid_counts)
            self.metrics['filtered_max_patches'] = np.max(valid_counts)
        
        print(f"Complete: {len(valid_clusters)} clusters saved after filtering.")
        print(f"Results saved to {self.output_dir}")
        
        return summary_path
    
    def run(self):
        """Run the entire clustering process"""
        self.collect_patches()
        self.perform_clustering()
        return self.save_results()