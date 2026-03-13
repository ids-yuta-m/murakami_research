import os
import scipy.io
import numpy as np

def load_bit_matrices(base_dir):
    """
    Load bit matrices from .mat files in the base directory
    
    Args:
        base_dir: Directory containing .mat files
        
    Returns:
        bit_matrices: List of flattened bit matrices
        file_paths: List of (dir_name, filename) tuples
    """
    bit_matrices = []
    file_paths = []  # Store (dir_name, filename)
    
    mat_files = sorted([f for f in os.listdir(base_dir) if f.endswith('.mat')])
    
    for mat_file in mat_files:
        full_path = os.path.join(base_dir, mat_file)
        mat_data = scipy.io.loadmat(full_path)
        
        # Find usable key
        key = next(k for k in mat_data.keys() if not k.startswith('__'))
        matrix = mat_data[key]
        
        if matrix.shape == (16, 8, 8):
            bit_matrices.append(matrix.flatten())
            file_paths.append((base_dir, mat_file))  # Save origin
    
    print(f"Loaded {len(bit_matrices)} bit matrices.")
    return bit_matrices, file_paths

def save_cluster_centers(cluster_results, file_paths, base_dir, out_dir_name):
    """
    Save cluster centers as .mat files
    
    Args:
        cluster_results: Dictionary of clustering results
        file_paths: List of (dir_name, filename) tuples
        base_dir: Original directory containing .mat files
        out_dir_name: Output directory for clustered masks
    """
    for k, result in cluster_results.items():
        closest_indices = result['closest_indices']
        
        # Output directory
        out_dir = os.path.join(out_dir_name, f"{k}masks")
        os.makedirs(out_dir, exist_ok=True)
        
        used_names = set()
        for idx, closest_idx in enumerate(closest_indices):
            dir_name, file_name = file_paths[closest_idx]
            full_path = os.path.join(base_dir, file_name)
            
            # Generate unique filename
            new_file_name = f"{file_name}"
            used_names.add(new_file_name)
            
            out_path = os.path.join(out_dir, new_file_name)
            
            # Save the actual bit matrix (not intermediate values)
            mat_data = scipy.io.loadmat(full_path)
            key = next(k for k in mat_data.keys() if not k.startswith('__'))
            matrix = mat_data[key]
            scipy.io.savemat(out_path, {'mask': matrix})
        
        print(f"✅ Saved {k} cluster centers to: {out_dir}")