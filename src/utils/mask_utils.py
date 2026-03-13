import torch
import scipy.io as scio
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path


# シャッタパターン読み込み
def generate_masks(mask_path,mask_name):
    mask = scio.loadmat(mask_path + '/' + mask_name)
    mask = mask['ExpPtn']
    #print(mask.shape)
    mask = np.transpose(mask, [2, 0, 1])
    mask_s = np.sum(mask, axis=0)
    index = np.where(mask_s == 0)
    mask_s[index] = 1
    mask_s = mask_s.astype(np.uint8)
    mask = torch.from_numpy(mask)
    mask = mask.float()
    mask = mask.cuda()
    mask_s = torch.from_numpy(mask_s)
    mask_s = mask_s.float()
    mask_s = mask_s.cuda()
    return mask, mask_s

def generate_mask_s(mask):
    mask_s = torch.sum(mask, axis=0)
    index = torch.where(mask_s == 0)
    mask_s[index] = 1
    # mask_s = mask_s.astype(np.uint8)
    # mask_s = torch.from_numpy(mask_s)
    mask_s = mask_s.float()
    mask_s = mask_s.cuda()
    return mask_s

def generate_bit_matrix(reference_patch):
    """
    Args:
        reference_patch: shape [16, 8, 8] のビット行列
    Returns:
        new_patch: shape [16, 8, 8] の新しいビット行列
    """
    assert reference_patch.shape == (16, 8, 8), "Input must be [16, 8, 8] shape"
    
    # GPUテンソルをCPUに移動（必要な場合）
    if reference_patch.device.type != 'cpu':
        reference_patch = reference_patch.cpu()
    
    new_patch = torch.zeros_like(reference_patch)
    
    # 各フレームのオン率を計算
    on_ratios = [torch.sum(reference_patch[i]).item() for i in range(16)]
    
    # フレーム間の重なりピクセル数を計算
    overlap_counts = []
    for i in range(15):
        overlap = torch.logical_and(reference_patch[i], reference_patch[i+1])
        overlap_counts.append(torch.sum(overlap).item())
    
    # 1フレーム目: オン率のみ考慮
    total_pixels = 64  # 8x8
    frame0_on_pixels = int(on_ratios[0])
    
    # フレーム0のピクセル設定（修正: np.arrayからindexing）
    if frame0_on_pixels > 0:
        on_indices = np.random.choice(total_pixels, size=frame0_on_pixels, replace=False)
        # 重要な修正: flattenしたテンソルのインデックスとして使用する前に、Pythonのリストに変換
        flat_tensor = new_patch[0].flatten()
        for idx in on_indices:
            flat_tensor[idx] = 1
        new_patch[0] = flat_tensor.reshape(8, 8)
    
    # 2フレーム目以降: 直前フレームとの重なりとオン率を考慮
    for i in range(1, 16):
        prev_frame = new_patch[i-1]
        required_on_pixels = int(on_ratios[i])
        required_overlap = int(overlap_counts[i-1])
        
        # 直前フレームでオンだったピクセルのインデックス
        prev_on_indices = torch.nonzero(prev_frame.flatten()).squeeze()
        
        # 直前フレームのオンピクセル数をチェック（0次元テンソルの場合の処理も含む）
        if prev_on_indices.ndim == 0 and prev_on_indices.numel() > 0:
            prev_on_indices = prev_on_indices.unsqueeze(0)  # 単一の値を1次元テンソルに変換
            prev_on_count = 1
        elif prev_on_indices.numel() == 0:  # 空のテンソル
            prev_on_count = 0
        else:
            prev_on_count = prev_on_indices.numel()
        
        # 直前フレームでオフだったピクセルのインデックス
        prev_off_indices = torch.nonzero(prev_frame.flatten() == 0).squeeze()
        
        # オフピクセル数もチェック
        if prev_off_indices.ndim == 0 and prev_off_indices.numel() > 0:
            prev_off_indices = prev_off_indices.unsqueeze(0)
            prev_off_count = 1
        elif prev_off_indices.numel() == 0:
            prev_off_count = 0
        else:
            prev_off_count = prev_off_indices.numel()
        
        # 重なりの調整: 前のフレームにオンピクセルが足りない場合
        if prev_on_count < required_overlap:
            required_overlap = prev_on_count
        
        # 現在のフレームをflatten
        current_frame_flat = new_patch[i].flatten()
        
        # 重なるピクセルをランダムに選択
        if required_overlap > 0 and prev_on_count > 0:
            # NumPy配列への変換
            prev_on_np = prev_on_indices.cpu().numpy()
            
            # 1つだけの場合は特別処理
            if prev_on_count == 1:
                overlap_indices = np.array([prev_on_np])
            else:
                overlap_indices = np.random.choice(prev_on_np, 
                                                size=required_overlap, 
                                                replace=False)
            
            # 重なるピクセルをオンに設定
            for idx in overlap_indices:
                current_frame_flat[idx] = 1
        
        # 残りのオンピクセルを直前フレームでオフだったピクセルから選択
        remaining_on = required_on_pixels - required_overlap
        
        if remaining_on > 0 and prev_off_count > 0:
            # NumPy配列への変換
            prev_off_np = prev_off_indices.cpu().numpy()
            
            # 利用可能なオフピクセルがrequired_on_pixelsより少ない場合の調整
            if remaining_on > prev_off_count:
                remaining_on = prev_off_count
            
            # 1つだけの場合は特別処理
            if prev_off_count == 1:
                additional_indices = np.array([prev_off_np])
            else:
                additional_indices = np.random.choice(prev_off_np, 
                                                    size=remaining_on, 
                                                    replace=False)
            
            # 選択したピクセルをオンに設定
            for idx in additional_indices:
                current_frame_flat[idx] = 1
        
        # 変更を反映
        new_patch[i] = current_frame_flat.reshape(8, 8)
    
    return new_patch

"""
Utility functions for handling masks in video reconstruction
"""

def load_mask(mask_path):
    """
    Load a mask from a .mat file
    
    Args:
        mask_path: Path to the .mat file containing the mask
        
    Returns:
        mask: Mask as a numpy array
    """
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    
    try:
        mat_data = scio.loadmat(mask_path)
        # Try common mask variable names
        for key in ['mask', 'data', 'binary_mask']:
            if key in mat_data:
                return mat_data[key]
        
        # If no standard name is found, take the first non-metadata field
        for key in mat_data.keys():
            if not key.startswith('__'):
                return mat_data[key]
                
        raise KeyError("No valid mask data found in .mat file")
    except Exception as e:
        raise IOError(f"Error loading mask from {mask_path}: {e}")


def save_mask(mask, save_path, metadata=None):
    """
    Save a mask to a .mat file
    
    Args:
        mask: Mask data (numpy array or torch tensor)
        save_path: Path where to save the mask
        metadata: Optional dictionary of additional data to save
    """
    # Convert to numpy if tensor
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Prepare data to save
    save_data = {'mask': mask}
    if metadata:
        save_data.update(metadata)
    
    # Save to file
    try:
        scio.savemat(save_path, save_data)
    except Exception as e:
        raise IOError(f"Error saving mask to {save_path}: {e}")


def visualize_mask(mask, save_path=None, title="Binary Sampling Mask"):
    """
    Visualize a binary sampling mask
    
    Args:
        mask: Mask data (numpy array or torch tensor)
        save_path: Optional path to save the visualization
        title: Title for the plot
    """
    # Convert to numpy if tensor
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    
    # Handle different mask shapes
    if mask.ndim == 3:
        # For 3D masks (typically [T, H, W]), show as subplot grid
        T, H, W = mask.shape
        fig, axes = plt.subplots(1, T, figsize=(T*3, 3))
        
        if T == 1:
            axes = [axes]  # Make axes iterable for single frame
            
        for t, ax in enumerate(axes):
            im = ax.imshow(mask[t], cmap='binary', vmin=0, vmax=1)
            ax.set_title(f"Frame {t+1}")
            ax.axis('off')
            
        plt.suptitle(title)
        plt.tight_layout()
        
    elif mask.ndim == 2:
        # For 2D masks
        plt.figure(figsize=(6, 6))
        plt.imshow(mask, cmap='binary', vmin=0, vmax=1)
        plt.title(title)
        plt.axis('off')
        
    else:
        raise ValueError(f"Unexpected mask shape: {mask.shape}")
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def analyze_mask_statistics(mask_dir, output_dir=None):
    """
    Analyze statistics of a collection of masks
    
    Args:
        mask_dir: Directory containing mask files
        output_dir: Directory to save analysis results and plots
    
    Returns:
        stats: Dictionary of mask statistics
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.mat')])
    if not mask_files:
        raise FileNotFoundError(f"No mask files found in {mask_dir}")
    
    sampling_rates = []
    mask_similarities = []
    
    # Load all masks
    masks = []
    for mask_file in mask_files:
        mask_path = os.path.join(mask_dir, mask_file)
        mask = load_mask(mask_path)
        masks.append(mask)
        
        # Calculate sampling rate (ratio of 1s)
        sampling_rate = np.mean(mask)
        sampling_rates.append(sampling_rate)
    
    # Calculate similarity between masks (if more than one)
    if len(masks) > 1:
        for i in range(len(masks)):
            for j in range(i+1, len(masks)):
                # Intersection over Union as similarity measure
                intersection = np.sum(np.logical_and(masks[i], masks[j]))
                union = np.sum(np.logical_or(masks[i], masks[j]))
                similarity = intersection / union if union > 0 else 0
                mask_similarities.append(similarity)
    
    # Compute statistics
    stats = {
        'num_masks': len(masks),
        'avg_sampling_rate': np.mean(sampling_rates) if sampling_rates else 0,
        'min_sampling_rate': np.min(sampling_rates) if sampling_rates else 0,
        'max_sampling_rate': np.max(sampling_rates) if sampling_rates else 0,
        'std_sampling_rate': np.std(sampling_rates) if sampling_rates else 0,
        'avg_similarity': np.mean(mask_similarities) if mask_similarities else 0,
        'min_similarity': np.min(mask_similarities) if mask_similarities else 0,
        'max_similarity': np.max(mask_similarities) if mask_similarities else 0,
    }
    
    # Generate plots if output directory is provided
    if output_dir:
        # Sampling rate distribution
        plt.figure(figsize=(10, 6))
        plt.hist(sampling_rates, bins=20, alpha=0.7)
        plt.xlabel('Sampling Rate')
        plt.ylabel('Number of Masks')
        plt.title('Distribution of Mask Sampling Rates')
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'sampling_rate_distribution.png'), dpi=150)
        plt.close()
        
        # Similarity distribution (if applicable)
        if mask_similarities:
            plt.figure(figsize=(10, 6))
            plt.hist(mask_similarities, bins=20, alpha=0.7)
            plt.xlabel('Mask Similarity (IoU)')
            plt.ylabel('Number of Mask Pairs')
            plt.title('Distribution of Mask Similarities')
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(output_dir, 'mask_similarity_distribution.png'), dpi=150)
            plt.close()
            
        # Save statistics to file
        with open(os.path.join(output_dir, 'mask_statistics.txt'), 'w') as f:
            f.write("Mask Statistics Summary\n")
            f.write("======================\n\n")
            f.write(f"Number of masks analyzed: {stats['num_masks']}\n\n")
            f.write("Sampling Rate Statistics:\n")
            f.write(f"  Average: {stats['avg_sampling_rate']:.4f}\n")
            f.write(f"  Minimum: {stats['min_sampling_rate']:.4f}\n")
            f.write(f"  Maximum: {stats['max_sampling_rate']:.4f}\n")
            f.write(f"  Standard Deviation: {stats['std_sampling_rate']:.4f}\n\n")
            
            if mask_similarities:
                f.write("Mask Similarity Statistics (IoU):\n")
                f.write(f"  Average: {stats['avg_similarity']:.4f}\n")
                f.write(f"  Minimum: {stats['min_similarity']:.4f}\n")
                f.write(f"  Maximum: {stats['max_similarity']:.4f}\n")
    
    return stats


def generate_visualization_grid(mask_dir, output_path, max_masks=16):
    """
    Create a grid visualization of multiple masks
    
    Args:
        mask_dir: Directory containing mask files
        output_path: Path to save the visualization
        max_masks: Maximum number of masks to include
    """
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.mat')])
    if not mask_files:
        raise FileNotFoundError(f"No mask files found in {mask_dir}")
    
    # Limit number of masks to display
    if len(mask_files) > max_masks:
        mask_files = mask_files[:max_masks]
    
    # Determine grid size
    grid_size = int(np.ceil(np.sqrt(len(mask_files))))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size*3, grid_size*3))
    axes = axes.flatten()
    
    for i, mask_file in enumerate(mask_files):
        if i >= len(axes):
            break
            
        mask_path = os.path.join(mask_dir, mask_file)
        mask = load_mask(mask_path)
        
        # If mask is 3D, use the first frame
        if mask.ndim == 3:
            mask = mask[0]
        
        axes[i].imshow(mask, cmap='binary', vmin=0, vmax=1)
        axes[i].set_title(f"Mask {i+1}", fontsize=10)
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(mask_files), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f"Optimized Mask Visualization (showing {len(mask_files)} masks)")
    plt.tight_layout()
    
    # Create directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()