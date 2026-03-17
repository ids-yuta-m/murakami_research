"""
Record PSNR results for each fixed-mask reconstruction model.

This script evaluates all 16 fixed-mask models on test patches and creates
an NPZ file summarizing reconstruction results. The format is compatible with
train_decision_next_8x8_direct.py which uses these PSNR values for training
the mask selection network.

Usage:
    python ./scripts/training/mask_selection/record_each_model_psnrs_fixed.py

Input:
    - Models: ./checkpoints/reconstruction/fixed_8x8/best_models/*.pth (16 models)
    - Data: ./data/raw/full_size/gt_mat/*.mat (video files)

Output:
    - File: ./recon_results_next.npz
    - Format:
        * Keys: "{video_name}_{patch_idx}_{block_idx}"
        * Values: array of 16 PSNR values (one for each mask model)
        * Example: psnr_results["video001_0_0"] -> [32.1, 31.8, ..., 33.2] (16 values)

Processing:
    - Extracts 5x5 grid of 8x8 patches from each 256x256 video frame
    - Processes 16-frame temporal blocks
    - Evaluates each patch with all 16 fixed-mask models
    - Records PSNR for each (patch, mask) combination
"""

import os
import torch
import numpy as np
import scipy.io as scio
from torch.nn.functional import mse_loss
from tqdm import tqdm
import sys
from pathlib import Path
import glob

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from models.reconstruction.ADMM_net import ADMM_net

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if not torch.cuda.is_available():
    raise Exception('NO GPU!')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Paths
MODEL_DIR = './checkpoints/reconstruction/fixed_8x8_best'
INPUT_DIR = './data/raw/256x256/test_mat' # train : './data/raw/256x256/train_mat' 
SAVE_PATH = './recon_results_next_88_test.npz'


def load_fixed_models(model_dir):
    """
    Load all 16 fixed-mask reconstruction models.

    Returns:
        models: list of loaded ADMM_net models
        model_names: list of mask names (e.g., 'mask0034')
    """
    model_files = sorted(glob.glob(os.path.join(model_dir, '*.pth')))

    if len(model_files) == 0:
        raise FileNotFoundError(f"No model files found in {model_dir}")

    print(f"Found {len(model_files)} model files")

    models = []
    model_names = []

    for model_file in tqdm(model_files, desc="Loading models"):
        # Extract mask name from filename
        mask_name = Path(model_file).stem

        # Load model
        model = ADMM_net(t=16, s=8).to(DEVICE)
        checkpoint = torch.load(model_file, map_location=DEVICE)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.eval()

        models.append(model)
        model_names.append(mask_name)

    print(f"Loaded {len(models)} models: {model_names}")
    return models, model_names


def record_psnr_fixed(models, model_names, input_dir, save_path):
    """
    Evaluate all models on all patches and record PSNR values.

    Args:
        models: list of ADMM_net models
        model_names: list of mask names
        input_dir: directory containing .mat video files
        save_path: path to save NPZ file
    """
    mat_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.mat')])

    if len(mat_files) == 0:
        raise FileNotFoundError(f"No .mat files found in {input_dir}")

    print(f"Found {len(mat_files)} video files to process")

    # Dictionary to store PSNR results
    # Key: "{video_name}_{patch_idx}_{block_idx}"
    # Value: list of 16 PSNR values (one per mask)
    psnr_dict = {}

    # Process each video file
    for mat_file in tqdm(mat_files, desc="Processing videos"):
        file_path = os.path.join(input_dir, mat_file)
        fname = os.path.splitext(mat_file)[0]

        # Load video data
        data = scio.loadmat(file_path)
        keys = [k for k in data.keys() if not k.startswith('__')]

        if len(keys) != 1:
            print(f"Warning: {mat_file} has unexpected keys: {keys}, skipping...")
            continue

        tensor = data[keys[0]]  # Expected shape: [t, 256, 256] or similar

        # Handle different possible shapes - convert to [T, H, W]
        if tensor.ndim == 3:
            shape = tensor.shape
            # If shape is (256, 256, T) -> transpose to (T, 256, 256)
            if shape[0] >= 256 and shape[1] >= 256 and shape[2] < 256:
                tensor = tensor.transpose(2, 0, 1)
            # If shape is (256, T, 256) -> transpose to (T, 256, 256)
            elif shape[0] >= 256 and shape[1] < 256 and shape[2] >= 256:
                tensor = tensor.transpose(1, 0, 2)

        t, h, w = tensor.shape

        # Extract 8x8 patches (5x5 grid covering the 256x256 frame)
        # Using the same patch extraction as the original script
        for patch_h in range(8):
            for patch_w in range(8):
                # Extract spatial patch: 8 pixels * 6 = 48 pixels spacing
                patch = tensor[:,
                              patch_h * 30:patch_h * 30 + 8,
                              patch_w * 30:patch_w * 30 + 8]

                # Extract temporal blocks (16 frames each)
                for start_frame in range(0, t - 15, 16):
                    sub_patch = patch[start_frame:start_frame + 16]  # [16, 8, 8]

                    # Normalize to [0, 1]
                    input_tensor = torch.from_numpy(sub_patch / 255.).float().unsqueeze(0).to(DEVICE)

                    # Evaluate with all 16 models
                    psnr_list = []
                    for model in models:
                        with torch.no_grad():
                            output = model(input_tensor)[-1]  # Take final reconstruction stage

                        mse = mse_loss(output, input_tensor)
                        psnr = 10 * torch.log10(1.0 / mse).item()
                        psnr_list.append(psnr)

                    # Create key: "{video_name}_{patch_idx}_{block_idx}"
                    patch_idx = patch_h * 8 + patch_w
                    block_idx = int(start_frame / 16)
                    key = f"{fname}_{patch_idx}_{block_idx}"

                    psnr_dict[key] = np.array(psnr_list, dtype=np.float32)

    # Save to NPZ file
    print(f"\nSaving {len(psnr_dict)} patch results to {save_path}")
    np.savez(save_path, **psnr_dict)

    # Print statistics
    if len(psnr_dict) > 0:
        all_psnrs = np.array([v for v in psnr_dict.values()])
        print(f"\nStatistics:")
        print(f"  Total patches: {len(psnr_dict)}")
        print(f"  PSNR per patch shape: {all_psnrs[0].shape}")
        print(f"  Average PSNR: {all_psnrs.mean():.2f} dB")
        print(f"  Min PSNR: {all_psnrs.min():.2f} dB")
        print(f"  Max PSNR: {all_psnrs.max():.2f} dB")

        # Per-model statistics
        print(f"\n  Per-model average PSNR:")
        for i, name in enumerate(model_names):
            avg_psnr = all_psnrs[:, i].mean()
            print(f"    {name}: {avg_psnr:.2f} dB")


def main():
    print("="*60)
    print("Recording PSNR results for fixed-mask reconstruction models")
    print("="*60)

    # Load all models
    print(f"\nLoading models from: {MODEL_DIR}")
    models, model_names = load_fixed_models(MODEL_DIR)

    # Record PSNR for all patches
    print(f"\nProcessing video patches from: {INPUT_DIR}")
    record_psnr_fixed(models, model_names, INPUT_DIR, SAVE_PATH)

    print(f"\n{'='*60}")
    print(f"PSNR results saved to: {SAVE_PATH}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
