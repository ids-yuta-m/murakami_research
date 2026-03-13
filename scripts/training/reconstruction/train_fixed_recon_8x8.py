"""
Training script for fixed-mask ADMM reconstruction models.

Unlike the universal reconstruction model that adapts to different masks,
this script trains separate models for each fixed mask from ./data/mask/16masks.
Each model specializes in reconstruction with its specific mask pattern.

This script uses the existing ADMM_net architecture and freezes the LearnableMask
parameters to create a fixed-mask model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
import os
import glob
from pathlib import Path
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import argparse
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from models.reconstruction.ADMM_net import ADMM_net

# CUDA configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Path configuration
MASK_DIR = "./data/masks/candidates/generated_from_bayer"
GT_DIR = "./data/raw/256x256/gt_mat"  # Training data
TEST_DIR = "./data/raw/256x256/test_mat"  # Test data for evaluation


def set_fixed_mask(model, mask_tensor):
    """
    Set a fixed mask in the ADMM_net's LearnableMask and freeze it.

    Args:
        model: ADMM_net instance
        mask_tensor: [16, 8, 8] binary mask tensor (values 0 or 1)

    The LearnableMask uses a weight parameter where positive values -> 1, negative -> 0.
    We set weight to large positive/negative values and freeze it.
    """
    with torch.no_grad():
        # Convert binary mask (0/1) to weight values (+10/-10)
        # Positive weight -> mask value 1, Negative weight -> mask value 0
        weight_values = torch.where(mask_tensor > 0.5,
                                     torch.tensor(10.0),
                                     torch.tensor(-10.0))
        model.mask.weight.copy_(weight_values)

    # Freeze the mask parameters
    model.mask.weight.requires_grad = False

    print(f"Fixed mask set and frozen:")
    print(f"  Mask shape: {mask_tensor.shape}")
    print(f"  Binary mask sparsity: {mask_tensor.mean():.3f}")
    print(f"  Weight range: {model.mask.weight.min():.1f} to {model.mask.weight.max():.1f}")

    return model


def load_mask(mask_path):
    """Load a single mask from .mat file"""
    mask_data = sio.loadmat(mask_path)
    key = [k for k in mask_data.keys() if not k.startswith('_')][0]
    mask = mask_data[key]  # Expected shape: [16, 8, 8]

    mask = np.array(mask, dtype=np.float32)

    # Ensure correct shape [16, 8, 8]
    if mask.shape != (16, 256, 256):
        if mask.ndim == 3:
            if mask.shape == (256, 256, 16):
                mask = mask.transpose(2, 0, 1)
            elif mask.shape == (256, 16, 256):
                mask = mask.transpose(1, 0, 2)
        print(f"Warning: Mask shape was {mask_data[key].shape}, reshaped to {mask.shape}")
    
    mask = mask[:,:8,:8]

    return torch.tensor(mask, dtype=torch.float32)


def load_ground_truth_data(gt_dir, max_files=None):
    """Load ground truth data (will extract 8x8 patches from videos)"""
    gt_files = glob.glob(os.path.join(gt_dir, "*.mat"))
    if max_files:
        gt_files = gt_files[:max_files]

    gt_data = []
    print(f"Loading {len(gt_files)} ground truth files...")
    for gt_file in tqdm(gt_files):
        data = sio.loadmat(gt_file)
        key = [k for k in data.keys() if not k.startswith('_')][0]
        video = data[key]

        video = np.array(video, dtype=np.float32)

        # Handle different possible shapes - convert to [T, H, W]
        if video.ndim == 3:
            # Determine which dimension is temporal (usually the smallest or equal to 16/similar)
            # and which are spatial (usually 256 or larger)
            shape = video.shape

            # If shape is (256, 256, T) -> transpose to (T, 256, 256)
            if shape[0] >= 256 and shape[1] >= 256 and shape[2] < 256:
                video = video.transpose(2, 0, 1)
            # If shape is (256, T, 256) -> transpose to (T, 256, 256)
            elif shape[0] >= 256 and shape[1] < 256 and shape[2] >= 256:
                video = video.transpose(1, 0, 2)
            # If shape is (T, 256, 256) -> keep as is
            elif shape[0] < 256 and shape[1] >= 256 and shape[2] >= 256:
                pass
            # Otherwise assume it's already (T, H, W)

        # Normalize to [0, 1] range
        video = (video - video.min()) / (video.max() - video.min() + 1e-8)
        gt_data.append(torch.tensor(video, dtype=torch.float32))

    return gt_data


def extract_8x8_patch(video_256, target_frames=16):
    """
    Randomly extract a 16-frame, 8x8 patch from a video

    Args:
        video_256: [T, H, W] tensor where T can be variable, H=W=256
        target_frames: number of frames to extract (default: 16)

    Returns:
        patch_8x8: [16, 8, 8] tensor
    """
    T, H, W = video_256.shape

    # Randomly select temporal starting point (ensure we have enough frames)
    if T >= target_frames:
        start_t = random.randint(0, T - target_frames)
    else:
        # If video has fewer than target_frames, pad with zeros
        start_t = 0

    # Randomly select spatial starting point
    start_h = random.randint(0, H - 8)
    start_w = random.randint(0, W - 8)

    # Extract temporal block
    if T >= target_frames:
        temporal_block = video_256[start_t:start_t+target_frames, :, :]
    else:
        # Pad with zeros if not enough frames
        temporal_block = torch.zeros(target_frames, H, W, dtype=video_256.dtype, device=video_256.device)
        temporal_block[:T, :, :] = video_256

    # Extract spatial patch
    patch = temporal_block[:, start_h:start_h+8, start_w:start_w+8]

    return patch


def load_test_data(test_dir, max_files=None):
    """Load test data for evaluation"""
    return load_ground_truth_data(test_dir, max_files=max_files)


def evaluate_model(model, test_data, num_test_patches=100):
    """
    Evaluate model reconstruction accuracy on test data using fixed patches

    Args:
        model: trained ADMM_net model
        test_data: list of test video tensors
        num_test_patches: number of patches to evaluate (default: 100)

    Returns:
        avg_psnr: average PSNR over test patches

    Note:
        Uses a fixed random seed (42) to ensure the same patches are used
        across different evaluations for fair comparison.
    """
    model.eval()
    criterion = nn.MSELoss()
    total_psnr = 0.0

    # Save current random state
    random_state = random.getstate()

    # Set fixed seed for reproducible patch selection
    random.seed(42)

    with torch.no_grad():
        for _ in range(num_test_patches):
            # Select test video (now deterministic due to fixed seed)
            video_idx = random.randint(0, len(test_data) - 1)
            test_video = test_data[video_idx]

            # Extract patch (now deterministic due to fixed seed)
            test_patch = extract_8x8_patch(test_video).unsqueeze(0).to(device)

            # Reconstruct
            reconstructed_list = model(test_patch)
            final_reconstruction = reconstructed_list[-1]

            # Calculate PSNR
            mse = criterion(final_reconstruction, test_patch)
            psnr = 10 * torch.log10(1.0 / mse)
            total_psnr += psnr.item()

    # Restore previous random state
    random.setstate(random_state)

    avg_psnr = total_psnr / num_test_patches
    return avg_psnr


def train_fixed_mask_model(mask_path, gt_data, test_data, mask_name, num_epochs=200, batch_size=8, lr=0.001, patches_per_epoch=20000):
    """
    Train a reconstruction model with a single fixed mask

    Args:
        mask_path: path to the .mat file containing the mask
        gt_data: list of ground truth video tensors for training
        test_data: list of test video tensors for evaluation
        mask_name: name identifier for saving checkpoints
        num_epochs: number of training epochs
        batch_size: batch size for training
        lr: learning rate
        patches_per_epoch: number of patches to train per epoch (default: 10000)

    Returns:
        trained model
    """
    print(f"\n{'='*60}")
    print(f"Training model for mask: {mask_name}")
    print(f"{'='*60}")

    # Load the fixed mask
    fixed_mask = load_mask(mask_path).to(device)
    print(f"Loaded mask:")
    print(f"  Shape: {fixed_mask.shape}")
    print(f"  Value range: {fixed_mask.min():.3f} - {fixed_mask.max():.3f}")
    print(f"  Sparsity: {fixed_mask.mean():.3f}")

    # Create ADMM_net model and set fixed mask
    model = ADMM_net(t=16, s=8).to(device)
    model = set_fixed_mask(model, fixed_mask)

    # Display parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Patches per epoch: {patches_per_epoch:,}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

    # Loss function
    criterion = nn.MSELoss()

    # Training logs
    train_losses = []
    train_psnrs = []

    print("Starting training...")

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_psnr = 0.0
        num_batches = 0
        start_time = time.time()

        print(f"\rTraining Epoch: {epoch}/{num_epochs}", end="", flush=True)

        # Generate patches_per_epoch patches by randomly sampling from videos
        num_batches_per_epoch = patches_per_epoch // batch_size

        for batch_idx in range(num_batches_per_epoch):
            # Randomly select videos for this batch
            batch_gt_8x8 = []
            for _ in range(batch_size):
                # Randomly select a video
                video_idx = random.randint(0, len(gt_data) - 1)
                video_256 = gt_data[video_idx]

                # Extract a random patch from this video
                patch_8x8 = extract_8x8_patch(video_256)
                batch_gt_8x8.append(patch_8x8)

            # Move batch to GPU
            batch_gt = torch.stack(batch_gt_8x8).to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            reconstructed_list = model(batch_gt)

            # Multi-stage loss (weighted combination of last 3 stages)
            recon_loss = (torch.sqrt(criterion(reconstructed_list[-1], batch_gt)) +
                         0.5 * torch.sqrt(criterion(reconstructed_list[-2], batch_gt)) +
                         0.5 * torch.sqrt(criterion(reconstructed_list[-3], batch_gt)))

            total_loss = recon_loss

            # Backward pass
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer step
            optimizer.step()

            # Calculate metrics
            with torch.no_grad():
                final_reconstruction = reconstructed_list[-1]
                batch_psnr = 0.0
                for i in range(len(batch_gt)):
                    psnr_val = 10 * torch.log10(1 / criterion(final_reconstruction[i], batch_gt[i]))
                    batch_psnr += psnr_val.item()
                batch_psnr /= len(batch_gt)

            epoch_loss += total_loss.item()
            epoch_psnr += batch_psnr
            num_batches += 1

        # Epoch averages
        avg_loss = epoch_loss / num_batches
        avg_psnr = epoch_psnr / num_batches
        time_taken = time.time() - start_time

        train_losses.append(avg_loss)
        train_psnrs.append(avg_psnr)

        # Update learning rate
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']

        # Logging
        if epoch % 1 == 0:
            print("\n" + "="*40)
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"Train Loss: {avg_loss:.6f}")
            print(f"Train PSNR: {avg_psnr:.2f} dB")
            print(f"Time: {time_taken:.2f}s")
            print(f"LR: {current_lr:.6f}")

        # Evaluation and checkpoint saving
        if epoch % 10 == 0:
            # Evaluate on test data
            if test_data and len(test_data) > 0:
                test_psnr = evaluate_model(model, test_data, num_test_patches=100)
                print(f"Test PSNR: {test_psnr:.2f} dB")
            else:
                test_psnr = 0.0

            # Save checkpoint
            checkpoint_dir = f'./checkpoints/reconstruction/fixed_8x8_add/{mask_name}'
            os.makedirs(checkpoint_dir, exist_ok=True)
            model_out_path = f'{checkpoint_dir}/fixed_admm_epoch_{epoch}_train{avg_psnr:.2f}_test{test_psnr:.2f}dB.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'train_psnr': avg_psnr,
                'test_psnr': test_psnr,
                'mask_name': mask_name,
                'mask_path': mask_path,
            }, model_out_path)
            print(f"Checkpoint saved to {model_out_path}")

    print(f"\nTraining completed for {mask_name}!")
    print(f"Final PSNR: {avg_psnr:.2f} dB")

    return model, train_losses, train_psnrs


def train_all_masks(gt_data, test_data, num_epochs=200, batch_size=8, lr=0.001, patches_per_epoch=20000):
    """Train models for all 16 masks"""
    mask_files = sorted(glob.glob(os.path.join(MASK_DIR, "*.mat")))
    print(f"\nFound {len(mask_files)} masks to train")

    results = {}

    for idx, mask_file in enumerate(mask_files):
        mask_name = Path(mask_file).stem
        # if mask_name == "mask0007":
        #     continue
        print(f"\n{'#'*60}")
        print(f"Training mask {idx+1}/{len(mask_files)}: {mask_name}")
        print(f"{'#'*60}")

        model, losses, psnrs = train_fixed_mask_model(
            mask_file, gt_data, test_data, mask_name,
            num_epochs=num_epochs, batch_size=batch_size, lr=lr, patches_per_epoch=patches_per_epoch
        )

        results[mask_name] = {
            'model': model,
            'losses': losses,
            'psnrs': psnrs,
            'final_psnr': psnrs[-1]
        }

    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    for mask_name, result in results.items():
        print(f"{mask_name}: Final PSNR = {result['final_psnr']:.2f} dB")

    return results


def main():
    parser = argparse.ArgumentParser(description='Train fixed-mask ADMM reconstruction models')
    parser.add_argument('--mask', type=str, default=None,
                       help='Specific mask name to train (e.g., mask0034). If not specified, trains all masks.')
    parser.add_argument('--epochs', type=int, default=150,
                       help='Number of training epochs (default: 200)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size (default: 8)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.0001)')
    parser.add_argument('--patches_per_epoch', type=int, default=20000,
                       help='Number of patches to train per epoch (default: 10000)')
    parser.add_argument('--max_files', type=int, default=None,
                       help='Maximum number of ground truth files to load (for testing)')

    args = parser.parse_args()

    # Load ground truth data
    print("Loading training data...")
    gt_data = load_ground_truth_data(GT_DIR, max_files=args.max_files)
    print(f"Loaded {len(gt_data)} training videos")

    if len(gt_data) > 0:
        print(f"\nTraining video statistics:")
        print(f"  First video shape: {gt_data[0].shape}")
        print(f"  First video value range: {gt_data[0].min():.3f} - {gt_data[0].max():.3f}")

        # Show shape distribution
        shapes = {}
        for video in gt_data:
            shape_key = str(video.shape)
            shapes[shape_key] = shapes.get(shape_key, 0) + 1

        print(f"\n  Video shape distribution:")
        for shape, count in sorted(shapes.items()):
            print(f"    {shape}: {count} videos")

    # Load test data
    print("\nLoading test data...")
    test_data = load_test_data(TEST_DIR, max_files=args.max_files)
    print(f"Loaded {len(test_data)} test videos")

    if len(test_data) > 0:
        print(f"\nTest video statistics:")
        print(f"  First video shape: {test_data[0].shape}")
        print(f"  First video value range: {test_data[0].min():.3f} - {test_data[0].max():.3f}")

    # Train specific mask or all masks
    if args.mask:
        mask_path = os.path.join(MASK_DIR, f"{args.mask}.mat")
        if not os.path.exists(mask_path):
            print(f"Error: Mask file not found: {mask_path}")
            return

        model, losses, psnrs = train_fixed_mask_model(
            mask_path, gt_data, test_data, args.mask,
            num_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            patches_per_epoch=args.patches_per_epoch
        )
    else:
        results = train_all_masks(
            gt_data, test_data, num_epochs=args.epochs,
            batch_size=args.batch_size, lr=args.lr,
            patches_per_epoch=args.patches_per_epoch
        )


if __name__ == "__main__":
    main()
