"""
Training script for mask selection model using the compressed image from the
previous block as input.

Based on train_decision_id_8x8_only_image.py (random-mask variant) and
train_decision_id_with_mask_index.py (sequential variant).

Blocks within each location sequence are processed in order.  The model
receives the compressed measurement of the *previous* block (compressed with
the mask predicted for that block) and learns to predict the optimal mask for
the *current* block.  For the first block in a sequence the current block is
compressed with mask 0 to initialise the context.

Key differences from the random-mask variant:
- Input is the previous block's compressed image instead of a randomly
  sampled compression of the current block.
- Blocks are processed sequentially within each location sequence.
- Evaluation uses a random mask to compress the sample as a proxy for the
  previous-block context (same approach as train_decision_id_with_mask_index).

Input Format:
- Compressed measurement: [B, 1, 8, 8] - previous block compressed with the
  previously predicted mask (mask 0 for the very first block)
- Target: [B] - index of the best mask (0-15)
- Output: [B, 16] - probability distribution over 16 mask candidates

Requirements:
- Masks: ./data/masks/candidates/new_16masks/*.mat  (16 masks)
- PSNR Data: ./recon_results_next_88.npz
- Videos: ./data/raw/256x256/gt_mat/*.mat

Usage:
    python ./scripts/training/mask_selection/train_decision_id_8x8_only_image.py
"""

import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import scipy.io as scio
import os
import numpy as np
import glob
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from models.mask_selection.DecisionNet_8x8_only_image import DecisionNet


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LocationSequenceDataset(Dataset):
    """Dataset that generates all (video, patch) location sequences."""
    def __init__(self, path, patch_size=8):
        super().__init__()
        self.patch_size = patch_size
        self.video_dir = path

        if not os.path.exists(path):
            raise FileNotFoundError(f'Path does not exist: {path}')

        video_files = sorted([f for f in os.listdir(path) if f.endswith('.mat')])
        self.location_sequences = []

        for video_file in video_files:
            video_path = os.path.join(path, video_file)
            video_id = os.path.splitext(video_file)[0]

            mat_data = scio.loadmat(video_path)
            keys = [k for k in mat_data.keys() if not k.startswith('__')]
            if len(keys) != 1:
                continue

            video = mat_data[keys[0]]
            if video.ndim == 3:
                s = video.shape
                if s[0] >= 256 and s[1] >= 256 and s[2] < 256:
                    video = video.transpose(2, 0, 1)
                elif s[0] >= 256 and s[1] < 256 and s[2] >= 256:
                    video = video.transpose(1, 0, 2)

            t = video.shape[0]

            for patch_h in range(8):
                for patch_w in range(8):
                    patch_idx = patch_h * 8 + patch_w
                    start_h = patch_h * 30
                    start_w = patch_w * 30

                    blocks = []
                    for start_frame in range(0, t - 15, 16):
                        block_idx = int(start_frame / 16)
                        blocks.append({
                            'video_id': video_id,
                            'patch_idx': patch_idx,
                            'block_idx': block_idx,
                            'start_h': start_h,
                            'start_w': start_w,
                            'start_frame': start_frame,
                            'key': f"{video_id}_{patch_idx}_{block_idx}",
                        })

                    if blocks:
                        self.location_sequences.append(blocks)

        print(f"Generated {len(self.location_sequences)} location sequences "
              f"from {len(video_files)} videos")
        print(f"Total blocks: {sum(len(s) for s in self.location_sequences)}")

    def __getitem__(self, index):
        return self.location_sequences[index]

    def __len__(self):
        return len(self.location_sequences)


class VideoDataset(Dataset):
    """Loads and caches video patches at specified locations."""
    def __init__(self, path):
        super().__init__()
        if not os.path.exists(path):
            raise FileNotFoundError(f'Path does not exist: {path}')
        self.video_files = {
            os.path.splitext(f)[0]: os.path.join(path, f)
            for f in os.listdir(path) if f.endswith('.mat')
        }
        self._cache = {}

    def load_video(self, video_id):
        if video_id not in self._cache:
            mat_data = scio.loadmat(self.video_files[video_id])
            video = next(v for k, v in mat_data.items() if not k.startswith('__'))
            video = torch.from_numpy(video).float()
            if video.dim() == 3:
                if video.shape[0] == 256 and video.shape[1] == 256:
                    video = video.permute(2, 0, 1)
            if video.max() > 1.0:
                video = video / 255.0
            self._cache[video_id] = video
        return self._cache[video_id]

    def extract_block(self, video_id, start_frame, start_h, start_w, patch_size=8):
        video = self.load_video(video_id)
        return video[start_frame:start_frame + 16,
                     start_h:start_h + patch_size,
                     start_w:start_w + patch_size]

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, index):
        return list(self.video_files.keys())[index]


# ---------------------------------------------------------------------------
# Mask loading
# ---------------------------------------------------------------------------

def load_all_masks(mask_dir):
    """Load all mask files from mask_dir and return as a list of tensors."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mask_files = sorted(glob.glob(os.path.join(mask_dir, '*.mat')))
    if not mask_files:
        raise FileNotFoundError(f"No mask files found in {mask_dir}")

    masks = []
    for mask_path in mask_files:
        mask_data = scio.loadmat(mask_path)
        key = [k for k in mask_data.keys() if not k.startswith('__')][0]
        mask = torch.from_numpy(mask_data[key]).float().to(device)
        if mask.shape != (16, 8, 8):
            if mask.shape == (8, 8, 16):
                mask = mask.permute(2, 0, 1)
            elif mask.shape == (8, 16, 8):
                mask = mask.permute(1, 0, 2)
        masks.append(mask)

    print(f"Loaded {len(masks)} masks from {mask_dir}")
    return masks


# ---------------------------------------------------------------------------
# PSNR analysis
# ---------------------------------------------------------------------------

def analyze_psnr_distribution(psnr_results):
    """Print PSNR statistics and best-mask distribution."""
    print("\n" + "=" * 60)
    print("PSNR DATA ANALYSIS")
    print("=" * 60)

    psnr_array = np.array([psnr_results[k] for k in psnr_results.keys()])
    print(f"Total patches: {len(psnr_array)}")
    print(f"\n{'Mask':<6} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8}")
    print("-" * 40)
    for i in range(16):
        col = psnr_array[:, i]
        print(f"{i:<6} {col.mean():<8.4f} {col.std():<8.4f} "
              f"{col.min():<8.4f} {col.max():<8.4f}")

    best_masks = np.argmax(psnr_array, axis=1)
    print(f"\nBest Mask Distribution:")
    for i in range(16):
        count = np.sum(best_masks == i)
        freq = count / len(best_masks)
        print(f"  Mask {i:2d}: {freq:.4f} ({freq*100:.2f}%)", end="")
        if (i + 1) % 4 == 0:
            print()

    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_selection(decision_net, masks, psnr_npz_path, test_video_dir,
                       num_samples=1000):
    """
    Evaluate mask selection accuracy on test data.

    Blocks are processed sequentially within each (video, patch) location,
    matching the training protocol:
    - First block: compressed with mask 0 as initial context.
    - Each subsequent block: use the previous block compressed with the
      previously predicted mask as input.

    Evaluation stops after num_samples total blocks have been scored.
    """
    device = next(decision_net.parameters()).device
    decision_net.eval()

    psnr_results = np.load(psnr_npz_path)

    # Build sequences grouped by (video, patch location)
    sequences = []
    for video_file in sorted(f for f in os.listdir(test_video_dir) if f.endswith('.mat')):
        video_path = os.path.join(test_video_dir, video_file)
        video_id = os.path.splitext(video_file)[0]

        mat_data = scio.loadmat(video_path)
        keys = [k for k in mat_data.keys() if not k.startswith('__')]
        if len(keys) != 1:
            continue
        video = mat_data[keys[0]]
        if video.ndim == 3:
            s = video.shape
            if s[0] >= 256 and s[1] >= 256 and s[2] < 256:
                video = video.transpose(2, 0, 1)
            elif s[0] >= 256 and s[1] < 256 and s[2] >= 256:
                video = video.transpose(1, 0, 2)

        t = video.shape[0]
        for ph in range(8):
            for pw in range(8):
                patch_idx = ph * 8 + pw
                sh, sw = ph * 30, pw * 30
                blocks = []
                for sf in range(0, t - 15, 16):
                    key = f"{video_id}_{patch_idx}_{int(sf / 16)}"
                    if key in psnr_results:
                        blocks.append({
                            'video': video,
                            'start_frame': sf,
                            'start_h': sh,
                            'start_w': sw,
                            'key': key,
                        })
                if blocks:
                    sequences.append(blocks)

    # Shuffle sequences for a representative sample
    rng = np.random.RandomState(42)
    rng.shuffle(sequences)

    correct = 0
    total = 0
    predicted_dist = np.zeros(16)
    best_mask_dist = np.zeros(16)
    per_mask_correct = np.zeros(16)
    per_mask_total = np.zeros(16)

    with torch.no_grad():
        for blocks in sequences:
            if num_samples > 0 and total >= num_samples:
                break

            prev_mask_idx = 0
            prev_compressed = None

            for block_idx, block in enumerate(blocks):
                if num_samples > 0 and total >= num_samples:
                    break

                video = block['video']
                patch = video[block['start_frame']:block['start_frame'] + 16,
                             block['start_h']:block['start_h'] + 8,
                             block['start_w']:block['start_w'] + 8]
                if patch.max() > 1.0:
                    patch = patch / 255.0

                gt_block = torch.from_numpy(patch).float().unsqueeze(0).to(device)

                if prev_compressed is None:
                    # First block: initialise context only; skip evaluation
                    prev_compressed = torch.sum(
                        gt_block * masks[prev_mask_idx].unsqueeze(0),
                        dim=1, keepdim=True,
                    )
                    continue

                logits = decision_net.get_logits(prev_compressed)
                predicted_idx = torch.argmax(logits, dim=1).item()
                best_mask_idx = int(np.argmax(psnr_results[block['key']]))

                predicted_dist[predicted_idx] += 1
                best_mask_dist[best_mask_idx] += 1
                per_mask_total[best_mask_idx] += 1
                if predicted_idx == best_mask_idx:
                    correct += 1
                    per_mask_correct[best_mask_idx] += 1
                total += 1

                # Update sequential context
                prev_mask_idx = predicted_idx
                prev_compressed = torch.sum(
                    gt_block * masks[prev_mask_idx].unsqueeze(0),
                    dim=1, keepdim=True,
                )

    accuracy = correct / total if total > 0 else 0.0
    return {
        'accuracy': accuracy,
        'predicted_dist': predicted_dist / total,
        'best_mask_dist': best_mask_dist / total,
        'per_mask_correct': per_mask_correct,
        'per_mask_total': per_mask_total,
    }


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------

def save_training_curves(epoch_losses, train_accs, test_accs, save_dir):
    """Save loss and accuracy curves to a PNG file."""
    os.makedirs(save_dir, exist_ok=True)
    epochs = list(range(len(epoch_losses)))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(epochs, epoch_losses, color='#1f77b4', linewidth=1.5, label='Loss')
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Curves', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_accs, color='#2ca02c', linewidth=1.5, label='Train Accuracy')
    ax2.plot(epochs, test_accs,  color='#d62728', linewidth=1.5, label='Test Accuracy')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_ylim(0, max(1.0, max(train_accs + test_accs, default=0) * 1.05))
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(save_dir, 'training_curves.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved training curves to: {out_path}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_decision_net(masks, sequence_loader, video_dataset, npz_dir, save_dir,
                       num_epochs=1000, lr=1e-4, resume_path=None):
    """
    Train decision network.

    At each step one mask is randomly sampled from `masks` to compress the
    current block.  The network is trained with cross-entropy + entropy
    regularization (identical structure to train_decision_id_8x8_single_mask.py).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    decision_net = DecisionNet(num_classes=16).to(device)
    optimizer = torch.optim.AdamW(decision_net.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.95)

    start_epoch = 0
    best_test_accuracy = 0.0
    epoch_losses: list = []
    train_accs: list = []
    test_accs: list = []

    if resume_path is not None and os.path.exists(resume_path):
        print(f"Loading checkpoint from: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        decision_net.load_state_dict(checkpoint['state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        if 'test_accuracy' in checkpoint:
            best_test_accuracy = checkpoint['test_accuracy']
        if 'history' in checkpoint:
            hist = checkpoint['history']
            epoch_losses = list(hist.get('losses', []))
            train_accs   = list(hist.get('train_accs', []))
            test_accs    = list(hist.get('test_accs', []))
        print(f"Resuming from epoch {start_epoch}, "
              f"best accuracy: {best_test_accuracy:.4f}")

    psnr_results = np.load(npz_dir)
    analyze_psnr_distribution(psnr_results)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    entropy_weight = 0.1
    best_model_state = None
    total_sequences = len(sequence_loader)

    debug_info = {
        'selection_counts': np.zeros(16),
        'target_counts': np.zeros(16),
    }

    for epoch in range(start_epoch, num_epochs):
        decision_net.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        batch_num = 0
        per_mask_correct = np.zeros(16)
        per_mask_total = np.zeros(16)
        debug_info['selection_counts'] = np.zeros(16)
        debug_info['target_counts'] = np.zeros(16)

        for batch_idx, sequence_metadata in enumerate(sequence_loader):
            sequence = sequence_metadata[0]

            prev_mask_idx = None
            prev_compressed = None

            for idx, block_meta in enumerate(sequence):
                psnr_key = block_meta['key']
                if psnr_key not in psnr_results:
                    continue

                # Extract [16, 8, 8] block
                gt_block = video_dataset.extract_block(
                    block_meta['video_id'],
                    block_meta['start_frame'],
                    block_meta['start_h'],
                    block_meta['start_w'],
                    patch_size=8,
                ).unsqueeze(0).to(device).float()

                if prev_compressed is None:
                    # First block: initialise context with mask 0; skip training
                    prev_mask_idx = 0
                    prev_compressed = torch.sum(
                        gt_block * masks[prev_mask_idx].unsqueeze(0),
                        dim=1, keepdim=True,
                    )
                    continue

                # Ground truth: best mask by PSNR
                best_mask_idx = int(np.argmax(psnr_results[psnr_key]))
                target = torch.tensor([best_mask_idx], dtype=torch.long, device=device)

                # Forward pass using the previous block's compressed image
                logits = decision_net.get_logits(prev_compressed)
                ce_loss = criterion(logits, target)

                # Entropy regularization (encourage diverse predictions)
                probs = F.softmax(logits, dim=1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1).mean()
                loss = ce_loss - entropy_weight * entropy

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(decision_net.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                batch_num += 1

                # Accuracy tracking
                predicted_idx = torch.argmax(logits, dim=1).item()
                debug_info['selection_counts'][predicted_idx] += 1
                debug_info['target_counts'][best_mask_idx] += 1
                per_mask_total[best_mask_idx] += 1
                if predicted_idx == best_mask_idx:
                    correct += 1
                    per_mask_correct[best_mask_idx] += 1
                total += 1

                # Update sequential context: compress current block with the
                # predicted mask → becomes input for the next block
                prev_mask_idx = predicted_idx
                prev_compressed = torch.sum(
                    gt_block * masks[prev_mask_idx].unsqueeze(0),
                    dim=1, keepdim=True,
                )

            # Intra-epoch progress
            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == total_sequences:
                running_loss = epoch_loss / batch_num if batch_num > 0 else 0
                running_acc  = correct / total if total > 0 else 0
                total_sel = np.sum(debug_info['selection_counts'])
                if total_sel > 0:
                    sel_dist = debug_info['selection_counts'] / total_sel * 100
                    dist_str = "[" + ",".join([f"{sel_dist[i]:.0f}" for i in range(16)]) + "]"
                else:
                    dist_str = "[N/A]"
                print(f"\r  Ep{epoch} [{batch_idx+1}/{total_sequences}] "
                      f"L={running_loss:.4f} A={running_acc:.4f} "
                      f"Dist%:{dist_str}                    ",
                      end="", flush=True)

        scheduler.step()

        # ---- Epoch summary ----
        avg_loss  = epoch_loss / batch_num if batch_num > 0 else 0
        train_acc = correct / total if total > 0 else 0

        print(f"\nEpoch {epoch}")
        print(f"Average Loss: {avg_loss:.6f}")
        print(f"Train Accuracy: {train_acc:.4f} ({correct}/{total})")

        print(f"Per-mask accuracy:")
        for i in range(16):
            if per_mask_total[i] > 0:
                acc_i = per_mask_correct[i] / per_mask_total[i]
                print(f"  Mask {i:2d}: {acc_i:.4f} "
                      f"({int(per_mask_correct[i])}/{int(per_mask_total[i])})", end="")
            else:
                print(f"  Mask {i:2d}:    N/A (   0/   0)", end="")
            if (i + 1) % 4 == 0:
                print()

        total_selections = np.sum(debug_info['selection_counts'])
        total_targets    = np.sum(debug_info['target_counts'])
        if total_selections > 0:
            print(f"Predicted Mask Distribution (argmax):")
            for i in range(16):
                freq = debug_info['selection_counts'][i] / total_selections
                print(f"  Mask {i:2d}: {freq:.4f} ({freq*100:.2f}%)", end="")
                if (i + 1) % 4 == 0:
                    print()
        if total_targets > 0:
            print(f"Target (Best) Mask Distribution:")
            for i in range(16):
                freq = debug_info['target_counts'][i] / total_targets
                print(f"  Mask {i:2d}: {freq:.4f} ({freq*100:.2f}%)", end="")
                if (i + 1) % 4 == 0:
                    print()

        # ---- Test evaluation ----
        print(f"\nTest Set Evaluation (1000 samples):")
        TEST_VIDEO_DIR = "./data/raw/256x256/test_mat"
        TEST_PSNR_NPZ  = "./recon_results_next_88_test.npz"

        metrics = evaluate_selection(
            decision_net, masks, TEST_PSNR_NPZ, TEST_VIDEO_DIR, num_samples=1000
        )
        print(f"  Test Accuracy: {metrics['accuracy']:.4f}")

        print(f"  Predicted Mask Distribution (argmax):")
        for i in range(16):
            print(f"  Mask {i:2d}: {metrics['predicted_dist'][i]:.4f} "
                  f"({metrics['predicted_dist'][i]*100:.2f}%)", end="")
            if (i + 1) % 4 == 0:
                print()

        print(f"  Ground Truth Best Mask Distribution:")
        for i in range(16):
            print(f"  Mask {i:2d}: {metrics['best_mask_dist'][i]:.4f} "
                  f"({metrics['best_mask_dist'][i]*100:.2f}%)", end="")
            if (i + 1) % 4 == 0:
                print()

        # ---- History ----
        epoch_losses.append(avg_loss)
        train_accs.append(train_acc)
        test_accs.append(metrics['accuracy'])

        # ---- Save best model ----
        max_pred_freq = metrics['predicted_dist'].max()
        if metrics['accuracy'] > best_test_accuracy:
            if max_pred_freq > 0.4:
                print(f"  Skipping save: most-frequent predicted mask is "
                      f"{max_pred_freq*100:.1f}% > 40% (collapsed prediction)")
            else:
                best_test_accuracy = metrics['accuracy']
                best_model_state = {
                    'state_dict': decision_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch,
                    'test_accuracy': best_test_accuracy,
                    'train_accuracy': train_acc,
                    'history': {
                        'losses': epoch_losses,
                        'train_accs': train_accs,
                        'test_accs': test_accs,
                    },
                }
                print(f"  New best test accuracy: {best_test_accuracy:.4f}")

        # ---- Periodic checkpoint ----
        if epoch % 10 == 0 and epoch > 0:
            os.makedirs(save_dir, exist_ok=True)

            if best_model_state is not None:
                save_path = (f'{save_dir}/select_model_ep{best_model_state["epoch"]}'
                             f'_acc{best_test_accuracy:.4f}.pth')
                torch.save(best_model_state, save_path)
                print(f"  Saved best model to {save_path}")

            latest_checkpoint = {
                'state_dict': decision_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'test_accuracy': best_test_accuracy,
                'train_accuracy': train_acc,
                'history': {
                    'losses': epoch_losses,
                    'train_accs': train_accs,
                    'test_accs': test_accs,
                },
            }
            latest_path = f'{save_dir}/latest_checkpoint.pth'
            torch.save(latest_checkpoint, latest_path)
            print(f"  Saved latest checkpoint to {latest_path}")

            save_training_curves(epoch_losses, train_accs, test_accs, save_dir)

    # Final training curves
    if epoch_losses:
        save_training_curves(epoch_losses, train_accs, test_accs, save_dir)

    return decision_net


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Train mask selection model (random mask input)'
    )
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint file to resume training from')
    parser.add_argument('--resume-latest', action='store_true',
                        help='Resume from latest_checkpoint.pth in save directory')
    parser.add_argument('--num-epochs', type=int, default=1000,
                        help='Number of training epochs (default: 1000)')
    args = parser.parse_args()

    print("=" * 60)
    print("Training Mask Selection Model (Random Mask Input)")
    print("=" * 60)

    MASK_DIR  = "./data/masks/candidates/new_16masks"
    VIDEO_DIR = "./data/raw/256x256/gt_mat"
    NPZ_DIR   = "./recon_results_next_88.npz"
    SAVE_DIR  = "./checkpoints/mask_selection/select_models_id_only_image"

    # Determine resume path
    resume_path = args.resume
    if args.resume_latest:
        resume_path = os.path.join(SAVE_DIR, 'latest_checkpoint.pth')
        if not os.path.exists(resume_path):
            print(f"Warning: latest_checkpoint.pth not found at {resume_path}, "
                  f"starting fresh")
            resume_path = None

    # Load masks and datasets
    masks = load_all_masks(MASK_DIR)

    print(f"\nLoading datasets...")
    sequence_dataset = LocationSequenceDataset(VIDEO_DIR)
    video_dataset    = VideoDataset(VIDEO_DIR)
    print(f"Dataset size: {len(sequence_dataset)} location sequences")

    def collate_fn(batch):
        return batch

    sequence_loader = DataLoader(
        dataset=sequence_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn,
    )

    print(f"\nStarting training...")
    train_decision_net(
        masks,
        sequence_loader,
        video_dataset,
        npz_dir=NPZ_DIR,
        save_dir=SAVE_DIR,
        num_epochs=args.num_epochs,
        lr=1e-6,
        resume_path=resume_path,
    )

    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
