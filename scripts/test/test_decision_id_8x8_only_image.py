"""
Test script for the mask selection model trained with
train_decision_id_8x8_only_image.py.

Loads a checkpoint, iterates over every patch in every test video, and
outputs the selected mask index for each (video, patch, block) triple.

Inference is sequential (autoregressive) within each patch location:
- Block 0  : compressed with mask 0; that compressed image is fed to the
             model to predict the mask for block 0.
- Block N>0: the previous block, compressed with the mask predicted for
             block N-1, is fed to the model to predict the mask for block N.

This mirrors the sequential training protocol in
train_decision_id_with_mask_index.py.

Requirements:
- Masks      : ./data/masks/candidates/16masks/*.mat
- Test videos: ./data/raw/256x256/test_mat/*.mat
- Checkpoint : ./checkpoints/mask_selection/select_models_id_only_image/latest_checkpoint.pth

Usage:
    python ./scripts/test/test_decision_id_8x8_only_image.py
"""

import os
import sys
import glob

import scipy.io as scio
import torch
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration  (edit here)
# ---------------------------------------------------------------------------

CHECKPOINT_PATH = './checkpoints/mask_selection/select_models_id_only_image/select_model_ep172_acc0.1210.pth'
MASK_DIR        = './data/masks/candidates/16masks'
VIDEO_DIR       = './data/raw/256x256/test_mat'

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from models.mask_selection.DecisionNet_8x8_only_image import DecisionNet


# ---------------------------------------------------------------------------
# Mask loading
# ---------------------------------------------------------------------------

def load_all_masks(mask_dir, device):
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
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ---- Resolve checkpoint ----
    checkpoint_path = CHECKPOINT_PATH
    if not os.path.exists(checkpoint_path):
        save_dir = os.path.dirname(checkpoint_path)
        candidates = sorted(glob.glob(os.path.join(save_dir, 'select_model_ep*.pth')))
        if candidates:
            checkpoint_path = candidates[-1]
            print(f"latest_checkpoint.pth not found; using: {checkpoint_path}")
        else:
            print(f"Error: no checkpoint found at {checkpoint_path}")
            sys.exit(1)

    # ---- Load model ----
    model = DecisionNet(num_classes=16).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    epoch     = checkpoint.get('epoch', 'unknown')
    saved_acc = checkpoint.get('test_accuracy', None)
    print(f"Checkpoint: {checkpoint_path}  (epoch={epoch}"
          + (f", saved_acc={saved_acc:.4f}" if saved_acc is not None else "") + ")")

    # ---- Load masks ----
    masks = load_all_masks(MASK_DIR, device)

    # ---- Iterate over all patches in all test videos ----
    video_files = sorted(f for f in os.listdir(VIDEO_DIR) if f.endswith('.mat'))
    if not video_files:
        raise FileNotFoundError(f"No .mat files found in {VIDEO_DIR}")

    print(f"\n{'video_id':<30} {'patch_idx':>9} {'block_idx':>9} {'selected_mask':>13}")
    print("-" * 66)

    with torch.no_grad():
        for video_file in video_files:
            video_id = os.path.splitext(video_file)[0]
            mat_data = scio.loadmat(os.path.join(VIDEO_DIR, video_file))
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

            if video.max() > 1.0:
                video = video / 255.0

            t, h, w = video.shape
            n_ph = h // 8
            n_pw = w // 8

            for ph in range(n_ph):
                for pw in range(n_pw):
                    patch_idx = ph * n_pw + pw
                    sh, sw = ph * 8, pw * 8

                    prev_mask_idx = 0
                    prev_compressed = None

                    for sf in range(0, t - 15, 16):
                        block_idx = sf // 16
                        patch = video[sf:sf + 16, sh:sh + 8, sw:sw + 8]
                        gt_block = torch.from_numpy(patch).float().unsqueeze(0).to(device)

                        if prev_compressed is None:
                            # First block: initialise with mask 0; skip prediction
                            prev_compressed = torch.sum(
                                gt_block * masks[prev_mask_idx].unsqueeze(0),
                                dim=1, keepdim=True,
                            )
                            continue

                        # Predict using the previous block's compressed image
                        logits = model.get_logits(prev_compressed)
                        selected_mask = torch.argmax(logits, dim=1).item()

                        print(f"{video_id:<30} {patch_idx:>9} {block_idx:>9} {selected_mask:>13}")

                        # Compress current block with the predicted mask → input for next block
                        prev_mask_idx = selected_mask
                        prev_compressed = torch.sum(
                            gt_block * masks[prev_mask_idx].unsqueeze(0),
                            dim=1, keepdim=True,
                        )


if __name__ == '__main__':
    main()
