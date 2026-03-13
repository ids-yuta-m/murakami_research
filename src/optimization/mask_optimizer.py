import os
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io as scio
import numpy as np


class MaskOptimizer:
    """マスクの最適化を行うクラス"""
    
    def __init__(self, network, config):
        """
        初期化
        
        Args:
            network: 最適化対象のネットワーク
            config: 設定辞書
        """
        self.network = network
        self.config = config
        self.criterion = nn.MSELoss()
        self.device = next(network.parameters()).device
        
        # 保存先ディレクトリの作成
        os.makedirs(config['io']['output_dir'], exist_ok=True)
        
        # 再構成部分のパラメータを凍結
        self._freeze_reconstruction_layers()
    
    def _freeze_reconstruction_layers(self):
        """再構成層のパラメータを凍結する"""
        for i in range(1, 10):  # unet1 ~ unet9
            getattr(self.network, f'unet{i}').requires_grad_(False)
            getattr(self.network, f'gamma{i}').requires_grad_(False)
        
        # マスクのみ最適化可能に設定
        self.network.mask.requires_grad_(True)
    
    def optimize_mask_for_patch(self, patch, patch_num):
        """
        単一パッチグループに対してマスクを最適化
        
        Args:
            patch: 形状[num_patches, t, h, w]のパッチデータ
            patch_num: パッチグループ番号
            
        Returns:
            最適化されたマスク
        """
        # 設定の取得
        mask_lr = self.config['optimization']['mask_lr']
        epochs = self.config['optimization']['epochs']
        lr_decay_interval = self.config['optimization']['lr_decay_interval']
        lr_decay_rate = self.config['optimization']['lr_decay_rate']
        print_interval = self.config['optimization']['print_interval']
        save_dir = self.config['io']['output_dir']
        
        num_patches = patch.shape[0]
        patch = patch / 255 if patch.max() > 1.0 else patch
        
        # 初期PSNRの計算
        with torch.no_grad():
            psnrs = []
            for j in range(num_patches):
                single_frame = patch[j].unsqueeze(0)
                outputs = self.network(single_frame)
                psnr = 10 * torch.log10(1 / self.criterion(outputs[-1], single_frame))
                psnrs.append(psnr.item())
            mean_psnr = sum(psnrs) / len(psnrs)
        
        print(f"初期PSNR: {mean_psnr:.4f}")
        
        # 最適化ループ
        best_psnr = 0
        best_mask = self.network.mask.get_binary_mask()
        best_epoch = 0
        
        for epoch in range(epochs):
            optimizer = optim.Adam([self.network.mask.weight], lr=mask_lr)
            optimizer.zero_grad()
            
            total_loss = 0.0
            total_psnr = 0.0
            
            for i in range(num_patches):
                single_frame = patch[i].unsqueeze(0)
                outputs = self.network(single_frame)
                
                # 損失関数：複数出力の重み付き平均
                recon_loss = (
                    torch.sqrt(self.criterion(outputs[-1], single_frame)) +
                    0.5 * torch.sqrt(self.criterion(outputs[-2], single_frame)) +
                    0.5 * torch.sqrt(self.criterion(outputs[-3], single_frame))
                )
                
                loss = recon_loss
                loss.backward()
                
                total_loss += loss.item()
                psnr = 10 * torch.log10(1 / self.criterion(outputs[-1], single_frame))
                total_psnr += psnr.item()
            
            optimizer.step()
            
            avg_loss = total_loss / num_patches
            avg_psnr = total_psnr / num_patches
            
            # 最良結果の保存
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                best_mask = self.network.mask.get_binary_mask()
                best_epoch = epoch + 1
            
            # 学習率の調整
            if (epoch + 1) % lr_decay_interval == 0 and epoch < 20000:
                mask_lr *= lr_decay_rate
            
            # 経過表示
            if (epoch + 1) % print_interval == 0:
                print(f"[{epoch+1}] LR: {mask_lr:.3e} | Loss: {avg_loss:.6f} | PSNR: {avg_psnr:.4f} | Best Epoch: {best_epoch}")
        
        # 最適化結果の保存
        self._save_mask(best_mask, save_dir, patch_num, patch, best_psnr)
        
        print(f"🔧 最適PSNR: {best_psnr:.4f}")
        return best_mask
    
    def _save_mask(self, mask, save_dir, patch_num, patch, best_psnr):
        """最適化されたマスクの保存"""
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"mask{patch_num}.mat")
        
        # 既存マスクとの比較
        if os.path.exists(save_path):
            saved_mask = scio.loadmat(save_path).get("mask")
            if saved_mask is not None:
                saved_mask_tensor = torch.from_numpy(saved_mask).float().to(self.device)
                self.network.mask.weight.data.copy_(saved_mask_tensor)
                with torch.no_grad():
                    outputs = self.network(patch[0].unsqueeze(0))
                    psnr_old = 10 * torch.log10(1 / self.criterion(outputs[-1], patch[0].unsqueeze(0)))
                if best_psnr > psnr_old.item():
                    scio.savemat(save_path, {'mask': mask.cpu().numpy()})
                    print("上書き保存")
                else:
                    print("既存のマスクが良好なため保存スキップ")
        else:
            scio.savemat(save_path, {'mask': mask.cpu().numpy()})
            print("保存完了")