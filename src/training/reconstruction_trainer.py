import torch
import torch.nn as nn
import time
import os
import scipy.io as scio
import numpy as np
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt  # 追加: グラフ描画用

class ReconstructionTrainer:
    def __init__(self, 
                 model, 
                 train_loader, 
                 test_loader=None,
                 val_loader=None,
                 criterion=None,
                 recon_optimizer=None,
                 recon_lr=1.0e-7, 
                 #mask_lr=0.0001,
                 device='cuda',
                 checkpoint_dir='./checkpoints/reconstruction',
                 #mask_save_dir='./mask',
                 #recon_save_dir='./recon',
                 result_save_dir='./result'): 
        """再構成モデルのトレーニングを管理するクラス
        
        Args:
            model: 学習するモデル
            train_loader: 学習データローダー
            test_loader: テストデータローダー
            val_loader: 検証データローダー
            criterion: 損失関数（Noneの場合はMSELossを使用）
            recon_optimizer: 再構成モデル用オプティマイザー
            recon_lr: 再構成モデルの学習率
            mask_lr: マスク最適化の学習率
            device: 計算デバイス
            checkpoint_dir: チェックポイント保存ディレクトリ
            mask_save_dir: マスク保存ディレクトリ
            recon_save_dir: 再構成結果保存ディレクトリ
            result_save_dir: 評価結果保存ディレクトリ
            log_dir: ログ保存ディレクトリ
        """
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.criterion = criterion if criterion is not None else nn.MSELoss()
        self.device = device
        
        # 学習率
        self.recon_lr = recon_lr
        #self.mask_lr = mask_lr
        
        # 保存先ディレクトリ
        self.checkpoint_dir = checkpoint_dir
        #self.mask_save_dir = mask_save_dir
        #self.recon_save_dir = recon_save_dir
        self.result_save_dir = result_save_dir
        
        # ディレクトリ作成
        os.makedirs(checkpoint_dir, exist_ok=True)
        #os.makedirs(mask_save_dir, exist_ok=True)
        #os.makedirs(recon_save_dir, exist_ok=True)
        os.makedirs(result_save_dir, exist_ok=True)
        
        # オプティマイザーの設定
        if recon_optimizer is None:
            self.setup_optimizer()
        else:
            self.recon_optimizer = recon_optimizer
        
        # 結果の記録用
        self.psnr_epoch = []
        self.val_psnr_epoch = []
        self.result_dict = []
        
        # 追加: PSNR標準偏差と平均値、学習率の履歴を記録
        self.epochs = []
        self.psnr_stds = []
        self.psnr_means = []  # 追加: PSNR平均値を記録
        self.learning_rates = []
        
    def setup_optimizer(self):
        """再構成モデル用のオプティマイザーを設定"""
        self.recon_optimizer = torch.optim.Adam([
            {'params': self.model.unet1.parameters()},
            {'params': self.model.unet2.parameters()},
            {'params': self.model.unet3.parameters()},
            {'params': self.model.unet4.parameters()},
            {'params': self.model.unet5.parameters()},
            {'params': self.model.unet6.parameters()},
            {'params': self.model.unet7.parameters()},
            {'params': self.model.unet8.parameters()},
            {'params': self.model.unet9.parameters()},
            {'params': self.model.gamma1},
            {'params': self.model.gamma2},
            {'params': self.model.gamma3},
            {'params': self.model.gamma4},
            {'params': self.model.gamma5},
            {'params': self.model.gamma6},
            {'params': self.model.gamma7},
            {'params': self.model.gamma8},
            {'params': self.model.gamma9}
        ], lr=self.recon_lr)
    
    def train_epoch(self, epoch, new_mask=None):
        """1エポックの学習を実行
        
        Args:
            epoch: 現在のエポック数
            new_mask: 使用する新しいマスク（Noneの場合は現在のマスクを使用）
            
        Returns:
            epoch_loss: エポックの平均損失
            time_taken: エポックの学習時間
            binary_mask: 現在のバイナリマスク
        """
        self.model.train()
        
        # 新しいマスクが提供された場合、モデルのマスクを更新
        if new_mask is not None:
            with torch.no_grad():
                self.model.mask.weight.copy_(new_mask)
        
        epoch_loss = 0
        start_time = time.time()
        
        # トレーニングループ
        for batch_idx, gt in enumerate(self.train_loader):
            gt = gt.to(self.device).float()
            
            # 勾配をゼロに
            self.recon_optimizer.zero_grad()
            
            # モデルの順伝播
            outputs = self.model(gt)
            
            # 再構成損失の計算（複数ステージの出力に対して重み付け）
            recon_loss = (torch.sqrt(self.criterion(outputs[-1], gt)) + 
                         0.5 * torch.sqrt(self.criterion(outputs[-2], gt)) + 
                         0.5 * torch.sqrt(self.criterion(outputs[-3], gt)))
            
            # 総損失
            total_loss = recon_loss
            
            # 逆伝播
            total_loss.backward()
            
            # パラメータ更新
            self.recon_optimizer.step()
            
            # 損失の蓄積
            epoch_loss += total_loss.item()
        
        # エポックの平均損失
        avg_epoch_loss = epoch_loss / len(self.train_loader)
        time_taken = time.time() - start_time
        
        # ログ出力（50エポックごと）
        if epoch % 50 == 0:
            print("====================================")
            print(f"Epoch {epoch}")
            print(f"Loss: {avg_epoch_loss:.6f}")
            print(f"Time: {time_taken:.2f}s")
            
            # マスクの使用状況をチェック
            binary_mask = self.model.mask.get_binary_mask()
            exposure_sum = torch.sum(binary_mask, dim=(1,2))
            print(f"Average exposure per frame: {exposure_sum.mean().item():.2f}")
        
        return avg_epoch_loss, time_taken, self.model.mask.get_binary_mask()
    
    def test(self, test_loader, epoch, psnr_dict=None):
        """テストデータでモデルを評価
        
        Args:
            test_loader: テストデータローダー
            epoch: 現在のエポック数
            psnr_dict: PSNRを記録する辞書
            
        Returns:
            pred: 予測結果
            psnr_values: PSNRの値
        """
        self.model.eval()
        test_list = test_loader.dataset.file_list if hasattr(test_loader.dataset, 'file_list') else []
        psnr_sample = torch.zeros(len(test_list) if test_list else len(test_loader))
        pred = []
        
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                # データローダーからデータを取得
                if isinstance(data, torch.Tensor):
                    pic0 = data.to(self.device).float()
                    file_name = test_list[i] if test_list else f"sample_{i}"
                else:
                    # 辞書型の場合
                    pic0 = data['image'].to(self.device).float()
                    file_name = data['file_name'] if 'file_name' in data else test_list[i] if test_list else f"sample_{i}"
                
                # モデル推論
                out_pic_list = self.model(pic0)
                out_pic = out_pic_list[-1]
                
                # PSNR計算
                psnr_1 = 10 * torch.log10(1 / self.criterion(out_pic, pic0))
                psnr_sample[i] = psnr_1
                
                # 結果の記録
                if psnr_dict is not None:
                    if isinstance(file_name, str):
                        psnr_dict[file_name] = float(psnr_1)
                    else:
                        psnr_dict[str(file_name)] = float(psnr_1)
                
                # 予測結果の保存
                pred.append(out_pic.cpu().numpy())
        
        # 平均PSNRの計算
        if psnr_dict is not None:
            test_type = 'test' if test_loader == self.test_loader else 'val'
            psnr_dict[f'{test_type}_avg'] = float(torch.mean(psnr_sample))
        
        return pred, psnr_sample
    
    def save_checkpoint(self, epoch, filename=None):
        """モデルチェックポイントを保存
        
        Args:
            epoch: 現在のエポック数
            filename: 保存ファイル名（Noneの場合は自動生成）
        """
        if filename is None:
            filename = f"model_epoch_{epoch}.pth"
        
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        
        torch.save(self.model, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def save_mask(self, epoch, binary_mask, date_time=None):
        """マスクを保存
        
        Args:
            epoch: 現在のエポック数
            binary_mask: バイナリマスク
            date_time: 日時文字列（Noneの場合は現在の日時を使用）
        """
        if date_time is None:
            date_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        
        mask_path = os.path.join(self.mask_save_dir, date_time)
        os.makedirs(mask_path, exist_ok=True)
        
        # マスクをMatファイルとして保存
        scio.savemat(os.path.join(mask_path, f"mask_epoch_{epoch}.mat"), {
            'mask': binary_mask.cpu().numpy(),
            'raw_weights': self.model.mask.weight.detach().cpu().numpy()
        })
    
    def save_reconstruction(self, epoch, pred, psnr_mean, mask_name="mask", date_time=None):
        """再構成結果を保存
        
        Args:
            epoch: 現在のエポック数
            pred: 予測結果
            psnr_mean: 平均PSNR
            mask_name: マスク名
            date_time: 日時文字列（Noneの場合は現在の日時を使用）
        """
        if date_time is None:
            date_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        
        # recon_path = os.path.join(self.recon_save_dir, 'monochrome', mask_name, date_time)
        # os.makedirs(recon_path, exist_ok=True)
        
        # name = os.path.join(recon_path, f'S9_pred_{epoch}_{psnr_mean:.4f}.mat')
        # scio.savemat(name, {'pred': pred})
    
    # 追加: PSNR平均値とLRのグラフを保存する関数
    def save_psnr_mean_graph(self, date_time):
        """PSNRの平均値と学習率のグラフを保存
        
        Args:
            date_time: 日時文字列
        """
        plt.figure(figsize=(10, 6))
        
        # 左のY軸（PSNR平均値用）
        ax1 = plt.gca()
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('PSNR Mean (dB)', color='green')
        ax1.plot(self.epochs, self.psnr_means, 'g-', label='PSNR Mean')
        ax1.tick_params(axis='y', labelcolor='green')
        
        # 右のY軸（学習率用）
        ax2 = ax1.twinx()
        ax2.set_ylabel('Learning Rate', color='red')
        ax2.plot(self.epochs, self.learning_rates, 'r-', label='Learning Rate')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # グラフのタイトルと凡例
        plt.title('PSNR Mean and Learning Rate during Training')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # グリッド表示
        plt.grid(True, alpha=0.3)
        
        # ファイル保存
        save_path = os.path.join(self.result_save_dir, f"psnr_{date_time}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"PSNR平均値グラフを保存しました: {save_path}")
    
    # 追加: PSNR標準偏差とLRのグラフを保存する関数
    def save_psnr_std_graph(self, date_time):
        """PSNRの標準偏差と学習率のグラフを保存
        
        Args:
            date_time: 日時文字列
        """
        plt.figure(figsize=(10, 6))
        
        # 左のY軸（PSNR標準偏差用）
        ax1 = plt.gca()
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('PSNR Standard Deviation (dB)', color='blue')
        ax1.plot(self.epochs, self.psnr_stds, 'b-', label='PSNR Std Dev')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # 右のY軸（学習率用）
        ax2 = ax1.twinx()
        ax2.set_ylabel('Learning Rate', color='red')
        ax2.plot(self.epochs, self.learning_rates, 'r-', label='Learning Rate')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # グラフのタイトルと凡例
        plt.title('PSNR Standard Deviation and Learning Rate during Training')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # グリッド表示
        plt.grid(True, alpha=0.3)
        
        # ファイル保存
        save_path = os.path.join(self.result_save_dir, f"psnr_sd_{date_time}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def train(self, 
              max_iter=1000, 
              start_epoch=0, 
              save_interval=50,
              test_interval=50,
              mask_generator=None,
              lr_decay=0.97,
              lr_decay_interval=100):
        """完全な学習プロセスを実行
        
        Args:
            max_iter: 最大エポック数
            start_epoch: 開始エポック
            save_interval: モデル保存間隔
            test_interval: テスト評価間隔
            mask_generator: マスク生成関数（Noneの場合はモデル内のマスクを使用）
            lr_decay: 学習率の減衰率
            lr_decay_interval: 学習率を減衰させる間隔
            
        Returns:
            result_dict: 学習結果の記録
        """
        date_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        
        for epoch in range(start_epoch + 1, start_epoch + max_iter + 1):
            # エポック進行状況を一行で表示（上書き形式）
            print(f"\rTraining Epoch: {epoch}/{start_epoch + max_iter}", end="", flush=True)
            
            # マスク生成関数が提供されている場合は新しいマスクを生成
            new_mask = None
            if mask_generator is not None:
                new_mask = mask_generator()
            
            # 1エポックの学習
            loss, time_taken, binary_mask = self.train_epoch(epoch, new_mask)
            
            # 定期的なテスト評価
            if epoch % test_interval == 0:
                psnr_dict = {}
                
                                    # テストデータでの評価
                if self.test_loader is not None:
                    test_pred, test_psnr = self.test(self.test_loader, epoch, psnr_dict)
                    self.psnr_epoch.append(test_psnr)
                    test_psnr_mean = torch.mean(test_psnr)
                    
                    # 追加: PSNRの標準偏差と平均値を計算して記録
                    test_psnr_std = torch.std(test_psnr)
                    self.epochs.append(epoch)
                    self.psnr_stds.append(test_psnr_std.item())
                    self.psnr_means.append(test_psnr_mean.item())  # 追加: PSNR平均値を記録
                    self.learning_rates.append(self.recon_lr)
                    
                    print(f"Test result: {test_psnr_mean:.4f} (Std: {test_psnr_std:.4f})")
                
                # 検証データでの評価
                if self.val_loader is not None:
                    val_pred, val_psnr = self.test(self.val_loader, epoch, psnr_dict)
                    self.val_psnr_epoch.append(val_psnr)
                    val_psnr_mean = torch.mean(val_psnr)
                    print(f"Validation result: {val_psnr_mean:.4f}")
                
                # 結果の記録
                self.result_dict.append({
                    'epoch': epoch,
                    'recon_lr': self.recon_lr,
                    #'mask_lr': self.mask_lr,
                    'loss': loss,
                    'time': time_taken,
                    'psnr': psnr_dict,
                    'psnr_std': test_psnr_std.item() if self.test_loader is not None else None  # 追加: 標準偏差も記録
                })
                
                # モデルの保存
                self.save_checkpoint(epoch)
                
                # 追加: PSNRの標準偏差と平均値のグラフを更新・保存
                if self.test_loader is not None:
                    self.save_psnr_std_graph(date_time)
                    self.save_psnr_mean_graph(date_time)  # 追加: PSNR平均値グラフも保存
            
            # 学習率の減衰
            if epoch % lr_decay_interval == 0:
                #self.mask_lr *= lr_decay
                self.recon_lr *= lr_decay
                
                # オプティマイザーの学習率更新
                for param_group in self.recon_optimizer.param_groups:
                    param_group['lr'] = self.recon_lr
        
        return self.result_dict