import os
import torch
import numpy as np
import scipy.io as scio
from torch.autograd import Variable
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

class MaskSelectionTrainer:
    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        
        # PSNR結果のロード
        self.psnr_results = np.load(config.psnr_results_path)
        
        # 設定
        self.num_epochs = config.num_epochs
        self.lr = config.learning_rate
        self.replay_batch_size = config.replay_batch_size
        self.min_experiences = config.min_experiences
        
    def load_recon_models(self, model_paths):
        """再構成ネットワークをロードする"""
        from models.reconstruction.ADMM_net import ADMM_net
        
        recon_nets = []
        for path in model_paths:
            network = ADMM_net().to(self.device)
            state_dict = torch.load('./pre-train_epoch_12300_state_dict.pth', map_location=self.device)
            if hasattr(state_dict, "state_dict"):
                network.load_state_dict(state_dict.state_dict())
            else:
                network.load_state_dict(state_dict)
            #network = torch.load(path, map_location=self.device)
            network = network.to(self.device)
            mat = scio.loadmat(path)
            mask = mat[list(mat.keys())[-1]]  # 最初のマスク変数を取得
            mask = torch.from_numpy(mask).float().to(self.device)
            with torch.no_grad():  # 勾配計算をオフにする
                network.mask.weight.copy_(mask)
            self._stop_grads(network)
            recon_nets.append(network)
            
        return recon_nets
    
    def _stop_grads(self, network):
        """ネットワークの勾配を止める"""
        # マスクのパラメータを凍結
        network.mask.requires_grad_(False)
        # 再構成部分のパラメータを凍結
        network.unet1.requires_grad_(False)
        network.unet2.requires_grad_(False)
        network.unet3.requires_grad_(False)
        network.unet4.requires_grad_(False)
        network.unet5.requires_grad_(False)
        network.unet6.requires_grad_(False)
        network.unet7.requires_grad_(False)
        network.gamma1.requires_grad_(False)
        network.gamma2.requires_grad_(False)
        network.gamma3.requires_grad_(False)
        network.gamma4.requires_grad_(False)
        network.gamma5.requires_grad_(False)
        network.gamma6.requires_grad_(False)
        network.gamma7.requires_grad_(False)
    
    def train(self, decision_net, recon_nets, train_loader):
        """マスク選択モデルの学習"""
        device = self.device
        
        # オプティマイザーとスケジューラーの設定
        optimizer = torch.optim.AdamW(
            decision_net.parameters(), 
            lr=self.lr, 
            weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=200, T_mult=1, eta_min=1e-7
        )
        
        # リファレンスネットワークの設定
        reference_net = type(decision_net)(
            num_classes=decision_net.num_classes,
            epsilon_start=decision_net.epsilon,
            epsilon_end=decision_net.epsilon_end,
            epsilon_decay=decision_net.epsilon_decay
        ).to(device)
        
        reference_state = {
            k: v.clone() for k, v in decision_net.state_dict().items()
        }
        reference_net.load_state_dict(reference_state)
        reference_net.eval()
        
        best_expected_psnr = float('-inf')
        best_model_state = None

        for epoch in range(self.num_epochs):
            decision_net.train()
            epoch_loss = 0.0
            batch_num = 0
            sum_expected_psnr = 0
            
            for batch_idx, (video_batch, ids) in enumerate(train_loader):
                gt = Variable(video_batch)
                gt = gt.to(device).float()
                #patch_idx = None
                #selected_recon_net = None
                #prev_recon = None

                filename = ids[0]
                num_frames = gt.shape[1]
                
                for patch_h in range(5):
                    for patch_w in range(5):
                        patch_idx = None
                        selected_recon_net = None
                        prev_recon = None
                        filename = ids[0] + f"_{patch_h*5+patch_w}"
                        gt_patch = gt[:, :, patch_h*6*8:(patch_h*6+1)*8, patch_w*6*8:(patch_w*6+1)*8]
                
                        for idx in range(int(num_frames/16)):
                            patch_idx = gt_patch[:, idx*16:(idx+1)*16, :, :]
                            if idx == 0:
                                selected_recon_net = recon_nets[np.random.randint(0, len(recon_nets))]
                                reconstructions = selected_recon_net(patch_idx)
                                prev_recon = reconstructions[-1]
                            else:
                                filename_idx = filename + f"_{idx}"
                                batch_psnrs = torch.from_numpy(self.psnr_results[filename_idx]).float().requires_grad_(True).to(device)
                                prev_recon.requires_grad_(True)
                                
                                # reference_netでの予測(探索なし)
                                with torch.no_grad():
                                    reference_probs = reference_net(prev_recon)
                                    reference_expected_psnrs = torch.sum(batch_psnrs * reference_probs, dim=1)
                                    
                                # 現在のネットワークでの予測(epsilon greedyあり)
                                current_probs = decision_net.forward_with_exploration(prev_recon, training=True)
                                current_expected_psnrs = torch.sum(batch_psnrs * current_probs, dim=1)

                                # improvementsを報酬として使用
                                improvements = current_expected_psnrs - reference_expected_psnrs
                                
                                experience = {
                                    'state': prev_recon.detach().cpu(),
                                    'action_probs': current_probs[0].detach().cpu(),
                                    'reward': improvements[0].item(),
                                    'psnr_values': batch_psnrs.detach().cpu(),
                                    'reference_psnr': reference_expected_psnrs[0].detach().cpu()
                                }
                                decision_net.experience_buffer.append(experience)
                                
                                selected_recon_net = recon_nets[torch.argmax(current_probs).item()]
                                reconstructions = selected_recon_net(patch_idx)
                                prev_recon = reconstructions[-1]
                                
                                sum_expected_psnr += current_expected_psnrs.mean()
                                batch_num += 1

                                # Experience Replayによる学習
                                if len(decision_net.experience_buffer) >= self.min_experiences and epoch > 0:
                                    experiences = decision_net.experience_buffer.sample(self.replay_batch_size)
                                    
                                    # 経験バッチの処理
                                    replay_states = torch.stack([exp['state'] for exp in experiences]).to(device)
                                    replay_psnrs = torch.stack([exp['psnr_values'] for exp in experiences]).to(device)
                                    replay_reference_psnrs = torch.tensor([exp['reference_psnr'] for exp in experiences]).to(device)
                                    
                                    # リプレイデータでの予測
                                    replay_states = replay_states.squeeze(1)
                                    replay_current_probs = decision_net(replay_states)

                                    replay_expected_psnrs = torch.sum(replay_psnrs * replay_current_probs, dim=1)
                                    
                                    # 損失関数
                                    if epoch == 0:
                                        loss = -torch.mean(current_expected_psnrs)
                                    else:
                                        improvements = replay_expected_psnrs - replay_reference_psnrs
                                        loss = -torch.mean(improvements)

                                    reference_state = {
                                        k: v.clone() for k, v in decision_net.state_dict().items()
                                    }
                                    optimizer.zero_grad()
                                    loss.backward()
                                    #torch.nn.utils.clip_grad_norm_(decision_net.parameters(), max_norm=1.0)
                                    optimizer.step()

                        reference_net.load_state_dict(reference_state)
                        reference_net.eval()

                
                
            # スケジューラーの更新
            scheduler.step()
            current_epsilon = decision_net.update_epsilon()
            current_expected_psnr = sum_expected_psnr / batch_num
            
            print(f"Epoch {epoch}")
            print(f"Epsilon: {current_epsilon:.4f}")
            print(f"Expected PSNR: {current_expected_psnr}")
            
            # 最良モデルの保存
            if current_expected_psnr > best_expected_psnr:
                best_expected_psnr = current_expected_psnr
                best_model_state = {
                    'state_dict': decision_net.state_dict(),
                    'epoch': epoch,
                    'expected_psnr': current_expected_psnr
                }
            
            # 定期的なモデル保存
            if epoch % 100 == 0 and epoch > 0:
                if not os.path.exists(self.config.save_dir):
                    os.makedirs(self.config.save_dir, exist_ok=True)
                if best_model_state is not None:
                    save_path = os.path.join(
                        self.config.save_dir, 
                        f'select_model_{best_model_state["epoch"]}_{best_model_state["expected_psnr"]:.4f}.pth'
                    )
                    torch.save(best_model_state, save_path)
        
        # 最終的な最良モデルを返す
        return decision_net, best_model_state