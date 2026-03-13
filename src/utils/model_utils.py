import os
import torch
import sys
from pathlib import Path

# プロジェクトルートをPYTHONPATHに追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from models.reconstruction.ADMM_net import ADMM_net  # 実際のパスに合わせて修正する必要あり
from models.reconstruction.LearnableMask import LearnableMask


def load_model(model_path, device=None):
    """
    state_dict を使用してモデルを読み込む
    
    Args:
        model_path (str): state_dict ファイルのパス
        device (torch.device or None): 使用するデバイス（None の場合は自動選択）
        
    Returns:
        torch.nn.Module: 構築されたモデルに読み込まれた state_dict
    """
    import torch
    import os
    from models.reconstruction.ADMM_net import ADMM_net

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # モデルインスタンスを作成
    model = ADMM_net().to(device)

    # state_dict を読み込んで設定
    state_dict = torch.load(model_path, map_location=device)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        raise RuntimeError(f"state_dict のロードに失敗しました: {e}")
    
    return model


def create_learnable_mask(t=16, s=8, device=None):
    """
    学習可能なマスクを作成
    
    Args:
        t: 時間方向のサイズ
        s: 空間方向のサイズ
        device: 使用するデバイス
        
    Returns:
        LearnableMask インスタンス
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return LearnableMask(t=t, s=s).to(device)


def reset_model_mask(model, t=16, s=8):
    """
    モデルのマスクをリセット
    
    Args:
        model: 対象モデル
        t: 時間方向のサイズ
        s: 空間方向のサイズ
    """
    new_mask = create_learnable_mask(t, s, next(model.parameters()).device)
    model.mask = new_mask
    
def stop_grads(self, network):
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