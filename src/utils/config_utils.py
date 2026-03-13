import os
import yaml


def load_config(config_path):
    """
    YAML形式の設定ファイルを読み込む
    
    Args:
        config_path: 設定ファイルパス
        
    Returns:
        dict: 設定情報
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config, save_path):
    """
    設定を保存する
    
    Args:
        config: 設定辞書
        save_path: 保存先パス
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False)