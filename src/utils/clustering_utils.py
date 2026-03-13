"""
クラスタリングに関連するユーティリティ関数
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from sklearn.metrics import silhouette_score


def visualize_cluster_distribution(labels, output_path):
    """
    クラスタごとのパッチ数の分布をヒストグラムで可視化
    
    Args:
        labels: クラスタリングラベル
        output_path: 保存先パス
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    plt.figure(figsize=(10, 6))
    plt.hist(counts, bins=30)
    plt.title('クラスタごとのパッチ数の分布')
    plt.xlabel('パッチ数')
    plt.ylabel('クラスタ数')
    plt.savefig(output_path)
    plt.close()
    
    return counts


def evaluate_clustering(data, labels, metric='silhouette'):
    """
    クラスタリング結果を評価する
    
    Args:
        data: クラスタリングに使われたデータ
        labels: クラスタリングラベル
        metric: 評価指標 ('silhouette' など)
    
    Returns:
        score: 評価スコア
    """
    if metric == 'silhouette':
        # シルエットスコアの計算 (値が高いほど良いクラスタリング)
        score = silhouette_score(data, labels, sample_size=min(10000, len(data)))
    else:
        raise ValueError(f"未サポートの評価指標: {metric}")
    
    return score


def load_mat_files(directory, expected_shape=None):
    """
    ディレクトリからmatファイルを読み込む
    
    Args:
        directory: 読み込むディレクトリパス
        expected_shape: 期待されるテンソルの形状 (チェック用)
    
    Returns:
        data_list: 読み込まれたデータのリスト
        file_names: ファイル名のリスト
    """
    data_list = []
    file_names = []
    
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".mat"):
            filepath = os.path.join(directory, filename)
            try:
                mat = loadmat(filepath)
                key = next((k for k in mat if not k.startswith("__")), None)
                
                if key is None:
                    continue
                
                data = mat[key]
                
                if expected_shape and data.shape != expected_shape:
                    print(f"警告: {filename} の形状が無効です: {data.shape}, 期待値: {expected_shape}")
                    continue
                
                data_list.append(data)
                file_names.append(filename)
                
            except Exception as e:
                print(f"ファイル {filename} の読み込み中にエラー: {e}")
    
    return data_list, file_names


def extract_representative_patches(cluster_data, n_samples=5, method='central'):
    """
    各クラスタから代表的なパッチを抽出
    
    Args:
        cluster_data: クラスタ内のパッチデータ
        n_samples: 抽出するサンプル数
        method: 抽出方法 ('central'=中心に近いもの, 'random'=ランダム)
    
    Returns:
        representative_patches: 代表的なパッチのリスト
    """
    if len(cluster_data) <= n_samples:
        return cluster_data
    
    if method == 'central':
        # クラスタの中心に近いパッチを選択
        centroid = np.mean(cluster_data.reshape(len(cluster_data), -1), axis=0)
        flat_data = cluster_data.reshape(len(cluster_data), -1)
        
        # 中心からの距離を計算
        distances = np.linalg.norm(flat_data - centroid, axis=1)
        
        # 最も近いn_samplesを取得
        closest_indices = np.argsort(distances)[:n_samples]
        representative_patches = [cluster_data[i] for i in closest_indices]
        
    elif method == 'random':
        # ランダムに選択
        indices = np.random.choice(len(cluster_data), n_samples, replace=False)
        representative_patches = [cluster_data[i] for i in indices]
        
    else:
        raise ValueError(f"未サポートの抽出方法: {method}")
    
    return representative_patches