import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from config import CLASS_LABELS, OUTPUT_DIR


plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def plot_class_distribution(y, title: str = "类别分布", save_path: str = None):
    unique, counts = np.unique(y, return_counts=True)
    
    plt.figure(figsize=(10, 6))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    
    bars = plt.bar(CLASS_LABELS, [counts[list(unique).index(i)] if i in unique else 0 for i in range(len(CLASS_LABELS))], 
                   color=colors, edgecolor='white', linewidth=1.2)
    
    plt.xlabel('情感类别', fontsize=12)
    plt.ylabel('样本数量', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 5,
                 f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"类别分布图已保存: {save_path}")
    
    plt.show()


def plot_feature_distribution(X, feature_indices: list = None, feature_names: list = None, 
                               title: str = "特征分布", save_path: str = None):
    if feature_indices is None:
        feature_indices = list(range(min(10, X.shape[1])))
    
    n_features = len(feature_indices)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for i, idx in enumerate(feature_indices):
        ax = axes[i]
        ax.hist(X[:, idx], bins=30, color='steelblue', alpha=0.7, edgecolor='white')
        
        name = feature_names[idx] if feature_names and idx < len(feature_names) else f'特征 {idx}'
        ax.set_xlabel(name, fontsize=10)
        ax.set_ylabel('频数', fontsize=10)
        ax.set_title(f'{name} 分布', fontsize=11)
        ax.grid(True, alpha=0.3)
    
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"特征分布图已保存: {save_path}")
    
    plt.show()


def plot_feature_correlation(X, feature_names: list = None, title: str = "特征相关性矩阵", 
                              save_path: str = None, max_features: int = 20):
    n_features = min(max_features, X.shape[1])
    X_subset = X[:, :n_features]
    
    if feature_names is None:
        feature_names = [f'特征 {i}' for i in range(n_features)]
    else:
        feature_names = feature_names[:n_features]
    
    corr_matrix = np.corrcoef(X_subset.T)
    
    plt.figure(figsize=(12, 10))
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    ax = sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='RdBu_r', 
                     center=0, square=True, linewidths=0.5,
                     xticklabels=feature_names, yticklabels=feature_names)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"相关性矩阵已保存: {save_path}")
    
    plt.show()


def plot_feature_importance(importance_scores: np.ndarray, feature_names: list = None,
                            title: str = "特征重要性", save_path: str = None, top_n: int = 20):
    if feature_names is None:
        feature_names = [f'特征 {i}' for i in range(len(importance_scores))]
    
    indices = np.argsort(importance_scores)[::-1][:top_n]
    
    plt.figure(figsize=(12, 8))
    
    bars = plt.barh(range(top_n), importance_scores[indices][::-1], 
                    color='steelblue', alpha=0.8, edgecolor='white')
    
    plt.yticks(range(top_n), [feature_names[i] for i in indices][::-1], fontsize=10)
    plt.xlabel('重要性得分', fontsize=12)
    plt.ylabel('特征', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"特征重要性图已保存: {save_path}")
    
    plt.show()


def generate_eda_report(X, y, feature_names: list = None):
    print(f"\n{'=' * 60}")
    print("探索性数据分析报告")
    print(f"{'=' * 60}")
    
    print(f"\n数据集基本信息:")
    print(f"  样本数量: {X.shape[0]}")
    print(f"  特征维度: {X.shape[1]}")
    print(f"  类别数量: {len(CLASS_LABELS)}")
    
    print(f"\n类别分布:")
    unique, counts = np.unique(y, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  {CLASS_LABELS[label]:10s}: {count} ({count/len(y)*100:.1f}%)")
    
    print(f"\n特征统计信息:")
    print(f"  均值范围: [{X.mean(axis=0).min():.4f}, {X.mean(axis=0).max():.4f}]")
    print(f"  标准差范围: [{X.std(axis=0).min():.4f}, {X.std(axis=0).max():.4f}]")
    print(f"  最小值: {X.min():.4f}")
    print(f"  最大值: {X.max():.4f}")
    
    plot_class_distribution(y, save_path=os.path.join(OUTPUT_DIR, 'class_distribution.png'))
    
    plot_feature_distribution(X, feature_indices=list(range(min(10, X.shape[1]))),
                              feature_names=feature_names,
                              save_path=os.path.join(OUTPUT_DIR, 'feature_distribution.png'))
    
    plot_feature_correlation(X, feature_names=feature_names,
                             save_path=os.path.join(OUTPUT_DIR, 'feature_correlation.png'))
