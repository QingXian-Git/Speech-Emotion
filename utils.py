import os
import sys
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from config import CLASS_LABELS, TRAINING_CONFIG, DATA_PATH, MODELS_DIR
from feature_extractor import FeatureExtractor

# 兼容不同版本的TensorFlow/Keras
try:
    from tensorflow.keras.utils import to_categorical
except ImportError:
    # 兼容旧版本
    from keras.utils import to_categorical

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set(font='SimHei')


def get_data(data_path: str = DATA_PATH,
             class_labels: Tuple = CLASS_LABELS,
             verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    extractor = FeatureExtractor()
    data = []
    labels = []

    cur_dir = os.getcwd()
    os.chdir(data_path)

    if verbose:
        print(f"\n{'=' * 60}")
        print("加载数据集")
        print(f"{'=' * 60}")
        print(f"数据集路径: {data_path}")

    for i, emotion in enumerate(class_labels):
        if not os.path.exists(emotion):
            if verbose:
                print(f"文件夹 '{emotion}' 不存在，跳过")
            continue

        if verbose:
            print(f"正在读取: {emotion} ...", end=" ")

        os.chdir(emotion)
        file_count = 0

        for filename in os.listdir('.'):
            if not filename.endswith('.wav'):
                continue

            filepath = os.path.join(os.getcwd(), filename)
            try:
                feature_vector = extractor.get_feature_vector(filepath)
                data.append(feature_vector)
                labels.append(i)
                file_count += 1
            except Exception as e:
                if verbose:
                    print(f"\n  警告: 处理 {filename} 时出错: {e}")
                continue

        if verbose:
            print(f"已加载 {file_count} 个文件")

        os.chdir('..')

    os.chdir(cur_dir)

    if verbose:
        print(f"\n总计加载: {len(data)} 个音频文件")

    return np.array(data), np.array(labels)


def prepare_data(data_path: str = DATA_PATH,
                 test_size: float = None,
                 random_state: int = None,
                 verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    
    参数:
        data_path: 数据集路径
        test_size: 测试集比例
        random_state: 随机种子
        verbose: 是否显示详细信息
    
    返回:
        X_train, X_test, y_train_cat, y_test_cat, y_train, y_test
    """
    test_size = test_size or TRAINING_CONFIG['test_size']
    random_state = random_state or TRAINING_CONFIG['random_state']

    X, y = get_data(data_path, verbose=verbose)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    y_train_cat = to_categorical(y_train, num_classes=len(CLASS_LABELS))
    y_test_cat = to_categorical(y_test, num_classes=len(CLASS_LABELS))

    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    if verbose:
        print(f"\n{'=' * 60}")
        print("数据集划分")
        print(f"{'=' * 60}")
        print(f"训练集: {X_train.shape[0]} 样本")
        print(f"测试集: {X_test.shape[0]} 样本")
        print(f"特征维度: {X_train.shape[1:3]}")
        print(f"测试集比例: {test_size * 100}%")
        print(f"\n训练集类别分布:")
        train_counts = np.bincount(y_train)
        for i, count in enumerate(train_counts):
            print(f"  {CLASS_LABELS[i]:10s}: {count}")
        print(f"\n测试集类别分布:")
        test_counts = np.bincount(y_test)
        for i, count in enumerate(test_counts):
            print(f"  {CLASS_LABELS[i]:10s}: {count}")

    return X_train, X_test, y_train_cat, y_test_cat, y_train, y_test


def plot_training_history(history, save_path: str = None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history['accuracy'], label='训练集', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='验证集', linewidth=2)
    axes[0].set_title('模型准确率', fontsize=12)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('准确率')
    axes[0].legend(prop={'family': 'SimHei'})
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history['loss'], label='训练集', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='验证集', linewidth=2)
    axes[1].set_title('模型损失', fontsize=12)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('损失')
    axes[1].legend(prop={'family': 'SimHei'})
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"训练曲线已保存: {save_path}")

    plt.show()


def plot_confidence_radar(confidence_result: dict, save_path: str = None):
    categories = ['情感自信度', '语速自信度', '停顿自信度']
    values = [
        confidence_result['emotion_confidence'],
        confidence_result['speech_rate_confidence'],
        confidence_result['pause_confidence']
    ]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='steelblue', alpha=0.25)
    ax.plot(angles, values, color='steelblue', linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title(f"自信度评分: {confidence_result['confidence_score']}分", size=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"自信度雷达图已保存: {save_path}")

    plt.show()


def plot_emotion_distribution(emotion_probs: np.ndarray, save_path: str = None):
    emotions = list(CLASS_LABELS)
    probs = emotion_probs.flatten()

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(emotions, probs, color=colors, edgecolor='white', linewidth=1.2)

    ax.set_xlabel('情感类别', fontsize=11)
    ax.set_ylabel('概率', fontsize=11)
    ax.set_title('情感识别概率分布', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')

    for bar, prob in zip(bars, probs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{prob:.2f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"情感分布图已保存: {save_path}")

    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_labels, save_path=None):
    cm = np.zeros((len(class_labels), len(class_labels)), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[true, pred] += 1

    plt.figure(figsize=(10, 8))

    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                     xticklabels=class_labels,
                     yticklabels=class_labels,
                     annot_kws={'size': 12})

    ax.set_xlabel('预测标签', fontsize=12)
    ax.set_ylabel('真实标签', fontsize=12)
    ax.set_title('混淆矩阵', fontsize=14, fontweight='bold')

    ax.set_xticklabels(class_labels, fontsize=10)
    ax.set_yticklabels(class_labels, fontsize=10, rotation=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"混淆矩阵已保存: {save_path}")

    plt.show()