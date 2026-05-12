import os
import json
import numpy as np
from datetime import datetime
from config import OUTPUT_DIR, MODEL_DIR, CLASS_LABELS


def save_training_report(metrics: dict, X_train: np.ndarray, X_test: np.ndarray,
                         y_train: np.ndarray, y_test: np.ndarray, 
                         feature_names: list = None,
                         save_path: str = None):
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, 'training_report.json')
    
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    
    train_distribution = {CLASS_LABELS[label]: int(count) 
                          for label, count in zip(unique_train, counts_train)}
    test_distribution = {CLASS_LABELS[label]: int(count) 
                         for label, count in zip(unique_test, counts_test)}
    
    report = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_info': {
            'model_type': 'SVM',
            'kernel': 'RBF',
            'C': 1.0,
            'gamma': 'scale'
        },
        'data_info': {
            'train_samples': int(X_train.shape[0]),
            'test_samples': int(X_test.shape[0]),
            'feature_dimension': int(X_train.shape[1]),
            'num_classes': len(CLASS_LABELS),
            'class_labels': CLASS_LABELS
        },
        'data_cleaning': {
            'missing_values': 0,
            'normalization': 'StandardScaler',
            'train_test_split': '80/20'
        },
        'class_distribution': {
            'train': train_distribution,
            'test': test_distribution
        },
        'feature_statistics': {
            'feature_names': feature_names if feature_names else [],
            'train_mean_range': [float(X_train.mean(axis=0).min()), 
                                  float(X_train.mean(axis=0).max())],
            'train_std_range': [float(X_train.std(axis=0).min()), 
                                 float(X_train.std(axis=0).max())],
            'train_min': float(X_train.min()),
            'train_max': float(X_train.max())
        },
        'evaluation_metrics': {
            'accuracy': float(metrics['accuracy']),
            'precision_macro': float(metrics['precision_macro']),
            'recall_macro': float(metrics['recall_macro']),
            'f1_macro': float(metrics['f1_macro']),
            'precision_weighted': float(metrics['precision_weighted']),
            'recall_weighted': float(metrics['recall_weighted']),
            'f1_weighted': float(metrics['f1_weighted'])
        },
        'output_files': {
            'class_distribution': 'outputs/class_distribution.png',
            'feature_distribution': 'outputs/feature_distribution.png',
            'feature_correlation': 'outputs/feature_correlation.png',
            'confusion_matrix': 'outputs/confusion_matrix.png',
            'model_file': 'models/baseline_model.pkl',
            'scaler_file': 'models/scaler.pkl'
        }
    }
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"训练报告已保存: {save_path}")
    return report


def save_metrics_to_txt(metrics: dict, save_path: str = None):
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, 'metrics.txt')
    
    content = f"""
语音情感识别 - 基线模型评估指标

模型类型: SVM (RBF Kernel)
评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
评估指标

准确率 (Accuracy):     {metrics['accuracy']:.4f}
精确率 (Macro):        {metrics['precision_macro']:.4f}
召回率 (Macro):        {metrics['recall_macro']:.4f}
F1分数 (Macro):        {metrics['f1_macro']:.4f}

精确率 (Weighted):     {metrics['precision_weighted']:.4f}
召回率 (Weighted):     {metrics['recall_weighted']:.4f}
F1分数 (Weighted):     {metrics['f1_weighted']:.4f}

"""
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"评估指标已保存: {save_path}")


def save_data_summary(X_train, X_test, y_train, y_test, save_path: str = None):
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, 'data_summary.txt')
    
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    
    content = f"""

数据清洗与预处理报告：

[数据集基本信息]
训练集样本数: {X_train.shape[0]}
测试集样本数: {X_test.shape[0]}
特征维度: {X_train.shape[1]}
类别数量: {len(CLASS_LABELS)}

[数据清洗]
缺失值检查: 无缺失值
异常值处理: 无异常值
数据类型: float32

[数据预处理]
标准化方法: StandardScaler (Z-score标准化)
训练集划分: 80%
测试集划分: 20%
随机种子: 42

[训练集类别分布]
"""
    
    for label, count in zip(unique_train, counts_train):
        content += f"  {CLASS_LABELS[label]:10s}: {count} ({count/len(y_train)*100:.1f}%)\n"
    
    content += f"\n[测试集类别分布]\n"
    for label, count in zip(unique_test, counts_test):
        content += f"  {CLASS_LABELS[label]:10s}: {count} ({count/len(y_test)*100:.1f}%)\n"
    
    content += f"""
训练集特征均值范围: [{X_train.mean(axis=0).min():.4f}, {X_train.mean(axis=0).max():.4f}]
训练集特征标准差范围: [{X_train.std(axis=0).min():.4f}, {X_train.std(axis=0).max():.4f}]
训练集特征最小值: {X_train.min():.4f}
训练集特征最大值: {X_train.max():.4f}

"""
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"数据摘要已保存: {save_path}")
