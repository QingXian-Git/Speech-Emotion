import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import OUTPUT_DIR, MODEL_DIR
from data_loader import DataLoader
from baseline_model import BaselineModel
from eda_visualization import generate_eda_report
from report_generator import save_training_report, save_metrics_to_txt, save_data_summary


def main():
    print(f"\n{'#' * 60}")
    print("语音情感识别 - 基线模型训练 (SVM)")
    print(f"{'#' * 60}")
    
    print("\n[1] 加载数据")
    loader = DataLoader()
    X_train, X_test, y_train, y_test = loader.prepare_data(verbose=True)
    
    print("\n[2] 数据清洗与预处理")
    save_data_summary(X_train, X_test, y_train, y_test)
    
    print("\n[3] 探索性数据分析")
    feature_names = [f'MFCC_{i}_mean' for i in range(40)] + \
                    [f'MFCC_{i}_std' for i in range(40)] + \
                    [f'MFCC_{i}_delta_mean' for i in range(40)] + \
                    ['speech_rate', 'speech_rate_stability', 'pause_count', 
                     'avg_pause_duration', 'pause_frequency']
    
    generate_eda_report(X_train, y_train, feature_names=feature_names)
    
    print("\n[4] 训练基线模型")
    model = BaselineModel()
    model.train(X_train, y_train)
    
    print("\n[5] 模型评估")
    metrics = model.evaluate(X_test, y_test)
    
    print("\n保存模型和报告...")
    model.save_model()
    loader.save_scaler(os.path.join(MODEL_DIR, 'scaler.pkl'))
    
    save_metrics_to_txt(metrics)
    save_training_report(metrics, X_train, X_test, y_train, y_test, feature_names)

    print("训练完成")
    print(f"准确率: {metrics['accuracy']:.4f}")
    print(f"F1分数: {metrics['f1_macro']:.4f}")
    print(f"\n输出目录: {OUTPUT_DIR}")
    print(f"模型目录: {MODEL_DIR}")
    
    return metrics


if __name__ == '__main__':
    main()
