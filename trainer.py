import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import time
import numpy as np
import tensorflow as tf

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from model import EmotionRecognitionModel
from utils import prepare_data, plot_training_history, plot_confusion_matrix
from config import MODELS_DIR, CLASS_LABELS
from sklearn.metrics import classification_report


def check_gpu():
    print("\n" + "=" * 60)
    print("设备检测")
    print("=" * 60)

    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        print(f"✓ 检测到 {len(gpus)} 个GPU:")
        for i, gpu in enumerate(gpus):
            print(f"  [{i}] {gpu.name}")

        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✓ GPU内存动态分配已启用")
        except RuntimeError as e:
            print(f"GPU配置错误: {e}")

        print("\n>>> 将使用 GPU 进行训练 <<<")
    else:
        print("✗ 未检测到GPU")
        print(">>> 将使用 CPU 进行训练 <<<")

    print("=" * 60)


def train():
    check_gpu()

    print(f"\n{'=' * 60}")
    print("面试自信度模型训练")
    print(f"{'=' * 60}")

    print("\n[1/5] 准备数据")
    X_train, X_test, y_train, y_test, y_train_raw, y_test_raw = prepare_data()

    print("\n[2/5] 构建模型")
    input_shape = X_train.shape[1:]
    print(f"输入形状: {input_shape}")
    model = EmotionRecognitionModel(input_shape=input_shape)
    model.summary()

    print("\n[3/5] 开始训练")
    start_time = time.time()
    history = model.train(X_train, y_train, X_test, y_test)
    training_time = time.time() - start_time

    print(f"\n训练耗时: {training_time:.2f} 秒")

    print("\n[4/5] 评估模型")
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"测试集准确率: {test_acc * 100:.2f}%")
    print(f"测试集损失: {test_loss:.4f}")

    print("\n[5/5] 详细评估报告")
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print("\n分类报告:")
    print(classification_report(y_test_raw, y_pred_classes, target_names=CLASS_LABELS))

    plot_confusion_matrix(y_test_raw, y_pred_classes, CLASS_LABELS,
                          save_path=os.path.join(MODELS_DIR, 'confusion_matrix.png'))

    print("\n保存模型...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save_model('emotion_model')

    plot_training_history(history, save_path=os.path.join(MODELS_DIR, 'training_history.png'))

    print(f"\n{'=' * 60}")
    print("训练完成!")
    print(f"{'=' * 60}")

    return history, test_acc


if __name__ == '__main__':
    train()