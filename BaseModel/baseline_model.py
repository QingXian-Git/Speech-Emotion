import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from config import MODEL_CONFIG, CLASS_LABELS, MODEL_DIR, OUTPUT_DIR


plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class BaselineModel:
    def __init__(self):
        self.model = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            random_state=MODEL_CONFIG['random_state'],
            decision_function_shape='ovr'
        )

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        print("训练基线模型: SVM (RBF Kernel)")
        self.model.fit(X_train, y_train)
        print("训练完成!")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        y_pred = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_macro': precision_score(y_test, y_pred, average='macro'),
            'recall_macro': recall_score(y_test, y_pred, average='macro'),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'precision_weighted': precision_score(y_test, y_pred, average='weighted'),
            'recall_weighted': recall_score(y_test, y_pred, average='weighted'),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
        }

        print("模型评估结果:")
        print(f"准确率 (Accuracy): {metrics['accuracy']:.4f}")
        print(f"精确率 (Macro): {metrics['precision_macro']:.4f}")
        print(f"召回率 (Macro): {metrics['recall_macro']:.4f}")
        print(f"F1分数 (Macro): {metrics['f1_macro']:.4f}")
        print(f"\n分类报告:")
        print(classification_report(y_test, y_pred, target_names=CLASS_LABELS))
        
        cm = confusion_matrix(y_test, y_pred)
        self._plot_confusion_matrix(cm, save_path=os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
        
        return metrics

    def _plot_confusion_matrix(self, cm: np.ndarray, save_path: str = None):
        plt.figure(figsize=(10, 8))
        
        ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                         xticklabels=CLASS_LABELS,
                         yticklabels=CLASS_LABELS,
                         annot_kws={'size': 12})
        
        ax.set_xlabel('预测标签', fontsize=12)
        ax.set_ylabel('真实标签', fontsize=12)
        ax.set_title('混淆矩阵 - SVM', fontsize=14, fontweight='bold')
        
        ax.set_xticklabels(CLASS_LABELS, fontsize=10)
        ax.set_yticklabels(CLASS_LABELS, fontsize=10, rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"混淆矩阵已保存: {save_path}")
        
        plt.show()

    def save_model(self, path: str = None):
        if path is None:
            path = os.path.join(MODEL_DIR, 'baseline_model.pkl')
        
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"模型已保存: {path}")

    def load_model(self, path: str = None):
        if path is None:
            path = os.path.join(MODEL_DIR, 'baseline_model.pkl')
        
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"模型已加载: {path}")
