import os
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
from config import DATA_CONFIG, CLASS_LABELS, MODEL_CONFIG, SAMPLE_DATA_DIR
from feature_extractor import FeatureExtractor


class DataLoader:
    def __init__(self, data_path: str = None):
        self.data_path = data_path or DATA_CONFIG['data_path']
        self.extractor = FeatureExtractor()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(CLASS_LABELS)

    def load_dataset(self, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        data = []
        labels = []
        
        cur_dir = os.getcwd()
        os.chdir(self.data_path)

        if verbose:
            print(f"\n{'=' * 60}")
            print("加载数据集")
            print(f"{'=' * 60}")
            print(f"数据集路径: {self.data_path}")

        for i, emotion in enumerate(CLASS_LABELS):
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
                    features = self.extractor.extract_all_features(filepath)
                    data.append(features)
                    labels.append(emotion)
                    file_count += 1
                except Exception as e:
                    if verbose:
                        print(f"错误!")
                    continue

            if verbose:
                print(f"已加载 {file_count} 个文件")

            os.chdir('..')

        os.chdir(cur_dir)

        if verbose:
            print(f"\n总计加载: {len(data)} 个音频文件")

        X = np.array(data)
        y = np.array(labels)
        
        return X, y

    def prepare_data(self, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X, y = self.load_dataset(verbose=verbose)
        
        y_encoded = self.label_encoder.transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded,
            test_size=MODEL_CONFIG['test_size'],
            random_state=MODEL_CONFIG['random_state'],
            stratify=y_encoded
        )
        
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        if verbose:
            print("数据集划分")
            print(f"训练集: {X_train.shape[0]} 样本")
            print(f"测试集: {X_test.shape[0]} 样本")
            print(f"特征维度: {X_train.shape[1]}")
            print(f"\n训练集类别分布:")
            unique, counts = np.unique(y_train, return_counts=True)
            for label, count in zip(unique, counts):
                print(f"  {CLASS_LABELS[label]:10s}: {count}")
            print(f"\n测试集类别分布:")
            unique, counts = np.unique(y_test, return_counts=True)
            for label, count in zip(unique, counts):
                print(f"  {CLASS_LABELS[label]:10s}: {count}")

        return X_train, X_test, y_train, y_test

    def save_scaler(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Scaler已保存: {path}")

    def load_scaler(self, path: str):
        with open(path, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"Scaler已加载: {path}")

    def create_sample_data(self, samples_per_class: int = 10, verbose: bool = True):
        cur_dir = os.getcwd()
        os.chdir(self.data_path)

        if verbose:
            print("采用精简版数据样本")

        for emotion in CLASS_LABELS:
            if not os.path.exists(emotion):
                continue

            os.chdir(emotion)
            files = [f for f in os.listdir('.') if f.endswith('.wav')]
            
            sample_files = files[:samples_per_class]
            
            emotion_dir = os.path.join(SAMPLE_DATA_DIR, emotion)
            os.makedirs(emotion_dir, exist_ok=True)
            
            import shutil
            for f in sample_files:
                src = os.path.join(os.getcwd(), f)
                dst = os.path.join(emotion_dir, f)
                shutil.copy2(src, dst)
            
            if verbose:
                print(f"  {emotion}: 复制了 {len(sample_files)} 个文件")

            os.chdir('..')

        os.chdir(cur_dir)
        
        if verbose:
            print(f"\n样本数据已保存到: {SAMPLE_DATA_DIR}")
