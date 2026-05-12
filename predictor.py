import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import numpy as np
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from model import EmotionRecognitionModel
from feature_extractor import FeatureExtractor
from confidence_evaluator import ConfidenceEvaluator

from config import CLASS_LABELS, MODELS_DIR


class ConfidencePredictor:
    def __init__(self, model_name: str = 'emotion_model',
                 emotion_weight: float = 0.5,
                 speech_rate_weight: float = 0.3,
                 pause_weight: float = 0.2):

        model_path = os.path.join(MODELS_DIR, f'{model_name}.keras')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}\n请先运行 trainer.py 训练模型")

        self.model = EmotionRecognitionModel.load_model(model_name)
        self.extractor = FeatureExtractor()
        self.evaluator = ConfidenceEvaluator(
            emotion_weight=emotion_weight,
            speech_rate_weight=speech_rate_weight,
            pause_weight=pause_weight
        )

    def set_weights(self, emotion: float, speech_rate: float, pause: float):
        self.evaluator.set_weights(emotion, speech_rate, pause)

    def predict(self, audio_path: str) -> dict:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        print(f"\n{'=' * 60}")
        print(f"分析音频: {os.path.basename(audio_path)}")
        print(f"{'=' * 60}")

        mfcc_features, speech_rate_features, pause_features = self.extractor.extract_all_features(audio_path)

        input_data = mfcc_features[np.newaxis, ..., np.newaxis]

        emotion_probs = self.model.predict(input_data, verbose=0)[0]

        result = self.evaluator.evaluate_overall_confidence(
            emotion_probs, speech_rate_features, pause_features
        )

        self._print_result(result)

        return result

    def predict_without_plot(self, audio_path: str) -> dict:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        mfcc_features, speech_rate_features, pause_features = self.extractor.extract_all_features(audio_path)

        input_data = mfcc_features[np.newaxis, ..., np.newaxis]

        emotion_probs = self.model.predict(input_data, verbose=0)[0]

        result = self.evaluator.evaluate_overall_confidence(
            emotion_probs, speech_rate_features, pause_features
        )

        return result

    def _print_result(self, result: dict):
        print(f"\n{'─' * 50}")
        print(f"自信度评分: {result['confidence_score']} 分")
        print(f"自信度等级: {result['confidence_level']}")
        print(f"主要情感: {result['predicted_emotion']}")
        print(f"建议: {result['suggestion']}")
        print(f"\n详细分析:")
        print(f"  - 情感自信度: {result['emotion_confidence']:.2f}")
        print(f"  - 语速自信度: {result['speech_rate_confidence']:.2f}")
        print(f"  - 停顿自信度: {result['pause_confidence']:.2f}")
        print(f"\n情感分布:")
        for emotion, prob in result['emotion_distribution'].items():
            bar = '=' * int(prob * 20)
            print(f"  {emotion:10s}: {prob:.2f} {bar}")
        print(f"\n语速特征:")
        print(f"  - 语速: {result['speech_rate_features']['speech_rate']:.2f} 音节/秒")
        print(f"  - 语速稳定性: {result['speech_rate_features']['speech_rate_stability']:.2f}")
        print(f"\n停顿特征:")
        print(f"  - 停顿次数: {result['pause_features']['pause_count']}")
        print(f"  - 平均停顿时长: {result['pause_features']['avg_pause_duration']:.2f} 秒")
        print(f"  - 停顿频率: {result['pause_features']['pause_frequency']:.2f} 次/秒")
        print(f"{'─' * 50}")


def main():
    try:
        predictor = ConfidencePredictor()
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return

    predictor.set_weights(emotion=0.5, speech_rate=0.3, pause=0.2)

    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
    else:
        audio_path = input("\n请输入音频文件路径: ").strip().strip('"\'')

    try:
        result = predictor.predict(audio_path)
    except FileNotFoundError as e:
        print(f"错误: {e}")


if __name__ == '__main__':
    main()