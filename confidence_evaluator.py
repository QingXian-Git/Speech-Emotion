import numpy as np
from typing import Dict, Tuple
from config import CONFIDENCE_WEIGHTS, EMOTION_CONFIDENCE_SCORES, CLASS_LABELS


class ConfidenceEvaluator:
    def __init__(self,
                 emotion_weight: float = None,
                 speech_rate_weight: float = None,
                 pause_weight: float = None):
        self.emotion_weight = emotion_weight or CONFIDENCE_WEIGHTS['emotion']
        self.speech_rate_weight = speech_rate_weight or CONFIDENCE_WEIGHTS['speech_rate']
        self.pause_weight = pause_weight or CONFIDENCE_WEIGHTS['pause_frequency']

        self._normalize_weights()

        self.emotion_scores = EMOTION_CONFIDENCE_SCORES

    def _normalize_weights(self):
        total = self.emotion_weight + self.speech_rate_weight + self.pause_weight
        if total > 0:
            self.emotion_weight /= total
            self.speech_rate_weight /= total
            self.pause_weight /= total

    def set_weights(self, emotion: float, speech_rate: float, pause: float):
        self.emotion_weight = emotion
        self.speech_rate_weight = speech_rate
        self.pause_weight = pause
        self._normalize_weights()

    def evaluate_emotion_confidence(self, emotion_probs: np.ndarray) -> Tuple[float, str]:
        confidence = 0.0
        for i, prob in enumerate(emotion_probs):
            emotion = CLASS_LABELS[i]
            confidence += prob * self.emotion_scores.get(emotion, 0.5)

        predicted_emotion = CLASS_LABELS[np.argmax(emotion_probs)]
        return confidence, predicted_emotion

    def evaluate_speech_rate_confidence(self, speech_rate_features: Dict) -> float:
        stability = speech_rate_features.get('speech_rate_stability', 0)
        speech_rate = speech_rate_features.get('speech_rate', 0)

        optimal_rate = 4.0
        rate_score = 1.0 - min(abs(speech_rate - optimal_rate) / max(optimal_rate, 0.001), 1.0)

        confidence = stability * 0.6 + rate_score * 0.4
        return confidence

    def evaluate_pause_confidence(self, pause_features: Dict) -> float:
        pause_frequency = pause_features.get('pause_frequency', 0)
        avg_pause_duration = pause_features.get('avg_pause_duration', 0)
        total_pause_ratio = pause_features.get('total_pause_ratio', 0)

        optimal_frequency = 0.5
        if pause_frequency <= optimal_frequency:
            freq_score = 1.0
        else:
            freq_score = 1.0 - min((pause_frequency - optimal_frequency) / optimal_frequency, 1.0)

        optimal_avg_pause = 0.5
        if avg_pause_duration <= optimal_avg_pause:
            avg_pause_score = 1.0
        else:
            avg_pause_score = 1.0 - min((avg_pause_duration - optimal_avg_pause) / optimal_avg_pause, 1.0)

        optimal_pause_ratio = 0.3
        if total_pause_ratio <= optimal_pause_ratio:
            ratio_score = 1.0
        else:
            ratio_score = 1.0 - min((total_pause_ratio - optimal_pause_ratio) / optimal_pause_ratio, 1.0)

        confidence = freq_score * 0.3 + avg_pause_score * 0.4 + ratio_score * 0.3
        return confidence

    def evaluate_overall_confidence(self,
                                    emotion_probs: np.ndarray,
                                    speech_rate_features: Dict,
                                    pause_features: Dict) -> Dict:
        emotion_conf, predicted_emotion = self.evaluate_emotion_confidence(emotion_probs)
        speech_rate_conf = self.evaluate_speech_rate_confidence(speech_rate_features)
        pause_conf = self.evaluate_pause_confidence(pause_features)

        overall_confidence = (
                emotion_conf * self.emotion_weight +
                speech_rate_conf * self.speech_rate_weight +
                pause_conf * self.pause_weight
        )

        confidence_score = int(overall_confidence * 100)

        if confidence_score >= 80:
            level = "非常自信"
            suggestion = "表现优秀，继续保持！"
        elif confidence_score >= 60:
            level = "较为自信"
            suggestion = "表现良好，可以适当提升语速稳定性。"
        elif confidence_score >= 40:
            level = "一般"
            suggestion = "建议放松心态，保持稳定的语速，减少不必要的停顿。"
        else:
            level = "需要提升"
            suggestion = "建议多加练习，注意控制情绪，保持平稳的语速和适度的停顿。"

        return {
            'confidence_score': confidence_score,
            'confidence_level': level,
            'suggestion': suggestion,
            'predicted_emotion': predicted_emotion,
            'emotion_distribution': {CLASS_LABELS[i]: float(prob) for i, prob in enumerate(emotion_probs)},
            'emotion_confidence': float(emotion_conf),
            'speech_rate_confidence': float(speech_rate_conf),
            'pause_confidence': float(pause_conf),
            'speech_rate_features': speech_rate_features,
            'pause_features': pause_features,
            'weights_used': {
                'emotion': self.emotion_weight,
                'speech_rate': self.speech_rate_weight,
                'pause': self.pause_weight
            }
        }