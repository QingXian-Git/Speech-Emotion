import numpy as np
import librosa
import warnings
from typing import Tuple, Dict
from config import AUDIO_CONFIG


class FeatureExtractor:
    def __init__(self):
        self.sample_rate = AUDIO_CONFIG['sample_rate']
        self.max_signal_length = AUDIO_CONFIG['max_signal_length']
        self.n_mfcc = AUDIO_CONFIG['n_mfcc']
        self.n_mels = AUDIO_CONFIG['n_mels']
        self.n_fft = AUDIO_CONFIG['n_fft']
        self.hop_length = AUDIO_CONFIG['hop_length']

    def load_audio(self, file_path: str) -> np.ndarray:
        # 忽略弃用警告
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                # 首先尝试标准加载方式
                signal, _ = librosa.load(file_path, sr=self.sample_rate)
            except Exception:
                try:
                    # 尝试使用audioread后端
                    signal, _ = librosa.load(file_path, sr=self.sample_rate, dtype=np.float32)
                except Exception as e:
                    raise RuntimeError(f"无法加载音频文件: {e}")
        return signal

    def normalize_length(self, signal: np.ndarray) -> np.ndarray:
        s_len = len(signal)
        if s_len < self.max_signal_length:
            pad_len = self.max_signal_length - s_len
            pad_rem = pad_len % 2
            pad_len //= 2
            signal = np.pad(signal, (pad_len, pad_len + pad_rem), 'constant', constant_values=0)
        else:
            pad_len = s_len - self.max_signal_length
            pad_len //= 2
            signal = signal[pad_len:pad_len + self.max_signal_length]
        return signal

    def extract_mfcc_with_delta(self, signal: np.ndarray) -> np.ndarray:
        mfcc = librosa.feature.mfcc(
            y=signal,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        features = np.vstack([mfcc, delta, delta2])
        return features.T

    def extract_speech_rate(self, signal: np.ndarray) -> Dict[str, float]:
        rms = librosa.feature.rms(y=signal, hop_length=self.hop_length)[0]
        threshold = np.mean(rms) * 0.3
        voiced_frames = rms > threshold

        frame_times = librosa.frames_to_time(
            np.arange(len(voiced_frames)),
            sr=self.sample_rate,
            hop_length=self.hop_length
        )

        voiced_changes = np.diff(voiced_frames.astype(int))
        voiced_starts = np.where(voiced_changes == 1)[0] + 1
        voiced_ends = np.where(voiced_changes == -1)[0] + 1

        if len(voiced_starts) == 0 or len(voiced_ends) == 0:
            return {
                'speech_rate': 0.0,
                'speech_rate_stability': 0.0,
                'avg_syllable_duration': 0.0,
                'syllable_duration_std': 0.0
            }

        if voiced_starts[0] > voiced_ends[0]:
            voiced_ends = voiced_ends[1:]
        if len(voiced_starts) > len(voiced_ends):
            voiced_starts = voiced_starts[:len(voiced_ends)]
        elif len(voiced_ends) > len(voiced_starts):
            voiced_ends = voiced_ends[:len(voiced_starts)]

        if len(voiced_starts) == 0:
            syllable_durations = []
        else:
            syllable_durations = []
            for start, end in zip(voiced_starts, voiced_ends):
                duration = frame_times[end] - frame_times[start]
                if duration > 0.05:
                    syllable_durations.append(duration)

        if len(syllable_durations) == 0:
            return {
                'speech_rate': 0.0,
                'speech_rate_stability': 0.0,
                'avg_syllable_duration': 0.0,
                'syllable_duration_std': 0.0
            }

        total_duration = len(signal) / self.sample_rate
        speech_rate = len(syllable_durations) / total_duration if total_duration > 0 else 0
        avg_duration = np.mean(syllable_durations)
        duration_std = np.std(syllable_durations)
        stability = 1.0 - min(duration_std / avg_duration, 1.0) if avg_duration > 0 else 0

        return {
            'speech_rate': speech_rate,
            'speech_rate_stability': stability,
            'avg_syllable_duration': avg_duration,
            'syllable_duration_std': duration_std
        }

    def extract_pause_features(self, signal: np.ndarray) -> Dict[str, float]:
        rms = librosa.feature.rms(y=signal, hop_length=self.hop_length)[0]
        threshold = np.mean(rms) * 0.3

        is_silence = rms < threshold
        silence_changes = np.diff(is_silence.astype(int))
        pause_starts = np.where(silence_changes == 1)[0] + 1
        pause_ends = np.where(silence_changes == -1)[0] + 1

        if len(pause_starts) == 0:
            return {
                'pause_count': 0,
                'avg_pause_duration': 0.0,
                'pause_frequency': 0.0,
                'total_pause_ratio': 0.0
            }

        if len(pause_starts) > 0 and len(pause_ends) > 0:
            if pause_starts[0] > pause_ends[0]:
                pause_ends = pause_ends[1:]
            if len(pause_starts) > len(pause_ends):
                pause_starts = pause_starts[:len(pause_ends)]
            elif len(pause_ends) > len(pause_starts):
                pause_ends = pause_ends[:len(pause_starts)]

        pause_durations = []
        frame_duration = self.hop_length / self.sample_rate

        for start, end in zip(pause_starts, pause_ends):
            duration = (end - start) * frame_duration
            if duration > 0.1:
                pause_durations.append(duration)

        total_duration = len(signal) / self.sample_rate
        pause_count = len(pause_durations)
        avg_pause = np.mean(pause_durations) if pause_durations else 0
        pause_frequency = pause_count / total_duration if total_duration > 0 else 0
        total_pause_ratio = sum(pause_durations) / total_duration if total_duration > 0 else 0

        return {
            'pause_count': pause_count,
            'avg_pause_duration': avg_pause,
            'pause_frequency': pause_frequency,
            'total_pause_ratio': total_pause_ratio
        }


    def extract_all_features(self, file_path: str) -> Tuple[np.ndarray, Dict, Dict]:
        signal = self.load_audio(file_path)
        
        # 语速和停顿分析使用完整音频（这是最重要的改进！）
        speech_rate_features = self.extract_speech_rate(signal)
        pause_features = self.extract_pause_features(signal)
        
        # MFCC特征提取：使用标准化长度的音频（保持模型兼容性）
        signal_norm = self.normalize_length(signal)
        mfcc_features = self.extract_mfcc_with_delta(signal_norm)

        return mfcc_features, speech_rate_features, pause_features

    def get_feature_vector(self, file_path: str) -> np.ndarray:
        mfcc, speech_rate, pause = self.extract_all_features(file_path)
        return mfcc