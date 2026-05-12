import numpy as np
import librosa
import warnings
from typing import Tuple, Dict
from config import DATA_CONFIG


class FeatureExtractor:
    def __init__(self):
        self.sample_rate = DATA_CONFIG['sample_rate']
        self.max_signal_length = DATA_CONFIG['max_signal_length']
        self.n_mfcc = DATA_CONFIG['n_mfcc']
        self.n_fft = DATA_CONFIG['n_fft']
        self.hop_length = DATA_CONFIG['hop_length']

    def load_audio(self, file_path: str) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                signal, _ = librosa.load(file_path, sr=self.sample_rate)
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

    def extract_mfcc_features(self, signal: np.ndarray) -> np.ndarray:
        mfcc = librosa.feature.mfcc(
            y=signal,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta_mean = np.mean(mfcc_delta, axis=1)
        features = np.concatenate([mfcc_mean, mfcc_std, mfcc_delta_mean])
        return features

    def extract_speech_rate(self, signal: np.ndarray) -> Dict[str, float]:
        rms = librosa.feature.rms(y=signal, hop_length=self.hop_length)[0]
        threshold = np.mean(rms) * 0.3
        voiced_frames = rms > threshold

        voiced_changes = np.diff(voiced_frames.astype(int))
        voiced_starts = np.where(voiced_changes == 1)[0] + 1
        voiced_ends = np.where(voiced_changes == -1)[0] + 1

        if len(voiced_starts) == 0 or len(voiced_ends) == 0:
            return {'speech_rate': 0.0, 'speech_rate_stability': 0.0}

        if voiced_starts[0] > voiced_ends[0]:
            voiced_ends = voiced_ends[1:]
        if len(voiced_starts) > len(voiced_ends):
            voiced_starts = voiced_starts[:len(voiced_ends)]
        elif len(voiced_ends) > len(voiced_starts):
            voiced_ends = voiced_ends[:len(voiced_starts)]

        syllable_durations = []
        frame_times = librosa.frames_to_time(
            np.arange(len(voiced_frames)),
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        for start, end in zip(voiced_starts, voiced_ends):
            duration = frame_times[end] - frame_times[start]
            if duration > 0.05:
                syllable_durations.append(duration)

        if len(syllable_durations) == 0:
            return {'speech_rate': 0.0, 'speech_rate_stability': 0.0}

        total_duration = len(signal) / self.sample_rate
        speech_rate = len(syllable_durations) / total_duration if total_duration > 0 else 0
        avg_duration = np.mean(syllable_durations)
        duration_std = np.std(syllable_durations)
        stability = 1.0 - min(duration_std / avg_duration, 1.0) if avg_duration > 0 else 0

        return {'speech_rate': speech_rate, 'speech_rate_stability': stability}

    def extract_pause_features(self, signal: np.ndarray) -> Dict[str, float]:
        rms = librosa.feature.rms(y=signal, hop_length=self.hop_length)[0]
        threshold = np.mean(rms) * 0.3

        is_silence = rms < threshold
        silence_changes = np.diff(is_silence.astype(int))
        pause_starts = np.where(silence_changes == 1)[0] + 1
        pause_ends = np.where(silence_changes == -1)[0] + 1

        if len(pause_starts) == 0:
            return {'pause_count': 0, 'avg_pause_duration': 0.0, 'pause_frequency': 0.0}

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

        return {'pause_count': pause_count, 'avg_pause_duration': avg_pause, 'pause_frequency': pause_frequency}

    def extract_all_features(self, file_path: str) -> np.ndarray:
        signal = self.load_audio(file_path)
        signal = self.normalize_length(signal)
        
        mfcc_features = self.extract_mfcc_features(signal)
        speech_rate_features = self.extract_speech_rate(signal)
        pause_features = self.extract_pause_features(signal)
        
        additional_features = np.array([
            speech_rate_features['speech_rate'],
            speech_rate_features['speech_rate_stability'],
            pause_features['pause_count'],
            pause_features['avg_pause_duration'],
            pause_features['pause_frequency']
        ])
        
        all_features = np.concatenate([mfcc_features, additional_features])
        return all_features
