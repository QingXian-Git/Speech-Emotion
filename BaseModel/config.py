import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

DATA_CONFIG = {
    'data_path': r'd:\AudioContext\wav',
    'sample_rate': 22050,
    'max_signal_length': 48000,
    'n_mfcc': 40,
    'n_fft': 2048,
    'hop_length': 512,
}

CLASS_LABELS = ("angry", "fear", "happy", "neutral", "sad", "surprise")
NUM_CLASSES = len(CLASS_LABELS)

MODEL_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'n_estimators': 100,
    'max_depth': 10,
}

OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
SAMPLE_DATA_DIR = os.path.join(BASE_DIR, 'data')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SAMPLE_DATA_DIR, exist_ok=True)
