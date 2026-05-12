import os

# ==================== 基础路径配置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==================== 数据集配置 ====================
DATA_PATH = r'd:\AudioContext\wav'  # 音频数据集路径

CLASS_LABELS = ("angry", "fear", "happy", "neutral", "sad", "surprise")  # 6种情感类别标签
NUM_CLASSES = len(CLASS_LABELS)  # 情感类别数量

# ==================== 模型保存目录 ====================
MODELS_DIR = os.path.join(BASE_DIR, 'Models')  # 模型保存目录
os.makedirs(MODELS_DIR, exist_ok=True)  # 如果目录不存在则创建

# ==================== 自信度评估权重配置 ====================
CONFIDENCE_WEIGHTS = {
    'emotion': 0.5,           # 情感因素在自信度评估中的权重
    'speech_rate': 0.3,        # 语速因素在自信度评估中的权重
    'pause_frequency': 0.2      # 停顿因素在自信度评估中的权重
}

# ==================== 情感自信度评分配置 ====================
# 每种情感对应的基础自信度分数（0.0-1.0）
EMOTION_CONFIDENCE_SCORES = {
    'neutral': 1.0,       # 正常语气：满分，自信度最高
    'happy': 0.8,         # 开心：略微加分，表现积极
    'surprise': 0.6,      # 惊讶：不扣分，属于中性偏积极
    'angry': 0.2,         # 生气：大幅减分，情绪不稳定
    'sad': 0.3,           # 悲伤：大幅减分，情绪低落
    'fear': 0.2          # 恐惧：大幅减分，缺乏自信
}

# ==================== 音频处理参数配置 ====================
AUDIO_CONFIG = {
    'sample_rate': 22050,       # 音频采样率 (Hz)
    'max_signal_length': 48000,     # 最大信号长度（用于统一音频长度
    'n_mfcc': 40,                # MFCC特征数量
    'n_mels': 128,                  # Mel滤波器数量
    'n_fft': 2048,                 # FFT窗口大小
    'hop_length': 512,             # 帧移长度
}

# ==================== 模型训练参数配置 ====================
TRAINING_CONFIG = {
    'batch_size': 32,              # 批次大小
    'epochs': 100,               # 最大训练轮数
    'learning_rate': 0.001,      # 学习率
    'dropout_rate': 0.01,          # Dropout比例
    'test_size': 0.2,              # 测试集比例
    'random_state': 42                # 随机种子
}

# ==================== API服务配置 ====================
API_CONFIG = {
    'host': '0.0.0.0',                 # API服务监听地址
    'port': 8002,                        # API服务端口号
    'upload_dir': os.path.join(BASE_DIR, 'uploads'),  # 上传文件保存目录
    'max_upload_size': 50 * 1024 * 1024  # 最大上传文件大小
}

os.makedirs(API_CONFIG['upload_dir'], exist_ok=True)  # 创建上传目录