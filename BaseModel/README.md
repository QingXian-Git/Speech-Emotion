# 语音情感识别 - 基线模型

## 项目结构

```
BaseModel/
├── config.py              # 配置文件
├── feature_extractor.py   # 特征提取模块
├── data_loader.py         # 数据加载模块
├── baseline_model.py      # 基线模型 (SVM)
├── eda_visualization.py   # EDA可视化模块
├── report_generator.py    # 报告生成模块
├── train.py               # 训练主脚本
├── 学号_姓名_第二次基线报告.ipynb  # Jupyter Notebook报告
├── requirements.txt       # 依赖文件
└── README.md              # 说明文档
```

## 环境要求

```bash
pip install numpy pandas matplotlib seaborn scikit-learn librosa
```

## 使用方法

### 训练基线模型

```bash
python train.py
```

### 运行 Jupyter Notebook

```bash
jupyter notebook 学号_姓名_第二次基线报告.ipynb
```

## 特征说明

### MFCC特征 (120维)
- MFCC均值 (40维)
- MFCC标准差 (40维)
- MFCC一阶差分均值 (40维)

### 附加特征 (5维)
- speech_rate: 语速
- speech_rate_stability: 语速稳定性
- pause_count: 停顿次数
- avg_pause_duration: 平均停顿时长
- pause_frequency: 停顿频率

## 基线模型

**支持向量机 (SVM)**
- 核函数: RBF (径向基函数)
- 适合中小规模数据集
- 在高维空间中表现优秀
- 对特征缩放敏感（已标准化处理）

## 评估指标

- Accuracy: 准确率
- Precision: 精确率
- Recall: 召回率
- F1-Score: F1分数

## 输出文件

### 报告文件
- `outputs/training_report.json`: 完整训练报告 (JSON格式)
- `outputs/metrics.txt`: 评估指标摘要
- `outputs/data_summary.txt`: 数据清洗与预处理报告

### 可视化图片
- `outputs/class_distribution.png`: 类别分布图
- `outputs/feature_distribution.png`: 特征分布图
- `outputs/feature_correlation.png`: 特征相关性矩阵
- `outputs/confusion_matrix.png`: 混淆矩阵

### 模型文件
- `models/baseline_model.pkl`: 保存的SVM模型
- `models/scaler.pkl`: 标准化预处理器
