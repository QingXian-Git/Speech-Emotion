import os
import numpy as np # <-- 修复：补上缺失的numpy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import time
import uuid
import shutil
import signal
from typing import Optional
from datetime import datetime

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config import API_CONFIG, MODELS_DIR, CONFIDENCE_WEIGHTS
from predictor import ConfidencePredictor

app = FastAPI(
    title="面试自信度检测 API",
    description="基于语音情感识别的面试自信度评估系统",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = None

# ====================== 修复：自动创建必要文件夹 ======================
os.makedirs(API_CONFIG['upload_dir'], exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
# ====================================================================

def get_predictor():
    global predictor
    if predictor is None:
        try:
            predictor = ConfidencePredictor()
        except FileNotFoundError:
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "MODEL_NOT_FOUND",
                    "message": "模型未训练，请先运行 trainer.py 训练模型"
                }
            )
    return predictor


class WeightsConfig(BaseModel):
    emotion: float = 0.5
    speech_rate: float = 0.3
    pause: float = 0.2


class AnalysisRequest(BaseModel):
    audio_path: str


class AnalysisResponse(BaseModel):
    success: bool
    data: dict
    message: str


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "service": "confidence-detection"
    }


@app.get("/api/model/status")
async def get_model_status():
    model_path = os.path.join(MODELS_DIR, 'emotion_model.keras')
    model_exists = os.path.exists(model_path)

    return {
        "success": True,
        "model_loaded": predictor is not None,
        "model_exists": model_exists,
        "model_path": model_path if model_exists else None
    }


@app.get("/api/weights")
async def get_weights():
    return {
        "success": True,
        "weights": {
            "emotion": CONFIDENCE_WEIGHTS['emotion'],
            "speech_rate": CONFIDENCE_WEIGHTS['speech_rate'],
            "pause": CONFIDENCE_WEIGHTS['pause_frequency']
        },
        "message": "当前权重配置"
    }


@app.post("/api/weights")
async def set_weights(config: WeightsConfig):
    try:
        p = get_predictor()
        p.set_weights(
            emotion=config.emotion,
            speech_rate=config.speech_rate,
            pause=config.pause
        )

        return {
            "success": True,
            "weights": {
                "emotion": p.evaluator.emotion_weight,
                "speech_rate": p.evaluator.speech_rate_weight,
                "pause": p.evaluator.pause_weight
            },
            "message": "权重设置成功"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "INTERNAL_ERROR",
                "message": str(e)
            }
        )


@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_audio(request: AnalysisRequest):
    try:
        if not os.path.exists(request.audio_path):
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "FILE_NOT_FOUND",
                    "message": f"音频文件不存在: {request.audio_path}"
                }
            )

        p = get_predictor()
        result = p.predict_without_plot(request.audio_path)
        
        # 生成评价内容
        evaluation = generate_evaluation(result)

        return {
            "success": True,
            "data": {"evaluation": evaluation},
            "message": "分析完成"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "INTERNAL_ERROR",
                "message": str(e)
            }
        )


@app.post("/api/analyze/upload")
async def analyze_uploaded_audio(
        file: UploadFile = File(...)
):
    try:
        if not file.filename.lower().endswith(('.wav', '.mp3', '.ogg', '.flac')):
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "INVALID_FILE_TYPE",
                    "message": "仅支持 WAV, MP3, OGG, FLAC 格式的音频文件"
                }
            )

        file_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_ext = os.path.splitext(file.filename)[1]
        saved_filename = f"upload_{timestamp}_{file_id}{file_ext}"
        saved_path = os.path.join(API_CONFIG['upload_dir'], saved_filename)

        with open(saved_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        p = get_predictor()
        result = p.predict_without_plot(saved_path)
        
        # 生成评价内容
        evaluation = generate_evaluation(result)

        return {
            "success": True,
            "data": {"evaluation": evaluation},
            "message": "分析完成"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "INTERNAL_ERROR",
                "message": str(e)
            }
        )


def generate_evaluation(result):
    """
    根据分析结果生成评价内容
    """
    confidence_score = result['confidence_score']
    predicted_emotion = result['predicted_emotion']
    speech_rate = result['speech_rate_features']['speech_rate']
    speech_rate_stability = result['speech_rate_features']['speech_rate_stability']
    pause_frequency = result['pause_features']['pause_frequency']
    avg_pause_duration = result['pause_features']['avg_pause_duration']
    
    # 基于分数段和各项指标生成评价
    if confidence_score >= 80:
        evaluation = "您的表现非常自信！"
        if predicted_emotion in ['neutral', 'happy']:
            evaluation += " 您的情绪稳定积极，给人良好的印象。"
        if speech_rate_stability > 0.8:
            evaluation += " 语速稳定，表达流畅自然。"
        if pause_frequency < 0.5:
            evaluation += " 停顿适中，逻辑清晰。"
    elif confidence_score >= 60:
        evaluation = "您的表现较为自信。"
        if predicted_emotion in ['angry', 'fear', 'sad']:
            evaluation += " 建议调整情绪状态，保持积极稳定。"
        if speech_rate_stability < 0.6:
            evaluation += " 语速有些不稳定，建议练习保持均匀的语速。"
        if pause_frequency > 1.0:
            evaluation += " 停顿稍多，可适当减少不必要的停顿。"
    elif confidence_score >= 40:
        evaluation = "您的表现一般，有提升空间。"
        if predicted_emotion in ['angry', 'fear', 'sad']:
            evaluation += " 注意控制情绪，保持平稳的心态。"
        if speech_rate < 2.0:
            evaluation += " 语速偏慢，可适当加快以展现自信。"
        elif speech_rate > 6.0:
            evaluation += " 语速偏快，建议适当放慢以提高表达清晰度。"
        if avg_pause_duration > 1.0:
            evaluation += " 停顿时长较长，建议缩短停顿时间。"
    else:
        evaluation = "您需要在自信度方面多加提升。"
        evaluation += " 建议多练习演讲技巧，培养自信心。"
        if predicted_emotion in ['fear', 'sad']:
            evaluation += " 注意调整心态，保持积极情绪。"
        if speech_rate_stability < 0.5:
            evaluation += " 语速不稳定，建议通过练习提高表达的流畅度。"
        if pause_frequency > 1.5:
            evaluation += " 停顿频繁，可通过准备和练习减少不必要的停顿。"
    
    return evaluation


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail
        }
    )


def signal_handler(sig, frame):
    print(f"\n\n{'=' * 60}")
    print("服务已停止")
    print(f"{'=' * 60}\n")
    sys.exit(0)


if __name__ == '__main__':
    import uvicorn

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print(f"\n{'=' * 60}")
    print("        面试自信度检测 API 服务")
    print(f"{'=' * 60}")
    print(f"")
    print(f"  服务地址: http://localhost:{API_CONFIG['port']}")
    print(f"  API 文档: http://localhost:{API_CONFIG['port']}/docs")
    print(f"  交互文档: http://localhost:{API_CONFIG['port']}/redoc")
    print(f"")
    print(f"{'─' * 60}")
    print(f"  停止服务: 按 Ctrl+C")
    print(f"{'=' * 60}\n")

    uvicorn.run(
        app,
        host=API_CONFIG['host'],
        port=API_CONFIG['port']
    )