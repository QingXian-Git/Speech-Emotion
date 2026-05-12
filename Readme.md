> 

# 基于CNN+LSTM的语音识别模型与面试自信度判断

本项目基于CNN和LSTM架构训练的语音识别模型，通过该模型进行面试自信度判断，实现了语音情感识别，停顿识别与自动打分，引入了L2正则化和早停来优化模型效果



## 基础信息

- 服务名称：面试自信度检测 API
- 服务地址：http://localhost:8002（具体端口在配置文件中定义）
- API 文档：http://localhost:8002/docs
- 交互文档：http://localhost:8002/redoc

## 接口列表

### 1. 健康检查
- **路径**：`GET /health`
- **功能**：检查服务是否正常运行
- **请求参数**：无
- **响应**：
  ```json
  {
    "status": "healthy",
    "version": "1.0.0",
    "service": "confidence-detection"
  }
  ```

### 2. 模型状态查询
- **路径**：`GET /api/model/status`
- **功能**：查询模型加载状态
- **请求参数**：无
- **响应**：
  ```json
  {
    "success": true,
    "model_loaded": true,
    "model_exists": true,
    "model_path": "D:\\PythonCode\\Speech_Emotion\\Models\\emotion_model.keras"
  }
  ```

### 3. 获取权重配置
- **路径**：`GET /api/weights`
- **功能**：获取当前自信度评估的权重配置
- **请求参数**：无
- **响应**：
  ```json
  {
    "success": true,
    "weights": {
      "emotion": 0.5,  // 情感权重
      "speech_rate": 0.3,  // 语速权重
      "pause": 0.2  // 停顿权重
    },
    "message": "当前权重配置"
  }
  ```

### 4. 设置权重配置
- **路径**：`POST /api/weights`

- **功能**：设置自信度评估的权重配置

- **请求参数**：
  
  ```json
  {
    "emotion": 0.5,
    "speech_rate": 0.3,
    "pause": 0.2
  }
  ```
  
- **响应**：
  
  ```json
  {
    "success": true,
    "weights": {
      "emotion": 0.5,
      "speech_rate": 0.3,
      "pause": 0.2
    },
    "message": "权重设置成功"
  }
  ```

### 5. 分析音频文件（主要，其他可以在config.py中修改参数，也可保持现有参数不变）
- **路径**：`POST /api/analyze`
- **功能**：分析指定路径的音频文件
- **请求参数**：
  
  ```json
  {
    "audio_path": "(文件路径)"
  }
  ```
- **响应**：
  
  ```json
  {
    "success": true,
    "data": {
      "evaluation": "您的表现非常自信！ 您的情绪稳定积极，给人良好的印象。 语速稳定，表达流畅自然。"
    },
    "message": "分析完成"
  }
  ```

### 6. 上传并分析音频文件
- **路径**：`POST /api/analyze/upload`

- **功能**：上传音频文件并进行分析

- **请求参数**：
  - 文件：`file`（必需，支持WAV、MP3、OGG、FLAC格式）
  - 查询参数：`save_plot`（可选，默认false，是否保存分析图表）
  
  curl -X POST "http://localhost:8002/api/analyze/upload" -F "file=@D:\audio\interview.wav"
  
  参数说明
  
  - -X POST ：指定使用 POST 方法
  - -F ：指定表单数据
  - file=@D:\audio\interview.wav ：上传文件， @ 符号后是文件路径
    - Windows路径格式： @D:\audio\interview.wav 或 @D:/audio/interview.wav
    - 相对路径格式： @interview.wav 或 @./audio/interview.wav
  
- **响应**：
  
  ```json
  {
    "success": true,
    "data": {
      "evaluation": "您的表现较为自信。 建议调整语速，保持均匀的节奏。"
    },
    "message": "分析完成"
  }
  ```

## 错误响应格式

当请求失败时，API会返回以下格式的错误响应：

```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "详细错误说明"
  }
}
```

常见错误代码：
- `MODEL_NOT_FOUND`：模型未训练
- `FILE_NOT_FOUND`：音频文件不存在
- `INVALID_FILE_TYPE`：不支持的文件类型
- `INTERNAL_ERROR`：内部错误

## 技术说明

1. **模型加载**：系统会在首次请求时自动加载模型，如果模型未训练，会返回503错误
2. **文件处理**：上传的音频文件会保存在配置的上传目录中
3. **图表生成**：当`save_plot=true`时，系统会生成情感分布图和雷达图
4. **权重配置**：自信度评估基于情感、语速和停顿三个维度的加权计算

## 接口调用示例

### 1. 使用 curl 上传并分析音频文件

```bash
curl -X POST "http://localhost:8000/api/analyze/upload" -F "file=@path/to/audio.wav" -F "save_plot=true"
```

### 2. 使用 curl 设置权重

```bash
curl -X POST "http://localhost:8000/api/weights" -H "Content-Type: application/json" -d '{"emotion": 0.6, "speech_rate": 0.2, "pause": 0.2}'
```

### 3. 使用 curl 分析指定路径的音频

```bash
curl -X POST "http://localhost:8000/api/analyze" -H "Content-Type: application/json" -d '{"audio_path": "path/to/audio.wav", "save_plot": true}'
```

以上就是当前 API 的完整接口规范，包括各个接口的路径、请求参数和响应格式。
