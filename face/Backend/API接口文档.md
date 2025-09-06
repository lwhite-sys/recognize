# 人脸识别与表情识别系统 API 接口文档

## 📋 目录
- [系统概述](#系统概述)
- [基础信息](#基础信息)
- [认证与安全](#认证与安全)
- [核心功能接口](#核心功能接口)
- [WebSocket实时流接口](#websocket实时流接口)
- [调试接口](#调试接口)
- [错误码说明](#错误码说明)
- [前端开发指南](#前端开发指南)

---

## 🎯 系统概述

本系统提供基于深度学习的人脸识别与表情识别服务，支持图片、视频和实时流处理。

### 主要功能
- **人脸识别**：识别已知人员身份
- **表情识别**：识别7种基本表情（高兴、悲伤、愤怒、惊讶、恐惧、厌恶、中性）
- **综合识别**：同时进行人脸识别和表情识别
- **实时处理**：支持RTSP流和本地摄像头实时处理
- **批量处理**：支持批量图片处理

### 技术栈
- **后端框架**：FastAPI + Uvicorn
- **深度学习**：PyTorch + YOLO + InsightFace
- **图像处理**：OpenCV + NumPy
- **实时通信**：WebSocket
- **设备支持**：自动CUDA/CPU切换

---

## 📊 基础信息

### 服务器信息
- **基础URL**：`http://127.0.0.1:8000`
- **API文档**：`http://127.0.0.1:8000/docs`
- **前端演示**：`http://127.0.0.1:8000/frontend`

### 支持的媒体格式
- **图片**：JPG, JPEG, PNG, BMP
- **视频**：MP4, AVI, MOV, MKV
- **实时流**：RTSP, 本地摄像头

### 响应格式
所有接口返回JSON格式数据，标准响应结构：
```json
{
  "success": true/false,
  "data": {...},
  "message": "操作结果描述",
  "error": "错误信息（如果有）"
}
```

---

## 🔐 认证与安全

### CORS配置
系统已配置CORS中间件，支持跨域请求：
- 允许所有来源：`*`
- 允许所有HTTP方法：GET, POST, PUT, DELETE
- 允许所有请求头

### 文件上传限制
- **最大文件大小**：100MB
- **支持的文件类型**：图片、视频文件
- **并发上传**：支持多文件同时上传

---

## 🎯 核心功能接口

### 1. 人脸识别 - 图片

**接口地址**：`POST /face/recognize_image`

**功能描述**：对上传的图片进行人脸识别，返回识别结果

**请求参数**：
- `file` (File, 必需)：上传的图片文件

**请求示例**：
```javascript
const formData = new FormData();
formData.append('file', imageFile);

fetch('http://127.0.0.1:8000/face/recognize_image', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

**响应示例**：
```json
{
  "success": true,
  "faces": [
    {
      "box": [220, 510, 724, 1212],
      "confidence": 0.7207632064819336,
      "identity": "张三",
      "similarity": 0.85
    }
  ],
  "total_faces": 1,
  "debug_info": {
    "image_shape": [1296, 972, 3],
    "database_size": 3,
    "database_persons": ["张三", "李四", "王五"]
  }
}
```

**响应字段说明**：
- `faces`：检测到的人脸列表
  - `box`：人脸边界框 [x1, y1, x2, y2]
  - `confidence`：人脸检测置信度
  - `identity`：识别出的身份（Unknown表示未知）
  - `similarity`：与数据库中最相似特征的相似度
- `total_faces`：检测到的人脸总数
- `debug_info`：调试信息

### 2. 人脸识别 - 视频

**接口地址**：`POST /face/recognize_video`

**功能描述**：对上传的视频进行人脸识别，逐帧分析

**请求参数**：
- `file` (File, 必需)：上传的视频文件

**响应示例**：
```json
{
  "success": true,
  "total_frames": 300,
  "processed_frames": 30,
  "faces": [
    {
      "frame": 0,
      "faces": [
        {
          "box": [220, 510, 724, 1212],
          "confidence": 0.7207632064819336,
          "identity": "张三",
          "similarity": 0.85
        }
      ]
    }
  ]
}
```

### 3. 表情识别 - 图片

**接口地址**：`POST /emotion/recognize_image`

**功能描述**：对上传的图片进行表情识别

**请求参数**：
- `file` (File, 必需)：上传的图片文件

**响应示例**：
```json
{
  "success": true,
  "faces": [
    {
      "box": [220, 510, 724, 1212],
      "confidence": 0.7207632064819336,
      "emotion": "happy",
      "emotion_confidence": 0.92,
      "emotion_label": "高兴"
    }
  ],
  "total_faces": 1
}
```

**表情类型**：
- `happy`：高兴
- `sad`：悲伤
- `angry`：愤怒
- `surprised`：惊讶
- `fearful`：恐惧
- `disgusted`：厌恶
- `neutral`：中性

### 4. 表情识别 - 视频

**接口地址**：`POST /emotion/recognize_video`

**功能描述**：对上传的视频进行表情识别，逐帧分析

**请求参数**：
- `file` (File, 必需)：上传的视频文件

### 5. 综合识别 - 图片

**接口地址**：`POST /combined/recognize_image`

**功能描述**：同时进行人脸识别和表情识别

**请求参数**：
- `file` (File, 必需)：上传的图片文件

**响应示例**：
```json
{
  "success": true,
  "faces": [
    {
      "box": [220, 510, 724, 1212],
      "confidence": 0.7207632064819336,
      "identity": "张三",
      "similarity": 0.85,
      "emotion": "happy",
      "emotion_confidence": 0.92,
      "emotion_label": "高兴"
    }
  ],
  "total_faces": 1
}
```

### 6. 综合识别 - 视频

**接口地址**：`POST /combined/recognize_video`

**功能描述**：对视频同时进行人脸识别和表情识别

**请求参数**：
- `file` (File, 必需)：上传的视频文件

---

## 🔄 WebSocket实时流接口

### 1. 人脸识别 - RTSP流

**WebSocket地址**：`ws://127.0.0.1:8000/face/recognize_rtsp`

**功能描述**：对RTSP流进行实时人脸识别

**连接示例**：
```javascript
const ws = new WebSocket('ws://127.0.0.1:8000/face/recognize_rtsp');

ws.onopen = function() {
  // 发送RTSP地址
  ws.send(JSON.stringify({
    "rtsp_url": "rtsp://192.168.1.100:554/stream"
  }));
};

ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  console.log('识别结果:', data);
};
```

**接收数据格式**：
```json
{
  "frame": 150,
  "faces": [
    {
      "box": [220, 510, 724, 1212],
      "confidence": 0.7207632064819336,
      "identity": "张三",
      "similarity": 0.85
    }
  ],
  "timestamp": 1701415860.123
}
```

### 2. 人脸识别 - 本地摄像头

**WebSocket地址**：`ws://127.0.0.1:8000/face/recognize_camera`

**功能描述**：对本地摄像头进行实时人脸识别

**连接示例**：
```javascript
const ws = new WebSocket('ws://127.0.0.1:8000/face/recognize_camera');

ws.onopen = function() {
  // 发送摄像头ID
  ws.send(JSON.stringify({
    "camera_id": 0
  }));
};
```

### 3. 表情识别 - RTSP流

**WebSocket地址**：`ws://127.0.0.1:8000/emotion/recognize_rtsp`

**功能描述**：对RTSP流进行实时表情识别

### 4. 表情识别 - 本地摄像头

**WebSocket地址**：`ws://127.0.0.1:8000/emotion/recognize_camera`

**功能描述**：对本地摄像头进行实时表情识别

### 5. 综合识别 - 实时流

**WebSocket地址**：`ws://127.0.0.1:8000/combined/recognize_stream`

**功能描述**：对RTSP流或本地摄像头进行综合识别

**连接示例**：
```javascript
const ws = new WebSocket('ws://127.0.0.1:8000/combined/recognize_stream');

ws.onopen = function() {
  // 发送配置
  ws.send(JSON.stringify({
    "type": "rtsp",  // 或 "camera"
    "rtsp_url": "rtsp://192.168.1.100:554/stream"
    // 或者 "camera_id": 0
  }));
};
```

---

## 🔧 调试接口

### 1. 系统健康检查

**接口地址**：`GET /health`

**功能描述**：检查系统运行状态

**响应示例**：
```json
{
  "status": "healthy",
  "models_loaded": true
}
```

### 2. 特征数据库调试

**接口地址**：`GET /debug/features`

**功能描述**：显示数据库中的特征信息

**响应示例**：
```json
{
  "database_size": 3,
  "database_content": {
    "张三": {
      "feature_count": 5,
      "feature_dimensions": [[512], [512], [512]],
      "feature_samples": [
        {
          "index": 0,
          "shape": [512],
          "min_value": -1.2,
          "max_value": 1.8,
          "mean_value": 0.1,
          "norm": 15.6
        }
      ]
    }
  },
  "model_info": {
    "model_path": "models/face_recognition/16_backbone.pth",
    "model_exists": true
  }
}
```

### 3. 特征提取测试

**接口地址**：`GET /debug/test_extraction`

**功能描述**：测试人脸特征提取功能

### 4. 预处理方法测试

**接口地址**：`GET /debug/test_preprocessing`

**功能描述**：测试不同的图像预处理方法

### 5. 模型输出测试

**接口地址**：`GET /debug/test_model_output`

**功能描述**：测试模型输出和特征质量

### 6. YOLO模型测试

**接口地址**：`GET /debug/test_yolo`

**功能描述**：测试YOLO模型的人脸检测和关键点检测能力

### 7. 图像处理流程说明

**接口地址**：`GET /debug/image_processing_flow`

**功能描述**：获取详细的图像处理流程说明

### 8. 保存的调试图片

**接口地址**：`GET /debug/saved_images`

**功能描述**：查看保存的调试图片列表

### 9. 特征比较

**接口地址**：`GET /debug/compare_features`

**功能描述**：比较不同图片的特征向量

### 10. 阈值测试

**接口地址**：`GET /debug/threshold_test`

**功能描述**：测试不同阈值对识别结果的影响

---

## ❌ 错误码说明

### HTTP状态码
- `200`：请求成功
- `400`：请求参数错误
- `404`：资源不存在
- `500`：服务器内部错误

### 错误响应格式
```json
{
  "success": false,
  "error": "错误描述信息"
}
```

### 常见错误
- `无法读取图片`：图片格式不支持或文件损坏
- `无法连接RTSP流`：RTSP地址无效或网络问题
- `无法打开摄像头`：摄像头被占用或不存在

---

## 🎨 前端开发指南

### 1. 基础设置

**API基础URL**：
```javascript
const API_BASE = 'http://127.0.0.1:8000';
```

**CORS配置**：
```javascript
// 系统已配置CORS，无需额外设置
```

### 2. 文件上传示例

**单文件上传**：
```javascript
async function uploadImage(file) {
  const formData = new FormData();
  formData.append('file', file);
  
  try {
    const response = await fetch(`${API_BASE}/face/recognize_image`, {
      method: 'POST',
      body: formData
    });
    
    const result = await response.json();
    return result;
  } catch (error) {
    console.error('上传失败:', error);
    throw error;
  }
}
```

**多文件上传**：
```javascript
async function uploadMultipleImages(files) {
  const formData = new FormData();
  
  for (let i = 0; i < files.length; i++) {
    formData.append('files', files[i]);
  }
  
  try {
    const response = await fetch(`${API_BASE}/batch/process_images`, {
      method: 'POST',
      body: formData
    });
    
    const result = await response.json();
    return result;
  } catch (error) {
    console.error('批量上传失败:', error);
    throw error;
  }
}
```

### 3. 实时流处理示例

**WebSocket连接**：
```javascript
class RealTimeProcessor {
  constructor(streamType, streamConfig) {
    this.ws = null;
    this.streamType = streamType;
    this.streamConfig = streamConfig;
    this.onResult = null;
  }
  
  connect() {
    const wsUrl = `${API_BASE.replace('http', 'ws')}/${this.streamType}`;
    this.ws = new WebSocket(wsUrl);
    
    this.ws.onopen = () => {
      console.log('WebSocket连接已建立');
      this.ws.send(JSON.stringify(this.streamConfig));
    };
    
    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (this.onResult) {
        this.onResult(data);
      }
    };
    
    this.ws.onerror = (error) => {
      console.error('WebSocket错误:', error);
    };
    
    this.ws.onclose = () => {
      console.log('WebSocket连接已关闭');
    };
  }
  
  disconnect() {
    if (this.ws) {
      this.ws.close();
    }
  }
}

// 使用示例
const processor = new RealTimeProcessor('face/recognize_rtsp', {
  rtsp_url: 'rtsp://192.168.1.100:554/stream'
});

processor.onResult = (data) => {
  console.log('识别结果:', data);
  // 更新UI显示
  updateUI(data);
};

processor.connect();
```

### 4. 系统状态监控

**获取系统状态**：
```javascript
async function getSystemStatus() {
  try {
    const response = await fetch(`${API_BASE}/health`);
    const result = await response.json();
    return result;
  } catch (error) {
    console.error('获取系统状态失败:', error);
    throw error;
  }
}

// 定期监控系统状态
setInterval(async () => {
  const status = await getSystemStatus();
  updateSystemStatusUI(status);
}, 5000); // 每5秒更新一次
```

### 5. 错误处理

**统一错误处理**：
```javascript
function handleApiError(error) {
  if (error.response) {
    // 服务器返回错误
    const errorData = error.response.data;
    console.error('API错误:', errorData.error);
    showErrorMessage(errorData.error);
  } else if (error.request) {
    // 网络错误
    console.error('网络错误:', error.message);
    showErrorMessage('网络连接失败，请检查网络设置');
  } else {
    // 其他错误
    console.error('未知错误:', error.message);
    showErrorMessage('发生未知错误，请稍后重试');
  }
}

// 使用示例
try {
  const result = await uploadImage(file);
  handleSuccess(result);
} catch (error) {
  handleApiError(error);
}
```

### 6. UI组件建议

**文件上传组件**：
```html
<div class="upload-area" id="uploadArea">
  <input type="file" id="fileInput" accept="image/*,video/*" multiple style="display: none;">
  <div class="upload-text">
    <i class="upload-icon">📁</i>
    <p>点击或拖拽文件到此处上传</p>
    <p class="upload-hint">支持 JPG, PNG, MP4, AVI 等格式</p>
  </div>
</div>
```

**结果显示组件**：
```html
<div class="results-container">
  <div class="result-item" v-for="face in faces" :key="face.id">
    <div class="face-box" :style="getBoxStyle(face.box)">
      <div class="face-info">
        <span class="identity">{{ face.identity }}</span>
        <span class="confidence">{{ (face.confidence * 100).toFixed(1) }}%</span>
        <span class="similarity" v-if="face.similarity">相似度: {{ (face.similarity * 100).toFixed(1) }}%</span>
        <span class="emotion" v-if="face.emotion">{{ face.emotion_label }}</span>
      </div>
    </div>
  </div>
</div>
```

### 7. 性能优化建议

**图片预处理**：
```javascript
function preprocessImage(file) {
  return new Promise((resolve) => {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const img = new Image();
    
    img.onload = () => {
      // 限制图片最大尺寸
      const maxSize = 1920;
      let { width, height } = img;
      
      if (width > maxSize || height > maxSize) {
        const ratio = Math.min(maxSize / width, maxSize / height);
        width *= ratio;
        height *= ratio;
      }
      
      canvas.width = width;
      canvas.height = height;
      ctx.drawImage(img, 0, 0, width, height);
      
      canvas.toBlob(resolve, 'image/jpeg', 0.9);
    };
    
    img.src = URL.createObjectURL(file);
  });
}
```

**批量处理优化**：
```javascript
async function batchProcessWithLimit(files, limit = 3) {
  const results = [];
  
  for (let i = 0; i < files.length; i += limit) {
    const batch = files.slice(i, i + limit);
    const batchPromises = batch.map(file => uploadImage(file));
    
    const batchResults = await Promise.all(batchPromises);
    results.push(...batchResults);
    
    // 添加延迟避免服务器过载
    if (i + limit < files.length) {
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
  }
  
  return results;
}
```

---

## 📝 更新日志

### v1.0.0 (2024-01-01)
- ✅ 初始版本发布
- ✅ 支持人脸识别和表情识别
- ✅ 支持图片、视频和实时流处理
- ✅ 支持CUDA/CPU自动切换
- ✅ 完整的调试接口
- ✅ WebSocket实时通信

---

## 📞 技术支持

如有问题或建议，请联系开发团队。

**系统信息**：
- 版本：1.0.0
- 更新时间：2024-01-01
- 技术支持：开发团队
