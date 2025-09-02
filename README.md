# 北外人脸识别与表情识别系统

## 🎯 项目简介

这是一个基于深度学习的人脸识别与表情识别系统，支持多种输入源（图片、视频、RTSP流、本地摄像头）的实时处理。

## ✨ 主要功能

- **人脸检测**: 使用YOLO模型进行高精度人脸检测
- **人脸识别**: 基于InsightFace的人脸特征提取与匹配
- **表情识别**: 支持7种基本表情的识别
- **多输入源支持**: 图片、本地视频、RTSP流、本地摄像头
- **实时处理**: WebSocket支持实时视频流处理
- **RESTful API**: 标准HTTP接口，易于集成
- **标准人脸对齐**: 使用InsightFace标准的5点关键点对齐方法

## 🏗️ 系统架构

```
前端 (HTML/JS) ←→ FastAPI后端 ←→ AI模型
                    ↓
                数据库管理
```

## 📁 项目结构

```
recognize/
├── api_server.py          # 主API服务器
├── start_server.py        # 服务器启动脚本
├── settings.py            # 配置文件
├── requirements.txt       # Python依赖
├── README.md             # 项目说明
├── API使用说明.md        # API使用文档
├── 配置使用说明.md       # 配置说明
├── PROJECT_STRUCTURE.md  # 项目结构说明
├── frontend_demo.html    # 前端演示页面
├── simple_test.html      # 简单测试页面
├── test_face_alignment.py # 人脸对齐测试脚本
├── models/               # AI模型目录
│   ├── face_detection/   # 人脸检测模型
│   ├── face_recognition/ # 人脸识别模型
│   └── emotion_recognition/ # 表情识别模型
└── database/             # 人脸数据库
    ├── aa/               # 人员A的照片
    ├── bb/               # 人员B的照片
    └── cc/               # 人员C的照片
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- CUDA支持（可选，用于GPU加速）

### 安装依赖

```bash
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

### 启动服务

```bash
python3 start_server.py
```

服务启动后，访问：
- API文档: http://localhost:8000/docs
- 前端演示: http://localhost:8000/frontend_demo.html

## 🔌 API接口

### 主要接口组

- **`/face/*`**: 人脸识别相关接口
- **`/emotion/*`**: 表情识别相关接口  
- **`/combined/*`**: 组合识别接口

### 支持的操作

- 图片识别
- 视频识别
- RTSP流识别
- 本地摄像头识别

## 📱 前端集成

系统提供了完整的前端演示页面，展示如何调用各种API接口：

- `frontend_demo.html`: 完整的API功能演示
- `simple_test.html`: 简单的接口测试

## ⚙️ 配置说明

所有配置参数都在 `settings.py` 文件中，包括：

- 识别阈值配置
- 性能参数配置
- 文件处理配置
- 安全配置

详细配置说明请参考 `配置使用说明.md`。

## 🔧 技术特性

### 人脸对齐
- 支持68点和5点关键点输入
- 使用标准InsightFace对齐方法
- 输出112x112标准化人脸图像

### 特征提取
- 使用测试时增强（TTA）
- 支持图像翻转增强
- 标准化的特征向量

### 模型架构
- 基于IResNet的InsightFace模型
- YOLO人脸检测模型
- 表情识别模型

## 🧪 测试

运行人脸对齐功能测试：
```bash
python3 test_face_alignment.py
```

## 📚 使用文档

- [API使用说明](API使用说明.md) - 详细的API调用指南
- [配置使用说明](配置使用说明.md) - 系统配置参数说明
- [项目结构说明](PROJECT_STRUCTURE.md) - 项目文件组织说明

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进项目。

## 📄 许可证

本项目仅供学习和研究使用。

## 📞 联系方式

如有问题，请通过GitHub Issues联系。

---

**北外项目** - 智能识别系统
