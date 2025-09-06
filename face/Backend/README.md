# 人脸识别与表情识别系统 - Backend

## 🎯 项目简介

这是一个基于深度学习的人脸识别与表情识别系统后端服务，为前端开发人员提供完整的API接口。

## 🚀 快速开始

### 1. 环境要求
- Python 3.8+
- 支持CUDA的GPU（可选，用于加速）

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 启动服务
```bash
python start_server.py
```

服务将在 `http://127.0.0.1:8000` 启动

## 📚 文档

- [API接口文档](./API接口文档.md) - 完整的API接口说明
- [前端开发指南](./前端开发指南.md) - 前端开发人员快速上手指南

## 🎨 前端示例

- `frontend_demo.html` - 完整的前端演示页面
- 在线演示：`http://127.0.0.1:8000/frontend`

## 📋 文件说明

- `api_server.py` - 主服务器文件
- `settings.py` - 配置文件
- `start_server.py` - 启动脚本
- `requirements.txt` - 依赖包列表
- `demo.py` - 演示脚本

## ⚠️ 注意事项

**模型文件需要单独下载**：
- `models/face_detection/best.pt`
- `models/face_recognition/16_backbone.pth`
- `models/emotion_recognition/emotion_best.pt`

这些模型文件较大（超过100MB），无法上传到GitHub，请单独获取。

## 📞 技术支持

如有问题，请查看API文档或联系开发团队。
