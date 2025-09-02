# 🏗️ 项目结构说明

## 📁 目录结构

```
/beiwai/
├── 📄 api_server.py                    # 主API服务器（核心文件）
├── 📄 start_server.py                  # 服务器启动脚本
├── 📄 requirements.txt                 # Python依赖包列表
├── 📄 PROJECT_STRUCTURE.md            # 项目结构说明（本文件）
│
├── 📁 models/                          # 模型文件目录
│   ├── 📁 face_detection/             # 人脸检测模型
│   │   └── 📄 best.pt                 # YOLO人脸检测模型
│   ├── 📁 face_recognition/           # 人脸识别模型
│   │   └── 📄 backbone.pth            # InsightFace特征提取模型
│   └── 📁 emotion_recognition/        # 表情识别模型
│       └── 📄 emotion_best.pt         # YOLO表情识别模型
│
├── 📁 database/                        # 人脸数据库目录
│   ├── 📁 aa/                         # 人员aa的照片文件夹
│   │   ├── 📄 photo1.jpg
│   │   ├── 📄 photo2.jpg
│   │   └── ...
│   ├── 📁 bb/                         # 人员bb的照片文件夹
│   │   ├── 📄 photo1.jpg
│   │   └── ...
│   └── ...                            # 其他人员文件夹
│
├── 📁 examples/                        # 前端演示示例
│   ├── 📄 frontend_demo.html          # 基础功能演示页面
│   └── 📄 real_time_video_demo.html   # 实时视频识别演示页面
│
└── 📁 docs/                           # 项目文档
    ├── 📄 README.md                   # 项目主要说明文档
    └── 📄 FRONTEND_USAGE_GUIDE.md     # 前端使用指南
```

## 🎯 核心文件说明

### **📄 api_server.py**
- **作用**: 主要的FastAPI服务器，提供所有识别接口
- **功能**: 人脸识别、表情识别、综合识别
- **接口**: REST API + WebSocket
- **必需**: ✅ 必须保留

### **📄 start_server.py**
- **作用**: 服务器启动脚本，包含依赖检查
- **功能**: 自动检查模型文件、启动服务器
- **必需**: ✅ 必须保留

### **📄 requirements.txt**
- **作用**: Python依赖包列表
- **功能**: 一键安装所有依赖
- **必需**: ✅ 必须保留

## 🗂️ 模型文件说明

### **📁 models/face_detection/**
- **best.pt**: YOLO人脸检测模型
- **大小**: 约6MB
- **用途**: 检测图片/视频中的人脸位置

### **📁 models/face_recognition/**
- **backbone.pth**: InsightFace特征提取模型
- **大小**: 约249MB
- **用途**: 提取人脸特征，进行身份识别

### **📁 models/emotion_recognition/**
- **emotion_best.pt**: YOLO表情识别模型
- **大小**: 约21MB
- **用途**: 识别人脸表情类型

## 🗑️ 已删除的多余文件

以下文件已被清理，不再需要：

- `NEW_API_DESIGN.md` - 设计文档（已实现）
- `FRONTEND_API_GUIDE.md` - 重复的API指南
- `PERFORMANCE_OPTIMIZATION.md` - 性能优化文档
- `project_analysis.py` - 项目分析工具
- `test_complete_project.py` - 测试脚本
- `EMOTION_USAGE.md` - 重复的使用说明
- `test_emotion.py` - 测试脚本
- `test_camera.py` - 测试脚本
- `pth_inspector.py` - 模型检查工具
- `face_recognition_demo.py` - 演示脚本
- `demo.py` - MMDetection演示

## 🚀 快速开始

### **1. 安装依赖**
```bash
pip install -r requirements.txt
```

### **2. 启动服务器**
```bash
python3 start_server.py
```

### **3. 访问服务**
- 服务器: http://localhost:8000
- API文档: http://localhost:8000/docs
- 前端演示: 打开 `examples/` 目录下的HTML文件

## 📝 注意事项

1. **所有路径都是相对路径**，确保在项目根目录运行
2. **模型文件必须放在对应目录**，不能随意移动
3. **数据库目录会自动创建**，无需手动创建
4. **启动脚本会自动检查依赖**，确保服务正常运行

## 🔧 自定义配置

如需修改配置，请编辑 `api_server.py` 中的以下参数：

```python
# 模型路径（相对路径）
yolo_model_path = 'models/face_detection/best.pt'
insightface_model_path = 'models/face_recognition/backbone.pth'
emotion_model_path = 'models/emotion_recognition/emotion_best.pt'

# 数据库目录
database_dir = 'database/'

# 设备配置
device_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# 相似度阈值
sim_thr = 0.42
```
