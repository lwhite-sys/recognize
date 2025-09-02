#!/usr/bin/env python3
"""
启动人脸识别与表情识别API服务器
"""

import os
import sys
from pathlib import Path

def check_models():
    """检查模型文件是否存在"""
    models_dir = Path('models')
    required_models = [
        'face_detection/best.pt',
        'face_recognition/16_backbone.pth', 
        'emotion_recognition/emotion_best.pt'
    ]
    
    missing_models = []
    for model_path in required_models:
        if not (models_dir / model_path).exists():
            missing_models.append(model_path)
    
    if missing_models:
        print("❌ 缺少以下模型文件:")
        for model in missing_models:
            print(f"   - {model}")
        print("\n请确保模型文件已放置在正确位置")
        return False
    
    print("✅ 所有模型文件检查通过")
    return True

def check_database():
    """检查数据库目录"""
    database_dir = Path('database')
    if not database_dir.exists():
        print("📁 创建数据库目录...")
        database_dir.mkdir(exist_ok=True)
        print("✅ 数据库目录创建完成")
    else:
        print("✅ 数据库目录已存在")
    return True

def main():
    print("🚀 启动人脸识别与表情识别系统...")
    print("=" * 50)
    
    # 检查模型文件
    if not check_models():
        sys.exit(1)
    
    # 检查数据库目录
    if not check_database():
        sys.exit(1)
    
    print("\n🔧 启动API服务器...")
    print("📖 API文档地址: http://localhost:8001/docs")
    print("🌐 前端演示地址: http://localhost:8001/frontend_demo.html")
    print("🔍 设备状态检查: http://localhost:8001/debug/device_status")
    print("⏹️  按 Ctrl+C 停止服务器")
    print("=" * 50)
    
    try:
        # 启动服务器
        import uvicorn
        from api_server import app
        
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8001,
            reload=False,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n\n🛑 服务器已停止")
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

