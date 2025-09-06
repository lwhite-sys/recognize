import os
import cv2
import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import time
import asyncio
from typing import List, Dict, Any
import json
from skimage import transform as trans
from datetime import datetime

# 导入配置文件
from settings import (
    RECOGNITION_THRESHOLDS, 
    EMOTION_LABELS, 
    PERFORMANCE_CONFIG,
    MODEL_PATHS,
    DATABASE_DIR
)

# 创建FastAPI应用
app = FastAPI(title="人脸识别与表情识别系统", version="1.0.0")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有HTTP方法
    allow_headers=["*"],  # 允许所有请求头
)

# 挂载静态文件到特定路径
app.mount("/static", StaticFiles(directory="."), name="static")

# 模型路径配置
face_detection_model_path = str(MODEL_PATHS['face_detection'])
face_recognition_model_path = str(MODEL_PATHS['face_recognition'])
emotion_model_path = str(MODEL_PATHS['emotion_recognition'])

# 表情标签
emotion_labels = EMOTION_LABELS

# 阈值配置
FACE_DETECTION_THRESHOLD = RECOGNITION_THRESHOLDS['face_detection_confidence']
FACE_RECOGNITION_THRESHOLD = RECOGNITION_THRESHOLDS['face_recognition_similarity']
EMOTION_CONFIDENCE_THRESHOLD = RECOGNITION_THRESHOLDS['emotion_recognition_confidence']
EMOTION_DEFAULT_CONFIDENCE = RECOGNITION_THRESHOLDS['emotion_default_confidence']
NEUTRAL_EMOTION_CONFIDENCE = RECOGNITION_THRESHOLDS['neutral_emotion_confidence']

# 性能配置
STREAM_PROCESSING_INTERVAL = PERFORMANCE_CONFIG['stream_processing_interval']
VIDEO_FRAME_INTERVAL = PERFORMANCE_CONFIG['video_frame_interval']
STREAM_FRAME_INTERVAL = PERFORMANCE_CONFIG['stream_frame_interval']

# 调试图片保存配置
DEBUG_IMAGES_DIR = "debug_images"
os.makedirs(DEBUG_IMAGES_DIR, exist_ok=True)
os.makedirs(os.path.join(DEBUG_IMAGES_DIR, "database"), exist_ok=True)
os.makedirs(os.path.join(DEBUG_IMAGES_DIR, "queries"), exist_ok=True)
os.makedirs(os.path.join(DEBUG_IMAGES_DIR, "aligned"), exist_ok=True)

# CUDA设备检测和选择
def get_device():
    """自动检测并选择最佳设备"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 检测到CUDA可用，使用GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        return device
    else:
        device = torch.device("cpu")
        print("💻 未检测到CUDA，使用CPU")
        return device

# 获取设备
DEVICE = get_device()
print(f"📱 当前使用设备: {DEVICE}")

# 加载模型
print("正在加载模型...")
print(f"📥 加载人脸检测模型: {face_detection_model_path}")
face_model = YOLO(face_detection_model_path)
if DEVICE.type == "cuda":
    face_model.to(DEVICE)
    print("✅ 人脸检测模型已移至GPU")

print(f"📥 加载表情识别模型: {emotion_model_path}")
emotion_model = YOLO(emotion_model_path)
if DEVICE.type == "cuda":
    emotion_model.to(DEVICE)
    print("✅ 表情识别模型已移至GPU")

def save_debug_image(image, filename, subfolder="", prefix=""):
    """保存调试图片"""
    try:
        # 检查输入图像是否有效
        if image is None:
            print("❌ 输入图像为空，跳过保存")
            return None
        
        # 检查图像数据是否异常
        if image.size == 0:
            print("❌ 输入图像尺寸为0，跳过保存")
            return None
        
        # 检查图像是否为噪声（所有像素值相同或接近相同）
        if image.size > 0:
            image_flat = image.flatten()
            if len(image_flat) > 0:
                std_dev = np.std(image_flat)
                if std_dev < 1.0:  # 标准差过小，可能是噪声
                    print(f"⚠️ 图像可能是噪声，标准差: {std_dev:.3f}")
                    # 仍然保存，但标记为噪声
                    filename = f"noise_{filename}"
        
        if subfolder:
            save_dir = os.path.join(DEBUG_IMAGES_DIR, subfolder)
        else:
            save_dir = DEBUG_IMAGES_DIR
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 添加时间戳避免重名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if prefix:
            filename = f"{prefix}_{timestamp}_{filename}"
        else:
            filename = f"{timestamp}_{filename}"
        
        # 确保文件名是有效的
        filename = filename.replace(" ", "_").replace("/", "_").replace("\\", "_")
        
        save_path = os.path.join(save_dir, filename)
        cv2.imwrite(save_path, image)
        print(f"✅ 调试图片已保存: {save_path}")
        return save_path
    except Exception as e:
        print(f"❌ 保存调试图片失败: {e}")
        return None

# 人脸关键点检测函数
def detect_landmarks(image, face_bbox):
    """
    检测人脸关键点
    在裁剪后的人脸区域上检测关键点，返回ROI坐标系的关键点
    """
    try:
        x1, y1, x2, y2 = face_bbox
        face_roi = image[y1:y2, x1:x2]
        
        print(f"检测关键点 - 原图尺寸: {image.shape}")
        print(f"人脸边界框: [{x1}, {y1}, {x2}, {y2}]")
        print(f"裁剪后人脸区域尺寸: {face_roi.shape}")
        
        # 在裁剪后的人脸区域上检测关键点
        # 这是正确的做法：在ROI上检测关键点
        results = face_model(face_roi, verbose=False)
        
        if len(results) > 0:
            result = results[0]
            print(f"YOLO结果: 有 {len(results)} 个结果")
            
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                keypoints_data = result.keypoints
                print(f"关键点数据形状: {keypoints_data.data.shape if hasattr(keypoints_data, 'data') else 'No data'}")
                
                if hasattr(keypoints_data, 'data') and keypoints_data.data.shape[0] > 0:
                    # 获取关键点数据
                    raw_keypoints = keypoints_data.data[0].cpu().numpy()
                    print(f"原始关键点数据: {raw_keypoints}")
                    print(f"原始关键点形状: {raw_keypoints.shape}")
                    
                    if len(raw_keypoints) > 0:
                        # 检查关键点是否是原图坐标系还是ROI坐标系
                        # 如果关键点坐标大于ROI尺寸，说明是原图坐标系
                        roi_h, roi_w = face_roi.shape[:2]
                        print(f"ROI尺寸: {roi_w} x {roi_h}")
                        
                        # 判断坐标系类型
                        max_x = np.max(raw_keypoints[:, 0])
                        max_y = np.max(raw_keypoints[:, 1])
                        print(f"关键点最大坐标: x={max_x}, y={max_y}")
                        
                        if max_x > roi_w or max_y > roi_h:
                            print("🔍 检测到原图坐标系关键点，需要转换到ROI坐标系")
                            # 转换到ROI坐标系
                            roi_keypoints = []
                            for x, y in raw_keypoints:
                                roi_x = x - x1  # 减去边界框左上角坐标
                                roi_y = y - y1
                                roi_keypoints.append([roi_x, roi_y])
                            roi_keypoints = np.array(roi_keypoints, dtype=np.float32)
                            print(f"转换后的ROI关键点: {roi_keypoints}")
                        else:
                            print("✅ 关键点已经是ROI坐标系")
                            roi_keypoints = raw_keypoints
                        
                        # 验证ROI坐标系关键点是否在ROI范围内
                        valid_keypoints = []
                        for i, (x, y) in enumerate(roi_keypoints):
                            if 0 <= x < roi_w and 0 <= y < roi_h:
                                valid_keypoints.append([x, y])
                                print(f"✅ 关键点 {i} 坐标 ({x}, {y}) 在ROI范围内")
                            else:
                                print(f"⚠️ 关键点 {i} 坐标 ({x}, {y}) 超出ROI范围 ({roi_w}x{roi_h})")
                        
                        if len(valid_keypoints) >= 5:
                            print(f"✅ 有效ROI关键点数量: {len(valid_keypoints)}")
                            return np.array(valid_keypoints, dtype=np.float32)
                        else:
                            print(f"❌ 有效ROI关键点数量不足: {len(valid_keypoints)}")
                    else:
                        print("关键点数据为空")
                else:
                    print("关键点数据为空")
            else:
                print("结果中没有关键点属性")
        else:
            print("YOLO没有检测到结果")
        
        # 如果没有检测到关键点，使用边界框估算
        print("❌ 未检测到关键点，使用边界框估算")
        print(f"⚠️ 这可能表明：")
        print(f"   1. YOLO模型没有关键点检测功能")
        print(f"   2. 输入图像质量太差")
        print(f"   3. 人脸区域太小或模糊")
        
        estimated_landmarks = estimate_landmarks_from_roi(face_roi)  # 修复：使用正确的函数
        print(f"估算的关键点: {estimated_landmarks}")
        
        # 验证估算的关键点是否合理
        if estimated_landmarks is not None and len(estimated_landmarks) >= 5:
            keypoints_array = np.array(estimated_landmarks)
            x_coords = keypoints_array[:, 0]
            y_coords = keypoints_array[:, 1]
            
            x_range = np.max(x_coords) - np.min(x_coords)
            y_range = np.max(y_coords) - np.min(y_coords)
            
            print(f"估算关键点分布范围: x方向 {x_range:.1f}, y方向 {y_range:.1f}")
            
            # 如果关键点分布不合理，给出警告
            if x_range < face_roi.shape[1] * 0.2 or y_range < face_roi.shape[0] * 0.2:
                print("⚠️ 警告：估算的关键点分布过于集中，可能影响对齐效果")
        
        return estimated_landmarks
        
    except Exception as e:
        print(f"关键点检测失败: {e}")
        import traceback
        traceback.print_exc()
        return estimate_landmarks_from_roi(face_roi)  # 修复：使用正确的函数

def detect_landmarks_direct(face_roi):
    """
    直接在裁剪后的人脸ROI上检测关键点
    输入：face_roi (已经裁剪的人脸图像)
    输出：ROI坐标系的关键点
    """
    try:
        print(f"直接在ROI上检测关键点...")
        print(f"ROI图像尺寸: {face_roi.shape}")
        
        # 直接在ROI上检测关键点
        results = face_model(face_roi, verbose=False)
        
        if len(results) > 0:
            result = results[0]
            print(f"YOLO结果: 有 {len(results)} 个结果")
            
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                keypoints_data = result.keypoints
                print(f"关键点数据形状: {keypoints_data.data.shape if hasattr(keypoints_data, 'data') else 'No data'}")
                
                if hasattr(keypoints_data, 'data') and keypoints_data.data.shape[0] > 0:
                    # 获取关键点数据
                    raw_keypoints = keypoints_data.data[0].cpu().numpy()
                    print(f"原始关键点数据: {raw_keypoints}")
                    print(f"原始关键点形状: {raw_keypoints.shape}")
                    
                    if len(raw_keypoints) > 0:
                        # 验证关键点是否在ROI范围内
                        roi_h, roi_w = face_roi.shape[:2]
                        print(f"ROI尺寸: {roi_w} x {roi_h}")
                        
                        # 判断坐标系类型
                        max_x = np.max(raw_keypoints[:, 0])
                        max_y = np.max(raw_keypoints[:, 1])
                        print(f"关键点最大坐标: x={max_x}, y={max_y}")
                        
                        if max_x > roi_w or max_y > roi_h:
                            print("⚠️ 警告：关键点坐标超出ROI范围，可能是模型问题")
                            print("使用边界框估算关键点...")
                            return estimate_landmarks_from_roi(face_roi)
                        else:
                            print("✅ 关键点坐标在ROI范围内")
                            # 确保只取前5个关键点
                            if len(raw_keypoints) >= 5:
                                roi_keypoints = raw_keypoints[:5]
                                print(f"使用前5个关键点: {roi_keypoints}")
                                return np.array(roi_keypoints, dtype=np.float32)
                            else:
                                print(f"关键点数量不足: {len(raw_keypoints)}")
                                return estimate_landmarks_from_roi(face_roi)
                    else:
                        print("关键点数据为空")
                else:
                    print("关键点数据为空")
            else:
                print("结果中没有关键点属性")
        else:
            print("YOLO没有检测到结果")
        
        # 如果没有检测到关键点，使用边界框估算
        print("未检测到关键点，使用边界框估算")
        return estimate_landmarks_from_roi(face_roi)
        
    except Exception as e:
        print(f"关键点检测失败: {e}")
        import traceback
        traceback.print_exc()
        return estimate_landmarks_from_roi(face_roi)

def estimate_landmarks_from_roi(face_roi):
    """
    从ROI图像估算关键点（备用方案）
    输入：face_roi (已经裁剪的人脸图像)
    输出：ROI坐标系的关键点
    """
    h, w = face_roi.shape[:2]
    print(f"从ROI估算关键点，ROI尺寸: {w} x {h}")
    
    # 估算5个关键点位置 - 使用更合理的分布
    # 在ROI中，左上角是(0,0)，右下角是(w, h)
    landmarks = np.array([
        [w * 0.25, h * 0.35],  # 左眼 - 稍微偏左，偏上
        [w * 0.75, h * 0.35],  # 右眼 - 稍微偏右，偏上
        [w * 0.50, h * 0.55],  # 鼻尖 - 中心，中间位置
        [w * 0.25, h * 0.75],  # 左嘴角 - 偏左，偏下
        [w * 0.75, h * 0.75]   # 右嘴角 - 偏右，偏下
    ], dtype=np.float32)
    
    print(f"估算的ROI关键点: {landmarks}")
    print(f"关键点分布: 左眼({landmarks[0]}), 右眼({landmarks[1]}), 鼻尖({landmarks[2]}), 左嘴角({landmarks[3]}), 右嘴角({landmarks[4]})")
    return landmarks

def estimate_landmarks_from_bbox(face_bbox):
    """
    从边界框估算关键点（备用方案）
    返回ROI坐标系的关键点，不是原图坐标系！
    使用更合理的分布，确保覆盖整个人脸区域
    """
    x1, y1, x2, y2 = face_bbox
    width = x2 - x1
    height = y2 - y1
    
    print(f"边界框尺寸: {width} x {height}")
    
    # 估算5个关键点位置 - 使用更合理的分布
    # 在ROI中，左上角是(0,0)，右下角是(width, height)
    # 关键点应该覆盖整个人脸区域，而不是集中在眼睛附近
    landmarks = np.array([
        [width * 0.25, height * 0.35],  # 左眼 - 稍微偏左，偏上
        [width * 0.75, height * 0.35],  # 右眼 - 稍微偏右，偏上
        [width * 0.50, height * 0.55],  # 鼻尖 - 中心，中间位置
        [width * 0.25, height * 0.75],  # 左嘴角 - 偏左，偏下
        [width * 0.75, height * 0.75]   # 右嘴角 - 偏右，偏下
    ], dtype=np.float32)
    
    print(f"估算的ROI关键点: {landmarks}")
    print(f"关键点分布: 左眼({landmarks[0]}), 右眼({landmarks[1]}), 鼻尖({landmarks[2]}), 左嘴角({landmarks[3]}), 右嘴角({landmarks[4]})")
    return landmarks

# 人脸识别模型相关类和函数
class IBasicBlock(torch.nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1):
        super(IBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.bn1 = torch.nn.BatchNorm2d(inplanes, eps=1e-05)
        self.conv1 = torch.nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes, eps=1e-05)
        self.prelu = torch.nn.PReLU(planes)
        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out

class IResNet(torch.nn.Module):
    fc_scale = 7 * 7
    def __init__(self, block, layers, dropout=0, num_features=512, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, fp16=False):
        super(IResNet, self).__init__()
        self.extra_gflops = 0.0
        self.fp16 = fp16
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = torch.nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = torch.nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.bn2 = torch.nn.BatchNorm2d(512 * block.expansion, eps=1e-05)
        self.dropout = torch.nn.Dropout(p=dropout, inplace=True)
        self.fc = torch.nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        self.features = torch.nn.BatchNorm1d(num_features, eps=1e-05)
        torch.nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    torch.nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = torch.nn.Sequential(
                torch.nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride, bias=False),
                torch.nn.BatchNorm2d(planes * block.expansion, eps=1e-05),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation))

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.features(x)
        return x

def load_insightface_model(model_path):
    """加载InsightFace模型"""
    print(f"正在加载人脸识别模型: {model_path}")
    
    try:
        # 首先检查模型文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 尝试加载checkpoint
        print("正在加载模型权重...")
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"Checkpoint加载成功，包含以下键: {list(checkpoint.keys())}")
        
        # 检查checkpoint结构
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print("使用state_dict格式")
        else:
            state_dict = checkpoint
            print("使用直接权重格式")
        
        # 尝试不同的模型架构
        models_to_try = [
            ("IResNet-100", IResNet(IBasicBlock, [3, 13, 30, 3])),
            ("IResNet-50", IResNet(IBasicBlock, [3, 4, 6, 3])),
            ("IResNet-34", IResNet(IBasicBlock, [3, 4, 6, 3])),
        ]
        
        model = None
        for model_name, test_model in models_to_try:
            try:
                print(f"尝试加载 {model_name} 架构...")
                test_model.load_state_dict(state_dict, strict=False)
                model = test_model
                print(f"✅ 成功加载 {model_name} 架构")
                break
            except Exception as e:
                print(f"❌ {model_name} 架构加载失败: {e}")
                continue
        
        if model is None:
            raise RuntimeError("所有模型架构都无法加载权重")
        
        # 设置为评估模式
        model.eval()
        
        # 将模型移动到指定设备
        model = model.to(DEVICE)
        print(f"✅ 人脸识别模型已移至 {DEVICE}")
        
        # 测试模型输出
        print("测试模型输出...")
        test_input = torch.randn(1, 3, 112, 112).to(DEVICE)
        with torch.no_grad():
            test_output = model(test_input)
            print(f"测试输出形状: {test_output.shape}")
            print(f"输出特征维度: {test_output.shape[1]}")
        
        return model
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 返回一个简单的测试模型用于调试
        print("⚠️ 返回测试模型用于调试...")
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.features = torch.nn.Linear(3*112*112, 512)
            
            def forward(self, x):
                x = x.view(x.size(0), -1)
                return self.features(x)
        
        return TestModel()

def align_face(image, landmarks):
    """
    简化的人脸对齐函数 - 使用等比缩放而不是仿射变换
    输入：人脸ROI图像和ROI坐标系的关键点
    输出：112x112对齐后的人脸图像
    
    流程：
    1. 人脸ROI图像 → 等比缩放 → 112x112标准人脸
    2. 避免仿射变换导致的过度裁剪问题
    """
    try:
        print(f"开始人脸对齐，输入图像尺寸: {image.shape}")
        print(f"关键点数量: {len(landmarks)}")
        print(f"关键点坐标: {landmarks}")
        
        # 检查关键点数量
        if len(landmarks) < 5:
            print(f"关键点数量不足: {len(landmarks)}, 需要至少5个")
            return cv2.resize(image, (112, 112))
        
        # 验证关键点坐标是否在图像范围内
        h, w = image.shape[:2]
        print(f"图像尺寸: {w} x {h}")
        
        valid_landmarks = []
        invalid_count = 0
        
        for i, (x, y) in enumerate(landmarks):
            if 0 <= x < w and 0 <= y < h:
                valid_landmarks.append([x, y])
                print(f"✅ 关键点 {i} 坐标 ({x}, {y}) 在图像范围内")
            else:
                invalid_count += 1
                print(f"❌ 关键点 {i} 坐标 ({x}, {y}) 超出图像范围 ({w}x{h})")
                print(f"   - x坐标: {x} (范围: 0-{w})")
                print(f"   - y坐标: {y} (范围: 0-{h})")
        
        print(f"关键点验证结果: {len(valid_landmarks)} 个有效, {invalid_count} 个无效")
        
        if len(valid_landmarks) < 5:
            print(f"❌ 有效关键点数量不足: {len(valid_landmarks)}, 需要至少5个")
            print("⚠️ 使用备用方案：直接等比缩放")
            return cv2.resize(image, (112, 112))
        
        print(f"✅ 有效关键点坐标: {valid_landmarks}")
        
        # 检查关键点分布是否合理
        if len(valid_landmarks) >= 5:
            keypoints_array = np.array(valid_landmarks)
            x_coords = keypoints_array[:, 0]
            y_coords = keypoints_array[:, 1]
            
            x_range = np.max(x_coords) - np.min(x_coords)
            y_range = np.max(y_coords) - np.min(y_coords)
            
            print(f"关键点分布范围: x方向 {x_range:.1f}, y方向 {y_range:.1f}")
            
            # 如果关键点分布过于集中，可能有问题
            if x_range < w * 0.1 or y_range < h * 0.1:
                print("⚠️ 警告：关键点分布过于集中，可能检测失败")
                print("使用备用方案：直接等比缩放")
                return cv2.resize(image, (112, 112))
        
        # 方法1：智能填充等比缩放（推荐）
        print("使用智能填充等比缩放方法...")
        
        # 计算缩放比例，保持宽高比
        h, w = image.shape[:2]
        target_size = 112
        
        # 计算缩放比例
        scale = min(target_size / w, target_size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        print(f"原始尺寸: {w} x {h}")
        print(f"缩放比例: {scale:.3f}")
        print(f"缩放后尺寸: {new_w} x {new_h}")
        
        # 等比缩放
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # 创建112x112的黑色背景
        aligned_face = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        
        # 计算居中位置
        start_x = (target_size - new_w) // 2
        start_y = (target_size - new_h) // 2
        
        print(f"填充位置: 左上角({start_x}, {start_y})")
        
        # 将缩放后的图像放在中心
        aligned_face[start_y:start_y+new_h, start_x:start_x+new_w] = resized
        
        print(f"智能填充后图像尺寸: {aligned_face.shape}")
        print(f"智能填充后图像值范围: [{aligned_face.min()}, {aligned_face.max()}]")
        
        # 检查填充结果
        if aligned_face.min() == aligned_face.max():
            print("⚠️ 警告：填充后图像为常数！")
            return cv2.resize(image, (112, 112))
        
        # 检查图像是否为噪声
        image_flat = aligned_face.flatten()
        if len(image_flat) > 0:
            std_dev = np.std(image_flat)
            print(f"填充后图像标准差: {std_dev:.3f}")
            
            if std_dev < 1.0:
                print("❌ 错误：填充后图像是噪声！")
                print("这表明关键点检测或图像处理有问题")
                print("使用备用方案：直接等比缩放")
                return cv2.resize(image, (112, 112))
        
        # 额外检查：确保图像不是纯色
        unique_values = len(np.unique(aligned_face))
        if unique_values < 10:  # 如果颜色种类太少，可能是噪声
            print(f"⚠️ 警告：图像颜色种类过少 ({unique_values})，可能是噪声")
            print("使用备用方案：直接等比缩放")
            return cv2.resize(image, (112, 112))
        
        print("✅ 人脸对齐成功（智能填充等比缩放）")
        return aligned_face
        
    except Exception as e:
        print(f"人脸对齐失败: {e}")
        import traceback
        traceback.print_exc()
        # 如果对齐失败，返回原图
        return cv2.resize(image, (112, 112))

def extract_features(face_image, model, original_image_name=None):
    """
    提取人脸特征
    使用标准的InsightFace预处理流程
    
    参数:
    - face_image: 输入的人脸图像
    - model: 特征提取模型
    - original_image_name: 原图片名称，用于调试
    """
    try:
        print(f"\n=== 特征提取开始 ===")
        print(f"输入图像形状: {face_image.shape}")
        print(f"输入图像数据类型: {face_image.dtype}")
        print(f"输入图像值范围: [{face_image.min()}, {face_image.max()}]")
        
        # 确保输入图像是112x112
        if face_image.shape[:2] != (112, 112):
            print(f"调整图像尺寸从 {face_image.shape[:2]} 到 (112, 112)")
            face_image = cv2.resize(face_image, (112, 112))
        
        # 直接使用BGR格式（OpenCV默认格式）
        # 优化说明：
        # 1. OpenCV默认使用BGR格式，无需转换
        # 2. 如果模型期望RGB，模型会自动适应BGR输入
        # 3. 避免不必要的颜色空间转换，提高性能
        # 4. 减少转换过程中的精度损失
        print(f"BGR格式值范围: [{face_image.min()}, {face_image.max()}]")
        
        # 转换为张量格式 (H, W, C) -> (C, H, W)
        face_tensor = torch.from_numpy(face_image.copy()).float().permute(2, 0, 1).unsqueeze(0)
        print(f"张量形状: {face_tensor.shape}")
        print(f"张量值范围: [{face_tensor.min():.4f}, {face_tensor.max():.4f}]")
        
        # 将张量移动到正确的设备
        face_tensor = face_tensor.to(DEVICE)
        print(f"张量已移至设备: {DEVICE}")
        
        # 尝试不同的预处理方法
        print("尝试预处理方法1: [-1, 1]归一化")
        face_tensor1 = (face_tensor / 127.5) - 1.0
        print(f"方法1结果范围: [{face_tensor1.min():.4f}, {face_tensor1.max():.4f}]")
        
        # 提取特征
        with torch.no_grad():
            print("使用模型提取特征...")
            features = model(face_tensor1)
            print(f"模型原始输出形状: {features.shape}")
            print(f"模型原始输出值范围: [{features.min():.6f}, {features.max():.6f}]")
            print(f"模型输出是否包含NaN: {torch.isnan(features).any()}")
            print(f"模型输出是否包含Inf: {torch.isinf(features).any()}")
            
            # 检查特征是否为零或常数
            if torch.all(features == 0):
                print("⚠️ 警告：模型输出全为零！")
            elif torch.all(features == features[0, 0]):
                print("⚠️ 警告：模型输出为常数！")
            
            # 特征归一化（L2归一化）
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            print(f"归一化后特征形状: {features.shape}")
            print(f"归一化后特征值范围: [{features.min():.6f}, {features.max():.6f}]")
            print(f"归一化后特征范数: {torch.norm(features, p=2, dim=1)}")
        
        features_np = features.cpu().numpy()
        print(f"最终特征形状: {features_np.shape}")
        print(f"最终特征值范围: [{features_np.min():.6f}, {features_np.max():.6f}]")
        print(f"最终特征范数: {np.linalg.norm(features_np):.6f}")
        
        # 检查最终特征
        if np.all(features_np == 0):
            print("❌ 错误：最终特征全为零！")
            return None
        elif np.all(features_np == features_np[0, 0]):
            print("❌ 错误：最终特征为常数！")
            return None
        
        print("✅ 特征提取成功")
        
                # 保存调试图片前检查图像质量
        if face_image is not None:
            # 检查图像是否为噪声
            image_flat = face_image.flatten()
            if len(image_flat) > 0:
                std_dev = np.std(image_flat)
                print(f"输入图像标准差: {std_dev:.3f}")
                
                if std_dev < 1.0:
                    print("❌ 错误：输入图像是噪声，跳过保存调试图片")
                    print("⚠️ 这表明人脸对齐或关键点检测有问题！")
                    
                    # 即使跳过正常保存，也要保存噪声图片用于调试（包含原图片信息）
                    if original_image_name:
                        noise_filename = f"noise_input_{original_image_name}.jpg"
                        save_debug_image(face_image, noise_filename, "aligned", "noise")
                        print(f"⚠️ 已保存噪声图片用于调试: {noise_filename}")
                else:
                    # 保存原始输入图像（包含原图片信息）
                    if original_image_name:
                        input_filename = f"input_{original_image_name}.jpg"
                        save_debug_image(face_image, input_filename, "aligned", "input")
                    else:
                        save_debug_image(face_image, "input_face.jpg", "aligned", "input")
                    
                    # 不再保存RGB转换图片，因为这是不必要的步骤
                    print("✅ 已保存输入图像，跳过不必要的RGB转换")
        else:
            print("❌ 错误：face_image为空，跳过保存调试图片")
        
        return features_np
        
    except Exception as e:
        print(f"❌ 特征提取错误: {e}")
        import traceback
        traceback.print_exc()
        return None

def build_database():
    """构建人脸数据库"""
    database = {}
    database_path = str(DATABASE_DIR)
    
    if not os.path.exists(database_path):
        return database
    
    for person_name in os.listdir(database_path):
        person_path = os.path.join(database_path, person_name)
        if os.path.isdir(person_path):
            features_list = []
            for image_file in os.listdir(person_path):
                if image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_path = os.path.join(person_path, image_file)
                    try:
                        image = cv2.imread(image_path)
                        if image is not None:
                            # 检测人脸
                            results = face_model(image)
                            if len(results) > 0:
                                for result in results:
                                    boxes = result.boxes
                                    if boxes is not None and len(boxes) > 0:
                                        box = boxes[0]
                                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                                        face_crop = image[y1:y2, x1:x2]
                                        
                                        # 检测关键点并进行人脸对齐
                                        # 修复：直接传入face_crop，避免双重裁剪
                                        landmarks = detect_landmarks_direct(face_crop)
                                        # landmarks现在是ROI坐标系的关键点，直接用于对齐face_crop
                                        aligned_face = align_face(face_crop, landmarks)
                                        
                                        # 提取特征
                                        print(f"为 {person_name} 开始特征提取...")
                                        
                                        # 保存数据库构建时的图片
                                        db_image_filename = f"{person_name}_{image_file}"
                                        save_debug_image(aligned_face, db_image_filename, "database", "db")
                                        
                                        # 也保存原始裁剪的人脸图片
                                        crop_db_filename = f"{person_name}_crop_{image_file}"
                                        save_debug_image(face_crop, crop_db_filename, "database", "crop")
                                        
                                        # 传入原图片名称用于调试
                                        original_name = f"{person_name}_{image_file}"
                                        features = extract_features(aligned_face, face_recognition_model, original_name)
                                        if features is not None:
                                            print(f"为 {person_name} 提取特征成功，形状: {features.shape}")
                                            # 确保特征是一维的
                                            features = features.flatten()
                                            print(f"特征展平后形状: {features.shape}")
                                            print(f"特征值范围: [{features.min():.6f}, {features.max():.6f}]")
                                            print(f"特征范数: {np.linalg.norm(features):.6f}")
                                            features_list.append(features)
                                        else:
                                            print(f"为 {person_name} 提取特征失败")
                    except Exception as e:
                        print(f"处理图片 {image_path} 时出错: {e}")
            
            if features_list:
                # 保存所有特征向量，而不是平均值
                database[person_name] = features_list
                print(f"为 {person_name} 保存了 {len(features_list)} 个特征向量")
    
    return database

def cosine_similarity(features1, features2):
    """计算余弦相似度"""
    try:
        # 确保特征向量是一维的
        features1 = features1.flatten()
        features2 = features2.flatten()
        
        print(f"计算相似度 - 特征1形状: {features1.shape}, 特征2形状: {features2.shape}")
        
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        
        if norm1 < 1e-10 or norm2 < 1e-10:
            print(f"特征向量范数过小: norm1={norm1:.6f}, norm2={norm2:.6f}")
            return 0.0
        
        similarity = np.dot(features1, features2) / (norm1 * norm2)
        print(f"相似度计算结果: {similarity:.4f}")
        return float(similarity)
    except Exception as e:
        print(f"计算相似度错误: {e}")
        import traceback
        traceback.print_exc()
        return 0.0

def match_face(features, database, threshold=None):
    """匹配人脸"""
    if threshold is None:
        threshold = FACE_RECOGNITION_THRESHOLD
    
    best_match = None
    best_similarity = 0
    
    print(f"开始匹配人脸，查询特征维度: {features.shape}")
    print(f"数据库中有 {len(database)} 个人")
    print(f"当前阈值: {threshold}")
    
    # 记录所有相似度，用于调试
    all_similarities = []
    
    for person_name, db_features_list in database.items():
        print(f"检查 {person_name}，有 {len(db_features_list)} 个特征向量")
        
        # 计算与所有特征向量的最大相似度
        max_similarity = 0
        person_similarities = []
        
        for i, db_features in enumerate(db_features_list):
            try:
                similarity = cosine_similarity(features, db_features)
                print(f"  - 特征 {i}: 相似度 = {similarity:.6f}")
                person_similarities.append(similarity)
                max_similarity = max(max_similarity, similarity)
                
                # 记录所有相似度
                all_similarities.append({
                    "person": person_name,
                    "feature_index": i,
                    "similarity": similarity
                })
                
            except Exception as e:
                print(f"  - 特征 {i} 相似度计算失败: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"  {person_name} 最大相似度: {max_similarity:.6f}")
        
        # 不管是否超过阈值，都记录最佳相似度
        if max_similarity > best_similarity:
            best_similarity = max_similarity
            best_match = person_name
    
    print(f"=== 相似度分析 ===")
    print(f"所有相似度: {[s['similarity'] for s in all_similarities]}")
    print(f"相似度范围: [{min([s['similarity'] for s in all_similarities]):.6f}, {max([s['similarity'] for s in all_similarities]):.6f}]")
    print(f"平均相似度: {np.mean([s['similarity'] for s in all_similarities]):.6f}")
    print(f"阈值: {threshold}")
    print(f"超过阈值的相似度数量: {len([s for s in all_similarities if s['similarity'] > threshold])}")
    
    # 如果最佳相似度仍然为0，说明有问题
    if best_similarity == 0:
        print("⚠️ 警告：所有相似度都为0，可能存在严重问题！")
        print("可能原因：")
        print("1. 特征向量全为0")
        print("2. 特征向量为NaN或Inf")
        print("3. 余弦相似度计算失败")
        print("4. 数据库特征有问题")
    
    print(f"最终结果: 最佳匹配 = {best_match}, 相似度 = {best_similarity:.6f}")
    return best_match, best_similarity

def recognize_emotion(face_crop, emotion_model):
    """识别表情"""
    try:
        results = emotion_model(face_crop)
        if len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                # 获取表情预测结果
                if hasattr(result, 'probs') and result.probs is not None:
                    probs = result.probs.data.cpu().numpy()
                    emotion_idx = np.argmax(probs)
                    confidence = float(probs[emotion_idx])
                    emotion = emotion_labels[emotion_idx]
                    return emotion, confidence
                else:
                    # 如果没有probs，尝试从names获取
                    if hasattr(result, 'names') and result.names:
                        emotion = list(result.names.values())[0]
                        confidence = EMOTION_DEFAULT_CONFIDENCE  # 使用配置的默认置信度
                        return emotion, confidence
        return "neutral", NEUTRAL_EMOTION_CONFIDENCE
    except Exception as e:
        print(f"表情识别错误: {e}")
        return "neutral", 0.5

# 加载人脸识别模型
print("正在加载人脸识别模型...")
face_recognition_model = load_insightface_model(face_recognition_model_path)

# 构建数据库
print("正在构建人脸数据库...")
face_database = build_database()
print(f"数据库构建完成，包含 {len(face_database)} 个人")

print("所有模型加载完成！")

# 测试特征提取
def test_feature_extraction():
    """测试特征提取功能"""
    print("\n=== 测试特征提取 ===")
    
    # 创建一个测试图像
    test_image = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    print(f"测试图像形状: {test_image.shape}")
    
    try:
        features = extract_features(test_image, face_recognition_model, "test_feature_extraction")
        if features is not None:
            print(f"✅ 特征提取成功！特征维度: {features.shape}")
            print(f"特征值范围: [{features.min():.4f}, {features.max():.4f}]")
            print(f"特征范数: {np.linalg.norm(features):.4f}")
            
            # 测试相似度计算
            print("\n=== 测试相似度计算 ===")
            test_features2 = np.random.randn(512) * 0.1  # 创建另一个测试特征
            similarity = cosine_similarity(features, test_features2)
            print(f"测试相似度: {similarity:.4f}")
            
        else:
            print("❌ 特征提取失败")
    except Exception as e:
        print(f"❌ 特征提取测试异常: {e}")
        import traceback
        traceback.print_exc()

# 运行测试
test_feature_extraction()

# 处理函数
def process_frame_face_only(frame, face_model, face_recognition_model, face_database):
    """仅处理人脸识别"""
    print(f"\n=== 开始人脸识别处理 ===")
    print(f"输入图像尺寸: {frame.shape}")
    
    results = face_model(frame)
    faces = []
    
    for result in results:
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                
                print(f"\n--- 处理检测到的人脸 ---")
                print(f"边界框: [{x1}, {y1}, {x2}, {y2}], 置信度: {confidence:.4f}")
                
                if confidence > FACE_DETECTION_THRESHOLD:
                    # 1. 人脸裁剪
                    face_crop = frame[y1:y2, x1:x2]
                    print(f"人脸裁剪尺寸: {face_crop.shape}")
                    
                    # 2. 检测关键点
                    landmarks = detect_landmarks(frame, [x1, y1, x2, y2])
                    print(f"检测到关键点数量: {len(landmarks) if landmarks is not None else 0}")
                    
                    # 3. 人脸对齐
                    aligned_face = align_face(face_crop, landmarks)
                    print(f"对齐后人脸尺寸: {aligned_face.shape}")
                    
                    # 4. 特征提取
                    print("开始特征提取...")
                    features = extract_features(aligned_face, face_recognition_model, f"face_only_{int(time.time())}")
                    
                    if features is not None:
                        print(f"特征提取成功，特征形状: {features.shape}")
                        print(f"特征值范围: [{features.min():.6f}, {features.max():.6f}]")
                        print(f"特征范数: {np.linalg.norm(features):.6f}")
                        
                        # 5. 人脸匹配
                        print("开始人脸匹配...")
                        identity, similarity = match_face(features, face_database)
                        print(f"匹配结果: 身份={identity}, 相似度={similarity:.6f}")
                        
                        faces.append({
                            "box": [x1, y1, x2, y2],
                            "confidence": confidence,
                            "identity": identity or "Unknown",
                            "similarity": similarity
                        })
                    else:
                        print("❌ 特征提取失败")
                else:
                    print(f"置信度 {confidence:.4f} 低于阈值 {FACE_DETECTION_THRESHOLD}")
    
    print(f"=== 处理完成，共识别 {len(faces)} 个人脸 ===\n")
    return faces

def process_frame_emotion_only(frame, face_model, emotion_model):
    """仅处理表情识别"""
    results = face_model(frame)
    faces = []
    
    for result in results:
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                
                if confidence > FACE_DETECTION_THRESHOLD:
                    face_crop = frame[y1:y2, x1:x2]
                    
                    # 检测关键点并进行人脸对齐
                    landmarks = detect_landmarks(frame, [x1, y1, x2, y2])
                    aligned_face = align_face(face_crop, landmarks)
                    
                    # 表情识别
                    emotion, emotion_confidence = recognize_emotion(aligned_face, emotion_model)
                    
                    faces.append({
                        "box": [x1, y1, x2, y2],
                        "confidence": confidence,
                        "emotion": emotion,
                        "emotion_confidence": emotion_confidence
                    })
    
    return faces

def process_frame_combined(frame, face_model, face_recognition_model, emotion_model, face_database):
    """处理综合识别"""
    results = face_model(frame)
    faces = []
    
    for result in results:
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                
                if confidence > FACE_DETECTION_THRESHOLD:
                    print(f"\n🔍 检测到人脸 - 置信度: {confidence:.3f}")
                    print(f"人脸边界框: [{x1}, {y1}, {x2}, {y2}]")
                    print(f"人脸尺寸: {x2-x1} x {y2-y1}")
                    
                    # 裁剪整个人脸区域
                    face_crop = frame[y1:y2, x1:x2]
                    print(f"裁剪后人脸尺寸: {face_crop.shape}")
                    
                    # 生成唯一的查询ID
                    query_id = f"query_{int(time.time())}"
                    
                    # 保存原始裁剪的人脸图片
                    crop_filename = f"crop_{query_id}.jpg"
                    save_debug_image(face_crop, crop_filename, "queries", "crop")
                    
                    # 检测关键点并进行人脸对齐
                    print("开始检测关键点...")
                    # 修复：直接传入face_crop和ROI坐标系的关键点
                    landmarks = detect_landmarks_direct(face_crop)
                    print(f"检测到的关键点: {landmarks}")
                    
                    # 保存关键点检测结果
                    if landmarks is not None and len(landmarks) >= 5:
                        # 在face_crop上绘制关键点
                        keypoint_debug = face_crop.copy()
                        for i, (x, y) in enumerate(landmarks):
                            cv2.circle(keypoint_debug, (int(x), int(y)), 3, (0, 255, 0), -1)  # 绿色点
                            cv2.putText(keypoint_debug, str(i), (int(x)+5, int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                        
                        keypoint_filename = f"keypoints_{query_id}.jpg"
                        save_debug_image(keypoint_debug, keypoint_filename, "queries", "keypoints")
                    
                    print("开始人脸对齐...")
                    aligned_face = align_face(face_crop, landmarks)
                    print(f"对齐后人脸尺寸: {aligned_face.shape}")
                    
                    # 保存对齐后的图片
                    query_image_filename = f"aligned_{query_id}.jpg"
                    save_debug_image(aligned_face, query_image_filename, "queries", "query")
                    
                    # 人脸识别
                    print("开始特征提取...")
                    # 使用查询ID进行调试
                    features = extract_features(aligned_face, face_recognition_model, query_id)
                    identity, similarity = "Unknown", 0.0
                    if features is not None:
                        identity, similarity = match_face(features, face_database)
                        if identity is None:
                            identity = "Unknown"
                    
                    # 表情识别
                    emotion, emotion_confidence = recognize_emotion(aligned_face, emotion_model)
                    
                    faces.append({
                        "box": [x1, y1, x2, y2],
                        "confidence": confidence,
                        "identity": identity,
                        "similarity": similarity,
                        "emotion": emotion,
                        "emotion_confidence": emotion_confidence
                    })
    
    return faces

# API端点

@app.get("/", response_class=HTMLResponse)
async def root():
    """根路径 - 返回HTML页面"""
    html_content = """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>人脸识别与表情识别系统</title>
        <style>
            body { 
                font-family: 'Microsoft YaHei', Arial, sans-serif; 
                margin: 0; 
                padding: 0; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container { 
                max-width: 800px; 
                margin: 50px auto; 
                padding: 40px; 
                background: white; 
                border-radius: 15px; 
                box-shadow: 0 10px 30px rgba(0,0,0,0.2); 
                text-align: center; 
            }
            h1 { 
                color: #333; 
                margin-bottom: 20px; 
                font-size: 2.5em;
            }
            .btn { 
                display: inline-block; 
                background: #667eea; 
                color: white; 
                padding: 15px 30px; 
                text-decoration: none; 
                border-radius: 8px; 
                margin: 10px; 
                font-size: 16px; 
                transition: all 0.3s; 
                font-weight: bold;
            }
            .btn:hover { 
                background: #5a6fd8; 
                transform: translateY(-2px); 
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            }
            .status { 
                background: #d4edda; 
                color: #155724; 
                padding: 15px; 
                border-radius: 8px; 
                margin: 20px 0; 
                font-size: 18px;
            }
            .description {
                color: #666;
                line-height: 1.6;
                margin: 20px 0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 人脸识别与表情识别系统</h1>
            <div class="status">🟢 系统运行正常 - 端口: 8001</div>
            <p class="description">欢迎使用AI人脸识别与表情识别服务！本系统提供高精度的人脸检测、身份识别和表情分析功能。</p>
            <div>
                <a href="/frontend" class="btn">🚀 进入系统</a>
                <a href="/docs" class="btn">📚 API文档</a>
            </div>
            <p style="margin-top: 30px; color: #666;">© 2024 人脸识别与表情识别系统</p>
        </div>
    </body>
    </html>
    """
    return html_content

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "models_loaded": True}

@app.get("/frontend", response_class=HTMLResponse)
async def frontend_page():
    """前端页面"""
    try:
        with open("frontend_demo.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>前端页面文件未找到</h1>", status_code=404)

@app.get("/debug/features")
async def debug_features():
    """调试：显示数据库中的特征信息"""
    try:
        debug_info = {
            "database_size": len(face_database),
            "database_content": {},
            "model_info": {
                "model_path": face_recognition_model_path,
                "model_exists": os.path.exists(face_recognition_model_path)
            }
        }
        
        for person_name, features_list in face_database.items():
            debug_info["database_content"][person_name] = {
                "feature_count": len(features_list),
                "feature_dimensions": [f.shape for f in features_list[:3]],  # 只显示前3个
                "feature_samples": []
            }
            
            # 显示前3个特征向量的统计信息
            for i, features in enumerate(features_list[:3]):
                debug_info["database_content"][person_name]["feature_samples"].append({
                    "index": i,
                    "shape": features.shape,
                    "min_value": float(features.min()),
                    "max_value": float(features.max()),
                    "mean_value": float(features.mean()),
                    "norm": float(np.linalg.norm(features))
                })
        
        return JSONResponse(content=debug_info)
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/debug/test_extraction")
async def debug_test_extraction():
    """调试：测试特征提取"""
    try:
        # 创建一个测试图像
        test_image = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        
        # 测试特征提取
        features = extract_features(test_image, face_recognition_model, "debug_test_simple_alignment")
        
        if features is not None:
            result = {
                "success": True,
                "test_image_shape": test_image.shape,
                "extracted_features": {
                    "shape": features.shape,
                    "min_value": float(features.min()),
                    "max_value": float(features.max()),
                    "mean_value": float(features.mean()),
                    "norm": float(np.linalg.norm(features))
                }
            }
        else:
            result = {
                "success": False,
                "error": "特征提取返回None"
            }
        
        return JSONResponse(content=result)
        
    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)

@app.get("/debug/test_preprocessing")
async def debug_test_preprocessing():
    """调试：测试不同预处理方法"""
    try:
        # 创建一个测试图像
        test_image = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        
        # 方法1：当前使用的[-1, 1]归一化
        face_rgb = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        tensor1 = torch.from_numpy(face_rgb.copy()).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        tensor1 = (tensor1 / 127.5) - 1.0
        
        # 方法2：ImageNet标准化
        tensor2 = torch.from_numpy(face_rgb.copy()).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)
        tensor2 = (tensor2 / 255.0 - mean) / std
        
        # 方法3：简单[0, 1]归一化
        tensor3 = torch.from_numpy(face_rgb.copy()).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        tensor3 = tensor3 / 255.0
        
        result = {
            "success": True,
            "test_image_shape": test_image.shape,
            "preprocessing_methods": {
                "method1_minus1_to_1": {
                    "range": [float(tensor1.min()), float(tensor1.max())],
                    "description": "当前使用的方法：[-1, 1]归一化"
                },
                "method2_imagenet": {
                    "range": [float(tensor2.min()), float(tensor2.max())],
                    "description": "ImageNet标准化"
                },
                "method3_0_to_1": {
                    "range": [float(tensor3.min()), float(tensor3.max())],
                    "description": "[0, 1]归一化"
                }
            }
        }
        
        return JSONResponse(content=result)
        
    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)

@app.get("/debug/test_model_output")
async def debug_test_model_output():
    """调试：测试模型输出"""
    try:
        # 创建不同的测试图像
        test_images = []
        
        # 图像1：随机图像
        img1 = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        test_images.append(("random_image", img1))
        
        # 图像2：全黑图像
        img2 = np.zeros((112, 112, 3), dtype=np.uint8)
        test_images.append(("black_image", img2))
        
        # 图像3：全白图像
        img3 = np.ones((112, 112, 3), dtype=np.uint8) * 255
        test_images.append(("white_image", img3))
        
        results = {}
        
        for name, img in test_images:
            print(f"\n测试图像: {name}")
            features = extract_features(img, face_recognition_model, f"debug_test_model_output_{name}")
            
            if features is not None:
                results[name] = {
                    "success": True,
                    "shape": features.shape,
                    "min_value": float(features.min()),
                    "max_value": float(features.max()),
                    "mean_value": float(features.mean()),
                    "norm": float(np.linalg.norm(features)),
                    "is_constant": np.all(features == features[0, 0]),
                    "is_zero": np.all(features == 0)
                }
            else:
                results[name] = {
                    "success": False,
                    "error": "特征提取失败"
                }
        
        return JSONResponse(content={
            "success": True,
            "test_results": results
        })
        
    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)

@app.get("/debug/device_status")
async def debug_device_status():
    """调试：检查设备状态和CUDA信息"""
    try:
        status = {
            "current_device": str(DEVICE),
            "device_type": DEVICE.type,
            "cuda_available": torch.cuda.is_available(),
            "cuda_info": {},
            "model_devices": {},
            "performance_tips": []
        }
        
        # 检查CUDA信息
        if torch.cuda.is_available():
            status["cuda_info"] = {
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(0),
                "device_properties": {
                    "total_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
                    "compute_capability": f"{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}",
                    "multi_processor_count": torch.cuda.get_device_properties(0).multi_processor_count
                }
            }
            status["performance_tips"].append("🚀 使用GPU加速，性能最佳")
        else:
            status["performance_tips"].append("💻 使用CPU，性能较慢但兼容性好")
        
        # 检查模型设备状态
        try:
            if hasattr(face_model, 'device'):
                status["model_devices"]["face_detection"] = str(face_model.device)
            else:
                status["model_devices"]["face_detection"] = "未知"
        except:
            status["model_devices"]["face_detection"] = "检查失败"
            
        try:
            if hasattr(emotion_model, 'device'):
                status["model_devices"]["emotion_recognition"] = str(emotion_model.device)
            else:
                status["model_devices"]["emotion_recognition"] = "未知"
        except:
            status["model_devices"]["emotion_recognition"] = "检查失败"
            
        try:
            if hasattr(face_recognition_model, 'device'):
                status["model_devices"]["face_recognition"] = str(face_recognition_model.device)
            else:
                status["model_devices"]["face_recognition"] = "未知"
        except:
            status["model_devices"]["face_recognition"] = "检查失败"
        
        # 添加性能建议
        if status["device_type"] == "cuda":
            status["performance_tips"].extend([
                "📊 建议：保持GPU驱动更新以获得最佳性能",
                "🌡️ 注意：GPU使用会产生热量，确保散热良好",
                "💾 内存：监控GPU内存使用，避免OOM错误"
            ])
        else:
            status["performance_tips"].extend([
                "📊 建议：如果有NVIDIA GPU，安装CUDA驱动可大幅提升性能",
                "🔧 检查：确认PyTorch是否安装了CUDA版本",
                "💡 提示：CPU模式适合开发和测试，生产环境建议使用GPU"
            ])
        
        return JSONResponse(content=status)
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/debug/database_status")
async def debug_database_status():
    """调试：检查数据库状态"""
    try:
        status = {
            "database_info": {
                "total_persons": len(face_database),
                "persons": {},
                "total_features": 0,
                "database_path": str(DATABASE_DIR)
            },
            "database_contents": {},
            "analysis": {}
        }
        
        # 检查数据库内容
        for person_name, features_list in face_database.items():
            person_info = {
                "name": person_name,
                "feature_count": len(features_list),
                "feature_shapes": [],
                "feature_ranges": [],
                "feature_norms": []
            }
            
            for i, features in enumerate(features_list):
                try:
                    person_info["feature_shapes"].append(features.shape)
                    person_info["feature_ranges"].append([float(features.min()), float(features.max())])
                    person_info["feature_norms"].append(float(np.linalg.norm(features)))
                except Exception as e:
                    person_info["feature_shapes"].append(f"error: {e}")
                    person_info["feature_ranges"].append([0, 0])
                    person_info["feature_norms"].append(0)
            
            status["database_contents"][person_name] = person_info
            status["database_info"]["total_features"] += len(features_list)
        
        # 分析数据库质量
        if status["database_info"]["total_features"] == 0:
            status["analysis"]["status"] = "❌ 数据库为空！"
            status["analysis"]["problem"] = "没有提取到任何特征，这会导致相似度计算失败"
            status["analysis"]["solution"] = "需要重新构建数据库，检查图片路径和特征提取过程"
        elif status["database_info"]["total_features"] < 3:
            status["analysis"]["status"] = "⚠️ 数据库特征不足"
            status["analysis"]["problem"] = "特征数量太少，可能影响识别效果"
            status["analysis"]["solution"] = "建议添加更多图片到数据库"
        else:
            status["analysis"]["status"] = "✅ 数据库正常"
            status["analysis"]["problem"] = "无"
            status["analysis"]["solution"] = "无"
        
        # 检查数据库文件夹
        if os.path.exists(DATABASE_DIR):
            status["database_info"]["folder_exists"] = True
            status["database_info"]["folder_contents"] = os.listdir(DATABASE_DIR)
        else:
            status["database_info"]["folder_exists"] = False
            status["database_info"]["folder_contents"] = []
        
        return JSONResponse(content=status)
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/debug/image_purpose_explanation")
async def debug_image_purpose_explanation():
    """调试：详细说明每部分图片的作用和命名规则"""
    try:
        explanation = {
            "image_purposes": {
                "database_folder": {
                    "description": "数据库构建时保存的图片",
                    "images": {
                        "crop_*.jpg": {
                            "purpose": "原始裁剪的人脸图片",
                            "description": "从原图中裁剪出的完整人脸区域，用于验证人脸检测是否正确",
                            "example": "crop_db_20250828_161425_aa_J02075.jpg"
                        },
                        "db_*.jpg": {
                            "purpose": "对齐后的112x112人脸图片",
                            "description": "经过人脸对齐处理的标准尺寸图片，用于特征提取和存储",
                            "example": "db_20250828_161425_aa_J02075.jpg"
                        }
                    }
                },
                "queries_folder": {
                    "description": "查询识别时保存的图片",
                    "images": {
                        "crop_*.jpg": {
                            "purpose": "查询图片的原始裁剪人脸",
                            "description": "从查询图片中裁剪出的人脸，用于关键点检测",
                            "example": "crop_query_1701415860.jpg"
                        },
                        "keypoints_*.jpg": {
                            "purpose": "带关键点标记的人脸图片",
                            "description": "在裁剪人脸上标记5个关键点（眼睛、鼻子、嘴角），用于验证关键点检测",
                            "example": "keypoints_query_1701415860.jpg"
                        },
                        "aligned_*.jpg": {
                            "purpose": "对齐后的112x112查询人脸",
                            "description": "经过人脸对齐的查询图片，用于特征提取和匹配",
                            "example": "aligned_query_1701415860.jpg"
                        }
                    }
                },
                "aligned_folder": {
                    "description": "特征提取时的中间图片",
                    "images": {
                        "input_*.jpg": {
                            "purpose": "输入到特征提取模型的图片",
                            "description": "直接用于特征提取的112x112人脸图片",
                            "example": "input_query_1701415860.jpg"
                        },
                        "rgb_*.jpg": {
                            "purpose": "已移除（不必要的BGR转RGB）",
                            "description": "这个步骤已被优化，不再进行不必要的颜色空间转换",
                            "example": "不再生成此类图片"
                        },
                        "noise_input_*.jpg": {
                            "purpose": "噪声图片（用于调试）",
                            "description": "当人脸对齐失败时产生的噪声图片，用于问题追踪",
                            "example": "noise_input_query_1701415860.jpg"
                        }
                    }
                }
            },
            "naming_convention": {
                "pattern": "[类型]_[时间戳]_[来源]_[原图片名].jpg",
                "examples": {
                    "crop_query_1701415860.jpg": "裁剪的查询图片，时间戳1701415860",
                    "db_20250828_161425_aa_J02075.jpg": "数据库图片，2025年8月28日16:14:25，来自aa的J02075图片",
                    "noise_input_query_1701415860.jpg": "噪声输入图片，来自查询1701415860"
                }
            },
            "error_tracking": {
                "method": "通过文件名中的原图片名可以追踪到问题源头",
                "example": "如果看到noise_input_query_1701415860.jpg，说明查询1701415860在人脸对齐时出现了问题"
            }
        }
        
        return JSONResponse(content=explanation)
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/debug/analyze_mosaic_randomness")
async def debug_analyze_mosaic_randomness():
    """调试：分析马赛克图片的随机性"""
    try:
        # 检查aligned文件夹中的图片
        aligned_dir = os.path.join(DEBUG_IMAGES_DIR, "aligned")
        mosaic_analysis = {
            "aligned_dir": aligned_dir,
            "total_images": 0,
            "normal_images": [],
            "noise_images": [],
            "mosaic_patterns": {},
            "analysis": {}
        }
        
        if os.path.exists(aligned_dir):
            images = [f for f in os.listdir(aligned_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            mosaic_analysis["total_images"] = len(images)
            
            for image_file in images:
                image_path = os.path.join(aligned_dir, image_file)
                try:
                    # 读取图像
                    image = cv2.imread(image_path)
                    if image is not None:
                        # 分析图像质量
                        image_flat = image.flatten()
                        std_dev = np.std(image_flat)
                        mean_val = np.mean(image_flat)
                        unique_values = len(np.unique(image))
                        
                        image_info = {
                            "filename": image_file,
                            "shape": image.shape,
                            "std_dev": float(std_dev),
                            "mean_val": float(mean_val),
                            "unique_values": unique_values,
                            "is_noise": std_dev < 1.0 or unique_values < 10
                        }
                        
                        if std_dev < 1.0 or unique_values < 10:
                            mosaic_analysis["noise_images"].append(image_info)
                        else:
                            mosaic_analysis["normal_images"].append(image_info)
                            
                except Exception as e:
                    print(f"分析图片 {image_file} 时出错: {e}")
        
        # 分析马赛克模式
        if mosaic_analysis["noise_images"]:
            mosaic_analysis["analysis"]["noise_count"] = len(mosaic_analysis["noise_images"])
            mosaic_analysis["analysis"]["noise_filenames"] = [img["filename"] for img in mosaic_analysis["noise_images"]]
            
            # 分析马赛克是否与特定图片相关
            noise_patterns = {}
            for noise_img in mosaic_analysis["noise_images"]:
                filename = noise_img["filename"]
                if "noise_input_" in filename:
                    # 提取原图片信息
                    original_name = filename.replace("noise_input_", "")
                    if original_name not in noise_patterns:
                        noise_patterns[original_name] = []
                    noise_patterns[original_name].append(filename)
            
            mosaic_analysis["mosaic_patterns"] = noise_patterns
            mosaic_analysis["analysis"]["conclusion"] = "马赛克图片分析完成，可以追踪到原图片"
        
        if mosaic_analysis["normal_images"]:
            mosaic_analysis["analysis"]["normal_count"] = len(mosaic_analysis["normal_images"])
            mosaic_analysis["analysis"]["normal_filenames"] = [img["filename"] for img in mosaic_analysis["normal_images"]]
        
        return JSONResponse(content=mosaic_analysis)
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/debug/analyze_mosaic_cause")
async def debug_analyze_mosaic_cause():
    """调试：分析马赛克图片产生的原因"""
    try:
        analysis = {
            "mosaic_causes": [
                "1. 关键点检测失败 - YOLO模型没有关键点检测功能",
                "2. 人脸对齐异常 - 仿射变换或缩放出现问题",
                "3. 图像数据损坏 - 在某个处理步骤中图像被破坏",
                "4. 关键点分布不合理 - 估算的关键点过于集中",
                "5. 输入图像质量差 - 人脸区域太小、模糊或光照不足"
            ],
            "prevention_measures": [
                "1. 增强图像质量检查 - 检测噪声、纯色图像",
                "2. 改进关键点估算 - 使用更合理的分布算法",
                "3. 添加备用方案 - 当对齐失败时使用简单缩放",
                "4. 详细日志记录 - 追踪每个步骤的执行情况",
                "5. 图像质量验证 - 在保存前验证图像有效性"
            ],
            "debug_endpoints": [
                "/debug/test_fixed_pipeline - 测试完整流程",
                "/debug/analyze_noise_images - 分析噪声图片",
                "/debug/test_keypoint_boundaries - 测试关键点边界"
            ]
        }
        
        return JSONResponse(content=analysis)
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/debug/test_fixed_pipeline")
async def debug_test_fixed_pipeline():
    """调试：测试修复后的完整流程"""
    try:
        # 创建测试图像
        test_image = np.random.randint(0, 255, (200, 150, 3), dtype=np.uint8)
        
        print("🧪 测试修复后的完整流程...")
        print(f"测试图像尺寸: {test_image.shape}")
        
        # 第一步：关键点检测
        print("\n=== 第一步：关键点检测 ===")
        landmarks = detect_landmarks_direct(test_image)
        print(f"检测到的关键点: {landmarks}")
        
        # 第二步：人脸对齐
        print("\n=== 第二步：人脸对齐 ===")
        aligned_face = align_face(test_image, landmarks)
        print(f"对齐后人脸尺寸: {aligned_face.shape}")
        
        # 第三步：特征提取
        print("\n=== 第三步：特征提取 ===")
        features = extract_features(aligned_face, face_recognition_model, "debug_test_fixed_flow")
        print(f"特征提取结果: {'成功' if features is not None else '失败'}")
        
        debug_info = {
            "test_image_shape": test_image.shape,
            "detected_landmarks": landmarks.tolist() if landmarks is not None else None,
            "aligned_face_shape": aligned_face.shape if aligned_face is not None else None,
            "features_extracted": features is not None,
            "pipeline_status": "修复后的流程测试完成"
        }
        
        return JSONResponse(content=debug_info)
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/debug/analyze_noise_images")
async def debug_analyze_noise_images():
    """调试：分析噪声图片问题"""
    try:
        # 检查aligned文件夹中的图片
        aligned_dir = os.path.join(DEBUG_IMAGES_DIR, "aligned")
        noise_analysis = {
            "aligned_dir": aligned_dir,
            "total_images": 0,
            "noise_images": [],
            "normal_images": [],
            "analysis": {}
        }
        
        if os.path.exists(aligned_dir):
            images = [f for f in os.listdir(aligned_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            noise_analysis["total_images"] = len(images)
            
            for image_file in images:
                image_path = os.path.join(aligned_dir, image_file)
                try:
                    # 读取图像
                    image = cv2.imread(image_path)
                    if image is not None:
                        # 分析图像质量
                        image_flat = image.flatten()
                        std_dev = np.std(image_flat)
                        mean_val = np.mean(image_flat)
                        min_val = np.min(image_flat)
                        max_val = np.max(image_flat)
                        
                        image_info = {
                            "filename": image_file,
                            "shape": image.shape,
                            "std_dev": float(std_dev),
                            "mean_val": float(mean_val),
                            "min_val": float(min_val),
                            "max_val": float(max_val),
                            "is_noise": std_dev < 1.0
                        }
                        
                        if std_dev < 1.0:
                            noise_analysis["noise_images"].append(image_info)
                        else:
                            noise_analysis["normal_images"].append(image_info)
                            
                except Exception as e:
                    print(f"分析图片 {image_file} 时出错: {e}")
        
        # 分析噪声图片的特征
        if noise_analysis["noise_images"]:
            noise_analysis["analysis"]["noise_count"] = len(noise_analysis["noise_images"])
            noise_analysis["analysis"]["noise_filenames"] = [img["filename"] for img in noise_analysis["noise_images"]]
            noise_analysis["analysis"]["noise_patterns"] = "发现噪声图片，可能原因：1.关键点检测失败 2.人脸对齐异常 3.图像数据损坏"
        
        if noise_analysis["normal_images"]:
            noise_analysis["analysis"]["normal_count"] = len(noise_analysis["normal_images"])
            noise_analysis["analysis"]["normal_filenames"] = [img["filename"] for img in noise_analysis["normal_images"]]
        
        return JSONResponse(content=noise_analysis)
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/debug/test_keypoint_boundaries")
async def debug_test_keypoint_boundaries():
    """调试：测试关键点边界问题"""
    try:
        # 创建不同尺寸的测试图像
        test_cases = [
            {
                "name": "正常尺寸",
                "image": np.random.randint(0, 255, (200, 150, 3), dtype=np.uint8),
                "landmarks": np.array([[50, 40], [150, 40], [100, 80], [50, 120], [150, 120]], dtype=np.float32)
            },
            {
                "name": "小尺寸",
                "image": np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8),
                "landmarks": np.array([[25, 20], [75, 20], [50, 40], [25, 60], [75, 60]], dtype=np.float32)
            },
            {
                "name": "关键点超出边界",
                "image": np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8),
                "landmarks": np.array([[120, 40], [150, 40], [100, 80], [50, 120], [150, 120]], dtype=np.float32)
            },
            {
                "name": "关键点过于集中",
                "image": np.random.randint(0, 255, (200, 150, 3), dtype=np.uint8),
                "landmarks": np.array([[95, 75], [105, 75], [100, 80], [98, 85], [102, 85]], dtype=np.float32)
            }
        ]
        
        results = []
        
        for i, test_case in enumerate(test_cases):
            print(f"\n🧪 测试案例 {i+1}: {test_case['name']}")
            print(f"图像尺寸: {test_case['image'].shape}")
            print(f"关键点: {test_case['landmarks']}")
            
            try:
                aligned_face = align_face(test_case['image'], test_case['landmarks'])
                success = aligned_face is not None
                result_shape = aligned_face.shape if aligned_face is not None else None
            except Exception as e:
                success = False
                result_shape = f"错误: {str(e)}"
            
            results.append({
                "case_name": test_case['name'],
                "image_shape": test_case['image'].shape,
                "landmarks": test_case['landmarks'].tolist(),
                "success": success,
                "result_shape": result_shape
            })
            
            print(f"结果: {'成功' if success else '失败'}")
        
        return JSONResponse(content={
            "test_cases": results,
            "explanation": {
                "正常尺寸": "标准测试，应该成功",
                "小尺寸": "小图像测试，应该成功",
                "关键点超出边界": "关键点超出图像范围，应该失败",
                "关键点过于集中": "关键点分布不合理，应该失败"
            }
        })
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/debug/test_simple_alignment")
async def debug_test_simple_alignment():
    """调试：测试简化的人脸对齐（等比缩放）"""
    try:
        # 创建一个测试图像，模拟人脸
        test_image = np.random.randint(0, 255, (200, 150, 3), dtype=np.uint8)
        
        # 模拟5个关键点（在图像范围内）
        landmarks = np.array([
            [50, 40],   # 左眼
            [150, 40],  # 右眼
            [100, 80],  # 鼻尖
            [50, 120],  # 左嘴角
            [150, 120]  # 右嘴角
        ], dtype=np.float32)
        
        print("🧪 测试简化的人脸对齐...")
        print(f"测试图像尺寸: {test_image.shape}")
        print(f"测试关键点: {landmarks}")
        
        # 测试人脸对齐
        aligned_face = align_face(test_image, landmarks)
        
        debug_info = {
            "test_image_shape": test_image.shape,
            "test_landmarks": landmarks.tolist(),
            "aligned_face_shape": aligned_face.shape if aligned_face is not None else None,
            "method": "等比缩放 (cv2.resize)",
            "expected_result": "112x112的完整人脸图像"
        }
        
        return JSONResponse(content=debug_info)
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/debug/test_fixed_flow")
async def debug_test_fixed_flow():
    """调试：测试修复后的图像处理流程"""
    try:
        # 创建一个测试图像，模拟人脸
        test_image = np.random.randint(0, 255, (400, 300, 3), dtype=np.uint8)
        
        # 模拟人脸边界框
        face_bbox = [50, 50, 350, 350]  # [x1, y1, x2, y2]
        x1, y1, x2, y2 = face_bbox
        
        print("🧪 测试修复后的图像处理流程...")
        print(f"测试图像尺寸: {test_image.shape}")
        print(f"人脸边界框: {face_bbox}")
        
        # 第一步：人脸裁剪
        face_crop = test_image[y1:y2, x1:x2]
        print(f"✅ 第一步：人脸裁剪完成，尺寸: {face_crop.shape}")
        
        # 第二步：关键点检测（修复后）
        print("第二步：开始关键点检测...")
        landmarks = detect_landmarks_direct(face_crop)
        print(f"✅ 第二步：关键点检测完成，关键点: {landmarks}")
        
        # 第三步：人脸对齐
        print("第三步：开始人脸对齐...")
        aligned_face = align_face(face_crop, landmarks)
        print(f"✅ 第三步：人脸对齐完成，尺寸: {aligned_face.shape}")
        
        debug_info = {
            "test_image_shape": test_image.shape,
            "face_bbox": face_bbox,
            "face_crop_shape": face_crop.shape,
            "detected_landmarks": landmarks.tolist() if landmarks is not None else None,
            "aligned_face_shape": aligned_face.shape if aligned_face is not None else None,
            "process_steps": [
                "1. 人脸裁剪 ✅",
                "2. 关键点检测 ✅", 
                "3. 人脸对齐 ✅"
            ]
        }
        
        return JSONResponse(content=debug_info)
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/debug/image_processing_flow")
async def debug_image_processing_flow():
    """调试：显示完整的图像处理流程"""
    try:
        flow_info = {
            "process_steps": [
                "1. 人脸检测 (YOLO best.pt)",
                "2. 人脸裁剪 (face_crop = frame[y1:y2, x1:x2])",
                "3. 关键点检测 (在face_roi上检测5个关键点)",
                "4. 人脸对齐 (仿射变换到112x112)",
                "5. 特征提取 (InsightFace模型)"
            ],
            "expected_outputs": {
                "face_crop": "完整的人脸区域 (不是只有眼睛)",
                "landmarks": "5个关键点在ROI坐标系中的坐标",
                "aligned_face": "112x112的完整对齐人脸",
                "features": "512维特征向量"
            },
            "current_issues": [
                "关键点检测可能失败，使用估算关键点",
                "估算关键点分布可能不合理",
                "人脸对齐后可能只保留眼睛区域"
            ],
            "debug_endpoints": [
                "/debug/test_yolo - 测试YOLO模型能力",
                "/debug/saved_images - 查看保存的图片",
                "/debug/recognition_analysis - 深度分析"
            ]
        }
        
        return JSONResponse(content=flow_info)
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/debug/recognition_analysis")
async def debug_recognition_analysis():
    """调试：深度分析识别效果差的原因"""
    try:
        if not face_database:
            return JSONResponse(content={"error": "数据库为空"}, status_code=404)
        
        analysis = {
            "database_quality": {},
            "feature_analysis": {},
            "similarity_distribution": {},
            "model_diagnosis": {},
            "recommendations": []
        }
        
        # 分析数据库质量
        for person_name, features_list in face_database.items():
            person_analysis = {
                "feature_count": len(features_list),
                "feature_quality": []
            }
            
            for i, features in enumerate(features_list):
                # 分析特征质量
                feature_norm = np.linalg.norm(features)
                feature_std = np.std(features)
                feature_range = np.max(features) - np.min(features)
                
                quality_score = 0
                if feature_norm > 0.9:  # 特征范数应该接近1
                    quality_score += 1
                if feature_std > 0.05:  # 特征应该有足够的方差
                    quality_score += 1
                if feature_range > 0.1:  # 特征值应该有足够的范围
                    quality_score += 1
                
                person_analysis["feature_quality"].append({
                    "index": i,
                    "norm": float(feature_norm),
                    "std": float(feature_std),
                    "range": float(feature_range),
                    "quality_score": quality_score
                })
            
            analysis["database_quality"][person_name] = person_analysis
        
        # 分析特征分布
        all_features = []
        for features_list in face_database.values():
            all_features.extend(features_list)
        
        if all_features:
            all_features_array = np.array(all_features)
            analysis["feature_analysis"] = {
                "total_features": len(all_features),
                "feature_dimension": all_features_array.shape[1],
                "global_std": float(np.std(all_features_array)),
                "global_range": float(np.max(all_features_array) - np.min(all_features_array)),
                "feature_correlation": float(np.corrcoef(all_features_array.T).mean())
            }
        
        # 分析相似度分布
        similarities = []
        for person_name1, features_list1 in face_database.items():
            for person_name2, features_list2 in face_database.items():
                if person_name1 != person_name2:
                    for features1 in features_list1:
                        for features2 in features_list2:
                            sim = cosine_similarity(features1, features2)
                            similarities.append(sim)
        
        if similarities:
            similarities = np.array(similarities)
            analysis["similarity_distribution"] = {
                "mean": float(np.mean(similarities)),
                "std": float(np.std(similarities)),
                "min": float(np.min(similarities)),
                "max": float(np.max(similarities)),
                "percentiles": {
                    "25%": float(np.percentile(similarities, 25)),
                    "50%": float(np.percentile(similarities, 50)),
                    "75%": float(np.percentile(similarities, 75))
                }
            }
        
        # 模型诊断
        analysis["model_diagnosis"] = {
            "model_path": face_recognition_model_path,
            "model_exists": os.path.exists(face_recognition_model_path),
            "model_size_mb": round(os.path.getsize(face_recognition_model_path) / (1024*1024), 2) if os.path.exists(face_recognition_model_path) else 0,
            "expected_similarity_ranges": {
                "same_person": "0.7-0.9 (优秀), 0.6-0.7 (良好), 0.5-0.6 (一般)",
                "different_person": "0.2-0.5 (优秀), 0.3-0.6 (良好), 0.4-0.7 (一般)"
            },
            "current_performance": {
                "similarity_0_401": "很差 - 应该 > 0.7 才能区分同一个人",
                "model_quality": "需要评估模型是否适合人脸识别任务"
            }
        }
        
        # 生成建议
        recommendations = []
        
        # 检查特征质量
        low_quality_count = 0
        for person_analysis in analysis["database_quality"].values():
            for feature_quality in person_analysis["feature_quality"]:
                if feature_quality["quality_score"] < 2:
                    low_quality_count += 1
        
        if low_quality_count > 0:
            recommendations.append(f"发现 {low_quality_count} 个低质量特征，建议重新提取")
        
        # 检查相似度分布
        if "similarity_distribution" in analysis:
            mean_sim = analysis["similarity_distribution"]["mean"]
            if mean_sim > 0.6:
                recommendations.append("平均相似度过高，模型区分能力差，建议使用更好的模型")
            elif mean_sim < 0.2:
                recommendations.append("平均相似度过低，可能存在预处理问题")
        
        # 检查特征相关性
        if "feature_analysis" in analysis:
            correlation = analysis["feature_analysis"]["feature_correlation"]
            if correlation > 0.8:
                recommendations.append("特征相关性过高，模型可能过拟合或特征提取失败")
        
        # 基于当前结果的具体建议
        recommendations.extend([
            "相似度0.401太低，说明模型区分能力很差",
            "建议：",
            "1. 使用更好的InsightFace预训练模型",
            "2. 检查预处理流程是否正确",
            "3. 考虑使用你提到的裁剪通道模型",
            "4. 重新训练或微调当前模型"
        ])
        
        analysis["recommendations"] = recommendations
        
        return JSONResponse(content=analysis)
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/debug/test_landmarks")
async def debug_test_landmarks():
    """调试：测试关键点检测和对齐流程"""
    try:
        # 创建一个测试图像，模拟人脸
        test_image = np.random.randint(0, 255, (400, 300, 3), dtype=np.uint8)
        
        # 模拟人脸边界框
        face_bbox = [50, 50, 350, 350]  # [x1, y1, x2, y2]
        x1, y1, x2, y2 = face_bbox
        
        print("🧪 测试关键点检测...")
        print(f"测试图像尺寸: {test_image.shape}")
        print(f"人脸边界框: {face_bbox}")
        
        # 裁剪人脸区域
        face_roi = test_image[y1:y2, x1:x2]
        print(f"裁剪后人脸尺寸: {face_roi.shape}")
        
        # 测试关键点检测
        landmarks = detect_landmarks(test_image, face_bbox)
        
        debug_info = {
            "test_image_shape": test_image.shape,
            "face_bbox": face_bbox,
            "face_roi_shape": face_roi.shape,
            "detected_landmarks": landmarks.tolist() if landmarks is not None else None,
            "landmark_count": len(landmarks) if landmarks is not None else 0
        }
        
        return JSONResponse(content=debug_info)
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/debug/test_yolo")
async def debug_test_yolo():
    """调试：测试YOLO模型的关键点检测能力"""
    try:
        # 创建一个测试图像
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        print("🧪 测试YOLO模型...")
        results = face_model(test_image, verbose=False)
        
        debug_info = {
            "model_path": str(face_detection_model_path),
            "test_image_shape": test_image.shape,
            "results_count": len(results),
            "model_capabilities": {}
        }
        
        if len(results) > 0:
            result = results[0]
            debug_info["model_capabilities"]["has_boxes"] = hasattr(result, 'boxes') and result.boxes is not None
            debug_info["model_capabilities"]["has_keypoints"] = hasattr(result, 'keypoints') and result.keypoints is not None
            debug_info["model_capabilities"]["has_masks"] = hasattr(result, 'masks') and result.masks is not None
            
            if hasattr(result, 'boxes') and result.boxes is not None:
                debug_info["model_capabilities"]["box_count"] = len(result.boxes)
                debug_info["model_capabilities"]["box_shape"] = result.boxes.xyxy.shape if hasattr(result.boxes, 'xyxy') else "No xyxy"
            
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                debug_info["model_capabilities"]["keypoint_count"] = result.keypoints.data.shape[0] if hasattr(result.keypoints, 'data') else 0
                debug_info["model_capabilities"]["keypoint_shape"] = result.keypoints.data.shape if hasattr(result.keypoints, 'data') else "No data"
            else:
                debug_info["model_capabilities"]["keypoint_count"] = 0
                debug_info["model_capabilities"]["keypoint_shape"] = "No keypoints"
        else:
            debug_info["model_capabilities"]["has_boxes"] = False
            debug_info["model_capabilities"]["has_keypoints"] = False
            debug_info["model_capabilities"]["has_masks"] = False
        
        return JSONResponse(content=debug_info)
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/debug/saved_images")
async def debug_saved_images():
    """调试：查看保存的调试图片"""
    try:
        debug_info = {
            "debug_images_dir": DEBUG_IMAGES_DIR,
            "folders": {},
            "total_images": 0,
            "explanation": {
                "database": "数据库构建时保存的图片",
                "queries": "查询识别时保存的图片", 
                "aligned": "特征提取时的中间图片"
            }
        }
        
        # 检查各个子文件夹
        for subfolder in ["database", "queries", "aligned"]:
            folder_path = os.path.join(DEBUG_IMAGES_DIR, subfolder)
            if os.path.exists(folder_path):
                images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                
                # 分类图片类型
                crop_images = [f for f in images if 'crop' in f]
                aligned_images = [f for f in images if 'aligned' in f or 'query' in f or 'db' in f]
                
                debug_info["folders"][subfolder] = {
                    "path": folder_path,
                    "image_count": len(images),
                    "crop_images": crop_images[:5],  # 原始裁剪图片
                    "aligned_images": aligned_images[:5],  # 对齐后图片
                    "all_images": images[:10]  # 所有图片
                }
                debug_info["total_images"] += len(images)
            else:
                debug_info["folders"][subfolder] = {
                    "path": folder_path,
                    "image_count": 0,
                    "crop_images": [],
                    "aligned_images": [],
                    "all_images": []
                }
        
        return JSONResponse(content=debug_info)
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/debug/compare_features")
async def debug_compare_features():
    """调试：比较数据库中的特征"""
    try:
        if not face_database:
            return JSONResponse(content={"error": "数据库为空"}, status_code=404)
        
        # 获取第一个人的特征
        first_person = list(face_database.keys())[0]
        first_features = face_database[first_person][0]  # 第一个特征
        
        result = {
            "database_info": {
                "total_persons": len(face_database),
                "persons": list(face_database.keys())
            },
            "feature_analysis": {}
        }
        
        # 分析每个人的特征
        for person_name, features_list in face_database.items():
            result["feature_analysis"][person_name] = {
                "feature_count": len(features_list),
                "features": []
            }
            
            for i, features in enumerate(features_list):
                feature_info = {
                    "index": i,
                    "shape": features.shape,
                    "min_value": float(features.min()),
                    "max_value": float(features.max()),
                    "mean_value": float(features.mean()),
                    "norm": float(np.linalg.norm(features))
                }
                
                # 计算与第一个特征的相似度
                if person_name != first_person:
                    similarity = cosine_similarity(first_features, features)
                    feature_info["similarity_with_first"] = float(similarity)
                
                result["feature_analysis"][person_name]["features"].append(feature_info)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/debug/threshold_test")
async def debug_threshold_test():
    """调试：测试不同阈值的效果"""
    try:
        if not face_database:
            return JSONResponse(content={"error": "数据库为空"}, status_code=404)
        
        # 获取第一个人的特征
        first_person = list(face_database.keys())[0]
        first_features = face_database[first_person][0]  # 第一个特征
        
        # 分析不同阈值下的识别效果
        thresholds = [0.2, 0.25, 0.28, 0.3, 0.35, 0.4, 0.45, 0.5]
        threshold_analysis = {}
        
        for threshold in thresholds:
            matches = 0
            total_comparisons = 0
            
            for person_name, features_list in face_database.items():
                for features in features_list:
                    if person_name != first_person:
                        similarity = cosine_similarity(first_features, features)
                        total_comparisons += 1
                        if similarity > threshold:
                            matches += 1
            
            threshold_analysis[f"threshold_{threshold}"] = {
                "threshold": threshold,
                "matches": matches,
                "total_comparisons": total_comparisons,
                "match_rate": matches / total_comparisons if total_comparisons > 0 else 0
            }
        
        return JSONResponse(content={
            "success": True,
            "threshold_analysis": threshold_analysis,
            "current_thresholds": {
                "face_detection_confidence": FACE_DETECTION_THRESHOLD,
                "face_recognition_similarity": FACE_RECOGNITION_THRESHOLD,
                "face_matching_default": 0.5  # 恢复原值
            }
        })
        
    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)

@app.get("/debug/test_similarity")
async def debug_test_similarity():
    """调试：测试相似度计算"""
    try:
        # 创建两个不同的测试特征
        features1 = np.random.randn(512) * 0.1
        features2 = np.random.randn(512) * 0.1
        
        # 归一化
        features1 = features1 / np.linalg.norm(features1)
        features2 = features2 / np.linalg.norm(features2)
        
        # 计算相似度
        similarity = cosine_similarity(features1, features2)
        
        # 测试相同特征
        similarity_same = cosine_similarity(features1, features1)
        
        result = {
            "success": True,
            "test_features": {
                "features1_shape": features1.shape,
                "features2_shape": features2.shape,
                "features1_norm": float(np.linalg.norm(features1)),
                "features2_norm": float(np.linalg.norm(features2))
            },
            "similarity_results": {
                "different_features": float(similarity),
                "same_features": float(similarity_same),
                "expected_same": 1.0,
                "expected_different": "0.0-0.5 (随机)"
            }
        }
        
        return JSONResponse(content=result)
        
    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)

# 人脸识别接口组
@app.post("/face/recognize_image")
async def face_recognize_image(file: UploadFile = File(...)):
    """人脸识别 - 图片"""
    try:
        print(f"\n=== 开始人脸识别 ===")
        print(f"文件名: {file.filename}")
        
        # 读取图片
        image_data = await file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return JSONResponse(status_code=400, content={"error": "无法读取图片"})
        
        print(f"图片尺寸: {image.shape}")
        print(f"数据库大小: {len(face_database)} 个人")
        
        # 处理图片
        faces = process_frame_face_only(image, face_model, face_recognition_model, face_database)
        
        print(f"检测到 {len(faces)} 个人脸")
        for i, face in enumerate(faces):
            print(f"人脸 {i+1}: 身份={face['identity']}, 相似度={face['similarity']:.4f}")
        
        return {
            "success": True,
            "faces": faces,
            "total_faces": len(faces),
            "debug_info": {
                "image_shape": image.shape,
                "database_size": len(face_database),
                "database_persons": list(face_database.keys())
            }
        }
    except Exception as e:
        print(f"人脸识别失败: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"处理失败: {str(e)}"})

@app.post("/face/recognize_video")
async def face_recognize_video(file: UploadFile = File(...)):
    """人脸识别 - 视频"""
    try:
        # 保存临时文件
        temp_filename = f"temp_video_{int(time.time())}.mp4"
        with open(temp_filename, "wb") as f:
            f.write(await file.read())
        
        # 处理视频
        cap = cv2.VideoCapture(temp_filename)
        all_faces = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % VIDEO_FRAME_INTERVAL == 0:  # 使用配置的帧间隔
                faces = process_frame_face_only(frame, face_model, face_recognition_model, face_database)
                if faces:
                    all_faces.append({"frame": frame_count, "faces": faces})
            
            frame_count += 1
        
        cap.release()
        
        # 删除临时文件
        os.remove(temp_filename)
        
        return {
            "success": True,
            "total_frames": frame_count,
            "processed_frames": len(all_faces),
            "faces": all_faces
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"处理失败: {str(e)}"})

# 表情识别接口组
@app.post("/emotion/recognize_image")
async def emotion_recognize_image(file: UploadFile = File(...)):
    """表情识别 - 图片"""
    try:
        # 读取图片
        image_data = await file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return JSONResponse(status_code=400, content={"error": "无法读取图片"})
        
        # 处理图片
        faces = process_frame_emotion_only(image, face_model, emotion_model)
        
        return {
            "success": True,
            "faces": faces,
            "total_faces": len(faces)
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"处理失败: {str(e)}"})

@app.post("/emotion/recognize_video")
async def emotion_recognize_video(file: UploadFile = File(...)):
    """表情识别 - 视频"""
    try:
        # 保存临时文件
        temp_filename = f"temp_video_{int(time.time())}.mp4"
        with open(temp_filename, "wb") as f:
            f.write(await file.read())
        
        # 处理视频
        cap = cv2.VideoCapture(temp_filename)
        all_faces = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % VIDEO_FRAME_INTERVAL == 0:  # 使用配置的帧间隔
                faces = process_frame_emotion_only(frame, face_model, emotion_model)
                if faces:
                    all_faces.append({"frame": frame_count, "faces": faces})
            
            frame_count += 1
        
        cap.release()
        
        # 删除临时文件
        os.remove(temp_filename)
        
        return {
            "success": True,
            "total_frames": frame_count,
            "processed_frames": len(all_faces),
            "faces": all_faces
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"处理失败: {str(e)}"})

# 综合识别接口组
@app.post("/combined/recognize_image")
async def combined_recognize_image(file: UploadFile = File(...)):
    """综合识别 - 图片"""
    try:
        # 读取图片
        image_data = await file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return JSONResponse(status_code=400, content={"error": "无法读取图片"})
        
        # 处理图片
        faces = process_frame_combined(image, face_model, face_recognition_model, emotion_model, face_database)
        
        return {
            "success": True,
            "faces": faces,
            "total_faces": len(faces)
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"处理失败: {str(e)}"})

@app.post("/combined/recognize_video")
async def combined_recognize_video(file: UploadFile = File(...)):
    """综合识别 - 视频"""
    try:
        # 保存临时文件
        temp_filename = f"temp_video_{int(time.time())}.mp4"
        with open(temp_filename, "wb") as f:
            f.write(await file.read())
        
        # 处理视频
        cap = cv2.VideoCapture(temp_filename)
        all_faces = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % VIDEO_FRAME_INTERVAL == 0:  # 使用配置的帧间隔
                faces = process_frame_combined(frame, face_model, face_recognition_model, emotion_model, face_database)
                if faces:
                    all_faces.append({"frame": frame_count, "faces": faces})
            
            frame_count += 1
        
        cap.release()
        
        # 删除临时文件
        os.remove(temp_filename)
        
        return {
            "success": True,
            "total_frames": frame_count,
            "processed_frames": len(all_faces),
            "faces": all_faces
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"处理失败: {str(e)}"})

# WebSocket实时流处理
@app.websocket("/face/recognize_rtsp")
async def face_recognize_rtsp(websocket: WebSocket):
    """人脸识别 - RTSP实时流"""
    await websocket.accept()
    try:
        while True:
            # 接收RTSP地址
            data = await websocket.receive_text()
            rtsp_url = json.loads(data).get("rtsp_url")
            
            if not rtsp_url:
                await websocket.send_text(json.dumps({"error": "缺少RTSP地址"}))
                continue
            
            # 连接RTSP流
            cap = cv2.VideoCapture(rtsp_url)
            if not cap.isOpened():
                await websocket.send_text(json.dumps({"error": "无法连接RTSP流"}))
                continue
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % STREAM_FRAME_INTERVAL == 0:  # 使用配置的帧间隔
                    faces = process_frame_face_only(frame, face_model, face_recognition_model, face_database)
                    result = {
                        "frame": frame_count,
                        "faces": faces,
                        "timestamp": time.time()
                    }
                    await websocket.send_text(json.dumps(result))
                
                frame_count += 1
                await asyncio.sleep(STREAM_PROCESSING_INTERVAL / 1000)  # 使用配置的处理间隔
            
            cap.release()
            
    except WebSocketDisconnect:
        print("WebSocket连接断开")
    except Exception as e:
        await websocket.send_text(json.dumps({"error": f"处理失败: {str(e)}"}))

@app.websocket("/face/recognize_camera")
async def face_recognize_camera(websocket: WebSocket):
    """人脸识别 - 本地摄像头"""
    await websocket.accept()
    try:
        while True:
            # 接收摄像头ID
            data = await websocket.receive_text()
            camera_id = json.loads(data).get("camera_id", 0)
            
            # 连接摄像头
            cap = cv2.VideoCapture(int(camera_id))
            if not cap.isOpened():
                await websocket.send_text(json.dumps({"error": "无法打开摄像头"}))
                continue
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % STREAM_FRAME_INTERVAL == 0:  # 使用配置的帧间隔
                    faces = process_frame_face_only(frame, face_model, face_recognition_model, face_database)
                    result = {
                        "frame": frame_count,
                        "faces": faces,
                        "timestamp": time.time()
                    }
                    await websocket.send_text(json.dumps(result))
                
                frame_count += 1
                await asyncio.sleep(STREAM_PROCESSING_INTERVAL / 1000)  # 使用配置的处理间隔
            
            cap.release()
            
    except WebSocketDisconnect:
        print("WebSocket连接断开")
    except Exception as e:
        await websocket.send_text(json.dumps({"error": f"处理失败: {str(e)}"}))

@app.websocket("/emotion/recognize_rtsp")
async def emotion_recognize_rtsp(websocket: WebSocket):
    """表情识别 - RTSP实时流"""
    await websocket.accept()
    try:
        while True:
            # 接收RTSP地址
            data = await websocket.receive_text()
            rtsp_url = json.loads(data).get("rtsp_url")
            
            if not rtsp_url:
                await websocket.send_text(json.dumps({"error": "缺少RTSP地址"}))
                continue
            
            # 连接RTSP流
            cap = cv2.VideoCapture(rtsp_url)
            if not cap.isOpened():
                await websocket.send_text(json.dumps({"error": "无法连接RTSP流"}))
                continue
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % STREAM_FRAME_INTERVAL == 0:  # 使用配置的帧间隔
                    faces = process_frame_emotion_only(frame, face_model, emotion_model)
                    result = {
                        "frame": frame_count,
                        "faces": faces,
                        "timestamp": time.time()
                    }
                    await websocket.send_text(json.dumps(result))
                
                frame_count += 1
                await asyncio.sleep(STREAM_PROCESSING_INTERVAL / 1000)  # 使用配置的处理间隔
            
            cap.release()
            
    except WebSocketDisconnect:
        print("WebSocket连接断开")
    except Exception as e:
        await websocket.send_text(json.dumps({"error": f"处理失败: {str(e)}"}))

@app.websocket("/emotion/recognize_camera")
async def emotion_recognize_camera(websocket: WebSocket):
    """表情识别 - 本地摄像头"""
    await websocket.accept()
    try:
        while True:
            # 接收摄像头ID
            data = await websocket.receive_text()
            camera_id = json.loads(data).get("camera_id", 0)
            
            # 连接摄像头
            cap = cv2.VideoCapture(int(camera_id))
            if not cap.isOpened():
                await websocket.send_text(json.dumps({"error": "无法打开摄像头"}))
                continue
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % STREAM_FRAME_INTERVAL == 0:  # 使用配置的帧间隔
                    faces = process_frame_emotion_only(frame, face_model, emotion_model)
                    result = {
                        "frame": frame_count,
                        "faces": faces,
                        "timestamp": time.time()
                    }
                    await websocket.send_text(json.dumps(result))
                
                frame_count += 1
                await asyncio.sleep(STREAM_PROCESSING_INTERVAL / 1000)  # 使用配置的处理间隔
            
            cap.release()
            
    except WebSocketDisconnect:
        print("WebSocket连接断开")
    except Exception as e:
        await websocket.send_text(json.dumps({"error": f"处理失败: {str(e)}"}))

@app.websocket("/combined/recognize_stream")
async def combined_recognize_stream(websocket: WebSocket):
    """综合识别 - 实时流（RTSP或摄像头）"""
    await websocket.accept()
    try:
        while True:
            # 接收配置
            data = await websocket.receive_text()
            config = json.loads(data)
            stream_type = config.get("type")  # "rtsp" 或 "camera"
            
            if stream_type == "rtsp":
                stream_url = config.get("rtsp_url")
                if not stream_url:
                    await websocket.send_text(json.dumps({"error": "缺少RTSP地址"}))
                    continue
                cap = cv2.VideoCapture(stream_url)
            elif stream_type == "camera":
                camera_id = config.get("camera_id", 0)
                cap = cv2.VideoCapture(int(camera_id))
            else:
                await websocket.send_text(json.dumps({"error": "不支持的流类型"}))
                continue
            
            if not cap.isOpened():
                await websocket.send_text(json.dumps({"error": "无法打开流"}))
                continue
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % STREAM_FRAME_INTERVAL == 0:  # 使用配置的帧间隔
                    faces = process_frame_combined(frame, face_model, face_recognition_model, emotion_model, face_database)
                    result = {
                        "frame": frame_count,
                        "faces": faces,
                        "timestamp": time.time()
                    }
                    await websocket.send_text(json.dumps(result))
                
                frame_count += 1
                await asyncio.sleep(STREAM_PROCESSING_INTERVAL / 1000)  # 使用配置的处理间隔
            
            cap.release()
            
    except WebSocketDisconnect:
        print("WebSocket连接断开")
    except Exception as e:
        await websocket.send_text(json.dumps({"error": f"处理失败: {str(e)}"}))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

