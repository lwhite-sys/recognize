"""
项目配置文件

本文件包含了人脸识别与表情识别系统的所有配置参数，
包括阈值设置、性能配置、文件处理配置等。

主要配置说明：
1. RECOGNITION_THRESHOLDS: 识别相关的阈值配置
2. PERFORMANCE_CONFIG: 性能相关的配置参数
3. FILE_PROCESSING_CONFIG: 文件处理相关的配置
4. WEBSOCKET_CONFIG: WebSocket连接相关的配置
5. SECURITY_CONFIG: 安全相关的配置参数

修改配置后需要重启服务器才能生效。
"""

from pathlib import Path

# 基础路径
BASE_DIR = Path(__file__).parent

# 模型配置
MODEL_PATHS = {
    'face_detection': BASE_DIR / 'models' / 'face_detection' / 'best.pt',
    'face_recognition': BASE_DIR / 'models' / 'face_recognition' / '16_backbone.pth',
    'emotion_recognition': BASE_DIR / 'models' / 'emotion_recognition' / 'emotion_best.pt',
    'action_detection': BASE_DIR / 'models' / 'action_detect' / 'action_best.pt'
}

# 数据库配置
DATABASE_DIR = BASE_DIR / 'database'

# 识别配置 - 阈值设置
RECOGNITION_THRESHOLDS = {
    # 人脸检测阈值 - 只有置信度超过此值的人脸才会被处理
    # 范围：0.0 - 1.0，建议值：0.3-0.6
    # 值越低，检测到的人脸越多，但可能包含一些低质量人脸
    # 降低阈值以检测更多人脸
    'face_detection_confidence': 0.4,
    
    # 人脸识别相似度阈值 - 只有相似度超过此值才会被认为是同一个人
    # 范围：0.0 - 1.0，建议值：0.4-0.7
    # 值越高，识别准确性越高，但可能增加"Unknown"结果
    'face_recognition_similarity': 0.5,
    
    # 表情识别置信度阈值 - 只有置信度超过此值的表情才会被返回
    # 范围：0.0 - 1.0，建议值：0.5-0.8
    # 值越高，表情识别准确性越高，但可能增加"neutral"结果
    'emotion_recognition_confidence': 0.6,
    
    # 表情显示阈值 - 只有置信度超过此值的表情才会显示，否则显示为"unknown"
    # 范围：0.0 - 1.0，建议值：0.3-0.6
    # 值越高，显示的表情越可靠，但可能增加"unknown"结果
    'emotion_display_confidence': 0.415,
    
    # 人脸识别默认阈值（用于match_face函数）
    # 当没有指定阈值时使用此值
    'face_matching_default': 0.5,
    
    # 表情识别默认置信度（当模型无法提供置信度时使用）
    # 用于处理模型输出异常的情况
    'emotion_default_confidence': 0.8,
    
    # 中性表情默认置信度
    # 当无法识别具体表情时使用
    'neutral_emotion_confidence': 0.5,
    
    # 行为识别置信度阈值 - 只有置信度超过此值的行为才会被返回
    # 范围：0.0 - 1.0，建议值：0.3-0.7
    # 值越高，行为识别准确性越高，但可能增加"unknown"结果
    'action_detection_confidence': 0.4,
    
    # 行为显示阈值 - 只有置信度超过此值的行为才会显示，否则显示为"unknown"
    # 范围：0.0 - 1.0，建议值：0.3-0.6
    # 值越高，显示的行为越可靠，但可能增加"unknown"结果
    'action_display_confidence': 0.35
}

# 表情标签配置
# 支持的表情类型，顺序必须与模型输出一致
# 根据测试结果，模型实际输出9个类别
EMOTION_LABELS = ['angry', 'contempt', 'disgust', 'fear', 'happy', 'natural', 'sad', 'sleepy', 'surprised']

# 情绪标签映射（用于兼容旧版本）
EMOTION_LABEL_MAPPING = {
    'angry': 'angry',
    'contempt': 'disgust',  # 映射到disgust
    'disgust': 'disgust',
    'fear': 'fear',
    'happy': 'happy',
    'natural': 'neutral',   # 映射到neutral
    'sad': 'sad',
    'sleepy': 'neutral',    # 映射到neutral
    'surprised': 'surprise' # 映射到surprise
}

# 行为标签配置
# 支持的行为类型，顺序必须与模型输出一致
ACTION_LABELS = ['writing', 'reading', 'listening', 'turning_around', 'raising_hand', 'standing', 'discussing', 'guiding']

# 行为标签映射（中文显示）
ACTION_LABEL_MAPPING = {
    'writing': '写作',
    'reading': '阅读',
    'listening': '听讲',
    'turning_around': '转身',
    'raising_hand': '举手',
    'standing': '站立',
    'discussing': '讨论',
    'guiding': '指导'
}

# 行为颜色配置（用于雷达图显示）
ACTION_COLORS = {
    'writing': '#3B82F6',      # 蓝色
    'reading': '#F97316',      # 橙色
    'listening': '#10B981',    # 绿色
    'turning_around': '#EF4444', # 红色
    'raising_hand': '#8B5CF6', # 紫色
    'standing': '#A0522D',    # 棕色
    'discussing': '#EC4899',   # 粉色
    'guiding': '#6B7280'       # 灰色
}

# 服务器配置
HOST = "127.0.0.1"  # 监听本地回环地址
PORT = 8001        # 服务端口（改为8001避免冲突）

# 性能配置
PERFORMANCE_CONFIG = {
    # 视频处理帧率限制
    # 每秒最多处理的帧数，值越小性能越好但处理越慢
    'frame_rate_limit': 1,
    
    # 实时流处理间隔（毫秒）
    # 控制实时流处理的频率，值越小延迟越低但CPU占用越高
    'stream_processing_interval': 100,
    
    # 视频处理帧间隔（每N帧处理一次）
    # 值越大处理越快但可能错过一些人脸
    'video_frame_interval': 10,
    
    # 实时流处理帧间隔（每N帧处理一次）
    # 值越大性能越好但实时性越差
    'stream_frame_interval': 5
}

# 人脸质量阈值
FACE_QUALITY_THRESHOLDS = {
    'eye_distance': 30,      # 眼睛间最小距离（像素）
    'mouth_nose_distance': 20,  # 嘴鼻间最小距离（像素）
    'min_face_size': 80      # 最小人脸尺寸（像素）
}

# 文件处理配置
FILE_PROCESSING_CONFIG = {
    # 支持的文件格式
    'supported_image_formats': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'],
    'supported_video_formats': ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv'],
    
    # 文件大小限制（字节）
    'max_image_size': 50 * 1024 * 1024,  # 50MB
    'max_video_size': 500 * 1024 * 1024,  # 500MB
    
    # 临时文件配置
    'temp_file_prefix': 'temp_',
    'temp_file_suffix': '.mp4'
}

# WebSocket配置
WEBSOCKET_CONFIG = {
    # 连接超时时间（秒）
    'connection_timeout': 30,
    
    # 消息超时时间（秒）
    'message_timeout': 10,
    
    # 重连间隔（秒）
    'reconnect_interval': 5,
    
    # 最大重连次数
    'max_reconnect_attempts': 3
}

# 日志配置
LOGGING_CONFIG = {
    'level': 'INFO',  # 日志级别：DEBUG, INFO, WARNING, ERROR, CRITICAL
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'recognition_system.log'
}

# 缓存配置
CACHE_CONFIG = {
    # 结果缓存时间（秒）
    'result_cache_time': 300,  # 5分钟
    
    # 特征缓存时间（秒）
    'feature_cache_time': 3600,  # 1小时
    
    # 最大缓存条目数
    'max_cache_entries': 1000
}

# 安全配置
SECURITY_CONFIG = {
    # 允许的源域名（CORS）
    # 设置为 ['*'] 允许所有域名访问
    'allowed_origins': ['*'],
    
    # 请求频率限制
    'rate_limit': {
        'requests_per_minute': 60,  # 每分钟最大请求数
        'burst_size': 10            # 突发请求最大数量
    },
    
    # API密钥验证（可选）
    'require_api_key': False,       # 是否要求API密钥
    'api_key_header': 'X-API-Key'  # API密钥请求头名称
}

# 调试配置
DEBUG_CONFIG = {
    # 是否启用调试模式
    'enabled': False,
    
    # 是否保存中间结果
    'save_intermediate_results': False,
    
    # 是否显示详细日志
    'verbose_logging': False,
    
    # 是否启用性能分析
    'performance_profiling': False
}


