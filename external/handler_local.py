#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完全离线的语音识别处理模块
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Callable
import time

# 设置环境变量 - 强制离线模式
os.environ['HF_HUB_OFFLINE'] = '1'  # 强制离线模式
os.environ['HF_ENDPOINT'] = ''  # 清空端点
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'  # Transformers离线模式
os.environ['HF_DATASETS_OFFLINE'] = '1'  # Datasets离线模式

# 设置路径
ROOT_PATH = Path(__file__).parent.parent.absolute()
MODELS_PATH = ROOT_PATH.joinpath('models')
TEMP_PATH = ROOT_PATH.joinpath('temp')

# 确保临时目录存在
TEMP_PATH.mkdir(exist_ok=True)

# 模型映射
_MODELS = {
    "tiny.en": "Systran/faster-whisper-tiny.en",
    "tiny": "Systran/faster-whisper-tiny",
    "base.en": "Systran/faster-whisper-base.en",
    "base": "Systran/faster-whisper-base",
    "small.en": "Systran/faster-whisper-small.en",
    "small": "Systran/faster-whisper-small",
    "medium.en": "Systran/faster-whisper-medium.en",
    "medium": "Systran/faster-whisper-medium",
    "large-v1": "Systran/faster-whisper-large-v1",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large": "Systran/faster-whisper-large-v3",
    "distil-large-v2": "Systran/faster-distil-whisper-large-v2",
    "distil-medium.en": "Systran/faster-distil-whisper-medium.en",
    "distil-small.en": "Systran/faster-distil-whisper-small.en",
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
}

def get_local_model_path(model_name: str) -> Optional[Path]:
    """获取本地模型路径"""
    if model_name not in _MODELS:
        return None
    
    model_id = _MODELS[model_name]
    model_path = MODELS_PATH.joinpath(f"models--{model_id.replace('/', '--')}")
    
    if model_path.exists():
        # 查找snapshots目录下的模型文件
        snapshots_dir = model_path.joinpath("snapshots")
        if snapshots_dir.exists():
            # 获取第一个快照目录
            snapshots = list(snapshots_dir.iterdir())
            if snapshots:
                return snapshots[0]
    
    return None

def check_model_files(model_path: Path) -> bool:
    """检查模型文件是否完整"""
    required_files = ['config.json', 'model.bin']
    
    for file_name in required_files:
        if not model_path.joinpath(file_name).exists():
            return False
    
    return True

def get_available_models() -> List[str]:
    """获取可用的本地模型列表"""
    available_models = []
    
    for model_name in _MODELS.keys():
        model_path = get_local_model_path(model_name)
        if model_path and check_model_files(model_path):
            available_models.append(model_name)
    
    return available_models

def transcribe_audio(
    audio_file_path: str,
    model_size: str,
    device: str = "cpu",
    task: str = "transcribe",
    language: str = "en",
    source_language: Optional[str] = None,
    progress_callback: Optional[Callable[[float], None]] = None
) -> tuple[List[Dict], object]:
    """
    执行音频转录
    
    Args:
        audio_file_path: 音频文件路径
        model_size: 模型大小
        device: 设备类型 (cpu/cuda)
        task: 任务类型 (transcribe/translate)
        language: 目标语言
        source_language: 源语言 (None为自动检测)
        progress_callback: 进度回调函数
    
    Returns:
        (segments, info): 转录结果和模型信息
    """
    
    # 检查模型是否可用
    model_path = get_local_model_path(model_size)
    if not model_path or not check_model_files(model_path):
        raise ValueError(f"模型 {model_size} 不可用或文件不完整")
    
    # 导入faster-whisper
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise ImportError("请安装 faster-whisper: pip install faster-whisper")
    
    # 设置模型路径
    model_path_str = str(model_path)
    
    # 创建模型实例
    model = WhisperModel(
        model_path_str,
        device=device,
        compute_type="int8",  # 使用int8量化以提高性能
        download_root=None,  # 不使用下载
        local_files_only=True  # 只使用本地文件
    )
    
    # 执行转录
    segments, info = model.transcribe(
        audio_file_path,
        beam_size=5,
        language=source_language,
        task=task
    )
    
    # 收集结果
    result = []
    total_segments = 0
    
    # 估算总段数用于进度计算
    try:
        import av
        container = av.open(audio_file_path)
        audio_duration = container.duration / 1000000  # 转换为秒
        estimated_segments = max(1, int(audio_duration / 30))  # 假设每30秒一段
    except:
        estimated_segments = 100  # 默认值
    
    for segment in segments:
        total_segments += 1
        
        # 更新进度
        if progress_callback:
            progress = min(90, 10 + (total_segments / estimated_segments) * 80)
            progress_callback(progress)
        
        result.append({
            "text": segment.text,
            "start": segment.start,
            "end": segment.end
        })
    
    return result, info

def save_result_to_json(result: List[Dict], uuid: str) -> str:
    """保存结果到JSON文件"""
    output_file = TEMP_PATH.joinpath(f'{uuid}.json')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    return str(output_file)

def save_result_to_srt(result: List[Dict], output_path: str) -> str:
    """保存结果到SRT字幕文件"""
    def format_time(seconds: float) -> str:
        """格式化时间为SRT格式"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(result, 1):
            start_time = format_time(segment['start'])
            end_time = format_time(segment['end'])
            f.write(f"{i}\n{start_time} --> {end_time}\n{segment['text']}\n\n")
    
    return output_path

def main():
    """命令行入口点"""
    if len(sys.argv) < 4:
        print("用法: python handler_local.py <model_size> <audio_file> <uuid> [device]")
        print("示例: python handler_local.py small audio.wav 12345 cpu")
        sys.exit(1)
    
    model_size = sys.argv[1]
    audio_file_path = sys.argv[2]
    uuid = sys.argv[3]
    device = sys.argv[4] if len(sys.argv) > 4 else 'cpu'
    
    try:
        # 检查模型可用性
        available_models = get_available_models()
        if model_size not in available_models:
            print(f"错误: 模型 {model_size} 不可用")
            print(f"可用模型: {available_models}")
            sys.exit(1)
        
        # 执行转录
        result, info = transcribe_audio(audio_file_path, model_size, device)
        
        # 保存结果
        json_file = save_result_to_json(result, uuid)
        
        print(f"检测到语言: {info.language} (概率: {info.language_probability:.2f})")
        print(f"处理完成，结果保存到: {json_file}")
        
    except Exception as e:
        print(f"错误: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 