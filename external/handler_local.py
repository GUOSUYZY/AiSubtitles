import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT_PATH = Path(__file__).parent.parent.absolute()
EXTERNAL_PATH = ROOT_PATH.joinpath('external')
MODELS_PATH = ROOT_PATH.joinpath('models')
TEMP_PATH = ROOT_PATH.joinpath('temp')

# 设置环境变量 - 完全离线模式
os.environ['HF_HUB_OFFLINE'] = '1'  # 强制离线模式
os.environ['HF_ENDPOINT'] = ''  # 清空端点
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ["PATH"] = f"{os.environ['PATH']};{str(EXTERNAL_PATH.joinpath('ffmpeg'))}"

from faster_whisper import WhisperModel


def execute_with_stream(cmd):
    """执行命令并实时显示输出"""
    process = subprocess.Popen(
        cmd,
        encoding='utf-8',
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True
    )

    for line in iter(process.stdout.readline, ''):
        if line:
            sys.stdout.write('\r' + line)
            sys.stdout.flush()

    process.stdout.close()
    return process.wait()


def get_local_model_path(model_size):
    """获取本地模型路径"""
    model_paths = {
        "large-v3-turbo": MODELS_PATH.joinpath("models--mobiuslabsgmbh--faster-whisper-large-v3-turbo"),
        "large-v3": MODELS_PATH.joinpath("models--Systran--faster-whisper-large-v3"),
        "large-v2": MODELS_PATH.joinpath("models--Systran--faster-whisper-large-v2"),
        "medium": MODELS_PATH.joinpath("models--Systran--faster-whisper-medium"),
        "small": MODELS_PATH.joinpath("models--Systran--faster-whisper-small"),
        "base": MODELS_PATH.joinpath("models--Systran--faster-whisper-base"),
        "tiny": MODELS_PATH.joinpath("models--Systran--faster-whisper-tiny"),
    }
    return model_paths.get(model_size)


def check_model_files(model_path):
    """检查模型文件是否完整"""
    if not model_path.exists():
        return False
    
    # 检查snapshots目录
    snapshots_dir = model_path.joinpath("snapshots")
    if not snapshots_dir.exists():
        return False
    
    # 查找所有snapshot目录
    snapshot_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]
    if not snapshot_dirs:
        return False
    
    # 检查每个snapshot目录
    for snapshot_path in snapshot_dirs:
        required_files = ["config.json", "tokenizer.json", "model.bin"]
        all_files_exist = True
        
        for file_name in required_files:
            if not (snapshot_path / file_name).exists():
                all_files_exist = False
                break
        
        if all_files_exist:
            return True
    
    return False


def get_model_snapshot_path(model_path):
    """获取模型的snapshot路径"""
    if not model_path.exists():
        return None
    
    snapshots_dir = model_path.joinpath("snapshots")
    if not snapshots_dir.exists():
        return None
    
    # 查找所有snapshot目录
    snapshot_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]
    if not snapshot_dirs:
        return None
    
    # 检查每个snapshot目录
    for snapshot_path in snapshot_dirs:
        required_files = ["config.json", "tokenizer.json", "model.bin"]
        all_files_exist = True
        
        for file_name in required_files:
            if not (snapshot_path / file_name).exists():
                all_files_exist = False
                break
        
        if all_files_exist:
            return snapshot_path
    
    return None


def load_model(model_size, mode="cpu"):
    """加载模型 - 完全离线"""
    try:
        # 检查本地模型
        local_path = get_local_model_path(model_size)
        print(f"🔍 检查模型路径: {local_path}")
        
        if local_path and check_model_files(local_path):
            print(f"✅ 本地模型文件检查通过: {model_size}")
            
            # 获取正确的snapshot路径
            snapshot_path = get_model_snapshot_path(local_path)
            if snapshot_path:
                print(f"📁 使用snapshot路径: {snapshot_path}")
                
                # 检查模型文件大小
                model_bin_path = snapshot_path / "model.bin"
                if model_bin_path.exists():
                    model_size_mb = model_bin_path.stat().st_size / 1024 / 1024
                    print(f"📊 模型文件大小: {model_size_mb:.1f} MB")
                
                try:
                    model = WhisperModel(
                        str(snapshot_path),
                        device=mode,
                        compute_type="int8",
                        local_files_only=True
                    )
                    print(f"✅ 模型加载成功")
                    return model
                except Exception as e:
                    print(f"❌ 模型初始化失败: {str(e)}")
                    print(f"💡 可能的原因:")
                    print(f"   - 模型文件损坏")
                    print(f"   - 内存不足")
                    print(f"   - 模型版本不兼容")
                    raise
            else:
                print(f"❌ 模型文件不完整: {model_size}")
                print(f"📁 检查的路径: {local_path}")
                raise Exception(f"模型文件不完整: {model_size}")
        else:
            print(f"❌ 本地模型未找到或文件不完整: {model_size}")
            print(f"📁 检查的路径: {local_path}")
            print(f"💡 请下载模型到: {MODELS_PATH}")
            print(f"📥 下载地址: https://huggingface.co/Systran/faster-whisper-{model_size}")
            
            # 强制离线模式，不允许在线下载
            raise Exception(f"本地模型 {model_size} 未找到，请先下载模型文件")
        
    except Exception as e:
        print(f"❌ 模型加载失败: {str(e)}")
        print(f"🔍 错误类型: {type(e).__name__}")
        import traceback
        print(f"📋 详细错误信息:")
        traceback.print_exc()
        raise


def transcribe_audio(audio_file_path, model_size="small", mode="cpu", task="transcribe", target_language=None, source_language=None, progress_callback=None):
    """转录音频文件 - 支持多语言翻译"""
    try:
        # 检查音频文件是否存在
        if not os.path.exists(audio_file_path):
            raise Exception(f"音频文件不存在: {audio_file_path}")
        
        # 检查音频文件大小
        file_size = os.path.getsize(audio_file_path)
        if file_size == 0:
            raise Exception(f"音频文件为空: {audio_file_path}")
        
        print(f"📁 音频文件: {audio_file_path}")
        print(f"📊 文件大小: {file_size / 1024 / 1024:.2f} MB")
        
        # 加载模型
        print(f"🧠 正在加载模型: {model_size}")
        model = load_model(model_size, mode)
        print(f"✅ 模型加载成功")
        
        # 设置语言参数
        language = source_language if source_language else None
        
        # 获取音频时长用于进度计算
        try:
            import av
            container = av.open(audio_file_path)
            audio_duration = container.duration / 1000000  # 转换为秒
            print(f"⏱️ 音频时长: {audio_duration:.1f}秒")
        except Exception as e:
            print(f"⚠️ 无法获取音频时长: {str(e)}")
            audio_duration = 0
        
        # 执行转录
        print(f"🎤 开始转录...")
        try:
            segments, info = model.transcribe(
                audio_file_path, 
                beam_size=5,
                language=language,  # 源语言（可选）
                task=task  # "transcribe" 或 "translate"
            )
        except Exception as e:
            print(f"❌ 转录过程出错: {str(e)}")
            print(f"💡 可能的原因:")
            print(f"   - 模型文件损坏")
            print(f"   - 内存不足")
            print(f"   - 音频文件格式不支持")
            print(f"   - 模型版本兼容性问题")
            raise
        
        print(f"✅ 转录完成")
        print(f"🌍 检测到语言: {info.language} (概率: {info.language_probability:.2f})")
        print(f"📝 任务类型: {task}")
        if target_language and task == "translate":
            print(f"🎯 目标语言: {target_language}")
        
        # 收集结果并更新进度
        result = []
        segment_count = 0
        for segment in segments:
            print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
            result.append({
                "text": segment.text.strip(),
                "start": segment.start,
                "end": segment.end
            })
            
            # 更新进度
            segment_count += 1
            if progress_callback and audio_duration > 0:
                # 基于当前处理的时间段计算进度
                current_progress = min(10 + (segment.end / audio_duration) * 80, 90)
                progress_callback(current_progress)
        
        print(f"📊 总共处理了 {len(result)} 个片段")
        return result, info
        
    except Exception as e:
        print(f"❌ 转录失败: {str(e)}")
        print(f"🔍 错误类型: {type(e).__name__}")
        import traceback
        print(f"📋 详细错误信息:")
        traceback.print_exc()
        raise


def main():
    """主函数"""
    if len(sys.argv) < 4:
        print("用法: python handler_local.py <model_size> <audio_file> <uuid> [mode] [task] [target_language] [source_language]")
        sys.exit(1)
    
    model_size = sys.argv[1]
    audio_file_path = sys.argv[2]
    uuid = sys.argv[3]
    mode = sys.argv[4] if len(sys.argv) > 4 else 'cpu'
    task = sys.argv[5] if len(sys.argv) > 5 else 'transcribe'
    target_language = sys.argv[6] if len(sys.argv) > 6 else None
    source_language = sys.argv[7] if len(sys.argv) > 7 else None
    
    # 检查音频文件是否存在
    if not Path(audio_file_path).exists():
        print(f"音频文件不存在: {audio_file_path}")
        sys.exit(1)
    
    try:
        # 执行转录
        result, info = transcribe_audio(audio_file_path, model_size, mode, task, target_language, source_language)
        
        # 保存结果
        output_file = TEMP_PATH.joinpath(f'{uuid}.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "input_file": audio_file_path,
                "model": model_size,
                "device": mode,
                "task": task,
                "target_language": target_language,
                "source_language": source_language,
                "language": info.language,
                "language_probability": info.language_probability,
                "segments": result,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }, f, ensure_ascii=False, indent=2)
        
        print(f"转录完成，结果保存到: {output_file}")
        
    except Exception as e:
        print(f"处理失败: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 