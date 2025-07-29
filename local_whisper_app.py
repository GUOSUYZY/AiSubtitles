#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地Whisper语音识别应用 - 完全离线版本
"""

import os
import sys
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import threading
import subprocess
import time

# 设置环境变量 - 完全离线模式
os.environ['HF_HUB_OFFLINE'] = '1'  # 强制离线模式
os.environ['HF_ENDPOINT'] = ''  # 清空端点
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

from external.handler_local import transcribe_audio, get_local_model_path, check_model_files

class LocalWhisperApp:
    def __init__(self, root):
        self.root = root
        self.root.title("本地Whisper语音识别工具 - 完全离线版")
        self.root.geometry("1000x800")
        
        # 设置路径
        self.current_dir = Path(__file__).parent
        self.temp_dir = self.current_dir.joinpath("temp")
        self.temp_dir.mkdir(exist_ok=True)
        self.models_dir = self.current_dir.joinpath("models")
        
        # 可用模型
        self.available_models = {
            "tiny": "最小模型 (39MB) - 速度快，精度较低",
            "base": "基础模型 (74MB) - 平衡速度和精度",
            "small": "小型模型 (244MB) - 好精度",
            "medium": "中型模型 (769MB) - 很好精度",
            "large-v3-turbo": "推荐模型 (1.5GB) - 最佳精度"
        }
        
        self.setup_ui()
        self.check_local_models()
        self.check_gpu_availability()
    
    def setup_ui(self):
        """设置用户界面"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 标题
        title_label = ttk.Label(main_frame, text="🤖 本地Whisper语音识别工具 - 完全离线版", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # 文件选择区域
        file_frame = ttk.LabelFrame(main_frame, text="📁 音频文件选择", padding="10")
        file_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.file_path_var = tk.StringVar()
        file_entry = ttk.Entry(file_frame, textvariable=self.file_path_var, width=80)
        file_entry.grid(row=0, column=0, padx=(0, 10))
        
        browse_btn = ttk.Button(file_frame, text="浏览", command=self.browse_file)
        browse_btn.grid(row=0, column=1)
        
        # 模型选择区域
        model_frame = ttk.LabelFrame(main_frame, text="🧠 模型选择", padding="10")
        model_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.model_var = tk.StringVar(value="small")
        for i, (model_name, description) in enumerate(self.available_models.items()):
            rb = ttk.Radiobutton(model_frame, text=f"{model_name}: {description}", 
                                variable=self.model_var, value=model_name)
            rb.grid(row=i, column=0, sticky=tk.W, pady=2)
        
        # 设备选择
        device_frame = ttk.Frame(model_frame)
        device_frame.grid(row=len(self.available_models), column=0, sticky=tk.W, pady=(10, 0))
        
        ttk.Label(device_frame, text="设备类型:").grid(row=0, column=0, padx=(0, 10))
        self.device_var = tk.StringVar(value="cpu")
        self.gpu_button = ttk.Radiobutton(device_frame, text="GPU (如果可用)", variable=self.device_var, value="cuda")
        self.gpu_button.grid(row=0, column=1)
        ttk.Radiobutton(device_frame, text="CPU", variable=self.device_var, value="cpu").grid(row=0, column=2)
        
        # 自动检测GPU
        # self.check_gpu_availability() # Moved to __init__
        
        # 控制按钮
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, columnspan=2, pady=20)
        
        self.start_btn = ttk.Button(button_frame, text="🚀 开始处理", command=self.start_transcription)
        self.start_btn.grid(row=0, column=0, padx=(0, 10))
        
        self.stop_btn = ttk.Button(button_frame, text="⏹️ 停止", command=self.stop_transcription, state="disabled")
        self.stop_btn.grid(row=0, column=1, padx=(0, 10))
        
        ttk.Button(button_frame, text="📁 打开文件目录", command=self.open_results_folder).grid(row=0, column=2)
        
        # 进度条
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 状态标签
        self.status_var = tk.StringVar(value="就绪")
        status_label = ttk.Label(main_frame, textvariable=self.status_var, font=("Arial", 10))
        status_label.grid(row=6, column=0, columnspan=2)
        
        # 日志区域
        log_frame = ttk.LabelFrame(main_frame, text="📝 处理日志", padding="10")
        log_frame.grid(row=7, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        
        self.log_text = tk.Text(log_frame, height=15, width=100)
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(7, weight=1)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        # 添加底部提示
        bottom_note = ttk.Label(main_frame, text="💡 注意：所有输出均为英文格式", 
                               foreground="orange", font=("Arial", 10, "bold"))
        bottom_note.grid(row=8, column=0, columnspan=2, pady=(10, 0))
        
        # 转录线程
        self.transcription_thread = None
        self.stop_flag = False
    
    def log(self, message):
        """添加日志消息"""
        timestamp = time.strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        self.log_text.insert(tk.END, log_message)
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def check_local_models(self):
        """检查本地模型"""
        self.log("🔍 检查本地模型...")
        
        available_models = []
        for model_name in self.available_models.keys():
            model_path = get_local_model_path(model_name)
            if model_path and check_model_files(model_path):
                available_models.append(model_name)
                self.log(f"✅ 找到本地模型: {model_name}")
            else:
                self.log(f"❌ 未找到本地模型: {model_name}")
        
        if not available_models:
            self.log("⚠️ 警告: 未找到任何本地模型!")
            self.log("💡 请下载模型文件到 models/ 目录")
            self.start_btn.config(state="disabled")
        else:
            self.log(f"✅ 找到 {len(available_models)} 个本地模型")
            # 设置默认模型为第一个可用的
            if available_models:
                self.model_var.set(available_models[0])
    

    
    def browse_file(self):
        """浏览文件或文件夹"""
        from tkinter import messagebox
        
        # 创建一个选择对话框
        choice_window = tk.Toplevel(self.root)
        choice_window.title("选择处理方式")
        choice_window.geometry("400x200")
        choice_window.transient(self.root)
        choice_window.grab_set()
        
        # 居中显示
        choice_window.geometry("+%d+%d" % (self.root.winfo_rootx() + 50, self.root.winfo_rooty() + 50))
        
        # 标题
        title_label = ttk.Label(choice_window, text="请选择处理方式", font=("Arial", 14, "bold"))
        title_label.pack(pady=(20, 30))
        
        # 按钮框架
        button_frame = ttk.Frame(choice_window)
        button_frame.pack(pady=20)
        
        def select_single_file():
            """选择单个文件"""
            choice_window.destroy()
            file_types = [
                ("所有支持的文件", "*.*"),
                ("视频文件", "*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm"),
                ("音频文件", "*.wav *.mp3 *.m4a *.flac *.ogg")
            ]
            
            filename = filedialog.askopenfilename(
                title="选择音频或视频文件（单个处理）",
                filetypes=file_types
            )
            
            if filename:
                self.file_path_var.set(filename)
                self.log(f"📁 选择文件: {filename}")
        
        def select_folder():
            """选择文件夹进行批量处理"""
            choice_window.destroy()
            folder_path = filedialog.askdirectory(title="选择包含视频的文件夹（批量处理）")
            if folder_path:
                self.file_path_var.set(folder_path)
                self.log(f"📁 选择文件夹: {folder_path}")
                self.log("💡 将处理文件夹中的所有视频文件")
        
        def cancel_selection():
            """取消选择"""
            choice_window.destroy()
        
        # 创建按钮
        ttk.Button(button_frame, text="📄 选择单个文件", 
                  command=select_single_file, width=20).pack(pady=5)
        ttk.Button(button_frame, text="📁 选择文件夹（批量处理）", 
                  command=select_folder, width=20).pack(pady=5)
        ttk.Button(button_frame, text="❌ 取消", 
                  command=cancel_selection, width=20).pack(pady=5)
    
    def process_folder(self, folder_path):
        """处理文件夹中的所有视频"""
        video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm']
        folder = Path(folder_path)
        
        # 查找所有视频文件
        video_files = []
        for ext in video_extensions:
            video_files.extend(folder.glob(f"*{ext}"))
            video_files.extend(folder.glob(f"*{ext.upper()}"))
        
        if not video_files:
            self.log(f"❌ 在文件夹中未找到视频文件: {folder_path}")
            return
        
        self.log(f"📁 找到 {len(video_files)} 个视频文件")
        
        # 处理每个视频文件
        for i, video_file in enumerate(video_files, 1):
            if self.stop_flag:
                break
                
            self.log(f"🎬 处理第 {i}/{len(video_files)} 个文件: {video_file.name}")
            
            # 设置当前文件路径
            self.file_path_var.set(str(video_file))
            
            # 更新批量处理进度
            batch_progress = (i - 1) / len(video_files) * 100
            self.progress_var.set(batch_progress)
            self.status_var.set(f"批量处理中... {i}/{len(video_files)}")
            
            # 处理单个文件
            try:
                self.process_single_file(str(video_file))
            except Exception as e:
                self.log(f"❌ 处理文件失败 {video_file.name}: {str(e)}")
                continue
            
            self.log(f"✅ 完成第 {i}/{len(video_files)} 个文件")
        
        self.progress_var.set(100)
        self.status_var.set("批量处理完成")
        self.log(f"🎉 批量处理完成！共处理 {len(video_files)} 个文件")
    
    def process_single_file(self, input_path):
        """处理单个文件"""
        model_name = self.model_var.get()
        device = self.device_var.get()
        
        # 简化语言设置
        source_language = None  # 默认自动检测
        
        self.log(f"🚀 开始处理...")
        self.log(f"📁 输入文件: {input_path}")
        self.log(f"🧠 使用模型: {model_name}")
        self.log(f"💻 设备类型: {device}")
        
        # 重置进度条
        self.progress_var.set(0)
        
        # 检查文件类型
        file_ext = Path(input_path).suffix.lower()
        if file_ext in ['.mp4', '.avi', '.mkv', '.mov', '.wmv']:
            self.log("🎬 检测到视频文件，正在提取音频...")
            self.progress_var.set(5)
            self.status_var.set("正在提取音频...")
            audio_path = self.extract_audio(input_path)
            if not audio_path:
                raise Exception("音频提取失败")
        else:
            audio_path = input_path
            self.progress_var.set(5)
        
        # 执行转录
        self.log("🎤 开始语音识别...")
        self.progress_var.set(10)
        self.status_var.set("正在识别语音...")
        
        # 获取音频时长用于进度计算
        try:
            import av
            container = av.open(audio_path)
            audio_duration = container.duration / 1000000  # 转换为秒
            self.log(f"⏱️ 音频时长: {audio_duration:.1f}秒")
        except:
            audio_duration = 0
        
        # 定义进度回调函数
        def update_progress(progress):
            if not self.stop_flag:
                self.progress_var.set(progress)
                self.status_var.set(f"正在识别语音... {progress:.0f}%")
                self.root.update_idletasks()
        
        result, info = transcribe_audio(audio_path, model_name, device, "transcribe", "en", source_language, progress_callback=update_progress)
        
        if self.stop_flag:
            return
        
        self.progress_var.set(90)
        self.status_var.set("正在生成SRT文件...")
        
        self.log(f"✅ 处理完成!")
        self.log(f"🌍 检测到语言: {info.language} (概率: {info.language_probability:.2f})")
        self.log(f"📝 处理片段数: {len(result)}")
        
        # 生成SRT文件 - 保存在与输入文件相同的位置
        input_file_path = Path(input_path)
        srt_file = input_file_path.with_suffix('.srt')
        
        with open(srt_file, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(result, 1):
                start_time = self.format_time(segment['start'])
                end_time = self.format_time(segment['end'])
                f.write(f"{i}\n{start_time} --> {end_time}\n{segment['text']}\n\n")
        
        self.log(f"📄 SRT字幕已生成: {srt_file}")
        
        # 清理临时音频文件
        if audio_path != input_path:
            try:
                Path(audio_path).unlink()
                self.log("🧹 临时音频文件已清理")
            except:
                pass
        
        self.progress_var.set(100)
        self.status_var.set("处理完成")
    
    def extract_audio(self, video_path):
        """从视频中提取音频"""
        audio_path = self.temp_dir.joinpath(f"extracted_audio_{int(time.time())}.wav")
        
        ffmpeg_cmd = [
            str(self.current_dir.joinpath("external", "ffmpeg", "ffmpeg.exe")),
            "-y",  # 覆盖输出文件
            "-i", video_path,
            "-f", "wav",
            "-vn",  # 不包含视频
            str(audio_path)
        ]
        
        self.log(f"🎵 提取音频: {video_path}")
        
        try:
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                self.log(f"✅ 音频提取成功: {audio_path}")
                return str(audio_path)
            else:
                self.log(f"❌ 音频提取失败: {result.stderr}")
                return None
        except subprocess.TimeoutExpired:
            self.log("❌ 音频提取超时")
            return None
        except Exception as e:
            self.log(f"❌ 音频提取错误: {str(e)}")
            return None
    
    def start_transcription(self):
        """开始转录"""
        if not self.file_path_var.get():
            messagebox.showerror("错误", "请选择音频或视频文件")
            return
        
        if not Path(self.file_path_var.get()).exists():
            messagebox.showerror("错误", "文件不存在")
            return
        
        # 禁用开始按钮，启用停止按钮
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.stop_flag = False
        
        # 在新线程中执行转录
        self.transcription_thread = threading.Thread(target=self.transcription_worker)
        self.transcription_thread.daemon = True
        self.transcription_thread.start()
    
    def transcription_worker(self):
        """转录工作线程"""
        try:
            input_path = self.file_path_var.get()
            
            if Path(input_path).is_dir():
                self.process_folder(input_path)
            else:
                self.process_single_file(input_path)
            
        except Exception as e:
            self.log(f"❌ 处理失败: {str(e)}")
            self.status_var.set("处理失败")
        finally:
            # 恢复按钮状态
            self.start_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
    
    def format_time(self, seconds):
        """格式化时间为SRT格式"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
    
    def show_results_preview(self, result):
        """显示结果预览"""
        preview_window = tk.Toplevel(self.root)
        preview_window.title("处理结果预览")
        preview_window.geometry("600x400")
        
        # 创建文本框显示结果
        text_widget = tk.Text(preview_window, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(preview_window, orient="vertical", command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # 显示前10个片段
        for i, segment in enumerate(result[:10], 1):
            text_widget.insert(tk.END, f"片段 {i} [{segment['start']:.2f}s - {segment['end']:.2f}s]:\n")
            text_widget.insert(tk.END, f"{segment['text']}\n\n")
        
        if len(result) > 10:
            text_widget.insert(tk.END, f"... 还有 {len(result) - 10} 个片段\n")
        
        # 配置网格权重
        preview_window.columnconfigure(0, weight=1)
        preview_window.rowconfigure(0, weight=1)
    
    def stop_transcription(self):
        """停止转录"""
        self.stop_flag = True
        self.log("⏹️ 正在停止处理...")
        self.status_var.set("正在停止...")
    
    def open_results_folder(self):
        """打开输入文件所在文件夹"""
        import subprocess
        try:
            if self.file_path_var.get():
                input_path = self.file_path_var.get()
                input_file_path = Path(input_path)
                
                # 如果是文件夹，直接打开文件夹
                if input_file_path.is_dir():
                    folder_path = input_file_path
                else:
                    # 如果是文件，打开文件所在的文件夹
                    folder_path = input_file_path.parent
                
                # 使用更安全的方式打开文件夹
                folder_str = str(folder_path)
                
                # 检查路径是否存在
                if not folder_path.exists():
                    messagebox.showerror("错误", f"文件夹不存在: {folder_str}")
                    return
                
                # 使用 os.startfile 替代 subprocess.run
                import os
                try:
                    os.startfile(folder_str)
                except Exception as e:
                    # 如果 os.startfile 失败，尝试使用 subprocess
                    try:
                        subprocess.run(['explorer', folder_str], check=True, shell=True)
                    except subprocess.CalledProcessError:
                        # 如果还是失败，尝试使用 cmd
                        subprocess.run(['cmd', '/c', 'start', folder_str], check=False)
                        
            else:
                messagebox.showinfo("提示", "请先选择文件")
        except Exception as e:
            messagebox.showerror("错误", f"无法打开文件夹: {str(e)}")

    def check_gpu_availability(self):
        """检测GPU可用性"""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                self.log(f"✅ GPU可用: {gpu_name} (共{gpu_count}个GPU)")
                # 自动选择GPU
                self.device_var.set("cuda")
                # 更新GPU按钮文本
                self.gpu_button.config(text=f"GPU ({gpu_name})")
            else:
                self.log("❌ GPU不可用，使用CPU模式")
                self.device_var.set("cpu")
                # 更新GPU按钮文本
                self.gpu_button.config(text="GPU (没有)")
        except Exception as e:
            self.log(f"⚠️ GPU检测失败: {str(e)}")
            self.device_var.set("cpu")
            # 更新GPU按钮文本
            self.gpu_button.config(text="GPU (检测失败)")

def main():
    """主函数"""
    root = tk.Tk()
    app = LocalWhisperApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 