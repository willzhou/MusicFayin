# Author: Ningbo Wise Effects, Inc. (汇视创影) & Will Zhou
# License: Apache 2.0

"""
通用工具函数模块
包含与音乐生成相关的辅助函数和工具
"""

import streamlit as st
import json
import re
import torch
import psutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional


# 从config.py导入常量
from config import (
    MUSIC_SECTION_TEMPLATES,
    EMOTIONS, SINGER_GENDERS, GENRES, INSTRUMENTATIONS, TIMBRES, 
    STRUCTURE_TEMPLATES, SECTION_DEFINITIONS
)

# 在文件顶部添加项目根目录定义
PROJECT_ROOT = Path(__file__).parent  # 假设musicfayin.py现在放在SongGeneration的父目录
SONG_GEN_DIR = PROJECT_ROOT / "SongGeneration"


def get_absolute_path(relative_path: str, project_root: Path = PROJECT_ROOT, song_gen_dir: Path = SONG_GEN_DIR) -> Path:
    """将相对路径转换为绝对路径"""
    path = Path(relative_path)
    if relative_path.startswith("ckpt/"):
        return song_gen_dir / path.relative_to("ckpt/")
    return project_root / path

def clean_generated_lyrics(raw_lyrics: str) -> str:
    """格式化原始歌词文本"""
    # 替换规则：中文标点和空格都改为英文句点
    replace_rules = {
        ' ': '.',  # 空格
        '，': '.', '。': '.', '、': '.', '；': '.', '：': '.',
        '？': '.', '！': '.', '「': '.', '」': '.', '『': '.',
        '』': '.', '（': '.', '）': '.', '《': '.', '》': '.',
        '【': '.', '】': '.', '『': '.', '』': '.', '〔': '.',
        '〕': '.', '—': '.', '～': '.', '…': '.', '·': '.'
    }
    
    sections = []
    current_section = None
    current_lines = []
    
    for line in raw_lyrics.split('\n'):
        line = line.strip()
        if not line:
            continue
        
        # 检测段落标记如[verse]
        section_match = re.match(r'^\[([a-z\-]+)\]$', line)
        if section_match:
            if current_section is not None:
                sections.append((current_section, current_lines))
            current_section = section_match.group(1)
            current_lines = []
        elif current_section is not None:
            # 替换所有指定字符为句点
            cleaned_line = ''.join(
                replace_rules.get(char, char) 
                for char in line
            ).strip('.')  # 去除首尾多余的句点
            
            # 合并连续的句点为一个
            cleaned_line = re.sub(r'\.+', '.', cleaned_line)
            
            if cleaned_line:
                current_lines.append(cleaned_line)
    
    # 添加最后一段
    if current_section is not None:
        sections.append((current_section, current_lines))
    
    # 格式化各段落
    formatted_sections = []
    for section_type, lines in sections:
        if section_type in ['verse', 'chorus', 'bridge']:
            # 人声段落：用"."连接行，并确保不重复
            content = ".".join(
                line.rstrip('.') for line in lines 
                if line and line != '.'
            )
            formatted = f"[{section_type}] {content}" if content else f"[{section_type}]"
        else:
            # 器乐段落：不包含内容
            formatted = f"[{section_type}]"
        formatted_sections.append(formatted)
    
    return " ; ".join(formatted_sections)

@st.cache_resource(ttl=300)  # 缓存5分钟
def get_gpu_memory() -> Optional[Dict[str, float]]:
    """获取GPU显存信息（单位：GB）"""
    try:
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            total_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)  # 转换为GB
            used_memory = torch.cuda.memory_allocated(device) / (1024**3)
            free_memory = total_memory - used_memory
            return {
                "total": total_memory,
                "used": used_memory,
                "free": free_memory
            }
        return None
    except Exception:
        return None

def save_jsonl(entries: List[Dict], filename: str) -> str:
    """保存JSONL文件"""
    output_dir = get_absolute_path("output")
    output_dir.mkdir(exist_ok=True)
    filepath = output_dir / filename
    
    with open(filepath, "w", encoding="utf-8") as f:
        for entry in entries:
            # 确保所有值都是可序列化的
            serializable_entry = {
                k: str(v) if not isinstance(v, (str, int, float, bool, list, dict)) else v
                for k, v in entry.items()
            }
            f.write(json.dumps(serializable_entry, ensure_ascii=False) + "\n")
    
    return str(filepath)


def show_system_monitor():
    """显示系统资源监控"""
    st.sidebar.subheader("系统资源监控")
    
    # CPU使用率
    cpu_percent = psutil.cpu_percent()
    st.sidebar.metric("CPU使用率", f"{cpu_percent}%")
    st.sidebar.progress(cpu_percent / 100)
    
    # 内存使用
    mem = psutil.virtual_memory()
    st.sidebar.metric("内存使用", 
                     f"{mem.used/1024/1024:.1f}MB / {mem.total/1024/1024:.1f}MB",
                     f"{mem.percent}%")
    
    # GPU信息（如果可用）
    if torch.cuda.is_available():
        gpu_info = get_gpu_memory()
        if gpu_info:
            st.sidebar.subheader("GPU显存信息")
            st.sidebar.metric(
                "总显存", 
                f"{gpu_info['total']:.1f} GB",
                f"已用: {gpu_info['used']:.1f} GB"
            )
            st.sidebar.progress(gpu_info['used'] / gpu_info['total'])

