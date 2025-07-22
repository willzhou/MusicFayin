# Author: Ningbo Wise Effects, Inc. (汇视创影) & Will Zhou
# License: Apache 2.0

"""
通用工具函数模块
包含与音乐生成相关的辅助函数和工具
"""

import os
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

def get_absolute_path(relative_path: str, project_root: Path, song_gen_dir: Path) -> Path:
    """将相对路径转换为绝对路径"""
    path = Path(relative_path)
    if relative_path.startswith("ckpt/"):
        return song_gen_dir / path.relative_to("ckpt/")
    return project_root / path

def parse_duration_to_seconds(duration_str: str) -> int:
    """将中文时长字符串转换为秒数"""
    try:
        # 处理"X分Y秒"格式
        if "分" in duration_str and "秒" in duration_str:
            minutes = int(re.search(r"(\d+)分", duration_str).group(1))
            seconds = int(re.search(r"(\d+)秒", duration_str).group(1))
            return minutes * 60 + seconds
        
        # 处理只有分钟的格式
        if "分" in duration_str:
            return int(duration_str.replace("分", "")) * 60
        
        # 处理纯秒数格式
        if "秒" in duration_str:
            return int(duration_str.replace("秒", ""))
        
        # 默认处理纯数字
        return int(duration_str)
    except Exception as e:
        raise ValueError(f"无效的时长格式: '{duration_str}'") from e

def calculate_section_timings(sections: List[str], total_seconds: int) -> Dict[str, int]:
    """计算每个段落的时长分配"""
    # 1. 验证所有段落是否定义
    for section in sections:
        if section not in MUSIC_SECTION_TEMPLATES:
            raise ValueError(f"未定义的段落类型: {section}")
    
    # 2. 计算总基准时长
    total_baseline = sum(
        MUSIC_SECTION_TEMPLATES[sec]["duration_avg"] 
        for sec in sections
    )
    
    # 3. 检查是否包含bridge段落
    has_bridge = "bridge" in sections
    
    # 4. 分配时长
    section_timings = {}
    remaining_seconds = total_seconds
    
    # 先分配verse和chorus段落
    for section in [sec for sec in sections if sec in ["verse", "chorus"]]:
        allocated = int(MUSIC_SECTION_TEMPLATES[section]["duration_avg"] * total_seconds / total_baseline)
        allocated = max(15, min(45, allocated))  # 限制15-45秒
        section_timings[section] = allocated
        remaining_seconds -= allocated
    
    # 如果有bridge段落，分配时长
    if has_bridge:
        bridge_seconds = int(MUSIC_SECTION_TEMPLATES["bridge"]["duration_avg"] * total_seconds / total_baseline)
        bridge_seconds = max(10, min(30, bridge_seconds))  # 限制10-30秒
        section_timings["bridge"] = bridge_seconds
        remaining_seconds -= bridge_seconds
    
    # 分配器乐段落
    instrumental_sections = [sec for sec in sections if sec not in ["verse", "chorus", "bridge"]]
    for section in instrumental_sections:
        allocated = int(MUSIC_SECTION_TEMPLATES[section]["duration_avg"] * total_seconds / total_baseline)
        allocated = max(5, min(30, allocated))  # 限制5-30秒
        section_timings[section] = allocated
        remaining_seconds -= allocated
    
    # 处理剩余时间（加到最后一个段落）
    if remaining_seconds > 0:
        last_section = sections[-1]
        section_timings[last_section] += remaining_seconds
    
    return section_timings

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

def save_jsonl(entries: List[Dict], filename: str, output_dir: Path) -> str:
    """保存JSONL文件"""
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
