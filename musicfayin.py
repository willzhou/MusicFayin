# Author: Ningbo Wise Effects, Inc. (汇视创影) & Will Zhou
# License: Apache 2.0

import streamlit as st
import json
import requests
from datetime import datetime
import os
import subprocess
import time
import torch

from typing import Dict, Any, List, Tuple, Optional
import psutil
import sys
from pathlib import Path
import re
import glob

# 常量定义
DEEPSEEK_API_KEY = st.secrets['DEEPSEEK_API_KEY'] # 换成你自己的API KEY
DEEPSEEK_URL = st.secrets['DEEPSEEK_URL']

# “悲伤的”、“情绪的”、“愤怒的”、“快乐的”、“令人振奋的”、“强烈的”、“浪漫的”、“忧郁的”
EMOTIONS = [
    "sad", "emotional", "angry", "happy", 
    "uplifting", "intense", "romantic", "melancholic"
]

SINGER_GENDERS = ["male", "female"]

# “自动”、“中国传统”、“金属”、“雷鬼”、“中国戏曲”、“流行”、“电子”、“嘻哈”、“摇滚”、
# “爵士”、“蓝调”、“古典”、“说唱”、“乡村”、“经典摇滚”、“硬摇滚”、“民谣”、“灵魂乐”、
# “舞曲电子”、“乡村摇滚”、“舞曲、舞曲流行、浩室、流行”、“雷鬼”、“实验”、“舞曲、
# 流行”、“舞曲、深浩室、电子”、“韩国流行音乐”、“实验流行”、“流行朋克”、“摇滚乐”、
# “节奏布鲁斯”、“多样”、“流行摇滚”
GENRES = [
    'Auto', 'Chinese Tradition', 'Metal', 'Reggae', 'Chinese Opera',
    "pop", "electronic", "hip hop", "rock", "jazz", "blues", "classical",
    "rap", "country", "classic rock", "hard rock", "folk", "soul",
    "dance, electronic", "rockabilly", "dance, dancepop, house, pop",
    "reggae", "experimental", "dance, pop", "dance, deephouse, electronic",
    "k-pop", "experimental pop", "pop punk", "rock and roll", "R&B",
    "varies", "pop rock",
]

# “合成器与钢琴”，“钢琴与鼓”，“钢琴与合成器”，
# “合成器与鼓”，“钢琴与弦乐”，“吉他与鼓”，
# “吉他与钢琴”，“钢琴与低音提琴”，“钢琴与吉他”，
# “原声吉他与钢琴”，“原声吉他与合成器”，
# “合成器与吉他”，“钢琴与萨克斯风”，“萨克斯风与钢琴”，
# “钢琴与小提琴”，“电吉他与鼓”，“原声吉他与鼓”，
# “合成器”，“吉他与小提琴”，“吉他与口琴”，
# “合成器与原声吉他”，“节拍”，“钢琴”，
# “原声吉他与小提琴”，“铜管与钢琴”，“贝斯与鼓”，
# “小提琴”，“原声吉他与口琴”，“钢琴与大提琴”，
# “萨克斯风与小号”，“吉他与班卓琴”，“吉他与合成器”，
# “萨克斯风”，“小提琴与钢琴”，“合成器与贝斯”，
# “合成器与电吉他”，“电吉他与钢琴”，
# “节拍与钢琴”，“合成器与吉他”
INSTRUMENTATIONS = [
    "synthesizer and piano", "piano and drums", "piano and synthesizer",
    "synthesizer and drums", "piano and strings", "guitar and drums",
    "guitar and piano", "piano and double bass", "piano and guitar",
    "acoustic guitar and piano", "acoustic guitar and synthesizer",
    "synthesizer and guitar", "piano and saxophone", "saxophone and piano",
    "piano and violin", "electric guitar and drums", "acoustic guitar and drums",
    "synthesizer", "guitar and fiddle", "guitar and harmonica",
    "synthesizer and acoustic guitar", "beats", "piano",
    "acoustic guitar and fiddle", "brass and piano", "bass and drums",
    "violin", "acoustic guitar and harmonica", "piano and cello",
    "saxophone and trumpet", "guitar and banjo", "guitar and synthesizer",
    "saxophone", "violin and piano", "synthesizer and bass",
    "synthesizer and electric guitar", "electric guitar and piano",
    "beats and piano", "synthesizer and guitar"
]

# 音色：“黑暗的”、“明亮的”、“温暖的”、“岩石”、“变化的”、“柔和的”、“嗓音”
TIMBRES = ["dark", "bright", "warm", "rock", "varies", "soft", "vocal"]

AUTO_PROMPT_TYPES = ['Pop', 'R&B', 'Dance', 'Jazz', 'Folk', 'Rock', 
                    'Chinese Style', 'Chinese Tradition', 'Metal', 
                    'Reggae', 'Chinese Opera', 'Auto']


# 在常量定义部分添加音乐段落时长配置
MUSIC_SECTION_TEMPLATES = {
    # 纯器乐段落
    "intro-short": {
        "description": "前奏超短版(0-10秒)",
        "duration": "5-10秒",
        "lyric_required": False
    },
    "intro-medium": {
        "description": "前奏中等版(10-20秒)",
        "duration": "15-20秒",
        "lyric_required": False
    },
    "intro-long": {
        "description": "前奏完整版(20-30秒)",
        "duration": "20-30秒",
        "lyric_required": False
    },
    "outro-short": {
        "description": "尾奏超短版(0-10秒)", 
        "duration": "5-10秒",
        "lyric_required": False
    },
    "outro-medium": {
        "description": "尾奏中等版(10-20秒)",
        "duration": "15-20秒",
        "lyric_required": False
    },
    "outro-long": {
        "description": "尾奏完整版(20-30秒)",
        "duration": "20-30秒",
        "lyric_required": False
    },
    "inst-short": {
        "description": "间奏短版(5-10秒)",
        "duration": "5-10秒",
        "lyric_required": False
    },
    "inst-medium": {
        "description": "间奏中等版(10-20秒)",
        "duration": "15-20秒",
        "lyric_required": False
    },
    "inst-long": {
        "description": "间奏完整版(20-30秒)",
        "duration": "20-30秒",
        "lyric_required": False
    },
    "silence": {
        "description": "空白停顿(1-3秒)",
        "duration": "1-3秒",
        "lyric_required": False
    },
    
    # 人声段落
    "verse": {
        "description": "主歌段落(20-30秒)",
        "duration": "20-30秒",
        "lyric_required": True,
        "lines": "4-8行"
    },
    "chorus": {
        "description": "副歌(高潮段落)", 
        "duration": "20-30秒",
        "lyric_required": True,
        "lines": "4-8行"
    },
    "bridge": {
        "description": "过渡桥段",
        "duration": "15-25秒",
        "lyric_required": True,
        "lines": "2-4行"
    }
}


# - '[verse]'
# - '[chorus]'
# - '[bridge]'
# - '[intro-short]'
# - '[intro-medium]'
# - '[intro-long]'
# - '[outro-short]'
# - '[outro-medium]'
# - '[outro-long]'
# - '[inst-short]'
# - '[inst-medium]'
# - '[inst-long]'
# - '[silence]'

# 典型结构模板
# 音乐结构模板库 (36种)
STRUCTURE_TEMPLATES = {
    # 基础流行结构 (5种)
    "pop_basic": {
        "name": "流行基础结构",
        "sections": ["intro-medium", "verse", "chorus", "verse", "chorus", "outro-medium"]
    },
    "pop_with_bridge": {
        "name": "流行带桥段结构", 
        "sections": ["intro-medium", "verse", "chorus", "verse", "chorus", "bridge", "chorus", "outro-medium"]
    },
    "pop_with_prechorus": {
        "name": "流行带预副歌结构",
        "sections": ["intro-short", "verse", "verse", "chorus", "verse", "verse", "chorus", "outro-short"]
    },
    "pop_doublechorus": {
        "name": "流行双副歌结构",
        "sections": ["intro-short", "verse", "chorus", "chorus", "verse", "chorus", "chorus", "outro-short"]
    },
    "pop_postchorus": {
        "name": "流行带后副歌结构",
        "sections": ["intro-medium", "verse", "verse", "chorus", "inst-short", "verse", "verse", "chorus", "inst-short", "outro-medium"]
    },
    
    # 摇滚/金属结构 (8种)
    "rock_classic": {
        "name": "经典摇滚结构",
        "sections": ["intro-long", "verse", "chorus", "verse", "chorus", "inst-long", "chorus", "outro-long"]
    },
    "metal_progressive": {
        "name": "前卫金属结构",
        "sections": ["intro-long", "verse", "bridge", "chorus", "inst-long", "verse", "bridge", "chorus", "inst-long", "outro-long"]
    },
    "punk": {
        "name": "朋克结构",
        "sections": ["intro-short", "verse", "chorus", "verse", "chorus", "bridge", "chorus", "outro-short"]
    },
    "hardrock": {
        "name": "硬摇滚结构",
        "sections": ["intro-long", "verse", "chorus", "verse", "chorus", "inst-long", "inst-long", "chorus", "outro-long"]
    },
    "rock_ballad": {
        "name": "摇滚抒情曲结构",
        "sections": ["intro-long", "verse", "verse", "chorus", "inst-long", "verse", "chorus", "outro-long"]
    },
    "metalcore": {
        "name": "金属核结构",
        "sections": ["intro-short", "verse", "chorus", "verse", "chorus", "inst-short", "chorus", "outro-short"]
    },
    "blues_rock": {
        "name": "蓝调摇滚结构",
        "sections": ["intro-medium", "verse", "verse", "chorus", "inst-medium", "verse", "chorus", "outro-medium"]
    },
    "rock_instrumental": {
        "name": "摇滚器乐曲结构",
        "sections": ["intro-long", "inst-long", "inst-medium", "inst-long", "inst-medium", "inst-long", "inst-long", "outro-long"]
    },
    
    # 电子音乐结构 (7种)
    "edm_builddrop": {
        "name": "EDM构建-高潮结构",
        "sections": ["intro-long", "inst-medium", "inst-short", "inst-medium", "inst-medium", "inst-short", "outro-medium"]
    },
    "house": {
        "name": "浩室结构",
        "sections": ["intro-long", "inst-long", "inst-medium", "inst-long", "inst-medium", "inst-short", "outro-long"]
    },
    "trance": {
        "name": "迷幻结构",
        "sections": ["intro-long", "inst-long", "inst-medium", "inst-short", "inst-medium", "inst-medium", "inst-short", "outro-long"]
    },
    "dubstep": {
        "name": "回响贝斯结构",
        "sections": ["intro-medium", "verse", "inst-short", "inst-medium", "verse", "inst-short", "outro-short"]
    },
    "techno": {
        "name": "科技结构",
        "sections": ["intro-long", "inst-long", "inst-medium", "inst-long", "inst-short", "inst-long", "outro-long"]
    },
    "drum_bass": {
        "name": "鼓打贝斯结构",
        "sections": ["intro-medium", "inst-short", "verse", "inst-short", "inst-medium", "inst-short", "outro-medium"]
    },
    "ambient": {
        "name": "氛围结构",
        "sections": ["intro-long", "inst-long", "inst-medium", "inst-short", "inst-medium", "outro-long"]
    },
    
    # 嘻哈/说唱结构 (5种)
    "hiphop_classic": {
        "name": "经典嘻哈结构",
        "sections": ["intro-short", "verse", "chorus", "verse", "chorus", "bridge", "verse", "chorus", "outro-short"]
    },
    "trap": {
        "name": "陷阱结构",
        "sections": ["intro-short", "verse", "chorus", "verse", "chorus", "inst-short", "chorus", "outro-short"]
    },
    "rap_storytelling": {
        "name": "叙事说唱结构",
        "sections": ["intro-medium", "verse", "chorus", "verse", "chorus", "verse", "chorus", "outro-medium"]
    },
    "hiphop_jazzy": {
        "name": "爵士嘻哈结构",
        "sections": ["intro-medium", "verse", "chorus", "verse", "chorus", "inst-medium", "chorus", "outro-medium"]
    },
    "rap_battle": {
        "name": "对战说唱结构",
        "sections": ["intro-short", "verse", "verse", "verse", "verse", "outro-short"]
    },
    
    # 中国传统/民族结构 (6种)
    "chinese_folk": {
        "name": "中国民谣结构",
        "sections": ["intro-long", "verse", "inst-medium", "verse", "inst-medium", "outro-long"]
    },
    "chinese_opera": {
        "name": "戏曲结构",
        "sections": ["intro-long", "verse", "inst-short", "verse", "inst-medium", "inst-short", "verse", "outro-long"]
    },
    "guqin": {
        "name": "古琴曲结构",
        "sections": ["intro-long", "inst-long", "inst-medium", "inst-long", "inst-medium", "outro-long"]
    },
    "ethnic_fusion": {
        "name": "民族融合结构",
        "sections": ["intro-long", "verse", "chorus", "verse", "chorus", "inst-long", "outro-long"]
    },
    "chinese_pop": {
        "name": "中国流行结构",
        "sections": ["intro-medium", "verse", "verse", "chorus", "inst-medium", "verse", "verse", "chorus", "outro-medium"]
    },
    "mongolian_throat": {
        "name": "蒙古呼麦结构",
        "sections": ["intro-long", "verse", "inst-long", "inst-short", "verse", "inst-short", "outro-long"]
    },
    
    # 爵士/蓝调结构 (5种)
    "jazz_standard": {
        "name": "爵士标准结构",
        "sections": ["intro-medium", "inst-medium", "inst-long", "inst-medium", "inst-medium", "outro-medium"]
    },
    "blues_12bar": {
        "name": "12小节蓝调结构",
        "sections": ["intro-short", "verse", "verse", "verse", "inst-medium", "verse", "outro-short"]
    },
    "jazz_fusion": {
        "name": "爵士融合结构",
        "sections": ["intro-long", "inst-medium", "inst-long", "inst-medium", "inst-short", "inst-medium", "outro-long"]
    },
    "bebop": {
        "name": "比博普结构",
        "sections": ["intro-short", "inst-short", "inst-medium", "inst-long", "inst-medium", "inst-short", "outro-short"]
    },
    "jazz_ballad": {
        "name": "爵士抒情曲结构",
        "sections": ["intro-long", "inst-long", "inst-medium", "inst-long", "outro-long"]
    }
}

# 特殊段落说明
SECTION_DEFINITIONS = {
    "skank": "雷鬼特有的反拍节奏段落",
    "guitar-solo": "吉他独奏部分",
    "post-chorus": "副歌后的记忆点段落",
    "drop": "电子舞曲的高潮部分",
    "head": "爵士乐主题段落",
    "ad-lib": "即兴演唱部分",
    "12bar": "12小节蓝调进行",
    "build-up": "电子乐中的情绪构建段落",
    "breakdown": "电子乐中的分解段落",
    "call-response": "非洲音乐中的呼应段落",
    "copla": "弗拉门戈中的歌唱段落",
    "falseta": "弗拉门戈吉他独奏段落"
}

# 初始化session state
if 'app_state' not in st.session_state:
    st.session_state.app_state = {
        'lyrics': None,
        'analysis_result': None,
        'singer_gender': SINGER_GENDERS[0],
        'generated_jsonl': None,
        'music_files': []
    }

    
# ========================
# 应用界面函数
# ========================
def call_deepseek_api(prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> str:
    """调用DeepSeek API生成歌词"""
    headers = {
        "Authorization": f"Bearer {st.secrets['DEEPSEEK_API_KEY']}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    try:
        response = requests.post(st.secrets['DEEPSEEK_URL'], headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        st.error(f"API调用失败: {str(e)}")
        return None

def analyze_lyrics(lyrics: str) -> Dict[str, str]:
    """分析歌词并返回音乐参数建议
    
    Args:
        lyrics: 要分析的歌词文本
        
    Returns:
        包含音乐参数的字典，格式为:
        {
            "emotion": str,
            "genre": str,
            "instrumentation": str,
            "timbre": str,
            "gender_suggestion": str
        }
        
    Raises:
        ValueError: 当API返回无效结果时
    """
    prompt = f"""请严格按以下JSON格式分析歌词特征：
    {lyrics}
    
    返回格式必须为：
    {{
        "emotion": "从{sorted(EMOTIONS)}中选择",
        "genre": "从{sorted(GENRES)}中选择1-2种",
        "instrumentation": "从{sorted(INSTRUMENTATIONS)}中选择",
        "timbre": "从{sorted(TIMBRES)}中选择",
        "gender_suggestion": "从{sorted(SINGER_GENDERS)}中选择"
    }}
    
    注意：
    1. 必须返回合法JSON
    2. 所有值必须来自给定选项
    3. 不要包含任何额外文字"""
    
    max_retries = 3
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            result = call_deepseek_api(
                prompt,
                temperature=0.1,  # 降低随机性确保稳定输出
                max_tokens=500
            )
            
            if not result:
                raise ValueError("API返回空结果")
            
            # 预处理API响应
            cleaned_result = result.strip()
            
            # 处理可能的代码块标记
            if cleaned_result.startswith("```json"):
                cleaned_result = cleaned_result[7:].strip()
            if cleaned_result.endswith("```"):
                cleaned_result = cleaned_result[:-3].strip()
            
            # 解析JSON
            analysis = json.loads(cleaned_result)
            
            # 验证结果
            required_keys = ["emotion", "genre", "instrumentation", 
                           "timbre", "gender_suggestion"]
            if not all(key in analysis for key in required_keys):
                raise ValueError(f"缺少必要字段，应有: {required_keys}")
            
            # 验证字段值有效性
            if analysis["emotion"] not in EMOTIONS:
                raise ValueError(f"无效情绪: {analysis['emotion']}，应为: {EMOTIONS}")
                
            if not any(g in analysis["genre"] for g in GENRES):
                raise ValueError(f"无效类型: {analysis['genre']}，应为: {GENRES}")
                
            if analysis["instrumentation"] not in INSTRUMENTATIONS:
                raise ValueError(f"无效乐器组合: {analysis['instrumentation']}，应为: {INSTRUMENTATIONS}")
                
            if analysis["timbre"] not in TIMBRES:
                raise ValueError(f"无效音色: {analysis['timbre']}，应为: {TIMBRES}")
                
            if analysis["gender_suggestion"] not in SINGER_GENDERS:
                raise ValueError(f"无效性别建议: {analysis['gender_suggestion']}，应为: {SINGER_GENDERS}")
            
            # 返回验证通过的结果
            return {
                "emotion": analysis["emotion"],
                "genre": analysis["genre"],
                "instrumentation": analysis["instrumentation"],
                "timbre": analysis["timbre"],
                "gender_suggestion": analysis["gender_suggestion"]
            }
            
        except json.JSONDecodeError as e:
            last_exception = f"JSON解析失败: {str(e)}，原始响应: {result}"
            st.warning(f"尝试 {attempt + 1}/{max_retries}: {last_exception}")
            continue
            
        except ValueError as e:
            last_exception = str(e)
            st.warning(f"尝试 {attempt + 1}/{max_retries}: {last_exception}")
            continue
            
        except Exception as e:
            last_exception = str(e)
            st.warning(f"尝试 {attempt + 1}/{max_retries}: 未知错误: {last_exception}")
            continue
    
    # 所有重试都失败后的处理
    error_msg = f"歌词分析失败，将使用默认参数。最后错误: {last_exception}"
    st.error(error_msg)
    
    # 返回保守默认值
    return {
        "emotion": "emotional",
        "genre": "pop",
        "instrumentation": "piano and strings",
        "timbre": "warm",
        "gender_suggestion": "female"
    }


# ========================
# 辅助函数
# ========================
def generate_jsonl_entries(prefix: str, lyrics: str, analysis: Dict[str, Any], prompt_audio_path: str = "input/sample_prompt_audio.wav") -> List[Dict]:
    """生成所有JSONL条目"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    entries = [
        {
            "idx": f"{prefix}_autoprompt_{timestamp}",
            "gt_lyric": lyrics,
            "auto_prompt_audio_type": "Auto"
        },
        {
            "idx": f"{prefix}_noprompt_{timestamp}",
            "gt_lyric": lyrics
        },
        {
            "idx": f"{prefix}_textprompt_{timestamp}",
            "descriptions": (
                f"{analysis['gender_suggestion']}, {analysis['timbre']}, "
                f"{analysis['genre']}, {analysis['emotion']}, "
                f"{analysis['instrumentation']}, the bpm is 125"
            ),
            "gt_lyric": lyrics
        },
        {
            "idx": f"{prefix}_audioprompt_{timestamp}",
            "gt_lyric": lyrics,
            "prompt_audio_path": prompt_audio_path  # 使用传入的路径
        }
    ]
    
    return entries

def save_jsonl(entries: List[Dict], filename: str) -> str:
    """保存JSONL文件"""
    os.makedirs("output", exist_ok=True)
    filepath = os.path.join("output", filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    return filepath

def run_music_generation(jsonl_path: str, output_dir: str = "output"):
    """执行音乐生成命令并处理输出"""
    cmd = [
        "bash",
        "generate_lowmem.sh",
        "ckpt/songgeneration_base/",
        jsonl_path,
        output_dir
    ]
    
    # 创建进度显示
    progress_bar = st.progress(0)
    status_text = st.empty()
    output_container = st.expander("生成日志", expanded=True)
    
    # 执行命令
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    # 实时显示输出
    full_output = ""
    while True:
        line = process.stdout.readline()
        if line == '' and process.poll() is not None:
            break
        if line:
            full_output += line
            output_container.code(full_output, language="bash")
            
            # 更新进度
            if "Generating:" in line:
                progress_bar.progress(min(100, progress_bar.progress_value + 20))
    
    # 处理结果
    if process.returncode == 0:
        st.success("🎵 音乐生成完成！")
        display_generated_files(output_dir)
    else:
        st.error(f"❌ 生成失败 (返回码: {process.returncode})")
        st.text(full_output)

def display_generated_files(output_dir: str):
    """显示生成的音乐文件"""
    audio_files = list(Path(output_dir).glob("audios/*.flac"))
    if not audio_files:
        st.warning("未找到生成的音频文件")
        return
    
    st.subheader("生成的音乐")
    for audio_file in sorted(audio_files):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.audio(str(audio_file))
        with col2:
            with open(audio_file, "rb") as f:
                st.download_button(
                    "下载",
                    data=f.read(),
                    file_name=audio_file.name,
                    mime="audio/flac"
                )


def clean_generated_lyrics(raw_lyrics: str) -> str:
    """
    Format raw lyrics into the specified structure:
    - Sections separated by ' ; '
    - Each line in vocal sections ends with a period
    - No spaces around periods
    - Instrumental sections without content
    
    Args:
        raw_lyrics: Raw lyrics text with section markers
        
    Returns:
        Formatted string with strict section formatting
    """
    sections = []
    current_section = None
    current_lines = []
    
    for line in raw_lyrics.split('\n'):
        line = line.strip()
        if not line:
            continue
        
        # Detect section headers like [verse]
        section_match = re.match(r'^\[([a-z\-]+)\]$', line)
        if section_match:
            if current_section is not None:
                sections.append((current_section, current_lines))
            current_section = section_match.group(1)
            current_lines = []
        elif current_section is not None:
            # Clean lyric line and add to current section
            cleaned_line = line.replace(' ', '.').replace('，', '.').replace('。', '.').strip('. ')
            if cleaned_line:
                current_lines.append(cleaned_line)
    
    # Add the final section if exists
    if current_section is not None:
        sections.append((current_section, current_lines))
    
    # Format each section according to its type
    formatted_sections = []
    for section_type, lines in sections:
        if section_type in ['verse', 'chorus', 'bridge']:
            # Vocal sections: join lines with periods
            content = ".".join(line.rstrip('.') for line in lines if line)
            formatted = f"[{section_type}] {content}" if content else f"[{section_type}]"
        else:
            # Instrumental/other sections: no content
            formatted = f"[{section_type}]"
        formatted_sections.append(formatted)
    
    return " ; ".join(formatted_sections)

def replace_chinese_punctuation(text):
    """替换中文标点为英文标点"""
    punctuation_map = {
        '，': ',', '。': '.', '、': ',', '；': ';', '：': ':',
        '？': '?', '！': '!', '「': '"', '」': '"', '『': '"',
        '』': '"', '（': '(', '）': ')', '《': '"', '》': '"'
    }
    
    # 逐个字符替换
    result = []
    for char in text:
        if char in punctuation_map:
            # 在标点前后添加空格确保分割
            result.append(f" {punctuation_map[char]} ")
        else:
            result.append(char)
    
    # 合并并标准化空格
    return re.sub(r'\s+', ' ', "".join(result)).strip()

# 典型结构模板
# ========================
# Streamlit 界面
# ========================
def setup_ui():
    """设置Streamlit用户界面"""
    st.set_page_config(page_title="MusicFayIn", layout="wide")
    st.title("🎵 MusicFayIn 人工智能音乐生成系统")
    
    # 步骤1: 歌词生成
    st.header("第一步: 生成歌词")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        lyric_prompt = st.text_area("输入歌词主题", "孩子们的三色枣糕")
        
        # 结构模板选择
        selected_template = st.selectbox(
            "选择歌曲结构模板",
            options=list(STRUCTURE_TEMPLATES.keys()),
            format_func=lambda x: STRUCTURE_TEMPLATES[x]["name"]
        )
        
        # 显示选中的模板结构
        if selected_template:
            template = STRUCTURE_TEMPLATES[selected_template]
            st.markdown("**当前结构:**")
            for i, section in enumerate(template["sections"]):
                info = MUSIC_SECTION_TEMPLATES[section]
                st.markdown(f"{i+1}. `[{section}]` - {info['description']}")
                
    with col2:
        # 显示段落时长说明
        st.markdown("### 🎵 音乐段落时长规范")
        st.markdown("""
        **一、纯器乐段落（不含歌词）:**
        - `[intro-short]`: 前奏超短版 (0-10秒)
        - `[outro-short]`: 尾奏超短版 (0-10秒)
        - `[intro-medium]`: 前奏中等版 (10-20秒)
        - `[outro-medium]`: 尾奏中等版 (10-20秒)
        
        **二、人声段落（必须含歌词）:**
        - `[verse]`: 主歌 (20-30秒, 4-8行)
        - `[chorus]`: 副歌 (高潮段落, 20-30秒)
        - `[bridge]`: 过渡桥段 (15-25秒, 2-4行)
        """)
        
        st.markdown("### 📝 使用建议")
        st.markdown("""
        - 器乐段落严格按秒数范围控制
        - 人声段落通过歌词行数控制时长
        - 前奏/尾奏若超过20秒，可组合使用:
          `[intro-medium][intro-short]` ≈ 25秒
        """)
    # 生成歌词按钮
    if st.button("生成歌词"):
        with st.spinner("正在生成歌词..."):
            # 获取选中的模板结构
            template = STRUCTURE_TEMPLATES[selected_template]
            
            # 构建详细的提示词
            prompt = f"""请根据以下要求生成一首中文歌曲的完整歌词：
                        
            主题：{lyric_prompt}
            歌曲结构：
            {", ".join([f"[{section}]" for section in template["sections"]])}
            具体要求：
            1. 严格按照给定的结构标签分段
            2. 器乐段落([intro-*]/[outro-*])不需要填歌词
            3. 人声段落([verse]/[chorus]/[bridge])必须包含歌词
            4. 主歌([verse])每段4-8行
            5. 副歌([chorus])要突出高潮部分
            6. 桥段([bridge])2-4行
            7. 整体要有押韵和节奏感
            8. 不要包含歌曲标题
            9. 不要包含韵脚分析等额外说明
            返回格式示例：
            [intro-medium]
            [verse]
            第一行歌词
            第二行歌词
            ...
            [chorus]
            副歌第一行
            副歌第二行
            ...
            """
            lyrics = call_deepseek_api(prompt)
            if lyrics:
                cleaned_lyrics = clean_generated_lyrics(lyrics)
                st.session_state.app_state['lyrics'] = cleaned_lyrics
                st.text_area("生成的歌词", cleaned_lyrics, height=400)
                
                # 自动分析歌词参数
                with st.spinner("正在分析歌词特征..."):
                    analysis = analyze_lyrics(cleaned_lyrics)
                    if analysis:
                        st.session_state.app_state['analysis_result'] = analysis
                        st.success("歌词分析完成！")

    # 步骤2: 分析歌词
    if st.session_state.app_state.get('lyrics'):
        st.header("第二步: 分析歌词")
        
        if st.button("分析歌词参数"):
            with st.spinner("正在分析歌词..."):
                analysis = analyze_lyrics(st.session_state.app_state['lyrics'])
                if analysis:
                    st.session_state.app_state['analysis_result'] = analysis
                    st.json(analysis)

    # 步骤3: 参数调整
    if st.session_state.app_state.get('analysis_result'):
        st.header("第三步: 参数调整")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 使用分析结果或提供默认值
            default_gender = st.session_state.app_state['analysis_result'].get(
                'gender_suggestion', SINGER_GENDERS[0]
            )
            st.session_state.app_state['singer_gender'] = st.radio(
                "歌手性别", SINGER_GENDERS,
                index=SINGER_GENDERS.index(default_gender),
                horizontal=True
            )
            
            default_emotion = st.session_state.app_state['analysis_result'].get(
                'emotion', EMOTIONS[0]
            )
            st.session_state.app_state['analysis_result']['emotion'] = st.selectbox(
                "情绪", EMOTIONS,
                index=EMOTIONS.index(default_emotion)
            )
            
            default_timbre = st.session_state.app_state['analysis_result'].get(
                'timbre', TIMBRES[0]
            )
            st.session_state.app_state['analysis_result']['timbre'] = st.selectbox(
                "音色", TIMBRES,
                index=TIMBRES.index(default_timbre)
            )
        
        with col2:
            default_genre = st.session_state.app_state['analysis_result'].get(
                'genre', GENRES[0]
            )
            st.session_state.app_state['analysis_result']['genre'] = st.selectbox(
                "歌曲类型", GENRES,
                index=GENRES.index(default_genre.split(",")[0])  # 取第一个类型
            )
            
            default_instrument = st.session_state.app_state['analysis_result'].get(
                'instrumentation', INSTRUMENTATIONS[0]
            )
            st.session_state.app_state['analysis_result']['instrumentation'] = st.selectbox(
                "乐器组合", INSTRUMENTATIONS,
                index=INSTRUMENTATIONS.index(default_instrument)
            )

    # 步骤4: 生成JSONL
    if st.session_state.app_state.get('analysis_result'):
        st.header("第四步: 生成配置")
        
        prefix = st.text_input("ID前缀", "sample_01")
        
        # 添加音频文件选择器
        uploaded_file = st.file_uploader(
            "选择音频提示文件（默认：input/sample_prompt_audio.wav）",
            type=["wav","mp3","flac"],
            help="请选择用于音频提示的.wav文件"
        )
        # 设置默认路径或用户选择的路径
        prompt_audio_path = "input/sample_prompt_audio.wav"  # 默认值
        if uploaded_file is not None:
            # 保存上传的文件到input目录
            os.makedirs("input", exist_ok=True)
            prompt_audio_path = os.path.join("input", uploaded_file.name)
            with open(prompt_audio_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"文件已保存: {prompt_audio_path}")
        
        if st.button("生成JSONL配置"):
            entries = generate_jsonl_entries(
                prefix,
                st.session_state.app_state['lyrics'],
                st.session_state.app_state['analysis_result'],
                prompt_audio_path  # 传入自定义的音频路径
            )
            
            filename = f"{prefix}_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
            filepath = save_jsonl(entries, filename)
            
            st.session_state.app_state['generated_jsonl'] = filepath
            st.success(f"JSONL文件已生成: {filepath}")
            
            for entry in entries:
                st.json(entry)

    # 步骤5: 生成音乐
    if st.session_state.app_state.get('generated_jsonl'):
        st.header("第五步: 生成音乐")
        
        # 输出目录设置
        output_dir = st.text_input("输出目录", "output")
        
        # 检查模型文件
        try:
            # 验证模型文件是否存在
            required_files = [
                "ckpt/songgeneration_base/config.yaml",
                "ckpt/songgeneration_base/model.pt",
                "ckpt/model_1rvq/model_2_fixed.safetensors",
                "ckpt/model_septoken/model_2.safetensors",
                "ckpt/prompt.pt"
            ]
            
            missing_files = [f for f in required_files if not os.path.exists(f)]
            
            if missing_files:
                raise FileNotFoundError(
                    f"缺少必要的模型文件:\n{chr(10).join(missing_files)}\n"
                    "请确保文件结构如下:\n"
                    "ckpt/\n"
                    "├── model_1rvq/\n"
                    "│   └── model_2_fixed.safetensors\n"
                    "├── model_septoken/\n"
                    "│   └── model_2.safetensors\n"
                    "├── prompt.pt\n"
                    "└── songgeneration_base/\n"
                    "    ├── config.yaml\n"
                    "    └── model.pt"
                )
            
            st.success("✅ 模型文件验证通过")
            
            if st.button("运行音乐生成"):
                # 准备生成命令
                jsonl_path = st.session_state.app_state['generated_jsonl']
                cmd = [
                    "bash", 
                    "generate_lowmem.sh",
                    "ckpt/songgeneration_base/",
                    jsonl_path,
                    output_dir
                ]

                # 显示执行的命令
                st.code(" ".join(cmd), language="bash")
                
                # 创建进度条
                progress_bar = st.progress(0)
                status_text = st.empty()
                status_text.text("音乐生成中...")
                
                try:
                    # 执行生成命令，将输出重定向到终端
                    process = subprocess.Popen(
                        cmd,
                        stdout=sys.stdout,  # 输出到终端
                        stderr=sys.stderr,  # 错误输出到终端
                        universal_newlines=True
                    )
                    
                    # 简单进度模拟
                    for i in range(10):
                        time.sleep(0.5)  # 模拟处理时间
                        progress_bar.progress((i + 1) * 10)
                    
                    # 等待命令完成
                    return_code = process.wait()
                    
                    # 无论返回码如何，都检查是否有音频文件生成
                    audio_files = glob.glob(f"{output_dir}/audios/*.flac")
                    
                    if audio_files:
                        status_text.empty()
                        st.subheader("生成的音乐文件")
                        for audio_file in sorted(audio_files):
                            st.audio(audio_file)
                            st.download_button(
                                f"下载 {os.path.basename(audio_file)}",
                                data=open(audio_file, "rb").read(),
                                file_name=os.path.basename(audio_file),
                                mime="audio/flac"
                            )
                        
                        # 即使有文件生成，也根据返回码显示不同状态
                        if return_code == 0:
                            st.success("🎵 音乐生成完成！")
                        else:
                            st.warning(f"⚠️ 生成过程出现错误 (返回码: {return_code})，但已生成部分音频文件")
                            st.info("请查看终端输出获取详细错误信息")
                    else:
                        status_text.empty()
                        if return_code == 0:
                            st.error("❌ 生成过程完成但未找到音频文件")
                        else:
                            st.error(f"❌ 生成失败 (返回码: {return_code})")
                        st.info("请查看终端输出获取详细错误信息")
                        
                except Exception as e:
                    status_text.empty()
                    st.error(f"❌ 生成过程中发生错误: {str(e)}")
                    st.info("请查看终端输出获取详细错误信息")
                    
                    # 即使出错也检查是否有部分文件生成
                    audio_files = glob.glob(f"{output_dir}/audios/*.flac")
                    if audio_files:
                        st.subheader("部分生成的音乐文件")
                        for audio_file in sorted(audio_files):
                            st.audio(audio_file)
                            st.download_button(
                                f"下载 {os.path.basename(audio_file)}",
                                data=open(audio_file, "rb").read(),
                                file_name=os.path.basename(audio_file),
                                mime="audio/flac"
                            )
                                
        except FileNotFoundError as e:
            st.error(str(e))
            st.warning("请确保所有模型文件已正确下载并放置在指定位置")
        except Exception as e:
            st.error(f"生成过程中发生错误: {str(e)}")


    # 侧边栏说明
    st.sidebar.markdown("""
    ### 使用流程
    1. **生成歌词**：输入主题生成歌词
    2. **分析歌词**：自动分析音乐参数
    3. **调整参数**：根据需要修改参数
    4. **生成配置**：创建JSONL配置文件
    5. **生成音乐**：运行生成脚本

    ### 生成选项
    - 自动生成 (autoprompt)
    - 无提示生成 (noprompt)
    - 文本提示生成 (textprompt)
    - 音频提示生成 (audioprompt)
    """)

    # 系统监控
    if st.sidebar.checkbox("显示系统资源"):
        show_system_monitor()

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
    
    # 磁盘空间
    disk = psutil.disk_usage('/')
    st.sidebar.metric("磁盘空间", 
                     f"{disk.used/1024/1024:.1f}MB / {disk.total/1024/1024:.1f}MB",
                     f"{disk.percent}%")
    
    # GPU信息（如果可用）
    if torch.cuda.is_available():
        st.sidebar.subheader("GPU信息")
        for i in range(torch.cuda.device_count()):
            mem = torch.cuda.mem_get_info(i)
            total = mem[1] / 1024**3
            free = mem[0] / 1024**3
            used = total - free
            st.sidebar.metric(
                f"GPU {i} ({torch.cuda.get_device_name(i)})",
                f"{used:.1f}GB / {total:.1f}GB",
                f"{used/total*100:.1f}%"
            )

# ========================
# 主程序
# ========================
if __name__ == "__main__":
    os.environ.update({
        'USER': 'root',
        'PYTHONDONTWRITEBYTECODE': '1',
        'TRANSFORMERS_CACHE': str(Path.cwd() / "third_party/hub"),
        'NCCL_HOME': '/usr/local/tccl',
        'PYTHONPATH': ":".join([
            str(Path.cwd() / "codeclm/tokenizer"),
            str(Path.cwd()),
            str(Path.cwd() / "codeclm/tokenizer/Flow1dVAE"),
            os.getenv('PYTHONPATH', '')
        ])
    })
    Path(os.environ['TRANSFORMERS_CACHE']).mkdir(exist_ok=True)  # 确保目录存在

    # 确保必要的目录存在
    os.makedirs("output/audios", exist_ok=True)
    os.makedirs("output/jsonl", exist_ok=True)
    os.makedirs("input", exist_ok=True)
    
    # 设置并运行UI
    setup_ui()
