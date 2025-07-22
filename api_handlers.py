# Author: Ningbo Wise Effects, Inc. (汇视创影) & Will Zhou
# License: Apache 2.0

"""
API处理模块
包含所有与外部API交互的函数
"""

import streamlit as st

import json
import re
import requests
from typing import Dict, Optional, List, Any
from config import EMOTIONS, GENRES, INSTRUMENTATIONS, TIMBRES, SINGER_GENDERS, DEFAULT_BPM
from config import MUSIC_SECTION_TEMPLATES

import plotly.express as px

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


def format_section_timing(sections: List[str], timings: Dict[str, int]) -> str:
    """格式化段落时长信息"""
    return "\n".join(
        f"- [{sec}]: {timings[sec]}秒" + 
        f" ({MUSIC_SECTION_TEMPLATES[sec]['description']})" 
        for sec in sections
    )

def calc_lines_from_seconds(seconds: int) -> str:
    """根据秒数计算建议行数"""
    min_lines = max(2, seconds // 5)  # 每行最多5秒
    max_lines = max(4, seconds // 3)  # 每行最少3秒
    return f"{min_lines}-{max_lines}行"


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

def display_duration_breakdown(sections: List[str], total_seconds: int):
    """显示时长分配饼图"""
    timings = calculate_section_timings(sections, total_seconds)
    
    fig = px.pie(
        names=[f"[{sec}]" for sec in sections],
        values=[timings[sec] for sec in sections],
        title=f"时长分配 (总计: {total_seconds}秒)",
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    st.plotly_chart(fig, use_container_width=True)

def generate_lyrics_with_duration(
    lyric_prompt: str,
    template: Dict[str, Any],
    song_length: str
) -> Optional[str]:
    """生成带时长控制的歌词"""
    try:
        # 解析总时长
        total_seconds = parse_duration_to_seconds(song_length)
        
        # 计算段落时长
        section_timings = calculate_section_timings(template["sections"], total_seconds)
        
        # 构建提示词
        prompt_lines = [
            f"请根据以下要求生成一首中文歌曲的完整歌词：\n"
            f"主题：{lyric_prompt}",
            f"""歌曲结构：
            {", ".join([f"[{section}]" for section in template["sections"]])}
            具体要求：
            1. 严格按照给定的结构标签分段
            2. 器乐段落([intro-*]/[outro-*])不需要填歌词
            3. 人声段落([verse]/[pre-chorus]/[chorus]/[bridge])必须包含歌词
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
            ...""",
            f"总时长：{song_length} ({total_seconds}秒)",
            "段落时长分配："
        ]
        
        # 添加各段落信息
        for section in template["sections"]:
            desc = MUSIC_SECTION_TEMPLATES[section]["description"]
            prompt_lines.append(f"- [{section}]: {section_timings[section]}秒 ({desc})")
        
        # 添加歌词行数要求
        prompt_lines.append("\n歌词要求：")
        prompt_lines.append(f"1. 主歌([verse]): 每段{calc_lines_from_seconds(section_timings['verse'])}行")
        prompt_lines.append(f"2. 副歌([chorus]): 每段{calc_lines_from_seconds(section_timings['chorus'])}行")
        
        # 只有模板包含bridge时才添加bridge要求
        if "bridge" in template["sections"]:
            prompt_lines.append(f"3. 桥段([bridge]): {calc_lines_from_seconds(section_timings['bridge'])}行")
        
        prompt_lines.append("4. 器乐段落不需要歌词")
        prompt_lines.append("5. 注意押韵和节奏")
        
        prompt = "\n".join(prompt_lines)
        
        return call_deepseek_api(prompt)
    except Exception as e:
        st.error(f"歌词生成失败: {str(e)}")
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
        "gender_suggestion": "从{sorted(SINGER_GENDERS)}中选择",
        "bpm_suggestion": "建议的BPM值(60-160之间的整数)",
        "tempo": "从['slow', 'medium', 'fast']中选择速度类型"
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
                temperature=0.5,  # 降低随机性确保稳定输出
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
                "gender_suggestion": analysis["gender_suggestion"],
                "bpm": int(analysis.get("bpm_suggestion", DEFAULT_BPM)),
                "tempo": analysis.get("tempo", "medium")
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
