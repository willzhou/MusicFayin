# Author: Ningbo Wise Effects, Inc. (汇视创影) & Will Zhou
# License: Apache 2.0

import streamlit as st
import json
import requests
from datetime import datetime
import os
import subprocess
import torch

from typing import Dict, Any, List, Tuple, Optional
import psutil
import sys
from pathlib import Path
import re

# import glob
# import time
# import threading
# from .config import DEEPSEEK_API_KEY, DEEPSEEK_URL

from config import EMOTIONS, SINGER_GENDERS, GENRES, INSTRUMENTATIONS, TIMBRES, AUTO_PROMPT_TYPES
from config import MUSIC_SECTION_TEMPLATES, STRUCTURE_TEMPLATES, SECTION_DEFINITIONS

from api_handlers import call_deepseek_api, analyze_lyrics
from func_utils import (
    get_absolute_path,
    parse_duration_to_seconds,
    calculate_section_timings,
    clean_generated_lyrics,
    get_gpu_memory,
    save_jsonl
)
from config import DEFAULT_BPM

# 在文件顶部添加项目根目录定义
PROJECT_ROOT = Path(__file__).parent  # 假设musicfayin.py现在放在SongGeneration的父目录
SONG_GEN_DIR = PROJECT_ROOT / "SongGeneration"
 
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
# 辅助函数
# ========================
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
                f"{analysis['instrumentation']}, the bpm is {analysis.get('bpm', DEFAULT_BPM)}"
            ),
            "gt_lyric": lyrics
        },
        {
            "idx": f"{prefix}_audioprompt_{timestamp}",
            "gt_lyric": lyrics,
            "prompt_audio_path": prompt_audio_path
        }
    ]
    
    return entries


# 在全局变量或session_state中添加运行状态标志
if 'running_process' not in st.session_state:
    st.session_state.running_process = None

def run_music_generation(jsonl_path: str, output_dir: str = "output", 
                        force_standard: bool = False, gen_type: str = ""):
    """执行音乐生成命令（带防止重复运行机制）"""
    
    # 检查是否已有进程在运行
    if st.session_state.running_process is not None:
        st.warning("⚠️ 已有生成任务正在运行，请等待完成")
        return
    
    try:
        # 获取显存信息
        gpu_info = get_gpu_memory()
        
        # 决定使用哪个脚本
        if force_standard:
            script = "generate.sh"
            st.info("已强制使用标准生成模式(generate.sh)")
        elif gpu_info and gpu_info["total"] >= 30:
            script = "generate.sh"
            st.info(f"检测到充足显存 ({gpu_info['total']:.1f}GB)，将使用标准生成模式")
        else:
            script = "generate_lowmem.sh"
            st.warning(f"显存不足30GB ({gpu_info['total']:.1f}GB if available)，使用低显存模式")
        
        # 使用绝对路径
        cmd = [
            "bash",
            str(SONG_GEN_DIR / script),
            str(SONG_GEN_DIR / "ckpt/songgeneration_base/"),
            str(get_absolute_path(jsonl_path)),
            str(get_absolute_path(output_dir))
        ]
        
        # 添加生成类型参数
        if gen_type:
            cmd.append(gen_type)
        
        # 显示执行命令
        st.code(" ".join(cmd), language="bash")

        # 显示状态信息
        status_text = st.empty()
        status_text.text("音乐生成中，请查看终端输出...")
        
        # 创建进程锁文件
        lock_file = Path(output_dir) / "generation.lock"
        if lock_file.exists():
            st.error("❌ 检测到已有生成进程运行，请删除lock文件后再试")
            return
            
        try:
            # 创建锁文件
            lock_file.touch()
            
            # 执行命令 - 直接输出到终端
            process = subprocess.Popen(
                cmd,
                cwd=str(SONG_GEN_DIR),
                stdout=sys.stdout,
                stderr=sys.stderr,
                universal_newlines=True
            )
            
            # 保存进程到session state
            st.session_state.running_process = process
            
            # 等待命令完成
            return_code = process.wait()
            
        finally:
            # 无论成功失败都清理锁文件和进程状态
            if lock_file.exists():
                lock_file.unlink()
            st.session_state.running_process = None
            status_text.empty()
        
        # 检查是否有生成的音频文件
        audio_files = list(Path(get_absolute_path(output_dir)).glob("audios/*.flac"))
        
        # 处理结果
        if audio_files:
            st.success("🎵 音乐生成完成！")
            display_generated_files(output_dir)
            
            if return_code != 0:
                st.warning(f"⚠️ 生成过程出现警告 (返回码: {return_code})")
        else:
            if return_code == 0:
                st.error("❌ 生成过程完成但未找到音频文件")
            else:
                st.error(f"❌ 生成失败 (返回码: {return_code})")
                
    except Exception as e:
        st.error(f"生成过程中发生错误: {str(e)}")
        if 'lock_file' in locals() and lock_file.exists():
            lock_file.unlink()
        st.session_state.running_process = None


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


import plotly.express as px
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
        lyric_prompt = st.text_area("输入歌词主题", "如果能重来")
        
        # 新增时长选择器
        length_min = st.slider(
            "歌曲时长（分钟）", 
            min_value=1, 
            max_value=10, 
            value=2,
            step=1
        )
        length_sec = st.slider(
            "歌曲时长（秒）", 
            min_value=0, 
            max_value=59, 
            value=30,
            step=5
        )
        song_length = f"{length_min}分{length_sec}秒"
        
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
        with st.spinner(f"正在生成{song_length}的歌词..."):
            template = STRUCTURE_TEMPLATES[selected_template]
            lyrics = generate_lyrics_with_duration(
                lyric_prompt=lyric_prompt,
                template=template,
                song_length=song_length
            )

            if lyrics:
                cleaned_lyrics = clean_generated_lyrics(lyrics)
                st.session_state.app_state['lyrics'] = cleaned_lyrics
                st.text_area("生成的歌词", cleaned_lyrics, height=200)
                
                # 显示时长分配
                total_seconds = parse_duration_to_seconds(song_length)
                st.subheader("时长分配详情")
                display_duration_breakdown(template["sections"], total_seconds)

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
        
        col1, col2, col3 = st.columns(3)
        
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
        with col3:
            # 添加BPM控制
            default_bpm = st.session_state.app_state['analysis_result'].get('bpm', DEFAULT_BPM)
            st.session_state.app_state['analysis_result']['bpm'] = st.slider(
                "BPM (每分钟节拍数)",
                min_value=60,
                max_value=160,
                value=default_bpm,
                step=1,
                help="建议值: 慢速60-80, 中速80-120, 快速120-160"
            )
            
            # 显示BPM对应的音乐类型
            bpm = st.session_state.app_state['analysis_result']['bpm']
            tempo_type = "slow" if bpm < 80 else ("fast" if bpm > 120 else "medium")
            st.markdown(f"**速度类型**: {tempo_type} ({bpm} BPM)")
            
            # 可视化BPM
            st.markdown("**节奏参考**:")
            if bpm < 80:
                st.markdown("🎵 慢速 (抒情、民谣)")
            elif 80 <= bpm <= 120:
                st.markdown("🎵 中速 (流行、摇滚)")
            else:
                st.markdown("🎵 快速 (舞曲、电子)")

    # 步骤4: 生成JSONL
    if st.session_state.app_state.get('analysis_result'):
        st.header("第四步: 生成配置")
        
        prefix = st.text_input("ID前缀", "sample_01")
        
        # 添加生成类型选择
        gen_type = st.radio(
            "生成类型",
            options=["", "bgm", "vocal"],
            format_func=lambda x: {
                "": "完整歌曲",
                "bgm": "纯音乐(BGM)", 
                "vocal": "纯人声"
            }[x],
            horizontal=True,
            help="选择生成完整歌曲、纯背景音乐或纯人声"
        )
        
        # 设置默认路径或用户选择的路径
        prompt_audio_path = "input/sample_prompt_audio.wav"  # 默认值
        
        # 添加音频文件选择器
        uploaded_file = st.file_uploader(
            "选择音频提示文件（默认：input/sample_prompt_audio.wav）",
            type=["wav","mp3","flac"],
            help="请选择用于音频提示的.wav文件"
        )

        if uploaded_file is not None:
            input_dir = get_absolute_path("input")
            input_dir.mkdir(parents=True, exist_ok=True)
            prompt_audio_path = input_dir / uploaded_file.name
            with open(prompt_audio_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"文件已保存: {prompt_audio_path}")
            prompt_audio_path = str(prompt_audio_path)  # 转换为字符串供后续使用

                
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
        
        # 添加强制使用generate.sh的选项
        col1, col2 = st.columns(2)
        with col1:
            force_standard = st.checkbox(
                "强制使用标准模式(generate.sh)", 
                help="忽略显存检测，强制使用标准生成模式(需要30GB以上显存)"
            )
        
        # 检查模型文件
        try:
            # 验证模型文件是否存在
            required_files = [
                SONG_GEN_DIR / "ckpt/songgeneration_base/config.yaml",
                SONG_GEN_DIR / "ckpt/songgeneration_base/model.pt",
                SONG_GEN_DIR / "ckpt/model_1rvq/model_2_fixed.safetensors",
                SONG_GEN_DIR / "ckpt/model_septoken/model_2.safetensors",
                SONG_GEN_DIR / "ckpt/prompt.pt"
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
                # # 准备生成命令
                jsonl_path = st.session_state.app_state['generated_jsonl']
                
                gpu_info = get_gpu_memory()
                if gpu_info:
                    st.info(f"当前GPU显存: {gpu_info['total']:.1f}GB (已用: {gpu_info['used']:.1f}GB)")
                    
                # 修改run_music_generation调用，传入force_standard参数
                run_music_generation(
                    jsonl_path=jsonl_path,
                    output_dir=output_dir,
                    force_standard=force_standard,
                    gen_type=gen_type  # 传入生成类型
                )     
                
                # 在生成音乐部分添加取消按钮
                if st.session_state.get('running_process'):
                    if st.button("取消生成"):
                        try:
                            st.session_state.running_process.terminate()
                            st.success("已发送终止信号，请等待进程结束")
                        except Exception as e:
                            st.error(f"终止失败: {str(e)}")
                                           
                # 创建进度条
                # progress_bar = st.progress(0)
                status_text = st.empty()
                status_text.text("音乐生成中...")

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


# ========================
# 主程序
# ========================
if __name__ == "__main__":

    # 在全局变量或session_state中添加运行状态标志
    if 'running_process' not in st.session_state:
        st.session_state.running_process = None

    os.environ.update({
        'PYTHONDONTWRITEBYTECODE': '0',
        'TRANSFORMERS_CACHE': str(SONG_GEN_DIR / "third_party/hub"),
        'HF_HOME': str(SONG_GEN_DIR / "third_party/hub"),
        'NCCL_HOME': '/usr/local/tccl',
        'PYTHONPATH': ":".join([
            str(SONG_GEN_DIR / "codeclm/tokenizer"),
            str(PROJECT_ROOT),
            str(SONG_GEN_DIR / "codeclm/tokenizer/Flow1dVAE"),
            os.getenv('PYTHONPATH', '')
        ])
    })
    Path(os.environ['TRANSFORMERS_CACHE']).mkdir(exist_ok=True)  # 确保目录存在

    # 确保必要的目录存在
    (PROJECT_ROOT / "output/audios").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "output/jsonl").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "input").mkdir(parents=True, exist_ok=True)
    
    # 设置并运行UI
    setup_ui()
