# Author: Ningbo Wise Effects, Inc. (汇视创影) & Will Zhou
# License: Apache 2.0

import streamlit as st

from datetime import datetime
import os
import subprocess

from typing import Dict, Any, List, Tuple, Optional
import sys
from pathlib import Path

from config import EMOTIONS, SINGER_GENDERS, GENRES, INSTRUMENTATIONS, TIMBRES
from config import MUSIC_SECTION_TEMPLATES, STRUCTURE_TEMPLATES
from config import DEFAULT_MODEL, SUPPORTED_MODELS

from api_handlers import (
    generate_lyrics_with_duration, 
    analyze_lyrics,
    parse_duration_to_seconds,
    display_duration_breakdown
)
from func_utils import (
    get_absolute_path,
    clean_generated_lyrics,
    get_gpu_memory,
    show_system_monitor,
    save_jsonl
)
from config import DEFAULT_BPM

# 在文件顶部添加项目根目录定义
PROJECT_ROOT = Path(__file__).parent  # 假设musicfayin.py现在放在SongGeneration的父目录
SONG_GEN_DIR = PROJECT_ROOT / "SongGeneration"

st.set_page_config(page_title="MusicFayIn", layout="wide")

# 初始化session state
if 'app_state' not in st.session_state:
    st.session_state.app_state = {
        'lyrics': None,
        'analysis_result': None,
        'singer_gender': SINGER_GENDERS[0],
        'generated_jsonl': None,
        'music_files': []
    }

def generate_jsonl_entries(prefix: str, lyrics: str, analysis: Dict[str, Any], 
                         prompt_audio_path: str = "input/sample_prompt_audio.wav") -> List[Dict]:
    """生成所有JSONL条目，并添加类型标识"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    entries = [
        {
            "idx": f"{prefix}_autoprompt_{timestamp}",
            "gt_lyric": lyrics,
            "auto_prompt_audio_type": "Auto",
            "bpm": analysis.get('bpm', DEFAULT_BPM),
            "entry_type": "autoprompt"  # 添加类型标识
        },
        {
            "idx": f"{prefix}_noprompt_{timestamp}",
            "gt_lyric": lyrics,
            "bpm": analysis.get('bpm', DEFAULT_BPM),
            "entry_type": "noprompt"  # 添加类型标识
        },
        {
            "idx": f"{prefix}_textprompt_{timestamp}",
            "descriptions": (
                f"{analysis['gender_suggestion']}, {analysis['timbre']}, "
                f"{analysis['genre']}, {analysis['emotion']}, "
                f"{analysis['instrumentation']}, the bpm is {analysis.get('bpm', DEFAULT_BPM)}"
            ),
            "gt_lyric": lyrics,
            "bpm": analysis.get('bpm', DEFAULT_BPM),
            "entry_type": "textprompt"  # 添加类型标识
        },
        {
            "idx": f"{prefix}_audioprompt_{timestamp}",
            "gt_lyric": lyrics,
            "prompt_audio_path": prompt_audio_path,
            "bpm": analysis.get('bpm', DEFAULT_BPM),
            "entry_type": "audioprompt"  # 添加类型标识
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
            script = "generate_safe.sh"
            st.info("已强制使用标准生成模式(generate_safe.sh)")
        elif gpu_info and gpu_info["total"] >= 30:
            script = "generate_safe.sh"
            st.info(f"检测到充足显存 ({gpu_info['total']:.1f}GB)，将使用标准生成模式")
        else:
            script = "generate_lowmem_safe.sh"
            st.warning(f"显存不足30GB ({gpu_info['total']:.1f}GB if available)，使用低显存模式")
        
        # 使用绝对路径
        cmd = [
            "bash",
            str(get_absolute_path(str(PROJECT_ROOT / script))),  # 显式转换为字符串
            str(get_absolute_path(str(SONG_GEN_DIR / "ckpt/songgeneration_base/"))),
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


# 典型结构模板
# ========================
# Streamlit 界面
# ========================
def model_management_tab():
    """模型管理标签页"""
    tab1, tab2, tab3 = st.tabs(["选择模型", "添加模型", "删除模型"])
    
    with tab1:
        selected = st.selectbox(
            "当前模型",
            options=list(SUPPORTED_MODELS.keys()),
            index=list(SUPPORTED_MODELS.keys()).index(
                st.session_state.get('selected_model', DEFAULT_MODEL)
            ),
            key='model_selector'
        )
        st.session_state.selected_model = selected
        st.info(f"已选择: {selected}")
        
        # 显示模型详情
        #if selected in SUPPORTED_MODELS:
        #    st.json(SUPPORTED_MODELS[selected])
    
    with tab2: # TODO:
        with st.form("add_model_form"):
            model_name = st.text_input("模型名称", 
                help="如: anthropic/claude-3")
            api_base = st.text_input("API地址",
                help="如: https://api.anthropic.com/v1")
            max_tokens = st.number_input("最大token数", 
                min_value=512, max_value=32768, value=4096)
            temp_min = st.slider("最小温度", 0.0, 1.0, 0.1)
            temp_max = st.slider("最大温度", 0.0, 1.0, 1.0)
            api_key = st.text_input("API密钥", type="password")
            
            if st.form_submit_button("添加模型"):
                if model_name and api_base:
                    SUPPORTED_MODELS[model_name] = {
                        "api_base": api_base,
                        "max_tokens": max_tokens,
                        "temperature_range": (temp_min, temp_max)
                    }
                    # 保存到secrets (需要手动处理)
                    st.session_state.secrets[f"{model_name.replace('/', '_')}_API_KEY"] = api_key
                    st.success(f"已添加模型: {model_name}")
                else:
                    st.error("请填写完整信息")
    
    with tab3: # TODO:
        model_to_delete = st.selectbox(
            "选择要删除的模型",
            options=[m for m in SUPPORTED_MODELS.keys() if m != DEFAULT_MODEL],
            index=0
        )
        if st.button("删除模型"):
            if model_to_delete in SUPPORTED_MODELS:
                del SUPPORTED_MODELS[model_to_delete]
                st.success(f"已删除模型: {model_to_delete}")

def setup_ui():
    """设置Streamlit用户界面"""
    st.title("🎵 MusicFayIn 人工智能音乐生成系统")
    
    # 步骤1: 歌词生成
    st.header("第一步: 生成歌词")
    
    col1, col2 = st.columns([5, 2])
    
    with col1:
        lyric_prompt = st.text_area("输入歌词主题", "若有一天，星空炸裂，乾坤颠覆，故人无数，红颜白发，魂归黄土……")
        
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
                song_length=song_length,
                model=st.session_state.selected_model
            )

            if lyrics:
                cleaned_lyrics = clean_generated_lyrics(lyrics)
                # cleaned_lyrics = st.text_area("生成的歌词", cleaned_lyrics, height=200)
                st.info(cleaned_lyrics)
                st.session_state.app_state['lyrics'] = cleaned_lyrics
                
                # 显示时长分配
                total_seconds = parse_duration_to_seconds(song_length)
                st.subheader("时长分配详情")
                display_duration_breakdown(template["sections"], total_seconds)

                # # 自动分析歌词参数
                # with st.spinner("正在分析歌词特征..."):
                #     analysis = analyze_lyrics(cleaned_lyrics)
                #     if analysis:
                #         st.session_state.app_state['analysis_result'] = analysis
                #         st.success("歌词分析完成！")

    if st.session_state.app_state.get('lyrics'):
        st.session_state.app_state['lyrics']  = st.text_area("生成的歌词", st.session_state.app_state.get('lyrics'), height=200)

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
                min_value=10,
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
        
        prefix = st.text_input("ID前缀", lyric_prompt[:5])
        
        # 添加生成类型选择
        gen_type = st.radio(
            "生成类型",
            options=["mixed", "bgm", "vocal", "separate"],
            format_func=lambda x: {
                "mixed": "完整歌曲",
                "bgm": "纯音乐(BGM)", 
                "vocal": "纯人声",
                "separate": "生成单独的人声和伴奏音轨"
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


        # 添加条目选择界面
        st.subheader("选择要保留的配置条目")
        col1, col2, col3, col4 = st.columns(4)
        
        # 默认选中第3和第4条
        default_selected = [False, False, True, True if uploaded_file else False]
        selected = [
            col1.checkbox("自动提示", value=default_selected[0], key="autoprompt"),
            col2.checkbox("无提示", value=default_selected[1], key="noprompt"),
            col3.checkbox("文本提示", value=default_selected[2], key="textprompt"),
            col4.checkbox("音频提示", value=default_selected[3], key="audioprompt")
        ]

        entries = generate_jsonl_entries(
            prefix,
            st.session_state.app_state['lyrics'],
            st.session_state.app_state['analysis_result'],
            prompt_audio_path
        )       
                
        if st.button("生成JSONL配置"):
        
            # 过滤选中的条目
            filtered_entries = [entry for entry, select in zip(entries, selected) if select]

            if not filtered_entries:
                st.warning("请至少选择一条配置")
                return
            
                        
            filename = f"{prefix}_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
            filepath = save_jsonl(filtered_entries, filename)
            
            st.session_state.app_state['generated_jsonl'] = filepath
            st.session_state.app_state['selected_entries'] = filtered_entries  # 保存选中的条目
            
            st.success(f"JSONL文件已生成: {filepath}")
            
            # 显示选中的条目
            st.subheader("选中的配置条目")
            for entry in filtered_entries:
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
                "强制使用标准模式(generate_safe.sh)", 
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

    # 在侧边栏添加模型管理
    with st.sidebar:
        st.header("模型管理")
        model_management_tab()

    # 侧边栏说明
    st.sidebar.markdown("""
    ### 使用流程
    1. **生成歌词**：输入主题生成歌词
    2. **分析歌词**：自动分析音乐参数
    3. **调整参数**：根据需要修改参数
    4. **生成配置**：创建JSONL配置文件
    5. **生成音乐**：运行生成脚本
    """)

    # 系统监控
    if st.sidebar.checkbox("显示系统资源"):
        show_system_monitor()


# ========================
# 主程序
# ========================
if __name__ == "__main__":

    # 在全局变量或session_state中添加运行状态标志
    if 'running_process' not in st.session_state:
        st.session_state.running_process = None

    os.environ.update({
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
