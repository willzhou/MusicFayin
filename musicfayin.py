# Author: Ningbo Wise Effects, Inc. (汇视创影) & Will Zhou
# Date: 2025-07-07
# License: Apache 2.0

import streamlit as st
import json
import requests
from datetime import datetime
import os
import subprocess
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import Tensor

import torchaudio
import numpy as np
from omegaconf import OmegaConf
from typing import Dict, Any, List, Tuple, Optional
import psutil
import sys
from pathlib import Path
import re

# 常量定义
DEEPSEEK_API_KEY = "sk-" # 换成你自己的API KEY
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

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
# 音乐生成核心逻辑
# ========================
class MusicGenerator:
    def __init__(self, ckpt_path="./ckpt/"):  # 修改基础路径为ckpt根目录
        self.ckpt_path = ckpt_path
        # 更新所有模型路径指向正确位置
        self.cfg_path = os.path.join(ckpt_path, "songgeneration_base/config.yaml")
        self.model_ckpt = os.path.join(ckpt_path, "songgeneration_base/model.pt")
        self.auto_prompt_path = os.path.join(ckpt_path, "prompt.pt")
        
        # 添加音频tokenizer路径配置
        self.audio_tokenizer_checkpoint = os.path.join(ckpt_path, "model_1rvq/model_2_fixed.safetensors")
        self.separate_tokenizer_checkpoint = os.path.join(ckpt_path, "model_septoken/model_2.safetensors")
        
        # 显式初始化所有模型属性为None
        self._audio_tokenizer = None
        self._separate_tokenizer = None
        self._separator = None
        self._model = None
        self._auto_prompt = None
        
        # 验证关键文件是否存在
        self._verify_files()
        
    def _verify_files(self):
        required_files = [
            self.cfg_path,
            self.model_ckpt,
            self.auto_prompt_path,
            self.audio_tokenizer_checkpoint,
            self.separate_tokenizer_checkpoint
        ]
        missing = [f for f in required_files if not os.path.exists(f)]
        if missing:
            raise FileNotFoundError(
                f"缺少必要的模型文件: {missing}\n"
                "请确保：\n"
                "1. 所有模型文件已下载\n"
                "2. 文件路径配置正确\n"
                f"当前模型目录：{self.ckpt_path}"
            )
        
    @property
    def separator(self):
        if self._separator is None:
            self._separator = Separator()
        return self._separator
    
    @property
    def auto_prompt(self):
        if self._auto_prompt is None and os.path.exists(self.auto_prompt_path):
            self._auto_prompt = torch.load(self.auto_prompt_path)
        return self._auto_prompt
    
    def load_models(self, cfg):
        """加载音频tokenizer和模型"""
        if self._audio_tokenizer is None:
            self._audio_tokenizer = builders.get_audio_tokenizer_model(
                cfg.audio_tokenizer_checkpoint, cfg
            ).eval().cuda()
            
        if "audio_tokenizer_checkpoint_sep" in cfg.keys() and self._separate_tokenizer is None:
            self._separate_tokenizer = builders.get_audio_tokenizer_model(
                cfg.audio_tokenizer_checkpoint_sep, cfg
            ).eval().cuda()
            
        return self._audio_tokenizer, self._separate_tokenizer
    
    def generate_music(self, jsonl_path, save_dir="output"):
        """执行音乐生成流程"""
        # 初始化配置
        torch.backends.cudnn.enabled = False
        OmegaConf.register_new_resolver("eval", lambda x: eval(x))
        OmegaConf.register_new_resolver("concat", lambda *x: [xxx for xx in x for xxx in xx])
        OmegaConf.register_new_resolver("get_fname", lambda: os.path.splitext(os.path.basename(jsonl_path))[0])
        
        cfg = OmegaConf.load(self.cfg_path)
        cfg.mode = 'inference'
        max_duration = cfg.max_dur
        
        # 加载模型
        audio_tokenizer, separate_tokenizer = self.load_models(cfg)
        
        # 处理输入数据
        new_items = self.process_input_items(jsonl_path, save_dir, cfg, audio_tokenizer, separate_tokenizer)
        
        # 生成音乐
        self.run_generation(new_items, cfg, max_duration, save_dir)
        
        # 清理资源
        self.cleanup()
        
        # 返回生成的文件列表
        return [item['wav_path'] for item in new_items]
    
    def process_input_items(self, jsonl_path, save_dir, cfg, audio_tokenizer, separate_tokenizer):
        """处理输入JSONL文件"""
        with open(jsonl_path, "r") as fp:
            lines = fp.readlines()
            
        new_items = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            try:
                item = json.loads(line)
            except json.JSONDecoderError:
                st.error(f"Invalid JSON line: {line}")
                continue
                
            target_wav_name = f"{save_dir}/audios/{item['idx']}.flac"
            
            # 处理提示音频
            if "prompt_audio_path" in item:
                pmt_wav, vocal_wav, bgm_wav = self.process_audio_prompt(item, audio_tokenizer, separate_tokenizer)
            elif "auto_prompt_audio_type" in item:
                pmt_wav, vocal_wav, bgm_wav = self.process_auto_prompt(item)
            else:
                pmt_wav, vocal_wav, bgm_wav = None, None, None
                
            item.update({
                'pmt_wav': pmt_wav,
                'vocal_wav': vocal_wav,
                'bgm_wav': bgm_wav,
                'melody_is_wav': pmt_wav is not None,
                'idx': f"{item['idx']}",
                'wav_path': target_wav_name
            })
            new_items.append(item)
            
        return new_items
    
    def process_audio_prompt(self, item, audio_tokenizer, separate_tokenizer):
        """处理音频提示"""
        assert os.path.exists(item['prompt_audio_path']), f"Prompt audio not found: {item['prompt_audio_path']}"
        
        pmt_wav, vocal_wav, bgm_wav = self.separator.run(item['prompt_audio_path'])
        
        # 保存原始波形用于后续处理
        item['raw_pmt_wav'] = pmt_wav
        item['raw_vocal_wav'] = vocal_wav
        item['raw_bgm_wav'] = bgm_wav
        
        # 编码音频
        pmt_wav = self.prepare_audio_tensor(pmt_wav)
        vocal_wav = self.prepare_audio_tensor(vocal_wav)
        bgm_wav = self.prepare_audio_tensor(bgm_wav)
        
        pmt_wav = pmt_wav.cuda()
        vocal_wav = vocal_wav.cuda()
        bgm_wav = bgm_wav.cuda()
        
        pmt_wav, _ = audio_tokenizer.encode(pmt_wav)
        if separate_tokenizer is not None:
            vocal_wav, bgm_wav = separate_tokenizer.encode(vocal_wav, bgm_wav)
            
        return pmt_wav, vocal_wav, bgm_wav
    
    def process_auto_prompt(self, item):
        """处理自动提示"""
        assert item["auto_prompt_audio_type"] in AUTO_PROMPT_TYPES, f"Invalid auto prompt type: {item['auto_prompt_audio_type']}"
        
        if self.auto_prompt is None:
            raise ValueError("Auto prompt file not found")
            
        if item["auto_prompt_audio_type"] == "Auto":
            prompt_token = self.auto_prompt[np.random.randint(0, len(self.auto_prompt))]
        else:
            prompt_token = self.auto_prompt[item["auto_prompt_audio_type"]][np.random.randint(0, len(self.auto_prompt[item["auto_prompt_audio_type"]]))]
            
        return prompt_token[:,[0],:], prompt_token[:,[1],:], prompt_token[:,[2],:]
    
    def prepare_audio_tensor(self, audio):
        """准备音频张量"""
        if audio.dim() == 2:
            audio = audio[None]
        if audio.dim() != 3:
            raise ValueError("Audio should have shape [B, C, T]")
        return audio
    
    def run_generation(self, items, cfg, max_duration, save_dir):
        """运行音乐生成"""
        # 创建输出目录
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(f"{save_dir}/audios", exist_ok=True)
        os.makedirs(f"{save_dir}/jsonl", exist_ok=True)
        
        # 初始化模型
        model_light = CodecLM_PL(cfg, self.model_ckpt).eval()
        model_light.audiolm.cfg = cfg
        model = CodecLM(
            name="tmp",
            lm=model_light.audiolm,
            audiotokenizer=None,
            max_duration=max_duration,
            seperate_tokenizer=None,
        )
        model.lm = model.lm.cuda().to(torch.float16)
        
        # 设置生成参数
        model.set_generation_params(
            duration=max_duration,
            extend_stride=5,
            temperature=0.9,
            cfg_coef=1.5,
            top_k=50,
            top_p=0.0,
            record_tokens=True,
            record_window=50
        )
        
        # 生成音乐
        for item in items:
            self.generate_single_item(item, model)
        
        # 清理模型
        del model
        del model_light
        torch.cuda.empty_cache()
        
        # 分离音频生成
        self.generate_separate_audio(items, cfg, save_dir)
        
        # 保存结果
        src_jsonl_name = os.path.basename(jsonl_path)
        with open(f"{save_dir}/jsonl/{src_jsonl_name}.jsonl", "w", encoding='utf-8') as fw:
            for item in items:
                # 清理临时字段
                for key in ['tokens', 'pmt_wav', 'vocal_wav', 'bgm_wav', 'melody_is_wav', 
                          'raw_pmt_wav', 'raw_vocal_wav', 'raw_bgm_wav']:
                    item.pop(key, None)
                fw.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    def generate_single_item(self, item, model):
        """生成单个音乐项"""
        generate_inp = {
            'lyrics': [item["gt_lyric"].replace("  ", " ")],
            'descriptions': [item.get("descriptions")],
            'melody_wavs': item['pmt_wav'],
            'vocal_wavs': item['vocal_wav'],
            'bgm_wavs': item['bgm_wav'],
            'melody_is_wav': item['melody_is_wav'],
        }
        
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            item['tokens'] = model.generate(**generate_inp, return_tokens=True)
    
    def generate_separate_audio(self, items, cfg, save_dir):
        """生成分离的音频"""
        separate_tokenizer = builders.get_audio_tokenizer_model(cfg.audio_tokenizer_checkpoint_sep, cfg).eval().cuda()
        
        model = CodecLM(
            name="tmp",
            lm=None,
            audiotokenizer=None,
            max_duration=cfg.max_dur,
            seperate_tokenizer=separate_tokenizer,
        )
        
        for item in items:
            with torch.no_grad():
                if 'raw_pmt_wav' in item:   
                    wav_seperate = model.generate_audio(
                        item['tokens'], 
                        item['raw_pmt_wav'], 
                        item['raw_vocal_wav'], 
                        item['raw_bgm_wav'],
                        chunked=True
                    )
                else:
                    wav_seperate = model.generate_audio(item['tokens'], chunked=True)
                    
            torchaudio.save(item['wav_path'], wav_seperate[0].cpu().float(), cfg.sample_rate)
        
        del model
        torch.cuda.empty_cache()
    
    def cleanup(self):
        """清理资源"""
        self._separator = None
        self._audio_tokenizer = None
        self._separate_tokenizer = None
        self._model = None
        torch.cuda.empty_cache()

class Separator:
    """音频分离器"""
    def __init__(self, dm_model_path='third_party/demucs/ckpt/htdemucs.pth', 
                 dm_config_path='third_party/demucs/ckpt/htdemucs.yaml', gpu_id=0):
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() and gpu_id < torch.cuda.device_count() else "cpu")
        self.demucs_model = self.init_demucs_model(dm_model_path, dm_config_path)
    
    def init_demucs_model(self, model_path, config_path):
        model = get_model_from_yaml(config_path, model_path)
        model.to(self.device)
        model.eval()
        return model
    
    def load_audio(self, f):
        a, fs = torchaudio.load(f)
        if fs != 48000:
            a = torchaudio.functional.resample(a, fs, 48000)
        if a.shape[-1] >= 48000*10:
            a = a[..., :48000*10]
        else:
            a = torch.cat([a, a], -1)
        return a[:, 0:48000*10]
    
    def run(self, audio_path, output_dir='tmp', ext=".flac"):
        os.makedirs(output_dir, exist_ok=True)
        name = os.path.splitext(os.path.basename(audio_path))[0]
        
        # 分离音频
        drums_path, bass_path, other_path, vocal_path = self.demucs_model.separate(
            audio_path, output_dir, device=self.device
        )
        
        # 清理不需要的轨道
        for path in [drums_path, bass_path, other_path]:
            os.remove(path)
            
        # 加载并处理音频
        full_audio = self.load_audio(audio_path)
        vocal_audio = self.load_audio(vocal_path)
        bgm_audio = full_audio - vocal_audio
        
        return full_audio, vocal_audio, bgm_audio

class AudioEncoder(nn.Module):
    """音频编码器网络"""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=15, stride=5, padding=7),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=15, stride=5, padding=7),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=15, stride=5, padding=7),
            nn.ReLU()
        )
        self.proj = nn.Linear(256, cfg.embedding_dim)
    def forward(self, x):
        # x: [B, 1, T]
        x = self.conv_layers(x)  # [B, 256, T']
        x = x.permute(0, 2, 1)  # [B, T', 256]
        return self.proj(x)  # [B, T', embedding_dim]
class AudioDecoder(nn.Module):
    """音频解码器网络""" 
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Linear(cfg.embedding_dim, 256)
        self.conv_trans_layers = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=15, stride=5, padding=7, output_padding=4),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=15, stride=5, padding=7, output_padding=4),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 1, kernel_size=15, stride=5, padding=7, output_padding=4),
            nn.Tanh()
        )
    def forward(self, x):
        # x: [B, T', embedding_dim]
        x = self.proj(x)  # [B, T', 256]
        x = x.permute(0, 2, 1)  # [B, 256, T']
        return self.conv_trans_layers(x)  # [B, 1, T]
    
# ========================
# 应用界面函数
# ========================
def call_deepseek_api(prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> str:
    """调用DeepSeek API生成歌词"""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    try:
        response = requests.post(DEEPSEEK_URL, headers=headers, json=payload)
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
def generate_jsonl_entries(prefix: str, lyrics: str, analysis: Dict[str, Any]) -> List[Dict]:
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
            "prompt_audio_path": "input/sample_prompt_audio.wav"
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
            cleaned_line = line.replace('，', '.').replace('。', '.').strip('. ')
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

# ========================
# 模型构建器定义
# ========================
class builders:
    """模型构建工具类"""
    
    @staticmethod
    def get_audio_tokenizer_model(checkpoint_path: str, cfg: OmegaConf):
        """支持自动补全相对路径"""
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = os.path.join("./ckpt", checkpoint_path)
            
        if not os.path.exists(checkpoint_path):
            # 尝试在已知路径中查找
            possible_locations = [
                "model_1rvq/model_2_fixed.safetensors",
                "models--lengyue233--content-vec-best/snapshots/c0b9ba13db21beaa4053faae94c102ebe326fd68/model.safetensors"
            ]
            for loc in possible_locations:
                test_path = os.path.join("./ckpt", loc)
                if os.path.exists(test_path):
                    checkpoint_path = test_path
                    break
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"无法找到tokenizer模型文件，已尝试路径: {checkpoint_path}\n"
                f"请检查以下位置是否存在模型文件:\n"
                f"- ckpt/model_1rvq/model_2_fixed.safetensors\n"
                f"- ckpt/models--lengyue233--content-vec-best/.../model.safetensors"
            )
        
        model = AudioTokenizer(cfg)
        state_dict = torch.load(checkpoint_path, map_location='cuda') # cpu -> cuda
        model.load_state_dict(state_dict)
        return model
    
    @staticmethod
    def get_lm_model(cfg: OmegaConf):
        """加载语言模型"""
        model = AudioLM(
            n_vocab=cfg.n_vocab,
            dim=cfg.dim,
            depth=cfg.depth,
            heads=cfg.heads,
            ff_mult=cfg.ff_mult,
            max_seq_len=cfg.max_seq_len,
            use_flash_attn=cfg.get('use_flash_attn', False)
        )
        return model
    
    @staticmethod
    def get_separate_tokenizer_model(checkpoint_path: str, cfg: OmegaConf):
        """加载分离tokenizer模型"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Separate tokenizer checkpoint not found: {checkpoint_path}")
            
        model = SeparateAudioTokenizer(cfg)
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict)
        return model

# ========================
# 核心模型类定义
# ========================
class CodecLM(nn.Module):
    """音乐生成核心模型"""
    
    def __init__(self, 
                 name: str, 
                 lm: nn.Module, 
                 audiotokenizer: nn.Module = None,
                 max_duration: float = 30.0,
                 seperate_tokenizer: nn.Module = None):
        super().__init__()
        self.name = name
        self.lm = lm
        self.audiotokenizer = audiotokenizer
        self.seperate_tokenizer = seperate_tokenizer
        self.max_duration = max_duration
        self.sample_rate = 24000  # 默认采样率
        
        # 默认生成参数
        self.generation_params = {
            'duration': max_duration,
            'extend_stride': 5,
            'temperature': 0.9,
            'cfg_coef': 1.5,
            'top_k': 50,
            'top_p': 0.0,
            'record_tokens': True,
            'record_window': 50
        }
        
    def set_generation_params(self, **kwargs):
        """设置生成参数"""
        self.generation_params.update(kwargs)
        
    def generate(self, 
                lyrics: List[str],
                descriptions: List[str] = None,
                melody_wavs: torch.Tensor = None,
                vocal_wavs: torch.Tensor = None,
                bgm_wavs: torch.Tensor = None,
                melody_is_wav: bool = False,
                return_tokens: bool = False):
        """
        生成音乐
        
        Args:
            lyrics: 歌词列表
            descriptions: 描述文本列表
            melody_wavs: 旋律音频提示
            vocal_wavs: 人声音频提示
            bgm_wavs: 背景音乐音频提示
            melody_is_wav: 是否为原始波形
            return_tokens: 是否返回token序列
            
        Returns:
            生成的音频或token序列
        """
        # 准备输入
        inputs = self._prepare_inputs(
            lyrics, descriptions, 
            melody_wavs, vocal_wavs, bgm_wavs,
            melody_is_wav
        )
        
        # 生成token
        with torch.no_grad():
            tokens = self.lm.generate(**inputs, **self.generation_params)
            
        if return_tokens:
            return tokens
            
        # 解码为音频
        return self.generate_audio(tokens)
    
    def generate_audio(self, 
                      tokens: torch.Tensor,
                      pmt_wav: torch.Tensor = None,
                      vocal_wav: torch.Tensor = None,
                      bgm_wav: torch.Tensor = None,
                      chunked: bool = False):
        """
        从token生成音频
        
        Args:
            tokens: 生成的token序列
            pmt_wav: 原始提示音频(用于分离模型)
            vocal_wav: 原始人声音频(用于分离模型)
            bgm_wav: 原始背景音频(用于分离模型)
            chunked: 是否分块处理
            
        Returns:
            生成的音频波形
        """
        if self.seperate_tokenizer is not None and pmt_wav is not None:
            # 使用分离模型生成
            return self._generate_separate_audio(
                tokens, pmt_wav, vocal_wav, bgm_wav, chunked
            )
        elif self.audiotokenizer is not None:
            # 使用普通tokenizer生成
            return self._generate_normal_audio(tokens, chunked)
        else:
            raise ValueError("No valid tokenizer available for audio generation")
    
    def _generate_normal_audio(self, tokens: torch.Tensor, chunked: bool):
        """使用普通tokenizer生成音频"""
        if chunked:
            # 分块处理大音频
            chunk_size = 1024  # 根据GPU内存调整
            wavs = []
            for i in range(0, tokens.shape[1], chunk_size):
                chunk = tokens[:, i:i+chunk_size]
                wav = self.audiotokenizer.decode(chunk)
                wavs.append(wav)
            return torch.cat(wavs, dim=-1)
        else:
            return self.audiotokenizer.decode(tokens)
    
    def _generate_separate_audio(self, 
                                tokens: torch.Tensor,
                                pmt_wav: torch.Tensor,
                                vocal_wav: torch.Tensor,
                                bgm_wav: torch.Tensor,
                                chunked: bool):
        """使用分离tokenizer生成音频"""
        if chunked:
            # 分块处理大音频
            chunk_size = 1024  # 根据GPU内存调整
            wavs = []
            for i in range(0, tokens.shape[1], chunk_size):
                chunk = tokens[:, i:i+chunk_size]
                wav = self.seperate_tokenizer.decode(
                    chunk, pmt_wav, vocal_wav, bgm_wav
                )
                wavs.append(wav)
            return torch.cat(wavs, dim=-1)
        else:
            return self.seperate_tokenizer.decode(
                tokens, pmt_wav, vocal_wav, bgm_wav
            )
    
    def _prepare_inputs(self,
                       lyrics: List[str],
                       descriptions: List[str],
                       melody_wavs: torch.Tensor,
                       vocal_wavs: torch.Tensor,
                       bgm_wavs: torch.Tensor,
                       melody_is_wav: bool):
        """准备模型输入"""
        inputs = {
            'texts': lyrics,
            'descriptions': descriptions if descriptions else [None] * len(lyrics)
        }
        
        # 处理音频提示
        if melody_wavs is not None:
            if melody_is_wav:
                # 原始波形需要编码
                melody_tokens = self.audiotokenizer.encode(melody_wavs)
                inputs['melody_tokens'] = melody_tokens
            else:
                # 已经是token形式
                inputs['melody_tokens'] = melody_wavs
                
        if vocal_wavs is not None and bgm_wavs is not None:
            if self.seperate_tokenizer is not None:
                inputs['vocal_tokens'], inputs['bgm_tokens'] = \
                    self.seperate_tokenizer.encode(vocal_wavs, bgm_wavs)
            else:
                inputs['vocal_tokens'] = self.audiotokenizer.encode(vocal_wavs)
                inputs['bgm_tokens'] = self.audiotokenizer.encode(bgm_wavs)
                
        return inputs

class CodecLM_PL(pl.LightningModule):
    """PyTorch Lightning版本的CodecLM模型"""
    
    def __init__(self, cfg: OmegaConf, checkpoint_path: str = None):
        super().__init__()
        self.cfg = cfg
        self.audiolm = builders.get_lm_model(cfg)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load_checkpoint(checkpoint_path)
            
        # 初始化损失函数
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=cfg.pad_token_id)
        
    def load_checkpoint(self, path: str):
        """加载预训练权重"""
        state_dict = torch.load(path, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        self.audiolm.load_state_dict(state_dict)
        
    def forward(self, 
               texts: List[str],
               descriptions: List[str] = None,
               melody_tokens: torch.Tensor = None,
               vocal_tokens: torch.Tensor = None,
               bgm_tokens: torch.Tensor = None,
               labels: torch.Tensor = None):
        """前向传播"""
        return self.audiolm(
            texts=texts,
            descriptions=descriptions,
            melody_tokens=melody_tokens,
            vocal_tokens=vocal_tokens,
            bgm_tokens=bgm_tokens,
            labels=labels
        )
    
    def training_step(self, batch, batch_idx):
        """训练步骤"""
        inputs = {
            'texts': batch['texts'],
            'descriptions': batch.get('descriptions'),
            'melody_tokens': batch.get('melody_tokens'),
            'vocal_tokens': batch.get('vocal_tokens'),
            'bgm_tokens': batch.get('bgm_tokens'),
            'labels': batch['labels']
        }
        
        outputs = self(**inputs)
        loss = self.loss_fn(outputs.logits.view(-1, outputs.logits.size(-1)), 
                           inputs['labels'].view(-1))
        
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        inputs = {
            'texts': batch['texts'],
            'descriptions': batch.get('descriptions'),
            'melody_tokens': batch.get('melody_tokens'),
            'vocal_tokens': batch.get('vocal_tokens'),
            'bgm_tokens': batch.get('bgm_tokens'),
            'labels': batch['labels']
        }
        
        outputs = self(**inputs)
        loss = self.loss_fn(outputs.logits.view(-1, outputs.logits.size(-1)), 
                           inputs['labels'].view(-1))
        
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        """配置优化器"""
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.cfg.max_steps,
            eta_min=self.cfg.min_lr
        )
        
        return [optimizer], [scheduler]

class VectorQuantizer(nn.Module):
    """向量量化层"""
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        # 初始化码本
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)
    def forward(self, inputs):
        # 计算输入与码本的距离
        distances = (torch.sum(inputs**2, dim=-1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=-1)
                    - 2 * torch.matmul(inputs, self.embedding.weight.t()))
        
        # 获取最近邻编码
        encoding_indices = torch.argmin(distances, dim=-1)
        quantized = self.embedding(encoding_indices)
        
        # 计算损失
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # 直通估计器
        quantized = inputs + (quantized - inputs).detach()
        
        return quantized, loss, (encoding_indices, distances)
    @torch.no_grad()
    def quantize(self, inputs):
        distances = (torch.sum(inputs**2, dim=-1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=-1)
                    - 2 * torch.matmul(inputs, self.embedding.weight.t()))
        return torch.argmin(distances, dim=-1)
    
# ========================
# 辅助模型类
# ========================
class AudioTokenizer(nn.Module):
    """音频tokenizer模型 (VQ-VAE架构)
    
    特性：
    - 自动处理缺失配置
    - 详细的参数验证
    - 支持动态重配置
    """
    
    def __init__(self, cfg: Optional[OmegaConf] = None):
        super().__init__()
        # 初始化完整配置
        self.cfg = self._build_complete_config(cfg)
        
        # 验证配置有效性
        self._validate_config()
        
        # 初始化组件
        self.encoder = AudioEncoder(self.cfg.audio_tokenizer)
        self.decoder = AudioDecoder(self.cfg.audio_tokenizer)
        self.quantizer = VectorQuantizer(
            num_embeddings=self.cfg.audio_tokenizer.num_embeddings,
            embedding_dim=self.cfg.audio_tokenizer.embedding_dim,
            commitment_cost=self.cfg.audio_tokenizer.commitment_cost
        )
    
    def _build_complete_config(self, cfg: Optional[OmegaConf]) -> OmegaConf:
        """构建完整配置，合并默认值和用户配置"""
        default_config = OmegaConf.create({
            'audio_tokenizer': {
                'embedding_dim': 256,
                'num_embeddings': 1024,
                'commitment_cost': 0.25,
                'in_channels': 1,
                'sample_rate': 24000,
                'encoder': {
                    'channels': [64, 128, 256],
                    'kernel_sizes': [15, 15, 15],
                    'strides': [5, 5, 5]
                },
                'decoder': {
                    'channels': [256, 128, 64, 1],
                    'kernel_sizes': [15, 15, 15],
                    'strides': [5, 5, 5]
                }
            }
        })
        
        if cfg is None:
            return default_config
            
        # 合并配置
        if 'audio_tokenizer' not in cfg:
            # 处理平铺配置
            merged = OmegaConf.merge(default_config, {'audio_tokenizer': cfg})
        else:
            merged = OmegaConf.merge(default_config, cfg)
            
        return merged
    
    def _validate_config(self) -> None:
        """验证配置完整性"""
        required_keys = {
            'top_level': ['audio_tokenizer'],
            'audio_tokenizer': [
                'embedding_dim', 'num_embeddings', 'commitment_cost',
                'in_channels', 'sample_rate', 'encoder', 'decoder'
            ],
            'encoder': ['channels', 'kernel_sizes', 'strides'],
            'decoder': ['channels', 'kernel_sizes', 'strides']
        }
        
        errors = []
        
        # 检查顶级配置
        for section in required_keys['top_level']:
            if section not in self.cfg:
                errors.append(f"Missing top-level section: {section}")
        
        # 检查audio_tokenizer配置
        tokenizer_cfg = self.cfg.audio_tokenizer
        for key in required_keys['audio_tokenizer']:
            if key not in tokenizer_cfg:
                errors.append(f"Missing audio_tokenizer.{key}")
        
        # 检查encoder/decoder配置
        for component in ['encoder', 'decoder']:
            if component in tokenizer_cfg:
                for key in required_keys[component]:
                    if key not in tokenizer_cfg[component]:
                        errors.append(f"Missing audio_tokenizer.{component}.{key}")
            else:
                errors.append(f"Missing audio_tokenizer.{component} section")
        
        # 检查参数维度一致性
        if 'encoder' in tokenizer_cfg:
            enc_cfg = tokenizer_cfg.encoder
            if len(enc_cfg.channels) != len(enc_cfg.kernel_sizes) or \
               len(enc_cfg.channels) != len(enc_cfg.strides):
                errors.append("Encoder config mismatch: channels/kernel_sizes/strides must have same length")
            
            if enc_cfg.channels[-1] != self.cfg.audio_tokenizer.embedding_dim:
                errors.append(
                    f"Encoder output channels ({enc_cfg.channels[-1]}) "
                    f"must match embedding_dim ({self.cfg.audio_tokenizer.embedding_dim})"
                )
        
        if errors:
            raise ValueError(
                "Invalid audio tokenizer configuration:\n  - " + 
                "\n  - ".join(errors) + 
                f"\nCurrent config:\n{OmegaConf.to_yaml(self.cfg)}"
            )
    
    def forward(self, x):
        """处理音频输入"""
        # [实现您的forward逻辑]
        pass
    @property
    def config_summary(self) -> Dict[str, Any]:
        """获取配置摘要"""
        return {
            'embedding_dim': self.cfg.audio_tokenizer.embedding_dim,
            'num_embeddings': self.cfg.audio_tokenizer.num_embeddings,
            'latent_ratio': self._calculate_latent_ratio(),
            'encoder_params': sum(p.numel() for p in self.encoder.parameters()),
            'decoder_params': sum(p.numel() for p in self.decoder.parameters())
        }
    
    def _calculate_latent_ratio(self) -> float:
        """计算潜在空间下采样率"""
        stride_product = 1
        for s in self.cfg.audio_tokenizer.encoder.strides:
            stride_product *= s
        return float(stride_product)
# 辅助组件定义
class AudioEncoder(nn.Module):
    def __init__(self, cfg: OmegaConf):
        super().__init__()
        layers = []
        in_ch = cfg.in_channels
        for out_ch, k, s in zip(cfg.encoder.channels, 
                               cfg.encoder.kernel_sizes,
                               cfg.encoder.strides):
            layers += [
                nn.Conv1d(in_ch, out_ch, k, stride=s, padding=k//2),
                nn.ReLU()
            ]
            in_ch = out_ch
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)
class AudioDecoder(nn.Module):
    def __init__(self, cfg: OmegaConf):
        super().__init__()
        layers = []
        in_ch = cfg.embedding_dim
        for i, (out_ch, k, s) in enumerate(zip(cfg.decoder.channels,
                                             cfg.decoder.kernel_sizes,
                                             cfg.decoder.strides)):
            layers += [
                nn.ConvTranspose1d(in_ch, out_ch, k, stride=s, 
                                 padding=k//2,
                                 output_padding=s-1),
                nn.ReLU() if i < len(cfg.decoder.channels)-2 else nn.Tanh()
            ]
            in_ch = out_ch
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

class SeparateAudioTokenizer(nn.Module):
    """分离音频tokenizer模型"""
    
    def __init__(self, cfg: OmegaConf):
        super().__init__()
        self.cfg = cfg
        self.vocal_encoder = AudioEncoder(cfg)
        self.bgm_encoder = AudioEncoder(cfg)
        self.decoder = AudioDecoder(cfg)
        self.quantizer = VectorQuantizer(
            num_embeddings=cfg.num_embeddings,
            embedding_dim=cfg.embedding_dim,
            commitment_cost=cfg.commitment_cost
        )
        
    def encode(self, 
              vocal: torch.Tensor, 
              bgm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """编码分离的音频"""
        vocal_z = self.vocal_encoder(vocal)
        bgm_z = self.bgm_encoder(bgm)
        
        vocal_z_q, _, _ = self.quantizer(vocal_z)
        bgm_z_q, _, _ = self.quantizer(bgm_z)
        
        return vocal_z_q, bgm_z_q
    
    def decode(self, 
              tokens: torch.Tensor,
              pmt_wav: torch.Tensor,
              vocal_wav: torch.Tensor,
              bgm_wav: torch.Tensor) -> torch.Tensor:
        """解码token为分离的音频"""
        # 这里实现分离音频的特殊解码逻辑
        # 实际实现可能更复杂，这里简化处理
        mixed = pmt_wav + 0.5 * vocal_wav + 0.3 * bgm_wav
        return mixed

class AudioLM(nn.Module):
    """音频语言模型"""
    
    def __init__(self,
                 n_vocab: int,
                 dim: int,
                 depth: int,
                 heads: int,
                 ff_mult: int,
                 max_seq_len: int,
                 use_flash_attn: bool = False):
        super().__init__()
        self.token_emb = nn.Embedding(n_vocab, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        
        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            ff_mult=ff_mult,
            flash_attn=use_flash_attn
        )
        
        self.to_logits = nn.Linear(dim, n_vocab)
        
    def forward(self,
               texts: List[str],
               descriptions: List[str] = None,
               melody_tokens: torch.Tensor = None,
               vocal_tokens: torch.Tensor = None,
               bgm_tokens: torch.Tensor = None,
               labels: torch.Tensor = None):
        """前向传播"""
        # 文本嵌入
        text_emb = self._embed_text(texts, descriptions)
        
        # 合并所有嵌入
        x = text_emb
        if melody_tokens is not None:
            melody_emb = self.token_emb(melody_tokens)
            x = x + melody_emb
            
        if vocal_tokens is not None and bgm_tokens is not None:
            vocal_emb = self.token_emb(vocal_tokens)
            bgm_emb = self.token_emb(bgm_tokens)
            x = x + 0.5 * vocal_emb + 0.3 * bgm_emb
            
        # 位置编码
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device)
        pos_emb = self.pos_emb(positions)
        x = x + pos_emb
        
        # Transformer处理
        x = self.transformer(x)
        
        # 输出logits
        logits = self.to_logits(x)
        
        # 计算损失
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=self.cfg.pad_token_id
            )
            
        return {'logits': logits, 'loss': loss}
    
    def generate(self,
                texts: List[str],
                descriptions: List[str] = None,
                melody_tokens: torch.Tensor = None,
                vocal_tokens: torch.Tensor = None,
                bgm_tokens: torch.Tensor = None,
                duration: float = 30.0,
                extend_stride: int = 5,
                temperature: float = 1.0,
                cfg_coef: float = 1.5,
                top_k: int = 50,
                top_p: float = 0.0,
                record_tokens: bool = False,
                record_window: int = 50):
        """生成音乐token"""
        # 初始化生成状态
        generated = []
        if record_tokens:
            all_tokens = []
            
        # 文本嵌入
        text_emb = self._embed_text(texts, descriptions)
        
        # 初始输入
        x = text_emb
        if melody_tokens is not None:
            melody_emb = self.token_emb(melody_tokens)
            x = x + melody_emb
            
        if vocal_tokens is not None and bgm_tokens is not None:
            vocal_emb = self.token_emb(vocal_tokens)
            bgm_emb = self.token_emb(bgm_tokens)
            x = x + 0.5 * vocal_emb + 0.3 * bgm_emb
            
        # 生成循环
        for i in range(int(duration * self.cfg.sample_rate / extend_stride)):
            # 位置编码
            positions = torch.arange(x.shape[1], device=x.device)
            pos_emb = self.pos_emb(positions)
            x = x + pos_emb
            
            # Transformer处理
            x = self.transformer(x)
            
            # 采样下一个token
            logits = self.to_logits(x[:, -1:])
            next_token = self._sample_token(
                logits, temperature, top_k, top_p, cfg_coef
            )
            
            # 更新生成序列
            generated.append(next_token)


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
        lyric_prompt = st.text_area("输入歌词主题", "冬天的爱情故事")
        
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
        
        if st.button("生成JSONL配置"):
            entries = generate_jsonl_entries(
                prefix,
                st.session_state.app_state['lyrics'],
                st.session_state.app_state['analysis_result']
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
                
                # 创建进度条和输出容器
                progress_bar = st.progress(0)
                output_container = st.empty()
                
                # 执行生成命令
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True
                )
                
                # 实时显示输出
                full_output = ""
                progress_value = 0  # 初始化进度值
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        full_output += output
                        output_container.text(full_output)
                        progress_value = min(100, progress_value + 5)  # 更新进度值
                        progress_bar.progress(progress_value)  # 使用更新后的值
                
                # 检查执行结果
                return_code = process.poll()
                if return_code == 0:
                    st.success("🎵 音乐生成完成！")
                    
                    # 显示生成的音频文件
                    audio_files = glob.glob(f"{output_dir}/audios/*.flac")
                    if audio_files:
                        st.subheader("生成的音乐文件")
                        for audio_file in sorted(audio_files):
                            st.audio(audio_file)
                            st.download_button(
                                f"下载 {os.path.basename(audio_file)}",
                                data=open(audio_file, "rb").read(),
                                file_name=os.path.basename(audio_file),
                                mime="audio/flac"
                            )
                    else:
                        st.warning("未找到生成的音频文件")
                else:
                    st.error(f"❌ 生成失败 (返回码: {return_code})")
                    st.text(full_output)
                    
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
    # 确保必要的目录存在
    os.makedirs("output/audios", exist_ok=True)
    os.makedirs("output/jsonl", exist_ok=True)
    os.makedirs("input", exist_ok=True)
    
    # 设置并运行UI
    setup_ui()
