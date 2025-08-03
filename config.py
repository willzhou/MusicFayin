# Author: Ningbo Wise Effects, Inc. (汇视创影) & Will Zhou
# License: Apache 2.0

import json
import streamlit as st
from typing import Dict, Any

# 常量定义
# DEEPSEEK_API_KEY = st.secrets['DEEPSEEK_API_KEY'] # 换成你自己的API KEY
# DEEPSEEK_URL = st.secrets['DEEPSEEK_URL']
API_URL = st.secrets['API_URL']
API_KEY = st.secrets['API_KEY']

# 支持的模型列表
SUPPORTED_MODELS = {
    "moonshotai/kimi-k2:free": {
        "api_base": "https://openrouter.ai/api/v1",
        "max_tokens": 4096,
        "temperature_range": (0.1, 1.0)
    },
    "deepseek-chat": {
        "api_base": "https://api.deepseek.com/v1",
        "max_tokens": 2048,
        "temperature_range": (0.5, 1.0)
    }
}

def update_supported_models(config_path: str) -> Dict[str, Any]:
    """
    从JSON配置文件更新SUPPORTED_MODELS字典
    
    :param config_path: JSON配置文件路径
    :return: 更新后的模型配置字典
    """   
    try:
        # 读取JSON配置文件
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        # 更新配置 - 这里采用合并策略，新配置会覆盖旧配置
        for model_name, model_config in config_data.items():
            if model_name in SUPPORTED_MODELS:
                SUPPORTED_MODELS[model_name].update(model_config)
            else:
                SUPPORTED_MODELS[model_name] = model_config
                
        return SUPPORTED_MODELS
    
    except FileNotFoundError:
        print(f"警告: 配置文件 {config_path} 未找到，返回原始配置")
        return SUPPORTED_MODELS
    except json.JSONDecodeError:
        print(f"错误: 配置文件 {config_path} 不是有效的JSON格式")
        return SUPPORTED_MODELS
 
# 使用示例
SUPPORTED_MODELS = update_supported_models("models_config.json")

# 默认模型配置
DEFAULT_MODEL = "deepseek-chat"

# 情绪
EMOTIONS = [
    # 基础情绪 “悲伤的”、“情绪的”、“愤怒的”、“快乐的”、
    "sad", "emotional", "angry", "happy", 
    # “令人振奋的”、“强烈的”、“浪漫的”、“忧郁的”
    "uplifting", "intense", "romantic", "melancholic",

    # 更多基础情绪
    "joyful", # 纯粹的快乐（流行舞曲）
    "heartbroken", # 心碎（抒情 ballad）
    "furious", # 暴怒（金属/摇滚）
    "peaceful", # 宁静（轻音乐）

    # 能量型
    "empowering",     # 励志（ anthem 歌曲）
    "euphoric",       # 狂喜（EDM 高潮）
    "defiant",        # 叛逆（朋克摇滚）

    # 爱情相关
    "romantic",       # 浪漫（经典情歌）
    "sensual",        # 情欲（R&B）
    "bittersweet",    # 苦乐参半（分手歌）
    "unrequited",     # 单相思（忧郁情歌）

    # 忧郁/深沉
    "melancholic",    # 忧郁（民谣）
    "nostalgic",      # 怀旧（复古风）
    "lonely",         # 孤独（钢琴曲）
    "regretful",      # 悔恨（蓝调）

    # 特殊状态
    "anxious",        # 焦虑（另类摇滚）
    "dreamy",         # 梦幻（shoegaze）
    "mysterious",     # 神秘（电影配乐）

    # 积极能量
    "hopeful",        # 充满希望（福音）
    "playful",        # 俏皮（泡泡糖流行）
    "carefree",       # 无忧无虑（夏日单曲）

    # 强烈情感
    "desperate",      # 绝望（emo）
    "vengeful",       # 复仇（暗黑系）
    "triumphant",     # 凯旋（史诗音乐）

    # 复杂情绪
    "conflicted",     # 矛盾（另类R&B）
    "wistful",        # 惆怅（独立民谣）
    "vulnerable",     # 脆弱（灵魂乐）

    # 氛围型
    "ethereal",       # 空灵（氛围电子）
    "haunting",       # 萦绕（恐怖配乐）
    "hypnotic"        # 催眠（迷幻音乐）

    # 组合型 --- 
    # 1. 爱情混合体
    "love-hate",              # 爱恨交织（另类R&B）
    "passionate-fear",        # 热烈又恐惧（拉丁情歌）
    "romantic-nostalgia",     # 浪漫怀旧（复古流行）
    
    # 2. 痛苦与治愈
    "broken-but-healing",     # 破碎中自愈（民谣摇滚）
    "sad-yet-hopeful",        # 悲伤但抱有希望（钢琴 ballad）
    "tears-of-joy",           # 喜极而泣（福音流行）
    
    # 3. 矛盾心理
    "wanting-but-fearing",    # 渴望又害怕（另类摇滚）
    "guilty-pleasure",        # 罪恶快感（暗黑流行）
    "selfish-devotion",       # 自私的奉献（灵魂乐）
    
    # 4. 社会情绪
    "angry-empathy",          # 愤怒的共情（抗议歌曲）
    "isolated-but-connected", # 孤独却共鸣（电子民谣）
    "numb-but-feeling",       # 麻木中感知（emo rap）
    
    # 5. 成长阵痛
    "proud-and-ashamed",      # 骄傲与羞愧（成长叙事）
    "lost-but-finding",       # 迷失中探索（独立摇滚）
    "scared-but-brave",       # 恐惧却勇敢（电影主题曲）
    
    # 6. 超现实组合
    "dreamlike-terror",       # 梦幻式恐惧（另类电子）
    "mechanical-loneliness",  # 机械孤独（赛博朋克）
    "violent-tenderness",     # 暴烈的温柔（后硬核）
    
    # 7. 时间相关
    "nostalgic-dread",        # 怀旧式焦虑（合成器浪潮）
    "future-nostalgia",       # 未来怀旧（Dua Lipa 风格）
    "present-absent",         # 身在心不在（迷幻流行）
    
    # 8. 自然隐喻
    "stormy-calm",            # 暴风雨前的平静（史诗摇滚）
    "frozen-fire",            # 冰与火（力量金属）
    "sunshine-melancholy",    # 阳光忧郁（独立流行）
    
    # 9. 感官冲突
    "sweet-pain",             # 甜蜜的痛苦（另类R&B）
    "bitter-bliss",           # 苦涩的幸福（爵士民谣）
    "soft-destruction",       # 温柔的毁灭（氛围后摇）
    
    # 10. 终极矛盾
    "holy-sinful",            # 神圣与罪恶（福音摇滚）
    "chaotic-order",          # 混乱中的秩序（数学摇滚）
    "loud-silence"            # 震耳欲聋的沉默（后朋克）
]

SINGER_GENDERS = ["male", "female"]

# “自动”、“中国传统”、“金属”、“雷鬼”、“中国戏曲”、“流行”、“电子”、“嘻哈”、“摇滚”、
# “爵士”、“蓝调”、“古典”、“说唱”、“乡村”、“经典摇滚”、“硬摇滚”、“民谣”、“灵魂乐”、
# “舞曲电子”、“乡村摇滚”、“舞曲、舞曲流行、浩室、流行”、“雷鬼”、“实验”、“舞曲、
# 流行”、“舞曲、深浩室、电子”、“韩国流行音乐”、“实验流行”、“流行朋克”、“摇滚乐”、
# “节奏布鲁斯”、“多样”、“流行摇滚”
# GENRES = [
#     'Auto', 'Chinese Tradition', 'Metal', 'Reggae', 'Chinese Opera',
#     "pop", "electronic", "hip hop", "rock", "jazz", "blues", "classical",
#     "rap", "country", "classic rock", "hard rock", "folk", "soul",
#     "dance, electronic", "rockabilly", "dance, dancepop, house, pop",
#     "reggae", "experimental", "dance, pop", "dance, deephouse, electronic",
#     "k-pop", "experimental pop", "pop punk", "rock and roll", "R&B",
#     "varies", "pop rock",
# ]

auto_prompt_type = ['Pop', 'R&B', 'Dance', 'Jazz', 'Folk', 'Rock', 'Chinese Style', 'Chinese Tradition', 'Metal', 'Reggae', 'Chinese Opera', 'Auto']
GENRES = auto_prompt_type

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
TIMBRES = [
    # 1. 明暗度
    "dark",        # 幽暗（大混响/低频率主导）
    "bright",      # 明亮（高频突出/清脆）
    "muted",       # 闷哑（频段削减）
    
    # 2. 温度感
    "warm",        # 温暖（模拟设备/磁带饱和）
    "cold",        # 冰冷（数字感/金属质感）
    "neutral",     # 中性
    
    # 3. 力度
    "aggressive",  # 侵略性（失真/瞬态强）
    "soft",        # 柔和（动态平缓）
    "dynamic",     # 动态大（强弱对比明显）
    
    # 4. 音乐风格
    "rock",        # 摇滚类音色
    "electronic",  # 电子合成音色
    "organic",     # 原声乐器质感
    "hybrid",      # 混合音色
    
    # 5. 音色复杂度
    "simple",      # 简约（纯音/正弦波）
    "complex",     # 复杂（多层叠加）
    "textured",    # 有纹理（颗粒感/噪声层）
    
    # 6. 空间感
    "dry",         # 干声（无效果）
    "ambient",     # 氛围感（长混响）
    "intimate",    # 亲密感（近距离收音）
    
    # 7. 人声相关
    "vocal",       # 人声特征
    "spoken",      # 念白质感
    "choir",       # 合唱团效果
    
    # 8. 特殊状态
    "distorted",   # 失真
    "glitchy",     # 故障音
    "lo-fi",       # 低保真
    "crystalline"  # 水晶般透明

    "varies"       # “变化的”
]

AUTO_PROMPT_TYPES = ['Pop', 'R&B', 'Dance', 'Jazz', 'Folk', 'Rock', 
                    'Chinese Style', 'Chinese Tradition', 'Metal', 
                    'Reggae', 'Chinese Opera', 'Auto']

BPM_RANGES = {
    'slow': (60, 80),
    'medium': (80, 120),
    'fast': (120, 160)
}
DEFAULT_BPM = 100

# 在常量定义部分添加音乐段落时长配置
MUSIC_SECTION_TEMPLATES = {
    # 纯器乐段落
    "intro-short": {
        "description": "前奏超短版(0-10秒)",
        "duration": "0-10秒",
        "duration_avg": 5,  # (0+10)/2 = 5 取整
        "lyric_required": False
    },
    "intro-medium": {
        "description": "前奏中等版(10-20秒)",
        "duration": "10-20秒",
        "duration_avg": 15,  # (10+20)/2 = 15 取整
        "lyric_required": False
    },
    "intro-long": {
        "description": "前奏完整版(20-30秒)",
        "duration": "20-30秒",
        "duration_avg": 25,  # (20+30)/2 = 25
        "lyric_required": False
    },
    "outro-short": {
        "description": "尾奏超短版(0-10秒)", 
        "duration": "0-10秒",
        "duration_avg": 5,
        "lyric_required": False
    },
    "outro-medium": {
        "description": "尾奏中等版(10-20秒)",
        "duration": "10-20秒",
        "duration_avg": 15,
        "lyric_required": False
    },
    "outro-long": {
        "description": "尾奏完整版(20-30秒)",
        "duration": "20-30秒",
        "duration_avg": 25,
        "lyric_required": False
    },
    "inst-short": {
        "description": "间奏短版(5-12秒)",
        "duration": "5-12秒",
        "duration_avg": 8,
        "lyric_required": False
    },
    "inst-medium": {
        "description": "间奏中等版(10-20秒)",
        "duration": "10-20秒",
        "duration_avg": 15,
        "lyric_required": False
    },
    "inst-long": {
        "description": "间奏完整版(20-30秒)",
        "duration": "20-30秒",
        "duration_avg": 25,
        "lyric_required": False
    },
    "silence": {
        "description": "空白停顿(1-3秒)",
        "duration": "1-3秒",
        "duration_avg": 2,  # 取中间值
        "lyric_required": False
    },
    
    # 人声段落
    "verse": {
        "description": "主歌段落(20-30秒)",
        "duration": "20-30秒",
        "duration_avg": 25,
        "lyric_required": True,
        "lines": "4-8行"
    },
    "pre-chorus": {
        "description": "预副歌", 
        "duration": "6-12秒",
        "duration_avg": 9,
        "lyric_required": True,
        "lines": "1-5行"
    },
    "chorus": {
        "description": "副歌(高潮段落)", 
        "duration": "20-30秒",
        "duration_avg": 25,
        "lyric_required": True,
        "lines": "4-8行"
    },
    "bridge": {
        "description": "过渡桥段",
        "duration": "15-25秒",
        "duration_avg": 20,  # (15+25)/2 = 20
        "lyric_required": True,
        "lines": "2-4行"
    },
}

# 典型结构模板

STRUCTURE_TEMPLATES = {
    # 基础流行结构 (5种)
    "pop_basic": {
        "name": "默认结构",
        "sections": ["intro-short", "verse", "chorus", "inst-short","verse", "chorus", "outro-short"]
    },
    "pop_with_bridge": {
        "name": "流行带桥段结构", 
        "sections": ["intro-short", "verse", "chorus", "verse", "inst-short", "chorus", "bridge", "chorus", "outro-short"]
    },
    "pop_with_prechorus": {
        "name": "流行带预副歌结构",
        "sections": ["intro-short", "verse", "pre-chorus", "chorus", "verse", "inst-short", "pre-chorus", "chorus", "bridge","outro-short"]
    },
    "pop_doublechorus": {
        "name": "流行双副歌结构",
        "sections": ["intro-short", "verse", "pre-chorus", "chorus", "inst-short", "verse", "chorus", "bridge", "outro-short"]
    },
    "pop_postchorus": {
        "name": "流行带后副歌结构",
        "sections": ["intro-short", "verse", "verse", "chorus", "inst-short", "verse", "verse", "chorus", "inst-short", "outro-short"]
    },

    # 短视频/广告音乐 (15-30秒)
    "short_ad": {
        "name": "短视频广告结构",
        "sections": ["intro-short", "chorus", "outro-short"],
        "duration": "15-20秒",
        "use_case": "抖音/快手短视频背景音乐"
    },
    "product_jingle": {
        "name": "产品广告旋律",
        "sections": ["intro-short", "verse", "chorus", "silence", "chorus", "outro-short"],
        "duration": "20-30秒",
        "use_case": "广告口号+品牌记忆点"
    },
 
    # 主流流行音乐 (2.5-3.5分钟)
    "pop_radio": {
        "name": "电台流行结构",
        "sections": ["intro-medium", "verse", "chorus", "verse", "chorus", "bridge", "chorus", "outro-medium"],
        "duration": "约3分钟",
        "feature": "包含桥段增强记忆点"
    },
    "tiktok_pop": {
        "name": "短视频平台流行结构",
        "sections": ["intro-short", "chorus", "verse", "chorus", "inst-short", "chorus", "outro-short"],
        "duration": "约2分30秒",
        "feature": "副歌前置+重复高潮"
    },
 
    # 电子音乐结构
    "edm_build": {
        "name": "EDM情绪构建结构",
        "sections": ["intro-long", "build-up", "drop", "verse", "build-up", "drop", "outro-long"],
        "duration": "3分30秒-4分钟",
        "feature": "包含build-up和drop段落"
    },
    "dance_floor": {
        "name": "舞池专用结构",
        "sections": ["intro-long", "inst-long", "verse", "inst-long", "chorus", "inst-long", "outro-long"],
        "duration": "5-6分钟",
        "feature": "长器乐段落方便DJ混音"
    },
 
    # 嘻哈/说唱结构
    "hiphop_classic": {
        "name": "经典嘻哈结构",
        "sections": ["intro-short", "verse", "chorus", "verse", "chorus", "verse", "outro-short"],
        "duration": "3-4分钟",
        "feature": "三段verse展示歌词技巧"
    },
    "trap_modern": {
        "name": "现代陷阱结构",
        "sections": ["intro-medium", "verse", "hook", "verse", "hook", "bridge", "hook", "outro-medium"],
        "duration": "2分45秒",
        "feature": "使用hook替代传统副歌"
    },
 
    # 影视/游戏配乐
    "film_emotional": {
        "name": "电影情绪结构",
        "sections": ["intro-long", "verse", "inst-long", "verse", "chorus", "outro-long"],
        "duration": "4-5分钟",
        "feature": "长器乐段落营造氛围"
    },
    "game_battle": {
        "name": "游戏战斗音乐",
        "sections": ["intro-short", "inst-medium", "verse", "inst-long", "chorus", "inst-long", "outro-short"],
        "duration": "循环结构",
        "feature": "无尾奏方便循环播放"
    },
 
    # 实验性结构
    "deconstructed": {
        "name": "解构主义结构",
        "sections": ["verse", "silence", "chorus", "inst-short", "silence", "verse"],
        "duration": "不定",
        "feature": "非常规段落排列"
    },
    "minimalist": {
        "name": "极简主义结构",
        "sections": ["intro-long", "verse", "silence", "verse", "silence", "outro-long"],
        "duration": "6-8分钟",
        "feature": "大量留白空间"
    },
 
    # 中国风结构
    "chinese_modern": {
        "name": "现代中国风",
        "sections": ["intro-long", "verse", "chorus", "inst-medium", "verse", "chorus", "outro-long"],
        "duration": "4分钟",
        "feature": "加入传统乐器间奏"
    },
    "folk_ballad": {
        "name": "民谣叙事结构",
        "sections": ["intro-medium", "verse", "verse", "bridge", "verse", "outro-medium"],
        "duration": "5分钟",
        "feature": "多段verse讲述故事"
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
    "falseta": "弗拉门戈吉他独奏段落",
    "build-up": "EDM中逐渐增强音效和节奏的段落(通常8-16小节)",
    "drop": "EDM中高潮释放段落(通常16-32小节)",
    "hook": "重复的简短记忆点(通常2-4小节重复)",
    "silence": "刻意留白(1-3秒制造戏剧效果)",
    "inst-short": "短器乐过渡(5-10秒保持能量)",
    "inst-long": "展示性器乐独奏(20-30秒)"
}