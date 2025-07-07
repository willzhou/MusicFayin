# MusicFayIn - AI Music Generation System

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.12+-green.svg)
![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)

MusicFayIn is an advanced AI-powered music generation system that creates complete musical compositions from lyrics and style parameters.

## Features

- 🎵 **Lyrics-to-Music Generation**: Transform text lyrics into complete musical compositions
- 🎚️ **Style Control**: Adjust genre, emotion, instrumentation, and vocal characteristics
- 🎛️ **Structure Templates**: 36 predefined song structure templates across multiple genres
- 🎙️ **Multi-Prompt Generation**: Supports text, audio, and automatic prompting
- 🖥️ **Web Interface**: Streamlit-based UI for easy interaction

## Installation

1. Install core dependencies (requires SongGeneration)
```bash
git clone https://github.com/tencent-ailab/SongGeneration.git
cd SongGeneration
pip install -r requirements.txt
```

2. Install MusicFayIn extensions
```git clone https://github.com/your-repo/MusicFayIn.git
cd MusicFayIn
pip install .
```

3. Download model checkpoints and place them in the `ckpt/` directory following this structure:
```
ckpt/
├── model_1rvq/
│   └── model_2_fixed.safetensors
├── model_septoken/
│   └── model_2.safetensors
├── prompt.pt
└── songgeneration_base/
    ├── config.yaml
    └── model.pt
```

## Usage

Run the Streamlit application:
```bash
streamlit run MusicFayIn/musicfayin.py
```

The workflow consists of 5 steps:

1. **Lyrics Generation**: Input a theme and select a song structure template
2. **Lyrics Analysis**: AI analyzes lyrics for emotion, genre, and instrumentation
3. **Parameter Adjustment**: Fine-tune musical parameters
4. **Configuration Generation**: Create JSONL configuration files
5. **Music Generation**: Generate complete musical compositions

## Supported Music Styles

### Genres
- Pop (5 structure variations)
- Rock/Metal (8 variations)
- Electronic (7 variations)
- Hip-hop/Rap (5 variations)
- Chinese Traditional (6 variations)
- Jazz/Blues (5 variations)

### Instrumentations
36 combinations including piano, guitar, synthesizer, strings, and more

## Technical Architecture

![System Architecture](docs/architecture.png)

The system uses a multi-stage generation pipeline:
1. **Lyrics Processing**: DeepSeek API for lyric generation and analysis
2. **Tokenization**: Custom audio tokenizers for melody and accompaniment
3. **Generation**: Transformer-based music generation model
4. **Separation**: Audio source separation for enhanced quality

## Configuration

Key configuration files:
- `STRUCTURE_TEMPLATES`: 36 predefined song structures
- `MUSIC_SECTION_TEMPLATES`: Duration and content specifications
- `DEEPSEEK_API_KEY`: Set your API key for lyric generation

## Requirements

- Python 3.8+
- PyTorch 1.10+
- Streamlit 1.12+
- CUDA 11.3+ (for GPU acceleration)
- 16GB+ RAM (32GB recommended)
- NVIDIA GPU with 8GB+ VRAM

## License

Copyright 2025 Ningbo Wise Effects, Inc. (汇视创影)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

## Acknowledgements

We sincerely thank:
- **Tencent AI Lab** for open-sourcing the foundational SongGeneration framework
- The open-source community for valuable contributions and feedback
- Our professional music consultants for structure validation

Special gratitude to all contributors who made this project possible.

# MusicFayIn - AI 音乐生成系统

## 项目简介
MusicFayIn 是基于腾讯 AI Lab 开源项目 [SongGeneration](https://github.com/tencent-ailab/SongGeneration) 开发的 AI 音乐生成系统。

## 核心技术
本系统核心算法基于以下开源项目：
- **SongGeneration**：腾讯 AI Lab 开发的音乐生成框架 [GitHub 链接](https://github.com/tencent-ailab/SongGeneration)
- **改进点**：
  - 新增 36 种专业音乐结构模板
  - 优化了中文歌词适配能力
  - 增强了风格控制模块

## 主要功能
1. **智能音乐生成**
   - 基于 SongGeneration 核心引擎
   - 支持歌词驱动和风格引导两种创作模式

2. **36 种专业音乐结构**
   - 流行/摇滚/电子/中国风/爵士等分类
   - 每种结构都经过专门验证

3. **增强功能**
   - 段落时长精确控制
   - 风格混合与转换
   - 华语音乐特别优化

## 致谢
特别感谢腾讯 AI Lab 团队的开源贡献，SongGeneration 项目为本系统提供了核心技术支持。

## 许可证
本项目遵循Apache2.0开源协议；SongGeneration 的部分请遵循相关协议。
