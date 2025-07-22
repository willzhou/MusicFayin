#!/bin/bash
## 修复fairseq安装脚本
pip install pip==24.0
pip cache purge
pip install \
	"omegaconf==2.2.0" \
	"hydra-core==1.0.7" \
	fairseq==0.12.2
pip install git+https://github.com/descriptinc/audiotools

