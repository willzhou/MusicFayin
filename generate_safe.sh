#!/bin/bash

# 参数校验必须放在最前面
if [ $# -lt 3 ]; then
    echo "Usage: $0 <ckpt_path> <input_jsonl> <save_dir> [gen_type]" >&2
    exit 2
fi

# 先获取基础参数
CKPT_PATH="$1"
JSONL="$2"
SAVE_DIR="$3"
GEN_TYPE="${4:-all}"

# 现在可以安全使用JSONL变量创建锁文件
LOCK_FILE="/tmp/musicgen_$(echo "$JSONL" | md5sum | cut -d' ' -f1).lock"
if [ -f "$LOCK_FILE" ]; then
    echo "Error: Another instance is processing $JSONL" >&2
    exit 1
fi
trap 'rm -f "$LOCK_FILE"' EXIT
echo $$ > "$LOCK_FILE"

# 环境变量设置
export PYTHONDONTWRITEBYTECODE=1
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$(pwd)/third_party/hub}"
export NCCL_HOME="${NCCL_HOME:-/usr/local/tccl}"

# 安全获取脚本目录
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/SongGeneration"
export PYTHONPATH="$BASE_DIR:${BASE_DIR}/codeclm/tokenizer:${BASE_DIR}:${BASE_DIR}/codeclm/tokenizer/Flow1dVAE${PYTHONPATH:+:$PYTHONPATH}"

# GPU检测（可选）
if [ -x "$(command -v nvidia-smi)" ]; then
    GPU_COUNT=$(nvidia-smi -L | wc -l)
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPU_COUNT-1)))
fi

# 创建输出目录
mkdir -p "$SAVE_DIR"

# 执行日志记录
LOG_FILE="${SAVE_DIR}/generation_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[$(date)] Starting generation with:"
echo "CKPT_PATH: $CKPT_PATH"
echo "JSONL: $JSONL"
echo "SAVE_DIR: $SAVE_DIR"
echo "GEN_TYPE: $GEN_TYPE"

# 执行主程序
python3 "${BASE_DIR}/generate.py" \
    "$CKPT_PATH" \
    "$JSONL" \
    "$SAVE_DIR" \
    "$GEN_TYPE"

echo "[$(date)] Generation completed successfully"
