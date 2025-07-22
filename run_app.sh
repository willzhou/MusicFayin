#!/usr/bin/env bash
# =========== run.sh ===========

# --- 信号处理 ---
cleanup() {
    echo "正在终止进程..."
    kill -TERM "$PID" 2>/dev/null || true
    exit 0
}
trap cleanup INT TERM
 
# --- 主脚本 ---
set -euo pipefail

APP="musicfayin.py"
DEFAULT_PORT=8501
EXTRA_ARGS=("$@")      # 允许用户额外传参给 streamlit

# ---------- 工具函数 ----------
detect_os() {
    case "$(uname -s)" in
        Linux*)
            if grep -qi microsoft /proc/version 2>/dev/null; then
                echo "WSL"
            else
                echo "Linux"
            fi
            ;;
        Darwin*)   echo "macOS" ;;
        CYGWIN*|MINGW*|MSYS*) echo "Windows" ;;
        *)         echo "Unknown" ;;
    esac
}

command -v streamlit >/dev/null 2>&1 || {
    echo "未检测到 streamlit，请先安装："
    echo "   pip install streamlit"
    exit 1
}

# ---------- 主逻辑 ----------
OS=$(detect_os)
BROWSER_CMD=""

echo "当前环境：$OS"

case "$OS" in
    Windows|WSL)
        # 设定 Windows 浏览器
        if [[ "$OS" == "WSL" ]]; then
            WSL_IP=$(hostname -I | awk '{print $1}')
            echo "从 Windows 访问请用：http://${WSL_IP}:${DEFAULT_PORT}"
            echo "（按 Ctrl+Click 可直接在 Windows 浏览器中打开）"
            # 在 WSL 里使用 Windows 侧的 chrome
            BROWSER_PATH="/mnt/c/Program Files/Google/Chrome/Application/chrome.exe"
            [[ -f "$BROWSER_PATH" ]] && BROWSER_CMD="$BROWSER_PATH"
        else
            # 纯 Windows (Git-Bash / MSYS)
            BROWSER_CMD="start"
        fi
        ;;
    macOS)
        BROWSER_CMD="open"
        ;;
    Linux)
        # Linux GUI / 无 GUI 都会识别
        if [[ -n "${DISPLAY:-}" ]]; then
            BROWSER_CMD="xdg-open"
        else
            echo "检测到无 GUI，自动使用 headless 模式。" >&2
            export STREAMLIT_SERVER_HEADLESS=true
        fi
        ;;
    *)
        echo "未知环境，按无 GUI headless 运行。" >&2
        export STREAMLIT_SERVER_HEADLESS=true
        ;;
esac

# ---------- 启动 ----------
HOST=0.0.0.0
PORT=$DEFAULT_PORT

if [[ "${STREAMLIT_SERVER_HEADLESS:-false}" == "false" ]] && [[ -n "$BROWSER_CMD" ]]; then
    # 正常浏览器环境：先起进程，再打开
    streamlit run "$APP" --server.address="$HOST" --server.port="$PORT" "${EXTRA_ARGS[@]}" &
    PID=$!
    sleep 2
    $BROWSER_CMD "http://localhost:$PORT" || true
    wait $PID
else
    # 无 GUI / Docker / CI
    exec streamlit run "$APP" \
         --server.address="$HOST" \
         --server.port="$PORT" \
         --server.headless=true \
         "${EXTRA_ARGS[@]}"
fi
