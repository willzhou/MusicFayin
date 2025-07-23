#!/bin/bash
echo "=== PyTorch / CUDA / libc10.so ==="
python - <<'PY'
import torch, pathlib, os
print("PyTorch:", torch.__version__)
print("CUDA   :", torch.version.cuda)
lib_path = pathlib.Path(torch.__file__).parent/"lib"
print("libc10.so found:", (lib_path/"libc10.so").exists())
print("TORCH_LIB:", lib_path)
PY

