#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Linux / WSL. The runtime lives outside /tmp so WSL restarts do not remove it.
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/var/tmp/tiny-vla-python}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/var/tmp/tiny-vla-cache}"
VLA_ENV="${VLA_ENV:-/var/tmp/tiny-vla-env}"
if [[ ! -x "$VLA_ENV/bin/python" ]]; then uv venv "$VLA_ENV" --python 3.11; fi
uv pip install --python "$VLA_ENV/bin/python" torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python "$VLA_ENV/bin/python" -r requirements.txt
"$VLA_ENV/bin/python" -c 'import torch, pybullet; print("Runtime ready:", torch.__version__)'
