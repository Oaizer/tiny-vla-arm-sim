#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/var/tmp/tiny-vla-python}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/var/tmp/tiny-vla-cache}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/var/tmp/tiny-vla-project-env}"
uv sync --locked
uv run --locked tinyvla run "$@"
