#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY="${VLA_ENV:-/var/tmp/tiny-vla-env}/bin/python"
"$PY" simulator.py --episodes 12 --video teacher.mp4
"$PY" generate.py --scenes 1200 --output data/train.npz
"$PY" generate.py --scenes 120 --start-seed 10000 --output data/val.npz
"$PY" train.py --steps 3000 --batch 48
"$PY" refit_head.py
"$PY" evaluate.py --episodes 100 --start-seed 20000 --output evaluation
"$PY" evaluate.py --episodes 30 --start-seed 21000 --videos 0 --output ablations/control
"$PY" evaluate.py --episodes 30 --start-seed 21000 --videos 0 --ablate wrong_language --output ablations/wrong_language
"$PY" evaluate.py --episodes 30 --start-seed 21000 --videos 0 --ablate blank_image --output ablations/blank_image
