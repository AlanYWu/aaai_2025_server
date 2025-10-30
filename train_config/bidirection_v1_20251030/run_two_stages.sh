#!/usr/bin/env bash
set -euo pipefail

# One-command training for bidirectional v1 (2025-10-30)
# Stage 1: Chinese→Braille, Stage 2: Braille→Chinese

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${THIS_DIR}/../../" && pwd)"

# If you use a different launcher (e.g., python -m llamafactory), replace the command below accordingly.
LAUNCH() {
	FORCE_TORCHRUN=1 llamafactory-cli train "$@"
}

# Stage 1
echo "[Stage 1] Training Chinese→Braille using ${THIS_DIR}/stage1_c2b.yaml"
LAUNCH "${THIS_DIR}/stage1_c2b.yaml"

# Optionally, you can parse the latest checkpoint to pass to stage 2.
# For now, stage2 YAML points to a fixed checkpoint path. Adjust if needed.

# Stage 2
echo "[Stage 2] Training Braille→Chinese using ${THIS_DIR}/stage2_b2c.yaml"
LAUNCH "${THIS_DIR}/stage2_b2c.yaml"

echo "Both stages completed. Outputs under saves/Qwen2.5-7B-Instruct-Braille/"
