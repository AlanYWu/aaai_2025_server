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

# Function to find the latest checkpoint in a directory
find_latest_checkpoint() {
	local base_dir="$1"
	if [[ ! -d "${base_dir}" ]]; then
		echo "Error: Directory ${base_dir} does not exist" >&2
		exit 1
	fi
	
	# Find all checkpoint directories and sort by number (highest first)
	local latest_checkpoint=$(find "${base_dir}" -maxdepth 1 -type d -name "checkpoint-*" | \
		sort -t'-' -k2 -n | tail -1)
	
	if [[ -z "${latest_checkpoint}" ]]; then
		echo "Error: No checkpoint found in ${base_dir}" >&2
		exit 1
	fi
	
	echo "${latest_checkpoint}"
}

# Function to update model_name_or_path in a YAML file
update_checkpoint_path() {
	local yaml_file="$1"
	local checkpoint_path="$2"
	
	# Use sed to update the model_name_or_path line
	sed -i.bak "s|model_name_or_path:.*|model_name_or_path: ${checkpoint_path}|" "${yaml_file}"
	rm -f "${yaml_file}.bak"  # Remove backup file
}

# Stage 1
echo "[Stage 1] Training Chinese→Braille using ${THIS_DIR}/stage1_c2b.yaml"
LAUNCH "${THIS_DIR}/stage1_c2b.yaml"

# Find latest checkpoint from Stage 1 and update Stage 2 sentence config
# STAGE1_OUTPUT_DIR="${ROOT_DIR}/saves/Qwen2.5-7B-Instruct-Braille/bidir_v1_20251030_stage1_c2b"
# LATEST_STAGE1_CHECKPOINT=$(find_latest_checkpoint "${STAGE1_OUTPUT_DIR}")
# echo "[Stage 1] Latest checkpoint: ${LATEST_STAGE1_CHECKPOINT}"
# update_checkpoint_path "${THIS_DIR}/stage2_b2c_sentence.yaml" "${LATEST_STAGE1_CHECKPOINT}"

# Stage 2 - Sentence training first
echo "[Stage 2 - Sentence] Training Braille→Chinese (sentence) using ${THIS_DIR}/stage2_b2c_sentence.yaml"
LAUNCH "${THIS_DIR}/stage2_b2c_sentence.yaml"

# Find latest checkpoint from Sentence training and update Stage 2 passage config
SENTENCE_OUTPUT_DIR="${ROOT_DIR}/saves/Qwen2.5-7B-Instruct-Braille/bidir_v1_20251030_stage2_b2c_sentence"
LATEST_SENTENCE_CHECKPOINT=$(find_latest_checkpoint "${SENTENCE_OUTPUT_DIR}")
echo "[Stage 2 - Sentence] Latest checkpoint: ${LATEST_SENTENCE_CHECKPOINT}"
update_checkpoint_path "${THIS_DIR}/stage2_b2c_passage.yaml" "${LATEST_SENTENCE_CHECKPOINT}"

# Stage 2 - Passage training second
echo "[Stage 2 - Passage] Training Braille→Chinese (passage) using ${THIS_DIR}/stage2_b2c_passage.yaml"
LAUNCH "${THIS_DIR}/stage2_b2c_passage.yaml"

echo "Both stages completed. Outputs under saves/Qwen2.5-7B-Instruct-Braille/"



