#!/usr/bin/env python3

import argparse
import json
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Iterable


def read_json_list(path: Path, head_limit: int = 0) -> List[Dict[str, Any]]:
	with path.open('r', encoding='utf-8') as f:
		data = json.load(f)
		if not isinstance(data, list):
			raise ValueError(f"Expected a list at {path}")
		if head_limit > 0:
			return data[:head_limit]
		return data


def extract_braille_and_chinese_from_messages(messages: List[Dict[str, str]]) -> Tuple[str, str]:
	"""
	Given a `messages` chat array with the format observed in datasets,
	return (braille_text, chinese_text) without the trailing tokens.
	Robustly handles both Chinese and English prompt wordings.
	"""
	if not messages or len(messages) < 3:
		raise ValueError("messages must contain at least system/user/assistant")

	user_content = messages[1].get('content', '')
	assistant_content = messages[2].get('content', '')

	# Extract braille between header and <BRAILLE_END>
	# Support two headers: "Braille content:" and "盲文内容是:" (or minor variations)
	braille_text = ''
	# Common marker
	braille_end_token = '<BRAILLE_END>'
	# Try to locate the content after a header line
	# We'll search the last occurrence of header keyword to be robust
	user_lines = user_content.splitlines()
	start_idx = None
	for i, line in enumerate(user_lines):
		if re.search(r"Braille\s*content\s*:\s*$", line) or re.search(r"盲文内容是\s*:\s*$", line):
			start_idx = i + 1
		# Some data may inline immediately after colon
		if re.search(r"Braille\s*content\s*:\s*", line) or re.search(r"盲文内容是\s*:\s*", line):
			# If inline, take substring after colon
			m = re.search(r":\s*(.*)$", line)
			if m:
				braille_line = m.group(1)
				if braille_end_token in braille_line:
					braille_text = braille_line.split(braille_end_token)[0].strip()
					start_idx = None
					break
				else:
					start_idx = i + 1

	if braille_text == '' and start_idx is not None:
		concat = '\n'.join(user_lines[start_idx:])
		if braille_end_token in concat:
			braille_text = concat.split(braille_end_token)[0].strip()
		else:
			braille_text = concat.strip()

	# Extract chinese content before <TRANSLATION_END>
	translation_end_token = '<TRANSLATION_END>'
	# assistant typically starts with a header line like "对应的中文内容是："
	if translation_end_token in assistant_content:
		body = assistant_content.split(translation_end_token)[0]
	else:
		body = assistant_content
	# Remove common leading header
	body = re.sub(r"^\s*对应的中文内容是：\s*", "", body)
	chinese_text = body.strip()

	if not braille_text or not chinese_text:
		raise ValueError("Failed to parse braille/chinese from messages")

	return braille_text, chinese_text


def make_b2c_sample(braille_text: str, chinese_text: str) -> Dict[str, Any]:
	return {
		"messages": [
			{"role": "system", "content": "你是一个中国盲文翻译助手，请把通用盲文转换成为汉字。"},
			{"role": "user", "content": f"请把以下通用盲文转换成为汉字：\n盲文内容是:\n{braille_text}<BRAILLE_END>"},
			{"role": "assistant", "content": f"对应的中文内容是：\n{chinese_text}<TRANSLATION_END>"},
		]
	}


def make_c2b_sample(braille_text: str, chinese_text: str) -> Dict[str, Any]:
	return {
		"messages": [
			{"role": "system", "content": "你是一个中国盲文翻译助手，请把汉字转换成为通用盲文。"},
			{"role": "user", "content": f"请把以下中文转换成为通用盲文：\n中文内容是:\n{chinese_text}<CHINESE_END>"},
			{"role": "assistant", "content": f"对应的盲文内容是：\n{braille_text}<TRANSLATION_END>"},
		]
	}


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
	with path.open('w', encoding='utf-8') as f:
		for row in rows:
			f.write(json.dumps(row, ensure_ascii=False))
			f.write('\n')


def write_metadata(path: Path, *, direction: str, total: int, split_counts: Dict[str, int], sources: List[str], special_tokens: List[str]) -> None:
	created = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
	# Minimal YAML without external deps
	yaml_lines = []
	yaml_lines.append(f"name: {path.parent.name}\n")
	yaml_lines.append(f"direction: {direction}\n")
	yaml_lines.append(f"num_samples: {total}\n")
	yaml_lines.append("splits:\n")
	for k in ["train", "validation", "test"]:
		v = split_counts.get(k, 0)
		yaml_lines.append(f"  {k}: {v}\n")
	yaml_lines.append(f"created_utc: {created}\n")
	yaml_lines.append(f"sources:\n")
	for s in sources:
		yaml_lines.append(f"  - {s}\n")
	yaml_lines.append(f"prompt_schema: chat-3role-v1\n")
	yaml_lines.append(f"special_tokens:\n")
	for t in special_tokens:
		yaml_lines.append(f"  - {t}\n")
	with path.open('w', encoding='utf-8') as f:
		f.write(''.join(yaml_lines))


def write_readme(path: Path, *, direction: str, sources: List[str]) -> None:
	lines = []
	lines.append(f"# {path.parent.name}\n")
	lines.append(f"方向: {direction}\n")
	lines.append("数据来源:\n")
	for s in sources:
		lines.append(f"- {s}\n")
	lines.append("\n样本格式: 三段消息 (system/user/assistant)。输出以 <TRANSLATION_END> 结束，输入侧根据模态包含 <BRAILLE_END> 或 <CHINESE_END>。\n")
	lines.append("\n划分: 8:1:1 (train:validation:test)。每个方向下分别输出 sentence_* 与 passage_* 三个切分文件。\n")
	with path.open('w', encoding='utf-8') as f:
		f.write(''.join(lines))


def ensure_dir(path: Path) -> None:
	path.mkdir(parents=True, exist_ok=True)


def split_rows(rows: List[Dict[str, Any]], *, seed: int, ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)) -> Dict[str, List[Dict[str, Any]]]:
	# No random shuffle: keep original order
	n = len(rows)
	n_train = int(n * ratios[0])
	n_val = int(n * ratios[1])
	n_test = n - n_train - n_val
	return {
		"train": rows[:n_train],
		"validation": rows[n_train:n_train + n_val],
		"test": rows[n_train + n_val:],
	}


def load_many(paths: Iterable[Path], head_limit: int = 0) -> List[Dict[str, Any]]:
	items: List[Dict[str, Any]] = []
	for p in paths:
		if not p.exists():
			continue
		items.extend(read_json_list(p, head_limit=head_limit))
	return items


def build_datasets(
		passage_paths: List[Path],
		sentence_paths: List[Path],
		output_root: Path,
		name_prefix: str = "bidirectional",
		random_seed: int = 17,
		limit_each: int = 0,
	) -> None:
	# Read (order preserved)
	passage_items = load_many(passage_paths, head_limit=limit_each)
	sentence_items = load_many(sentence_paths, head_limit=limit_each)

	def to_pairs(items: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
		pairs: List[Tuple[str, str]] = []
		for it in items:
			try:
				braille, chinese = extract_braille_and_chinese_from_messages(it.get('messages', []))
				pairs.append((braille, chinese))
			except Exception:
				# Skip malformed
				continue
		return pairs

	passage_pairs = to_pairs(passage_items)
	sentence_pairs = to_pairs(sentence_items)

	if not passage_pairs:
		raise RuntimeError("No valid pairs parsed from passage datasets")
	if not sentence_pairs:
		raise RuntimeError("No valid pairs parsed from sentence datasets")

	# Split each source into halves: first half → b2c, second half → c2b
	def halves(pairs: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
		n = len(pairs)
		h = n // 2
		return pairs[:h], pairs[h:]

	passage_b2c_pairs, passage_c2b_pairs = halves(passage_pairs)
	sentence_b2c_pairs, sentence_c2b_pairs = halves(sentence_pairs)

	# Build rows per source and direction (order preserved)
	passage_b2c_rows: List[Dict[str, Any]] = [make_b2c_sample(b, c) for (b, c) in passage_b2c_pairs]
	passage_c2b_rows: List[Dict[str, Any]] = [make_c2b_sample(b, c) for (b, c) in passage_c2b_pairs]
	sentence_b2c_rows: List[Dict[str, Any]] = [make_b2c_sample(b, c) for (b, c) in sentence_b2c_pairs]
	sentence_c2b_rows: List[Dict[str, Any]] = [make_c2b_sample(b, c) for (b, c) in sentence_c2b_pairs]

	# Prepare output dirs
	date_tag = datetime.utcnow().strftime('%Y%m%d')
	b2c_name = f"{name_prefix}_braille_to_chinese_v1_{date_tag}"
	c2b_name = f"{name_prefix}_chinese_to_braille_v1_{date_tag}"

	b2c_dir = output_root / b2c_name
	c2b_dir = output_root / c2b_name
	ensure_dir(b2c_dir)
	ensure_dir(c2b_dir)

	# Split and write: B2C
	b2c_sentence_splits = split_rows(sentence_b2c_rows, seed=random_seed)
	b2c_passage_splits = split_rows(passage_b2c_rows, seed=random_seed)
	write_jsonl(b2c_dir / 'sentence_train.jsonl', b2c_sentence_splits['train'])
	write_jsonl(b2c_dir / 'sentence_validation.jsonl', b2c_sentence_splits['validation'])
	write_jsonl(b2c_dir / 'sentence_test.jsonl', b2c_sentence_splits['test'])
	write_jsonl(b2c_dir / 'passage_train.jsonl', b2c_passage_splits['train'])
	write_jsonl(b2c_dir / 'passage_validation.jsonl', b2c_passage_splits['validation'])
	write_jsonl(b2c_dir / 'passage_test.jsonl', b2c_passage_splits['test'])
	write_metadata(
		b2c_dir / 'metadata.yaml',
		direction='Braille→Chinese',
		total=len(sentence_b2c_rows) + len(passage_b2c_rows),
		split_counts={
			"train": len(b2c_sentence_splits['train']) + len(b2c_passage_splits['train']),
			"validation": len(b2c_sentence_splits['validation']) + len(b2c_passage_splits['validation']),
			"test": len(b2c_sentence_splits['test']) + len(b2c_passage_splits['test']),
		},
		sources=[str(p) for p in passage_paths + sentence_paths],
		special_tokens=['<BRAILLE_END>', '<CHINESE_END>', '<TRANSLATION_END>'],
	)
	write_readme(b2c_dir / 'README.md', direction='Braille→Chinese', sources=[str(p) for p in passage_paths + sentence_paths])

	# Split and write: C2B
	c2b_sentence_splits = split_rows(sentence_c2b_rows, seed=random_seed)
	c2b_passage_splits = split_rows(passage_c2b_rows, seed=random_seed)
	write_jsonl(c2b_dir / 'sentence_train.jsonl', c2b_sentence_splits['train'])
	write_jsonl(c2b_dir / 'sentence_validation.jsonl', c2b_sentence_splits['validation'])
	write_jsonl(c2b_dir / 'sentence_test.jsonl', c2b_sentence_splits['test'])
	write_jsonl(c2b_dir / 'passage_train.jsonl', c2b_passage_splits['train'])
	write_jsonl(c2b_dir / 'passage_validation.jsonl', c2b_passage_splits['validation'])
	write_jsonl(c2b_dir / 'passage_test.jsonl', c2b_passage_splits['test'])
	write_metadata(
		c2b_dir / 'metadata.yaml',
		direction='Chinese→Braille',
		total=len(sentence_c2b_rows) + len(passage_c2b_rows),
		split_counts={
			"train": len(c2b_sentence_splits['train']) + len(c2b_passage_splits['train']),
			"validation": len(c2b_sentence_splits['validation']) + len(c2b_passage_splits['validation']),
			"test": len(c2b_sentence_splits['test']) + len(c2b_passage_splits['test']),
		},
		sources=[str(p) for p in passage_paths + sentence_paths],
		special_tokens=['<BRAILLE_END>', '<CHINESE_END>', '<TRANSLATION_END>'],
	)
	write_readme(c2b_dir / 'README.md', direction='Chinese→Braille', sources=[str(p) for p in passage_paths + sentence_paths])

	print(f"[B2C] sentence: train {len(b2c_sentence_splits['train'])} val {len(b2c_sentence_splits['validation'])} test {len(b2c_sentence_splits['test'])}")
	print(f"[B2C] passage:  train {len(b2c_passage_splits['train'])} val {len(b2c_passage_splits['validation'])} test {len(b2c_passage_splits['test'])}")
	print(f"[C2B] sentence: train {len(c2b_sentence_splits['train'])} val {len(c2b_sentence_splits['validation'])} test {len(c2b_sentence_splits['test'])}")
	print(f"[C2B] passage:  train {len(c2b_passage_splits['train'])} val {len(c2b_passage_splits['validation'])} test {len(c2b_passage_splits['test'])}")


def main() -> None:
	parser = argparse.ArgumentParser(description='Generate bidirectional datasets (Braille↔Chinese) splitting each source (sentence/passage) into halves without shuffling.')
	parser.add_argument('--passage', type=str, nargs='*', default=[], help='Paths to passage JSON files (train/val/test etc)')
	parser.add_argument('--sentence', type=str, nargs='*', default=[], help='Paths to sentence JSON files (train/val/test etc)')
	parser.add_argument('--auto_old_100pc', action='store_true', help='Auto-include standard 100pc train/val/test files for sentence and passage')
	parser.add_argument('--out_root', type=str, default='datasets', help='Output root directory')
	parser.add_argument('--seed', type=int, default=17)
	parser.add_argument('--limit_each', type=int, default=0, help='Limit records per source file (0 means all)')
	args = parser.parse_args()

	passage_paths: List[Path] = [Path(p).expanduser().resolve() for p in args.passage]
	sentence_paths: List[Path] = [Path(s).expanduser().resolve() for s in args.sentence]

	if args.auto_old_100pc:
		passage_paths.extend([
			Path('data/Passage_dataset/passage_100pc_train_0727_v2.json').resolve(),
			Path('data/Passage_dataset/passage_100pc_val_0727_v2.json').resolve(),
			Path('data/Passage_dataset/passage_100pc_test_0727_v2.json').resolve(),
		])
		sentence_paths.extend([
			Path('data/Sentence_dataset/set-Full-Tone/sentence_100pc_train_0727_v2.json').resolve(),
			Path('data/Sentence_dataset/set-Full-Tone/sentence_100pc_val_0727_v2.json').resolve(),
			Path('data/Sentence_dataset/set-Full-Tone/sentence_100pc_test_0727_v2.json').resolve(),
		])

	# Filter non-existent
	passage_paths = [p for p in passage_paths if p.exists()]
	sentence_paths = [s for s in sentence_paths if s.exists()]

	if not passage_paths:
		print("ERROR: no passage files provided/found", file=sys.stderr)
		sys.exit(1)
	if not sentence_paths:
		print("ERROR: no sentence files provided/found", file=sys.stderr)
		sys.exit(1)

	out_root = Path(args.out_root).expanduser().resolve()
	ensure_dir(out_root)
	build_datasets(passage_paths, sentence_paths, out_root, random_seed=args.seed, limit_each=args.limit_each)


if __name__ == '__main__':
	main()
