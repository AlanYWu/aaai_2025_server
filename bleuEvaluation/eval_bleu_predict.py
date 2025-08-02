# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import time
import os

import fire
from datasets import load_dataset


try:
    import jieba  # type: ignore
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu  # type: ignore
    from rouge_chinese import Rouge  # type: ignore

    jieba.setLogLevel(logging.CRITICAL)
    jieba.initialize()
except ImportError:
    print("Please install llamafactory with `pip install -e .[metrics]`.")
    raise


def compute_metrics_old(sample):
    hypothesis = list(jieba.cut(sample["predict"]))
    reference = list(jieba.cut(sample["label"]))

    bleu_score = sentence_bleu(
        [list(sample["label"])],
        list(sample["predict"]),
        smoothing_function=SmoothingFunction().method3,
    )

    if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
        result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
    else:
        rouge = Rouge()
        scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
        result = scores[0]

    metric_result = {}
    for k, v in result.items():
        metric_result[k] = round(v["f"] * 100, 4)

    metric_result["bleu-4"] = round(bleu_score * 100, 4)

    return metric_result

def extract_cn(text: str) -> str:
    """
    Keep only the part after '对应的中文内容是:\\n'.
    Falls back to the whole string if the tag is missing.
    """
    return text.lower().split("对应的中文内容是:\n")[-1].strip()

def compute_metrics(sample):
    # -------- plain strings --------
    ref_text = extract_cn(sample["label"])
    # hyp_text = extract_cn(sample["polish_output"])   #  <-- your field
    hyp_text = extract_cn(sample["predict"])   #  <-- your field

    # -------- char‑level tokens for BLEU‑4 --------
    ref_chars = list(ref_text)
    hyp_chars = list(hyp_text)

    bleu_score = sentence_bleu(
        [ref_chars],
        hyp_chars,
        smoothing_function=SmoothingFunction().method3,
    )

    # -------- word‑level tokens (jieba) for ROUGE --------
    reference  = list(jieba.cut(ref_text))
    hypothesis = list(jieba.cut(hyp_text))

    if not reference or not hypothesis:
        rouge_result = {"rouge-1": {"f": 0.0},
                        "rouge-2": {"f": 0.0},
                        "rouge-l": {"f": 0.0}}
    else:
        rouge = Rouge()
        rouge_result = rouge.get_scores(" ".join(hypothesis),
                                        " ".join(reference))[0]

    # -------- pack the numbers the same way the template expects --------
    metric_result = {k: round(v["f"] * 100, 4) for k, v in rouge_result.items()}
    metric_result["bleu-4"] = round(bleu_score * 100, 4)
    return metric_result


def main(filename: str):
    start_time = time.time()
    dataset = load_dataset("json", data_files=filename, split="train")
    # Compute metrics and keep original fields
    def add_metrics(sample):
        metrics = compute_metrics(sample)
        sample.update(metrics)
        return sample

    dataset = dataset.map(add_metrics, num_proc=16)
    # Save all results
    dataset.to_json("detailed_predictions.json", force_ascii=False)

    score_dict = dataset.to_dict()

    average_score = {}
    for task, scores in sorted(score_dict.items(), key=lambda x: x[0]):
        # Skip non-numeric values and calculate average only for numeric scores
        numeric_scores = [s for s in scores if isinstance(s, (int, float))]
        if numeric_scores:
            avg = sum(numeric_scores) / len(numeric_scores)
            print(f"{task}: {avg:.4f}")
            average_score[task] = avg

    # Get the directory of the input file
    output_dir = os.path.dirname(filename)
    score_file = os.path.join(output_dir, "predictions_score.json")
    
    with open(score_file, "w", encoding="utf-8") as f:
        json.dump(average_score, f, indent=4)

    print(f"\nDone in {time.time() - start_time:.3f}s.\nScore file saved to {score_file}")


if __name__ == "__main__":
    fire.Fire(main)