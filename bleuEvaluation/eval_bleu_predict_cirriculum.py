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
import math

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


def evaluate_segment(segment_data, segment_name):
    """Evaluate a single segment and return average metrics"""
    if not segment_data:
        return None
    
    # Compute metrics for each sample in the segment
    segment_with_metrics = []
    for sample in segment_data:
        metrics = compute_metrics(sample)
        sample.update(metrics)
        segment_with_metrics.append(sample)
    
    # Calculate average scores for the segment
    score_dict = {}
    for key in segment_with_metrics[0].keys():
        if key in ['rouge-1', 'rouge-2', 'rouge-l', 'bleu-4']:
            scores = [sample[key] for sample in segment_with_metrics if isinstance(sample[key], (int, float))]
            if scores:
                avg = sum(scores) / len(scores)
                score_dict[key] = round(avg, 4)
    
    return score_dict, segment_with_metrics


def main(filename: str):
    start_time = time.time()
    dataset = load_dataset("json", data_files=filename, split="train")
    
    # Convert to list for easier manipulation
    data_list = list(dataset)
    total_samples = len(data_list)
    
    # Calculate segment size (10 segments total)
    segment_size = math.ceil(total_samples / 10)
    
    # Define segment names corresponding to tone percentages
    segment_names = ["100pc", "90pc", "80pc", "70pc", "60pc", "50pc", "40pc", "30pc", "20pc", "10pc"]
    
    print(f"Total samples: {total_samples}")
    print(f"Segment size: {segment_size}")
    print("=" * 50)
    
    all_results = {}
    detailed_results = {}
    
    # Process each segment
    for i, segment_name in enumerate(segment_names):
        start_idx = i * segment_size
        end_idx = min((i + 1) * segment_size, total_samples)
        
        if start_idx >= total_samples:
            break
            
        segment_data = data_list[start_idx:end_idx]
        print(f"\nProcessing {segment_name} (samples {start_idx+1}-{end_idx}): {len(segment_data)} samples")
        
        # Evaluate the segment
        segment_result = evaluate_segment(segment_data, segment_name)
        
        if segment_result:
            score_dict, detailed_segment = segment_result
            all_results[segment_name] = score_dict
            detailed_results[segment_name] = detailed_segment
            
            # Print segment results
            print(f"{segment_name} Results:")
            for metric, score in score_dict.items():
                print(f"  {metric}: {score:.4f}")
    
    # Save detailed results for each segment
    for segment_name, segment_data in detailed_results.items():
        segment_filename = f"detailed_predictions_{segment_name}.json"
        with open(segment_filename, "w", encoding="utf-8") as f:
            json.dump(segment_data, f, indent=2, ensure_ascii=False)
        print(f"Detailed results for {segment_name} saved to {segment_filename}")
    
    # Save overall results
    output_dir = os.path.dirname(filename)
    score_file = os.path.join(output_dir, "predictions_score_segmented.json")
    
    with open(score_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
    
    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY BY SEGMENT:")
    print("=" * 50)
    for segment_name, scores in all_results.items():
        print(f"\n{segment_name}:")
        for metric, score in scores.items():
            print(f"  {metric}: {score:.4f}")
    
    print(f"\nDone in {time.time() - start_time:.3f}s.")
    print(f"Score file saved to {score_file}")


if __name__ == "__main__":
    fire.Fire(main)