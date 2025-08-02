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
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
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


def extract_cn(text: str) -> str:
    """
    Keep only the part after '对应的中文内容是:\\n'.
    Falls back to the whole string if the tag is missing.
    """
    return text.lower().split("对应的中文内容是:\n")[-1].strip()


def compute_metrics(sample):
    # -------- plain strings --------
    ref_text = extract_cn(sample["label"])
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


def perform_anova_analysis(model_results, metrics=['rouge-1', 'rouge-2', 'rouge-l', 'bleu-4']):
    """Perform ANOVA analysis for each metric across models"""
    anova_results = {}
    
    for metric in metrics:
        # Prepare data for ANOVA
        groups = []
        group_names = []
        
        for model_name, model_data in model_results.items():
            # Extract all individual scores for this metric across all segments
            scores = []
            for segment_name, segment_data in model_data['detailed_results'].items():
                for sample in segment_data:
                    if metric in sample and isinstance(sample[metric], (int, float)):
                        scores.append(sample[metric])
            
            if scores:  # Only include if we have scores
                groups.append(scores)
                group_names.append(model_name)
        
        if len(groups) >= 2:  # Need at least 2 groups for ANOVA
            # Perform one-way ANOVA
            f_stat, p_value = stats.f_oneway(*groups)
            
            # Calculate effect size (eta squared)
            total_n = sum(len(group) for group in groups)
            total_mean = np.mean([score for group in groups for score in group])
            ss_between = sum(len(group) * ((np.mean(group) - total_mean) ** 2) for group in groups)
            ss_total = sum((score - total_mean) ** 2 for group in groups for score in group)
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
            
            anova_results[metric] = {
                'f_statistic': f_stat,
                'p_value': p_value,
                'eta_squared': eta_squared,
                'significant': p_value < 0.05,
                'group_means': {name: np.mean(group) for name, group in zip(group_names, groups)},
                'group_stds': {name: np.std(group) for name, group in zip(group_names, groups)}
            }
    
    return anova_results


def create_comparison_plots(model_results, anova_results, output_dir="."):
    """Create comparison plots with colored areas under the lines"""
    metrics = ['rouge-1', 'rouge-2', 'rouge-l', 'bleu-4']
    segment_names = ["100pc", "90pc", "80pc", "70pc", "60pc", "50pc", "40pc", "30pc", "20pc", "10pc"]
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    for metric in metrics:
        plt.figure(figsize=(6, 4))
        
        # Create the main plot
        ax = plt.gca()
        # Store handles for manual legend control
        handles = []
        labels = []
        
        # Plot each model
        for i, (model_name, model_data) in enumerate(model_results.items()):
            # Extract segment averages for this metric
            segment_scores = []
            for segment_name in segment_names:
                if segment_name in model_data['results'] and metric in model_data['results'][segment_name]:
                    segment_scores.append(model_data['results'][segment_name][metric])
                else:
                    segment_scores.append(0)  # or np.nan if you prefer
            
            # Create x-axis values (percentage values)
            x_values = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
            
            # Plot the line with colored area underneath
            color = plt.cm.Set1(i / len(model_results))
            line, = ax.plot(x_values, segment_scores, marker='o', linewidth=3, 
                           label=model_name, color=color, markersize=8)
            
            # Fill area under the line
            ax.fill_between(x_values, segment_scores, alpha=0.3, color=color)
            
            handles.append(line)
            labels.append(model_name)
        
        # Customize the plot
        ax.set_xlabel('Tone Percentage', fontsize=14, fontweight='bold')
        # ax.set_ylabel(f'{metric.upper()} Score', fontsize=14, fontweight='bold')
        # ax.set_title(f'{metric.upper()} Comparison Across Models', fontsize=16, fontweight='bold')
        ax.set_ylabel('BLEU Score', fontsize=14, fontweight='bold')
        ax.set_title('BLEU Comparison Across Models', fontsize=16, fontweight='bold')

        order = [1, 0]  # if you want second plotted line first
        ax.legend([handles[i] for i in order],
                [labels[i] for i in order],
                fontsize=12, frameon=True, fancybox=True, shadow=True)
        ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
        # Set y-axis range from 80 to 100
        ax.set_ylim(80, 100)
        
        # Set x-axis to reverse order (100% to 10%)
        ax.set_xlim(105, 5)
        ax.invert_xaxis()
        
        # Add ANOVA results as text if available
        if metric in anova_results:
            anova_info = anova_results[metric]
            if anova_info['significant']:
                significance_text = f"ANOVA: F={anova_info['f_statistic']:.3f}, p={anova_info['p_value']:.4f}*"
            else:
                significance_text = f"ANOVA: F={anova_info['f_statistic']:.3f}, p={anova_info['p_value']:.4f}"
            
            plt.figtext(0.02, 0.02, significance_text, fontsize=10, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Save the plot
        plot_filename = os.path.join(output_dir, f"{metric}_model_comparison.png")
        plt.tight_layout()
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plot saved: {plot_filename}")


def main(model_files: str, output_dir: str = "."):
    """
    Compare multiple models using ANOVA and create plots.
    
    Args:
        model_files: Comma-separated list of model prediction files
        output_dir: Directory to save results and plots
    """
    start_time = time.time()
    
    # Parse model files
    model_file_list = [f.strip() for f in model_files.split(',')]
    model_names = [os.path.splitext(os.path.basename(f))[0] for f in model_file_list]
    
    print(f"Comparing {len(model_file_list)} models:")
    for name, file in zip(model_names, model_file_list):
        print(f"  {name}: {file}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each model
    model_results = {}
    
    for model_name, filename in zip(model_names, model_file_list):
        print(f"\nProcessing {model_name}...")
        
        dataset = load_dataset("json", data_files=filename, split="train")
        data_list = list(dataset)
        total_samples = len(data_list)
        segment_size = math.ceil(total_samples / 10)
        segment_names = ["100pc", "90pc", "80pc", "70pc", "60pc", "50pc", "40pc", "30pc", "20pc", "10pc"]
        
        all_results = {}
        detailed_results = {}
        
        # Process each segment
        for i, segment_name in enumerate(segment_names):
            start_idx = i * segment_size
            end_idx = min((i + 1) * segment_size, total_samples)
            
            if start_idx >= total_samples:
                break
                
            segment_data = data_list[start_idx:end_idx]
            
            # Evaluate the segment
            segment_result = evaluate_segment(segment_data, segment_name)
            
            if segment_result:
                score_dict, detailed_segment = segment_result
                all_results[segment_name] = score_dict
                detailed_results[segment_name] = detailed_segment
        
        model_results[model_name] = {
            'results': all_results,
            'detailed_results': detailed_results
        }
        
        print(f"Completed {model_name} evaluation")
    
    # Perform ANOVA analysis
    print("\nPerforming ANOVA analysis...")
    anova_results = perform_anova_analysis(model_results)
    
    # Print ANOVA results
    print("\n" + "=" * 60)
    print("ANOVA RESULTS:")
    print("=" * 60)
    for metric, result in anova_results.items():
        print(f"\n{metric.upper()}:")
        print(f"  F-statistic: {result['f_statistic']:.4f}")
        print(f"  p-value: {result['p_value']:.6f}")
        print(f"  Significant: {'Yes' if result['significant'] else 'No'}")
        print(f"  Effect size (η²): {result['eta_squared']:.4f}")
        print("  Group means:")
        for model, mean in result['group_means'].items():
            std = result['group_stds'][model]
            print(f"    {model}: {mean:.4f} ± {std:.4f}")
    
    # Create comparison plots
    print("\nCreating comparison plots...")
    create_comparison_plots(model_results, anova_results, output_dir)
    
    # Save detailed results
    results_file = os.path.join(output_dir, "model_comparison_results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        # Prepare data for JSON serialization
        json_data = {
            'anova_results': anova_results,
            'model_results': {}
        }
        
        for model_name, model_data in model_results.items():
            json_data['model_results'][model_name] = {
                'results': model_data['results']
            }
        
        json.dump(json_data, f, indent=4, ensure_ascii=False)
    
    print(f"\nResults saved to: {results_file}")
    print(f"Done in {time.time() - start_time:.3f}s.")


if __name__ == "__main__":
    fire.Fire(main) 