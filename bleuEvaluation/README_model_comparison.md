# Model Comparison Tool with ANOVA Analysis

This tool allows you to compare multiple model predictions using statistical analysis (ANOVA) and create visualization plots with colored areas under the lines.

## Features

- **Statistical Analysis**: Performs one-way ANOVA to test for significant differences between models
- **Multiple Metrics**: Evaluates ROUGE-1, ROUGE-2, ROUGE-L, and BLEU-4 scores
- **Segmented Analysis**: Divides data into 10 segments (100%, 90%, 80%, ..., 10% tone percentages)
- **Visualization**: Creates plots with colored areas under the lines for each model
- **Effect Size**: Calculates eta-squared (η²) effect size for ANOVA results

## Installation

Install the required dependencies:

```bash
pip install -r requirements_model_comparison.txt
```

## Usage

### Basic Usage

```bash
python eval_bleu_model_comparison.py --model_files "model1.json,model2.json,model3.json" --output_dir results
```

### Parameters

- `--model_files`: Comma-separated list of model prediction files (JSON format)
- `--output_dir`: Directory to save results and plots (default: current directory)

### Input Format

Each model prediction file should be in JSON format with the following structure:

```json
[
  {
    "label": "对应的中文内容是:\n原始中文文本",
    "predict": "对应的中文内容是:\n预测的中文文本"
  },
  ...
]
```

## Output

The tool generates:

1. **ANOVA Results**: Statistical analysis showing F-statistic, p-value, and significance
2. **Comparison Plots**: PNG files for each metric (rouge-1, rouge-2, rouge-l, bleu-4)
3. **Detailed Results**: JSON file with all analysis results

### Plot Features

- **Colored Lines**: Each model has a different colored line
- **Filled Areas**: Colored areas under each line for better visualization
- **ANOVA Information**: Statistical results displayed on each plot
- **Reversed X-axis**: Tone percentage from 100% to 10%

## Example Output

### ANOVA Results
```
ROUGE-1:
  F-statistic: 15.2345
  p-value: 0.000123
  Significant: Yes
  Effect size (η²): 0.2345
  Group means:
    model1: 45.2345 ± 12.3456
    model2: 52.1234 ± 11.2345
    model3: 48.5678 ± 13.4567
```

### Generated Files
- `rouge-1_model_comparison.png`
- `rouge-2_model_comparison.png`
- `rouge-l_model_comparison.png`
- `bleu-4_model_comparison.png`
- `model_comparison_results.json`

## Statistical Interpretation

- **F-statistic**: Higher values indicate more significant differences between groups
- **p-value**: < 0.05 indicates statistically significant differences
- **Effect size (η²)**: 
  - 0.01-0.06: Small effect
  - 0.06-0.14: Medium effect
  - > 0.14: Large effect

## Example Script

See `example_usage.py` for a complete example of how to use the tool.

## Dependencies

- pandas: Data manipulation
- numpy: Numerical computations
- scipy: Statistical analysis
- matplotlib: Plotting
- seaborn: Enhanced plotting
- fire: Command-line interface
- datasets: Data loading
- jieba: Chinese text segmentation
- nltk: Natural language processing
- rouge-chinese: Chinese ROUGE evaluation 