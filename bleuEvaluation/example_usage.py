#!/usr/bin/env python3
"""
Example usage of the model comparison script.

This script demonstrates how to use the eval_bleu_model_comparison.py script
to compare multiple model predictions using ANOVA analysis and create plots.
"""

import os
import subprocess
import sys

def run_model_comparison():
    """
    Example of how to run the model comparison script.
    
    Replace the file paths with your actual model prediction files.
    """
    
    # Example model files (replace with your actual files)
    model_files = [
        "saves/Qwen2.5-7B-Instruct-Braille/7B_qwen2.5i_train_fullsft_sentence_100pc_20pc_10pc_0730_resume_from_v1_v3/test/generated_predictions.jsonl",
        "saves/Qwen2.5-7B-Instruct-Braille/7B_sentence_10pc_qwen2.5i_train_fullsft_0731_v1/test/generated_predictions.jsonl"
    ]
    
    # Join the files with commas for the script input
    model_files_str = ",".join(model_files)
    
    # Output directory for results and plots
    output_dir = "model_comparison_results"
    
    # Run the comparison script
    cmd = [
        "python", "eval_bleu_model_comparison.py",
        "--model_files", model_files_str,
        "--output_dir", output_dir
    ]
    
    print("Running model comparison...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Model comparison completed successfully!")
        print("Output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error running model comparison:")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False
    
    return True

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn',
        'fire', 'datasets', 'jieba', 'nltk', 'rouge_chinese'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install them using:")
        print("pip install -r requirements_model_comparison.txt")
        return False
    
    print("\nAll dependencies are installed!")
    return True

if __name__ == "__main__":
    print("Model Comparison Tool - Example Usage")
    print("=" * 50)
    
    # Check dependencies first
    print("Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    print("\nTo use the model comparison tool:")
    print("1. Prepare your model prediction files in JSON format")
    print("2. Update the file paths in this script")
    print("3. Run the comparison:")
    print("   python eval_bleu_model_comparison.py --model_files 'file1.json,file2.json,file3.json' --output_dir results")
    
    # Uncomment the line below to run the actual comparison
    run_model_comparison() 