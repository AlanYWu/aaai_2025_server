import json
import random
import os
from pathlib import Path

def sample_data(input_file, output_file, sample_ratio=0.1, random_seed=42):
    """
    Sample a percentage of data from a JSON file and save to a new file.
    
    Args:
        input_file (str): Path to input JSON file
        output_file (str): Path to output JSON file
        sample_ratio (float): Ratio of data to sample (default: 0.1 for 10%)
        random_seed (int): Random seed for reproducibility
    """
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    print(f"Processing {input_file}...")
    
    # Load the data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Ensure data is a list
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array, got {type(data)}")
    
    total_samples = len(data)
    sample_size = int(total_samples * sample_ratio)
    
    print(f"Total samples: {total_samples}")
    print(f"Sample size: {sample_size} ({sample_ratio*100}%)")
    
    # Sample the data
    sampled_data = random.sample(data, sample_size)
    
    # Create output directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the sampled data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sampled_data, f, ensure_ascii=False, indent=2)
    
    print(f"Saved sampled data to {output_file}")
    print(f"Sampled {len(sampled_data)} samples out of {total_samples}")
    print("-" * 50)

def main():
    # Define the files to process
    files_to_sample = [
        # Passage dataset files
        "data/Passage_dataset/passage_10pc_test_0727_v2.json",
        "data/Passage_dataset/passage_10pc_train_0727_v2.json", 
        "data/Passage_dataset/passage_10pc_val_0727_v2.json",
        
        # Sentence dataset files
        "data/Sentence_dataset/set-Full-Tone/sentence_10pc_test_0727_v2.json",
        "data/Sentence_dataset/set-Full-Tone/sentence_10pc_train_0727_v2.json",
        "data/Sentence_dataset/set-Full-Tone/sentence_10pc_val_0727_v2.json"
    ]
    
    # Create output directory
    output_dir = "data/sampled_data"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting data sampling process...")
    print("=" * 60)
    
    # Process each file
    for input_file in files_to_sample:
        # Check if input file exists
        if not os.path.exists(input_file):
            print(f"Warning: Input file {input_file} does not exist. Skipping...")
            continue
        
        # Create output filename
        filename = os.path.basename(input_file)
        output_file = os.path.join(output_dir, filename)
        
        try:
            sample_data(input_file, output_file, sample_ratio=0.1)
        except Exception as e:
            print(f"Error processing {input_file}: {str(e)}")
            continue
    
    print("=" * 60)
    print("Data sampling complete!")
    print(f"All sampled files saved to: {output_dir}")
    
    # List the created files
    print("\nCreated files:")
    for file in os.listdir(output_dir):
        file_path = os.path.join(output_dir, file)
        if os.path.isfile(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"  - {file} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    main() 