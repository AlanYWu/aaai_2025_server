import json
import random
from pathlib import Path
from tqdm import tqdm
import argparse

# Set random seed for reproducibility
random.seed(42)

# Define tone and punctuation mappings
PUNCTUATION_MARKS = {"句号": "⠐⠆", "问号": "⠐⠄", "叹号": "⠰⠂"}
TONES = {"一声": "⠁", "二声": "⠂", "三声": "⠄", "四声": "⠆"}
PUNCTUATION_CHARS = set("".join(PUNCTUATION_MARKS.values()))
TONE_CHARS = set(TONES.values())

# Braille number prefix and digit mappings
NUMBER_PREFIX = "⠼"  # Number sign prefix
DIGITS = {
    "0": "⠴", "1": "⠁", "2": "⠃", "3": "⠉", "4": "⠙",
    "5": "⠑", "6": "⠋", "7": "⠛", "8": "⠓", "9": "⠊"
}
DIGIT_CHARS = set(DIGITS.values())


def load_json(path: Path) -> dict:
    """Load the Braille JSON file into a Python dict."""
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: dict, path: Path) -> None:
    """Save data to JSON, ensuring UTF-8 encoding."""
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def remove_tones_chunk(chunk: str, keep_ratio: float) -> str:
    """
    Remove tone marks from a chunk of Braille, keeping a given ratio of them.
    - If a tone follows punctuation, always keep it.
    - If we're in a number sequence (after ⠼), don't remove any characters until we exit the number.
    - Otherwise randomly keep each tone with probability keep_ratio.
    """
    new_chunk = []
    i = 0
    while i < len(chunk):
        ch = chunk[i]
        
        # Check if we're entering a number sequence
        if ch == NUMBER_PREFIX:
            # Add the number prefix
            new_chunk.append(ch)
            i += 1
            
            # Skip all characters until we find a non-digit
            while i < len(chunk) and chunk[i] in DIGIT_CHARS:
                new_chunk.append(chunk[i])
                i += 1
            continue
        
        # Handle tone characters
        if ch in TONE_CHARS:
            # If preceding char is punctuation, keep
            if i > 0 and chunk[i-1] in PUNCTUATION_CHARS:
                new_chunk.append(ch)
            else:
                # Randomly decide to keep this tone
                if random.random() < keep_ratio:
                    new_chunk.append(ch)
        else:
            new_chunk.append(ch)
        
        i += 1
    
    return ''.join(new_chunk)


def process_single_file(
    input_path: Path,
    output_dir: Path,
    split_name: str
) -> dict:
    """
    Process a single JSON file and create a new file with only 10% of tone marks kept.
    Returns metadata about the processing.
    """
    print(f"\nProcessing {split_name} file: {input_path}")
    
    data = load_json(input_path)
    
    # Handle both list format (new) and dict format (old)
    if isinstance(data, list):
        # New format: list of conversation objects with 'messages' key
        conversations = data
    else:
        # Old format: single object with 'messages' key
        conversations = [data]

    # Prepare metadata container
    metadata = {
        'input_file': str(input_path), 
        'split': split_name,
        'total_tones_original': 0,
        'total_tones_kept': 0,
        'total_numbers_found': 0,
        'tone_ratio': 0.10
    }

    # Count total tones and numbers in the original dataset
    print(f"Counting original tones and numbers for {split_name}...")
    for conv in tqdm(conversations, desc=f"Counting {split_name} tones"):
        if 'messages' in conv:
            for msg in conv['messages']:
                text = msg.get('content', '')
                metadata['total_tones_original'] += sum(ch in TONE_CHARS for ch in text)
                # Count number sequences (⠼ followed by digits)
                i = 0
                while i < len(text):
                    if text[i] == NUMBER_PREFIX:
                        metadata['total_numbers_found'] += 1
                        # Skip to end of number sequence
                        i += 1
                        while i < len(text) and text[i] in DIGIT_CHARS:
                            i += 1
                    else:
                        i += 1

    # Process all conversations to keep only 10% of tones
    print(f"Processing {split_name} conversations to keep 10% of tones...")
    processed_conversations = []
    kept_tones = 0
    
    for conv in tqdm(conversations, desc=f"Processing {split_name} conversations"):
        if 'messages' in conv:
            processed_messages = []
            for msg in conv['messages']:
                content = msg['content']
                new_content = remove_tones_chunk(content, 0.10)
                kept_tones += sum(ch in TONE_CHARS for ch in new_content)
                processed_messages.append({
                    'role': msg['role'],
                    'content': new_content
                })
            processed_conversations.append({
                'messages': processed_messages
            })

    # Save the processed dataset
    output_file = output_dir / f'passage_{split_name}_10pc_0801_v1.json'
    save_json(processed_conversations, output_file)
    
    # Update metadata
    metadata['total_tones_kept'] = kept_tones
    metadata['total_conversations'] = len(processed_conversations)
    metadata['actual_ratio'] = kept_tones / metadata['total_tones_original'] if metadata['total_tones_original'] > 0 else 0
    
    # Save metadata
    meta_file = output_dir / f'passage_{split_name}_10pc_0801_v1_metadata.json'
    save_json(metadata, meta_file)
    
    print(f"Processed {split_name} dataset saved to: {output_file}")
    print(f"Total conversations: {len(processed_conversations)}")
    print(f"Original tones: {metadata['total_tones_original']}")
    print(f"Kept tones: {kept_tones}")
    print(f"Number sequences found: {metadata['total_numbers_found']}")
    print(f"Actual ratio: {metadata['actual_ratio']:.2%}")
    
    return metadata


def batch_process_files(
    input_dir: Path,
    output_dir: Path
) -> None:
    """
    Process all three files (train, val, test) in batch.
    """
    # Define the files to process
    files_to_process = [
        ("passage_100pc_train_0727_v2.json", "train"),
        ("passage_100pc_val_0727_v2.json", "val"),
        ("passage_100pc_test_0727_v2.json", "test")
    ]
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Process each file
    all_metadata = {}
    
    for filename, split_name in files_to_process:
        input_path = input_dir / filename
        
        if not input_path.exists():
            print(f"Warning: File {input_path} not found, skipping...")
            continue
            
        metadata = process_single_file(input_path, output_dir, split_name)
        all_metadata[split_name] = metadata
    
    # Save combined metadata
    combined_meta_file = output_dir / 'passage_all_splits_10pc_0801_v1_metadata.json'
    save_json(all_metadata, combined_meta_file)
    
    print(f"\nBatch processing complete!")
    print(f"Combined metadata saved to: {combined_meta_file}")
    
    # Print summary
    total_original = sum(meta['total_tones_original'] for meta in all_metadata.values())
    total_kept = sum(meta['total_tones_kept'] for meta in all_metadata.values())
    total_conversations = sum(meta['total_conversations'] for meta in all_metadata.values())
    
    print(f"\nSummary:")
    print(f"Total conversations processed: {total_conversations}")
    print(f"Total original tones: {total_original}")
    print(f"Total kept tones: {total_kept}")
    print(f"Overall ratio: {total_kept/total_original:.2%}" if total_original > 0 else "No tones found")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Batch process Braille JSON files to keep only 10% of tone marks.'
    )
    parser.add_argument('input_dir', nargs='?', default="Passage_dataset", type=str, help='Directory containing input JSON files')
    parser.add_argument('output_dir', nargs='?', default="Passage_dataset/passage_10pc", type=str, help='Directory to save processed JSON files')
    args = parser.parse_args()

    batch_process_files(Path(args.input_dir), Path(args.output_dir))
    print('Batch processing complete.') 