import json
import random
from pathlib import Path
from tqdm import tqdm

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


def process_file_10pc(
    input_path: Path,
    output_dir: Path
) -> None:
    """
    Process the input JSON and create a new file with only 10% of tone marks kept.
    """
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
        'total_tones_original': 0,
        'total_tones_kept': 0,
        'total_numbers_found': 0,
        'tone_ratio': 0.10
    }

    # Count total tones and numbers in the original dataset
    print("Counting original tones and numbers...")
    for conv in tqdm(conversations, desc="Counting original tones"):
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

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Process all conversations to keep only 10% of tones
    print("Processing conversations to keep 10% of tones...")
    processed_conversations = []
    kept_tones = 0
    
    for conv in tqdm(conversations, desc="Processing conversations"):
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
    
    # Generate output filename based on input filename
    input_stem = input_path.stem
    output_file = output_dir / 'sentence_train_10pc_0730_v1.json'

    save_json(processed_conversations, output_file)
    
    # Update metadata
    metadata['total_tones_kept'] = kept_tones
    metadata['total_conversations'] = len(processed_conversations)
    metadata['actual_ratio'] = kept_tones / metadata['total_tones_original'] if metadata['total_tones_original'] > 0 else 0
    
    # Save metadata
    meta_file = output_dir / 'sentence_train_10pc_0730_v1_metadata.json'
    save_json(metadata, meta_file)
    
    print(f"Processed dataset saved to: {output_file}")
    print(f"Total conversations: {len(processed_conversations)}")
    print(f"Original tones: {metadata['total_tones_original']}")
    print(f"Kept tones: {kept_tones}")
    print(f"Number sequences found: {metadata['total_numbers_found']}")
    print(f"Actual ratio: {metadata['actual_ratio']:.2%}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Process Braille JSON to keep only 10% of tone marks.'
    )
    parser.add_argument('input', nargs='?', default="Passage_dataset/passage_100pc_train_0727_v2.json", type=str, help='Path to input JSON file')
    parser.add_argument('output_dir', nargs='?', default="Passage_dataset/passage_10pc_train_0727_v2", type=str, help='Directory to save processed JSON')
    args = parser.parse_args()

    process_file_10pc(Path(args.input), Path(args.output_dir))
    print('Processing complete.') 