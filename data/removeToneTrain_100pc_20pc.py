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


def process_file(
    input_path: Path,
    output_dir: Path,
    ratios: list[float]
) -> None:
    """
    Create a mixed dataset where different portions have different tone ratios.
    For example: first 1/9 has 100% tones, next 1/9 has 90% tones, etc.
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
        'ratios': {}, 
        'total_tones': 0,
        'total_numbers_found': 0
    }

    # Count total tones and numbers in the dataset
    print("Counting total tones and numbers...")
    for conv in tqdm(conversations, desc="Counting tones"):
        if 'messages' in conv:
            for msg in conv['messages']:
                text = msg.get('content', '')
                metadata['total_tones'] += sum(ch in TONE_CHARS for ch in text)
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

    # Calculate chunk size for each ratio
    total_conversations = len(conversations)
    chunk_size = total_conversations // len(ratios)
    print(f"Total conversations: {total_conversations}")
    print(f"Chunk size per ratio: {chunk_size}")
    print(f"Number of ratios: {len(ratios)}")

    # Create mixed dataset
    processed_conversations = []
    kept_tones_total = 0
    
    print("Creating mixed dataset with different tone ratios...")
    for i, ratio in enumerate(tqdm(ratios, desc="Processing ratios")):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < len(ratios) - 1 else total_conversations
        
        chunk_conversations = conversations[start_idx:end_idx]
        ratio_pct = int(ratio * 100)
        
        print(f"Processing {ratio_pct}% ratio: conversations {start_idx}-{end_idx-1}")
        
        kept_tones_chunk = 0
        for conv in tqdm(chunk_conversations, desc=f"Processing {ratio_pct}% chunk", leave=False):
            if 'messages' in conv:
                processed_messages = []
                for msg in conv['messages']:
                    content = msg['content']
                    new_content = remove_tones_chunk(content, ratio)
                    kept_tones_chunk += sum(ch in TONE_CHARS for ch in new_content)
                    processed_messages.append({
                        'role': msg['role'],
                        'content': new_content
                    })
                processed_conversations.append({
                    'messages': processed_messages
                })
        
        kept_tones_total += kept_tones_chunk
        
        # Record metadata for this chunk
        metadata['ratios'][f'{ratio_pct}%'] = {
            'kept_tones': kept_tones_chunk,
            'ratio': ratio,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'conversation_count': len(chunk_conversations)
        }

    # Save the mixed dataset
    output_file = output_dir / 'sentence_train_100pc_20pc_0730_v1.json'
    save_json(processed_conversations, output_file)
    
    # Update metadata
    metadata['total_kept_tones'] = kept_tones_total
    metadata['total_conversations'] = len(processed_conversations)
    
    # Save metadata
    meta_file = output_dir / 'sentence_train_100pc_20pc_0730_v1_metadata.json'
    save_json(metadata, meta_file)
    
    print(f"Mixed dataset saved to: {output_file}")
    print(f"Total conversations in mixed dataset: {len(processed_conversations)}")
    print(f"Total tones kept: {kept_tones_total}")
    print(f"Number sequences found: {metadata['total_numbers_found']}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Process Braille JSON to keep varying proportions of tone marks.'
    )
    parser.add_argument('input', type=str, help='Path to input JSON file')
    parser.add_argument('output_dir', type=str, help='Directory to save processed JSONs')
    args = parser.parse_args()

    # Ratios: 100%, 90%, ..., 20%
    ratios = [x / 100 for x in range(100, 10, -10)]  # 100,90,...20

    process_file(Path(args.input), Path(args.output_dir), ratios)
    print('Processing complete.')
