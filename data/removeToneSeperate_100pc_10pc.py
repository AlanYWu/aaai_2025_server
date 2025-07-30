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
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data: dict, path: Path) -> None:
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def remove_tones_chunk(chunk: str, keep_ratio: float) -> str:
    new_chunk = []
    i = 0
    while i < len(chunk):
        ch = chunk[i]
        if ch == NUMBER_PREFIX:
            new_chunk.append(ch)
            i += 1
            while i < len(chunk) and chunk[i] in DIGIT_CHARS:
                new_chunk.append(chunk[i])
                i += 1
            continue
        if ch in TONE_CHARS:
            if i > 0 and chunk[i-1] in PUNCTUATION_CHARS:
                new_chunk.append(ch)
            else:
                if random.random() < keep_ratio:
                    new_chunk.append(ch)
        else:
            new_chunk.append(ch)
        i += 1
    return ''.join(new_chunk)

def process_file_separate(
    input_path: Path,
    output_dir: Path
) -> None:
    data = load_json(input_path)
    
    # Handle both list format (new) and dict format (old)
    if isinstance(data, list):
        # New format: list of conversation objects with 'messages' key
        conversations = data
    else:
        # Old format: single object with 'messages' key
        conversations = [data]

    total_conversations = len(conversations)
    chunk_size = total_conversations // 10
    print(f"Total conversations: {total_conversations}, chunk size: {chunk_size}")

    output_dir.mkdir(parents=True, exist_ok=True)

    for i, ratio in enumerate([x / 10 for x in range(10, 0, -1)]):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < 9 else total_conversations
        chunk_conversations = conversations[start_idx:end_idx]
        processed_conversations = []
        kept_tones = 0
        
        for conv in tqdm(chunk_conversations, desc=f"Processing {int(ratio*100)}% chunk"):
            if 'messages' in conv:
                processed_messages = []
                for msg in conv['messages']:
                    content = msg['content']
                    new_content = remove_tones_chunk(content, ratio)
                    kept_tones += sum(ch in TONE_CHARS for ch in new_content)
                    processed_messages.append({
                        'role': msg['role'],
                        'content': new_content
                    })
                processed_conversations.append({
                    'messages': processed_messages
                })
        
        output_file = output_dir / f'validation_{int(ratio*100)}pc.json'
        save_json(processed_conversations, output_file)
        print(f"Saved {output_file}, kept tones: {kept_tones}, conversations: {len(processed_conversations)}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Split input into 10 parts, each with different tone retention (100%~10%).'
    )
    parser.add_argument('input', type=str, help='Path to input JSON file')
    parser.add_argument('output_dir', type=str, help='Directory to save processed JSONs')
    args = parser.parse_args()
    process_file_separate(Path(args.input), Path(args.output_dir))
    print('Processing complete.') 