import json
import os

# Input and output paths
input_path = "Passage_dataset/passage_10pc/passage_test_10pc_0801_v1.json"
output_path = "Passage_dataset/passage_10pc/passage_test_10pc_no_special_token_0802_v1.json"

# Load the JSON file
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Define tokens to remove
tokens_to_remove = ["<TRANSLATION_END>", "<BRAILLE_END>"]

def remove_tokens(text):
    for token in tokens_to_remove:
        text = text.replace(token, "")
    return text

# Recursively clean all string fields
def clean_obj(obj):
    if isinstance(obj, str):
        return remove_tokens(obj)
    elif isinstance(obj, list):
        return [clean_obj(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: clean_obj(value) for key, value in obj.items()}
    else:
        return obj

cleaned_data = clean_obj(data)

# Write cleaned data to output
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

print(f"Saved cleaned JSON to: {output_path}")