from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--original_model_dir", help="The path to the original model", default="./models/Qwen2.5-3B-Instruct"
)
parser.add_argument(
    "--output_dir", help="The path to save the new model", default="./models/Qwen2.5-3B-Instruct-Braille"
)
args = parser.parse_args()

model_name = args.original_model_dir
# The output_dir is the path that the tokenizer will be saved.
output_dir = args.output_dir


def directory_exists_and_not_empty(path):
    return os.path.exists(path) and os.path.isdir(path) and len(os.listdir(path)) > 0


# model_name = "./Qwen2.5-3B-Instruct"  # Or local path to your Qwen model
# ./Qwen2.5-3B-Instruct-Braille
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

# Add special tokens. Updated. 
new_chars = "⠠⠐⠰⠈⠨⠘⠸⠄⠤⠔⠴⠌⠬⠜⠼⠂⠢⠒⠲⠊⠪⠚⠺⠆⠦⠖⠶⠎⠮⠞⠾⠁⠡⠑⠱⠉⠩⠙⠹⠅⠥⠕⠵⠍⠭⠝⠽⠃⠣⠓⠳⠋⠫⠛⠻⠇⠧⠗⠷⠏⠯⠟⠿"
new_tokens = list(new_chars)
new_tokens.append("<BRAILLE_END>")
new_tokens.append("<TRANSLATION_END>")
assert len(new_tokens) == 63 + 2

# Add special tokens to the tokenizer
num_added_toks = tokenizer.add_tokens(new_tokens)
print(f"Number of tokens added: {num_added_toks}")


example = (
    "⠼⠁ ⠌⠢⠆⠛⠢⠆ ⠼⠉⠙ ⠎⠺⠆ ⠙⠢ ⠙⠔⠆⠇⠡⠂ ⠅⠡⠁⠝⠩⠂⠐ ⠙⠖⠆ ⠓⠩⠆⠵⠪⠆ ⠇⠩⠂ ⠝⠬⠄⠓⠪⠂ ⠙⠢⠱⠷⠄ ⠙⠷⠁ ⠍⠮⠂ ⠇⠔⠁ ⠛⠕⠆ ⠐⠆"
)
print(example)
print(tokenizer.decode(tokenizer(example)["input_ids"]))

# Test the added tokens
text = "⠢⠖⠶ ⠦⠔⠴⠁⠃⠉⠙⠑⠋⠛⠓⠊⠚⠅⠇⠍⠝⠕⠏⠟⠗⠎⠞⠥⠧⠺⠭  ⠽⠵⠮⠐⠼⠫⠯⠄⠷⠾⠾⠡⠬⠠⠤⠨⠌⠆⠰⠣⠿⠜⠹⠈⠪⠳⠻⠘⠸⠲⠒⠆⠂<BRAILLE_END>这是视觉盲文-助力教育公平的工具。<TRANSLATION_END>"
print("Original input",text)
encoding = tokenizer.encode(text, max_length=100)
print("Encoded input", encoding)
print("Decoded input (should equal to the original input", tokenizer.decode(encoding))
assert text == tokenizer.decode(encoding), (f"Assertion failed: encoding != "
                                                f"decoded\nEncoding: {text}"
                                                f"\nDecoded: {tokenizer.decode(encoding)}")


model.resize_token_embeddings(len(tokenizer))
if num_added_toks > 0:
    model.resize_token_embeddings(len(tokenizer))

tokenizer.save_pretrained(output_dir)
model.save_pretrained(output_dir)
print("Model and tokenizer saved successfully!")

