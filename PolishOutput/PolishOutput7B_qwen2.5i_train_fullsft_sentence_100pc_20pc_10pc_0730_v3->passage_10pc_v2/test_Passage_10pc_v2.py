import json
from tqdm import tqdm
from vllm import LLM, SamplingParams
import os

# For this file
# saves/Qwen2.5-7B-Instruct-Braille/
# 7B_qwen2.5i_train_fullsft_sentence_100pc_20pc_10pc_0730_v3->passage_10pc_v2/test_Passage_10pc_v2/generated_predictions.jsonl

# === Configurations ===
INPUT_JSON = "saves/Qwen2.5-7B-Instruct-Braille/7B_qwen2.5i_train_fullsft_sentence_100pc_20pc_10pc_0730_v3->passage_10pc_v2/test_Passage_10pc_v2/generated_predictions.jsonl"
# Get the directory of the input file
INPUT_DIR = os.path.dirname(os.path.abspath(INPUT_JSON))
OUTPUT_JSON = os.path.join(INPUT_DIR, "generated_predictions_polishedOutput_7bbraille_0802_v1.json")
MODEL_NAME = "models/Qwen2.5-7B-Instruct"
BATCH_SIZE = 20  # You can adjust this based on your GPU memory

# Example for prompt construction
with open("PolishOutput/example_ori.txt", "r", encoding="utf-8") as f:
    example_ori = f.read().strip()
with open("PolishOutput/example_target.txt", "r", encoding="utf-8") as f:
    example_target = f.read().strip()

# === Load input JSON ===
print(f"Loading input data from {INPUT_JSON}")
with open(INPUT_JSON, 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]

if not isinstance(data, list):
    data = [data]

# === Generate and save ===
results = []

# Construct prompts
all_prompts = [
    """<|im_start|>system
你是一个擅长润色中文语句的助手，我的其它模型生成了一些中文语句，但其中可能存在语言模型常见的重复生成问题，这通常发生在语句末尾。我需要你识别出来这些重复生成并去掉。其它地方尽量维持原始语句不变。你**必须**直接输出修改过的语句，**禁止**输出任何解释。
原始文本：{}
目标输出：{}
<|im_end|>
<|im_start|>user
现在如下是我要处理的文本：
{}

请给我你的输出。
请注意：我需要你识别出来这些重复生成并去掉。其它地方尽量维持原始语句不变。你**必须**直接输出修改过的语句，**禁止**输出任何解释。
<|im_end|>
<|im_start|>assistant""".format(example_ori, example_target, item.get("predict", ""))
    for item in data
]   

llm = LLM(model=MODEL_NAME)
sampling_params = SamplingParams(max_tokens=512, temperature=0.7, top_p=1.0)

print("Generating responses with vLLM...")
for i in tqdm(range(0, len(all_prompts), BATCH_SIZE)):
    batch_prompts = all_prompts[i:i + BATCH_SIZE]
    batch_data = data[i:i + BATCH_SIZE]
    outputs = llm.generate(batch_prompts, sampling_params)
    for item, output in zip(batch_data, outputs):
        result = {
            "predict": item.get("predict", ""),
            "polish_output": output.outputs[0].text if output.outputs else "",
            "label": item.get("label", ""),
            "prompt": item.get("prompt", "")
        }
        results.append(result)

# === Save output ===
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"Done! Saved results to {OUTPUT_JSON}")