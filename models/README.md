Prerequisite: git lfs

Download the huggingface model from this mirror
```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://hf-mirror.com/Qwen/Qwen2.5-7B-Instruct
cd https://hf-mirror.com/Qwen/Qwen2.5-7B-Instruct
git lfs pull
```

Then, run 
```bash
python pre_processing/addSpecialTokens.py --original_model_dir ./models/Qwen2.5-7B-Instruct --output_dir ./models/Qwen2.5-7B-Instruct-Braille
```
