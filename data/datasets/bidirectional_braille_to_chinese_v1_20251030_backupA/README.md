# bidirectional_braille_to_chinese_v1_20251030
方向: Braille→Chinese
数据来源:
- /workspace/wuyou/aaai_2025_server/data/Passage_dataset/passage_100pc_train_0727_v2.json
- /workspace/wuyou/aaai_2025_server/data/Passage_dataset/passage_100pc_val_0727_v2.json
- /workspace/wuyou/aaai_2025_server/data/Passage_dataset/passage_100pc_test_0727_v2.json
- /workspace/wuyou/aaai_2025_server/data/Sentence_dataset/set-Full-Tone/sentence_100pc_train_0727_v2.json
- /workspace/wuyou/aaai_2025_server/data/Sentence_dataset/set-Full-Tone/sentence_100pc_val_0727_v2.json
- /workspace/wuyou/aaai_2025_server/data/Sentence_dataset/set-Full-Tone/sentence_100pc_test_0727_v2.json

样本格式: 三段消息 (system/user/assistant)。输出以 <TRANSLATION_END> 结束，输入侧根据模态包含 <BRAILLE_END> 或 <CHINESE_END>。

划分: 8:1:1 (train:validation:test)。每个方向下分别输出 sentence_* 与 passage_* 三个切分文件。
