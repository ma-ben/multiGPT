# train_gpt2_tokenizer.py
# pip install tokenizers transformers

from tokenizers.implementations import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast ,GPT2TokenizerFast
import json, os
from pathlib import Path

# === 1. 数据与配置 ===
corpus_files = ["input_ZH.txt"]  # 语料文件

vocab_size = 2000
output_dir = f"tokenizer_gpt2_{vocab_size}"
min_frequency = 2
os.makedirs(output_dir, exist_ok=True)

# === 2. 训练 ByteLevelBPE 分词器（GPT-2 风格） ===
tokenizer = ByteLevelBPETokenizer()
tokenizer.train(
    files=corpus_files,
    vocab_size=vocab_size,
    min_frequency=min_frequency,
    special_tokens=[
        "<|endoftext|>",  # GPT-2 原始结束符
        "<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>",  # 现代兼容符号
    ],
)
tokenizer.save_model(output_dir)  # 导出 vocab.json + merges.txt

# === 3. 转为 transformers 格式 ===
fast_tok = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    bos_token="<|bos|>",
    eos_token="<|eos|>",
    unk_token="<|unk|>",
    pad_token="<|pad|>",
    additional_special_tokens=["<|endoftext|>"],
)

# 写出配置：声明 model_type=gpt2，prefix 空格策略
fast_tok.save_pretrained(output_dir)
with open(os.path.join(output_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
    json.dump(
        {
            "model_type": "gpt2",
            "add_prefix_space": True,
            "padding_side": "right",
            "truncation_side": "right"
        },
        f,
        ensure_ascii=False,
        indent=2
    )

print(f"[+] GPT2 tokenizer trained and saved to {output_dir}")

# === 4. 验证加载 ===
import numpy as np
tokenizer = GPT2TokenizerFast.from_pretrained(f"tokenizer_gpt2_{vocab_size}")

all_ids = []

for fp in corpus_files:
    text = Path(fp).read_text()
    ids = tokenizer.encode(text)
    all_ids.extend(ids + [tokenizer.eos_token_id])  # 文件间加上分隔符

print(f"Total tokens: {len(all_ids):,}")

np.array(all_ids, dtype=np.uint16).tofile(f"train_{vocab_size}.bin")

