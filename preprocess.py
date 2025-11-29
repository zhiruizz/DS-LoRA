import os
import json
import argparse
from types import SimpleNamespace
from datasets import load_from_disk
from transformers import AutoTokenizer


def load_config(config_path):
    """从 JSON 配置文件加载并合并所有配置部分"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    merged_config = {}
    for section in config_dict.values():
        merged_config.update(section)
    return SimpleNamespace(**merged_config)


def main():
    parser = argparse.ArgumentParser(description="Pre-process text dataset based on a config file.")
    parser.add_argument('--config', type=str, default='config.json', help="Path to the JSON config file.")
    args = parser.parse_args()
    config = load_config(args.config)

    # [MODIFIED] 从 config 中读取所有路径和参数
    tokenizer_path = config.tokenizer_path
    processed_dataset_path = config.processed_dataset_path
    max_seq_len = config.max_seq_len

    # [KEPT] 您可以保持硬编码，或也将其加入 config.json
    local_raw_dataset_path = "./wikitext_dataset_local"

    if os.path.exists(processed_dataset_path):
        print(f"Processed dataset already exists at '{processed_dataset_path}'. Skipping.")
        print("If you want to re-process, please delete this folder manually.")
        return

    print(f"Loading tokenizer from unified path: '{tokenizer_path}'...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    print(f"Loading raw dataset from disk: '{local_raw_dataset_path}'...")
    raw_datasets = load_from_disk(local_raw_dataset_path)

    # [MODIFIED] 函数现在使用从 config 加载的 max_seq_len
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=max_seq_len, padding=False)

    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // max_seq_len) * max_seq_len
        result = {
            k: [t[i: i + max_seq_len] for i in range(0, total_length, max_seq_len)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    print("Starting data processing (this may take some time)...")

    cpu_cores = os.cpu_count()
    num_proc = cpu_cores // 2 if cpu_cores and cpu_cores > 1 else 1
    print(f"Using {num_proc} CPU cores for processing.")

    print("Step 1: Tokenizing...")
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        num_proc=num_proc
    )

    print("Step 2: Grouping texts...")
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=num_proc
    )

    print(f"Processing finished. Saving processed dataset to '{processed_dataset_path}'...")
    lm_datasets.save_to_disk(processed_dataset_path)
    print("Done.")


if __name__ == '__main__':
    main()