from transformers import AutoTokenizer
import argparse
import json, os
from tqdm import tqdm
from datasets import load_dataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default="el")
    parser.add_argument("--tok", type=str,
                        default="meta-llama/Llama-3.2-3B")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to CulturaX parquet data directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for sampled data")
    parser.add_argument("--document_nums", type=int, default=50000,
                        help="Number of documents to sample")
    return parser.parse_args()


def collect(data, tokenizer, document_nums=50000):
    res = []
    for k in tqdm(data):
        res.append(k)
        if len(res) == document_nums:
            break
    return res


if __name__ == "__main__":
    args = get_args()

    lang = args.lang
    src = load_dataset(os.path.join(args.data_dir, lang))['train'].shuffle(seed=22)
    print(src)
    print(len(src))

    tokenizer = AutoTokenizer.from_pretrained(args.tok)

    final = collect(src, tokenizer, document_nums=args.document_nums)

    save_path = args.output_dir
    os.makedirs(save_path, exist_ok=True)

    tok_model_path = args.tok
    if "llama" in tok_model_path.lower():
        save_path_z = os.path.join(save_path, f"{lang}-llama")
    elif "qwen" in tok_model_path.lower():
        save_path_z = os.path.join(save_path, f"{lang}-qwen")
    else:
        save_path_z = os.path.join(save_path, lang)

    with open(f"{save_path_z}-100K.jsonl", 'w', encoding="utf-8") as f:
        for k in tqdm(final):
            f.write(json.dumps(k) + '\n')
