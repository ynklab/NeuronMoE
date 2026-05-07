from datasets import load_dataset
import json
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to SlimPajama dataset directory")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to output sampled JSONL file")
    parser.add_argument("--num_samples", type=int, default=50000,
                        help="Number of samples to extract")
    parser.add_argument("--seed", type=int, default=22,
                        help="Random seed for shuffling")
    args = parser.parse_args()

    print("Loading dataset...")
    src = load_dataset(args.data_path)['train'].shuffle(seed=args.seed)
    print(f"Dataset size: {len(src)}")

    res = []
    for k in tqdm(src):
        res.append(k)
        if len(res) == args.num_samples:
            break

    with open(args.output, 'w', encoding="utf-8") as f:
        for k in res:
            f.write(json.dumps(k) + '\n')

    print(f"Saved {len(res)} samples to {args.output}")

if __name__ == "__main__":
    main()
