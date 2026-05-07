import random
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input JSONL file (e.g., 2023-14_zh_middle_0009.jsonl)")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to output sampled JSONL file")
    parser.add_argument("--num_samples", type=int, default=50000,
                        help="Number of samples to extract")
    parser.add_argument("--seed", type=int, default=22,
                        help="Random seed")
    args = parser.parse_args()

    with open(args.input, 'r', encoding="utf-8") as f:
        lines = f.readlines()

    random.seed(args.seed)
    random.shuffle(lines)

    with open(args.output, 'w', encoding="utf-8") as fw:
        for line in lines[:args.num_samples]:
            fw.write(line)

    print(f"Sampled {min(args.num_samples, len(lines))} lines from {len(lines)} total")

if __name__ == "__main__":
    main()
