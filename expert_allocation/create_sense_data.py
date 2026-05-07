#!/usr/bin/env python3
# Create sense datasets for new languages (e.g., tr, el, hu)

import json
import jsonlines
import argparse
import random
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared_data_path", type=str,
                        required=True,
                        help="Path to prepared language data (JSONL files)")
    parser.add_argument("--output_dir", type=str,
                        default="./neuron_analysis/assets/Language/sense",
                        help="Output directory for sense data")
    parser.add_argument("--original_dir", type=str,
                        default="./neuron_analysis/assets/Language/sense",
                        help="Original sense data directory (for existing 6 languages)")
    parser.add_argument("--num_sentences", type=int, default=500,
                        help="Number of sentences per language")
    parser.add_argument("--languages", nargs='+', default=["tr", "el", "hu"],
                        help="Languages to process")
    parser.add_argument("--file_pattern", type=str, default="{lang}-llama-2B.jsonl",
                        help="File pattern for language files (use {lang} placeholder)")
    return parser.parse_args()

def extract_sentences_from_jsonl(jsonl_path, num_sentences=1000):
    """Extract sentences from JSONL file"""
    sentences = []

    try:
        with jsonlines.open(jsonl_path) as reader:
            for line in reader:
                if 'text' in line:
                    text = line['text'].strip()
                    if text and len(text) > 20 and len(text) < 500:
                        import re
                        sents = re.split(r'[.!?]+\s+', text)
                        for sent in sents[:3]:
                            if len(sent.strip()) > 20:
                                sentences.append(sent.strip())

                if len(sentences) >= num_sentences * 2:
                    break

    except Exception as e:
        print(f"Error reading {jsonl_path}: {e}")
        return []

    if len(sentences) > num_sentences:
        sentences = random.sample(sentences, num_sentences)

    return sentences

def create_sense_data(language, positive_sentences, negative_sentences):
    """Create JSON in sense data format"""
    sense_data = {
        "concept": language,
        "group": "sense",
        "source": "language",
        "sentences": {
            "positive": positive_sentences,
            "negative": negative_sentences
        }
    }
    return sense_data

def load_existing_sense_data(sense_dir):
    """Extract sentences from existing sense data"""
    existing_sentences = {}
    sense_path = Path(sense_dir)

    if not sense_path.exists():
        return existing_sentences

    for json_file in sense_path.glob("*.json"):
        lang_code = json_file.stem
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'sentences' in data and 'positive' in data['sentences']:
                    existing_sentences[lang_code] = data['sentences']['positive']
                    print(f"Loaded existing {lang_code}: {len(existing_sentences[lang_code])} sentences")
        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    return existing_sentences

def main():
    args = get_args()

    print("=== Multilingual Sense Dataset Creation ===")
    print(f"Target languages: {args.languages}")
    print(f"File pattern: {args.file_pattern}")

    # Load original 6 languages
    print("Loading original 6 languages...")
    original_sentences = load_existing_sense_data(args.original_dir)

    # Extract sentences for new languages
    new_sentences = {}
    for lang in args.languages:
        filename = args.file_pattern.format(lang=lang)
        jsonl_path = Path(args.prepared_data_path) / filename

        if not jsonl_path.exists():
            print(f"Warning: {jsonl_path} does not exist")
            continue

        print(f"Loading new language {lang}: {jsonl_path}")
        sentences = extract_sentences_from_jsonl(jsonl_path, args.num_sentences * 2)

        if not sentences:
            print(f"Error: No sentences extracted for {lang}")
            continue

        new_sentences[lang] = sentences
        print(f"Loaded {len(sentences)} sentences for {lang}")

    # Combine original 6 languages + new languages
    all_sentences = {}
    all_sentences.update(original_sentences)
    all_sentences.update(new_sentences)

    all_langs = list(all_sentences.keys())
    print(f"\nAll languages ({len(all_langs)}): {all_langs}")
    print(f"Per language: {args.num_sentences} positive + {(len(all_langs)-1) * args.num_sentences} negative sentences")

    # Clean up and recreate output directory
    output_path = Path(args.output_dir)
    if output_path.exists():
        print(f"Cleaning {args.output_dir}")
        import shutil
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create sense data for all languages
    for target_lang in all_langs:
        print(f"\nCreating sense data for {target_lang}")

        positive_sentences = random.sample(all_sentences[target_lang],
                                         min(args.num_sentences, len(all_sentences[target_lang])))

        negative_sentences = []
        other_langs = [lang for lang in all_langs if lang != target_lang]

        for other_lang in other_langs:
            if other_lang in all_sentences:
                other_sentences = random.sample(all_sentences[other_lang],
                                              min(args.num_sentences, len(all_sentences[other_lang])))
                negative_sentences.extend(other_sentences)

        print(f"  Positive: {len(positive_sentences)} sentences")
        print(f"  Negative: {len(negative_sentences)} sentences (from {len(other_langs)} other languages)")

        sense_data = create_sense_data(target_lang, positive_sentences, negative_sentences)

        output_file = output_path / f"{target_lang}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sense_data, f, ensure_ascii=False, indent=2)

        print(f"  Created: {output_file}")

    print(f"\n=== Sense dataset creation complete (total: {len(all_langs)} languages) ===")

if __name__ == "__main__":
    main()
