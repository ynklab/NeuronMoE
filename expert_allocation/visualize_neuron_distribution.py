#!/usr/bin/env python3
"""
Language neuron layer-wise distribution histogram visualization script.
Displays the per-layer distribution of top-1000 language-specific neurons for each language.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import re
from pathlib import Path
import seaborn as sns

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang_neuron_dir", type=str,
                        required=True,
                        help="Path to lang_neuron results directory (containing Language/ subdirectory)")
    parser.add_argument("--model_name", type=str,
                        default="meta-llama_Llama-3.2-3B",
                        help="Model name (with underscore)")
    parser.add_argument("--languages", nargs='+', default=["en", "de", "fr", "es", "ja", "zh", "el"],
                        help="Languages to analyze")
    parser.add_argument("--expertise_type", type=str, default="expertise_limited_1000_top",
                        help="Expertise type directory name")
    parser.add_argument("--output_path", type=str, default="neuron_distribution.png",
                        help="Output image path")
    parser.add_argument("--num_layers", type=int, default=28,
                        help="Number of model layers (Llama 3.2-3B = 28)")
    return parser.parse_args()

def load_expertise_data(expertise_file):
    """Load neuron data from expertise.csv or .json"""
    try:
        if expertise_file.suffix == '.csv':
            df = pd.read_csv(expertise_file)
            layer_counts = {}

            if 'layer' in df.columns:
                for _, row in df.iterrows():
                    layer_name = str(row['layer'])
                    match = re.search(r'model\.layers\.(\d+)\.', layer_name)
                    if match:
                        layer_num = int(match.group(1))
                        layer_counts[layer_num] = layer_counts.get(layer_num, 0) + 1
            elif 'responses' in df.columns:
                for _, row in df.iterrows():
                    response_name = str(row['responses'])
                    match = re.search(r'layers\.(\d+)\.', response_name)
                    if match:
                        layer_num = int(match.group(1))
                        layer_counts[layer_num] = layer_counts.get(layer_num, 0) + 1

            return layer_counts

        else:
            with open(expertise_file, 'r') as f:
                data = json.load(f)

            layer_counts = {}

            if 'layers' in data:
                for layer_info in data['layers']:
                    if 'layer' in layer_info and 'neurons' in layer_info:
                        layer_num = layer_info['layer']
                        neuron_count = len(layer_info['neurons'])
                        layer_counts[layer_num] = neuron_count
            else:
                for key, value in data.items():
                    if key.startswith('layer_') and isinstance(value, list):
                        layer_num = int(key.split('_')[1])
                        layer_counts[layer_num] = len(value)

            return layer_counts

    except Exception as e:
        print(f"Error loading {expertise_file}: {e}")
        return {}

def find_expertise_file(lang_neuron_dir, model_name, language, expertise_type):
    """Search for expertise.csv or .json file"""
    possible_paths = [
        Path(lang_neuron_dir) / "Language" / model_name / "sense" / language / "expertise" / f"{expertise_type}.csv",
        Path(lang_neuron_dir) / model_name / "sense" / language / "expertise" / f"{expertise_type}.csv",
        Path(lang_neuron_dir) / model_name / "sense" / language / "expertise" / f"{expertise_type}.json",
        Path(lang_neuron_dir) / "results" / model_name / "sense" / language / "expertise" / f"{expertise_type}.json",
    ]

    for path in possible_paths:
        if path.exists():
            return path

    # Search recursively if not found at expected paths
    base_path = Path(lang_neuron_dir)
    for expertise_file in base_path.rglob(f"*{language}*/expertise/{expertise_type}.*"):
        return expertise_file

    return None

def plot_neuron_distribution(language_data, languages, num_layers, output_path):
    """Visualize as histogram"""
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=(6, 4))

    colors = ['#1f77b4', '#ff7f0e'] * (len(languages) // 2 + 1)

    for i, lang in enumerate(languages):
        if lang in language_data:
            layers = list(range(num_layers))
            counts = [language_data[lang].get(layer, 0) for layer in layers]

            ax.bar([x + i * 0.1 for x in layers], counts,
                  width=0.2, label=lang.upper(), alpha=0.8, color=colors[i])

            print(f"{lang.upper()}: Total neurons = {sum(counts)}")

    ax.set_xlabel('Layer Number', fontsize=12)
    ax.set_ylabel('Number of Language-Specific Neurons', fontsize=12)
    ax.set_title(r'Neuron Distribution Across Layers Finetuning' + '\n(Top 1000 Language-Specific Neurons)', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, num_layers - 0.5)

    ax.set_xticks(range(0, num_layers, 4))
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Histogram saved to: {output_path}")

def main():
    args = get_args()

    print("=== Language Neuron Distribution Analysis ===")
    print(f"Model: {args.model_name}")
    print(f"Languages: {args.languages}")
    print(f"Expertise type: {args.expertise_type}")

    language_data = {}

    for lang in args.languages:
        print(f"\nProcessing {lang}...")

        expertise_file = find_expertise_file(
            args.lang_neuron_dir,
            args.model_name,
            lang,
            args.expertise_type
        )

        if expertise_file:
            print(f"Found: {expertise_file}")
            layer_counts = load_expertise_data(expertise_file)
            if layer_counts:
                language_data[lang] = layer_counts
                print(f"Loaded {len(layer_counts)} layers for {lang}")
            else:
                print(f"No data loaded for {lang}")
        else:
            print(f"Expertise file not found for {lang}")
            base_path = Path(args.lang_neuron_dir)
            if base_path.exists():
                print("Available directories:")
                for d in base_path.rglob(f"*{lang}*"):
                    if d.is_dir():
                        print(f"  {d}")

    if not language_data:
        print("No expertise data found! Please check the paths and file structure.")
        return

    print(f"\nLoaded data for {len(language_data)} languages: {list(language_data.keys())}")

    plot_neuron_distribution(language_data, args.languages, args.num_layers, args.output_path)

    # Statistics
    print("\n=== Statistics ===")
    for lang in language_data:
        total_neurons = sum(language_data[lang].values())
        max_layer = max(language_data[lang], key=language_data[lang].get)
        max_count = language_data[lang][max_layer]
        print(f"{lang.upper()}: Total={total_neurons}, Peak=Layer{max_layer}({max_count} neurons)")

if __name__ == "__main__":
    main()
