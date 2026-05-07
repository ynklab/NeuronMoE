#!/usr/bin/env python3
# Multilingual neuron distribution analysis - Expert number determination script
# Determines mutually beneficial optimized expert allocation from integrated neuron distribution
# of existing and new languages

import os
import json
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang_neuron_results", type=str,
                        default="./neuron_analysis_results",
                        help="Path to lang_neuron results directory")
    parser.add_argument("--model_name", type=str,
                        default="meta-llama/Llama-3.2-3B",
                        help="Model name used in lang_neuron")
    parser.add_argument("--languages", type=str,
                        help="All languages to analyze (space-separated)")
    parser.add_argument("--min_experts", type=int, default=1,
                        help="Minimum number of experts per layer")
    parser.add_argument("--max_experts", type=int, default=6,
                        help="Maximum number of experts per layer")
    parser.add_argument("--output_config", type=str,
                        default="./expert_allocation/configs/expert_config.txt",
                        help="Output expert configuration file")
    return parser.parse_args()

def load_language_neuron_data(lang_neuron_path, model_name, language):
    """Load language neuron data for the specified language"""
    # Result path structure: {base_path}/Language/{model_name}/sense/{lang}/expertise/
    model_path = model_name.replace("/", "_")
    expertise_path = Path(lang_neuron_path) / "Language" / model_path / "sense" / language / "expertise"

    # Search for expertise_limited_1000_top.csv files
    expertise_files = list(expertise_path.glob("*expertise_limited*.csv"))

    if not expertise_files:
        print(f"Warning: No expertise files found for {language} at {expertise_path}")
        return None

    expertise_file = expertise_files[0]
    print(f"Loading expertise data for {language}: {expertise_file}")

    try:
        df = pd.read_csv(expertise_file)
        return df
    except Exception as e:
        print(f"Error loading {expertise_file}: {e}")
        return None

def analyze_layer_neuron_distribution(expertise_data_dict, num_layers=24):
    """Analyze the unique neuron count distribution per layer"""
    layer_neuron_counts = {}

    for layer_idx in range(num_layers):
        all_layer_neurons = []

        for lang, df in expertise_data_dict.items():
            if df is not None:
                if 'layer' in df.columns and 'unit' in df.columns:
                    df_with_layer_num = df.copy()
                    df_with_layer_num['layer_num'] = df['layer'].str.extract(r'model\.layers\.(\d+)\.').astype(int)
                    layer_data = df_with_layer_num[df_with_layer_num['layer_num'] == layer_idx]
                    neurons = layer_data['unit'].tolist()
                    all_layer_neurons.extend(neurons)
                elif f'layer_{layer_idx}' in df.columns:
                    layer_data = df[f'layer_{layer_idx}'].dropna()
                    neurons = layer_data.tolist()
                    all_layer_neurons.extend(neurons)

        unique_neurons = len(set(all_layer_neurons)) if all_layer_neurons else 0
        layer_neuron_counts[f'layer_{layer_idx}'] = unique_neurons
        print(f"Layer {layer_idx}: {unique_neurons} unique neurons")

    return layer_neuron_counts

def determine_expert_numbers(layer_neuron_counts, min_experts=1, max_experts=6):
    """Determine the number of experts based on unique neuron counts"""
    neuron_counts = list(layer_neuron_counts.values())

    if not neuron_counts or all(count == 0 for count in neuron_counts):
        print("Warning: No neuron data found, using default configuration")
        return [3] * 28

    max_neurons = max(neuron_counts) if max(neuron_counts) > 0 else 1
    min_neurons = min([c for c in neuron_counts if c > 0]) if any(c > 0 for c in neuron_counts) else 0

    expert_numbers = []

    for i, neuron_count in enumerate(neuron_counts):
        if neuron_count == 0:
            experts = min_experts
        else:
            normalized_count = (neuron_count - min_neurons) / (max_neurons - min_neurons) if max_neurons > min_neurons else 0
            experts = min_experts + int(normalized_count * (max_experts - min_experts))
            experts = max(min_experts, min(max_experts, experts))

        print(f"Layer {i}: {neuron_count} unique neurons -> {experts} experts")
        expert_numbers.append(experts)

    return expert_numbers

def save_expert_configuration(expert_numbers, output_path, languages=None):
    """Save expert configuration to file"""
    expert_str = ','.join(map(str, expert_numbers))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(f"# Expert configuration based on multilingual unique neuron distribution\n")
        f.write(f"# Target languages: {languages}\n")
        f.write(f"# Generated at: {pd.Timestamp.now()}\n")
        f.write(f"ADA_MOE_NUM_EXPERTS_LIST={expert_str}\n")

    return expert_str

def main():
    args = get_args()

    languages = args.languages.split() if args.languages else []

    print("=== Multilingual Neuron Distribution Analysis ===")
    print(f"Target languages: {languages}")
    print(f"Lang neuron results: {args.lang_neuron_results}")

    expertise_data_dict = {}
    for lang in languages:
        expertise_data = load_language_neuron_data(
            args.lang_neuron_results, args.model_name, lang
        )
        expertise_data_dict[lang] = expertise_data

    valid_data = {lang: df for lang, df in expertise_data_dict.items() if df is not None}
    if not valid_data:
        print("Error: No valid expertise data found for any language")
        print("Using default expert configuration")
        expert_numbers = [3] * 28
    else:
        print(f"Successfully loaded data for languages: {list(valid_data.keys())}")
        layer_neuron_counts = analyze_layer_neuron_distribution(valid_data)
        expert_numbers = determine_expert_numbers(
            layer_neuron_counts, args.min_experts, args.max_experts
        )

    expert_str = save_expert_configuration(
        expert_numbers, args.output_config, languages
    )

    print("\n=== Expert Configuration Complete ===")
    print(f"Expert configuration: {expert_str}")
    print(f"Saved to: {args.output_config}")
    print(f"\nExpert distribution:")
    print(f"- Min experts: {min(expert_numbers)}")
    print(f"- Max experts: {max(expert_numbers)}")
    print(f"- Average: {np.mean(expert_numbers):.2f}")

    return expert_numbers

if __name__ == "__main__":
    main()
