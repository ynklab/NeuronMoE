#!/usr/bin/env python3
# Multilingual neuron distribution analysis - Expert number determination script
# Determines mutually beneficial optimized expert allocation from integrated neuron distribution
# of existing and new languages (high-resource + low-resource language variant)

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
    parser.add_argument("--high_resource_lang", type=str, required=True,
                        help="High resource language (base)")
    parser.add_argument("--low_resource_langs", type=str, required=True,
                        help="Low resource languages (space-separated)")
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
    model_path = model_name.replace("/", "_")
    expertise_path = Path(lang_neuron_path) / "Language" / model_path / "sense" / language / "expertise"

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

def extract_layer_neurons(df, layer_idx):
    """Extract neuron list for a specific layer from DataFrame"""
    if df is None:
        return []

    if 'layer' in df.columns and 'unit' in df.columns:
        df_copy = df.copy()
        df_copy['layer_num'] = df['layer'].str.extract(r'model\.layers\.(\d+)\.').astype(int)
        layer_data = df_copy[df_copy['layer_num'] == layer_idx]
        return layer_data['unit'].tolist()

    elif f'layer_{layer_idx}' in df.columns:
        return df[f'layer_{layer_idx}'].dropna().tolist()

    return []

def analyze_neuron_with_averaged_lowresource(expertise_data_dict,
                                              high_resource_lang,
                                              low_resource_langs,
                                              num_layers=24):
    """Determine expert count per layer using high-resource neurons + averaged low-resource neurons"""
    layer_neuron_counts = {}

    for layer_idx in range(num_layers):
        # High-resource language neuron count
        hr_neurons = set(extract_layer_neurons(
            expertise_data_dict.get(high_resource_lang), layer_idx
        ))
        hr_neuron_count = len(hr_neurons)

        # Low-resource language neuron counts
        lr_neuron_counts = []
        for lang in low_resource_langs:
            neurons = set(extract_layer_neurons(
                expertise_data_dict.get(lang), layer_idx
            ))
            lr_neuron_counts.append(len(neurons))

        # Average neuron count across low-resource languages
        lr_avg_neuron_count = np.mean(lr_neuron_counts) if lr_neuron_counts else 0

        # Total = high-resource + average low-resource
        total_neuron_count = hr_neuron_count + lr_avg_neuron_count

        layer_neuron_counts[f'layer_{layer_idx}'] = total_neuron_count

        print(f"Layer {layer_idx}: HR={hr_neuron_count}, "
              f"LR_avg={lr_avg_neuron_count:.0f}, "
              f"Total={total_neuron_count:.0f}")

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

    high_resource_lang = args.high_resource_lang
    low_resource_langs = args.low_resource_langs.split()
    all_languages = [high_resource_lang] + low_resource_langs

    print("=== Multilingual Neuron Distribution Analysis ===")
    print(f"High-resource language (base): {high_resource_lang}")
    print(f"Low-resource languages: {low_resource_langs}")
    print(f"Lang neuron results: {args.lang_neuron_results}")

    expertise_data_dict = {}
    for lang in all_languages:
        expertise_data = load_language_neuron_data(
            args.lang_neuron_results, args.model_name, lang
        )
        expertise_data_dict[lang] = expertise_data

    valid_data = {lang: df for lang, df in expertise_data_dict.items() if df is not None}
    if not valid_data:
        print("Error: No valid expertise data found for any language")
        print("Using default expert configuration")
        expert_numbers = [3] * 28
    elif high_resource_lang not in valid_data:
        print(f"Error: High resource language '{high_resource_lang}' data not found")
        print("Using default expert configuration")
        expert_numbers = [3] * 28
    else:
        print(f"Successfully loaded data for languages: {list(valid_data.keys())}")

        # Determine number of layers based on model architecture
        if "llama" in args.model_name.lower():
            num_layers = 28
        elif "qwen" in args.model_name.lower():
            num_layers = 24
        else:
            num_layers = 24  # default
        layer_neuron_counts = analyze_neuron_with_averaged_lowresource(
            valid_data,
            high_resource_lang,
            low_resource_langs,
            num_layers=num_layers
        )

        expert_numbers = determine_expert_numbers(
            layer_neuron_counts, args.min_experts, args.max_experts
        )

    expert_str = save_expert_configuration(
        expert_numbers, args.output_config,
        f"{high_resource_lang} (base) + {' '.join(low_resource_langs)}"
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
