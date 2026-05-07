#!/bin/bash
# Step 3: Neuron-guided expert allocation

set -e

NEURON_RESULTS_DIR="${NEURON_RESULTS_DIR:?Please set NEURON_RESULTS_DIR (path to neuron analysis output)}"
MODE="${MODE:-3lang}"  # "single" or "3lang"

echo "=== Step 3: Expert Allocation (mode=$MODE) ==="

if [ "$MODE" = "3lang" ]; then
    HIGH_RESOURCE_LANG="${HIGH_RESOURCE_LANG:-en}"
    LOW_RESOURCE_LANGS="${LOW_RESOURCE_LANGS:-el hu tr}"
    echo "High-resource: $HIGH_RESOURCE_LANG"
    echo "Low-resource: $LOW_RESOURCE_LANGS"

    uv run python expert_allocation/analyze_neuron_distribution_3lang.py \
        --lang_neuron_results "$NEURON_RESULTS_DIR" \
        --high_resource_lang "$HIGH_RESOURCE_LANG" \
        --low_resource_langs "$LOW_RESOURCE_LANGS" \
        --output_config expert_allocation/configs/expert_config_neuron_guided.txt
else
    LANGUAGES="${LANGUAGES:?Please set LANGUAGES (space-separated)}"
    echo "Languages: $LANGUAGES"

    uv run python expert_allocation/analyze_neuron_distribution.py \
        --lang_neuron_results "$NEURON_RESULTS_DIR" \
        --languages "$LANGUAGES" \
        --output_config expert_allocation/configs/expert_config_neuron_guided.txt
fi

# Visualize
ALL_LANGUAGES="${ALL_LANGUAGES:-en es zh el hu tr}"
echo "--- Visualizing neuron distribution ---"
uv run python expert_allocation/visualize_neuron_distribution.py \
    --lang_neuron_dir "$NEURON_RESULTS_DIR" \
    --languages $ALL_LANGUAGES

echo "=== Expert allocation complete ==="
echo "Config saved to: expert_allocation/configs/expert_config_neuron_guided.txt"
