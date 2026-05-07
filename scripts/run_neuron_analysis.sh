#!/bin/bash
# Step 2: Language-specific neuron analysis

set -e

NEURONMOE_OUTPUT_DIR="${NEURONMOE_OUTPUT_DIR:?Please set NEURONMOE_OUTPUT_DIR}"
SAMPLE_DATA_DIR="${SAMPLE_DATA_DIR:?Please set SAMPLE_DATA_DIR (path to sample-data)}"
LANGUAGES="${LANGUAGES:-tr el hu en es zh}"
MODEL="${MODEL:-Llama-3.2-3B}"

echo "=== Step 2: Neuron Analysis ==="
echo "Output: $NEURONMOE_OUTPUT_DIR"
echo "Languages: $LANGUAGES"
echo "Model: $MODEL"

# Create sense data for new languages
echo "--- Creating sense data ---"
uv run python expert_allocation/create_sense_data.py \
    --prepared_data_path "$SAMPLE_DATA_DIR" \
    --output_dir neuron_analysis/assets/Language/sense \
    --languages $LANGUAGES

# Run neuron analysis for each language
echo "--- Running neuron analysis ---"
NEURONMOE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd neuron_analysis
export NEURONMOE_OUTPUT_DIR
# main_prod_env.sh invokes scripts/*.py directly; route them through uv so the
# project venv (with deepspeed, seaborn, etc.) is used.
export PYTHON_BIN="uv run --project $NEURONMOE_DIR python"

for lang in $LANGUAGES; do
    echo "  Analyzing $lang..."
    bash main_prod_env.sh "$MODEL compute_responses Language $lang 1000 on_p50 expertise_limited_1000_top off flores200 0"
    bash main_prod_env.sh "$MODEL compute_expertise Language $lang 1000 on_p50 expertise_limited_1000_top off flores200 0"
    bash main_prod_env.sh "$MODEL limit_expertise Language $lang 1000 on_p50 expertise_limited_1000_top off flores200 0"
done

cd ..
echo "=== Neuron analysis complete ==="
