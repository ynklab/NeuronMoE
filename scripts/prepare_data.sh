#!/bin/bash
# Step 1: Download and preprocess multilingual training data

set -e

OUTPUT_DIR="${OUTPUT_DIR:?Please set OUTPUT_DIR (path to save data)}"
NEW_LANGS="${NEW_LANGS:-el hu tr}"
OLD_LANGS="${OLD_LANGS:-es}"

echo "=== Step 1: Data Preparation ==="
echo "Output directory: $OUTPUT_DIR"
echo "New languages: $NEW_LANGS"
echo "Old languages to sample: $OLD_LANGS"

# Download data from HuggingFace
echo "--- Downloading data ---"
bash data/download.sh

# Preprocess CulturaX data (for new languages)
echo "--- Preprocessing CulturaX data ---"
for lang in $NEW_LANGS; do
    echo "  Processing $lang..."
    uv run python data/CulturaX_preprocess.py \
        --lang "$lang" \
        --data_dir "$OUTPUT_DIR/CulturaX" \
        --output_dir "$OUTPUT_DIR/sample-data"
done

# Sample existing language data
echo "--- Sampling existing language data ---"
for lang in $OLD_LANGS; do
    echo "  Sampling $lang..."
    uv run python data/CulturaX_sample.py \
        --lang "$lang" \
        --data_dir "$OUTPUT_DIR/CulturaX" \
        --output_dir "$OUTPUT_DIR/sample-data"
done

# Sample English (SlimPajama)
echo "  Sampling en (SlimPajama)..."
uv run python data/SlimPajama_sample.py \
    --data_path "$OUTPUT_DIR/SlimPajama-627B" \
    --output "$OUTPUT_DIR/sample-data/SlimPajama-50K.jsonl"

# Sample Chinese (SkyPile)
echo "  Sampling zh (SkyPile)..."
uv run python data/SkyPile_sample.py \
    --input "$OUTPUT_DIR/SkyPile-150B/data/2023-14_zh_middle_0009.jsonl" \
    --output "$OUTPUT_DIR/sample-data/2023-14_zh_middle_0009-50K.jsonl"

echo "=== Data preparation complete ==="
echo "Data saved to: $OUTPUT_DIR/sample-data"
