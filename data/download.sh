#!/bin/bash

# Download training data from HuggingFace
# Set OUTPUT_DIR to specify where to save the data

OUTPUT_DIR="${OUTPUT_DIR:?Please set OUTPUT_DIR}"
HF_TOKEN="${HF_TOKEN:-}"

TOKEN_ARG=""
if [ -n "$HF_TOKEN" ]; then
    TOKEN_ARG="--token $HF_TOKEN"
fi

# CulturaX: multilingual data (first 30 parquet files following MoE-LPR)
# Supported languages: el/hu/tr/bn/hi/ne/es
for L in el hu tr es; do
  echo "Downloading CulturaX ${L}..."
  huggingface-cli download \
    --repo-type dataset \
    $TOKEN_ARG \
    uonlp/CulturaX \
    --include "${L}/${L}_part_000[0-1][0-9].parquet" \
    --include "${L}/${L}_part_0002[0-9].parquet" \
    --local-dir "${OUTPUT_DIR}/CulturaX"
done

# English: SlimPajama-627B
echo "Downloading SlimPajama (English)..."
huggingface-cli download \
    --repo-type dataset \
    $TOKEN_ARG \
    cerebras/SlimPajama-627B \
    --include "train/chunk1*" \
    --local-dir "${OUTPUT_DIR}/SlimPajama-627B"

# Chinese: SkyPile-150B
echo "Downloading SkyPile (Chinese)..."
huggingface-cli download \
    --repo-type dataset \
    $TOKEN_ARG \
    Skywork/SkyPile-150B \
    --include "data/2023-14_zh_middle_0009.jsonl" \
    --local-dir "${OUTPUT_DIR}/SkyPile-150B"

echo "Download complete. Data saved to: ${OUTPUT_DIR}"
