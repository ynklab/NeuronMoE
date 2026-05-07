#!/bin/bash

# NeuronMoE Evaluation Script
# Evaluate the trained MoE model on multilingual benchmarks

# ===== Configuration =====
# BASE_MODEL_PATH: path to base model
# PEFT_MODEL_PATH: path to trained MoE adapter
# OUTPUT_PATH: directory for evaluation results

BASE_MODEL_PATH="${BASE_MODEL_PATH:?Please set BASE_MODEL_PATH}"
PEFT_MODEL_PATH="${PEFT_MODEL_PATH:?Please set PEFT_MODEL_PATH}"
OUTPUT_PATH="${OUTPUT_PATH:-./eval_results}"
LOG_DIR="${LOG_DIR:-./logs-eval}"

# lm-evaluation-harness lm_eval command path
LM_EVAL="${LM_EVAL:-lm_eval}"

mkdir -p "$OUTPUT_PATH"
mkdir -p "$LOG_DIR"

CUR_LOG="$LOG_DIR/$(basename $PEFT_MODEL_PATH)"
mkdir -p "$CUR_LOG"

echo "=== Evaluation Started $(date) ==="
echo "Base Model: $BASE_MODEL_PATH"
echo "PEFT Model: $PEFT_MODEL_PATH"

# MMLU (multilingual)
export CUDA_VISIBLE_DEVICES=0
nohup $LM_EVAL --model hf \
        --model_args pretrained=$BASE_MODEL_PATH,peft=$PEFT_MODEL_PATH,dtype="float16" \
        --tasks mmlu_tr,m_mmlu_es,m_mmlu_hu \
        --device cuda:0 \
        --num_fewshot 4 \
        --output_path $OUTPUT_PATH \
        --batch_size auto:4 \
        >> ${CUR_LOG}/mmlu-tr_es_hu.log 2>&1 &

export CUDA_VISIBLE_DEVICES=1
nohup $LM_EVAL --model hf \
        --model_args pretrained=$BASE_MODEL_PATH,peft=$PEFT_MODEL_PATH,dtype="float16" \
        --tasks mmlu \
        --device cuda:0 \
        --num_fewshot 5 \
        --output_path $OUTPUT_PATH \
        --batch_size 4 \
        >> ${CUR_LOG}/mmlu-en.log 2>&1 &

export CUDA_VISIBLE_DEVICES=2
nohup $LM_EVAL --model hf \
        --model_args pretrained=$BASE_MODEL_PATH,peft=$PEFT_MODEL_PATH,dtype="float16" \
        --tasks cmmlu \
        --device cuda:0 \
        --num_fewshot 5 \
        --output_path $OUTPUT_PATH \
        --batch_size auto:4 \
        >> ${CUR_LOG}/mmlu-zh.log 2>&1 &

export CUDA_VISIBLE_DEVICES=3
nohup $LM_EVAL --model hf \
        --model_args pretrained=$BASE_MODEL_PATH,peft=$PEFT_MODEL_PATH,dtype="float16" \
        --tasks mmlu_el \
        --num_fewshot 5 \
        --output_path $OUTPUT_PATH \
        --batch_size 1 \
        >> ${CUR_LOG}/mmlu-el.log 2>&1 &

# HellaSwag (multilingual)
export CUDA_VISIBLE_DEVICES=4
nohup $LM_EVAL --model hf \
        --model_args pretrained=$BASE_MODEL_PATH,peft=$PEFT_MODEL_PATH,dtype="float16" \
        --tasks hellaswag_el,hellaswag_hu,hellaswag_tr \
        --device cuda:0 \
        --num_fewshot 10 \
        --output_path $OUTPUT_PATH \
        --batch_size auto:4 \
        >> ${CUR_LOG}/hellaswag-el_hu_tr.log 2>&1 &

export CUDA_VISIBLE_DEVICES=5
nohup $LM_EVAL --model hf \
        --model_args pretrained=$BASE_MODEL_PATH,peft=$PEFT_MODEL_PATH,dtype="float16" \
        --tasks hellaswag,hellaswag_es,hellaswag_zh  \
        --device cuda:0 \
        --num_fewshot 10 \
        --output_path $OUTPUT_PATH \
        --batch_size auto:4 \
        >> ${CUR_LOG}/hellaswag-en_es_zh.log 2>&1 &

# ARC Challenge (multilingual)
export CUDA_VISIBLE_DEVICES=6
nohup $LM_EVAL --model hf \
        --model_args pretrained=$BASE_MODEL_PATH,peft=$PEFT_MODEL_PATH,dtype="float16" \
        --tasks arc_challenge,arc_es,arc_hu,arc_zh,arc_el,arc_tr \
        --device cuda:0 \
        --num_fewshot 25 \
        --output_path $OUTPUT_PATH \
        --batch_size 4 \
        >> ${CUR_LOG}/arc-G1.log 2>&1 &

# Belebele (multilingual)
export CUDA_VISIBLE_DEVICES=7
nohup $LM_EVAL --model hf \
        --model_args pretrained=$BASE_MODEL_PATH,peft=$PEFT_MODEL_PATH,dtype="float16" \
        --tasks belebele_zho_Hans,belebele_ell_Grek,belebele_hun_Latn,belebele_tur_Latn,belebele_spa_Latn,belebele_eng_Latn \
        --device cuda:0 \
        --num_fewshot 5 \
        --output_path $OUTPUT_PATH \
        --batch_size 4 \
        >> ${CUR_LOG}/belebele-G1.log 2>&1 &

echo "All evaluation jobs launched. Check logs in $CUR_LOG"
