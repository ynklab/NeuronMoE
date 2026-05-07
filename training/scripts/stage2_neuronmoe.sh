#!/bin/bash

# NeuronMoE Stage 2: Router Training
# Train the MoE router on mixed old+new language data

# ===== Configuration (set these environment variables before running) =====
# LLAMA_FACTORY_DIR: path to LLaMA-Factory installation
# DATA_DIR: path to prepared training data
# OUTPUT_BASE_DIR: base directory for model outputs
# MOE_MODEL_PATH: path to Stage 1 checkpoint

LLAMA_FACTORY_DIR="${LLAMA_FACTORY_DIR:?Please set LLAMA_FACTORY_DIR}"
DATA_DIR="${DATA_DIR:?Please set DATA_DIR}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:?Please set OUTPUT_BASE_DIR}"
MOE_MODEL_PATH="${MOE_MODEL_PATH:?Please set MOE_MODEL_PATH (Stage 1 checkpoint)}"

export PYTHONPATH="${LLAMA_FACTORY_DIR}"
ROOT_DIR="${LLAMA_FACTORY_DIR}"

BASE_MODEL_PATH="${BASE_MODEL_PATH:-meta-llama/Llama-3.2-3B}"
PREPARED_DATA_PATH="${DATA_DIR}"
OUTPUT_DIR="${OUTPUT_BASE_DIR}/NeuronMoE-stage2"
LOG_DIR="${OUTPUT_BASE_DIR}/logs"
LOG_PATH="${LOG_DIR}/train-stage2.log"

# Dataset: new language(s) + existing languages
DATASET="${G1_DATASETS:-el2b},en50k,zh50k,es50k"
GPU_NUM=${GPU_NUM:-8}
echo "Using dataset configuration: $DATASET"

# Create output and log directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

echo "=== Stage 2 Router Training Started $(date) ===" >> ${LOG_PATH}
echo "Base Model: $BASE_MODEL_PATH" >> ${LOG_PATH}
echo "MoE Model: $MOE_MODEL_PATH" >> ${LOG_PATH}
nvidia-smi >> ${LOG_PATH} 2>&1

# Dynamically generate dataset_info.json
DATA_LINK_DIR="$ROOT_DIR/data-prepared"
mkdir -p "$DATA_LINK_DIR"
DATASET_INFO_PATH="$DATA_LINK_DIR/dataset_info.json"
echo "Generating dataset_info.json for datasets: $DATASET"
echo "Using language files: ${G1_LANG_FILES:-el-llama-2B.jsonl}"

cat > $DATASET_INFO_PATH << EOF
{
$(IFS=',' read -ra DATASETS_ARRAY <<< "$DATASET"
  IFS=',' read -ra FILES_ARRAY <<< "${G1_LANG_FILES:-el-llama-2B.jsonl}"
  for i in "${!DATASETS_ARRAY[@]}"; do
    dataset_name="${DATASETS_ARRAY[i]}"
    if [[ "$dataset_name" == "en50k" ]]; then
      file_name="SlimPajama-50K.jsonl"
      language="old"
      group_tag="g0"
    elif [[ "$dataset_name" == "zh50k" ]]; then
      file_name="2023-14_zh_middle_0009-50K.jsonl"
      language="old"
      group_tag="g0"
    elif [[ "$dataset_name" == "es50k" ]]; then
      file_name="es-llama-50K.jsonl"
      language="old"
      group_tag="g0"
    else
      file_name="${FILES_ARRAY[i]:-${FILES_ARRAY[0]}}"
      language="new"
      group_tag="g1"
    fi
    echo "    \"$dataset_name\": {"
    echo "        \"file_name\": \"$file_name\","
    echo "        \"file_sha1\": \"\","
    echo "        \"language\": \"$language\","
    echo "        \"group_tag\": \"$group_tag\","
    echo "        \"columns\": {"
    echo "            \"prompt\": \"text\""
    echo "        }"
    if [ $i -lt $((${#DATASETS_ARRAY[@]}-1)) ]; then
      echo "    },"
    else
      echo "    }"
    fi
  done)
}
EOF

# Create symlinks for new language data
IFS=',' read -ra FILES_ARRAY <<< "${G1_LANG_FILES:-el-llama-2B.jsonl}"
for file_name in "${FILES_ARRAY[@]}"; do
    source_file="$PREPARED_DATA_PATH/$file_name"
    target_file="$DATA_LINK_DIR/$file_name"
    if [ -f "$source_file" ]; then
        echo "Creating symlink: $source_file -> $target_file"
        ln -sf $source_file $target_file
    else
        echo "Warning: Source file not found: $source_file"
    fi
done

# Create symlinks for existing language data
old_files=("SlimPajama-50K.jsonl" "2023-14_zh_middle_0009-50K.jsonl" "es-llama-50K.jsonl")
for file_name in "${old_files[@]}"; do
    source_file="$PREPARED_DATA_PATH/$file_name"
    target_file="$DATA_LINK_DIR/$file_name"
    if [ -f "$source_file" ]; then
        echo "Creating symlink for existing language: $source_file -> $target_file"
        ln -sf $source_file $target_file
    else
        echo "Warning: Source file not found: $source_file"
    fi
done

# Load expert configuration
if [ -n "$ADA_MOE_NUM_EXPERTS_LIST" ]; then
    echo "Using ADA_MOE_NUM_EXPERTS_LIST from environment: $ADA_MOE_NUM_EXPERTS_LIST"
else
    EXPERT_CONFIG_PATH="${EXPERT_CONFIG_PATH:-./expert_allocation/configs/expert_config_neuron_guided.txt}"
    if [ -f "$EXPERT_CONFIG_PATH" ]; then
        source "$EXPERT_CONFIG_PATH"
        echo "Using optimized expert config from file: $ADA_MOE_NUM_EXPERTS_LIST"
    else
        ADA_MOE_NUM_EXPERTS_LIST="5,5,5,4,3,3,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4"
        echo "Using default expert config: $ADA_MOE_NUM_EXPERTS_LIST"
    fi
fi

# WandB configuration (optional)
export WANDB_PROJECT="${WANDB_PROJECT:-neuronmoe-stage2}"
export WANDB_RUN_NAME="NeuronMoE-stage2-$(date +%Y%m%d-%H%M%S)"

# DeepSpeed Router training
deepspeed --num_gpus ${GPU_NUM} --master_port=9903 $ROOT_DIR/src/train_bash.py \
    --deepspeed $ROOT_DIR/config/ds_config.json \
    --stage pt \
    --model_name_or_path $BASE_MODEL_PATH \
    --adapter_name_or_path $MOE_MODEL_PATH \
    --finetuning_type moe \
    --ada_moe_num_experts_list $ADA_MOE_NUM_EXPERTS_LIST \
    --lpr_loss_coef 0.1 \
    --train_only_router \
    --do_train \
    --dataset_dir $DATA_LINK_DIR \
    --dataset $DATASET \
    --max_samples 100000 \
    --generate_lang_mask \
    --preprocessing_num_workers 128 \
    --cutoff_len 512 \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_total_limit 10 \
    --save_steps 100 \
    --learning_rate 5e-5 \
    --num_train_epochs 1.0 \
    --report_to wandb \
    --plot_loss \
    --bf16 \
    >> ${LOG_PATH} 2>&1

if [ $? -eq 0 ]; then
    echo "=== Stage 2 Router Training Complete $(date) ===" >> ${LOG_PATH}
    echo "Output model: $OUTPUT_DIR" >> ${LOG_PATH}
else
    echo "=== Stage 2 Router Training Failed $(date) ===" >> ${LOG_PATH}
    exit 1
fi
