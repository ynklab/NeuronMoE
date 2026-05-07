#!/bin/bash

# NeuronMoE Stage 1: MoE Expert Training
# Train MoE experts for new language group using neuron-guided expert allocation

# ===== Configuration (set these environment variables before running) =====
# LLAMA_FACTORY_DIR: path to LLaMA-Factory installation
# DATA_DIR: path to prepared training data
# OUTPUT_BASE_DIR: base directory for model outputs
# EXPERT_CONFIG_PATH: path to expert configuration file (optional)

LLAMA_FACTORY_DIR="${LLAMA_FACTORY_DIR:?Please set LLAMA_FACTORY_DIR}"
DATA_DIR="${DATA_DIR:?Please set DATA_DIR}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:?Please set OUTPUT_BASE_DIR}"

NAMING="${NAMING:-neuronmoe}"
export PYTHONPATH="${LLAMA_FACTORY_DIR}"
ROOT_DIR="${LLAMA_FACTORY_DIR}"

BASE_MODEL_PATH="${BASE_MODEL_PATH:-meta-llama/Llama-3.2-3B}"
PREPARED_DATA_PATH="${DATA_DIR}"
OUTPUT_DIR="${OUTPUT_BASE_DIR}/NeuronMoE-stage1-${NAMING}"
LOG_DIR="${OUTPUT_BASE_DIR}/logs"
LOG_PATH="${LOG_DIR}/train-stage1-${NAMING}.log"

# Dataset configuration (from environment or default)
DATASET=${G1_DATASETS:-"el2b"}
GPU_NUM=${GPU_NUM:-8}
echo "Using dataset configuration: $DATASET"

# Load expert configuration
if [ -n "$ADA_MOE_NUM_EXPERTS_LIST" ]; then
    echo "Using ADA_MOE_NUM_EXPERTS_LIST from environment: $ADA_MOE_NUM_EXPERTS_LIST"
else
    EXPERT_CONFIG_PATH="${EXPERT_CONFIG_PATH:-./expert_allocation/configs/expert_config_neuron_guided.txt}"
    if [ -f "$EXPERT_CONFIG_PATH" ]; then
        source "$EXPERT_CONFIG_PATH"
        echo "Using optimized expert config from file: $ADA_MOE_NUM_EXPERTS_LIST"
    else
        # Default configuration (before neuron analysis)
        ADA_MOE_NUM_EXPERTS_LIST="5,5,5,4,3,3,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4"
        echo "Using default expert config: $ADA_MOE_NUM_EXPERTS_LIST"
    fi
fi

# Create output and log directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

echo "=== Stage 1 MoE Expert Training Started $(date) ===" >> ${LOG_PATH}
echo "Base Model: $BASE_MODEL_PATH" >> ${LOG_PATH}
echo "Expert Config: $ADA_MOE_NUM_EXPERTS_LIST" >> ${LOG_PATH}
nvidia-smi >> ${LOG_PATH} 2>&1

# Create data symlinks
DATA_LINK_DIR="$ROOT_DIR/data-prepared"
mkdir -p "$DATA_LINK_DIR"

# Dynamically generate dataset_info.json
DATASET_INFO_PATH="$DATA_LINK_DIR/dataset_info.json"
echo "Generating dataset_info.json for datasets: $DATASET"
echo "Using language files: ${G1_LANG_FILES:-el-llama-2B.jsonl}"
cat > $DATASET_INFO_PATH << EOF
{
$(IFS=',' read -ra DATASETS_ARRAY <<< "$DATASET"
  IFS=',' read -ra FILES_ARRAY <<< "${G1_LANG_FILES:-el-llama-2B.jsonl}"
  for i in "${!DATASETS_ARRAY[@]}"; do
    dataset_name="${DATASETS_ARRAY[i]}"
    file_name="${FILES_ARRAY[i]:-${FILES_ARRAY[0]}}"
    echo "    \"$dataset_name\": {"
    echo "        \"file_name\": \"$file_name\","
    echo "        \"file_sha1\": \"\","
    echo "        \"language\": \"new\","
    echo "        \"group_tag\": \"g1\","
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

# Create symlinks to data files
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

# WandB configuration (optional)
export WANDB_PROJECT="${WANDB_PROJECT:-neuronmoe-stage1}"
export WANDB_RUN_NAME="NeuronMoE-stage1-${NAMING}-$(date +%Y%m%d-%H%M%S)"

# DeepSpeed MoE training
deepspeed --num_gpus ${GPU_NUM} --master_port=9903 $ROOT_DIR/src/train_bash.py \
    --deepspeed $ROOT_DIR/config/ds_config.json \
    --stage pt \
    --model_name_or_path $BASE_MODEL_PATH \
    --finetuning_type moe \
    --topk 2 \
    --ada_moe_num_experts_list $ADA_MOE_NUM_EXPERTS_LIST \
    --aux_loss_coef 0.01 \
    --do_train \
    --dataset_dir $DATA_LINK_DIR \
    --dataset $DATASET \
    --preprocessing_num_workers 128 \
    --cutoff_len 1024 \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_total_limit 10 \
    --save_steps 50 \
    --learning_rate 1e-4 \
    --num_train_epochs 0.5 \
    --report_to wandb \
    --plot_loss \
    --bf16 \
    >> ${LOG_PATH} 2>&1

if [ $? -eq 0 ]; then
    echo "=== Stage 1 MoE Expert Training Complete $(date) ===" >> ${LOG_PATH}
    echo "Output model: $OUTPUT_DIR" >> ${LOG_PATH}
    echo "Expert config used: $ADA_MOE_NUM_EXPERTS_LIST" >> ${LOG_PATH}
else
    echo "=== Stage 1 MoE Expert Training Failed $(date) ===" >> ${LOG_PATH}
    exit 1
fi
