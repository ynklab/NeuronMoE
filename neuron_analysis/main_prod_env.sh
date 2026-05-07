#!/bin/bash

#------- Program execution -------#

# ARGUMENTS
echo ${1}
args=(${1})

model_name=${args[0]}
phase=${args[1]}
datapath=${args[2]}
language=${args[3]}
num_units=${args[4]}
force_value=${args[5]}
expert_file=${args[6]}
japanese_control=${args[7]}
translation_task=${args[8]}
prompt_format_id_for_translation=${args[9]}

echo ${model_name}
echo ${phase}
echo ${datapath}
echo ${language}
echo ${num_units}
echo ${force_value}
echo ${expert_file}
echo ${japanese_control}
echo ${translation_task}

# Environment variables (set these before running)
# NEURONMOE_MODEL_DIR: directory containing model files (optional, for local models)
# NEURONMOE_OUTPUT_DIR: directory for output files
model_path="${NEURONMOE_MODEL_DIR:-}"
base_path="${NEURONMOE_OUTPUT_DIR:-./output/}"

# BOS setting
# xglms: </s> is automatically set.
# bloom: nothing is automatically set. So we explicitly set </s>.
# llama2: <s> is automatically set.
if [[ ${model_name} == *"xglm"* ]]; then
  model_name2="facebook/${model_name}"
  prompt=""
elif [[ ${model_name} == *"bloom"* ]]; then
  model_name2="bigscience/${model_name}"
  prompt="</s>"
elif [[ ${model_name} == *"Swallow"* ]]; then
  model_name2="tokyotech-llm/${model_name}"
  prompt=""
elif [[ ${model_name} == *"Llama-2"* ]]; then
  model_name2="meta-llama/${model_name}"
  prompt=""
elif [[ ${model_name} == *"Llama-3"* ]]; then
  model_name2="meta-llama/${model_name}"
  prompt=""
elif [[ ${model_name} == *"EvoLLM"* ]]; then
  model_name2="SakanaAI/${model_name}"
  prompt=""
elif [[ ${model_name} == *"DeepSeek"* ]]; then
  model_name2="deepseek-ai/${model_name}"
  prompt=""
else
  echo "Unsupported model: ${model_name}"
  exit 1
fi

if [ ${phase} == "compute_responses" ] || [ ${phase} == "compute_all" ]; then
  ${PYTHON_BIN:-python} scripts/compute_responses.py --model-name-or-path ${model_name2} --data-path assets/${datapath} --responses-path ${base_path}${datapath} --concepts sense/${language}
fi

if [ ${phase} == "compute_expertise" ] || [ ${phase} == "compute_all" ]; then
  ${PYTHON_BIN:-python} scripts/compute_expertise.py --root-dir ${base_path}${datapath} --model-name ${model_name2} --concepts assets/${datapath}/${language} --concepts sense/${language}
fi

if [ ${phase} == "limit_expertise" ] || [ ${phase} == "compute_all" ]; then
  ${PYTHON_BIN:-python} scripts/make_limited_expert_exe.py --model-name ${model_name} --language ${language} --num-units ${num_units} --base-path ${base_path}${datapath}/
fi

if [ ${phase} == "generate_activated" ]; then
  ${PYTHON_BIN:-python} scripts/generate_seq_lang.py --model-name-or-path ${model_name2} --expertise ${base_path}${datapath}/${model_name}/sense/${language}/expertise/${expert_file}.csv --expertise2 ${base_path}${datapath}/${model_name}/sense/ja/expertise/${expert_file}.csv --length 64 --seed 1 101 --metric ap --forcing ${force_value} --num-units ${num_units} --eos --top-n 1 --results-file ${base_path}${datapath}/${model_name}/sense/${language}/expertise/created_sentence_${force_value}_${num_units}_${expert_file}_${japanese_control}.csv --temperature 0.8 --prompt "${prompt}" --japanese_control ${japanese_control}
fi

if [ ${phase} == "generate_activated_condition" ]; then
  translation_file="assets_translation/translation_text_${translation_task}_ja_${language}.pkl"
  ${PYTHON_BIN:-python} scripts/generate_seq_lang.py --model-name-or-path ${model_name2} --expertise ${base_path}${datapath}/${model_name}/sense/${language}/expertise/${expert_file}.csv --expertise2 ${base_path}${datapath}/${model_name}/sense/ja/expertise/${expert_file}.csv --length 256 --seed 1 31 --metric ap --forcing ${force_value} --num-units ${num_units} --eos --top-n 1 --results-file ${base_path}${datapath}/${model_name}/sense/${language}/expertise/created_sentence_${force_value}_${num_units}_${expert_file}_${japanese_control}_${translation_task}_condition_${prompt_format_id_for_translation}.csv --temperature 0.0 --prompt ${translation_file} --prompt_format_id_for_translation ${prompt_format_id_for_translation} --japanese_control ${japanese_control}
fi
