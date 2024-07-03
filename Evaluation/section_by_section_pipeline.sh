#!/bin/bash
set -e

# Check if exactly all arguments are provided
if [ $# -ne 7 ]; then
    echo "Usage: $0 <dataset_path> <max_tokens> <prompt_template_name> <model_name> <exp_name> <limits> <regeneration>"
    exit 1
fi

# Assign arguments to descriptive named variables
dataset_path="$1"
max_tokens="$2"
prompt_template_name="$3"
model_name="$4"
exp_name="$5"
limits="$6"
regeneration="$7"

# Print the variable names and values
echo "Dataset path: $dataset_path"
echo "Max tokens: $max_tokens"
echo "Prompt template name: $prompt_template_name"
echo "Generation model name: $model_name"
echo "Experiment name: $exp_name"
echo "Selection limits: $limits"
echo "Regeneration flag: $regeneration"
echo "=================================================================="

echo "START SECTION BY SECTION PIPELINE"
echo "=================================================================="

# Create prompt dict SbS for generation
python3 pre_process/dataset2prompt_dict_SbS.py --dataset "$dataset_path" --prompt_template_name "$prompt_template_name" --prompt_dict_name "$exp_name" --limit_k "$limits" --output_path "./generation/prompts/section_by_section/"

echo "Finish first round prompt dict generation..."
echo "=================================================================="

# LLM Generation
python3 generation/generation.py --prompt_path "generation/prompts/section_by_section/${exp_name}/prompt.pickle" --output_path "generation/outputs/${exp_name}" --max_workers 25 --timeout_seconds 240 --generation_model ${model_name} --temperature 0 --max_tokens ${max_tokens} &
pid=$!
echo "Generation PID: $pid"
wait $pid

echo "Finish first round generation..."
echo "=================================================================="

# First round parsing and aggregation
python3 post_process/parsing_aggregation_sbs.py --model_name ${model_name} --model_output_path "generation/outputs/${exp_name}/collected_results.pickle" --parsed_mapping_path "generation/prompts/section_by_section/${exp_name}/pid2cid_aggregation.pickle" --original_prompt_dict_path "generation/prompts/section_by_section/${exp_name}/" --regen_prompt_dict_path "generation/prompts/section_by_section/${exp_name}/prompt_aggregation.pickle" --dataset_path "$dataset_path" --prompt_template_name "sbs_aggregation" --limit_k ${limits}

echo "Finish first round parsing and aggregation..."
echo "=================================================================="

# LLM Generation aggregation first round
python3 generation/generation.py --prompt_path "generation/prompts/section_by_section/${exp_name}/prompt_aggregation.pickle" --output_path "generation/outputs/${exp_name}_aggregation" --max_workers 25 --timeout_seconds 240 --generation_model ${model_name} --temperature 0 --max_tokens ${max_tokens}

echo "=================================================================="
echo "Finish first round agggregation generation..."
echo "=================================================================="

python3 post_process/parsing_first_round_SbS.py --model_name ${model_name} --model_output_path "generation/outputs/${exp_name}_aggregation/collected_results.pickle" --parsed_output_path "./post_process/parsed_output/${exp_name}.pickle" --original_prompt_dict_path "generation/prompts/section_by_section/${exp_name}/prompt_aggregation.pickle" --regen_prompt_dict_path "generation/prompts/section_by_section/${exp_name}/prompt_aggregation_regen.pickle" --limit_k ${limits} --pid2cid_path "generation/prompts/section_by_section/${exp_name}/pid2cid_aggregation.pickle" --dataset_path "$dataset_path" 

echo "Finish first round aggregation generation parsing..."
echo "=================================================================="


if [[ $regeneration == "True" ]]; then
    echo "Start second round generation and parsing..."

    # LLM Generation second round
    python3 generation/generation.py --prompt_path "generation/prompts/section_by_section/${exp_name}/prompt_aggregation_regen.pickle" --output_path "generation/outputs/${exp_name}_aggregation_regen" --max_workers 25 --timeout_seconds 240 --generation_model ${model_name} --temperature 0 --max_tokens ${max_tokens} --multi_turn ${regeneration}

    echo "Finish second round aggregation generation..."
    echo "=================================================================="

    # Second round parsing
    python3 post_process/parsing_second_round_SbS.py --model_name ${model_name} --model_output_path "generation/outputs/${exp_name}_aggregation_regen/collected_results.pickle" --parsed_output_path "post_process/parsed_output/${exp_name}.pickle" --original_prompt_dict_path "generation/prompts/section_by_section/${exp_name}/prompt_aggregation_regen.pickle" --pid2cid_path "generation/prompts/section_by_section/${exp_name}/pid2cid_aggregation.pickle"

    echo "Finish second round parsing..."
    echo "=================================================================="
    
fi

echo "Start evaluation..."

# Post process and evaluation
python3 post_process/evaluate.py --model_name ${model_name} --input_path "./post_process/parsed_output/${exp_name}.pickle" --limit_k ${limits} --dataset_path ${dataset_path} --exp_name ${exp_name} --logging_csv "post_process/logs.csv"

echo "Finish post process and evaluation..."
echo "The results will be added to post_process/logs.csv"
echo "===============================Finished==================================="

# Note: The script will stop executing at the first command that fails due to 'set -e'.

# bash section_by_section_pipeline.sh ./dataset/test_long_prompt_dataset.pickle 1000 original gpt-3.5-turbo-0125 SbS_test_pipeline 20 True
