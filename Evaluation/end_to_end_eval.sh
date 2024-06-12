#!/bin/bash
set -e


# Check if exactly all arguments are provided
if [ $# -ne 8 ]; then
    echo "Usage: $0 <dataset_path> <instruction_template_name> <model_name> <exp_name> <limits> <use_api> <cuda> <batch_size>"
    exit 1
fi

# Assign arguments to descriptive named variables
dataset_path="$1"
instruction_template_name="$2"
model_name="$3"
exp_name="$4"
limits="$5"
use_api="$6"
cuda="$7"
batch_size="$8"

# Get the current date and time in Eastern Time
current_time=$(TZ="America/Los_Angeles" date +"%Y-%m-%d_%H-%M-%S")
pid_file="logs/PIDs/PIDs_from_embedding_pipeline_${exp_name}_${current_time}.pids"

# Function to log the PID of the current script and subprocesses
log_pid() {
    echo "$1: $2" >> "$pid_file"
}

# Log the PID of the current script
log_pid "Main Script" $$

echo "PIPELINE STARTED"
start_time=$(TZ="America/Los_Angeles" date +"%Y-%m-%d_%H-%M-%S")
echo $start_time
echo "=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-"

# Print the variable names and values
echo "Dataset path: $dataset_path"
echo "Instruction template name: $instruction_template_name"
echo "Embedding model name: $model_name"
echo "Experiment name: $exp_name"
echo "Selection limits: $limits"
echo "Use API: $use_api"
echo "CUDA: $cuda"
echo "batch_size: $batch_size"

echo "=================================================================="

# Create text_list for embedding
python3 pre_process/dataset2embedding_pre.py --dataset ${dataset_path} --instruction_name ${instruction_template_name} --exp_name ${exp_name} &
pid=$!
log_pid "pre_process/dataset2embedding_pre.py" $pid
wait $pid
echo "=================================================================="


# Embed the text_list
python3 embedding/embed.py --text_path "./embedding/text_to_emb/${exp_name}.pickle" --model_name ${model_name} --exp_name ${exp_name} --use_api ${use_api} --cuda ${cuda} --batch_size ${batch_size} &
pid=$!
log_pid "embedding/embed.py" $pid
wait $pid

echo "Finish embedding..."
echo "=================================================================="

# Parse the embedding output

python3 post_process/embedding_scoring.py --model_name ${model_name} --model_output_path "./embedding/embedding_results/${exp_name}.pickle" --parsed_output_path ./post_process/parsed_output/${exp_name}.pickle --dataset_path ${dataset_path} --limit_k ${limits} &
pid=$!
log_pid "post_process/embedding_scoring.py" $pid
wait $pid


# Post process and evaluation
python3 post_process/evaluate.py --model_name ${model_name} --input_path "./post_process/parsed_output/${exp_name}.pickle" --limit_k ${limits} --dataset_path ${dataset_path} --exp_name ${exp_name} --logging_csv "post_process/logs.csv" &
pid=$!
log_pid "post_process/evaluate.py" $pid
wait $pid

echo "Finish post process and evaluation..."
echo "The results will be added to post_process/logs.csv"
echo "===============================Finished==================================="

# Note: The script will stop executing at the first command that fails due to 'set -e'.

