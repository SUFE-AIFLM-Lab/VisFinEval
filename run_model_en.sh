#!/bin/bash

# Error handling
set -e
trap 'echo "Error occurred at line $LINENO"; exit 1' ERR

# ========== Configuration Section ==========
# General settings
PYTHON_SCRIPT="visfineval.py"   # Modify with actual path
INPUT_FILE="your_test_file.tsv"  # Modify with actual path

# Extract TSV filename (without extension and path) for subdirectory
TSV_NAME=$(basename "$INPUT_FILE" .tsv)

# 1. First define all parameters
USE_API=true   # true/false
API_MODEL_NAME="your_model_name"
MODEL_PATH="your_model_address"

# 2. Then set MODEL_NAME based on USE_API
if [ "${USE_API}" = true ]; then
    MODEL_NAME="$API_MODEL_NAME"
else
    MODEL_NAME=$(basename "$MODEL_PATH")
fi

# Output and log directories
BASE_OUTPUT_DIR="your_output_address"
BASE_LOG_DIR="your_logs_address"
OUTPUT_DIR="$BASE_OUTPUT_DIR/$TSV_NAME/$MODEL_NAME"
LOG_DIR="$BASE_LOG_DIR/$TSV_NAME/$MODEL_NAME"

N=20000  # Only evaluate first N samples, modify as needed

# Local model parameters
GPU_IDS="0,1,2,3"  # Modify with available GPU IDs
TOP_K=1
MODEL_TYPE="qwen"
TP=4

# API model parameters
API_KEY="your_model_key"
API_BASE_URL="your_url"

# Judge model parameters (Qwen2.5-72B)
JUDGE_API_KEY="your_key"

# ========== Logging Functions ==========
log() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $message"
}

info() { log "INFO" "$@"; }
warn() { log "WARN" "$@"; }
error() { log "ERROR" "$@"; }

# ========== GPU Availability Check ==========
check_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
        error "nvidia-smi command not available, ensure NVIDIA drivers are properly installed"
        exit 1
    fi
    
    # Get available GPU count
    GPU_COUNT=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
    if [ "$GPU_COUNT" -eq 0 ]; then
        error "No available GPUs detected"
        exit 1
    fi
    
    info "Detected $GPU_COUNT GPU devices"
    
    # Check if specified GPUs are available
    IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
    for gpu_id in "${GPU_ARRAY[@]}"; do
        if ! nvidia-smi -i "$gpu_id" &> /dev/null; then
            error "GPU $gpu_id is not available"
            exit 1
        fi
        
        # Get GPU utilization
        local gpu_util=$(nvidia-smi -i "$gpu_id" --query-gpu=utilization.gpu --format=csv,noheader,nounits)
        local gpu_mem=$(nvidia-smi -i "$gpu_id" --query-gpu=memory.used --format=csv,noheader,nounits)
        local gpu_total_mem=$(nvidia-smi -i "$gpu_id" --query-gpu=memory.total --format=csv,noheader,nounits)
        
        info "GPU $gpu_id status:"
        info "  - Utilization: ${gpu_util}%"
        info "  - Memory usage: ${gpu_mem}MB / ${gpu_total_mem}MB"
    done
    
    #nvidia-smi
}

# ========== Model File Check ==========
check_model() {
    if [ "$USE_API" = false ]; then
        if [ ! -d "$MODEL_PATH" ]; then
            error "Model path does not exist: $MODEL_PATH"
            exit 1
        fi
        info "Local model path check passed: $MODEL_PATH"
        
        # Check required config files
        if [ ! -f "$MODEL_PATH/config.json" ]; then
            error "Missing config file: config.json"
            exit 1
        fi
        
        # Check for safetensors files
        if ! ls "$MODEL_PATH"/*.safetensors 1> /dev/null 2>&1; then
            error "No model weight files found (*.safetensors)"
            exit 1
        fi
        
        # Display model file information
        info "Model file list:"
        for f in "$MODEL_PATH"/*.safetensors; do
            info "  - $(basename "$f") ($(du -h "$f" | cut -f1))"
        done
        info "Model file integrity check passed"
        
        # Display model configuration
        if [ -f "$MODEL_PATH/config.json" ]; then
            info "Model configuration:"
            python -c "
import json
with open('$MODEL_PATH/config.json') as f:
    config = json.load(f)
print(f\"  - Model type: {config.get('model_type', 'unknown')}\")
print(f\"  - Hidden size: {config.get('hidden_size', 'unknown')}\")
print(f\"  - Number of layers: {config.get('num_hidden_layers', 'unknown')}\")
print(f\"  - Attention heads: {config.get('num_attention_heads', 'unknown')}\")
"
        fi
    else
        info "Using API mode, skipping local model check"
    fi
}

# ========== Preprocessing: Extract first N samples ==========
info "Starting data preprocessing..."
TMP_DIR="$BASE_OUTPUT_DIR/$TSV_NAME/tmp"
mkdir -p "$TMP_DIR"
TMP_INPUT="$TMP_DIR/${TSV_NAME}_head${N}.tsv"
head -n 1 "$INPUT_FILE" > "$TMP_INPUT"
tail -n +2 "$INPUT_FILE" | head -n $N >> "$TMP_INPUT"
info "Data preprocessing complete, temporary file created: $TMP_INPUT"
info "Sample count: $N"

# ========== Directory Check ==========
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
info "Created output directory structure:"
info "  - Dataset directory: $BASE_OUTPUT_DIR/$TSV_NAME"
info "  - Model output directory: $OUTPUT_DIR"
info "  - Log directory: $LOG_DIR"

# ========== Run Evaluation ==========
info "================================"
info "Evaluation configuration:"
info "Evaluation script: $PYTHON_SCRIPT"
info "Input file: $TMP_INPUT"
info "Output directory: $OUTPUT_DIR"
info "Log directory: $LOG_DIR"
info "GPU configuration: $GPU_IDS"
info "Model type: $MODEL_TYPE"
info "TP configuration: $TP"
info "================================"

# Check GPU availability
info "Checking GPU availability..."
check_gpu

# Check model
info "Checking model..."
check_model

# Set CUDA visible devices
export CUDA_VISIBLE_DEVICES=$GPU_IDS
info "Set CUDA_VISIBLE_DEVICES=$GPU_IDS"

# 4. Finally call python and pass --run_model_name "$MODEL_NAME"
if [ "$USE_API" = true ]; then
    info ">>> Running API model evaluation <<<"
    info "API base URL: $API_BASE_URL"
    info "API model name: $API_MODEL_NAME"
    
    python "$PYTHON_SCRIPT" \
        --input_file "$TMP_INPUT" \
        --output_dir "$OUTPUT_DIR" \
        --log_dir "$LOG_DIR" \
        --log_level INFO \
        --gpu_ids "$GPU_IDS" \
        --use_api \
        --api_key "$API_KEY" \
        --api_base_url "$API_BASE_URL" \
        --api_model_name "$API_MODEL_NAME" \
        --top_k $TOP_K \
        --model_type $MODEL_TYPE \
        --zhipu_key "$JUDGE_API_KEY" \
        --run_model_name "$MODEL_NAME" 2>&1 | grep -v "AttributeError: 'ActorHandle' object has no attribute 'exit'"
else
    info ">>> Running local model evaluation <<<"
    info "Local model path: $MODEL_PATH"
    
    python "$PYTHON_SCRIPT" \
        --input_file "$TMP_INPUT" \
        --output_dir "$OUTPUT_DIR" \
        --log_dir "$LOG_DIR" \
        --log_level INFO \
        --gpu_ids "$GPU_IDS" \
        --model_path "$MODEL_PATH" \
        --top_k $TOP_K \
        --model_type $MODEL_TYPE \
        --tp $TP \
        --zhipu_key "$JUDGE_API_KEY" \
        --run_model_name "$MODEL_NAME" 2>&1 | grep -v "AttributeError: 'ActorHandle' object has no attribute 'exit'"
fi

# Check evaluation results
RESULT_FILE=$(ls -t "$OUTPUT_DIR"/*.json | head -n1)
if [ -f "$RESULT_FILE" ]; then
    info "Evaluation complete! Result file: $RESULT_FILE"
    
    # Display summary statistics
    info "Evaluation statistics:"
    python -c "
import json
with open('$RESULT_FILE') as f:
    data = json.load(f)
print(f\"Dataset: {data['metadata']['args']['input_file']}\")
print(f\"Model: {data['metadata']['args']['model_path'] if not data['metadata']['args']['use_api'] else data['metadata']['args']['api_model_name']}\")
print(f\"Total samples: {data['summary']['total_samples']}\")
print(f\"Perfect dialogues: {data['summary']['perfect_dialogues']}\")
print(f\"Perfect dialogue rate: {data['summary']['perfect_dialogue_rate']:.2%}\")
print(f\"Average sample accuracy: {data['summary']['average_sample_accuracy']:.2%}\")
print('\nRound statistics:')
for round_name, stats in data['summary']['round_statistics'].items():
    print(f\"{round_name}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})\")"

    # Clean up temporary files
    if [ -d "$TMP_DIR" ]; then
        rm -rf "$TMP_DIR"
        info "Cleaned up temporary directory: $TMP_DIR"
    fi
else
    error "No evaluation result file found!"
    exit 1
fi