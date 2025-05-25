#!/bin/bash

# 错误处理
set -e
trap 'echo "错误发生在第 $LINENO 行"; exit 1' ERR

# ========== 配置区 ==========
# 通用
PYTHON_SCRIPT="visfineval.py"   # 按实际路径修改
INPUT_FILE="your_test_file.tsv"  # 按实际路径修改

# 从INPUT_FILE提取TSV文件名（不含扩展名和路径）作为子目录
TSV_NAME=$(basename "$INPUT_FILE" .tsv)

# 1. 先定义所有参数
USE_API=true   # true/false
API_MODEL_NAME="your_model_name"
MODEL_PATH="your_model_address"

# 2. 再根据 USE_API 赋值 MODEL_NAME
if [ "${USE_API}" = true ]; then
    MODEL_NAME="$API_MODEL_NAME"
else
    MODEL_NAME=$(basename "$MODEL_PATH")
fi

# 输出和日志目录
BASE_OUTPUT_DIR="your_output_address"
BASE_LOG_DIR="your_logs_address"
OUTPUT_DIR="$BASE_OUTPUT_DIR/$TSV_NAME/$MODEL_NAME"
LOG_DIR="$BASE_LOG_DIR/$TSV_NAME/$MODEL_NAME"

N=20000  # 只评测前N条，按需修改

# 本地模型参数
GPU_IDS="0,1,2,3"  # 修改为实际可用的GPU ID
TOP_K=1
MODEL_TYPE="qwen"
TP=4

# API模型参数
API_KEY="your_model_key"
API_BASE_URL="your_url"

# 裁判模型参数（Qwen2.5-72B）
JUDGE_API_KEY="your_key"

# ========== 日志函数 ==========
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

# ========== 检查GPU可用性 ==========
check_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
        error "nvidia-smi 命令不可用，请确保NVIDIA驱动已正确安装"
        exit 1
    fi
    
    # 获取可用的GPU数量
    GPU_COUNT=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
    if [ "$GPU_COUNT" -eq 0 ]; then
        error "未检测到可用的GPU"
        exit 1
    fi
    
    info "检测到 $GPU_COUNT 个GPU设备"
    
    # 检查指定的GPU是否可用
    IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
    for gpu_id in "${GPU_ARRAY[@]}"; do
        if ! nvidia-smi -i "$gpu_id" &> /dev/null; then
            error "GPU $gpu_id 不可用"
            exit 1
        fi
        
        # 获取GPU使用情况
        local gpu_util=$(nvidia-smi -i "$gpu_id" --query-gpu=utilization.gpu --format=csv,noheader,nounits)
        local gpu_mem=$(nvidia-smi -i "$gpu_id" --query-gpu=memory.used --format=csv,noheader,nounits)
        local gpu_total_mem=$(nvidia-smi -i "$gpu_id" --query-gpu=memory.total --format=csv,noheader,nounits)
        
        info "GPU $gpu_id 状态:"
        info "  - 使用率: ${gpu_util}%"
        info "  - 显存使用: ${gpu_mem}MB / ${gpu_total_mem}MB"
    done
    
    #nvidia-smi
}

# ========== 检查模型文件 ==========
check_model() {
    if [ "$USE_API" = false ]; then
        if [ ! -d "$MODEL_PATH" ]; then
            error "模型路径不存在: $MODEL_PATH"
            exit 1
        fi
        info "本地模型路径检查通过: $MODEL_PATH"
        
        # 检查必需的配置文件
        if [ ! -f "$MODEL_PATH/config.json" ]; then
            error "缺少配置文件: config.json"
            exit 1
        fi
        
        # 检查是否存在safetensors文件
        if ! ls "$MODEL_PATH"/*.safetensors 1> /dev/null 2>&1; then
            error "未找到模型权重文件 (*.safetensors)"
            exit 1
        fi
        
        # 显示模型文件信息
        info "模型文件列表:"
        for f in "$MODEL_PATH"/*.safetensors; do
            info "  - $(basename "$f") ($(du -h "$f" | cut -f1))"
        done
        info "模型文件完整性检查通过"
        
        # 显示模型配置信息
        if [ -f "$MODEL_PATH/config.json" ]; then
            info "模型配置信息:"
            python -c "
import json
with open('$MODEL_PATH/config.json') as f:
    config = json.load(f)
print(f\"  - 模型类型: {config.get('model_type', 'unknown')}\")
print(f\"  - 隐藏层大小: {config.get('hidden_size', 'unknown')}\")
print(f\"  - 层数: {config.get('num_hidden_layers', 'unknown')}\")
print(f\"  - 注意力头数: {config.get('num_attention_heads', 'unknown')}\")
"
        fi
    else
        info "使用API模式，跳过本地模型检查"
    fi
}

# ========== 预处理：截取前N条 ==========
info "开始数据预处理..."
TMP_DIR="$BASE_OUTPUT_DIR/$TSV_NAME/tmp"
mkdir -p "$TMP_DIR"
TMP_INPUT="$TMP_DIR/${TSV_NAME}_head${N}.tsv"
head -n 1 "$INPUT_FILE" > "$TMP_INPUT"
tail -n +2 "$INPUT_FILE" | head -n $N >> "$TMP_INPUT"
info "数据预处理完成，生成临时文件: $TMP_INPUT"
info "样本数量: $N"

# ========== 目录检查 ==========
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
info "创建输出目录结构:"
info "  - 数据集目录: $BASE_OUTPUT_DIR/$TSV_NAME"
info "  - 模型输出目录: $OUTPUT_DIR"
info "  - 日志目录: $LOG_DIR"

# ========== 运行评测 ==========
info "================================"
info "评测配置信息:"
info "评测脚本: $PYTHON_SCRIPT"
info "输入文件: $TMP_INPUT"
info "输出目录: $OUTPUT_DIR"
info "日志目录: $LOG_DIR"
info "GPU配置: $GPU_IDS"
info "模型类型: $MODEL_TYPE"
info "TP配置: $TP"
info "================================"

# 检查GPU可用性
info "开始检查GPU..."
check_gpu

# 检查模型
info "开始检查模型..."
check_model

# 设置CUDA可见设备
export CUDA_VISIBLE_DEVICES=$GPU_IDS
info "设置CUDA_VISIBLE_DEVICES=$GPU_IDS"

# 4. 最后调用 python，并传递 --run_model_name "$MODEL_NAME"
if [ "$USE_API" = true ]; then
    info ">>> 运行API模型评测 <<<"
    info "API基础URL: $API_BASE_URL"
    info "API模型名称: $API_MODEL_NAME"
    
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
    info ">>> 运行本地模型评测 <<<"
    info "本地模型路径: $MODEL_PATH"
    
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

# 检查评测结果
RESULT_FILE=$(ls -t "$OUTPUT_DIR"/*.json | head -n1)
if [ -f "$RESULT_FILE" ]; then
    info "评测完成！结果文件: $RESULT_FILE"
    
    # 显示简要统计信息
    info "评测统计信息:"
    python -c "
import json
with open('$RESULT_FILE') as f:
    data = json.load(f)
print(f\"数据集: {data['metadata']['args']['input_file']}\")
print(f\"模型: {data['metadata']['args']['model_path'] if not data['metadata']['args']['use_api'] else data['metadata']['args']['api_model_name']}\")
print(f\"总样本数: {data['summary']['total_samples']}\")
print(f\"完美对话数: {data['summary']['perfect_dialogues']}\")
print(f\"完美对话率: {data['summary']['perfect_dialogue_rate']:.2%}\")
print(f\"平均样本准确率: {data['summary']['average_sample_accuracy']:.2%}\")
print('\n按轮次统计:')
for round_name, stats in data['summary']['round_statistics'].items():
    print(f\"{round_name}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})\")"

    # 清理临时文件
    if [ -d "$TMP_DIR" ]; then
        rm -rf "$TMP_DIR"
        info "清理临时文件目录: $TMP_DIR"
    fi
else
    error "未找到评测结果文件！"
    exit 1
fi 