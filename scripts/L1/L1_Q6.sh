#!/bin/bash

# 错误处理
set -e
trap 'echo "错误发生在第 $LINENO 行"; exit 1' ERR

# 设置模型路径
MODEL_PATH_1="本地模型1路径"
MODEL_PATH_2="本地模型2路径"

# 设置API配置
API_KEY="your_key"
API_BASE_URL="api model的base url"
API_MODEL_NAME="api_model_name"

GPU_IDS="4,5"

# --------------------------
# 配置文件路径（相对项目根目录）
# --------------------------
# 获取脚本绝对路径
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
# 计算项目根目录
PROJECT_ROOT=$(dirname  "$(dirname "$SCRIPT_DIR")")
# 输入文件
INPUT_FILE="${PROJECT_ROOT}/data/L1_Q6.tsv"
# 输出目录
OUTPUT_DIR="${PROJECT_ROOT}/output/L1_Q6"
LOG_DIR="${PROJECT_ROOT}/logs/L1_Q6"
# Python脚本
PYTHON_SCRIPT="${PROJECT_ROOT}/src/L1/L1_Q6.py"
# --------------------------
# 目录结构验证
# --------------------------
check_prerequisites() {
    # 检查Python脚本
    if [ ! -f "$PYTHON_SCRIPT" ]; then
        echo "错误：Python脚本不存在，预期路径：$PYTHON_SCRIPT"
        echo "当前项目根目录：$PROJECT_ROOT"
        exit 1
    fi
    # 检查输入文件
    if [ ! -f "$INPUT_FILE" ]; then
        echo "错误：输入文件不存在，预期路径：$INPUT_FILE"
        exit 1
    fi
    # 创建输出目录
    mkdir -p "$OUTPUT_DIR" "$LOG_DIR" || {
        echo "目录创建失败"
        exit 1
    }
}
# --------------------------
# 构建命令
# --------------------------
get_base_cmd() {
    echo "python \"${PYTHON_SCRIPT}\" \
    --input_file \"${INPUT_FILE}\" \
    --output_dir \"${OUTPUT_DIR}\" \
    --log_dir \"${LOG_DIR}\" \
    --log_level INFO \
    --version v1.1 \
    --prompt_template \"你是一位专业的金融分析师，擅长图表趋势分析和数据解读。我将提供一张金融图表和相关的单选题(A/B/C/D四个选项)。请你:

1. 仔细分析图表中的趋势变化、波动特征和关键转折点
2. 结合专业知识对数据走势进行判断
3. 只需输出一个正确选项的字母(A/B/C/D)
4. 确保基于客观数据得出结论，避免主观臆测

注：这是趋势分析的单选题，请只选择一个最准确的答案。\""
}
# --------------------------
# 主执行流程
# --------------------------
check_prerequisites

echo "================================"
echo "项目根目录: $PROJECT_ROOT"
echo "Python脚本路径: $PYTHON_SCRIPT"
echo "输入文件路径: $INPUT_FILE"
echo "输出目录: $OUTPUT_DIR"
echo "日志目录: $LOG_DIR"
echo "================================"

# 运行模型评测
run_model() {
    local model_name=$1
    local model_path=$2
    local model_type=$3
    shift 3
    
    echo "开始运行 ${model_name} 模型评测..."
    echo "----------------------------------------"
    
    local cmd="$(get_base_cmd) \
        --model_path \"${model_path}\" \
        --top_k 1 \
        --model_type ${model_type} $@"
    echo $cmd
    if ! eval "$cmd"; then
        echo "警告：${model_name} 模型运行失败"
        return 1
    fi
    
    echo "${model_name} 模型评测完成"
    echo "----------------------------------------"
    sleep 10
}

# 运行API模型评测
run_api_model() {
    echo "开始运行 API 模型评测..."
    echo "----------------------------------------"
    
    local cmd="$(get_base_cmd) \
        --use_api \
        --api_key \"${API_KEY}\" \
        --api_base_url \"${API_BASE_URL}\" \
        --api_model_name \"${API_MODEL_NAME}\" \
        --gpu_ids \"${GPU_IDS}\""
    
    if ! eval "$cmd"; then
        echo "警告：API 模型运行失败"
        return 1
    fi
    
    echo "API 模型评测完成"
    echo "----------------------------------------"
}

# 主函数
main() {
    check_prerequisites
    # 取消注释以下行来运行相应的模型
    #run_api_model
    #取消注释以运行相应的本地模型
    #run_model "模型1名称" "$MODEL_PATH_1" "model_type" "--tp" "2" "--gpu_ids" "4,5"
    #run_model "模型2名称" "$MODEL_PATH_2" "model_type" "--tp" "2" "--gpu_ids" "4,5"
    echo "所有模型评测完成！"
}

# 执行主函数
main