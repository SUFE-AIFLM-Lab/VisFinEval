#!/bin/bash

# 错误处理
set -e
trap 'echo "错误发生在第 $LINENO 行"; exit 1' ERR

#------------------------------
# 用户配置区（必须修改）
#------------------------------
# 设置GPU ID（多个GPU用逗号分隔）
GPU_IDS="0,1,2,3,4,5,6,7"
# 获取脚本绝对路径
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
# 计算项目根目录
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
# 数据根目录
DATA_ROOT="${PROJECT_ROOT}/data"
# 脚本根目录
SCRIPTS_ROOT="${PROJECT_ROOT}/scripts"
# Python脚本根目录
SRC_ROOT="${PROJECT_ROOT}/src"
# 模型配置区
#------------------------------

# API 模型配置（格式：API_KEY, API_BASE_URL, 模型名称）
API_CONFIGS=(
    #"your_api_key_1;https://api.provider1.com/v1;model_name_1"
    #"your_api_key_2;https://api.provider2.com/v1;model_name_2"
)


# 本地模型配置（格式：模型名称, 模型路径, 模型类型, 额外参数）
LOCAL_MODEL_CONFIGS=(
    #"InternVL2_5-8B;/root/autodl-tmp/model/InternVL2_5-8B;qwen;--tp 2 --gpu_ids 4,5"
   # "llava-v1.6-mistral-7b;/root/new_data/llava-v1.6-mistral-7b;qwen;--tp 2 --gpu_ids 6,7"
)

#------------------------------
# 路径配置（自动生成）
#------------------------------

#------------------------------
declare -A SCRIPT_MAPPING=(
    # L1基础
    ["L1_Q1"]="L1/L1_Q1.py"
    ["L1_Q2"]="L1/L1_Q2.py"
    ["L1_Q3"]="L1/L1_Q3.py"
    ["L1_Q4"]="L1/L1_Q4.py"
    ["L1_Q5"]="L1/L1_Q5.py"
    ["L1_Q6"]="L1/L1_Q6.py"
    # L2复杂
    ["L2_Q1"]="L2/L2_Q1_Q2.py"    # 共享脚本
    ["L2_Q2"]="L2/L2_Q1_Q2.py"    # 共享脚本
    ["L2_Q3"]="L2/L2_Q3.py"
    # L3极限
    ["L3_Q1"]="L3/L3_Q1.py"
    ["L3_Q2_1"]="L3/L3_Q2_1_2.py"  # 共享脚本
    ["L3_Q2_2"]="L3/L3_Q2_1_2.py"  # 共享脚本
    ["L3_Q2_3_1"]="L3/L3_Q2_3_1.py"
    ["L3_Q2_3_2"]="L3/L3_Q2_3_2.py"
    ["L3_Q3"]="L3/L3_Q3.py"
    ["L3_Q4"]="L3/L3_Q4.py"
)

declare -A INPUT_FILES=(
    ["L1_Q1"]="${DATA_ROOT}/L1_Q1.tsv"
    ["L1_Q2"]="${DATA_ROOT}/L1_Q2.tsv"
    ["L1_Q3"]="${DATA_ROOT}/L1_Q3.tsv"
    ["L1_Q4"]="${DATA_ROOT}/L1_Q4.tsv"
    ["L1_Q5"]="${DATA_ROOT}/L1_Q5.tsv"
    ["L1_Q6"]="${DATA_ROOT}/L1_Q6.tsv"
    ["L2_Q1"]="${DATA_ROOT}/L2_Q1.tsv"
    ["L2_Q2"]="${DATA_ROOT}/L2_Q2.tsv"
    ["L2_Q3"]="${DATA_ROOT}/L2_Q3.tsv"
    ["L3_Q1"]="${DATA_ROOT}/L3_Q1.tsv"
    ["L3_Q2_1"]="${DATA_ROOT}/L3_Q2_1.tsv"
    ["L3_Q2_2"]="${DATA_ROOT}/L3_Q2_2.tsv"
    ["L3_Q2_3_1"]="${DATA_ROOT}/L3_Q2_3_1.tsv"
    ["L3_Q2_3_2"]="${DATA_ROOT}/L3_Q2_3_2.tsv"
    ["L3_Q3"]="${DATA_ROOT}/L3_Q3.tsv"
    ["L3_Q4"]="${DATA_ROOT}/L3_Q4.tsv"
)

#------------------------------
# Prompt模板配置
#------------------------------

# L3-Q1 特殊提示
L3_Q1_IMAGE_PROMPT='你是一位专业熟悉金融图表并拥有广泛金融和财务知识的金融学专家，我将提供给你一道四个选项的单项选择题。请回答此问题。

注意事项：
1. 只需输出正确选项的字母(A/B/C/D)
2. 请基于图表数据进行分析
3. 确保答案的准确性和客观性
4. 题目为单选题，请只输出一个选项。'
L3_Q1_MARKDOWN_PROMPT='你是一位专业的金融分析师，我将提供给你一个markdown格式的表格数据和一道四个选项的单项选择题。请根据表格数据回答此问题。

注意事项：
1. 只需输出正确选项的字母(A/B/C/D)
2. 请仔细分析markdown中的数据
3. 确保答案的准确性和客观性
4. 题目为单选题，请只输出一个选项。'

# 通用提示模板
declare -A PROMPT_TEMPLATES=(
    # L1 系列
    ["L1_Q1"]='你是一位专业熟悉金融图表并拥有广泛金融和财务知识的金融学专家，我将提供给你一道三选的单项选择题。请回答此问题。

注意事项：
1. 只需输出正确选项的字母(A/B/C)
2. 请基于图表数据进行分析
3. 确保答案的准确性和客观性
4. 题目为单选题，请只输出一个选项。'
    ["L1_Q2"]='你是一位专业熟悉金融图表并拥有广泛金融和财务知识的金融学专家，我将提供给你一道四个选项的单项选择题。请回答此问题。

注意事项：
1. 只需输出正确选项的字母(A/B/C/D)
2. 请基于图表数据进行分析
3. 确保答案的准确性和客观性
4. 题目为单选题，请只输出一个选项。'
    ["L1_Q3"]='你是一位专业熟悉金融图表并拥有广泛金融和财务知识的金融学专家，我将提供给你一道四个选项的多选题。请回答此问题。

注意事项：
1. 只需输出正确选项的字母(A/B/C/D)
2. 请基于图表数据进行分析
3. 确保答案的准确性和客观性
4. 题目为多选题，多个选项使用,进行分隔。'
    ["L1_Q4"]='你是一位专业熟悉金融图表并拥有广泛金融和财务知识的金融学专家，请基于图表严谨地分析以下判断题。

请仔细观察图表中的数据趋势、数值变化和关键特征,运用专业的数据分析能力,对判断内容的正确性做出准确评估。

注意事项:
1. 需要基于客观数据进行判断,避免主观臆测
2. 关注判断中涉及的具体时间点、数值区间或趋势变化
3. 确保判断的严谨性和准确性
4. 请只回答\"是\"或\"否\"。'
    ["L1_Q5"]='你是一位专业熟悉金融图表并拥有广泛金融和财务知识的金融学专家，我将提供给你一道四个选项的单项选择题。请回答此问题。

注意事项：
1. 只需输出正确选项的字母(A/B/C/D)
2. 请基于图表数据进行分析
3. 确保答案的准确性和客观性
4. 题目为单选题，请只输出一个选项。'
    ["L1_Q6"]='你是一位专业的金融分析师，擅长图表趋势分析和数据解读。我将提供一张金融图表和相关的单选题(A/B/C/D四个选项)。请你:

1. 仔细分析图表中的趋势变化、波动特征和关键转折点
2. 结合专业知识对数据走势进行判断
3. 只需输出一个正确选项的字母(A/B/C/D)
4. 确保基于客观数据得出结论,避免主观臆测

注:这是趋势分析的单选题,请只选择一个最准确的答案。'
    # L2 系列
    ["L2_Q1"]='你是一位专业熟悉金融图表并拥有广泛金融和财务知识的金融学专家，我将提供给你一道三个选项的单项选择题。请回答此问题.

注意事项：
1. 只需输出正确选项的字母(A/B/C)
2. 请基于图表数据进行分析
3. 确保答案的准确性和客观性
4. 题目为单选题，请只输出一个选项。'
    ["L2_Q2"]='你是一位专业熟悉金融图表并拥有广泛金融和财务知识的金融学专家，我将提供给你一道四个选项的单项选择题。请回答此问题.

注意事项：
1. 只需输出正确选项的字母(A/B/C/D)
2. 请基于图表数据进行分析
3. 确保答案的准确性和客观性
4. 题目为单选题，请只输出一个选项.'
    ["L2_Q3"]='你是一位专业熟悉金融图表并拥有广泛金融和财务知识的金融学专家，请仔细阅读背景故事和图表信息，并回答以下单项选择题.

请注意：
1. 只需输出正确选项的字母(A/B/C/D)
2. 请基于图表数据和背景信息进行分析
3. 确保答案的准确性和客观性
4. 题目为单选题，请只输出一个选项.'
    # L3 系列
    ["L3_Q2_1"]='你是一位专业熟悉金融图表并拥有广泛金融和财务知识的金融学专家，我将提供给你一道四个选项的单项选择题。请回答此问题.

注意事项：
1. 只需输出正确选项的字母(A/B/C/D)
2. 请基于图表数据进行分析
3. 确保答案的准确性和客观性
4. 题目为单选题，请只输出一个选项.'
    ["L3_Q2_2"]='请是一位专业熟悉金融图表并拥有广泛金融和财务知识的金融学专家，我将提供给你一道四个选项的单项选择题。请回答此问题.

注意事项：
1. 只需输出正确选项的字母(A/B/C/D)
2. 请基于图表数据进行分析
3. 确保答案的准确性和客观性
4. 题目为单选题，请只输出一个选项.'
    ["L3_Q2_3_1"]='你是一位专业熟悉金融图表并拥有广泛金融和财务知识的金融学专家，请仔细阅读背景故事和图表信息，并回答以下单项选择题.

请注意：
1. 只需输出正确选项的字母(A/B/C/D)
2. 请基于图表数据和背景信息进行分析
3. 确保答案的准确性和客观性
4. 题目为单选题，请只输出一个选项.'
    ["L3_Q2_3_2"]='你是一位专业熟悉金融图表并拥有广泛金融和财务知识的金融学专家，我将提供给你一道四个选项的单项选择题。请回答此问题.

注意事项：
1. 只需输出正确选项的字母(A/B/C/D)
2. 请基于图表数据进行分析
3. 确保答案的准确性和客观性
4. 题目为单选题，请只输出一个选项.'
    ["L3_Q3"]='作为一位专业的金融分析师，请仔细阅读背景故事和图表信息，并回答以下选择题.

请注意：
1. 只需输出正确选项的字母(A/B/C/D)
2. 请基于图表数据和背景信息进行分析
3. 确保答案的准确性和客观性
4. 题目为单选题，请只输出一个选项.'
    ["L3_Q4"]='你是一位专业熟悉金融图表并拥有广泛金融和财务知识的金融学专家，我将提供给你一道金融问题。请基于图表数据进行详细分析，确保答案的准确性和客观性，并归纳出最终结论。请你一步步详细推理，并给出最终结论。请务必将你的最终答案将数字和单位一起使用 \\boxed{{}} 包裹返回，且答案中不包含其他多余内容。\'
)


#------------------------------
# 核心函数
#------------------------------
run_evaluation() {
    local task=$1
    local model_type=$2
    local model_name=$3
    local api_key=${4:-""}
    local api_url=${5:-""}
    local model_path=${6:-""}
    local extra_args=${7:-""}

    # 构造输出路径
    local output_dir="${PROJECT_ROOT}/output/${task}/${model_name}"
    local log_dir="${PROJECT_ROOT}/logs/${task}/${model_name}"
    mkdir -p "$output_dir" "$log_dir"

    # 获取脚本路径
    local script_path="${SRC_ROOT}/${SCRIPT_MAPPING[$task]}"
    if [[ ! -f "$script_path" ]]; then
        echo "错误：Python脚本不存在，预期路径：$script_path"
        exit 1
    fi

    # 构造基础命令
    local cmd="python \"${script_path}\" \
        --input_file \"${INPUT_FILES[$task]}\" \
        --output_dir \"${output_dir}\" \
        --log_dir \"${log_dir}\" \
        --version \"v1.0\""

    # 添加特殊参数
    case "$task" in
        "L3_Q1")
            cmd+=" --image_prompt \"${PROMPT_TEMPLATES[L3_Q1_IMAGE]}\""
            cmd+=" --markdown_prompt \"${PROMPT_TEMPLATES[L3_Q1_MARKDOWN]}\""
            ;;
    esac

    # 添加模型参数
    if [[ "$model_type" == "API" ]]; then
        cmd+=" --use_api \
              --api_key \"${api_key}\" \
              --api_base_url \"${api_url}\" \
              --api_model_name \"${model_name}\" \
              --gpu_ids \"${GPU_IDS}\""
    else
        cmd+=" --model_path \"${model_path}\" \
              --model_type \"${model_name}\" \
              ${extra_args}"
    fi

    # 执行命令
    echo "▶ 正在运行: $task | 模型: $model_name"
    eval "$cmd" && echo "✓ 成功" || { echo "✗ 失败"; return 1; }
    sleep 1
}

#------------------------------
# 主控制逻辑
#------------------------------
main() {
    echo "======= 开始评测 ======="
    
    # 配置检查
    if [[ ${#API_CONFIGS[@]} -eq 0 && ${#LOCAL_MODEL_CONFIGS[@]} -eq 0 ]]; then
        echo "错误：未配置任何模型，请检查 API_CONFIGS 和 LOCAL_MODEL_CONFIGS"
        exit 1
    fi

    # 处理API模型
    for config in "${API_CONFIGS[@]}"; do
        IFS=';' read -r api_key api_url model_name <<< "$config"
        for task in "${!SCRIPT_MAPPING[@]}"; do
            run_evaluation "$task" "API" "$model_name" "$api_key" "$api_url" "" "" &
            (( $(jobs -r | wc -l) >= MAX_PARALLEL )) && wait -n
        done
    done

    # 处理本地模型
    for config in "${LOCAL_MODEL_CONFIGS[@]}"; do
        IFS=';' read -r name path type args <<< "$config"
        for task in "${!SCRIPT_MAPPING[@]}"; do
            run_evaluation "$task" "LOCAL" "$type" "" "" "$path" "$args" &
            (( $(jobs -r | wc -l) >= MAX_PARALLEL )) && wait -n
        done
    done

    wait
    echo "======= 评测完成 ======="
}

main