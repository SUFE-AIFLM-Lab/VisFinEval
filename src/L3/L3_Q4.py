# -*- coding: utf-8 -*-
import logging
from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig
from lmdeploy.vl.constants import IMAGE_TOKEN
import json
import os
import yaml
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import random
from zhipuai import ZhipuAI
import argparse
from openai import OpenAI
import base64
import time
import pandas as pd
import re
# 指定使用的GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
# 初始化chatglmapi
client = ZhipuAI(api_key="your_key")
logging.info("chatglmapi初始化成功")

def setup_logging(config):
    """配置日志记录器"""
    log_dir = Path(config['logging']['log_dir'])
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'image_analysis_{timestamp}.log'
    
    # 修改日志配置
    logging.basicConfig(
        level=logging.INFO,  # 将默认级别改为INFO
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # 输出到控制台
            logging.FileHandler(str(log_file), encoding='utf-8')
        ],
        force=True
    )
    
    # 获取根日志记录器并设置级别
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)  # 改为INFO级别
    
    # 特别设置一些第三方库的日志级别
    logging.getLogger('openai').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)  # 改为INFO级别
    
    logging.info("日志记录器初始化成功")
    return logger



def init_model(model_name, backend_config, log_level='INFO'):
    """初始化模型管道"""
    logger = logging.getLogger(__name__)
    logger.info(f"正在初始化模型: {model_name}")
    try:
        pipe = pipeline(model_name, log_level=log_level, backend_config=backend_config)
        logger.info("模型初始化成功")
        return pipe
    except Exception as e:
        logger.error(f"模型初始化失败: {str(e)}")
        raise

def analyze_images(pipe, image_paths, questions, prompt_template="作为一位专业的金融分析师，请根据图片回答以下问题：", top_k=1, model_type="qwen"):
    """分析图片并进行多轮对话"""
    logger = logging.getLogger(__name__)
    
    # 确保 image_paths 是列表格式
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    
    messages = []
    answers = []
    
    # 第一轮对话需要包含完整信息
    if model_type.lower() == "minicpm":
        content = [dict(type='text', text=f"{prompt_template}\n题目描述：{questions[0]['information']}\n问题：{questions[0]['question']}")]
        for image_path in image_paths:
            content.append({
                'type': 'image_url',
                'image_url': {
                    'max_slice_nums': 11,
                    'url': image_path
                }
            })
    else:
        full_prompt = f"{prompt_template}\n题目描述：{questions[0]['information']}\n问题：{questions[0]['question']}"
        tokens_and_prompt = f"{IMAGE_TOKEN * len(image_paths)}\n{full_prompt}"
        content = [{'type': 'text', 'text': tokens_and_prompt}]
        
        for image_path in image_paths:
            content.append({
                'type': 'image_url',
                'image_url': {
                    'max_dynamic_patch': 12,
                    'url': image_path
                }
            })
    
    messages.append({'role': 'user', 'content': content})
    
    try:
        response = pipe(messages, gen_config=GenerationConfig(top_k=top_k,max_new_tokens = 2048))
        first_answer = response.text
        answers.append(first_answer)
        messages.append({'role': 'assistant', 'content': first_answer})
        
        # 后续轮次只包含问题
        for question in questions[1:]:
            messages.append({
                'role': 'user',
                'content': f"问题：{question['question']}"
            })
            
            response = pipe(messages, gen_config=GenerationConfig(top_k=top_k,max_new_tokens = 2048))
            answer = response.text
            answers.append(answer)
            messages.append({'role': 'assistant', 'content': answer})
            
        return answers
        
    except Exception as e:
        logger.error(f"多轮对话过程中发生错误: {str(e)}")
        raise

def get_output_filename(config, model_name):
    """生成输出文件名"""
    # 创建输出目录
    output_dir = Path(config['data']['output_dir'])
    output_dir.mkdir(exist_ok=True, parents=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = Path(model_name).name
    version = config['evaluation']['version']
    # 返回完整的输出路径
    return output_dir / f"results_{model_name}_v{version}_{timestamp}.jsonl"

def init_api_client(api_key, base_url=None):
    """初始化API客户端"""
    logger = logging.getLogger(__name__)
    try:
        if base_url:
            client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            client = OpenAI(api_key=api_key)
        logger.info("API客户端初始化成功")
        return client
    except Exception as e:
        logger.error(f"API客户端初始化失败: {str(e)}")
        raise

def encode_image(image_path):
    """将图片转换为base64编码"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def analyze_images_api(client, image_paths, questions, model_name, prompt_template="作为一位专业的金融分析师，请根据图片回答以下问题：", max_retries=5, retry_delay=1):
    """使用API进行图片分析的多轮对话"""
    logger = logging.getLogger(__name__)
    
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    
    messages = []
    answers = []
    
    # 准备第一轮对话的图片内容
    user_content = []
    for image_path in image_paths:
        try:
            image_format = Path(image_path).suffix.lower().replace('.', '')
            if image_format == 'jpg':
                image_format = 'jpeg'
            if image_format == 'png':
                image_format = 'png'
            logger.info(f"图片格式: {image_format}")
            base64_image = encode_image(image_path)
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{image_format};base64,{base64_image}"
                }
            })
        except Exception as e:
            logger.error(f"处理图片 {image_path} 时出错: {str(e)}")
            print(image_format)

            raise
    
    # 添加第一个问题（包含完整信息）
    user_content.append({
        "type": "text",
        "text": f"{prompt_template}\n题目描述：{questions[0]['information']}\n问题：{questions[0]['question']}"
    })
    
    messages.append({"role": "user", "content": user_content})
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages
            )
            first_answer = response.choices[0].message.content
            answers.append(first_answer)
            messages.append({"role": "assistant", "content": first_answer})
            break
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                logger.warning(f"第一轮对话失败: {str(e)}，{wait_time}秒后重试...")
                time.sleep(wait_time)
                continue
            else:
                logger.error(f"第一轮对话在 {max_retries} 次尝试后仍然失败")
                raise
    
    # 后续轮次的对话
    for idx, question in enumerate(questions[1:], 2):
        messages.append({
            "role": "user",
            "content": f"问题：{question['question']}"
        })
        
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages
                )
                answer = response.choices[0].message.content
                answers.append(answer)
                messages.append({"role": "assistant", "content": answer})
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    logger.warning(f"第 {idx} 轮对话失败: {str(e)}，{wait_time}秒后重试...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"第 {idx} 轮对话在 {max_retries} 次尝试后仍然失败")
                    raise
    
    return answers

def extract_answer(text):
    """使用正则表达式从文本中提取答案"""
    # 尝试匹配 \boxed{} 中的内容
    boxed_pattern = r'\\boxed{([^}]*)}'
    boxed_match = re.search(boxed_pattern, text)
    if boxed_match:
        return boxed_match.group(1).strip()
    
    # 如果没有 \boxed{}，尝试提取单个字母答案（A/B/C/D）
    letter_pattern = r'\b[A-D]\b'
    letter_matches = re.findall(letter_pattern, text)
    if letter_matches:
        return ' '.join(letter_matches)
    
    # 如果上述都没匹配到，返回原始文本
    return text.strip()

def evaluate_answer(client, model_answer, correct_answer, max_retries=3, retry_delay=1):
    """使用ChatGLM API评估答案是否正确"""
    logger = logging.getLogger(__name__)
    
    # 添加对client的检查
    if client is None:
        logger.error("API客户端未初始化")
        raise ValueError("API客户端未初始化")
        
    prompt = f"""你作为一名专业的答案检察员，请比对以下答案是否一致：
【评估规则】
1. 基础原则：仅判断核心答案是否一致,即判断模型输出答案和正确答案是否一致
2. 数值类：
   - 不同分隔符视为等价（如2,200 = 2200）
   - 若没有单位，则数值必须相同
   - 若带有单位，进行换算表示相同即可
4. 判断类：
   - 核心语义必须准确
6. 特殊处理：
   - 忽略答案后的解释文字

答案检察有效样例1：
模型输出答案: 222,00
正确答案: 22200
评测结果: True

答案检察有效样例2：
模型输出答案: 2224545元钱
正确答案: 2224545元
评测结果: True

答案检察有效样例3：
模型输出答案: 《分析解释一大堆》最后答案是18515元
正确答案: 18515
评测结果: True

答案检察无效样例1：
模型输出答案:2224545元钱
正确答案: 4545155
评测结果: False

答案检察无效样例2：
模型输出答案: 2224545元钱
正确答案: 4545155151
评测结果: False



模型输出答案：{model_answer}
正确答案：{correct_answer}

请仅输出True或False，True表示答案选项完全一致（不区分大小写），False表示答案选项不同。
注意：只需比对选项字母是否一致，忽略分隔符的差异。"""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="glm-4-flash",
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个严格的答案评判专家，判断模型输出答案和正确答案是否一致，只输出True或False。"
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=10,
                stream=False
            )
            
            result = response.choices[0].message.content.strip()
            return result == "True"
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                logger.warning(f"评估答案时第 {attempt + 1} 次调用失败: {str(e)}，{wait_time}秒后重试...")
                time.sleep(wait_time)
                continue
            else:
                logger.error(f"评估答案在 {max_retries} 次尝试后仍然失败: {str(e)}")
                raise

def process_jsonl(input_path, output_path, model_pipe, config):
    """处理JSONL文件并保存多轮对话结果"""
    logger = logging.getLogger(__name__)
    logger.info("开始处理JSONL文件")
    logger.info(f"输入文件路径: {input_path}")
    logger.info(f"输出文件路径: {output_path}")
    
    # 检查model_pipe是否正确初始化
    if model_pipe is None and not config['model'].get('use_api', False):
        logger.error("模型管道未正确初始化")
        raise ValueError("模型管道未正确初始化")
        
    # 初始化评测用的API客户端
    evaluation_client = None
    try:
        evaluation_client = ZhipuAI(api_key="326f7b85389f0c0a57ad0edebbe2a222.fC3o7Xuo7QsGGEYG")
        logger.info("评测用API客户端初始化成功")
    except Exception as e:
        logger.error(f"评测用API客户端初始化失败: {str(e)}")
        raise

    # 如果使用API，初始化推理用API客户端
    api_client = None
    if config['model'].get('use_api', False):
        logger.info("正在初始化推理用API客户端...")
        api_client = init_api_client(
            config['model']['api_key'],
            config['model'].get('base_url')
        )
        logger.info("推理用API客户端初始化成功")

    # 读取TSV文件
    logger.info("正在读取TSV文件...")
    df = pd.read_csv(input_path, sep='\t')
    logger.info(f"成功读取数据，共 {len(df)} 行")
    
    # 将image列中的逗号分隔字符串转换为列表
    logger.info("正在处理图片路径...")
    df['image'] = df['image'].apply(lambda x: [path.strip() for path in x.split(',')])
    
    # 按index分组，获取每组的所有轮次问题
    grouped_data = df.groupby('index')
    logger.info(f"数据分组完成，共 {len(grouped_data)} 个样本")
    
    results = []
    processed_count = 0
    error_count = 0
    
    metadata = {
        "model": config['model'].get('path') or config['model'].get('api_model_name'),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": config
    }
    logger.info(f"元数据准备完成: {metadata}")
    
    # 初始化评测结果统计
    total_evaluated = 0
    correct_count = 0
    round_correct_counts = {}
    
    for index, group in tqdm(grouped_data):
        logger.info(f"\n开始处理样本 {index}")
        # 按round排序
        group_sorted = group.sort_values('round')
        logger.info(f"样本 {index} 包含 {len(group_sorted)} 轮对话")
        
        # 获取第一行的information
        information = group_sorted['information'].iloc[0]
        logger.debug(f"样本 {index} 的题目描述: {information}")
        
        # 构建问题列表
        questions = []
        for _, row in group_sorted.iterrows():
            question_dict = {
                'information': information if row['round'] == 1 else None,
                'question': row['question']
            }
            questions.append(question_dict)
            logger.debug(f"第 {row['round']} 轮问题: {question_dict}")
            
        correct_answers = group_sorted['answer'].tolist()
        logger.debug(f"正确答案列表: {correct_answers}")
        
        # 获取图片路径列表
        image_paths = group_sorted['image'].iloc[0]
        logger.info(f"样本 {index} 的图片路径: {image_paths}")
        
        try:
            logger.info(f"开始处理样本 {index} 的多轮对话")
            # 根据是否使用API选择不同的处理函数
            if config['model'].get('use_api', False):
                logger.info("使用API模式进行推理")
                predicted_answers = analyze_images_api(
                    api_client,
                    image_paths,
                    questions,
                    config['model']['api_model_name'],
                    prompt_template=config['evaluation']['prompt_template']
                )
            else:
                logger.info("使用本地模型进行推理")
                predicted_answers = analyze_images(
                    model_pipe,
                    image_paths,
                    questions,
                    prompt_template=config['evaluation']['prompt_template'],
                    top_k=config['model'].get('top_k', 1),
                    model_type=config['model'].get('model_type', 'qwen')
                )
            
            # 打印每个样本的多轮对话结果
            logger.info(f"\n样本 {index} 的多轮对话结果：")
            round_results = []
            for round_idx, (pred, true) in enumerate(zip(predicted_answers, correct_answers), 1):
                logger.info(f"第 {round_idx} 轮：")
                logger.info(f"问题：{questions[round_idx-1]['question']}")
                if round_idx == 1:
                    logger.info(f"题目描述：{questions[0]['information']}")
                logger.info(f"模型预测：{pred.strip()}")
                logger.info(f"正确答案：{true}")
                
                # 提取答案并保留原始预测
                extracted_pred = extract_answer(pred.strip())
                extracted_true = str(true).strip()
                
                # 使用GLM-4进行答案评估（后续统一评估）
                is_correct = False  # 先初始化为False，后续统一评估
                
                logger.info(f"提取的预测答案：{extracted_pred}")
                logger.info(f"处理后的正确答案：{extracted_true}")
                logger.info(f"待评估状态：等待统一评估")
                
                round_results.append({
                    'round': round_idx,
                    'predicted': pred.strip(),
                    'extracted_predicted': extracted_pred,
                    'correct': str(true),
                    'is_correct': is_correct,
                    'model_raw_answer': pred.strip(),  # 新增原始模型输出
                    'evaluation_method': 'GLM-4'  # 标明评估方式
                })
            
            # 计算该样本的正确率
            correct_count = sum(1 for r in round_results if r['is_correct'])
            sample_accuracy = correct_count / len(round_results) if round_results else 0
            logger.info(f"样本 {index} 的正确率: {sample_accuracy:.2%}")
            
            result = {
                "index": int(index),
                "image_path": image_paths,
                "rounds": round_results,
                "all_correct": all(r['is_correct'] for r in round_results),
                "sample_accuracy": sample_accuracy
            }
            
            processed_count += 1
            logger.info(f"样本 {index} 处理完成")
            
        except Exception as e:
            error_msg = f"处理样本 {index} 时发生错误: {str(e)}"
            logger.error(error_msg)
            logger.exception("详细错误信息：")
            result = {
                "index": int(index),
                "image_path": image_paths,
                "error": error_msg
            }
            error_count += 1
            
        results.append(result)
        
    # 对每个样本进行评测
    for result in results:
        if "error" not in result:
            for round_result in result["rounds"]:
                try:
                    round_idx = round_result["round"]
                    extracted_pred = round_result["extracted_predicted"]
                    model_answer = extracted_pred if len(extracted_pred) < 10 else round_result["predicted"]
                    correct_answer = round_result["correct"]
                    
                    # 使用evaluation_client而不是api_client进行评测
                    is_correct = evaluate_answer(evaluation_client, model_answer, correct_answer)
                    round_result["is_correct"] = is_correct
                    
                    round_correct_counts[round_idx] = round_correct_counts.get(round_idx, 0) + (1 if is_correct else 0)
                    
                except Exception as e:
                    logger.error(f"评测样本 {result['index']} 第 {round_idx} 轮对话出错: {str(e)}")
                    round_result["is_correct"] = False

    # 新增：重新计算样本正确率指标
    for result in results:
        if "error" not in result:
            # 重新计算正确数量和正确率
            correct_count = sum(1 for r in result["rounds"] if r["is_correct"])
            total_rounds = len(result["rounds"])
            result["sample_accuracy"] = correct_count / total_rounds if total_rounds > 0 else 0
            result["all_correct"] = correct_count == total_rounds
            
            # 记录日志
            logger.info(f"重新计算样本 {result['index']} 正确率: {result['sample_accuracy']:.2%}")
            logger.info(f"样本 {result['index']} 是否全对: {result['all_correct']}")

    # 计算评测指标
    total_samples = len([r for r in results if "error" not in r])
    round_accuracies = {
        round_idx: count / total_samples 
        for round_idx, count in round_correct_counts.items()
    }
    
    # 计算完美对话数（所有轮次都正确的对话）
    perfect_dialogues = sum(
        1 for r in results 
        if "error" not in r and all(round_r["is_correct"] for round_r in r["rounds"])
    )
    
    # 计算所有样本的平均正确率
    valid_samples = [r for r in results if "error" not in r]
    avg_sample_accuracy = sum(s["sample_accuracy"] for s in valid_samples) / len(valid_samples) if valid_samples else 0
    logger.info(f"\n样本平均正确率: {avg_sample_accuracy:.2%}")
    
    # 打印评测统计
    logger.info("\n=== 评测结果汇总 ===")
    logger.info(f"总样本数: {len(results)}")
    logger.info(f"成功处理样本数: {processed_count}")
    logger.info(f"失败样本数: {error_count}")
    logger.info(f"完美对话数: {perfect_dialogues}")
    logger.info(f"完美对话率: {perfect_dialogues / total_samples if total_samples > 0 else 0:.2%}")
    logger.info("\n各轮次正确率：")
    for round_idx, accuracy in sorted(round_accuracies.items()):
        logger.info(f"第 {round_idx} 轮正确率: {accuracy:.2%}")
    
    # 修改summary部分
    output_data = {
        "metadata": metadata,
        "results": results,
        "summary": {
            "total": processed_count + error_count,
            "success": processed_count,
            "error": error_count,
            "evaluation": {
                "total_evaluated": total_samples,
                "perfect_dialogues": perfect_dialogues,
                "perfect_dialogue_rate": perfect_dialogues / total_samples if total_samples > 0 else 0,
                "round_accuracies": round_accuracies,
                "average_sample_accuracy": avg_sample_accuracy,  # 添加平均正确率
                "evaluation_model": "GLM-4"  # 标明评估模型
            }
        }
    }
    
    # 保存结果
    logger.info(f"正在保存结果到文件: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"处理完成。成功: {processed_count}, 失败: {error_count}")
    logger.info(f"结果已保存至: {output_path}")



if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='图片分析任务')
    parser.add_argument('--use_api', action='store_true', help='是否使用API模型')
    parser.add_argument('--api_key', type=str, help='API密钥')
    parser.add_argument('--api_base_url', type=str, help='API基础URL')
    parser.add_argument('--api_model_name', type=str, default='qwen-vl-max-latest', help='API模型名称')
    parser.add_argument('--model_path', type=str, help='模型路径')  # 改为非必需
    parser.add_argument('--tp', type=int, default=2, help='tensor parallel degree')
    parser.add_argument('--top_k', type=int, default=1, help='生成配置参数')
    parser.add_argument('--input_file', type=str, required=True, help='输入文件路径')
    parser.add_argument('--output_dir', type=str, default='outputs', help='输出目录')
    parser.add_argument('--version', type=str, default='v1.1', help='评测版本')
    parser.add_argument('--prompt_template', type=str, 
                       default='作为一位专业的金融分析师，请根据我提供的图片回答以下问题，只需回答选项字母即可\n',
                       help='提示模板')
    parser.add_argument('--log_dir', type=str, default='logs', help='日志目录')
    parser.add_argument('--log_level', type=str, default='INFO', help='日志级别')
    parser.add_argument('--gpu_ids', type=str, default='4,5', help='指定使用的GPU ID，用逗号分隔')
    parser.add_argument('--model_type', type=str, default='qwen', help='模型类型 (qwen/minicpm)')
    
    args = parser.parse_args()
    
    # 设置GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    
    # 构建配置字典
    config = {
        'model': {
            'path': args.model_path,
            'tp': args.tp,
            'top_k': args.top_k,
            'use_api': args.use_api,
            'api_key': args.api_key,
            'base_url': args.api_base_url,
            'api_model_name': args.api_model_name,
            'model_type': args.model_type
        },
        'data': {
            'input_file': args.input_file,
            'output_dir': args.output_dir
        },
        'evaluation': {
            'version': args.version,
            'prompt_template': args.prompt_template
        },
        'logging': {
            'level': args.log_level,
            'log_dir': args.log_dir
        }
    }
    
    # 配置日志记录器
    logger = setup_logging(config)
    logger.info("开始运行图片分析任务")
    
    try:
        # 根据是否使用API决定是否初始化本地模型
        pipeline_instance = None
        if not config['model'].get('use_api', False):
            if not config['model'].get('path'):
                raise ValueError("使用本地模型时必须提供model_path参数")
            logger.info(f"正在初始化本地模型: {config['model']['path']}")
            backend_cfg = TurbomindEngineConfig(tp=config['model']['tp'])
            pipeline_instance = init_model(config['model']['path'], backend_cfg)
        else:
            logger.info("使用API模式，跳过本地模型初始化")
        
        # 生成输出文件名（现在会包含完整路径）
        output_file = get_output_filename(config, config['model'].get('path') or config['model'].get('api_model_name'))
        
        # 确保输出目录存在
        output_file.parent.mkdir(exist_ok=True, parents=True)
        
        process_jsonl(
            config['data']['input_file'],
            str(output_file),  # 转换为字符串
            pipeline_instance,
            config
        )
        logger.info(f"任务完成，结果已保存至 {output_file}")
    except Exception as e:
        logger.critical(f"程序执行过程中发生严重错误: {str(e)}")
        raise