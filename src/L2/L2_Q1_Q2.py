
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

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
# 初始化chatglmapi
client = ZhipuAI(api_key="your_key")
logging.info("chatglmapi初始化成功")

def setup_logging(config):
    """配置日志记录器"""
    log_dir = Path(config['logging']['log_dir'])
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"image_analysis_{timestamp}.log"


    
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

def analyze_images(pipe, image_paths, questions, prompt_template="作为一位专业的金融分析师，请根据图片回答以下问题，只需回答选项字母即可：", top_k=1, model_type="qwen"):
    """分析图片并进行多轮对话"""
    logger = logging.getLogger(__name__)
    
    # 确保 image_paths 是列表格式
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    
    logger.debug(f"开始分析 {len(image_paths)} 张图片")
    
    # 初始化对话历史
    messages = []
    answers = []
    
    # 第一轮对话
    if model_type.lower() == "minicpm":
        content = [dict(type='text', text=f"{prompt_template}\n\n问题：{questions[0]}")]
        for image_path in image_paths:
            content.append({
                'type': 'image_url',
                'image_url': {
                    'max_slice_nums': 11,
                    'url': image_path
                }
            })
    else:
        full_prompt = f"{prompt_template}\n\n问题：{questions[0]}"
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
        # 进行第一轮对话
        response = pipe(messages, gen_config=GenerationConfig(top_k=top_k))
        first_answer = response.text
        answers.append(first_answer)
        messages.append({'role': 'assistant', 'content': first_answer})
        
        # 后续轮次的对话
        for question in questions[1:]:
            # 添加新的问题
            messages.append({
                'role': 'user',
                'content': f"{prompt_template}\n{question}"
            })
            
            # 获取回答
            response = pipe(messages, gen_config=GenerationConfig(top_k=top_k))
            answer = response.text
            answers.append(answer)
            messages.append({'role': 'assistant', 'content': answer})
            
        logger.info(f"多轮对话完成，共 {len(answers)} 轮")
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

def analyze_images_api(client, image_paths, questions, model_name, prompt_template="作为一位专业的金融分析师，请根据图片回答以下问题，只需回答选项字母即可：", max_retries=5, retry_delay=1):
    """使用API进行图片分析的多轮对话"""
    logger = logging.getLogger(__name__)
    
    # 确保 image_paths 是列表格式
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    
    # 验证图片路径是否存在且是文件
    for path in image_paths:
        if not os.path.isfile(path):
            raise ValueError(f"无效的图片路径: {path}")
            
    logger.info(f"开始处理图片: {image_paths}")
    
    # 初始化对话历史和答案列表
    messages = []
    answers = []
    
    # 准备第一轮对话的图片内容
    user_content = []
    for image_path in image_paths:
        try:
            image_format = Path(image_path).suffix.lower().replace('.', '')
            if image_format == 'jpg':
                image_format = 'jpeg'
            
            if not os.path.exists(image_path):
                logger.error(f"图片文件不存在: {image_path}")
                raise FileNotFoundError(f"图片文件不存在: {image_path}")
                
            base64_image = encode_image(image_path)
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{image_format};base64,{base64_image}"
                }
            })
        except Exception as e:
            logger.error(f"处理图片 {image_path} 时出错: {str(e)}")
            raise
    
    # 添加第一个问题
    user_content.append({
        "type": "text",
        "text": f"{prompt_template}\n\n问题：{questions[0]}"
    })
    
    # 第一轮对话
    messages.append({"role": "user", "content": user_content})
    
    logger.info("\n=================== 第1轮对话 ===================")
    logger.info(f"提示词：{prompt_template}\n\n问题：{questions[0]}")
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages
            )
            first_answer = response.choices[0].message.content
            logger.info(f"模型回答：{first_answer}")
            logger.info("===============================================")
            
            answers.append(first_answer)
            messages.append({"role": "assistant", "content": first_answer})
            break
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                logger.warning(f"第一轮对话第 {attempt + 1} 次API调用失败: {str(e)}，{wait_time}秒后重试...")
                time.sleep(wait_time)
                continue
            else:
                logger.error(f"第一轮对话在 {max_retries} 次尝试后仍然失败: {str(e)}")
                raise
    
    # 后续轮次的对话
    for idx, question in enumerate(questions[1:], 2):
        messages.append({
            "role": "user",
            "content": f"{prompt_template}\n\n问题：{question}"
        })
        
        logger.info(f"\n=================== 第{idx}轮对话 ===================")
        logger.info(f"提示词：{prompt_template}\n\n问题：{question}")
        
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages
                )
                answer = response.choices[0].message.content
                logger.info(f"模型回答：{answer}")
                logger.info("===============================================")
                
                answers.append(answer)
                messages.append({"role": "assistant", "content": answer})
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    logger.warning(f"后续轮次对话第 {attempt + 1} 次API调用失败: {str(e)}，{wait_time}秒后重试...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"后续轮次对话在 {max_retries} 次尝试后仍然失败: {str(e)}")
                    raise
    
    logger.info(f"多轮对话完成，共 {len(answers)} 轮")
    return answers

def evaluate_answer(client, model_answer, correct_answer, question_with_options, max_retries=3, retry_delay=1):
    """使用zhipuai评估答案正确性"""
    prompt = f"""你作为一名专业的金融答案检察员，请比对以下答案是否一致，请仔细检测模型输出答案，不要把选项当作答案，只有<正确答案：>之类的字样才是正确答案：

答案检察有效样例1：
模型输出答案: A. 1993
正确答案: A
评测结果: True

答案检察有效样例2：
模型输出答案: A
正确答案: A 
评测结果: True

答案检察无效样例1：
模型输出答案: A
正确答案: B
评测结果: False

答案检察无效样例2：
模型输出答案: C
正确答案: B
评测结果: False


问题和选项：{question_with_options}
模型输出答案：{model_answer}
正确答案：{correct_answer}

请仔细分析模型输出的内容，如果模型输出包含了正确选项对应的内容（即使没有直接输出选项字母），也应该判定为正确。
如果模型输出的内容为空，则直接判定为False。
请仅输出True或False，True表示答案实质相同，False表示答案不同。"""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="glm-4-flash",
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个严格的答案评判专家，只输出True或False。"
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
    
    # 检查model_pipe是否正确初始化
    if model_pipe is None and not config['model'].get('use_api', False):
        raise ValueError("模型管道未正确初始化")
        
    # 如果使用API，初始化API客户端
    api_client = None
    if config['model'].get('use_api', False):
        api_client = init_api_client(
            config['model']['api_key'],
            config['model'].get('base_url')
        )

    # 读取TSV文件并处理图片路径列
    df = pd.read_csv(input_path, sep='\t')
    # 将image列中的逗号分隔字符串转换为列表
    df['image'] = df['image'].apply(lambda x: [path.strip() for path in x.split(',')])
    
    # 按index分组，获取每组的所有轮次问题
    grouped_data = df.groupby('index')
    
    results = []
    processed_count = 0
    error_count = 0
    
    metadata = {
        "model": config['model'].get('path') or config['model'].get('api_model_name'),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": config
    }
    
    for index, group in tqdm(grouped_data):
        # 查看是否可以通过sort_values
        group_sorted = group.sort_values('round')
        questions = group_sorted['question'].tolist()
        correct_answers = group_sorted['answer'].tolist()
       
        if 'D' in group_sorted.columns and group_sorted['D'].notna().any():
            questions_with_options = group_sorted.apply(
                lambda row: f"{row['question']}\nA. {row.get('A', '')}\nB. {row.get('B', '')}\nC. {row.get('C', '')}\nD. {row.get('D', '')}",
                axis=1
            ).tolist()
        else:
            questions_with_options = group_sorted.apply(
                lambda row: f"{row['question']}\nA. {row.get('A', '')}\nB. {row.get('B', '')}\nC. {row.get('C', '')}",
                axis=1
            ).tolist()
        # 获取第一行的图片路径列表
        image_paths = group_sorted['image'].iloc[0]
        
        try:
            # 根据是否使用API选择不同的处理函数
            if config['model'].get('use_api', False):
                predicted_answers = analyze_images_api(
                    api_client,
                    image_paths,  # 现在直接传入图片路径列表
                    questions_with_options,
                    config['model']['api_model_name'],
                    prompt_template=config['evaluation']['prompt_template']
                )
            else:
                predicted_answers = analyze_images(
                    model_pipe,
                    image_paths,  # 现在直接传入图片路径列表
                    questions_with_options,
                    prompt_template=config['evaluation']['prompt_template'],
                    top_k=config['model'].get('top_k', 1),
                    model_type=config['model'].get('model_type', 'qwen')
                )
            
            # 打印每个样本的多轮对话结果
            logger.info(f"\n样本 {index} 的多轮对话结果：")
            for round_idx, (pred, true, q_with_opts) in enumerate(zip(predicted_answers, correct_answers, questions_with_options), 1):
                logger.info(f"第 {round_idx} 轮：")
                logger.info(f"模型预测：{pred.strip()}")
                logger.info(f"正确答案：{true.strip()}")
                logger.info(f"问题和选项：{q_with_opts}")
                logger.info("-----------------------------")
            logger.info("开始使用zhipuai评估每轮答案")
            # 使用zhipuai评估每轮答案
            round_results = []
            for round_idx, (pred, true, q_with_opts) in enumerate(zip(predicted_answers, correct_answers, questions_with_options), 1):
                try:
                    is_correct = evaluate_answer(client, pred.strip(), true.strip(), q_with_opts)
                    round_results.append({
                        'round': round_idx,
                        'predicted': pred.strip(),
                        'correct': true.strip(),
                        'question_with_options': q_with_opts,
                        'is_correct': is_correct
                    })
                    logger.info(f"评估样本 {index} 第 {round_idx} 轮答案完成，结果: {is_correct}")
                except Exception as e:
                    logger.error(f"评估样本 {index} 第 {round_idx} 轮答案时出错: {str(e)}")
                    round_results.append({
                        'round': round_idx,
                        'predicted': pred.strip(),
                        'correct': true.strip(),
                        'question_with_options': q_with_opts,
                        'is_correct': False,
                        'evaluation_error': str(e)
                    })
            
            # 计算该样本的正确率
            correct_count = sum(1 for r in round_results if r['is_correct'])
            sample_accuracy = correct_count / len(round_results) if round_results else 0
            
            result = {
                "index": int(index),
                "image_path": image_paths,
                "rounds": round_results,
                "all_correct": all(r['is_correct'] for r in round_results),
                "sample_accuracy": sample_accuracy  # 添加样本正确率
            }
            
            processed_count += 1
            
        except Exception as e:
            error_msg = f"处理样本 {index} 时发生错误: {str(e)}"
            logger.error(error_msg)
            result = {
                "index": int(index),
                "image_path": image_paths,
                "error": error_msg
            }
            error_count += 1
            
        results.append(result)
        
    # 评测统计
    total_evaluated = 0
    perfect_dialogues = 0
    round_correct_counts = {}
    
    for result in results:
        if "error" not in result:
            total_evaluated += 1
            all_rounds_correct = True
            
            for round_result in result["rounds"]:
                try:
                    round_idx = round_result["round"]
                    is_correct = round_result["is_correct"]
                    round_correct_counts[round_idx] = round_correct_counts.get(round_idx, 0) + (1 if is_correct else 0)
                    if not is_correct:
                        all_rounds_correct = False
                except Exception as e:
                    logger.error(f"评测样本 {result['index']} 第 {round_idx} 轮对话出错: {str(e)}")
                    all_rounds_correct = False
            
            if all_rounds_correct:
                perfect_dialogues += 1
    
    # 计算评测指标
    perfect_dialogue_rate = perfect_dialogues / total_evaluated if total_evaluated > 0 else 0
    round_accuracies = {
        round_idx: count / total_evaluated 
        for round_idx, count in round_correct_counts.items()
    }
    
    # 修改评测统计的打印
    logger.info("\n=== 评测结果汇总 ===")
    logger.info(f"完美对话率: {perfect_dialogue_rate:.2%}")
    logger.info("\n各轮次正确率：")
    for round_idx, accuracy in sorted(round_accuracies.items()):
        logger.info(f"第 {round_idx} 轮正确率: {accuracy:.2%}")
    
    # 计算所有样本的平均正确率
    valid_samples = [r for r in results if "error" not in r]
    avg_sample_accuracy = sum(s["sample_accuracy"] for s in valid_samples) / len(valid_samples) if valid_samples else 0
    logger.info(f"\n样本平均正确率: {avg_sample_accuracy:.2%}")
    logger.info("=============================")
    
    # 准备输出数据
    output_data = {
        "metadata": metadata,
        "results": results,
        "summary": {
            "total": processed_count + error_count,
            "success": processed_count,
            "error": error_count,
            "evaluation": {
                "total_evaluated": total_evaluated,
                "perfect_dialogues": perfect_dialogues,
                "perfect_dialogue_rate": perfect_dialogue_rate,
                "round_accuracies": round_accuracies,
                "average_sample_accuracy": avg_sample_accuracy  # 添加样本平均正确率
            }
        }
    }
    
    # 保存结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"处理完成。成功: {processed_count}, 失败: {error_count}")



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
            if "llava" in config['model']['path']:
                backend_cfg = TurbomindEngineConfig(tp=config['model']['tp'],session_len=8096)
            else:
                backend_cfg = TurbomindEngineConfig(tp=config['model']['tp'],session_len=4096)
            
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