
"""
Copyright (c) 2024 Liu Zhaowei, Shanghai University of Finance and Economics
All rights reserved.

Created on: 2024-02-01
Author: Zhaowei Liu 
Organization: Shanghai University of Finance and Economics
Description: 这是一个用于多模态金融数据分析的评测程序。主要功能包括：
    1. 支持多种视觉语言模型(Qwen-VL, MiniCPM等)的图片分析
    2. 提供本地模型和API调用两种方式
    3. 实现了金融图表的智能分析和判断
    4. 包含完整的日志记录和错误处理机制
    5. 支持批量处理和结果评估

Version: 1.0
"""

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
import csv
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

def analyze_images(pipe, image_paths, question, prompt_template="""你是一位专业熟悉金融图表并拥有广泛金融和财务知识的金融学专家，我将提供给你一道四个选项的单项选择题。请回答此问题。

注意事项：
1. 只需输出正确选项的字母(A/B/C/D)
2. 请基于图表数据进行分析
3. 确保答案的准确性和客观性
4. 题目为单选题，请只输出一个选项。
     
""", top_k=1, model_type="qwen"):
    """分析一张或多张图片并回答特定问题"""
    logger = logging.getLogger(__name__)
    
    # 确保 image_paths 是列表格式
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    
    logger.debug(f"开始分析 {len(image_paths)} 张图片")
    logger.debug(f"问题内容: {question}")
    
    # 根据model_type使用不同的处理逻辑
    if model_type.lower() == "minicpm":
        content = [
            dict(type='text', text=f"{prompt_template}\n问题：{question}")
        ]
        for image_path in image_paths:
            content.append({
                'type': 'image_url',
                'image_url': {
                    'max_slice_nums': 11,
                    'url': image_path
                }
            })
    else:  # 默认使用qwen的处理逻辑
        full_prompt = f"{prompt_template}\n问题：{question}"
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
    
    messages = [{
        'role': 'user',
        'content': content
    }]
    
    try:
        response = pipe(messages, gen_config=GenerationConfig(top_k=top_k))
        logger.info(f"图片分析完成，图片路径: {image_paths}")
        logger.info(f"图片分析问题: {question}")
        logger.info(f"图片分析响应结果: {response.text}")
        return response.text
    except Exception as e:
        logger.error(f"图片分析失败: {str(e)}")
        raise

def get_output_filename(config, model_name):
    """生成输出文件名"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = Path(model_name).name
    version = config['evaluation']['version']
    return f"results_{model_name}_v{version}_{timestamp}.jsonl"

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

def analyze_images_api(client, image_paths, question, model_name, prompt_template="""你是一位专业熟悉金融图表并拥有广泛金融和财务知识的金融学专家，我将提供给你一道四个选项的单项选择题。请回答此问题。

注意事项：
1. 只需输出正确选项的字母(A/B/C/D)
2. 请基于图表数据进行分析
3. 确保答案的准确性和客观性
4. 题目为单选题，请只输出一个选项。
     
""", max_retries=5, retry_delay=1):
    """使用API进行图片分析，包含重试机制"""
    logger = logging.getLogger(__name__)
    
    # 记录处理信息时只打印图片路径，不打印base64数据
    logger.info(f"开始处理图片: {image_paths}")
    logger.info(f"问题: {question}")
    
    user_content = []
    for image_path in image_paths:
        image_format = Path(image_path).suffix.lower().replace('.', '')
        if image_format == 'jpg':
            image_format = 'jpeg'
        
        base64_image = encode_image(image_path)
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{image_format};base64,{base64_image}"
            }
        })
    
    user_content.append({
        "type": "text",
        "text": f"{prompt_template}\n{question}"
    })
    
    messages = [
        {"role": "user", "content": user_content}
    ]
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages
            )
            answer = response.choices[0].message.content
            logger.info(f"API分析完成，响应结果: {answer}")
            return answer
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                logger.warning(f"第 {attempt + 1} 次API调用失败: {str(e)}，{wait_time}秒后重试...")
                time.sleep(wait_time)
                continue
            else:
                logger.error(f"API调用在 {max_retries} 次尝试后仍然失败: {str(e)}")
                raise

def analyze_text_api(client, text_content, question, model_name, prompt_template="""你是一位专业的金融分析师，我将提供给你一个markdown格式的表格数据和一道四个选项的单项选择题。请根据表格数据回答此问题。

注意事项：
1. 只需输出正确选项的字母(A/B/C/D)
2. 请仔细分析markdown中的数据
3. 确保答案的准确性和客观性
4. 题目为单选题，请只输出一个选项。

""", max_retries=5, retry_delay=1):
    """使用API进行文本分析，包含重试机制"""
    logger = logging.getLogger(__name__)
    logger.info(f"开始处理文本分析")
    logger.info(f"文本分析问题: {question}")
    logger.info(f"文本内容: {text_content[:200]}...")  # 只记录前200个字符避免日志过长

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"{prompt_template}\n\n文本内容：{text_content}\n\n{question}"
                }
            ]
        }
    ]

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages
            )
            answer = response.choices[0].message.content
            logger.info(f"文本分析完成")
            logger.info(f"文本分析响应结果: {answer}")
            return answer
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                logger.warning(f"第 {attempt + 1} 次API调用失败: {str(e)}，{wait_time}秒后重试...")
                time.sleep(wait_time)
                continue
            else:
                logger.error(f"API调用在 {max_retries} 次尝试后仍然失败: {str(e)}")
                raise

def process_jsonl(input_path, output_path, model_pipe, config):
    """处理TSV文件并保存结果，包含图片和markdown两种分析方式"""
    logger = logging.getLogger(__name__)
    logger.info(f"开始处理tsv文件: {input_path}")
    
    # 初始化API客户端（如果使用API模式）
    api_client = None
    if config['model'].get('use_api', False):
        api_client = init_api_client(
            api_key=config['model']['api_key'],
            base_url=config['model'].get('base_url')
        )
    
    # 设置固定的随机种子
    random.seed(42)
    
    # 加载数据
    data = load_data(input_path)
    total_lines = len(data)
    logger.info(f"输入文件总条数: {total_lines}")
    
    results = []
    processed_count = 0
    error_count = 0
    
    # 添加评测元信息
    metadata = {
        "evaluation_version": config['evaluation']['version'],
        "model_name": Path(config['model']['path']).name if not config['model'].get('use_api', False) else config['model']['api_model_name'],
        "timestamp": datetime.now().isoformat(),
        "config": config
    }
    
    for line_num, entry in enumerate(tqdm(data, total=total_lines, desc="处理进度"), 1):
        logging.info(f"正在处理第 {line_num} 条数据")
        
        try:
            # 拼接完整问题
            full_question = f"""
问题：{entry['question']}

选项：
A. {entry['A']}
B. {entry['B']}
C. {entry['C']}
D. {entry['D']}"""
            
            # 分别进行图片和文本分析
            image_answer = None
            text_answer = None
            
            # 图片分析
            if entry['image']:
                if config['model'].get('use_api', False):
                    image_answer = analyze_images_api(
                        api_client,
                        [entry['image']],
                        full_question,
                        config['model']['api_model_name'],
                        prompt_template=config['evaluation']['image_prompt_template']  # 使用图片专用prompt
                    )
                else:
                    image_answer = analyze_images(
                        model_pipe,
                        [entry['image']],
                        full_question,
                        prompt_template=config['evaluation']['image_prompt_template'],  # 使用图片专用prompt
                        top_k=config['model']['top_k'],
                        model_type=config['model'].get('model_type', 'qwen')
                    )
            
            # 文本分析
            if entry['md_content']:
                if config['model'].get('use_api', False):
                    text_answer = analyze_text_api(
                        api_client,
                        entry['md_content'],
                        full_question,
                        config['model']['api_model_name'],
                        prompt_template=config['evaluation']['markdown_prompt_template']  # 使用markdown专用prompt
                    )
                else:
                    text_answer = analyze_text(
                        model_pipe,
                        entry['md_content'],
                        full_question,
                        prompt_template=config['evaluation']['markdown_prompt_template'],  # 使用markdown专用prompt
                        top_k=config['model']['top_k']
                    )
            
            results.append({
                "image_path": entry['image'],
                "md_content": bool(entry['md_content']),  # 只记录是否有markdown内容
                "question": entry['question'],
                "type": entry['type'],
                "index": entry['index'],
                "full_prompt": full_question,
                "image_answer": image_answer,
                "text_answer": text_answer,
                "correct_answer": entry['answer']
            })
            
            processed_count += 1
            
        except Exception as e:
            error_msg = f"处理第 {line_num} 条数据时出错: {str(e)}"
            logger.error(error_msg)
            results.append({
                "index": entry.get('index'),
                "error": str(e)
            })
            error_count += 1

    # 加入chatglmapi进行模型评估 根据回答和正确选项比对，输出结果
    def evaluate_answer(client, model_answer, correct_answer, question_with_options, max_retries=3, retry_delay=1):
        logger = logging.getLogger(__name__)
        prompt = f"""你作为一名专业的金融答案检察员，请比对以下答案是否一致，请仔细检测模型输出答案，不要把选项当作答案，只有<正确答案：>之类的字样才是正确答案：

判断正确样例1：
模型输出答案: A. 1993
正确答案: A
评测结果: True

判断正确样例2：
模型输出答案: A；D
正确答案: A D
评测结果: True

评测错误样例1：
模型输出答案: A, C, D
正确答案: B;C;D
评测结果: False

评测错误样例2：
模型输出答案: B, C
正确答案: B;C
评测结果: False

问题和选项：{question_with_options}
模型输出答案：{model_answer}
正确答案：{correct_answer}

请仔细分析模型输出的内容，如果模型输出包含了正确选项对应的内容（即使没有直接输出选项字母），也应该判定为正确。
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

    # 评估结果
    image_correct = 0
    text_correct = 0
    total_image = 0
    total_text = 0
    
    for result in results:
        if "error" not in result:
            # 添加评估结果到每个结果项
            if result["image_answer"]:
                result["image_evaluation"] = evaluate_answer(client, result["image_answer"], result["correct_answer"], result["full_prompt"])
            
            if result["text_answer"]:
                result["text_evaluation"] = evaluate_answer(client, result["text_answer"], result["correct_answer"], result["full_prompt"])
            
            # 计算准确率部分保持不变
            if result["image_answer"]:
                total_image += 1
                if result.get("image_evaluation", False):  # 使用新添加的评估结果
                    image_correct += 1
            
            if result["text_answer"]:
                total_text += 1
                if result.get("text_evaluation", False):  # 使用新添加的评估结果
                    text_correct += 1
    
    # 计算准确率
    image_accuracy = image_correct / total_image if total_image > 0 else 0
    text_accuracy = text_correct / total_text if total_text > 0 else 0
    
    # 更新输出数据
    output_data = {
        "metadata": metadata,
        "results": results,
        "summary": {
            "total": processed_count + error_count,
            "success": processed_count,
            "error": error_count,
            "evaluation": {
                "image_analysis": {
                    "total": total_image,
                    "correct": image_correct,
                    "accuracy": image_accuracy
                },
                "text_analysis": {
                    "total": total_text,
                    "correct": text_correct,
                    "accuracy": text_accuracy
                }
            }
        }
    }

    # 确保输出目录存在
    output_dir = Path(config['data']['output_dir'])
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / output_path
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    logger.info(f"结果已保存至: {output_path}")

    # 在处理完所有数据后打印统计信息
    logger.info(f"处理完成！总条数: {total_lines}, 成功处理: {processed_count}, 错误数: {error_count}")

def load_data(input_path):
    """加载TSV文件数据，并读取对应的markdown文件"""
    logger = logging.getLogger(__name__)
    data = []
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            
            logger.info(f"文件中的列名: {reader.fieldnames}")
            
            for row in reader:
                # 读取markdown文件内容
                md_content = ""
                if row.get('md_path'):
                    try:
                        with open(row['md_path'], 'r', encoding='utf-8') as md_file:
                            md_content = md_file.read()
                    except Exception as e:
                        logger.warning(f"无法读取markdown文件 {row['md_path']}: {str(e)}")
                
                data_entry = {
                    'type': row.get('type', ''),
                    'index': row.get('index', ''),
                    'image': row.get('image_path', ''),  # 注意这里改成了image_path
                    'md_content': md_content,
                    'question': row.get('question', ''),
                    'A': row.get('A', ''),
                    'B': row.get('B', ''),
                    'C': row.get('C', ''),
                    'D': row.get('D', ''),
                    'answer': row.get('answer', '')
                }
                data.append(data_entry)
        
        logger.info(f"成功加载 {len(data)} 条数据")
        return data
        
    except Exception as e:
        logger.error(f"加载数据文件时出错: {str(e)}")
        raise

def analyze_text(pipe, text_content, question, prompt_template="""你是一位专业的金融分析师，我将提供给你一个markdown格式的表格数据和一道四个选项的单项选择题。请根据表格数据回答此问题。

注意事项：
1. 只需输出正确选项的字母(A/B/C/D)
2. 请仔细分析markdown中的数据
3. 确保答案的准确性和客观性
4. 题目为单选题，请只输出一个选项。

""", top_k=1):
    """分析文本内容并回答特定问题"""
    logger = logging.getLogger(__name__)
    logger.info("开始文本分析")
    logger.debug(f"问题内容: {question}")
    
    full_prompt = f"{prompt_template}\n\n文本内容：{text_content}\n\n{question}"
    messages = [{
        'role': 'user',
        'content': [{'type': 'text', 'text': full_prompt}]
    }]
    
    try:
        response = pipe(messages, gen_config=GenerationConfig(top_k=top_k))
        logger.info(f"文本分析完成")
        logger.info(f"文本分析响应结果: {response.text}")
        return response.text
    except Exception as e:
        logger.error(f"文本分析失败: {str(e)}")
        raise

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
    parser.add_argument('--output_dir', type=str, default='outputs_L1_Q2', help='输出目录')
    parser.add_argument('--version', type=str, default='v1.1', help='评测版本')
    parser.add_argument('--image_prompt_template', type=str, 
                       default='''你是一位专业熟悉金融图表并拥有广泛金融和财务知识的金融学专家，我将提供给你一道四个选项的单项选择题。请回答此问题。

注意事项：
1. 只需输出正确选项的字母(A/B/C/D)
2. 请基于图表数据进行分析
3. 确保答案的准确性和客观性
4. 题目为单选题，请只输出一个选项。 

     ''',
                       help='图片分析提示模板')
    parser.add_argument('--markdown_prompt_template', type=str, 
                       default='''你是一位专业的金融分析师，我将提供给你一个markdown格式的表格数据和一道四个选项的单项选择题。请根据表格数据回答此问题。

注意事项：
1. 只需输出正确选项的字母(A/B/C/D)
2. 请仔细分析markdown中的数据
3. 确保答案的准确性和客观性
4. 题目为单选题，请只输出一个选项。

''',
                       help='Markdown分析提示模板')
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
            'image_prompt_template': args.image_prompt_template,     # 添加图片prompt
            'markdown_prompt_template': args.markdown_prompt_template  # 添加markdown prompt
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
            if "llava" in config['model']['path']:
                backend_cfg = TurbomindEngineConfig(tp=config['model']['tp'],session_len=8000)
            else:
                backend_cfg = TurbomindEngineConfig(tp=config['model']['tp'])
            pipeline_instance = init_model(config['model']['path'], backend_cfg)
        
        # 生成输出文件名
        output_file = get_output_filename(config, config['model'].get('path') or config['model'].get('api_model_name'))
        
        # 处理整个JSONL文件
        process_jsonl(
            config['data']['input_file'],
            output_file,
            pipeline_instance,
            config
        )
        logger.info(f"任务完成，结果已保存至 {output_file}")
    except Exception as e:
        logger.critical(f"程序执行过程中发生严重错误: {str(e)}")
        raise