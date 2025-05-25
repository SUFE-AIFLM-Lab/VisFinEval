# -*- coding: utf-8 -*-
"""
Main script for multi-round evaluation of all question types, automatically adapting to unified TSV format and multi-turn dialogues.
Supports both API and local model inference, with automatic prompt selection and evaluation rules.
Detailed comments for future extension.
"""
import logging
import os
import json
import time
import base64
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
from tqdm import tqdm
#from zhipuai import ZhipuAI
from openai import OpenAI
from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig, PytorchEngineConfig
from lmdeploy.vl.constants import IMAGE_TOKEN
import sys
from typing import List, Dict, Any, Tuple
from PIL import Image

# ===================== PROMPT AND EVALUATION RULES DICTIONARY =====================
# Automatically extracted scripts, can be extended for new question types
PROMPT_RULES = {
    'L1_Q1': {
        'prompt': """你是一位专业熟悉金融图表并拥有广泛金融和财务知识的金融学专家，我将提供给你一道三选的单项选择题。请回答此问题。\n\n注意事项：\n1. 只需输出正确选项的字母(A/B/C)\n2. 请基于图表数据进行分析\n3. 确保答案的准确性和客观性\n4. 题目为单选题，请只输出一个选项。""",
        'eval_type': 'single_choice_abc',
    },
    'L1_Q2': {
        'prompt': """你是一位专业熟悉金融图表并拥有广泛金融和财务知识的金融学专家，我将提供给你一道四个选项的单项选择题。请回答此问题。\n\n注意事项：\n1. 只需输出正确选项的字母(A/B/C/D)\n2. 请基于图表数据进行分析\n3. 确保答案的准确性和客观性\n4. 题目为单选题，请只输出一个选项。""",
        'eval_type': 'single_choice_abcd',
    },
    'L1_Q3': {
        'prompt': """你是一位专业熟悉金融图表并拥有广泛金融和财务知识的金融学专家，我将提供给你一道四个选项的多选题。请回答此问题。\n\n注意事项：\n1. 只需输出正确选项的字母(A/B/C/D)\n2. 请基于图表数据进行分析\n3. 确保答案的准确性和客观性\n4. 题目为多选题，多个选项使用,进行分隔。""",
        'eval_type': 'multi_choice',
    },
    'L1_Q4': {
        'prompt': """你是一位专业熟悉金融图表并拥有广泛金融和财务知识的金融学专家，请基于图表严谨地分析以下判断题。\n\n请仔细观察图表中的数据趋势、数值变化和关键特征,运用专业的数据分析能力,对判断内容的正确性做出准确评估。\n\n注意事项:\n1. 需要基于客观数据进行判断,避免主观臆测\n2. 关注判断中涉及的具体时间点、数值区间或趋势变化\n3. 确保判断的严谨性和准确性\n4. 请只回答\"是\"或\"否\"。""",
        'eval_type': 'judgement',
    },
    'L1_Q5': {
        'prompt': """你是一位专业熟悉金融图表并拥有广泛金融和财务知识的金融学专家，我将提供给你一道四个选项的单项选择题。请回答此问题。\n\n注意事项：\n1. 只需输出正确选项的字母(A/B/C/D)\n2. 请基于图表数据进行分析\n3. 确保答案的准确性和客观性\n4. 题目为单选题，请只输出一个选项。""",
        'eval_type': 'multi_choice',
    },
    'L1_Q6': {
        'prompt': """你是一位专业的金融分析师，擅长图表趋势分析和数据解读。我将提供一张金融图表和相关的单选题(A/B/C/D四个选项)。请你:\n\n1. 仔细分析图表中的趋势变化、波动特征和关键转折点\n2. 结合专业知识对数据走势进行判断\n3. 只需输出一个正确选项的字母(A/B/C/D)\n4. 确保基于客观数据得出结论,避免主观臆测\n\n注:这是趋势分析的单选题,请只选择一个最准确的答案。""",
        'eval_type': 'trend_single_choice',
    },
    'L2_Q1': {
        'prompt': """你是一位专业熟悉金融图表并拥有广泛金融和财务知识的金融学专家，我将提供给你一道三个选项的单项选择题。请回答此问题.\n\n注意事项：\n1. 只需输出正确选项的字母(A/B/C)\n2. 请基于图表数据进行分析\n3. 确保答案的准确性和客观性\n4. 题目为单选题，请只输出一个选项。""",
        'eval_type': 'multi_round_single_choice',
    },
    'L2_Q2': {
        'prompt': """你是一位专业熟悉金融图表并拥有广泛金融和财务知识的金融学专家，我将提供给你一道四个选项的单项选择题。请回答此问题.\n\n注意事项：\n1. 只需输出正确选项的字母(A/B/C/D)\n2. 请基于图表数据进行分析\n3. 确保答案的准确性和客观性\n4. 题目为单选题，请只输出一个选项.""",
        'eval_type': 'multi_round_single_choice',
    },
    'L2_Q3': {
        'prompt': """你是一位专业熟悉金融图表并拥有广泛金融和财务知识的金融学专家，请仔细阅读背景故事和图表信息，并回答以下单项选择题.\n\n请注意：\n1. 只需输出正确选项的字母(A/B/C/D)\n2. 请基于图表数据和背景信息进行分析\n3. 确保答案的准确性和客观性\n4. 题目为单选题，请只输出一个选项.""",
        'eval_type': 'multi_round_single_choice_with_bg',
    },
    'L3_Q1': {
        'prompt': """你是一位专业熟悉金融图表并拥有广泛金融和财务知识的金融学专家，我将提供给你一道四个选项的单项选择题。请回答此问题。\n\n注意事项：\n1. 只需输出正确选项的字母(A/B/C/D)\n2. 请基于图表数据进行分析\n3. 确保答案的准确性和客观性\n4. 题目为单选题，请只输出一个选项。\n""",
        'eval_type': 'single_choice_abcd',
    },
    'L3_Q2': {
        'prompt': """你是一位专业熟悉金融图表并拥有广泛金融和财务知识的金融学专家，请仔细阅读背景故事和图表信息，我将提供给你一道四个选项的单项选择题。请回答此问题。\n\n注意事项：\n1. 只需输出正确选项的字母(A/B/C/D)\n2. 请基于图表数据进行分析\n3. 确保答案的准确性和客观性\n4. 题目为单选题，请只输出一个选项。\n""",
        'eval_type': 'single_choice_abcd',
    },
    'L3_Q3': {
        'prompt': """你是一位专业熟悉金融图表并拥有广泛金融和财务知识的金融学专家，请仔细阅读背景故事和图表信息，并回答以下选择题.\n\n请注意：\n1. 只需输出正确选项的字母(A/B/C/D)\n2. 请基于图表数据和背景信息进行分析\n3. 确保答案的准确性和客观性\n4. 题目为单选题，请只输出一个选项.""",
        'eval_type': 'single_choice_abcd_with_bg',
    },
    'L3_Q4': {
        'prompt': "你是一位专业熟悉金融图表并拥有广泛金融和财务知识的金融学专家，我将提供给你一道金融问题。请基于图表数据进行详细分析，确保答案的准确性和客观性，并归纳出最终结论。请你一步步详细推理，并给出最终结论。请务必将你的最终答案将数字和单位一起使用 \\boxed{{}} 包裹返回，且答案中不包含其他多余内容。",
        'eval_type': 'multi_round_qa',
    },
}


# ===================== JUDGE MODEL (Qwen2.5-72B) =====================
# Configure judge model API Key and Base URL here
JUDGE_CLIENT = OpenAI(
    api_key="your_key",  # Directly pass API Key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
JUDGE_MODEL = "qwen-max-latest"

# ===================== UTILITY FUNCTIONS =====================
# ========== Question Type Number Prefix Mapping ==========
INDEX_PREFIX_MAP = {
    '11': ('L1_Q1', '一级题型1'),
    '12': ('L1_Q2', '一级题型2'),
    '13': ('L1_Q3', '一级题型3'),
    '14': ('L1_Q4', '一级题型4'),
    '15': ('L1_Q5', '一级题型5'),
    '16': ('L1_Q6', '一级题型6'),
    '21': ('L2_Q1', '二级题型1'),
    '22': ('L2_Q2', '二级题型2'),
    '23': ('L2_Q3', '二级题型3'),
    '31': ('L3_Q1', '三级题型1(双模态)'),
    '32': ('L3_Q2', '三级题型2'),
    '33': ('L3_Q3', '三级题型3'),
    '34': ('L3_Q4', '三级题型4(多轮)')
}

def get_prompt_and_eval_type(index):
    """Get prompt and evaluation type based on question index"""
    prefix = str(index)[:2]
    key_name = INDEX_PREFIX_MAP.get(prefix)
    if key_name:
        key, typename = key_name
        if key in PROMPT_RULES:
            return PROMPT_RULES[key]['prompt'], PROMPT_RULES[key]['eval_type'], key, typename
    # fallback
    return PROMPT_RULES['L2_Q1']['prompt'], PROMPT_RULES['L2_Q1']['eval_type'], 'L2_Q1', 'Default Level 2 Type 1'

def setup_logging(log_dir, log_level="INFO"):
    """Configure logger with both file and console output"""
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'L1L2eval_{timestamp}.log'
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(str(log_file), encoding='utf-8')
        ],
        force=True
    )
    logging.info("Logger initialized successfully")

def load_and_group_tsv(tsv_path):
    """
    Read TSV, group by index, sort by round, assemble multi-round questions.
    Returns: {index: [dict(each round data)]}
    """
    df = pd.read_csv(tsv_path, sep='\t', dtype=str).fillna('')
    # Handle comma-separated multiple images
    df['image'] = df['image'].apply(lambda x: [os.path.join('/root/VisFinEval', p.strip()) for p in x.split(',')] if x else [])
    # Convert round to int for sorting
    df['round'] = df['round'].apply(lambda x: int(x) if str(x).isdigit() else 0)
    grouped = {}
    for idx, group in df.groupby('index'):
        group_sorted = group.sort_values('round')
        grouped[idx] = group_sorted.to_dict(orient='records')
    return grouped

def analyze_multi_round(pipe, api_client, image_paths, questions, prompt, use_api, model_name, top_k=1, model_type='qwen'):
    """Multi-round dialogue inference, automatically routes local/remote"""
    answers = []
    messages = []
    # Record complete dialogue for each round
    dialogue_history = []
    
    # First round with images
    if use_api:
        # API mode: first round with images, subsequent rounds text only
        user_content = []
        for image_path in image_paths:
            image_format = Path(image_path).suffix.lower().replace('.', '')
            if image_format == 'jpg':
                image_format = 'jpeg'
            with open(image_path, "rb") as f:
                base64_img = base64.b64encode(f.read()).decode("utf-8")
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/{image_format};base64,{base64_img}"}
            })
        user_content.append({"type": "text", "text": f"{prompt}\n\nQuestion: {questions[0]}"})
        messages.append({"role": "user", "content": user_content})
        for i, q in enumerate(questions):
            if i > 0:
                messages.append({"role": "user", "content": f"{prompt}\n\nQuestion: {q}"})
            response = api_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.1
            )
            answer = response.choices[0].message.content
            answers.append(answer)
            messages.append({"role": "assistant", "content": answer})
            # Record this round's dialogue
            dialogue_history.append({
                "user_prompt": f"{prompt}\n\nQuestion: {q}",
                "model_response": answer
            })
    else:
        # Local mode: first round with images, subsequent rounds text only
        if model_type.lower() == "minicpm":
            content = [dict(type='text', text=f"{prompt}\n\nQuestion: {questions[0]}")]
            for image_path in image_paths:
                content.append({'type': 'image_url', 'image_url': {'max_slice_nums': 11, 'url': image_path}})
        else:
            full_prompt = f"{prompt}\n\nQuestion: {questions[0]}"
            tokens_and_prompt = f"{IMAGE_TOKEN * len(image_paths)}\n{full_prompt}"
            content = [{'type': 'text', 'text': tokens_and_prompt}]
            for image_path in image_paths:
                content.append({'type': 'image_url', 'image_url': {'max_dynamic_patch': 12, 'url': image_path}})
        messages.append({'role': 'user', 'content': content})
        for i, q in enumerate(questions):
            if i > 0:
                messages.append({'role': 'user', 'content': f"{prompt}\n\nQuestion: {q}"})
            response = pipe(messages, gen_config=GenerationConfig(top_k=top_k, temperature=0.1))
            answer = response.text
            answers.append(answer)
            messages.append({'role': 'assistant', 'content': answer})
            # Record this round's dialogue
            dialogue_history.append({
                "user_prompt": f"{prompt}\n\nQuestion: {q}",
                "model_response": answer
            })
    return answers, dialogue_history

def evaluate_answer_judge(client, model_answer, correct_answer, question_with_options, eval_type, max_retries=3, retry_delay=1):
    """
    用Qwen2.5-72B评测答案，自动适配不同题型的prompt
    """
    # 可根据eval_type定制不同的prompt
    if eval_type.startswith('single_choice'):
        prompt = f"""你是金融考试的自动判分专家。请判断模型输出的答案与标准答案是否实质一致。\n\n【题目与选项】\n{question_with_options}\n\n【模型输出】\n{model_answer}\n\n【标准答案】\n{correct_answer}\n\n判分规则：\n- 只要模型输出明确表达了标准答案的内容（无论是选项字母还是内容本身），即判为True。\n- 如果模型输出与标准答案内容不一致，或表达模糊、无法判断，则判为False。\n- 只输出True或False，不要输出其他内容。"""
    elif eval_type == 'multi_choice':
        prompt = f"""你是金融考试的自动判分专家。请判断模型输出的多选答案与标准答案是否实质一致。\n\n【题目与选项】\n{question_with_options}\n\n【模型输出】\n{model_answer}\n\n【标准答案】\n{correct_answer}\n\n判分规则：\n- 只要模型输出包含所有标准答案内容（无论是选项字母还是内容本身），即判为True。\n- 如果模型输出缺少标准答案中的任何一项，或有多余选项，判为False。\n- 只输出True或False，不要输出其他内容。"""
    elif eval_type == 'judgement':
        prompt = f"""你是金融考试的自动判分专家。请判断模型输出的判断题与标准答案是否实质一致。\n\n【题目】\n{question_with_options}\n\n【模型输出】\n{model_answer}\n\n【标准答案】\n{correct_answer}\n\n判分规则：\n- 只要模型输出表达的意思与标准答案一致，即判为True。\n- 否则判为False。\n- 只输出True或False，不要输出其他内容。"""
    elif eval_type == 'multi_round_qa':
        prompt = f"""你是一名专业的答案检察员。请你从模型输出的答案中，提取唯一一个被 \\boxed{{}} 包裹的内容（即 \\boxed{{答案}} ），然后将其与标准答案进行对比：
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

【题目与选项】
{question_with_options}
【模型输出】
{model_answer}
【标准答案】
{correct_answer}

请只输出True或False，不要输出其他内容。"""
    else:
        prompt = f"""你作为一名专业的金融答案检察员，请比对以下答案是否一致...\n问题和选项：{question_with_options}\n模型输出答案：{model_answer}\n正确答案：{correct_answer}\n只要模型输出明确表达了标准答案的内容（无论是选项字母还是内容本身），即判为True。\n- 如果模型输出与标准答案内容不一致，或表达模糊、无法判断，则判为False。\n- 只输出True或False，不要输出其他内容。"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": "你是一个严格的答案评判专家，只输出True或False。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=10,
                stream=False
            )
            result = response.choices[0].message.content.strip()
            return result == "True"
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2 ** attempt))
                continue
            else:
                raise

def compress_image(input_path: str, output_path: str, max_size=(384, 384), quality=85):
    """Compress image to specified resolution and quality"""
    try:
        img = Image.open(input_path)
        img = img.convert("RGB")
        # Compatibility for different Pillow versions
        if hasattr(Image, 'Resampling'):
            resample = Image.Resampling.LANCZOS
        else:
            resample = Image.ANTIALIAS
        img.thumbnail(max_size, resample)
        img.save(output_path, format="JPEG", quality=quality)
    except Exception as e:
        logging.warning(f"Image compression failed: {input_path}, error: {e}")

def check_image_paths(image_paths: List[str]) -> List[str]:
    """Check image paths and formats, auto-compress if >5 images, return valid paths"""
    supported_formats = {'jpg', 'jpeg', 'png', 'webp', 'bmp'}
    valid_paths = []
    need_compress = len(image_paths) > 5
    for image_path in image_paths:
        if not os.path.exists(image_path):
            logging.warning(f"Image file not found: {image_path}")
            continue
        image_format = Path(image_path).suffix.lower().replace('.', '')
        if image_format not in supported_formats:
            logging.warning(f"Unsupported image format: {image_path}")
            continue
        # Compress if more than 5 images
        if need_compress:
            compress_image(image_path, image_path, max_size=(384, 384), quality=85)
        valid_paths.append(image_path)
    return valid_paths

def preprocess_questions(rounds: List[Dict[str, Any]]) -> List[str]:
    """拼接每轮问题、选项、背景、补充信息，保证鲁棒性"""
    questions = []
    for r in rounds:
        opts = [f"{opt}. {r.get(opt, '')}" for opt in ['A', 'B', 'C', 'D'] if r.get(opt, '')]
        ref_info = []
        if r.get('background_story', ''):
            ref_info.append(f"[背景信息] {r.get('background_story', '')}")
        if r.get('information', ''):
            ref_info.append(f"[补充信息] {r.get('information', '')}")
        qopt = ""
        if ref_info:
            qopt += "\n".join(ref_info) + "\n"
        qopt += r.get('question', '')
        if opts:
            qopt += "\n" + "\n".join(opts)
        questions.append(qopt)
    return questions

def single_sample_eval(
    idx: str,
    rounds: List[Dict[str, Any]],
    pipe: Any,
    api_client: Any,
    prompt: str,
    eval_type: str,
    type_key: str,
    type_name: str,
    args: argparse.Namespace,
    skipped_indices: List[str],
    skipped_errors: List[str],
    completed_indices: set
) -> Tuple[Any, bool]:
    """Evaluate single sample, return result and success status"""
    image_paths = check_image_paths(rounds[0]['image'])
    if not image_paths:
        logging.warning(f"Sample index={idx} all images invalid, skipping")
        skipped_indices.append(idx)
        skipped_errors.append(f"All images invalid")
        return None, False
    questions = preprocess_questions(rounds)
    correct_answers = [r.get('answer', '') for r in rounds]
    md_path = rounds[0].get('md_path', '')
    md_content = ''
    if type_key == 'L3_Q1' and md_path:
        # Path handling: add prefix like images
        if not os.path.isabs(md_path):
            md_path = os.path.join('/root/VisFinEval', md_path)
        try:
            with open(md_path, 'r', encoding='utf-8') as f:
                md_content = f.read()
        except Exception as e:
            logging.warning(f"Failed to read markdown file {md_path}: {e}")
            md_content = ''
    # Log detailed input
    logging.info(f"\n===== Sample index: {idx} =====")
    logging.info(f"Type key: {type_key} | Type name: {type_name}")
    logging.info(f"fintype: {rounds[0].get('fintype', '')}")
    logging.info(f"type: {rounds[0].get('type', '')}")
    logging.info(f"md_path: {md_path}")
    logging.info(f"background_story: {rounds[0].get('background_story', '')}")
    logging.info(f"information: {rounds[0].get('information', '')}")
    logging.info(f"Image paths: {image_paths}")
    logging.info(f"Prompt: \n{prompt.strip()}")
    for i, r in enumerate(rounds, 1):
        logging.info(f"  [Round {i}]")
        logging.info(f"    round: {r.get('round', '')}")
        logging.info(f"    Question: {r.get('question', '')}")
        for opt in ['A', 'B', 'C', 'D']:
            logging.info(f"    Option {opt}: {r.get(opt, '')}")
        logging.info(f"    Correct answer: {r.get('answer', '')}")
        logging.info(f"    Question+Options: {questions[i-1]}")
    # Special handling for L3_Q1
    try:
       if type_key == 'L3_Q1':
            pred_img, dialogue_img = analyze_multi_round(pipe, api_client, image_paths, questions, prompt, args.use_api, args.api_model_name, args.top_k, args.model_type)
            is_img_correct = evaluate_answer_judge(JUDGE_CLIENT, pred_img[0].strip(), correct_answers[0].strip(), questions[0], eval_type)
            md_prompt = "你是一位专业的金融分析师，我将提供给你一个markdown格式的表格数据和一道四个选项的单项选择题。请根据表格数据回答此问题。\n\n注意事项：\n1. 只需输出正确选项的字母(A/B/C/D)\n2. 请仔细分析markdown中的数据\n3. 确保答案的准确性和客观性\n4. 题目为单选题，请只输出一个选项。\n"
            md_question = questions[0]
            md_qopt = questions[0]
            if md_content:
                if args.use_api:
                    messages = [
                        {"role": "user", "content": [
                            {"type": "text", "text": f"{md_prompt}\n\n文本内容：{md_content}\n\n{md_question}"}
                        ]}
                    ]
                    response = api_client.chat.completions.create(
                        model=args.api_model_name,
                        messages=messages
                    )
                    pred_md = response.choices[0].message.content
                else:
                    full_prompt = f"{md_prompt}\n\n文本内容：{md_content}\n\n{md_question}"
                    messages = [
                        {'role': 'user', 'content': [{'type': 'text', 'text': full_prompt}]}
                    ]
                    response = pipe(messages, gen_config=GenerationConfig(top_k=args.top_k))
                    pred_md = response.text
                is_md_correct = evaluate_answer_judge(JUDGE_CLIENT, pred_md.strip(), correct_answers[0].strip(), md_qopt, eval_type)
            else:
                pred_md = ""
                is_md_correct = False
                logging.warning(f"L3_Q1: markdown file missing, md_path={md_path}")
            all_correct = is_img_correct and is_md_correct
            sample_accuracy = 1.0 if all_correct else 0.0
            logging.info(f"[L3_Q1] Image QA output: {pred_img[0]}")
            logging.info(f"[L3_Q1] MD QA output: {pred_md}")
            logging.info(f"[L3_Q1] Image score: {is_img_correct} | MD score: {is_md_correct} | all_correct: {all_correct}")
            result = {
                'index': idx,
                'type_key': type_key,
                'fintype': rounds[0].get('fintype', ''),
                'type': rounds[0].get('type', ''),
                'image_path': image_paths,
                'md_path': md_path,
                'prompt': prompt,
                'eval_type': eval_type,
                'background_story': rounds[0].get('background_story', ''),
                'information': rounds[0].get('information', ''),
                'image_answer': pred_img[0],
                'md_answer': pred_md,
                'image_correct': is_img_correct,
                'md_correct': is_md_correct,
                'all_correct': all_correct,
                'sample_accuracy': sample_accuracy
            }
            completed_indices.add(idx)
            return result, True
        # Other question types
        pred_answers, dialogue_history = analyze_multi_round(pipe, api_client, image_paths, questions, prompt, args.use_api, args.api_model_name, args.top_k, args.model_type)
        for i, ans in enumerate(pred_answers, 1):
             logging.info(f"  [Round {i}] Model output: {ans}")
        round_results = []
        for i, (pred, true, qopt, dialogue) in enumerate(zip(pred_answers, correct_answers, questions, dialogue_history), 1):
            try:
                is_correct = evaluate_answer_judge(JUDGE_CLIENT, pred.strip(), true.strip(), qopt, eval_type)
            except Exception as e:
                logging.error(f"Evaluation error: {e}")
                is_correct = False
            logging.info(f"  [Round {i}] Judgment: Predicted={pred.strip()} | Correct={true.strip()} | IsCorrect={is_correct}")
            round_results.append({
                'round': i,
                'predicted': pred.strip(),
                'correct': true.strip(),
                'question_with_options': qopt,
                'is_correct': is_correct,
                'dialogue': dialogue
            })
        sample_accuracy = sum(r['is_correct'] for r in round_results) / len(round_results) if round_results else 0
        result = {
            'index': idx,
            'type_key': type_key,
            'fintype': rounds[0].get('fintype', ''),
            'type': rounds[0].get('type', ''),
            'image_path': image_paths,
            'prompt': prompt,
            'eval_type': eval_type,
            'background_story': rounds[0].get('background_story', ''),
            'information': rounds[0].get('information', ''),
            'rounds': round_results,
            'all_correct': all(r['is_correct'] for r in round_results),
            'sample_accuracy': sample_accuracy
        }
        completed_indices.add(idx)
        return result, True
    except Exception as e:
        err_msg = f"Sample index={idx} evaluation error: {e}"
        logging.error(err_msg)
        skipped_indices.append(idx)
        skipped_errors.append(err_msg)
        return None, False

def main(args: argparse.Namespace):
    setup_logging(args.log_dir, args.log_level)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    api_client = None
    if args.use_api:
        api_client = OpenAI(api_key=args.api_key, base_url=args.api_base_url) if args.api_base_url else OpenAI(api_key=args.api_key)
    pipe = None
    if not args.use_api:
        if not args.model_path:
            raise ValueError("Local model requires --model_path")
        if "Phi-3.5-vision" in args.model_path:
            backend_cfg = PytorchEngineConfig()
        elif "llava" in args.model_path:
            backend_cfg = TurbomindEngineConfig(tp=args.tp, session_len=4096)
        else:
            backend_cfg = TurbomindEngineConfig(tp=args.tp)
        pipe = pipeline(args.model_path, backend_config=backend_cfg)
    grouped = load_and_group_tsv(args.input_file)
    results = []
    skipped = 0
    skipped_indices = []
    skipped_errors = []
    completed_indices = set()
    # Resume from checkpoint: read completed indices if output file exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    tmp_file_name = f"{args.run_model_name or args.api_model_name}_tmp.json"
    out_path = output_dir / tmp_file_name
    if out_path.exists():
        with open(out_path, 'r', encoding='utf-8') as f:
            try:
                prev = json.load(f)
                for r in prev.get('results', []):
                    completed_indices.add(r['index'])
                results = prev.get('results', [])
                logging.info(f"Resuming from checkpoint, completed samples: {len(completed_indices)}")
            except Exception as e:
                logging.warning(f"Failed to read checkpoint file: {e}")
    total = len(grouped)
    for idx, rounds in tqdm(grouped.items(), desc="Evaluation progress", total=total):
        if idx in completed_indices:
            continue
        result, success = single_sample_eval(idx, rounds, pipe, api_client, *get_prompt_and_eval_type(idx), args, skipped_indices, skipped_errors, completed_indices)
        if success:
            results.append(result)
            # Write checkpoint file in real-time
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump({'results': results}, f, ensure_ascii=False, indent=2)
    skipped = len(skipped_indices)
    # Statistics
    total = len(results)
    perfect = sum(1 for r in results if r['all_correct'])
    perfect_rate = perfect / total if total else 0
    avg_sample_acc = sum(r['sample_accuracy'] for r in results) / total if total else 0
    # Normalize round structure for all samples: those without 'rounds' field (e.g. L3_Q1) are treated as single round    
    normalized_results = []
    for r in results:
        if 'rounds' in r:
            normalized_results.append(r)
        else:
            single_round = {
                'round': 1,
                'predicted': r.get('image_answer', ''),
                'correct': r.get('md_answer', ''),
                'question_with_options': '',
                'is_correct': r.get('all_correct', False),
                'dialogue': []
            }
            r['rounds'] = [single_round]
            normalized_results.append(r)
    max_rounds = max(len(r['rounds']) for r in normalized_results) if normalized_results else 0
    round_stats = {}
    for i in range(max_rounds):
        round_correct = sum(1 for r in normalized_results if i < len(r['rounds']) and r['rounds'][i]['is_correct'])
        round_total = sum(1 for r in normalized_results if i < len(r['rounds']))
        round_stats[f'round_{i+1}'] = {
            'correct': round_correct,
            'total': round_total,
            'accuracy': round_correct / round_total if round_total else 0
        }
    # Output
    if args.use_api:
        run_model_name = args.api_model_name
    else:
        run_model_name = args.run_model_name
    output = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'args': vars(args),
            'run_model_name': run_model_name,
            'skipped_samples': skipped,
            'skipped_indices': skipped_indices,
            'skipped_errors': skipped_errors
        },
        'results': results,
        'summary': {
            'total_samples': total,
            'perfect_dialogues': perfect,
            'perfect_dialogue_rate': perfect_rate,
            'average_sample_accuracy': avg_sample_acc,
            'round_statistics': round_stats,
            'skipped_samples': skipped,
            'skipped_indices': skipped_indices,
            'skipped_errors': skipped_errors
        }
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    final_out_path = output_dir / f"{run_model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(final_out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logging.info(f"Evaluation completed, results saved to {final_out_path}")

    # Automatically delete checkpoint file (tmp) only when all samples are completed
    tmp_path = out_path
    if len(completed_indices) == len(grouped):
        if tmp_path.exists():
            try:
                os.remove(tmp_path)
                logging.info(f"Automatically deleted checkpoint file {tmp_path}")
            except Exception as e:
                logging.warning(f"Failed to delete checkpoint file: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='L1/L2 Multi-round Evaluation Integration Script')
    parser.add_argument('--use_api', action='store_true', help='Whether to use API model')
    parser.add_argument('--api_key', type=str, help='API key')
    parser.add_argument('--api_base_url', type=str, help='API base URL')
    parser.add_argument('--api_model_name', type=str, default='none', help='API model name')
    parser.add_argument('--model_path', type=str, help='Local model path')
    parser.add_argument('--tp', type=int, default=2, help='Tensor parallel degree')
    parser.add_argument('--top_k', type=int, default=1, help='Generation configuration parameter')
    parser.add_argument('--input_file', type=str, required=True, help='Input TSV file path')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--log_dir', type=str, default='logs', help='Log directory')
    parser.add_argument('--log_level', type=str, default='INFO', help='Log level')
    parser.add_argument('--gpu_ids', type=str, default='0', help='Specify GPU IDs to use')
    parser.add_argument('--model_type', type=str, default='qwen', help='Model type (qwen/minicpm)')
    parser.add_argument('--zhipu_key', type=str, required=True, help='ZhipuAI evaluation API key')
    parser.add_argument('--run_model_name', type=str, default='', help='Runtime model name (for recording)')
    args = parser.parse_args()
    main(args)