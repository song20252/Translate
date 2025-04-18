import re
from typing import List, Dict, Tuple
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import os
from functools import partial

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenAITranslator:
    def __init__(self, api_key: str, base_url: str, model: str = "deepseek-chat"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_retries = 5
        self.timeout = 90

    def translate_batch(self, prompt: str, texts: List[str]) -> List[str]:
        """批量翻译文本，带重试机制"""
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": self._format_batch_input(texts)}
                    ],
                    temperature=0.0,
                    timeout=self.timeout
                )
                responsed_message = response.choices[0].message.content
                parsed = self._parse_response(responsed_message, len(texts))
                # 检查解析结果是否有效
                if not self._is_valid_translation(parsed, len(texts)):
                    logger.info(f"Submitting to model - Prompt: {prompt}, Texts: {texts}")
                    logger.info(f"Received from model - Raw Response: {responsed_message}")
                    raise ValueError("Translation result does not match expected output.")
                return parsed
            except Exception as e:
                logger.warning(f"Attempt {attempt}/{self.max_retries} failed: {str(e)}")
        return ["[TRANSLATION FAILED]"] * len(texts)

    def _format_batch_input(self, texts: List[str]) -> str:
        return "\n".join(f"{idx}. {text}" for idx, text in enumerate(texts, 1))

    def _parse_response(self, response: str, expected: int) -> List[str]:
        lines = [line.split(".", 1)[1].strip() for line in response.split("\n") if re.match(r"^\d+\.", line)]
        return lines[:expected] or ["[PARSE ERROR]"] * expected
        
    def _is_valid_translation(self, translations: List[str], expected_count: int) -> bool:
        if len(translations) != expected_count:
            return False
        for translation in translations:
            if not translation.strip() or "[PARSE ERROR]" in translation or "[TRANSLATION FAILED]" in translation:
                return False
        return True

def _should_translate(line: str) -> bool:
    """判断是否需要翻译该行"""
    line = line.strip()
    return bool(line) and not re.match(
        r"^(\d+|\d{2}:\d{2}:\d{2},\d{3}\s-->\s\d{2}:\d{2}:\d{2},\d{3})", 
        line
    )

def _process_chunk(
    chunk: List[str], 
    chunk_start_idx: int,
    translator: OpenAITranslator, 
    prompt: str
) -> Tuple[int, Dict[int, str]]:
    """
    处理单个文本块，返回:
    - chunk的起始行号
    - {行号: 翻译后的文本} 的字典
    """
    translations = {}
    positions = [i for i, line in enumerate(chunk) if _should_translate(line)]
    texts_to_translate = [chunk[pos] for pos in positions]
    
    if texts_to_translate:
        translated_texts = translator.translate_batch(prompt, texts_to_translate)
        for pos, translated in zip(positions, translated_texts):
            absolute_pos = chunk_start_idx + pos
            translations[absolute_pos] = translated
    
    return chunk_start_idx, translations

def do_translate(
    input_file: str,
    output_file: str,
    prompt: str,
    api_key: str,
    base_url: str,
    max_workers: int = 10,
    chunk_size: int = 500
) -> None:
    translator = OpenAITranslator(api_key, base_url)
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = [line.rstrip('\n') for line in f]
    except IOError as e:
        logger.error(f"无法读取输入文件: {str(e)}")
        raise

    # 分块处理并保留原始行号
    chunks = [(i, lines[i:i + chunk_size]) for i in range(0, len(lines), chunk_size)]
    all_translations = {}  # 存储所有翻译结果 {行号: 翻译文本}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        process_func = partial(_process_chunk, translator=translator, prompt=prompt)
        futures = {
            executor.submit(process_func, chunk, start_idx): start_idx
            for start_idx, chunk in chunks
        }
        
        for future in as_completed(futures):
            start_idx, translations = future.result()
            all_translations.update(translations)
            logger.info(f"Processed chunk starting at line {start_idx}, translated {len(translations)} lines")

    # 构建最终结果，保持原始顺序
    final_lines = []
    for idx, original_line in enumerate(lines):
        if idx in all_translations:
            final_lines.append(all_translations[idx])
        else:
            final_lines.append(original_line)

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(final_lines))
        logger.info(f"翻译完成，结果已保存到 {output_file}")
    except IOError as e:
        logger.error(f"无法写入输出文件: {str(e)}")
        raise

if __name__ == '__main__':
    config = {
        "prompt": "你是个专业翻译，请直接给出中文译文，不需要推理过程。你必须严格保持原格式，每条以数字编号开头",
        "api_key": "sk-67f60cd559b34f0b9caf6c4cb3167256",
        "base_url": "http://211.90.219.175:8000/v1/",
        "max_workers": 30,
        "chunk_size": 20
    }

    try:
        # 获取 srt 子目录路径
        srt_dir = os.path.join(os.getcwd(), "srt")
        if not os.path.exists(srt_dir):
            logger.error("srt 子目录不存在")
            raise FileNotFoundError("srt 子目录不存在")

        # 遍历 srt 子目录中的所有 .srt 文件
        for file_name in os.listdir(srt_dir):
            if file_name.endswith(".srt"):
                input_file = os.path.join(srt_dir, file_name)
                output_file = os.path.join(srt_dir, f"{os.path.splitext(file_name)[0]}_translated.srt")
                
                # 调用翻译函数
                try:
                    do_translate(
                        input_file=input_file,
                        output_file=output_file,
                        prompt=config["prompt"],
                        api_key=config["api_key"],
                        base_url=config["base_url"],
                        max_workers=config["max_workers"],
                        chunk_size=config["chunk_size"]
                    )
                except Exception as e:
                    logger.error(f"翻译文件 {file_name} 时发生错误: {str(e)}")
        
        logger.info("所有文件翻译完成")
    except Exception as e:
        logger.error(f"主程序发生错误: {str(e)}")
