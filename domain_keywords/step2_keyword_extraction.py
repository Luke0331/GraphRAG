 #!/usr/bin/env python3
"""
第2步：自动提取 - 使用KeyBERT批量挖掘候选术语
从所有文本文件中提取关键词和短语
"""

import os
from pathlib import Path
import logging
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import re
import threading
import time

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 配置参数
MAX_TEXT_LENGTH = 50000  # 最大文本长度（字符）
MAX_PROCESSING_TIME = 300  # 最大处理时间（秒）
CHUNK_SIZE = 3000  # 分块大小

def load_keybert_model():
    """加载KeyBERT模型"""
    logger.info("正在加载KeyBERT模型...")
    # 使用适合科学文献的模型
    model = SentenceTransformer('all-MiniLM-L6-v2')
    kw_model = KeyBERT(model=model)
    logger.info("KeyBERT模型加载完成")
    return kw_model

def clean_keyword(keyword):
    """清理关键词"""
    # 移除首尾空白
    keyword = keyword.strip()
    # 移除特殊字符但保留连字符和空格
    keyword = re.sub(r'[^\w\s\-]', '', keyword)
    # 转换为小写
    keyword = keyword.lower()
    return keyword

def extract_keywords_from_text(text, kw_model, top_n=30):
    """从单个文本中提取关键词"""
    try:
        # 如果文本太长，进行分块处理
        if len(text) > MAX_TEXT_LENGTH:
            logger.warning(f"文本过长 ({len(text)} 字符)，进行分块处理")
            return extract_keywords_from_chunks(text, kw_model, top_n)
        
        # 提取关键词，支持1-3个词的短语
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 3),  # 1-3个词的短语
            stop_words='english',
            use_mmr=True,  # 使用MMR增加多样性
            diversity=0.7,  # 多样性参数
            top_n=top_n
        )
        
        # 清理和过滤关键词
        cleaned_keywords = []
        for keyword, score in keywords:
            cleaned = clean_keyword(keyword)
            # 过滤掉太短或太长的关键词
            if len(cleaned) >= 2 and len(cleaned) <= 50:
                cleaned_keywords.append(cleaned)
        
        return cleaned_keywords
        
    except Exception as e:
        logger.error(f"提取关键词时出错: {str(e)}")
        return []

def extract_keywords_from_chunks(text, kw_model, top_n=30):
    """从分块文本中提取关键词"""
    # 按句子分割文本
    sentences = re.split(r'[.!?]+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < CHUNK_SIZE:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # 限制块数
    chunks = chunks[:10]  # 最多处理10个块
    
    all_keywords = []
    for i, chunk in enumerate(chunks):
        try:
            keywords = kw_model.extract_keywords(
                chunk,
                keyphrase_ngram_range=(1, 3),
                stop_words='english',
                use_mmr=True,
                diversity=0.7,
                top_n=min(10, top_n // len(chunks))  # 每块提取较少的关键词
            )
            
            for keyword, score in keywords:
                cleaned = clean_keyword(keyword)
                if len(cleaned) >= 2 and len(cleaned) <= 50:
                    all_keywords.append(cleaned)
                    
        except Exception as e:
            logger.warning(f"处理第 {i+1} 块时出错: {str(e)}")
            continue
    
    # 去重并限制数量
    unique_keywords = list(set(all_keywords))[:top_n]
    return unique_keywords

def process_file_with_timeout(text_file, kw_model, top_n=30):
    """带超时控制的文件处理"""
    result = {'keywords': [], 'error': None}
    
    def process_file():
        try:
            start_time = time.time()
            
            # 读取文本文件，增强容错
            try:
                with open(text_file, 'r', encoding='utf-8') as f:
                    text = f.read()
            except UnicodeDecodeError:
                result['error'] = f"无法用utf-8解码文件: {text_file.name}"
                return
            
            # 如果文本太短，跳过
            if len(text.strip()) < 100:
                result['error'] = f"文件 {text_file.name} 内容太短"
                return
            
            # 检查文件大小
            if len(text) > MAX_TEXT_LENGTH * 2:
                logger.warning(f"文件 {text_file.name} 过大 ({len(text)} 字符)，仅处理前 {MAX_TEXT_LENGTH} 字符")
                text = text[:MAX_TEXT_LENGTH]
            
            # 提取关键词
            keywords = extract_keywords_from_text(text, kw_model, top_n)
            
            processing_time = time.time() - start_time
            logger.info(f"处理时间: {processing_time:.1f} 秒")
            
            result['keywords'] = keywords
            
        except Exception as e:
            result['error'] = str(e)
    
    # 创建线程并设置超时
    thread = threading.Thread(target=process_file)
    thread.daemon = True
    thread.start()
    
    # 等待线程完成或超时
    thread.join(timeout=MAX_PROCESSING_TIME)
    
    if thread.is_alive():
        logger.error(f"处理文件 {text_file.name} 超时 ({MAX_PROCESSING_TIME} 秒)，已跳过")
        return []
    
    if result['error']:
        logger.error(f"处理 {text_file.name} 时出错: {result['error']}")
        return []
    
    return result['keywords']

def main():
    """主函数"""
    # 设置路径
    text_dir = Path("extracted_texts")
    output_file = Path("extracted_keywords.txt")
    
    # 检查文本目录是否存在
    if not text_dir.exists():
        logger.error(f"文本目录不存在: {text_dir}")
        logger.info("请先运行 step1_pdf_to_text.py")
        return
    
    # 获取所有文本文件
    text_files = list(text_dir.glob("*.txt"))
    
    if not text_files:
        logger.error(f"在 {text_dir} 中没有找到文本文件")
        return
    
    logger.info(f"找到 {len(text_files)} 个文本文件")
    
    # 加载KeyBERT模型
    kw_model = load_keybert_model()
    
    # 存储所有关键词
    all_keywords = set()
    
    # 处理每个文本文件
    for i, text_file in enumerate(text_files, 1):
        logger.info(f"正在处理文件 {i}/{len(text_files)}: {text_file.name}")
        
        # 使用超时控制处理文件
        keywords = process_file_with_timeout(text_file, kw_model, top_n=30)
        
        # 添加到总集合中
        all_keywords.update(keywords)
        
        logger.info(f"从 {text_file.name} 提取了 {len(keywords)} 个关键词")
    
    # 保存结果
    sorted_keywords = sorted(list(all_keywords))
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for keyword in sorted_keywords:
            f.write(keyword + '\n')
    
    logger.info(f"关键词提取完成！")
    logger.info(f"总共提取了 {len(sorted_keywords)} 个唯一关键词")
    logger.info(f"结果保存在: {output_file.absolute()}")
    
    # 显示前20个关键词作为示例
    logger.info("前20个关键词示例:")
    for i, keyword in enumerate(sorted_keywords[:20], 1):
        logger.info(f"{i:2d}. {keyword}")

if __name__ == "__main__":
    main()