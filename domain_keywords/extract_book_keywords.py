#!/usr/bin/env python3
"""
对books目录下每本书分块提取关键词，并与extracted_keywords.txt合并去重输出all_extracted_keywords.txt
"""

import os
from pathlib import Path
import logging
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import re

# 日志设置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 分块参数
CHUNK_SIZE = 3000  # 每块字符数
MAX_CHUNKS = 100   # 每本书最多处理100块（约30万字符，防止极端大书）

# 关键词参数
TOP_N_PER_CHUNK = 10

# 路径
BOOKS_DIR = Path('../books') if not Path('books').exists() else Path('books')
PAPER_KEYWORDS_FILE = Path('extracted_keywords.txt')
OUTPUT_FILE = Path('all_extracted_keywords.txt')

# 加载KeyBERT模型
def load_keybert_model():
    logger.info("正在加载KeyBERT模型...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    kw_model = KeyBERT(model=model)
    logger.info("KeyBERT模型加载完成")
    return kw_model

# 清理关键词
def clean_keyword(keyword):
    keyword = keyword.strip()
    keyword = re.sub(r'[^\w\s\-]', '', keyword)
    keyword = keyword.lower()
    return keyword

# 分块
def split_text(text, chunk_size=CHUNK_SIZE, max_chunks=MAX_CHUNKS):
    chunks = [text[i:i+chunk_size] for i in range(0, min(len(text), chunk_size*max_chunks), chunk_size)]
    return chunks

# 对单本书分块提取关键词
def extract_keywords_from_book(book_path, kw_model):
    logger.info(f"处理书籍: {book_path.name}")
    try:
        with open(book_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        logger.error(f"读取 {book_path.name} 失败: {e}")
        return set()
    
    chunks = split_text(text)
    logger.info(f"分为 {len(chunks)} 块")
    all_keywords = set()
    for i, chunk in enumerate(chunks, 1):
        try:
            keywords = kw_model.extract_keywords(
                chunk,
                keyphrase_ngram_range=(1, 3),
                stop_words='english',
                use_mmr=True,
                diversity=0.7,
                top_n=TOP_N_PER_CHUNK
            )
            for keyword, score in keywords:
                cleaned = clean_keyword(keyword)
                if 2 <= len(cleaned) <= 50:
                    all_keywords.add(cleaned)
        except Exception as e:
            logger.warning(f"第{i}块提取失败: {e}")
            continue
    logger.info(f"{book_path.name} 共提取 {len(all_keywords)} 个关键词")
    return all_keywords

# 合并论文关键词
def load_paper_keywords():
    keywords = set()
    if PAPER_KEYWORDS_FILE.exists():
        with open(PAPER_KEYWORDS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                kw = line.strip().lower()
                if kw:
                    keywords.add(kw)
    logger.info(f"已加载论文关键词 {len(keywords)} 个")
    return keywords

# 主流程
def main():
    if not BOOKS_DIR.exists():
        logger.error(f"books目录不存在: {BOOKS_DIR}")
        return
    book_files = list(BOOKS_DIR.glob('*.txt'))
    if not book_files:
        logger.error(f"books目录下没有txt文件")
        return
    logger.info(f"共发现 {len(book_files)} 本书")
    kw_model = load_keybert_model()
    all_keywords = set()
    # 处理每本书
    for book_path in book_files:
        book_keywords = extract_keywords_from_book(book_path, kw_model)
        all_keywords.update(book_keywords)
    logger.info(f"所有书籍共提取 {len(all_keywords)} 个唯一关键词")
    # 合并论文关键词
    paper_keywords = load_paper_keywords()
    all_keywords.update(paper_keywords)
    logger.info(f"合并后总关键词数: {len(all_keywords)}")
    # 保存
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for kw in sorted(all_keywords):
            f.write(kw + '\n')
    logger.info(f"已保存到: {OUTPUT_FILE.absolute()}")
    logger.info("前20个关键词示例:")
    for i, kw in enumerate(sorted(all_keywords)[:20], 1):
        logger.info(f"{i:2d}. {kw}")

if __name__ == "__main__":
    main() 