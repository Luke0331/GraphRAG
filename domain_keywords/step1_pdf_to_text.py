#!/usr/bin/env python3
"""
第1步：预处理 - 将所有文献转换为纯文本 (.txt)
使用PyMuPDF (fitz) 将PDF文件转换为纯文本
"""

import os
import fitz  # PyMuPDF
import re
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_text(text):
    """清理提取的文本"""
    # 移除多余的空白字符
    text = re.sub(r'\s+', ' ', text)
    # 移除特殊字符但保留基本标点
    text = re.sub(r'[^\w\s\.\,\;\:\!\?\(\)\[\]\{\}\-\+\=\*\/\@\#\$\%\&\|]', '', text)
    # 移除行首行尾空白
    text = text.strip()
    return text

def pdf_to_text(pdf_path, output_dir):
    """将单个PDF文件转换为文本文件"""
    try:
        # 打开PDF文件
        doc = fitz.open(pdf_path)
        text_content = []
        
        logger.info(f"正在处理: {pdf_path}")
        
        # 逐页提取文本
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            if text.strip():  # 只添加非空页面
                text_content.append(text)
        
        doc.close()
        
        # 合并所有页面文本
        full_text = '\n\n'.join(text_content)
        full_text = clean_text(full_text)
        
        # 生成输出文件名
        pdf_name = Path(pdf_path).stem
        output_file = output_dir / f"{pdf_name}.txt"
        
        # 保存文本文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(full_text)
        
        logger.info(f"成功转换: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"处理 {pdf_path} 时出错: {str(e)}")
        return False

def main():
    """主函数"""
    # 设置路径
    pdf_dir = Path("../zotero")  # 上一级目录的zotero文件夹
    output_dir = Path("extracted_texts")
    
    # 创建输出目录
    output_dir.mkdir(exist_ok=True)
    
    # 获取所有PDF文件
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        logger.error(f"在 {pdf_dir} 中没有找到PDF文件")
        return
    
    logger.info(f"找到 {len(pdf_files)} 个PDF文件")
    
    # 处理每个PDF文件
    success_count = 0
    for pdf_file in pdf_files:
        if pdf_to_text(pdf_file, output_dir):
            success_count += 1
    
    logger.info(f"转换完成！成功转换 {success_count}/{len(pdf_files)} 个文件")
    logger.info(f"文本文件保存在: {output_dir.absolute()}")

if __name__ == "__main__":
    main() 