# -*- coding: utf-8 -*-
"""
修复metadata问题：从文档内容中提取来源信息
"""

import re
from typing import Dict, Any, Optional

def extract_source_from_content(content: str) -> Optional[str]:
    """
    从文档内容中提取来源信息
    
    Args:
        content: 文档内容
        
    Returns:
        来源ID或None
    """
    # 尝试从内容中提取文件名
    patterns = [
        r'文件:([^,]+)',  # 匹配 "文件:xxx.pdf"
        r'【图表】文件:([^,]+)',  # 匹配 "【图表】文件:xxx.pdf"
        r'【表格】文件:([^,]+)',  # 匹配 "【表格】文件:xxx.pdf"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, content)
        if match:
            filename = match.group(1).strip()
            # 移除扩展名
            source_id = filename.replace('.pdf', '').replace('.txt', '')
            return source_id
    
    return None

def enhance_source_documents_with_metadata(source_docs: list) -> list:
    """
    为source_documents添加metadata
    
    Args:
        source_docs: 原始source_documents列表
        
    Returns:
        增强后的source_documents列表
    """
    enhanced_docs = []
    
    for doc in source_docs:
        content = doc.get('content', '')
        
        # 提取来源信息
        source_id = extract_source_from_content(content)
        
        # 创建增强的metadata
        enhanced_metadata = {
            'source': source_id,
            'file_name': f"{source_id}.pdf" if source_id else None,
            'doc_id': source_id,
            'document_id': source_id
        }
        
        # 创建增强的文档
        enhanced_doc = {
            'index': doc.get('index', 0),
            'content': content,
            'metadata': enhanced_metadata
        }
        
        enhanced_docs.append(enhanced_doc)
    
    return enhanced_docs

def test_metadata_extraction():
    """测试metadata提取功能"""
    print("🧪 测试metadata提取功能")
    print("="*60)
    
    # 测试用例
    test_contents = [
        "efficiency, and because they can be prepared and processed under \nmild conditions. These incentives have triggered academic interest to \nachieve upscaling to larger areas, improve their device lifetim...",
        "【表格】文件:Handbook_of_Photovoltaic_Silicon.pdf, 页码:1, 表格序号:1\n数据内容...",
        "【图表】文件:Solar_Cell_Technology.pdf, 页码:5, 图片序号:2, 路径:./extracted_images/xxx.png",
        "普通文本内容，没有文件信息..."
    ]
    
    for i, content in enumerate(test_contents, 1):
        source_id = extract_source_from_content(content)
        print(f"测试 {i}:")
        print(f"  内容: {content[:100]}...")
        print(f"  提取的来源ID: {source_id}")
        print()

if __name__ == "__main__":
    test_metadata_extraction()
