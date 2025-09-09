#!/usr/bin/env python3
"""
硅电池领域词典提取Pipeline测试脚本
验证各个步骤的基本功能
"""

import os
import sys
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_dependencies():
    """测试依赖包是否安装"""
    logger.info("🔍 测试依赖包...")
    
    required_packages = {
        'PyMuPDF': 'fitz',
        'keybert': 'keybert',
        'sentence-transformers': 'sentence_transformers',
        'pandas': 'pandas',
        'openpyxl': 'openpyxl'
    }
    
    missing_packages = []
    
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
            logger.info(f"✅ {package_name}")
        except ImportError:
            logger.error(f"❌ {package_name} - 未安装")
            missing_packages.append(package_name)
    
    if missing_packages:
        logger.error(f"缺少依赖包: {', '.join(missing_packages)}")
        logger.info("请运行: pip install -r requirements_dictionary.txt")
        return False
    
    logger.info("✅ 所有依赖包已安装")
    return True

def test_file_structure():
    """测试文件结构"""
    logger.info("🔍 测试文件结构...")
    
    # 检查脚本文件
    script_files = [
        'step1_pdf_to_text.py',
        'step2_keyword_extraction.py', 
        'step3_create_spreadsheet.py',
        'step4_generate_json.py',
        'run_pipeline.py'
    ]
    
    missing_scripts = []
    for script in script_files:
        if Path(script).exists():
            logger.info(f"✅ {script}")
        else:
            logger.error(f"❌ {script} - 文件不存在")
            missing_scripts.append(script)
    
    if missing_scripts:
        logger.error(f"缺少脚本文件: {', '.join(missing_scripts)}")
        return False
    
    # 检查PDF目录
    pdf_dir = Path("../zotero")
    if pdf_dir.exists():
        pdf_files = list(pdf_dir.glob("*.pdf"))
        logger.info(f"✅ PDF目录存在，包含 {len(pdf_files)} 个PDF文件")
    else:
        logger.error("❌ PDF目录不存在: ../zotero")
        return False
    
    logger.info("✅ 文件结构检查通过")
    return True

def test_pdf_extraction():
    """测试PDF文本提取功能"""
    logger.info("🔍 测试PDF文本提取...")
    
    try:
        import fitz
        from pathlib import Path
        
        # 查找一个PDF文件进行测试
        pdf_dir = Path("../zotero")
        pdf_files = list(pdf_dir.glob("*.pdf"))
        
        if not pdf_files:
            logger.error("❌ 没有找到PDF文件")
            return False
        
        # 测试第一个PDF文件
        test_pdf = pdf_files[0]
        logger.info(f"测试文件: {test_pdf.name}")
        
        doc = fitz.open(test_pdf)
        text_content = []
        
        # 只读取前3页进行测试
        for page_num in range(min(3, len(doc))):
            page = doc.load_page(page_num)
            text = page.get_text()
            if text.strip():
                text_content.append(text)
        
        doc.close()
        
        if text_content:
            logger.info(f"✅ PDF文本提取成功，提取了 {len(text_content)} 页内容")
            logger.info(f"   文本长度: {sum(len(text) for text in text_content)} 字符")
            return True
        else:
            logger.error("❌ PDF文本提取失败，没有提取到内容")
            return False
            
    except Exception as e:
        logger.error(f"❌ PDF文本提取测试失败: {str(e)}")
        return False

def test_keyword_extraction():
    """测试关键词提取功能"""
    logger.info("🔍 测试关键词提取...")
    
    try:
        from keybert import KeyBERT
        from sentence_transformers import SentenceTransformer
        
        # 测试文本
        test_text = """
        Silicon heterojunction solar cells have achieved remarkable efficiency improvements 
        through advanced passivation techniques and optimized carrier transport. 
        The PERC technology demonstrates superior performance in commercial applications.
        Anti-reflection coatings enhance light absorption and improve conversion efficiency.
        """
        
        # 加载模型
        model = SentenceTransformer('all-MiniLM-L6-v2')
        kw_model = KeyBERT(model=model)
        
        # 提取关键词
        keywords = kw_model.extract_keywords(
            test_text,
            keyphrase_ngram_range=(1, 3),
            stop_words='english',
            use_mmr=True,
            top_n=10
        )
        
        if keywords:
            logger.info(f"✅ 关键词提取成功，提取了 {len(keywords)} 个关键词")
            logger.info(f"   示例关键词: {[kw[0] for kw in keywords[:5]]}")
            return True
        else:
            logger.error("❌ 关键词提取失败")
            return False
            
    except Exception as e:
        logger.error(f"❌ 关键词提取测试失败: {str(e)}")
        return False

def test_excel_operations():
    """测试Excel操作功能"""
    logger.info("🔍 测试Excel操作...")
    
    try:
        import pandas as pd
        
        # 创建测试数据
        test_data = {
            'Candidate_Term': ['silicon heterojunction', 'perc cell', 'anti-reflection coating'],
            'Is_Keep': [1, 1, 1],
            'Standard_Name': ['Silicon Heterojunction', 'PERC Cell', 'Anti-Reflection Coating'],
            'Category': ['Technology', 'Technology', 'Process']
        }
        
        df = pd.DataFrame(test_data)
        
        # 测试保存Excel
        test_file = 'test_excel.xlsx'
        df.to_excel(test_file, index=False)
        
        # 测试读取Excel
        df_read = pd.read_excel(test_file)
        
        # 清理测试文件
        Path(test_file).unlink()
        
        if len(df_read) == len(df):
            logger.info("✅ Excel操作测试成功")
            return True
        else:
            logger.error("❌ Excel操作测试失败")
            return False
            
    except Exception as e:
        logger.error(f"❌ Excel操作测试失败: {str(e)}")
        return False

def main():
    """主测试函数"""
    logger.info("🧪 开始Pipeline功能测试")
    logger.info("=" * 50)
    
    tests = [
        ("依赖包检查", test_dependencies),
        ("文件结构检查", test_file_structure),
        ("PDF文本提取", test_pdf_extraction),
        ("关键词提取", test_keyword_extraction),
        ("Excel操作", test_excel_operations)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 {test_name}")
        try:
            if test_func():
                passed += 1
            else:
                logger.error(f"❌ {test_name} 失败")
        except Exception as e:
            logger.error(f"❌ {test_name} 异常: {str(e)}")
    
    logger.info("=" * 50)
    logger.info(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        logger.info("🎉 所有测试通过！Pipeline可以正常运行")
        logger.info("💡 运行命令: python run_pipeline.py")
    else:
        logger.error("⚠️  部分测试失败，请检查环境配置")
    
    return passed == total

if __name__ == "__main__":
    main() 