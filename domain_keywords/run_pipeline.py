#!/usr/bin/env python3
"""
硅电池领域词典提取Pipeline主控制脚本
一键运行完整的四步流程
"""

import subprocess
import sys
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_step(step_name, script_name):
    """运行单个步骤"""
    logger.info(f"🚀 开始执行 {step_name}...")
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, encoding='utf-8')
        
        if result.returncode == 0:
            logger.info(f"✅ {step_name} 执行成功")
            if result.stdout:
                logger.info(result.stdout)
            return True
        else:
            logger.error(f"❌ {step_name} 执行失败")
            if result.stderr:
                logger.error(result.stderr)
            return False
            
    except Exception as e:
        logger.error(f"❌ 执行 {step_name} 时出错: {str(e)}")
        return False

def check_dependencies():
    """检查依赖是否安装"""
    required_packages = [
        'PyMuPDF',
        'keybert', 
        'sentence-transformers',
        'pandas',
        'openpyxl'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.lower().replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"❌ 缺少以下依赖包: {', '.join(missing_packages)}")
        logger.info("请运行: pip install " + " ".join(missing_packages))
        return False
    
    logger.info("✅ 所有依赖包已安装")
    return True

def main():
    """主函数"""
    logger.info("🎯 硅电池领域词典提取Pipeline")
    logger.info("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        return
    
    # 定义步骤
    steps = [
        ("第1步：PDF转文本", "step1_pdf_to_text.py"),
        ("第2步：关键词提取", "step2_keyword_extraction.py"),
        ("第3步：创建电子表格", "step3_create_spreadsheet.py")
    ]
    
    # 执行前3步
    for step_name, script_name in steps:
        if not run_step(step_name, script_name):
            logger.error(f"Pipeline在第{step_name}停止")
            return
    
    logger.info("=" * 50)
    logger.info("🎉 自动化步骤完成！")
    logger.info("📋 接下来需要人工操作:")
    logger.info("1. 打开 curated_dictionary.xlsx")
    logger.info("2. 按照 curation_instructions.md 的说明进行人工整理")
    logger.info("3. 完成后运行: python step4_generate_json.py")
    logger.info("=" * 50)

if __name__ == "__main__":
    main() 