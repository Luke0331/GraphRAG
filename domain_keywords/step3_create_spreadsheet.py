#!/usr/bin/env python3
"""
第3步：人工精修 - 创建电子表格模板
将提取的关键词导入到Excel文件中，方便人工整理和标准化
"""

import pandas as pd
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_spreadsheet_template():
    """创建电子表格模板"""
    # 读取提取的关键词
    keywords_file = Path("all_extracted_keywords.txt")
    
    if not keywords_file.exists():
        logger.error(f"关键词文件不存在: {keywords_file}")
        logger.info("请先运行 step2_keyword_extraction.py")
        return
    
    # 读取关键词
    with open(keywords_file, 'r', encoding='utf-8') as f:
        keywords = [line.strip() for line in f if line.strip()]
    
    logger.info(f"读取了 {len(keywords)} 个关键词")
    
    # 创建DataFrame
    df = pd.DataFrame({
        'Candidate_Term': keywords,  # A列：候选术语
        'Is_Keep': '',              # B列：是否保留 (1=保留, 0=丢弃)
        'Standard_Name': '',        # C列：标准名称
        'Category': ''              # D列：类别
    })
    
    # 保存为Excel文件
    output_file = Path("all_curated_dictionary.xlsx")
    df.to_excel(output_file, index=False, sheet_name='Keywords')
    
    logger.info(f"电子表格模板已创建: {output_file.absolute()}")
    logger.info("请按照以下步骤进行人工精修:")
    logger.info("1. 打开 curated_dictionary.xlsx")
    logger.info("2. 在B列中标记 1(保留) 或 0(丢弃)")
    logger.info("3. 在C列中填写标准名称")
    logger.info("4. 在D列中填写类别 (如: Technology, Material, Metric, Process)")
    logger.info("5. 保存文件后运行 step4_generate_json.py")

def create_instructions_file():
    """创建操作说明文件"""
    instructions = """
# 硅电池领域词典人工精修指南

## 文件说明
- curated_dictionary.xlsx: 需要人工整理的电子表格
- extracted_keywords.txt: 自动提取的原始关键词

## 操作步骤

### 第一步：快速筛选
1. 打开 curated_dictionary.xlsx
2. 按A列（候选术语）排序，相似术语会聚集在一起
3. 在B列中快速标记：
   - 1 = 保留（有价值的术语）
   - 0 = 丢弃（无意义或重复的术语）

### 第二步：标准化命名
1. 对标记为1的行，在C列填写标准名称
2. 相似术语使用统一的标准名，例如：
   - "perc", "perc cell", "perc solar cell" → "PERC Cell"
   - "silicon heterojunction", "heterojunction solar cell" → "Silicon Heterojunction Solar Cell"
   - "anti-reflection coating", "antireflection coating" → "Anti-Reflection Coating"

### 第三步：分类标注
在D列中填写类别，建议使用以下分类：
- Technology: 技术类型（如 PERC, HJT, TOPCon）
- Material: 材料（如 Silicon, ITO, TCO）
- Process: 工艺过程（如 Texturing, Doping, Metallization）
- Metric: 性能指标（如 Efficiency, Fill Factor, Open Circuit Voltage）
- Device: 器件结构（如 Solar Cell, Module, Wafer）
- Property: 物理性质（如 Band Gap, Absorption, Reflection）

### 第四步：质量检查
1. 确保所有保留的术语都有标准名称
2. 检查分类是否合理
3. 保存文件

## 注意事项
- 保持术语的准确性和专业性
- 优先保留硅电池领域的核心术语
- 可以添加同义词和缩写形式
- 完成后运行 step4_generate_json.py 生成最终词典
"""
    
    with open("curation_instructions.md", 'w', encoding='utf-8') as f:
        f.write(instructions)
    
    logger.info("操作说明已保存到: curation_instructions.md")

def main():
    """主函数"""
    logger.info("开始创建电子表格模板...")
    
    # 创建电子表格模板
    create_spreadsheet_template()
    
    # 创建操作说明
    create_instructions_file()
    
    logger.info("模板创建完成！")

if __name__ == "__main__":
    main() 