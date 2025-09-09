#!/usr/bin/env python3
"""
第4步：自动生成 - 从表格一键生成最终的JSON文件
将人工整理的Excel表格转换为结构化的JSON词典
"""

import pandas as pd
import json
from pathlib import Path
import logging
from collections import defaultdict

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_curated_data():
    """加载人工整理的数据"""
    excel_file = Path("curated_dictionary.xlsx")
    
    if not excel_file.exists():
        logger.error(f"Excel文件不存在: {excel_file}")
        logger.info("请先运行 step3_create_spreadsheet.py 并完成人工整理")
        return None
    
    try:
        # 读取Excel文件
        df = pd.read_excel(excel_file)
        logger.info(f"成功读取Excel文件，包含 {len(df)} 行数据")
        return df
    except Exception as e:
        logger.error(f"读取Excel文件时出错: {str(e)}")
        return None

def process_curated_data(df):
    """处理人工整理的数据"""
    # 筛选出保留的词条
    df_kept = df[df['Is_Keep'] == 1].copy()
    
    if df_kept.empty:
        logger.error("没有找到标记为保留的词条")
        return []
    
    logger.info(f"找到 {len(df_kept)} 个保留的词条")
    
    # 按标准名分组
    grouped_data = defaultdict(list)
    
    for _, row in df_kept.iterrows():
        candidate_term = str(row['Candidate_Term']).strip()
        standard_name = str(row['Standard_Name']).strip()
        category = str(row['Category']).strip()
        
        # 跳过空的标准名
        if not standard_name or standard_name == 'nan':
            logger.warning(f"跳过没有标准名的词条: {candidate_term}")
            continue
        
        # 添加到分组中
        grouped_data[standard_name].append({
            'alias': candidate_term,
            'category': category if category and category != 'nan' else 'Unknown'
        })
    
    # 构建最终词典
    final_dictionary = []
    
    for standard_name, aliases_data in grouped_data.items():
        # 收集所有别名
        aliases = [item['alias'] for item in aliases_data]
        
        # 确保标准名在别名列表中
        if standard_name not in aliases:
            aliases.append(standard_name)
        
        # 获取类别（使用第一个非空的类别）
        categories = [item['category'] for item in aliases_data if item['category'] != 'Unknown']
        category = categories[0] if categories else 'Unknown'
        
        entry = {
            "standard_name": standard_name,
            "aliases": sorted(list(set(aliases))),  # 去重并排序
            "category": category
        }
        
        final_dictionary.append(entry)
    
    # 按标准名排序
    final_dictionary.sort(key=lambda x: x['standard_name'])
    
    return final_dictionary

def generate_statistics(dictionary):
    """生成统计信息"""
    stats = {
        'total_terms': len(dictionary),
        'categories': defaultdict(int),
        'total_aliases': 0
    }
    
    for entry in dictionary:
        stats['categories'][entry['category']] += 1
        stats['total_aliases'] += len(entry['aliases'])
    
    return stats

def save_json_dictionary(dictionary, output_file):
    """保存JSON词典"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dictionary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"JSON词典已保存: {output_file.absolute()}")
        return True
    except Exception as e:
        logger.error(f"保存JSON文件时出错: {str(e)}")
        return False

def save_summary_report(dictionary, stats, output_file):
    """保存摘要报告"""
    report = f"""# 硅电池领域词典生成报告

## 统计信息
- 总术语数: {stats['total_terms']}
- 总别名数: {stats['total_aliases']}
- 平均别名数: {stats['total_aliases'] / stats['total_terms']:.1f}

## 分类统计
"""
    
    for category, count in sorted(stats['categories'].items()):
        percentage = (count / stats['total_terms']) * 100
        report += f"- {category}: {count} ({percentage:.1f}%)\n"
    
    report += "\n## 术语列表\n"
    
    for entry in dictionary:
        report += f"\n### {entry['standard_name']} ({entry['category']})\n"
        report += f"别名: {', '.join(entry['aliases'])}\n"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"摘要报告已保存: {output_file.absolute()}")

def main():
    """主函数"""
    logger.info("开始生成最终词典...")
    
    # 加载人工整理的数据
    df = load_curated_data()
    if df is None:
        return
    
    # 处理数据
    dictionary = process_curated_data(df)
    if not dictionary:
        return
    
    # 生成统计信息
    stats = generate_statistics(dictionary)
    
    # 保存JSON词典
    json_file = Path("domain_dictionary.json")
    if save_json_dictionary(dictionary, json_file):
        logger.info("✅ 词典生成成功！")
    
    # 保存摘要报告
    report_file = Path("dictionary_report.md")
    save_summary_report(dictionary, stats, report_file)
    
    # 显示统计信息
    logger.info("📊 词典统计:")
    logger.info(f"   - 总术语数: {stats['total_terms']}")
    logger.info(f"   - 总别名数: {stats['total_aliases']}")
    logger.info(f"   - 平均别名数: {stats['total_aliases'] / stats['total_terms']:.1f}")
    
    logger.info("📂 分类统计:")
    for category, count in sorted(stats['categories'].items()):
        percentage = (count / stats['total_terms']) * 100
        logger.info(f"   - {category}: {count} ({percentage:.1f}%)")
    
    logger.info(f"\n🎉 词典生成完成！")
    logger.info(f"📄 JSON文件: {json_file.absolute()}")
    logger.info(f"📋 报告文件: {report_file.absolute()}")

if __name__ == "__main__":
    main() 