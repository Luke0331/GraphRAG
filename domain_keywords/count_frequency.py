import pandas as pd
import os
import re
from collections import Counter

def count_keyword_frequencies(corpus_dir, keywords_file, output_excel_file):
    """
    计算关键词列表中每个词在语料库中出现的频率。
    """
    print(f"--- 开始关键词频率统计 ---")
    
    # 1. 加载我们最终筛选出的高质量关键词
    print(f"正在加载关键词: '{keywords_file}'...")
    with open(keywords_file, 'r', encoding='utf-8') as f:
        # 我们将在这里把关键词转换为小写，以便不区分大小写地进行匹配
        keywords = {line.strip().lower() for line in f if line.strip()}
    print(f"共加载 {len(keywords)} 个唯一关键词。")

    # 2. 读取语料库中的所有文本内容
    print(f"正在读取语料库文本: '{corpus_dir}'...")
    all_text = ""
    for filename in os.listdir(corpus_dir):
        if filename.endswith(".txt"):
            try:
                with open(os.path.join(corpus_dir, filename), 'r', encoding='utf-8') as f:
                    all_text += f.read().lower() + " " # 转换为小写
            except Exception as e:
                print(f"Warning: Could not read file {filename}. Error: {e}")
    print("语料库读取完成。")
    
    # 3. 高效地统计每个关键词的频率
    print("正在计算词频 (这可能需要几分钟，请耐心等待)...")
    keyword_counts = Counter()
    total_keywords = len(keywords)
    for i, keyword in enumerate(keywords):
        # 使用正则表达式来确保我们匹配的是整个单词/短语 (\b 是单词边界)
        # re.escape() 会处理关键词中的特殊字符 (例如, 'state-of-the-art')
        try:
            count = len(re.findall(r'\b' + re.escape(keyword) + r'\b', all_text))
            if count > 0:
                keyword_counts[keyword] = count
        except re.error:
            # print(f"Warning: Invalid regex for keyword '{keyword}'. Skipping.")
            pass # 跳过无法构成有效正则表达式的词条
        
        # 打印进度
        if (i + 1) % 100 == 0:
            print(f"  已处理 {i + 1}/{total_keywords} 个关键词...")

    print("词频计算完成！")
    
    # 4. 创建DataFrame并保存到Excel
    # 从原始大小写列表中恢复大小写，以方便阅读
    with open(keywords_file, 'r', encoding='utf-8') as f:
        original_case_keywords = {k.lower(): k for k in (line.strip() for line in f if line.strip())}

    df_data = [{'Keyword': original_case_keywords.get(kw, kw), 'Frequency': freq} for kw, freq in keyword_counts.items()]
    
    df = pd.DataFrame(df_data)
    df_sorted = df.sort_values(by='Frequency', ascending=False)
    
    df_sorted.to_excel(output_excel_file, index=False)
    
    print(f"\n--- 统计完成 ---")
    print(f"结果已保存至: '{output_excel_file}'")

# --- 执行脚本 ---
if __name__ == "__main__":
    # 【请修改这里】指定包含你61个txt文件的文件夹路径
    corpus_directory = 'extracted_texts' 
    
    # 输入文件是上一轮精修后生成的包含7022个词条的文件
    keywords_to_count_file = 'keywords_polished_final.txt'
    
    # 输出的Excel文件名
    output_excel_filename = 'keywords_with_frequency.xlsx'
    
    if not os.path.isdir(corpus_directory):
        print(f"\n错误: 语料库文件夹 '{corpus_directory}' 不存在。")
        print("请创建一个名为 'corpus_txt' 的文件夹，并将你的61个.txt文件放入其中，然后重新运行脚本。")
    else:
        count_keyword_frequencies(corpus_directory, keywords_to_count_file, output_excel_filename)