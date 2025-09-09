import re

def final_polishing_filter(input_file, output_file):
    """
    对已经初步清洗的关键词列表进行最后一轮、更严格的精修。
    """
    print("--- 开始最终精修流程 (V4 - Polishing) ---")
    with open(input_file, 'r', encoding='utf-8') as f:
        keywords = sorted(list(set(line.strip() for line in f if line.strip())))
    
    original_count = len(keywords)
    print(f"1. 精修前关键词数量: {original_count}")

    # 定义一系列严格的过滤规则
    
    # 规则1: 定义不应出现在开头或结尾的常见虚词/助词
    # 注意每个词后面都有一个空格，以匹配单词边界
    bad_starts = ('a ', 'an ', 'the ', 'of ', 'in ', 'on ', 'for ', 'with ', 'to ', 'at ', 'by ', 'is ', 'are ', 'its ')
    bad_ends = (' a', ' the', ' of', ' and', ' or', ' for', ' in', ' on', ' is', ' are', ' to')

    # 规则2: 定义不应出现在术语中的常见动词和非技术性填充词
    forbidden_words = {
        'is', 'are', 'was', 'were', 'has', 'have', 'had', 'be', 'been', 'being', 
        'do', 'does', 'did', 'show', 'shows', 'shown', 'study', 'studied', 'studies',
        'investigate', 'investigated', 'analyze', 'analyzed', 'compare', 'compared',
        'report', 'reported', 'reports', 'use', 'used', 'using', 'based', 'due', 'via',
        'various', 'different', 'several', 'however', 'therefore', 'according'}

    polished_keywords = []
    for term in keywords:
        term_lower = term.lower()
        
        # 应用规则1: 检查开头和结尾
        if term_lower.startswith(bad_starts) or term_lower.endswith(bad_ends):
            continue
            
        # 应用规则2: 检查是否包含禁用的动词或填充词
        # 使用简单的split()来避免分词库的不一致性
        tokens = set(term_lower.split())
        if not tokens.isdisjoint(forbidden_words): # 如果两个集合有交集
            continue
            
        # 应用规则3: 移除过长的纯大写术语（通常是文章标题或摘要）
        if term.isupper() and len(term) > 12:
            continue
            
        # 应用规则4: 移除包含多个问号或奇怪标点的词条
        if '?' in term or '->' in term or '<' in term:
            continue

        polished_keywords.append(term)
        
    final_count = len(polished_keywords)
    print(f"2. 精修后关键词数量: {final_count} (移除: {original_count - final_count})")

    # --- 保存最终的精修列表 ---
    with open(output_file, 'w', encoding='utf-8') as f:
        for keyword in sorted(polished_keywords):
            f.write(keyword + '\n')
            
    print("\n--- 精修完成 ---")
    print(f"结果已保存至: '{output_file}'")


# --- 执行脚本 ---
if __name__ == "__main__":
    # 输入文件是你上一轮生成的包含7464个词条的文件
    input_filename = 'keywords_final_for_review.txt' 
    output_filename = 'keywords_polished_final.txt'
    
    try:
        final_polishing_filter(input_filename, output_filename)
        print(f"\n最终建议：将'{output_filename}'导入Excel或Google Sheets进行人工标准化。这个列表的质量现在非常高了。")
    except FileNotFoundError:
        print(f"\n错误: 输入文件 '{input_filename}' 未找到。请确保它和脚本在同一个目录下。")