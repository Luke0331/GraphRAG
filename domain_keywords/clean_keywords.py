import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# --- 确保NLTK数据包已下载 ---
try:
    stopwords.words('english')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("正在下载NLTK数据包，请稍候...")
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    print("NLTK数据包下载完成。")
# --------------------------------

def is_good_keyword_heuristic(term, stop_words):
    """
    使用启发式规则来验证关键词，完全不依赖POS Tagging。
    """
    # 统一处理，转换为小写
    term_lower = term.lower()
    tokens = word_tokenize(term_lower)
    
    # 规则 1: 边界词检查 - 好的术语通常不以停用词开始或结束
    if len(tokens) > 1:
        if tokens[0] in stop_words or tokens[-1] in stop_words:
            return False
            
    # 规则 2: 实质性检查 - 必须包含至少一个非停用词
    non_stop_word_count = sum(1 for word in tokens if word not in stop_words)
    if non_stop_word_count == 0:
        return False
        
    # 规则 3: 构成检查 - 字母字符应占多数
    alpha_chars = sum(c.isalpha() for c in term)
    total_chars = len(term)
    if total_chars > 0 and (alpha_chars / total_chars) < 0.7:
        return False
        
    # 规则 4: 噪音词检查 - 拒绝包含明确噪音词的术语
    noise_words = {'fig', 'figure', 'table', 'abstract', 'introduction', 'conclusion', 
                   'copyright', 'ieee', 'wiley', 'et al', 'chapter', 'results',
                   'supplementary', 'information'}
    if any(noise in tokens for noise in noise_words):
        return False

    return True

def final_filter_pipeline(input_file, output_file):
    """
    最终版的关键词过滤流程。
    """
    print("--- 开始自动化清洗流程 (V3 - Heuristic Filtering) ---")
    with open(input_file, 'r', encoding='utf-8') as f:
        # 在这里不转小写，保留原始大小写，方便人工审查
        keywords = sorted(list(set(line.strip() for line in f if line.strip())))
    
    original_count = len(keywords)
    print(f"1. 初始关键词数量 (去重后): {original_count}")

    # --- 过滤器1: 长度过滤 (基础) ---
    min_len, max_len = 3, 80
    keywords = [k for k in keywords if min_len <= len(k) <= max_len]
    count_after_len = len(keywords)
    print(f"2. 长度过滤后 ({min_len}-{max_len}字符): {count_after_len} (移除: {original_count - count_after_len})")

    # --- 过滤器2: 正则表达式过滤 (精细) ---
    filtered_keywords_re = []
    for k in keywords:
        # 拒绝纯数字、符号组合
        if re.fullmatch(r'[\d\s\.\-\(\)\[\]\%\/]+', k):
            continue
        # 拒绝包含多个连在一起的非字母数字字符 (比如 '---' 或 '...')
        if re.search(r'[^a-zA-Z0-9\s]{2,}', k):
            continue
        filtered_keywords_re.append(k)
    keywords = filtered_keywords_re
    count_after_re = len(keywords)
    print(f"3. Regex过滤后: {count_after_re} (移除: {count_after_len - count_after_re})")
    
    # --- 过滤器3: 启发式规则过滤 (核心) ---
    english_stop_words = set(stopwords.words('english'))
    final_keywords = [k for k in keywords if is_good_keyword_heuristic(k, english_stop_words)]
    final_count = len(final_keywords)
    print(f"4. 启发式规则过滤后: {final_count} (移除: {count_after_re - final_count})")

    # --- 保存清洗后的列表 ---
    with open(output_file, 'w', encoding='utf-8') as f:
        for keyword in sorted(final_keywords):
            f.write(keyword + '\n')
            
    print("\n--- 清洗完成 ---")
    print(f"总计移除: {original_count - final_count} 个词条")
    print(f"最终保留: {final_count} 个高质量候选词条")
    print(f"结果已保存至: '{output_file}'")

# --- 执行脚本 ---
if __name__ == "__main__":
    # 使用你上传的文件名
    input_filename = 'all_extracted_keywords.txt' 
    output_filename = 'keywords_final_for_review.txt'
    
    try:
        final_filter_pipeline(input_filename, output_filename)
        print("\n下一步：将'keywords_final_for_review.txt'导入Excel或Google Sheets进行最后的人工精修。")
    except FileNotFoundError:
        print(f"\n错误: 输入文件 '{input_filename}' 未找到。请确保它和脚本在同一个目录下。")