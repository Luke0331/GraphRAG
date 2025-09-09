# 硅电池领域词典提取Pipeline

基于61篇硅电池相关文献，自动提取和构建专业领域词典的完整pipeline。

## 🎯 项目目标

从大量学术文献中自动提取硅电池领域的专业术语，构建结构化的领域词典，包含：
- 标准术语名称
- 同义词和别名
- 术语分类
- 可扩展的JSON格式

## 📋 Pipeline流程

### 第1步：预处理 - PDF转纯文本
- **输入**: 61篇PDF文献（位于`../zotero/`目录）
- **工具**: PyMuPDF (fitz)
- **输出**: `extracted_texts/`目录下的61个.txt文件
- **功能**: 高质量文本提取，支持双栏格式

### 第2步：自动提取 - 关键词挖掘
- **输入**: 提取的纯文本文件
- **工具**: KeyBERT + Sentence Transformers
- **输出**: `extracted_keywords.txt`
- **功能**: 自动识别技术术语和短语

### 第3步：人工精修 - 电子表格整理
- **输入**: 自动提取的关键词
- **工具**: Excel/Google Sheets
- **输出**: `curated_dictionary.xlsx`
- **功能**: 人工筛选、标准化和分类

### 第4步：自动生成 - JSON词典
- **输入**: 人工整理的Excel表格
- **工具**: Python pandas + json
- **输出**: `domain_dictionary.json`
- **功能**: 生成结构化词典文件

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements_dictionary.txt
```

### 2. 一键运行Pipeline

```bash
python run_pipeline.py
```

这将自动执行前3步，然后提示你进行人工整理。

### 3. 人工整理

1. 打开生成的 `curated_dictionary.xlsx`
2. 按照 `curation_instructions.md` 的说明进行整理
3. 保存Excel文件

### 4. 生成最终词典

```bash
python step4_generate_json.py
```

## 📁 文件结构

```
RAG-pR-main/
├── run_pipeline.py              # 主控制脚本
├── step1_pdf_to_text.py         # 第1步：PDF转文本
├── step2_keyword_extraction.py  # 第2步：关键词提取
├── step3_create_spreadsheet.py  # 第3步：创建电子表格
├── step4_generate_json.py       # 第4步：生成JSON
├── requirements_dictionary.txt   # 依赖包列表
├── curation_instructions.md     # 人工整理指南
├── README_dictionary.md         # 本文件
├── extracted_texts/             # 提取的文本文件（自动生成）
├── extracted_keywords.txt       # 提取的关键词（自动生成）
├── curated_dictionary.xlsx      # 人工整理的表格
├── domain_dictionary.json       # 最终词典（自动生成）
└── dictionary_report.md         # 词典报告（自动生成）
```

## 📊 输出格式

### JSON词典格式

```json
[
  {
    "standard_name": "PERC Cell",
    "aliases": ["perc", "perc cell", "perc solar cell", "passivated emitter and rear cell"],
    "category": "Technology"
  },
  {
    "standard_name": "Silicon Heterojunction Solar Cell",
    "aliases": ["heterojunction solar cell", "silicon heterojunction", "hjt"],
    "category": "Technology"
  }
]
```

### 分类体系

- **Technology**: 技术类型（PERC, HJT, TOPCon等）
- **Material**: 材料（Silicon, ITO, TCO等）
- **Process**: 工艺过程（Texturing, Doping, Metallization等）
- **Metric**: 性能指标（Efficiency, Fill Factor, Open Circuit Voltage等）
- **Device**: 器件结构（Solar Cell, Module, Wafer等）
- **Property**: 物理性质（Band Gap, Absorption, Reflection等）

## 🔧 技术细节

### 关键词提取算法

- **模型**: all-MiniLM-L6-v2 (适合科学文献)
- **方法**: KeyBERT + MMR (Maximal Marginal Relevance)
- **参数**: 
  - ngram_range: (1, 3) - 支持1-3个词的短语
  - top_n: 30 - 每篇文献提取30个关键词
  - diversity: 0.7 - 确保关键词多样性

### 文本预处理

- 移除特殊字符，保留基本标点
- 统一空白字符处理
- 过滤过短或过长的关键词
- 转换为小写标准化

## 📈 预期效果

基于61篇高质量文献，预期提取：
- **候选关键词**: 500-1000个
- **最终术语**: 200-400个
- **平均别名数**: 3-5个
- **覆盖领域**: 硅电池技术、材料、工艺、性能等

## 🛠️ 自定义配置

### 调整关键词提取参数

编辑 `step2_keyword_extraction.py`:

```python
# 调整提取参数
keywords = kw_model.extract_keywords(
    text,
    keyphrase_ngram_range=(1, 4),  # 支持1-4个词
    top_n=50,                      # 每篇提取50个
    diversity=0.8                  # 更高多样性
)
```

### 修改分类体系

编辑 `step3_create_spreadsheet.py` 中的分类说明。

## 🐛 故障排除

### 常见问题

1. **PDF提取失败**
   - 检查PDF文件是否损坏
   - 确保PyMuPDF版本兼容

2. **关键词提取慢**
   - 首次运行需要下载模型
   - 考虑使用GPU加速

3. **Excel文件无法读取**
   - 确保安装了openpyxl
   - 检查文件是否被其他程序占用

### 日志查看

所有脚本都会输出详细的日志信息，包括：
- 处理进度
- 错误信息
- 统计结果

## 📞 支持

如有问题，请检查：
1. 依赖包是否正确安装
2. 文件路径是否正确
3. 日志输出中的错误信息

## 📄 许可证

本项目遵循MIT许可证。 