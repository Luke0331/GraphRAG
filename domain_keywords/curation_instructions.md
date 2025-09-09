
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
