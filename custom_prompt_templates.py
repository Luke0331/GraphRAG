from langchain.prompts import PromptTemplate
from typing import Dict, Any

class SiliconBatteryPromptTemplates:
    """
    硅电池领域自定义提示模板
    """
    
    @staticmethod
    def get_qa_prompt() -> PromptTemplate:
        """获取问答提示模板"""
        return PromptTemplate(
            input_variables=["context", "question"],
            template="""
            你是一个硅电池领域的专业助手。请基于以下相关文档回答用户问题。
            
            重要约束：
            1. 仅限2020年后的文献信息
            2. 优先使用硅电池相关术语
            3. 如果信息不足，请明确说明
            
            相关文档:
            {context}
            
            用户问题: {question}
            
            请提供准确、详细的答案，并尽可能引用文档中的具体信息。
            如果文档信息不足或不是2020年后的内容，请说明。
            
            答案:
            """
        )
    
    @staticmethod
    def get_query_rewrite_prompt() -> PromptTemplate:
        """获取查询重写提示模板"""
        return PromptTemplate(
            input_variables=["original_query", "domain_terms"],
            template="""
            基于硅电池领域术语，重写用户查询以更好地匹配相关文档。
            
            领域术语: {domain_terms}
            原始查询: {original_query}
            
            请重写查询，使用更准确的硅电池领域术语，保持原意。
            只返回重写后的查询，不要其他内容。
            
            重写后的查询:
            """
        )
    
    @staticmethod
    def get_efficiency_analysis_prompt() -> PromptTemplate:
        """获取效率分析提示模板"""
        return PromptTemplate(
            input_variables=["context", "question"],
            template="""
            你是硅电池效率分析专家。请分析以下文档中关于电池效率的信息。
            
            约束条件：
            1. 仅分析2020年后的技术进展
            2. 重点关注效率提升方法
            3. 提供具体的数据和指标
            
            相关文档:
            {context}
            
            分析问题: {question}
            
            请提供结构化的效率分析，包括：
            1. 当前效率水平
            2. 提升方法
            3. 技术趋势
            4. 具体数据
            
            分析结果:
            """
        )
    
    @staticmethod
    def get_manufacturing_prompt() -> PromptTemplate:
        """获取制造工艺提示模板"""
        return PromptTemplate(
            input_variables=["context", "question"],
            template="""
            你是硅电池制造工艺专家。请分析以下文档中的制造工艺信息。
            
            约束条件：
            1. 仅关注2020年后的制造技术
            2. 重点关注工艺改进
            3. 提供具体的工艺参数
            
            相关文档:
            {context}
            
            工艺问题: {question}
            
            请提供详细的制造工艺分析，包括：
            1. 工艺流程
            2. 关键参数
            3. 工艺改进
            4. 质量控制
            
            工艺分析:
            """
        )
    
    @staticmethod
    def get_material_analysis_prompt() -> PromptTemplate:
        """获取材料分析提示模板"""
        return PromptTemplate(
            input_variables=["context", "question"],
            template="""
            你是硅电池材料专家。请分析以下文档中的材料相关信息。
            
            约束条件：
            1. 仅分析2020年后的材料技术
            2. 重点关注新材料应用
            3. 提供材料性能数据
            
            相关文档:
            {context}
            
            材料问题: {question}
            
            请提供详细的材料分析，包括：
            1. 材料类型
            2. 性能指标
            3. 应用效果
            4. 发展趋势
            
            材料分析:
            """
        )


class PromptTemplateManager:
    """
    提示模板管理器
    """
    
    def __init__(self):
        self.templates = {
            "qa": SiliconBatteryPromptTemplates.get_qa_prompt(),
            "query_rewrite": SiliconBatteryPromptTemplates.get_query_rewrite_prompt(),
            "efficiency_analysis": SiliconBatteryPromptTemplates.get_efficiency_analysis_prompt(),
            "manufacturing": SiliconBatteryPromptTemplates.get_manufacturing_prompt(),
            "material_analysis": SiliconBatteryPromptTemplates.get_material_analysis_prompt(),
            "fusion": PromptTemplate(
                input_variables=["question", "graph_context", "vector_context"],
                template="""
请根据以下提供的结构化知识图谱路径和相关的非结构化文献上下文，全面而深入地回答用户的问题。
在你的回答中，请优先利用知识图谱提供的因果和逻辑关系，并用文献上下文中的具体信息来丰富和支撑你的论点。

[用户问题]:
{question}

[知识图谱揭示的关系路径]:
{graph_context}

[相关文献上下文]:
{vector_context}

[你的回答]:
"""
            )
        }
        self.prompt_cache = {}
    
    def get_prompt(self, name: str) -> PromptTemplate:
        """
        获取指定类型的提示模板
        
        Args:
            prompt_type: 提示模板类型
            **kwargs: 额外参数
            
        Returns:
            提示模板
        """
        if name == "qa":
            return self.templates["qa"]
        elif name == "query_rewrite":
            return self.templates["query_rewrite"]
        elif name == "efficiency_analysis":
            return self.templates["efficiency_analysis"]
        elif name == "manufacturing":
            return self.templates["manufacturing"]
        elif name == "material_analysis":
            return self.templates["material_analysis"]
        elif name == "fusion":
            return self.templates["fusion"]
        else:
            raise ValueError(f"未知的提示模板类型: {name}")
    
    def create_custom_prompt(self, 
                           template: str, 
                           input_variables: list,
                           constraints: Dict[str, Any] = None) -> PromptTemplate:
        """
        创建自定义提示模板
        
        Args:
            template: 模板字符串
            input_variables: 输入变量列表
            constraints: 约束条件
            
        Returns:
            自定义提示模板
        """
        if constraints:
            # 添加约束条件到模板
            constraint_text = "\n".join([f"{k}: {v}" for k, v in constraints.items()])
            template = f"约束条件:\n{constraint_text}\n\n{template}"
        
        return PromptTemplate(
            input_variables=input_variables,
            template=template
        )


def test_prompt_templates():
    """测试提示模板"""
    manager = PromptTemplateManager()
    
    # 测试QA提示模板
    qa_prompt = manager.get_prompt("qa")
    print("QA提示模板:")
    print(qa_prompt.template)
    print("\n" + "="*50)
    
    # 测试效率分析提示模板
    efficiency_prompt = manager.get_prompt("efficiency_analysis")
    print("效率分析提示模板:")
    print(efficiency_prompt.template)
    print("\n" + "="*50)
    
    # 测试自定义提示模板
    custom_prompt = manager.create_custom_prompt(
        template="基于以下信息回答问题：\n{context}\n\n问题：{question}\n\n答案：",
        input_variables=["context", "question"],
        constraints={
            "时间限制": "仅限2020年后信息",
            "领域限制": "硅电池相关",
            "数据要求": "提供具体数据"
        }
    )
    print("自定义提示模板:")
    print(custom_prompt.template)


if __name__ == "__main__":
    test_prompt_templates() 