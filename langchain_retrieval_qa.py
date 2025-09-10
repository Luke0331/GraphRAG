import json
import os
import re
import time
from typing import List, Dict, Any
from collections import defaultdict

# LangChain imports
from langchain_community.retrievers import LlamaIndexRetriever
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
# from langchain_community.llms import ZhipuAI
from llama_index.llms.zhipuai import ZhipuAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document as LCDocument
from langchain_core.retrievers import BaseRetriever

# LlamaIndex imports (for compatibility)
from llama_index.core import (
    Document, 
    VectorStoreIndex, 
    StorageContext,
    Settings
)
from llama_index.vector_stores.deeplake import DeepLakeVectorStore # Make sure this is imported
from llama_index.llms.zhipuai import ZhipuAI as LlamaIndexZhipuAI

# Custom embedding solution
from embedding_solution import create_embedding_model
from langchain_core.language_models.llms import LLM
from pydantic import PrivateAttr

from custom_prompt_templates import PromptTemplateManager

class LangChainZhipuAI(LLM):
    _llm: ZhipuAI = PrivateAttr()

    def __init__(self, api_key, model, **kwargs):
        super().__init__(**kwargs)
        self._llm = ZhipuAI(api_key=api_key, model=model)

    def _call(self, prompt, stop=None, run_manager=None, **kwargs):
        return self._llm.complete(prompt).text

    @property
    def _llm_type(self):
        return "zhipuai"

class LlamaIndexRetrieverAdapter(BaseRetriever):
    _llama_retriever: any = PrivateAttr()

    def __init__(self, llama_retriever, **kwargs):
        super().__init__(**kwargs)
        self._llama_retriever = llama_retriever

    def _get_relevant_documents(self, query, *, run_manager=None, **kwargs):
        # 获取k参数，默认为5
        k = kwargs.get('k', 5)
        # 设置检索器的top_k
        self._llama_retriever.similarity_top_k = k
        nodes = self._llama_retriever.retrieve(query)
        return [LCDocument(page_content=node.text, metadata=node.metadata) for node in nodes]

    def _call(self, query, **kwargs):
        return self._get_relevant_documents(query, **kwargs)

class LangChainDomainRAG:
    """
    基于LangChain RetrievalQA的领域感知RAG系统
    """
    
    def __init__(self, 
                 domain_dict_path: str = "domain_keywords/domain_dictionary.json",
                 vector_store_path: str = "RAG-Wikipedia/dataset/vector_storage_new",
                 llm_api_key: str = "41b29e65745d4110a018c5d616b0012f.A6CEwmornnYXSVLC",
                 llm_model: str = "glm-4-flash"):
        """
        初始化LangChain领域感知RAG系统
        
        Args:
            domain_dict_path: 领域词典文件路径
            vector_store_path: 向量存储路径
            llm_api_key: LLM API密钥
            llm_model: LLM模型名称
        """
        self.domain_dict_path = domain_dict_path
        self.vector_store_path = vector_store_path
        
        # prompt templates
        self.prompt_manager = PromptTemplateManager()

        # 加载领域词典
        self.domain_dictionary = self._load_domain_dictionary()

        # 设置本地 embedding 模型
        model_path = r"F:\Intern\EDF\EmbeddingModels\Qwen3-Embedding-0.6B"
        embed_model = create_embedding_model(model_path, fallback=True)
        Settings.embed_model = embed_model
        
        # 初始化LlamaIndex索引（用于LangChain检索器）
        self._initialize_llama_index()
        
        # 初始化LangChain组件
        self._initialize_langchain_components(llm_api_key, llm_model)
        
        print(f"✓ 领域词典加载完成，共 {len(self.domain_dictionary)} 个术语")
        print(f"✓ LangChain RetrievalQA初始化完成")
        print(f"✓ 向量存储路径: {vector_store_path}")
    
    def _load_domain_dictionary(self) -> Dict[str, Dict[str, Any]]:
        """加载领域词典"""
        try:
            with open(self.domain_dict_path, 'r', encoding='utf-8') as f:
                domain_dict = json.load(f)
            
            processed_dict = {}
            for item in domain_dict:
                standard_name = item.get('standard_name', '')
                aliases = item.get('aliases', [])
                category = item.get('category', '')
                
                processed_dict[standard_name] = {
                    'aliases': aliases,
                    'category': category,
                    'all_terms': [standard_name] + aliases
                }
            
            return processed_dict
            
        except Exception as e:
            print(f"✗ 加载领域词典失败: {e}")
            return {}
    
    def _initialize_llama_index(self):
        """初始化LlamaIndex索引"""
        try:
            # Reverted to the correct DeepLake loading logic
            print(f"💾 Loading DeepLake index from path: {self.vector_store_path}")
            vector_store = DeepLakeVectorStore(dataset_path=self.vector_store_path, read_only=True)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            self.index = VectorStoreIndex.from_vector_store(
                vector_store,
                storage_context=storage_context
            )
            self.retriever = self.index.as_retriever()
            
            print("✓ LlamaIndex索引初始化完成")
            
        except Exception as e:
            print(f"✗ LlamaIndex索引初始化失败: {e}")
            self.index = None
            self.retriever = None
    
    def _initialize_langchain_components(self, llm_api_key: str, llm_model: str):
        """初始化LangChain组件"""
        try:
            if self.retriever is None:
                raise RuntimeError("LlamaIndex retriever 初始化失败，无法创建 LangChain 检索器。")
        
            # 创建LangChain检索器 
            self.lc_retriever = LlamaIndexRetrieverAdapter(self.retriever)

            # 创建LangChain LLM
            self.llm = LangChainZhipuAI(
                api_key=llm_api_key,
                model=llm_model
            )
            
            # 创建自定义提示模板
            # self.prompt_template = PromptTemplate(
            #     input_variables=["context", "question"],
            #     template="""
            #     你是一个硅电池领域的专家助手。基于以下相关文档，回答用户的问题。
                
            #     相关文档:
            #     {context}
                
            #     用户问题: {question}
                
            #     请提供准确、详细的答案，引用文档中的具体信息，并提供引用文档的详细信息。
                
            #     答案:
            #     """
            # )
            self.prompt_template = self.prompt_manager.get_prompt("qa")
            
            # 创建RetrievalQA链
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.lc_retriever,
                chain_type_kwargs={
                    "prompt": self.prompt_template
                },
                verbose=True
            )
            
            print("✓ LangChain组件初始化完成")
            
        except Exception as e:
            print(f"✗ LangChain组件初始化失败: {e}")
    
    def _extract_domain_terms(self, query: str) -> List[str]:
        """从查询中提取领域术语"""
        extracted_terms = []
        query_lower = query.lower()
        
        for standard_name, term_info in self.domain_dictionary.items():
            all_terms = term_info['all_terms']
            
            for term in all_terms:
                if term.lower() in query_lower:
                    extracted_terms.append(standard_name)
                    break
        
        return list(set(extracted_terms))
    
    def _expand_query_with_domain_terms(self, query: str) -> str:
        """使用领域术语扩展查询"""
        extracted_terms = self._extract_domain_terms(query)
        
        if not extracted_terms:
            return query
        
        expansion_parts = []
        for term in extracted_terms:
            term_info = self.domain_dictionary[term]
            aliases = term_info['aliases']
            expansion_parts.extend(aliases)
        
        expansion_terms = list(set(expansion_parts))[:10]
        
        if expansion_terms:
            expanded_query = f"{query} {' '.join(expansion_terms)}"
            print(f"🔍 查询扩展: {query} -> {expanded_query}")
            return expanded_query
        
        return query

    def _rewrite_query_with_domain_context(self, query: str) -> str:
        """
        使用领域上下文重写查询
        Args:
            query: 原始查询
        Returns:
            重写后的查询
        """
        extracted_terms = self._extract_domain_terms(query)
        if not extracted_terms:
            return query
        # 构建领域上下文提示
        context_parts = []
        for term in extracted_terms[:3]:  # 最多3个主要术语
            term_info = self.domain_dictionary[term]
            category = term_info['category']
            context_parts.append(f"{term} ({category})")
        context = ", ".join(context_parts)
        # 使用LLM重写查询
        prompt = f"""
        基于以下硅电池领域术语的上下文，重写用户查询以更好地匹配相关文档：
        领域上下文: {context}
        用户查询: {query}
        请重写查询，保持原意但使用更准确的领域术语。只返回重写后的查询，不要其他内容。
        """
        try:
            rewritten_query = self.llm._call(prompt).strip()
            print(f"🔄 查询重写: {query} -> {rewritten_query}")
            return rewritten_query
        except Exception as e:
            print(f"⚠️ 查询重写失败: {e}")
            return query

    def _translate_to_english(self, text: str) -> str:
        """
        用 LLM 把输入翻译成英文
        """
        prompt = f"Translate the following text to English. Only return the translation:\n{text}"
        try:
            if hasattr(self.llm, "complete"):
                return self.llm.complete(prompt).text.strip()
            else:
                return self.llm._call(prompt).strip()
        except Exception as e:
            print(f"⚠️ 英文翻译失败: {e}")
            return text

    def _translate_to_chinese(self, text: str) -> str:
        """
        用 LLM 把输入翻译成中文
        """
        prompt = f"请将下列内容翻译成中文，只返回翻译结果：\n{text}"
        try:
            if hasattr(self.llm, "complete"):
                return self.llm.complete(prompt).text.strip()
            else:
                return self.llm._call(prompt).strip()
        except Exception as e:
            print(f"⚠️ 中文翻译失败: {e}")
            return text


    def _extract_structured_query(self, user_query: str) -> Dict[str, str]:
        prompt = f"""
        请从下列问题中提取出材料名称、指标和关系，返回JSON格式，如：{{"material": "...", "metric": "...", "relation": "..."}}
        问题：{user_query}
        只返回JSON，不要其他内容。
        """
        try:
            result = self.llm._call(prompt)
            print(f"LLM结构化抽取原始返回: {result!r}")
            # 尝试用正则提取JSON
            match = re.search(r'\{.*\}', result, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            else:
                print("未找到JSON结构，返回空")
                return {}
        except Exception as e:
            print(f"结构化抽取失败: {e}")
            return {}

    def _structured_retrieve(self, structured_query: Dict[str, str], top_k: int = 5) -> str:
        """
        用结构化条件检索文档。这里简单用AND关键词过滤，也可扩展为多字段检索。
        返回拼接后的文档内容字符串。
        """
        # 这里只是示例，实际应用可用更复杂的检索逻辑
        keywords = [v for v in structured_query.values() if v]
        # 假设有 self.index/documents，或用 retriever 支持多关键词
        # 这里直接用原有检索器做一次多关键词拼接
        query = " ".join(keywords)
        docs = self.lc_retriever._get_relevant_documents(query)
        context = "\n".join([doc.page_content for doc in docs[:top_k]])
        return context

    # def query(self, 
    #           user_query: str, 
    #           use_query_expansion: bool = True,
    #           use_query_rewriting: bool = True,
    #           top_k: int = 5,
    #           prompt_type: str = "qa",
    #           use_structured_query: bool = False) -> Dict[str, Any]:
    #     """
    #     执行LangChain RetrievalQA查询
    #     Args:
    #         user_query: 用户查询
    #         use_query_expansion: 是否使用查询扩展
    #         use_query_rewriting: 是否使用查询重写
    #         top_k: 检索文档数量
    #         prompt_type：问答类型
    #         use_structured_query: 是否使用结构化检索
    #     Returns:
    #         查询结果字典
    #     """
    #     start_time = time.time()
    #     print(f"\n🔍 处理查询: {user_query}")
    #     print("="*50)

    #     # 0. 翻译
    #     user_query = self._translate_to_english(user_query)
    #     print(f"🌐 翻译后英文查询: {user_query}")

    #     # 1. 提取领域术语
    #     domain_terms = self._extract_domain_terms(user_query)
    #     print(f"📚 提取的领域术语: {domain_terms}")

    #     # 2. 查询扩展
    #     expanded_query = user_query
    #     if use_query_expansion and domain_terms:
    #         expanded_query = self._expand_query_with_domain_terms(user_query)
        
    #     # 3. 查询重写
    #     final_query = expanded_query
    #     if use_query_rewriting and domain_terms:
    #         final_query = self._rewrite_query_with_domain_context(expanded_query)
        
    #     # 4. 使用LangChain RetrievalQA执行查询
    #     print(f"🔍 使用LangChain RetrievalQA查询: {final_query}")
    #     try:
    #         # 1. 动态获取模板
    #         self.prompt_template = self.prompt_manager.get_prompt(prompt_type)

    #         # 2. 动态重建 RetrievalQA 链（可选优化：只在prompt_type变化时重建）
    #         self.qa_chain = RetrievalQA.from_chain_type(
    #             llm=self.llm,
    #             chain_type="stuff",
    #             retriever=self.lc_retriever,
    #             chain_type_kwargs={
    #                 "prompt": self.prompt_template
    #             },
    #             verbose=True
    #         )

    #         # 3. 执行查询
    #         response = self.qa_chain.invoke({"query": final_query})
    #         answer = response.get("result", "抱歉，没有找到相关答案。")
    #     except Exception as e:
    #         print(f"✗ LangChain查询失败: {e}")
    #         answer = "抱歉，查询过程中出现错误。"

    #     # 5. 答案翻译回中文
    #     answer_zh = self._translate_to_chinese(answer)
        
    #     # 计算执行时间
    #     execution_time = time.time() - start_time
        
    #     # 构建结果
    #     result = {
    #         'original_query': user_query,
    #         'expanded_query': expanded_query,
    #         'final_query': final_query,
    #         'domain_terms': domain_terms,
    #         'answer': answer_zh,
    #         'execution_time': execution_time,
    #         'framework': 'LangChain RetrievalQA'
    #     }
    #     print(f"⏱️ 执行时间: {execution_time:.2f}秒")
    #     print("="*50)
    #     return result
    
    def query(self, 
            user_query: str, 
            use_query_expansion: bool = True,
            use_query_rewriting: bool = True,
            top_k: int = 5,
            prompt_type: str = "qa",
            use_structured_query: bool = False) -> Dict[str, Any]:
        start_time = time.time()
        print(f"\n🔍 处理查询: {user_query}")
        print("="*50)

        # 0. 翻译
        user_query_en = self._translate_to_english(user_query)
        print(f"🌐 翻译后英文查询: {user_query_en}")

        if use_structured_query:
            # 结构化检索分支
            structured_query = self._extract_structured_query(user_query_en)
            print(f"🧩 结构化检索条件: {structured_query}")
            context = self._structured_retrieve(structured_query, top_k=top_k)
            self.prompt_template = self.prompt_manager.get_prompt(prompt_type)
            prompt_text = self.prompt_template.format(context=context, question=user_query_en)
            try:
                answer = self.llm._call(prompt_text)
            except Exception as e:
                print(f"✗ 结构化检索失败: {e}")
                answer = "抱歉，结构化检索过程中出现错误。"
            expanded_query = json.dumps(structured_query, ensure_ascii=False)
            final_query = expanded_query
            domain_terms = list(structured_query.values())
        else:
            # 原有自然语言检索分支
            domain_terms = self._extract_domain_terms(user_query_en)
            print(f"📚 提取的领域术语: {domain_terms}")

            expanded_query = user_query_en
            if use_query_expansion and domain_terms:
                expanded_query = self._expand_query_with_domain_terms(user_query_en)
            
            final_query = expanded_query
            if use_query_rewriting and domain_terms:
                final_query = self._rewrite_query_with_domain_context(expanded_query)
            
            print(f"🔍 使用LangChain RetrievalQA查询: {final_query}")
            try:
                self.prompt_template = self.prompt_manager.get_prompt(prompt_type)
                self.qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=self.lc_retriever,
                    chain_type_kwargs={
                        "prompt": self.prompt_template
                    },
                    verbose=True
                )
                response = self.qa_chain.invoke({"query": final_query})
                answer = response.get("result", "抱歉，没有找到相关答案。")
                
                # 获取检索到的文档信息，使用指定的top_k
                retrieved_docs = self.lc_retriever.get_relevant_documents(final_query, k=top_k)
                source_docs = []
                for i, doc in enumerate(retrieved_docs):
                    source_info = {
                        "index": i + 1,
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    source_docs.append(source_info)
                
            except Exception as e:
                print(f"✗ LangChain查询失败: {e}")
                answer = "抱歉，查询过程中出现错误。"
                source_docs = []

        # 5. 答案翻译回中文
        answer_zh = self._translate_to_chinese(answer)
        execution_time = time.time() - start_time

        result = {
            'original_query': user_query_en,
            'expanded_query': expanded_query,
            'final_query': final_query,
            'domain_terms': domain_terms,
            'answer': answer_zh,
            'source_documents': source_docs,
            'retrieved_docs_count': len(source_docs),
            'execution_time': execution_time,
            'framework': 'LangChain RetrievalQA'
        }
        print(f"⏱️ 执行时间: {execution_time:.2f}秒")
        print("="*50)
        return result


    def get_domain_statistics(self) -> Dict[str, Any]:
        """获取领域词典统计信息"""
        if not self.domain_dictionary:
            return {}
        
        categories = defaultdict(int)
        total_terms = len(self.domain_dictionary)
        total_aliases = 0
        
        for term_info in self.domain_dictionary.values():
            category = term_info['category']
            categories[category] += 1
            total_aliases += len(term_info['aliases'])
        
        return {
            'total_terms': total_terms,
            'total_aliases': total_aliases,
            'categories': dict(categories),
            'avg_aliases_per_term': total_aliases / total_terms if total_terms > 0 else 0
        }


def main():
    """主函数 - 演示LangChain RetrievalQA系统"""
    # 初始化系统
    rag_system = LangChainDomainRAG()
    
    # 显示领域词典统计
    stats = rag_system.get_domain_statistics()
    print("\n📊 领域词典统计:")
    print(f"总术语数: {stats['total_terms']}")
    print(f"总同义词数: {stats['total_aliases']}")
    print(f"平均同义词数: {stats['avg_aliases_per_term']:.2f}")
    
    # 示例查询
    test_queries = [
        "How to improve the efficiency of solar cells?",
        "What are the manufacturing processes of silicon cells?",
        "What is anti-reflective coating?",
        "How to measure the conversion efficiency of solar cells?"
    ]
    
    print("\n" + "="*60)
    print("🧪 测试LangChain RetrievalQA系统")
    print("="*60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n测试 {i}: {query}")
        result = rag_system.query(query)
        
        print(f"\n📝 答案:")
        print(result['answer'])
        print(f"\n🔧 框架: {result['framework']}")


if __name__ == "__main__":
    main() 