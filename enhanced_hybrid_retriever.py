# -*- coding: utf-8 -*-
"""
增强混合检索器：整合LangChain RetrievalQA和图谱检索
基于现有的langchain_retrieval_qa.py系统
"""

import json
import re
import time
from typing import List, Dict, Any, Optional
from langchain_retrieval_qa import LangChainDomainRAG
from graph_retriever import GraphRetriever
from entity_linker import EntityLinker
from query_parser import QueryParser # I am adding this import

class EnhancedHybridRetriever:
    """
    增强混合检索器：整合LangChain RetrievalQA和图谱检索
    """
    
    def __init__(self, 
                 domain_dict_path: str = "domain_keywords/domain_dictionary_cleaned.json",
                 vector_store_path: str = "RAG-Wikipedia/dataset/vector_storage_with_metadata",
                 neo4j_uri: str = "bolt://localhost:7687",
                 neo4j_user: str = "neo4j",
                 neo4j_password: str = "domain-kg123",
                 llm_api_key: str = "41b29e65745d4110a018c5d616b0012f.A6CEwmornnYXSVLC",
                 llm_model: str = "glm-4-flash"):
        
        # 初始化LangChain RAG系统
        self.langchain_rag = LangChainDomainRAG(
            domain_dict_path=domain_dict_path,
            vector_store_path=vector_store_path,
            llm_api_key=llm_api_key,
            llm_model=llm_model
        )
        
        # 初始化图谱检索器
        self.graph_retriever = GraphRetriever(neo4j_uri, neo4j_user, neo4j_password)
        
        # 初始化实体链接器
        self.entity_linker = EntityLinker(domain_dict_path)

        # 初始化Query Parser
        self.query_parser = QueryParser(entity_linker=self.entity_linker, api_key=llm_api_key)
        
        print("✓ 增强混合检索器初始化完成")
    
    def _is_english(self, text: str) -> bool:
        """
        简单判断文本是否为英文
        """
        # 检查是否包含中文字符
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
        return len(chinese_chars) == 0 and len(text.strip()) > 0
    
    def graph_guided_retrieval(self, query: str, top_k: int = 3, graph_limit: int = 15) -> Dict[str, Any]:
        """
        Orchestrates the 'Parse -> Graph Search -> Refine -> Generate' pipeline.
        """
        print(f"\n🚀 执行图谱引导的检索: {query}")
        print("="*50)
        start_time = time.time()

        # 0. 翻译查询为英文（确保实体匹配）
        print("0. 准备英文查询...")
        # 如果查询已经是英文，就不需要翻译
        if self._is_english(query):
            english_query = query
            print(f"   - 查询已经是英文: {english_query}")
        else:
            english_query = self.langchain_rag._translate_to_english(query)
            print(f"   - 翻译后的英文查询: {english_query}")

        # 1. 问题解析 (Parse)
        print("1. 解析查询...")
        parsed_query = self.query_parser.parse_query(english_query)
        if parsed_query.get("error"):
            print(f"✗ 查询解析失败: {parsed_query.get('error')}")
            return {"error": "Query parsing failed", "details": parsed_query}
        print(f"   - 意图: {parsed_query.get('intent')}")
        print(f"   - 实体: {parsed_query.get('source_entities')}")

        # 2. 图检索 (Graph Search)
        print("2. 执行图检索...")
        graph_results = self.graph_retriever.search_from_parsed_query(parsed_query, limit=graph_limit)
        if graph_results.get("error") or not graph_results.get("results"):
            print("   - 结构化图检索未找到结果，尝试自然语言图检索...")
            # 尝试自然语言图检索作为备用
            fallback_graph_results = self.graph_retriever.query(english_query)
            if fallback_graph_results.get("results"):
                print(f"   - 自然语言图检索找到 {len(fallback_graph_results['results'])} 个结果")
                graph_results = fallback_graph_results
            else:
                print("   - 所有图检索方法都未找到结果，将回退到标准向量检索。")
                # Fallback to standard vector search if graph search fails
                return self.langchain_vector_search(query, top_k=top_k)
        
        print(f"   - Cypher 查询: {graph_results.get('cypher_query', '').strip()}")
        print(f"   - 找到 {graph_results.get('count')} 个图谱结果。")

        # 3. 向量精排 (Vector Refinement via Query Generation)
        # We refine the vector search by giving it a much more specific query
        # synthesized from the graph results.
        print("3. 基于图谱结果生成精确查询以进行精排...")
        refined_query = self._synthesize_query_from_graph(query, graph_results)
        print(f"   - 精炼后的查询: \"{refined_query}\"")

        # 4. LLM生成 (LLM Generation)
        # Use the refined query to get the final answer from the LLM.
        print("4. 使用精确查询调用LLM生成最终答案...")
        final_result = self.langchain_vector_search(
            refined_query, 
            top_k=top_k, 
            use_query_expansion=False, # Expansion is not needed for our specific query
            use_query_rewriting=False
        )

        execution_time = time.time() - start_time
        print(f"⏱️ 图谱引导检索总执行时间: {execution_time:.2f}s")
        print("="*50)

        # Combine all information for a comprehensive output
        return {
            "pipeline": "graph_guided_retrieval",
            "original_query": query,
            "english_query": english_query,
            "parsed_query": parsed_query,
            "graph_search_results": graph_results,
            "refined_query_for_llm": refined_query,
            "final_answer_and_sources": final_result,
            "execution_time": execution_time
        }

    def _synthesize_query_from_graph(self, original_query: str, graph_results: Dict[str, Any]) -> str:
        """
        Creates a new, more specific query for the LLM based on graph findings.
        """
        graph_findings = []
        for res in graph_results.get("results", []):
            # Handles 'find_relation' results
            if "source" in res and "target" in res:
                finding = f"{res['source']} -> {res['relationship']} -> {res['target']}"
            # Handles results from our new constrained search
            elif "found_entity" in res and "related_to" in res:
                entity = res['found_entity']
                entity_type = res.get('entity_type', ['Unknown'])[0] # Safely get type
                relation = res['relationship']
                related_to = res['related_to']
                finding = f"Found a '{entity_type}' named '{entity}', which '{relation}' a metric related to '{related_to}'."
            # Handles 'find_entity' results
            elif "found_entity" in res:
                finding = f"Found entity '{res['found_entity']}' related via '{res['relationship']}'"
            else:
                continue
            graph_findings.append(finding)

        if not graph_findings:
            findings_str = "No specific facts were found in the knowledge graph for this query."
        else:
            findings_str = "\n".join([f"- {f}" for f in graph_findings])


        # This prompt asks the LLM to synthesize an answer using the graph data as a hard context.
        refined_query = f"""
        Original question: "{original_query}"

        From a knowledge graph, I have retrieved the following verified facts:
        {findings_str}

        Based *only* on these facts and the documents related to them, please provide a comprehensive answer to the original question. Summarize the findings and explain how these facts answer the question.
        """
        return refined_query.strip()

    def langchain_vector_search(self, query: str, 
                              use_query_expansion: bool = True, 
                              use_query_rewriting: bool = True,
                              top_k: int = 5,
                              prompt_type: str = "qa",
                              use_structured_query: bool = False) -> Dict[str, Any]:
        """
        使用LangChain进行向量检索
        """
        try:
            # 调用LangChain RAG系统的query方法
            result = self.langchain_rag.query(
                query,
                use_query_expansion=use_query_expansion,
                use_query_rewriting=use_query_rewriting,
                top_k=top_k,
                prompt_type=prompt_type,
                use_structured_query=use_structured_query
            )
            
            return {
                "method": "langchain_vector_search",
                "query": query,
                "answer": result.get('answer', ''),
                "original_query": result.get('original_query', ''),
                "expanded_query": result.get('expanded_query', ''),
                "final_query": result.get('final_query', ''),
                "domain_terms": result.get('domain_terms', []),
                "execution_time": result.get('execution_time', 0),
                "framework": result.get('framework', 'LangChain RetrievalQA'),
                "top_k_used": top_k,
                "prompt_type_used": prompt_type,
                "use_structured_query": use_structured_query,
                "source_documents": result.get('source_documents', []),
                "retrieved_docs_count": result.get('retrieved_docs_count', 0)
            }
        except Exception as e:
            print(f"✗ LangChain向量检索失败: {e}")
            return {
                "method": "langchain_vector_search",
                "query": query,
                "answer": "检索失败",
                "error": str(e),
                "execution_time": 0,
                "framework": "LangChain RetrievalQA",
                "top_k_used": top_k,
                "prompt_type_used": prompt_type,
                "use_structured_query": use_structured_query,
                "source_documents": [],
                "retrieved_docs_count": 0
            }
    
    def graph_knowledge_search(self, query: str, limit: int = None) -> Dict[str, Any]:
        """
        使用图谱进行知识检索（使用结构化查询解析）
        """
        try:
            # 使用与graph_guided_retrieval相同的逻辑
            print(f"🕸️ 执行结构化图谱检索: {query}")
            
            # 翻译查询为英文（确保实体匹配）
            if self._is_english(query):
                english_query = query
                print(f"   - 查询已经是英文: {english_query}")
            else:
                english_query = self.langchain_rag._translate_to_english(query)
                print(f"   - 翻译后的英文查询: {english_query}")

            # 解析查询
            print("   - 解析查询...")
            parsed_query = self.query_parser.parse_query(english_query)
            if parsed_query.get("error"):
                print(f"   - 查询解析失败: {parsed_query.get('error')}")
                # 回退到旧方法
                graph_results = self.graph_retriever.query(english_query, limit=limit)
            else:
                print(f"   - 解析成功，意图: {parsed_query.get('intent')}")
                # 使用结构化查询
                graph_results = self.graph_retriever.search_from_parsed_query(parsed_query, limit=limit if limit is not None else 15)
            
            # 提取查询中的实体（用于兼容性）
            entities = self.entity_linker.extract_entities_from_text(query)
            
            return {
                "method": "structured_graph_retrieval",
                "query": query,
                "english_query": english_query,
                "parsed_query": parsed_query,
                "extracted_entities": entities,
                "graph_results": graph_results,
                "cypher_query": graph_results.get('cypher_query', ''),
                "entities_found": len(graph_results.get('results', [])),
                "relationships": graph_results.get('results', []),
                "limit_used": limit if limit is not None else 15
            }
        except Exception as e:
            print(f"✗ 结构化图谱检索失败: {e}")
            # 回退到旧方法
            try:
                entities = self.entity_linker.extract_entities_from_text(query)
                graph_results = self.graph_retriever.query(query, limit=limit)
                return {
                    "method": "fallback_graph_retrieval",
                    "query": query,
                    "extracted_entities": entities,
                    "graph_results": graph_results,
                    "cypher_query": graph_results.get('cypher_query', ''),
                    "entities_found": graph_results.get('count', 0),
                    "relationships": graph_results.get('results', []),
                    "limit_used": graph_results.get('limit_used', limit if limit is not None else 50)
                }
            except Exception as fallback_error:
                print(f"✗ 回退图谱检索也失败: {fallback_error}")
                return {
                    "method": "graph_retrieval",
                    "query": query,
                    "error": str(e),
                    "entities_found": 0,
                    "relationships": [],
                    "limit_used": limit if limit is not None else 50
                }
    
    def hybrid_search(self, query: str, 
                     vector_weight: float = 0.6,
                     graph_weight: float = 0.4,
                     use_query_expansion: bool = True,
                     use_query_rewriting: bool = True,
                     graph_limit: int = None,
                     top_k: int = 5,
                     prompt_type: str = "qa",
                     use_structured_query: bool = False) -> Dict[str, Any]:
        """
        混合检索：结合LangChain向量检索和图谱检索
        """
        start_time = time.time()
        
        print(f"\n🔍 混合检索查询: {query}")
        print("="*50)
        
        # 1. LangChain向量检索
        print("📚 执行LangChain向量检索...")
        vector_results = self.langchain_vector_search(
            query, 
            use_query_expansion=use_query_expansion,
            use_query_rewriting=use_query_rewriting,
            top_k=top_k,
            prompt_type=prompt_type,
            use_structured_query=use_structured_query
        )
        
        # 2. 图谱知识检索
        print("🕸️ 执行图谱知识检索...")
        graph_results = self.graph_knowledge_search(query, limit=graph_limit)
        
        # 3. 融合结果
        execution_time = time.time() - start_time
        
        # 改进的评分计算
        vector_score = self._calculate_vector_score(vector_results, vector_weight)
        graph_score = self._calculate_graph_score(graph_results, graph_weight)
        combined_score = vector_score + graph_score
        
        # 质量评估
        quality_metrics = self._assess_quality(vector_results, graph_results)
        
        # 4. 上下文融合生成最终答案
        print("🤝 执行上下文融合...")
        final_answer, fusion_prompt = self._context_fusion_generation(
            query, 
            vector_results, 
            graph_results
        )
        
        hybrid_results = {
            "query": query,
            "vector_results": vector_results,
            "graph_results": graph_results,
            "combined_info": {
                "vector_score": vector_score,
                "graph_score": graph_score,
                "combined_score": combined_score,
                "quality_metrics": quality_metrics,
                "execution_time": execution_time,
                "vector_weight": vector_weight,
                "graph_weight": graph_weight,
                "graph_limit_used": graph_results.get('limit_used', graph_limit if graph_limit is not None else 50),
                "top_k_used": top_k,
                "prompt_type_used": prompt_type,
                "use_structured_query": use_structured_query
            },
            "final_answer": final_answer,
            "fusion_prompt": fusion_prompt
        }
        
        print(f"⏱️ 混合检索执行时间: {execution_time:.2f}秒")
        print(f"📊 综合评分: {combined_score:.2f}")
        print(f"🎯 质量评估: {quality_metrics}")
        print(f"🕸️ 图谱限制: {graph_results.get('limit_used', 'N/A')}")
        print(f"📚 向量检索top_k: {top_k}")
        print(f"🎭 提示模板类型: {prompt_type}")
        print(f"🧩 结构化查询: {use_structured_query}")
        print("="*50)
        
        return hybrid_results
    
    def _context_fusion_generation(self, query: str, vector_results: Dict[str, Any], graph_results: Dict[str, Any]) -> tuple[str, str]:
        """
        使用融合的上下文生成最终答案
        """
        graph_context_str = self._format_graph_results_for_prompt(graph_results)
        vector_context_str = self._format_vector_results_for_prompt(vector_results)

        fusion_prompt_template = self.langchain_rag.prompt_manager.get_prompt("fusion")
        
        final_prompt = fusion_prompt_template.format(
            question=query,
            graph_context=graph_context_str,
            vector_context=vector_context_str
        )
        
        print("--- FUSION PROMPT ---")
        print(final_prompt)
        print("---------------------")

        try:
            final_answer = self.langchain_rag.llm._call(final_prompt)
            return final_answer, final_prompt
        except Exception as e:
            print(f"✗ 上下文融合生成失败: {e}")
            return "Failed to generate a fused answer.", final_prompt

    def _calculate_vector_score(self, vector_results: Dict[str, Any], vector_weight: float) -> float:
        """
        计算向量检索评分（改进版）
        """
        answer = vector_results.get('answer', '')
        
        if not answer or answer == "检索失败":
            return 0.0
        
        # 基础分数
        base_score = vector_weight
        
        # 质量加分（基于答案长度、关键词匹配等）
        quality_bonus = 0.0
        
        # 答案长度加分（适中的长度）
        answer_length = len(answer)
        if 100 <= answer_length <= 1000:
            quality_bonus += 0.1
        elif answer_length > 1000:
            quality_bonus += 0.05
        
        # 关键词匹配加分
        query_terms = set(vector_results.get('domain_terms', []))
        if query_terms:
            quality_bonus += 0.1
        
        return min(vector_weight + quality_bonus, vector_weight * 1.5)
    
    def _calculate_graph_score(self, graph_results: Dict[str, Any], graph_weight: float) -> float:
        """
        计算图谱检索评分（改进版）
        """
        entities_found = graph_results.get('entities_found', 0)
        relationships = graph_results.get('relationships', [])
        
        if entities_found == 0:
            return 0.0
        
        # 基础分数
        base_score = graph_weight * min(entities_found, 10)  # 限制最大实体数
        
        # 质量加分
        quality_bonus = 0.0
        
        # 关系多样性加分
        unique_relationships = set()
        for rel in relationships:
            relationship_type = rel.get('relationship', '')
            if relationship_type:
                unique_relationships.add(relationship_type)
        
        if len(unique_relationships) >= 3:
            quality_bonus += 0.2
        elif len(unique_relationships) >= 2:
            quality_bonus += 0.1
        
        # 实体类型多样性加分
        entity_types = set()
        for rel in relationships:
            source_type = rel.get('source_type', '')
            target_type = rel.get('target_type', '')
            
            # 处理source_type可能是列表的情况
            if isinstance(source_type, list):
                if source_type:
                    entity_types.add(source_type[0])  # 取第一个类型
            elif source_type:
                entity_types.add(source_type)
            
            # 处理target_type可能是列表的情况
            if isinstance(target_type, list):
                if target_type:
                    entity_types.add(target_type[0])  # 取第一个类型
            elif target_type:
                entity_types.add(target_type)
        
        if len(entity_types) >= 3:
            quality_bonus += 0.1
        
        return min(base_score + quality_bonus, graph_weight * 15)  # 限制最大分数
    
    def _assess_quality(self, vector_results: Dict[str, Any], graph_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估检索质量
        """
        vector_answer = vector_results.get('answer', '')
        graph_entities = graph_results.get('entities_found', 0)
        
        # 计算实体类型多样性，处理列表类型
        entity_types = set()
        for rel in graph_results.get('relationships', []):
            source_type = rel.get('source_type', '')
            target_type = rel.get('target_type', '')
            
            # 处理source_type可能是列表的情况
            if isinstance(source_type, list):
                if source_type:
                    entity_types.add(source_type[0])  # 取第一个类型
            elif source_type:
                entity_types.add(source_type)
            
            # 处理target_type可能是列表的情况
            if isinstance(target_type, list):
                if target_type:
                    entity_types.add(target_type[0])  # 取第一个类型
            elif target_type:
                entity_types.add(target_type)
        
        quality_metrics = {
            "vector_quality": {
                "has_answer": bool(vector_answer and vector_answer != "检索失败"),
                "answer_length": len(vector_answer) if vector_answer else 0,
                "domain_terms_matched": len(vector_results.get('domain_terms', [])),
                "quality_level": "high" if len(vector_answer) > 500 else "medium" if len(vector_answer) > 100 else "low"
            },
            "graph_quality": {
                "entities_found": graph_entities,
                "relationships_found": len(graph_results.get('relationships', [])),
                "entity_diversity": len(entity_types),
                "relationship_diversity": len(set([rel.get('relationship', '') for rel in graph_results.get('relationships', [])])),
                "quality_level": "high" if graph_entities >= 10 else "medium" if graph_entities >= 5 else "low"
            },
            "overall_quality": {
                "has_vector_answer": bool(vector_answer and vector_answer != "检索失败"),
                "has_graph_entities": graph_entities > 0,
                "balanced_retrieval": bool(vector_answer and graph_entities > 0),
                "recommendation": self._get_quality_recommendation(vector_results, graph_results)
            }
        }
        
        return quality_metrics
    
    def _get_quality_recommendation(self, vector_results: Dict[str, Any], graph_results: Dict[str, Any]) -> str:
        """
        根据检索结果提供质量建议
        """
        vector_answer = vector_results.get('answer', '')
        graph_entities = graph_results.get('entities_found', 0)
        
        if vector_answer and vector_answer != "检索失败" and graph_entities > 0:
            return "✅ 优秀：向量检索和图谱检索都成功"
        elif vector_answer and vector_answer != "检索失败":
            return "📚 良好：向量检索成功，但图谱检索无结果"
        elif graph_entities > 0:
            return "🕸️ 良好：图谱检索成功，但向量检索无结果"
        else:
            return "❌ 需要改进：两种检索方法都未找到相关结果"
    
    def _combine_answers(self, vector_results: Dict[str, Any], 
                        graph_results: Dict[str, Any]) -> str:
        """
        融合向量检索和图谱检索的答案，包含知识来源信息
        """
        vector_answer = vector_results.get('answer', '')
        graph_entities = graph_results.get('entities_found', 0)
        graph_relationships = graph_results.get('relationships', [])
        source_documents = vector_results.get('source_documents', [])
        retrieved_docs_count = vector_results.get('retrieved_docs_count', 0)
        
        # 如果向量检索有答案，以它为主
        if vector_answer and vector_answer != "检索失败":
            combined_answer = vector_answer
            
            # 添加知识来源信息
            if source_documents:
                # 增强metadata
                from fix_metadata_issue import enhance_source_documents_with_metadata
                enhanced_source_docs = enhance_source_documents_with_metadata(source_documents)
                
                combined_answer += f"\n\n📚 知识来源信息："
                combined_answer += f"\n- 检索到 {retrieved_docs_count} 个相关文档片段"
                
                # 添加前3个文档来源
                for i, doc in enumerate(enhanced_source_docs[:3]):
                    combined_answer += f"\n- 来源 {doc['index']}: {doc['content']}"
                    if doc.get('metadata'):
                        metadata = doc['metadata']
                        if metadata.get('source'):
                            combined_answer += f" (来源: {metadata['source']})"
                        elif metadata.get('file_name'):
                            combined_answer += f" (文件: {metadata['file_name']})"
            
            # 如果有图谱信息，添加到答案中
            if graph_entities > 0:
                combined_answer += f"\n\n📊 知识图谱补充信息："
                combined_answer += f"\n- 发现 {graph_entities} 个相关实体"
                
                # 添加前3个关系
                for i, rel in enumerate(graph_relationships[:3]):
                    source = rel.get('source', '')
                    target = rel.get('target', '')
                    relationship = rel.get('relationship', '')
                    if source and target and relationship:
                        combined_answer += f"\n- {source} {relationship} {target}"
        else:
            # 如果向量检索失败，使用图谱信息
            if graph_entities > 0:
                combined_answer = f"基于知识图谱检索到 {graph_entities} 个相关实体：\n"
                for i, rel in enumerate(graph_relationships[:5]):
                    source = rel.get('source', '')
                    target = rel.get('target', '')
                    relationship = rel.get('relationship', '')
                    if source and target and relationship:
                        combined_answer += f"- {source} {relationship} {target}\n"
            else:
                combined_answer = "抱歉，没有找到相关信息。"
        
        return combined_answer
    
    def _format_vector_results_for_prompt(self, vector_results: Dict[str, Any]) -> str:
        """Formats vector search results for the fusion prompt."""
        source_docs = vector_results.get('source_documents', [])
        if not source_docs:
            return "No relevant documents found."
        
        formatted_docs = []
        for i, doc in enumerate(source_docs[:3]):  # Limit to top 3 for conciseness
            source = doc['metadata'].get('source', 'Unknown Source')
            content = doc['content'].strip()
            formatted_docs.append(f"Source {i+1} ({source}):\n---\n{content}\n---")
            
        return "\n\n".join(formatted_docs)

    def _format_graph_results_for_prompt(self, graph_results: Dict[str, Any]) -> str:
        """Formats graph search results for the fusion prompt."""
        relationships = graph_results.get('relationships', [])
        if not relationships:
            return "No direct relationships found in the knowledge graph."
            
        formatted_rels = []
        for rel in relationships[:5]:  # Limit to top 5 for conciseness
            source = rel.get('source', 'N/A')
            target = rel.get('target', 'N/A')
            relationship = rel.get('relationship', 'RELATED_TO')
            formatted_rels.append(f"- {source} --[{relationship}]--> {target}")
            
        return "\n".join(formatted_rels)

    def get_entity_context(self, entity_name: str) -> Dict[str, Any]:
        """
        获取实体的完整上下文信息
        """
        # 实体链接信息
        entity_context = self.entity_linker.get_entity_context(entity_name)
        
        # 图谱信息
        graph_info = self.graph_retriever.get_entity_info(entity_name)
        
        # LangChain向量检索信息
        vector_results = self.langchain_vector_search(entity_name)
        
        return {
            "entity": entity_name,
            "entity_linking": entity_context,
            "graph_info": graph_info,
            "vector_context": vector_results
        }
    
    def explain_retrieval(self, query: str) -> Dict[str, Any]:
        """
        详细解释检索过程
        """
        hybrid_results = self.hybrid_search(query)
        
        explanation = {
            "query": query,
            "vector_explanation": {
                "method": "langchain_retrieval_qa",
                "framework": hybrid_results["vector_results"].get("framework", ""),
                "domain_terms": hybrid_results["vector_results"].get("domain_terms", []),
                "query_expansion": hybrid_results["vector_results"].get("expanded_query", ""),
                "final_query": hybrid_results["vector_results"].get("final_query", ""),
                "execution_time": hybrid_results["vector_results"].get("execution_time", 0)
            },
            "graph_explanation": {
                "method": "cypher_query",
                "cypher_used": hybrid_results["graph_results"].get("cypher_query", ""),
                "entities_found": hybrid_results["graph_results"].get("entities_found", 0),
                "extracted_entities": hybrid_results["graph_results"].get("extracted_entities", []),
                "relationships": [
                    {
                        "source": rel.get("source", ""),
                        "relationship": rel.get("relationship", ""),
                        "target": rel.get("target", "")
                    }
                    for rel in hybrid_results["graph_results"].get("relationships", [])[:5]
                ]
            },
            "combined_explanation": {
                "vector_weight": hybrid_results["combined_info"]["vector_weight"],
                "graph_weight": hybrid_results["combined_info"]["graph_weight"],
                "combined_score": hybrid_results["combined_info"]["combined_score"],
                "total_execution_time": hybrid_results["combined_info"]["execution_time"]
            }
        }
        
        return explanation
    
    def compare_retrieval_methods(self, query: str) -> Dict[str, Any]:
        """
        比较不同检索方法的效果
        """
        print(f"\n🔍 比较检索方法: {query}")
        print("="*50)
        
        # 1. 仅LangChain检索
        print("📚 仅LangChain检索...")
        start_time = time.time()
        langchain_only = self.langchain_vector_search(query)
        langchain_time = time.time() - start_time
        
        # 2. 仅图谱检索
        print("🕸️ 仅图谱检索...")
        start_time = time.time()
        graph_only = self.graph_knowledge_search(query)
        graph_time = time.time() - start_time
        
        # 3. 混合检索
        print("🔄 混合检索...")
        start_time = time.time()
        hybrid = self.hybrid_search(query)
        hybrid_time = time.time() - start_time
        
        comparison = {
            "query": query,
            "langchain_only": {
                "answer": langchain_only.get("answer", ""),
                "execution_time": langchain_time,
                "domain_terms": langchain_only.get("domain_terms", [])
            },
            "graph_only": {
                "entities_found": graph_only.get("entities_found", 0),
                "execution_time": graph_time,
                "relationships": graph_only.get("relationships", [])
            },
            "hybrid": {
                "answer": hybrid.get("final_answer", ""),
                "execution_time": hybrid_time,
                "combined_score": hybrid["combined_info"]["combined_score"]
            }
        }
        
        print(f"📊 比较结果:")
        print(f"  LangChain: {langchain_time:.2f}s, 领域术语: {len(langchain_only.get('domain_terms', []))}")
        print(f"  图谱检索: {graph_time:.2f}s, 实体数: {graph_only.get('entities_found', 0)}")
        print(f"  混合检索: {hybrid_time:.2f}s, 综合评分: {hybrid['combined_info']['combined_score']:.2f}")
        
        return comparison 