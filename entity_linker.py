# -*- coding: utf-8 -*-
"""
实体链接与标准化模块
完善NodeParser和实体标准化流程，保证图谱和RAG一致性
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

class EntityLinker:
    """
    实体链接器：实现实体标准化和消歧
    """
    
    def __init__(self, domain_dictionary_path: str):
        # 加载领域词典
        with open(domain_dictionary_path, 'r', encoding='utf-8') as f:
            self.domain_dict = json.load(f)
        
        # 构建索引
        self._build_indexes()
    
    def _build_indexes(self):
        """
        构建实体索引
        """
        self.standard_to_info = {}
        self.alias_to_standard = {}
        self.category_entities = defaultdict(list)
        
        for entry in self.domain_dict:
            standard_name = entry["standard_name"]
            category = entry["category"]
            aliases = entry["aliases"]
            
            # 标准名到信息的映射
            self.standard_to_info[standard_name] = {
                "category": category,
                "aliases": aliases,
                "standard_name": standard_name
            }
            
            # 别名到标准名的映射
            for alias in aliases:
                self.alias_to_standard[alias.lower()] = standard_name
            
            # 按类别分组
            self.category_entities[category].append(standard_name)
    
    def normalize_entity(self, entity_text: str) -> Optional[Dict[str, Any]]:
        """
        实体标准化
        """
        entity_lower = entity_text.lower()
        
        # 直接匹配标准名
        if entity_text in self.standard_to_info:
            return {
                "original": entity_text,
                "standard_name": entity_text,
                "category": self.standard_to_info[entity_text]["category"],
                "confidence": 1.0
            }
        
        # 匹配别名
        if entity_lower in self.alias_to_standard:
            standard_name = self.alias_to_standard[entity_lower]
            return {
                "original": entity_text,
                "standard_name": standard_name,
                "category": self.standard_to_info[standard_name]["category"],
                "confidence": 0.9
            }
        
        # 模糊匹配
        best_match = self._fuzzy_match(entity_text)
        if best_match:
            return best_match
        
        return None
    
    def _fuzzy_match(self, entity_text: str) -> Optional[Dict[str, Any]]:
        """
        模糊匹配实体
        """
        entity_lower = entity_text.lower()
        best_score = 0
        best_match = None
        
        for standard_name in self.standard_to_info:
            # 计算相似度
            score = self._calculate_similarity(entity_lower, standard_name.lower())
            if score > best_score and score > 0.7:  # 阈值
                best_score = score
                best_match = {
                    "original": entity_text,
                    "standard_name": standard_name,
                    "category": self.standard_to_info[standard_name]["category"],
                    "confidence": score
                }
        
        return best_match
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        计算文本相似度
        """
        # 简单的Jaccard相似度
        set1 = set(text1.split())
        set2 = set(text2.split())
        
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def extract_entities_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        从文本中提取并标准化实体
        """
        entities = []
        text_lower = text.lower()
        
        # 按长度排序，优先匹配长实体
        all_aliases = sorted(self.alias_to_standard.keys(), key=len, reverse=True)
        
        for alias in all_aliases:
            if alias in text_lower:
                standard_name = self.alias_to_standard[alias]
                entity_info = self.standard_to_info[standard_name]
                
                entities.append({
                    "original": alias,
                    "standard_name": standard_name,
                    "category": entity_info["category"],
                    "confidence": 0.9,
                    "position": text_lower.find(alias)
                })
        
        # 去重和排序
        unique_entities = []
        seen_standards = set()
        
        for entity in sorted(entities, key=lambda x: x["position"]):
            if entity["standard_name"] not in seen_standards:
                unique_entities.append(entity)
                seen_standards.add(entity["standard_name"])
        
        return unique_entities
    
    def get_entity_context(self, entity_name: str) -> Dict[str, Any]:
        """
        获取实体的上下文信息
        """
        normalized = self.normalize_entity(entity_name)
        if not normalized:
            return {}
        
        standard_name = normalized["standard_name"]
        entity_info = self.standard_to_info[standard_name]
        
        return {
            "entity": entity_name,
            "normalized": normalized,
            "aliases": entity_info["aliases"],
            "category": entity_info["category"],
            "related_entities": self._get_related_entities(standard_name)
        }
    
    def _get_related_entities(self, standard_name: str) -> List[str]:
        """
        获取相关实体
        """
        category = self.standard_to_info[standard_name]["category"]
        return [entity for entity in self.category_entities[category] 
                if entity != standard_name][:5]

class NodeParser:
    """
    节点解析器：用于文档分块和实体标注
    """
    
    def __init__(self, entity_linker: EntityLinker, chunk_size: int = 1000, overlap: int = 200):
        self.entity_linker = entity_linker
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def parse_document(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        解析文档为节点
        """
        # 分块
        chunks = self._split_text(text)
        
        nodes = []
        for i, chunk in enumerate(chunks):
            # 提取实体
            entities = self.entity_linker.extract_entities_from_text(chunk)
            
            # 创建节点
            node = {
                "id": f"chunk_{i}",
                "content": chunk,
                "metadata": metadata or {},
                "entities": entities,
                "entity_count": len(entities),
                "chunk_index": i
            }
            
            nodes.append(node)
        
        return nodes
    
    def _split_text(self, text: str) -> List[str]:
        """
        智能文本分块
        """
        # 按句子分割
        sentences = re.split(r'(?<=[.!?。！？])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def create_entity_nodes(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        为实体创建专门的节点
        """
        entity_nodes = []
        entity_seen = set()
        
        for node in nodes:
            for entity in node["entities"]:
                standard_name = entity["standard_name"]
                
                if standard_name not in entity_seen:
                    entity_node = {
                        "id": f"entity_{standard_name}",
                        "type": "entity",
                        "name": standard_name,
                        "category": entity["category"],
                        "aliases": self.entity_linker.standard_to_info[standard_name]["aliases"],
                        "related_chunks": [node["id"]],
                        "confidence": entity["confidence"]
                    }
                    entity_nodes.append(entity_node)
                    entity_seen.add(standard_name)
                else:
                    # 更新已存在的实体节点
                    for existing_node in entity_nodes:
                        if existing_node["name"] == standard_name:
                            existing_node["related_chunks"].append(node["id"])
                            break
        
        return entity_nodes

class ConsistencyChecker:
    """
    一致性检查器：确保图谱和RAG的一致性
    """
    
    def __init__(self, entity_linker: EntityLinker):
        self.entity_linker = entity_linker
    
    def check_entity_consistency(self, graph_entities: List[str], 
                               rag_entities: List[str]) -> Dict[str, Any]:
        """
        检查实体一致性
        """
        # 标准化实体
        graph_normalized = [self.entity_linker.normalize_entity(e) for e in graph_entities]
        rag_normalized = [self.entity_linker.normalize_entity(e) for e in rag_entities]
        
        # 提取标准名
        graph_standards = {e["standard_name"] for e in graph_normalized if e}
        rag_standards = {e["standard_name"] for e in rag_normalized if e}
        
        # 计算一致性指标
        intersection = graph_standards.intersection(rag_standards)
        union = graph_standards.union(rag_standards)
        
        consistency_score = len(intersection) / len(union) if union else 0.0
        
        return {
            "consistency_score": consistency_score,
            "graph_entities": len(graph_standards),
            "rag_entities": len(rag_standards),
            "common_entities": len(intersection),
            "graph_only": list(graph_standards - rag_standards),
            "rag_only": list(rag_standards - graph_standards),
            "common": list(intersection)
        }
    
    def suggest_improvements(self, consistency_report: Dict[str, Any]) -> List[str]:
        """
        基于一致性报告提出改进建议
        """
        suggestions = []
        
        if consistency_report["consistency_score"] < 0.8:
            suggestions.append("实体一致性较低，建议检查实体抽取和标准化流程")
        
        if consistency_report["graph_only"]:
            suggestions.append(f"图谱中有 {len(consistency_report['graph_only'])} 个实体未在RAG中出现")
        
        if consistency_report["rag_only"]:
            suggestions.append(f"RAG中有 {len(consistency_report['rag_only'])} 个实体未在图谱中出现")
        
        return suggestions 