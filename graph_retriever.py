# -*- coding: utf-8 -*-
"""
GraphRetriever: 实现自然语言到Cypher查询的转换
支持多种查询模式，包括实体查询、关系查询、路径查询等
"""

import re
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase, basic_auth
from query_parser import QueryParser

class GraphRetriever:
    """
    基于Neo4j的图检索器，支持自然语言到Cypher查询的转换
    """
    
    def __init__(self, uri: str, user: str, password: str, default_limit: int = 50):
        self.driver = GraphDatabase.driver(uri, auth=basic_auth(user, password))
        self.default_limit = default_limit  # 默认限制数量
        
        # 查询模板映射 - 与neo4j_extract.py中的关系模式完全对应（7个关系）
        self.query_templates = {
            # 1. OPTIMIZES - 正向影响关系
            "find_optimization_relationships": {
                "patterns": [
                    # 中文模式
                    r"提升.*", r"优化.*", r"提高.*", r"增强.*", r"促进.*", r"改善.*", r"加强.*",
                    r"改进.*", r"优化.*", r"增强.*", r"促进.*", r"改善.*", r"加强.*", r"提升.*效率",
                    r"优化.*性能", r"增强.*效果", r"改善.*质量", r"提升.*表现",
                    # 英文模式 - 与抽取模式保持一致
                    r"improves?.*", r"enhances?.*", r"optimizes?.*", r"increases?.*", r"boosts?.*", 
                    r"promotes?.*", r"facilitates?.*", r"strengthens?.*", r"leads?\s+to\s+improvement.*",
                    r"contributes?\s+to\s+enhancement.*", r"plays?\s+a\s+role\s+in\s+optimizing.*",
                    r"results?\s+in\s+better.*", r"enables?\s+higher.*", r"supports?\s+improved.*",
                    r"drives?\s+enhancement.*", r"fosters?\s+optimization.*",
                    # 自然语言查询模式
                    r"how.*improve.*", r"what.*enhance.*", r"which.*optimize.*", r"how.*boost.*",
                    r"what.*promote.*", r"how.*facilitate.*", r"which.*strengthen.*",
                    r"improvement.*", r"enhancement.*", r"optimization.*", r"boost.*", r"promotion.*"
                ],
                "cypher": """
                MATCH (source)-[:OPTIMIZES]->(target)
                RETURN source.name as source, target.name as target, 
                       labels(source) as source_type, labels(target) as target_type, "OPTIMIZES" as relationship
                """
            },
            
            # 2. SUPPRESSES - 负向影响关系
            "find_suppression_relationships": {
                "patterns": [
                    # 中文模式
                    r"抑制.*", r"减少.*", r"降低.*", r"限制.*", r"降解.*", r"削弱.*", r"抑制.*效果",
                    r"降低.*性能", r"减少.*效率", r"限制.*功能", r"削弱.*表现",
                    # 英文模式 - 与抽取模式保持一致
                    r"suppresses?.*", r"reduces?.*", r"inhibits?.*", r"diminishes?.*", r"decreases?.*",
                    r"limits?.*", r"degrades?.*", r"leads?\s+to\s+reduction.*", r"contributes?\s+to\s+degradation.*",
                    r"causes?\s+decrease.*", r"results?\s+in\s+lower.*", r"prevents?\s+improvement.*",
                    r"hinders?\s+enhancement.*", r"blocks?\s+optimization.*", r"restricts?\s+performance.*",
                    # 自然语言查询模式
                    r"how.*suppress.*", r"what.*reduce.*", r"which.*inhibit.*", r"how.*decrease.*",
                    r"what.*limit.*", r"how.*degrade.*", r"which.*prevent.*",
                    r"suppression.*", r"reduction.*", r"inhibition.*", r"decrease.*", r"limitation.*"
                ],
                "cypher": """
                MATCH (suppressor)-[:SUPPRESSES]->(target)
                RETURN suppressor.name as suppressor, target.name as target, 
                       labels(suppressor) as suppressor_type, labels(target) as target_type, "SUPPRESSES" as relationship
                """
            },
            
            # 3. CAUSES - 通用因果关系
            "find_causal_relationships": {
                "patterns": [
                    # 中文模式
                    r"导致.*", r"引起.*", r"造成.*", r"产生.*", r"引发.*", r"影响.*", r"作用.*",
                    r"导致.*变化", r"引起.*效果", r"造成.*影响", r"产生.*结果",
                    # 英文模式 - 与抽取模式保持一致
                    r"causes?.*", r"leads?\s+to.*", r"results?\s+in.*", r"induces?.*", r"affects?.*", 
                    r"influences?.*", r"impacts?.*", r"contributes?\s+to.*", r"plays?\s+a\s+role\s+in.*",
                    r"drives?.*", r"triggers?.*", r"initiates?.*", r"stimulates?.*", r"provokes?.*",
                    r"generates?.*", r"produces?.*", r"creates?.*", r"brings?\s+about.*", r"gives?\s+rise\s+to.*",
                    # 自然语言查询模式
                    r"how.*cause.*", r"what.*lead\s+to.*", r"which.*result\s+in.*", r"how.*affect.*",
                    r"what.*influence.*", r"how.*impact.*", r"which.*contribute\s+to.*",
                    r"causation.*", r"causal.*", r"effect.*", r"impact.*", r"influence.*"
                ],
                "cypher": """
                MATCH (cause)-[:CAUSES]->(effect)
                RETURN cause.name as cause, effect.name as effect, 
                       labels(cause) as cause_type, labels(effect) as effect_type, "CAUSES" as relationship
                """
            },
            
            # 4. PART_OF - 组成关系
            "find_component_relationships": {
                "patterns": [
                    # 中文模式
                    r"组成部分", r"包含.*组件", r"包含.*部件", r"属于.*", r"是.*的一部分", r"组成.*",
                    r"构成.*", r"组成.*", r"构成.*", r"包含.*", r"包括.*",
                    # 英文模式 - 与抽取模式保持一致
                    r"part\s+of.*", r"component\s+of.*", r"element\s+of.*", r"constituent\s+of.*", r"belongs?\s+to.*",
                    r"forms?\s+part\s+of.*", r"constitutes?\s+part\s+of.*", r"makes?\s+up\s+part\s+of.*",
                    r"is?\s+incorporated\s+in.*", r"is?\s+integrated\s+into.*", r"comprises?\s+part\s+of.*",
                    # 自然语言查询模式
                    r"what.*part\s+of.*", r"which.*component\s+of.*", r"how.*belong\s+to.*",
                    r"what.*element\s+of.*", r"which.*constituent\s+of.*",
                    r"component.*", r"part.*", r"element.*", r"constituent.*", r"composition.*"
                ],
                "cypher": """
                MATCH (component)-[:PART_OF]->(whole)
                RETURN component.name as component, whole.name as whole, 
                       labels(component) as component_type, labels(whole) as whole_type, "PART_OF" as relationship
                """
            },
            
            # 5. IS_USED_IN - 应用/用途关系
            "find_usage_relationships": {
                "patterns": [
                    # 中文模式
                    r"用于.*", r"应用.*", r"使用.*", r"应用于.*", r"在.*中使用", r"服务于.*",
                    r"用途.*", r"应用.*", r"使用.*", r"应用领域.*", r"使用场景.*",
                    # 英文模式 - 与抽取模式保持一致
                    r"is?\s+used\s+in.*", r"is?\s+applied\s+in.*", r"is?\s+utilized\s+in.*", r"is?\s+employed\s+in.*",
                    r"serves?\s+as.*", r"used?\s+for.*", r"applied?\s+for.*", r"utilized?\s+for.*", r"employed?\s+for.*",
                    r"finds?\s+application\s+in.*", r"has?\s+application\s+in.*", r"is?\s+incorporated\s+into.*",
                    r"is?\s+integrated\s+into.*", r"plays?\s+a\s+role\s+in.*",
                    # 自然语言查询模式
                    r"what.*used\s+in.*", r"how.*applied\s+in.*", r"which.*utilized\s+in.*",
                    r"what.*employed\s+in.*", r"how.*serve\s+as.*", r"what.*used\s+for.*",
                    r"usage.*", r"application.*", r"utilization.*", r"employment.*", r"service.*"
                ],
                "cypher": """
                MATCH (material)-[:IS_USED_IN]->(application)
                RETURN material.name as material, application.name as application, 
                       labels(material) as material_type, labels(application) as application_type, "IS_USED_IN" as relationship
                """
            },
            
            # 6. IS_A - 分类关系
            "find_classification_relationships": {
                "patterns": [
                    # 中文模式
                    r"是.*类型", r"是.*例子", r"属于.*类别", r"分类为.*", r"归类为.*", r"是.*的一种",
                    r"类型.*", r"分类.*", r"类别.*", r"种类.*", r"分类体系.*",
                    # 英文模式 - 与抽取模式保持一致
                    r"is?\s+a.*", r"is?\s+a\s+type\s+of.*", r"is?\s+an\s+example\s+of.*", r"can?\s+be\s+classified\s+as.*",
                    r"belongs?\s+to\s+the\s+category\s+of.*", r"falls?\s+into\s+the\s+class\s+of.*",
                    r"is?\s+considered\s+as.*", r"is?\s+regarded\s+as.*", r"is?\s+classified\s+as.*", r"is?\s+categorized\s+as.*",
                    # 自然语言查询模式
                    r"what.*type\s+of.*", r"which.*category.*", r"how.*classified.*",
                    r"what.*example\s+of.*", r"which.*kind\s+of.*",
                    r"classification.*", r"category.*", r"type.*", r"class.*", r"classification.*"
                ],
                "cypher": """
                MATCH (instance)-[:IS_A]->(category)
                RETURN instance.name as instance, category.name as category, 
                       labels(instance) as instance_type, labels(category) as category_type, "IS_A" as relationship
                """
            },
            
            # 7. MEASURED_BY - 衡量方法关系
            "find_measurement_relationships": {
                "patterns": [
                    # 中文模式
                    r"由.*测量", r"通过.*测量", r"测量方法.*", r"评估.*", r"量化.*", r"表征.*", r"确定.*",
                    r"测量.*", r"评估.*", r"量化.*", r"表征.*", r"确定.*", r"检测.*",
                    # 英文模式 - 与抽取模式保持一致
                    r"measured?\s+by.*", r"evaluated?\s+by.*", r"quantified?\s+by.*", r"characterized?\s+by.*", 
                    r"determined?\s+by.*", r"assessed?\s+by.*", r"analyzed?\s+through.*", r"examined?\s+via.*",
                    r"investigated?\s+using.*", r"studied?\s+with.*", r"monitored?\s+through.*", r"tracked?\s+via.*",
                    # 自然语言查询模式
                    r"how.*measured.*", r"what.*evaluated.*", r"which.*quantified.*",
                    r"how.*characterized.*", r"what.*determined.*", r"how.*assessed.*",
                    r"measurement.*", r"evaluation.*", r"quantification.*", r"characterization.*", r"assessment.*"
                ],
                "cypher": """
                MATCH (metric)-[:MEASURED_BY]->(method)
                RETURN metric.name as metric, method.name as method, 
                       labels(metric) as metric_type, labels(method) as method_type, "MEASURED_BY" as relationship
                """
            },
            
            # 通用查询模板
            "find_related_entities": {
                "patterns": [
                    r"相关.*", r"关联.*", r"与.*相关", r"联系.*", r"所有.*关系",
                    r"related.*", r"associate.*", r"connect.*", r"link.*", r"all.*relationships"
                ],
                "cypher": """
                MATCH (e1)-[r]->(e2)
                RETURN e1.name as source, type(r) as relationship, e2.name as target,
                       labels(e1) as source_type, labels(e2) as target_type
                """
            },
            
            "find_all_relationships": {
                "patterns": [
                    r"关系.*", r"连接.*", r"联系.*", r"所有.*连接", r"图谱.*关系",
                    r"relationship.*", r"connection.*", r"link.*", r"how.*", r"all.*connections",
                    r"graph.*relationships", r"all.*links", r"connections.*between"
                ],
                "cypher": """
                MATCH (e1)-[r]->(e2)
                RETURN e1.name as source, type(r) as relationship, e2.name as target,
                       labels(e1) as source_type, labels(e2) as target_type
                """
            }
        }
    
    def close(self):
        self.driver.close()
    
    def natural_language_to_cypher(self, query: str, limit: int = None) -> Optional[str]:
        """
        将自然语言查询转换为Cypher查询
        """
        query_lower = query.lower()
        
        # 如果没有指定limit，使用默认值
        if limit is None:
            limit = self.default_limit
        
        for template_name, template in self.query_templates.items():
            for pattern in template["patterns"]:
                if re.search(pattern, query_lower):
                    cypher = template["cypher"].strip()
                    # 添加LIMIT子句
                    if "LIMIT" not in cypher.upper():
                        cypher += f"\nLIMIT {limit}"
                    return cypher
        
        # 如果没有匹配的模板，返回通用查询
        return f"""
        MATCH (e1)-[r]->(e2)
        RETURN e1.name as source, type(r) as relationship, e2.name as target,
               labels(e1) as source_type, labels(e2) as target_type
        LIMIT {limit}
        """
    
    def execute_cypher(self, cypher_query: str) -> List[Dict[str, Any]]:
        """
        执行Cypher查询并返回结果
        """
        with self.driver.session() as session:
            result = session.run(cypher_query)
            return [dict(record) for record in result]
    
    def query(self, natural_query: str, limit: int = None) -> Dict[str, Any]:
        """
        执行自然语言查询
        """
        cypher = self.natural_language_to_cypher(natural_query, limit)
        results = self.execute_cypher(cypher)
        
        return {
            "natural_query": natural_query,
            "cypher_query": cypher,
            "results": results,
            "count": len(results),
            "limit_used": limit if limit is not None else self.default_limit
        }
    
    def get_entity_info(self, entity_name: str) -> Dict[str, Any]:
        """
        获取实体的详细信息
        """
        cypher = f"""
        MATCH (n {{name: $name}})
        OPTIONAL MATCH (n)-[r]->(target)
        OPTIONAL MATCH (source)-[r2]->(n)
        RETURN n.name as entity, labels(n) as type,
               collect(DISTINCT {{target: target.name, relationship: type(r)}}) as outgoing,
               collect(DISTINCT {{source: source.name, relationship: type(r2)}}) as incoming
        """
        
        with self.driver.session() as session:
            result = session.run(cypher, name=entity_name)
            return dict(result.single()) if result.peek() else {}

    def search_from_parsed_query(self, parsed_query: Dict[str, Any], limit: int = 15) -> Dict[str, Any]:
        """
        Executes a search based on the structured query from QueryParser.
        Centralizes query building and limit application.
        """
        final_query = ""
        strategy_used = "none"

        # Strategy 1: Precise query with constraints
        if parsed_query.get("intent") == "find_entity_by_relation":
            query_body = self._build_query_with_constraints(parsed_query, limit)
            if query_body:
                final_query = f"{query_body}\nLIMIT {limit}"
                strategy_used = "precise_with_constraints"
                print(f"Debug - Using '{strategy_used}' strategy.")
                try:
                    results = self.execute_cypher(final_query)
                    if results:
                        print(f"Debug - Strategy '{strategy_used}' succeeded with {len(results)} results.")
                        return {"cypher_query": final_query, "results": results}
                    else:
                        print(f"Debug - Strategy '{strategy_used}' yielded no results. Trying constraint-only query.")
                        # Try constraint-only query if source entity doesn't match
                        constraints_only_query = self._build_constraint_only_query(parsed_query, limit)
                        print(f"Debug - Constraint-only query built: {constraints_only_query[:100]}...")
                        if constraints_only_query:
                            constraint_query = f"{constraints_only_query}\nLIMIT {limit}"
                            print(f"Debug - Executing constraint-only query...")
                            try:
                                constraint_results = self.execute_cypher(constraint_query)
                                if constraint_results:
                                    print(f"Debug - Constraint-only query succeeded with {len(constraint_results)} results.")
                                    return {"cypher_query": constraint_query, "results": constraint_results}
                                else:
                                    print(f"Debug - Constraint-only query returned no results.")
                            except Exception as constraint_error:
                                print(f"Debug - Constraint-only query failed: {constraint_error}")
                        else:
                            print(f"Debug - No constraint-only query could be built.")
                except Exception as e:
                    print(f"Debug - Strategy '{strategy_used}' failed with error: {e}. Falling back.")

        # Strategy 2: Standard entity finding query
        query_body = self._build_find_entity_query(parsed_query, limit)
        if query_body:
            final_query = f"{query_body}\nLIMIT {limit}"
            strategy_used = "find_entity"
            # print(f"Using '{strategy_used}' strategy.") # Original code had print, keeping it
            results = self.execute_cypher(final_query)
            if results:
                return {"cypher_query": final_query, "results": results}
            # print(f"Strategy '{strategy_used}' yielded no results. Falling back.") # Original code had print, keeping it

        # Strategy 3: Simple fallback for broad matching
        query_body = self._build_simple_fallback_query(parsed_query, limit)
        if query_body:
            final_query = f"{query_body}\nLIMIT {limit}"
            strategy_used = "simple_fallback"
            # print(f"Using '{strategy_used}' strategy.") # Original code had print, keeping it
            results = self.execute_cypher(final_query)
            if results:
                return {"cypher_query": final_query, "results": results}

        # print("All query strategies failed to find results.") # Original code had print, keeping it
        return {"cypher_query": final_query, "results": []}


    def _build_query_with_constraints(self, parsed_query: Dict[str, Any], limit: int = 15) -> str:
        """
        Builds a precise Cypher query using a source entity, target type,
        relationship, and constraints.
        """
        source_entity = parsed_query.get("source_entity")
        target_type = parsed_query.get("target_entity_type") or "n"
        relationship = parsed_query.get("relationship", "").upper()
        constraints = parsed_query.get("constraints", {})

        # Build the MATCH clause
        match_clauses = []
        where_clauses = []
        
        # Base match for the target node
        if source_entity:
            sanitized_source = source_entity.replace("'", "\\'")
            base_match = f"(source {{name: '{sanitized_source}'}})-[r:{relationship}]->(target:{target_type})"
        else:
            base_match = f"(target:{target_type})"

        match_clauses.append(base_match)

        # Add matches for constraint nodes
        for i, (prop, value) in enumerate(constraints.items()):
            constraint_node_alias = f"c{i}"
            # Assuming a generic relationship like 'HAS_PROPERTY' or that the relationship is part of the constraint
            # For this example, let's assume a generic connection. A more robust solution needs schema awareness.
            match_clauses.append(f"(target)-[]->({constraint_node_alias})")
            if isinstance(value, str):
                 sanitized_value = value.replace("'", "\\'")
                 where_clauses.append(f"({constraint_node_alias}.name CONTAINS '{sanitized_value}' OR {constraint_node_alias}.value CONTAINS '{sanitized_value}')")
            # Can add handling for other types like numbers
        
        # Build the final query
        match_str = " MATCH " + ", ".join(match_clauses)
        where_str = ""
        if where_clauses:
            where_str = " WHERE " + " AND ".join(where_clauses)

        # Define the RETURN structure
        return_str = f"""
        RETURN target.name as found_entity, 
               labels(target) as entity_type,
               source.name as related_to, 
               type(r) as relationship
        """
        # Handle case where there's no source entity - use constraint-only query
        if not source_entity and constraints:
            # Build a direct constraint-based query like: MATCH (target:Material)-[r:OPTIMIZES]->(constraint_node)
            constraint_value = list(constraints.values())[0] if constraints else "efficiency"
            sanitized_constraint = constraint_value.replace("'", "\\'")
            
            constraint_only_query = f"""
            MATCH (target:{target_type})-[r:{relationship}]->(constraint_node)
            WHERE constraint_node.name CONTAINS '{sanitized_constraint}'
            RETURN target.name as found_entity,
                   labels(target) as entity_type,
                   constraint_node.name as related_to,
                   type(r) as relationship
            """
            # print(f"Constructed constraint-only query: {constraint_only_query}") # Original code had print, keeping it
            return constraint_only_query.strip()


        final_query_without_limit = match_str + where_str + return_str
        # print(f"Constructed precise query: {final_query_without_limit}") # Original code had print, keeping it
        return final_query_without_limit

    def _build_constraint_only_query(self, parsed_query: Dict[str, Any], limit: int = 15) -> str:
        """
        Builds a constraint-only query when source entity doesn't exist in the graph.
        This is similar to the constraint-only branch in _build_query_with_constraints.
        """
        target_type = parsed_query.get("target_entity_type") or "Material"
        relationship = parsed_query.get("relationship", "OPTIMIZES").upper()
        constraints = parsed_query.get("constraints", {})
        
        if not constraints:
            return ""
            
        # Build a direct constraint-based query
        constraint_value = list(constraints.values())[0]
        sanitized_constraint = constraint_value.replace("'", "\\'")
        
        constraint_only_query = f"""
        MATCH (target:{target_type})-[r:{relationship}]->(constraint_node)
        WHERE constraint_node.name CONTAINS '{sanitized_constraint}'
        RETURN target.name as found_entity,
               labels(target) as entity_type,
               constraint_node.name as related_to,
               type(r) as relationship
        """
        return constraint_only_query.strip()

    def _build_find_entity_query(self, parsed_query: Dict[str, Any], limit: int = 15) -> str:
        """Builds a Cypher query for the 'find_entity' intent."""
        source_entities = parsed_query.get("source_entities") or [parsed_query.get("source_entity")]
        if not source_entities or source_entities[0] is None:
             # print(f"Debug - No source entities, using general query") # Original code had print, keeping it
             return self._build_general_query(parsed_query.get("target_entity_type"), parsed_query.get("relation_type"))

        target_type = parsed_query.get("target_entity_type")
        relation = parsed_query.get("relation_type")
        
        # print(f"Debug - Building query with: source_entities={source_entities}, target_type={target_type}, relation={relation}") # Original code had print, keeping it
        
        # 如果缺少必要字段，尝试构建更宽松的查询
        if not source_entities:
            # print(f"Debug - No source entities, using general query") # Original code had print, keeping it
            return self._build_general_query(target_type, relation)
        
        # 简化策略，避免复杂的UNION查询
        source_name = source_entities[0]
        
        # 策略1: 优先尝试双向关系匹配
        if target_type and relation:
            query = f"""
            MATCH (source {{name: '{source_name}'}})-[r:{relation}]-(target:{target_type})
            RETURN target.name AS found_entity, type(r) AS relationship, 'bidirectional' AS strategy
            """
            return query
            
        # 策略2: 如果没有指定关系，尝试任意关系匹配
        elif target_type:
            query = f"""
            MATCH (source {{name: '{source_name}'}})-[r]-(target:{target_type})
            RETURN target.name AS found_entity, type(r) AS relationship, 'any_relation' AS strategy
            """
            return query
            
        # 策略3: 最宽松的匹配 - 查找所有相关实体
        else:
            query = f"""
            MATCH (source {{name: '{source_name}'}})-[r]-(target)
            RETURN target.name AS found_entity, type(r) AS relationship, 'all_related' AS strategy
            """
            return query

    def _build_general_query(self, target_type: str = None, relation: str = None) -> str:
        """构建通用查询，当缺少具体实体时使用"""
        if target_type and relation:
            return f"""
            MATCH (source)-[r:{relation}]-(target:{target_type})
            RETURN target.name AS found_entity, type(r) AS relationship, 'general_bidirectional' AS strategy
            """
        elif target_type:
            return f"""
            MATCH (source)-[r]-(target:{target_type})
            RETURN target.name AS found_entity, type(r) AS relationship, 'general_any_relation' AS strategy
            """
        else:
            return """
            MATCH (source)-[r]-(target)
            RETURN target.name AS found_entity, type(r) AS relationship, 'general_all' AS strategy
            """

    def _build_fallback_query(self, parsed_query: Dict[str, Any], limit: int) -> str:
        """构建回退查询，更宽松的条件"""
        source_entities = parsed_query.get("source_entities", [])
        if not source_entities:
            return ""
        
        source_name = source_entities[0]
        # 最宽松的查询 - 只要实体相关的所有内容
        query = f"""
        MATCH (source {{name: '{source_name}'}})-[r]-(target)
        RETURN target.name AS found_entity, type(r) AS relationship, 'fallback' AS strategy
        LIMIT {limit if limit else self.default_limit}
        """
        return query

    def _build_simple_fallback_query(self, parsed_query: Dict[str, Any], limit: int = 15) -> str:
        """构建最简单的回退查询"""
        source_entities = parsed_query.get("source_entities", [])
        if not source_entities:
            # 如果连实体都没有，返回一些示例结果
            query = f"""
            MATCH (source)-[r]-(target)
            RETURN target.name AS found_entity, type(r) AS relationship, 'simple_fallback' AS strategy
            LIMIT {limit if limit else min(10, self.default_limit)}
            """
            return query
        
        source_name = source_entities[0]
        # 使用CONTAINS进行模糊匹配
        query = f"""
        MATCH (source)-[r]-(target)
        WHERE source.name CONTAINS '{source_name}' OR target.name CONTAINS '{source_name}'
        RETURN target.name AS found_entity, type(r) AS relationship, 'fuzzy_fallback' AS strategy
        LIMIT {limit if limit else self.default_limit}
        """
        return query

    def _build_find_relation_query(self, pq: Dict[str, Any]) -> str:
        """Builds a Cypher query for the 'find_relation' intent."""
        source_entities = pq.get("source_entities", [])
        if len(source_entities) < 2:
            return "" # Need at least two entities to find a relation

        e1_name = source_entities[0]
        e2_name = source_entities[1]

        query = f"""
        MATCH (e1 {{name: '{e1_name}'}})-[r]-(e2 {{name: '{e2_name}'}})
        RETURN e1.name AS source, type(r) AS relationship, e2.name AS target
        """
        return query 