
import json
import re
from typing import Dict, Any, List
import logging
from zhipuai import ZhipuAI
from entity_linker import EntityLinker

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QueryParser:
    def __init__(self, entity_linker: EntityLinker, api_key: str):
        self.entity_linker = entity_linker
        self.llm = ZhipuAI(api_key=api_key)

    def _build_parsing_prompt(self, query: str, entities: List[Dict[str, Any]]) -> str:
        entity_list = [f"- `{e['standard_name']}` (type: {e.get('category', 'Unknown')})" for e in entities]
        entity_str = "\n".join(entity_list)

        return f"""
        **Task:** Analyze the user's query about solar cells and convert it into a structured JSON object for a knowledge graph query.

        **User Query:** "{query}"

        **Extracted Entities:**
        {entity_str}

        **Instructions:**
        Your goal is to create a JSON object that captures the core intent of the query. Follow these steps:
        1.  **Identify Intent (`intent`):** Determine the user's primary goal. 
            - Use `find_entity_by_relation` if the user is looking for entities (like materials, processes) that have a specific relationship with another entity and meet certain criteria. **This is the most common and preferred intent.**
            - Use `get_entity_details` only if the user just wants general information about a single, specific entity.
        2.  **Identify Source Entity (`source_entity`):** Find the main subject of the query. This is usually the most specific **Device, Material, or Process** mentioned. It must be one of the extracted entities. **Metrics (like 'efficiency') or generic concepts (like 'material') are NOT source entities; they belong in `target_entity_type` or `constraints`.**
        3.  **Identify Target Entity Type (`target_entity_type`):** Determine the *type* of entity the user is looking for (e.g., "Material", "Process", "Metric"). This is required for `find_entity_by_relation`.
        4.  **Identify Relationship (`relationship`):** Extract the verb or action that connects the source and target entities. **CRITICAL: Always use these standard relationships:** "OPTIMIZES" (for improves/enhances/increases), "SUPPRESSES" (for reduces/decreases), "CAUSES" (for general causality), "PART_OF", "IS_USED_IN", "IS_A", "MEASURED_BY". This is required for `find_entity_by_relation`.
        5.  **Define Constraints (`constraints`):** This is critical. **Place any metrics (like 'efficiency') or properties here.** The key is the property to filter on (e.g., "name") and the value is the desired keyword (e.g., "efficiency"). This field is required for `find_entity_by_relation`. Do NOT use a separate `target_property` field.

        **Example 1: Complex Query**
        - **Query:** "What material improves the efficiency of crystalline silicon solar cells?"
        - **Extracted Entities:**
          - `crystalline silicon solar cells` (type: Device)
          - `material` (type: Concept)
          - `efficiency` (type: Metric)
        - **Analysis:** The user wants to find a `Material` (target) that `OPTIMIZES` (relationship) the main subject, `crystalline silicon solar cells` (source). The optimization is constrained by the metric `efficiency` (constraint). **IMPORTANT: Always map "improves", "enhances", "increases" to "OPTIMIZES".**
        - **Correct JSON Output (MUST follow this structure):**
        ```json
        {{
            "intent": "find_entity_by_relation",
            "source_entity": "crystalline silicon solar cells",
            "target_entity_type": "Material",
            "relationship": "OPTIMIZES",
            "constraints": {{
                "name": "efficiency"
            }}
        }}
        ```
        
        **Example 2: Simple Query**
        - **Query:** "Tell me about anti-reflection coatings."
        - **Extracted Entities:** 
          - `anti-reflection coatings` (type: Material)
        - **Correct JSON Output:**
        ```json
        {{
            "intent": "get_entity_details",
            "source_entity": "anti-reflection coatings"
        }}
        ```

        Now, analyze the user query and provide the JSON output. Remember to strictly follow the rules for identifying the source entity and constraints.
        """

    def _process_llm_response(self, response_text: str) -> Dict[str, Any]:
        """
        Processes the raw text response from the LLM, extracting the JSON block.
        """
        try:
            # Use regex to find the JSON block, which is more robust
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
            else:
                logger.warning("No JSON block found in ```json ... ``` format. Trying to parse the whole response.")
                # As a fallback, try to parse the whole string
                return json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON from LLM response: {e}")
            return {"error": "Invalid JSON format", "raw_response": response_text}
        except Exception as e:
            logger.error(f"An unexpected error occurred during LLM response processing: {e}", exc_info=True)
            return {"error": f"An unexpected error occurred: {e}", "raw_response": response_text}

    def _fallback_parse(self, query: str, entities: list) -> dict:
        """
        A rule-based fallback parser if the LLM fails.
        """
        logger.info("Executing rule-based fallback parser.")
        
        # Simple keyword-based intent detection
        intent = "find_entity_by_relation"
        target_entity_type = "Material" # Default assumption for "what material" questions
        relationship = "OPTIMIZES" # Default assumption
        
        # Map common relationship variations to standard relationships
        query_lower = query.lower()
        if any(word in query_lower for word in ["improve", "enhance", "increase", "boost", "optimize"]):
            relationship = "OPTIMIZES"
        
        source_entity = None
        for entity in entities:
            if entity.get("category") in ["Device", "Technology"]:
                source_entity = entity.get("standard_name")
                break
        if not source_entity and entities:
             for entity in entities:
                 if entity.get("category") != "Metric":
                     source_entity = entity.get("standard_name")
                     break

        constraints = {}
        for entity in entities:
            if entity.get("category") == "Metric":
                constraints["name"] = entity.get("standard_name")

        if source_entity and source_entity.lower() in ['efficiency', 'stability', 'performance']:
             constraints["name"] = source_entity
             source_entity = None

        if not source_entity and not constraints:
            if constraints:
                source_entity = None
            else:
                return {"error": "Fallback parser could not identify a clear source or constraint."}

        result = {
            "intent": intent,
            "source_entity": source_entity,
            "target_entity_type": target_entity_type,
            "relationship": relationship,
            "constraints": constraints,
            "parsing_method": "rule_based_fallback"
        }
        logger.info(f"Fallback parser generated: {result}")
        return result

    def parse_query(self, query: str) -> dict:
        """
        Parses a natural language query into a structured JSON format.
        Tries LLM parsing first, then falls back to a rule-based method.
        """
        logger.info(f"Parsing query: '{query}'")
        linked_entities = self.entity_linker.extract_entities_from_text(query)
        logger.debug(f"Linked entities: {linked_entities}")

        try:
            prompt = self._build_parsing_prompt(query, linked_entities)
            response = self.llm.chat.completions.create(
                model="glm-4-flash",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            raw_response_text = response.choices[0].message.content
            logger.debug(f"Raw LLM response: {raw_response_text}")
            
            parsed_response = self._process_llm_response(raw_response_text)
            
            if parsed_response.get("error"):
                logger.warning(f"LLM parsing failed: {parsed_response['error']}. Trying fallback parser.")
                return self._fallback_parse(query, linked_entities)
            
            parsed_response['parsing_method'] = 'llm'
            return parsed_response

        except Exception as e:
            logger.error(f"Error parsing query with ZhipuAI LLM: {e}", exc_info=True)
            return self._fallback_parse(query, linked_entities)


if __name__ == '__main__':
    ZHIPU_API_KEY = "41b29e65745d4110a018c5d616b0012f.A6CEwmornnYXSVLC" 
    
    if "YOUR_ZHIPU_API_KEY" in ZHIPU_API_KEY:
        print("Please replace 'YOUR_ZHIPU_API_KEY' with your actual Zhipu AI API key.")
    else:
        linker = EntityLinker(domain_dict_path="domain_keywords/domain_dictionary_cleaned.json")
        parser = QueryParser(entity_linker=linker, api_key=ZHIPU_API_KEY)
        
        test_queries = [
            "What material improves the efficiency of crystalline silicon solar cells?",
            "Tell me about anti-reflection coatings.",
            "How can I increase the stability of perovskite solar cells?",
            "What are the manufacturing processes for silicon solar cells?"
        ]
        
        for query in test_queries:
            parsed = parser.parse_query(query)
            print("\n--- Final Result ---")
            print(f"Query: {query}")
            print(f"Parsed Result: {json.dumps(parsed, indent=2, ensure_ascii=False)}")
            print("-" * 20)
