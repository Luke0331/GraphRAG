import streamlit as st
import json
import pandas as pd # Import pandas for dataframes
import streamlit.components.v1 as components
from pyvis.network import Network
from enhanced_hybrid_retriever import EnhancedHybridRetriever
from evaluation import Evaluator, get_sample_ground_truth # Import evaluation tools

# --- 1. 初始化与缓存 ---

# 使用Streamlit的缓存功能，避免每次交互都重新加载模型
@st.cache_resource
def get_retriever():
    """
    加载并缓存EnhancedHybridRetriever实例。
    """
    # 确保您的API密钥和密码是安全的，例如使用Streamlit的secrets管理
    # 这里为了演示方便，直接写在代码里
    retriever = EnhancedHybridRetriever(
        neo4j_password="domain-kg123",
        llm_api_key="41b29e65745d4110a018c5d616b0012f.A6CEwmornnYXSVLC" # 请替换为您的智谱AI API Key
    )
    return retriever

# --- 2. UI界面构建 ---

def main():
    """
    Streamlit应用主函数
    """
    st.set_page_config(page_title="GraphRAG 知识检索系统", layout="wide")
    
    st.sidebar.title("导航")
    page = st.sidebar.radio("选择一个页面:", [
        "🔍 交互式检索", 
        "📊 系统评估"
    ])

    st.title("🔬 领域知识图谱增强的RAG系统")
    st.markdown("一个结合了向量检索与知识图谱的智能问答系统，用于太阳能领域的文献研究。")

    # 初始化retriever
    try:
        retriever = get_retriever()
        # 安全地检查API key，防止NoneType错误
        api_key = getattr(getattr(retriever.langchain_rag.llm, '_llm', None), 'api_key', None)
        if api_key and "YOUR_ZHIPU_API_KEY" in api_key:
            st.error("请在 `app.py` 代码中替换为您的智谱AI API Key。")
            return
    except Exception as e:
        st.error(f"初始化检索器失败: {e}")
        st.error("请确保Neo4j数据库正在运行，并且API Key有效。")
        return

    if page == "🔍 交互式检索":
        run_interactive_retrieval(retriever)
    elif page == "📊 系统评估":
        run_system_evaluation(retriever)


def run_interactive_retrieval(retriever):
    """
    运行交互式检索页面
    """
    st.sidebar.header("⚙️ 检索设置")
    retrieval_mode = st.sidebar.radio(
        "选择检索模式:",
        ('Graph-Guided Retrieval (顺序引导)', 'Hybrid Search (并行融合)'),
        help="**顺序引导**: 问题→图检索→向量精排→LLM生成 (更精确)。\n\n**并行融合**: 同时进行向量和图谱检索，然后融合结果 (可能更快)。"
    )
    
    top_k = st.sidebar.slider("返回Top-K文档:", min_value=1, max_value=10, value=3)
    graph_limit = st.sidebar.slider("返回图谱关系数量:", min_value=5, max_value=50, value=15, help="设置从知识图谱中检索的最大关系/实体数量。")

    st.header("💬 提出您的问题")
    query = st.text_input("例如：什么材料可以提高晶体硅太阳能电池的效率？", "")

    if st.button("🚀 开始检索"):
        if not query:
            st.warning("请输入您的问题。")
        else:
            with st.spinner('正在检索中，请稍候...'):
                if retrieval_mode == 'Graph-Guided Retrieval (顺序引导)':
                    results = retriever.graph_guided_retrieval(query, top_k=top_k, graph_limit=graph_limit)
                    display_graph_guided_results(results)
                else: # Hybrid Search
                    results = retriever.hybrid_search(query, top_k=top_k, graph_limit=graph_limit)
                    display_hybrid_results(results)

def run_system_evaluation(retriever):
    """
    运行系统评估页面
    """
    st.header("📊 系统性能评估")
    st.markdown("在这里，我们使用一个预定义的“标准答案”数据集来评估和比较两种检索模式的性能。")

    ground_truth_data = get_sample_ground_truth()
    
    with st.expander("查看评估数据集"):
        st.json(ground_truth_data)

    if st.button("🚀 运行评估套件"):
        with st.spinner("正在运行评估... 这可能需要一些时间..."):
            evaluator = Evaluator(retriever, ground_truth_data)
            results = evaluator.run_evaluation()

            st.success("评估完成！")
            st.subheader("📈 评估结果摘要")

            # Prepare data for dataframe
            summary_data = []
            for res in results:
                summary_data.append({
                    "Query": res['query'],
                    "Method": "Hybrid Search",
                    "BLEU-4": res['hybrid_search']['bleu_4'],
                    "Graph Recall": res['hybrid_search']['graph_recall']
                })
                summary_data.append({
                    "Query": res['query'],
                    "Method": "Graph-Guided",
                    "BLEU-4": res['graph_guided']['bleu_4'],
                    "Graph Recall": res['graph_guided']['graph_recall']
                })
            
            df = pd.DataFrame(summary_data)
            st.dataframe(df.style.format({"BLEU-4": "{:.4f}", "Graph Recall": "{:.4f}"}))
            
            st.subheader("📋 详细结果")
            st.json(results)

# --- 3. 结果显示函数 ---

def display_graph_guided_results(results):
    """
    显示Graph-Guided Retrieval的结果
    """
    st.success("顺序引导检索完成！")
    
    if "error" in results:
        st.error(f"检索失败: {results['error']}")
        return

    # 展示推理路径
    st.subheader("🧠 推理路径")
    
    path_data = {
        "1. 原始问题": results.get("original_query"),
        "1.5. 英文翻译": results.get("english_query"),
        "2. 解析后的意图": results.get("parsed_query"),
        "3. 生成的Cypher查询": results.get("graph_search_results", {}).get("cypher_query"),
        "4. 图谱检索结果": results.get("graph_search_results", {}).get("results"),
        "5. 精炼后用于LLM的查询": results.get("refined_query_for_llm"),
        "6. 最终答案": results.get("final_answer_and_sources", {}).get("answer")
    }

    # 使用expander来展示每一步的细节
    with st.expander("点击查看详细推理步骤", expanded=True):
        st.markdown(f"**1. 原始问题:**\n`{path_data['1. 原始问题']}`")
        st.markdown(f"**1.5. 英文翻译:**\n`{path_data['1.5. 英文翻译']}`")
        st.markdown("**2. 问题解析:**")
        st.json(path_data["2. 解析后的意图"])
        st.markdown(f"**3. 动态生成的Cypher查询:**")
        st.code(path_data["3. 生成的Cypher查询"], language="cypher")
        st.markdown("**4. 图谱返回的关键事实:**")
        st.json(path_data["4. 图谱检索结果"])
        st.markdown("**5. 用于指导LLM的精炼查询:**")
        st.info(path_data["5. 精炼后用于LLM的查询"])

    st.subheader("✅ 最终答案")
    st.markdown(path_data["6. 最终答案"])

    # 显示图谱可视化
    if path_data["4. 图谱检索结果"]:
        with st.expander("知识图谱可视化", expanded=True):
            generate_and_display_graph(path_data["4. 图谱检索结果"])
    
    # 显示源文档
    display_source_documents(results.get("final_answer_and_sources", {}))


def display_hybrid_results(results):
    """
    显示Hybrid Search的结果
    """
    st.success("并行融合检索完成！")
    
    # 最终答案
    st.subheader("✅ 最终答案")
    st.markdown(results.get("final_answer", "未能生成答案。"))

    # 两路结果对比
    st.subheader("🔍 双路检索结果对比")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📚 向量检索")
        vector_res = results.get("vector_results", {})
        st.metric("执行时间 (秒)", f"{vector_res.get('execution_time', 0):.2f}")
        with st.expander("查看向量检索详情", expanded=False):
            st.json(vector_res)

    with col2:
        st.markdown("#### 🕸️ 图谱检索")
        graph_res = results.get("graph_results", {})
        st.metric("发现实体数", graph_res.get("entities_found", 0))
        with st.expander("查看图谱检索详情", expanded=False):
            st.code(graph_res.get('cypher_query', ''), language='cypher')
            st.json(graph_res.get("relationships", []))
            
    # 新增：显示融合提示
    st.subheader("🤝 上下文融合提示")
    fusion_prompt = results.get("fusion_prompt", "未生成融合提示。")
    with st.expander("点击查看发送给LLM的最终融合提示", expanded=False):
        st.code(fusion_prompt, language='markdown')
        
    # 显示图谱可视化
    graph_rels = results.get("graph_results", {}).get("relationships")
    if graph_rels:
        st.subheader("🎨 知识图谱可视化")
        with st.expander("点击展开/折叠图谱", expanded=True):
            generate_and_display_graph(graph_rels)

    # 显示源文档
    display_source_documents(results.get("vector_results", {}))


def generate_and_display_graph(relationships: list):
    """
    使用pyvis生成并显示图谱。
    """
    if not relationships:
        st.warning("没有可供可视化的图谱数据。")
        return

    net = Network(height="450px", width="100%", bgcolor="#222222", font_color="white", notebook=True, cdn_resources='in_line')
    
    # 设置物理引擎以获得更好的布局
    net.set_options("""
    var options = {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 100,
          "springConstant": 0.08
        },
        "maxVelocity": 50,
        "minVelocity": 0.1,
        "solver": "forceAtlas2Based",
        "stabilization": {
          "enabled": true,
          "iterations": 1000
        }
      }
    }
    """)

    # 添加节点和边
    nodes = set()
    for rel in relationships:
        # 兼容多种结果格式
        source = rel.get("source") or rel.get("found_entity")
        target = rel.get("target") or rel.get("related_to")  # 新增对related_to的支持
        label = rel.get("relationship")

        # 如果还是没有target，尝试其他回退策略
        if not target and source:
            target = rel.get("source")  # 原来的回退逻辑

        if source and target and label:
            if source not in nodes:
                net.add_node(source, label=source, title=source)
                nodes.add(source)
            if target not in nodes:
                net.add_node(target, label=target, title=target)
                nodes.add(target)
            
            net.add_edge(source, target, label=label, title=label)

    # 生成并显示HTML
    try:
        path = 'tmp'
        import os
        if not os.path.exists(path):
            os.makedirs(path)
        
        file_path = os.path.join(path, 'graph.html')
        
        # Manually generate HTML and write with UTF-8 encoding to fix GBK error
        html_content = net.generate_html()
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        with open(file_path, 'r', encoding='utf-8') as f:
            html_data = f.read()
        
        components.html(html_data, height=470)
        
    except Exception as e:
        st.error(f"生成图谱可视化失败: {e}")


def display_source_documents(results_with_sources):
    """
    显示源文档信息,并利用新的metadata格式提供更丰富的引用。
    """
    source_documents = results_with_sources.get("source_documents")
    
    if source_documents:
        with st.expander("📚 查看引用的源文档"):
            for doc in source_documents:
                # Correctly access data from the dictionary
                metadata = doc.get('metadata', {})
                content = doc.get('content', '内容不可用')
                doc_index = doc.get('index', 0)

                # --- 构建丰富的引用信息 ---
                citation_parts = []
                
                # 作者
                author = metadata.get('pdf_author')
                if author and author != 'N/A':
                    citation_parts.append(f"**Author(s):** {author}")
                
                # 标题
                title = metadata.get('pdf_title')
                if title and title != 'N/A':
                    citation_parts.append(f"**Title:** *{title}*")
                
                # 文件来源和页码
                source_file = metadata.get('source_file', '未知来源')
                page_number = metadata.get('page_number', 'N/A')
                citation_parts.append(f"**Source:** `{source_file}` (Page: {page_number})")
                
                # 块类型
                chunk_type = metadata.get('chunk_type', 'text').capitalize()
                citation_parts.append(f"**Type:** {chunk_type}")

                # 最终引用字符串
                citation = " | ".join(citation_parts)
                st.markdown(citation)

                # --- 显示内容片段 ---
                st.text_area(
                    "内容片段:", 
                    content, 
                    height=150, 
                    key=f"doc_{doc_index}_{source_file}_{page_number}" # Improved key for uniqueness
                )
                st.divider()


if __name__ == "__main__":
    main()
