"""
增强心血管专科应用 - 展示完整的RAG过程和LangChain功能
"""
import os
import sys
import time
import json
import warnings
from datetime import datetime
from typing import List, Dict, Any

# 设置环境变量以避免兼容性问题
os.environ['STREAMLIT_SERVER_ENABLE_STATIC_SERVING'] = 'false'
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# 忽略警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import streamlit as st
import json
from cardio_specialist import CardiovascularSpecialist
from session_manager import SessionManager
from utils import format_medical_response, validate_medical_query, format_timestamp
from config import settings


def setup_page_config():
    """设置页面配置"""
    st.set_page_config(
        page_title="💓 心智医 - 增强心血管专科系统",
        page_icon="💓",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 自定义CSS
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }

    .rag-step {
        background: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }

    .thinking-step {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 0.8rem;
        margin: 0.3rem 0;
        border-radius: 5px;
    }

    .reference-card {
        background: #e7f3ff;
        border: 1px solid #b3d9ff;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
    }

    .langchain-feature {
        background: #f0f8f0;
        border: 1px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def initialize_enhanced_system():
    """初始化增强心血管专科系统"""
    try:
        # 检查心血管数据文件
        cardio_data_file = "cardiovascular_qa_data.json"
        if not os.path.exists(cardio_data_file):
            st.error("❌ 心血管数据文件不存在，请先运行: python analyze_huatuo_dataset.py cardiovascular")
            return None, None, False

        # 加载心血管数据
        with open(cardio_data_file, 'r', encoding='utf-8') as f:
            cardio_data = json.load(f)

        # 初始化心血管专科系统
        specialist = CardiovascularSpecialist()
        specialist.build_knowledge_base(cardio_data)

        # 初始化会话管理器
        session_manager = SessionManager()

        return specialist, session_manager, True
    except Exception as e:
        st.error(f"系统初始化失败: {e}")
        return None, None, False


def render_main_header():
    """渲染主标题"""
    st.markdown("""
    <div class="main-header">
        <h1>💓 心智医 - 增强心血管专科系统</h1>
        <p>集成RAG过程展示、LangChain功能、会话管理的智能医疗问答系统</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar(specialist, session_manager):
    """渲染侧边栏"""
    with st.sidebar:
        st.markdown("## 🎛️ 系统控制台")

        # 功能选择
        st.markdown("### 🔧 功能模式")
        mode = st.selectbox(
            "选择功能模式",
            ["💬 智能问答", "📊 RAG过程展示", "🔗 LangChain演示", "📝 会话管理", "📄 长文本摘要"],
            key="function_mode"
        )

        # RAG设置
        st.markdown("### ⚙️ RAG设置")
        use_enhanced_rag = st.checkbox("启用增强RAG", value=True, help="显示完整的检索和推理过程")
        show_thinking = st.checkbox("显示AI思考过程", value=True, help="展示DeepSeek的推理过程")
        show_references = st.checkbox("显示引用来源", value=True, help="展示检索到的参考资料")

        # 会话管理
        st.markdown("### 💾 会话管理")
        sessions = session_manager.get_sessions_list()

        if sessions:
            session_options = ["新建会话"] + [f"{s['title']} ({s['message_count']}条)" for s in sessions]
            selected_session = st.selectbox("选择会话", session_options)

            if selected_session != "新建会话":
                session_idx = session_options.index(selected_session) - 1
                if st.button("加载选中会话"):
                    session_manager.load_session(sessions[session_idx]['session_id'])
                    st.success("会话已加载")
                    st.rerun()

        # 系统统计
        st.markdown("### 📊 系统统计")
        stats = specialist.get_cardio_statistics()
        session_stats = session_manager.get_session_stats()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("知识库大小", f"{stats.get('knowledge_base_size', 0):,}")
            st.metric("会话总数", session_stats['total_sessions'])
        with col2:
            st.metric("关键词数", stats.get('total_keywords', 0))
            st.metric("消息总数", session_stats['total_messages'])

        return mode, use_enhanced_rag, show_thinking, show_references


def render_chat_interface(specialist, session_manager, use_enhanced_rag, show_thinking, show_references):
    """渲染聊天界面"""
    st.markdown("## 💬 智能心血管问答")

    # 显示当前会话信息
    if session_manager.current_session:
        st.info(f"当前会话: {session_manager.current_session.title}")
    else:
        if st.button("创建新会话"):
            session_manager.create_session()
            st.success("新会话已创建")
            st.rerun()

    # 聊天历史
    if session_manager.current_session and session_manager.current_session.messages:
        st.markdown("### 📜 聊天历史")
        for message in session_manager.current_session.messages[-5:]:  # 显示最近5条
            with st.chat_message(message.role):
                st.write(message.content)
                st.caption(f"时间: {message.timestamp.strftime('%H:%M:%S')}")

    # 用户输入
    user_input = st.chat_input("请输入您的心血管相关问题...")

    if user_input:
        # 添加用户消息
        session_manager.add_message("user", user_input)

        # 显示用户消息
        with st.chat_message("user"):
            st.write(user_input)

        # 处理问题
        with st.chat_message("assistant"):
            with st.spinner("🤔 正在思考..."):
                result = specialist.get_cardiovascular_answer(user_input, use_enhanced_rag=use_enhanced_rag)

            # 显示回答
            st.write(result['answer'])

            # 显示思考过程
            if show_thinking and 'thinking_steps' in result:
                render_thinking_process(result['thinking_steps'])

            # 显示RAG过程
            if 'rag_steps' in result:
                render_rag_process(result['rag_steps'])

            # 显示引用来源
            if show_references and result.get('references'):
                render_references(result['references'])

        # 保存助手回复
        session_manager.add_message("assistant", result['answer'], {
            "confidence": result.get('confidence', 0),
            "references_count": len(result.get('references', [])),
            "rag_enabled": use_enhanced_rag
        })


def render_thinking_process(thinking_steps):
    """渲染AI思考过程 - 优化展示"""
    if not thinking_steps:
        return

    with st.expander("🧠 DeepSeek AI思考过程", expanded=False):
        # 创建思考过程的时间线
        st.markdown("### 🔄 推理时间线")

        # 按步骤类型分组
        step_groups = {
            'analysis': [],
            'retrieval': [],
            'reasoning': [],
            'synthesis': []
        }

        for step in thinking_steps:
            step_type = step.step_type.lower()
            if step_type in step_groups:
                step_groups[step_type].append(step)

        # 渲染每个阶段
        phases = [
            ('analysis', '🔍 问题分析', '#e3f2fd'),
            ('retrieval', '📚 知识检索', '#f3e5f5'),
            ('reasoning', '🤔 逻辑推理', '#fff3e0'),
            ('synthesis', '✨ 答案合成', '#e8f5e8')
        ]

        for phase_key, phase_name, bg_color in phases:
            steps = step_groups.get(phase_key, [])
            if steps:
                st.markdown(f"""
                <div style="background-color: {bg_color}; padding: 1rem; margin: 0.5rem 0; border-radius: 8px; border-left: 4px solid #1976d2;">
                    <h4 style="margin: 0; color: #1976d2;">{phase_name}</h4>
                </div>
                """, unsafe_allow_html=True)

                for i, step in enumerate(steps, 1):
                    confidence_color = "#4caf50" if step.confidence > 0.7 else "#ff9800" if step.confidence > 0.4 else "#f44336"

                    st.markdown(f"""
                    <div style="margin-left: 1rem; padding: 0.8rem; background: white; border-radius: 6px; margin-bottom: 0.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <strong>步骤 {i}</strong>
                            <span style="background: {confidence_color}; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.8em;">
                                置信度: {step.confidence:.1%}
                            </span>
                        </div>
                        <p style="margin: 0.5rem 0; color: #333;">{step.content}</p>
                        {f'<small style="color: #666;">来源: {", ".join(step.sources)}</small>' if step.sources else ''}
                    </div>
                    """, unsafe_allow_html=True)


def render_rag_process(rag_steps):
    """渲染RAG检索过程 - 优化展示"""
    if not rag_steps:
        return

    with st.expander("🔍 RAG检索过程详情", expanded=False):
        st.markdown("### 📊 检索流程概览")

        # 创建流程图式的展示
        total_time = sum(step.duration for step in rag_steps)

        # 流程概览
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("总步骤", len(rag_steps))
        with col2:
            st.metric("总耗时", f"{total_time:.2f}s")
        with col3:
            retrieval_steps = [s for s in rag_steps if '检索' in s.step_name]
            st.metric("检索步骤", len(retrieval_steps))
        with col4:
            if retrieval_steps:
                avg_time = sum(s.duration for s in retrieval_steps) / len(retrieval_steps)
                st.metric("平均检索时间", f"{avg_time:.2f}s")

        st.markdown("---")
        st.markdown("### 🔄 详细流程")

        # 渲染每个步骤
        for i, step in enumerate(rag_steps, 1):
            # 计算进度百分比
            progress = (step.duration / total_time) * 100 if total_time > 0 else 0

            # 根据步骤类型选择图标和颜色
            if '分析' in step.step_name:
                icon = "🔍"
                color = "#2196f3"
            elif '检索' in step.step_name:
                icon = "📚"
                color = "#4caf50"
            elif '推理' in step.step_name:
                icon = "🤔"
                color = "#ff9800"
            elif '生成' in step.step_name:
                icon = "✨"
                color = "#9c27b0"
            else:
                icon = "⚙️"
                color = "#607d8b"

            st.markdown(f"""
            <div style="background: linear-gradient(90deg, {color}15 0%, {color}05 100%);
                        border-left: 4px solid {color};
                        padding: 1rem;
                        margin: 0.8rem 0;
                        border-radius: 8px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <h4 style="margin: 0; color: {color};">
                        {icon} 步骤 {i}: {step.step_name}
                    </h4>
                    <div style="text-align: right;">
                        <span style="background: {color}; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.8em;">
                            {step.duration:.2f}s
                        </span>
                        <br>
                        <small style="color: #666;">{step.timestamp.strftime('%H:%M:%S')}</small>
                    </div>
                </div>

                <div style="background: {color}20; height: 4px; border-radius: 2px; margin: 0.5rem 0;">
                    <div style="background: {color}; height: 100%; width: {min(progress, 100)}%; border-radius: 2px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # 显示步骤详情
            if step.metadata:
                with st.container():
                    if step.metadata.get('retrieved_count'):
                        st.success(f"✅ 检索到 {step.metadata['retrieved_count']} 条相关文档")

                    if step.metadata.get('strategies_used'):
                        strategies = step.metadata['strategies_used']
                        st.info(f"🎯 使用策略: {', '.join(strategies)}")

                    # 其他元数据
                    other_metadata = {k: v for k, v in step.metadata.items()
                                    if k not in ['retrieved_count', 'strategies_used']}
                    if other_metadata:
                        with st.expander(f"📋 {step.step_name} 详细信息"):
                            st.json(other_metadata)


def render_references(references):
    """渲染引用来源 - 优化展示"""
    if not references:
        return

    with st.expander("📚 知识库引用来源", expanded=False):
        st.markdown("### 📊 引用概览")

        # 引用统计
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("引用数量", len(references))
        with col2:
            avg_similarity = sum(ref.get('similarity', 0) for ref in references) / len(references)
            st.metric("平均相似度", f"{avg_similarity:.3f}")
        with col3:
            max_similarity = max(ref.get('similarity', 0) for ref in references)
            st.metric("最高相似度", f"{max_similarity:.3f}")

        st.markdown("---")
        st.markdown("### 📋 详细引用")

        for i, ref in enumerate(references, 1):
            similarity = ref.get('similarity', 0)
            strategy = ref.get('strategy', 'unknown')

            # 根据相似度设置颜色
            if similarity >= 0.8:
                similarity_color = "#4caf50"  # 绿色 - 高相似度
                similarity_label = "高度相关"
            elif similarity >= 0.6:
                similarity_color = "#2196f3"  # 蓝色 - 中等相似度
                similarity_label = "相关"
            elif similarity >= 0.4:
                similarity_color = "#ff9800"  # 橙色 - 低相似度
                similarity_label = "部分相关"
            else:
                similarity_color = "#f44336"  # 红色 - 很低相似度
                similarity_label = "弱相关"

            # 根据策略设置图标
            strategy_icons = {
                'original': '🎯',
                'keywords': '🔑',
                'enhanced': '⚡',
                'semantic': '🧠',
                'unknown': '❓'
            }
            strategy_icon = strategy_icons.get(strategy, '❓')

            st.markdown(f"""
            <div style="background: white;
                        border: 1px solid #e0e0e0;
                        border-radius: 12px;
                        padding: 1.2rem;
                        margin: 1rem 0;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                        border-left: 4px solid {similarity_color};">

                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                    <h4 style="margin: 0; color: #333;">
                        📄 引用 {i}
                    </h4>
                    <div style="display: flex; gap: 0.5rem; align-items: center;">
                        <span style="background: {similarity_color};
                                   color: white;
                                   padding: 4px 12px;
                                   border-radius: 16px;
                                   font-size: 0.8em;
                                   font-weight: bold;">
                            {similarity:.3f} - {similarity_label}
                        </span>
                        <span style="background: #f5f5f5;
                                   color: #666;
                                   padding: 4px 8px;
                                   border-radius: 12px;
                                   font-size: 0.8em;">
                            {strategy_icon} {strategy}
                        </span>
                    </div>
                </div>

                <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
                    <strong style="color: #495057;">❓ 原始问题:</strong>
                    <p style="margin: 0.5rem 0; color: #333; font-style: italic;">
                        "{ref.get('question', 'N/A')}"
                    </p>
                </div>

                <div style="background: #e8f5e8; padding: 1rem; border-radius: 8px;">
                    <strong style="color: #28a745;">💡 参考答案:</strong>
                    <p style="margin: 0.5rem 0; color: #333; line-height: 1.6;">
                        {ref.get('answer', 'N/A')[:300]}{'...' if len(ref.get('answer', '')) > 300 else ''}
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # 如果答案很长，提供展开选项
            if len(ref.get('answer', '')) > 300:
                with st.expander(f"📖 查看引用 {i} 完整内容"):
                    st.markdown(f"**完整问题:** {ref.get('question', 'N/A')}")
                    st.markdown(f"**完整答案:** {ref.get('answer', 'N/A')}")

                    # 显示其他元数据
                    if 'keywords' in ref:
                        st.markdown(f"**关键词:** {ref['keywords']}")
                    if 'source' in ref:
                        st.markdown(f"**来源:** {ref['source']}")

        # 引用质量分析
        st.markdown("---")
        st.markdown("### 📈 引用质量分析")

        high_quality = sum(1 for ref in references if ref.get('similarity', 0) >= 0.7)
        medium_quality = sum(1 for ref in references if 0.4 <= ref.get('similarity', 0) < 0.7)
        low_quality = sum(1 for ref in references if ref.get('similarity', 0) < 0.4)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("高质量引用", high_quality, help="相似度 ≥ 0.7")
        with col2:
            st.metric("中等质量引用", medium_quality, help="0.4 ≤ 相似度 < 0.7")
        with col3:
            st.metric("低质量引用", low_quality, help="相似度 < 0.4")

        # 质量建议
        if high_quality >= len(references) * 0.6:
            st.success("✅ 引用质量良好，检索结果高度相关")
        elif high_quality + medium_quality >= len(references) * 0.7:
            st.info("ℹ️ 引用质量中等，建议优化检索策略")
        else:
            st.warning("⚠️ 引用质量较低，可能需要调整问题描述或检索参数")


def render_langchain_demo(specialist):
    """渲染LangChain功能演示"""
    st.markdown("## 🔗 LangChain功能演示")

    # 功能选择
    langchain_feature = st.selectbox(
        "选择LangChain功能",
        ["提示词模板", "示例选择器", "任务链", "长文本摘要"]
    )

    if langchain_feature == "提示词模板":
        st.markdown("### 📝 LangChain提示词模板")
        st.markdown("""
        <div class="langchain-feature">
        <h4>功能说明</h4>
        <p>使用LangChain的PromptTemplate来构建结构化的提示词，确保输入变量的正确格式化。</p>
        </div>
        """, unsafe_allow_html=True)

        # 演示提示词模板
        question = st.text_input("输入问题", "高血压的症状有哪些？")
        context = st.text_area("输入上下文", "高血压是常见的心血管疾病...")

        if st.button("生成提示词"):
            from langchain.prompts import PromptTemplate

            template = PromptTemplate(
                input_variables=["question", "context"],
                template="""基于以下上下文回答问题：

上下文: {context}

问题: {question}

回答:"""
            )

            formatted_prompt = template.format(question=question, context=context)
            st.code(formatted_prompt, language="text")

    elif langchain_feature == "示例选择器":
        st.markdown("### 🎯 LangChain示例选择器")
        st.markdown("""
        <div class="langchain-feature">
        <h4>功能说明</h4>
        <p>使用SemanticSimilarityExampleSelector根据输入问题自动选择最相关的示例，实现Few-shot学习。</p>
        </div>
        """, unsafe_allow_html=True)

        query = st.text_input("输入查询", "心脏病的预防方法")

        if st.button("选择相关示例"):
            # 这里会调用specialist中的示例选择器
            st.info("示例选择器功能已集成在增强RAG系统中")

    elif langchain_feature == "任务链":
        st.markdown("### ⛓️ LangChain任务链")
        st.markdown("""
        <div class="langchain-feature">
        <h4>功能说明</h4>
        <p>使用SequentialChain将多个任务串联，实现复杂的工作流程：问题分析 → 信息检索 → 医学推理 → 答案生成。</p>
        </div>
        """, unsafe_allow_html=True)

        st.info("任务链功能已集成在增强RAG系统中，在智能问答模式下自动运行")

    elif langchain_feature == "长文本摘要":
        st.markdown("### 📄 LangChain长文本摘要")
        st.markdown("""
        <div class="langchain-feature">
        <h4>功能说明</h4>
        <p>使用LangChain的summarize_chain对长文本进行智能摘要，支持map-reduce策略。</p>
        </div>
        """, unsafe_allow_html=True)

        long_text = st.text_area("输入长文本", height=200, placeholder="输入需要摘要的长文本...")

        if st.button("生成摘要") and long_text:
            with st.spinner("正在生成摘要..."):
                if hasattr(specialist, 'enhanced_rag'):
                    summary = specialist.enhanced_rag.summarize_long_text(long_text)
                    st.success("摘要生成完成")
                    st.write(summary)
                else:
                    st.error("请先在智能问答模式下初始化增强RAG系统")


def render_session_management(session_manager):
    """渲染会话管理界面"""
    st.markdown("## 📝 会话管理")

    # 会话列表
    sessions = session_manager.get_sessions_list()

    if sessions:
        st.markdown("### 📋 会话列表")
        for session in sessions:
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

            with col1:
                st.write(f"**{session['title']}**")
                st.caption(f"创建: {session['created_at'][:16]} | 消息: {session['message_count']}条")

            with col2:
                if st.button("加载", key=f"load_{session['session_id']}"):
                    session_manager.load_session(session['session_id'])
                    st.success("会话已加载")
                    st.rerun()

            with col3:
                if st.button("导出", key=f"export_{session['session_id']}"):
                    exported = session_manager.export_session(session['session_id'], 'txt')
                    st.download_button(
                        "下载",
                        exported,
                        f"session_{session['session_id'][:8]}.txt",
                        key=f"download_{session['session_id']}"
                    )

            with col4:
                if st.button("删除", key=f"delete_{session['session_id']}"):
                    session_manager.delete_session(session['session_id'])
                    st.success("会话已删除")
                    st.rerun()

    else:
        st.info("暂无会话记录")

    # 搜索会话
    st.markdown("### 🔍 搜索会话")
    search_query = st.text_input("搜索关键词")

    if search_query:
        search_results = session_manager.search_sessions(search_query)
        if search_results:
            st.write(f"找到 {len(search_results)} 个相关会话:")
            for result in search_results:
                st.write(f"- {result['title']} ({result['match_type']})")
        else:
            st.write("未找到相关会话")


def main():
    """主函数"""
    setup_page_config()
    render_main_header()

    # 初始化系统
    specialist, session_manager, init_success = initialize_enhanced_system()

    if not init_success:
        st.error("❌ 系统初始化失败，请检查配置")
        return

    # 渲染侧边栏
    mode, use_enhanced_rag, show_thinking, show_references = render_sidebar(specialist, session_manager)

    # 根据模式渲染不同界面
    if mode == "💬 智能问答":
        render_chat_interface(specialist, session_manager, use_enhanced_rag, show_thinking, show_references)
    elif mode == "📊 RAG过程展示":
        st.info("RAG过程展示已集成在智能问答模式中，请启用'显示AI思考过程'和相关选项")
    elif mode == "🔗 LangChain演示":
        render_langchain_demo(specialist)
    elif mode == "📝 会话管理":
        render_session_management(session_manager)
    elif mode == "📄 长文本摘要":
        render_langchain_demo(specialist)  # 复用LangChain演示中的摘要功能


if __name__ == "__main__":
    main()
