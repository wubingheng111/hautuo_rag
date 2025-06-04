"""
RAG系统核心模块
"""
from typing import List, Dict, Any, Optional
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.schema import BaseMessage, HumanMessage, AIMessage

from llm_client import DeepSeekLLM, DeepSeekChatClient
from embeddings import EmbeddingManager
from api_embeddings import APIEmbeddingManager
from config import settings

try:
    from langchain_selector import MedicalExampleSelector
    LANGCHAIN_SELECTOR_AVAILABLE = True
except ImportError:
    print("⚠️ LangChain示例选择器不可用")
    LANGCHAIN_SELECTOR_AVAILABLE = False

try:
    from cardio_specialist import CardiovascularSpecialist
    CARDIO_SPECIALIST_AVAILABLE = True
except ImportError:
    print("⚠️ 心血管专科模块不可用")
    CARDIO_SPECIALIST_AVAILABLE = False


class MedicalRAGSystem:
    """医学RAG系统"""

    def __init__(self, use_api_embedding: bool = False, api_provider: str = "huggingface", enable_langchain_selector: bool = True):
        self.llm_client = DeepSeekChatClient()

        # 选择向量化方式
        if use_api_embedding:
            print(f"🌐 使用API向量化: {api_provider}")
            self.embedding_manager = APIEmbeddingManager(api_provider)
        else:
            print("💻 使用本地向量化")
            self.embedding_manager = EmbeddingManager()

        self.chat_history = ChatMessageHistory()
        self.max_history_length = settings.MAX_HISTORY_LENGTH

        # 初始化LangChain示例选择器
        self.example_selector = None
        if enable_langchain_selector and LANGCHAIN_SELECTOR_AVAILABLE:
            try:
                self.example_selector = MedicalExampleSelector()
                print("✅ LangChain示例选择器已启用")
            except Exception as e:
                print(f"⚠️ LangChain示例选择器初始化失败: {e}")

        # 初始化心血管专科模块
        self.cardio_specialist = None
        if CARDIO_SPECIALIST_AVAILABLE:
            try:
                self.cardio_specialist = CardiovascularSpecialist()
                print("✅ 心血管专科模块已启用")
            except Exception as e:
                print(f"⚠️ 心血管专科模块初始化失败: {e}")

        self._setup_prompts()

    def _setup_prompts(self):
        """设置提示词模板"""

        # CoT推理提示词
        self.cot_prompt_template = """你是华佗医学AI助手，请使用Chain of Thought推理来回答医学问题。

## 参考资料
{context}

## 对话历史
{chat_history}

## 用户问题
{question}

## 推理要求
请按照以下步骤进行深度推理，并在回答中展示你的思考过程：

**第一步：问题理解与分析**
- 分析用户问题的核心要点
- 识别涉及的医学领域和关键概念
- 判断问题的复杂程度和紧急程度

**第二步：资料检索与评估**
- 评估参考资料的相关性和可信度
- 提取与问题直接相关的关键信息
- 识别资料中的重要医学事实和数据

**第三步：医学推理**
- 基于医学知识进行逻辑推理
- 考虑可能的诊断、治疗方案或解释
- 分析不同选项的优缺点和适用性

**第四步：综合判断**
- 结合参考资料和医学推理得出结论
- 考虑患者安全和最佳实践
- 提供个性化的建议和注意事项

请在回答中清晰地展示每个推理步骤，使用"🤔 **思考**："来标记你的推理过程。

## 我的推理和回答："""

        # 简化的RAG提示词（用于非CoT模式）
        self.rag_prompt_template = """你是华佗医学AI助手，基于权威医学知识库为用户提供准确的医学信息。

## 参考资料
{context}

## 对话历史
{chat_history}

## 用户问题
{question}

请基于参考资料提供专业、准确的医学回答，并提醒用户本回答仅供参考，具体诊疗请咨询专业医生。

## 我的回答："""

        # 摘要提示词
        self.summary_prompt_template = """请对以下医学文本进行摘要，提取关键信息：

原文：
{text}

请提供一个简洁明了的摘要，包含：
1. 主要医学概念
2. 关键信息点
3. 重要的诊疗建议

摘要："""

        # 任务链提示词
        self.chain_prompt_template = """作为医学AI助手，请按照以下步骤分析用户的医学问题：

用户问题：{question}
参考信息：{context}

请按以下步骤进行分析：
1. 问题理解：理解用户询问的医学问题
2. 信息检索：从参考信息中提取相关内容
3. 专业分析：基于医学知识进行分析
4. 建议提供：给出专业的医学建议
5. 注意事项：提醒相关注意事项

分析结果："""

    def query(
        self,
        question: str,
        use_history: bool = True,
        top_k: int = None,
        compare_mode: bool = False
    ) -> Dict[str, Any]:
        """
        RAG查询

        Args:
            question: 用户问题
            use_history: 是否使用对话历史
            top_k: 检索文档数量
            compare_mode: 是否启用对比模式（显示检索前后差异）

        Returns:
            查询结果
        """
        try:
            # 1. 生成无检索的基础回答（检索前）
            base_answer = ""
            if compare_mode:
                base_answer = self._generate_base_answer(question)

            # 2. 检索相关文档
            similar_docs = self.embedding_manager.search_similar(
                query=question,
                top_k=top_k or settings.TOP_K_RETRIEVAL
            )

            # 3. 构建上下文
            context = self._build_context(similar_docs)

            # 4. 获取对话历史
            chat_history = ""
            if use_history:
                chat_history = self._get_chat_history()

            # 5. 构建RAG提示词
            prompt = self.rag_prompt_template.format(
                context=context,
                chat_history=chat_history,
                question=question
            )

            # 6. 生成RAG增强回答（检索后）
            rag_answer = self.llm_client.generate_response(
                user_message=prompt,
                system_prompt="你是一个专业的医学AI助手。"
            )

            # 7. 更新对话历史
            if use_history:
                self.chat_history.add_user_message(question)
                self.chat_history.add_ai_message(rag_answer)

            result = {
                'question': question,
                'answer': rag_answer,
                'context': context,
                'retrieved_docs': similar_docs,
                'doc_count': len(similar_docs)
            }

            # 8. 如果是对比模式，添加对比信息
            if compare_mode:
                result.update({
                    'base_answer': base_answer,
                    'rag_answer': rag_answer,
                    'comparison': self._compare_answers(base_answer, rag_answer, similar_docs)
                })

            return result

        except Exception as e:
            print(f"RAG查询失败: {e}")
            return {
                'question': question,
                'answer': f"抱歉，处理您的问题时出现错误：{str(e)}",
                'context': "",
                'retrieved_docs': [],
                'doc_count': 0
            }

    def _build_context(self, similar_docs: List[Dict[str, Any]]) -> str:
        """构建上下文"""
        if not similar_docs:
            return "未找到相关的医学信息。"

        # 过滤低相关性文档
        high_quality_docs = [
            doc for doc in similar_docs
            if doc['similarity'] >= settings.SIMILARITY_THRESHOLD
        ]

        if not high_quality_docs:
            return "未找到足够相关的医学信息。"

        context_parts = []
        for i, doc in enumerate(high_quality_docs, 1):
            # 添加更结构化的上下文，包含文档ID用于跳转
            doc_id = doc.get('id', f"doc_{i}")
            source = doc.get('metadata', {}).get('source', '华佗医学知识库')
            dataset_info = doc.get('metadata', {}).get('dataset_info', '')

            context_parts.append(
                f"【参考资料{i}】（文档ID：{doc_id}，相关度：{doc['similarity']:.3f}）\n"
                f"内容：{doc['text']}\n"
                f"来源：{source}\n"
                f"数据集：{dataset_info}\n"
            )

        return "\n".join(context_parts)

    def get_document_details(self, doc_id: str) -> Dict[str, Any]:
        """
        获取文档详细信息

        Args:
            doc_id: 文档ID

        Returns:
            文档详细信息
        """
        try:
            # 从向量数据库获取文档详情
            doc_details = self.embedding_manager.get_document_by_id(doc_id)

            if doc_details:
                return {
                    'id': doc_id,
                    'content': doc_details.get('text', ''),
                    'metadata': doc_details.get('metadata', {}),
                    'source': doc_details.get('metadata', {}).get('source', '华佗医学知识库'),
                    'dataset_info': doc_details.get('metadata', {}).get('dataset_info', ''),
                    'question': doc_details.get('metadata', {}).get('question', ''),
                    'answer': doc_details.get('metadata', {}).get('answer', ''),
                    'found': True
                }
            else:
                return {
                    'id': doc_id,
                    'found': False,
                    'error': '文档未找到'
                }

        except Exception as e:
            return {
                'id': doc_id,
                'found': False,
                'error': f'获取文档详情失败: {str(e)}'
            }

    def _get_chat_history(self) -> str:
        """获取对话历史"""
        try:
            messages = self.chat_history.messages
            if not messages:
                return "无对话历史"

            history_parts = []
            # 获取最近的对话（限制数量）
            recent_messages = messages[-(self.max_history_length * 2):]  # 每轮对话包含用户和助手消息

            for msg in recent_messages:
                if isinstance(msg, HumanMessage):
                    role = "用户"
                elif isinstance(msg, AIMessage):
                    role = "助手"
                else:
                    role = "系统"
                history_parts.append(f"{role}: {msg.content}")

            return "\n".join(history_parts)
        except Exception as e:
            print(f"获取对话历史失败: {e}")
            return "无对话历史"

    def summarize_text(self, text: str) -> str:
        """
        文本摘要功能

        Args:
            text: 需要摘要的文本

        Returns:
            摘要结果
        """
        try:
            prompt = self.summary_prompt_template.format(text=text)
            summary = self.llm_client.generate_response(
                user_message=prompt,
                system_prompt="你是一个专业的医学文本摘要助手。"
            )
            return summary
        except Exception as e:
            return f"摘要生成失败：{str(e)}"

    def chain_analysis(self, question: str) -> str:
        """
        任务链分析

        Args:
            question: 用户问题

        Returns:
            分析结果
        """
        try:
            # 检索相关信息
            similar_docs = self.embedding_manager.search_similar(question)
            context = self._build_context(similar_docs)

            # 使用任务链提示词
            prompt = self.chain_prompt_template.format(
                question=question,
                context=context
            )

            analysis = self.llm_client.generate_response(
                user_message=prompt,
                system_prompt="你是一个专业的医学分析助手，请按步骤进行分析。"
            )
            return analysis
        except Exception as e:
            return f"任务链分析失败：{str(e)}"

    def query_with_cot_reasoning(
        self,
        question: str,
        use_history: bool = True,
        top_k: int = None
    ) -> Dict[str, Any]:
        """
        使用Chain of Thought推理的RAG查询

        Args:
            question: 用户问题
            use_history: 是否使用对话历史
            top_k: 检索文档数量

        Returns:
            包含推理过程的查询结果
        """
        try:
            # 1. 检索相关文档
            similar_docs = self.embedding_manager.search_similar(
                query=question,
                top_k=top_k or settings.TOP_K_RETRIEVAL
            )

            # 2. 构建上下文
            context = self._build_context(similar_docs)

            # 3. 获取对话历史
            chat_history = ""
            if use_history:
                chat_history = self._get_chat_history()

            # 4. 使用专门的CoT推理提示
            cot_reasoning_prompt = f"""你是华佗医学AI助手，请对以下医学问题进行深度Chain of Thought推理。

## 参考资料
{context}

## 用户问题
{question}

## 推理要求
请按照以下格式进行推理，每个步骤都要详细展示你的思考过程：

🤔 **第一步：问题分析**
[详细分析用户问题的核心要点、涉及的医学领域]

🤔 **第二步：资料评估**
[评估参考资料的相关性，提取关键医学信息]

🤔 **第三步：医学推理**
[基于医学知识进行逻辑推理，考虑可能的解释或方案]

🤔 **第四步：综合判断**
[结合所有信息得出最终结论和建议]

💡 **最终回答**
[基于推理过程给出专业的医学回答]

请开始你的推理："""

            # 5. 生成推理回答
            reasoning_response = self.llm_client.generate_response(
                user_message=cot_reasoning_prompt,
                system_prompt="你是一个专业的医学AI助手，请详细展示你的Chain of Thought推理过程。"
            )

            # 6. 更新对话历史
            if use_history:
                self.chat_history.add_user_message(question)
                self.chat_history.add_ai_message(reasoning_response)

            return {
                'question': question,
                'reasoning_response': reasoning_response,
                'context': context,
                'retrieved_docs': similar_docs,
                'doc_count': len(similar_docs),
                'reasoning_type': 'Chain of Thought'
            }

        except Exception as e:
            error_msg = f"抱歉，推理过程中出现错误：{str(e)}"
            return {
                'question': question,
                'reasoning_response': error_msg,
                'context': "",
                'retrieved_docs': [],
                'doc_count': 0,
                'reasoning_type': 'Error'
            }

    def query_with_langchain_selector(
        self,
        question: str,
        use_history: bool = True,
        top_k: int = None
    ) -> Dict[str, Any]:
        """
        使用LangChain示例选择器的RAG查询

        Args:
            question: 用户问题
            use_history: 是否使用对话历史
            top_k: 检索文档数量

        Returns:
            包含示例选择过程的查询结果
        """
        try:
            if not self.example_selector:
                return self.query(question, use_history, top_k)

            # 1. 分析示例选择过程
            selection_analysis = self.example_selector.analyze_selection_process(question)

            # 2. 生成包含示例的提示词
            few_shot_prompt = self.example_selector.generate_prompt(question)

            # 3. 检索相关文档
            similar_docs = self.embedding_manager.search_similar(
                query=question,
                top_k=top_k or settings.TOP_K_RETRIEVAL
            )

            # 4. 构建上下文
            context = self._build_context(similar_docs)

            # 5. 获取对话历史
            chat_history = ""
            if use_history:
                chat_history = self._get_chat_history()

            # 6. 结合Few-Shot提示和RAG上下文
            enhanced_prompt = f"""
{few_shot_prompt}

参考资料：
{context}

对话历史：
{chat_history}

请基于以上示例和参考资料，详细回答用户的问题。
"""

            # 7. 生成回答
            answer = self.llm_client.generate_response(
                user_message=enhanced_prompt,
                system_prompt="你是华佗医学AI助手，请参考示例和资料提供专业回答。"
            )

            # 8. 更新对话历史
            if use_history:
                self.chat_history.add_user_message(question)
                self.chat_history.add_ai_message(answer)

            return {
                'question': question,
                'answer': answer,
                'context': context,
                'retrieved_docs': similar_docs,
                'doc_count': len(similar_docs),
                'selection_analysis': selection_analysis,
                'few_shot_prompt': few_shot_prompt,
                'method': 'LangChain示例选择器 + RAG'
            }

        except Exception as e:
            error_msg = f"抱歉，使用示例选择器查询时出现错误：{str(e)}"
            return {
                'question': question,
                'answer': error_msg,
                'context': "",
                'retrieved_docs': [],
                'doc_count': 0,
                'selection_analysis': {},
                'few_shot_prompt': "",
                'method': 'Error'
            }

    def analyze_example_selection(self, question: str) -> Dict[str, Any]:
        """
        分析示例选择过程

        Args:
            question: 用户问题

        Returns:
            详细的选择过程分析
        """
        if not self.example_selector:
            return {"error": "示例选择器未启用"}

        return self.example_selector.analyze_selection_process(question)

    def add_custom_example(self, question: str, thinking: str, answer: str):
        """
        添加自定义示例

        Args:
            question: 问题
            thinking: 思考过程
            answer: 答案
        """
        if self.example_selector:
            self.example_selector.add_example(question, thinking, answer)
        else:
            print("⚠️ 示例选择器未启用")

    def query_with_stream(
        self,
        question: str,
        use_history: bool = True,
        top_k: int = None
    ):
        """
        流式RAG查询

        Args:
            question: 用户问题
            use_history: 是否使用对话历史
            top_k: 检索文档数量

        Yields:
            流式回复的文本片段和元数据
        """
        try:
            # 1. 检索相关文档
            similar_docs = self.embedding_manager.search_similar(
                query=question,
                top_k=top_k or settings.TOP_K_RETRIEVAL
            )

            # 2. 构建上下文
            context = self._build_context(similar_docs)

            # 3. 获取对话历史
            chat_history = ""
            if use_history:
                chat_history = self._get_chat_history()

            # 4. 构建完整上下文
            full_context = ""
            if context:
                full_context += f"参考资料：\n{context}\n\n"
            if chat_history:
                full_context += f"对话历史：\n{chat_history}\n\n"

            # 5. 先返回元数据
            yield {
                'type': 'metadata',
                'question': question,
                'doc_count': len(similar_docs),
                'retrieved_docs': similar_docs,
                'context': context
            }

            # 6. 流式生成回答
            full_answer = ""
            for chunk in self.llm_client.generate_response_stream(
                user_message=question,
                context=full_context,
                system_prompt="你是华佗医学AI助手，请基于提供的参考资料进行专业的医学回答。"
            ):
                full_answer += chunk
                yield {
                    'type': 'content',
                    'chunk': chunk,
                    'full_content': full_answer
                }

            # 7. 更新对话历史
            if use_history:
                self.chat_history.add_user_message(question)
                self.chat_history.add_ai_message(full_answer)

            # 8. 返回完成信号
            yield {
                'type': 'complete',
                'final_answer': full_answer,
                'method': '流式RAG查询'
            }

        except Exception as e:
            yield {
                'type': 'error',
                'error': str(e),
                'message': f"抱歉，流式查询过程中出现错误：{str(e)}"
            }

    def query_cardiovascular_specialist(
        self,
        question: str,
        use_history: bool = True,
        top_k: int = None
    ) -> Dict[str, Any]:
        """
        心血管专科查询

        Args:
            question: 用户问题
            use_history: 是否使用对话历史
            top_k: 检索文档数量

        Returns:
            心血管专科查询结果
        """
        try:
            if not self.cardio_specialist:
                return self.query(question, use_history, top_k)

            # 1. 心血管相关性分析
            cardio_analysis = self.cardio_specialist.is_cardiovascular_related(question)

            # 2. 急症评估
            emergency_analysis = self.cardio_specialist.assess_emergency_level(question)

            # 3. 风险因素分析
            risk_analysis = self.cardio_specialist.analyze_risk_factors(question)

            # 4. 检索相关文档（优先心血管相关）
            cardio_query = question
            if cardio_analysis['is_cardiovascular']:
                # 增强查询词，提高心血管相关文档的检索精度
                top_keywords = [kw for kw_list in cardio_analysis['matched_keywords'].values() for kw in kw_list]
                if top_keywords:
                    cardio_query = f"{question} {' '.join(top_keywords[:3])}"

            similar_docs = self.embedding_manager.search_similar(
                query=cardio_query,
                top_k=top_k or settings.TOP_K_RETRIEVAL
            )

            # 5. 构建上下文
            context = self._build_context(similar_docs)

            # 6. 获取对话历史
            chat_history = ""
            if use_history:
                chat_history = self._get_chat_history()

            # 7. 生成心血管专科提示词
            specialist_prompt = self.cardio_specialist.generate_cardio_prompt(question)

            # 8. 构建完整提示
            full_prompt = f"""
{specialist_prompt}

参考资料：
{context}

对话历史：
{chat_history}

请基于以上信息，作为心血管专科AI助手回答用户问题。
"""

            # 9. 生成回答
            answer = self.llm_client.generate_response(
                user_message=full_prompt,
                system_prompt="你是心智医AI，专业的心血管疾病智能问答助手。"
            )

            # 10. 更新对话历史
            if use_history:
                self.chat_history.add_user_message(question)
                self.chat_history.add_ai_message(answer)

            return {
                'question': question,
                'answer': answer,
                'context': context,
                'retrieved_docs': similar_docs,
                'doc_count': len(similar_docs),
                'cardio_analysis': cardio_analysis,
                'emergency_analysis': emergency_analysis,
                'risk_analysis': risk_analysis,
                'specialist_prompt': specialist_prompt,
                'method': '心血管专科RAG查询',
                'rag_process': {
                    'original_query': question,
                    'enhanced_query': cardio_query,
                    'retrieval_method': 'semantic_similarity',
                    'knowledge_base': 'huatuo_medical_qa',
                    'embedding_model': 'sentence-transformers',
                    'vector_db': 'chromadb',
                    'retrieval_time': 'real-time',
                    'context_length': len(context),
                    'fusion_method': 'context_injection'
                }
            }

        except Exception as e:
            error_msg = f"抱歉，心血管专科查询时出现错误：{str(e)}"
            return {
                'question': question,
                'answer': error_msg,
                'context': "",
                'retrieved_docs': [],
                'doc_count': 0,
                'cardio_analysis': {},
                'emergency_analysis': {},
                'risk_analysis': {},
                'specialist_prompt': "",
                'method': 'Error'
            }

    def get_cardio_statistics(self) -> Dict[str, Any]:
        """获取心血管专科统计信息"""
        if self.cardio_specialist:
            return self.cardio_specialist.get_cardio_statistics()
        else:
            return {"error": "心血管专科模块未启用"}

    def clear_history(self):
        """清空对话历史"""
        self.chat_history.clear()

    def _preprocess_question(self, question: str) -> str:
        """预处理用户问题"""
        # 去除多余空格
        question = question.strip()

        # 添加医学相关关键词提取
        medical_keywords = ['症状', '疾病', '治疗', '药物', '诊断', '病因', '预防']

        # 如果问题太短，建议用户提供更多信息
        if len(question) < 5:
            return question + "（建议提供更详细的描述以获得更准确的回答）"

        return question

    def _postprocess_answer(self, answer: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """后处理回答"""
        # 确保回答包含安全提醒
        safety_reminder = "\n\n⚠️ **重要提醒**：本回答仅供参考，不能替代专业医生的诊断和治疗建议。如有健康问题，请及时就医咨询专业医生。"

        if "仅供参考" not in answer and "咨询医生" not in answer:
            answer += safety_reminder

        # 如果有参考资料，添加资料来源说明
        if retrieved_docs:
            answer += f"\n\n📚 **参考资料**：基于{len(retrieved_docs)}条华佗医学知识库资料"

        return answer

    def query_with_native_cot(
        self,
        question: str,
        use_history: bool = True,
        top_k: int = None
    ) -> Dict[str, Any]:
        """
        使用DeepSeek原生CoT推理的RAG查询

        Args:
            question: 用户问题
            use_history: 是否使用对话历史
            top_k: 检索文档数量

        Returns:
            包含原生CoT推理过程的查询结果
        """
        try:
            # 1. 检索相关文档
            similar_docs = self.embedding_manager.search_similar(
                query=question,
                top_k=top_k or settings.TOP_K_RETRIEVAL
            )

            # 2. 构建上下文
            context = self._build_context(similar_docs)

            # 3. 获取对话历史
            chat_history = ""
            if use_history:
                chat_history = self._get_chat_history()

            # 4. 构建完整上下文
            full_context = ""
            if context:
                full_context += f"参考资料：\n{context}\n\n"
            if chat_history:
                full_context += f"对话历史：\n{chat_history}\n\n"

            # 5. 使用DeepSeek原生CoT推理
            cot_result = self.llm_client.generate_response_with_cot(
                user_message=question,
                context=full_context,
                system_prompt="你是华佗医学AI助手，请基于提供的参考资料进行专业的医学回答。"
            )

            # 6. 更新对话历史
            if use_history:
                self.chat_history.add_user_message(question)
                self.chat_history.add_ai_message(cot_result['final_answer'])

            return {
                'question': question,
                'thinking_process': cot_result['thinking_process'],
                'answer': cot_result['final_answer'],
                'full_response': cot_result['full_response'],
                'has_explicit_cot': cot_result['has_explicit_cot'],
                'context': context,
                'retrieved_docs': similar_docs,
                'doc_count': len(similar_docs),
                'method': 'DeepSeek原生CoT推理'
            }

        except Exception as e:
            error_msg = f"抱歉，CoT推理过程中出现错误：{str(e)}"
            return {
                'question': question,
                'thinking_process': f"推理过程出现错误：{str(e)}",
                'answer': error_msg,
                'full_response': error_msg,
                'has_explicit_cot': False,
                'context': "",
                'retrieved_docs': [],
                'doc_count': 0,
                'method': 'Error'
            }

    def query_with_thinking_process(
        self,
        question: str,
        use_history: bool = True,
        top_k: int = None,
        show_thinking: bool = True
    ) -> Dict[str, Any]:
        """
        带思考过程的RAG查询

        Args:
            question: 用户问题
            use_history: 是否使用对话历史
            top_k: 检索文档数量
            show_thinking: 是否显示思考过程

        Returns:
            包含思考过程的查询结果
        """
        thinking_steps = []

        try:
            # 步骤1: 问题分析
            if show_thinking:
                thinking_steps.append("🤔 正在分析您的问题...")

            # 步骤2: 知识检索
            if show_thinking:
                thinking_steps.append("📚 正在检索相关医学知识...")

            similar_docs = self.embedding_manager.search_similar(
                query=question,
                top_k=top_k or settings.TOP_K_RETRIEVAL
            )

            if show_thinking:
                thinking_steps.append(f"✅ 找到 {len(similar_docs)} 条相关资料")

            # 步骤3: 构建上下文
            if show_thinking:
                thinking_steps.append("🔗 正在整理参考信息...")

            context = self._build_context(similar_docs)

            # 步骤4: 获取对话历史
            chat_history = ""
            if use_history:
                if show_thinking:
                    thinking_steps.append("💭 正在回顾对话历史...")
                chat_history = self._get_chat_history()

            # 步骤5: CoT推理生成回答
            if show_thinking:
                thinking_steps.append("🧠 正在进行Chain of Thought推理...")

            # 使用CoT推理提示词
            cot_prompt = self.cot_prompt_template.format(
                context=context,
                chat_history=chat_history,
                question=question
            )

            # 生成包含推理过程的回答
            answer = self.llm_client.generate_response(
                user_message=cot_prompt,
                system_prompt="你是一个专业的医学AI助手，请展示你的推理思考过程。"
            )

            if show_thinking:
                thinking_steps.append("✨ 回答生成完成！")

            # 更新对话历史
            if use_history:
                self.chat_history.add_user_message(question)
                self.chat_history.add_ai_message(answer)

            return {
                'question': question,
                'answer': answer,
                'context': context,
                'retrieved_docs': similar_docs,
                'doc_count': len(similar_docs),
                'thinking_steps': thinking_steps
            }

        except Exception as e:
            error_msg = f"抱歉，处理您的问题时出现错误：{str(e)}"
            thinking_steps.append(f"❌ 处理过程中出现错误: {str(e)}")

            return {
                'question': question,
                'answer': error_msg,
                'context': "",
                'retrieved_docs': [],
                'doc_count': 0,
                'thinking_steps': thinking_steps
            }

    def query_with_comparison(
        self,
        question: str,
        use_history: bool = True,
        top_k: int = None
    ) -> Dict[str, Any]:
        """
        带对比的RAG查询 - 展示检索前后的差异

        Args:
            question: 用户问题
            use_history: 是否使用对话历史
            top_k: 检索文档数量

        Returns:
            包含对比信息的查询结果
        """
        try:
            # 1. 生成无检索的基础回答（检索前）
            print("🔍 正在生成基础回答（无知识检索）...")
            base_prompt = f"""你是一个医学AI助手，请仅基于你的基础知识回答以下问题，不要使用任何外部参考资料：

用户问题：{question}

请提供你的回答："""

            base_answer = self.llm_client.generate_response(
                user_message=base_prompt,
                system_prompt="你是一个医学AI助手，仅使用基础医学知识回答问题。"
            )

            # 2. 检索相关文档
            print("📚 正在检索相关医学知识...")
            similar_docs = self.embedding_manager.search_similar(
                query=question,
                top_k=top_k or settings.TOP_K_RETRIEVAL
            )

            # 3. 构建上下文
            context = self._build_context(similar_docs)

            # 4. 获取对话历史
            chat_history = ""
            if use_history:
                chat_history = self._get_chat_history()

            # 5. 构建RAG提示词
            rag_prompt = self.rag_prompt_template.format(
                context=context,
                chat_history=chat_history,
                question=question
            )

            # 6. 生成RAG增强回答（检索后）
            print("🧠 正在生成RAG增强回答（基于检索知识）...")
            rag_answer = self.llm_client.generate_response(
                user_message=rag_prompt,
                system_prompt="你是一个专业的医学AI助手，基于提供的参考信息回答问题。"
            )

            # 7. 生成对比分析
            comparison_analysis = self._generate_comparison_analysis(
                question, base_answer, rag_answer, similar_docs
            )

            # 8. 更新对话历史
            if use_history:
                self.chat_history.add_user_message(question)
                self.chat_history.add_ai_message(rag_answer)

            return {
                'question': question,
                'base_answer': base_answer,
                'rag_answer': rag_answer,
                'final_answer': rag_answer,  # 最终采用RAG答案
                'context': context,
                'retrieved_docs': similar_docs,
                'doc_count': len(similar_docs),
                'comparison_analysis': comparison_analysis,
                'improvement_summary': self._summarize_improvements(base_answer, rag_answer, similar_docs)
            }

        except Exception as e:
            print(f"RAG对比查询失败: {e}")
            return {
                'question': question,
                'base_answer': f"基础回答生成失败：{str(e)}",
                'rag_answer': f"RAG回答生成失败：{str(e)}",
                'final_answer': f"抱歉，处理您的问题时出现错误：{str(e)}",
                'context': "",
                'retrieved_docs': [],
                'doc_count': 0,
                'comparison_analysis': "对比分析失败",
                'improvement_summary': "改进总结失败"
            }

    def _generate_comparison_analysis(
        self,
        question: str,
        base_answer: str,
        rag_answer: str,
        retrieved_docs: List[Dict[str, Any]]
    ) -> str:
        """生成对比分析"""

        analysis_prompt = f"""请分析以下两个医学回答的差异和改进：

原始问题：{question}

基础回答（无知识检索）：
{base_answer}

RAG增强回答（基于检索知识）：
{rag_answer}

检索到的参考文档数量：{len(retrieved_docs)}

请从以下几个方面进行对比分析：
1. 准确性：哪个回答更准确？
2. 完整性：哪个回答更全面？
3. 专业性：哪个回答更专业？
4. 具体性：哪个回答提供了更具体的信息？
5. 可信度：哪个回答更可信？

分析结果："""

        try:
            analysis = self.llm_client.generate_response(
                user_message=analysis_prompt,
                system_prompt="你是一个医学专家，请客观分析两个回答的差异。"
            )
            return analysis
        except Exception as e:
            return f"对比分析生成失败：{str(e)}"

    def _summarize_improvements(
        self,
        base_answer: str,
        rag_answer: str,
        retrieved_docs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """总结RAG带来的改进"""

        improvements = {
            'knowledge_enhancement': len(retrieved_docs) > 0,
            'doc_count': len(retrieved_docs),
            'answer_length_change': len(rag_answer) - len(base_answer),
            'has_references': len(retrieved_docs) > 0,
            'confidence_boost': len(retrieved_docs) > 0
        }

        # 计算相似度（简单版本）
        base_words = set(base_answer.split())
        rag_words = set(rag_answer.split())

        improvements['content_overlap'] = len(base_words.intersection(rag_words)) / len(base_words.union(rag_words)) if base_words.union(rag_words) else 0
        improvements['new_content_ratio'] = len(rag_words - base_words) / len(rag_words) if rag_words else 0

        return improvements

    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        embedding_stats = self.embedding_manager.get_collection_stats()

        return {
            'vector_db_stats': embedding_stats,
            'memory_length': len(self.chat_history.messages),
            'model_info': {
                'llm_model': settings.DEEPSEEK_MODEL,
                'embedding_model': settings.EMBEDDING_MODEL
            }
        }
