"""
增强RAG系统 - 显示完整的检索和推理过程
集成LangChain功能
"""
import json
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass

from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.chains import LLMChain, SequentialChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains.summarize import load_summarize_chain

from enhanced_retriever import EnhancedRetriever
from llm_client import LLMClient
from config import settings


@dataclass
class RAGStep:
    """RAG步骤记录"""
    step_name: str
    timestamp: datetime
    input_data: Any
    output_data: Any
    duration: float
    metadata: Dict[str, Any] = None


@dataclass
class ThinkingStep:
    """思考步骤记录"""
    step_type: str  # analysis, retrieval, reasoning, synthesis
    content: str
    confidence: float
    sources: List[str] = None


class EnhancedRAGSystem:
    """增强RAG系统"""
    
    def __init__(self):
        self.retriever = EnhancedRetriever()
        self.llm_client = LLMClient()
        self.rag_steps = []
        self.thinking_steps = []
        
        # LangChain组件
        self.setup_langchain_components()
    
    def setup_langchain_components(self):
        """设置LangChain组件"""
        
        # 1. 基础提示词模板
        self.base_prompt = PromptTemplate(
            input_variables=["question", "context", "thinking_process"],
            template="""你是一个专业的心血管疾病AI助手。请基于以下信息回答问题：

问题: {question}

检索到的相关信息:
{context}

思考过程:
{thinking_process}

请提供专业、准确的回答，并说明你的推理过程：
"""
        )
        
        # 2. 示例选择器（用于Few-shot学习）
        self.setup_example_selector()
        
        # 3. 任务链
        self.setup_task_chains()
        
        # 4. 文本分割器（用于长文本摘要）
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
    
    def setup_example_selector(self):
        """设置示例选择器"""
        
        # 心血管问答示例
        examples = [
            {
                "question": "高血压的症状有哪些？",
                "context": "高血压常见症状包括头痛、头晕、心悸等",
                "answer": "高血压的主要症状包括：1. 头痛，特别是后脑勺痛；2. 头晕目眩；3. 心悸胸闷；4. 疲劳乏力。需要注意的是，很多高血压患者早期可能没有明显症状。"
            },
            {
                "question": "心肌梗死的急救措施是什么？",
                "context": "心肌梗死需要立即急救，包括呼叫120、服用硝酸甘油等",
                "answer": "心肌梗死的急救措施：1. 立即拨打120急救电话；2. 让患者平躺休息；3. 如有硝酸甘油可舌下含服；4. 保持呼吸道通畅；5. 不要随意搬动患者。"
            }
        ]
        
        # 创建示例选择器
        try:
            from langchain.vectorstores import FAISS
            from langchain.embeddings import HuggingFaceEmbeddings
            
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
            
            self.example_selector = SemanticSimilarityExampleSelector.from_examples(
                examples,
                embeddings,
                FAISS,
                k=2
            )
            
            # Few-shot提示词模板
            self.few_shot_prompt = FewShotPromptTemplate(
                example_selector=self.example_selector,
                example_prompt=PromptTemplate(
                    input_variables=["question", "context", "answer"],
                    template="问题: {question}\n上下文: {context}\n回答: {answer}"
                ),
                prefix="以下是一些心血管问答示例：",
                suffix="现在请回答新问题：\n问题: {question}\n上下文: {context}\n回答:",
                input_variables=["question", "context"]
            )
            
        except Exception as e:
            print(f"示例选择器初始化失败: {e}")
            self.example_selector = None
            self.few_shot_prompt = None
    
    def setup_task_chains(self):
        """设置任务链"""
        
        # 分析链 - 分析问题类型
        analysis_prompt = PromptTemplate(
            input_variables=["question"],
            template="""分析以下医学问题的类型和关键信息：

问题: {question}

请分析：
1. 问题类型（症状咨询、治疗建议、药物询问、检查解读等）
2. 涉及的医学领域
3. 紧急程度
4. 关键词提取

分析结果："""
        )
        
        # 检索链 - 基于分析结果检索
        retrieval_prompt = PromptTemplate(
            input_variables=["question", "analysis"],
            template="""基于问题分析结果，制定检索策略：

原问题: {question}
分析结果: {analysis}

请制定：
1. 检索关键词
2. 检索策略
3. 预期结果类型

检索策略："""
        )
        
        # 推理链 - 基于检索结果推理
        reasoning_prompt = PromptTemplate(
            input_variables=["question", "context", "analysis"],
            template="""基于检索到的信息进行医学推理：

问题: {question}
分析: {analysis}
检索信息: {context}

请进行推理：
1. 信息相关性评估
2. 医学逻辑推理
3. 答案可信度评估
4. 注意事项和免责声明

推理过程："""
        )
        
        try:
            # 创建链
            self.analysis_chain = LLMChain(
                llm=self.llm_client.get_langchain_llm(),
                prompt=analysis_prompt,
                output_key="analysis"
            )
            
            self.retrieval_chain = LLMChain(
                llm=self.llm_client.get_langchain_llm(),
                prompt=retrieval_prompt,
                output_key="retrieval_strategy"
            )
            
            self.reasoning_chain = LLMChain(
                llm=self.llm_client.get_langchain_llm(),
                prompt=reasoning_prompt,
                output_key="reasoning"
            )
            
            # 顺序链
            self.sequential_chain = SequentialChain(
                chains=[self.analysis_chain, self.reasoning_chain],
                input_variables=["question", "context"],
                output_variables=["analysis", "reasoning"],
                verbose=True
            )
            
        except Exception as e:
            print(f"任务链初始化失败: {e}")
            self.analysis_chain = None
            self.sequential_chain = None
    
    def record_rag_step(self, step_name: str, input_data: Any, output_data: Any, 
                       duration: float, metadata: Dict = None):
        """记录RAG步骤"""
        step = RAGStep(
            step_name=step_name,
            timestamp=datetime.now(),
            input_data=input_data,
            output_data=output_data,
            duration=duration,
            metadata=metadata or {}
        )
        self.rag_steps.append(step)
    
    def add_thinking_step(self, step_type: str, content: str, confidence: float, 
                         sources: List[str] = None):
        """添加思考步骤"""
        step = ThinkingStep(
            step_type=step_type,
            content=content,
            confidence=confidence,
            sources=sources or []
        )
        self.thinking_steps.append(step)
    
    def query_with_full_process(self, question: str) -> Dict[str, Any]:
        """带完整过程的查询"""
        start_time = time.time()
        self.rag_steps = []  # 重置步骤记录
        self.thinking_steps = []  # 重置思考记录
        
        try:
            # 步骤1: 问题分析
            analysis_start = time.time()
            self.add_thinking_step("analysis", f"开始分析问题: {question}", 1.0)
            
            if self.analysis_chain:
                analysis_result = self.analysis_chain.run(question=question)
                self.add_thinking_step("analysis", f"问题分析完成: {analysis_result}", 0.9)
            else:
                analysis_result = "基础分析：这是一个心血管相关问题"
            
            analysis_duration = time.time() - analysis_start
            self.record_rag_step("问题分析", question, analysis_result, analysis_duration)
            
            # 步骤2: 知识检索
            retrieval_start = time.time()
            self.add_thinking_step("retrieval", "开始检索相关知识", 1.0)
            
            retrieved_docs = self.retriever.search(question, top_k=8, debug=True)
            
            retrieval_duration = time.time() - retrieval_start
            self.record_rag_step("知识检索", question, retrieved_docs, retrieval_duration, {
                "retrieved_count": len(retrieved_docs),
                "strategies_used": list(set([doc.get('strategy', 'unknown') for doc in retrieved_docs]))
            })
            
            if retrieved_docs:
                self.add_thinking_step("retrieval", 
                    f"成功检索到 {len(retrieved_docs)} 条相关信息，最高相似度: {retrieved_docs[0].get('final_score', 0):.3f}", 
                    0.8, 
                    [doc.get('id', 'unknown') for doc in retrieved_docs[:3]]
                )
            else:
                self.add_thinking_step("retrieval", "未检索到相关信息", 0.1)
            
            # 步骤3: 上下文构建
            context_start = time.time()
            context = self.build_context(retrieved_docs)
            context_duration = time.time() - context_start
            self.record_rag_step("上下文构建", retrieved_docs, context, context_duration)
            
            # 步骤4: 推理过程
            reasoning_start = time.time()
            self.add_thinking_step("reasoning", "开始基于检索信息进行推理", 0.9)
            
            if self.reasoning_chain and context:
                reasoning_result = self.reasoning_chain.run(
                    question=question, 
                    context=context, 
                    analysis=analysis_result
                )
                self.add_thinking_step("reasoning", f"推理完成: {reasoning_result[:100]}...", 0.8)
            else:
                reasoning_result = "基于检索到的信息进行基础推理"
            
            reasoning_duration = time.time() - reasoning_start
            self.record_rag_step("医学推理", context, reasoning_result, reasoning_duration)
            
            # 步骤5: 答案生成
            generation_start = time.time()
            self.add_thinking_step("synthesis", "开始生成最终答案", 0.9)
            
            # 构建思考过程文本
            thinking_process = self.format_thinking_process()
            
            # 使用Few-shot提示词（如果可用）
            if self.few_shot_prompt and context:
                try:
                    prompt_text = self.few_shot_prompt.format(question=question, context=context)
                    final_answer = self.llm_client.generate_response(prompt_text)
                except Exception as e:
                    print(f"Few-shot生成失败，使用基础模板: {e}")
                    prompt_text = self.base_prompt.format(
                        question=question, 
                        context=context, 
                        thinking_process=thinking_process
                    )
                    final_answer = self.llm_client.generate_response(prompt_text)
            else:
                prompt_text = self.base_prompt.format(
                    question=question, 
                    context=context or "未找到相关信息", 
                    thinking_process=thinking_process
                )
                final_answer = self.llm_client.generate_response(prompt_text)
            
            generation_duration = time.time() - generation_start
            self.record_rag_step("答案生成", prompt_text, final_answer, generation_duration)
            
            self.add_thinking_step("synthesis", "答案生成完成", 0.9)
            
            total_duration = time.time() - start_time
            
            return {
                "answer": final_answer,
                "retrieved_docs": retrieved_docs,
                "rag_steps": self.rag_steps,
                "thinking_steps": self.thinking_steps,
                "analysis": analysis_result,
                "reasoning": reasoning_result,
                "context": context,
                "total_duration": total_duration,
                "metadata": {
                    "question": question,
                    "timestamp": datetime.now().isoformat(),
                    "retrieval_count": len(retrieved_docs),
                    "langchain_used": True
                }
            }
            
        except Exception as e:
            error_msg = f"查询处理失败: {e}"
            self.add_thinking_step("error", error_msg, 0.0)
            
            return {
                "answer": f"抱歉，处理您的问题时出现错误: {e}",
                "retrieved_docs": [],
                "rag_steps": self.rag_steps,
                "thinking_steps": self.thinking_steps,
                "error": str(e),
                "total_duration": time.time() - start_time
            }
    
    def build_context(self, retrieved_docs: List[Dict]) -> str:
        """构建上下文"""
        if not retrieved_docs:
            return ""
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs[:5], 1):
            metadata = doc.get('metadata', {})
            question = metadata.get('question', '')
            answer = metadata.get('answer', '')
            similarity = doc.get('final_score', doc.get('similarity', 0))
            
            context_parts.append(f"""
参考资料 {i} (相似度: {similarity:.3f}):
问题: {question}
答案: {answer}
""")
        
        return "\n".join(context_parts)
    
    def format_thinking_process(self) -> str:
        """格式化思考过程"""
        if not self.thinking_steps:
            return "无思考记录"
        
        formatted_steps = []
        for i, step in enumerate(self.thinking_steps, 1):
            sources_text = f" (来源: {', '.join(step.sources)})" if step.sources else ""
            formatted_steps.append(
                f"{i}. [{step.step_type.upper()}] {step.content} "
                f"(置信度: {step.confidence:.1%}){sources_text}"
            )
        
        return "\n".join(formatted_steps)
    
    def summarize_long_text(self, text: str) -> str:
        """使用LangChain总结长文本"""
        try:
            # 分割文本
            docs = [Document(page_content=text)]
            split_docs = self.text_splitter.split_documents(docs)
            
            # 加载摘要链
            summarize_chain = load_summarize_chain(
                llm=self.llm_client.get_langchain_llm(),
                chain_type="map_reduce"
            )
            
            # 生成摘要
            summary = summarize_chain.run(split_docs)
            
            return summary
            
        except Exception as e:
            return f"摘要生成失败: {e}"
