"""
心血管疾病专科模块
专门处理心血管相关的医学问答
基于华佗数据集的心血管专项数据
"""
import re
import json
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class CardiovascularRiskFactor:
    """心血管风险因素"""
    name: str
    category: str  # 'modifiable' or 'non_modifiable'
    severity: str  # 'low', 'medium', 'high'
    description: str

class CardiovascularSpecialist:
    """心血管疾病专科处理器"""

    def __init__(self):
        self.cardio_keywords = self._load_cardio_keywords()
        self.risk_factors = self._load_risk_factors()
        self.emergency_keywords = self._load_emergency_keywords()
        self.knowledge_base = None
        self.vector_store = None

    def _load_cardio_keywords(self) -> Dict[str, List[str]]:
        """加载心血管关键词库"""
        return {
            "疾病": [
                "冠心病", "心肌梗死", "心绞痛", "高血压", "低血压",
                "心律不齐", "心房颤动", "心力衰竭", "心肌病", "心包炎",
                "动脉硬化", "动脉瘤", "静脉曲张", "血栓", "栓塞",
                "心脏病", "先心病", "风心病", "肺心病"
            ],
            "症状": [
                "胸痛", "胸闷", "心悸", "气短", "呼吸困难",
                "头晕", "晕厥", "水肿", "乏力", "心慌",
                "胸部不适", "心跳快", "心跳慢", "心跳不规律"
            ],
            "检查": [
                "心电图", "心脏彩超", "冠脉造影", "心肌酶", "肌钙蛋白",
                "血压", "血脂", "心率", "动态心电图", "运动试验",
                "CT血管造影", "核磁共振", "心导管"
            ],
            "药物": [
                "降压药", "硝酸甘油", "阿司匹林", "他汀", "β受体阻滞剂",
                "ACE抑制剂", "ARB", "利尿剂", "钙通道阻滞剂",
                "抗凝药", "抗血小板", "强心药", "抗心律失常药"
            ],
            "治疗": [
                "支架", "搭桥", "球囊扩张", "起搏器", "除颤器",
                "射频消融", "心脏移植", "介入治疗", "手术治疗"
            ]
        }

    def _load_risk_factors(self) -> List[CardiovascularRiskFactor]:
        """加载心血管风险因素"""
        return [
            # 不可控因素
            CardiovascularRiskFactor("年龄", "non_modifiable", "high", "男性≥45岁，女性≥55岁"),
            CardiovascularRiskFactor("性别", "non_modifiable", "medium", "男性风险高于女性"),
            CardiovascularRiskFactor("家族史", "non_modifiable", "high", "直系亲属有早发心血管疾病"),

            # 可控因素
            CardiovascularRiskFactor("吸烟", "modifiable", "high", "吸烟是最重要的可控危险因素"),
            CardiovascularRiskFactor("高血压", "modifiable", "high", "收缩压≥140mmHg或舒张压≥90mmHg"),
            CardiovascularRiskFactor("糖尿病", "modifiable", "high", "血糖控制不良"),
            CardiovascularRiskFactor("血脂异常", "modifiable", "high", "LDL-C升高，HDL-C降低"),
            CardiovascularRiskFactor("肥胖", "modifiable", "medium", "BMI≥28或腰围过大"),
            CardiovascularRiskFactor("缺乏运动", "modifiable", "medium", "久坐不动的生活方式"),
            CardiovascularRiskFactor("不良饮食", "modifiable", "medium", "高盐、高脂、高糖饮食"),
            CardiovascularRiskFactor("慢性压力", "modifiable", "medium", "长期精神紧张"),
            CardiovascularRiskFactor("睡眠不足", "modifiable", "low", "睡眠质量差")
        ]

    def _load_emergency_keywords(self) -> List[str]:
        """加载急症关键词"""
        return [
            "胸痛剧烈", "胸痛持续", "压榨性胸痛", "撕裂样胸痛",
            "呼吸困难加重", "不能平卧", "大汗淋漓", "面色苍白",
            "意识模糊", "晕厥", "心跳停止", "血压下降",
            "急性", "突发", "剧烈", "持续不缓解"
        ]

    def is_cardiovascular_related(self, text: str) -> Dict[str, Any]:
        """判断是否为心血管相关问题"""
        text_lower = text.lower()

        # 统计各类关键词出现次数
        category_scores = {}
        matched_keywords = {}

        for category, keywords in self.cardio_keywords.items():
            score = 0
            matched = []

            for keyword in keywords:
                if keyword in text_lower:
                    score += 1
                    matched.append(keyword)

            category_scores[category] = score
            matched_keywords[category] = matched

        # 计算总分和置信度
        total_score = sum(category_scores.values())
        confidence = min(total_score / 10.0, 1.0)  # 归一化到0-1

        # 判断是否为心血管相关
        is_cardio = confidence > 0.3 or any(score > 0 for score in category_scores.values())

        return {
            "is_cardiovascular": is_cardio,
            "confidence": confidence,
            "category_scores": category_scores,
            "matched_keywords": matched_keywords,
            "total_matches": total_score
        }

    def assess_emergency_level(self, text: str) -> Dict[str, Any]:
        """评估急症程度"""
        emergency_score = 0
        matched_emergency = []

        for keyword in self.emergency_keywords:
            if keyword in text:
                emergency_score += 1
                matched_emergency.append(keyword)

        # 特殊急症模式检测
        emergency_patterns = [
            r"胸痛.*持续.*\d+.*小时",
            r"胸痛.*放射.*左臂",
            r"呼吸困难.*不能.*平卧",
            r"血压.*\d+.*\d+.*很高",
            r"心跳.*停止|心脏.*停跳"
        ]

        pattern_matches = 0
        for pattern in emergency_patterns:
            if re.search(pattern, text):
                pattern_matches += 1

        # 计算急症等级
        total_emergency_score = emergency_score + pattern_matches * 2

        if total_emergency_score >= 3:
            level = "high"
            recommendation = "🚨 建议立即就医或拨打120急救电话"
        elif total_emergency_score >= 1:
            level = "medium"
            recommendation = "⚠️ 建议尽快到医院心内科就诊"
        else:
            level = "low"
            recommendation = "💡 可先进行咨询，必要时就医"

        return {
            "emergency_level": level,
            "emergency_score": total_emergency_score,
            "matched_keywords": matched_emergency,
            "recommendation": recommendation
        }

    def analyze_risk_factors(self, text: str) -> Dict[str, Any]:
        """分析心血管风险因素"""
        mentioned_factors = []

        # 检测文本中提到的风险因素
        for factor in self.risk_factors:
            if any(keyword in text for keyword in [factor.name, factor.description]):
                mentioned_factors.append(factor)

        # 按严重程度分类
        high_risk = [f for f in mentioned_factors if f.severity == "high"]
        medium_risk = [f for f in mentioned_factors if f.severity == "medium"]
        low_risk = [f for f in mentioned_factors if f.severity == "low"]

        # 计算总体风险等级
        risk_score = len(high_risk) * 3 + len(medium_risk) * 2 + len(low_risk) * 1

        if risk_score >= 6:
            overall_risk = "high"
        elif risk_score >= 3:
            overall_risk = "medium"
        else:
            overall_risk = "low"

        return {
            "overall_risk": overall_risk,
            "risk_score": risk_score,
            "high_risk_factors": [f.name for f in high_risk],
            "medium_risk_factors": [f.name for f in medium_risk],
            "low_risk_factors": [f.name for f in low_risk],
            "modifiable_factors": [f.name for f in mentioned_factors if f.category == "modifiable"],
            "non_modifiable_factors": [f.name for f in mentioned_factors if f.category == "non_modifiable"]
        }

    def generate_cardio_prompt(self, question: str) -> str:
        """生成心血管专科提示词"""

        # 分析问题特征
        cardio_analysis = self.is_cardiovascular_related(question)
        emergency_analysis = self.assess_emergency_level(question)
        risk_analysis = self.analyze_risk_factors(question)

        # 构建专科提示词
        prompt = f"""你是心智医AI，专门的心血管疾病智能问答助手。请基于以下分析回答用户的心血管相关问题：

问题分析：
- 心血管相关性: {cardio_analysis['confidence']:.2f}
- 急症等级: {emergency_analysis['emergency_level']}
- 风险评估: {risk_analysis['overall_risk']}

{emergency_analysis['recommendation']}

请按照以下格式回答：

1. **问题理解**: 简要分析用户问题的核心要点
2. **专业解答**: 提供准确的心血管医学信息
3. **风险提示**: 如有必要，提醒相关风险因素
4. **生活建议**: 给出实用的生活方式建议
5. **就医指导**: 明确是否需要就医及科室选择

注意事项：
- 如果是急症情况，优先强调紧急就医
- 提供的信息要准确、专业但易懂
- 强调这是咨询参考，不能替代医生诊断
- 重点关注心血管疾病的预防和管理

用户问题：{question}
"""

        return prompt

    def build_knowledge_base(self, cardio_data: List[Dict[str, Any]]):
        """构建心血管知识库（支持增量更新）"""
        print("🔧 正在初始化心血管知识向量数据库...")

        try:
            from rag_system import MedicalRAGSystem
            from embeddings import EmbeddingManager

            # 初始化RAG系统和嵌入管理器
            self.rag_system = MedicalRAGSystem()
            self.embedding_manager = EmbeddingManager()

            # 检查现有数据库状态
            existing_stats = self.embedding_manager.get_collection_stats()
            existing_count = existing_stats.get('total_documents', 0)

            print(f"📊 现有向量数据库状态: {existing_count} 条记录")

            # 如果数据库已有足够的心血管数据，跳过重建
            expected_count = len(cardio_data)
            if existing_count >= expected_count * 0.95:  # 允许5%的误差
                print(f"✅ 检测到现有心血管知识库包含 {existing_count} 条记录")
                print(f"📋 预期记录数: {expected_count}")
                print("🚀 跳过重建，直接使用现有数据库")
                self.knowledge_base = cardio_data
                return

            # 如果数据不完整或为空，重新构建
            print(f"🔄 数据库记录不完整 ({existing_count}/{expected_count})，开始重建...")

            # 清空现有集合（如果需要）
            if existing_count > 0:
                print("🗑️ 清空现有数据...")
                self.embedding_manager.clear_collection()

            # 准备知识库数据 - 简化但优化的方法
            documents = []
            for i, item in enumerate(cardio_data):
                question = item['question']
                answer = item['answer']
                keywords = item.get('matched_keywords', [])

                # 智能文本增强：为短文本添加上下文，长文本保持原样
                if len(answer) < 30:  # 极短答案（如"无特殊人群"）
                    # 添加问题上下文和关键词
                    keyword_context = f" 相关概念: {', '.join(keywords[:3])}" if keywords else ""
                    enhanced_text = f"医学问答: {question} 答案: {answer}{keyword_context}"
                elif len(answer) < 100:  # 短答案
                    # 简单增强
                    enhanced_text = f"问题: {question} 答案: {answer}"
                else:  # 长答案，保持原样
                    enhanced_text = f"问题: {question}\n答案: {answer}"

                # 只创建一个优化的文档
                documents.append({
                    "text": enhanced_text,
                    "chunk_id": f"cardio_{i:06d}",
                    "metadata": {
                        "question": question,
                        "answer": answer,
                        "keywords": ", ".join(keywords),
                        "keyword_count": len(keywords),
                        "source": "华佗数据集-心血管专项",
                        "dataset_info": "心血管疾病问答数据",
                        "data_type": "cardiovascular",
                        "text_length": len(enhanced_text),
                        "original_answer_length": len(answer)
                    }
                })

            # 添加文档到向量数据库
            self.embedding_manager.add_documents(documents)
            self.knowledge_base = cardio_data

            print(f"✅ 成功构建包含 {len(documents)} 条记录的心血管知识库")
            print("💾 向量数据库已持久化，下次启动将直接加载")

        except Exception as e:
            print(f"❌ 知识库构建失败: {e}")
            raise

    def get_cardiovascular_answer(self, question: str, use_enhanced_rag: bool = True) -> Dict[str, Any]:
        """获取心血管专科回答"""

        if not self.knowledge_base:
            return {
                "answer": "❌ 心血管知识库未初始化，请先构建知识库",
                "confidence": 0.0,
                "references": []
            }

        try:
            # 如果启用增强RAG，使用完整的RAG流程
            if use_enhanced_rag:
                from enhanced_rag_system import EnhancedRAGSystem

                if not hasattr(self, 'enhanced_rag'):
                    self.enhanced_rag = EnhancedRAGSystem()

                # 使用增强RAG系统获取完整的处理结果
                rag_result = self.enhanced_rag.query_with_full_process(question)

                # 添加心血管专科分析
                cardio_analysis = self.is_cardiovascular_related(question)
                emergency_analysis = self.assess_emergency_level(question)

                # 如果是急症，优先返回急症建议
                if emergency_analysis['emergency_level'] == 'high':
                    rag_result['answer'] = f"🚨 {emergency_analysis['recommendation']}\n\n{rag_result['answer']}"
                    rag_result['emergency'] = True

                # 添加心血管专科信息
                rag_result.update({
                    "cardio_confidence": cardio_analysis['confidence'],
                    "emergency_level": emergency_analysis['emergency_level'],
                    "emergency_recommendation": emergency_analysis['recommendation'],
                    "cardio_analysis": cardio_analysis
                })

                return rag_result

            else:
                # 使用简化的检索流程（向后兼容）
                return self._get_simple_answer(question)

        except Exception as e:
            return {
                "answer": f"处理问题时出现错误: {e}",
                "confidence": 0.0,
                "references": [],
                "error": str(e)
            }

    def _get_simple_answer(self, question: str) -> Dict[str, Any]:
        """简化的回答流程（向后兼容）"""

        # 分析问题特征
        cardio_analysis = self.is_cardiovascular_related(question)
        emergency_analysis = self.assess_emergency_level(question)

        # 如果是急症，优先返回急症建议
        if emergency_analysis['emergency_level'] == 'high':
            return {
                "answer": f"🚨 {emergency_analysis['recommendation']}\n\n根据您描述的症状，这可能是心血管急症。请立即就医或拨打120急救电话，不要延误治疗时机。",
                "confidence": 1.0,
                "emergency": True,
                "references": []
            }

        # 使用增强检索器检索相关信息
        from enhanced_retriever import EnhancedRetriever

        if not hasattr(self, 'enhanced_retriever'):
            self.enhanced_retriever = EnhancedRetriever()

        # 使用增强检索，支持调试模式
        similar_docs = self.enhanced_retriever.search(question, top_k=8, debug=True)

        if not similar_docs:
            # 如果增强检索也没找到，尝试更宽松的检索
            print("🔄 尝试更宽松的检索策略...")
            try:
                fallback_docs = self.embedding_manager.search_similar(
                    question,
                    top_k=10,
                    threshold=0.05  # 非常宽松的阈值
                )
                if fallback_docs:
                    similar_docs = fallback_docs
                    print(f"✅ 备用检索找到 {len(fallback_docs)} 条结果")
            except Exception as e:
                print(f"备用检索也失败: {e}")

        if not similar_docs:
            return {
                "answer": "抱歉，我在心血管知识库中没有找到相关信息。这可能是因为您的问题比较特殊，建议您咨询专业的心血管科医生获得更准确的建议。",
                "confidence": 0.0,
                "references": [],
                "search_info": {
                    "query": question,
                    "results_count": 0,
                    "search_strategies": ["enhanced", "fallback"]
                }
            }

        # 提取参考资料
        references = []
        for doc in similar_docs[:5]:  # 增加参考资料数量
            if 'metadata' in doc:
                references.append({
                    "question": doc['metadata'].get('question', ''),
                    "answer": doc['metadata'].get('answer', ''),
                    "similarity": doc.get('final_score', doc.get('similarity', 0.0)),
                    "strategy": doc.get('strategy', 'unknown')
                })

        # 生成专业回答
        answer = self._generate_professional_answer(question, references, cardio_analysis, emergency_analysis)

        return {
            "answer": answer,
            "confidence": cardio_analysis['confidence'],
            "references": references,
            "emergency": emergency_analysis['emergency_level'] != 'low',
            "emergency_recommendation": emergency_analysis['recommendation'],
            "search_info": {
                "query": question,
                "results_count": len(similar_docs),
                "search_strategies": ["enhanced"] + (["fallback"] if len(similar_docs) > 8 else []),
                "top_similarity": similar_docs[0].get('final_score', similar_docs[0].get('similarity', 0)) if similar_docs else 0
            }
        }

    def _generate_professional_answer(self, question: str, references: List[Dict],
                                    cardio_analysis: Dict, emergency_analysis: Dict) -> str:
        """生成专业的心血管回答 - 基于检索结果进行知识扩展"""

        # 使用LLM进行知识扩展
        try:
            from llm_client import LLMClient

            llm_client = LLMClient()

            # 构建上下文信息
            context_parts = []
            if references:
                context_parts.append("检索到的相关医学信息:")
                for i, ref in enumerate(references[:3], 1):
                    context_parts.append(f"{i}. 问题: {ref['question']}")
                    context_parts.append(f"   答案: {ref['answer']}")
                    context_parts.append(f"   相似度: {ref.get('similarity', 0):.3f}")
                    context_parts.append("")

            context = "\n".join(context_parts)

            # 构建专业的医学提示词
            system_prompt = """你是一位专业的心血管科医生AI助手。请基于检索到的医学信息，结合你的专业知识，为患者提供全面、专业的回答。

要求：
1. 基于检索结果，但不局限于检索结果
2. 结合医学专业知识进行扩展说明
3. 提供实用的建议和指导
4. 保持专业性和准确性
5. 包含必要的注意事项和免责声明

回答格式：
- 使用清晰的结构化格式
- 包含专业解释、相关知识、注意事项等
- 语言通俗易懂，但保持专业性"""

            # 生成扩展回答
            user_message = f"""
患者问题: {question}

心血管相关性: {cardio_analysis['confidence']:.1%}
紧急程度: {emergency_analysis['emergency_level']}

请基于以上信息和检索结果，提供专业的心血管医学回答。
"""

            # 使用CoT推理生成回答
            response = llm_client.generate_response_with_cot(
                user_message=user_message,
                context=context,
                system_prompt=system_prompt
            )

            # 如果CoT成功，返回最终答案；否则使用备用方案
            if response.get('final_answer'):
                return response['final_answer']
            else:
                return self._generate_fallback_answer(question, references, cardio_analysis, emergency_analysis)

        except Exception as e:
            print(f"LLM扩展回答失败: {e}")
            return self._generate_fallback_answer(question, references, cardio_analysis, emergency_analysis)

    def _generate_fallback_answer(self, question: str, references: List[Dict],
                                cardio_analysis: Dict, emergency_analysis: Dict) -> str:
        """备用回答生成方案"""

        answer_parts = []

        # 1. 基于检索结果的核心回答
        if references:
            answer_parts.append("## 💊 专业解答")

            # 主要答案
            main_ref = references[0]
            answer_parts.append(f"**核心信息**: {main_ref['answer']}")

            # 补充信息
            if len(references) > 1:
                answer_parts.append("\n**相关信息**:")
                for ref in references[1:3]:
                    if ref['answer'] != main_ref['answer']:
                        answer_parts.append(f"• {ref['answer']}")

        # 2. 急症提示
        if emergency_analysis['emergency_level'] != 'low':
            answer_parts.append(f"\n## ⚠️ 重要提示")
            answer_parts.append(emergency_analysis['recommendation'])

        # 3. 通用建议
        answer_parts.append("\n## 📋 建议")
        answer_parts.append("• 如有疑问，请咨询专业心血管科医生")
        answer_parts.append("• 定期进行心血管健康检查")
        answer_parts.append("• 保持健康的生活方式")

        # 4. 免责声明
        answer_parts.append("\n## 📋 重要声明")
        answer_parts.append("以上信息仅供参考，不能替代专业医生的诊断和治疗建议。")

        return "\n".join(answer_parts)

    def get_cardio_statistics(self) -> Dict[str, Any]:
        """获取心血管专科统计信息"""
        stats = {
            "total_keywords": sum(len(keywords) for keywords in self.cardio_keywords.values()),
            "keyword_categories": len(self.cardio_keywords),
            "risk_factors": len(self.risk_factors),
            "modifiable_factors": len([f for f in self.risk_factors if f.category == "modifiable"]),
            "emergency_keywords": len(self.emergency_keywords),
            "coverage_areas": [
                "冠心病及急性冠脉综合征",
                "高血压及血压管理",
                "心律失常及心电异常",
                "心力衰竭及心功能不全",
                "心血管风险因素评估",
                "心血管急症识别",
                "心血管药物指导",
                "心血管康复指导"
            ]
        }

        # 如果有知识库，添加知识库统计
        if self.knowledge_base:
            stats.update({
                "knowledge_base_size": len(self.knowledge_base),
                "avg_keywords_per_qa": sum(item.get('keyword_count', 0) for item in self.knowledge_base) / len(self.knowledge_base)
            })

        return stats
