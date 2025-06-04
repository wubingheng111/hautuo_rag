"""
LangChain示例选择器 - 生成精准提示词
"""
import os
import json
from typing import List, Dict, Any, Optional
import numpy as np

try:
    from langchain.prompts import PromptTemplate, FewShotPromptTemplate
    from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
    from langchain.vectorstores import Chroma
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain_core.example_selectors import BaseExampleSelector
    LANGCHAIN_AVAILABLE = True
except ImportError:
    print("⚠️ LangChain未安装，请运行: python install_langchain.py")
    LANGCHAIN_AVAILABLE = False

from config import settings


class MedicalExampleSelector:
    """医学示例选择器"""
    
    def __init__(self):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain未安装")
        
        self.examples = self._load_medical_examples()
        self.example_selector = None
        self.few_shot_prompt = None
        self._setup_selector()
    
    def _load_medical_examples(self) -> List[Dict[str, str]]:
        """加载医学示例数据"""
        examples = [
            {
                "question": "高血压的症状有哪些？",
                "thinking": "用户询问高血压的症状，这是一个常见的心血管疾病问题。我需要从症状表现、严重程度、并发症等角度来回答。",
                "answer": "高血压的主要症状包括：\n1. 头痛、头晕\n2. 心悸、胸闷\n3. 疲劳乏力\n4. 视力模糊\n5. 耳鸣\n注意：轻度高血压可能无明显症状，建议定期监测血压。"
            },
            {
                "question": "糖尿病患者饮食注意事项？",
                "thinking": "这是关于糖尿病管理的问题，饮食控制是糖尿病治疗的重要组成部分。我需要提供具体的饮食指导。",
                "answer": "糖尿病患者饮食要点：\n1. 控制总热量摄入\n2. 少食多餐，定时定量\n3. 选择低升糖指数食物\n4. 增加膳食纤维摄入\n5. 限制糖分和精制碳水化合物\n6. 适量优质蛋白质\n建议咨询营养师制定个性化饮食方案。"
            },
            {
                "question": "感冒和流感的区别？",
                "thinking": "用户想了解感冒和流感的区别，这是常见的呼吸道疾病鉴别问题。我需要从症状、病因、严重程度等方面对比。",
                "answer": "感冒与流感的主要区别：\n\n感冒：\n- 症状较轻，逐渐出现\n- 主要是鼻塞、流涕、咽痛\n- 很少发热或低热\n- 病程3-7天\n\n流感：\n- 症状较重，突然发作\n- 高热、全身酸痛、乏力\n- 可能有并发症\n- 病程1-2周\n\n建议出现高热等严重症状时及时就医。"
            },
            {
                "question": "心脏病的早期症状？",
                "thinking": "心脏病早期识别很重要，我需要列出主要的早期警示症状，帮助用户及时发现问题。",
                "answer": "心脏病的早期症状：\n1. 胸痛或胸闷（特别是活动时）\n2. 气短、呼吸困难\n3. 心悸、心律不齐\n4. 疲劳乏力\n5. 下肢水肿\n6. 头晕、晕厥\n7. 夜间咳嗽\n\n⚠️ 出现这些症状应及时就医检查，早期诊断和治疗很重要。"
            },
            {
                "question": "如何预防癌症？",
                "thinking": "癌症预防是重要的健康话题，我需要从生活方式、饮食、环境等多个角度提供预防建议。",
                "answer": "癌症预防措施：\n\n生活方式：\n- 戒烟限酒\n- 规律运动\n- 保持健康体重\n- 充足睡眠\n\n饮食方面：\n- 多吃蔬菜水果\n- 减少加工肉类\n- 限制高脂食物\n- 适量膳食纤维\n\n其他：\n- 定期体检筛查\n- 避免有害环境\n- 接种相关疫苗\n- 管理慢性疾病"
            },
            {
                "question": "药物过敏怎么办？",
                "thinking": "药物过敏是紧急医疗情况，我需要提供应急处理方法和预防措施。",
                "answer": "药物过敏处理：\n\n立即措施：\n1. 停止使用可疑药物\n2. 轻度：观察症状变化\n3. 重度：立即就医或拨打120\n\n严重症状（需紧急就医）：\n- 呼吸困难\n- 面部肿胀\n- 全身皮疹\n- 血压下降\n\n预防：\n- 记录过敏药物\n- 就医时主动告知\n- 佩戴过敏提示卡"
            },
            {
                "question": "失眠的原因和治疗？",
                "thinking": "失眠是常见问题，我需要分析可能的原因并提供治疗建议，包括非药物和药物治疗。",
                "answer": "失眠的常见原因：\n1. 压力和焦虑\n2. 不良睡眠习惯\n3. 环境因素\n4. 疾病影响\n5. 药物副作用\n\n治疗方法：\n\n非药物治疗：\n- 建立规律作息\n- 改善睡眠环境\n- 放松训练\n- 认知行为疗法\n\n药物治疗：\n- 需医生指导\n- 短期使用\n- 注意副作用\n\n建议先尝试非药物方法，必要时咨询医生。"
            }
        ]
        
        return examples
    
    def _setup_selector(self):
        """设置示例选择器"""
        try:
            # 使用HuggingFace嵌入模型
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
            
            # 创建语义相似度示例选择器
            self.example_selector = SemanticSimilarityExampleSelector.from_examples(
                self.examples,
                embeddings,
                Chroma,
                k=2  # 选择最相似的2个示例
            )
            
            # 创建示例模板
            example_prompt = PromptTemplate(
                input_variables=["question", "thinking", "answer"],
                template="""
问题：{question}
思考过程：{thinking}
回答：{answer}
"""
            )
            
            # 创建Few-Shot提示模板
            self.few_shot_prompt = FewShotPromptTemplate(
                example_selector=self.example_selector,
                example_prompt=example_prompt,
                prefix="""你是华佗医学AI助手，请参考以下示例来回答医学问题。

示例：""",
                suffix="""
现在请回答以下问题：
问题：{question}
思考过程：""",
                input_variables=["question"]
            )
            
            print("✅ LangChain示例选择器初始化成功")
            
        except Exception as e:
            print(f"❌ 示例选择器初始化失败: {e}")
            self.example_selector = None
            self.few_shot_prompt = None
    
    def select_examples(self, question: str, k: int = 2) -> List[Dict[str, str]]:
        """
        选择相似示例
        
        Args:
            question: 用户问题
            k: 选择示例数量
            
        Returns:
            相似示例列表
        """
        if not self.example_selector:
            return []
        
        try:
            # 更新选择数量
            self.example_selector.k = k
            
            # 选择示例
            selected_examples = self.example_selector.select_examples({"question": question})
            
            return selected_examples
            
        except Exception as e:
            print(f"示例选择失败: {e}")
            return []
    
    def generate_prompt(self, question: str) -> str:
        """
        生成包含示例的提示词
        
        Args:
            question: 用户问题
            
        Returns:
            完整的提示词
        """
        if not self.few_shot_prompt:
            return f"请回答以下医学问题：{question}"
        
        try:
            # 生成Few-Shot提示
            prompt = self.few_shot_prompt.format(question=question)
            return prompt
            
        except Exception as e:
            print(f"提示词生成失败: {e}")
            return f"请回答以下医学问题：{question}"
    
    def analyze_selection_process(self, question: str) -> Dict[str, Any]:
        """
        分析示例选择过程
        
        Args:
            question: 用户问题
            
        Returns:
            选择过程分析
        """
        if not self.example_selector:
            return {"error": "示例选择器未初始化"}
        
        try:
            # 选择示例
            selected_examples = self.select_examples(question, k=3)
            
            # 生成提示词
            prompt = self.generate_prompt(question)
            
            # 分析过程
            analysis = {
                "user_question": question,
                "total_examples": len(self.examples),
                "selected_count": len(selected_examples),
                "selected_examples": selected_examples,
                "generated_prompt": prompt,
                "selection_method": "语义相似度匹配",
                "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
                "similarity_scores": []
            }
            
            # 计算相似度分数（如果可能）
            for example in selected_examples:
                analysis["similarity_scores"].append({
                    "example_question": example["question"],
                    "similarity": "高相似度"  # 实际实现中可以计算具体分数
                })
            
            return analysis
            
        except Exception as e:
            return {"error": f"分析失败: {e}"}
    
    def add_example(self, question: str, thinking: str, answer: str):
        """
        添加新示例
        
        Args:
            question: 问题
            thinking: 思考过程
            answer: 答案
        """
        new_example = {
            "question": question,
            "thinking": thinking,
            "answer": answer
        }
        
        self.examples.append(new_example)
        
        # 重新设置选择器
        self._setup_selector()
        
        print(f"✅ 已添加新示例: {question}")
    
    def save_examples(self, filepath: str = None):
        """保存示例到文件"""
        if not filepath:
            filepath = os.path.join(settings.DATA_DIR, "medical_examples.json")
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.examples, f, ensure_ascii=False, indent=2)
            print(f"✅ 示例已保存到: {filepath}")
        except Exception as e:
            print(f"❌ 保存失败: {e}")
    
    def load_examples(self, filepath: str = None):
        """从文件加载示例"""
        if not filepath:
            filepath = os.path.join(settings.DATA_DIR, "medical_examples.json")
        
        if not os.path.exists(filepath):
            print(f"⚠️ 示例文件不存在: {filepath}")
            return
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.examples = json.load(f)
            
            # 重新设置选择器
            self._setup_selector()
            
            print(f"✅ 已加载 {len(self.examples)} 个示例")
        except Exception as e:
            print(f"❌ 加载失败: {e}")


class CustomMedicalExampleSelector(BaseExampleSelector):
    """自定义医学示例选择器"""
    
    def __init__(self, examples: List[Dict[str, str]], k: int = 2):
        self.examples = examples
        self.k = k
    
    def add_example(self, example: Dict[str, str]) -> None:
        """添加示例"""
        self.examples.append(example)
    
    def select_examples(self, input_variables: Dict[str, str]) -> List[Dict[str, str]]:
        """
        基于关键词匹配选择示例
        """
        question = input_variables.get("question", "")
        
        # 简单的关键词匹配
        scored_examples = []
        
        for example in self.examples:
            score = 0
            example_question = example["question"]
            
            # 计算关键词重叠度
            question_words = set(question)
            example_words = set(example_question)
            
            overlap = len(question_words.intersection(example_words))
            score = overlap / max(len(question_words), len(example_words))
            
            scored_examples.append((score, example))
        
        # 按分数排序并选择前k个
        scored_examples.sort(key=lambda x: x[0], reverse=True)
        
        return [example for score, example in scored_examples[:self.k]]
