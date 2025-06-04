"""
增强检索器 - 支持多种检索策略
解决检索过于死板的问题
"""
import re
import jieba
from typing import List, Dict, Any, Optional
from embeddings import EmbeddingManager
from config import settings


class EnhancedRetriever:
    """增强检索器"""
    
    def __init__(self):
        self.embedding_manager = EmbeddingManager()
        
        # 医学同义词映射
        self.medical_synonyms = {
            "高血压": ["高压", "血压高", "血压升高", "高血压病"],
            "低血压": ["低压", "血压低", "血压降低", "低血压症"],
            "心脏病": ["心病", "心脏疾病", "心脏问题"],
            "心肌梗死": ["心梗", "心肌梗塞", "急性心梗"],
            "心绞痛": ["胸痛", "心前区疼痛", "心口痛"],
            "心律失常": ["心律不齐", "心跳不规律", "心律紊乱"],
            "心力衰竭": ["心衰", "心功能不全"],
            "冠心病": ["冠状动脉疾病", "冠状动脉病变"],
            "动脉硬化": ["动脉粥样硬化", "血管硬化"],
            "房颤": ["心房颤动", "房性心律失常"],
            "胸闷": ["胸部不适", "胸部憋闷"],
            "气短": ["呼吸困难", "喘气", "气促"],
            "心悸": ["心慌", "心跳快", "心跳加速"]
        }
        
        # 心血管关键词
        self.cardio_keywords = [
            "心脏", "心血管", "心肌", "心房", "心室", "血压", "血管",
            "动脉", "静脉", "冠状动脉", "主动脉", "肺动脉",
            "胸痛", "胸闷", "心悸", "气短", "呼吸困难",
            "高血压", "低血压", "心律失常", "心绞痛", "心肌梗死",
            "心力衰竭", "冠心病", "房颤", "心电图", "心脏彩超"
        ]
    
    def preprocess_query(self, query: str) -> List[str]:
        """预处理查询，生成多个查询变体"""
        queries = [query]  # 原始查询
        
        # 1. 添加同义词变体
        for term, synonyms in self.medical_synonyms.items():
            if term in query:
                for synonym in synonyms:
                    variant = query.replace(term, synonym)
                    if variant != query:
                        queries.append(variant)
            
            # 反向替换
            for synonym in synonyms:
                if synonym in query:
                    variant = query.replace(synonym, term)
                    if variant != query:
                        queries.append(variant)
        
        # 2. 分词后的关键词组合
        words = list(jieba.cut(query))
        if len(words) > 1:
            # 提取心血管相关词汇
            cardio_words = [w for w in words if any(kw in w for kw in self.cardio_keywords)]
            if cardio_words:
                queries.append(" ".join(cardio_words))
        
        # 3. 去除标点符号的版本
        clean_query = re.sub(r'[^\w\s]', '', query)
        if clean_query != query:
            queries.append(clean_query)
        
        # 4. 添加问题格式
        if not query.endswith('？') and not query.endswith('?'):
            queries.append(query + "？")
        
        return list(set(queries))  # 去重
    
    def multi_strategy_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """多策略检索"""
        all_results = []
        
        # 策略1: 原始查询
        try:
            results1 = self.embedding_manager.search_similar(
                query, 
                top_k=top_k, 
                threshold=0.1  # 非常宽松的阈值
            )
            for result in results1:
                result['strategy'] = 'original'
                result['boost'] = 1.0
            all_results.extend(results1)
        except Exception as e:
            print(f"原始查询失败: {e}")
        
        # 策略2: 查询变体
        query_variants = self.preprocess_query(query)
        for i, variant in enumerate(query_variants[1:], 1):  # 跳过原始查询
            try:
                results = self.embedding_manager.search_similar(
                    variant, 
                    top_k=max(5, top_k//2), 
                    threshold=0.1
                )
                for result in results:
                    result['strategy'] = f'variant_{i}'
                    result['boost'] = 0.8  # 变体权重稍低
                all_results.extend(results)
            except Exception as e:
                print(f"变体查询失败 ({variant}): {e}")
        
        # 策略3: 关键词检索
        keywords = self.extract_keywords(query)
        if keywords:
            keyword_query = " ".join(keywords)
            try:
                results3 = self.embedding_manager.search_similar(
                    keyword_query, 
                    top_k=max(5, top_k//2), 
                    threshold=0.05  # 更宽松
                )
                for result in results3:
                    result['strategy'] = 'keywords'
                    result['boost'] = 0.6
                all_results.extend(results3)
            except Exception as e:
                print(f"关键词查询失败: {e}")
        
        # 合并和去重
        unique_results = self.merge_and_deduplicate(all_results)
        
        # 重新排序
        final_results = self.rerank_results(unique_results, query)
        
        return final_results[:top_k]
    
    def extract_keywords(self, query: str) -> List[str]:
        """提取关键词"""
        words = list(jieba.cut(query))
        
        # 过滤停用词
        stop_words = {'的', '了', '是', '在', '有', '和', '与', '或', '但', '如果', '因为', '所以', '什么', '怎么', '为什么', '吗', '呢', '吧'}
        
        keywords = []
        for word in words:
            if len(word) > 1 and word not in stop_words:
                keywords.append(word)
        
        # 优先保留医学相关词汇
        medical_keywords = [w for w in keywords if any(kw in w for kw in self.cardio_keywords)]
        other_keywords = [w for w in keywords if w not in medical_keywords]
        
        return medical_keywords + other_keywords[:3]  # 医学词汇 + 最多3个其他词汇
    
    def merge_and_deduplicate(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """合并和去重结果"""
        seen_texts = set()
        unique_results = []
        
        for result in results:
            text = result.get('text', '')
            # 使用文本的前100个字符作为去重标识
            text_key = text[:100]
            
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                unique_results.append(result)
        
        return unique_results
    
    def rerank_results(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """重新排序结果"""
        query_lower = query.lower()
        
        for result in results:
            text = result.get('text', '').lower()
            metadata = result.get('metadata', {})
            
            # 基础分数
            base_score = result.get('similarity', 0) * result.get('boost', 1.0)
            
            # 文本匹配加分
            text_bonus = 0
            for word in jieba.cut(query_lower):
                if len(word) > 1 and word in text:
                    text_bonus += 0.1
            
            # 关键词匹配加分
            keyword_bonus = 0
            keywords = metadata.get('keywords', '')
            if keywords:
                for word in jieba.cut(query_lower):
                    if len(word) > 1 and word in keywords.lower():
                        keyword_bonus += 0.15
            
            # 问题类型匹配加分
            question = metadata.get('question', '').lower()
            if question:
                # 如果查询和问题有相似的结构
                if any(marker in query_lower for marker in ['什么', '怎么', '为什么', '如何']):
                    if any(marker in question for marker in ['什么', '怎么', '为什么', '如何']):
                        text_bonus += 0.2
            
            # 计算最终分数
            final_score = base_score + text_bonus + keyword_bonus
            result['final_score'] = final_score
        
        # 按最终分数排序
        return sorted(results, key=lambda x: x.get('final_score', 0), reverse=True)
    
    def search(self, query: str, top_k: int = None, debug: bool = False) -> List[Dict[str, Any]]:
        """主要搜索接口"""
        top_k = top_k or settings.TOP_K_RETRIEVAL
        
        if debug:
            print(f"🔍 原始查询: {query}")
            variants = self.preprocess_query(query)
            print(f"🔄 查询变体: {variants}")
            keywords = self.extract_keywords(query)
            print(f"🔑 提取关键词: {keywords}")
        
        # 使用多策略检索
        results = self.multi_strategy_search(query, top_k)
        
        if debug:
            print(f"📊 检索结果数量: {len(results)}")
            for i, result in enumerate(results[:3], 1):
                print(f"   {i}. 分数: {result.get('final_score', 0):.3f}, 策略: {result.get('strategy', 'unknown')}")
                print(f"      文本: {result.get('text', '')[:100]}...")
        
        return results
