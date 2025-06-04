"""
配置文件
"""
import os
from typing import Optional


class Settings:
    """系统配置"""

    def __init__(self):
        # DeepSeek API配置
        self.DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "your-api-key")
        self.DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        self.DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

        # 数据配置
        self.DATA_DIR = os.getenv("DATA_DIR", "data")
        self.VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR", "data/vectordb")
        self.CHAT_HISTORY_DIR = os.getenv("CHAT_HISTORY_DIR", "data/chat_history")

        # 嵌入模型配置
        # 可选模型（按推荐程度排序）：
        # 1. "BAAI/bge-large-zh-v1.5" - 中文最佳，1024维
        # 2. "BAAI/bge-base-zh-v1.5" - 中文优秀，768维
        # 3. "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" - 多语言，384维（当前）
        # 4. "shibing624/text2vec-base-chinese" - 中文专用，768维
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))

        # 嵌入优化配置
        self.EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

        # RAG配置 - 针对短文本医学问答优化
        self.CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "200"))  # 适合短文本
        self.CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "20"))  # 减少重叠
        self.TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "15"))  # 增加检索数量
        self.SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.2"))  # 大幅降低阈值

        # 对话配置
        self.MAX_HISTORY_LENGTH = int(os.getenv("MAX_HISTORY_LENGTH", "10"))
        self.MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2000"))
        self.TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

        # 华佗数据集配置
        self.HUATUO_DATASET = os.getenv("HUATUO_DATASET", "FreedomIntelligence/huatuo_knowledge_graph_qa")


# 全局配置实例
settings = Settings()

# 创建必要的目录
os.makedirs(settings.DATA_DIR, exist_ok=True)
os.makedirs(settings.VECTOR_DB_DIR, exist_ok=True)
os.makedirs(settings.CHAT_HISTORY_DIR, exist_ok=True)
