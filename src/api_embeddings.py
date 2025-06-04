"""
API向量化管理器 - 支持多种免费API
"""
import requests
import numpy as np
from typing import List, Dict, Any
import time
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import chromadb
from chromadb.config import Settings as ChromaSettings

from config import settings
from vector_cache import VectorCache


class APIEmbeddingManager:
    """API向量化管理器"""

    def __init__(self, api_provider: str = "huggingface", enable_cache: bool = True):
        self.api_provider = api_provider
        self.enable_cache = enable_cache
        self.chroma_client = None
        self.collection = None
        self.api_configs = self._setup_api_configs()

        # 初始化向量缓存
        if enable_cache:
            self.vector_cache = VectorCache()
            print("💾 API向量缓存已启用")
        else:
            self.vector_cache = None
            print("⚠️ API向量缓存已禁用")

        self._init_vector_db()

    def _setup_api_configs(self) -> Dict[str, Dict]:
        """设置API配置"""
        return {
            "huggingface": {
                "url": "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "headers": {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY', '')}"},
                "free": True,
                "rate_limit": 1000,  # 每小时请求数
                "batch_size": 50
            },
            "cohere": {
                "url": "https://api.cohere.ai/v1/embed",
                "headers": {
                    "Authorization": f"Bearer {os.getenv('COHERE_API_KEY', '')}",
                    "Content-Type": "application/json"
                },
                "free": True,
                "rate_limit": 100,  # 每月免费额度
                "batch_size": 96
            },
            "openai": {
                "url": "https://api.openai.com/v1/embeddings",
                "headers": {
                    "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', '')}",
                    "Content-Type": "application/json"
                },
                "free": False,  # 付费但便宜
                "rate_limit": 3000,
                "batch_size": 100
            },
            "jina": {
                "url": "https://api.jina.ai/v1/embeddings",
                "headers": {
                    "Authorization": f"Bearer {os.getenv('JINA_API_KEY', '')}",
                    "Content-Type": "application/json"
                },
                "free": True,  # 有免费额度
                "rate_limit": 1000,
                "batch_size": 100
            }
        }

    def _init_vector_db(self):
        """初始化向量数据库"""
        print("正在初始化向量数据库...")
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=settings.VECTOR_DB_DIR,
                settings=ChromaSettings(anonymized_telemetry=False)
            )

            self.collection = self.chroma_client.get_or_create_collection(
                name="huatuo_medical_qa_api",
                metadata={"description": "华佗医学知识图谱QA向量集合(API版)"}
            )
            print("向量数据库初始化成功")
        except Exception as e:
            print(f"向量数据库初始化失败: {e}")
            raise

    def encode_texts_huggingface(self, texts: List[str]) -> np.ndarray:
        """使用HuggingFace API编码"""
        config = self.api_configs["huggingface"]

        def query_api(payload):
            response = requests.post(config["url"], headers=config["headers"], json=payload)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"HuggingFace API错误: {response.status_code} - {response.text}")
                return None

        all_embeddings = []
        batch_size = config["batch_size"]

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            print(f"🌐 HuggingFace API处理批次 {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")

            result = query_api({"inputs": batch_texts})
            if result:
                all_embeddings.extend(result)

            # 避免API限制
            time.sleep(1)

        return np.array(all_embeddings)

    def encode_texts_cohere(self, texts: List[str]) -> np.ndarray:
        """使用Cohere API编码"""
        config = self.api_configs["cohere"]

        payload = {
            "texts": texts,
            "model": "embed-multilingual-v2.0",
            "input_type": "search_document"
        }

        print(f"🌐 Cohere API处理 {len(texts)} 个文本...")
        response = requests.post(config["url"], headers=config["headers"], json=payload)

        if response.status_code == 200:
            result = response.json()
            return np.array(result["embeddings"])
        else:
            print(f"Cohere API错误: {response.status_code} - {response.text}")
            raise Exception(f"Cohere API调用失败: {response.status_code}")

    def encode_texts_jina(self, texts: List[str]) -> np.ndarray:
        """使用Jina AI API编码"""
        config = self.api_configs["jina"]

        payload = {
            "input": texts,
            "model": "jina-embeddings-v2-base-zh"  # 支持中文
        }

        print(f"🌐 Jina AI API处理 {len(texts)} 个文本...")
        response = requests.post(config["url"], headers=config["headers"], json=payload)

        if response.status_code == 200:
            result = response.json()
            embeddings = [item["embedding"] for item in result["data"]]
            return np.array(embeddings)
        else:
            print(f"Jina AI API错误: {response.status_code} - {response.text}")
            raise Exception(f"Jina AI API调用失败: {response.status_code}")

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """编码文本为向量（支持缓存）"""
        try:
            # 如果启用缓存，使用智能缓存
            if self.enable_cache and self.vector_cache:
                return self.vector_cache.get_or_compute_vectors(
                    texts,
                    lambda missing_texts: self._compute_api_embeddings(missing_texts),
                    f"api_{self.api_provider}"
                )
            else:
                # 直接调用API计算向量
                return self._compute_api_embeddings(texts)

        except Exception as e:
            print(f"API向量化失败: {e}")
            print("🔄 尝试使用备用方案...")
            # 可以在这里添加备用的本地模型
            raise

    def _compute_api_embeddings(self, texts: List[str]) -> np.ndarray:
        """实际调用API计算向量"""
        print(f"🌐 使用 {self.api_provider} API进行向量化...")

        if self.api_provider == "huggingface":
            return self.encode_texts_huggingface(texts)
        elif self.api_provider == "cohere":
            return self.encode_texts_cohere(texts)
        elif self.api_provider == "jina":
            return self.encode_texts_jina(texts)
        else:
            raise ValueError(f"不支持的API提供商: {self.api_provider}")

    def add_documents(self, chunks: List[Dict[str, Any]]) -> None:
        """添加文档到向量数据库"""
        if not chunks:
            print("没有文档需要添加")
            return

        print(f"🌐 使用API向量化 {len(chunks)} 个文档块...")

        # 准备数据
        texts = [chunk['text'] for chunk in chunks]
        ids = [chunk['chunk_id'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]

        # API向量化
        embeddings = self.encode_texts(texts)

        # 智能调整数据库批处理大小
        total_chunks = len(chunks)
        if total_chunks > 50000:
            batch_size = 100  # 超大数据集用小批次
        elif total_chunks > 10000:
            batch_size = 200  # 大数据集用中等批次
        elif total_chunks > 1000:
            batch_size = 300  # 中等数据集
        else:
            batch_size = 500  # 小数据集用大批次

        print(f"📊 API数据库批处理大小: {batch_size} (总数据: {total_chunks})")

        for i in range(0, len(chunks), batch_size):
            end_idx = min(i + batch_size, len(chunks))

            batch_texts = texts[i:end_idx]
            batch_ids = ids[i:end_idx]
            batch_embeddings = embeddings[i:end_idx].tolist()
            batch_metadatas = metadatas[i:end_idx]

            try:
                self.collection.add(
                    documents=batch_texts,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
                print(f"✅ 已添加批次 {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
            except Exception as e:
                print(f"❌ 添加批次失败: {e}")
                continue

        print("🎉 API向量化完成！")

    def search_similar(self, query: str, top_k: int = None, threshold: float = None) -> List[Dict[str, Any]]:
        """搜索相似文档"""
        top_k = top_k or settings.TOP_K_RETRIEVAL
        threshold = threshold or settings.SIMILARITY_THRESHOLD

        if not self.collection:
            raise ValueError("向量数据库未初始化")

        try:
            # 编码查询文本
            query_embedding = self.encode_texts([query])[0].tolist()

            # 搜索相似文档
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )

            # 处理结果
            similar_docs = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    similarity = 1 - distance
                    if similarity >= threshold:
                        similar_docs.append({
                            'text': doc,
                            'metadata': metadata,
                            'similarity': similarity,
                            'rank': i + 1
                        })

            return similar_docs

        except Exception as e:
            print(f"相似文档搜索失败: {e}")
            return []

    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        if not self.collection:
            return {}

        try:
            count = self.collection.count()
            return {
                'total_documents': count,
                'collection_name': self.collection.name,
                'api_provider': self.api_provider,
                'embedding_type': 'API'
            }
        except Exception as e:
            print(f"获取统计信息失败: {e}")
            return {}
