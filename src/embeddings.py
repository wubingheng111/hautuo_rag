"""
嵌入模型处理模块
"""
import numpy as np
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings as ChromaSettings
import os

from config import settings
from vector_cache import VectorCache

# 尝试导入sentence_transformers，如果失败则使用备用方案
from sentence_transformers import SentenceTransformer
SENTENCE_TRANSFORMERS_AVAILABLE = True
print("✅ sentence-transformers 可用")


class EmbeddingManager:
    """嵌入管理器"""

    def __init__(self, enable_cache: bool = True):
        self.model = None
        self.chroma_client = None
        self.collection = None
        self.use_lite_mode = False
        self.enable_cache = enable_cache

        # 初始化向量缓存
        if enable_cache:
            self.vector_cache = VectorCache()
            print("💾 向量缓存已启用")
        else:
            self.vector_cache = None
            print("⚠️ 向量缓存已禁用")

        # 如果sentence_transformers不可用，自动切换到轻量级模式
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            print("🔄 自动切换到轻量级嵌入模式")
            self._init_lite_mode()
        else:
            try:
                self._load_embedding_model()
            except Exception as e:
                print(f"⚠️ 标准嵌入模型加载失败: {e}")
                print("🔄 切换到轻量级嵌入模式")
                self._init_lite_mode()

        self._init_vector_db()

    def _init_lite_mode(self):
        """初始化轻量级模式"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            import jieba

            self.use_lite_mode = True
            self.vectorizer = TfidfVectorizer(
                max_features=384,  # 匹配原始embedding维度
                tokenizer=self._chinese_tokenizer,
                lowercase=False,
                ngram_range=(1, 2)
            )
            print("✅ TF-IDF轻量级模式初始化成功")
        except ImportError as e:
            print(f"❌ 轻量级模式也无法初始化: {e}")
            raise Exception("无法初始化任何嵌入模式，请检查依赖安装")

    def _chinese_tokenizer(self, text):
        """中文分词器"""
        try:
            import jieba
            return list(jieba.cut(text))
        except:
            # 如果jieba也不可用，使用简单的字符分割
            return list(text)

    def _load_embedding_model(self):
        """加载嵌入模型"""
        print(f"正在加载嵌入模型: {settings.EMBEDDING_MODEL}")
        try:
            # 使用缓存目录加速模型加载
            cache_dir = os.path.join(settings.DATA_DIR, "model_cache")
            os.makedirs(cache_dir, exist_ok=True)

            self.model = SentenceTransformer(settings.EMBEDDING_MODEL)

            # 检查并设置设备
            import torch
            if torch.cuda.is_available():
                device = 'cuda'
                gpu_name = torch.cuda.get_device_name(0)
                print(f"🚀 使用GPU加速: {gpu_name}")
            else:
                device = 'cpu'
                print("💻 使用CPU模式")

            # 将模型移动到GPU
            self.model = self.model.to(device)
            print(f"✅ 嵌入模型加载成功，设备: {device}")
        except Exception as e:
            print(f"嵌入模型加载失败: {e}")
            # 备用模型
            print("尝试加载备用模型...")
            try:
                cache_dir = os.path.join(settings.DATA_DIR, "model_cache")
                self.model = SentenceTransformer('all-MiniLM-L6-v2')

                # 备用模型也要移动到GPU
                import torch
                if torch.cuda.is_available():
                    device = 'cuda'
                    gpu_name = torch.cuda.get_device_name(0)
                    print(f"🚀 备用模型使用GPU加速: {gpu_name}")
                else:
                    device = 'cpu'
                    print("💻 备用模型使用CPU模式")

                self.model = self.model.to(device)
                print(f"✅ 备用模型加载成功，设备: {device}")
            except Exception as e2:
                print(f"备用模型也加载失败: {e2}")
                raise

    def _init_vector_db(self):
        """初始化向量数据库"""
        print("正在初始化向量数据库...")
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=settings.VECTOR_DB_DIR,
                settings=ChromaSettings(anonymized_telemetry=False)
            )

            # 创建或获取集合
            self.collection = self.chroma_client.get_or_create_collection(
                name="huatuo_medical_qa",
                metadata={"description": "华佗医学知识图谱QA向量集合"}
            )
            print("向量数据库初始化成功")
        except Exception as e:
            print(f"向量数据库初始化失败: {e}")
            raise

    def encode_texts(self, texts: List[str], batch_size: int = None) -> np.ndarray:
        """
        编码文本为向量（支持缓存）

        Args:
            texts: 文本列表
            batch_size: 批处理大小

        Returns:
            向量数组
        """
        try:
            # 获取模型名称用于缓存
            model_name = "tfidf" if self.use_lite_mode else getattr(self.model, 'model_name', settings.EMBEDDING_MODEL)

            # 如果启用缓存，使用智能缓存
            if self.enable_cache and self.vector_cache:
                return self.vector_cache.get_or_compute_vectors(
                    texts,
                    lambda missing_texts: self._compute_embeddings(missing_texts, batch_size),
                    model_name
                )
            else:
                # 直接计算向量
                return self._compute_embeddings(texts, batch_size)

        except Exception as e:
            print(f"文本编码失败: {e}")
            raise

    def _compute_embeddings(self, texts: List[str], batch_size: int = None) -> np.ndarray:
        """实际计算向量的方法"""
        if self.use_lite_mode:
            # 使用TF-IDF模式
            if not hasattr(self.vectorizer, 'vocabulary_'):
                print("首次使用，正在训练TF-IDF模型...")
                embeddings = self.vectorizer.fit_transform(texts)
            else:
                embeddings = self.vectorizer.transform(texts)

            # 转换为密集矩阵
            dense_embeddings = embeddings.toarray()

            # 归一化
            norms = np.linalg.norm(dense_embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1  # 避免除零
            normalized_embeddings = dense_embeddings / norms

            return normalized_embeddings
        else:
            # 使用sentence_transformers模式
            if not self.model:
                raise ValueError("嵌入模型未加载")

            # 根据设备和文本数量自动调整批处理大小
            if batch_size is None:
                import torch
                if torch.cuda.is_available() and self.model.device.type == 'cuda':
                    # GPU模式：使用更大的批处理大小
                    batch_size = min(256, len(texts))  # 最大256，避免显存不足
                    print(f"🚀 GPU模式，设备: {self.model.device}，批处理大小: {batch_size}")
                else:
                    # CPU模式：使用较小的批处理大小
                    batch_size = min(64, len(texts))
                    print(f"💻 CPU模式，设备: {self.model.device}，批处理大小: {batch_size}")

            # 显示GPU内存使用情况
            import torch
            if torch.cuda.is_available() and self.model.device.type == 'cuda':
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_cached = torch.cuda.memory_reserved() / 1024**3
                print(f"📊 GPU内存使用: 已分配 {memory_allocated:.1f}GB, 已缓存 {memory_cached:.1f}GB")

            # 使用优化参数加速编码
            embeddings = self.model.encode(
                texts,
                show_progress_bar=True,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,  # 归一化向量，提高检索效果
                device=self.model.device  # 确保使用正确的设备
            )
            return embeddings

    def add_documents(self, chunks: List[Dict[str, Any]]) -> None:
        """
        添加文档到向量数据库

        Args:
            chunks: 文档块列表
        """
        if not chunks:
            print("没有文档需要添加")
            return

        print(f"正在添加 {len(chunks)} 个文档块到向量数据库...")

        # 准备数据
        texts = [chunk['text'] for chunk in chunks]
        ids = [chunk['chunk_id'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]

        # 生成嵌入向量
        embeddings = self.encode_texts(texts)

        # 智能调整数据库批处理大小
        total_chunks = len(chunks)
        if total_chunks > 50000:
            db_batch_size = 100  # 超大数据集用小批次
        elif total_chunks > 10000:
            db_batch_size = 200  # 大数据集用中等批次
        elif total_chunks > 1000:
            db_batch_size = 300  # 中等数据集
        else:
            db_batch_size = 500  # 小数据集用大批次

        print(f"📊 数据库批处理大小: {db_batch_size} (总数据: {total_chunks})")

        for i in range(0, len(chunks), db_batch_size):
            end_idx = min(i + db_batch_size, len(chunks))

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
                print(f"已添加批次 {i//db_batch_size + 1}/{(len(chunks)-1)//db_batch_size + 1}")
            except Exception as e:
                print(f"添加批次失败: {e}")
                continue

        print("文档添加完成")

    def search_similar(
        self,
        query: str,
        top_k: int = None,
        threshold: float = None
    ) -> List[Dict[str, Any]]:
        """
        搜索相似文档

        Args:
            query: 查询文本
            top_k: 返回结果数量
            threshold: 相似度阈值

        Returns:
            相似文档列表
        """
        top_k = top_k or settings.TOP_K_RETRIEVAL
        threshold = threshold or settings.SIMILARITY_THRESHOLD

        if not self.collection:
            raise ValueError("向量数据库未初始化")

        try:
            # 编码查询文本
            query_embedding = self.encode_texts([query])[0].tolist()

            # 搜索相似文档 - 先获取更多结果
            search_k = max(top_k * 3, 50)  # 搜索更多结果以确保有足够的候选
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=search_k,
                include=['documents', 'metadatas', 'distances']
            )

            # 处理结果
            similar_docs = []
            if results['documents'] and results['documents'][0]:
                print(f"🔍 ChromaDB返回 {len(results['documents'][0])} 条原始结果")

                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    # 调试信息：显示前几个结果的距离
                    if i < 3:
                        print(f"   结果 {i+1}: 距离={distance:.4f}")

                    # ChromaDB使用的是距离（越小越相似），需要转换为相似度
                    # 根据距离范围判断使用的距离类型并相应转换
                    if distance <= 2.0:  # 可能是余弦距离或归一化的欧几里得距离
                        # 对于余弦距离: similarity = 1 - distance
                        # 但如果是欧几里得距离，需要不同的转换
                        if distance >= 1.0:  # 很可能是欧几里得距离
                            # 使用指数衰减转换：距离越大，相似度越小
                            similarity = max(0, 1 - (distance - 1) * 0.5)  # 线性衰减
                        else:
                            similarity = 1 - distance  # 余弦距离
                    else:  # 大距离值，使用倒数转换
                        similarity = 1 / (1 + distance * 0.5)

                    # 动态调整阈值 - 根据距离分布自适应
                    if i == 0:  # 第一个结果，设置基准
                        min_distance = distance
                        # 如果最小距离都很大，说明没有很相似的结果，降低阈值
                        if min_distance > 1.5:
                            effective_threshold = 0.01  # 极低阈值
                        elif min_distance > 1.0:
                            effective_threshold = 0.1   # 低阈值
                        else:
                            effective_threshold = min(threshold, 0.3)  # 正常阈值
                    else:
                        effective_threshold = getattr(locals(), 'effective_threshold', 0.1)

                    if similarity >= effective_threshold:
                        # 生成文档ID（如果metadata中没有的话）
                        doc_id = metadata.get('id', f"doc_{hash(doc[:100]) % 100000}")

                        similar_docs.append({
                            'id': doc_id,
                            'text': doc,
                            'metadata': metadata,
                            'similarity': similarity,
                            'distance': distance,  # 保留原始距离用于调试
                            'rank': i + 1
                        })

                print(f"📊 阈值 {effective_threshold} 过滤后: {len(similar_docs)} 条结果")

                # 如果仍然没有结果，降低阈值再试一次
                if not similar_docs and threshold > 0.05:
                    print("🔄 降低阈值重试...")
                    for i, (doc, metadata, distance) in enumerate(zip(
                        results['documents'][0][:10],  # 只检查前10个
                        results['metadatas'][0][:10],
                        results['distances'][0][:10]
                    )):
                        if distance <= 2.0:
                            similarity = 1 - distance
                        else:
                            similarity = 1 / (1 + distance)

                        if similarity >= 0.01:  # 极低阈值
                            doc_id = metadata.get('id', f"doc_{hash(doc[:100]) % 100000}")
                            similar_docs.append({
                                'id': doc_id,
                                'text': doc,
                                'metadata': metadata,
                                'similarity': similarity,
                                'distance': distance,
                                'rank': i + 1
                            })

                    print(f"🆘 极低阈值 0.01 结果: {len(similar_docs)} 条")

            # 按相似度排序并返回top_k
            similar_docs.sort(key=lambda x: x['similarity'], reverse=True)
            return similar_docs[:top_k]

        except Exception as e:
            print(f"相似文档搜索失败: {e}")
            import traceback
            traceback.print_exc()
            return []

    def get_document_by_id(self, doc_id: str) -> Dict[str, Any]:
        """
        根据文档ID获取文档详情

        Args:
            doc_id: 文档ID

        Returns:
            文档详情
        """
        if not self.collection:
            return None

        try:
            # 尝试通过ID直接查询
            results = self.collection.get(
                ids=[doc_id],
                include=['documents', 'metadatas']
            )

            if results['documents'] and len(results['documents']) > 0:
                return {
                    'id': doc_id,
                    'text': results['documents'][0],
                    'metadata': results['metadatas'][0] if results['metadatas'] else {}
                }

            # 如果直接查询失败，尝试通过元数据查询
            all_results = self.collection.get(
                include=['documents', 'metadatas']
            )

            if all_results['documents']:
                for i, (doc, metadata) in enumerate(zip(all_results['documents'], all_results['metadatas'] or [])):
                    # 检查是否匹配文档ID
                    if metadata.get('id') == doc_id or f"doc_{hash(doc[:100]) % 100000}" == doc_id:
                        return {
                            'id': doc_id,
                            'text': doc,
                            'metadata': metadata or {}
                        }

            return None

        except Exception as e:
            print(f"获取文档详情失败: {e}")
            return None

    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        if not self.collection:
            return {}

        try:
            count = self.collection.count()
            return {
                'total_documents': count,
                'collection_name': self.collection.name,
                'embedding_dimension': settings.EMBEDDING_DIMENSION
            }
        except Exception as e:
            print(f"获取统计信息失败: {e}")
            return {}

    def clear_collection(self) -> None:
        """清空集合"""
        if self.collection:
            try:
                # 删除现有集合
                self.chroma_client.delete_collection(name="huatuo_medical_qa")
                # 重新创建集合
                self.collection = self.chroma_client.create_collection(
                    name="huatuo_medical_qa",
                    metadata={"description": "华佗医学知识图谱QA向量集合"}
                )
                print("集合已清空")
            except Exception as e:
                print(f"清空集合失败: {e}")
