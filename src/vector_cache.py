"""
向量缓存管理器 - 本地存储向量，避免重复计算
"""
import os
import json
import pickle
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
import time

from config import settings


class VectorCache:
    """向量缓存管理器"""
    
    def __init__(self, cache_dir: str = None):
        self.cache_dir = Path(cache_dir or os.path.join(settings.DATA_DIR, "vector_cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 缓存文件路径
        self.vectors_file = self.cache_dir / "vectors.pkl"
        self.metadata_file = self.cache_dir / "metadata.json"
        self.index_file = self.cache_dir / "text_index.json"
        
        # 加载现有缓存
        self.vectors_cache = self._load_vectors()
        self.metadata_cache = self._load_metadata()
        self.text_index = self._load_text_index()
        
        print(f"📦 向量缓存初始化完成，缓存目录: {self.cache_dir}")
        print(f"📊 当前缓存: {len(self.vectors_cache)} 个向量")
    
    def _load_vectors(self) -> Dict[str, np.ndarray]:
        """加载向量缓存"""
        if self.vectors_file.exists():
            try:
                with open(self.vectors_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"⚠️ 加载向量缓存失败: {e}")
        return {}
    
    def _load_metadata(self) -> Dict[str, Dict]:
        """加载元数据缓存"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ 加载元数据缓存失败: {e}")
        return {}
    
    def _load_text_index(self) -> Dict[str, str]:
        """加载文本索引"""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ 加载文本索引失败: {e}")
        return {}
    
    def _save_vectors(self):
        """保存向量缓存"""
        try:
            with open(self.vectors_file, 'wb') as f:
                pickle.dump(self.vectors_cache, f)
        except Exception as e:
            print(f"❌ 保存向量缓存失败: {e}")
    
    def _save_metadata(self):
        """保存元数据缓存"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"❌ 保存元数据缓存失败: {e}")
    
    def _save_text_index(self):
        """保存文本索引"""
        try:
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(self.text_index, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"❌ 保存文本索引失败: {e}")
    
    def _get_text_hash(self, text: str, model_name: str = "default") -> str:
        """生成文本哈希值"""
        # 包含模型名称，确保不同模型的向量不会混淆
        content = f"{model_name}:{text}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def get_cached_vectors(self, texts: List[str], model_name: str = "default") -> tuple:
        """
        获取缓存的向量
        
        Returns:
            (cached_vectors, missing_texts, missing_indices)
        """
        cached_vectors = []
        missing_texts = []
        missing_indices = []
        
        for i, text in enumerate(texts):
            text_hash = self._get_text_hash(text, model_name)
            
            if text_hash in self.vectors_cache:
                cached_vectors.append(self.vectors_cache[text_hash])
            else:
                cached_vectors.append(None)
                missing_texts.append(text)
                missing_indices.append(i)
        
        hit_rate = (len(texts) - len(missing_texts)) / len(texts) * 100
        print(f"📊 缓存命中率: {hit_rate:.1f}% ({len(texts) - len(missing_texts)}/{len(texts)})")
        
        return cached_vectors, missing_texts, missing_indices
    
    def cache_vectors(self, texts: List[str], vectors: np.ndarray, model_name: str = "default"):
        """缓存向量"""
        if len(texts) != len(vectors):
            raise ValueError("文本数量和向量数量不匹配")
        
        cached_count = 0
        for text, vector in zip(texts, vectors):
            text_hash = self._get_text_hash(text, model_name)
            
            # 缓存向量
            self.vectors_cache[text_hash] = vector
            
            # 缓存元数据
            self.metadata_cache[text_hash] = {
                'model_name': model_name,
                'vector_dim': len(vector),
                'cached_time': time.time(),
                'text_length': len(text)
            }
            
            # 缓存文本索引（用于调试和查看）
            self.text_index[text_hash] = text[:100] + "..." if len(text) > 100 else text
            
            cached_count += 1
        
        # 保存到磁盘
        self._save_vectors()
        self._save_metadata()
        self._save_text_index()
        
        print(f"💾 已缓存 {cached_count} 个新向量")
    
    def get_or_compute_vectors(self, texts: List[str], compute_func, model_name: str = "default") -> np.ndarray:
        """
        获取或计算向量（智能缓存）
        
        Args:
            texts: 文本列表
            compute_func: 计算向量的函数
            model_name: 模型名称
            
        Returns:
            向量数组
        """
        print(f"🔍 检查 {len(texts)} 个文本的向量缓存...")
        
        # 检查缓存
        cached_vectors, missing_texts, missing_indices = self.get_cached_vectors(texts, model_name)
        
        if not missing_texts:
            print("✅ 所有向量都已缓存，直接返回")
            return np.array([v for v in cached_vectors if v is not None])
        
        print(f"🔄 需要计算 {len(missing_texts)} 个新向量...")
        
        # 计算缺失的向量
        new_vectors = compute_func(missing_texts)
        
        # 缓存新向量
        self.cache_vectors(missing_texts, new_vectors, model_name)
        
        # 合并结果
        result_vectors = []
        new_vector_idx = 0
        
        for i, cached_vector in enumerate(cached_vectors):
            if cached_vector is not None:
                result_vectors.append(cached_vector)
            else:
                result_vectors.append(new_vectors[new_vector_idx])
                new_vector_idx += 1
        
        return np.array(result_vectors)
    
    def clear_cache(self, model_name: str = None):
        """清空缓存"""
        if model_name:
            # 清空特定模型的缓存
            to_remove = []
            for text_hash, metadata in self.metadata_cache.items():
                if metadata.get('model_name') == model_name:
                    to_remove.append(text_hash)
            
            for text_hash in to_remove:
                self.vectors_cache.pop(text_hash, None)
                self.metadata_cache.pop(text_hash, None)
                self.text_index.pop(text_hash, None)
            
            print(f"🗑️ 已清空模型 {model_name} 的缓存 ({len(to_remove)} 个向量)")
        else:
            # 清空所有缓存
            count = len(self.vectors_cache)
            self.vectors_cache.clear()
            self.metadata_cache.clear()
            self.text_index.clear()
            print(f"🗑️ 已清空所有缓存 ({count} 个向量)")
        
        # 保存更改
        self._save_vectors()
        self._save_metadata()
        self._save_text_index()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        if not self.metadata_cache:
            return {
                'total_vectors': 0,
                'models': {},
                'cache_size_mb': 0,
                'oldest_cache': None,
                'newest_cache': None
            }
        
        # 按模型统计
        models = {}
        cache_times = []
        
        for metadata in self.metadata_cache.values():
            model_name = metadata.get('model_name', 'unknown')
            if model_name not in models:
                models[model_name] = 0
            models[model_name] += 1
            cache_times.append(metadata.get('cached_time', 0))
        
        # 计算缓存文件大小
        cache_size = 0
        for file_path in [self.vectors_file, self.metadata_file, self.index_file]:
            if file_path.exists():
                cache_size += file_path.stat().st_size
        
        return {
            'total_vectors': len(self.vectors_cache),
            'models': models,
            'cache_size_mb': cache_size / 1024 / 1024,
            'oldest_cache': time.ctime(min(cache_times)) if cache_times else None,
            'newest_cache': time.ctime(max(cache_times)) if cache_times else None,
            'cache_dir': str(self.cache_dir)
        }
    
    def export_cache_info(self) -> str:
        """导出缓存信息"""
        stats = self.get_cache_stats()
        
        info = f"""
📦 向量缓存信息
{'='*50}
总向量数: {stats['total_vectors']}
缓存大小: {stats['cache_size_mb']:.2f} MB
缓存目录: {stats['cache_dir']}

📊 按模型统计:
"""
        for model, count in stats['models'].items():
            info += f"  • {model}: {count} 个向量\n"
        
        if stats['oldest_cache']:
            info += f"\n⏰ 最早缓存: {stats['oldest_cache']}"
            info += f"\n⏰ 最新缓存: {stats['newest_cache']}"
        
        return info
