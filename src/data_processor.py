"""
数据处理模块 - 处理华佗医学知识图谱QA数据
"""
import json
import pandas as pd
from typing import List, Dict, Any
from datasets import load_dataset
from tqdm import tqdm
import jieba
import re

from config import settings


class HuatuoDataProcessor:
    """华佗数据处理器"""

    def __init__(self):
        self.dataset = None
        self.processed_data = []

    def load_huatuo_dataset(self, split: str = "train", sample_size: int = None) -> None:
        """
        加载华佗数据集

        Args:
            split: 数据集分割 (train/validation/test)
            sample_size: 采样大小，None表示加载全部数据
        """
        print(f"正在加载华佗数据集 ({split})...")

        try:
            self.dataset = load_dataset(settings.HUATUO_DATASET, split=split)

            if sample_size and sample_size < len(self.dataset):
                self.dataset = self.dataset.select(range(sample_size))

            print(f"成功加载 {len(self.dataset)} 条数据")

        except Exception as e:
            print(f"加载数据集失败: {e}")
            raise

    def process_qa_pairs(self) -> List[Dict[str, Any]]:
        """
        处理问答对数据

        Returns:
            处理后的问答对列表
        """
        if not self.dataset:
            raise ValueError("请先加载数据集")

        processed_data = []

        print("正在处理问答对数据...")
        for item in tqdm(self.dataset):
            questions = item.get('questions', [])
            answers = item.get('answers', [])

            # 确保问题和答案都存在
            if questions and answers:
                question = questions[0] if isinstance(questions, list) else questions
                answer = "; ".join(answers) if isinstance(answers, list) else answers

                # 数据清洗
                question = self._clean_text(question)
                answer = self._clean_text(answer)

                if question and answer:
                    processed_item = {
                        'question': question,
                        'answer': answer,
                        'question_tokens': self._tokenize_chinese(question),
                        'answer_tokens': self._tokenize_chinese(answer),
                        'combined_text': f"问题: {question}\n答案: {answer}",
                        'metadata': {
                            'source': 'huatuo_kg_qa',
                            'question_length': len(question),
                            'answer_length': len(answer)
                        }
                    }
                    processed_data.append(processed_item)

        self.processed_data = processed_data
        print(f"成功处理 {len(processed_data)} 条问答对")
        return processed_data

    def _clean_text(self, text: str) -> str:
        """清洗文本"""
        if not text:
            return ""

        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text.strip())

        # 移除特殊字符（保留中文、英文、数字、常用标点）
        text = re.sub(r'[^\u4e00-\u9fff\w\s，。！？；：""''（）【】\-\+\*\/\=\<\>\%]', '', text)

        return text

    def _tokenize_chinese(self, text: str) -> List[str]:
        """中文分词"""
        return list(jieba.cut(text))

    def create_chunks(self, chunk_size: int = None, overlap: int = None) -> List[Dict[str, Any]]:
        """
        创建文本块用于向量化

        Args:
            chunk_size: 块大小
            overlap: 重叠大小

        Returns:
            文本块列表
        """
        chunk_size = chunk_size or settings.CHUNK_SIZE
        overlap = overlap or settings.CHUNK_OVERLAP

        if not self.processed_data:
            raise ValueError("请先处理问答对数据")

        chunks = []

        print("正在创建文本块...")
        for idx, item in enumerate(tqdm(self.processed_data)):
            combined_text = item['combined_text']

            # 如果文本较短，直接作为一个块
            if len(combined_text) <= chunk_size:
                chunk = {
                    'text': combined_text,
                    'chunk_id': f"{idx}_0",
                    'source_id': idx,
                    'metadata': item['metadata'].copy()
                }
                chunks.append(chunk)
            else:
                # 分割长文本
                text_chunks = self._split_text(combined_text, chunk_size, overlap)
                for chunk_idx, chunk_text in enumerate(text_chunks):
                    chunk = {
                        'text': chunk_text,
                        'chunk_id': f"{idx}_{chunk_idx}",
                        'source_id': idx,
                        'metadata': item['metadata'].copy()
                    }
                    chunks.append(chunk)

        print(f"成功创建 {len(chunks)} 个文本块")
        return chunks

    def _split_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """分割文本"""
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)

            if end >= len(text):
                break

            start = end - overlap

        return chunks

    def save_processed_data(self, filepath: str) -> None:
        """保存处理后的数据"""
        if not self.processed_data:
            raise ValueError("没有可保存的数据")

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.processed_data, f, ensure_ascii=False, indent=2)

        print(f"数据已保存到: {filepath}")

    def load_processed_data(self, filepath: str) -> List[Dict[str, Any]]:
        """加载已处理的数据"""
        with open(filepath, 'r', encoding='utf-8') as f:
            self.processed_data = json.load(f)

        print(f"从 {filepath} 加载了 {len(self.processed_data)} 条数据")
        return self.processed_data
