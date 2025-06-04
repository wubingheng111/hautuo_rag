"""
工具函数模块
"""
import re
import json
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
import jieba
from functools import wraps


def timer(func):
    """计时装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} 执行时间: {end_time - start_time:.2f}秒")
        return result
    return wrapper


def clean_medical_text(text: str) -> str:
    """
    清洗医学文本

    Args:
        text: 原始文本

    Returns:
        清洗后的文本
    """
    if not text:
        return ""

    # 移除多余的空白字符
    text = re.sub(r'\s+', ' ', text.strip())

    # 移除HTML标签
    text = re.sub(r'<[^>]+>', '', text)

    # 移除特殊字符，保留中文、英文、数字、常用标点
    text = re.sub(r'[^\u4e00-\u9fff\w\s，。！？；：""''（）【】\-\+\*\/\=\<\>\%]', '', text)

    # 移除多余的标点符号
    text = re.sub(r'[，。；：]{2,}', '，', text)

    return text


def extract_medical_keywords(text: str) -> List[str]:
    """
    提取医学关键词

    Args:
        text: 输入文本

    Returns:
        关键词列表
    """
    # 使用jieba分词
    words = jieba.cut(text)

    # 医学相关关键词模式
    medical_patterns = [
        r'.*病$',      # 以"病"结尾
        r'.*症$',      # 以"症"结尾
        r'.*炎$',      # 以"炎"结尾
        r'.*癌$',      # 以"癌"结尾
        r'.*瘤$',      # 以"瘤"结尾
        r'.*药$',      # 以"药"结尾
        r'.*疗$',      # 以"疗"结尾
        r'.*治$',      # 以"治"结尾
        r'.*检$',      # 以"检"结尾
        r'.*查$',      # 以"查"结尾
    ]

    keywords = []
    for word in words:
        word = word.strip()
        if len(word) >= 2:  # 至少2个字符
            # 检查是否匹配医学模式
            for pattern in medical_patterns:
                if re.match(pattern, word):
                    keywords.append(word)
                    break
            # 或者是常见医学词汇
        elif word in ['诊断', '治疗', '症状', '病因', '预防', '手术', '药物', '检查']:
                keywords.append(word)

    return list(set(keywords))  # 去重


def format_medical_response(response: str) -> str:
    """
    格式化医学回复

    Args:
        response: 原始回复

    Returns:
        格式化后的回复
    """
    if not response:
        return ""

    # 添加医学免责声明
    disclaimer = "\n\n⚠️ 免责声明：以上信息仅供参考，不能替代专业医生的诊断和治疗建议。如有健康问题，请及时就医咨询专业医生。"

    # 如果回复中没有免责声明，则添加
    if "免责" not in response and "仅供参考" not in response:
        response += disclaimer

    return response


def validate_medical_query(query: str) -> Dict[str, Any]:
    """
    验证医学查询 - 优化版

    Args:
        query: 用户查询

    Returns:
        验证结果
    """
    result = {
        'is_valid': True,
        'is_medical': True,  # 默认认为是医学相关
        'confidence': 0.8,   # 默认高置信度
        'keywords': [],
        'suggestions': []
    }

    if not query or len(query.strip()) < 2:
        result['is_valid'] = False
        result['suggestions'].append("请输入有效的问题")
        return result

    # 扩展的医学关键词库
    medical_keywords = {
        # 疾病相关
        '疾病': ['病', '症', '综合征', '炎', '癌', '瘤', '肿瘤', '感染', '中毒', '损伤'],
        # 症状相关
        '症状': ['疼', '痛', '酸', '胀', '麻', '痒', '肿', '红', '热', '冷', '乏力', '疲劳',
                '头晕', '头痛', '恶心', '呕吐', '腹泻', '便秘', '发烧', '发热', '咳嗽',
                '气短', '胸闷', '心悸', '失眠', '多梦'],
        # 身体部位
        '部位': ['头', '颈', '胸', '腹', '背', '腰', '臀', '腿', '脚', '手', '臂', '肩',
                '心脏', '肺', '肝', '肾', '胃', '肠', '脑', '眼', '耳', '鼻', '口', '喉'],
        # 医疗行为
        '医疗': ['治疗', '诊断', '检查', '化验', '手术', '用药', '服药', '注射', '输液',
                '康复', '护理', '预防', '保健', '体检', '复查', '随访'],
        # 药物相关
        '药物': ['药', '片', '胶囊', '注射液', '口服液', '软膏', '贴剂', '滴剂'],
        # 医学专业词汇
        '专业': ['血压', '血糖', '血脂', '心率', '体温', '白细胞', '红细胞', '血小板',
                'CT', 'MRI', 'B超', 'X光', '心电图', '化疗', '放疗']
    }

    # 非医学关键词（用于排除）
    non_medical_keywords = [
        '天气', '股票', '新闻', '娱乐', '游戏', '电影', '音乐', '旅游', '美食',
        '购物', '学习', '工作', '编程', '技术', '数学', '物理', '化学'
    ]

    query_lower = query.lower()

    # 检查是否包含非医学关键词
    non_medical_score = 0
    for keyword in non_medical_keywords:
        if keyword in query_lower:
            non_medical_score += 1

    # 如果包含明显的非医学关键词，降低医学相关性
    if non_medical_score > 0:
        result['is_medical'] = False
        result['confidence'] = 0.2
        result['suggestions'].append("您的问题似乎不是医学相关，但我仍会尝试回答")
        return result

    # 计算医学相关性得分
    medical_score = 0
    found_keywords = []

    for category, keywords in medical_keywords.items():
        for keyword in keywords:
            if keyword in query:
                medical_score += 1
                found_keywords.append(keyword)

    result['keywords'] = found_keywords

    # 特殊医学问题模式检测
    medical_patterns = [
        r'.*怎么治.*',
        r'.*什么原因.*',
        r'.*有什么症状.*',
        r'.*如何预防.*',
        r'.*能吃.*药.*',
        r'.*需要检查.*',
        r'.*是什么病.*',
        r'.*危险.*因素.*',
        r'.*注意.*事项.*'
    ]

    pattern_match = False
    for pattern in medical_patterns:
        if re.search(pattern, query):
            pattern_match = True
            medical_score += 2
            break

    # 计算最终置信度
    if medical_score >= 3 or pattern_match:
        result['confidence'] = 0.9
        result['is_medical'] = True
    elif medical_score >= 1:
        result['confidence'] = 0.7
        result['is_medical'] = True
    else:
        # 即使没有明确的医学关键词，也给一个基础的医学相关性
        # 因为用户使用的是医学RAG系统
        result['confidence'] = 0.6
        result['is_medical'] = True
        result['suggestions'].append("如果您的问题是医学相关的，建议提供更详细的描述")

    return result


def chunk_text_by_sentences(text: str, max_length: int = 500) -> List[str]:
    """
    按句子分割文本

    Args:
        text: 输入文本
        max_length: 最大长度

    Returns:
        文本块列表
    """
    if not text:
        return []

    # 按句号分割
    sentences = re.split(r'[。！？]', text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # 如果添加这个句子不会超过最大长度
        if len(current_chunk + sentence) <= max_length:
            current_chunk += sentence + "。"
        else:
            # 保存当前块
            if current_chunk:
                chunks.append(current_chunk.strip())

            # 开始新块
            if len(sentence) <= max_length:
                current_chunk = sentence + "。"
            else:
                # 句子太长，强制分割
                chunks.append(sentence[:max_length])
                current_chunk = ""

    # 添加最后一块
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    计算文本相似度（简单版本）

    Args:
        text1: 文本1
        text2: 文本2

    Returns:
        相似度分数 (0-1)
    """
    if not text1 or not text2:
        return 0.0

    # 分词
    words1 = set(jieba.cut(text1))
    words2 = set(jieba.cut(text2))

    # 计算交集和并集
    intersection = words1.intersection(words2)
    union = words1.union(words2)

    if not union:
        return 0.0

    # Jaccard相似度
    similarity = len(intersection) / len(union)
    return similarity


def save_json(data: Any, filepath: str) -> bool:
    """
    保存JSON文件

    Args:
        data: 要保存的数据
        filepath: 文件路径

    Returns:
        是否保存成功
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"保存JSON文件失败: {e}")
        return False


def load_json(filepath: str) -> Optional[Any]:
    """
    加载JSON文件

    Args:
        filepath: 文件路径

    Returns:
        加载的数据，失败返回None
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载JSON文件失败: {e}")
        return None


def format_timestamp(timestamp: str = None) -> str:
    """
    格式化时间戳

    Args:
        timestamp: ISO格式时间戳，None表示当前时间

    Returns:
        格式化的时间字符串
    """
    if timestamp:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    else:
        dt = datetime.now()

    return dt.strftime('%Y-%m-%d %H:%M:%S')


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    截断文本

    Args:
        text: 输入文本
        max_length: 最大长度
        suffix: 后缀

    Returns:
        截断后的文本
    """
    if not text or len(text) <= max_length:
        return text

    return text[:max_length - len(suffix)] + suffix
