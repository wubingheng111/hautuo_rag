#!/usr/bin/env python3
"""
分析心血管数据的长度分布
"""
import json

def analyze_data_length():
    """分析数据长度"""
    print("📊 分析心血管数据长度分布")
    print("=" * 50)
    
    try:
        with open('cardiovascular_qa_data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"总数据量: {len(data):,} 条")
        
        # 分析长度
        question_lengths = []
        answer_lengths = []
        combined_lengths = []
        
        for item in data[:5000]:  # 分析前5000条
            question = item['question']
            answer = item['answer']
            combined = f"问题: {question}\n答案: {answer}"
            
            question_lengths.append(len(question))
            answer_lengths.append(len(answer))
            combined_lengths.append(len(combined))
        
        # 统计信息
        print(f"\n📏 长度统计 (基于前5000条数据):")
        print(f"问题平均长度: {sum(question_lengths)/len(question_lengths):.1f} 字符")
        print(f"答案平均长度: {sum(answer_lengths)/len(answer_lengths):.1f} 字符")
        print(f"组合平均长度: {sum(combined_lengths)/len(combined_lengths):.1f} 字符")
        
        print(f"\n📊 长度范围:")
        print(f"问题: {min(question_lengths)} - {max(question_lengths)} 字符")
        print(f"答案: {min(answer_lengths)} - {max(answer_lengths)} 字符")
        print(f"组合: {min(combined_lengths)} - {max(combined_lengths)} 字符")
        
        # 长度分布
        print(f"\n📈 组合文本长度分布:")
        very_short = sum(1 for l in combined_lengths if l < 50)
        short = sum(1 for l in combined_lengths if 50 <= l < 150)
        medium = sum(1 for l in combined_lengths if 150 <= l < 300)
        long = sum(1 for l in combined_lengths if 300 <= l < 500)
        very_long = sum(1 for l in combined_lengths if l >= 500)
        
        total = len(combined_lengths)
        print(f"极短文本(<50字符): {very_short:,} 条 ({very_short/total*100:.1f}%)")
        print(f"短文本(50-150字符): {short:,} 条 ({short/total*100:.1f}%)")
        print(f"中等文本(150-300字符): {medium:,} 条 ({medium/total*100:.1f}%)")
        print(f"长文本(300-500字符): {long:,} 条 ({long/total*100:.1f}%)")
        print(f"超长文本(>=500字符): {very_long:,} 条 ({very_long/total*100:.1f}%)")
        
        # 中位数
        sorted_lengths = sorted(combined_lengths)
        median = sorted_lengths[len(sorted_lengths)//2]
        p75 = sorted_lengths[int(len(sorted_lengths)*0.75)]
        p90 = sorted_lengths[int(len(sorted_lengths)*0.90)]
        
        print(f"\n📊 长度分位数:")
        print(f"中位数(50%): {median} 字符")
        print(f"75分位数: {p75} 字符")
        print(f"90分位数: {p90} 字符")
        
        # 建议的chunk设置
        print(f"\n💡 建议的配置:")
        if median < 100:
            print(f"建议CHUNK_SIZE: 200-300 (当前中位数: {median})")
        elif median < 200:
            print(f"建议CHUNK_SIZE: 300-400 (当前中位数: {median})")
        else:
            print(f"建议CHUNK_SIZE: 400-500 (当前中位数: {median})")
        
        print(f"建议CHUNK_OVERLAP: 20-50")
        print(f"建议SIMILARITY_THRESHOLD: 0.3-0.5 (当前可能过高)")
        
        # 显示一些示例
        print(f"\n📝 数据示例:")
        for i, item in enumerate(data[:3], 1):
            combined = f"问题: {item['question']}\n答案: {item['answer']}"
            print(f"\n示例 {i} (长度: {len(combined)} 字符):")
            print(f"问题: {item['question']}")
            print(f"答案: {item['answer'][:100]}{'...' if len(item['answer']) > 100 else ''}")
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")

if __name__ == "__main__":
    analyze_data_length()
