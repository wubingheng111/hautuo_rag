#!/usr/bin/env python3
"""
心血管专科问答系统主程序
基于华佗数据集的心血管专项数据构建的RAG系统
"""
import os
import sys
import json
import argparse
from pathlib import Path

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def setup_cardiovascular_system():
    """初始化心血管专科系统"""
    print("🫀 心血管专科问答系统")
    print("=" * 50)
    
    try:
        from cardio_specialist import CardiovascularSpecialist
        from data_processor import HuatuoDataProcessor
        
        # 检查心血管数据是否存在
        cardio_data_file = "cardiovascular_qa_data.json"
        if not os.path.exists(cardio_data_file):
            print("❌ 心血管数据文件不存在，请先运行数据抽取:")
            print("   python analyze_huatuo_dataset.py cardiovascular")
            return None
        
        # 加载心血管数据
        print("📥 正在加载心血管专项数据...")
        with open(cardio_data_file, 'r', encoding='utf-8') as f:
            cardio_data = json.load(f)
        
        print(f"✅ 成功加载 {len(cardio_data):,} 条心血管问答数据")
        
        # 初始化心血管专科系统
        specialist = CardiovascularSpecialist()
        
        # 构建向量数据库
        print("🔧 正在构建心血管知识向量数据库...")
        specialist.build_knowledge_base(cardio_data)
        
        print("✅ 心血管专科系统初始化完成!")
        return specialist
        
    except Exception as e:
        print(f"❌ 系统初始化失败: {e}")
        return None

def interactive_mode(specialist):
    """交互式问答模式"""
    print("\n💬 进入心血管专科问答模式")
    print("输入 'quit' 或 'exit' 退出")
    print("-" * 50)
    
    while True:
        try:
            question = input("\n🩺 请输入您的心血管相关问题: ").strip()
            
            if question.lower() in ['quit', 'exit', '退出']:
                print("👋 感谢使用心血管专科问答系统!")
                break
            
            if not question:
                continue
            
            print("\n🤔 正在分析您的问题...")
            
            # 获取专业回答
            response = specialist.get_cardiovascular_answer(question)
            
            print(f"\n🫀 心血管专科回答:")
            print("-" * 30)
            print(response['answer'])
            
            if response.get('references'):
                print(f"\n📚 参考资料:")
                for i, ref in enumerate(response['references'][:3], 1):
                    print(f"{i}. {ref['question']} -> {ref['answer'][:100]}...")
            
            if response.get('confidence'):
                print(f"\n📊 置信度: {response['confidence']:.2%}")
                
        except KeyboardInterrupt:
            print("\n\n👋 感谢使用心血管专科问答系统!")
            break
        except Exception as e:
            print(f"\n❌ 处理问题时出错: {e}")

def batch_mode(specialist, questions_file, output_file):
    """批量问答模式"""
    print(f"\n📋 批量处理模式")
    print(f"输入文件: {questions_file}")
    print(f"输出文件: {output_file}")
    
    try:
        # 读取问题列表
        with open(questions_file, 'r', encoding='utf-8') as f:
            if questions_file.endswith('.json'):
                questions = json.load(f)
            else:
                questions = [line.strip() for line in f if line.strip()]
        
        results = []
        
        for i, question in enumerate(questions, 1):
            print(f"处理问题 {i}/{len(questions)}: {question[:50]}...")
            
            response = specialist.get_cardiovascular_answer(question)
            
            result = {
                "question": question,
                "answer": response['answer'],
                "confidence": response.get('confidence', 0),
                "references": response.get('references', [])
            }
            results.append(result)
        
        # 保存结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 批量处理完成，结果已保存到: {output_file}")
        
    except Exception as e:
        print(f"❌ 批量处理失败: {e}")

def web_mode():
    """启动Web界面模式"""
    print("\n🌐 启动Web界面...")
    
    try:
        import subprocess
        subprocess.run([sys.executable, "-m", "streamlit", "run", "ui/cardio_app.py"], check=True)
    except Exception as e:
        print(f"❌ Web界面启动失败: {e}")
        print("请确保已安装streamlit: pip install streamlit")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="心血管专科问答系统")
    parser.add_argument("--mode", choices=["interactive", "batch", "web"], 
                       default="interactive", help="运行模式")
    parser.add_argument("--questions", help="批量模式的问题文件")
    parser.add_argument("--output", help="批量模式的输出文件")
    parser.add_argument("--init-only", action="store_true", help="仅初始化系统")
    
    args = parser.parse_args()
    
    # 初始化系统
    specialist = setup_cardiovascular_system()
    
    if not specialist:
        return 1
    
    if args.init_only:
        print("✅ 系统初始化完成，退出")
        return 0
    
    # 根据模式运行
    if args.mode == "interactive":
        interactive_mode(specialist)
    elif args.mode == "batch":
        if not args.questions or not args.output:
            print("❌ 批量模式需要指定 --questions 和 --output 参数")
            return 1
        batch_mode(specialist, args.questions, args.output)
    elif args.mode == "web":
        web_mode()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
