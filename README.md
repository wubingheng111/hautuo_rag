# 💓 心智医 - 心血管疾病智能问答系统

基于华佗数据集的心血管专科RAG系统，专门针对心血管疾病提供智能问答服务。

## 🎯 系统特色

### 📊 数据基础
- **数据来源**: 华佗医学知识图谱问答数据集
- **专项数据**: 131,188 条心血管相关问答对
- **覆盖领域**: 高血压、冠心病、心律失常、心力衰竭等
- **数据质量**: 经过专业筛选和关键词匹配

### 🫀 专科功能
- **心血管疾病咨询**: 基于专业医学知识的问答
- **急症识别**: 智能识别心血管急症并提供紧急建议
- **风险评估**: 分析心血管风险因素
- **RAG检索**: 实时检索相关医学知识
- **专业回答**: 结构化的专业医学回答

## 🚀 快速开始

### 1. 环境准备
```bash
# 克隆项目
git clone https://github.com/wubingheng111/hautuo_rag.git
cd hautuo_rag

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备
```bash
# 抽取心血管专项数据
python analyze_huatuo_dataset.py cardiovascular
```

### 3. 启动系统

#### 命令行模式
```bash
# 交互式问答
python cardio_main.py --mode interactive

# 批量处理
python cardio_main.py --mode batch --questions questions.txt --output results.json

# 仅初始化系统
python cardio_main.py --init-only
```

#### Web界面模式
```bash
# 启动Streamlit应用
python cardio_main.py --mode web
# 或直接运行
streamlit run ui/cardio_app.py
```

## 📁 项目结构

```
hautuo_rag/
├── src/                          # 核心代码
│   ├── cardio_specialist.py      # 心血管专科模块
│   ├── rag_system.py            # RAG系统
│   ├── data_processor.py        # 数据处理
│   └── ...
├── ui/                          # 用户界面
│   └── cardio_app.py           # 心血管专科Web应用
├── cardio_main.py              # 主程序入口
├── analyze_huatuo_dataset.py   # 数据分析和抽取工具
├── cardiovascular_qa_data.json # 心血管专项数据
└── requirements.txt            # 依赖包
```

## 💡 使用示例

### 命令行交互
```bash
$ python cardio_main.py --mode interactive

🫀 心血管专科问答系统
==================================================
📥 正在加载心血管专项数据...
✅ 成功加载 131,188 条心血管问答数据
🔧 正在构建心血管知识向量数据库...
✅ 心血管专科系统初始化完成!

💬 进入心血管专科问答模式
输入 'quit' 或 'exit' 退出
--------------------------------------------------

🩺 请输入您的心血管相关问题: 高血压应该注意什么？

🤔 正在分析您的问题...

🫀 心血管专科回答:
------------------------------
## 🔍 问题分析
这是一个典型的心血管相关问题。

## 💊 专业解答
高血压患者需要注意以下几个方面：
1. 定期监测血压
2. 坚持服用降压药物
3. 控制饮食，减少盐分摄入
...
```

### Web界面功能
- **💬 智能对话**: 自然语言问答交互
- **🔍 RAG检索**: 实时显示知识检索过程
- **📊 专科分析**: 心血管相关性和风险评估
- **🚨 急症识别**: 自动识别紧急情况
- **📎 引用管理**: 收集和管理参考文献

## 🔧 系统配置

### 模型配置
在 `src/config.py` 中配置：
```python
# LLM配置
LLM_MODEL = "deepseek-chat"
LLM_API_KEY = "your-api-key"

# 嵌入模型配置
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# 向量数据库配置
VECTOR_DB_TYPE = "chroma"
```

### 心血管专科配置
系统自动加载心血管关键词库和风险因素，包括：
- **疾病关键词**: 冠心病、高血压、心律失常等
- **症状关键词**: 胸痛、心悸、气短等
- **检查关键词**: 心电图、心脏彩超等
- **治疗关键词**: 支架、搭桥、药物治疗等

## 📊 数据统计

### 心血管专项数据
- **总数据量**: 131,188 条问答对
- **数据占比**: 16.47% (从796,444条总数据中筛选)
- **主要子领域**:
  - 高血压: 30.4%
  - 冠心病: 24.6%
  - 心律失常: 12.4%
  - 心脏检查: 10.7%
  - 心力衰竭: 10.2%

### 高频关键词
1. 动脉: 35,379次
2. 血管: 35,111次
3. 静脉: 20,994次
4. 血压: 15,009次
5. 高血压: 11,251次

## 🚨 重要声明

⚠️ **医疗免责声明**
- 本系统仅供医学信息参考，不能替代专业医生的诊断和治疗
- 急症情况请立即就医或拨打120急救电话
- 任何医疗决策都应咨询专业医生

+

💓 **心智医** - 让AI成为您的心血管健康守护者
