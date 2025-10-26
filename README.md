# 中医智能辅助诊断助手

基于检索增强生成（RAG）技术的中医智能辅助诊断助手。

## 项目介绍

该系统利用向量数据库和大语言模型，为用户提供专业的中医药知识查询服务。系统支持中医疾病、证候、诊断等信息的智能检索与问答。

## 主要功能

- 中医知识向量检索
- 基于大语言模型的专业问答
- 批量数据处理与索引

## 安装依赖

```bash
pip install langchain chromadb langchain_community
```

## 使用方法
首先在rag.py中将“Your_key”改为你的GLM API KEY：
```bash
python rag.py
```
## 技术栈

- Python
- ChromaDB（向量数据库）
- LangChain（LLM应用框架）
- GLM-4.5（大语言模型）

## 注意事项

- 首次使用需要构建向量数据库
- 确保网络连接正常以访问大语言模型API
- 支持批量处理大量中医数据
