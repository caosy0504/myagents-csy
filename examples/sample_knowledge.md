# MyAgents CSY

MyAgents CSY 是一个从基础接口开始实现的轻量 Agent 框架。

## 核心接口

### Message

Message 规定消息的角色、正文、时间和元数据，并可以转换为模型 API 所需的字典格式。

### Config

Config 集中管理温度、历史长度、最大步骤数和反思次数等公共参数。

### Agent

所有 Agent 都继承统一的 Agent 抽象基类，并通过 `run(input_text)` 执行任务。统一接口使未来的 Router 和 Workflow Engine 能够调度不同 Agent。

## Agent 模式

- SimpleAgent 用于基础问答。
- ReActAgent 可以调用计算器、搜索和 RAG 等工具。
- ReflectionAgent 通过生成、反思和改进提高内容质量。
- PlanAndSolveAgent 先生成计划，再逐步执行复杂任务。
- DocumentQAAgent 从 Qdrant 知识库检索文档片段，再让模型基于证据回答。

## RAG 流程

文档首先被转换为 Markdown，然后按照标题和段落进行结构感知分块。每个分块通过本地中文 BGE Embedding 模型转换为 512 维向量并写入 Qdrant。用户提问时，系统使用查询专用指令将问题向量化，检索最相关分块，最后将参考资料交给本地 Ollama 模型生成带来源的答案。
