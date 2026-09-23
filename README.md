# WorldAI

模块化、离线可验证、CPU 可跑的端到端 RAG + Agent 系统。
整合业界领先开源成果（faiss / ONNX Runtime / llama.cpp / BM25），不重造轮子。

作者：晨星 · License: MIT

## 特性

- **混合检索**：稠密向量（faiss）+ 稀疏词项（BM25），RRF 融合
- **交叉编码器重排**：bge-reranker-base（ONNX，无 torch 依赖）
- **本地生成**：Qwen2.5-1.5B-Instruct GGUF（llama.cpp，CPU 线程数已调优）
- **中文原生**：CJK 分词、中文嵌入（bge-small-zh）、中文黄金评测集
- **离线可验证**：每个外部依赖都有零依赖兜底实现，`verify.py` 无网无 Key 全绿
- **单一职责**：10 个模块，接口为 Protocol，实现运行时注入，可独立单测

## 系统架构

```
                 ┌─────────────┐
   HTTP  ──────▶ │  api (FastAPI) │
                 └──┬───────┬───┘
              ┌─────┘       └──────┐
        ┌─────▼─────┐       ┌──────▼─────┐
        │ rag 编排   │◀──────│ agent 工具路由│
        └──┬───┬───┘       └────────────┘
   ┌───────┘   └──────────────┐
   ▼            ▼              ▼
embed      vectorstore      sparse
(ONNX/哈希) (faiss/内存)    (BM25+CJK)
   ▲            ▲              ▲
   └──────── ingest ───────────┘
        (pdf/txt/md → 分块)
        rerank (ONNX/直通) ── 在融合之后
        llm   (GGUF/Mock)  ── 在检索之后
```

模块接口定义见 `src/worldai/types.py`；装配关系见 `src/worldai/assembly.py`；
详细架构决策见 `ARCHITECTURE.md` 与 `docs/decisions/`。

## 快速开始（干净环境一键复现）

```bash
# 1. 安装核心依赖（全部有预编译 wheel，无需编译器）
pip install -r requirements.lock.txt -r requirements-dev.txt

# 2. 离线验证（不下载任何模型，使用兜底实现）
python verify.py
# 期望输出：SUMMARY: unit=OK e2e_offline=OK e2e_online=SKIP => ALL GREEN
```

## 生产模式（真实模型）

```bash
# 3. 下载模型（ModelScope 直链，约 2.3GB，可断点续传）
python tools/download_models.py

# 4. （可选）本地 GGUF 生成：见 requirements-llm.txt 内说明
pip install -r requirements-llm.txt

# 5. 再次验证 —— 自动检测到模型后追加在线链路 E2E
python verify.py
# 期望输出：... e2e_online=OK => ALL GREEN
```

## 运行服务

```bash
# 离线兜底模式（默认，无需模型）
python server.py

# 生产模式（需先下载模型）
WORLDAI_EMBED=onnx WORLDAI_STORE=faiss WORLDAI_RERANK=onnx WORLDAI_LLM=gguf \
  python server.py
# Windows PowerShell:
# $env:WORLDAI_EMBED="onnx"; $env:WORLDAI_STORE="faiss"; `
# $env:WORLDAI_RERANK="onnx"; $env:WORLDAI_LLM="gguf"; python server.py
```

服务监听 `127.0.0.1:8000`（`WORLDAI_PORT` 可改）。

## 使用指南

```bash
# 健康检查（含实际生效的 provider 与回退原因）
curl http://127.0.0.1:8000/health

# 摄入文档（ texts 直接传文本；paths 传服务器侧文件路径，支持 pdf/txt/md ）
curl -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"texts": ["光合作用是指绿色植物利用光能把二氧化碳和水转化成有机物并释放氧气的过程。"]}'

# RAG 问答（带引用）
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "光合作用释放什么气体？"}'

# Agent（自动路由 search / calc 工具）
curl -X POST http://127.0.0.1:8000/agent \
  -H "Content-Type: application/json" \
  -d '{"question": "12*12+6"}'

# 检索质量评测（独立黄金集，全新索引，不受生产数据污染）
curl http://127.0.0.1:8000/eval
```

## 配置项（环境变量）

| 变量 | 取值 | 默认 | 说明 |
|---|---|---|---|
| `WORLDAI_EMBED` | `onnx`/`hash` | `hash` | 嵌入实现 |
| `WORLDAI_STORE` | `faiss`/`memory` | `memory` | 向量库实现 |
| `WORLDAI_RERANK` | `onnx`/`rrf` | `rrf` | 重排实现 |
| `WORLDAI_LLM` | `gguf`/`mock` | `mock` | 生成实现 |
| `WORLDAI_LLM_THREADS` | 整数 | `3` | llama.cpp 线程（CPU 上 2-4 最优，瓶颈在内存带宽） |
| `WORLDAI_CHUNK_SIZE` / `WORLDAI_CHUNK_OVERLAP` | 整数 | `400`/`60` | 分块参数 |
| `WORLDAI_RETRIEVE_K` / `WORLDAI_FINAL_K` | 整数 | `8`/`4` | 召回/最终条数 |

任何生产实现加载失败都会显式回退到底兜实现，并在 `/health` 的
`fallback_errors` 字段中报告原因——**不允许静默降级**。

## 目录结构

```
worldai/
├── server.py               # 入口（只装配，零业务逻辑）
├── verify.py               # 一键验收：单测 + 在线/离线双路 E2E + 评测
├── requirements.lock.txt   # 核心锁定依赖（全 wheel，免编译）
├── requirements-dev.txt    # 测试依赖
├── requirements-llm.txt    # 可选：本地 GGUF 生成
├── src/worldai/            # 10 个模块（types/config/ingest/embed/vectorstore/
│                           #   sparse/rerank/llm/rag/agent/api/evals/assembly）
├── tests/                  # 31 个单元/集成测试
├── tools/download_models.py# ModelScope 模型下载（断点续传）
├── docs/SPEC.md            # 接口与行为规范
├── docs/decisions/         # 架构决策记录（ADR）
└── models/                 # 模型文件（git 忽略，脚本下载）
```

## 验证哲学

- 单测在**进程内**运行，按收集到的用例计数判定，不依赖退出码或文本匹配
- E2E 用真实子进程起服务、轮询健康检查、逐条断言成功流与错误流
- 评测永远在**全新内存索引**上跑黄金集，灌入干扰文档后指标逐位不变（有断言）
- 离线兜底不是玩具：它让"干净环境一键复现"不依赖任何模型与网络
