# WorldAI 架构文档

作者：晨星

## 1. 设计目标

| 目标 | 落地手段 |
|---|---|
| 复用业界领先开源成果 | faiss / ONNX Runtime / llama.cpp / rank-bm25 / FastAPI |
| 单一职责、可独立验证 | 10 模块，接口为 Protocol（`types.py`），实现运行时注入 |
| 干净环境一键复现 | 全 wheel 锁版依赖 + 零依赖兜底实现 + `verify.py` |
| 不允许静默降级 | 每个工厂返回 `(实现, 实际provider, error)`，错误进 `/health` |
| CPU 可跑 | 免 torch；ONNX + GGUF q4；llama.cpp 线程锁 2-4 |

## 2. 模块清单与调用关系

```
server.py (入口，只装配)
   └─ assembly.build_system(cfg)
        ├─ embed.build_embedder   → OnnxEmbedder   | HashEmbedder
        ├─ vectorstore.build_store→ FaissStore     | MemoryStore
        ├─ sparse.BM25Index        (生产/离线同一实现)
        ├─ rerank.build_reranker  → OnnxReranker   | PassthroughReranker
        └─ llm.build_llm          → LlamaCppLLM    | MockLLM
   └─ api.create_app(system)
        ├─ /chat  → RagPipeline.query
        │             ├─ retrieve: dense(store) + sparse(bm25)
        │             │            → RRF 融合 → reranker.rerank
        │             └─ llm.generate(带 [n] 编号的资料块 prompt)
        ├─ /agent → ToolAgent.run (确定性路由: 算术→calc, 其余→search)
        ├─ /ingest→ ingest.chunk_document / ingest_paths
        └─ /eval  → evals.evaluate (全新内存索引 + 黄金集)
```

数据流：`文档 → ingest(分块) → embed(向量化) → store+sparse(双索引)`
→ 查询时 `dense+sparse → RRF → rerank → top-k → prompt → llm → 答案+引用`。

## 3. 模块职责与接口

| 模块 | 职责 | 接口（Protocol） | 生产实现 | 兜底实现 |
|---|---|---|---|---|
| types | 数据结构与接口定义 | Embedder/VectorStore/SparseIndex/Reranker/LLM | — | — |
| config | 环境变量装配配置 | `load_config()` | — | — |
| ingest | 解析+分块 | 函数式 | pypdf/txt/md | 同左 |
| embed | 文本→向量 | `embed(texts)->vectors`, `dim` | OnnxEmbedder (bge-small-zh, CLS pooling) | HashEmbedder (char-bigram 哈希) |
| vectorstore | 稠密检索 | `add/search/count/reset` | FaissStore (IndexFlatIP) | MemoryStore (numpy 余弦) |
| sparse | 稀疏检索 | 同上 | BM25Index (rank-bm25 + CJK 分词) | 同左 |
| rerank | 候选重排 | `rerank(query, candidates)` | OnnxReranker (bge-reranker-base) | PassthroughReranker |
| llm | 文本生成 | `generate(prompt, max_tokens)` | LlamaCppLLM (Qwen2.5-1.5B q4_k_m) | MockLLM (抽取式模板) |
| rag | 检索编排 | `RagPipeline` | 组合以上模块 | 组合兜底 |
| agent | 工具路由 | `ToolAgent.plan/run` | search/calc 双工具 | 同左 |
| api | HTTP 服务 | `create_app(system)` | FastAPI | 同左 |
| evals | 检索评测 | `evaluate(embedder, top_k)` | 黄金集 recall@k / MRR | 同左 |
| assembly | 依赖注入装配 | `build_system(cfg)` | — | — |

## 4. 关键决策（详见 docs/decisions/）

- **ADR-0001 双实现策略**：每个外部依赖一个生产实现 + 一个零依赖兜底，
  兜底让 CI 与干净环境不依赖模型/网络/Key。
- **ADR-0002 模型供应链**：HuggingFace 不可达 → 全部模型走 ModelScope 直链；
  嵌入/重排用 ONNX Runtime 直接加载（tokenizers 读 tokenizer.json），
  绕开 fastembed/transformers 对 HF 的隐式依赖。
- **ADR-0003 CPU 推理线程**：llama.cpp 默认线程数会让小量化模型慢约 4 倍
  （瓶颈在内存带宽），锁定 2-4 线程。
- **ADR-0004 评测隔离**：评测永远新建内存索引重灌黄金集，禁止复用生产索引；
  有专门断言证明"生产索引被污染后评测指标逐位不变"。

## 5. 故障与回退语义

工厂函数签名统一为 `(instance, actual_provider, error)`：
- `error is None` → 请求的 provider 生效；
- `error 非空` → 已回退兜底实现，`/health.fallback_errors` 可见原因。

测试 `test_build_embedder_fallback_reports_error` 锁死该语义，
防止"静默回退但断言全绿"的假成功。

## 6. 性能特征（实测环境：Ryzen 7 H 255, 16GB RAM, CPU only）

| 项 | 数值 |
|---|---|
| bge-small-zh ONNX 嵌入 | ~50-100 条/秒（批量 512 维） |
| bge-reranker-base ONNX | ~8 对/秒（fp32，batch 8） |
| Qwen2.5-1.5B q4_k_m 生成 | ~15-25 tok/s（3 线程） |
| faiss IndexFlatIP | 万级 chunk 毫秒级 |
| 离线 verify.py 全程 | < 30 秒 |

## 7. 已知边界

- HashEmbedder 只近似字面重合度，离线模式下的"语义"检索质量有限——
  它存在的意义是可验证性而非质量；质量由 ONNX 嵌入承担。
- MockLLM 是抽取式摘要，不做推理；生成质量由 GGUF 承担。
- bge-reranker-base 为 fp32（1.1GB），如需更快可换量化版本（接口不变）。
- Agent 是确定性双工具路由，不是通用 ReAct；加工具只需注册进 `ToolAgent._tools`。
