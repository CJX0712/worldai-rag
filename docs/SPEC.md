# WorldAI 接口与行为规范（SPEC）

作者：晨星

## 1. HTTP API

### GET /health
响应：
```json
{
  "status": "ok",
  "providers": {"embed": "onnx|hash", "store": "faiss|memory",
                "rerank": "onnx|rrf", "llm": "gguf|mock"},
  "fallback_errors": {"embed": "...", "llm": "..."},
  "chunks": 42
}
```
- `fallback_errors` 只包含发生了回退的模块；为空对象表示全部按请求生效。

### POST /ingest
请求：`{"texts": ["..."], "paths": ["..."], "chunk_size": 400, "overlap": 60}`
- `texts` 与 `paths` 至少一个非空，否则 `400`。
- `paths` 支持 `.txt/.md/.pdf`，不支持或读取失败 `400`。
响应：`{"ingested_chunks": 3, "total_chunks": 10}`

### POST /chat
请求：`{"question": "...", "max_tokens": 512}`
- 空白问题 `400`。
响应：
```json
{
  "answer": "...",
  "provider": "gguf|mock",
  "citations": [{"chunk_id": "...", "doc_id": "...",
                  "text": "前200字符", "score": 0.83}]
}
```
- 知识库为空时返回 200，answer 为提示语，citations 为空。

### POST /agent
请求/响应同 /chat。行为：
- 问题匹配纯算术表达式 → `calc` 工具，`provider="calc"`，answer 含数值结果。
- 其余 → `search` 工具（等价于 /chat）。
- 算术表达式支持 `+ - * / % ** 括号`（AST 白名单，拒绝任意代码执行）。

### GET /eval?top_k=4
响应：`{"recall@4": 1.0, "mrr": 1.0, "queries": 5.0}`
- 在全新内存索引上重灌内置黄金集（5 篇中文文档、5 条问答），
  不读不写生产索引；连续两次调用结果必须逐位相等。

## 2. Python 接口（src/worldai/types.py）

```python
class Embedder(Protocol):
    @property
    def dim(self) -> int: ...
    def embed(self, texts: Sequence[str]) -> List[List[float]]: ...
    # 约定：返回向量已 L2 归一化；空输入返回零向量

class VectorStore(Protocol):
    def add(self, chunks, vectors) -> None: ...
    def search(self, vector, k) -> List[ScoredChunk]: ...  # 按分数降序
    def count(self) -> int: ...
    def reset(self) -> None: ...

class SparseIndex(Protocol):   # 同 VectorStore，query 为原始文本
class Reranker(Protocol):
    def rerank(self, query, candidates) -> List[ScoredChunk]: ...  # 重排后降序
class LLM(Protocol):
    def generate(self, prompt: str, max_tokens: int = 512) -> str: ...
```

工厂约定：`build_*(...) -> (instance, actual_provider, error)`。
`error is not None` 表示发生回退，调用方必须暴露该错误。

## 3. 不变量（测试锁死的行为）

| 不变量 | 测试 |
|---|---|
| 嵌入向量 L2 范数 = 1 | test_hash_embedder_output_shape_and_norm |
| 相同输入嵌入逐位相同 | test_hash_embedder_deterministic |
| 分块 id 唯一、长度 ≤ chunk_size | test_chunk_document_respects_size_and_overlap |
| 向量维度不匹配抛 ValueError | test_store_dim_mismatch_raises |
| RRF 融合去重 | test_rrf_fuse_dedupes_and_orders |
| 空知识库问答有提示不崩溃 | test_rag_empty_kb |
| calc 拒绝任意代码执行 | test_safe_calc |
| 回退必须报告 error | test_build_embedder_fallback_reports_error |
| 评测不受生产索引污染且逐位确定 | test_eval_fresh_index_deterministic |
| 摄入文本超过分块阈值时必须产出多块 | e2e: ingest_multi_chunk |
