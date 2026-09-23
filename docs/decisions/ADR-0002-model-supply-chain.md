# ADR-0002: 模型供应链走 ModelScope 直链 + ONNX Runtime 直接加载

状态：已接受 · 作者：晨星

## 背景

目标部署环境（国内网络）实测：HuggingFace 超时不可达；fastembed /
transformers 等高层库默认从 HF 拉模型，会在运行时才失败。

## 决策

1. 模型文件一律从 ModelScope 直链下载
   （`/api/v1/models/{repo}/repo?Revision=master&FilePath={path}`），
   下载脚本 `tools/download_models.py` 幂等、可断点续传、校验大小。
2. 嵌入与重排不经过 fastembed/transformers，而是
   `tokenizers.Tokenizer.from_file()` + `onnxruntime.InferenceSession`
   直接加载，ONNX 输入名运行时探测（不同导出带/不带 token_type_ids）。
3. 选型：bge-small-zh-v1.5（中文嵌入，95MB）、bge-reranker-base（重排）、
   Qwen2.5-1.5B-Instruct q4_k_m GGUF（生成，1.1GB）。
4. `models/` 目录进 .gitignore，不进仓库。

## 后果

- 好：模型获取在目标网络下可靠；依赖面缩小（无 torch/transformers）。
- 坏：需要自维护 pooling/归一化逻辑（BGE 用 CLS pooling + L2 归一化，
  已在 embed.py 实现并有范数=1 的不变量测试）。
