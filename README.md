# lsglang_q3.8fn

Qwen3.8 Flash Next NVFP4 单卡推理管线：一张 RTX PRO 5000 72GB + 双路 EPYC，262K 上下文，MTP 投机解码，支持图片。
框架是 [guqiong96/Lsglang](https://github.com/guqiong96/Lsglang)（sglang 的 CPU/GPU 混合推理分支），CPU 层的路由专家由受
[lk_moe](https://github.com/guqiong96/Lsglang)（lsglang 作者 guqiong96 的 CPU MoE 后端）启发、并针对 NVFP4 格式专项加速的
**lkqmoe** 承担计算（本仓库附编译好的闭源版本）。

*English: a single-GPU Qwen3.8 Flash Next NVFP4 pipeline (RTX PRO 5000 72GB + 2x EPYC 9334, 262K context, MTP speculative
decoding, image input) on guqiong96/Lsglang, with the CPU-side routed experts on lkqmoe, an NVFP4-specialised MoE kernel inspired by lk_moe
(by guqiong96, the author of lsglang); a compiled, closed-source build is included here. Speeds at 8 context lengths from 4K to 253K are below.*

## 组成

| 部分 | 来源 | 本仓库内容 |
|---|---|---|
| 模型 | [lovedheart/Qwen3.8-Flash-Next-NVFP4-W4A16-ATTN-FP8-MTP-NVFP4](https://huggingface.co/lovedheart/Qwen3.8-Flash-Next-NVFP4-W4A16-ATTN-FP8-MTP-NVFP4)：lovedheart 对 Qwen 团队 **Qwen3.8 Flash Next** 的量化版（专家 NVFP4 W4A16，注意力 FP8，MTP 草稿专家 NVFP4），121 GB | — |
| 推理框架 | guqiong96/Lsglang v1.4.12 @ [`c49d8f3703`](https://github.com/guqiong96/Lsglang/commit/c49d8f3703b1b829d0eedbd53f2a4c2443fa8800)；Qwen3.8 支持来自 sglang PR #36497（Qiaolin-Yu, "Introduce Qwen 3.8 Flash Next"），另有本地调试修改与验证过的 Split-K | `patches/lsglang-c49d8f37-qwen38fn.patch`（81 个文件） |
| CPU 专家内核 | **lkqmoe** 0.3.2（闭源，编译版）：受 lk_moe 启发、针对 NVFP4 格式专项加速的 CPU/GPU 混合 MoE 内核。lk_moe 来自 lsglang 作者 guqiong96（[guqiong96/Lsglang](https://github.com/guqiong96/Lsglang)）；lkqmoe 独立实现其 `MOE_NVFP4` 接口，不含 lk_moe 代码 | `lkqmoe/` |
| MTP 草稿热词表 | 从实际输出统计的 49152 个高频 token（只影响草稿接受率，目标模型仍验证每个 token） | `lkqmoe/draft-token-map-49152.pt`，`bench/build_token_map.py` |
| 启动 | 本文测速所用的完整参数 | `launch/run-qwen38fn-nvfp4.sh` |

## 本机硬件

| | |
|---|---|
| CPU | 2× AMD EPYC 9334（32 核/颗，共 64 核 128 线程），NPS2 → 4 个 NUMA 节点，256 MB L3，AVX512-BF16/VBMI |
| 内存 | 320 GB DDR5（20× 16 GB，4800 MT/s），实测带宽约 511 GB/s |
| GPU | NVIDIA RTX PRO 5000 72GB Blackwell（SM120），驱动 595.84，CUDA 13.2 |
| 存储 | NVMe |
| 系统 | Ubuntu 24.04.4，内核 7.0.0-31，Python 3.12.3 |

## 速度

测试条件：单请求，温度 0，`ignore_eos`，每点输出 1024 token（含思考内容）；输入是长日志文本，在 10% 深度埋一条事实并在末尾提问（`bench/speed_points.py`）。MTP 开启，32 层 GPU 常驻，其余见“运行方式”。2026-09-30 用本仓库的 lkqmoe 包实测，原始数据 `results/speed-points-20260930.json`。

| 输入 token | 首 token 时间 | 预填充 tok/s | decode tok/s | 平均接受长度 | 找回埋藏事实 | 显存峰值 |
|---:|---:|---:|---:|---:|:---:|---:|
| 4,109 | 1.1 s | 3,676 | 200.3 | 3.42 | 是 | 66,940 MiB |
| 8,315 | 1.4 s | 5,749 | 180.4 | 3.25 | 是 | 67,920 MiB |
| 17,123 | 2.8 s | 6,179 | 170.1 | 3.12 | 是 | 69,882 MiB |
| 33,265 | 5.6 s | 5,923 | 162.8 | 3.03 | 是 | 71,446 MiB |
| 68,103 | 12.6 s | 5,410 | 169.3 | 3.01 | 是 | 71,932 MiB |
| 134,941 | 27.4 s | 4,923 | 147.7 | 2.95 | 是 | 72,274 MiB |
| 210,634 | 36.1 s | 5,837 | 154.5 | 2.90 | 是 | 72,396 MiB |
| 252,609 | 44.4 s | 5,693 | 147.6 | 2.86 | 是 | 72,396 MiB |

最后一档（253K 输入 + 1K 输出）接近 262,144 的上下文上限。接受长度按服务端累计平均统计。

## 运行方式

- 模型 48 层（36 层线性注意力 GDN + 12 层全注意力 QSA），MoE 512 专家 top-10（hidden 2560，专家中间维 640）。
- 前 32 个 MoE 层常驻 GPU，其余 16 层的路由专家在 CPU（lkqmoe，64 线程）；MTP 草稿全在 GPU。
- KV 缓存 NVFP4，草稿 KV FP8；PLE（逐层嵌入）卸载到内存；32K 分块预填充；单请求并发。
- MTP（NEXTN）：3 步、每步验证 4 个 token，草稿热词表 49152。

## 环境搭建

1. 框架：
   ```sh
   git clone https://github.com/guqiong96/Lsglang /opt/Lsglang && cd /opt/Lsglang
   git checkout c49d8f3703b1b829d0eedbd53f2a4c2443fa8800
   git apply /path/to/lsglang_q3.8fn/patches/lsglang-c49d8f37-qwen38fn.patch
   ```
2. Python 3.12 虚拟环境（`/opt/Lsglang/env`），按 Lsglang v1.4.12 的 wheel 发布包安装，版本见 `requirements-lock.txt`
   （torch 2.13.0、triton 3.7.1、flashinfer 0.6.17、transformers 5.12.1）。
   **`lk_moe` 2.4.3 需要另外安装**（Lsglang 作者发布的闭源库，本仓库不再分发；MoE 计算由 lkqmoe 负责）。
3. 下载模型，按需修改 `launch/run-qwen38fn-nvfp4.sh` 开头的路径，启动：
   ```sh
   bash launch/run-qwen38fn-nvfp4.sh
   ```
   服务在 `127.0.0.1:18080`，OpenAI 兼容接口，模型名 `Qwen3.8-Flash-Next`，思考输出由 `qwen3` 解析器分离。

启动前需要约 230 GB 可用内存、70 GB 空闲显存。

## 主要启动参数

```
--context-length 262144 --max-total-tokens 262144 --max-running-requests 1 --chunked-prefill-size 32768
--mem-fraction-static 0.90 --page-size 64 --quantization modelopt_mixed --moe-runner-backend flashinfer_cutlass
--kv-cache-dtype nvfp4 --speculative-draft-kv-cache-dtype fp8_e4m3 --ple-offload-embedding
--prefill-attention-backend triton --decode-attention-backend trtllm_mha
--linear-attn-prefill-backend triton --linear-attn-decode-backend flashinfer
--mamba-ssm-dtype bfloat16 --mamba-scheduler-strategy extra_buffer --max-mamba-cache-size 16
--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
--speculative-token-map lkqmoe/draft-token-map-49152.pt --speculative-draft-model-quantization modelopt_mixed
--cuda-graph-backend-decode full --cuda-graph-bs-decode 1 --reasoning-parser qwen3 --tool-call-parser qwen3_coder
--json-model-override-args '{"text_config":{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":262144}}}'
LVLLM_GPU_RESIDENT_MOE_LAYERS=0-31  LK_THREADS=64  LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=4096
LKQMOE_MODE=standalone  LKQMOE_ZERO_COPY=1  LKQMOE_MTP_QUANT=1  LKQMOE_MTP_ROUTER_GUARD=1  LKQMOE_PREFILL_PREFETCH=1
```
完整列表见 `launch/run-qwen38fn-nvfp4.sh`。

## lkqmoe（`lkqmoe/`）

- `liblkqmoe.so`（CPU 内核）、`liblkqmoe_cuda.so`（CUDA 信箱桥，可被 CUDA Graph 捕获）、`python/lkqmoe/*.pyc`（lsglang 适配层与
  Triton GPU 预填充，编译版；含 Triton 内核的模块把源码压缩内嵌，因为 Triton 编译时需要读取源码）。
- 开源部分（Apache-2.0）：`python/sitecustomize.py`、`python/lkqmoe/gpu/mtp_quant.py`（让 MTP 草稿保留 NVFP4 量化，省约 2.9 GB 显存）、
  `python/lkqmoe/gpu/router_dependency.py`（MTP 路由的生产者/消费者依赖保护）。
- 二进制许可见 `lkqmoe/LICENSE`：可免费使用、原样再分发；源码不公开。
- 与闭源 `lk_moe` 在 118 个真实权重用例上逐位一致（相同的 FP32 数值路径），MTP 验证步较 lk_moe 基线快约 20%，decode 约 +24%。
- 硬件要求：x86-64 AVX512-BF16，4 个 NUMA 节点 × 16 物理核（其他拓扑未验证），CUDA GPU。

## 测试脚本（`bench/`）

- `speed_points.py`：任意长度打点测速（本页速度表）。
- `mtp_speed_bench.py`：4 类任务 2K 输入 / 512 输出的 MTP 速度，加一次 32K 冷启动首 token 时间。
- `build_token_map.py`：从实际输出生成草稿热词表。

## 许可

本仓库开源部分按 Apache-2.0（与 Lsglang/sglang 相同）；`lkqmoe/` 下的二进制见其 LICENSE。模型权重遵循各自许可。
