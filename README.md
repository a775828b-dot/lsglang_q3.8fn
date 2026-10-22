# lsglang_q3.8fn

Qwen3.8 Flash Next NVFP4 单卡推理管线：一张 RTX PRO 5000 72GB + 双路 EPYC，512K 上下文、2 并发，MTP 投机解码，
**LKQstrata 热专家池**（CPU 层的热门专家常驻 GPU、自适应换入换出，与 CPU 同时计算），支持图片。
框架是 [guqiong96/Lsglang](https://github.com/guqiong96/Lsglang)（sglang 的 CPU/GPU 混合推理分支），CPU 层的路由专家由受
[lk_moe](https://github.com/guqiong96/Lsglang)（lsglang 作者 guqiong96 的 CPU MoE 后端）启发、并针对 NVFP4 格式专项加速的
**lkqmoe** 承担计算（本仓库附编译好的闭源版本 0.5.6）。

*English: a single-GPU Qwen3.8 Flash Next NVFP4 pipeline (RTX PRO 5000 72GB + 2x EPYC 9334, 512K context, 2 concurrent requests,
MTP speculative decoding, LKQstrata GPU expert pool for the hot experts of the CPU MoE layers, image input) on guqiong96/Lsglang,
with the CPU-side routed experts on lkqmoe 0.5.6, an NVFP4-specialised MoE kernel inspired by lk_moe (by guqiong96, the author of
lsglang); a compiled, closed-source build is included here. With 24 memory channels and NPS4 use
`launch/run-qwen38fn-nvfp4-24ch-nps4.sh` (112 lkqmoe threads, host CPUs resolved by lkqmoe); release notes are under "版本更新",
details in `reports/`.*

## 版本更新

- **2026-10-22**：lkqmoe 0.5.6（与 lsglang_ds4.1 同一份编译版）；内存补满 24 通道、BIOS 改 NPS4，新增 `launch/run-qwen38fn-nvfp4-24ch-nps4.sh`
  （112 线程、`LKQMOE_MAIN_CPUS=auto`）。decode 受 GPU 限制，按每步耗时与上一版持平。详见 [reports/2026-10-22-24ch-nps4-lkqmoe056.md](reports/2026-10-22-24ch-nps4-lkqmoe056.md)。
- **2026-10-17**：lkqmoe 0.5.3：只有 AVX2 的 CPU 与多卡张量并行。详见 [reports/2026-10-17-lkqmoe053-avx2-tp.md](reports/2026-10-17-lkqmoe053-avx2-tp.md)。
- **2026-10-16**：lkqmoe 0.5.2：NUMA 拓扑从系统读取（NPS2 / NPS4），`LKQMOE_MAIN_CPUS=auto`。详见 [reports/2026-10-16-lkqmoe052-numa.md](reports/2026-10-16-lkqmoe052-numa.md)。
- **2026-10-14**：LKQstrata 热专家池：上下文 256K → 512K，并发 1 → 2，单请求 decode +10%、2 并发合计 +89%。
  详见 [reports/2026-10-14-lkqstrata.md](reports/2026-10-14-lkqstrata.md)。
- **2026-10-01**：GPU 预填充按块 FP32 累加路由，GPU 常驻 34 层（原 32 层），预填充和 decode 都更快。数据 `results/speed-points-20261001-l34-accum.json`。
- **2026-09-30**：首次发布；lkqmoe 0.4.1（与 0.3.2 在 118 个用例上逐位相同，验证步 16.69 → 16.35 ms）；不需要闭源 lk_moe。
  数据 `results/speed-points-20260930.json`、`results/lkqmoe032-vs-041/`。

## 组成

| 部分 | 来源 | 本仓库内容 |
|---|---|---|
| 模型 | [lovedheart/Qwen3.8-Flash-Next-NVFP4-W4A16-ATTN-FP8-MTP-NVFP4](https://huggingface.co/lovedheart/Qwen3.8-Flash-Next-NVFP4-W4A16-ATTN-FP8-MTP-NVFP4)：lovedheart 对 Qwen 团队 **Qwen3.8 Flash Next** 的量化版（专家 NVFP4 W4A16，注意力 FP8，MTP 草稿专家 NVFP4），121 GB | — |
| 推理框架 | guqiong96/Lsglang v1.4.12 @ [`c49d8f3703`](https://github.com/guqiong96/Lsglang/commit/c49d8f3703b1b829d0eedbd53f2a4c2443fa8800)；Qwen3.8 支持来自 sglang PR #36497（Qiaolin-Yu, "Introduce Qwen 3.8 Flash Next"），另有本地调试修改与验证过的 Split-K | `patches/lsglang-c49d8f37-qwen38fn.patch` |
| CPU 专家内核 | **lkqmoe**（闭源，编译版）：受 lk_moe 启发、针对 NVFP4 格式专项加速的 CPU/GPU 混合 MoE 内核。lk_moe 来自 lsglang 作者 guqiong96（[guqiong96/Lsglang](https://github.com/guqiong96/Lsglang)）；lkqmoe 独立实现其 `MOE_NVFP4` 接口，不含 lk_moe 代码。0.5.0 起含 **LKQstrata 热专家池**：GPU 上缓存每层的热门专家，换入换出策略参考 [Strata](https://github.com/Niko1221/Strata)（MIT，独立实现） | `lkqmoe/` |
| 热专家池排名表 | 从 Qwen 实际 decode 路由统计的每层专家使用排名（52 个请求：中英文、代码、数学、agent 排错等短提示，约 11 万 token 的长文问答，5 轮编程对话；温度 1.0）；启动时按它装填，之后自适应 | `lkqmoe/lkqstrata-qwen38.json` |
| MTP 草稿热词表 | 从实际输出统计的 49152 个高频 token（只影响草稿接受率，目标模型仍验证每个 token） | `lkqmoe/draft-token-map-49152.pt` |
| 启动 | 测速所用的完整参数 | `launch/run-qwen38fn-nvfp4-24ch-nps4.sh`（24 通道 / NPS4）；`launch/run-qwen38fn-nvfp4.sh`（20 通道 / NPS2） |

## 本机硬件

| | |
|---|---|
| CPU | 2× AMD EPYC 9334（共 64 核 128 线程），AVX512-BF16/VBMI |
| 内存 / NUMA | 24 × 16 GiB DDR5-4800（384 GiB），BIOS NPS4 → 8 个 NUMA 节点（2026-10-22 起；之前为 20 × 16 GB、320 GB、NPS2 4 节点） |
| GPU | NVIDIA RTX PRO 5000 72GB Blackwell（SM120），驱动 595.99，CUDA 13.2；功耗墙 350 W |
| 系统 | Ubuntu 24.04.4，内核 7.0.0-31，Python 3.12.3 |

## 速度

单请求：温度 0，每点输出 1024 token，输入是埋了一条事实的长文本（`bench/speed_points.py`），所有长度都找回事实。
并发：短提示，温度 1.0，每个请求 512 token，同时开始，各 4 轮（`bench/conc_bench.py`）。两个请求共用 524,288 token 的 KV 池。

### 24 通道 / NPS4（2026-10-22，lkqmoe 0.5.6，`launch/run-qwen38fn-nvfp4-24ch-nps4.sh`）

| 输入 token | decode tok/s | 接受长度 | 首 token |
|---:|---:|---:|---:|
| 4K | 220.3 | 3.14 | 1.07 s |
| 8K | 234.6 | 3.23 | 1.49 s |
| 17K | 192.1 | 3.06 | 2.72 s |
| 33K | 192.4 | 2.98 | 5.66 s |
| 68K | 191.5 | 2.92 | 12.58 s |
| 135K | 177.3 | 2.86 | 31.87 s |
| 211K | 180.5 | 2.83 | 35.33 s |
| 253K | 210.0 | 2.86 | 43.57 s |
| 415K | 169.8 | 2.84 | 81.49 s |
| 500K | 165.1 | 2.55 | 106.24 s |

- 4K–253K 平均 decode 199.8 tok/s（10 个点平均 193.4）；500K 一点用 `--exact` 在 WebUI 正式启动后测得，其余各点同一配置。
- 并发：1 个请求 145.9 tok/s；2 个请求每个 109.5、合计 **202.2 tok/s**（温度 1.0 下接受长度 2.07 / 1.9）。
- 困惑度 3.4779（6 × 4096 token）。
- 原始数据：`results/speed-points-20261022-24ch-nps4.json`、`results/concurrency-20261022-24ch-nps4.json`。

### 20 通道 / NPS2（2026-10-14，lkqmoe 0.5.0，`launch/run-qwen38fn-nvfp4.sh`）

| 输入 token | decode tok/s | 首 token |
|---:|---:|---:|
| 4K | 241.6 | 1.19 s |
| 8K | 219.0 | 1.47 s |
| 17K | 209.2 | 2.67 s |
| 33K | 190.2 | 5.74 s |
| 68K | 181.7 | 12.46 s |
| 135K | 165.8 | 28.87 s |
| 211K | 166.5 | 34.82 s |
| 253K | 207.6 | 43.80 s |
| 415K | 189.5 | 82.17 s |
| 500K | 167.5 | 106.47 s |

- 4K–253K 平均 decode 197.7 tok/s；并发：1 个请求 154.1 tok/s，2 个请求合计 217.0 tok/s（接受长度 2.20 / 2.02）。
- 原始数据：`results/speed-points-20261014-lkqstrata.json`、`results/concurrency-20261014.json`。

## 运行方式

- 模型 48 层（36 层线性注意力 GDN + 12 层全注意力 QSA），MoE 512 专家 top-10（hidden 2560，专家中间维 640）。
- 第 0 层和第 1、2、5–8、10–14、17–22、26、28、29、33、34、37、38 层（共 25 层）整层常驻 GPU（Marlin W4A16）；其余 23 层的路由
  专家由 lkqmoe 承担：CPU（112 线程，NPS4 下每个 NUMA 节点 14 个；20 通道 NPS2 脚本为 64 线程）加上 GPU 上的 LKQstrata 热专家池（每层 155 个槽位）；MTP 草稿全在 GPU。
- LKQstrata：每层 decode 时，一个打包内核读本层的驻留表（显存，CUDA graph 重放时读取，驻留变化不用重新抓图），把路由分成两半：
  在槽位里的专家交给 GPU（W4A16），其余写给 CPU；GPU 在通知 CPU 之后、等待 CPU 之前计算，最后把两部分相加。CPU 侧的信箱线程
  统计每层路由（带衰减的计数），策略线程每 4 个验证窗口把最常用的未驻留专家换进槽位、换出最少用的（收益门槛 1.5，每窗口最多
  32 个）；换入换出经映射内存里的命令环和每窗口开头的提交内核生效：换出立即生效，换入在拷贝落地后生效，正在写的槽位不会被读取。
- 预填充：长输入走 lkqmoe 的 Triton GPU 内核，按 32 个专家一块上传权重，路由结果按块 FP32 累加（工作区约 1.1 GiB）；32K 分块预填充。
- KV 缓存 NVFP4，草稿 KV FP8；PLE（逐层嵌入）卸载到内存；最多 2 个请求同时运行，KV 池 512K 共享。
- MTP（NEXTN）：3 步、每步验证 4 个 token，草稿热词表 49152。

## 环境搭建

1. 框架：
   ```sh
   git clone https://github.com/guqiong96/Lsglang /opt/Lsglang && cd /opt/Lsglang
   git checkout c49d8f3703b1b829d0eedbd53f2a4c2443fa8800
   git apply /path/to/lsglang_q3.8fn/patches/lsglang-c49d8f37-qwen38fn.patch
   ```
2. Python 3.12 虚拟环境（`/opt/Lsglang/env`），按 Lsglang v1.4.12 的 wheel 发布包安装，版本见 `requirements-lock.txt`
   （torch 2.13.0、triton 3.7.1、flashinfer 0.6.17、transformers 5.12.1）。不需要安装 `lk_moe`：lkqmoe 以 `LKQMOE_MODE=standalone`
   直接提供 `lk_moe` 模块。
3. 下载模型，按需修改启动脚本开头的路径后启动：24 通道 / NPS4 用 `bash launch/run-qwen38fn-nvfp4-24ch-nps4.sh`，20 通道 / NPS2 用 `bash launch/run-qwen38fn-nvfp4.sh`。
   服务在 `127.0.0.1:18080`，OpenAI 兼容接口，模型名 `Qwen3.8-Flash-Next`，思考输出由 `qwen3` 解析器分离。

启动前需要约 230 GB 可用内存、70 GB 空闲显存。

## 主要启动参数

```
--context-length 524288 --max-total-tokens 524288 --max-running-requests 2 --chunked-prefill-size 32768
--mem-fraction-static 0.90 --page-size 64 --quantization modelopt_mixed --moe-runner-backend flashinfer_cutlass
--kv-cache-dtype nvfp4 --speculative-draft-kv-cache-dtype fp8_e4m3 --ple-offload-embedding
--prefill-attention-backend triton --decode-attention-backend trtllm_mha
--linear-attn-prefill-backend triton --linear-attn-decode-backend flashinfer
--mamba-ssm-dtype bfloat16 --mamba-scheduler-strategy extra_buffer --max-mamba-cache-size 16
--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
--speculative-token-map lkqmoe/draft-token-map-49152.pt --speculative-draft-model-quantization modelopt_mixed
--cuda-graph-backend-decode full --cuda-graph-max-bs-decode 2 --cuda-graph-bs-decode 1 2
--reasoning-parser qwen3 --tool-call-parser qwen3_coder
--json-model-override-args '{"text_config":{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":262144}}}'
LVLLM_GPU_RESIDENT_MOE_LAYERS=0,1,2,5,6,7,8,10,11,12,13,14,17,18,19,20,21,22,26,28,29,33,34,37,38
LK_THREADS=64  LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=4096
LKQMOE_THREADS=112  LKQMOE_PROBE_DISPATCH_CPU=95  LKQMOE_MAIN_CPUS=auto
LKQMOE_MODE=standalone  LKQMOE_ZERO_COPY=1  LKQMOE_MTP_QUANT=1  LKQMOE_MTP_ROUTER_GUARD=1  LKQMOE_PREFILL_PREFETCH=1
LKQMOE_PREFILL_EXPERT_CHUNK=32  LKQMOE_PREFILL_ROUTE_ACCUM=1
LKQSTRATA=1  LKQSTRATA_SLOTS=155  LKQSTRATA_SWAPS=32  LKQSTRATA_PROFILE=lkqmoe/lkqstrata-qwen38.json
```
完整列表见 `launch/run-qwen38fn-nvfp4-24ch-nps4.sh`（20 通道 NPS2 用 `launch/run-qwen38fn-nvfp4.sh`；`RESIDENT` / `LKQSTRATA` / `LKQSTRATA_SLOTS` / `CONTEXT` / `MAX_REQUESTS` 可用环境变量覆盖）。

## lkqmoe（`lkqmoe/`）

- 编译版 0.5.6：`liblkqmoe.so`（分发库）+ `liblkqmoe_avx512.so` / `liblkqmoe_avx2.so`（CPU 内核）、`liblkqmoe_cuda.so`（CUDA 信箱桥与热专家池的换入换出）、`python/lkqmoe/*.pyc`（lsglang
  适配层、Triton GPU 预填充、LKQstrata 热专家池；含 Triton 内核的模块内嵌压缩源码，因为 Triton 编译时要读源码）。许可见
  `lkqmoe/LICENSE`：可免费使用、原样再分发。
- 版本 0.5.6（2026-10-22，与 lsglang_ds4.1、lsglang_glm5.3 是同一份编译版）：CPU 内核按 NUMA 拓扑自动分配（NPS2 4 节点、NPS4 8 节点），
  `LKQMOE_MAIN_CPUS=auto` 自动取"允许的 CPU − 专家线程 − 信箱调度线程"；同时支持只有 AVX2 的 CPU 与多卡张量并行（见下面的使用建议）；
  CPU Down 一次读一行（`-DLKQMOE_DOWN_R4=1`），本模型实权重单层比四行版快 3–13%、输出逐位相同。各版本的改动见上面「版本更新」。
- 开源部分（Apache-2.0）：`python/sitecustomize.py`、`python/lkqmoe/gpu/mtp_quant.py`（MTP 草稿保留 NVFP4，省约 2.9 GB 显存）、
  `python/lkqmoe/gpu/router_dependency.py`（MTP 路由的生产者/消费者依赖保护）。
- 精度：CPU 专家走 FP32 数值路径；热专家池的 GPU 部分是 W4A16（硬件 FP4→FP16，FP32 累加，BF16 中间结果），与常驻层同一精度；
  GPU 预填充对 FP64 参考的误差 1.66e-3。`LKQSTRATA=0` 时与上一版（0.4.1）在 118 个用例上逐位相同。
- 硬件要求：x86-64，AVX512-BF16/VBMI 或 AVX2/FMA（运行时自动选择），NUMA 拓扑自动识别（已在本机 NPS2 的 4 节点 × 16 核与 NPS4 的 8 节点 × 8 核上实测；hidden / intermediate 须是 16 × 节点数的倍数）。`launch/run-qwen38fn-nvfp4-24ch-nps4.sh` 用 `LKQMOE_MAIN_CPUS=auto`，`launch/run-qwen38fn-nvfp4.sh` 的列表按本机 NPS2 写好，CUDA GPU。

- AVX2 / 多卡使用建议：
  - 只有 AVX2 的 CPU 上专家计算更慢（同一台 Zen4 上实权重单层的 AVX2 路径约慢 1.4–1.6 倍），decode 更依赖热专家池，显存允许时多给槽位；
    `LKQMOE_THREADS` 取“每个 NUMA 节点的物理核数 − 1”乘节点数，留出信箱调度线程和主线程。
  - 多卡：设 `LKQMOE_TP_NODES=auto` 与 `LKQMOE_MAIN_CPUS=auto`，去掉写死的 `LKQMOE_PROBE_DISPATCH_CPU`；
    每张卡对应的 NUMA 节点最好属于同一颗 CPU（`nvidia-smi topo -m` 可查）。消费级卡之间没有 P2P，加 `--disable-custom-all-reduce`。
  - TP 下草稿热词表（`--speculative-token-map`）不能用：草稿 lm_head 按完整词表的 id 从目标 lm_head 取行，而 TP 下每个 rank
    只有自己那一半词表，会越界；去掉后草稿直接共用目标 lm_head，投机解码仍然无损。
  - lkqmoe 的 GPU 预填充每个分块都要把全部专家权重传到 GPU，PCIe 4.0 主机上 prefill 速度大致与分块大小成正比，显存不够时
    可减小 `LKQMOE_PREFILL_EXPERT_CHUNK` 和 DSA/索引器类的分块参数来换更大的预填充分块。

## 测试脚本（`bench/`）

- `speed_points.py`：任意长度打点测速（本页长上下文表）。
- `conc_bench.py`：并发测速（本页并发表）。
- `mtp_speed_bench.py`：4 类任务 2K 输入 / 512 输出的 MTP 速度，加一次 32K 冷启动首 token 时间。
- `multi_image_check.py`：多图请求，记录是否读对、首 token 时间与显存峰值。
- `build_token_map.py`：从实际输出生成草稿热词表。

## 许可

本仓库开源部分按 Apache-2.0（与 Lsglang/sglang 相同）；`lkqmoe/` 下的二进制见其 LICENSE。模型权重遵循各自许可。
