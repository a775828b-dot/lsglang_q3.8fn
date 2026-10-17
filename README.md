# lsglang_q3.8fn

Qwen3.8 Flash Next NVFP4 单卡推理管线：一张 RTX PRO 5000 72GB + 双路 EPYC，512K 上下文、2 并发，MTP 投机解码，
**LKQstrata 热专家池**（CPU 层的热门专家常驻 GPU、自适应换入换出，与 CPU 同时计算），支持图片。
框架是 [guqiong96/Lsglang](https://github.com/guqiong96/Lsglang)（sglang 的 CPU/GPU 混合推理分支），CPU 层的路由专家由受
[lk_moe](https://github.com/guqiong96/Lsglang)（lsglang 作者 guqiong96 的 CPU MoE 后端）启发、并针对 NVFP4 格式专项加速的
**lkqmoe** 承担计算（本仓库附编译好的闭源版本）。

*English: a single-GPU Qwen3.8 Flash Next NVFP4 pipeline (RTX PRO 5000 72GB + 2x EPYC 9334, 512K context, 2 concurrent requests,
MTP speculative decoding, LKQstrata GPU expert pool for the hot experts of the CPU MoE layers, image input) on guqiong96/Lsglang,
with the CPU-side routed experts on lkqmoe, an NVFP4-specialised MoE kernel inspired by lk_moe (by guqiong96, the author of lsglang);
a compiled, closed-source build is included here.*

## 组成

| 部分 | 来源 | 本仓库内容 |
|---|---|---|
| 模型 | [lovedheart/Qwen3.8-Flash-Next-NVFP4-W4A16-ATTN-FP8-MTP-NVFP4](https://huggingface.co/lovedheart/Qwen3.8-Flash-Next-NVFP4-W4A16-ATTN-FP8-MTP-NVFP4)：lovedheart 对 Qwen 团队 **Qwen3.8 Flash Next** 的量化版（专家 NVFP4 W4A16，注意力 FP8，MTP 草稿专家 NVFP4），121 GB | — |
| 推理框架 | guqiong96/Lsglang v1.4.12 @ [`c49d8f3703`](https://github.com/guqiong96/Lsglang/commit/c49d8f3703b1b829d0eedbd53f2a4c2443fa8800)；Qwen3.8 支持来自 sglang PR #36497（Qiaolin-Yu, "Introduce Qwen 3.8 Flash Next"），另有本地调试修改与验证过的 Split-K | `patches/lsglang-c49d8f37-qwen38fn.patch` |
| CPU 专家内核 | **lkqmoe**（闭源，编译版）：受 lk_moe 启发、针对 NVFP4 格式专项加速的 CPU/GPU 混合 MoE 内核。lk_moe 来自 lsglang 作者 guqiong96（[guqiong96/Lsglang](https://github.com/guqiong96/Lsglang)）；lkqmoe 独立实现其 `MOE_NVFP4` 接口，不含 lk_moe 代码。0.5.0 起含 **LKQstrata 热专家池**：GPU 上缓存每层的热门专家，换入换出策略参考 [Strata](https://github.com/Niko1221/Strata)（MIT，独立实现） | `lkqmoe/` |
| 热专家池排名表 | 从 Qwen 实际 decode 路由统计的每层专家使用排名（52 个请求：中英文、代码、数学、agent 排错等短提示，约 11 万 token 的长文问答，5 轮编程对话；温度 1.0）；启动时按它装填，之后自适应 | `lkqmoe/lkqstrata-qwen38.json` |
| MTP 草稿热词表 | 从实际输出统计的 49152 个高频 token（只影响草稿接受率，目标模型仍验证每个 token） | `lkqmoe/draft-token-map-49152.pt` |
| 启动 | 测速所用的完整参数 | `launch/run-qwen38fn-nvfp4.sh` |

## 本机硬件

| | |
|---|---|
| CPU | 2× AMD EPYC 9334（共 64 核 128 线程），NPS2 → 4 个 NUMA 节点，AVX512-BF16/VBMI |
| 内存 | 320 GB DDR5-4800（20× 16 GB），实测带宽约 511 GB/s |
| GPU | NVIDIA RTX PRO 5000 72GB Blackwell（SM120），驱动 595.99，CUDA 13.2；本版测速时功耗墙 350 W（上一版 10-01 测速时为默认的 300 W） |
| 系统 | Ubuntu 24.04.4，内核 7.0.0-31，Python 3.12.3 |

## 速度

**本版升级（2026-10-14，lkqmoe 0.5.0 LKQstrata 热专家池）**

- 上一版把前 34 个 MoE 层整层放在 GPU，上下文 256K，同时只能跑 1 个请求。
- 本版把显存重新分配：第 0 层和热专家池帮助最小的 24 层整层常驻，其余 23 层每层 155 个"热专家"槽位。这些层 decode 时，槽位里的
  专家由 GPU 计算（W4A16），其余由 CPU 计算，两边同时进行后相加；槽位按排名表装填，运行中自适应换入换出，实测 82–90% 的专家由 GPU 承担。
- 上下文 256K → **512K**，同时运行的请求 1 → **2**。
- 精度不变：困惑度 3.4751 → 3.4817（在两次测量的波动内），4K–500K 的埋藏事实都找回。代价是首 token 慢 −1% ~ +13%（4K 多 0.32 s；池里的层预填充要从内存上传权重）。
- 回到上一版：`LKQSTRATA=0 RESIDENT=0-33 CONTEXT=262144 MAX_REQUESTS=1 bash launch/run-qwen38fn-nvfp4.sh`。

**单请求：上一版 → 本版**（温度 0，每点输出 1024 token，输入是埋了一条事实的长文本，`bench/speed_points.py`；
上一版 2026-10-01，本版 2026-10-14；原始数据 `results/speed-points-20261001-l34-accum.json`、`results/speed-points-20261014-lkqstrata.json`）

| 输入 token | decode tok/s | 首 token 时间 |
|---:|---:|---:|
| 4K | 191.2 → **241.6**（+26%） | 0.87 → 1.19 s |
| 8K | 211.1 → **219.0**（+4%） | 1.37 → 1.47 s |
| 17K | 190.8 → **209.2**（+10%） | 2.63 → 2.67 s |
| 33K | 184.1 → **190.2**（+3%） | 5.43 → 5.74 s |
| 68K | 160.0 → **181.7**（+14%） | 11.67 → 12.46 s |
| 135K | 163.3 → **165.8**（+2%） | 25.64 → 28.87 s |
| 211K | 161.6 → **166.5**（+3%） | 35.11 → 34.82 s |
| 253K | 171.4 → **207.6**（+21%） | 43.10 → 43.80 s |
| **平均** | 179.2 → **197.7（+10%）** | |
| 415K（上一版不支持） | **189.5** | 82.17 s |
| 500K（上一版不支持） | **167.5** | 106.47 s |

**并发：上一版 → 本版**（短提示，温度 1.0，每个请求 512 token，同时开始，各 4 轮，`bench/conc_bench.py`，同日测得，
`results/concurrency-20261014.json`）

| 并发 | 上一版 单请求 / 总吞吐 | 本版 单请求 / 总吞吐 | 总吞吐 |
|---:|---:|---:|---:|
| 1 | 114.0 / 114.3 tok/s | 153.8 / 154.1 tok/s | **+35%** |
| 2 | 118.8 / 114.9 tok/s（第二个请求排队） | 115.0 / 217.0 tok/s | **+89%** |

两个请求共用 524,288 token 的 KV 池，单个对话可以用满 512K。

## 运行方式

- 模型 48 层（36 层线性注意力 GDN + 12 层全注意力 QSA），MoE 512 专家 top-10（hidden 2560，专家中间维 640）。
- 第 0 层和第 1、2、5–8、10–14、17–22、26、28、29、33、34、37、38 层（共 25 层）整层常驻 GPU（Marlin W4A16）；其余 23 层的路由
  专家由 lkqmoe 承担：CPU（64 线程）加上 GPU 上的 LKQstrata 热专家池（每层 155 个槽位）；MTP 草稿全在 GPU。
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
3. 下载模型，按需修改 `launch/run-qwen38fn-nvfp4.sh` 开头的路径后启动：`bash launch/run-qwen38fn-nvfp4.sh`。
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
LKQMOE_MODE=standalone  LKQMOE_ZERO_COPY=1  LKQMOE_MTP_QUANT=1  LKQMOE_MTP_ROUTER_GUARD=1  LKQMOE_PREFILL_PREFETCH=1
LKQMOE_PREFILL_EXPERT_CHUNK=32  LKQMOE_PREFILL_ROUTE_ACCUM=1
LKQSTRATA=1  LKQSTRATA_SLOTS=155  LKQSTRATA_SWAPS=32  LKQSTRATA_PROFILE=lkqmoe/lkqstrata-qwen38.json
```
完整列表见 `launch/run-qwen38fn-nvfp4.sh`（`RESIDENT` / `LKQSTRATA` / `LKQSTRATA_SLOTS` / `CONTEXT` / `MAX_REQUESTS` 可用环境变量覆盖）。

## lkqmoe（`lkqmoe/`）

- 编译版 0.5.3：`liblkqmoe.so`（分发库）+ `liblkqmoe_avx512.so` / `liblkqmoe_avx2.so`（CPU 内核）、`liblkqmoe_cuda.so`（CUDA 信箱桥与热专家池的换入换出）、`python/lkqmoe/*.pyc`（lsglang
  适配层、Triton GPU 预填充、LKQstrata 热专家池；含 Triton 内核的模块内嵌压缩源码，因为 Triton 编译时要读源码）。许可见
  `lkqmoe/LICENSE`：可免费使用、原样再分发。
- 版本 0.5.3（2026-10-17）：**同时支持只有 AVX2 的 CPU 和多卡张量并行（TP）**。
  - 指令集：`liblkqmoe.so` 启动时检测 CPU，有 AVX512-BF16/VBMI（Zen4、Sapphire Rapids 等）就加载 `liblkqmoe_avx512.so`，
    只有 AVX2/FMA（Zen2、Zen3 等）就加载 `liblkqmoe_avx2.so`，三个文件放在同一目录；启动日志打印 `[lkqmoe] CPU kernels: avx512`
    或 `avx2`，`LKQMOE_ISA=avx2` 可强制（例如在 AVX-512 机器上测 AVX2 路径）。AVX-512 路径与 0.5.2 逐位相同、速度持平
    （三个模型的实权重单层，64 / 112 线程，m = 1…2048）。AVX2 内核按 Zen2 优化（`vpshufb` 查表、每步 32 个权重），FP32 累加，
    误差与 AVX-512 同量级；同一输入在 Zen2 与 Zen4 上的 AVX2 输出逐位相同。
  - 张量并行：每个 TP rank 一个进程，各自持有每个专家 1/tp 的中间维，部分和由框架原有的 all-reduce 相加，CPU 侧不需要跨进程通信。
    `LKQMOE_TP_NODES=auto` 让每个 rank 只用其 GPU 所在 CPU 的 NUMA 节点（权重读本地内存）、并各自挑选信箱调度线程与主线程
    （也可按 rank 写，如 `0-3|4-7`）；`LKQMOE_THREADS` 按每个 rank 计。LKQstrata 热专家池在每张卡上各存半个专家。
  - 本管线验证：本机（AVX-512）换 0.5.3 后，关闭热专家池自适应时 PPL 3.4709（生产版 0.5.0 为 3.4721；开着自适应时两次启动之间差 0.2–0.4%，来自池内专家不同），500K 首 token 106.0 s、decode 200.3 tok/s，埋藏事实召回。**AVX2 主机和多卡 TP 没有在本管线上测试**（内核与 lsglang_glm5.3 是同一份二进制，GLM-5.3 有 AVX2 + TP2 实测）。
- 版本 0.5.2（2026-10-16）：CPU 内核按 NUMA 拓扑自动分配——从系统读取每个有 CPU 的 NUMA 节点（本机 NPS2 为 4 个、NPS4 为 8 个），每个节点存每个专家 1/N 的行，线程先用该节点的物理核、再用其 SMT 兄弟线程，不再写死 4 节点 × 16 核。`LKQMOE_MAIN_CPUS=auto` 自动取“允许的 CPU − 专家线程 − 信箱调度线程”，换拓扑或线程数不用手改列表。本机（NPS2）上绑核与 0.5.0 完全相同；三个模型的实权重单层在 64 / 112 线程、默认与生产开关、m = 1…2048 下输出与 0.5.0 逐位相同，速度持平；在本机把每个节点拆成两半模拟 NPS4 的 8 节点布局，输出同样逐位相同。启动日志会打印 `[lkqmoe] N worker nodes ...`。
- 开源部分（Apache-2.0）：`python/sitecustomize.py`、`python/lkqmoe/gpu/mtp_quant.py`（MTP 草稿保留 NVFP4，省约 2.9 GB 显存）、
  `python/lkqmoe/gpu/router_dependency.py`（MTP 路由的生产者/消费者依赖保护）。
- 精度：CPU 专家走 FP32 数值路径；热专家池的 GPU 部分是 W4A16（硬件 FP4→FP16，FP32 累加，BF16 中间结果），与常驻层同一精度；
  GPU 预填充对 FP64 参考的误差 1.66e-3。`LKQSTRATA=0` 时与上一版（0.4.1）在 118 个用例上逐位相同。
- 硬件要求：x86-64，AVX512-BF16/VBMI 或 AVX2/FMA（运行时自动选择），NUMA 拓扑自动识别（已验证 NPS2 的 4 节点 × 16 核，NPS4 的 8 节点 × 8 核在本机模拟验证；hidden / intermediate 须是 16 × 节点数的倍数）。启动脚本里的 `LKQMOE_MAIN_CPUS` 列表按本机 NPS2 写好，换到其他拓扑时改成 `auto`，CUDA GPU。

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
