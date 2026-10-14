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

### 本版（2026-10-14）：512K + 2 并发，LKQstrata 热专家池

上一版把前 34 个 MoE 层整层放在 GPU（约 50 GB），上下文 256K，同时只跑 1 个请求。要支持 512K 和 2 并发，KV 缓存要多 2.2 GB，
bs=2 的 CUDA graph 和两个请求的运行时缓冲也要显存，只靠整层常驻只能放 32 层。本版把这块显存重新分配：第 0 层（与 MTP 草稿层共用
层号）和热专家池帮助最小的 24 层整层常驻，其余 23 层每层 155 个"热专家"槽位（共约 9.2 GiB）。这 23 层每层 decode 时，在槽位里的
专家由 GPU 计算（W4A16），其余由 CPU 计算，两边同时进行，最后相加；槽位启动时按排名表装填，运行中按路由统计自适应换入换出
（每 4 个验证窗口一轮，每窗口最多换入 32 个专家，拷贝不阻塞计算）。实测这 23 层每层 82–90% 的专家由 GPU 承担。

**长上下文打点（温度 0）**：单请求，`ignore_eos`，每点输出 1024 token（含思考内容）；输入是长日志文本，在 10% 深度埋一条事实并在
末尾提问（`bench/speed_points.py`，前 8 个点与上一版是同一批文档）。上一版数据 2026-10-01（34 层常驻、256K，
`results/speed-points-20261001-l34-accum.json`），本版 2026-10-14（`results/speed-points-20261014-lkqstrata.json`，本机部署与本仓库
`lkqmoe/` 是同一份源码与二进制；本仓库的编译版另在本机服务上复测，1 / 2 并发与 4K、32K 两点与下表一致）。接受长度是服务端从启动起的平均值，两版都是刚启动的服务只跑打点。

| 输入 token | 首 token 时间 上一版 → 本版 | 预填充 tok/s 上一版 → 本版 | decode tok/s 上一版 → 本版 | 接受长度 上一版 → 本版 | 找回埋藏事实 | 显存峰值 上一版 → 本版 |
|---:|---:|---:|---:|---:|:---:|---:|
| 4,109 | 0.87 → 1.19 s | 4,707 → 3,462 | 191.2 → **241.6**（+26%） | 3.10 → 3.44 | 是 | 71,060 → 63,646 MiB |
| 8,315 | 1.37 → 1.47 s | 6,051 → 5,658 | 211.1 → **219.0**（+4%） | 3.29 → 3.35 | 是 | 71,060 → 64,706 MiB |
| 17,123 | 2.63 → 2.67 s | 6,505 → 6,405 | 190.8 → **209.2**（+10%） | 3.22 → 3.22 | 是 | 71,060 → 67,028 MiB |
| 33,265 | 5.43 → 5.74 s | 6,132 → 5,793 | 184.1 → **190.2**（+3%） | 3.17 → 3.10 | 是 | 71,062 → 71,174 MiB |
| 68,103 | 11.67 → 12.46 s | 5,834 → 5,464 | 160.0 → **181.7**（+14%） | 3.06 → 3.00 | 是 | 71,708 → 71,820 MiB |
| 134,941 | 25.64 → 28.87 s | 5,263 → 4,675 | 163.3 → **165.8**（+2%） | 2.98 → 2.89 | 是 | 71,732 → 71,944 MiB |
| 210,634 | 35.11 → 34.82 s | 6,000 → 6,049 | 161.6 → **166.5**（+3%） | 2.94 → 2.81 | 是 | 71,972 → 71,964 MiB |
| 252,609 | 43.10 → 43.80 s | 5,861 → 5,767 | 171.4 → **207.6**（+21%） | 2.94 → 2.85 | 是 | 71,972 → 71,966 MiB |
| 414,774 | — → 82.17 s | — → 5,048 | — → **189.5** | — → 2.82 | 是 | — → 71,968 MiB |
| 499,966 | — → 106.47 s | — → 4,696 | — → **167.5** | — → 2.81 | 是 | — → 71,968 MiB |

- decode 8 点平均 179.2 → **197.7 tok/s（+10%）**；新增 384K、500K 两个点（上一版上限 256K）。
- 4 类任务（`bench/mtp_speed_bench.py`，2K 输入 / 512 输出，温度 0）：decode 171.4 → **191.2 tok/s（+12%）**，两版接受长度都是 2.72，
  即 MTP 验证一步 15.87 → 14.25 ms；32K 冷启动首 token 4.8 → 4.9 s（`results/tasks-20261014-lkqstrata.json`）。
- 多图（`bench/multi_image_check.py`，1920×1080 JPEG）：1 / 4 / 8 / 16 / 24 张全部读对，24 张（49K token）首 token 20.7 → 20.4 s，
  显存峰值 72,572 → 72,204 MiB（`results/multi-image-20261014-lkqstrata.json`）。
- 预填充变慢：上一版 14 层走 lkqmoe 的 GPU 预填充，本版 23 层（池里的层预填充时仍要从内存上传整层权重，并用 BF16 计算，常驻层
  直接用 FP4），首 token 时间 −1% ~ +13%，4K 多 0.32 s（+37%）。两次打点的 GPU 功耗墙不同（上一版 300 W，本版 350 W）；decode 用不到
  300 W，不受影响，预填充在 350 W 下约快 4%，所以同功耗墙下预填充的差距比上表再大约 4%。
- 精度：GPU 承担的专家是 W4A16（与常驻层同一精度），CPU 承担的仍是 FP32 数值路径。困惑度（13 段 × 4096 token，中英文文档与
  代理会话，各测 2 次）：同显存的 32 层常驻 3.4751 → 本版 3.4817，在两次之间的波动（单段约 ±1%）以内；4K–500K 的埋藏事实全部找回。
- 回到上一版：`LKQSTRATA=0 RESIDENT=0-33 CONTEXT=262144 MAX_REQUESTS=1 bash launch/run-qwen38fn-nvfp4.sh`。
- decode 速度随生成内容变化较大：接受长度不同，每步验证通过的 token 数就不同，单点 ±10% 很常见，比较时看多点平均或同一时段的 A/B。

**并发**（短提示，温度 1.0，每个请求 512 token，同时开始，各 4 轮；`bench/conc_bench.py`，同一天测得，
`results/concurrency-20261014.json`）：

| 并发 | 上一版（34 层常驻，256K，最多 1 个请求在跑）单请求 / 总吞吐 | 同显存不开池（32 层常驻，512K）单请求 / 总吞吐 | 本版（LKQstrata）单请求 / 总吞吐 | 总吞吐 本版 vs 上一版 |
|---:|---:|---:|---:|---:|
| 1 | 114.0 / 114.3 tok/s（接受长度 1.87） | 125.9 / 126.1（2.08） | **153.8 / 154.1**（2.20） | +35% |
| 2 | 118.8 / 114.9（第二个请求排队，平均首 token 2.6 s） | 86.5 / 163.4（1.91） | **115.0 / 217.0**（2.02） | **+89%** |

- 温度 1.0 的接受长度随内容波动；按每个验证步折算（tok/s ÷ 接受长度），1 并发时本版比上一版快约 15%，比同显存不开池快约 15%。
- 2 并发配置：`--max-running-requests 2 --max-mamba-cache-size 16`（每个请求占 4 个 mamba 状态槽），bs=1、2 的 decode CUDA graph。
  两个请求共用 524,288 token 的 KV 池：单个对话可以用满 512K；实测 500K 输入的请求与另一个短请求同时 decode，KV 用到 96%，
  没有请求被撤回。长输入预填充期间，另一个请求的预填充需排队等它完成。

**布局选择**（同一天，显存预算相同，512K + 2 并发；`results/layout-20261014.json`）：

| 布局 | 1 并发 decode（温度 1.0） | 2 并发总吞吐 | 4K 首 token | 500K 首 token | 500K decode（温度 0） |
|---|---:|---:|---:|---:|---:|
| 32 层整层常驻，不开池 | 125.9 | 163.4 | 0.99 s | 100.6 s | 130.7 |
| 全换：只常驻第 0 层，47 层 × 337 槽位 | 146.7 | 208.8 | 1.62 s | 123.3 s | 161.8 |
| **混用（本版）**：25 层常驻，23 层 × 155 槽位 | **148.1** | **212.9** | 1.08 s | 107.0 s | **185.6** |

全换时池命中率最高（95–99%），decode 与混用相当，但 47 层都要走 lkqmoe 的预填充，预填充慢 18–39%；混用把池命中率最低
（路由最分散）的层整层常驻，decode 不减、预填充只慢 6–8%。常驻哪些层由路由记录离线模拟选出（`lkqmoe/lkqstrata-qwen38.json`
同一批数据）。

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

- 编译版 0.5.0：`liblkqmoe.so`（CPU 内核）、`liblkqmoe_cuda.so`（CUDA 信箱桥与热专家池的换入换出）、`python/lkqmoe/*.pyc`（lsglang
  适配层、Triton GPU 预填充、LKQstrata 热专家池；含 Triton 内核的模块内嵌压缩源码，因为 Triton 编译时要读源码）。许可见
  `lkqmoe/LICENSE`：可免费使用、原样再分发。
- 开源部分（Apache-2.0）：`python/sitecustomize.py`、`python/lkqmoe/gpu/mtp_quant.py`（MTP 草稿保留 NVFP4，省约 2.9 GB 显存）、
  `python/lkqmoe/gpu/router_dependency.py`（MTP 路由的生产者/消费者依赖保护）。
- 精度：CPU 专家走 FP32 数值路径；热专家池的 GPU 部分是 W4A16（硬件 FP4→FP16，FP32 累加，BF16 中间结果），与常驻层同一精度；
  GPU 预填充对 FP64 参考的误差 1.66e-3。`LKQSTRATA=0` 时与上一版（0.4.1）在 118 个用例上逐位相同。
- 硬件要求：x86-64 AVX512-BF16，4 个 NUMA 节点 × 16 物理核（其他拓扑未验证），CUDA GPU。

## 测试脚本（`bench/`）

- `speed_points.py`：任意长度打点测速（本页长上下文表）。
- `conc_bench.py`：并发测速（本页并发表）。
- `mtp_speed_bench.py`：4 类任务 2K 输入 / 512 输出的 MTP 速度，加一次 32K 冷启动首 token 时间。
- `multi_image_check.py`：多图请求，记录是否读对、首 token 时间与显存峰值。
- `build_token_map.py`：从实际输出生成草稿热词表。

## 许可

本仓库开源部分按 Apache-2.0（与 Lsglang/sglang 相同）；`lkqmoe/` 下的二进制见其 LICENSE。模型权重遵循各自许可。
