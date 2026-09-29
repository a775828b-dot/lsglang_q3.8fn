#!/usr/bin/env bash
# Qwen3.8 Flash Next NVFP4 + MTP (NEXTN) on one RTX PRO 5000 72GB (SM120) + 2x EPYC 9334 (4 NUMA nodes).
# 32 MoE layers on the GPU; routed experts of the remaining layers run on lkqmoe (compiled, ../lkqmoe).
# This is the launch used for the measurements in README.md; edit the paths block first.
set -Eeuo pipefail

# ---- paths ------------------------------------------------------------------------------
REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
LSGLANG=${LSGLANG:-/opt/Lsglang}      # guqiong96/Lsglang @ c49d8f3703 (v1.4.12) + patches/lsglang-c49d8f37-qwen38fn.patch
VENV=${VENV:-$LSGLANG/env}            # python 3.12, see README "Environment"
MODEL=${MODEL:-/models/lovedheart/Qwen3.8-Flash-Next-NVFP4-W4A16-ATTN-FP8-MTP-NVFP4}
PORT=${PORT:-18080}
STATS_DIR=${STATS_DIR:-$REPO/run/lkqmoe-stats}

cuda_root="$VENV/lib/python3.12/site-packages/nvidia/cu13"
lkq="$REPO/lkqmoe"
mkdir -p "$STATS_DIR" "$REPO/run/cache"
cd "$LSGLANG"
exec env \
  LKQMOE_MODE=standalone \
  LKQMOE_MTP_ROUTER_GUARD=1 \
  LKQMOE_PREFILL_RELEASE_WORKSPACE=1 \
  LKQMOE_LIBRARY="$lkq/liblkqmoe.so" \
  LKQMOE_STATS_DIR="$STATS_DIR" LKQMOE_ZERO_COPY=1 LKQMOE_MTP_QUANT=1 LKQMOE_PREFILL_PREFETCH=1 LKQMOE_MAIN_CPUS=64-79,81-127 \
  LKQMOE_ORIGINAL_WARMUP=0 \
  LKQMOE_CPU_PREFILL_BATCH=1024 \
  LKQMOE_SPIN_COUNT=1048576 \
  LKQMOE_PREFILL_UNPACK=1 \
  LKQMOE_GPU_PREFILL_CHUNK=8192 \
  PYTHONPATH="$lkq/python:$LSGLANG/python" \
  LVLLM_ENABLE_VALIDATED_SPLITK=1 \
  SGLANG_W4A16_RESIDENT_MARLIN=1 \
  CUDA_HOME="$cuda_root" PATH="$cuda_root/bin:$PATH" \
  LD_LIBRARY_PATH="$cuda_root/lib:${LD_LIBRARY_PATH:-}" \
  TVM_FFI_GPU_BACKEND=cuda TVM_FFI_CUDA_ARCH_LIST=12.0 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  LVLLM_MOE_NUMA_ENABLED=1 LVLLM_ENABLE_MOE_LAYERWISE_LOAD=1 \
  LVLLM_ENABLE_NUMA_INTERLEAVE=1 LK_THREADS=64 OMP_NUM_THREADS=1 \
  LK_THREAD_BINDING=CPU_CORE LK_POWER_SAVING=1 \
  LVLLM_GPU_PREFETCH_WINDOW=1 LVLLM_GPU_PREFILL_MIN_BATCH_SIZE=4096 \
  LVLLM_GPU_RESIDENT_MOE_LAYERS=0-31 \
  LVLLM_GPU_RESIDENT_MOE_LAYERS_DSPARK=0 \
  FLASHINFER_DISABLE_VERSION_CHECK=1 \
  XDG_CACHE_HOME="$REPO/run/cache" \
  "$VENV/bin/sglang" serve \
    --model-path "$MODEL" \
    --served-model-name Qwen3.8-Flash-Next \
    --host 127.0.0.1 --port "$PORT" --trust-remote-code \
    --tp-size 1 \
    --weight-loader-drop-cache-after-load \
    --context-length 262144 --max-running-requests 1 --max-total-tokens 262144 \
    --json-model-override-args '{"text_config":{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":262144}}}' \
    --chunked-prefill-size 32768 --mem-fraction-static 0.90 --page-size 64 \
    --quantization modelopt_mixed --moe-runner-backend flashinfer_cutlass \
    --kv-cache-dtype nvfp4 --speculative-draft-kv-cache-dtype fp8_e4m3 \
    --ple-offload-embedding \
    --prefill-attention-backend triton \
    --decode-attention-backend trtllm_mha \
    --linear-attn-prefill-backend triton \
    --linear-attn-decode-backend flashinfer \
    --mamba-ssm-dtype bfloat16 --mamba-scheduler-strategy extra_buffer --max-mamba-cache-size 16 \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 3 --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 --speculative-token-map "$lkq/draft-token-map-49152.pt" \
    --speculative-draft-model-quantization modelopt_mixed \
    --disable-flashinfer-autotune \
    --cuda-graph-backend-decode full --cuda-graph-max-bs-decode 1 \
    --cuda-graph-bs-decode 1 --cuda-graph-backend-prefill disabled \
    --image-processor-backend pil \
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
    --watchdog-timeout 1200 --enable-metrics \
    --enable-cache-report "$@"
