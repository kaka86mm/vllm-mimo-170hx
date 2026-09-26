#!/bin/bash
# 官方权重版: XiaomiMiMo/MiMo-V2.6-Flash-RL 原版 fp8 checkpoint (161GB, 比 NVFP4 省 ~33GB
# → KV 池 10GiB→16GiB)。在 v1.3 生产配置之上叠加三件官方解锁补丁:
#   1. patches/mimo_v2.py        avtc 版 _shard_fp8_qkv_proj(NB=4 官方布局, 双 scale 布局自适应)
#                                + OPROJ_FP8(对 NVFP4 休眠)
#   2. patches/mimo_v2_mtp.py    官方 model_mtp.safetensors 加载修复(fp8 配对分片 + NB=4
#                                去交错 + 未建层跳过)。仅官方权重可挂; NVFP4 勿挂此文件
#   3. patches/fp8.py            离线序列化 fp8 dense 在无原生 fp8 的部件上强制 MarlinFP8
#                                W8A16(官方权重 -73% decode 的根因修复; 对 NVFP4 休眠)
# 回滚: git checkout main && bash scripts/launch-omni.sh
set -e
MODEL_DIR=${MODEL_DIR:-/home/matri/models/MiMo-V2.6-Flash-RL-official}
PORT=${PORT:-8099}
IMAGE=${IMAGE:-docker.m.daocloud.io/lazymio/vllm-backport:v0.13.0-sm80}
PATCHES="$(cd "$(dirname "$0")/../patches" && pwd)"
for _p in "$PATCHES/mimo_v2_omni_model.py" "$PATCHES/mimo_v2_omni.py" "$PATCHES/triton_attn_diffkv.py" "$PATCHES/mimo_v2.py" "$PATCHES/online_fp8.py" "$PATCHES/mimo_v2_mtp.py" "$PATCHES/fp8.py"; do
  [ -f "$_p" ] || { echo "缺补件: $_p"; exit 1; }
done

docker rm -f mimo26 2>/dev/null || true
docker run -d --name mimo26 --restart unless-stopped \
  --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0,1,2,3 \
  -e HF_HUB_OFFLINE=1 -e VLLM_DIFFKV_FULL_ATTN_SEGMENTS=64 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_MIMO_OPROJ_FP8=1 \
  -e LMCACHE_MP_TRANSFER_MODE=engine_driven -e VLLM_PP_LAYER_PARTITION=11,13,12,12 \
  -v "$MODEL_DIR":/model:ro \
  -v "$PATCHES/triton_attn_diffkv.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/triton_attn_diffkv.py:ro" \
  -v "$PATCHES/mimo_v2_omni_model.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/mimo_v2_omni.py:ro" \
  -v "$PATCHES/online_fp8.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/online/fp8.py:ro" \
  -v "$PATCHES/mimo_v2.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/mimo_v2.py:ro" \
  -v "$PATCHES/mimo_v2_mtp.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/mimo_v2_mtp.py:ro" \
  -v "$PATCHES/fp8.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/fp8.py:ro" \
  -v "$PATCHES/mimo_v2_omni.py:/usr/local/lib/python3.12/dist-packages/vllm/transformers_utils/processors/mimo_v2_omni.py:ro" \
  --network host --ipc host \
  "$IMAGE" \
  /model --served-model-name mimo26 --port 8099 \
  --pipeline-parallel-size 4 \
  --kv-cache-dtype fp8_e4m3 --block-size 128 \
  --max-model-len 1048576 \
  --max-num-batched-tokens 1024 \
  --trust-remote-code \
  --kv-cache-memory 16106127360 --gpu-memory-utilization 0.94 \
  --max-num-seqs 32 \
  --reasoning-parser mimo --tool-call-parser mimo \
  --enable-auto-tool-choice \
  --enable-prefix-caching --prefix-cache-retention-interval 1024 \
  --async-scheduling \
  --no-enable-flashinfer-autotune \
  --media-io-kwargs '{"video": {"num_frames": 128}}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2,"draft_sample_method":"probabilistic","rejection_sample_method":"block"}'
echo started
