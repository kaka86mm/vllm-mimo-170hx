#!/bin/bash
# 全模态版: 官方权重 + Omni arch + fp8 KV + MTP 投机 + 重平衡 PP
# v1.3: 移植 avtc/vllm-backport PR#106 三件套 —— OPROJ_FP8(在线fp8 o_proj, env 开)
#       + sm80 online-fp8 Marlin 路由 + 音频塔懒加载(休眠: 需要 audio 输入时保持默认;
#       想省 1.9GB 显存/启动时间可加 --limit-mm-per-prompt '{"audio": 0}')
set -e
MODEL_DIR=${MODEL_DIR:-/home/matri/models/MiMo-V2.6-Flash-RL-NVFP4}
PORT=${PORT:-8099}
IMAGE=${IMAGE:-docker.m.daocloud.io/lazymio/vllm-backport:v0.13.0-sm80}
# 前置检查: 生产补丁(视觉修复+split-KV+OPROJ_FP8)必须存在
for _p in "$(dirname "$0")/../patches/mimo_v2_omni_model.py" "$(dirname "$0")/../patches/mimo_v2_omni.py" "$(dirname "$0")/../patches/triton_attn_diffkv.py" "$(dirname "$0")/../patches/mimo_v2.py" "$(dirname "$0")/../patches/online_fp8.py"; do
  [ -f "$_p" ] || { echo "缺生产补件: $_p"; exit 1; }
done

docker rm -f mimo26 2>/dev/null || true
docker run -d --name mimo26 --restart unless-stopped \
  --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0,1,2,3 \
  -e HF_HUB_OFFLINE=1 -e VLLM_DIFFKV_FULL_ATTN_SEGMENTS=64 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_MIMO_OPROJ_FP8=1 \
  -e LMCACHE_MP_TRANSFER_MODE=engine_driven -e VLLM_PP_LAYER_PARTITION=11,13,12,12 \
  -v "$MODEL_DIR":/model:ro \
  -v "$(dirname "$0")/../patches/triton_attn_diffkv.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/triton_attn_diffkv.py:ro" \
  -v "$(dirname "$0")/../patches/mimo_v2_omni_model.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/mimo_v2_omni.py:ro" \
  -v "$(dirname "$0")/../patches/online_fp8.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/online/fp8.py:ro" \
  -v "$(dirname "$0")/../patches/mimo_v2.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/mimo_v2.py:ro" \
  -v "$(dirname "$0")/../patches/mimo_v2_omni.py:/usr/local/lib/python3.12/dist-packages/vllm/transformers_utils/processors/mimo_v2_omni.py:ro" \
  --network host --ipc host \
  "$IMAGE" \
  /model --served-model-name mimo26 --port 8099 \
  --pipeline-parallel-size 4 \
  --kv-cache-dtype fp8_e4m3 --block-size 128 \
  --max-model-len 1048576 \
  --max-num-batched-tokens 1024 \
  --trust-remote-code \
  --kv-cache-memory 10737418240 --gpu-memory-utilization 0.94 \
  --max-num-seqs 32 \
  --reasoning-parser mimo --tool-call-parser mimo \
  --enable-auto-tool-choice \
  --enable-prefix-caching --prefix-cache-retention-interval 1024 \
  --async-scheduling \
  --no-enable-flashinfer-autotune \
  --media-io-kwargs '{"video": {"num_frames": 128}}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2,"draft_sample_method":"probabilistic","rejection_sample_method":"block"}'
echo started
