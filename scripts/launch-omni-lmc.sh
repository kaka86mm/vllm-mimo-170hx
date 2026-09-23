#!/bin/bash
# 全模态版: 官方权重 + Omni arch + fp8 KV + MTP 投机 + 重平衡 PP
set -e
MODEL_DIR=${MODEL_DIR:-$HOME/models/MiMo-V2.6-Flash-RL-NVFP4}
PORT=${PORT:-8099}
IMAGE=${IMAGE:-docker.m.daocloud.io/lazymio/vllm-backport:v0.13.0-sm80}
docker rm -f mimo26 2>/dev/null || true
docker run -d --name mimo26 --restart unless-stopped \
  --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0,1,2,3 \
  -e HF_HUB_OFFLINE=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e LMCACHE_MP_TRANSFER_MODE=engine_driven -e VLLM_PP_LAYER_PARTITION=11,13,12,12 \
  -v "$MODEL_DIR":/model:ro \
  --network host --ipc host \
  "$IMAGE" \
  /model --served-model-name mimo26 --port 8099 \
  --pipeline-parallel-size 4 \
  --kv-cache-dtype fp8_e4m3 --block-size 128 \
  --max-model-len 262144 \
  --max-num-batched-tokens 8192 \
  --trust-remote-code \
  --kv-cache-memory 9663676416 --gpu-memory-utilization 0.94 \
  --max-num-seqs 32 \
  --reasoning-parser mimo --tool-call-parser mimo \
  --enable-auto-tool-choice \
  --enable-prefix-caching --prefix-cache-retention-interval 1024 \
  --async-scheduling \
  --no-enable-flashinfer-autotune \
  --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_connector_module_path":"lmcache.integration.vllm.lmcache_mp_connector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"127.0.0.1","lmcache.mp.port":5556,"lmcache.mp.mp_transfer_mode":"engine_driven","kv_buffer_size":268435456}}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"greedy","rejection_sample_method":"standard"}'
echo started
