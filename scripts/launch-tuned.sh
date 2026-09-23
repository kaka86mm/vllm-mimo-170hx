#!/bin/bash
# 进阶版：冒烟通过后用 —— fp8 KV + MTP 投机解码 + 本机已验证的调度参数
# 参数来源: dsv4 runbook 红线 + glm-exl3 容器实测命令
set -e

MODEL_DIR=/home/matri/models/MiMo-V2.6-Flash-RL-NVFP4
PORT=8099
IMAGE=${IMAGE:-docker.m.daocloud.io/lazymio/vllm-backport:v0.13.0-sm80}

docker rm -f mimo26 2>/dev/null || true
docker run -d --name mimo26 --restart unless-stopped \
  --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0,1,2,3 \
  -e HF_HUB_OFFLINE=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_PP_LAYER_PARTITION=11,12,12,13 \
  -v "$MODEL_DIR":/model:ro \
  --shm-size=16g -p $PORT:8000 \
  "$IMAGE" \
  /model --served-model-name mimo26 \
  --pipeline-parallel-size 4 \
  --kv-cache-dtype fp8_e4m3 --block-size 256 \
  --max-model-len 131072 \
  --max-num-batched-tokens 4096 \
  --trust-remote-code \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 32 \
  --reasoning-parser mimo --tool-call-parser mimo \
  --enable-auto-tool-choice \
  --enable-prefix-caching\
  --async-scheduling \
  --no-enable-flashinfer-autotune \
  --speculative-config '{"method":"dflash","model":"/model/dflash","num_speculative_tokens":3,"draft_sample_method":"greedy","rejection_sample_method":"standard","attention_backend":"TRITON_ATTN"}'

echo "容器已启动，日志: docker logs -f mimo26"
# 变体:
#  A) DFlash drafter（diffbot 冠军，glm-exl3 同款方法）:
#     --speculative-config '{"method":"dflash","model":"/model/dflash","num_speculative_tokens":3,"draft_sample_method":"greedy","rejection_sample_method":"standard","attention_backend":"TRITON_ATTN"}'
#     (先确认 dflash/ 的 index 指向的权重文件已就位)
#  B) 无投机解码对照（量化投机收益）: 去掉 --speculative-config
#  C) 512K 上下文: --max-model-len 524288（GA 层 fp8 KV ~9KB/token，262K 单序列 ~2.4GB，池子够）
#  D) 失败回退: 若 fp8 KV 与 hybrid SWA 组合报错，去掉 --kv-cache-dtype/--block-size 再试
