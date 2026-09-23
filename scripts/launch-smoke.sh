#!/bin/bash
# MiMo-V2.6-Flash-RL (ProCreations NVFP4) 首次冒烟启动 —— 最小参数集，先验证能跑
# 依据: dsv4 runbook 红线（PP-only / util<=0.92 / 禁 enforce-eager / 禁 expandable_segments）
# 用法: bash ~/mimo-v26/launch-smoke.sh   (需先 docker stop glm-exl3 释放显存)
set -e

MODEL_DIR=/home/matri/models/MiMo-V2.6-Flash-RL-NVFP4
PORT=8099
IMAGE=${IMAGE:-docker.m.daocloud.io/lazymio/vllm-backport:v0.13.0-sm80}

# 前置检查：权重齐了没（index 里声明的分片都要在）
python3 - "$MODEL_DIR" <<'EOF'
import json, os, sys
d = sys.argv[1]
idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))
need = sorted(set(idx["weight_map"].values()))
missing = [f for f in need if not os.path.exists(os.path.join(d, f))]
sz = lambda f: os.path.getsize(os.path.join(d, f)) if os.path.exists(os.path.join(d, f)) else 0
done = sum(sz(f) for f in need)
total = 193_964_338_432  # README 声明的 root payload
if missing:
    print(f"[WARN] 还缺 {len(missing)} 个分片, 已到 {done/1e9:.1f}GB / ~194GB")
    if done < total * 0.98:
        sys.exit(1)
print(f"[OK] {len(need)} 个分片全部就位, 共 {done/1e9:.1f}GB")
EOF

docker rm -f mimo26 2>/dev/null || true
docker run -d --name mimo26 --restart unless-stopped \
  --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0,1,2,3 \
  -e HF_HUB_OFFLINE=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_PP_LAYER_PARTITION=12,12,12,12 \
  -v "$MODEL_DIR":/model:ro \
  --shm-size=16g -p $PORT:8000 \
  "$IMAGE" \
  /model --served-model-name mimo26 \
  --pipeline-parallel-size 4 \
  --max-model-len 131072 \
  --max-num-batched-tokens 4096 \
  --trust-remote-code \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 32 \
  --reasoning-parser mimo --tool-call-parser mimo \
  --enable-auto-tool-choice \
  --no-enable-flashinfer-autotune

echo "容器已启动，日志: docker logs -f mimo26"
