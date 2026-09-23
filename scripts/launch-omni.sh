#!/bin/bash
# 最终验证配置：ProCreations NVFP4(Omni) + PP=4 + fp8 KV + MTP 投机 + mnbt 1024
# 所有数字的来源见仓库 README；mnbt=1024 是 KV 池扩容的关键（原因见 docs/RESULT.md）
set -e
cd "$(dirname "$0")"
[ -f env.sh ] && . ./env.sh
MODEL_DIR=${MODEL_DIR:-$HOME/models/MiMo-V2.6-Flash-RL-NVFP4}
PORT=${PORT:-8099}
CONTAINER=${CONTAINER:-mimo26}
IMAGE=${IMAGE:-docker.m.daocloud.io/lazymio/vllm-backport:v0.13.0-sm80}
GPUS=${GPUS:-0,1,2,3}

# 前置检查：分片与 Omni 架构
python3 - "$MODEL_DIR" <<'PYEOF'
import json, os, sys
d = sys.argv[1]
idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))
need = sorted(set(idx["weight_map"].values()))
missing = [f for f in need if not os.path.exists(os.path.join(d, f))]
assert not missing, f"缺 {len(missing)} 个分片，先跑 ./download-weights.sh"
arch = json.load(open(os.path.join(d, "config.json")))["architectures"]
assert arch == ["MiMoV2OmniForCausalLM"], f"arch={arch}，需先跑 ./download-weights.sh 切换 Omni"
print(f"[OK] {len(need)} 分片 + Omni 架构")
PYEOF

docker rm -f "$CONTAINER" 2>/dev/null || true
docker run -d --name "$CONTAINER" --restart unless-stopped \
  --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=$GPUS \
  -e HF_HUB_OFFLINE=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_PP_LAYER_PARTITION=11,13,12,12 \
  -v "$MODEL_DIR":/model:ro \
  --network host --ipc host \
  "$IMAGE" \
  /model --served-model-name mimo26 --port "$PORT" \
  --pipeline-parallel-size 4 \
  --kv-cache-dtype fp8_e4m3 --block-size 128 \
  --max-model-len 262144 \
  --max-num-batched-tokens 1024 \
  --trust-remote-code \
  --kv-cache-memory 10737418240 --gpu-memory-utilization 0.94 \
  --max-num-seqs 32 \
  --reasoning-parser mimo --tool-call-parser mimo \
  --enable-auto-tool-choice \
  --enable-prefix-caching --prefix-cache-retention-interval 1024 \
  --async-scheduling \
  --no-enable-flashinfer-autotune \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"greedy","rejection_sample_method":"standard"}'

echo "容器已启动，加载约 7 分钟。日志: docker logs -f $CONTAINER"
echo "就绪探测: curl http://127.0.0.1:$PORT/health"
# 注意：本镜像 ENTRYPOINT 已是 [vllm serve]，容器参数不要再写 "vllm serve" 前缀
# 期望日志: GPU KV cache size: 1,160,104 tokens ... 4.43x
