#!/bin/bash
# 下载 MiMo-V2.6-Flash-RL 权重并完成部署前的两处必需加工：
#   1. 排除 gguf/ 子目录（约 183GB，vLLM 用不上，且可能写满磁盘）
#   2. config.json 架构从 MiMoV2ForCausalLM 改为 MiMoV2OmniForCausalLM
#      （ProCreations NVFP4 转码版默认声明文本架构；不改则无视觉/音频输入）
# 结束时按 index 校验 65 个分片完整性。
set -e
cd "$(dirname "$0")"
[ -f env.sh ] && . ./env.sh

REPO=ProCreations/MiMo-V2.6-Flash-RL-NVFP4
DIR=${MODEL_DIR:-$HOME/models/MiMo-V2.6-Flash-RL-NVFP4}
mkdir -p "$DIR"

echo "[1/3] 下载权重到 $DIR （~194GB，hf-mirror 直连 HF 常不通）"
HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com} \
  "${HFENV:-$HOME/hfenv}/bin/hf" download "$REPO" \
  --local-dir "$DIR" --max-workers 12 --exclude "gguf/*"

echo "[2/3] 切换 config.json 架构为 Omni（全模态）"
python3 - "$DIR/config.json" <<'PYEOF'
import json, sys
p = sys.argv[1]
c = json.load(open(p))
if c["architectures"] != ["MiMoV2OmniForCausalLM"]:
    c["architectures"] = ["MiMoV2OmniForCausalLM"]
    json.dump(c, open(p, "w"), indent=2, ensure_ascii=False)
    print("  已切换:", c["architectures"])
else:
    print("  已是 Omni，跳过")
PYEOF

echo "[3/3] 校验分片完整性"
python3 - "$DIR" <<'PYEOF'
import json, os, sys
d = sys.argv[1]
idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))
need = sorted(set(idx["weight_map"].values()))
missing = [f for f in need if not os.path.exists(os.path.join(d, f))
           or os.path.getsize(os.path.join(d, f)) < 1e6]
total = sum(os.path.getsize(f) for f in need if os.path.exists(os.path.join(d, f)))
print(f"  {len(need)-len(missing)}/{len(need)} 分片就位, 合计 {total/1e9:.1f}GB (预期 ~194GB)")
assert not missing, f"缺失: {missing[:5]}"
small = [f for f in ["config.json","tokenizer.json","tokenizer_config.json",
                     "generation_config.json","preprocessor_config.json"]
         if not os.path.exists(os.path.join(d, f))]
assert not small, f"关键文件缺失: {small}"
print("  校验通过")
PYEOF
echo "完成。启动: ./launch-omni.sh"
