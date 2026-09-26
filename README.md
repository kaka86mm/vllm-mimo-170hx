# vllm-mimo-170hx

Serve **XiaomiMiMo/MiMo-V2.6-Flash-RL** — a 309B-MoE (15B active) omni-modal model with 1M-token context — on a cluster of **4× NVIDIA CMP 170HX** (GA100 sm_80, 64 GB unlocked each, PCIe 2.0 x4, no P2P, pipeline-parallel only), using [wtdcode/vllm-backport](https://github.com/wtdcode/vllm-backport).

Text, image, video and audio in; tool calls and a reasoning parser enabled. Everything below is measured on the real box (Sep 2026), with the exact scripts in this repo.

## Performance

Production config = official 161 GB fp8 checkpoint, PP=4, fp8 KV cache (15 GiB), MTP 2-token speculative decoding, fp8 o_proj. `scripts/bench-full.py` reproduces every number (unique-content prompts, thinking off, warm decode):

| Metric | Result |
|---|---|
| Single-stream decode (greedy) | **107–109 tok/s** (MTP k=2; k=3 measured 100–104) |
| Single-stream decode (temp 1.0, streaming) | 74–82 tok/s |
| 8 / 16-stream aggregate | 243 / 353 tok/s |
| Prefill, 18.7K-token prompt (cold) | ~4.0 s ≈ 4.7K tok/s |
| Prefill, 75K-token prompt (cold) | 26–29 s ≈ 2.6–2.9K tok/s |
| KV pool | **1,994,875 tokens** = 1.90× a 1M-token request |
| TTFT (short prompt) | 0.1–0.5 s |

The ProCreations NVFP4 transcode (194 GB) is kept as a fallback (`scripts/launch-omni.sh`): 92–94 tok/s decode, 1,329,835-token KV pool — same prefill. Frozen state: tag `v1.3-prod`.

## Requirements

- 4× CMP 170HX with [CMPUnlocker](https://github.com/tinygrad/CMPUnlocker)-class unlock (64 GB visible; PCIe Gen2 restored; sm_80)
- ~200 GB disk for weights, Docker + NVIDIA container toolkit
- `lazymio/vllm-backport:v0.13.0-sm80` image (pulled via the launch script's registry mirror)

## Quick start

```bash
# 1. weights (161 GB, official checkpoint)
huggingface-cli download XiaomiMiMo/MiMo-V2.6-Flash-RL --local-dir ~/models/MiMo-V2.6-Flash-RL-official
# fix the hidden 2048-token output cap shipped in generation_config.json
python3 -c "import json; p='$HOME/models/MiMo-V2.6-Flash-RL-official/generation_config.json'; \
g=json.load(open(p)); g['max_new_tokens']=65536; json.dump(g, open(p,'w'), indent=2)"

# 2. launch (mounts the patches, starts serving on :8099)
bash scripts/launch-official.sh

# 3. verify
curl -s http://127.0.0.1:8099/v1/models
python3 scripts/bench-full.py
```

Step-by-step bring-up with verification checkpoints and a troubleshooting table of every error actually hit: **[DEPLOY.md](DEPLOY.md)**.

## Layout

- `scripts/launch-official.sh` — production launcher (official weights)
- `scripts/launch-omni.sh` — NVFP4 fallback launcher
- `scripts/bench-full.py` — decode + prefill benchmark (real token counts)
- `scripts/omni-test*.py`, `video-diag.py` — multimodal verification
- `scripts/kv-stress2.py` — long-context admission stress
- `patches/` — vLLM patches mounted by the launchers (see below)
- `docs/RESULT.md` — full engineering log: every pitfall, decision and dead end

## The patches (what and why)

| File | Purpose |
|---|---|
| `patches/mimo_v2.py` | Official-checkpoint fused-QKV loader: **exact (lossless) sharding below the checkpoint's TP=4 chunking** — SWA layers load as a pure scale-block permutation, GA layers keep each chunk's K/V rows on their checkpoint scales in a zero-padded layout the forward splits (`VLLM_MIMO_EXACT_QKV=0` restores the requantizing path; audit: `scripts/audit-exact-qkv.py`, 0 weights changed on all 48 layers). Also carries the opt-in fp8 o_proj quantization (`VLLM_MIMO_OPROJ_FP8=1`) |
| `patches/mimo_v2_mtp.py` | MTP drafter loader for the official `model_mtp.safetensors` (fp8 weight/scale paired sharding) |
| `patches/fp8.py` + `patches/online_fp8.py` | Route fp8 GEMMs to Marlin W8A16 on sm80 (no native fp8); without this the stock selection falls into a runtime-dequant path that costs 73% decode |
| `patches/mimo_v2_omni_model.py`, `patches/mimo_v2_omni.py` | ViT attention-sink fix (upstream [vllm#58235](https://github.com/vllm-project/vllm/pull/58235) port — without it the model is color-blind) + processor fixes; optional audio-tower skip at `audio=0` |
| `patches/triton_attn_diffkv.py` | Split-KV knob for the DiffKV attention verify step (`VLLM_DIFFKV_FULL_ATTN_SEGMENTS=64`) |

All patches are inactive on the NVFP4 track, so both launchers coexist.

## Operational notes

- **KV sizing ceiling**: 15 GiB is the max on this box — 16 GiB leaves <3 MB at cudagraph capture on the drafter rank and fails to boot.
- **Multimodal + thinking**: send `"chat_template_kwargs": {"enable_thinking": false}` with image/video input; the reasoning parser intermittently swallows content otherwise.
- **Audio input**: use `audio_url` + wav (not OpenAI's `input_audio`).
- GPU memory is intentionally uneven (encoders on PP0, drafter + lm_head on the last rank); `VLLM_PP_LAYER_PARTITION=11,13,12,12` is the measured balance and further tuning gains <1%.
- Prompt-prefix caching is on; expect ~33K tok/s on cache-hit re-prefills.

## Credits

- [wtdcode/vllm-backport](https://github.com/wtdcode/vllm-backport) — the sm80 fork this runs on
- [avtc/vllm-backport PR#106](https://github.com/wtdcode/vllm-backport/pull/106) — the official-checkpoint sharding, MTP-loader and dense-fp8 Marlin fixes ported here
- Upstream vLLM [#58235](https://github.com/vllm-project/vllm/pull/58235) — the ViT sink fix
- Related upstream contributions from this work: [#103](https://github.com/wtdcode/vllm-backport/pull/103), [#104](https://github.com/wtdcode/vllm-backport/pull/104), [#105](https://github.com/wtdcode/vllm-backport/issues/105)

## 中文摘要

4×CMP 170HX（无 P2P、只能 PP4）上跑 MiMo-V2.6-Flash-RL 全模态的生产部署。现役配置 = 官方 161GB fp8 权重：单流 greedy 100–104 tok/s、16 流聚合 ~300 tok/s、KV 池 199 万 token（可同时容纳 2 个完整 1M 上下文）。文/图/视/音四模态 + 工具调用全部可用。核心坑（官方 QKV 的 NB=4 导出布局、sm80 上 fp8 落入运行时反量化路径导致 -73% decode、滑窗 KV 在途预留随流水线深度膨胀、ViT sink 色彩 bug）均已修复并沉淀为挂载补丁。完整工程日志见 `docs/RESULT.md`；NVFP4 回退轨保留在 `launch-omni.sh` / tag `v1.3-prod`。
