# vllm-mimo-170hx

Serve **XiaomiMiMo/MiMo-V2.6-Flash-RL** — a 309B-MoE (15B active) omni-modal model with 1M-token context — on a cluster of **4× NVIDIA CMP 170HX** (GA100 sm_80, 64 GB unlocked each, PCIe 2.0 x4, no P2P, pipeline-parallel only), using [wtdcode/vllm-backport](https://github.com/wtdcode/vllm-backport).

Text, image, video and audio in; tool calls and a reasoning parser enabled. Everything below is measured on the real box (Sep 2026), with the exact scripts in this repo.

## Performance

Production config = official 161 GB fp8 checkpoint, PP=4 (`12,12,12,12`), fp8 KV cache (**16 GiB**), fp8 o_proj, **speculative decoding OFF** (see the long-context note below), and a **three-tier KV storage stack** (GPU → RAM → disk, see below). `scripts/bench-full.py` reproduces the decode/prefill numbers (unique-content prompts, thinking off, warm decode):

| Metric | Result |
|---|---|
| Single-stream decode (greedy, ~0 ctx) | 64.5 tok/s (no spec; MTP k=2 gives 106 but only at trivial contexts — see below) |
| Single-stream decode (temp 1.0, streaming) | 74–82 tok/s |
| Single-stream decode @19K / @67K ctx | 60 / 56 tok/s |
| Aggregate, short ctx ×2/4/8/16 streams | 110 / 137 / 210 / 302 tok/s |
| Aggregate, long ctx: 19K×2/×4/×8, 67K×2/×4 | 99 / 143 / 260, 103 / 136 tok/s |
| Prefill, 18.7K-token prompt (cold) | ~4.0 s ≈ 4.7K tok/s |
| Prefill, 75K-token prompt (cold) | 26–29 s ≈ 2.6–2.9K tok/s |
| Prefill, 200K-token prompt (cold, q-tiling kernel) | 74.7 s ≈ 2.7K tok/s (stock kernel: ~170 tok/s — see notes) |
| KV pool (GPU) | **2,127,834 tokens** = 2.03× a 1M-token request |
| KV tier 2 — host RAM (LMCache, SHM transport) | 100 GB ≈ **8.7M tokens**; engine restart → prefix back in **2.6 s** per 69K |
| KV tier 3 — disk (fs adapter) | bounded by free disk; full server+engine restart → 23.8 s per 69K (partial-hit, degrades to recompute) |
| TTFT (short prompt) | 0.1–0.5 s |

The ProCreations NVFP4 transcode (194 GB) is kept as a fallback (`scripts/launch-omni.sh`): 92–94 tok/s decode, 1,329,835-token KV pool — same prefill. Frozen state: tag `v1.3-prod`.

## Requirements

- 4× CMP 170HX with [CMPUnlocker](https://github.com/tinygrad/CMPUnlocker)-class unlock (64 GB visible; PCIe Gen2 restored; sm_80)
- ~200 GB disk for weights, Docker + NVIDIA container toolkit
- The engine image `vllm-mimo:lmc-pr4410c` built from `lazymio/vllm-backport:v0.13.0-sm80` + [LMCache PR#4410](https://github.com/LMCache/LMCache/pull/4410) (hybrid-model engine-driven transfers — build recipe in `docs/RESULT.md`)
- ≥64 GB free RAM for the CPU KV tier (100 GB pinned by default; see storage notes) and a spare directory on disk for tier 3

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

- `scripts/launch-official.sh` — production launcher (official weights; starts the CPU-tier server first, then the engine)
- `scripts/lmc-server-cmd.sh` — the CPU-tier server command, mounted as the container entrypoint (inline `-c` JSON quoting is unmanageable — don't inline it back)
- `scripts/launch-omni.sh` — NVFP4 fallback launcher
- `scripts/bench-full.py` — decode + prefill benchmark (real token counts)
- `scripts/bench-ctx-long.py` — steady-state decode vs context length (default 18600 words ≈ 67K tokens; pass N as argv[1])
- `scripts/bench-ctx-multi.py` — N-concurrent long-context decode (aggregate + per-stream tok/s)
- `scripts/kernel-ab-spec3d.py` — kernel-level A/B: stock vs spec-3D `unified_attention_diffkv` (CUDA events, GA shape)
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
| `patches/triton_unified_attention_diffkv.py` | Spec-3D verify kernel **+ prefill q-tiling** (from the [MiaAI-Lab recipe](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks)). Spec-3D env: `VLLM_DIFFKV_SPEC_3D_MAX_Q`/`_BLOCK_M`/`_NUM_WARPS`/`_TILE`. Prefill q-tiling: `max_seqlen_q >= 64` batches use `BLOCK_M=128` (8 query tokens per program) via `VLLM_DIFFKV_PREFILL_BLOCK_M`/`_TILE`/`_NUM_WARPS` — fixes the long-context chunked-prefill cliff (see below). Upstream: [vllm#59054](https://github.com/vllm-project/vllm/issues/59054) + [PR#59085](https://github.com/vllm-project/vllm/pull/59085) + [backport#110](https://github.com/wtdcode/vllm-backport/pull/110) |

All patches are inactive on the NVFP4 track, so both launchers coexist.

The CPU/disk tiers run **stock PR#4410 LMCache code** (`lmcache server --chunk-size 8192 --shm-name ... --l2-adapter '{"type":"fs","base_path":"/lmcache-l2"}'`); no local patches in the store/retrieve chain. Chunk size 8192 and SHM transport are both load-bearing: each removes about half of a ~0.55 s per-transfer-object fixed cost that otherwise dominates retrieval on PCIe-constrained hosts (measurements in [LMCache#5440](https://github.com/LMCache/LMCache/issues/5440)).

## Operational notes

- **KV sizing**: 16 GiB with the `12,12,12,12` layer split (one layer moved off the 13-layer rank; the drafter that used to OOM 16 GiB is gone in no-spec mode). Rank 2 emits allocator warnings at capture time with ~7 MB free — they self-recover; verify decode ITL ≈15 ms after any change here. 17 GiB is not reachable without per-layer rebalancing (gain +3%, declined).
- **Multimodal + thinking**: send `"chat_template_kwargs": {"enable_thinking": false}` with image/video input; the reasoning parser intermittently swallows content otherwise.
- **Audio input**: use `audio_url` + wav (not OpenAI's `input_audio`).
- GPU memory is intentionally uneven (vision encoders on PP0, lm_head on the last rank); `VLLM_PP_LAYER_PARTITION=12,12,12,12` is the measured balance with 16 GiB KV.
- Prompt-prefix caching is on; expect ~33K tok/s on cache-hit re-prefills.
- **Storage tiers & restart semantics**: engine restart loses nothing (prefixes retrieve from RAM, 2.6 s per 69K); a `lmcache-server` restart wipes the RAM tier but tier 3 reloads it from disk at boot (23.8 s per 69K, graceful partial-hit); a machine reboot starts from tier 3 only. The L1 pool is **eagerly pinned** on some backend configurations (RSS == `--l1-size-gb` from boot) — size it against free RAM, not workload, and do not combine `--chunk-size 8192` with `--l1-size-gb 200` on a 404 GB host (OOM-kills the server).
- **Orphaned GPU workers**: `docker rm -f` on the engine can leak `VLLM::Worker` processes that hold ~65 GB per GPU; the next boot then dies with CUDA OOM at `set_device_index`. After any forced teardown, check `nvidia-smi --query-compute-apps=pid,used_memory` and `sudo kill -9` the strays.
- **Forced kill corrupts the SHM tier**: `docker rm -f` on the engine leaves stale slot/semaphore state in the named SHM segment the server owns; the next engine attaches, gets rejected, and spins in an LMCache transfer-context init loop (symptom: GPUs busy, zero tokens, `Auto-selected backend` log lines flooding). Graceful `docker stop`/`restart` is clean (verified). After any forced kill, remove stale `/dev/shm/mimo26*` segments **and recreate `lmcache-server` too** — tier 3 (disk) restores its contents at boot.
- **Speculative decoding is off, deliberately.** The MTP verify pass (any q_len>1 attention) hits a kernel cliff on this stack: decode at 67K context collapses from 56 (no spec) to 16.5 tok/s (k=2) / 17.5 (k=1) — a 3.4x penalty that dwarfs the acceptance win. The cliff starts by 5K context (42.5 tok/s at k=1). Spec only pays below ~2K context. Measured solo, steady-state; agents live at 5K–100K effective contexts, so production runs without spec until the verify kernel is fixed. **The kernel fix exists in this repo** (`patches/triton_unified_attention_diffkv.py`): mount it + `VLLM_DIFFKV_FULL_ATTN_SEGMENTS=64` + enable MTP — that takes 67K-context spec decode from 16.5 to **33.7 tok/s** (kernel-level 7-20× per verify call, `q=1` unaffected). Still loses to no-spec (56.1) on this rig — the remaining gap sits elsewhere in the verify path (SWA-layer verify, draft forward, PP hop overhead). Full analysis: [vllm#59054](https://github.com/vllm-project/vllm/issues/59054); upstream port: [vllm#59085](https://github.com/vllm-project/vllm/pull/59085) (with the sm80 measurements from this repo folded in).

- **Long-context chunked prefill hits the same cliff family (2026-09-30).** With agent contexts at 150–250K, every 1024-token chunk's new queries attend over the full history on the 2D kernel path (`BLOCK_M=16` = one query token per program, zero KV-dim parallelism): 620 ms per GA layer at 200K context on sm80 — ~5.6 s per chunk step across 9 GA layers, which stalls the whole mixed batch (decode riders included) to ~0.2 tok/s for minutes at a time. Prefix caching does not help: it skips recomputing history K/V but the new queries still scan the full KV. **Fix (same file): prefill q-tiling** — `BLOCK_M=128` packs 8 query tokens per program so each KV tile is read once instead of 8 times. Measured in-container on the 4×170HX rig, chunk q=1024 @200K prior KV, bf16: stock 624 ms → 210 ms (`BLOCK_M=128`, `TILE=64`, 8 warps) = **3.0×**; mixed batch (chunk + 4 giant-KV decodes) 637 → 309 ms; SWA-layer chunks 0.35 → 0.15 ms; decode-only and spec-verify shapes bit-identical. `VLLM_DIFFKV_PREFILL_TILE=64` is set by the launcher (sm80 has 164 KB smem; `BLOCK_M≥256`+`TILE=64` exceeds it and crashes).

  **End-to-end (production restart 2026-09-30):** a 201,913-token unique-context probe prefills in **74.7 s = 2,703 tok/s** (was ~170 tok/s projected on the stock kernel — a ~16× recovery to the healthy 75K-class rate); 67K-context decode 49.5 tok/s under concurrent agent load (no regression); tool calls, reasoning parse and the router-precision edge case all pass. Multi-minute 0.2 tok/s stalls are gone; agent context reloads now take 1–2 min each instead of ~20.

## Credits

- [wtdcode/vllm-backport](https://github.com/wtdcode/vllm-backport) — the sm80 fork this runs on
- [avtc/vllm-backport PR#106](https://github.com/wtdcode/vllm-backport/pull/106) — the official-checkpoint sharding, MTP-loader and dense-fp8 Marlin fixes ported here
- Upstream vLLM [#58235](https://github.com/vllm-project/vllm/pull/58235) — the ViT sink fix
- [LMCache PR#4410](https://github.com/LMCache/LMCache/pull/4410) — hybrid-model multi-group engine-driven transfers (the CPU/disk tiers run its head build); field data in [#4410](https://github.com/LMCache/LMCache/pull/4410#issuecomment-5922932290) and [#5440](https://github.com/LMCache/LMCache/issues/5440)
- Related upstream contributions from this work: [wtdcode#103](https://github.com/wtdcode/vllm-backport/pull/103), [wtdcode#104](https://github.com/wtdcode/vllm-backport/pull/104), [wtdcode#105](https://github.com/wtdcode/vllm-backport/issues/105), [wtdcode#110](https://github.com/wtdcode/vllm-backport/pull/110), [vllm#59054](https://github.com/vllm-project/vllm/issues/59054)

## 中文摘要

4×CMP 170HX（无 P2P、只能 PP4）上跑 MiMo-V2.6-Flash-RL 全模态的生产部署——**首个完全可用的版本**。现役配置 = 官方 161GB fp8 权重、16GiB fp8 KV（层切分 12,12,12,12）、投机解码关闭（长上下文内核悬崖，详见英文注意事项）、**三级 KV 存储**（GPU 213 万 + 内存层 870 万 + 磁盘层）：单流 greedy 64.5（短）/ 56（67K 上下文）tok/s、16 流聚合 302、200K 冷 prefill 2703 tok/s（q-tiling 内核修复后 ~16×）、**引擎重启零缓存损失（69K 前缀 2.6 秒取回）**。文/图/视/音四模态 + 工具调用全部可用。核心坑（官方 QKV 的 NB=4 导出布局、fp8 反量化路径 -73% decode、滑窗 KV 在途预留、ViT sink 色彩 bug、spec 验证/长上下文 prefill 两条内核悬崖）均已修复并沉淀为挂载补丁，测量数据同步回馈上游（vllm#59054/#59085、LMCache#4410/#5440）。完整工程日志见 `docs/RESULT.md`；NVFP4 回退轨保留在 `launch-omni.sh` / tag `v1.3-prod`。
