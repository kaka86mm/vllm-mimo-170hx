# vllm-mimo-170hx

Serve **XiaomiMiMo/MiMo-V2.6-Flash-RL** (309B MoE, 15B active, omni-modal, 1M context) on a cluster of **4× NVIDIA CMP 170HX** (GA100 sm_80, 64 GB unlocked each, PCIe 2.0 x4, **no P2P — pipeline parallel only**) using [wtdcode/vllm-backport](https://github.com/wtdcode/vllm-backport).

Everything here was measured on real hardware over one deployment session (Sep 2026). The scripts are runnable as-is on a matching box; the numbers below are the acceptance results, not aspirations.

## TL;DR

| Metric | Result |
|---|---|
| Weights | `XiaomiMiMo/MiMo-V2.6-Flash-RL` **official fp8 checkpoint** (161 GB) — unlocked by this repo's port of avtc's [PR#106](https://github.com/wtdcode/vllm-backport/pull/106) dense-fp8 Marlin fix, official qkv sharder and MTP loader |
| Launcher | `scripts/launch-official.sh` |
| Topology | PP=4, layer partition `11,13,12,12`, fp8_e4m3 KV (15 GiB), MTP 3-token spec decode, OPROJ_FP8 on |
| Single-stream decode | **100.6–104.0 tok/s** warm |
| 4-stream aggregate | 189 tok/s |
| KV pool | **1,994,875 tokens** (1.90× a 1M-token request — two full 1M contexts at once) |
| Modalities | text ✓ image ✓ video ✓ audio ✓ (accurate sine-wave & testsrc descriptions) |
| Tool calls / reasoning parser | ✓ (`mimo` parsers) |

The NVFP4 transcode track (ProCreations, 194 GB; 93 tok/s, KV 1,329,835) is kept as a fallback: `scripts/launch-omni.sh`, or `git checkout v1.3-prod` for the exact frozen state.

## The KV pool story (why this repo exists)

Out of the box the KV pool was 447K tokens — ~89 KB/token for a model whose GA layers only need ~11.5 KB/token. Root cause (found by instrumenting `get_max_concurrency_for_kv_cache_config`, patch in `patches/`):

> Each sliding-window group reserves `window + max_in_flight_tokens` per request, and `max_in_flight_tokens = max_concurrent_batches × max_num_batched_tokens` **scales with pipeline depth** (`max_concurrent_batches == pp_size`). With PP=4 and mnbt=8192 that is 40,960 in-flight tokens against a 128-token window — 99.7% of the reservation. The hybrid KV manager works; the capacity is pipeline-bound, not window-bound.

Fix: shrink `--max-num-batched-tokens`. The whole curve improved monotonically:

| mnbt | KV pool (tokens) | 32K prefill | single-stream decode |
|------|------------------|-------------|----------------------|
| 8192 | 447,350 | 9.7 s | 73–75 tok/s |
| 4096 | 780,571 | 8.3 s | 75 tok/s |
| 2048 | 998,304 | 7.9 s | 75 tok/s |
| **1024** | **1,160,104** | **7.7 s** | **75 tok/s** |

Smaller chunks also *reduce* head-of-line blocking on the pipeline, so prefill got faster at every step. The real fix (per-stage sliding-window recycling) is proposed upstream in [wtdcode/vllm-backport#105](https://github.com/wtdcode/vllm-backport/issues/105).

## Deployment

**Step-by-step guide: [DEPLOY.md](DEPLOY.md) — zero to serving, with verification checkpoints and a troubleshooting table of every error actually hit during bring-up.**

## Layout

- `scripts/launch-official.sh` — **the production config** (all TL;DR numbers come from it)
- `scripts/launch-omni.sh` — NVFP4 fallback (identical apart from MODEL_DIR, KV size and the three official-only mounts)
- `scripts/launch-omni-lmc.sh` — LMCache CPU-offload variant (works, archived; see docs)
- `scripts/bench-full.py` — decode (1/2/4/8/16 streams) + prefill (8K/32K) suite with TTFT
- `scripts/omni-test*.py`, `video-diag.py` — multimodal verification (needs `ffmpeg` inside the image: `testsrc` + `sine` lavfi inputs make good probes)
- `scripts/kv-stress2.py` — 4×191K unique-content long-context admission stress
- `scripts/patch-kvcap.py` — the capacity probe that found the root cause
- `docs/RESULT.md` — full runbook: pitfalls, tuning decisions, LMCache postmortem, restore procedures

## Findings worth reusing

1. **Weights choice**: the official checkpoint ships attention fused-QKV **TP=4-interleaved** (NB=4 export super-groups); the stock backport loader mishandles it at TP=1 (`_shard_fp8_qkv_proj`, 1856 vs 1792 rows). This repo fixes the loader (finding #9); the ProCreations NVFP4 transcode (already de-interleaved) remains the zero-surgery fallback via `launch-omni.sh`.
2. **DFlash does not work under PP** in current builds: the drafter consumes 5 target-layer hidden states (EAGLE3 interface) and PP boundaries only carry the final hidden state. MTP is the PP answer.
3. **`--kv-cache-memory`**: `human_readable_int` rejected `11GiB` (upstream PR [#103](https://github.com/wtdcode/vllm-backport/pull/103)); and 11 GiB is too aggressive anyway — 10 GiB keeps long-prefill activation headroom (11 GiB → OOM 500s under 4×191K).
4. **PP layer partition**: weigh per-rank *total* load (encoders live on PP0, lm_head + drafter on the last rank). `VLLM_PP_LAYER_PARTITION=11,13,12,12` was the measured balance.
5. **Multimodal + thinking**: requests with mm content should set `"chat_template_kwargs": {"enable_thinking": false}` — the `mimo` reasoning parser intermittently swallows content under thinking mode with mm inputs.
6. **Audio input**: use `audio_url` + wav (not OpenAI's `input_audio` format).
7. **LMCache MP on CMP-unlocked drivers**: AUTO transfer mode walks GPU-IPC (`cudaErrorMapBufferObjectFailed`); the working combo is a CPU-only `lmcache server` + `--supported-transfer-mode engine_driven` + `"lmcache.mp.mp_transfer_mode":"engine_driven"` in the connector extra config + `--prefix-cache-retention-interval <chunk>` for hybrid models. But over PCIe 2.0 x4, retrieval (222 tok/s under load) loses to recomputation (19K tok/s) until SWA storage is fixed — capacity feature, not a latency feature.
8. **fp8 o_proj (OPROJ_FP8)** — ported from [wtdcode/vllm-backport#106](https://github.com/wtdcode/vllm-backport/pull/106) (avtc's commit `ced6985f`): the checkpoint's dense-bf16 `o_proj` is quantized to per-tensor fp8 at load and runs Marlin **W8A16** on sm80 (the fork's online-fp8 path needs an explicit `force_kernel` route there — see `patches/online_fp8.py`). Decode GEMM is memory-bound in PP, so halving the weight read wins: **75 → 93 tok/s single-stream (+24%)** on NVFP4, warm, KV pool unchanged, quality smoke clean (incl. the 9.11 vs 9.9 trap). Lossy but quality-neutral (upstream GSM8K parity 82.7 vs 82.0). Off-switch: remove `VLLM_MIMO_OPROJ_FP8=1` — the patched files are byte-identical to stock with the env unset. Same port ships an **audio-tower lazy-load gate** (`patches/mimo_v2_omni_model.py`): with `--limit-mm-per-prompt '{"audio": 0}'` the 1.9 GB tower is never built; we keep audio enabled by default since it's a verified modality.
9. **Official-weights unlock (v2.0)** — the official checkpoint served at **-73% decode** on the stock image, and the cause was *not* the loader: `MarlinFP8ScaledMMLinearKernel.can_implement` returns True unconditionally and natively handles the 128×128 block scales, but on sm80 the natural kernel selection landed on the torch dequant fallback, so every dense GEMM decoded its weights at runtime. Forcing MarlinFP8 for serialized fp8 (`patches/fp8.py`, same one-line `force_kernel` pattern as the online path) plus avtc's rewritten `_shard_fp8_qkv_proj` (NB=4 layout, both observed scale layouts) and MTP-loader fix (fp8 weight/scale paired sharding) turn the 161 GB checkpoint into the *fastest* config here: **104 tok/s, +50% KV pool** (the checkpoint is 33 GB smaller than the NVFP4 transcode — that difference is what buys the 15 GiB pool). All three patches are dormant on NVFP4, so the NVFP4 fallback launcher keeps working unchanged. KV sizing note: 16 GiB left <3 MB headroom at cudagraph capture on the drafter rank (thousands of boot-phase allocator retries); 15 GiB is the ceiling on this box. Also patch the official `generation_config.json` (`max_new_tokens: 2048` → 65536) — same hidden cap as the transcode.

## Related upstream contributions

- [#103](https://github.com/wtdcode/vllm-backport/pull/103) — accept IEC suffixes in `human_readable_int`
- [#104](https://github.com/wtdcode/vllm-backport/pull/104) — warn when sliding-window capacity is pipeline-bound
- [#105](https://github.com/wtdcode/vllm-backport/issues/105) — RFC: per-stage sliding-window recycling

## 中文摘要

4×CMP 170HX（无 P2P、只能 PP）上跑 MiMo-V2.6-Flash-RL 全模态的完整部署与调优记录，现役生产 = **官方 161GB fp8 权重**。核心成果：KV 池从 44.7 万 token 一路扩到 **199 万（1.9× 百万上下文）**——先是修滑窗在途预留随流水线膨胀的 bug（mnbt 压到 1024），再靠官方权重比 NVFP4 转码省的 33GB 把池子提到 15GiB。单流 decode 75 → 93（OPROJ fp8 o_proj）→ **104 tok/s（离线 fp8 强制 Marlin）**——官方权重 -73% 慢速路径的根因是 sm80 内核选择落在 torch 反量化，而非 loader。文/图/视/音四模态全部可用。所有坑（NB=4 交错布局、DFlash×PP 结构性缺失、LMCache 排雷、层切分配平、cudagraph 捕获余量红线）都写在 `docs/RESULT.md`；NVFP4 回退轨保留在 `launch-omni.sh` / tag `v1.3-prod`。
