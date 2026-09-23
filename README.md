# vllm-mimo-170hx

Serve **XiaomiMiMo/MiMo-V2.6-Flash-RL** (309B MoE, 15B active, omni-modal, 1M context) on a cluster of **4× NVIDIA CMP 170HX** (GA100 sm_80, 64 GB unlocked each, PCIe 2.0 x4, **no P2P — pipeline parallel only**) using [wtdcode/vllm-backport](https://github.com/wtdcode/vllm-backport).

Everything here was measured on real hardware over one deployment session (Sep 2026). The scripts are runnable as-is on a matching box; the numbers below are the acceptance results, not aspirations.

## TL;DR

| Metric | Result |
|---|---|
| Weights | `ProCreations/MiMo-V2.6-Flash-RL-NVFP4` (value-identical NVFP4 transcode, arch switched to `MiMoV2OmniForCausalLM`) |
| Topology | PP=4, layer partition `11,13,12,12`, fp8_e4m3 KV, MTP 3-token spec decode |
| Single-stream decode | 69–75 tok/s (MTP acceptance ~0.57–1.0 per position) |
| 16-stream aggregate | 290 tok/s |
| Prefill | 8K: 2.6 s · 32K: 7.8 s (~4.2K tok/s) |
| KV pool | **1,160,104 tokens** (1.2× a 1,048,576-token request; peak 20.1% under 4×191K concurrent) |
| Modalities | text ✓ image ✓ video ✓ audio ✓ (accurate sine-wave & testsrc descriptions) |
| Tool calls / reasoning parser | ✓ (`mimo` parsers) |

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

- `scripts/launch-omni.sh` — the final config (the one all numbers above come from)
- `scripts/launch-omni-lmc.sh` — LMCache CPU-offload variant (works, archived; see docs)
- `scripts/bench-full.py` — decode (1/2/4/8/16 streams) + prefill (8K/32K) suite with TTFT
- `scripts/omni-test*.py`, `video-diag.py` — multimodal verification (needs `ffmpeg` inside the image: `testsrc` + `sine` lavfi inputs make good probes)
- `scripts/kv-stress2.py` — 4×191K unique-content long-context admission stress
- `scripts/patch-kvcap.py` — the capacity probe that found the root cause
- `docs/RESULT.md` — full runbook: pitfalls, tuning decisions, LMCache postmortem, restore procedures

## Findings worth reusing

1. **Weights choice**: the official checkpoint ships attention fused-QKV **TP=4-interleaved**; the backport loader mishandles it at TP=1 (`_shard_fp8_qkv_proj`, 1856 vs 1792 rows). The ProCreations transcode is already de-interleaved — use it and skip the mine.
2. **DFlash does not work under PP** in current builds: the drafter consumes 5 target-layer hidden states (EAGLE3 interface) and PP boundaries only carry the final hidden state. MTP is the PP answer.
3. **`--kv-cache-memory`**: `human_readable_int` rejected `11GiB` (upstream PR [#103](https://github.com/wtdcode/vllm-backport/pull/103)); and 11 GiB is too aggressive anyway — 10 GiB keeps long-prefill activation headroom (11 GiB → OOM 500s under 4×191K).
4. **PP layer partition**: weigh per-rank *total* load (encoders live on PP0, lm_head + drafter on the last rank). `VLLM_PP_LAYER_PARTITION=11,13,12,12` was the measured balance.
5. **Multimodal + thinking**: requests with mm content should set `"chat_template_kwargs": {"enable_thinking": false}` — the `mimo` reasoning parser intermittently swallows content under thinking mode with mm inputs.
6. **Audio input**: use `audio_url` + wav (not OpenAI's `input_audio` format).
7. **LMCache MP on CMP-unlocked drivers**: AUTO transfer mode walks GPU-IPC (`cudaErrorMapBufferObjectFailed`); the working combo is a CPU-only `lmcache server` + `--supported-transfer-mode engine_driven` + `"lmcache.mp.mp_transfer_mode":"engine_driven"` in the connector extra config + `--prefix-cache-retention-interval <chunk>` for hybrid models. But over PCIe 2.0 x4, retrieval (222 tok/s under load) loses to recomputation (19K tok/s) until SWA storage is fixed — capacity feature, not a latency feature.

## Related upstream contributions

- [#103](https://github.com/wtdcode/vllm-backport/pull/103) — accept IEC suffixes in `human_readable_int`
- [#104](https://github.com/wtdcode/vllm-backport/pull/104) — warn when sliding-window capacity is pipeline-bound
- [#105](https://github.com/wtdcode/vllm-backport/issues/105) — RFC: per-stage sliding-window recycling

## 中文摘要

4×CMP 170HX（无 P2P、只能 PP）上跑 MiMo-V2.6-Flash-RL 全模态的完整部署与调优记录。核心成果：KV 池从 44.7 万 token 扩到 116 万（根因是滑窗组的在途预留随流水线深度膨胀，修法是把 mnbt 压到 1024，prefill 反而更快）；单流 75 tok/s、16 流 290 tok/s；文/图/视/音四模态全部可用。所有坑（官方权重 TP4 交错 loader bug、DFlash×PP 结构性缺失、LMCache 三种传输模式的排雷、层切分按 rank 实测权重配平）都写在 `docs/RESULT.md`。
