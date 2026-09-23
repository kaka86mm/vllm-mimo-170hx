# Deploy: MiMo-V2.6-Flash-RL (omni) on 4× CMP 170HX

Zero-to-serving guide. Every command below was executed on the reference box
(4× CMP 170HX 64 GB unlocked, PP-only PCIe 2.0 x4, driver 610.57.04); every
error in the troubleshooting table was actually hit and resolved during the
deployment session.

## 0. Prerequisites

| Item | Requirement | Why |
|---|---|---|
| GPUs | 4× 64 GB sm_80-class (A100 also works), **no P2P required** | Weights ~194 GB on disk, ~183 GB in VRAM (fp8 KV + MTP drafter included). TP is impossible without P2P — everything here is PP=4. |
| Driver | ≥ 570 (we used 610.57.04 with CMPUnlocker 64 GB unlock) | sm_80 FP4-Marlin path needs current CUDA userspace in the image, not host toolkit |
| Docker | with `nvidia` runtime (`--runtime=nvidia`) | |
| Disk | ≥ 350 GB free on the model path | weights 194 GB + image 30 GB + headroom |
| RAM | ≥ 64 GB (we had 373 GB) | worker host memory; more if you use the LMCache variant |
| Power | cap cards at ≤ 230 W (`sudo nvidia-smi -pl 230`) | 250 W transient spikes → Xid 43 on CMP 170HX |

## 1. Clone and configure

```bash
git clone https://github.com/kaka86mm/vllm-mimo-170hx.git
cd vllm-mimo-170hx/scripts
cp env.example env.sh
$EDITOR env.sh        # MODEL_DIR / PORT / GPUS / HF_ENDPOINT
```

## 2. Prepare the download venv (one-time)

The box's system Python may lack `huggingface_hub`. Python 3.14 has no
`hf_transfer` wheel — plain `hf download` with `--max-workers` is fast enough
(~45 MB/s via mirror ⇒ ~75 min).

```bash
python3 -m venv ~/hfenv
~/hfenv/bin/pip install -U huggingface_hub \
  -i https://pypi.tuna.tsinghua.edu.cn/simple   # or your index
```

## 3. Download weights

```bash
./download-weights.sh
```

This does three things, in order:
1. Downloads `ProCreations/MiMo-V2.6-Flash-RL-NVFP4` **excluding `gguf/`** —
   the repo carries a ~183 GB llama.cpp tree that vLLM never reads and that
   filled our root disk on the first attempt.
2. Switches `config.json` architectures to `MiMoV2OmniForCausalLM`
   (the transcode ships with the text-only class; without the switch you get
   no vision/audio inputs).
3. Verifies all 65 shards + key small files.

**Do not use the official `XiaomiMiMo/MiMo-V2.6-Flash-RL` weights** with this
stack: their fused-QKV attention tensors are TP=4-interleaved and the
backport's `_shard_fp8_qkv_proj` fails at TP=1 (`tensor a (1856) vs b (1792)`).
The ProCreations transcode is value-identical with attention already in
global order.

Expected tail:

```
  65/65 分片就位, 合计 194.0GB (预期 ~194GB)
  校验通过
```

## 4. Pull the image

```bash
docker pull docker.m.daocloud.io/lazymio/vllm-backport:v0.13.0-sm80
# outside CN: docker pull lazymio/vllm-backport:v0.13.0-sm80
```

Note: the image **ENTRYPOINT is `["vllm serve"]`** — container args must NOT
repeat `vllm serve` (classic `unrecognized arguments: serve /model` mistake;
the launch script handles this).

## 5. Launch

```bash
./launch-omni.sh
```

Boot takes ~7 min (weight load ~2 min, torch.compile + cudagraph capture,
MTP drafter). Success looks like:

```
(Worker_PPx) Model loading took 4x.xx GiB memory ...
(EngineCore) GPU KV cache size: 1,160,104 tokens, Maximum concurrency for 262,144 tokens per request: 4.43x
(APIServer) Application startup complete.
```

```bash
curl http://127.0.0.1:8099/health   # -> 200
```

The config in one line: PP=4 with `VLLM_PP_LAYER_PARTITION=11,13,12,12`
(PP0 hosts encoders, last rank hosts lm_head + MTP drafter), fp8_e4m3 KV +
block 128, `--max-num-batched-tokens 1024` (the KV-pool lever — see README),
`--kv-cache-memory 10737418240` (raw bytes; `11GiB` is rejected by this
build and 11 GiB OOMs anyway), util 0.94 (0.95+ risks cudagraph-capture
OOM), MTP 3-token spec decode.

## 6. Verify

```bash
./make-test-assets.sh          # ffmpeg lavfi testsrc video + 440 Hz wav + png, via the image
python3 final-check.py         # 3× single-stream decode + tool call
python3 omni-test2.py          # audio + video
python3 video-diag.py          # text baseline + video (default & enable_thinking=false)
```

Expected: decode ≈ 70 tok/s per stream; tool call returns
`{"name":"get_weather","arguments":"{\"city\":\"北京\"}"}`; the video answer
describes the testsrc countdown digits (8→5→2) and colors; the audio answer
identifies a continuous high-frequency beep.

**Video frame budget**: default sampling pins any video at ~1.4K prompt
tokens (~16 effective frames) regardless of duration - fine for "what is
this video about", too sparse for per-second detail. The knob is
`--media-io-kwargs '{"video": {"num_frames": 128}}'` (~5.6K tokens,
empowered frame math: frames / temporal-group-2 x ~88 tok/frame; NOT
`--mm-processor-kwargs`, which this processor ignores). Note the model's
duration estimate is a frames-x-assumed-fps heuristic, not ground truth.

Multimodal requests should carry `"chat_template_kwargs": {"enable_thinking": false}` —
under thinking mode the `mimo` reasoning parser intermittently returns empty
content with mm inputs. Audio input uses `{"type":"audio_url","audio_url":{"url":"data:audio/wav;base64,..."}}`
(not OpenAI's `input_audio` shape, which 400s).

## 7. Benchmark

```bash
python3 bench-full.py          # decode 1/2/4/8/16 streams + prefill 8K/32K
python3 kv-stress2.py          # 4×191K unique-content admission stress (~10 min)
```

Reference numbers (acceptance run, mnbt 1024):

| Scenario | Result |
|---|---|
| single stream | 69–75 tok/s, TTFT 0.63 s |
| 4 / 8 / 16 streams aggregate | 151 / 213 / 290 tok/s |
| 8K / 32K prefill | 2.6 s / 7.8 s |
| 4×191K concurrent | all admitted, peak KV 20.1%, 0 preemptions |

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `unrecognized arguments: serve /model` | image ENTRYPOINT is already `vllm serve` | pass args without the `vllm serve` prefix |
| disk fills mid-download (~+183 GB surprise) | repo's hidden `gguf/` tree | `--exclude "gguf/*"` (script does this) |
| `tensor a (1856) ... b (1792)` at load | official checkpoint's TP=4-interleaved fp8 QKV | use the ProCreations transcode |
| `ModuleNotFoundError: xxhash` | `--prefix-caching-hash-algo xxhash` not in image | drop the flag (default hash is fine) |
| boot OOM / allocator spam on one rank | util ≥ 0.95, or `--kv-cache-memory` ≥ 11 GiB | util 0.94 + 10737418240 bytes; rebalance `VLLM_PP_LAYER_PARTITION` |
| HTTP 500 on long concurrent prefills at 11 GiB KV | activation headroom gone | 10 GiB is the ceiling on 64 GB ranks |
| KV pool only ~450K tokens | `max_num_batched_tokens` too high (pipeline-bound SWA reservation) | `--max-num-batched-tokens 1024`; see [RFC](https://github.com/wtdcode/vllm-backport/issues/105) |
| `Model does not support EAGLE3 interface` | DFlash drafter under PP | use `"method":"mtp"` (DFlash needs all target layers local, i.e. TP) |
| empty `content` with image/video input | thinking-mode + `mimo` reasoning parser | `enable_thinking:false` in `chat_template_kwargs` |
| HTTP 400 on audio input | OpenAI `input_audio` shape | `audio_url` + wav payload |
| engine hangs after LMCache connector init | transfer-mode mismatch (`cudaErrorMapBufferObjectFailed` / missing engine-driven handler) | see `launch-omni-lmc.sh` header for the working combo; simplest: don't enable LMCache until SWA storage is fixed |

## 9. Ops

- Container is `--restart unless-stopped` — it self-heals; `./launch-omni.sh` for config changes.
- To restore whatever ran before: `docker stop mimo26 && docker start <your-old-container>`.
- 230 W power cap is a box property (`nvidia-smi -pl`), not per-container.
