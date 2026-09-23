import json, threading, time, urllib.request
import os
BASE = os.environ.get("BASE", f"http://127.0.0.1:{os.environ.get('PORT', '8099')}")

def req_stream(content, max_tokens, out, idx):
    t0 = time.time(); first = None; ntok = 0
    body = {"model": "mimo26", "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens, "temperature": 1.0, "stream": True,
            "stream_options": {"include_usage": True}}
    r = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(),
                               {"Content-Type": "application/json"})
    try:
        buf = b""
        for chunk in urllib.request.urlopen(r, timeout=600):
            buf += chunk
            if first is None and b'"delta":{"content"' in buf.replace(b" ", b""):
                first = time.time() - t0
            if b'"usage"' in chunk:
                for line in chunk.decode(errors="ignore").split("\n"):
                    if line.startswith("data: {") and '"usage"' in line and '"completion_tokens"' in line:
                        try:
                            j = json.loads(line[6:])
                            if j.get("usage") and j["usage"].get("completion_tokens"):
                                ntok = j["usage"]["completion_tokens"]
                        except Exception:
                            pass
        out[idx] = (first, time.time() - t0, ntok)
    except Exception as e:
        out[idx] = (first, time.time() - t0, 0)

FILLER = "大模型推理系统的性能取决于显存带宽、算力和互联拓扑三者的平衡。" * 420

def bench(name, nstreams, prompt, mtoks):
    outs = [None] * nstreams
    th = [threading.Thread(target=req_stream, args=(prompt, mtoks, outs, i)) for i in range(nstreams)]
    t0 = time.time()
    for t in th: t.start()
    for t in th: t.join()
    wall = time.time() - t0
    tot = sum(o[2] for o in outs)
    ttfts = [o[0] for o in outs if o[0] is not None]
    line = f"[{name}] 并发{nstreams} | 墙钟{wall:.1f}s | 总{tot} tok | 聚合{tot/wall:.1f} tok/s"
    if ttfts:
        line += f" | TTFT均值{sum(ttfts)/len(ttfts):.2f}s"
    print(line, flush=True)

print("=== decode 单流 ===", flush=True)
bench("单流700", 1, "详细讲解 PCIe 流水线并行中的气泡问题，以及微批次如何缓解。600字以上。", 700)
print("=== decode 多流 ===", flush=True)
for n in [2, 4, 8, 16]:
    bench(f"{n}流300", n, "写一段关于推理引擎调度器的技术分析，250字左右。", 300)
print("=== prefill ===", flush=True)
bench("8K预填充", 1, FILLER + "\n\n总结上文要点，三条。", 64)
bench("32K预填充", 1, FILLER * 4 + "\n\n总结上文要点，三条。", 64)
