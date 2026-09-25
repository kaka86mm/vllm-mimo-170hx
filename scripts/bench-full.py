"""Benchmark suite: decode concurrency curve + prefill at real token counts.

All numbers reported with actual `usage.prompt_tokens` / `completion_tokens`
(no assumed labels), unique-content prompts (prefix-cache cold), thinking off.
Run against a live server:  python3 scripts/bench-full.py [BASE_URL]
"""
import json, random, sys, threading, time, urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8099"
URL = BASE.rstrip("/") + "/v1/chat/completions"

WORDS = ["显卡", "带宽", "算子", "调度", "缓存", "拓扑", "量化", "流水线", "显存碎片",
         "内核融合", "批处理", "注意力", "稀疏性", "蒸馏", "集群", "推理引擎", "词元",
         "向量", "梯度", "预热"]


def filler(n_words: int) -> str:
    random.seed()
    return " ".join(random.choice(WORDS) for _ in range(n_words))


def req_stream(content, max_tokens, out, idx):
    t0 = time.time(); first = None; ntok = 0
    body = {"model": "mimo26", "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens, "temperature": 1.0, "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    r = urllib.request.Request(URL, json.dumps(body).encode(),
                               {"Content-Type": "application/json"})
    try:
        buf = b""
        for chunk in urllib.request.urlopen(r, timeout=600):
            buf += chunk
            if first is None and b'"delta":{"content"' in buf.replace(b" ", b""):
                first = time.time() - t0
            if b'"usage"' in chunk:
                for line in chunk.decode(errors="ignore").split("\n"):
                    if line.startswith("data: {") and '"completion_tokens"' in line:
                        try:
                            j = json.loads(line[6:])
                            if j.get("usage", {}).get("completion_tokens"):
                                ntok = j["usage"]["completion_tokens"]
                        except Exception:
                            pass
        out[idx] = (first, time.time() - t0, ntok)
    except Exception:
        out[idx] = (first, time.time() - t0, 0)


def bench_decode(nstreams, mtoks=300, warm=True):
    prompt = "写一段关于推理引擎调度器的技术分析，250字左右。"
    def once():
        outs = [None] * nstreams
        th = [threading.Thread(target=req_stream, args=(prompt, mtoks, outs, i))
              for i in range(nstreams)]
        t0 = time.time()
        for t in th: t.start()
        for t in th: t.join()
        wall = time.time() - t0
        tot = sum(o[2] for o in outs)
        ttfts = [o[0] for o in outs if o[0] is not None]
        ttft = f" TTFT {sum(ttfts)/len(ttfts):.2f}s" if ttfts else ""
        print(f"decode {nstreams:>2} 流 | 聚合 {tot/wall:6.1f} tok/s | "
              f"总 {tot} tok / {wall:.1f}s{ttft}", flush=True)
    if warm: once()  # warm-up run, then the reported one
    once()


def bench_prefill(n_words, label):
    body = {"model": "mimo26",
            "messages": [{"role": "user", "content": filler(n_words) + "\n\n用一个词概括上文主题。"}],
            "max_tokens": 16, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time()
    j = json.load(urllib.request.urlopen(urllib.request.Request(
        URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
    wall = time.time() - t0
    pt = j["usage"]["prompt_tokens"]
    print(f"prefill {label} | 实测 {pt} tok | 墙钟 {wall:.2f}s | ≈{pt/wall:.0f} tok/s", flush=True)


if __name__ == "__main__":
    print("=== decode（temp 1.0, thinking off, 暖态报告值）===", flush=True)
    for n in [1, 2, 4, 8, 16]:
        bench_decode(n, 700 if n == 1 else 300)
    print("=== prefill（唯一内容，前缀缓存冷）===", flush=True)
    bench_prefill(5200, "~19K")   # 5200 words ≈ 18.7K tokens
    bench_prefill(5200, "~19K")
    bench_prefill(20800, "~75K")  # 20800 words ≈ 75K tokens
    bench_prefill(20800, "~75K")
