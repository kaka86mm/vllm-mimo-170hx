import json, random, time, urllib.request

URL = "http://127.0.0.1:8099/v1/chat/completions"
WORDS = ["显卡","带宽","算子","调度","缓存","拓扑","量化","流水线","碎片","融合",
         "批处理","注意力","稀疏性","蒸馏","集群","引擎","词元","向量","梯度","预热"]

def filler(n_words):
    random.seed()
    return " ".join(random.choice(WORDS) for _ in range(n_words))

def bench_ctx(n_words, label):
    prompt = filler(n_words) + "\n\n现在从1数到250，只输出数字，用空格分隔，不要思考。"
    body = {"model": "mimo26", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 260, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time()
    times = []
    pt = None
    r = urllib.request.urlopen(urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}), timeout=1200)
    for raw in r:
        line = raw.decode("utf-8", "ignore").strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        try:
            j = json.loads(line[6:])
        except Exception:
            continue
        if j.get("usage") and j["usage"].get("prompt_tokens"):
            pt = j["usage"]["prompt_tokens"]
        ch = (j.get("choices") or [{}])[0]
        d = ch.get("delta") or {}
        if d.get("content"):
            times.append(time.time())
    if len(times) <= 60:
        print(f"[{label}] INSUFFICIENT tokens={len(times)} pt={pt}", flush=True)
        return
    ttft = times[0] - t0
    seg = times[40:]                      # drop warmup
    dur = seg[-1] - seg[0]
    tps = (len(seg) - 1) / dur if dur > 0 else 0
    print(f"[{label}] prompt={pt} tok | TTFT {ttft:.1f}s | decode稳态 {tps:.1f} tok/s", flush=True)

import sys
ctx_words = int(sys.argv[1]) if len(sys.argv) > 1 else 18600
bench_ctx(ctx_words, f"~{ctx_words} words context")
