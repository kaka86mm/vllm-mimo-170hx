import json, threading, time, urllib.request

BASE = "http://127.0.0.1:8099"
FILLER = "大模型推理系统的性能取决于显存带宽、算力和互联拓扑三者的平衡。" * 420  # ~8.4K tok

def req(idx, out):
    body = {"model": "mimo26",
            "messages": [{"role": "user", "content": FILLER * 15 + "\n\n用一句话总结。"}],
            "max_tokens": 24, "temperature": 0.2}
    r = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(),
                               {"Content-Type": "application/json"})
    t0 = time.time()
    try:
        d = json.loads(urllib.request.urlopen(r, timeout=900).read())
        out[idx] = f"req{idx}: {time.time()-t0:.1f}s finish={d['choices'][0]['finish_reason']} prompt_tok={d['usage']['prompt_tokens']}"
    except Exception as e:
        out[idx] = f"req{idx}: FAIL {str(e)[:120]}"

out = [None] * 4
ths = [threading.Thread(target=req, args=(i, out)) for i in range(4)]
t0 = time.time()
for t in ths: t.start()
for t in ths: t.join()
print(f"4路126K并发 总墙钟 {time.time()-t0:.0f}s")
for line in out: print(" ", line)
