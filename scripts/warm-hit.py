import json, random, string, time, urllib.request

BASE = "http://127.0.0.1:8099"
BASE_SENT = "大模型推理系统的性能取决于显存带宽、算力和互联拓扑三者的平衡。"

def uniq_filler(seed, reps):
    rnd = random.Random(seed)
    return "".join(BASE_SENT + "".join(rnd.choices(string.digits, k=8)) for _ in range(reps))

body = {"model": "mimo26",
        "messages": [{"role": "user", "content": uniq_filler(13, 6600) + "\n\n用一句话总结。"}],
        "max_tokens": 24, "temperature": 0.2}
r = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(),
                           {"Content-Type": "application/json"})
t0 = time.time()
d = json.loads(urllib.request.urlopen(r, timeout=1800).read())
dt = time.time() - t0
u = d["usage"]
print(f"热命中 126K: 总耗时 {dt:.1f}s | prompt {u['prompt_tokens']} tok | 有效 prefill {u['prompt_tokens']/dt:.0f} tok/s")
print("回答:", (d["choices"][0]["message"].get("content") or "")[:100])
