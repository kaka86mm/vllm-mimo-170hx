import json, time, urllib.request
BASE = "http://127.0.0.1:8099"
def chat(msg, max_tokens, stream=False):
    req = urllib.request.Request(BASE + "/v1/chat/completions",
        json.dumps({"model":"mimo26","messages":[{"role":"user","content":msg}],
                    "max_tokens":max_tokens,"temperature":1.0,"top_p":0.95,"stream":stream}).encode(),
        {"Content-Type":"application/json"})
    return urllib.request.urlopen(req, timeout=600)

# 短请求: 总延迟
t0=time.time(); r=json.loads(chat("用一句话解释流水线并行。",200).read()); dt=time.time()-t0
u=r["usage"]; c=u["completion_tokens"]
print(f"[短] 总耗时 {dt:.1f}s | 输出 {c} tok | 端到端 {c/dt:.1f} tok/s")

# 长生成: 稳态 decode 速度 (流式首 token + 后续)
t0=time.time(); resp=chat("详细讲解 PCIe 流水线并行中的气泡问题，以及微批次如何缓解它。写600字以上。",800,stream=True)
first=None; n=0; buf=b""
for chunk in resp:
    buf+=chunk
    if first is None and b"\"delta\":{}" not in chunk and b"content" in chunk:
        first=time.time()-t0
    if chunk==b"\n\n": n+=1
total=time.time()-t0
print(f"[长/流式] TTFT {first:.1f}s | chunks {n} | 总 {total:.1f}s")

# 非流式长生成核对 token 数
t0=time.time(); r=json.loads(chat("从零解释 MoE 路由的负载不均问题与辅助损失，800字以上。",700).read()); dt=time.time()-t0
u=r["usage"]; c=u["completion_tokens"]
print(f"[长/非流式] 输出 {c} tok | 端到端 {c/dt:.1f} tok/s (含生成全程)")
