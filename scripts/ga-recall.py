import json, os, time, urllib.request

BASE = os.environ.get("BASE", f"http://127.0.0.1:{os.environ.get('PORT', '8099')}")
FILLER_UNIT = "推理引擎的调度器需要在延迟与吞吐之间权衡，这是每一代系统都要重新回答的问题。"

# ~100K token 上下文：开头埋密码，结尾提问。GA 层若被错误滑窗化（128 窗口），必然答不出。
needle = "【重要】系统密码是 紫色菠萝7788，请记住。"
ctx = needle + ("\n" + FILLER_UNIT * 30 + "\n") * 160
body = {"model": "mimo26",
        "messages": [{"role": "user", "content": ctx + "\n\n上文开头的系统密码是什么？只回答密码本身。"}],
        "max_tokens": 60, "temperature": 0.2,
        "chat_template_kwargs": {"enable_thinking": False}}
r = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(),
                           {"Content-Type": "application/json"})
t0 = time.time()
d = json.loads(urllib.request.urlopen(r, timeout=900).read())
print(f"prompt_tok={d['usage']['prompt_tokens']} 耗时{time.time()-t0:.0f}s")
print("回答:", repr((d["choices"][0]["message"].get("content") or "")[:120]))
print("判定:", "PASS - GA 层全注意力工作正常" if "紫色菠萝7788" in (d["choices"][0]["message"].get("content") or "") else "FAIL - 长程记忆丢失!")
