import json, time, urllib.request
BASE = "http://127.0.0.1:8099"

def go(msg, mt, tools=None):
    body = {"model": "mimo26", "messages": [{"role": "user", "content": msg}],
            "max_tokens": mt, "temperature": 1.0}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=300).read())
    return time.time() - t0, r

go("你好", 8)  # 预热
for i in range(3):
    dt, r = go("详细讲解流水线并行的气泡问题和微批次缓解手段，600字。", 700)
    u = r["usage"]
    print(f"单流#{i+1}: {u['completion_tokens']}tok / {dt:.1f}s = {u['completion_tokens']/dt:.1f} tok/s")

dt, r = go("北京天气？", 200, tools=[{"type": "function", "function":
        {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}])
tc = r["choices"][0]["message"].get("tool_calls")
print("工具调用:", json.dumps(tc[0]["function"], ensure_ascii=False) if tc else "FAIL")
