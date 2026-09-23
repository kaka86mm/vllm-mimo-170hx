#!/bin/bash
# 冒烟测试：等健康检查 -> 计时 chat -> 简单吞吐 -> 工具调用格式
PORT=${1:-8099}
BASE=http://127.0.0.1:$PORT

echo "== 等待服务健康 (最长 20 分钟，首次加载 194GB 需时间) =="
for i in $(seq 1 120); do
  if curl -s --max-time 3 "$BASE/health" >/dev/null 2>&1; then echo "healthy!"; break; fi
  if ! docker ps --format '{{.Names}}' | grep -q mimo26; then echo "容器挂了:"; docker logs --tail 30 mimo26; exit 1; fi
  [ $((i % 10)) -eq 0 ] && echo "...still waiting ($i/120), latest: $(docker logs --tail 1 mimo26 2>/dev/null | head -c 120)"
  sleep 10
done

echo "== 1) 短对话 (计时) =="
python3 - "$BASE" <<'EOF'
import json, time, sys
base = sys.argv[1]
req = {"model":"mimo26","messages":[{"role":"user","content":"用一句话解释什么是流水线并行。"}],"max_tokens":200,"temperature":1.0,"top_p":0.95}
t0=time.time()
r=__import__("urllib.request",fromlist=["x"]).urllib.request.urlopen(
    __import__("urllib.request",fromlist=["x"]).urllib.request.Request(
        base+"/v1/chat/completions", json.dumps(req).encode(), {"Content-Type":"application/json"}), timeout=300)
d=json.loads(r.read())
dt=time.time()-t0
msg=d["choices"][0]["message"]
content=msg.get("content") or ""
reason=msg.get("reasoning_content") or ""
u=d["usage"]
print(f"总耗时 {dt:.1f}s | prompt {u['prompt_tokens']} tok, 输出 {u['completion_tokens']} tok, 吞吐 {u['completion_tokens']/dt:.1f} tok/s")
print(f"reasoning: {reason[:100]}...")
print(f"answer: {content[:200]}")
EOF

echo "== 2) 代码生成 (验证模型能力在线) =="
curl -s --max-time 300 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "model":"mimo26",
  "messages":[{"role":"user","content":"写一个 Python 函数：判断二叉树是否对称，只给代码。"}],
  "max_tokens":400,"temperature":1.0}' | python3 -c "import json,sys; d=json.load(sys.stdin); print((d['choices'][0]['message'].get('content') or '')[:400])"

echo "== 3) 工具调用 =="
curl -s --max-time 120 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "model":"mimo26",
  "messages":[{"role":"user","content":"北京现在几度？"}],
  "tools":[{"type":"function","function":{"name":"get_weather","description":"查询城市天气","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],
  "max_tokens":300}' | python3 -c "import json,sys; d=json.load(sys.stdin); tc=d['choices'][0]['message'].get('tool_calls'); print('tool_calls OK:', json.dumps(tc,ensure_ascii=False)[:200] if tc else 'NONE!')"

echo "== GPU 状态 =="
ssh_local=1 nvidia-smi --query-gpu=index,memory.used,power.draw,utilization.gpu --format=csv,noheader 2>/dev/null || docker exec mimo26 nvidia-smi --query-gpu=index,memory.used,power.draw --format=csv,noheader
echo "== 完成 =="
