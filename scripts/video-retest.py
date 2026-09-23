import base64, json, urllib.request
BASE = "http://127.0.0.1:8099"
b64 = lambda p: base64.b64encode(open(p, "rb").read()).decode()

req = urllib.request.Request(
    BASE + "/v1/chat/completions",
    json.dumps({"model": "mimo26",
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "视频里画面如何随时间变化？有什么运动元素？请直接回答。"},
                    {"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + b64("/tmp/mimo-test-video.mp4")}}]}],
                "max_tokens": 800, "temperature": 1.0}).encode(),
    {"Content-Type": "application/json"})
r = json.loads(urllib.request.urlopen(req, timeout=300).read())
m = r["choices"][0]["message"]
print("finish:", r["choices"][0]["finish_reason"], "| usage:", r["usage"]["completion_tokens"])
print("reasoning 长度:", len(m.get("reasoning_content") or ""))
print("content:", (m.get("content") or "")[:500])
