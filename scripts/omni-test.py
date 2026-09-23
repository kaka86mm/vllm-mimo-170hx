import base64, json, time, urllib.request
BASE = "http://127.0.0.1:8099"

def b64(p):
    return base64.b64encode(open(p, "rb").read()).decode()

def chat(content, max_tokens=200, tag=""):
    t0 = time.time()
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        json.dumps({"model": "mimo26", "messages": [{"role": "user", "content": content}],
                    "max_tokens": max_tokens, "temperature": 1.0}).encode(),
        {"Content-Type": "application/json"})
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=300).read())
        m = r["choices"][0]["message"]
        txt = (m.get("content") or "")[:160].replace("\n", " ")
        u = r["usage"]
        print(f"[{tag}] {time.time()-t0:.1f}s | {u['completion_tokens']} tok | {txt}")
        return True
    except Exception as e:
        print(f"[{tag}] FAIL: {str(e)[:250]}")
        return False

ok = True
ok &= chat("用一句话解释流水线并行。", 150, "文本")
ok &= chat([{"type": "text", "text": "描述这张图片的内容和颜色分布。"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64("/tmp/mimo-test-img.png")}}], 250, "图片")
ok &= chat([{"type": "text", "text": "这个视频里画面如何随时间变化？有什么运动元素？"},
            {"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + b64("/tmp/mimo-test-video.mp4")}}], 300, "视频")
ok &= chat([{"type": "text", "text": "这段音频的频率特征是怎样的？"},
            {"type": "input_audio", "input_audio": {"data": "data:audio/mp4;base64," + b64("/tmp/mimo-test-video.mp4"), "format": "mp4"}}], 200, "音频")
print("ALL OK" if ok else "有失败项")
