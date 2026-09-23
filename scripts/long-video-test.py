import base64, json, os, time, urllib.request

BASE = os.environ.get("BASE", f"http://127.0.0.1:{os.environ.get('PORT', '8099')}")
b64 = lambda p: base64.b64encode(open(p, "rb").read()).decode()

def ask_video(path, question, tag, max_tokens=500):
    body = {"model": "mimo26",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": question},
                {"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + b64(path)}}]}],
            "max_tokens": max_tokens, "temperature": 0.2,
            "chat_template_kwargs": {"enable_thinking": False}}
    r = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(),
                               {"Content-Type": "application/json"})
    t0 = time.time()
    try:
        d = json.loads(urllib.request.urlopen(r, timeout=1800).read())
        dt = time.time() - t0
        m = d["choices"][0]["message"]
        u = d["usage"]
        print(f"[{tag}] {dt:.0f}s | prompt_tok={u['prompt_tokens']} completion={u['completion_tokens']} finish={d['choices'][0]['finish_reason']}")
        print("   内容:", (m.get("content") or "")[:220].replace("\n", " "))
    except Exception as e:
        print(f"[{tag}] FAIL {time.time()-t0:.0f}s: {str(e)[:200]}")

# 60 秒: testsrc 计时器数字会从 0 涨到 59 左右
ask_video("/tmp/mimo-vid-60s.mp4",
          "这个视频有多长？画面里的计时器数字范围是多少？开头和结尾画面有什么不同？", "60秒")
# 5 分钟: testsrc2 有彩球运动+计时器
ask_video("/tmp/mimo-vid-5min.mp4",
          "这个视频有多长？描述画面中的运动元素和计时器显示的数字范围。", "5分钟")
