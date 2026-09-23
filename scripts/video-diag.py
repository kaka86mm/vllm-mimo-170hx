import base64, json, urllib.request
BASE = "http://127.0.0.1:8099"
b64 = lambda p: base64.b64encode(open(p, "rb").read()).decode()

def go(content, label, extra=None):
    body = {"model": "mimo26",
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 400, "temperature": 0.2}
    if extra: body.update(extra)
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=300).read())
        m = r["choices"][0]["message"]
        print(f"--- {label} | prompt_tok={r['usage']['prompt_tokens']} completion={r['usage']['completion_tokens']} finish={r['choices'][0]['finish_reason']}")
        print("   content:", repr((m.get("content") or "")[:200]))
        print("   reasoning:", repr((m.get("reasoning_content") or "")[:100]))
    except Exception as e:
        print(f"--- {label} FAIL: {str(e)[:200]}")

vid = [{"type": "text", "text": "视频画面如何变化？"},
       {"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + b64("/tmp/mimo-test-video.mp4")}}]
go("你好，一句话自我介绍。", "纯文本基线")
go(vid, "视频默认")
go(vid, "视频关思考", {"chat_template_kwargs": {"enable_thinking": False}})
