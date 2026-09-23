import base64, json, urllib.request
import os
BASE = os.environ.get("BASE", f"http://127.0.0.1:{os.environ.get('PORT', '8099')}")
def b64(p): return base64.b64encode(open(p,"rb").read()).decode()
def chat(content, max_tokens, tag):
    req = urllib.request.Request(BASE+"/v1/chat/completions",
        json.dumps({"model":"mimo26","messages":[{"role":"user","content":content}],"max_tokens":max_tokens,"temperature":1.0}).encode(),
        {"Content-Type":"application/json"})
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=300).read())
        m = r["choices"][0]["message"]
        print(f"[{tag}] reasoning:{len(m.get(chr(34)+chr(114)+chr(101)+chr(97)+chr(115)+chr(111)+chr(110)+chr(105)+chr(110)+chr(103)+chr(95)+chr(99)+chr(111)+chr(110)+chr(116)+chr(101)+chr(110)+chr(116)+chr(34),chr(34)) or chr(34)+chr(34))}字 content:{(m.get(chr(99)+chr(111)+chr(110)+chr(116)+chr(101)+chr(110)+chr(116)) or chr(34)+chr(34))[:400]}")
    except Exception as e:
        print(f"[{tag}] FAIL: {str(e)[:300]}")
chat([{"type":"text","text":"视频里画面如何随时间变化？"},
      {"type":"video_url","video_url":{"url":"data:video/mp4;base64,"+b64("/tmp/mimo-test-video.mp4")}}], 350, "视频详测")
chat([{"type":"text","text":"这段音频听起来是什么样的？"},
      {"type":"audio_url","audio_url":{"url":"data:audio/wav;base64,"+b64("/tmp/mimo-test-audio.wav")}}], 250, "音频wav")
