import json, random, time, urllib.request, threading

URL = "http://127.0.0.1:8099/v1/chat/completions"
WORDS = ["显卡","带宽","算子","调度","缓存","拓扑","量化","流水线","碎片","融合",
         "批处理","注意力","稀疏性","蒸馏","集群","引擎","词元","向量","梯度","预热"]

def one_stream(i, n_words, max_tok, results):
    random.seed(i * 7919 + n_words)
    prompt = " ".join(random.choice(WORDS) for _ in range(n_words)) + f"\n\n现在从{i*100}开始往上数{max_tok}个数，只输出数字，用空格分隔。"
    body = {"model":"mimo26","messages":[{"role":"user","content":prompt}],
            "max_tokens":max_tok+20,"temperature":0,"stream":True,
            "stream_options":{"include_usage":True},
            "chat_template_kwargs":{"enable_thinking":False}}
    t0=time.time(); times=[]
    try:
        r = urllib.request.urlopen(urllib.request.Request(URL,
            data=json.dumps(body).encode(), headers={"Content-Type":"application/json"}), timeout=1800)
        for raw in r:
            line = raw.decode("utf-8","ignore").strip()
            if not line.startswith("data: ") or line=="data: [DONE]": continue
            try: j=json.loads(line[6:])
            except Exception: continue
            d=(j.get("choices") or [{}])[0].get("delta") or {}
            if d.get("content"): times.append(time.time())
    except Exception as e:
        results[i]=("ERR",str(e),0,0); return
    if len(times)>40:
        seg=times[30:]; dur=seg[-1]-seg[0]
        per=(len(seg)-1)/dur if dur>0 else 0
        wall=times[-1]-t0
        results[i]=("OK","",per,wall)
    else:
        results[i]=("SHORT","",len(times),0)

def bench(label, n_streams, n_words, max_tok=150):
    results={}
    th=[threading.Thread(target=one_stream,args=(i,n_words,max_tok,results)) for i in range(n_streams)]
    t0=time.time()
    for t in th: t.start()
    for t in th: t.join()
    wall=time.time()-t0
    oks=[r for r in results.values() if r[0]=="OK"]
    agg=sum(r[2] for r in oks)
    print(f"[{label}] 流数{n_streams} | 全部墙钟{wall:.0f}s | 聚合{agg:.0f} tok/s | 单流均值{agg/max(len(oks),1):.1f} | 成功{len(oks)}/{n_streams}", flush=True)

for n in [2,4]:
    bench(f"19K×{n}并发", n, 5600)
for n in [2,4]:
    bench(f"67K×{n}并发", n, 18600)
bench("19K×8并发", 8, 5600)
