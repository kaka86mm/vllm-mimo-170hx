import os
import json, struct

MODEL_DIR = os.environ.get("MODEL_DIR", "/tmp/MiMo-V2.6-Flash-RL-official") + "/"
idx = json.load(open(MODEL_DIR + "model.safetensors.index.json"))
wm = idx["weight_map"]

def header(shard):
    with open(MODEL_DIR + "" + shard, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))

targets = ["layers.1.self_attn", "layers.2.self_attn"]
names = sorted(k for k in wm if any(t in k for t in targets))
print("相关张量:", names[:14])

shards = {wm[k] for k in names}
info = {}
for sh in shards:
    h = header(sh)
    for name, meta in h.items():
        if name == "__metadata__":
            continue
        if any(t in name for t in targets):
            info[name] = (meta["shape"], meta["dtype"])
for n in sorted(info):
    print(f"{n}: {info[n][0]} {info[n][1]}")
