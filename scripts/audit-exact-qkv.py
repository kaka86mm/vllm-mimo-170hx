"""TP=1 exactness audit for the lossless fused-QKV loader.

Runs INSIDE the serving container against the mounted official checkpoint:
  docker cp patches/mimo_v2.py mimo26:/usr/local/lib/python3.12/dist-packages/\
    vllm/model_executor/models/mimo_v2_audit.py
  docker cp scripts/audit-exact-qkv.py mimo26:/tmp/audit.py
  docker exec mimo26 python3 /tmp/audit.py

Classifies layers by qkv row count (GA=13568, SWA=14848), shards each with
the patched loader at tp_size=1, and counts every (fp8 byte, scale) pair
that differs from the checkpoint. Expect 0 changed everywhere.
"""
import json
import sys

import torch
from safetensors import safe_open

import vllm.model_executor.models.mimo_v2_audit as m

MODEL = "/model"
idx = json.load(open(f"{MODEL}/model.safetensors.index.json"))
wm = idx["weight_map"]


def load(layer, suffix):
    key = f"model.layers.{layer}.self_attn.qkv_proj.{suffix}"
    with safe_open(f"{MODEL}/{wm[key]}", framework="pt") as f:
        return f.get_tensor(key)


def audit(layer, num_heads, num_kv, hd, vhd, kv_chunk_rows):
    w, s = load(layer, "weight"), load(layer, "weight_scale_inv")
    out_w, out_s = m._shard_fp8_qkv_proj(
        w, s,
        num_heads=num_heads, num_kv_heads=num_kv, head_dim=hd, v_head_dim=vhd,
        tp_rank=0, tp_size=1, kv_chunk_rows=kv_chunk_rows,
    )
    nb, q_sblk = 4, 24
    q = num_heads // nb * hd
    k = num_kv // nb * hd
    v = num_kv // nb * vhd
    chunk = q + k + v
    srows = -(-chunk // 128)
    changed = total = 0

    def cmp(ow, cw, os_, cs_):
        nonlocal changed, total
        changed += int((ow != cw).sum()) + int((os_ != cs_).sum())
        total += ow.numel() + os_.numel()

    for c in range(nb):  # Q rows are at the chunk head in both layouts
        cmp(
            out_w[c * q : (c + 1) * q],
            w[c * chunk : c * chunk + q],
            out_s[c * q_sblk : (c + 1) * q_sblk],
            s[c * srows : c * srows + q_sblk],
        )

    if kv_chunk_rows:  # [Q.. | K_c V_c pad | ..]
        kv_sblk = kv_chunk_rows // 128
        for c in range(nb):
            base = nb * q + c * kv_chunk_rows
            cmp(
                out_w[base : base + k + v],
                w[c * chunk + q : (c + 1) * chunk],
                out_s[nb * q_sblk + c * kv_sblk : nb * q_sblk + (c + 1) * kv_sblk],
                s[c * srows + q_sblk : (c + 1) * srows],
            )
            pad = out_w[base + k + v : base + kv_chunk_rows]
            assert int((pad != 0).sum()) == 0, "pad rows must stay zero"
    else:  # [Q.. | K.. | V..] block permutation
        k_sblk, v_sblk = k // 128, v // 128
        for c in range(nb):
            cmp(
                out_w[nb * q + c * k : nb * q + (c + 1) * k],
                w[c * chunk + q : c * chunk + q + k],
                out_s[nb * q_sblk + c * k_sblk : nb * q_sblk + (c + 1) * k_sblk],
                s[c * srows + q_sblk : c * srows + q_sblk + k_sblk],
            )
            cmp(
                out_w[nb * q + nb * k + c * v : nb * q + nb * k + (c + 1) * v],
                w[c * chunk + q + k : (c + 1) * chunk],
                out_s[nb * (q_sblk + k_sblk) + c * v_sblk : nb * (q_sblk + k_sblk) + (c + 1) * v_sblk],
                s[c * srows + q_sblk + k_sblk : c * srows + q_sblk + k_sblk + v_sblk],
            )
    kind = "GA padded" if kv_chunk_rows else "SWA perm"
    print(f"layer {layer:>2} ({kind}): {total} elements compared, changed = {changed}")
    return changed


bad = 0
ga = swa = 0
for layer in range(48):
    rows = load(layer, "weight").shape[0]
    if rows == 13568:
        bad += audit(layer, 64, 4, 192, 128, kv_chunk_rows=384)
        ga += 1
    elif rows == 14848:
        bad += audit(layer, 64, 8, 192, 128, kv_chunk_rows=0)
        swa += 1
    else:
        print(f"layer {layer}: unexpected rows {rows}")
        bad += 1

print(f"audited {ga} GA + {swa} SWA layers")
print("AUDIT", "PASS (0 weights changed)" if bad == 0 else "FAIL")
sys.exit(0 if bad == 0 else 1)
