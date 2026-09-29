"""Kernel-level A/B: stock unified_attention_diffkv (q>1 -> 2D) vs diffbot
spec-3D version, synthetic GA-layer shapes, no serving stack.

Run inside the serving container.
"""
import importlib.util
import pathlib
import torch

torch.cuda.init()
dev = torch.device("cuda")

# ---- import both kernel modules
import vllm.v1.attention.ops.triton_unified_attention_diffkv as stock_mod

script_dir = pathlib.Path(__file__).parent
spec = importlib.util.spec_from_file_location(
    "tuad_spec3d", script_dir / ".." / "patches" / "triton_unified_attention_diffkv.py")
s3d_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s3d_mod)

# ---- GA-layer shapes (MiMo global attention)
HQ, HKV, DQK, DV = 64, 4, 192, 128
BLOCK = 128
KV = 67000
NBLK = (KV + BLOCK - 1) // BLOCK

pool = torch.randn(NBLK, BLOCK, HKV, DQK + DV, device=dev, dtype=torch.bfloat16)
kc = pool[..., :DQK]
vc = pool[..., DQK:]
block_table = torch.arange(NBLK, device=dev, dtype=torch.int32).view(1, -1)
seqused_k = torch.tensor([KV], device=dev, dtype=torch.int32)

SEG = 64
SEQ_TH = 8  # num_seqs=1 < threshold -> 3D eligible
buf_shape = (SEQ_TH, HQ, SEG)
segm_out = torch.empty(*buf_shape, DV, device=dev, dtype=torch.float32)
segm_max = torch.empty(*buf_shape, device=dev, dtype=torch.float32)
segm_sum = torch.empty(*buf_shape, device=dev, dtype=torch.float32)


def run(mod, q_len, kv=KV, iters=30):
    q = torch.randn(q_len, HQ, DQK, device=dev, dtype=torch.bfloat16)
    out = torch.empty(q_len, HQ, DV, device=dev, dtype=torch.bfloat16)
    cu_q = torch.tensor([0, q_len], device=dev, dtype=torch.int32)
    nblk = (kv + BLOCK - 1) // BLOCK
    bt = block_table[:, :nblk].contiguous()
    sk = torch.tensor([kv], device=dev, dtype=torch.int32)

    def call():
        mod.unified_attention_diffkv(
            q=q, k=kc, v=vc, out=out,
            cu_seqlens_q=cu_q, seqused_k=sk,
            softmax_scale=DQK ** -0.5, causal=True,
            alibi_slopes=None, use_alibi_sqrt=False,
            window_size=(-1, -1), block_table=bt, softcap=0.0,
            sinks=None, max_seqlen_q=q_len,
            seq_threshold_3D=SEQ_TH,
            num_par_softmax_segments=SEG,
            softmax_segm_output=segm_out,
            softmax_segm_max=segm_max,
            softmax_segm_expsum=segm_sum,
        )
    for _ in range(5):
        call()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
    st.record()
    for _ in range(iters):
        call()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / iters


print(f"shapes: HQ={HQ} HKV={HKV} DQK={DQK} DV={DV} KV={KV}", flush=True)
print(f"{'q_len':>5} | {'stock ms':>9} | {'spec3D ms':>9} | speedup")
for q_len in (1, 2, 3, 4):
    t_stock = run(stock_mod, q_len)
    t_s3d = run(s3d_mod, q_len)
    print(f"{q_len:>5} | {t_stock:>9.2f} | {t_s3d:>9.2f} | {t_stock/t_s3d:>6.2f}x", flush=True)

print()
print("KV=19K:")
pool2_nblk = (19000 + BLOCK - 1) // BLOCK
print(f"{'q_len':>5} | {'stock ms':>9} | {'spec3D ms':>9} | speedup")
for q_len in (1, 2, 3, 4):
    t_stock = run(stock_mod, q_len, kv=19000)
    t_s3d = run(s3d_mod, q_len, kv=19000)
    print(f"{q_len:>5} | {t_stock:>9.2f} | {t_s3d:>9.2f} | {t_stock/t_s3d:>6.2f}x", flush=True)
