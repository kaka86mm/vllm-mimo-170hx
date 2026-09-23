import re

SRC = "/tmp/mimo_v2_orig.py"
DST = "/home/matri/mimo-v26/patches/mimo_v2.py"

import os
os.makedirs(os.path.dirname(DST), exist_ok=True)
s = open(SRC).read()

# 1) 在 for 循环前插入布局检测与连续布局的整表展开
old_block = """    qs, ks, vs = [], [], []
    for g_idx in range(tp_rank * kv_heads_per_rank, (tp_rank + 1) * kv_heads_per_rank):
        row_start = g_idx * rows_per_group
        scale_row_start = g_idx * scale_rows_per_group
        # Dequantize this group's weights.
        w_g = w_full[row_start : row_start + rows_per_group].to(torch.float32)
        s_g = s_full[scale_row_start : scale_row_start + scale_rows_per_group].to(
            torch.float32
        )
        s_g_expanded = s_g.repeat_interleave(block, dim=0).repeat_interleave(
            block, dim=1
        )[:rows_per_group]
        w_g_dequant = w_g * s_g_expanded"""

new_block = """    # The checkpoint's fp8 scales follow one of two layouts, detectable from
    # the tensor shape: contiguous over the whole fused tensor (SWA layers:
    # ceil(total_rows / block) blocks, block boundaries straddle KV groups),
    # or per-KV-group aligned (GA layers: ceil(rows_per_group / block) blocks
    # per group). Dividing total scale rows by num_kv_heads (the original
    # assumption) truncates and mismatches rows on the contiguous layout.
    total_scale_rows_contig = (w_full.shape[0] + block - 1) // block
    contiguous_scales = s_full.shape[0] == total_scale_rows_contig
    full_row_scales = None
    if contiguous_scales:
        full_row_scales = (
            s_full.to(torch.float32)
            .repeat_interleave(block, dim=0)
            .repeat_interleave(block, dim=1)
        )

    qs, ks, vs = [], [], []
    for g_idx in range(tp_rank * kv_heads_per_rank, (tp_rank + 1) * kv_heads_per_rank):
        row_start = g_idx * rows_per_group
        # Dequantize this group's weights.
        w_g = w_full[row_start : row_start + rows_per_group].to(torch.float32)
        if contiguous_scales:
            # Absolute-row indexing: scale row r covers fused rows
            # [r * block, (r + 1) * block).
            s_g_expanded = full_row_scales[row_start : row_start + rows_per_group]
        else:
            scale_row_start = g_idx * scale_rows_per_group
            s_g = s_full[scale_row_start : scale_row_start + scale_rows_per_group].to(
                torch.float32
            )
            s_g_expanded = s_g.repeat_interleave(block, dim=0).repeat_interleave(
                block, dim=1
            )[:rows_per_group]
        w_g_dequant = w_g * s_g_expanded"""

assert old_block in s, "目标代码块没找到，文件版本可能不同"
s = s.replace(old_block, new_block)
open(DST, "w").write(s)
print("补丁已写:", DST)
