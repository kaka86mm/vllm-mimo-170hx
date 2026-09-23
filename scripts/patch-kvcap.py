import re

SRC = "/tmp/kvcap_orig.py"
DST = "/home/matri/mimo-v26/patches/kv_cache_utils.py"

import os
os.makedirs(os.path.dirname(DST), exist_ok=True)
s = open(SRC).read()

old = """    num_blocks_per_request = sum(
        cdiv(
            group.kv_cache_spec.max_memory_usage_bytes(vllm_config),
            group.kv_cache_spec.page_size_bytes,
        )
        for group in kv_cache_config.kv_cache_groups
    )
    max_concurrency = kv_cache_config.num_blocks / num_blocks_per_request
    return max_concurrency"""

new = """    parts = []
    for group in kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        parts.append(
            cdiv(spec.max_memory_usage_bytes(vllm_config), spec.page_size_bytes)
        )
    num_blocks_per_request = sum(parts)
    import logging as _lg
    _lg.getLogger("vllm.v1.core.kv_cache_utils").warning(
        "PROBE groups=%s",
        [
            (
                type(g.kv_cache_spec).__name__,
                "blk",
                g.kv_cache_spec.block_size,
                "page",
                g.kv_cache_spec.page_size_bytes,
                "sw",
                getattr(g.kv_cache_spec, "sliding_window", None),
                "reqblocks",
                p,
                "layers",
                len(g.layer_names),
            )
            for g, p in zip(kv_cache_config.kv_cache_groups, parts)
        ],
    )
    _lg.getLogger("vllm.v1.core.kv_cache_utils").warning(
        "PROBE num_blocks=%s per_request=%s mml=%s inflight=%s",
        kv_cache_config.num_blocks,
        num_blocks_per_request,
        vllm_config.model_config.max_model_len,
        getattr(vllm_config, "max_in_flight_tokens", "?"),
    )
    max_concurrency = kv_cache_config.num_blocks / num_blocks_per_request
    return max_concurrency"""

assert old in s, "capacity 函数没匹配上"
s = s.replace(old, new)
open(DST, "w").write(s)
print("探针补丁已写:", DST)
