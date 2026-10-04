# v3.3 — the eight runtime bind-mount patches baked into the image layer,
# md5-identical to the files the v3.0 launchers mount. The attention kernel
# now also carries the vllm#59673 QK-dot split (192 = 128 + 64, no 256-pad;
# bitwise-identical outputs, sm80 kernel 1.10-1.11x, E2E 200K prefill +5.7%).
# Build from the repo root:   docker build -f docker/Dockerfile -t vllm-mimo:v3.3 .
# Base carries the LMCache PR#4410 build (vllm-mimo:lmc-pr4410c lineage).
FROM vllm-mimo:lmc-pr4410c
COPY patches/triton_attn_diffkv.py /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/triton_attn_diffkv.py
COPY patches/triton_unified_attention_diffkv.py /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/triton_unified_attention_diffkv.py
COPY patches/mimo_v2_omni_model.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/mimo_v2_omni.py
COPY patches/online_fp8.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/online/fp8.py
COPY patches/mimo_v2.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/mimo_v2.py
COPY patches/mimo_v2_mtp.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/mimo_v2_mtp.py
COPY patches/fp8.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/fp8.py
COPY patches/mimo_v2_omni.py /usr/local/lib/python3.12/dist-packages/vllm/transformers_utils/processors/mimo_v2_omni.py
