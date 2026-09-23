# MiMo-V2.6-Flash-RL @ 4x170HX 部署结果 (2026-09-22晚)

## 最终配置 (launch-omni.sh, 已在跑, 端口 8099)
- 权重: ProCreations NVFP4 转码版 (arch 已改 MiMoV2OmniForCausalLM = 全模态)
- 镜像: lazymio/vllm-backport:v0.13.0-sm80
- PP=4, VLLM_PP_LAYER_PARTITION=11,13,12,12, util 0.94
- fp8_e4m3 KV + block 256, max-model-len 262144, mnbt 8192, max-seqs 32
- MTP 投机 3 token (greedy/standard), async-scheduling, prefix-caching
- KV 池: 447,350 token (262K 并发 1.71x)

## 实测
- 单流 decode: 73-75 tok/s (热态)
- 并发 decode: 2流111 / 4流87 / 8流169 / 16流267 tok/s 聚合
- prefill: 32K prompt 9.7s (~3400 tok/s), 8K 4.2s
- 模态: 文本/图片/视频/音频 全通 (视频描述准确; 音频正确识别 440Hz 正弦波)
- 工具调用/推理解析: 正常
- 多模态请求建议带 chat_template_kwargs.enable_thinking=false (思考模式+mm 时 reasoning parser 偶发吞 content)

## 已知边界
- DFlash 不可用: 本构建中 drafter 需要目标 5 层中间态(EAGLE3接口), PP 下无跨rank传输路径; MTP 是 PP 机器正解
- 官方权重触发 mimo_v2.py:527 _shard_fp8_qkv_proj 形状 bug (TP4交错 vs TP=1, 1856vs1792) — 用 ProCreations 版规避
- util 0.95+ 有 cudagraph 捕获 OOM 风险 (dsv4 runbook 红线), 0.94 实测稳定

## 恢复
- glm-exl3 (8093): docker stop mimo26 && docker start glm-exl3
- 重启 mimo26: bash ~/mimo-v26/launch-omni.sh (容器 restart=unless-stopped 自愈)

## KV 池扩展 (09-23 凌晨追加)
- --kv-cache-memory 10737418240 (10GiB) 显式压掉 profiler 安全余量: 池 447,350 -> 543,495 token (+21.5%), 262K 并发 2.07x
- 注意: 11GiB 太激进, 4x126K 长上下文 prefill 会 OOM 500; 10GiB 实测 4 路并发零异常
- 根因未修: SWA 层(39/48)按全历史存 KV (~89KB/token vs 理论 ~11.5KB), hybrid 分配的容量计算把它当全量; 上游 fork 截至 09-22 只推进了 1 个无关提交, 无修复
- 后续大杀器: 若把滑窗 KV 回收真正打通(需改 kv_cache_utils 容量模型+PP 跨 rank 一致性), 池可到 ~3M token 量级

## LMCache CPU offload 攻坚记录 (09-23 凌晨, 已存档未上线)
- 最终打通的组合: CPU-only lmcache server (host网络+host IPC, --supported-transfer-mode engine_driven)
  + vllm 侧 kv_connector_extra_config 里 "lmcache.mp.mp_transfer_mode":"engine_driven" + "kv_buffer_size":268435456
  + --prefix-cache-retention-interval 1024 (滑窗模型必需) + kv-cache-memory 降到 9GiB
- 完整配置见 launch-omni-lmc.sh; 坑: AUTO模式走GPU IPC在CMP解锁驱动上挂 cudaErrorMapBufferObjectFailed;
  transfer mode 环境变量无效必须走 extra config JSON; 经典 LMCacheConnector 已被删; server 侧模式是独立 CLI 参数
- 实测: CPU层真实填充(RAM 21→172GB)、Deferred取回队列工作; 但 engine_driven 经 PCIe2.0x4 搬 SWA 全历史KV
  (~89KB/token), 取回速度(排队后 222 tok/s)远低于重算(19K tok/s) — 当前只适合当容量备胎, 不适合当加速器
- 下一步若要可用: 先修 SWA 全历史存储(把 89KB/token 降到 11.5KB, 传输量降8x), 或调 chunk/并发传输

## KV 池根治 (09-23 上午) — 543K -> 780,571 token (+43.5%)
根因(探针实证): hybrid 管理器一直在正常工作(1 FullAttention组 + 5 SlidingWindow组 sw=128),
但每滑窗组每请求占用 = 窗口127 + max_in_flight_tokens; 后者 = mnbt×(PP深度+1) = 8192×5 = 40960,
占99.7%. chunked prefill 块穿越4级流水线期间不可释放, 保守计入容量.
修复: --max-num-batched-tokens 8192 -> 4096 (in_flight减半, 每请求2634->1834块, 池+43.5%).
意外收益: prefill 反而更快 (8K: 4.2->3.2s, 32K: 9.7->8.3s, 小chunk减少PP队头阻塞).
实测: 4x191K 并发全准入, 峰值用量仅41.5% (实际滑窗稳态远优于保守公式, 实际可容纳~8路191K).
还可再换: mnbt 2048 -> 池~100万 (prefill 代价待测).
上游PR方向: 滑窗块按stage释放(块在本stage处理完即可释放, 不必等全流水线退出), 可把in_flight项再除以PP倍数.

## 最终冻结 (09-23 上午): mnbt=1024, KV 池 1,160,104 token
调参曲线(全维度单调向好, 无拐点直到1024):
| mnbt  | 池token   | 32K prefill | 单流decode |
|-------|-----------|-------------|------------|
| 8192  | 447,350   | 9.7s        | 73-75      |
| 4096  | 780,571   | 8.3s        | 75         |
| 2048  | 998,304   | 7.9s        | 75         |
| 1024  | 1,160,104 | 7.7s        | 75         |
认证: 4x191K 并发全完成, 峰值KV 20.1%, 零异常(等效实际容量~380万token并发长上下文)
注: 512 预期仅+9%且chunk过小有per-chunk开销风险, 不再往下. launch-omni.sh 已是最终版.
