# LMCache connector 修复族评估（#5657 + #5579 + #5627 + #5614）

评估日期：2026-10-15
基线镜像：`vllm-mimo:lmc-4410p5525`（LMCache 0.5.6.post4410 + 手工移植的 #5525）
验证方式：容器内离线打补丁 + 上游测试用例，**全程未触碰生产容器**

---

## 结论速览

| PR | 内容 | 改动量 | 我们这版是否存在该 bug | 建议 |
|---|---|---|---|---|
| **#5657** | abort 的请求释放 lookup 读锁 | connector +15/-0 | **存在，已实测复现** | **带上** |
| **#5579** | end_session 先于 cleanup_lookup_result | connector +2/-5 | **存在，已实测复现** | **带上** |
| #5627 | 协调器拒收已离开实例的 L1 事件 | 6 文件 +169/-7 | 不适用（未启用 coordinator） | 不带 |
| #5614 | ZMQ 主循环批处理（已合并上游） | mq.py +63/-51 | **架构不同，不适用** | 不带 |

---

## 一、#5657 + #5579：已离线验证通过，可直接采用

### 实测证据（决定性）

用上游 PR 自带的两个测试用例，在**同一个生产镜像**里对比：

```
基线（未打补丁）   : 29 failed, 41 passed
打补丁后          : 17 failed, 53 passed
```

失败差集：

```
补丁修好的（12 个，全部为目标用例）：
  test_abort_after_lookup_hit_frees_lookup_locks[kv_{both,consumer,producer}-{False,True}]      ×6
  test_abort_with_pending_lookup_ends_session_before_cleanup[kv_{both,consumer,producer}-{False,True}] ×6

新引入的回归：无
```

**剩余 17 个失败在基线里就是失败的**（上游 dev 测试对更新代码的期望，如 `test_decode_store_policy_*`、`test_kv_events_*`），与本次补丁无关。

### 两个 bug 在我们代码里的位置

镜像内 `lmcache/integration/vllm/lmcache_mp_connector.py` 的 `request_finished()` 原文：

```python
self._cleanup_request_tracker(request.request_id)
self.scheduler_adapter.cleanup_lookup_result(request.request_id)   # ← #5579: 太早

if self.lazy_offload:
    ...
    self.scheduler_adapter.end_session(request_id)                 # ← 顺序反了
    return False, (return_params or None)

self.scheduler_adapter.end_session(request.request_id)
return self._can_store, (return_params or None)
```

- **#5657**：函数里完全没有 `free_lookup_locks` 分支 → abort 的请求其 lookup 读锁挂到 L1 read TTL（默认 300s），期间对象不可驱逐/删除。
- **#5579**：`cleanup_lookup_result()` 在 `end_session()` 之前 → abort 的请求过早发 END_SESSION，迟到的 LOOKUP 会在服务端重建 session。

### 为什么必须合并应用

两个补丁改**同一函数的相邻区域**（#5657 在 tracker cleanup 前插入，#5579 重排 cleanup/end_session），单独应用会上下文冲突。已合并为一次整体替换。

### 关键结构事实

`request_finished_all_groups()`（HMA 多组入口）**只是委托**：

```python
return self.request_finished(request, block_ids[0] if block_ids else [])
```

→ **改一处覆盖全部路径**，无需分别打补丁。

### 依赖接口核实（全部存在）

| 接口 | 位置 |
|---|---|
| `LMCacheMPRequestState.PREFETCHING` | `lmcache_mp_metadata.py:39` |
| `self.request_trackers` | connector:706 |
| `tracker.num_lmcache_hit_tokens` | connector:1208/1219 |
| `tracker.cache_salt` / `.request_configs` | connector:1228 起 |
| `free_lookup_locks(token_ids, start, end, request_id, cache_salt, request_configs)` | `vllm_multi_process_adapter.py:1107` |

补丁脚本：`apply-connector-fixes.py`（幂等，带 `--check` 干跑、自动备份、语法失败回滚）

---

## 二、#5614：不适用（架构不同）

上游修的是"**单 client** 的 `MessageQueueServer._main_loop` 每 `poll()` 只读一个请求"（#5613：~2,000 lookups/s 后不扩展）。

我们镜像的 `mq.py` 是**另一套架构**：

```
我们（909 行）:  共享 poller + 多 client
                _socket_to_client / ClientPollingLoop
                _process_ops() → process_inbound() / process_outbound_task()

上游 #5614:      单 server 循环重构
                MAX_REQUESTS_PER_POLL = 64 / _handle_request() / _send_responses()
```

实测我们镜像里 `MAX_REQUESTS_PER_POLL`、`_handle_request`、`_send_responses` **均不存在** → 该 PR 的重构无法套用，且**我们的多 client 共享 poller 架构本身就不是它描述的那个瓶颈**。

结论：**不带**。若将来整体 rebase 到含该修复的 dev，自然获得。

---

## 三、#5627：不适用

改 `lmcache/v1/mp_coordinator/`（事件门控的 incarnation 栅栏、健康循环）。我们未启用 mp_coordinator（生产为 engine_driven + fs L2），不适用。

---

## 四、落地方案（待用户批准窗口）

采用方式与 #5525 一致——**烘焙进镜像**：

1. 在 `vllm-mimo:v3.3-qksplit`（或当前生产基线）基础上，容器内运行 `apply-connector-fixes.py`
2. `docker commit` 出新镜像（如 `vllm-mimo:v3.5-connfix`）
3. 用 `make-diag-launcher.sh` 同款方式生成候选 launcher（单 sed 改 IMAGE）
4. 窗口内 rollout + 验证指纹 + 观察 L1 锁相关行为
5. 保留回滚路径（现役 v3.3 launcher 不动）

预期收益：消除 abort 请求的 lookup 读锁泄漏（<300s 内不可驱逐的对象）、修正 abort 时 session 生命周期错乱。属于**健壮性修复**，非性能提升，风险低。

## 证据文件

- `apply-connector-fixes.py` — 补丁脚本
- 盒子 `~/mimo-v26/experiments/lmc-connector-fixes/{base_fail.txt,patched_fail.txt}` — 对比结果
