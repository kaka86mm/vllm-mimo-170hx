#!/usr/bin/env python3
"""Apply LMCache connector fixes #5657 + #5579 to lmcache_mp_connector.py.

Both upstream patches touch request_finished() in adjacent regions, so they
are applied as one replacement.  Run inside the container:

    python3 apply-connector-fixes.py            # patch in place
    python3 apply-connector-fixes.py --check    # dry run only
"""
import sys

PATH = "/usr/local/lib/python3.12/dist-packages/lmcache/integration/vllm/" \
       "lmcache_mp_connector.py"

# Covers the whole tail of request_finished(): tracker cleanup, the stale
# cleanup_lookup_result() placement (#5579), and both the lazy and eager
# exit paths.  The replacement rewrites this region in the upstream order.
OLD = """        # Clean up request tracker to prevent memory leak
        self._cleanup_request_tracker(request.request_id)

        # Drop lookup state for a request aborted before its lookup was
        # consumed (update_state_after_alloc never ran for it). Both the eager
        # and the lazy path need this, so it runs before they diverge.
        self.scheduler_adapter.cleanup_lookup_result(request.request_id)

        if self.lazy_offload:
            # Blocks return to the free queue (False) and remain observable;
            # the manager ends the LMCache session after all deferred stores
            # have either completed or been dropped.
            actions = self._lazy_offload_manager.on_request_finished(request.request_id)
            for request_id in actions.sessions_to_end:
                self.scheduler_adapter.end_session(request_id)
            return False, (return_params or None)

        # Notify LMCache to end the session for this request
        self.scheduler_adapter.end_session(request.request_id)
        return self._can_store, (return_params or None)
"""

NEW = """        # #5657: the request ended before update_state_after_alloc() ran, so
        # the chunks matched by its lookup will never be retrieved.  Release
        # their read locks now instead of waiting for the L1 read TTL.
        tracker = self.request_trackers.get(request.request_id)
        if (
            tracker is not None
            and tracker.state == LMCacheMPRequestState.PREFETCHING
            and tracker.num_lmcache_hit_tokens > 0
        ):
            self.scheduler_adapter.free_lookup_locks(
                token_ids=tracker.get_token_ids(),
                start=0,
                end=tracker.num_lmcache_hit_tokens,
                request_id=request.request_id,
                cache_salt=tracker.cache_salt,
                request_configs=tracker.request_configs,
            )

        # Clean up request tracker to prevent memory leak
        self._cleanup_request_tracker(request.request_id)

        if self.lazy_offload:
            # Blocks return to the free queue (False) and remain observable;
            # the manager ends the LMCache session after all deferred stores
            # have either completed or been dropped.
            actions = self._lazy_offload_manager.on_request_finished(request.request_id)
            for request_id in actions.sessions_to_end:
                self.scheduler_adapter.end_session(request_id)
            # #5579: end the session before dropping lookup state, otherwise a
            # late LOOKUP can recreate the session server-side.
            self.scheduler_adapter.cleanup_lookup_result(request.request_id)
            return False, (return_params or None)

        # Notify LMCache to end the session for this request
        self.scheduler_adapter.end_session(request.request_id)
        # #5579: see above.
        self.scheduler_adapter.cleanup_lookup_result(request.request_id)
        return self._can_store, (return_params or None)
"""


def main() -> int:
    check = "--check" in sys.argv
    src = open(PATH).read()

    if src.count("free_lookup_locks") and "free_lookup_locks" in \
            src.split("def request_finished(")[1].split(
                "def request_finished_all_groups")[0]:
        print("ALREADY PATCHED")
        return 0

    n = src.count(OLD)
    if n != 1:
        print(f"FAIL: anchor matched {n} times (expected 1)")
        return 1
    if "LMCacheMPRequestState" not in src:
        print("FAIL: LMCacheMPRequestState not imported")
        return 1

    out = src.replace(OLD, NEW)
    print(f"anchor ok; {len(src)} -> {len(out)} bytes")
    if check:
        print("DRY RUN — nothing written")
        return 0

    import py_compile, shutil, time
    bak = f"{PATH}.bak-{int(time.time())}"
    shutil.copy2(PATH, bak)
    open(PATH, "w").write(out)
    try:
        py_compile.compile(PATH, doraise=True)
    except py_compile.PyCompileError as e:
        shutil.copy2(bak, PATH)
        print(f"SYNTAX ERROR, rolled back: {e}")
        return 1
    print(f"PATCHED + syntax OK (backup: {bak})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
