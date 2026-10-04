#!/bin/bash
exec lmcache server --host 0.0.0.0 --port 5555 --chunk-size 8192 --shm-name mimo26shm --separate-object-groups --max-gpu-workers 4 --supported-transfer-mode engine_driven --l1-size-gb 100 --l1-init-size-gb 96 --no-l1-use-lazy --eviction-policy LRU --l2-adapter '{"type":"fs","base_path":"/lmcache-l2","relative_tmp_dir":"tmp"}'
