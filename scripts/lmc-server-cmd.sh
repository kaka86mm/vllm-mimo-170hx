#!/bin/bash
exec lmcache server --host 0.0.0.0 --port 5555 --chunk-size 8192 --shm-name mimo26shm --separate-object-groups --supported-transfer-mode engine_driven --l1-size-gb 100 --eviction-policy LRU --l2-adapter '{"type":"fs","base_path":"/lmcache-l2"}'
