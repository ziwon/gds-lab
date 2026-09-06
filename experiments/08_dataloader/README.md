# 08 - DataLoader

Use `python/dataloader_bench.py` to sweep worker count, prefetching, pinned memory and sequential/random access against a real file-backed dataset.

The dataset from `make dataset` is deterministic finite float32, not `/dev/zero`. Sequential checksums (no `--random-access`) must match across the worker sweep; a mismatch means the loader returned the wrong bytes. `--random-access` shuffles, so checksums are not comparable.
