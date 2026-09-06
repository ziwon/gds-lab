# 04 - Block size

`--read-chunk` sets the storage read granularity for `pageable`, `pinned`, `direct` and `cufile`. It is separate from `--chunk-bytes`, which sizes the overlap pipeline's buffers - sweeping the pipeline buffer would change three things at once.

```bash
for c in 4K 64K 256K 1M 4M 16M 64M; do
  ./build/gds_lab --backend direct --file /data/gds-lab.bin --bytes 1G --read-chunk "$c"
done
```

Read `read_GBps`, not `effective_GBps`: the H2D copy is constant across the sweep and would flatten the curve.

Two things bound the result and belong in the write-up:

- `max_hw_sectors_kb` from `make inspect` is where the kernel splits a request. A 64 MiB `--read-chunk` is not a 64 MiB device request, so the upper half of the curve measures syscall and queue behaviour, not device transfer size.
- Below 4 KiB the `direct` backend refuses the run, because `O_DIRECT` requires aligned sizes. Use the buffered backends for smaller blocks and label them as such.
