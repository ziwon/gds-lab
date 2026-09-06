# 02 - Sync vs async

`pageable` issues a blocking `cudaMemcpy`; `pinned` issues `cudaMemcpyAsync` on a non-blocking stream and then synchronizes. The `copy_seconds` field isolates that transfer, so the comparison no longer includes the storage read:

```bash
./build/gds_lab --backend pageable --file /data/gds-lab.bin --bytes 1G
./build/gds_lab --backend pinned   --file /data/gds-lab.bin --bytes 1G
```

Asynchrony by itself buys nothing here: with one buffer and an immediate synchronize there is no other work to overlap with. The number to read is `h2d_GBps`, and what it shows is the pinned-vs-pageable DMA difference. Overlap is what experiment 03 adds on top.
