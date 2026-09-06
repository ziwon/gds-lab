# 01 - Pageable vs pinned

Compare `--backend pageable` with `--backend pinned`.

Both backends report the storage read and the H2D copy as separate numbers, because a single total cannot say which of the two page-locking actually moved:

```bash
./build/gds_lab --backend pageable --file /data/gds-lab.bin --bytes 1G
./build/gds_lab --backend pinned   --file /data/gds-lab.bin --bytes 1G
```

```text
read_seconds / read_GBps   storage -> host buffer
copy_seconds / h2d_GBps    host buffer -> device
```

The expected finding is that `read_GBps` barely moves while `h2d_GBps` roughly triples, because page-locked memory changes the DMA path, not the filesystem path. If `read_GBps` also changes, the host buffer allocation is interfering and should be investigated before drawing conclusions.

Watch CPU and GPU utilization separately, and keep the `checksum` field in view: both backends must report the same value for the same region.
