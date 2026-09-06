# 03 - Double buffering

Use `--backend overlap`. The pipeline alternates two pinned host buffers and two device buffers while separate CUDA streams coordinate H2D copies and a GPU reduction kernel, with events guarding slot reuse.

```bash
./build/gds_lab --backend overlap --file /data/gds-lab.bin --bytes 1G --chunk-bytes 64M
./build/gds_lab --backend overlap --file /data/gds-lab.bin --bytes 1G --chunk-bytes 64M --direct
```

`overlap` reports no `read_seconds` / `copy_seconds` split: read and copy are pipelined on purpose, so reporting them separately would count overlapped time twice. Compare its `effective_GBps` against the sum of experiment 01's read and copy times instead - that difference is what the overlap bought.

`--direct` runs the same pipeline on `O_DIRECT`. Without it the pipeline is a page-cache measurement on any host where the region fits in RAM, which makes 03 and 07 look far better than the storage path allows.
