# Results

Generated benchmark/profiler artifacts are ignored by default.

When publishing a result set, include at least:

- GPU model
- driver and CUDA/GDS version
- kernel / distribution
- filesystem and mount options
- NVMe device
- PCIe topology
- cache state / `O_DIRECT` status
- backend name and chunk/read size
- whether cuFile was forced into compatibility mode
- evidence used to verify a true GDS path, if claimed
