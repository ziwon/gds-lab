# 00 - Capability

Run `make inspect` and save the host, GPU, filesystem, NVMe and PCIe topology before benchmarking. A cuFile library being present is not equivalent to a verified direct-GDS path.

`inspect.sh` reports facts, but three of its lines are verdicts that gate the other experiments. Read them first:

- **`arch check`** - whether the selected `nvcc` can emit code for this GPU's compute capability. `FAIL` means the build produces no runnable kernel, so 01-07 are not measurable regardless of what compiles.
- **page-cache budget** - the dataset size against `MemTotal`. A dataset smaller than RAM turns repeated buffered reads into a DRAM measurement, which invalidates 01-05 unless the run is labelled accordingly.
- **block device queue** - `max_hw_sectors_kb` is the size the kernel splits reads at. Experiment 04's "block size" sweep is bounded by this, so a 64 MiB application chunk is never a 64 MiB device request.

The `Tooling` section exists for the same reason: a missing `fio` means there is no independent device baseline to compare `gds_lab` numbers against, and a CPU-only `torch` means experiment 08 measures nothing about H2D transfer.
