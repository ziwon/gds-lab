# Experiments

The experiment order is deliberate: establish the normal host-staged path before introducing cuFile/GDS terminology.

| ID | Path | Variables |
|---|---|---|
| 00 | capability | GPU, CUDA, cuFile, nvidia-fs, filesystem, NVMe, PCIe topology |
| 01 | pageable vs pinned | pageable host allocation vs page-locked host allocation |
| 02 | sync vs async | synchronous H2D vs asynchronous H2D |
| 03 | double buffer | chunk size, copy stream, compute stream |
| 04 | block size | 4 KiB -> 16 MiB+ read sizes |
| 05 | access pattern | sequential vs randomized offsets |
| 06 | cuFile compat | cuFile API with forced compatibility mode |
| 07 | I/O + compute | overlap efficiency and GPU idle gaps |
| 08 | DataLoader | workers, prefetch, pin_memory, persistent workers |

## Measurement rules

- Record filesystem, mount options, kernel, CUDA/driver and GPU model with every result set.
- Label page-cache state. Do not call a hot-cache result "NVMe throughput".
- Treat `O_DIRECT` and buffered I/O as different experiments.
- `cuFile` means API usage; it does **not** by itself prove direct GDS.
- Direct GDS results require path verification, not inference from throughput.
- Unsupported configurations use `SKIPPED`.

## Suggested first run

```bash
source scripts/cuda_env.sh
make inspect
make dataset SIZE_GIB=4
make build
make bench BYTES=1G CHUNK_BYTES=64M
```
