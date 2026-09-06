# gds-lab

A small systems lab for understanding how storage data reaches NVIDIA GPU memory.

The project is intentionally broader than a single GPUDirect Storage benchmark. It walks through the host-staged path first, then removes bottlenecks step by step:

```text
File / NVMe
   |
   +--> POSIX + pageable RAM + synchronous H2D
   |
   +--> POSIX + pinned RAM + asynchronous H2D
   |
   +--> O_DIRECT + pinned RAM + asynchronous H2D
   |
   +--> double-buffered read / copy / GPU compute overlap
   |
   +--> cuFile compatibility path
   |
   `--> true GDS direct path                    [supported platforms only]
```

## Why this lab exists

`cuFile` is an application-facing I/O API. `nvidia-fs` is a kernel-side component used by the traditional GDS path. They are not competing technologies and they live at different layers of the stack.

On an unsupported GPU or filesystem, cuFile can use compatibility mode and stage I/O through host memory. The lab therefore follows three rules:

1. **Measured** numbers are reported as measured.
2. **Simulated** paths are explicitly labelled as simulated.
3. **Unsupported** direct-GDS tests print `SKIPPED` instead of publishing host-staged throughput as GDS throughput.

## Target machines

The initial development target is Ubuntu on GeForce RTX 5080 / RTX 5090. These machines are useful for studying pageable vs pinned memory, `O_DIRECT`, asynchronous H2D copies, overlap, DataLoader tuning, cuFile APIs, and compatibility mode.

Do not interpret a successful cuFile call on a GeForce system as proof that storage DMA bypassed host memory. Confirm the active path with `gdscheck`, cuFile statistics/logging, system topology, and the current NVIDIA support matrix.

A later run on a supported data-center / professional GPU can fill in the final true-GDS comparison without changing the overall experiment model.

## Quick start

### 1. Inspect the host

```bash
make inspect
```

### 2. Create a test file

By default this writes a 4 GiB file. Put it on the NVMe/filesystem you actually want to measure.

```bash
make dataset DATASET=/data/gds-lab.bin SIZE_GIB=4
```

### 3. Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

If `cufile.h` and `libcufile.so` are available, the cuFile backend is enabled automatically.

### 4. Run the staged I/O experiments

```bash
./build/gds_lab --backend pageable --file /data/gds-lab.bin --bytes 1G
./build/gds_lab --backend pinned   --file /data/gds-lab.bin --bytes 1G
./build/gds_lab --backend direct   --file /data/gds-lab.bin --bytes 1G
./build/gds_lab --backend overlap  --file /data/gds-lab.bin --bytes 1G --chunk-bytes 64M
```

### 5. Run cuFile in compatibility mode

```bash
CUFILE_ENV_PATH_JSON=$PWD/configs/cufile.json \
CUFILE_FORCE_COMPAT_MODE=true \
./build/gds_lab --backend cufile --file /data/gds-lab.bin --bytes 1G
```

This command is intentionally labelled **cuFile compatibility**, not true GDS.

### 6. PyTorch DataLoader sweep

```bash
python3 python/dataloader_bench.py \
  --file /data/gds-lab.bin \
  --workers 0,2,4,8 \
  --batches 100
```

## Experiment map

| ID | Experiment | Main question |
|---|---|---|
| 00 | Capability | What storage/GPU/GDS paths can this host actually use? |
| 01 | Pageable vs pinned | What does page-locked host memory change? |
| 02 | Sync vs async | When does non-blocking H2D help? |
| 03 | Double buffering | Can read/copy/compute overlap hide latency? |
| 04 | Block size | Where is the throughput knee for this NVMe/filesystem? |
| 05 | Sequential vs random | How much does access pattern matter? |
| 06 | cuFile compatibility | What does the same application-facing API look like without direct GDS? |
| 07 | I/O + compute overlap | Does the GPU remain fed while work executes? |
| 08 | DataLoader | How do workers, prefetch, pinned memory and persistent workers affect GPU starvation? |

See [`experiments/README.md`](experiments/README.md) for the run plan.

## Page cache warning

Buffered POSIX and `mmap` benchmarks can turn into RAM/page-cache benchmarks after the first iteration. This repository keeps buffered and `O_DIRECT` paths separate for that reason.

Treat results as one of:

```text
hot cache        page cache -> application -> GPU
buffered/cold    NVMe -> page cache -> application -> GPU
O_DIRECT         NVMe -> aligned host buffer -> GPU
cuFile compat    cuFile -> POSIX/host staging -> GPU
true GDS         storage DMA -> GPU memory                 [supported hosts]
```

Do not compare those labels as if they represented the same data path.

## Profiling

```bash
scripts/profile_nsys.sh /data/gds-lab.bin overlap
scripts/collect_system_metrics.sh
```

Useful external tools include:

- `gdscheck.py -p`
- `nvidia-smi topo -m`
- `nvidia-smi dmon`
- `iostat -xz 1`
- `pidstat -dru 1`
- Nsight Systems
- cuFile statistics / TRACE logging when diagnosing the actual cuFile path

## Repository layout

```text
gds-lab/
├── src/                 C++/CUDA benchmark backends
├── experiments/         experiment plans and interpretation notes
├── python/              PyTorch/DataLoader experiments
├── scripts/             inspection, dataset, profiling and metric helpers
├── configs/             per-workload cuFile configuration
├── docs/                architecture notes
└── results/             local result artifacts (ignored except README)
```

## References

- NVIDIA GPUDirect Storage Overview Guide: https://docs.nvidia.com/gpudirect-storage/overview-guide/
- NVIDIA cuFile API Reference: https://docs.nvidia.com/gpudirect-storage/api-reference-guide/
- NVIDIA GDS Benchmarking and Configuration Guide: https://docs.nvidia.com/gpudirect-storage/configuration-guide/
- NVIDIA GDS O_DIRECT Requirements Guide: https://docs.nvidia.com/gpudirect-storage/o-direct-guide/
- Inspiration for staged storage / preprocessing experiments: https://github.com/cfregly/ai-performance-engineering/tree/main/code/ch05

## Status

Initial scaffold. The first milestone is reproducible single-GPU storage-path characterization on Ubuntu + RTX 5080/5090. True direct-GDS validation is deliberately deferred to a supported host.
