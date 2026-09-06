# GPU storage I/O paths

## Host-staged baseline

```text
NVMe -> Linux I/O path -> CPU RAM -> cudaMemcpy -> GPU VRAM -> kernel
```

This path can still be fast, especially with pinned buffers, asynchronous copies and overlap. It also gives us the baseline needed to understand what GDS is removing.

## cuFile compatibility

```text
application -> libcufile -> POSIX/host staging -> GPU VRAM
```

The application uses the cuFile API, but the data can still traverse system memory. NVIDIA documents compatibility mode as the fallback when a direct path is unavailable or disabled.

## Traditional GDS kernel path

```text
application -> libcufile -> nvidia-fs / filesystem / storage driver
                                  |
                                  `---- DMA ----> GPU VRAM
```

The CPU remains involved in the control path. The important bypass is the CPU-system-memory bounce buffer in the data path.

## PCI P2PDMA path

Recent GDS releases can use Linux PCI P2PDMA for supported local-NVMe configurations, reducing dependence on the traditional `nvidia-fs` path. Treat this as a separate path to verify rather than assuming `nvidia-fs` must always be loaded.
