# GDS support notes

This repository targets RTX 5080/5090 for the host-staged and cuFile-compatibility experiments, but it does not use those machines to publish true direct-GDS performance claims.

Before a direct-GDS run, verify the **current** NVIDIA support matrix and all of the following:

- GPU product support
- Linux distribution and kernel
- NVIDIA driver / open kernel module requirements
- CUDA/GDS version
- filesystem and mount options
- local NVMe vs remote storage path
- `nvidia-fs` vs PCI P2PDMA path
- PCIe/NUMA topology

Useful references:

- https://docs.nvidia.com/gpudirect-storage/release-notes/
- https://docs.nvidia.com/gpudirect-storage/overview-guide/
- https://docs.nvidia.com/gpudirect-storage/configuration-guide/

The lab intentionally prints `SKIPPED` where it cannot prove the requested capability.
