# cuFile vs nvidia-fs

They live at different layers.

| Component | Layer | Role |
|---|---|---|
| cuFile / `libcufile.so` | user space | application-facing file I/O APIs for GPU buffers |
| `nvidia-fs.ko` | kernel space | kernel-side plumbing used by the traditional GDS path |

A useful shorthand is:

```text
cuFile:     "read this file into this GPU buffer"
nvidia-fs:  kernel-side support for mapping/routing the GPU DMA path
```

Compatibility mode is important because the same cuFile API can fall back to a CPU-staged POSIX path. Therefore `cuFileRead()` succeeding is not sufficient evidence that `nvidia-fs` or a direct GDS data path was used.

CUDA 12.8+ also introduced supported local-NVMe cases where PCI P2PDMA can be used without the traditional `nvidia-fs` requirement, which is another reason not to equate cuFile with nvidia-fs.

References:

- https://docs.nvidia.com/gpudirect-storage/overview-guide/
- https://docs.nvidia.com/gpudirect-storage/api-reference-guide/
