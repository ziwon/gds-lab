# 03 - Double buffering

Use `--backend overlap --chunk-bytes 64M`. The pipeline alternates two pinned host buffers and two device buffers while separate CUDA streams coordinate H2D copies and a GPU touch/reduction kernel.
