# 07 - I/O + compute overlap

Profile the `overlap` backend with Nsight Systems and look for whether reads/H2D work leave visible gaps between GPU kernels. The goal is pipeline utilization, not a single headline GB/s number.

```bash
scripts/profile_nsys.sh /data/gds-lab.bin overlap
```

Run it with `--direct` as well. A buffered run whose region fits in RAM keeps the GPU fed from the page cache, which hides exactly the starvation this experiment is meant to expose.

The kernel is a byte-sum reduction over each chunk, deliberately cheap. It exists to make the compute stream visible in the timeline and to produce the checksum, not to represent a real workload; a heavier kernel would change where the pipeline's bottleneck sits.
