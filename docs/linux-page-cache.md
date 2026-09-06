# Linux page cache and benchmark validity

Repeated buffered reads can stop measuring the storage device and start measuring memory/page-cache behavior.

Keep these cases separate:

```text
hot cache        page cache -> userspace -> GPU
buffered/cold    NVMe -> page cache -> userspace -> GPU
O_DIRECT         NVMe -> aligned userspace buffer -> GPU
```

## Practical rules

- Make the dataset larger than the tiny toy tensors commonly used in microbenchmarks.
- Record whether a run is first-touch or repeated.
- Use the `direct` backend when the question is storage-path throughput rather than cache reuse.
- If you intentionally drop caches, record that operation and understand its system-wide impact.
- Do not compare a hot-cache buffered run with an `O_DIRECT` run without labelling them.
