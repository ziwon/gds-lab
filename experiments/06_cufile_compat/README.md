# 06 - cuFile compatibility

Run the cuFile backend with `CUFILE_FORCE_COMPAT_MODE=true`. This validates application-facing cuFile usage without claiming storage-to-GPU direct DMA.

```bash
CUFILE_ENV_PATH_JSON=$PWD/configs/cufile.json \
CUFILE_FORCE_COMPAT_MODE=true \
  ./build/gds_lab --backend cufile --file /data/gds-lab.bin --bytes 1G
```

## Why the direct path is SKIPPED here

`SKIPPED` has two very different causes and they must not be recorded the same way:

- **not installed** - `libcufile` is missing, so the backend was never compiled. `make build` prints `cuFile backend: disabled` and `gds_lab` reports `cufile_compiled=no`. This is a setup gap, not a hardware finding. Fix it with `scripts/install_cuda_redist.sh`.
- **not supported** - the library is present and the API works, but no direct path exists on this host. This is a real result and belongs in the record.

On the RTX 5080 development host the second case applies, and `gdscheck.py -p` is the evidence:

```text
NVMe P2PDMA        : Unsupported
NVMe               : Unsupported
...
properties.use_compat_mode : true
GPU index 0 NVIDIA GeForce RTX 5080 ... IOMMU State: Pass-through or Enabled
WARN: GDS is not guaranteed to work functionally or in a performant way with iommu=on/pt
Platform verification succeeded
```

`nvidia-fs` is not loaded, every backing-store class reports `Unsupported`, and the driver falls back to POSIX host staging. So the numbers this experiment produces are `cufile-forced-compat`, and they are comparable to the `direct` backend - not to a true GDS result.

## Verifying the path rather than inferring it

Throughput alone never distinguishes compat mode from direct DMA. Use:

- `gdscheck.py -p` for driver/filesystem capability
- `gds_stats -p <pid>` while the benchmark runs, for the per-process cuFile path counters
- `configs/cufile.json` with `"cufile_stats": 3` and cuFile TRACE logging when a call unexpectedly succeeds
