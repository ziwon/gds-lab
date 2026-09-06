#!/usr/bin/env bash
set -u

TARGET="${1:-.}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Picks a toolkit that can target this GPU and exports CUDA_HOME/PATH/LD_LIBRARY_PATH.
# shellcheck source=scripts/cuda_env.sh
. "$SCRIPT_DIR/cuda_env.sh" 2>/dev/null || true

section() { printf '\n== %s ==\n' "$1"; }
has() { command -v "$1" >/dev/null 2>&1; }

# GiB with one decimal, from a byte count.
gib() { awk -v b="$1" 'BEGIN { printf "%.1f", b / 1073741824 }'; }

section "OS / kernel"
uname -a
if [[ -r /etc/os-release ]]; then grep -E '^(NAME|VERSION|PRETTY_NAME)=' /etc/os-release || true; fi

section "GPU"
if has nvidia-smi; then
  nvidia-smi --query-gpu=name,driver_version,pci.bus_id,memory.total --format=csv,noheader || true
  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
  if grep -qi 'GeForce' <<<"$GPU_NAME"; then
    echo "NOTE: GeForce detected. A successful cuFile call does not prove a true direct-GDS path."
  fi
else
  echo "nvidia-smi: not found"
fi

section "CUDA / cuFile"
NVCC=""
if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
  echo "CUDA_HOME=$CUDA_HOME"
  NVCC="$CUDA_HOME/bin/nvcc"
elif has nvcc; then
  echo "CUDA_HOME: unset/unusable"
  NVCC="$(command -v nvcc)"
fi
if [[ -n "$NVCC" ]]; then
  echo "nvcc: $NVCC"
  "$NVCC" --version | tail -n 4
else
  echo "nvcc: not found"
fi
if has nvidia-smi; then
  echo "driver CUDA version: $(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -1)"
  CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
  echo "compute capability: ${CC:-unknown}"
  # The verdict experiment 00 actually needs: can this nvcc emit code for this GPU?
  if [[ -n "$NVCC" && -n "$CC" ]]; then
    ARCH="compute_${CC//./}"
    if "$NVCC" --list-gpu-arch 2>/dev/null | grep -qx "$ARCH"; then
      echo "arch check: OK - nvcc can target $ARCH"
    else
      echo "arch check: FAIL - nvcc cannot target $ARCH (max $("$NVCC" --list-gpu-arch 2>/dev/null | tail -1))"
      echo "            builds will produce no runnable kernel; see scripts/install_cuda_redist.sh"
    fi
  fi
fi
if ldconfig -p 2>/dev/null | grep -q libcufile; then
  ldconfig -p | grep libcufile || true
else
  echo "libcufile: not visible through ldconfig"
fi
for lib in "${CUDA_HOME:-}/lib/libcufile.so" "${CUDA_HOME:-}/lib64/libcufile.so"; do
  [[ -e "$lib" ]] && { echo "cufile library: $lib"; break; }
done
for header in \
  "${CUDA_HOME:-}/include/cufile.h" \
  /usr/local/cuda/include/cufile.h \
  /usr/include/cufile.h
do
  [[ -f "$header" ]] && { echo "cufile header: $header"; break; }
done

section "nvidia-fs / NVIDIA kernel module"
lsmod | grep -E '^nvidia_fs\b' || echo "nvidia-fs: not loaded"
if has modinfo; then modinfo nvidia 2>/dev/null | grep -E '^(version|license):' || true; fi

section "GDS capability"
GDSCHECK=""
for candidate in \
  "${CUDA_HOME:-}/tools/gdscheck.py" \
  "${CUDA_HOME:-}/gds/tools/gdscheck.py" \
  /usr/local/cuda/gds/tools/gdscheck.py
do
  [[ -x "$candidate" ]] && { GDSCHECK="$candidate"; break; }
done
if [[ -z "$GDSCHECK" ]] && has gdscheck.py; then
  GDSCHECK="$(command -v gdscheck.py)"
fi
if [[ -z "$GDSCHECK" ]]; then
  GDSCHECK="$(find /usr/local -path '*/gds/tools/gdscheck.py' -type f 2>/dev/null | head -1 || true)"
fi
if [[ -n "$GDSCHECK" ]]; then
  "$GDSCHECK" -p || true
else
  echo "gdscheck.py: not found"
fi

section "PCIe topology"
if has nvidia-smi; then nvidia-smi topo -m || true; fi
if has lspci; then
  echo
  lspci | grep -Ei 'NVIDIA|Non-Volatile memory|NVMe' || true
fi
echo
if [[ -d /sys/class/iommu ]] && [[ -n "$(ls -A /sys/class/iommu 2>/dev/null)" ]]; then
  echo "IOMMU devices: $(ls /sys/class/iommu | tr '\n' ' ')"
else
  echo "IOMMU devices: none"
fi
cmdline_iommu="$(tr ' ' '\n' < /proc/cmdline | grep -E '^(intel_iommu|amd_iommu|iommu)=' || true)"
if [[ -n "$cmdline_iommu" ]]; then
  echo "kernel cmdline: $(printf '%s' "$cmdline_iommu" | tr '\n' ' ')"
else
  echo "kernel cmdline: no iommu=/intel_iommu=/amd_iommu="
fi
echo "GDS implication: see gdscheck.py IOMMU State (run gdscheck.py -p)"

section "NVMe"
if has nvme; then nvme list || true; else echo "nvme-cli: not installed"; fi

section "Host memory / page cache budget"
MEM_TOTAL_KB="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)"
MEM_TOTAL=$((MEM_TOTAL_KB * 1024))
grep -E '^(MemTotal|MemAvailable|SwapTotal|Dirty|Writeback):' /proc/meminfo
echo "MemTotal: $(gib "$MEM_TOTAL") GiB"
if [[ -f "$TARGET" ]]; then
  DATASET_BYTES="$(stat -c '%s' "$TARGET")"
  echo "dataset:  $TARGET  $(gib "$DATASET_BYTES") GiB"
  if (( DATASET_BYTES < MEM_TOTAL )); then
    echo "WARNING: dataset ($(gib "$DATASET_BYTES") GiB) is smaller than RAM ($(gib "$MEM_TOTAL") GiB)."
    echo "         Repeated buffered reads will be served from the page cache, so pageable/"
    echo "         pinned/overlap numbers measure DRAM, not NVMe. Use --backend direct for the"
    echo "         storage path, or drop caches between buffered runs and record that you did."
  else
    echo "dataset exceeds RAM: buffered reads cannot be fully cached."
  fi
else
  echo "dataset:  $TARGET (not created yet; make dataset SIZE_GIB=N)"
  echo "NOTE: with $(gib "$MEM_TOTAL") GiB of RAM, any dataset smaller than that is page-cache"
  echo "      resident after the first pass. See docs/linux-page-cache.md."
fi

section "Filesystem for target"
if has findmnt; then findmnt -T "$TARGET" || findmnt -T "$(dirname "$TARGET")" || true; fi
df -h "$(dirname "$TARGET")" 2>/dev/null

section "Block device queue"
if has findmnt; then
  SRC="$(findmnt -nro SOURCE -T "$TARGET" 2>/dev/null || findmnt -nro SOURCE -T "$(dirname "$TARGET")" 2>/dev/null || true)"
  if [[ -b "$SRC" ]]; then
    DISK="$(lsblk -nro PKNAME "$SRC" 2>/dev/null | head -1)"
    [[ -z "$DISK" ]] && DISK="$(basename "$SRC")"
    QUEUE="/sys/block/$DISK/queue"
    if [[ -d "$QUEUE" ]]; then
      echo "device: $SRC (disk $DISK)"
      for knob in scheduler nr_requests max_hw_sectors_kb max_sectors_kb logical_block_size physical_block_size rotational; do
        [[ -r "$QUEUE/$knob" ]] && printf '  %-22s %s\n' "$knob" "$(cat "$QUEUE/$knob")"
      done
      echo "NOTE: the kernel splits application reads at max_hw_sectors_kb. A 64 MiB chunk is"
      echo "      not a 64 MiB device request; experiment 04 measures the syscall/queue knee."
    fi
  fi
fi

section "Tooling"
for tool in cmake nvcc nsys ncu fio nvme iostat pidstat numactl gdscheck.py gds_stats gdsio; do
  if has "$tool"; then
    printf '  %-12s %s\n' "$tool" "$(command -v "$tool")"
  else
    printf '  %-12s MISSING\n' "$tool"
  fi
done
if has python3; then
  python3 - <<'PY' 2>/dev/null || echo "  torch    not importable"
import importlib.util
spec = importlib.util.find_spec("torch")
if spec is None:
    print("  torch    not installed (experiment 08 needs it)")
else:
    import torch
    ok = torch.cuda.is_available()
    print(f"  torch    {torch.__version__} cuda={torch.version.cuda} available={ok}")
    if not ok:
        print("           WARNING: CPU-only build. Experiment 08 cannot measure H2D/pin_memory.")
PY
fi

section "Useful next commands"
echo "iostat -xz 1"
echo "nvidia-smi dmon"
echo "pidstat -dru 1"
echo "CUFILE_FORCE_COMPAT_MODE=true ..."
