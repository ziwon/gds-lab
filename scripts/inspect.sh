#!/usr/bin/env bash
set -u

TARGET="${1:-.}"

section() { printf '\n== %s ==\n' "$1"; }
has() { command -v "$1" >/dev/null 2>&1; }

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
if has nvcc; then nvcc --version | tail -n 4; else echo "nvcc: not found"; fi
if ldconfig -p 2>/dev/null | grep -q libcufile; then
  ldconfig -p | grep libcufile || true
else
  echo "libcufile: not visible through ldconfig"
fi
for header in /usr/local/cuda/include/cufile.h /usr/include/cufile.h; do
  [[ -f "$header" ]] && echo "cufile header: $header"
done

section "nvidia-fs / NVIDIA kernel module"
lsmod | grep -E '^nvidia_fs\b' || echo "nvidia-fs: not loaded"
if has modinfo; then modinfo nvidia 2>/dev/null | grep -E '^(version|license):' || true; fi

section "GDS capability"
GDSCHECK=""
if has gdscheck.py; then
  GDSCHECK="$(command -v gdscheck.py)"
elif [[ -x /usr/local/cuda/gds/tools/gdscheck.py ]]; then
  GDSCHECK=/usr/local/cuda/gds/tools/gdscheck.py
else
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

section "NVMe"
if has nvme; then nvme list || true; else echo "nvme-cli: not installed"; fi

section "Filesystem for target"
if has findmnt; then findmnt -T "$TARGET" || findmnt -T "$(dirname "$TARGET")" || true; fi

section "Useful next commands"
echo "iostat -xz 1"
echo "nvidia-smi dmon"
echo "pidstat -dru 1"
echo "CUFILE_FORCE_COMPAT_MODE=true ..."
