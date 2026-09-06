#!/usr/bin/env bash
# Install a user-local CUDA 13.0.2 compiler+runtime+cuFile from NVIDIA redist
# archives. Does not touch the NVIDIA driver or Ubuntu's /usr/bin/nvcc.
set -euo pipefail

# CUDA_HOME is a compiler location, not an install destination.
PREFIX="${1:-${GDSLAB_CUDA_PREFIX:-/data/cuda-13.0}}"
RELEASE="${CUDA_REDIST_RELEASE:-13.0.2}"
BASE="https://developer.download.nvidia.com/compute/cuda/redist"
# Keep the download cache off any filesystem under measurement.
WORKDIR="${CUDA_REDIST_WORKDIR:-${XDG_CACHE_HOME:-$HOME/.cache}/gds-lab/cuda-redist-$RELEASE}"

resolved="$(readlink -f "$PREFIX" 2>/dev/null || printf '%s' "$PREFIX")"
case "$resolved" in
  /usr|/usr/*)
    echo "refusing to install into system prefix: $PREFIX" >&2
    echo "pass a user-writable path, e.g. $0 /data/cuda-13.0" >&2
    exit 2
    ;;
esac
if [[ -x "$PREFIX/bin/nvcc" ]]; then
  if [[ ! -f "$PREFIX/version.txt" ]] || ! grep -qx "CUDA Version $RELEASE" "$PREFIX/version.txt"; then
    echo "refusing to overlay existing nvcc at $PREFIX/bin/nvcc" >&2
    echo "it is not the CUDA $RELEASE redist this script installs." >&2
    exit 2
  fi
fi

mkdir -p "$PREFIX" "$WORKDIR"

# component|relative_path|sha256
# Compiler + runtime + cuFile. No driver, no Nsight.
# libcufile also carries include/cufile.h and tools/ (gdscheck.py, gds_stats,
# gdsio), which experiments 00 and 06 need. It does not provide nvidia-fs, so
# cuFile still runs in compatibility mode on an unsupported host.
COMPONENTS=$(cat <<'EOF'
cuda_cccl|cuda_cccl/linux-x86_64/cuda_cccl-linux-x86_64-13.0.85-archive.tar.xz|ed845eae8c1767706b6ee91e40c608a03f6f633551a849b63f7346d32d73ee60
cuda_crt|cuda_crt/linux-x86_64/cuda_crt-linux-x86_64-13.0.88-archive.tar.xz|5a3279a049ffc1cdb951c44cb95206acfdde9e9ae5e87825fc18d7e4a6878bb0
cuda_cudart|cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-13.0.96-archive.tar.xz|25b8071951baba827be1580b841d363464f6ee6c39f48d33a81646f90cc95ed1
cuda_culibos|cuda_culibos/linux-x86_64/cuda_culibos-linux-x86_64-13.0.85-archive.tar.xz|98fca11f9ab89be61385a67eee686c1ebf76fad9c7269d0cf9090edfcdb8a8ac
cuda_nvcc|cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-13.0.88-archive.tar.xz|48e35be3cfbf4b4fbc16828eaec8a7048ee789403049dc409f7b643d6259cf7a
libnvjitlink|libnvjitlink/linux-x86_64/libnvjitlink-linux-x86_64-13.0.88-archive.tar.xz|25f9763fd60122a4f728eec22ac4e64e46ea8679e140234f15eaec008e6c41a8
libnvptxcompiler|libnvptxcompiler/linux-x86_64/libnvptxcompiler-linux-x86_64-13.0.88-archive.tar.xz|3d2a51c6816278b90167550a7e0e9adfff9c8c919d87ef980565c0eb7fc23830
libnvvm|libnvvm/linux-x86_64/libnvvm-linux-x86_64-13.0.88-archive.tar.xz|17ef1665b63670887eeba7d908da5669fa8c66bb73b5b4c1367f49929c086353
libcufile|libcufile/linux-x86_64/libcufile-linux-x86_64-1.15.1.6-archive.tar.xz|2c1f544ab1b0e215590d943cbd1b0a192a066e9b7592e605c2e28b22e886688f
EOF
)

sha256_ok() {
  local file="$1" expect="$2"
  local actual
  actual="$(sha256sum "$file" | awk '{print $1}')"
  [[ "$actual" == "$expect" ]]
}

echo "Installing CUDA $RELEASE redist into $PREFIX"

while IFS='|' read -r name rel sha; do
  [[ -z "$name" ]] && continue
  archive="$WORKDIR/$(basename "$rel")"
  url="$BASE/$rel"
  if [[ -f "$archive" ]] && sha256_ok "$archive" "$sha"; then
    echo "cached $name"
  else
    echo "download $name"
    curl -fL --retry 3 --retry-delay 2 -o "$archive.partial" "$url"
    mv "$archive.partial" "$archive"
    sha256_ok "$archive" "$sha" || {
      echo "checksum mismatch for $archive" >&2
      exit 1
    }
  fi
  extract="$WORKDIR/extract-$name"
  rm -rf "$extract"
  mkdir -p "$extract"
  tar -xJf "$archive" -C "$extract"
  inner="$(find "$extract" -mindepth 1 -maxdepth 1 -type d | head -1)"
  cp -a "$inner"/. "$PREFIX/"
  rm -rf "$extract"
done <<<"$COMPONENTS"

if [[ ! -f "$PREFIX/version.txt" ]]; then
  printf 'CUDA Version %s\n' "$RELEASE" > "$PREFIX/version.txt"
fi
# Redist archives ship lib/, not lib64/. Only link when lib64 is absent:
# `ln -sfn lib lib64` against an existing directory would create lib64/lib.
if [[ ! -e "$PREFIX/lib64" ]]; then
  ln -sfn lib "$PREFIX/lib64"
fi

if [[ ! -x "$PREFIX/bin/nvcc" ]]; then
  echo "nvcc missing after install: $PREFIX/bin/nvcc" >&2
  exit 1
fi

echo
"$PREFIX/bin/nvcc" --version
echo "OK: $PREFIX"
echo "Use: source scripts/cuda_env.sh"
