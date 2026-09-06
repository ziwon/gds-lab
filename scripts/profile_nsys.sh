#!/usr/bin/env bash
set -euo pipefail

FILE="${1:-/data/gds-lab.bin}"
BACKEND="${2:-overlap}"
BYTES="${BYTES:-1G}"
CHUNK_BYTES="${CHUNK_BYTES:-64M}"
BIN="${GDSLAB_BIN:-./build/gds_lab}"
OUT="${OUT:-results/nsys-${BACKEND}}"

if ! command -v nsys >/dev/null 2>&1; then
  echo "nsys not found" >&2
  exit 3
fi
mkdir -p results

ARGS=(--backend "$BACKEND" --file "$FILE" --bytes "$BYTES")
if [[ "$BACKEND" == "overlap" ]]; then ARGS+=(--chunk-bytes "$CHUNK_BYTES"); fi

nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=process-tree \
  --force-overwrite=true \
  -o "$OUT" \
  "$BIN" "${ARGS[@]}"

echo "Nsight Systems output: ${OUT}.nsys-rep"
