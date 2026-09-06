#!/usr/bin/env bash
set -euo pipefail

FILE="${1:-/data/gds-lab.bin}"
SIZE_GIB="${2:-4}"

if ! [[ "$SIZE_GIB" =~ ^[0-9]+$ ]] || [[ "$SIZE_GIB" -lt 1 ]]; then
  echo "SIZE_GIB must be a positive integer" >&2
  exit 2
fi

mkdir -p "$(dirname "$FILE")"
BYTES=$((SIZE_GIB * 1024 * 1024 * 1024))

echo "Writing ${SIZE_GIB} GiB to $FILE ..."
dd if=/dev/zero of="$FILE" bs=16M count=$((SIZE_GIB * 64)) status=progress conv=fdatasync

ACTUAL=$(stat -c '%s' "$FILE")
if [[ "$ACTUAL" -ne "$BYTES" ]]; then
  echo "Unexpected file size: got $ACTUAL expected $BYTES" >&2
  exit 1
fi

echo "OK: $FILE ($ACTUAL bytes)"
