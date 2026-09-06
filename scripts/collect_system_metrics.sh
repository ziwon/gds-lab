#!/usr/bin/env bash
set -u

DURATION="${1:-5}"

echo "Collecting a ${DURATION}s snapshot. Run this in another terminal during a benchmark."

if command -v iostat >/dev/null 2>&1; then
  echo
  echo "== iostat =="
  timeout "${DURATION}s" iostat -xz 1 || true
else
  echo "iostat not found (package: sysstat)"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  echo
  echo "== nvidia-smi dmon =="
  timeout "${DURATION}s" nvidia-smi dmon -s pucvmet -d 1 || true
fi
