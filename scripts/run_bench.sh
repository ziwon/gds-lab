#!/usr/bin/env bash
set -u

FILE="${1:-/data/gds-lab.bin}"
BYTES="${2:-1G}"
CHUNK_BYTES="${3:-64M}"
BIN="${GDSLAB_BIN:-./build/gds_lab}"

if [[ ! -x "$BIN" ]]; then
  echo "Missing benchmark binary: $BIN" >&2
  echo "Run: make build" >&2
  exit 2
fi
if [[ ! -f "$FILE" ]]; then
  echo "Missing dataset: $FILE" >&2
  exit 2
fi

run() {
  echo
  echo ">>> $*"
  "$@"
}

run "$BIN" --backend pageable --file "$FILE" --bytes "$BYTES" || true
run "$BIN" --backend pinned   --file "$FILE" --bytes "$BYTES" || true
run "$BIN" --backend direct   --file "$FILE" --bytes "$BYTES" || true
run "$BIN" --backend overlap  --file "$FILE" --bytes "$BYTES" --chunk-bytes "$CHUNK_BYTES" || true

echo
echo ">>> cuFile compatibility probe/benchmark"
CUFILE_ENV_PATH_JSON="${CUFILE_ENV_PATH_JSON:-$PWD/configs/cufile.json}" \
CUFILE_FORCE_COMPAT_MODE=true \
  "$BIN" --backend cufile --file "$FILE" --bytes "$BYTES" || true

echo
echo "NOTE: buffered runs after the first access may be served partly or fully from Linux page cache."
