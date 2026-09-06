#!/usr/bin/env bash
set -u

FILE="${1:-/data/gds-lab.bin}"
BYTES="${2:-1G}"
CHUNK_BYTES="${3:-64M}"
READ_CHUNK="${READ_CHUNK:-16M}"
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

LOG="$(mktemp)"
trap 'rm -f "$LOG"' EXIT

run() {
  echo
  echo ">>> $*"
  "$@" | tee -a "$LOG"
}

run "$BIN" --backend pageable --file "$FILE" --bytes "$BYTES" --read-chunk "$READ_CHUNK" || true
run "$BIN" --backend pinned   --file "$FILE" --bytes "$BYTES" --read-chunk "$READ_CHUNK" || true
run "$BIN" --backend direct   --file "$FILE" --bytes "$BYTES" --read-chunk "$READ_CHUNK" || true
run "$BIN" --backend overlap  --file "$FILE" --bytes "$BYTES" --chunk-bytes "$CHUNK_BYTES" || true
run "$BIN" --backend overlap  --file "$FILE" --bytes "$BYTES" --chunk-bytes "$CHUNK_BYTES" --direct || true

echo
echo ">>> cuFile compatibility probe/benchmark"
CUFILE_ENV_PATH_JSON="${CUFILE_ENV_PATH_JSON:-$PWD/configs/cufile.json}" \
CUFILE_FORCE_COMPAT_MODE=true \
  "$BIN" --backend cufile --file "$FILE" --bytes "$BYTES" --read-chunk "$READ_CHUNK" \
  | tee -a "$LOG" || true

# Every backend reads the same region and sums the same bytes on the GPU, so a
# differing checksum means a backend did not deliver the data it reported.
echo
CHECKSUMS="$(grep -o 'checksum=[0-9]*' "$LOG" | sort -u)"
COUNT="$(printf '%s\n' "$CHECKSUMS" | grep -c . || true)"
if [[ "$COUNT" -gt 1 ]]; then
  echo "ERROR: backends disagree on the data they transferred:" >&2
  printf '%s\n' "$CHECKSUMS" >&2
  exit 1
fi
echo "checksum agreement: OK (${CHECKSUMS:-none reported})"

echo
echo "NOTE: buffered runs after the first access may be served partly or fully from Linux page cache."
