#!/usr/bin/env python3
"""Write a deterministic finite float32 file for gds-lab datasets.

/dev/zero makes experiment 08's checksum column identically 0.0, so a
wrong-offset read looks healthy. Raw random bytes reinterpreted as float32
produce NaN/Inf and poison the sum. Each 4-byte word is a finite value in
[1, 2) derived from its index, so sequential checksums change if the loader
returns the wrong bytes.
"""

from __future__ import annotations

import os
import sys

import numpy as np

CHUNK_BYTES = 16 * 1024 * 1024
GOLDEN = np.uint32(2654435761)


def fill_words(index0: int, nwords: int) -> np.ndarray:
    idx = np.arange(index0, index0 + nwords, dtype=np.uint32)
    mixed = idx * GOLDEN
    frac = (mixed >> 8).astype(np.float32) * np.float32(2.0**-24)
    return np.float32(1.0) + frac


def write_dataset(path: str, nbytes: int) -> None:
    if nbytes < 4 or nbytes % 4:
        raise SystemExit("byte count must be a positive multiple of 4")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    written = 0
    try:
        while written < nbytes:
            n = min(CHUNK_BYTES, nbytes - written)
            values = fill_words(written // 4, n // 4)
            os.write(fd, values.tobytes())
            written += n
            print(f"\r{written / nbytes:.1%}", end="", file=sys.stderr, flush=True)
        os.fsync(fd)
    finally:
        os.close(fd)
        print(file=sys.stderr)


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} PATH BYTES", file=sys.stderr)
        return 2
    write_dataset(sys.argv[1], int(sys.argv[2]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
