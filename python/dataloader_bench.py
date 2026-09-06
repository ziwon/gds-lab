#!/usr/bin/env python3
"""File-backed PyTorch DataLoader sweep for GPU starvation experiments."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class BinaryTensorDataset(Dataset):
    def __init__(self, path: Path, sample_bytes: int) -> None:
        if sample_bytes <= 0 or sample_bytes % 4:
            raise ValueError("sample_bytes must be a positive multiple of 4")
        self.path = path
        self.sample_bytes = sample_bytes
        self.length = path.stat().st_size // sample_bytes
        if self.length < 1:
            raise ValueError("file is smaller than one sample")
        self._fd = None

    def __len__(self) -> int:
        return self.length

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_fd"] = None
        return state

    def _fileno(self) -> int:
        if self._fd is None:
            self._fd = os.open(self.path, os.O_RDONLY)
        return self._fd

    def __getitem__(self, index: int):
        # pread does not use the kernel file offset, so a handle inherited
        # across fork (workers=0 then num_workers>0) cannot interleave seeks.
        raw = os.pread(self._fileno(), self.sample_bytes, index * self.sample_bytes)
        if len(raw) != self.sample_bytes:
            raise RuntimeError("short read")
        array = np.frombuffer(raw, dtype=np.float32).copy()
        return torch.from_numpy(array), index % 10


def make_loader(dataset: Dataset, workers: int, pin_memory: bool, prefetch: int, random_access: bool):
    kwargs = dict(
        dataset=dataset,
        batch_size=32,
        shuffle=random_access,
        num_workers=workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    if workers > 0:
        kwargs["prefetch_factor"] = prefetch
        kwargs["persistent_workers"] = True
    return DataLoader(**kwargs)


def bench(dataset: Dataset, workers: int, pin_memory: bool, prefetch: int, batches: int, random_access: bool):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = make_loader(dataset, workers, pin_memory, prefetch, random_access)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    seen = 0
    accumulator = torch.zeros((), device=device)

    for batch_index, (inputs, _targets) in enumerate(loader):
        inputs = inputs.to(device, non_blocking=pin_memory and device.type == "cuda")
        accumulator = accumulator + inputs.sum()
        seen += inputs.shape[0]
        if batch_index + 1 >= batches:
            break

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return seen / elapsed, elapsed, float(accumulator.item())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--file", type=Path, required=True)
    p.add_argument("--workers", default="0,2,4,8", help="comma-separated worker counts")
    p.add_argument("--sample-bytes", type=int, default=3 * 224 * 224 * 4)
    p.add_argument("--prefetch", type=int, default=2)
    p.add_argument("--batches", type=int, default=100)
    p.add_argument("--no-pin-memory", action="store_true")
    p.add_argument("--random-access", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.file.exists():
        raise SystemExit(f"missing file: {args.file}")

    dataset = BinaryTensorDataset(args.file, args.sample_bytes)
    workers = [int(x) for x in args.workers.split(",") if x.strip()]
    pin_memory = not args.no_pin_memory

    print(f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    print(f"file={args.file} size={os.path.getsize(args.file)} samples={len(dataset)}")
    print(f"pin_memory={pin_memory} prefetch={args.prefetch} random_access={args.random_access}")
    print("workers,samples_per_second,seconds,checksum")

    for worker_count in workers:
        rate, elapsed, checksum = bench(
            dataset, worker_count, pin_memory, args.prefetch, args.batches, args.random_access
        )
        print(f"{worker_count},{rate:.2f},{elapsed:.4f},{checksum:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
