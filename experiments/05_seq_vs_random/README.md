# 05 - Sequential vs random

`--random-access` visits the `--read-chunk`-sized blocks of the region in shuffled order (`--seed` selects the permutation). Real file offsets are used; nothing emulates latency with `sleep()`.

```bash
./build/gds_lab --backend direct --file /data/gds-lab.bin --bytes 1G --read-chunk 1M
./build/gds_lab --backend direct --file /data/gds-lab.bin --bytes 1G --read-chunk 1M --random-access
```

Each block still lands at its own position in the destination buffer, so **the checksum must be identical to the sequential run**. A different checksum means the randomized path read the wrong bytes, not that random access is slower.

Sweep `--read-chunk` together with `--random-access`: the gap between sequential and random closes as the block size grows, and where it closes is the more interesting result than either number alone. `--backend overlap --random-access` shuffles the pipeline's chunk order for the same comparison under overlap.
