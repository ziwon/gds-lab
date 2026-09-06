# 05 - Sequential vs random

Do not emulate random-storage latency with `sleep()`. This experiment will use real randomized file offsets so the filesystem/NVMe path is actually exercised.
