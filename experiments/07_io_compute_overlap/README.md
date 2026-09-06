# 07 - I/O + compute overlap

Profile the `overlap` backend with Nsight Systems and look for whether reads/H2D work leave visible gaps between GPU kernels. The goal is pipeline utilization, not a single headline GB/s number.
