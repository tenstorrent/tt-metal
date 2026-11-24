# width_split — Layout::ROW_MAJOR tilize, no combine: k column groups per tile-row

box=bh-50-special-adamjanovic-for-reservation-177924  arch=BH  grid=13x10  clock=unknown MHz  N=3 (median)  kernel-iters=1 (per-launch latency)  date=2026-11-24

Interleaved DRAM ROW_MAJOR bf16 [32 R_t, W] in, interleaved DRAM TILE out. k = 1: whole tile-rows per Tensix core; k > 1: each tile-row split into k column groups (pieces of 2W/k bytes per stick). ns per k (helper defaults); best = fastest k and its speedup over k = 1. trim = the tilize helper with NoReconfigure + InitOnly (no reconfig after compute_kernel_hw_startup, no trailing uninit), re-measured interleaved against the default at k = 1 and at the best k.

| R_t | W | stick B | k=1 | k=2 | k=4 | k=8 | k=16 | k=32 | k=64 | best k | trim at k=1: default -> trimmed ns | trim at best k | k=1 default -> best k trimmed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---:|
| 1 | 64 | 128 | 1799 | 1733 |  |  |  |  |  | k=2 (1.04x, 64 B pieces, 2 Tensix cores) | 1810 -> 1835 (0.99x) | 1764 -> 1732 (1.02x) | 1.05x |
| 1 | 128 | 256 | 1954 | 1838 | 1792 |  |  |  |  | k=4 (1.09x, 64 B pieces, 4 Tensix cores) | 1920 -> 1948 (0.99x) | 1781 -> 1792 (0.99x) | 1.07x |
| 1 | 256 | 512 | 2190 | 1956 | 1864 | 1790 |  |  |  | k=8 (1.22x, 64 B pieces, 8 Tensix cores) | 2141 -> 2176 (0.98x) | 1801 -> 1774 (1.02x) | 1.21x |
| 1 | 512 | 1024 | 2589 | 2175 | 1993 | 1899 | 1903 |  |  | k=8 (1.36x, 128 B pieces, 8 Tensix cores) | 2579 -> 2559 (1.01x) | 1910 -> 1894 (1.01x) | 1.36x |
| 1 | 2048 | 4096 | 4746 | 3539 | 2686 | 2392 | 2203 | 2679 | 4364 | k=16 (2.15x, 256 B pieces, 16 Tensix cores) | 4761 -> 4756 (1.00x) | 2211 -> 2208 (1.00x) | 2.16x |
| 1 | 8192 | 16384 | 12493 | 7437 | 5036 | 5145 | 4482 | 4287 | 5698 | k=32 (2.91x, 512 B pieces, 32 Tensix cores) | 12483 -> 12500 (1.00x) | 4405 -> 4279 (1.03x) | 2.92x |
| 2 | 64 | 128 | 1832 | 1796 |  |  |  |  |  | k=2 (1.02x, 64 B pieces, 4 Tensix cores) | 1824 -> 1859 (0.98x) | 1793 -> 1774 (1.01x) | 1.03x |
| 2 | 128 | 256 | 1972 | 1871 | 1811 |  |  |  |  | k=4 (1.09x, 64 B pieces, 8 Tensix cores) | 1972 -> 1967 (1.00x) | 1811 -> 1801 (1.01x) | 1.09x |
| 2 | 256 | 512 | 2189 | 2009 | 1889 | 1913 |  |  |  | k=4 (1.16x, 128 B pieces, 8 Tensix cores) | 2205 -> 2193 (1.01x) | 1896 -> 1899 (1.00x) | 1.16x |
| 2 | 1024 | 2048 | 3499 | 2653 | 2413 | 2201 | 2683 | 4392 |  | k=8 (1.59x, 256 B pieces, 16 Tensix cores) | 3484 -> 3464 (1.01x) | 2274 -> 2215 (1.03x) | 1.57x |
| 4 | 64 | 128 | 1849 | 1812 |  |  |  |  |  | k=2 (1.02x, 64 B pieces, 8 Tensix cores) | 1861 -> 1847 (1.01x) | 1803 -> 1801 (1.00x) | 1.03x |
| 4 | 256 | 512 | 2218 | 2062 | 2004 | 2592 |  |  |  | k=4 (1.11x, 128 B pieces, 16 Tensix cores) | 2217 -> 2204 (1.01x) | 2040 -> 2001 (1.02x) | 1.11x |
| 4 | 2048 | 4096 | 4917 | 4794 | 4303 | 4294 | 5771 | 9057 |  | k=8 (1.15x, 512 B pieces, 32 Tensix cores) | 4901 -> 4873 (1.01x) | 4350 -> 4368 (1.00x) | 1.12x |
| 16 | 128 | 256 | 2246 | 2724 | 4481 |  |  |  |  | k=1 (1.00x, 256 B pieces, 16 Tensix cores) | 2207 -> 2187 (1.01x) | 2241 -> 2182 (1.03x) | 1.01x |
| 16 | 512 | 1024 | 4335 | 4283 | 5627 | 9204 |  |  |  | k=2 (1.01x, 512 B pieces, 32 Tensix cores) | 4353 -> 4330 (1.01x) | 4247 -> 4359 (0.97x) | 1.00x |
| 16 | 2048 | 4096 | 11964 | 12296 | 12530 | 15653 |  |  |  | k=1 (1.00x, 4096 B pieces, 16 Tensix cores) | 11749 -> 11830 (0.99x) | 11721 -> 11968 (0.98x) | 0.98x |
| 47 | 256 | 512 | 5918 | 9038 |  |  |  |  |  | k=1 (1.00x, 512 B pieces, 47 Tensix cores) | 5783 -> 5710 (1.01x) | 5959 -> 5766 (1.03x) | 1.00x |
| 64 | 256 | 512 | 7716 | 11941 |  |  |  |  |  | k=1 (1.00x, 512 B pieces, 64 Tensix cores) | 7522 -> 7766 (0.97x) | 7556 -> 7793 (0.97x) | 0.97x |
| 64 | 2048 | 4096 | 44001 | 45670 |  |  |  |  |  | k=1 (1.00x, 4096 B pieces, 64 Tensix cores) | 45116 -> 43336 (1.04x) | 44074 -> 43945 (1.00x) | 1.03x |
