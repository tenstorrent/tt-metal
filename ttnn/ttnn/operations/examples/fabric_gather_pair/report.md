# fabric_gather_pair — device report

Illustrative, not a CI bound. Metric: `DEVICE KERNEL DURATION [ns]` (in-process device profiler), the slowest
chip of each launch, median of 3. GB/s per link-dir = (shard bytes / links) / kernel ns; GB/s per chip = shard
bytes / kernel ns (bytes each chip sends, and also receives). Correctness (every chip's output ==
[row-0 shard ; row-1 shard] of its column pair) passed for every row.

## Blackhole — `bh-qb-11-special-mstaletovic-for-reservation-93463` · 4× p150a (2×2 mesh, axis-0 pairs) · 2026-09-29 · `05d3e4d94b9`+

FABRIC_1D, 14,336 B router payload; shard 8192×4096 bf16 (64 MiB) per chip in DRAM; one core per link (reader on
RISCV_1 / NoC0, sender on RISCV_0 / NoC1); CB = 2 groups of chunks (≤ 112 KiB); links split the 8 DRAM banks.
`+copy:copyN`: the local copy is made by N separate copy cores (copy1 = logical (0,6), copy2 = (0,6),(1,6)), each
reading its share of the banks from DRAM and writing it to the local output; the link cores then only read and send.

```
placement        NoC coords      variant                      kernel ns  GB/s per link-dir  GB/s per chip
1link_under_eth  (3,2)           page_per_packet                5213279              12.87          12.87
1link_under_eth  (3,2)           page_per_packet+local_noc0     5280653              12.71          12.71
1link_under_eth  (3,2)           bank_run                       2060116              32.58          32.58
1link_under_eth  (3,2)           bank_run+local_noc0            1670643              40.17          40.17
1link_under_eth  (3,2)           page_per_packet+copy:copy1     4938313              13.59          13.59
1link_under_eth  (3,2)           page_per_packet+copy:copy2     4937827              13.59          13.59
1link_under_eth  (3,2)           bank_run+copy:copy1            1391934              48.21          48.21
1link_under_eth  (3,2)           bank_run+copy:copy2            1391842              48.22          48.22
2link_adjacent   (1,2) (2,2)     page_per_packet                2754925              12.18          24.36
2link_adjacent   (1,2) (2,2)     page_per_packet+local_noc0     2766005              12.13          24.26
2link_adjacent   (1,2) (2,2)     bank_run                       1283573              26.14          52.28
2link_adjacent   (1,2) (2,2)     bank_run+local_noc0            1217084              27.57          55.14
2link_adjacent   (1,2) (2,2)     page_per_packet+copy:copy1     2755721              12.18          24.35
2link_adjacent   (1,2) (2,2)     page_per_packet+copy:copy2     2748857              12.21          24.41
2link_adjacent   (1,2) (2,2)     bank_run+copy:copy1            1535416              21.85          43.71
2link_adjacent   (1,2) (2,2)     bank_run+copy:copy2            1137721              29.49          58.99
2link_under_eth  (3,2) (4,2)     page_per_packet                2645730              12.68          25.36
2link_under_eth  (3,2) (4,2)     page_per_packet+local_noc0     2687667              12.48          24.97
2link_under_eth  (3,2) (4,2)     bank_run                       1170825              28.66          57.32
2link_under_eth  (3,2) (4,2)     bank_run+local_noc0            1295895              25.89          51.79
2link_under_eth  (3,2) (4,2)     page_per_packet+copy:copy1     2717119              12.35          24.70
2link_under_eth  (3,2) (4,2)     page_per_packet+copy:copy2     2498376              13.43          26.86
2link_under_eth  (3,2) (4,2)     bank_run+copy:copy1            1342473              24.99          49.99
2link_under_eth  (3,2) (4,2)     bank_run+copy:copy2             706547              47.49          94.98
```

Ablations (1 link, `bank_run`, local copy on NoC1, trials=1; output incomplete, correctness not checked):

```
full op                          33.56 GB/s
without local copy               48.22
without fabric send              51.68
without DRAM read                34.52
without local copy + fabric      53.58
without DRAM read + local copy   48.23
```

Bank order (1 link, `bank_run`, trials=1): walking one bank at a time (all runs of bank b, then b+1)
**19.99 GB/s**; round-robin over the link's banks **33.70 GB/s**.

Copy-core count and placement (2 links under their Ethernet cores, `bank_run`, trials=1):

```
copy cores                          GB/s per link-dir   GB/s per chip
none (link core copies)                  28.49              56.99
2  (0,6),(1,6)                           47.58              95.16
4  (0..3,6)                              41.65              83.30
4  (0..3,9)                              41.45              82.91
4  column (6,3..6)                       36.03              72.06
4  spread (0,4),(4,6),(8,4),(9,7)        41.71              83.42
8  (0..7,6)                              37.76              75.52
```
