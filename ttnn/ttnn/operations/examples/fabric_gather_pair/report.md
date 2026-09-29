# fabric_gather_pair — device report

Illustrative, not a CI bound. Metric: `DEVICE KERNEL DURATION [ns]` (in-process device profiler), the slowest
chip of each launch, median of 3. GB/s per link-dir = (shard bytes / links) / kernel ns; GB/s per chip = shard
bytes / kernel ns (bytes each chip sends, and also receives). Correctness (every chip's output ==
[row-0 shard ; row-1 shard] of its column pair) passed for every row.

## Blackhole — `bh-qb-11-special-mstaletovic-for-reservation-93463` · 4× p150a (2×2 mesh, axis-0 pairs) · 2026-09-29 · `05d3e4d94b9`+

FABRIC_1D, 14,336 B router payload; shard 8192×4096 bf16 (64 MiB) per chip in DRAM; one core per link (reader on
RISCV_1 / NoC0, sender on RISCV_0 / NoC1); CB = 2 groups of chunks (≤ 112 KiB); links split the 8 DRAM banks.

```
placement        NoC coords      variant                      kernel ns  GB/s per link-dir  GB/s per chip
1link_under_eth  (3,2)           page_per_packet                5232988              12.82          12.82
1link_under_eth  (3,2)           page_per_packet+local_noc0     5279212              12.71          12.71
1link_under_eth  (3,2)           bank_run                       2061950              32.55          32.55
1link_under_eth  (3,2)           bank_run+local_noc0            1664686              40.31          40.31
2link_adjacent   (1,2) (2,2)     page_per_packet                2753384              12.19          24.37
2link_adjacent   (1,2) (2,2)     page_per_packet+local_noc0     2765502              12.13          24.27
2link_adjacent   (1,2) (2,2)     bank_run                       1281630              26.18          52.36
2link_adjacent   (1,2) (2,2)     bank_run+local_noc0            1275223              26.31          52.63
2link_under_eth  (3,2) (4,2)     page_per_packet                2640856              12.71          25.41
2link_under_eth  (3,2) (4,2)     page_per_packet+local_noc0     2681452              12.51          25.03
2link_under_eth  (3,2) (4,2)     bank_run                       1168080              28.73          57.45
2link_under_eth  (3,2) (4,2)     bank_run+local_noc0            1296919              25.87          51.74
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
