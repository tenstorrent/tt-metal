# bank_stagger — measured report

| stamp | value |
|---|---|
| box | `bh-qbge-09-special-dstoiljkovic-for-reservation-117054` |
| arch | Blackhole p300c (11×10 = 110 compute grid, 8 DRAM banks) |
| date | 2026-10-05 |
| metric | `DEVICE KERNEL DURATION [ns]`, read in-process (`ttnn.ReadDeviceProfiler` + `ttnn.get_latest_programs_perf_data`) |
| method | 3 warmup launches per variant, then 7 trials × 10 launches, variants interleaved per trial, median of trials; every variant checked bit-exact first |

> Illustrative, not a CI bound. Add another arch as a new block rather than replacing this one.

Op: tilize bf16, `ROW_MAJOR` width-sharded DRAM (one shard per bank) → `TILE` interleaved DRAM,
units of 32 rows × `chunk` tiles. `nt_h` = tile-rows, `n_w` = units per tile-row (= shards here),
`blk` = blocks per core. `spread` = (max − min) / median over trials.

## Blackhole (N=7, kernel-iters=1)

```
         shape chunk read B nt_h  n_w cores blk  variant        ns  spread  vs none
   3520x1024       4    256  110    8   110   8  none       101361    2.1%    base
   3520x1024       4    256  110    8   110   8  stagger     77087    1.7%  1.315x

   7040x1024       4    256  220    8   110  16  none       194525    0.8%    base
   7040x1024       4    256  220    8   110  16  stagger    163880    2.7%  1.187x

   3520x4096      16   1024  110    8   110   8  none       184246    1.1%    base
   3520x4096      16   1024  110    8   110   8  stagger    158186    1.8%  1.165x

   7040x4096      16   1024  220    8   110  16  none       357802    1.3%    base
   7040x4096      16   1024  220    8   110  16  stagger    316582    1.8%  1.130x
```

## Reading

- `stagger` is 1.32× / 1.19× at 256 B reads and 1.16× / 1.13× at 1024 B for 8 / 16 blocks per core.
- The saving grows with the work (24 → 31 µs and 26 → 41 µs): without the switch the grid sits on one
  bank for every block, not only at launch start.
