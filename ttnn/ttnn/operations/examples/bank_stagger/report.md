# bank_stagger — measured report

| stamp | value |
|---|---|
| box | `bh-qbge-09-special-dstoiljkovic-for-reservation-117054` |
| arch | Blackhole p300c (11×10 = 110 compute grid, 8 DRAM banks) |
| date | 2026-10-05 |
| metric | `DEVICE KERNEL DURATION [ns]`, read in-process (`ttnn.ReadDeviceProfiler` + `ttnn.get_latest_programs_perf_data`) |
| method | 3 warmup launches per variant, then 7 trials × 10 launches, variants interleaved per trial, median of trials; every variant checked bit-exact first |

> Illustrative, not a CI bound. Add another arch as a new block rather than replacing this one.

Op: tilize bf16, `ROW_MAJOR` DRAM → `TILE` interleaved DRAM, units of 32 rows × `chunk` tiles.
`src`: `il` = interleaved, `ws` = width-sharded with one shard per DRAM bank. `nt_h` = tile-rows,
`n_w` = units per tile-row, `blk` = blocks per core. `spread` = (max − min) / median over trials.
Shown: the baseline and the switch for each case (`blocks` on `ws`, `read` on `il`).

## Blackhole — default sweep (N=7, kernel-iters=1)

```
         shape src chunk read B nt_h  n_w cores blk  variant        ns  spread  vs none
     32x7040    il     2    128    1  110   110   1  none         7468    0.9%    base
     32x7040    il     2    128    1  110   110   1  read         7192    2.2%  1.038x

     32x56320   il    16   1024    1  110   110   1  none        18958    1.5%    base
     32x56320   il    16   1024    1  110   110   1  read        17768    1.2%  1.067x

   3520x512     il    16   1024  110    1   110   1  none        18571    1.4%    base
   3520x512     il    16   1024  110    1   110   1  read        17566    1.2%  1.057x

   3520x1024    ws     4    256  110    8   110   8  none       100789    1.2%    base
   3520x1024    ws     4    256  110    8   110   8  blocks      76722    2.0%  1.314x

   3520x4096    ws    16   1024  110    8   110   8  none       183685    1.0%    base
   3520x4096    ws    16   1024  110    8   110   8  blocks     157840    2.2%  1.164x
```

## Blackhole — width-sharded, more blocks per core (N=7, kernel-iters=1)

```
         shape src chunk read B nt_h  n_w cores blk  variant        ns  spread  vs none
   7040x1024    ws     4    256  220    8   110  16  none       193737    2.1%    base
   7040x1024    ws     4    256  220    8   110  16  blocks     162812    3.5%  1.190x

  14080x1024    ws     4    256  440    8   110  32  none       366165    2.2%    base
  14080x1024    ws     4    256  440    8   110  32  blocks     329894    1.1%  1.110x

   7040x4096    ws    16   1024  220    8   110  16  none       358202    1.5%    base
   7040x4096    ws    16   1024  220    8   110  16  blocks     316906    1.1%  1.130x

  14080x4096    ws    16   1024  440    8   110  32  none       696135    1.1%    base
  14080x4096    ws    16   1024  440    8   110  32  blocks     641448    1.6%  1.085x
```

## Reading

- `blocks` on a width-sharded source: 1.31× / 1.19× / 1.11× at 256 B reads and 1.16× / 1.13× /
  1.085× at 1024 B for 8 / 16 / 32 blocks per core. The saving grows with the work (24 → 36 µs and
  26 → 55 µs): the grid sits on one bank for every block, not only at launch start.
- `read` on an interleaved source: 1.04–1.07×, a fixed 0.3–1.2 µs per launch — only the first
  wave of reads is clustered.
- Not shown because they measure on par (within the spread): `read` and `write` on `ws`,
  `blocks` and `write` on `il`, and `combined` (≈ the winning single switch).

## Wormhole B0 (earlier, 64 cores, 12 banks)

Box `bgd-lab-16-special-dstoiljkovic-for-reservation-114068`, commit `3f32706f595`, 1 block per core.
`both` was 1.07–1.14× on width blocking with ≥ 256 B reads. Unlike BH, the write switch carried the
gain at `chunk = 4` (1.09–1.12×), where every core's first write landed on 3 of 12 banks.
