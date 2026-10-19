# bank_stagger — measured report

| stamp | value |
|---|---|
| box | `bh-qbge-09-special-dstoiljkovic-for-reservation-117054` (width-sharded), `bh-qbge-09-special-dstoiljkovic-for-reservation-131852` (interleaved) |
| arch | Blackhole p300c (11×10 = 110 compute grid, 8 DRAM banks) |
| date | 2026-10-05 (width-sharded), 2026-10-13 (interleaved) |
| metric | `DEVICE KERNEL DURATION [ns]`, read in-process (`ttnn.ReadDeviceProfiler` + `ttnn.get_latest_programs_perf_data`) |
| method | 3 warmup launches per variant, then 7 trials × 10 launches, variants interleaved per trial, median of trials; every variant checked bit-exact first |

> Illustrative, not a CI bound. Add another arch as a new block rather than replacing this one.

Op: tilize bf16, `ROW_MAJOR` DRAM → `TILE` interleaved DRAM, units of 32 rows × `chunk` tiles.
Source `ws` = width-sharded, one shard per bank; `il` = interleaved. `nt_h` = tile-rows, `n_w` =
units per tile-row, `blk` = blocks per core. `spread` = (max − min) / median over trials.

## Blackhole, width-sharded source — `stagger` (= `stagger_blocks`) (N=7, kernel-iters=1)

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
- Relative gains decrease when work doubles from 8 to 16 blocks per core. Runtime reduction falls
  from 24.0% to 15.8% at 256 B reads, and from 14.1% to 11.5% at 1024 B reads.
- Total time saved increases (24 → 31 µs and 26 → 41 µs), but savings per unit of work decrease.
  The benefit is stronger for the shorter workloads tested; the 8-block speedups should not be
  extrapolated to workloads with many blocks per core.
- These two workload sizes do not establish the cause of the declining relative gain or whether
  the benefit eventually disappears.

## Blackhole, interleaved source — `rows` (= `stagger_rows`) (N=7, kernel-iters=1)

Interleaved DRAM puts row `r` in bank `r % 8`. Width blocking (`32xW`, all 110 cores read the same
32 rows at different width offsets) and height blocking (`Hx512`, core k reads rows `32·k ..`, and
`32·k % 8 == 0`) both start every core on the same bank. `rows` starts core k at row `k % 32`.

```
         shape src chunk read B nt_h  n_w cores blk  variant        ns  spread  vs none
     32x7040    il     2    128    1  110   110   1  none         7447    0.7%    base
     32x7040    il     2    128    1  110   110   1  rows         7157    1.3%  1.041x

     32x56320   il    16   1024    1  110   110   1  none        19017    1.2%    base
     32x56320   il    16   1024    1  110   110   1  rows        17749    0.8%  1.071x

   3520x512     il    16   1024  110    1   110   1  none        18534    0.6%    base
   3520x512     il    16   1024  110    1   110   1  rows        17589    0.4%  1.054x

    128x56320   il    16   1024    4  110   110   4  none        66462    0.3%    base
    128x56320   il    16   1024    4  110   110   4  rows        66040    0.6%  1.006x
```

### Reading

- `rows` was faster on every shape measured: 1.04–1.07× with one block per core, 1.006× with four.
- The time saved does not grow with the work (0.3–1.3 µs at one block, 0.4 µs at four), so the
  ratio shrinks as the run gets longer. In the width-sharded case, by contrast, total time saved
  increases with blocks per core.
- These measurements do not establish why the saving stays flat; a pile-up that is limited to
  launch start would fit, but was not measured directly.

## Blackhole p150b — bh-qb-15-special-adamjanovic-for-reservation-139410 (2026-10-19)

| stamp | value |
|---|---|
| arch / board | Blackhole p150b, compute grid 12×10 = 120 Tensix cores, 8 DRAM banks, AICLK 1350 MHz |
| commit | `f74ee7aad64` + arch-agnostic regimes tests (branch `adamjanovic/ab-tilize-run-scoreboard-catalog`), Release build |
| metric / method | `test_bank_stagger_device_perf`: same op and metric as above, 5 trials × 10 launches (median of trials; the p300c blocks used 7), variants interleaved per trial, `kernel-iters=1`; shapes sized to 120 tile-rows (`3840`) instead of 110 (`3520`) |

### Width-sharded (`ws`) and interleaved (`il`) sources — all three variants

```
         shape src chunk read B nt_h  n_w cores blk  variant        ns  spread  vs none
   3840x1024    ws     4    256  120    8   120   8  none       111677    0.6%    base
   3840x1024    ws     4    256  120    8   120   8  stagger     84255    1.1%  1.325x
   3840x1024    ws     4    256  120    8   120   8  rows       111579    1.7%  1.001x

   7680x1024    ws     4    256  240    8   120  16  none       208627    1.4%    base
   7680x1024    ws     4    256  240    8   120  16  stagger    175265    1.6%  1.190x
   7680x1024    ws     4    256  240    8   120  16  rows       210400    1.6%  0.992x

   3840x4096    ws    16   1024  120    8   120   8  none       203545    1.0%    base
   3840x4096    ws    16   1024  120    8   120   8  stagger    171495    1.0%  1.187x
   3840x4096    ws    16   1024  120    8   120   8  rows       203979    0.9%  0.998x

   7680x4096    ws    16   1024  240    8   120  16  none       385965    0.6%    base
   7680x4096    ws    16   1024  240    8   120  16  stagger    345965    0.5%  1.116x
   7680x4096    ws    16   1024  240    8   120  16  rows       383476    1.0%  1.006x

     32x7680    il     2    128    1  120   120   1  none         8079    1.1%    base
     32x7680    il     2    128    1  120   120   1  stagger      8142    1.0%  0.992x
     32x7680    il     2    128    1  120   120   1  rows         7773    1.3%  1.039x

    32x61440    il    16   1024    1  120   120   1  none        21134    1.3%    base
    32x61440    il    16   1024    1  120   120   1  stagger     21135    1.2%  1.000x
    32x61440    il    16   1024    1  120   120   1  rows        19998    0.6%  1.057x

    3840x512    il    16   1024  120    1   120   1  none        20194    0.9%    base
    3840x512    il    16   1024  120    1   120   1  stagger     20122    0.7%  1.004x
    3840x512    il    16   1024  120    1   120   1  rows        19377    1.7%  1.042x
```

### Helps / does not help vs BH p300c

| case | BH p300c (110 Tensix cores) | BH p150b (120 Tensix cores) |
|---|--:|--:|
| ws, 256 B, 8 blocks per Tensix core, `stagger` | 1.32× | **1.33×** |
| ws, 256 B, 16 blocks, `stagger` | 1.19× | **1.19×** |
| ws, 1024 B, 8 blocks, `stagger` | 1.16× | **1.19×** |
| ws, 1024 B, 16 blocks, `stagger` | 1.13× | **1.12×** |
| ws, any, `rows` (does not help) | not measured | **0.99–1.01×** |
| il, 1 block, `rows` | 1.04–1.07× | **1.04–1.06×** |
| il, 1 block, `stagger` (does not help) | not measured | **0.99–1.00×** |
| il, 4 blocks, `rows` (does not help) | 1.01× | not measured |

### Reading it
- **Reproduces BH p300c.** Every shared case is within ~0.03×: the effect depends on the 8 DRAM banks, which
  both boards have, not on 110 vs 120 Tensix cores.
- **Each rotation only fixes its own source.** `stagger` (block rotation) helps only the width-sharded source and
  `rows` (row rotation) only the interleaved one; the crossed variants are flat (0.99–1.01×), which confirms the
  two pile-ups are different.
- **WH B0 n150 is untested** (12 DRAM banks); the ratios above should not be assumed there.
