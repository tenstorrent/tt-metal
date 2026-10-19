# mcast_topology — measured reports

Delivery only (no compute). One block per box/arch.

## Blackhole P150b — bh-49-special-mstaletovic-for-reservation-60064

Busy AICLK, enabled GDDR count, and firmware were not captured by the original run.

```
mcast_topology  box=bh-49-special-mstaletovic-for-reservation-60064  arch=Arch.BLACKHOLE  grid=11x10 (110 cores)  M=8t N=32t K=4t  delivery only (no compute)   N=5 (median of 5-launch windows)
  per_core_dram  split=8x8  cores= 64/110 ( 58%)  per-core DRAM reads                    8512 ns ±0.3%  -> 1.00x
  mcast_1d_pair  split=8x8  cores= 64/110 ( 58%)  2x Mcast1D (PerRow + PerColumn)        4450 ns ±1.1%  -> 1.91x
```

## Blackhole P100a — bh-43-special-sjovic-for-reservation-97381

1350 MHz busy AICLK · 7 enabled GDDR banks · firmware 19.12.0 · 2026-09-23

```
mcast_topology  box=bh-43-special-sjovic-for-reservation-97381  arch=blackhole  grid=11x10 (110 cores)  M=8t N=32t K=4t  delivery only (no compute)   N=5 (median of 5-launch windows)
  per_core_dram  split=8x8  cores= 64/110 ( 58%)  per-core DRAM reads                   10958 ns ±0.7%  -> 1.00x
  mcast_1d_pair  split=8x8  cores= 64/110 ( 58%)  2x Mcast1D (PerRow + PerColumn)        4683 ns ±0.4%  -> 2.34x
```

## Wormhole B0 n150 — regime measurements (2026-10-12)

`test_mcast_topology_regimes`; 8x8 compute grid; bf16; delivery only; median of 10 interleaved windows x 5 launches (ranges min-max):

```
Mt Nt Kt   split  cores  per_core_dram ns         mcast_1d_pair ns        speedup
 8 32  4   8x8    64     16340 (16055-16406)      8643 (8560-8693)        1.89x
 8  8  4   8x8    64      9189                    6849                    1.34x
 4  4  4   4x4    16      4098 (3980-4324)        4051 (4003-4152)        1.01x
 2  2  4   2x2     4      2230 (2123-2349)        3117 (3090-3163)        0.72x
 2  8  4   2x8    16      3216                    3721                    0.86x
 8  2  4   8x2    16      5135                    5180                    0.99x
 1  8  4   1x8     8      2496                    2966                    0.84x
 8  1  4   8x1     8      4716                    4854                    0.97x
16 64  4   8x8    64     29928                   12640                    2.37x
16 64  1   8x8    64     10349                    6213                    1.67x
 8 32  1   8x8    64      7232                    5446                    1.33x
 8 32 16   8x8    64     56780                   21236                    2.67x
 8 32 64   8x8    64    261544                   75342                    3.47x
 4  4  1   4x4    16      2673                    2909                    0.92x
 4  4 32   4x4    16     19640                   16021                    1.23x
16 16  8   8x8    64     24082                   13907                    1.73x
24 24  4   8x8    64     21680                   11888                    1.82x
56 56  2   8x8    64     21263                   13111                    1.62x
64 64 16   8x8    64    208738                   81644                    2.56x
```

Test runs (8x8 / 4x4 / 2x2 speedup): 1.87x / 1.04x / 0.70x; 1.87x / 1.04x / 0.73x; 1.89x / 1.03x / 0.75x. The 4x4 assertion is `<= 1.08`.

## Blackhole p150b — bh-qb-15-special-adamjanovic-for-reservation-139410 (2026-10-19)

| stamp | value |
|---|---|
| arch / board | Blackhole p150b, compute grid 12×10 = 120 Tensix cores, 8 DRAM banks, AICLK 1350 MHz |
| commit | `f74ee7aad64` + arch-agnostic regimes tests (branch `adamjanovic/ab-tilize-run-scoreboard-catalog`), Release build |
| metric / method | DEVICE KERNEL DURATION [ns], delivery only (no compute), bf16; `test_mcast_topology_device_perf` (median of 5-launch windows, N=5) and `test_mcast_topology_regimes` (median of 10 interleaved windows × 5 launches, ranges min-max), plus a probe using the regimes test's `_measure_interleaved_ns` |

### Sweep (`test_mcast_topology_device_perf`)

```
mcast_topology  box=bh-qb-15-special-adamjanovic-for-reservation-139410  arch=Arch.BLACKHOLE  grid=12x10 (120 cores)  M=8t N=32t K=4t  delivery only (no compute)   N=5 (median of 5-launch windows)
  per_core_dram  split=8x8  cores= 64/120 ( 53%)  per-core DRAM reads                    8739 ns ±0.3%  -> 1.00x
  mcast_1d_pair  split=8x8  cores= 64/120 ( 53%)  2x Mcast (RowMajor + ColumnMajor)       4350 ns ±0.9%  -> 2.01x
```

### Boundary sweep / regimes (`test_mcast_topology_regimes` on BH p150b, plus same-code probe)

```
Mt Nt Kt   split  cores  per_core_dram ns         mcast_1d_pair ns        speedup   source
 8 32  4   8x8    64      8698 (8653-8821)        4327 (4268-4374)        2.01x     regime helps_8x8_64cores (assert >= 1.45)
10 48  4   10x12 120     14171                    5849                    2.42x     BH-only regime, dropped — not asserted
 4  4  4   4x4    16      1791 (1765-1818)        2252 (2241-2283)        0.80x     regime does_not_help_4x4_16cores (assert <= 1.08)
 2  2  4   2x2     4      1261 (1248-1293)        1933 (1909-1953)        0.65x     regime does_not_help_2x2_4cores (assert <= 1.0)
 8 32 16   8x8    64      -                       -                       3.27x     probe
 8 32 64   8x8    64      -                       -                       3.90x     probe
10 48 16   10x12 120      -                       -                       4.18x     probe
 5 24  4   5x12   60      -                       -                       1.44x     probe
 1 32  4   1x8     8      -                       -                       0.75x     probe
 2 32  4   2x8    16      -                       -                       0.91x     probe
```

(`-` = ns not kept; only the speedup was recorded.)

### Reading it

- 4×4 (16 Tensix cores) **loses** on BH p150b (0.80x) where WH B0 n150 was flat (1.01-1.04x): the
  per-core DRAM baseline is 2.3x faster than on WH B0 (1791 vs 4186 ns) but the multicast path only 1.8x
  (2252 vs 4056 ns), so the two per-line semaphore handshakes are no longer paid for. The regimes test keeps
  `<= 1.08` for 4x4, the bound that held on both boards; the BH p150b loss is documented here.
- Every 8x8 point gains more than on WH B0 (Kt 4 / 16 / 64: 2.01x / 3.27x / 3.90x vs 1.87x / 2.67x / 3.47x),
  and the BH-only 10x12 split (lines of 10 and 12 Tensix cores) reaches 2.42x at Kt 4 and 4.18x at Kt 16.
- The boundary sits between 4x4 (0.80x) and 5x12 (1.44x); 1x8 and 2x8 lose on both boards (one grid
  dimension of 1-2 Tensix cores shares `B` at most 2-fold).
- The device_perf 8x8 point (2.01x) is close to the older 11x10 P150b box (1.91x, bh-49); absolute ns are within 3% (8739 vs 8512, 4350 vs 4450).
