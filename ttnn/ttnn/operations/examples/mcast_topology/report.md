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
