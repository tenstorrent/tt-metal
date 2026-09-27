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
