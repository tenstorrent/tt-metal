# mcast_topology — measured reports

Delivery only (no compute). One block per box/arch.

## Arch.BLACKHOLE — bh-49-special-mstaletovic-for-reservation-60064

```
mcast_topology  box=bh-49-special-mstaletovic-for-reservation-60064  arch=Arch.BLACKHOLE  grid=11x10 (110 cores)  M=8t N=32t K=4t  delivery only (no compute)   N=5 (median of 5-launch windows)
  per_core_dram  split=8x8  cores= 64/110 ( 58%)  per-core DRAM reads                    8512 ns ±0.3%  -> 1.00x
  mcast_1d_pair  split=8x8  cores= 64/110 ( 58%)  2x Mcast1D (PerRow + PerColumn)        4450 ns ±1.1%  -> 1.91x
```
