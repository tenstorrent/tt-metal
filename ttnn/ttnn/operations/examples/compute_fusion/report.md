# Compute fusion vs L1 round-trips — single-core report

box=bgd-lab-t3003-special-mstaletovic-for-reservation-40918  arch=Arch.WORMHOLE_B0  cores=1  placement=single-core sharded-L1  N=5 (median)  kernel-iters=100 (steady-state)

Metric: DEVICE KERNEL DURATION [ns] per expression evaluation. Speedup = unfused / variant.

| Scenario | Tiles | Block | Variant | Median ns | Std/med | Speedup vs unfused |
|---|---:|---:|---|---:|---:|---:|
| sfpu_chain | 4 | 1 | fused | 7929.6 | 0.1% | 1.07x |
| sfpu_chain | 4 | 1 | unfused | 8508.2 | 0.0% | 1.00x |
| sfpu_chain | 4 | 4 | fused | 7926.5 | 0.0% | 1.12x |
| sfpu_chain | 4 | 4 | unfused | 8854.6 | 0.0% | 1.00x |
| sfpu_chain | 16 | 1 | fused | 30978.2 | 0.0% | 1.04x |
| sfpu_chain | 16 | 1 | unfused | 32223.8 | 0.0% | 1.00x |
| sfpu_chain | 16 | 4 | fused | 30683.7 | 0.0% | 1.06x |
| sfpu_chain | 16 | 4 | unfused | 32530.0 | 0.0% | 1.00x |
| sfpu_chain | 64 | 1 | fused | 123184.6 | 0.0% | 1.03x |
| sfpu_chain | 64 | 1 | unfused | 127084.2 | 0.0% | 1.00x |
| sfpu_chain | 64 | 4 | fused | 121621.1 | 0.0% | 1.05x |
| sfpu_chain | 64 | 4 | unfused | 127218.8 | 0.0% | 1.00x |
| fpu_sfpu | 4 | 1 | dstreuse | 4736.3 | 0.0% | 0.97x |
| fpu_sfpu | 4 | 1 | sfpu | 7405.1 | 0.0% | 0.62x |
| fpu_sfpu | 4 | 1 | unfused | 4599.2 | 0.0% | 1.00x |
| fpu_sfpu | 4 | 4 | dstreuse | 4716.8 | 0.0% | 1.02x |
| fpu_sfpu | 4 | 4 | sfpu | 7410.1 | 0.1% | 0.65x |
| fpu_sfpu | 4 | 4 | unfused | 4828.3 | 0.0% | 1.00x |
| fpu_sfpu | 16 | 1 | dstreuse | 18193.1 | 0.0% | 0.94x |
| fpu_sfpu | 16 | 1 | sfpu | 29054.6 | 0.0% | 0.59x |
| fpu_sfpu | 16 | 1 | unfused | 17154.5 | 0.0% | 1.00x |
| fpu_sfpu | 16 | 4 | dstreuse | 17841.9 | 0.0% | 0.97x |
| fpu_sfpu | 16 | 4 | sfpu | 29770.9 | 0.1% | 0.58x |
| fpu_sfpu | 16 | 4 | unfused | 17327.3 | 0.0% | 1.00x |
| fpu_sfpu | 64 | 1 | dstreuse | 72048.9 | 0.0% | 0.94x |
| fpu_sfpu | 64 | 1 | sfpu | 115616.9 | 0.0% | 0.58x |
| fpu_sfpu | 64 | 1 | unfused | 67395.0 | 0.0% | 1.00x |
| fpu_sfpu | 64 | 4 | dstreuse | 70241.7 | 0.0% | 0.96x |
| fpu_sfpu | 64 | 4 | sfpu | 113978.8 | 0.1% | 0.59x |
| fpu_sfpu | 64 | 4 | unfused | 67316.6 | 0.0% | 1.00x |
| reduce_recip | 4 | 1 | fused | 2554.5 | 0.1% | 1.07x |
| reduce_recip | 4 | 1 | unfused | 2736.0 | 0.1% | 1.00x |
| reduce_recip | 16 | 1 | fused | 4428.7 | 0.1% | 1.04x |
| reduce_recip | 16 | 1 | unfused | 4606.8 | 0.4% | 1.00x |
| reduce_recip | 64 | 1 | fused | 11913.9 | 0.0% | 1.01x |
| reduce_recip | 64 | 1 | unfused | 12090.5 | 0.0% | 1.00x |

## 2026-10-12 - regimes (Wormhole B0 n150), `test_compute_fusion_regimes`
11 interleaved trials per variant, kernel-iters 100, ns per evaluation; three runs identical to within 0.1 percent.

| Scenario | n tiles | blk | baseline | median ns | lever | median ns | speedup | baseline min..max | lever min..max |
|---|---:|---:|---|---:|---|---:|---:|---|---|
| sfpu_chain | 4 | 4 | unfused | 9674 | fused | 8632 | 1.12x | 9671..9679 | 8631..8634 |
| fpu_sfpu | 64 | 1 | unfused | 77734 | dstreuse | 84105 | 0.92x | 77708..77746 | 83856..84816 |

Extra reading (not asserted): `reduce_recip` n=64, unfused 5284 -> fused 5099 ns = 1.04x (11 trials).
Absolute times at n=64 differ from the 2026-07-10 table above (fpu_sfpu unfused 67395 ns then, 77734 ns now; reduce_recip unfused 12090 ns then, 5284 ns now),
so the current kernel library/helper code is not the one that produced the older table; the qualitative rule (SFPU consumer fuses, FPU consumer loses) still holds, with dstreuse now 0.92x instead of 0.94x.

## Blackhole p150b — bh-qb-15-special-adamjanovic-for-reservation-139410 (2026-10-19)

| stamp | value |
|---|---|
| arch / board | Blackhole p150b, compute grid 12×10 = 120 Tensix cores, 8 DRAM banks, AICLK 1350 MHz (this test uses 1 Tensix core) |
| commit | `f74ee7aad64` + arch-agnostic regimes tests (branch `adamjanovic/ab-tilize-run-scoreboard-catalog`), Release build |
| metric / method | same as the blocks above: DEVICE KERNEL DURATION [ns] per expression evaluation; `test_compute_fusion_device_perf` (N=5 median, kernel-iters 100) and `test_compute_fusion_regimes` (11 interleaved trials, kernel-iters 100) |

### Sweep (`test_compute_fusion_device_perf`)

| Scenario | Tiles | Block | Variant | Median ns | Std/med | Speedup vs unfused |
|---|---:|---:|---|---:|---:|---:|
| sfpu_chain | 4 | 1 | fused | 5736.4 | 0.0% | 1.09x |
| sfpu_chain | 4 | 1 | unfused | 6234.1 | 0.0% | 1.00x |
| sfpu_chain | 4 | 4 | fused | 5744.9 | 0.0% | 1.11x |
| sfpu_chain | 4 | 4 | unfused | 6386.0 | 0.0% | 1.00x |
| sfpu_chain | 16 | 1 | fused | 22435.4 | 0.0% | 1.06x |
| sfpu_chain | 16 | 1 | unfused | 23720.4 | 0.0% | 1.00x |
| sfpu_chain | 16 | 4 | fused | 22128.7 | 0.0% | 1.05x |
| sfpu_chain | 16 | 4 | unfused | 23326.8 | 0.1% | 1.00x |
| sfpu_chain | 64 | 1 | fused | 89218.3 | 0.0% | 1.05x |
| sfpu_chain | 64 | 1 | unfused | 93658.9 | 0.0% | 1.00x |
| sfpu_chain | 64 | 4 | fused | 87657.9 | 0.0% | 1.04x |
| sfpu_chain | 64 | 4 | unfused | 91058.8 | 0.0% | 1.00x |
| fpu_sfpu | 4 | 1 | dstreuse | 3793.3 | 0.0% | 0.95x |
| fpu_sfpu | 4 | 1 | sfpu | 5567.0 | 0.0% | 0.65x |
| fpu_sfpu | 4 | 1 | unfused | 3610.9 | 0.0% | 1.00x |
| fpu_sfpu | 4 | 4 | dstreuse | 3749.6 | 0.1% | 0.99x |
| fpu_sfpu | 4 | 4 | sfpu | 5529.8 | 0.0% | 0.67x |
| fpu_sfpu | 4 | 4 | unfused | 3726.0 | 0.0% | 1.00x |
| fpu_sfpu | 16 | 1 | dstreuse | 14701.0 | 0.0% | 0.93x |
| fpu_sfpu | 16 | 1 | sfpu | 21862.1 | 0.0% | 0.63x |
| fpu_sfpu | 16 | 1 | unfused | 13697.0 | 0.0% | 1.00x |
| fpu_sfpu | 16 | 4 | dstreuse | 14255.6 | 0.0% | 0.96x |
| fpu_sfpu | 16 | 4 | sfpu | 21418.1 | 0.0% | 0.64x |
| fpu_sfpu | 16 | 4 | unfused | 13653.9 | 0.0% | 1.00x |
| fpu_sfpu | 64 | 1 | dstreuse | 58327.4 | 0.0% | 0.93x |
| fpu_sfpu | 64 | 1 | sfpu | 87035.1 | 0.0% | 0.62x |
| fpu_sfpu | 64 | 1 | unfused | 54022.4 | 0.0% | 1.00x |
| fpu_sfpu | 64 | 4 | dstreuse | 56193.7 | 0.0% | 0.95x |
| fpu_sfpu | 64 | 4 | sfpu | 84956.1 | 0.0% | 0.63x |
| fpu_sfpu | 64 | 4 | unfused | 53332.1 | 0.0% | 1.00x |
| reduce_recip | 4 | 1 | fused | 565.8 | 0.0% | 1.16x |
| reduce_recip | 4 | 1 | unfused | 657.2 | 0.0% | 1.00x |
| reduce_recip | 16 | 1 | fused | 1000.7 | 0.0% | 1.09x |
| reduce_recip | 16 | 1 | unfused | 1091.4 | 0.0% | 1.00x |
| reduce_recip | 64 | 1 | fused | 2743.6 | 0.0% | 1.03x |
| reduce_recip | 64 | 1 | unfused | 2836.5 | 0.0% | 1.00x |

### Regimes (`test_compute_fusion_regimes` on BH p150b)

| Scenario | n tiles | blk | baseline | median ns | lever | median ns | speedup | verdict | baseline min..max | lever min..max |
|---|---:|---:|---|---:|---|---:|---:|---|---|---|
| sfpu_chain | 4 | 4 | unfused | 6386 | fused | 5745 | 1.11x | helps (assert ≥ 1.05) | 6385..6401 | 5744..5746 |
| fpu_sfpu | 64 | 1 | unfused | 54023 | dstreuse | 58328 | 0.93x | loses (≤ 1.0) | 54022..54025 | 58327..58331 |

### Reading it
- Same verdicts as WH B0 n150: fuse when the consumer is an SFPU op (`sfpu_chain` 1.04–1.11x, `reduce_recip`
  1.03–1.16x), keep the L1 round-trip when it is an FPU op (`dstreuse` 0.93–0.99x), never move an FPU multiply onto
  the SFPU (`sfpu` 0.62–0.67x).
- Absolute times are ~0.7x of WH B0 n150's 2026-10-12 regime numbers (`fpu_sfpu` n=64 unfused 54023 vs 77734 ns),
  so the fixed saving from fusing at 4 tiles is ~0.64 µs on BH p150b vs ~1.04 µs on WH B0 n150; the ratio is unchanged.
- `blk=4` vs `blk=1` stays a second-order effect (within ~2 %), as on WH B0 n150.
