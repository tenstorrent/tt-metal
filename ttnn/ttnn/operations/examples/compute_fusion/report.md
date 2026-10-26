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
| commit | `f74ee7aad64` + arch-agnostic regimes tests, Release build |
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

## Blackhole p150b — bh-50-special-adamjanovic-for-reservation-143315 (2026-10-26): `fpu_chain` and work per call

| stamp | value |
|---|---|
| arch / board | Blackhole p150b, AICLK 1350 MHz (profiler `CHIP_FREQ`, stable all session: the `sfpu_chain` / `fpu_sfpu` regimes reproduced the 2026-10-19 numbers to within 0.1 %), 1 Tensix core |
| commit | `5eac7659e6c` + the `fpu_chain` scenario and `test_compute_fusion_per_call` (uncommitted), Release build |
| metric / method | DEVICE KERNEL DURATION [ns] ÷ calls per launch = ns per call. Each launch does 768 tiles as 768 / n calls of n tiles (one call = one pass of the expression over n tiles, per-op init re-run each call). 5 interleaved trials, median; a second full run matched every row to within 0.1 % |
| correctness | every variant vs a torch golden with the intermediate rounded to bf16 (`rtol = atol = 1/64`); `dstreuse` vs `unfused` max abs difference = 0 in every case (`fp32_dest_acc_en = False`) |

### Work per call (`test_compute_fusion_per_call`)

| Scenario | Tiles per call | Block | Calls per launch | Variant | ns per call | ns per tile | Std/med | Speedup vs unfused | max abs diff vs unfused |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| fpu_chain | 1 | 1 | 768 | dstreuse | 311.5 | 311.5 | 0.0% | 1.108x | 0 |
| fpu_chain | 1 | 1 | 768 | unfused | 345.1 | 345.1 | 0.0% | 1.000x |  |
| fpu_chain | 2 | 1 | 384 | dstreuse | 485.7 | 242.9 | 0.0% | 0.991x | 0 |
| fpu_chain | 2 | 1 | 384 | unfused | 481.5 | 240.7 | 0.0% | 1.000x |  |
| fpu_chain | 2 | 2 | 384 | dstreuse | 455.8 | 227.9 | 0.0% | 1.052x | 0 |
| fpu_chain | 2 | 2 | 384 | unfused | 479.7 | 239.8 | 0.0% | 1.000x |  |
| fpu_chain | 4 | 1 | 192 | dstreuse | 791.5 | 197.9 | 0.0% | 0.882x | 0 |
| fpu_chain | 4 | 1 | 192 | unfused | 697.9 | 174.5 | 0.0% | 1.000x |  |
| fpu_chain | 4 | 4 | 192 | dstreuse | 757.4 | 189.4 | 0.0% | 0.963x | 0 |
| fpu_chain | 4 | 4 | 192 | unfused | 729.5 | 182.4 | 0.0% | 1.000x |  |
| fpu_chain | 6 | 1 | 128 | dstreuse | 1099.5 | 183.3 | 0.0% | 0.829x | 0 |
| fpu_chain | 6 | 1 | 128 | unfused | 911.3 | 151.9 | 0.3% | 1.000x |  |
| fpu_chain | 6 | 4 | 128 | dstreuse | 1007.0 | 167.8 | 0.0% | 0.890x | 0 |
| fpu_chain | 6 | 4 | 128 | unfused | 896.3 | 149.4 | 0.0% | 1.000x |  |
| fpu_chain | 8 | 1 | 96 | dstreuse | 1406.4 | 175.8 | 0.1% | 0.803x | 0 |
| fpu_chain | 8 | 1 | 96 | unfused | 1128.9 | 141.1 | 0.0% | 1.000x |  |
| fpu_chain | 8 | 4 | 96 | dstreuse | 1268.4 | 158.5 | 0.1% | 0.857x | 0 |
| fpu_chain | 8 | 4 | 96 | unfused | 1087.2 | 135.9 | 0.0% | 1.000x |  |
| fpu_chain | 16 | 1 | 48 | dstreuse | 2620.4 | 163.8 | 0.0% | 0.756x | 0 |
| fpu_chain | 16 | 1 | 48 | unfused | 1981.8 | 123.9 | 0.0% | 1.000x |  |
| fpu_chain | 16 | 4 | 48 | dstreuse | 2255.6 | 141.0 | 0.0% | 0.788x | 0 |
| fpu_chain | 16 | 4 | 48 | unfused | 1777.0 | 111.1 | 0.1% | 1.000x |  |
| fpu_chain | 32 | 1 | 24 | dstreuse | 5068.5 | 158.4 | 0.1% | 0.731x | 0 |
| fpu_chain | 32 | 1 | 24 | unfused | 3707.4 | 115.9 | 0.1% | 1.000x |  |
| fpu_chain | 32 | 4 | 24 | dstreuse | 4245.8 | 132.7 | 0.0% | 0.747x | 0 |
| fpu_chain | 32 | 4 | 24 | unfused | 3172.0 | 99.1 | 0.0% | 1.000x |  |
| fpu_chain | 64 | 1 | 12 | dstreuse | 9958.2 | 155.6 | 0.0% | 0.717x | 0 |
| fpu_chain | 64 | 1 | 12 | unfused | 7139.3 | 111.6 | 0.0% | 1.000x |  |
| fpu_chain | 64 | 4 | 12 | dstreuse | 8216.8 | 128.4 | 0.0% | 0.723x | 0 |
| fpu_chain | 64 | 4 | 12 | unfused | 5937.8 | 92.8 | 0.1% | 1.000x |  |
| fpu_sfpu | 1 | 1 | 768 | dstreuse | 1046.9 | 1046.9 | 0.0% | 1.028x | 0 |
| fpu_sfpu | 1 | 1 | 768 | sfpu | 1484.8 | 1484.8 | 0.0% | 0.725x |  |
| fpu_sfpu | 1 | 1 | 768 | unfused | 1075.9 | 1075.9 | 0.0% | 1.000x |  |
| fpu_sfpu | 2 | 1 | 384 | dstreuse | 1976.7 | 988.4 | 0.0% | 0.978x | 0 |
| fpu_sfpu | 2 | 1 | 384 | sfpu | 2848.1 | 1424.1 | 0.1% | 0.679x |  |
| fpu_sfpu | 2 | 1 | 384 | unfused | 1933.9 | 966.9 | 0.0% | 1.000x |  |
| fpu_sfpu | 2 | 2 | 384 | dstreuse | 1930.7 | 965.4 | 0.0% | 1.008x | 0 |
| fpu_sfpu | 2 | 2 | 384 | sfpu | 2822.9 | 1411.5 | 0.0% | 0.689x |  |
| fpu_sfpu | 2 | 2 | 384 | unfused | 1945.2 | 972.6 | 0.0% | 1.000x |  |
| fpu_sfpu | 4 | 1 | 192 | dstreuse | 3792.4 | 948.1 | 0.0% | 0.952x | 0 |
| fpu_sfpu | 4 | 1 | 192 | sfpu | 5565.9 | 1391.5 | 0.0% | 0.649x |  |
| fpu_sfpu | 4 | 1 | 192 | unfused | 3609.8 | 902.5 | 0.0% | 1.000x |  |
| fpu_sfpu | 4 | 4 | 192 | dstreuse | 3748.6 | 937.2 | 0.0% | 0.994x | 0 |
| fpu_sfpu | 4 | 4 | 192 | sfpu | 5528.6 | 1382.2 | 0.0% | 0.674x |  |
| fpu_sfpu | 4 | 4 | 192 | unfused | 3725.0 | 931.2 | 0.0% | 1.000x |  |
| fpu_sfpu | 6 | 1 | 128 | dstreuse | 5611.4 | 935.2 | 0.0% | 0.944x | 0 |
| fpu_sfpu | 6 | 1 | 128 | sfpu | 8282.9 | 1380.5 | 0.0% | 0.639x |  |
| fpu_sfpu | 6 | 1 | 128 | unfused | 5296.3 | 882.7 | 0.0% | 1.000x |  |
| fpu_sfpu | 6 | 4 | 128 | dstreuse | 5501.0 | 916.8 | 0.0% | 0.969x | 0 |
| fpu_sfpu | 6 | 4 | 128 | sfpu | 8179.7 | 1363.3 | 0.0% | 0.652x |  |
| fpu_sfpu | 6 | 4 | 128 | unfused | 5332.7 | 888.8 | 0.0% | 1.000x |  |
| fpu_sfpu | 8 | 1 | 96 | dstreuse | 7429.1 | 928.6 | 0.0% | 0.939x | 0 |
| fpu_sfpu | 8 | 1 | 96 | sfpu | 10993.1 | 1374.1 | 0.0% | 0.635x |  |
| fpu_sfpu | 8 | 1 | 96 | unfused | 6976.0 | 872.0 | 0.0% | 1.000x |  |
| fpu_sfpu | 8 | 4 | 96 | dstreuse | 7264.0 | 908.0 | 0.0% | 0.968x | 0 |
| fpu_sfpu | 8 | 4 | 96 | sfpu | 10827.2 | 1353.4 | 0.0% | 0.650x |  |
| fpu_sfpu | 8 | 4 | 96 | unfused | 7034.9 | 879.4 | 0.0% | 1.000x |  |
| fpu_sfpu | 16 | 1 | 48 | dstreuse | 14702.5 | 918.9 | 0.0% | 0.932x | 0 |
| fpu_sfpu | 16 | 1 | 48 | sfpu | 21853.1 | 1365.8 | 0.0% | 0.627x |  |
| fpu_sfpu | 16 | 1 | 48 | unfused | 13699.3 | 856.2 | 0.0% | 1.000x |  |
| fpu_sfpu | 16 | 4 | 48 | dstreuse | 14257.8 | 891.1 | 0.0% | 0.958x | 0 |
| fpu_sfpu | 16 | 4 | 48 | sfpu | 21426.0 | 1339.1 | 0.0% | 0.637x |  |
| fpu_sfpu | 16 | 4 | 48 | unfused | 13656.8 | 853.6 | 0.0% | 1.000x |  |
| fpu_sfpu | 32 | 1 | 24 | dstreuse | 29248.6 | 914.0 | 0.0% | 0.928x | 0 |
| fpu_sfpu | 32 | 1 | 24 | sfpu | 43594.5 | 1362.3 | 0.0% | 0.623x |  |
| fpu_sfpu | 32 | 1 | 24 | unfused | 27143.2 | 848.2 | 0.0% | 1.000x |  |
| fpu_sfpu | 32 | 4 | 24 | dstreuse | 28242.5 | 882.6 | 0.0% | 0.952x | 0 |
| fpu_sfpu | 32 | 4 | 24 | sfpu | 42595.0 | 1331.1 | 0.0% | 0.631x |  |
| fpu_sfpu | 32 | 4 | 24 | unfused | 26886.2 | 840.2 | 0.0% | 1.000x |  |
| fpu_sfpu | 64 | 1 | 12 | dstreuse | 58338.3 | 911.5 | 0.0% | 0.926x | 0 |
| fpu_sfpu | 64 | 1 | 12 | sfpu | 87009.9 | 1359.5 | 0.0% | 0.621x |  |
| fpu_sfpu | 64 | 1 | 12 | unfused | 54032.2 | 844.3 | 0.0% | 1.000x |  |
| fpu_sfpu | 64 | 4 | 12 | dstreuse | 56210.6 | 878.3 | 0.0% | 0.949x | 0 |
| fpu_sfpu | 64 | 4 | 12 | sfpu | 84978.9 | 1327.8 | 0.0% | 0.628x |  |
| fpu_sfpu | 64 | 4 | 12 | unfused | 53350.5 | 833.6 | 0.0% | 1.000x |  |

### fpu_chain rows of the sweep (`test_compute_fusion_device_perf`, N=5 median, kernel-iters 100)

| Scenario | Tiles | Block | Variant | Median ns | Std/med | Speedup vs unfused |
|---|---:|---:|---|---:|---:|---:|
| fpu_chain | 4 | 1 | dstreuse | 792.5 | 0.0% | 0.88x |
| fpu_chain | 4 | 1 | unfused | 699.8 | 0.2% | 1.00x |
| fpu_chain | 4 | 4 | dstreuse | 758.4 | 0.2% | 0.96x |
| fpu_chain | 4 | 4 | unfused | 731.1 | 0.2% | 1.00x |
| fpu_chain | 16 | 1 | dstreuse | 2633.2 | 0.0% | 0.75x |
| fpu_chain | 16 | 1 | unfused | 1981.2 | 0.0% | 1.00x |
| fpu_chain | 16 | 4 | dstreuse | 2261.1 | 0.0% | 0.78x |
| fpu_chain | 16 | 4 | unfused | 1769.1 | 0.2% | 1.00x |
| fpu_chain | 64 | 1 | dstreuse | 9993.2 | 0.0% | 0.71x |
| fpu_chain | 64 | 1 | unfused | 7112.7 | 0.0% | 1.00x |
| fpu_chain | 64 | 4 | dstreuse | 8225.5 | 0.0% | 0.72x |
| fpu_chain | 64 | 4 | unfused | 5908.2 | 0.0% | 1.00x |

### Regimes (`test_compute_fusion_regimes`, 11 interleaved trials, kernel-iters 100, ns per call)

| Scenario | n tiles | blk | baseline | median ns | lever | median ns | speedup | verdict | baseline min..max | lever min..max |
|---|---:|---:|---|---:|---|---:|---:|---|---|---|
| sfpu_chain | 4 | 4 | unfused | 6386 | fused | 5745 | 1.11x | helps (assert ≥ 1.055) | 6385..6443 | 5744..5746 |
| fpu_sfpu | 64 | 1 | unfused | 54023 | dstreuse | 58328 | 0.93x | loses (≤ 1.0) | 54022..54023 | 58327..58331 |
| fpu_chain | 1 | 1 | unfused | 349 | dstreuse | 314 | 1.11x | helps (assert ≥ 1.055) | 349..350 | 314..314 |
| fpu_chain | 4 | 4 | unfused | 731 | dstreuse | 759 | 0.96x | loses (≤ 1.0) | 731..732 | 758..762 |
| fpu_chain | 64 | 4 | unfused | 5908 | dstreuse | 8226 | 0.72x | loses (≤ 1.0) | 5907..5912 | 8225..8226 |

### Reading it
- FPU → FPU chain (`fpu_chain`): DEST reuse wins only at 1 tile per call (1.11×) and at 2 tiles per call with
  block 2 (1.05×); at 2 tiles with block 1 it is even (0.99×), and from 4 tiles per call it loses (0.88–0.96× at 4,
  0.83–0.89× at 6, 0.72× at 64). Crossover 2–3 tiles per call.
- Straight-line fits (blk 1 from n = 1 and 64; blk 4 from n = 4 and 64): `unfused` ~237 / ~382 ns fixed per call
  + ~108 / ~87 ns per tile; `dstreuse` ~158 / ~260 ns fixed + ~153 / ~124 ns per tile. DEST reuse saves the second
  pass's fixed cost (~80–120 ns per call) and pays ~37–45 ns more per tile (a DEST→SrcA move and a DEST face clear
  before the multiply on each of the 4 faces), so crossover ≈ 80 / 45 ≈ 1.8 tiles (blk 1), 122 / 37.5 ≈ 3.3 tiles
  (blk 4).
- `fpu_sfpu` (SFPU sqrt producer): 1.03× at 1 tile per call, 1.01× at 2 tiles (blk 2), loses from 2 tiles at blk 1
  and everywhere from 4 tiles (0.93–0.99×). The sqrt pass costs ~0.8 µs per tile, so the saved fixed cost is small.
