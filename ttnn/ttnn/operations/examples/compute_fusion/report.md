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
