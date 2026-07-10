# Compute fusion — per-phase device-zone micro-benchmark

box=bgd-lab-t3003-special-mstaletovic-for-reservation-40918  arch=Arch.WORMHOLE_B0  cores=1  placement=single-core sharded-L1  num-tiles=32  kernel-iters=16  metric=median ns per phase (per launch)

Each phase is one `eltwise_chain` / `reduce` call. A compute zone records on all three TRISCs; `wall` is the slowest engine (they pipeline). `Σ wall` sums the variant's phases = the serial cost the whole-kernel number reflects.

| Scenario | Variant | Phase | unpack ns | math ns | pack ns | wall ns |
|---|---|---|---:|---:|---:|---:|
| sfpu_chain | fused | CF_FUSED | 58857 | 61688 | 61713 | 61713 |
| sfpu_chain | unfused | CF_SQRT | 26852 | 30625 | 30442 | 30625 |
| sfpu_chain | unfused | CF_ADD | 13313 | 10146 | 10646 | 13313 |
| sfpu_chain | unfused | CF_EXP | 20725 | 22991 | 22661 | 22991 |
| sfpu_chain | unfused | **Σ wall** | | | | **66929** |
| fpu_sfpu | dstreuse | CF_FUSED | 34998 | 36160 | 36174 | 36174 |
| fpu_sfpu | sfpu | CF_FUSED | 55387 | 57954 | 57976 | 57976 |
| fpu_sfpu | unfused | CF_SQRT | 26857 | 30663 | 29802 | 30663 |
| fpu_sfpu | unfused | CF_MUL | 6608 | 3196 | 4054 | 6608 |
| fpu_sfpu | unfused | **Σ wall** | | | | **37271** |
| reduce_recip | fused | CF_FUSED | 4582 | 6922 | 6947 | 6947 |
| reduce_recip | unfused | CF_REDUCE | 4581 | 4285 | 6784 | 6784 |
| reduce_recip | unfused | CF_RECIP | 722 | 2775 | 291 | 2775 |
| reduce_recip | unfused | **Σ wall** | | | | **9559** |
