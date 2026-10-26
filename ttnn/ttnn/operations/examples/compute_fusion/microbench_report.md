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

## Blackhole p150b — bh-50-special-adamjanovic-for-reservation-143315 (2026-10-26), `fpu_chain` only

arch=Blackhole p150b  AICLK=1350 MHz  cores=1  placement=single-core sharded-L1  num-tiles=32  block=1  kernel-iters=16  metric=median ns per phase (per launch)

| Scenario | Variant | Phase | unpack ns | math ns | pack ns | wall ns |
|---|---|---|---:|---:|---:|---:|
| fpu_chain | dstreuse | CF_FUSED | 4865 | 5053 | 5056 | 5056 |
| fpu_chain | unfused | CF_ADD | 1134 | 1369 | 1367 | 1369 |
| fpu_chain | unfused | CF_MUL | 2200 | 2293 | 2295 | 2295 |
| fpu_chain | unfused | **Σ wall** | | | | **3664** |

The fused FPU → FPU chain costs 158 ns per tile against 43 + 72 = 115 ns per tile for the two passes: the
DEST→SrcA move and DEST face clear the dest-reuse multiply runs on each face outweigh the scratch-CB unpack it
replaces once a call covers more than ~2–3 tiles (see README "Work per call").
