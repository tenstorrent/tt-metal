# wan_fused_distributed_rmsnorm — performance optimization log

Goal: speed up distributed RMSNorm for all Wan shapes without any loss of
compute fidelity / precision. Targets: smallest shapes 1.5×, largest 3×
vs the composite baseline.

Baseline (TP=4 LINE, BH 2x4), composite_us / fused_us / speedup:

| config | composite | fused | speedup |
|---|---|---|---|
| self_sp4_N18944    | 663.6 | 416.4 | 1.59× |
| self_sp8_N9472     | 334.0 | 271.0 | 1.23× |
| self_sp32_N2368    | 108.4 | 124.3 | 0.87× |
| cross_q_sp4_N18944 | 596.1 | 344.8 | 1.73× |
| cross_q_sp8_N9472  | 294.9 | 216.2 | 1.36× |
| cross_q_sp32_N2368 |  97.7 |  90.5 | 1.08× |
| cross_k_prompt_L512|  51.8 |  53.3 | 0.97× |

## Profiling (2026-05-29) — largest shape N18944, TP=4 LINE, with RoPE

Device-zone profiling (DeviceZoneScopedN markers, 64 workers). Per TRISC
thread, averaged over the 64 worker cores, summed over all chunks:

| phase | time | notes |
|---|---|---|
| PRE (square + sum-of-squares reduce) | 84us | 27% |
| AGWAIT (wait for fabric all-gather)  | **0us** | fabric fully hidden |
| P_NORM (reduce<AVG>+eps+rsqrt + ×1/rms) | 96us | 30% |
| P_WEIGHT (×weight)                   | 23us | 7% |
| P_ROPE (matmul+cos+sin+add)          | 111us | 35% |
| **TRISC kernel total**               | **~316us** | matches sum |

Per-RISC kernel span (max core / avg over cores):
- BRISC (writer/mux): 386 / 322 us  (max core is a MUX core, busy-wait)
- NCRISC (reader):    313 / 257 us  (sits *under* compute — not limiting)
- TRISC (compute):    369 / 316 us  (**the bottleneck**)

Wall-clock fused = 416us; compute critical path (slowest core) ≈ 369us +
output-drain/termination tail.

### Key findings
1. **Fabric AG is completely hidden** (AGWAIT ≈ 0). All prior fabric/MUX
   work paid off — fabric is free. The bottleneck is pure compute.
2. **Compute-bound**: TRISC 316us avg vs reader 257us. To go faster we
   must cut compute *and/or* add compute parallelism.
3. **Only 64 of 110 BH cores are used** (64 workers + 4 MUX). The
   `kMaxMuxWorkersPerChip = 64` cap leaves 42 cores idle. For N18944 that
   is 9.25 tile-rows/worker; more workers → fewer rows/worker → near
   linear compute speedup (fabric stays hidden).
4. RoPE is the single biggest compute phase (35%); P_NORM second (30%),
   much of it the serialized reduce→eps→rsqrt→×1/rms dependency chain.

## Session 2026-05-29 — small-shape worker threshold + multi-link dead-end

Same-session HEAD baseline reproduced PERF_LOG within 0.5% (N18944 fused
416.81us vs 416.4 logged), so the device is stable and cross-session
variance is NOT a confound. All numbers below were measured in one
process against that same-session HEAD.

### WIN: raise small-shape worker threshold 8 → 16 (committed)

`pick_num_workers_tp_gt_1` has two regimes split at `kSmallShapeRowsLimit`:
one-worker-per-row below, rows/2 packing above. Bumping the split from 8
to 16 tile-rows pulls `cross_k_prompt_L512` (16 tile-rows) into the
one-per-row regime (8 → 16 workers):

| config | before | after | speedup |
|---|---|---|---|
| cross_k_prompt_L512 | 53.3us / 0.97× | 42.4us / 1.22× | — |

Zero regression elsewhere (only 9–16 tile-row shapes change regime, and
L512 is the only bench config in that band). Re-benched all 7 configs:
N18944 1.59×, N9472 1.23×, N2368-rope 0.87×, cross_q N18944 1.73×,
cross_q N9472 1.38×, cross_q N2368 1.07×, cross_k_L512 1.22×.
Production-shape multihead correctness test (N=128/2368/12400) PASSES.

### DEAD END: multi-link MUX does NOT help large shapes

Hypothesis was that num_links=2 lifts the worker cap 64 → ~106, giving
the compute-bound large shapes near-linear speedup (finding #3 above).
Measured: it does NOT help — N18944 416 → 428us, N9472 271 → 310us
(regression). Root causes:
  - HEAD's bench harness **already** runs num_links=2; the "64 cores"
    profiling run was itself num_links=2, so there was no headroom to
    unlock.
  - Forcing one-worker-per-row on large shapes (to consume the extra
    cores) creates MORE fabric packets, and packet/MUX overhead grows
    faster than the per-worker compute shrinks. Net regression: N9472
    +13.6%, cross_q N2368 +24.6%.

This invalidates the simple "add cores → linear speedup" reading of
finding #3. Worker-count parallelism is **exhausted** as a lever for
large shapes; the rows/2 packing is already near-optimal for them. The
multi-link WIP is stashed (stash@{0}) but is a confirmed net regression —
do not re-apply.

### Revised bottleneck hypothesis for large shapes

The "compute-bound" reading (TRISC 316us > reader 257us) is real but the
reader floor (~257us DRAM-read) is close enough that simply parallelizing
compute won't reach 3×. Next lever must CUT work on the critical path
(fuse/shorten the P_NORM dependency chain or the RoPE matmul), not add
cores. Re-profile N18944 to confirm whether reader DRAM throughput or the
compute critical path dominates once workers are saturated.
