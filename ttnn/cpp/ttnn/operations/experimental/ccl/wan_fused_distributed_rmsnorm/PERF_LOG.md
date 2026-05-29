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

## Ceiling analysis (2026-05-29) — why 3×/1.5× targets are hard

After exhausting the worker-count and multi-link levers, the structural
ceilings appear to be:

**Large shapes (N9472/N18944) — ceiling ≈ 2×, currently 1.59–1.73×.**
The fused op's ONLY structural advantage over the composite is eliminating
the post-kernel DRAM re-read of the input (composite reads input twice:
once in pre, once in post; fused reads once, keeps it L1-resident). That
is a ~2× DRAM-traffic reduction → ~2× ceiling IF the composite is purely
DRAM-bound. Observed 1.59–1.73× is below that because:
  - Compute (TRISC ~316us avg) ≈ reader (~257us) ≈ writer-worker (~322us)
    are all balanced — no single lane has slack to absorb the others.
  - Fabric is already fully hidden (AGWAIT≈0); no fabric lever left.
  - The ~100us wall-vs-TRISC gap is pipeline fill (first chunk's AG must
    complete before the first POST) + drain (last chunk's output write
    after the last POST). With only ~3 chunks/worker, fill+drain is a
    large fraction and can't be amortized away without a deeper rewrite.
  - Fidelity is fixed (HiFi4) — cannot trade precision for speed.
Reaching 3× would require ~halving compute at fixed fidelity (not
possible with the same math) or a fundamentally different algorithm.

**Small sharded shapes (N2368, sp32) — currently 0.87×/1.07×, want 1.5×.**
These are dominated by fused-op FIXED overhead (~50us), not compute:
  - Composite "wastes" ~40us re-reading input yet still BEATS fused by
    16us → fused carries ~56us of fixed cost the composite doesn't.
  - That fixed cost is the MUX/fabric handshake (endpoint-ready wait +
    client-connect + per-chunk mcast + semaphore wait + DRAM round-trip)
    plus single-program scheduling — a per-chunk latency that does NOT
    amortize when there is little work per chip.
  - Worker count does NOT help: MORE workers (one-per-row) regressed
    N2368 (+24.6%, more fabric packets); FEWER workers add serial compute
    without cutting packet count (packet count ≈ rows/chunk_size, and
    chunk_size is capped at 128/num_tile_cols=3 by the L1-residency budget
    of input_cb). So the packet count — and thus the dominant fixed
    fabric cost — is essentially fixed regardless of worker count.
The only path to 1.5× here is cutting the per-chunk fabric fixed cost
(lighter handshake, batch multiple chunks per mcast, or skip the DRAM
round-trip for the gathered stats), or routing these tiny shapes to the
composite (which is faster for them).

**Conclusion / decision point.** The clean, in-scope levers are exhausted
(committed: L512 0.97→1.22×). The remaining gap to 3×/1.5× needs one of:
(a) relax the fixed-fidelity constraint, (b) an algorithmic change to the
RoPE/norm compute, or (c) cut the per-chunk fabric fixed cost (batched
mcast / no-DRAM-roundtrip AG). All three are larger efforts that change
correctness-sensitive code paths — surfacing to the user before pursuing.

## Worker-scaling sweep (2026-05-29) — N18944 is NOT compute-bound

Earlier I read "TRISC FW span 316us ≈ compute" and inferred compute-bound.
That inference is unsound: the TRISC span *includes* time stalled in
cb_wait_front (input_cb from the reader, stats_gathered_cb from the
writer/fabric) and cb_reserve_back back-pressure on output_cb. FW span
equality across RISCs does not imply compute-bound.

Decisive test instead of trusting zone spans: sweep the per-chip worker
cap (new `WAN_RMSNORM_WORKER_CAP` env override on
`pick_num_workers_tp_gt_1`) and watch how fused wall scales. chunk_size
stays pinned at 3 (H=1280, 40 cols, L1 budget) across the whole sweep, so
**fabric packet count is invariant to worker count** — fewer workers just
means more serial chunks per core. Pure compute would make wall ∝
rows/worker. Measured (self_sp4_N18944, TP=4 LINE, RoPE, num_links=2):

| cap | workers | rows/core | fused us | speedup | actual/compute-pred |
|---|---|---|---|---|---|
| 64 | 64 | 10 | 417.6 | 1.59× | 1.00 |
| 48 | 48 | 13 | 444.7 | 1.49× | 0.82 |
| 32 | 32 | 19 | 520.3 | 1.28× | 0.66 |
| 16 | 16 | 37 | 872.8 | 0.76× | 0.56 |

At 16 workers each core does 3.7× the rows of the 64-worker case yet wall
is only 2.1× — the rest of the per-core "compute" was hideable CB-stall.
The 48→64 slope (9.0 us per row/core) extrapolates to a **~327us floor**
(rows/core→0, i.e. infinite workers) → asymptotic best **2.03×**. Even
maxing the BH core budget (~106 workers, needs num_links≥2 for channel
headroom) lands ~378us → **1.76×**, only ~6% over today's 1.59×.

### Conclusion: 3× is structurally unreachable for large shapes

The ~327us floor is data-movement + pipeline fill/drain, not compute, so
adding cores cannot reach 3× (≈221us). This confirms the prior ceiling
analysis from a second, independent angle: the fused op's single
structural win over the composite is reading the input once instead of
twice (composite: pre-read + post-read + output-write = 3 DRAM passes;
fused: input-read + output-write = 2 passes), so the DRAM-traffic ceiling
is ~3/2 = 1.5× of composite's bandwidth-bound time — observed 1.59× is
already *at* that ceiling. Lifting the worker cap past 64 is therefore NOT
worth pursuing for the 3× goal (the confirmed-regression stash stays
parked). The env knob is retained purely as a diagnostic.

Next, to honor the "where does the time actually go" question: add
per-chunk stall-vs-math device zones (IN_WAIT / PRE / AG_WAIT / POST) to
attribute the ~327us floor to reader DRAM-read vs writer drain vs fabric
fill — localizes whether any lever remains (faster reader DRAM pattern,
faster writer drain) short of an algorithmic change.

## Compute-kernel zone split (2026-05-29) — direct stall-vs-phase zones

Per the user's note ("TRISC FW span includes CB stalls, not just math —
add detailed zones"), added three per-chunk `DeviceZoneScopedN` zones in
the compute kernel that tile the chunk timeline:
  - `RMS_PRE`  — PRE for-loop (input-read stall + square + row-reduce)
  - `RMS_AGWAIT` — the single `cb_wait_front(stats_gathered_cb)` (pure
    fabric all-gather stall)
  - `RMS_POST` — POST for-loop (norm + ×weight + RoPE math + output-drain
    back-pressure)

Measured on N12400 (388 tile rows, TP=4 LINE, RoPE, NON-traced device-op
test, 64 worker cores, num_links=2). Per-core totals summed over all
chunks, averaged across the 64 workers, stable across 8 op invocations
(run_host_id 5120–5127); a separate run that also wrapped each per-block
`cb_wait_front(input_cb)` in an `RMS_INWAIT` zone reproduced PRE/AGWAIT/
POST within <1%, so the zone instrumentation adds no measurable overhead:

| zone | avg us/core | share | what it is |
|---|---|---|---|
| RMS_PRE    | ~206 | 31% | input-read stall + sum-of-squares math |
| RMS_AGWAIT |  ~30 |  5% | fabric all-gather wait (per-chunk) |
| RMS_POST   | ~415 | 63% | norm + weight + RoPE math + output drain |
| (RMS_INWAIT, subset of PRE) | ~57 | — | reader-input stall portion of PRE |

(Absolute spans are ~2× the traced-bench TRISC FW span because this is
the NON-traced unit test — higher per-op dispatch latency, no inter-op
trace pipelining. The **relative split** is the signal.)

### Findings (confirm the worker-sweep conclusion from a 3rd angle)
1. **Fabric is hidden** — AGWAIT is only ~5% (≤42us even at its noisiest
   chunk boundary). Independently confirms the traced-profile AGWAIT≈0.
   There is no fabric lever left.
2. **Reader keeps up** — the input-read stall (INWAIT) is only ~57us of
   the 206us PRE; the other ~150us is the square+reduce math. Reader DRAM
   throughput is NOT the bottleneck.
3. **POST dominates (63%)** — norm + weight + RoPE compute plus the
   output-drain back-pressure. This is the phase to attack IF compute
   were the lever — but the worker sweep already showed adding workers
   (more POST parallelism) yields sublinear returns, i.e. POST is gated
   by shared DRAM-write bandwidth + pipeline fill/drain, not per-core
   math. So even the dominant phase can't be sped up by parallelism.

### Net: the "where does the time go" question is fully answered
Fabric ≈5% (hidden), reader-input ≈9%, the remaining ~86% is PRE-math +
POST(math+drain) — i.e. local compute and DRAM read/write traffic, the
exact ~327us data-movement floor the worker sweep extrapolated. All three
independent diagnostics (DRAM-pass-count ceiling = 1.5×, worker sweep
floor = ~327us/2.03×, zone split = fabric-hidden/compute+drain-bound)
agree: **the large-shape 3× target is not reachable without an
algorithmic change to the norm/RoPE math or relaxing fidelity (forbidden).
Current 1.59× is at the structural DRAM-traffic ceiling.**

The 3 coarse zones are kept in the kernel (zero-cost when the profiler is
disabled; the SDPA compute kernel keeps zones the same way) for future
re-profiling.

## POST decomposition (2026-05-29) — where exactly does POST's 63% go?

Per the user's follow-up ("dig deeper into POST — is it weight-input wait,
rope-input wait, weight+rope math, output-CB drain, or something else?"),
split RMS_POST into six per-row sub-zones mapping 1:1 to the POST
sub-phases. Same test (N12400, TP=4 LINE, RoPE, non-traced, 64 workers,
num_links=2). Sub-zone sum reproduces RMS_POST to **99%** (no blind spot).
Means/max/min are per-core totals over all chunks, averaged across 8 op
invocations (run_host_id 5123–5130 = the 8 mesh chips):

| sub-zone | mean us | max us | min us | share of POST | what it is |
|---|---|---|---|---|---|
| P_NORM   | 166 | 276 |  98 | 38% | reduce<AVG>(stats) + eps + rsqrt + first ×(1/rms) pass over all 40 cols |
| P_WEIGHT |  46 |  68 |  30 | 10% | weight wait + ×weight (mul_tiles_bcast_rows, 40 cols) |
| P_MM     |  36 |  53 |  24 |  8% | RoPE rotate matmul (×trans_mat, 40 cols) |
| P_COS    |  48 |  71 |  32 | 11% | cos wait + ×cos (40 cols) |
| P_SIN    |  46 |  68 |  30 | 10% | sin wait + ×sin (40 cols) |
| P_ADD    |  95 | 223 |  38 | 22% | add(rot,unrot)→output + **output_cb drain back-pressure** |
| RMS_POST | 440 | 599 | 321 | 100% | (sum = 440 ≈ RMS_POST) |

### Direct answer to "where does POST time go"
- **Normalization math (P_NORM): 38% — the single biggest bucket.** This
  is NOT a wait: input_cb is already resident (PRE didn't pop it) and stats
  are already gathered (AGWAIT covered that). It's the reduce<AVG> + the
  serial eps→rsqrt dependency chain (tile_regs round-trip + a mid-chain
  `cb_wait_front(reduce_result_cb)` per row) + the first full ×(1/rms) pass
  over all 40 col-tiles. Genuine TRISC compute.
- **Weight + RoPE math (P_WEIGHT+P_MM+P_COS+P_SIN): 39%.** Four separate
  full 40-col passes, each with its own reconfig + tile_regs cycle. The
  side-input waits folded in here (weight/cos/sin from the reader) are
  small — these are math-dominated, not input-wait-dominated. So the
  "rope-input wait" / "weight-input wait" buckets the question asked about
  are *not* where the time is.
- **Output-CB drain (P_ADD): 22%.** The add itself is trivial (40 adds);
  nearly all of P_ADD is `cb_reserve_back(output_cb)` back-pressure waiting
  for the writer to drain output to DRAM. **Smoking gun: P_ADD has the
  widest core-to-core spread of any zone — 38→223 us, 5.9×.** That extreme
  variance is the signature of contention on a shared resource (DRAM write
  bandwidth across all 64 workers), which is exactly what explains the
  sublinear core-scaling the worker sweep found. This is the "it can't all
  be compute" portion: ~22% is DRAM-write back-pressure, not math.

### Implications for the next step (no change made yet — awaiting go-ahead)
- The 22% output-drain (P_ADD) is shared-DRAM-write-bound at fixed output
  volume — hard to beat without writing less or rescheduling writes.
- The compute buckets (P_NORM 38% + weight/rope 39%) are 5 separate full
  40-col passes, each paying loop + reconfig_data_format + tile_regs
  fixed overhead. The cleanest fidelity-preserving lever is **pass fusion**:
  e.g. fold ×(1/rms) (sub-phase 1) and ×weight (sub-phase 2) into one pass
  (eliminates a full col-loop + a CB round-trip), and similarly chain the
  two RoPE muls. This cuts fixed per-tile overhead without touching the
  math precision (still HiFi4, same op order per element).

The six POST sub-zones are left in the kernel alongside the three coarse
zones (zero-cost when the profiler is disabled).

## Opt 1: fuse reduce + eps + rsqrt via post_reduce_op (2026-05-29)

P_NORM was 38% of POST and the largest single bucket. Its per-row cost
included a *separate* tile_regs cycle for the `mean + eps; rsqrt` step:
`add_tiles(reduce_result_cb, epsilon_cb)` → `rsqrt_tile` → pack → a
`reduce_result_cb` push/pop round-trip → a re-`cb_wait_front`, plus two
`reconfig_data_format` calls. That is a full DST acquire/commit/pack cycle
spent on one scalar per row.

**Change:** moved `+eps` and `rsqrt` into the reduce helper's
`post_reduce_op` callback, so they run on DST *after the reduce math,
before the pack* — inside the reduce's own tile_regs cycle. The separate
cycle, the CB round-trip, and the two eps reconfigs are gone. `+eps` is now
an SFPU scalar add (`add_unary_tile(dst_idx, eps_bits)`, fp32 scalar passed
as a compile-time arg) instead of a bf16 `epsilon_cb` `add_tiles`.

**Fidelity:** unchanged or better — the eps scalar is now fp32 (was bf16
`epsilon_cb`), and `rsqrt` sees the un-truncated fp32 mean still resident in
DST. Same HiFi4, same per-element op order. Correctness preserved:
COMPOSITE_VS_FUSED PCC 99.9955% (N2368) / 99.9932% (N12400); assert_quality
PCC 99.9996% — both unchanged vs pre-fusion.

**Perf (TP=4 LINE, BH 2x4), fused_us before → after / new speedup:**

| config | fused before | fused after | speedup |
|---|---|---|---|
| self_sp4_N18944  (RoPE) | 416.4 | 413.2 | 1.61× |
| self_sp8_N9472   (RoPE) | 271.0 | 271.7 | 1.23× |
| self_sp32_N2368  (RoPE) | 124.3 | 124.5 | 0.87× |
| cross_q_sp4_N18944      | 344.8 | 346.2 | 1.72× |
| cross_q_sp8_N9472       | 216.2 | 219.0 | 1.35× |
| cross_q_sp32_N2368      |  90.5 |  87.4 | 1.12× |
| cross_k_prompt_L512     |  53.3 |  42.6 | 1.22× |

The win lands where P_NORM is the biggest POST fraction — the **no-RoPE**
shapes. **L512 jumped 20% (0.97→1.22×, now over 1.0×)**; cross_q N2368
+3.4% (1.08→1.12×). RoPE shapes are flat: their POST is dominated by the
RoPE matmul + output-drain back-pressure, not norm, so removing norm fixed
cost barely moves them. Composite numbers reproduced near-exactly (663.4 vs
663.6), confirming the deltas are real, not run-to-run noise.

**Still short of targets** (small 1.5×, large 3×). N2368-RoPE (0.87×) is the
worst — its time is RoPE/drain-bound, addressed by the next opts (pass
fusion of the weight/RoPE muls; output-drain mitigation).

## Opt 2 (NEGATIVE RESULT): fuse RoPE P_COS + P_ADD via dest-reuse (2026-05-29)

Following Opt 1's recommendation to "chain the two RoPE muls", tried
collapsing the four trailing RoPE passes (P_MM, P_COS, P_SIN, P_ADD) into
three: reorder P_SIN ahead of cos so `rotated` already holds `rotated*sin`,
then in one tile_regs cycle compute `x_w*cos` into DST (HiFi4 `mul_tiles`)
and fold `+ rotated*sin` via
`binary_dest_reuse_tiles<ELWADD, DEST_TO_SRCB>` straight to `output_cb`.
This removes P_COS's pack-to-intermediate round-trip and P_ADD's separate
intermediate load — a full 40-col pass eliminated.

**Correctness:** preserved. 19/19 RoPE + COMPOSITE_VS_FUSED bisect tests
PASS at PCC ≥ 99.999% (the ELWADD dest-reuse is single-pass/fidelity-neutral;
the cos mul stays HiFi4, same per-element op order).

**Perf (TP=4 LINE, BH 2x4), fused_us before → after:**

| config | fused before | fused after | speedup |
|---|---|---|---|
| self_sp4_N18944 (RoPE) | 413.2 | 408.7 | 1.62× (was 1.61×) |
| self_sp8_N9472  (RoPE) | 271.7 | 274.5 | 1.22× (was 1.23×) |
| self_sp32_N2368 (RoPE) | 124.5 | 124.1 | 0.87× (unchanged) |

**Flat — within run-to-run noise.** Why it can't help: the POST
decomposition already showed P_ADD's 22% is almost entirely **output_cb
drain back-pressure** (DRAM write bandwidth), not the add math, and the
worker sweep showed POST is gated by shared DRAM-write bandwidth, not
per-core compute. Removing a compute pass frees TRISC math cycles that
were already hidden behind the writer drain — the writer still gates
wall-clock identically. Compute-pass fusion in the tail is a dead lever
at the current DRAM-traffic ceiling.

**Decision:** reverted (kept the simpler proven 4-pass kernel). The change
is preserved in git stash (message "RoPE P_COS+P_ADD dest-reuse fusion ...")
if a future change ever lifts the drain bottleneck and makes tail compute
the binding constraint. Same conclusion holds for the analogous
×(1/rms)+×weight fusion (both are broadcasts → blocked by
`binary_dest_reuse`'s BroadcastType::NONE limitation anyway). **The
remaining levers are algorithmic (reduce DRAM traffic, e.g. cross-op
fusion keeping output L1-resident for the consumer), not intra-kernel
pass fusion.**

## Opt 3: deep input read + single per-row barrier (2026-05-29)

The reader previously barriered after every `block_size`=2 input tiles, so
only 2 DRAM reads were ever outstanding and each barrier exposed a full
read round-trip. On the **no-RoPE** shapes (read-latency-bound, ~44% of
BH's 512 GB/s peak) this was the dominant cost.

**Change (reader.cpp):** issue a whole tile-row's `num_tile_cols`=40 reads,
then ONE `noc_async_read_barrier()` (40 reads in flight vs 2). `input_cb` is
sized to an integer multiple of `num_tile_cols`, so a row reservation never
wraps the ring (wr_ptr stays contiguous). For the RoPE path, cos/sin reads
are now issued *first* (few tiles), then the input row, under that same
single barrier; input is pushed before cos/sin so compute's PRE phase isn't
delayed (cos/sin aren't consumed until the much-later POST RoPE sub-phase).
This collapses two per-row barriers into one without delaying input.

**Correctness:** preserved. 15/15 RoPE + COMPOSITE_VS_FUSED bisect tests
PASS at PCC ≥ 99.999%; bench PCC checks PASS. Same tiles, same DST, only
the read-issue ordering and barrier granularity changed.

**Perf (TP=4 LINE, BH 2x4), fused_us before → after / new speedup:**

| config | fused before | fused after | speedup |
|---|---|---|---|
| self_sp4_N18944  (RoPE) | 413.2 | 416.4 | 1.59× (was 1.62×) |
| self_sp8_N9472   (RoPE) | 271.7 | 263.3 | 1.27× (was 1.23×) |
| self_sp32_N2368  (RoPE) | 124.5 | 124.2 | 0.87× (unchanged) |
| cross_q_sp4_N18944      | 346.2 | 321.9 | **1.85×** (was 1.72×) |
| cross_q_sp8_N9472       | 219.0 | 195.4 | **1.51×** (was 1.35×) |
| cross_q_sp32_N2368      |  87.4 |  85.4 | 1.14× (was 1.12×) |
| cross_k_prompt_L512     |  42.6 |  43.0 | 1.20× (~flat) |

**Big wins on the read-bound no-RoPE shapes** (cross_q_N18944 +7%,
cross_q_N9472 +12%, both well past 1.5×). RoPE shapes are compute-bound
(TRISC dominated by the RoPE matmul/cos/sin per the profiling) so deep
reads don't help them; the merged-barrier ordering recovered the small NoC
contention regression deep reads first introduced on RoPE-N18944 (was 421
with two barriers → 416). Net clearly positive; committed.

## Opt 4: deep output-write — row-deep output_cb + per-row flush (2026-05-29)

**Overturns the Opt-2 "output drain is a dead lever" conclusion.** That
conclusion was drawn while output_cb was only `block_size*2`=4 tiles, so
the writer flushed (`noc_async_writes_flushed`) every 2 tiles — write
concurrency was capped at 2 in-flight, which IS the back-pressure P_ADD
measured (22% of POST, 5.9× core-to-core spread = DRAM-write contention).
The drain wasn't a fixed floor; it was throttled by a too-small CB, exactly
like the reader was throttled by per-2-tile read barriers before Opt 3.

**Change (program factory + writer_mux):**
- `output_cb` sized `block_size*2` → `2 * padded_row_tiles` (2 full rows).
- MUX writer drains a row by issuing each col-block's writes as soon as
  compute pushes it (incremental `cb_wait_front` on the padded cumulative
  count keeps within-row overlap), then does ONE `noc_async_writes_flushed`
  + ONE `cb_pop_front(padded_row_tiles)` at the end of the row. Writes now
  pipeline at DRAM depth; the 2-row CB lets compute fill row r+1 while row r
  drains. This is the write-side mirror of Opt 3's deep read. (Note: this is
  NOT the earlier negative "Experiment 1" — that waited for the *whole row*
  before issuing any write, serializing compute→write; here writes issue
  per-block and only the flush/pop is batched.)

**Correctness:** preserved. 15/15 RoPE + COMPOSITE_VS_FUSED bisect tests
PASS, PCC ≥ 99.999%. Same tiles written to the same DRAM addresses; only
CB size + flush granularity changed.

**Perf (TP=4 LINE, BH 2x4), fused_us before(Opt3) → after / new speedup,
two stable runs:**

| config | fused before | fused after | speedup |
|---|---|---|---|
| self_sp4_N18944  (RoPE) | 416.4 | 417.0 | 1.59× (flat) |
| self_sp8_N9472   (RoPE) | 263.3 | **244.9** | **1.36×** (was 1.27×) |
| self_sp32_N2368  (RoPE) | 124.2 | 119.7 | 0.90× (was 0.87×) |
| cross_q_sp4_N18944      | 321.9 | **311.6** | **1.91×** (was 1.85×) |
| cross_q_sp8_N9472       | 195.4 | 193.8 | 1.52× (was 1.51×) |
| cross_q_sp32_N2368      |  85.4 |  84.1 | 1.16× (was 1.14×) |
| cross_k_prompt_L512     |  43.0 |  42.5 | 1.21× (flat) |

**Net positive, no regressions.** Biggest win is N9472 RoPE (+7%, gated on
output drain because at 296 rows/64 workers ≈ 4.6 rows/core the drain tail
is a larger fraction). cross_q_N18944 (no-RoPE target) 1.85→1.91×. The
RoPE-N18944 and L512 flats are expected — N18944-RoPE is compute-bound
(drain hidden behind the long RoPE math), L512 has 1 row/core (no
cross-row overlap to gain). Committed.

### Where this leaves the targets
no-RoPE: N9472 1.52× ✓, N18944 1.91× (→3× still needs ~40% more DRAM-write
util; now ~61% of 512 GB/s peak). small N2368 1.16× (no-RoPE) / 0.90×
(RoPE) — still short of 1.5×, the next focus. The Opt-2 "drain is a dead
lever / 1.5× structural ceiling" claim is now disproven by both Opt 3 (read)
and Opt 4 (write) beating 1.5× on no-RoPE — the real ceiling is DRAM
*utilization*, and there is still headroom.

## Re-profile after deep-r/w + two dead-end levers (2026-05-29)

Re-profiled N12400 RoPE (TP=4 LINE, non-traced, 64 workers) on the
post-Opt-4 (deep-write) kernel to find the *new* bottleneck. POST sub-zone
shares, before (pre-deep-write) → after:

| sub-zone | before share | after share | note |
|---|---|---|---|
| P_NORM   | 38% (166us) | **49–56% (194–258us)** | now dominant |
| P_WEIGHT | 10% | 12% (46us) | flat |
| P_MM     |  8% |  9% (36us) | flat |
| P_COS    | 11% | 12% (48us) | flat |
| P_SIN    | 10% | 11% (46us) | flat |
| P_ADD    | **22% (95us)** | **7% (27us)** | deep-write killed drain BP |

**Confirmed: Opt-4 (deep-write) crushed the output-drain back-pressure
(P_ADD 22%→7%).** P_NORM is now the single dominant POST bucket — and it's
~4–5× larger than the structurally-identical P_WEIGHT/P_COS/P_SIN passes
(all are full 40-col multiplies). The extra ~150us is the per-row
reduce<AVG>→eps→rsqrt→serial `cb_wait_front(reduce_result_cb)` chain that
gates sub-phase-1's ×(1/rms) multiply.

### Dead lever A — intra-kernel pass fusion (P_COS+P_ADD dest-reuse)
Re-tried the stashed COS+ADD fusion (proven PCC-correct) on the hypothesis
that deep-write removed the drain that had previously hidden the saved
compute. **Result: still a regression** — RoPE self_N18944 426 (vs 416),
N9472 290-ish, N2368 123; no-RoPE flat (path untouched). ~+2% on all RoPE
shapes. **Root cause (now understood):** DST holds only `block_size`=2 tiles
(fp32_dest_acc_en + half-sync), so fusing two ops forces *per-block*
interleaving (mul then dest-reuse-add inside one tile_regs cycle), which
needs a `reconfig_data_format` + init **per block** (20 blocks/row) instead
of once per pass. That per-block reconfig overhead exceeds the saved
pack+CB-roundtrip of one pass. Fusion is structurally dead here, not just
drain-masked. **Do not retry pass fusion** unless DST capacity grows.

### Dead lever B — dst_full_sync_en=true (block_size 2→4)
The DST-capacity insight suggested raising block_size by enabling
`dst_full_sync_en` (get_dest_reg_count: full-sync skips the ÷2, giving 4
tiles with fp32_dest_acc still on — **precision-neutral**, it's a math↔pack
scheduling mode, not a fidelity knob). Required changes in *two* places:
the device-op `init_device_compute_kernel_config` 7th arg AND the program
factory `ComputeConfig{.dst_full_sync_en=...}` (it was silently defaulting
false — a mismatch would corrupt). Built, RoPE PCC PASS. **Result:
regressed everything** — self_N18944 426→463, N9472 249→290, cross_q_N18944
310→327, cross_q_N9472 190→209 (~+6–16% across the board). Full-sync
serializes MATH and PACK on DST; that lost pipeline overlap costs more than
the halved per-block count saves. **Reverted both edits. Keep
dst_full_sync_en=false / block_size=2.**

### Takeaway
The compute kernel is at a local optimum for the per-pass structure:
half-sync math/pack overlap + minimal-reconfig single-pass loops beat both
fusion and bigger blocks. The remaining large bucket (P_NORM's reduce→rsqrt
serialization) and the no-RoPE DRAM-utilization gap are the open levers; the
intra-pass micro-structure is tapped out.

## P_NORM split (2026-05-29) — the dominant bucket is the bcast_cols mul, not the reduce

Split `P_NORM` into two sub-zones to see whether the dominant cost is the
reduce→eps→rsqrt chain or the ×(1/rms) multiply:
  - `P_NRED` — `reduce<AVG,REDUCE_ROW>` + fused eps+rsqrt + `cb_wait_front(reduce_result_cb)`
  - `P_NMUL` — sub-phase-1 `mul_tiles_bcast_cols(input, 1/rms)` over all 40 col-tiles

Measured (N12400, TP=4 LINE, RoPE, non-traced, 64 workers, per-core totals
averaged across cores):

| sub-zone | us | what it is |
|---|---|---|
| P_NRED | ~75 | reduce + eps + rsqrt + the serial reduce_result wait |
| P_NMUL | ~128 | the ×(1/rms) **COL-broadcast** multiply (40 col-tiles) |

**Key finding: P_NMUL (~128us) is the larger half of P_NORM, and it is
~2.8× the structurally-identical `P_WEIGHT` (~46us)** — which does the same
40-col multiply but with **ROW** broadcast (`mul_tiles_bcast_rows`). The
prior log framed P_NORM as "reduce→rsqrt serialization"; the split shows the
multiply dominates, and the multiply is expensive *only because it is a COL
broadcast*.

**Root cause (arch-confirmed, BH `tt_llk_blackhole/llk_lib/llk_math_eltwise_binary.h`):**
`_llk_math_eltwise_binary_standard_` cannot fold the 4 tile-faces into one
MOP run for `BroadcastType::COL` — it issues one MOP run per face-row plus a
serializing `SETRWC(CLR_B)` each (4 dispatches/tile). ROW broadcast and full
`mul_tiles` fold all 4 faces into a single MOP run (1 dispatch/tile). So
bcast_cols does ~4× the MOP dispatches of the equivalent bcast_rows/mul_tiles
— the measured ~2.8× wall cost. This is the one concrete, fidelity-neutral
compute lever the earlier "intra-pass micro-structure is tapped out" claim
missed: it predates this P_NRED/P_NMUL split.

### Why REDUCE_ROW forces COL broadcast
`reduce<REDUCE_ROW>` reduces the feature/W dim → produces a **column vector**
(per-token 1/rms down col 0 of the tile). Applying it to all features needs
COL broadcast. This is the canonical in-tree pattern (rmsnorm/layernorm all
use `bcast_cols`), so the cost is structural to the layout, not a bug.

### DEAD END (lever C): naive per-row `unary_bcast<COL>` splat + `mul_tiles`
The canonical *fast* alternative (layernorm_large_tensor.cpp:242-320) splats
the 1/rms column to a full tile once via `unary_bcast<COL>`, then uses cheap
full `mul_tiles` (1 MOP/tile) across all cols. Converting MOP dispatches
40×4=160 → 1×4(splat)+40×1=44 should be a ~3.6× reduction in multiply work.
Tried it (per-row): **regressed** — P_NMUL 128→141us, P_NRED 75→84us,
RMS_POST 505→527us. Correctness preserved (PCC ≥ 99.999%), but slower.

**Root cause of the regression (now understood):** two fixed costs the naive
per-row version pays that the layernorm reference hides:
  1. `unary_bcast_init<COL>` is a **full hw_configure** (heavy), paid once
     per row. The reference pays it per row too but amortizes it.
  2. The splat round-trip (MATH→DST→PACK→L1→UNPACK→DST) is **exposed**: in
     RMSNorm the 40 muls depend directly on the splatted tile, so the
     round-trip is on the critical path. The layernorm reference hides this
     latency behind its mandatory `x − E[x]` subtraction/copy pass (the
     `cb_xmm` round-trip at layernorm_large_tensor.cpp:264-312) — a bubble
     RMSNorm simply does not have (no mean-subtraction).

Reverted to `mul_bcast_cols_init_short` + `mul_tiles_bcast_cols`. The
P_NRED/P_NMUL diagnostic zones are kept (zero-cost when profiler off).

### DEAD END (lever C, take 2): *batched* reduce + COL-splat + `mul_tiles`
The per-row splat (above) was slow because of (1) per-row `unary_bcast_init`
hw_configure and (2) an *exposed* splat round-trip. Both should be fixable by
**batching across the chunk**: hoist `unary_bcast_init<COL>` to once-per-chunk,
do all `rows_in_chunk` reduces first, then all `rows_in_chunk` splats into a
new `splat_cb` (sized `chunk_size_rows` fp32 tiles), then per-row cheap full
`mul_tiles(input, splat_cb)`. The round-trips of consecutive rows hide behind
each other instead of stalling the muls.

Implemented (program factory: `splat_cb`=c_21, `reduce_result_cb` grown to
`chunk_size_rows`, CT arg 35; compute: batched PRE-reduce→splat→per-row mul).
Correctness: **43/43 device-op tests PASS** (all TP, weight/bias/per_head_*),
multihead-RoPE PCC ≥ 0.999. Perf (TP=4 LINE bench vs the c3ad603643c baseline):

| config | baseline us | batched-splat us | delta |
|---|---|---|---|
| self_N18944 (RoPE)  | 417.0 | 417.8 | +0.8 |
| self_N9472  (RoPE)  | 244.9 | 243.8 | −1.1 |
| self_N2368  (RoPE)  | 119.7 | 119.6 | −0.1 |
| cross_q_N18944      | 311.6 | 309.9 | −1.7 |
| cross_q_N9472       | 193.8 | 193.4 | −0.4 |
| cross_q_N2368       |  84.1 |  85.0 | +0.9 |
| cross_k_L512        |  42.5 |  42.8 | +0.3 |

**Result: a wash** — every delta is within run-to-run noise (±1–2us). The
large no-RoPE shapes improved ~0.5%; small shapes regressed ~1%. **Confirms
finding #3 (zone split): the POST norm-multiply is drain-hidden** — the 40
col-tile muls overlap the output drain / AG wait, so cutting their MOP-dispatch
count (160→44) does not move the wall clock. The batched version was numerically
clean and avoided the per-row regression, but the work it saved was already free.

**Reverted** (kept c3ad603643c as HEAD) because the wash carried real L1 cost:
the extra `splat_cb` + grown `reduce_result_cb` spend the L1 that is the binding
constraint for worker count and for any future *drain*-focused optimization. If
a drain win later re-exposes POST compute as the bottleneck, re-introduce this.

**Strategic takeaway:** stop optimizing POST *compute* (norm-mul, weight-mul,
RoPE matmul are all drain/fabric-hidden per findings #1–#5). The only remaining
levers that can move the wall are the **output drain (writer NoC)** and the
**fabric AG**. Next focus there.
