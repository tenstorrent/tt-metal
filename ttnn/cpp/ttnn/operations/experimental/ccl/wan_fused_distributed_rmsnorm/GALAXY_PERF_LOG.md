# wan_fused_distributed_rmsnorm — Wormhole Galaxy read-overlap optimization log

Machine: WH Galaxy (4x8), TP=4 LINE on a 1x4 submesh, 4 fabric links.
Bench: `test_wan_rmsnorm_bench_composite_tp4_ring_galaxy` (fused, traced, 100 iters).
Correctness: `test_wan_rmsnorm_correctness_tp4_galaxy` (fused vs composite PCC, must stay ~0.999).

All times are fused µs/iter (lower = better). Composite reference (committed baseline):
N18944-rope 1244, N9472-rope 609, N2368-rope 198, N18944 1037, N9472 517, N2368 151, L512 75.

## Baseline (commit 5e1eb846dea, fix landed) — fused µs/iter

| config | seq_len | RoPE | fused µs |
|---|---:|:--:|---:|
| self_sp4_N18944    | 18944 | Y | 981.7 |
| self_sp8_N9472     | 9472  | Y | 497.4 |
| self_sp32_N2368    | 2368  | Y | 244.1 |
| cross_q_sp4_N18944 | 18944 | N | 592.7 |
| cross_q_sp8_N9472  | 9472  | N | 321.8 |
| cross_q_sp32_N2368 | 2368  | N | 142.9 |
| cross_k_prompt_L512| 512   | N |  64.3 |

## Ablation findings (exposed cost = baseline − ablation, % of baseline)

| config | rope-rd | input-rd | out-wr | fabric | gather/scatter |
|---|---:|---:|---:|---:|---:|
| N18944 RoPE | 242 (25%) | 280 (29%) | 76 (8%) | 9 (1%) | 4 (0%) |
| N9472 RoPE  | 59 (12%)  | 109 (22%) | 4 (1%)  | 5 (1%) | 9 (2%) |
| N2368 RoPE  | 25 (10%)  | 40 (17%)  | 1 (1%)  | 5 (2%) | 8 (3%) |
| N18944 no-rope | — | 114 (19%) | 46 (8%) | 16 (3%) | 18 (3%) |
| N9472 no-rope  | — | 35 (11%)  | 23 (7%) | ~0 | ~0 |
| N2368 no-rope  | — | 17 (12%)  | 2 (1%)  | 3 (2%) | 7 (5%) |
| L512 no-rope   | — | 2 (3%)    | 3 (5%)  | 1 (1%) | 2 (3%) |

Conclusion: latency-bound reads on the critical path. cos/sin reads (RoPE) and input
reads are the two big exposed costs; fabric + gather/scatter are hidden. Ideas A–F target
read overlap. Each idea below: implement → correctness (PCC) → measure → keep if faster.

---

## Idea A — decouple cos/sin reads from the input barrier (reader)

Issue the input row first and barrier it alone (cos/sin not yet in flight) so
compute's PRE starts immediately; issue cos/sin reads + their own barrier AFTER
the input push, so cos/sin DRAM latency overlaps the PRE pass instead of gating
it. Reader-only change (JIT, no rebuild).

Correctness: fused-vs-baseline PCC = 1.000000 (worst 0.999997) — pure reorder,
output unchanged. PASS.

Perf (fused µs/iter, two runs, vs baseline):

| config | baseline | A (run1/run2) | Δ |
|---|---:|---:|---:|
| N18944 RoPE | 981.7 | 956.6 / 954.0 | **−2.8%** |
| N9472 RoPE  | 497.4 | 497.3 / 495.7 | ~flat |
| N2368 RoPE  | 244.1 | 230.2 / 230.5 | **−5.6%** |
| N18944 no-rope | 592.7 | 592.8 / 591.9 | 0 (no cos/sin) |
| N9472 no-rope  | 321.8 | 323.0 / 323.3 | noise |
| N2368 no-rope  | 142.9 | 143.3 / 143.0 | noise |
| L512 no-rope   | 64.3  | 64.6 / 64.7   | noise |

Win on RoPE shapes (cos/sin reads partially hidden behind PRE), no-rope
unaffected, no regressions. KEPT (committed). Doesn't fully reclaim the 25%
cos/sin tax — the reader still serializes on the per-row cos/sin barrier; only
compute's PRE overlaps it. Ideas B (dual-NoC) / C (deeper pipeline) target the rest.

## Idea B — dual-NoC input read (DEAD END: structural blocker)

Plan: run the reader (NCRISC) in DM_DYNAMIC_NOC mode and split the input row
across NoC0/NoC1 (alternate tiles) to use both NoCs' bandwidth (input read is
the #1 exposed cost, latency-bound at ~31% of peak on one NoC).

Result: program build FATAL — `program.cpp:731: noc_modes.size() <= 1`. A
program requires ALL its kernels to share one NoC mode. Putting just the reader
on dynamic NoC conflicts with the dedicated-NoC writer/compute AND the fabric
MUX kernels (created by FabricMuxConfig — NoC mode not controllable here).
Converting the whole program, including the fabric MUX, to dynamic NoC is out
of scope and risky. **Reverted.** Per-kernel dual-NoC is not available for this
op. (The input-read bandwidth lever is pursued via Idea C instead.)

## Idea C — deepen the read pipeline (DEAD END)

(i) Bigger input_cb (WAN_RMSNORM_INPUT_CB_CHUNKS 2->3/4): **L1-blocked**. At
nt=40/chunk=3 the resident CBs already sit near the cap; chunks=3 overflows
(1.63 MB > 1.50 MB), chunks=4 trips the auto-streaming path (incompatible with
chunk_size_rows=3). No headroom to prefetch deeper.

(ii) Chunk-batched input reads (issue all chunk rows under one barrier =
rows_in_batch*nt in flight vs one row): correct (PCC ~1.0) but **regresses
every shape** vs Idea A: N18944 no-rope 592->637 (+7.6%), N9472-RoPE 497->530
(+6.6%), N18944-RoPE 958->1001 (+4.5%), N2368 +5-7%. Confirms the Blackhole
finding on WH: more in-flight depth adds NoC/DRAM-controller contention AND
delays compute's PRE start (PRE now waits for the whole chunk's input), both
outweighing latency-hiding. The op sits at a delicate read local-optimum.
**Reverted.** Read-depth is not a lever here.

## Idea D — finer (per-block) input push, RoPE-gated (KEPT)

Push input in block_size groups (issue+barrier+push per block) so compute's PRE
(cumulative cb_wait_front) starts squaring block 0 while later blocks read. The
cos/sin reads that Idea A defers after the input then overlap more compute.

First tried for ALL shapes: big RoPE win but **+9.3% regression on N9472 no-rope**
(no cos/sin to overlap → finer push is pure per-block-barrier overhead, the
pre-Opt-3 cost). Gated on `fuse_rope`: RoPE uses finer push, no-rope keeps the
single-barrier row push (Idea A).

Correctness: fused-vs-baseline PCC worst 0.99988 (>0.999). PASS.

Perf (fused µs/iter, two runs, vs Idea A):

| config | Idea A | gated D | Δ |
|---|---:|---:|---:|
| N18944 RoPE | 958 | 842.6 / 844.4 | **−12%** |
| N9472 RoPE  | 497 | 480.3 / 478.5 | **−3.6%** |
| N2368 RoPE  | 229 | 232.7 / 232.7 | +1.7% |
| N18944 no-rope | 592 | 593.3 / 592.4 | ~0 |
| N9472 no-rope  | 323 | 322.8 / 322.7 | ~0 (regression gone) |
| N2368 no-rope  | 142 | 142.2 / 143.1 | ~0 |
| L512 no-rope   | 65  | 65.1 / 64.6   | ~0 |

KEPT (committed). RoPE shapes win big; no-rope untouched. The Idea-A + Idea-D
combination on RoPE is the headline: N18944-RoPE 982 (pre-opt) -> 843 (-14%).

## Running totals vs pre-optimization baseline (fused µs/iter)

| config | baseline | current (A+D) | speedup-of-fused |
|---|---:|---:|---:|
| N18944 RoPE | 981.7 | 843 | 1.16x |
| N9472 RoPE  | 497.4 | 479 | 1.04x |
| N2368 RoPE  | 244.1 | 233 | 1.05x |
| N18944 no-rope | 592.7 | 592 | ~1.0x |
| N9472 no-rope  | 321.8 | 323 | ~1.0x |
| N2368 no-rope  | 142.9 | 142 | ~1.0x |
| L512 no-rope   | 64.3  | 65  | ~1.0x |

## Idea E — reduce cos/sin cost (NOT pursued this round)

(i) **bf16 cos/sin** (halve the ~19 MB cos/sin read on N18944, the single
biggest lever left for RoPE): this is a **precision trade** — it lowers RoPE
rotation fidelity and would fail the fused-vs-baseline guard by construction
(needs a torch-reference check instead). It violates the standing "no loss of
compute fidelity/precision" constraint, so NOT done autonomously. Flagged for a
fidelity decision — highest remaining upside if precision can be relaxed.

(ii) **batch cos/sin reads** across the chunk: precision-neutral, but only cuts
the per-row cos/sin barrier count (3->1 per chunk), NOT the dominant cos/sin
*byte* cost; cos/sin already overlap compute via Idea A. Low expected upside vs
a moderate reader restructure that risks the committed A+D wins. Deferred.

## Idea F — reshard input / cross-op fusion (NOT pursued: large effort)

Reshard input to block/width-sharded for contiguous large reads, or fuse with
the producing op so input arrives L1-resident (eliminating the input read
entirely, ~19% on N18944 no-rope). Both are cross-op architectural changes far
beyond a kernel tweak — a separate workstream. Deferred.

## Session summary

Committed wins: Idea A (decouple cos/sin barrier) + Idea D (RoPE-gated finer
input push). Combined: **N18944-RoPE 982 -> 843 us (-14%)**, N9472-RoPE -4%,
N2368-RoPE -5% (A) net ~flat (D), no-rope shapes unchanged. Dead ends: B
(dynamic dual-NoC blocked by program-wide noc_mode constraint + fabric MUX),
C (deeper reads — L1-blocked / contention regression). Remaining levers need a
fidelity decision (E-bf16) or a cross-op rewrite (F).

## Idea G — move scalar/eps/trans_mat population reader -> writer (KEPT)

The reader generated the reduce-scalar SUM/AVG + epsilon CBs and read the
trans_mat tile (NoC read + barrier) BEFORE its input loop, so its first NoC op
was trans_mat, not input. Moved this population to the MUX writer (runs before
its fabric handshake, ready by the time compute starts), gated by a new
`scalars_in_writer` reader CT flag (=use_mux). Reader's first op is now the
input read. Legacy/TP=1 path unchanged (reader still populates).

Correctness: fused-vs-baseline PCC ~1.0 (worst 0.999998). PASS.

Perf (fused µs/iter, two runs, vs A+D):

| config | A+D | +writer-pop | Δ |
|---|---:|---:|---:|
| N18944 RoPE | 843.9 | 837.9 / 839.6 | −0.6% |
| N9472 RoPE  | 480.5 | 478.1 / 478.1 | −0.5% |
| N2368 RoPE  | 232.7 | 226.3 / 226.5 | **−2.7%** |
| N18944 no-rope | 594.6 | 588.8 / 588.9 | −1.0% |
| N9472 no-rope  | 324.9 | 321.0 / 321.1 | −1.2% |
| N2368 no-rope  | 142.9 | 140.8 / 141.0 | −1.4% |
| L512 no-rope   | 64.5  | 64.2 / 64.5   | ~0 |

Small but consistent win on EVERY shape (no regressions) — reader starts input
ASAP, improving pipeline fill. Biggest on small shapes (startup is a larger
fraction). KEPT (committed).

## Idea H — chunk-match the reader: cos/sin AFTER the chunk's input (KEPT)

The reader looped per-row (input row -> cos/sin row -> ...), but compute loops
per-chunk (PRE over all chunk rows -> POST). So each row's cos/sin read sat
*between* consecutive input rows, stalling the next row's input on the NoC even
though cos/sin aren't consumed until POST. Fix: read all the chunk's input rows
first (per-row, finer push from Idea D unchanged), THEN the chunk's cos/sin
(still pushed per row). Matches compute's loop structure; input flows
uninterrupted. Distinct from the failed Idea C — input is NOT batched/one-
barriered, only cos/sin is relocated. No-rope unaffected (compile-time skip).

Correctness: fused-vs-baseline PCC worst 0.999996. PASS.

Perf (fused µs/iter, two runs, vs Idea G):

| config | G | H | Δ |
|---|---:|---:|---:|
| N18944 RoPE | 838.8 | 837.3 / 840.2 | ~flat |
| N9472 RoPE  | 478.1 | 446.6 / 447.5 | **−6.5%** |
| N2368 RoPE  | 226.4 | 206.4 / 206.7 | **−8.7%** |
| no-rope (all) | — | unchanged | 0 |

KEPT (committed). N2368-RoPE now ~0.96x composite (was 0.85x), N9472-RoPE 1.36x
(was 1.27x). N18944-RoPE flat (most rows/worker → already amortized; input-bound).

## Idea I — read weight BEFORE rope (compute consumes weight first) (KEPT)

Extends Idea H: at the chunk boundary the reader now reads weight/bias first,
then cos/sin — matching compute's POST order (sub-phase 2 weight mul, then
sub-phase 3 rope). Broadcast weight was already chunk-gated (read once after the
first chunk); just relocated before the cos/sin block. Lets POST sub-phase 2 get
weight without waiting behind the chunk's cos/sin reads.

Correctness: fused-vs-baseline PCC worst 0.999994. PASS.

Perf (fused µs/iter, two runs, vs Idea H):

| config | H | I | Δ |
|---|---:|---:|---:|
| N18944 RoPE | 838.7 | 788.7 / 791.1 | **−5.9%** |
| N9472 RoPE  | 447.0 | 429.4 / 428.5 | **−4.0%** |
| N2368 RoPE  | 206.5 | 205.7 / 205.7 | ~flat |
| no-rope (all) | — | unchanged | 0 |

KEPT (committed). N18944-RoPE finally moved (was flat for H) -> 1.57x composite
(was 1.48x); N9472-RoPE 1.42x (was 1.36x). Session arc N18944-RoPE: 982 -> 790 (-20%).

## Ablation re-run with weight-read column (post A-I)

Added ablation 6 (WAN_ABLATION=6 -> skip weight/bias NoC reads, keep CB APIs).
Exposed cost = baseline - ablation, % of fused baseline:

| config | base µs | input | rope | weight | out-wr | fabric | gather |
|---|---:|---:|---:|---:|---:|---:|---:|
| N18944 RoPE | 788.1 | 19% | 13% | 2% | 10% | 3% | 2% |
| N9472 RoPE  | 429.6 | 13% | 3% | ~0 | 2% | 1% | 1% |
| N2368 RoPE  | 205.8 | 3% | 0.4% | ~0 | 3% | 2% | 2% |
| N18944 no-rope | 588.5 | 19% | — | ~0 | 8% | 3% | 3% |
| N9472 no-rope  | 321.0 | 10% | — | ~0 | 9% | ~0 | ~0 |
| N2368 no-rope  | 141.6 | 12% | — | 3% | 2% | 3% | 6% |
| L512 | 64.7 | ~0 | — | 1% | 8% | 1% | 1% |

Weight read is HIDDEN (<=2.4%, several deltas negative = noise): broadcast weight
is read once/worker (face-row partial reads), deferred to overlap chunk-0 compute
+ fabric, and Idea I placed it before rope. Not a nail. Ranking unchanged:
input #1 (-> Idea F), output-write #2 on no-rope/large (8-10%, best unaddressed
lever), rope-read #2 on N18944-RoPE only (13%).

## Barrier-read threshold sweep + heuristic (KEPT)

Made the input read's push+barrier granularity a tunable `input_barrier_tiles`
(CT arg; WAN_BARRIER_TILES env override) and swept {2,4,6,8,16,40} on all shapes.

Sweep (fused µs/iter), RoPE | no-rope:

| bar | N18944r | N9472r | N2368r | N18944 | N9472 | N2368 | L512 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2  | 739 | 432 | 216 | 613 | 354 | 144 | 66 |
| 4  | 791 | 430 | 206 | 578 | 353 | 137 | 64 |
| 8  | 845 | 442 | 204 | 570 | 350 | 136 | 64 |
| 16 | 889 | 461 | 205 | 557 | 338 | 134 | 64 |
| 40 (whole row) | 922 | 491 | 216 | 589 | 321 | 141 | 64 |

Findings:
1. **The reference DRAM-contention heuristic ((512/num_readers)*1152/tile_bytes,
   ~4 for 64 readers) does NOT transfer.** The threshold here is dominated by
   compute-overlap (how finely PRE is fed), not in-flight contention. The auto
   heuristic REGRESSED no-rope N9472 321->352 (+10%): no-rope wants the whole
   row in one push (no cos/sin to overlap; finer push is pure overhead).
2. **RoPE wants finer push the deeper the worker** (more rows/worker = more
   pipeline): N18944-RoPE (9.3 rows/wkr) opt=2, N9472-RoPE (4.6) opt=2-4,
   N2368-RoPE (2 rows/wkr) opt=4-8 (2 regresses it).

Heuristic kept (no-regression, net-positive):
  - no-rope -> whole-row push (= num_tile_cols).
  - RoPE -> rows_per_worker >= 4 ? 2 : block_size.

Correctness PCC ~1.0. Perf (two runs, vs prior committed):

| config | prev | heuristic | Δ |
|---|---:|---:|---:|
| N18944 RoPE | 790 | 739.5 / 740.1 | **-6.4%** |
| N9472 RoPE  | 430 | 432.1 / 431.6 | ~flat |
| N2368 RoPE  | 206 | 205.7 / 205.7 | flat |
| no-rope (all) | — | unchanged | 0 |

N18944-RoPE -> 1.68x composite (was 1.57x). NOTE: tuned on Wormhole; Blackhole
likely needs its own sweep (different DRAM BW / NoC), hence two heuristics.
Sweep harness (WAN_BARRIER_TILES + input_barrier_tiles CT arg) retained.

## Does fused compute differ from composite compute? (N2368 gap investigation)

Question: why do the small N2368 shapes barely beat / slightly lose to composite
(self_sp32_N2368 RoPE 0.96x, cross_q 1.08x) despite fused keeping input resident
+ packed AG? Compared the fused compute kernel (wan_rmsnorm_fused_compute.cpp)
to the composite pre/post kernels (rmsnorm_pre_allgather.cpp /
rmsnorm_post_allgather.cpp).

**1. POST math: IDENTICAL.** Same LLK ops, same order, same broadcast types:
reduce<AVG,REDUCE_ROW>; rsqrt; mul_tiles_bcast_cols (x*1/rms, COL bcast);
mul_tiles_bcast_rows (weight, ROW bcast); matmul_tiles (trans_mat); mul_tiles
(*cos, *sin); add_tiles (rot+unrot). PRE is the same x^2 + l1-acc + reduce<SUM>.

**2. Fidelity: IDENTICAL.** Both HiFi4, fp32_dest_acc_en=true, fp32
intermediate/reduce_result CBs. eps: fused folds +eps (fp32 SFPU immediate) +
rsqrt into the reduce's post_reduce_op (one DST cycle); composite does eps
(add_tiles vs a possibly-bf16 eps tile) + rsqrt in a SEPARATE DST cycle. So the
fused eps/rsqrt is slightly MORE efficient and at least as precise. Compute is
NOT less precise and NOT arithmetically heavier.

**3. POST loop STRUCTURE differs (the real difference):**
  - FUSED = phase-major: per row, each sub-phase (norm-mul, weight, matmul, cos,
    sin, add) sweeps ALL num_tile_cols, round-tripping the whole row through
    intermediate CBs between sub-phases. ~7 flat passes/row.
  - COMPOSITE = block-major: ONE `for col_tile += block_size` loop with weight +
    RoPE NESTED inside (post_allgather.cpp:108,127,166), so each col-block flows
    through all sub-phases staying warm; sub-phase reconfigs are per-block.
  Different working-set/pipeline tradeoff; same total tile-ops.

**4. Fused-only in-kernel structure** the composite POST never sees: the fused
kernel ALSO runs the PRE sum-of-squares + the per-chunk AG-wait
(cb_wait_front(stats_gathered_cb)) + chunk-loop bookkeeping on the SAME TRISC,
serialized in front of POST. The composite splits PRE / AG / POST into 3
separate ops/kernels.

**Conclusion.** The compute math + fidelity are equivalent (fused eps is even a
hair better). The fused does NOT do less-efficient arithmetic. N2368 is
latency/fixed-overhead bound (~11.8 MB; ~23 us at peak BW vs ~200 us actual), so
the input-resident DRAM saving is negligible there. The small-shape gap is
structural fixed cost, not math: (a) the in-kernel PRE + AG-wait serialized
before POST per chunk, and (b) the MUX-AG fixed setup (endpoint-ready wait +
client-connect + sem handshake + DRAM round-trip, ~tens of us, documented in
PERF_LOG.md) — both a large fraction when there are only ~3 rows/worker. The
composite's separate well-tuned all_gather_async + dedicated POST kernel carry
less fixed overhead at this size, which is why it stays ~even on N2368.

## Block-major POST (mirror composite) — DEAD END (regression)

Tried restructuring the fused POST to block-major like the composite
(rmsnorm_post_allgather.cpp): per row compute 1/rms, then for each col-block do
norm-mul -> weight -> matmul -> *cos -> *sin -> add nested (each block flows
through all sub-phases), reading input by index from the RESIDENT input_cb (not
popped). Gated to the common case (whole-row norm + broadcast weight/RoPE);
exotic variants kept phase-major.

Correctness: fused-vs-baseline PCC ~1.0 (output identical, as expected — same
math). PASS. But perf REGRESSED on every shape:

| config | phase-major (current) | block-major | Δ |
|---|---:|---:|---:|
| N18944 RoPE | 740 | 870 | +17.6% |
| N9472 RoPE  | 432 | 521 | +20.6% |
| N2368 RoPE  | 206 | 246 | +19.4% |
| N18944 no-rope | 589 | 623 | +5.7% |
| N2368 no-rope  | 140 | 152 | +8% |

**Why block-major is worse for the fused (reverted):** it is reconfig-bound.
Block-major issues ~6-7 reconfig_data_format per col-block -> ~70/row at 40
cols; the phase-major POST issues 7/row total (one reconfig per full-column
sub-phase pass). The composite tolerates block-major because it is DRAM-bound
(re-reads input from DRAM, which masks the reconfig cost); the FUSED keeps input
resident and has its reads/AG already optimized, so it is compute/reconfig-bound
-- fewer reconfigs (phase-major) wins decisively. This also confirms the N2368
gap vs composite is NOT the POST loop structure (block-major makes the fused
worse, not better); it is the fused's fixed in-kernel PRE+AG-wait + MUX-AG setup
overhead (documented above), which block-major does nothing to address.
Reverted; phase-major POST retained.

## Device-zone profile of the 4 sections (per worker, mean, non-traced)

Added W_SETUP (MUX fabric handshake: endpoint-ready wait + client connect) and
W_FABRIC (per-chunk fabric mcast send + semaphore wait + barriers) zones in the
MUX writer. PRE=RMS_PRE, POST=RMS_POST already existed. Profiled each shape with
TT_METAL_DEVICE_PROFILER=1 (non-traced; spans run larger than the traced wall,
breakdown is the signal). Per-worker us, mean across cores, summed over chunks:

| shape | RoPE | MUX setup | PRE | POST | fabric send+sem | AG-wait | W_AG | out-drain |
|---|:--:|---:|---:|---:|---:|---:|---:|---:|
| N18944 | Y | 0.95 | 162 | 391 | 38 | 21 | 176 | 385 |
| N9472  | Y | 0.95 | 84  | 193 | 14 | 9  | 88  | 198 |
| N2368  | Y | 0.94 | 31  | 118 | 8  | 5  | 32  | 113 |
| N18944 | N | 1.09 | 149 | 203 | 45 | 28 | 192 | 202 |
| N9472  | N | 1.09 | 79  | 84  | 14 | 12 | 89  | 113 |
| N2368  | N | 0.93 | 38  | 47  | 9  | 5  | 46  | 39 |
| L512   | N | 0.96 | 11  | 20  | 7  | 4  | 14  | 15 |

Findings:
1. **MUX setup is ~1 us and constant** for all shapes — a one-time handshake.
   This REFUTES the earlier hypothesis that MUX-AG setup was the small-shape
   fixed-cost tax; it is not.
2. **POST compute dominates** (TRISC), esp. RoPE. PRE ~half of POST.
3. **W_FABRIC (fabric send + sem wait) is small** (7-45 us); most of W_AG is the
   gather read-back + scatter, not the fabric. Output drain (BRISC) overlaps POST.
4. AG-wait mostly hidden (median ~0) with a straggler tail.
5. N2368 is compute(POST)+drain bound with balanced TRISC/BRISC lanes, not
   fabric/setup bound. The composite's edge at that size is its lighter per-row
   POST + separate tuned AG, not MUX setup cost.

---

## Compute-vs-I/O decomposition via two complementary ablations

Added two diagnostic ablations to isolate the critical path (program_factory
`ablation_defines()`, `WAN_ABLATION` env):

- **WAN_ABLATION=7 (`WAN_ABL_SKIP_COMPUTE`)** — compute kernel runs the exact
  external CB handshake (input / stats_local / stats_gathered / output / rope
  cos+sin) but issues **zero LLKs** (no math, reduce, tile_regs, pack, reconfig,
  init). A gated passthrough branch at the chunk-loop head; valid for the common
  AG path (whole-row norm, packed AG, non-streaming, is_tp_1==0 = galaxy bench).
  Gives the **I/O floor**: reader DRAM reads + writer fabric-AG + drain + output
  write + CB sync, with no compute.
- **WAN_ABLATION=8 (pure compute)** — sets all six read/write skips
  (INPUT/WEIGHT/ROPE read, OUTPUT write, FABRIC, GATHER_SCATTER). Every CB
  reserve/push/wait/pop lives *outside* the per-skip `#ifndef` guards, so CBs
  still flow (garbage data) and **compute runs full-speed LLKs** with all
  DRAM/fabric stubbed. Gives **pure compute time**.

Both passed with no deadlock, confirming the CB accounting in the passthrough
and the guard placement. Profiler off, WAN_GALAXY_LINKS=4, traced wall (us/iter):

| shape | RoPE | full | pure compute (A8) | I/O floor (A7) | bound by | overlap gap |
|---|:--:|---:|---:|---:|:--:|---:|
| N18944 | Y | 738.5 | 580.2 | 597.3 | balanced | 141 |
| N9472  | Y | 431.1 | 334.2 | 332.5 | balanced | 97 |
| N2368  | Y | 205.7 | 181.8 | 116.4 | **compute** | 24 |
| N18944 | N | 589.3 | 280.5 | 532.4 | **I/O** | 57 |
| N9472  | N | 320.6 | 176.6 | 303.5 | **I/O** | 17 |
| N2368  | N | 140.5 | 99.9  | 111.2 | balanced | 29 |
| L512   | N | 64.2  | 53.9  | 52.3  | balanced | 10 |

overlap gap = full - max(compute, I/O) = time lost to imperfect compute<->I/O
overlap (serialization). The wall = max(compute, I/O) + overlap gap.

Findings:
1. **No-RoPE big shapes are pure I/O-bound.** Compute (280 / 177) is ~half the
   I/O floor (532 / 304) and fully hideable. No compute headroom; DRAM/fabric
   sets the wall. Already 1.76x / 1.62x.
2. **N2368-RoPE is genuinely compute-bound.** Pure compute alone (181.8) is
   within 24us of the full 205.7 wall; its I/O floor is only 116. Compute does
   not fit behind the small I/O window. This is *the* reason it sits at 0.96x.
3. **RoPE ~doubles compute.** Pure-compute RoPE vs no-RoPE: N18944 580 vs 280
   (+300), N9472 334 vs 177 (+157), N2368 182 vs 100 (+82). trans-mat matmul +
   .cos + .sin + add per row. On big shapes it still (mostly) tucks under the
   I/O floor; on N2368 it blows past it.
4. **RoPE big shapes are balanced with overlap headroom.** N18944-RoPE compute
   580 ~= I/O 597 yet wall is 738 -> 141us lost to serialization; N9472-RoPE
   loses 97. Perfect overlap would approach ~597 / ~334 (~19% / ~23% faster).

Two distinct optimization targets:
- **N2368-RoPE** -> cut RoPE compute (binding constraint, 182 of 206us).
- **N18944 / N9472-RoPE** -> tighten compute<->I/O overlap (97-141us exposed
  serialization despite balanced costs).

---

## RoPE compute optimization: fuse cos/sin/add via FPU dst-accumulate

Targeted the RoPE eltwise tail (P_COS+P_SIN+P_ADD), which the sub-phase
profiling showed was 78% of the RoPE cost (the matmul P_MM is only 22%).
Per-row, shape-invariant: P_MM 5.5us, P_COS 6.5, P_SIN 6.1, P_ADD 6.6.

**Method (what worked):** collapse the three eltwise passes into ONE FPU pass
using dst accumulation:
  binary_tiles_init<true, ELWMUL>(intermediate, cos, /*acc_to_dest=*/false);
  mul_tiles(...)  -> dst[i] = x*cos
  binary_tiles_init<true, ELWMUL>(rotated, sin, /*acc_to_dest=*/true);
  mul_tiles(...)  -> dst[i] = rotate(x)*sin + dst[i]   (FPU adds for free)
  pack dst[i] -> output
The add is free (done by the FPU during the second multiply), runs in fp32
dest, so there is a SINGLE final rounding at pack -> precision-preserving
(same as the old fp32-intermediate add_tiles). 1 dst reg/output tile
(block_size tiles/acquire), 1 pack/tile, FPU-only.

**Dead-end first tried (recorded so we don't repeat):** SFPU add_binary_tile
(mul->dst0, mul->dst1, sfpu add->dst0). Correct but perf-NEUTRAL: P_ROPE 19.7us
vs old 19.2us. The SFPU add is quasi-serial per-datum and costs as much as the
packs it saves. There is no FPU dst+dst add (FPU binary ops only read CBs), so
the dst-ACCUMULATE-on-mul (acc_to_dest) is the right primitive, not SFPU.
NOTE: the 3-arg mul_tiles_init(cb,cb,acc) overload is ambiguous with the
call_line variant -> call binary_tiles_init<true,ELWMUL>(cb,cb,acc) directly.

**Result.** P_ROPE 19.2us/row -> 11.5us/row (-40%); POST/row 41.7 -> 32.6us.
Wall (traced, WAN_GALAXY_LINKS=4), RoPE shapes (no-RoPE unchanged):

| shape | fused before | fused after | speedup before | speedup after |
|---|---:|---:|---:|---:|
| N18944-RoPE | 738 | 692.6 | 1.67x | 1.79x |
| N9472-RoPE  | 431 | 403.5 | 1.39x | 1.51x |
| N2368-RoPE  | 206 | 180.8 | 0.96x | **1.09x** |

N2368-RoPE (the compute-bound shape) now BEATS composite. Correctness:
fused-vs-recorded-baseline PCC 0.999991-0.999996 on all RoPE shapes (no-RoPE
1.000000), i.e. ~bit-identical, no fidelity loss.
