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
