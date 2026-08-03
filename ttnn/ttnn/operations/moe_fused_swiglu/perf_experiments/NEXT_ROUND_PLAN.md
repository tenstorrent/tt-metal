# moe_fused_swiglu — next perf round: measured state, plan, and traps

Working notes for the optimisation round in progress. Everything here is MEASURED on
`blackhole_p150`, grid `11x8` = 88 cores, emb 7168, ND-sharded weights, unless stated. Companion
records: `changelog.md` (Perf 16 + 6 addenda), `dram_download/README.md` (the isolated floors),
`noc_trace/README.md` (tt-npe + phase timeline), `scatter_matmul/` (the compute floor).

---

## 0. STATUS AFTER ROUND 17 — read this before the rest of the document

**Every optimisation in §2 (#1-#5) and §7 (#6) has now been BUILT and MEASURED.** Full record with
methodology, all curves and every null: **`ROUND17_LOG.md`**. Sections 1, 2, 3 and 7 below are the
*pre-round* analysis and several of their predictions were wrong — the log supersedes them.

| knob | outcome |
|---|---|
| **#1 + #3** `MOE_SWIGLU_WD_SPLIT` (one lever, not two) | **SHIPPED at 3.** Interior optimum; lands at f = 0.561 against the isolated bench's predicted 0.55. Wholesale move (8) is +13 % at count 128. |
| #2 `MOE_SWIGLU_HSPLIT` (dual-NoC h) | **Built, correct, DROPPED.** Flat at the predicted split, -5 to -6 % worse by 6/8. The second NoC obligates an acked-barrier signal; the free linked-flag version is the already-measured off-loop-sender hang. |
| **#4** `MOE_SWIGLU_HPOSTED` (posted h mcast) | **SHIPPED at 1.** The plan's open correctness question is resolved: post the DATA, keep the flag non-posted and LINKED. The isolated bench's "+10 % under load" does not transfer. |
| #5 `MOE_SWIGLU_H_DTYPE=bfp4` | **Still broken**, default-off, and two of §5's stated diagnoses are corrected in the log. Cannot ship on precision regardless. |
| **#6** `MOE_SWIGLU_REDUCE_MECH` | **SHIPPED at `dest_acc`.** Not `pack_l1_pair` — DEST accumulation gets the same win with NO extra L1, so §7's +102 KB blocker is moot. |

**Measured end to end, shipped defaults, us at counts 128 / 256 / 512:**

| | 128 | 256 | 512 |
|---|---|---|---|
| bf16_rm | 91.06 -> **84.44** | 128.75 -> **120.23** | 225.82 -> **211.68** |
| bfp8_tile | 88.46 -> **84.34** | 122.64 -> **112.06** | 215.78 -> **200.14** |

Golden 45/45 throughout. **Count 128 MET with margin; count 256 is 4.06 us short (was 14.05);
count 512 is 38.3 us short (was 53.1).** §8 remains the only open track, and the log records a
concrete NEW blocker for it (phase 2's blanket read barriers drain any cross-block prefetch,
including trid-tagged ones, so they must be trid-scoped FIRST).

**§14 candidates: ALL FOUR now measured too** — 14.1 SiLU-in-DEST (null, the trade cancels),
14.2 peer-loop rotation (null on e2e AND on tt-npe), 14.3 drop the gather's trailing atomic barrier
(null), 14.4 last-K-block-packs-to-out (BLOCKED on a runtime-`m_eff` CB-wrap invariant). Each entry
in §14 now carries its result; full records in `ROUND17_LOG.md`. §6's "recommended order" is
superseded. **The remaining open track is §8 (count 512) alone.**

Two corrections that matter more than the numbers:

* **"reconfig before init, on BOTH sides."** Any raw compute block dropped between this kernel's
  helper chains must emit `reconfig_data_format(srcA, srcB)` AND `pack_reconfig_data_format(out)`
  *before* its `*_init` calls — the inits set the math MOP, not the unpacker's format registers.
  Omitting it yields `pcc = 1.000000` with `inf`: right pattern, wrong format. This is the same
  signature #5's bfp4-h still produces, and it is the lens to finish it with.
* **`skip_eltwise` cannot price a pack-count change.** Perf 7 used it to retract this whole family at
  "at most 712 ns"; it elides the eltwise MATH but keeps every PACK and CB round-trip, so it never
  measured the mechanism. `dest_acc` is worth 2.6-4.0 %.

---

## 1. Where the op stands

Current working-tree defaults include `XSTAGE_FIRST=1` + `XSTAGE_STAGGER=1` (Perf 16 addendum 2).
The changes are still uncommitted (§5); the default path is golden 45/45.

| count | target | bf16_rm | bfp8_tile | prior reference best MEASURED (feature_spec) |
|---|---|---|---|---|
| 128 | 91 800 | **91 000 — MET** | **87 660 — MET** | 102 000 |
| 256 | 108 000 | 128 430 | 122 050 | 120 000 |
| 512 | 161 816 | 226 000 | 214 910 | 179 795 |

`PERF_TARGET_NS = PERF_MEASURED_NS x 0.9`, and feature_spec says outright "**NO known implementation
hits a target**". So 128 leads the field; 256 is at parity (+1.7 % on bfp8); 512 is the genuinely weak
cell (+19.5 % vs measured best).

### The provisional accounting (count 256, 88 cores) — REOPENED

> **Review correction:** this ladder currently mixes configurations and cannot be used as a closed
> attribution. The heading used to say bf16, but 122.05 us is the bfp8 endpoint while 104.35 us and
> the 23.82 us matmul ablation are bf16 measurements. More importantly, `dl_realistic` puts W_down on
> the writer/NOC_1, while the real op puts it on the reader/NOC_0. It also contains no compute, whereas
> `ABLATE=skip_compute` removes the matmul LLK calls but deliberately leaves the eltwise chains running.
> The agreement between 103.37 and 104.35 us may therefore contain compensating errors; it does NOT yet
> price the semaphore rendezvous at 1 us.

| term | us | increment |
|---|---|---|
| DRAM download alone, optimal NoC split | 52.8 | |
| collectives alone, op assignment | 72.0 | |
| all DM, one trailing barrier | 94.78 | |
| + the op's dependency ordering | 103.37 | **+8.6** |
| + the semaphore rendezvous (`ABLATE=skip_compute`) | 104.35 | **+1.0** |
| + compute | **122.05** | **+17.7** |
| target | 108.0 | |

Do not currently read this as `op = DM 104.35 + exposed compute 17.7`. The required END-TO-END wall
reductions, which do not depend on an attribution model, are:

| count | bf16_rm -> target | bfp8_tile -> target |
|---|---|---|
| 256 | 128.43 -> 108.00 = **20.43 us** | 122.05 -> 108.00 = **14.05 us** |
| 512 | 226.00 -> 161.816 = **64.184 us** | 214.91 -> 161.816 = **53.094 us** |

A 14 us isolated-DM reduction is not guaranteed to buy 14 us of wall: shortening DM can expose more
compute. The success criterion is the measured op wall, not the sum of isolated savings.

### Exposed-cost inventory (count 256, each ablation stubs one payload, keeps all sync)

| payload | exposed us |
|---|---|
| matmul math | 23.82 |
| three bfp4 weight streams | 21.66 |
| **h all-gather** | **16.48** |
| x DRAM read (`xstage`) | 14.29 |
| column reduce all-to-all | 9.34 |
| output write | 6.18 |
| x row multicast | ~0 (fully hidden) |
| sum of payloads | 91.8 |
| residual (`baseline - sum`; NOT a partition) | ~36 |

`no_h_xfer` across M: **15.16 / 16.29 / 27.77 us** at M = 128 / 256 / 512 (12-17 % of the op). This is
the exact upper bound on ANY h-byte reduction.

The individual ablations above overlap. Their marginal exposed costs do not form an additive
decomposition, so `baseline - sum` must not be called "pure serialisation", and the expected wins of
the candidates below must not be added as if independent. A cumulative peel (in a declared order), or
an exact matched skeleton, is needed for an attribution.

### Accounting prerequisite — do this before optimising

Repair and re-run `dl_realistic` before treating any DM conclusion as closed:

1. Baseline W_down on reader/NOC_0, exactly like the op; add writer and split assignments only as
   explicit experimental variants.
2. Match the op's fixed/ragged h payload and every dependency barrier.
3. Produce separate bf16 and bfp8 ladders from matched sessions.
4. Compare like with like: either add the same eltwise skeleton to `dl_realistic`, or compare the op
   with both `skip_compute+skip_eltwise` and the appropriate transport ablations.
5. Validate the repaired stage against the real op before attributing the remainder to rendezvous.

---

## 2. The five optimisations

### #1 — W_down read off NOC_0 (OPEN; coupled with #2/#3)

**Why.** `WD_AHEAD=1` means only the prologue block is batched; **10 of 11 W_down K-blocks are read
INSIDE phase 2**, on the reader/NOC_0 — the same NoC and RISC that carries the entire h all-gather.
Meanwhile NOC_1 is measured idle: `writer_out_issue` shows ~100 % core occupancy through phase 2 but
`uni wr GB/s = 0` for almost all of it, with the output write bursting only in the last 3 trace buckets.

**Do NOT expect the win from W_down itself** — its own exposed cost is ~2.5 us total
(`reader_wd_issue` 1.07 + `p2_wdbar` 1.26 + `reader_wd_wait` 0.15). The thesis is **de-loading NOC_0's
request path during the h rounds**.

**Direct precedent, from the sibling `moe_matmul` op** (`BRISC_WEIGHT_K`, its PERF 1): it splits every
weight K-block's k-rows 4/8 onto BRISC/NOC_1 and measured, on an isolated weight-stream bench:

    0/8 (all NOC_0)  31747 ns  261 GB/s  1.00x
    2/8              25626     322       1.24x
    4/8  <- AUTO     19904     415       1.60x
    5/8              24496     337       1.30x
    8/8 (role swap)  39027     212       0.81x  REGRESSION

Its diagnosis: the stream is **transaction-rate bound, not byte bound** (shrinking payload 18x costs
+12 %), and "the reader is not instruction-bound; it is blocked inside `noc_async_read` on NOC0
REQUEST-PATH credit. BRISC brings a second, physically separate request path." Its stated precondition
is literally our situation: "`down`'s [writer] blocks on `cb_wait_front(cb_out_tiles, ...)`, and neither
output can exist before the last K-block."

**Counter-evidence that must stay visible:** the op's own NoC trace says phase 2 is a square wave with
1.5 us at 700-900 GB/s followed by ~3 us idle and concludes that the request path is nowhere near
saturated. tt-npe also reports no material link congestion. The sibling precedent therefore makes this
worth measuring, not likely by itself. A local request-credit stall may escape the link-congestion
model, but that is the hypothesis the A/B must prove.

**Implementation.** Behind a knob, e.g. `MOE_SWIGLU_WD_RISC = reader (default) | writer | split`.
W_down currently lives entirely in the reader: `wd_acc` (reader.cpp:232), `cb_w_down` (CT arg 33),
`issue_wd_batch` (~reader.cpp:646), `reader_wd_issue` / `reader_wd_wait` / `p2_wdbar` zones. Moving it
needs the writer to gain the `wd_args` accessor, the `w_down_addr` runtime arg, `jstart`/`ec`,
`SLOTS_E`/`SLOTS_H`, `EC_MAX`, `WD_AHEAD`, `cb_w_down`, plus the `BRD` bank-run binding — and the
per-round barrier must move with it (a `noc_async_read_barrier` is per-RISC). The compute side is
unchanged because `cb_w_down` is a CB.

**Watch out:** adding a CT arg SHIFTS every later index and `TA_BASE` in all three kernels. Pass new
scalars as **defines** (`dm_defines`), which is what the shard widths and `H_TILE_BYTES` do.

**Possible wall win:** 2-5 us, unproven. No precision cost. Wholesale movement keeps one CB producer
and is medium risk; a fine two-RISC split is higher risk because the two producers need separate CBs or
raw non-overlapping writes plus an explicit completion handoff.

### #2 — split the h multicast across both NoCs (OPEN)

**Why.** The h all-gather is the largest single traffic term (48.46 MB delivered per M-block, 38.44 us
isolated) and it rides NOC_0 alone (`HSEND="reader"`). NOC_1 is idle in phase 2.

**Evidence for and against.** Wholesale move to NOC_1 is measured WORSE: the op's own `HSEND=writer`
measured +5-7 % (110 299 / 152 919 / 264 829 vs reader's 104 813 / 146 741 / 246 790), and my isolated
bench has NOC_1 1.37x slower for this multicast (52.49 vs 38.44 us). So the direction is a SPLIT
(~58 % NOC_0 given r0/r1 = 1.37), never a swap. **A split has never been tried.**

**Caution:** my DRAM sweep found the two NoCs reach only 75 % of the additive bound, and multicast is a
different traffic class — do not extrapolate, measure.

This also interacts directly with #1: NOC_1 is idle only before W_down is moved there. Once W_down uses
NOC_1, an h split competes with it. Ultimately sweep the two phase-2 fractions together, rather than
measuring each once and adding the wins. Splitting one h block also needs a cross-RISC "both halves
landed" protocol while retaining exactly one owner of the destination CB's reserve/push state.

**Possible wall win:** 8-13 us only if the paths add partially and the completion handoff is cheap.
Treat this as an upper-range hypothesis, not an expected additive saving. Medium-high risk.

### #3 — finer weight split by K-rows (OPEN)

**Why.** We split weights by WHOLE MATRIX (W_gate -> reader, W_up -> writer, W_down -> reader), which
lands at **f = 0.682** of whole-op weight bytes on NOC_0. The isolated concurrent-stream sweep's
optimum is **f = 0.55** (469.5 GB/s, 91.7 % of peak), and `moe_matmul` splits every K-block's rows
50/50, which is strictly finer.

**CORRECTION to an earlier claim in this session:** I first reported the op at f = 0.318 and "~23 us on
the table". That was WRONG — I had W_down on the writer. `wd_acc` is in the READER
(reader.cpp:232) and the writer has no W_down references at all, so the reader carries
W_gate 96 768 + W_down 110 592 = 207 360 B against the writer's W_up 96 768 B. The trace confirms:
NOC_0 26.09 MB vs NOC_1 15.38 MB. **The isolated concurrent-stream headroom is ~3.4 us, and its
direction is to move a little OFF NOC_0.** Note #1 and #3 interact: moving W_down to the writer already
shifts f from 0.682 to 0.318, i.e.
past the optimum in the other direction. **Tune them together, not separately.**

**Review correction:** the global f is diagnostic, not the op's tuning target. W_gate/W_up run in phase
1, while W_down runs in phase 2 beside the h multicast; combining their bytes loses the temporal
schedule. Optimise each concurrency window. For phase 2, sweep W_down K-rows on NOC_1 and, later, h
tiles on NOC_1. #1 and #3 are two mechanisms/effects of the SAME assignment lever and their estimates
must not be added. A fine split also runs directly into the one-producer-per-CB rule, so it needs an
implementation design before it can be called low risk.

**Isolated headroom indication:** ~3.4 us, not an end-to-end prediction.

### #4 — POSTED h multicast (OPEN, needs a correctness gate first)

**Not dropped.** I briefly concluded it was already done because the shipped `Flag` path takes no acked
write barrier (`mcast_pipe.inl:send_data_` links data to the signal; the `async_write_barrier()` is only
on the `Counter` branch, which ships off). **That conflated two different things.** No barrier means the
SENDER never WAITS; the write is still NON-POSTED, so the 87 destinations still GENERATE write-acks that
travel back over the NoC and bump the sender's counter. `noc.h` blocks posted mcast at the library level:

    static_assert(!has_flag(opts, NocOptions::POSTED),
        "Mcasts with posted transactions are not supported");  // TODO: Make this an arch specific assertion

...marked TODO, i.e. a library guard, not necessarily a hardware limit — and my `dl_collective` bench
DID issue posted multicasts via the low-level primitive without hanging.

**The measurement, and its hole.** `ncrisc_noc_fast_write_any_len<noc_mode>(..., posted=true)` (the flag
IS forwarded to every packet of a multi-packet transfer), closed with `noc_async_writes_flushed()`:

| case | non-posted | posted | delta |
|---|---|---|---|
| h, 1 root (spread 2.05-2.08 / 1.04-1.09) | 2.06 us | 1.07 us | **-47.8 %** |
| h, 11 roots (median) | 50.17 | 40.36 | -19.6 % |
| all phases (median) | 68.75 | 75.61 | **+10.0 %** |

**The bench never verified the data arrived** — nothing consumes it. So the -48 % could be real or an
artifact of a degenerate transfer. A post-kernel checksum alone is NOT a sufficient correctness gate:
it can show that bytes arrived eventually, but posted `flushed` gives the receiver no point at which it
is safe to consume them. **NEXT STEP: add per-round unique data plus an independent, architecturally
valid receiver-completion mechanism, checksum on the receiver only after that completion, and re-run
posted vs non-posted including the completion cost.** A trailing non-posted signal is one candidate only
if same-VC data-before-signal ordering is guaranteed on this architecture. Until then, ~11 us is the
gross ack-removal upper bound, not an unlocked or net win.

**Also note:** the win reverses under load (+10 to +21 % with all collectives running), consistent with
posted removing back-pressure so senders overrun. So it may only pay if the h rounds are the only
traffic in flight.

### #5 — bfp4 h (DONE as a measurement exercise; result is INVALID, and that is the finding)

**Transport/CB plumbing implemented:** `MOE_SWIGLU_H_DTYPE=bfp8|bfp4` switches the three h CBs (`cb_h`,
`cb_h_local`, `cb_h_slice`) to `bfloat4_b`, plus an `H_TILE` byte constant for the h transport sites
ONLY — deliberately not the generic `BFP8_TILE`, which also sizes x, the output and the reduce operands.
Passed as the `H_TILE_BYTES` / `H_BFP4` **defines** (not CT args — see the TA_BASE trap).

**Timing:** 91.12 -> 88.87 / 128.42 -> 123.16 / 226.11 -> 215.16 us (-2.5 / -4.1 / -4.8 %).

**BUT the computation is broken:** `pcc=nan max_abs=inf` on 8 of 9 precision cells. The timing is of a
broken op and must not be used.

Two format boundaries found and fixed, NEITHER sufficient:
1. **SwiGLU pack target.** Phase 1's reconfigs are deliberately HOISTED out of the loop
   (`reconfig_data_format(cb_w_gate, cb_x_tiles)` / `pack_reconfig_data_format(cb_gate_acc)`), so
   `mul<..., blk_out(cb_h_slice)>` INHERITS the bfp8 pack format. Added
   `pack_reconfig_data_format(cb_h_slice)` + restore. Still inf.
2. **`down` matmul in0 unpack.** `InitMode::Short` skips format reconfiguration and in0 (`cb_h`) had
   just changed bfp8 -> bfp4. Added `reconfig_data_format(cb_w_down, cb_h)`. Still inf.

**Diagnosis:** bfp4 h is not a 3-CB change. The compute kernel's hoisted-reconfig design carries an
implicit "the h path is bfp8" invariant spread across phase 1 and the epilogue, and at least one more
format-dependent site remains unlocated. Fixing it needs the helper library's format plumbing
understood, not guessed.

**Why not pursued further:** the informational goal is already met and better. `no_h_xfer` gives an
EXACT upper bound (15.16 / 16.29 / 27.77 us) where bfp4 could only ever be a point estimate below it,
and the precision was never going to be acceptable. **h mcast IS confirmed a bottleneck: 12-17 % of the
op.** The measurement branch is defaulted off (`H_BFP4=0`, `H_TILE_BYTES=1088` verified in the compile
flags), and the default path is golden 45/45. Do not ship an exposed broken bfp4 path: either finish it
and add knob-on correctness coverage, or remove/isolate the branch before merging the op changes.

---

## 3. Measured negative results — do not re-walk without new evidence

**18 knobs** (Perf 16 addendum 4): `M_BLOCK=4` (+12.8 %), `GU_CHUNKS` 2/6 (+3.3/+12.2 %), `SBH_DN=2`,
`HN_BLOCK=2`, `DEPTH_X` 1/3 (3 is L1-dead at 1 806 016 B), `REDUCE=tree` (L1-dead, 1 678 592 B),
`SCATTER_NOC=one`, `XPRIO=0` (+1.8 %), `WD_LATE=1`, `HACK_AHEAD=4`, `REDUCE_SLOTS=2`, `XSTAGE_DIAG=1`,
`WG_TRID=1` (+4.1 %), `W_RESIDENT=0` (L1 +183 KB), `DEPTH_X=1 + W_RESIDENT=0` (fits, and STILL flat —
so CB depth was never the cause), `WSHARD_H` 2/4 (flat -> bank rotation is not the limiter), grid 11x9.

**Structural, closed:**
* **Subdividing the block is a regression.** `M_BLOCK=4` at count 256 IS the phase-boundary
  subdivision experiment (M_t 8/4 = two blocks, weights resident so no extra traffic) and costs
  +12.8 %; `GU_CHUNKS=6` +12.2 %. Doubling the round count costs more than the overlap buys.
* **Rendezvous-shortening knobs measured null or small, but its exact cost is REOPENED.** Perf 15 fitted
  a 3.12 us/round "fixed" term and called it rendezvous. The later 103.37 vs 104.35 comparison tried to
  price it at ~1 us, but the compared programs have the W_down-placement and compute mismatches listed
  in §1. Do not restore the old 34.3 us attribution either; repair the matched skeleton first.
* **No congestion.** tt-npe on a fresh trace: **congestion impact 0.0 %**, estimated 194 003 vs golden
  194 111 cycles (-0.1 %), DRAM BW util 56.1 %, avg/max link util 14.8/32.3 %, avg mcast link util
  0.4 %, avg NIU demand 4.3 %. One hotspot exists (max link demand **302.6 %** vs 24.5 % avg, almost
  certainly the 8:1-per-root gather incast) but costs nothing overall.
* **Request size is not the lever.** 1152-3456 B requests reach 340-469 GB/s isolated; `GU_CHUNKS=1`
  (3456 B, 3x bigger) is +6.8 % because it loses the chunk pipelining.
* **The reduce-scatter is healthy** — 10.21 us isolated, tight spread. Retired as a target.
* **Our phase-2 machinery is already at parity with `moe_matmul`'s `down`** — 36.6 vs 37.8 us, same
  88 cores, same delivered bytes. 11 rounds vs their 8 does not show up in the totals. The gap is the
  14 us phase-1 head, which is a fusion artifact their standalone op does not have.

---

## 4. Traps that cost real time — read before touching kernels

1. **READ `generated/tt-triage/triage.txt` FIRST on any hang.** I burned three device-hang cycles
   guessing. Each time triage named the fault in one line (e.g. "BRISC stuck at
   `noc_async_write_barrier`, dl_collective.cpp:78, logical (1,0)..(7,0); NCRISC in
   `wait_for_brisc_notification`").
2. **GATE CORRECTNESS BEFORE BELIEVING A TIMING.** Both #4 and #5 produced plausible speedups that were
   artifacts of unvalidated computation. A transfer that doesn't properly happen is also fast.
3. **Multicast rectangle corner order is NoC-DEPENDENT.** NOC_1 takes far->near, NOC_0 near->far
   (descriptor.py:1538 does this deliberately). Passing near->far to both hangs BRISC.
4. **`src_l1 == dst_l1` is what makes a multicast exclude-source.** A sender inside its own rectangle
   with `src != dst` is a LOOPBACK multicast needing the loopback API and a +1 fan-out
   (`mcast_pipe.inl`: `loopback = in_rect_ && src_l1 != dst_l1`). Mismatch = hang.
5. **A CB has ONE producer.** Two data-movement RISCs calling `cb_reserve_back` on the same CB hangs.
   Give each RISC its own CB set.
6. **Adding a CT arg shifts every later index AND `TA_BASE`** in all three kernels. Use `dm_defines` /
   `compute_defines` instead.
7. **CB total size must be an exact multiple of its page size**, and the x landing CB must be sized from
   the SLICE the kernel writes (1792 B), not the accessor's page (14 336 B).
8. **`cb_reserve_back` counts are format-specific** — bf16 lands 32 stick-slices per tile-row, bfp8
   lands `kr` whole tiles. Over-reserving blocks forever.
9. **tracy masks pytest's exit code under `--profile`** — a run can report `SAFE_PYTEST_RESULT: PASS`
   while having errored. Distrust a profiled PASS that produced no CSV.
10. **The precompile warmup OOMs** on multi-case sessions at capacity 5120 and the SIGKILL wedges the
    board. Always `--no-precompile` for these.
11. **tracy's host post-processing needs ~50 MB RSS per profiled dispatch** — 964 dispatches = 50 GB =
    OOM-killed AFTER the device work succeeded. Chunk profiled sessions to ~150 dispatches.
12. **tt-npe IS available** (I wrongly said it wasn't): built pybind in an earlier session's scratchpad,
    driver at `tt_metal/third_party/tt_ops_code_gen/skills/perf-ceiling-dm/tt_npe.sh`. **Pass
    `--device blackhole`** — the default `wormhole_b0` asserts on grid bounds.

---

## 5. Harnesses built this round (all uncommitted)

| path | what |
|---|---|
| `tests/.../test_moe_fused_swiglu_seqlen_sweep.py` | count sweep, chunked, manifest-mapped; `MOE_SWEEP_*` env |
| `perf_experiments/sweep_seqlen.sh` / `parse_seqlen_sweep.py` / `plot_seqlen_sweep.py` | driver + parser + 4-panel plot |
| `perf_experiments/knob_search.sh` | knob A/B at the graded cells, ND-sharded, 88 cores. **The workhorse** |
| `perf_experiments/dram_download/dl_bench.py` | isolated DRAM download floor |
| `perf_experiments/dram_download/dl_split.py` | NoC-split sweep (f = fraction on NOC_0) |
| `perf_experiments/dram_download/dl_collective.py` | x-mcast / reduce-scatter / h-gather floors, per-RISC assignment, posted switch |
| `perf_experiments/dram_download/dl_realistic.py` | the whole DM with the op's dependency barriers (stage 0/1) |
| `perf_experiments/seqlen_sweep/` | 320-point sweep data + plots + README |

Op changes uncommitted: `XSTAGE_FIRST`/`XSTAGE_STAGGER` defaults -> 1 (working-tree win), `MOE_SWIGLU_WSHARD_H`
probe, `MOE_SWIGLU_H_DTYPE` measurement knob + the two format reconfigs it guards, and the perf
harnesses switched to ND-sharded weights (`MOE_PERF_WPLACE`, worth up to 11 % on the graded numbers).

---

## 6. Recommended order from here

0. **Lock the measurement contract and repair the accounting.** Fix `dl_realistic` per §1, then record
   matched bf16 and bfp8 baselines with grid, weight placement, `input_m_tiles`, env knobs, warmup and
   profiler mode explicit. Do not compare numbers from different rows of that contract.
1. **#1/#3 as one real-op W_down assignment sweep.** Start with wholesale reader vs writer (one CB
   producer), then add a small K-row menu only after designing the two-producer handoff. Do not target
   whole-op f=0.55; measure the phase-2 critical path. Run count 128/256/512 in BOTH formats and run the
   golden suite with each candidate's env, not merely on the default path.
2. **Integrate #6 (`pack_l1_pair`) on the single-sized-M-block program.** It has the strongest direct
   compute evidence. Measure the whole op: the isolated ~3 us is a ceiling/starting estimate because
   the saved compute may already overlap DM.
3. **Joint phase-2 NoC sweep for #1/#2.** Once a W_down assignment is known, split h by a tile-aligned
   payload fraction and sweep it together with the W_down fraction. Include the cross-RISC completion
   cost and preserve one CB reserve/push owner.
4. **#4 only after a real completion protocol exists.** The test must prove receiver-safe consumption,
   not eventual arrival, and the reported win must be net of that protocol.
5. **Count 512 as a separate structural track.** First bound the overlap available at the existing
   M_BLOCK=8 granularity; only then build the cross-block schedule in §8.

Success is an end-to-end wall measurement: <=108 us at count 256 (14.05 us needed for the recorded
bfp8 baseline, 20.43 us for bf16). Isolated candidate savings are evidence and upper bounds, not an
additive budget.

---

## 7. #6 — the reduce ACCUMULATE mechanism (a candidate I left off the list above)

`scatter_matmul/` (phase 1's floor with all DRAM removed by construction, zero-copy L1-resident
operands) measured every reduce SHAPE x MECHANISM at the op's exact geometry — m=8 (count 256), n=6
(HN_PAD), kr=28 (KR_PAD), k=8 (KGROUPS), 88 cores, roofline 31 858 ns for gate+up. Re-run this session,
reproducing the recorded table to <= 0.13 %:

| shape | mech | ns | math util | PCC |
|---|---|---|---|---|
| `mm_only` (no reduce) | — | 35 102 | **90.8 %** | — |
| `scatter_dual` | **`pack_l1_pair`** | **44 116** | **72.2 %** | 0.999889 |
| `ring` | `addchain` | 44 979 | 70.8 % | 0.999759 |
| `scatter_dual` | `dest_acc` | 46 288 | 68.8 % | 0.999888 |
| **`scatter_dual`** | **`addchain`** <- SHIPPED | **47 080** | **67.7 %** | 0.999759 |
| `scatter` (single NoC) | `addchain` | 52 184 | 61.0 % | 0.999759 |
| `direct` (star) | `addchain` | 71 231 | 44.7 % | 0.999759 |

**The op ships `scatter_dual` + `addchain` at 67.7 % and the best measured is 72.2 %** — so
`pack_l1_pair` is worth **-2 973 ns (-6.3 %)** on phase 1's reduce, at PCC 0.999889 (BETTER than
addchain's 0.999759). `ring` is worth -2 097 ns AND saves 13 KB of L1.

**Mechanism:** `pack_l1_pair` folds TWO K-contributors per DEST window with one `BinaryFpu` add plus one
L1-accumulating pack, so the slice fold becomes `ceil(K/2)` `eltwise_chain` calls instead of `K` —
halving the PACK count and the per-call init/reconfig.

**The catch, and why it is affordable now:** it MUST use a bf16 accumulator (packer L1-accumulate on a
block-float tile is a correctness bug — a prior bench measured PCC 0.412), which the changelog priced at
**+102 KB** in the real op against 10 560 B free at 11x8. The descriptor already path-gates the actual
x depth to 1 when the HOST-SIZED `input_m_tiles` extent can reach only one M-block, freeing 195 584 B;
an explicit `DEPTH_X=1` knob is not needed on that program. I also measured the forced depth-1 path as
flat at actual counts 128/256 (91.13 / 128.51). Therefore `pack_l1_pair` fits only when the sized extent,
not merely the device-resident actual count, is one block. A capacity-sized program whose actual count
happens to be 128/256 still needs the second x slot for other legal counts and cannot claim this L1.

**Two orthogonal wins, measured additive** in that bench: dual-NoC transport alone -5 000 ns,
`pack_l1_pair` alone -3 074 ns, both -7 928 ns against a -8 074 ns sum of singles (2 % interaction). We
already ship the dual-NoC half.

**Isolated indication:** ~3 us at counts 128/256. Integrating the bf16 CB/mechanism is medium risk and
must be golden-gated; it improved PCC in the isolated bench, but the real-op precision result is not yet
measured.

---

## 8. Count 512 needs a different plan from 256

512 is the weak cell: **226 000 / 214 910 ns against 179 795 measured-best (+25.7 / +19.5 %)** and a
161 816 target. It is also the ONLY cell where the biggest structural idea applies.

* **`m_blocks = 2` at count 512**, so data movement for block b+1 may be interleaved with stalls in
  block b's phase 2 — named-next-step 1 in the changelog ("software-pipeline the M-block"), still open.
  Do not describe all of phase 1 and phase 2 as concurrently executable: the same TRISC cannot run
  block b's down compute and block b+1's gate/up compute at once, and each data-movement RISC also has a
  single instruction stream. Specify which reads/multicasts are hoisted or interleaved and which waits
  they cover. This must be done at the current M_BLOCK=8 granularity; `M_BLOCK=4` subdivision is +12.8 %.
* **W_RESIDENT already pays here** (-6.25 % at 512, -12.47 % at count=capacity), so the weights are read
  once and the second block's cost is genuinely the collectives + compute.
* The changelog's round-cost model puts **68.6 us across two M-blocks** of per-round period at 512,
  "27 % of that cell's wall and exactly why 512 is the worst cell" — though note §3 above corrects the
  attribution of that term (transport + barriers, not rendezvous).
* The recorded target gaps are format-specific: 64.184 us bf16 and 53.094 us bfp8. Do not sum the
  isolated §2/#6 estimates. Before implementation, build a two-block phase timeline without overflowing
  the NoC event buffer (ordinary device zones, reduced event scope, or separate block markers) and
  calculate a resource-feasible maximum overlap. Cross-block pipelining is a plausible necessary
  direction, not yet a quantified promise to close the target.

---

## 9. Caller-facing finding: latency is a STEP function of work

From the 320-point sweep (`seqlen_sweep/`, every tile-aligned count 32..5120, both formats, median of 3,
rep spread <= 2.9 %). **Configuration exception to the document header:** this full sweep is 110 cores
with interleaved weights; `sweep88` is the separate 88-core placement A/B. The step locations transfer,
but do not compare the absolute 110-core latencies below directly with the 88-core ND-sharded headline.

**The step locations are explained by `work_rows(count) = 8*floor(M_t/8) + next_pow2(M_t mod 8)`**,
where `M_t = count/32` and `next_pow2(0)` is defined as 0 — i.e. `M_BLOCK = 8` plus the descriptor's
power-of-two tail rounding (`m_tiles_eff`). This is a step/monotonicity result, not proof of one exact
linear coefficient: of the 159 consecutive 32-token steps, **80 are FREE** (|delta| <= 3.3 us, and every
one has zero work-row change) and the rest cost ~13 / ~19 / ~40 us for a 1 / 2 / 4 work-row increase —
**159/159 steps consistent for bfp8_tile, 158/159 for bf16_rm**.

Consequences for a caller that controls padding:
* Rounding count UP to a work-row plateau is free: **4768 -> 4864 tokens costs +0 us**.
* A single 32-token step across a boundary costs up to **+42 us** (4224 -> 4256).
* Tail regimes: `m_eff` in {1,2,4,8}, so counts with `M_t mod 8` in {5,6,7} all pay for 8.

Also measured: **`capacity` is free** (count 256 at capacity 1024/2048/5120 = 143.4 / 143.4 / 141.7 us,
a <=1.2 % span with no ordering), and the tight `input_m_tiles` bound is worth a real but small -0.6 %
at 256 / -0.3 % at 128.

Scaling shape: fixed floor ~64 us (weights only), marginal 363 ns/token (bf16) / 340 (bfp8) at 110
cores; DRAM read utilisation peaks at 61 % at count 32 and falls to 7-10 % by 5120; matmul throughput
saturates ~236 (bf16) / 251 (bfp8) TFLOP/s by ~2k tokens.

---

## 10. Per-phase rates and transfer sizes (count 256, 88 cores)

| phase | xfer B | dur us | MB moved | GB/s agg | GB/s per rx | GB/s per tx |
|---|---|---|---|---|---|---|
| W download, op NoC split | 3456 / 1728 | 72.87 | 24.77 | 340 | — | — |
| W download, optimal f=0.55 | 3456 / 1728 | **52.76** | 24.77 | **469.5** | — | — |
| x download (bf16 RM) | 1792 | 1.80 | 0.46 | 255 | — | — |
| x row-multicast (1->10) | 30 464 | 34.22 | 19.50 | 570 | **7.1** | — |
| reduce-scatter | 6528 | **10.21** | 4.60 | 450 | **5.1** | **5.8** |
| h all-gather, 11 rounds (1->87) | 52 224 | **38.44** | 49.98 | 1300 | **13.6** | — |
| h all-gather, ONE round | 52 224 | **2.06** | 4.54 | 2206 | **25.4** | — |

Transfer sizes: W_gate/W_up per K-row **3456 B** (ragged column 2304); W_down per hidden row **1728 B**
(some cores 1152); x bf16 RM sub-page slice **1792 B**; x bfp8 tile 1088 B; x mcast payload **30 464 B**;
reduce slice **6528 B**; h mcast payload **52 224 B**; output write 1088 B/tile. All far below the
~8160 B NoC saturation point, and the trace confirms 1152-1287 B mean per DRAM transaction.

**The two h rows are the key comparison:** one uncontended round delivers **25.4 GB/s per receiver**, 11
rounds in flight give **13.6** — the per-receiver rate HALVES. Since npe reports 0.0 % congestion, that
is serialisation and injection-port limits, not NoC contention.

### Phase timeline (from the tt-npe trace, 88 cores, count 256)

```
t=  0- 59us  x + W_gate + W_up, 330 GB/s, BOTH RISCs        DRAM busy
t= 59-104us  gate/up tail + reduce + scatter + h publish     DRAM = 0   <- peak L1 writes, 639 GB/s
t=104-144us  11 phase-2 rounds                              DRAM 25 % duty
t=144-159us  down tail + output write                        DRAM = 0
```

**50 % of the op has DRAM under 10 GB/s**, longest contiguous idle window 45 us, and "43 of the 46 extra
us" going from count 128 to 256 are DRAM-idle. Phase 2's DRAM is a square wave: ~1.5 us at 700-900 GB/s
then ~3 us of silence, x11.

### The gather (all columns AT ONCE, not serial with the rounds)

`writer_scatter` 36 350 ns mean (max 59 979, min 27 156) | `reader_reduce` 31 489 (41 403 / 18 851) |
`writer_hslice` 19 434 (27 810 / 12 496) — all with a **2.2x core-to-core spread**, the signature of the
8:1-per-root incast. All 11 columns gather concurrently in phase 1 and finish clustered at ~90 us; round
0 fires at 92.0. This is the peak-write window and the likely home of npe's 302.6 % max link demand.

---

## 11. Remaining transferable ideas from `moe_matmul`

Location: `/localdev/mstaletovic/2026_07_29/2358_mstaletovic_moe_matmul_codegen/clones/moe_matmul_run1/tt-metal/ttnn/ttnn/operations/moe_matmul/`
(also at `.../1708_.../`; the `1706` run dir has no op checked out). Its `down` bfp8 M=256 final is
**37 753 ns at 52.8 % util on 88 cores with INTERLEAVED weights**.

1. **A tunable multicast GRANULARITY knob, which we do not have.** `MCAST_MT_BLOCK` = M-tile-rows per
   multicast round, with `MCAST_MT_BLOCK_CAP` keyed by front-end (bf16_rm 2, bfp8_tile 8) "because the
   two front-ends have opposite curves". Their measured curve on gate/up bfp8 M=256:
   **1 -> 60.3 us | 2 -> 53.4 | 4 -> 52.4 | 8 -> 52.6** — i.e. coarsening is worth **13 %**. Their `down`
   already runs at maximum granularity ("one round per K-block carrying the whole in0 block"). Our h
   payload is FIXED at `m_eff * HN_PAD` with no knob. Worth investigating whether our x row-multicast
   (which IS per-tile-row, 8 rounds per grid row) has the same headroom — note our x mcast measures
   fully hidden (`no_x_xfer` ~0), so probably not, but the x DRAM read exposes 14.29 us.
2. **`DOWN_N_ROWS` — deliberately stranding cores.** "8 rows is 88 cores on this 11x10 grid... MEASURED:
   8 rows beats 10 on all three `down` cells." We independently found 110 cores SLOWER than 88 at counts
   128/256 (the extra rendezvous participants outweigh the added throughput) — same conclusion, and it
   is why `MOE_SWIGLU_GRID=11x8` is the right measurement grid.
3. **`SHARD_CORES = 8` decouples the mcast group count from the grid.** Their 8 shard owners in column 0
   each hold 8 whole HIDDEN tiles (`Kt / SHARD_CORES`, a divisor of 64), giving 8 clean K-block rounds
   and no padding. Ours is `HGROUPS = 11` (grid width, prime), forcing `HN_PAD = 6` with 66 >= 64 and 11
   rounds. **Changing ours is a scheme change, not a knob:** our `slice_assigned` splits the
   `m_eff x HN_PAD` TILE block across the 8 rows, so an owner holds an arbitrary 6-tile slice spanning
   both M and hidden — not "8 hidden tiles". Making it hidden-major means changing the shared
   `slice_assigned` contract all three kernels use to keep the column all-to-all deadlock-free. And
   since the totals are already level (36.6 vs 37.8 us), the round count is NOT where the prize is.
4. **Their weight stream is transaction-rate bound**, which is the same finding as our 1152-3456 B
   requests reaching only 340-469 GB/s. Their fix was the second request path (#1/#3), not bigger
   requests — matching our measured rejection of the request-size family.

---

## 12. Reproduction

```bash
# the workhorse: run each candidate in BOTH formats (88 cores, ND-sharded)
KNOB_COUNTS=128,256,512 KNOB_REPS=5 KNOB_FMT=bf16_rm \
  perf_experiments/knob_search.sh "label=ENV=val ENV2=val2" "label2=..."
KNOB_COUNTS=128,256,512 KNOB_REPS=5 KNOB_FMT=bfp8_tile \
  perf_experiments/knob_search.sh "label=ENV=val ENV2=val2" "label2=..."

# graded cells through the real perf harness, both weight placements
MOE_PERF_WPLACE=nd_shard MOE_SWIGLU_GRID=11x8 \
  MOE_R2_CASES="7168,5120,128,bf16_rm;7168,5120,256,bf16_rm;7168,5120,512,bf16_rm" \
  scripts/run_safe_pytest.sh --profile --no-precompile \
  tests/ttnn/unit_tests/operations/moe_fused_swiglu/test_moe_fused_swiglu_r2_perf.py

# isolated floors (separate exact invocations)
scripts/run_safe_pytest.sh --run-all --no-precompile tests/ttnn/unit_tests/operations/moe_fused_swiglu/test_perfexp_dram_download.py
scripts/run_safe_pytest.sh --run-all --no-precompile tests/ttnn/unit_tests/operations/moe_fused_swiglu/test_perfexp_dram_split.py
scripts/run_safe_pytest.sh --run-all --no-precompile tests/ttnn/unit_tests/operations/moe_fused_swiglu/test_perfexp_dram_collective.py
scripts/run_safe_pytest.sh --run-all --no-precompile tests/ttnn/unit_tests/operations/moe_fused_swiglu/test_perfexp_dram_realistic.py
scripts/run_safe_pytest.sh --run-all --no-precompile tests/ttnn/unit_tests/operations/moe_fused_swiglu/test_perfexp_dram_posted.py

# compute floor (reduce shape x mechanism)
scripts/run_safe_pytest.sh --run-all --no-precompile \
  ttnn/ttnn/operations/moe_fused_swiglu/perf_experiments/scatter_matmul/test_scatter_matmul.py \
  -k "focus_menu"        # or fidelity_probe

# NoC trace + tt-npe (ONE cell per run; count 512 overflows the event buffer -> NO device log at all)
TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1 TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=10000 \
  MOE_SWIGLU_GRID=11x8 MOE_R2_CASES="7168,5120,256,bf16_rm" \
  scripts/run_safe_pytest.sh --profile --no-precompile <r2 perf test>
python3 perf_experiments/noc_trace/noc_phases.py <.logs>/noc_trace_dev0_ID<n>.json
TT_NPE_HOME=<built tt-npe> bash tt_metal/third_party/tt_ops_code_gen/skills/perf-ceiling-dm/tt_npe.sh \
  <trace>.json --noc-trace --device blackhole --cong fast    # and --cong none for the delta

# per-stage device zones from any --profile run
python3 perf_experiments/parse_zones.py <report_dir>

# golden gate — run with the candidate env enabled, then again on defaults
scripts/run_safe_pytest.sh --run-all --no-precompile eval/golden_tests/moe_fused_swiglu/test_golden.py
```

Env knobs added this round: `MOE_SWIGLU_WSHARD_H` (shard height probe, default 1, measured flat),
`MOE_SWIGLU_H_DTYPE` (bfp8|bfp4, measurement only), `MOE_PERF_WPLACE` (nd_shard|interleaved, in both
perf harnesses), `MOE_SWEEP_*` (the seqlen sweep), `KNOB_*` (knob_search driver).

---

## 13. One-line summary for whoever picks this up

The old "DM = 104.35 us, therefore remove 14 us of DM" conclusion is REOPENED because its ladder mixes
formats, W_down placement and compute content; first repair the matched accounting, then optimise the
real phase-2 assignment of W_down and h jointly, integrate the evidence-backed single-block
`pack_l1_pair` candidate, and judge every result by end-to-end walls (count-256 gaps: 20.43 us bf16 /
14.05 us bfp8), never by adding isolated savings.

---

## 14. Candidates for round 18 (written up at the end of round 17)

Ranked by EVIDENCE, not novelty. Three of the four are direct extensions of a mechanism that won
this round, which is why they are ranked above anything new. Each entry says what would have to be
true for it to pay, so a null is still a result.

Not on this list, and closed on direct measurement — do not re-walk: the matmul (at its real
bfp8 x bfp4 LoFi limit, ~10.5 cycles/tile-MAC), the h round count, request sizing, and grid shape.

### 14.1 Fold the gate SiLU into the DEST accumulation — HIGH confidence, medium risk

**Why.** `REDUCE_MECH=dest_acc` won 2.6-4.0 % by removing accumulator L1 traffic, and the gate path
still pays a FULL L1 round-trip that the same mechanism can delete. Today:

    dest_acc folds KGROUPS-1 contributors in DEST -> pack cb_slice_gate
      -> UNPACK cb_slice_gate -> add_bias_bcast_rows(last contributor, +SiLU) -> pack cb_gate_silu

`cb_slice_gate` exists only to carry the running sum between those two passes. Fold **all KGROUPS**
contributors in DEST, apply SiLU **in DEST**, and pack ONCE straight to `cb_gate_silu`: the CB, its
pack and its unpack all disappear, and so does the separate add.

**THERE IS NO SiLU TRADE — an earlier draft of this entry claimed one and was wrong.** SiLU is SFPU
work in both shapes. "SiLU on the packer thread" in the kernel comment describes where
`add_bias_bcast_rows` ISSUES it, not a free packer-hardware activation; the changelog is explicit
that the epilogue is "dominated by the 48-tile SFPU SiLU". Fusing therefore adds no SFPU work at all
and the change is a pure removal of L1 traffic — the same shape as the win that already landed.

**DO NOT also fuse the SwiGLU multiply.** The obvious next step — keep the SiLU result in DEST and
multiply by `cb_slice_up` there — requires a DEST x L1 product, i.e. `binary_dest_reuse_tiles`
(`DEST_TO_SRCA`). That path is slow, and the kernel already carries a measurement saying so: the FPU
multiply through L1 was measured FASTER than DEST reuse (`examples/compute_fusion`). Pack after the
SiLU and leave the multiply reading two L1 operands, exactly as it does now.

**Prerequisite:** obey the round-17 rule — `reconfig_data_format(srcA, srcB)` AND
`pack_reconfig_data_format(out)` BEFORE any `*_init`, or the result is `pcc = 1.000000` with `inf`.

### 14.2 Rotate the reduce-scatter's peer-loop start — LOW cost, plausible, never tried

**Why.** Every core walks its KGROUPS column peers from index 0 at the same instant, so all 8
contributors hit peer 0 first, then peer 1, and so on. That IS the 8:1 incast: tt-npe measured **max
link demand 302.6 %** against a 24.5 % average, and the three gather stages carry a **2.2x
core-to-core spread** (`writer_scatter` 36 350 ns mean, 59 979 max, 27 156 min; `reader_reduce`
31 489 / 41 403 / 18 851; `writer_hslice` 19 434 / 27 810 / 12 496).

**The change is two lines** — core in row `r` starts its peer loop at index `r` and wraps — in the
writer's gather loops and the reader's `SCATTER_NOC_SPLIT` up-half loop. No protocol change: the
destinations and counts are identical, only the ORDER differs, and every wait is a monotone counter
that does not care about arrival order.

**What would have to be true:** that the stages are limited by the instantaneous incast fan-in rather
than by total bytes. npe reports 0.0 % overall congestion impact, so the hotspot may cost nothing —
that is exactly the open question. Cheap enough that the answer is worth having either way.

### 14.3 Remove one of the reduce-scatter's two grid traversals — MEDIUM confidence

**Why.** Per M-block the writer does: 8 unicast writes -> ACKED `noc_async_write_barrier()` ->
8 `noc_semaphore_inc` -> `noc_async_atomic_barrier()`. That is TWO full ack round-trips on a stage
measuring 36 us mean. `HPOSTED` just showed that removing return traffic from the h multicast is
worth 1-2 % end to end, and this is the same class of question on a bigger stage.

Two separable sub-questions:
1. **The trailing `noc_async_atomic_barrier()` guards nothing local** — no subsequent statement reads
   or reuses the incremented cells, and the kernel-exit barrier covers completion. Dropping it is a
   one-line A/B.
2. **The write barrier is the data-before-signal proof** and cannot simply be dropped. The only legal
   removal is the `HPOSTED` argument: if data and signal travel the same VC in issue order, the
   signal cannot overtake. But the signal here is an ATOMIC on `write_at_cmd_buf` — a different
   command buffer — which is precisely the configuration `mcast_pipe`'s Counter path documents as
   unable to terminate a link. So (2) is likely blocked; (1) is free to test.

### 14.4 The `cb_out_interm` -> `cb_out_tiles` copy — LOWEST confidence, read before assuming

**Why.** `compute_out_pack` is a full extra L1 pass over `m_eff * EC_MAX` tiles (24 at count 256)
that exists only because packer L1-accumulation needs a non-block-float target, so `down` accumulates
into a bf16 interm and a separate copy converts to bfp8.

**The lead:** `matmul_block_helpers.inl:407` has a last-block swap-to-out
`pack_reconfig_data_format(pack_target_id)`, and its own comment says that swap "is gated on
l1_acc / fp32 DEST and may not fire". So the helper already contemplates the last K-block packing
somewhere other than the interm. Whether the FINAL K-block can accumulate in DEST instead of L1 — and
therefore pack once, straight to bfp8 — needs the helper read, not assumed. If it can, the copy
disappears; if it cannot, this is structural and should be recorded as closed.

**Trap to respect:** the accumulator must not become block-float. A prior bench measured PCC 0.412
for packer L1-accumulate onto a bfp8 tile, and the changelog's L1 table prices the bf16 interm at the
cost it does for exactly this reason.
