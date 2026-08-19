# rms_norm — changelog

## Phase 0 — first implementation

Device: **blackhole p150b**, 13×10 compute grid (130 worker cores). All numbers are
`DEVICE KERNEL DURATION [ns]` from the Tracy per-op CSV
(`scripts/run_safe_pytest.sh --profile`), median of 10 fresh-cache dispatches after 2 warm-ups.
Bench harness: `_bench_rms_norm.py` + `tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_bench.py`.

### What shipped

| Piece | Content |
|---|---|
| `rms_norm.py` | `INPUT_TAGGERS` (alignment, rank) / `SUPPORTED` / `EXCLUSIONS` / `validate()` + entry point. `default_compute_kernel_config()` is the single exported precision factory. |
| `rms_norm_program_descriptor.py` | `blocking_plan()` — the ONLY place block factors, buffer depths, the regime and the core split are decided. Everything downstream reads a field. |
| `kernels/rms_norm_{reader,compute,writer}.cpp` | reader on NoC0 / writer on NoC1; compute is 100% `kernel_lib` helpers (no raw-LLK compute anywhere). |

**Blocking knobs (all live, none inlined):** `BLOCK_HT`, `WT_REDUCE_BLOCK`, `WT_SCALE_BLOCK`,
`DEST_BLOCK`, `IN_BUF_DEPTH`, `OUT_BUF_DEPTH`, `RM_BUF_DEPTH`, `GAMMA_INGEST_BLOCK`,
`ACTIVE_CORE_CAP`. Two regimes selected by the design's pinned predicate: **A** resident
single-read, **B** streaming masked 2-pass.

**Work distribution:** `Rt` (the independent axis) across the whole grid via
`ttnn.split_work_to_cores(grid, ceil(Rt/BLOCK_HT), row_wise=True)`. `(1,1,8192,1024)` runs on
**128 of 130 cores**, 1 row-block (BLOCK_HT=2 tile-rows) each.

### Correctness

| Suite | Result |
|---|---|
| `tests/.../rms_norm/test_rms_norm.py` (acceptance) | **73 / 73 pass**, default *and* `--dev` |
| `eval/golden_tests/rms_norm/test_golden.py` cartesian | **700 pass / 0 fail**, rest xfail (out of Phase-0 `SUPPORTED`) or skip (`INVALID`) |
| `test_golden.py` loose (resilience + pad-poison + perf shapes) | **3 pass / 0 fail / 382 xfail** |
| `test_regression.py` | **15 / 15 pass** |

### Bound classification — measured ablation

Payload stubbed, CB/barrier sync scaffolding intact (`stub_dm`, `stub_compute`, and the
all-payloads-stubbed floor `stub_both`).

| shape | full | `stub_dm` | `stub_compute` | `stub_both` (floor) | **verdict** |
|---|---|---|---|---|---|
| `(1,1,8192,1024)` TILE, 128 cores | 93 824 ns | 24 570 ns (26%) | 90 035 ns (96%) | 11 624 ns (12%) | **data-movement-bound** |
| `(1,1,32,7168)` TILE, 1 core, Regime B | 76 136 ns | 61 881 ns (81%) | 31 484 ns (41%) | 22 941 ns (30%) | **compute-bound** |

> `stub_compute` elides the `eltwise_chain` math only (`CKL_ELTWISE_CHAIN_SKIP_COMPUTE`); the
> `reduce` / `tilize` / `untilize` helpers still execute, so the compute share on the
> grid-starved shape is a **lower bound** of ≥58.6%.

**This refutes one design claim.** `op_design.md` classifies the op "movement-dominated in every
regime". That holds for prefill, but the grid-starved decode regime is compute-bound on one core —
and that reframes Lamp L1: its win there is not only fewer bytes per core, it is parallelising the
*compute*. It also makes F27 (`math_fidelity`, **1.45× measured** on that shape) a real Phase-1
lever a movement-only reading would have skipped.

### Data-movement ceiling and achieved

`noc_estimate` is a test target and is **not built in this tree** (the `--build-tests` targets fail
on unrelated nuked-op sources), so the NPE bracket could not replace the design's constants. The
ceiling below uses the design's own evidence-based anchor, `DRAM_ACHIEVABLE ≈ 350 GB/s`
(`op_design.md` → "Candidate algorithms").

| shape | bytes moved | DM target @350 GB/s | measured | `achieved = target / measured` |
|---|---|---|---|---|
| `(1,1,8192,1024)` Regime A | 33.55 MB useful (+8.39 MB gamma replication) | 95.9 µs (119.8 µs with replication) | **93.4 µs** | **1.03** — at/above the anchor (359 GB/s useful, 449 GB/s aggregate) |
| `(1,1,8192,7168)` Regime B | 234.9 MB useful → 352.3 MB with the 2nd read | 1006 µs | **1019 µs** | **0.99** — at the DM bound *for the 2-pass algorithm* |
| `(1,1,32,7168)` Regime B | 0.92 MB useful | — | **76.1 µs** | not applicable: **compute-bound**. Binding stage = the compute chain on one core (58.6%+ of the wall). |

### Reconciliation against the design's Mode-A ranking

| shape | design row-1 prediction | measured | reading |
|---|---|---|---|
| `(1,1,8192,1024)` | 116 µs full grid / 102 µs @cap 32 / 96 µs with Lamp L2 | **93.4 µs** | **beats the best prediction.** The A0 core sweep also *refutes* the "cap cores to the bandwidth knee" clause: full grid 93.7 µs, 96 cores 96.0, 64 cores 94.5, 32 cores 110.1, 16 cores 106.0 — **no knee below the full grid**. |
| `(1,1,8192,7168)` | 1006 µs for the streaming 2-pass | **1019 µs** | 0.99× — the byte model is accurate. Candidate #4 (Lamp L4, eliminate `cb_normed` → Regime A) predicts 672 µs = **1.5×**; top P2 item. |
| `(1,1,32,7168)` | ~46 µs (Regime B, 1 core) | **76.1 µs** | **1.65× worse than predicted, and 5.1× off the 14 894 ns gate.** The divergence is a *mispredicted bound*, not a tuning miss — the design assumed DM, the ablation says compute. Confirms the ranking's own verdict that row 1 cannot reach the gate: **Lamp L1 (cross-core W split) is required and stays scheduled as P1.** |

### Cumulative bench set (carried forward — every later phase re-measures ALL of these)

| name | shape | layout | regime | cores | **Phase 0 ns** | useful GB/s | reference |
|---|---|---|---|---|---|---|---|
| `grid_filling` | (1,1,8192,1024) | TILE | A | 128 | **93 415** | 359.2 | 96 744 → **1.04× better** |
| `wide_prefill` | (1,1,8192,7168) | TILE | B | 130 | **1 019 487** | 230.4 | 1 032 281 → **1.01× better** |
| `grid_starved` | (1,1,32,7168) | TILE | B | 1 | **76 149** | 12.0 | 104 259 → 1.37× better, **but 5.1× off the 7× gate** |
| `smallest` | (32,17) | TILE | B | 1 | **3 267** | — | — (master.md B0 counterfactual regime) |
| `row_major` | (1,1,8192,1024) | ROW_MAJOR | A | 128 | **95 053** | 353.0 | — |

References are the `feature_spec.LOOSE_CASES` `achievable_ns` values (blackhole p150b, 1350 MHz).
Note ours are measured at the **maxed-out** precision corner (HiFi4 + `fp32_dest_acc_en=True`) while
the references are HiFi2 + fp32-DEST-off — i.e. we beat them while paying more precision.

### Levers (full ledger: `lever_ledger.json`; table: `python3 -m eval.verify_levers <ledger> --report`)

`verify_levers` is **clean**: 0 blocking, 0 signal, 0 stale. 12 of 29 closed with evidence,
17 open on record.

Measured wins, each with a re-runnable `levers=dict(...)` off-arm in `_bench_rms_norm.py`:

| lever | knob | on → off | win |
|---|---|---|---|
| **B9** reader NoC0 / writer NoC1 | `noc_split` | 93 535 → 137 506 ns | **1.47×** (1.47× on the RM path too) |
| **B7** one barrier per block | `barrier_per_block` | 93 535 → 124 082 ns | **1.33×**; **3.56×** on `(1,1,32,7168)` |
| **A0** full grid, no core cap | `active_cores` | 93 535 → 109 520 ns (cap 32) | **1.17×** |
| **B5** whole-page transactions | `coalesce` | 93 535 → 102 044 ns | **1.09×**; 1.10× on `grid_starved` |
| **coarse_chunk** (block-size fidelity) | `coarse_chunk` | 76 127 → 279 975 ns | **3.68×** on `(1,1,32,7168)` |
| **compute_block_size** | `block_ht`, `dest_block` | 93 535 → 96 819 ns | 1.035×; 1.06× on `grid_starved` |

Kept-but-flat (correct levers, held at their working defaults, not reverted):

* **C16 double-buffering** — genuinely live (`IN_BUF_DEPTH=4`, OUT/RM=2 on `grid_filling`; forcing
  1/1/1 is a different plan, verified) but **+0.1%**. The shape is already DRAM-saturated at
  359 GB/s of useful traffic, so a deeper CB cannot create bandwidth. It is the lever that pays once
  Lamp L1/L3 move the wall — re-check then.
* **A1 `row_wise=True`** — **−0.6%**: with 128 of 130 cores in use, row-wise and column-wise select
  nearly the same set. Free, and correct the moment the op runs on a partial grid.

**B0 (smallest-regime) is satisfied:** every per-core-overhead lever was re-measured on `(32,17)`
(one tile of real work). Worst case ±1.7%, i.e. noise — no lever needs a work-per-core gate.

### Ranked opportunities carried into Phase 1+

1. **Lamp L1 — cross-core W split (P1, not optional).** The only path to the `(1,1,32,7168)`
   `minimum_expected_speedup = 7.0` gate. The ablation says the win is *compute* parallelism as much
   as bytes. Unlocks `WIDTH_SHARDED` / `BLOCK_SHARDED` too. (ledger C15/C14/A2/A3)
2. **Lamp L4 — eliminate `cb_normed`** so wide prefill fits Regime A: 1019 µs → predicted 672 µs
   (**1.5×**) on `(1,1,8192,7168)`. (ledger D20)
3. **F27 `math_fidelity`** — measured **1.45×** (HiFi2) / 1.51× (LoFi) on the compute-bound decode
   shape. *Not* changed in Phase 0: the default is pinned by `references/precision_convention.md`
   and F23 forbids downgrading a caller-supplied value.
4. **Lamp L3 — `HEIGHT_SHARDED`** zero-copy CB on the shard: cuts the read entirely for a regime
   whose DM payload is 74% of the wall.
5. **B12 / Lamp L2 — gamma mcast.** Measured upper bound **5.3%** on `grid_filling` (93 981 with
   gamma vs 89 030 with none, and that gap also contains a whole `mul`). The design's byte model
   predicted 21%; the measurement says the replication is largely absorbed. Real but small — ranked
   below L1/L4 on evidence, not on argument.
6. **B8 trid double-issue** — invisible on the benched shapes (1 row-block per core on
   `grid_filling`); needs `wide_prefill` (2 blocks/core) or `(99991,64)` (28 blocks/core). Recorded
   as a *bench* gap, not as evidence the lever is inapplicable.

### Notable implementation findings (kept in the git log)

* **Blocked `eltwise_chain` outputs need `ReservePolicy/PushPolicy::PerBlockSize`.** `PerTile` emits
  one reserve/push per *block* iteration, not per tile — it packs `block_size` tiles into a single
  CB page and pushes 1.
* **`sum_of_squares` only accumulates a tile-row element-wise.** Regime A needs a within-tile
  `reduce<SUM, REDUCE_ROW>` finalize after it (the catalog's `row_reduce_accumulate` shape).
* **CB-wrap invariant.** A multi-page `cb_reserve_back`/`cb_wait_front` plus a contiguous N-page
  access is legal only when the CB's page count is a multiple of N *and* the fifo pointer is
  N-aligned. A short trailing W-chunk and a partial last row-block both broke it, and the access ran
  past the end of the CB into the neighbouring one — deterministic, silent corruption of a handful of
  tiles. Fixed structurally: the W-chunk search is over **divisors** of `Wt_core`, and every
  row-block is exactly `BLOCK_HT` tile-rows (phantom rows clamped by the reader, dropped by the
  writer).
* **Blackhole DRAM NoC alignment is 64 B.** Placing gamma's tile row 0 with two per-face reads is
  illegal (the second face starts at stick offset +32 B); the staged `cb_gamma_rm` + `tilize` path,
  chunked at `GAMMA_INGEST_BLOCK`, is both legal and L1-bounded.

---

## Phase 0 — Verification pass

- **Date**: 2026-08-19
- **Device**: blackhole p150b, 13×10 compute grid.
- **What was done**: independent verification of the Phase 0 implementation (registry conformance,
  design/blocking-model conformance, helper usage, correctness mechanics), the golden suite +
  `eval.verify_supported`, a precision baseline, and the refinement queue.

### SUPPORTED at Phase 0 (unchanged by this pass — no drift found)

`dtype=[float32, bfloat16]`, `fp32_dest_acc_en=[True]`, `layout=[TILE, ROW_MAJOR]`,
`alignment=[tile_aligned, w_non_aligned, h_non_aligned]`, `rank=[2,3,4]`,
`gamma_mode=[gamma, no_gamma]`, `gamma_dtype=[float32, bfloat16, "none"]`,
`gamma_layout=[TILE, ROW_MAJOR, "none"]`, `memory_layout=[INTERLEAVED]`.
`EXCLUSIONS=[{dtype: float32, fp32_dest_acc_en: False}]`.

### Golden suite at Phase 0

752 / 40 828 passed, **0 failed, 0 errors, 0 hangs**. Per `verifier_report.json`:
**supported_pass 737**, xfail_expected 6 172, invalid_skipped 33 900, infeasible_skipped 2,
no_axes_found 15 (all `test_regression.py`, all passed), and **supported_fail 0 / xpass_drift 0 /
xfail_wrong_mode 0 / supported_marked_xfail 0**.
`invalid_unexpected 2` — two `test_translated.py` bf8b cells that match an *author-scoped* INVALID
entry but are not skipped by that file; the op refused them correctly. Feature-spec authoring issue,
not an op defect.

### Accuracy achieved (`test_rms_norm_precision_baseline.py`, 4 shapes × 2 dtypes)

| dtype | PCC gate | max_abs_err | mean_abs_err | rel_rms_err | got/true ratio (median, p5..p95) |
|---|---|---|---|---|---|
| bfloat16 | ≥0.995, all pass | 0.043–0.100 | ~1.7e-3 | 3.3e-3 – 3.5e-3 | 1.00000, 0.9944..1.0056 |
| float32 | ≥0.999, all pass | 0.013–0.029 | ~8e-4 | 1.2e-3 – 1.5e-3 | 0.99878, 0.9975..1.0000 |

Flat in shape and in `W` — no accumulation drift as the reduced axis widens. bf16 sits below one bf16
quantization step (0.39 %). fp32's tight 0.12 % low bias is the FPU tf32 truncation of the three
multiplied operands, **not** a scale bug: the `w_non_aligned` shape's ratio is identical to the
aligned shapes' (a padding-fold bug would be +15.5 % there), which is direct evidence risk R1 is
handled. The baseline test asserts `|ratio_median − 1| < 0.02` so a regression trips loudly.

### Issues found and fixed

1. **DRY violation** — the CB set was described twice (L1 budget solver vs descriptor creation).
   Unified into one `_cb_layout()`; the solver now *sums* it and `create_program_descriptor()`
   *instantiates* it. A knob turn can no longer drift the budget away from the allocation.
2. **`cb_sumsq` over-allocated in Regime A** (`2 × BLOCK_HT` pages where the second generation only
   exists for Regime B's cross-chunk `Accumulate`). Now exact per regime; marginally widens Regime A's
   L1 reach.
3. **Output-placement hole in `validate()`** — an interleaved input with an explicit sharded
   `memory_config=` request passed validation and would have been written through an interleaved
   `TensorAccessor`. Now gated against `SUPPORTED["memory_layout"]`.
4. **Attempted and reverted**: amortizing `tilize` init across the RM-gamma ingest loop
   (`InitOnly`/`Neither`/`UninitOnly`) is numerically wrong here — the chunks are separate CB
   reserve/push groups, and it corrupts every chunk after the first (PCC 0.24 / −0.018 / 0.0035).
   Reason recorded in the kernel.

No SUPPORTED value was added or removed (`xpass_drift = 0`), and no EXCLUSION was used to silence a
failure.

### Perf after the verification edits (non-regression baseline for Phase 1+)

| name | shape | layout | Phase 0 ns | after this pass | delta |
|---|---|---|---|---|---|
| `grid_filling` | (1,1,8192,1024) | TILE | 93 415 | **93 656** | +0.3 % |
| `wide_prefill` | (1,1,8192,7168) | TILE | 1 019 487 | **1 019 959** | +0.05 % |
| `grid_starved` | (1,1,32,7168) | TILE | 76 149 | **76 161** | +0.02 % |
| `smallest` | (32,17) | TILE | 3 267 | **3 221** | −1.4 % |
| `row_major` | (1,1,8192,1024) | ROW_MAJOR | 95 053 | **95 881** | +0.9 % |

All within measurement noise — the verification edits are perf-neutral.

### Tests added

`tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_precision_baseline.py` (PCC + abs/RMS error +
the got/true ratio-spread scale-bug detector, 4 shapes × 2 dtypes).
Existing: `test_rms_norm.py` (73 pass), `test_rms_norm_bench.py`. Whole op directory: **82 pass**.

### Refinement queue

`op_requirements.md` now carries 5 refinements at the 2:1 generality/perf cadence:
1. numeric configurability (`bfloat8_b`, `fp32_dest_acc_en=False`) — **the gate on every perf phase**,
   because every perf loose case runs at `fp32_dest_acc_en=False`;
2. sharding-native `memory_layout` (HEIGHT local zero-copy CB = knob-turn; WIDTH/BLOCK cross-core
   combine = scheme-change);
3. perf — the flagged `(1,1,32,7168)` interleaved decode case and its 7× gate, via a gated
   `Rt`→`Wt` split diversion;
4. perf — wide-`W` prefill, fuse the pass-B multiplies to fit Regime A (single-read) at `Wt = 224`;
5. perf completeness audit (`/perf-ceiling-dm` Mode D) over the full lever ledger.

---

## Refinement 1 — Numerical configurability expansion

- **Date**: 2026-08-19
- **Device**: blackhole p150b, 13×10 compute grid. All ns are `DEVICE KERNEL DURATION [ns]`
  from the Tracy per-op CSV (`scripts/run_safe_pytest.sh --profile`).

### What was done

**SUPPORTED grew by three axis values, and `EXCLUSIONS` did not grow at all.**

| axis | before | after |
|---|---|---|
| `dtype` | float32, bfloat16 | float32, bfloat16, **bfloat8_b** |
| `gamma_dtype` | float32, bfloat16, "none" | float32, bfloat16, **bfloat8_b**, "none" |
| `fp32_dest_acc_en` | [True] | **[True, False]** |

`EXCLUSIONS` is unchanged (`{dtype: float32, fp32_dest_acc_en: False}` only). **No bf8b corner
needed excluding** — every one the feature spec allows passes.

**Zero new kernel files, zero new CBs, zero second code paths.** The compute kernel is 100 %
`kernel_lib` helpers, so the dtype and DEST-width work is descriptor-level, exactly as
`/numeric-formats-metal`'s pass condition predicts. What was added:

1. **Two derived CB formats, each resolved in ONE place** (`_interm_dtype` / `_acc_dtype`), surfaced
   as the new `"interm"` / `"acc"` kinds in the single `_cb_layout()`:
   - a compute-only intermediate (`cb_squared`, `cb_normed`) is **never block-float** — 16 datums
     sharing an exponent would re-quantise a value the next phase reads straight back;
   - an accumulator CB (`cb_sumsq`, `cb_rms_recip`, `cb_sumsq_acc`) is `Float32` only when DEST
     itself accumulates in fp32. Both are byte-identical on the float32/bfloat16 paths, so no
     Phase-0 cell moved.
2. **`_elem_size()`** — `Tensor.element_size()` *raises* for bfp8, and the value is only used by the
   ROW_MAJOR stick arithmetic, which a block format can never reach (ttnn refuses to build a
   ROW_MAJOR bfloat8_b tensor). Derived from the tile size, with that invariant asserted.
3. **A reduce-datapath knob** (below) — the one real kernel change, and it is two template
   parameters.
4. **Four new live knobs**: `acc_narrow`, `reduce_via_add`, `wt_block`, plus `dest_acc` /
   `pack_precise` (the F25/F24 counterfactual arms). Every one has a re-runnable
   `levers=dict(...)` off-arm in `_bench_rms_norm.py`.

### The bug this refinement had to find: a 16-bit-DEST sum-of-squares bias

Turning `fp32_dest_acc_en=False` on is one line. Making it *correct* was the work.

At 16-bit DEST, Regime B's `reduce_tile` datapath carries a **systematic sum-of-squares
overestimate that grows with the reduced width**:

| `Wt` | 32 | 64 | 128 | 224 | 344 |
|---|---|---|---|---|---|
| implied `Σx²` bias | +0.84 % | +1.90 % | +5.56 % | +10.4 % | +28 % |

At `W = 7168` that is a **uniform 4.8 % low output scale on every row — while PCC still read
0.99995.** It failed the golden suite only through the `rms` metric, and it is invisible to PCC.

Diagnosis (probes 010–018, all preserved). Ruled out, each by measurement, not argument:

- **the W-chunk size** — `WT_REDUCE_BLOCK` swept 112 → 2 (2 to 112 chunks): **bit-identical** rms.
  This is what proved the loss is not the cross-chunk accumulation;
- **the accumulator CB format** (fp32 vs bf16): no difference at any chunk size;
- **`DEST_BLOCK`** (8/4/2/1): no difference;
- **input magnitude** (`Σx²` from 449 to 175 633): no difference — so not "adding small to large";
- **gamma** value, dtype and `math_fidelity`: no difference.

The isolating experiment was a **matched-width Regime A vs B comparison** (`W = 1016/1024`,
`2040/2048`, `4088/4096` — the non-aligned member forces Regime B at the same `Wt`): Regime A is
clean at 16-bit DEST (−0.09 % … +0.06 %, *better* than Regime A at fp32 DEST), Regime B is not. One
earlier mis-step is worth recording: a "clean" no-gamma control at `W = 7168` was actually **Regime
A** (without `cb_normed`/`cb_gamma` the resident path fits L1), which made the bug look
gamma-related for two probes.

**Fix**: route Regime B's reduce to `ReduceAlgorithm::AccumulateViaAdd` — pairwise FPU
`add_tiles` into one DST register plus a single SFPU within-tile finalize, i.e. the same
accumulate shape Regime A already proves clean, and which the helper documents as "more accurate …
wins for wide reduces". Result: rms **0.0061–0.0088, flat in W** (was 0.0086 → 0.1219).

Two follow-on traps, both found and fixed:

- **The two datapaths take different `ReducePartialScaler` forms.** `last_tile_at()` carries only a
  scaler-tile *index*; `AccumulateViaAdd` needs `partial_mask()`, which also carries the valid-element
  *count*. Passing the wrong one is **silent** — `valid_reduce_dim_elements` stays 0, the datapath
  reads "tile-aligned" and never masks, so poisoned tile padding enters the sum (rms ≈ 1.0 on every
  `w_non_aligned` pad-poison case). The reader now emits `prepare_reduce_mask` (0/1, row-0 broadcast
  layout) on that arm and the partial scaler on the other, both off the same knob.
- **`AccumulateViaAdd` folds that mask out of the scaler CB with no data-format reconfig**, so a
  masked reduce is only correct when the (mandatorily bfloat16) scaler CB matches the reduce input
  CB. A `float32` `cb_squared` makes it unpack the bf16 mask as fp32 — **rms 0.59 at W=17**. This
  regressed a *prior-phase* cell (float32 × `w_non_aligned`) that my first golden `-k` filter had
  deselected; the lesson is in the retro note below. `_reduce_via_add()` now gates on **needed**
  (16-bit DEST) **and correct** (`interm_dtype == bfloat16`, or no mask), which also means every
  Phase-0 configuration keeps Phase 0's datapath byte-for-byte.

### Accuracy achieved

`test_rms_norm_precision_matrix` (new, mandated by `/numeric-formats-metal` §10): 6 shapes ×
3 dtypes × 2 DEST widths × 2 fidelities = **48 passed, 24 skipped, 0 failed**. Full table in
`tests/ttnn/unit_tests/operations/rms_norm/precision_matrix_results.md`.

| metric | worst over all 48 cells | gate |
|---|---|---|
| PCC | **0.999944** | ≥ 0.99 |
| relative RMS error | **0.01316** | (printed) |
| **row-scale bias** | **0.00841** | < 0.02 |

The `row_scale_bias` assertion is new and is the point: it is the mean relative error of the per-row
`1/rms` factor, and it is the gate that would have caught the bug above at its first appearance,
where PCC would not. (Its regressor must include gamma — fitting `out ~ k·x` cancels to ~0 for a
random-sign gamma and reports bias ≈ −1.0 for a *correct* kernel. That mistake is in the git log.)

24 skips: 12 `{float32, fp32_dest_acc_en=False}` (op `EXCLUSIONS`) and 12 `{bfloat8_b ×
non-tile-aligned}` (`feature_spec.INVALID`).

### Golden test progress

| suite / slice | result |
|---|---|
| `test_golden.py` cartesian, **all INTERLEAVED cells** | **1500 passed, 0 failed**, 420 xfailed (sharded → Refinement 2) |
| `test_golden.py` loose, every `fp32_dest_acc_en=False` INTERLEAVED case | **0 failed** |
| corrected slice incl. float32 `w_non_aligned` | **436 passed, 0 failed** |
| `tests/.../rms_norm/` (whole dir, minus bench) | **129 passed**, 24 skipped |

Phase 0 had **700** interleaved cartesian cells passing; the supported rectangle roughly doubled to
**1500** because `bfloat8_b` and `fp32_dest_acc_en=False` are both live.

**The gate this refinement exists for is met**: the `(1,1,32,7168)` interleaved perf loose case now
**reaches the op and passes** at its exact config — bf16 / TILE / INTERLEAVED / HiFi2 /
`fp32_dest_acc_en=False` — rather than xfailing. Refinement 3 can now measure the real datapath.

### Perf — bound classification and the cumulative bench set

**No regression.** Every carried-forward shape, re-measured at the op's default corner (where both
new levers are no-ops by construction):

| name | shape | layout | Phase 0 ns | **Refinement 1 ns** | delta |
|---|---|---|---|---|---|
| `grid_filling` | (1,1,8192,1024) | TILE | 93 656 | **94 052** | +0.4 % |
| `wide_prefill` | (1,1,8192,7168) | TILE | 1 019 959 | **1 020 216** | +0.0 % |
| `grid_starved` | (1,1,32,7168) | TILE | 76 161 | **76 090** | −0.1 % |
| `smallest` | (32,17) | TILE | 3 221 | **3 270** | +1.5 % |
| `row_major` | (1,1,8192,1024) | ROW_MAJOR | 95 881 | **95 134** | −0.8 % |

**Added to the cumulative set** (new *config* points on the same five shapes, so the documented shape
set is unchanged; `RMS_BENCH_MODE=precision`):

| shape | default (HiFi4/fp32-DEST) | **loose corner** (HiFi2/fp32-DEST-off) | bfloat8_b | bf8b + loose |
|---|---|---|---|---|
| `grid_filling` | 93 536 | **91 011** | 71 113 | 68 216 |
| `grid_starved` | 76 080 | **47 995** | 65 987 | 41 732 |
| `wide_prefill` | 1 019 072 | **988 864** | 475 804 | 454 833 |

Bound classification is **unchanged** by this refinement and re-confirmed by the lever behaviour:
`grid_filling` stays **DRAM-bound** (every data-path lever flat; C16 flat even at bfloat8_b, which
halves the bytes — it is still DRAM-bound, just at a smaller total), `grid_starved` stays
**compute-bound** (it is the only shape where the compute-side levers move at all). No DM ceiling is
re-derived here: this refinement moved no bytes on the default corner.

`bfloat8_b` is much faster purely because it halves DRAM traffic — **2.17×** on `wide_prefill`
(988 864 → 454 833 at the loose corner), 1.31× on `grid_filling`, 1.15× on `grid_starved`.

### Levers (`lever_ledger.json`; `python3 -m eval.verify_levers <ledger> --bench <bench> --report`)

`verify_levers` is **clean: 0 blocking, 0 signal, 0 stale.** 18 of 29 closed with evidence
(was 12), by status `applied=8, deferred=14, measured-no-payoff=2, missed=2,
structurally-impossible=3`.

| lever | knob | on → off | result |
|---|---|---|---|
| **F25** `fp32_dest_acc_en` | `dest_acc` | 71 796 → 76 080 (`grid_starved`) | **1.06×**, attributed alone; flat on `grid_filling` (DRAM-bound) — Phase 0 deferred F25 *onto this refinement* |
| **reduce_via_add** | `reduce_via_add` | 47 995 → 51 465 (`grid_starved`) | **1.07×** — the correctness fix is also a compute-side win |
| **B6** whole-page transactions | `coalesce` | 71 273 → 80 440 (bf8b) | **1.129×** — **unlocked** by this refinement (see below) |
| **A1** `row_wise` | `row_wise` | 71 273 → 74 486 (bf8b) | 1.045× on bf8b, −1.7 % on bf16 — corners disagree, stays `measured-no-payoff` |
| **F24** `bfp8_pack_precise` | `pack_precise` | 71 113 → 71 135 | **flat** — the applied FAST packer is right, and now measured |
| **acc_narrow** | `acc_narrow` | 91 011 → 91 547 | **flat**, kept (see below) |
| **C16** double-buffering | `double_buffer` | 91 201 → 90 594 | still **flat**, now confirmed on both new regimes |

Three closures were **re-opened and re-decided because this refinement moved their premise** — which
is exactly what the ledger's topology stamp is for:

- **F24** was `structurally-impossible` on the premise "SUPPORTED[dtype] holds no block format, so
  there is no bfloat8_b pack for the knob to govern". Adding `bfloat8_b` voided that. Re-closed
  `applied`: the **fast** packer is the right setting (`/numeric-formats-metal` §1.7 gates the precise
  packer on a *float32 input*, and this op's output dtype always equals its input dtype, so a bf8b
  output implies a bf8b input), the accuracy margin confirms it (min PCC 0.999944), **and** both arms
  are now measured flat so the choice carries no throughput risk either way.
- **B6** was `measured-no-payoff` at a 2048 B bfloat16 tile. A bfloat8_b tile is 1088 B and the
  whole-page transaction now saves **1.129×**. Measuring it honestly required **fixing the
  counterfactual**: the off-arm split the page at `IN_TILE_BYTES/2` = 544 B for bf8b, which is *not* a
  multiple of Blackhole's 64 B DRAM alignment, so the second transfer was illegal — and the illegal
  arm over-reported the win as 1.249×. Rounding the first half down to a 64 B multiple (1088 → 512 +
  576) fixed both.
- **F23 / F26 / B11** were closed under an *unrecorded* topology and so could not be re-checked
  mechanically; all three are re-confirmed with the topology now stamped. **F23's wording needed
  tightening**, because this refinement added the op's only config-touching code: for every real call
  the caller's descriptor is still passed verbatim, and the sole rebuild site
  (`_apply_precision_levers`) is reachable only from the internal `_levers` bench hook. **F26 gained a
  second, independent disqualifier** found while auditing the `UnpackToDestFp32` candidates: the one
  CB that is otherwise eligible (`cb_sumsq`) is read through SrcA/SrcB by `reduce()`'s accumulator
  reload, which the helper documents as valid *only* for `UnpackToDestMode::Default` — tagging it
  would be incorrect, not merely slower. So **no CB in this op can carry the tag.**

**`acc_narrow` is a kept-but-flat lever, not a reverted one.** It narrows the accumulator CBs on the
16-bit-DEST path, and it is proven to cost **zero** precision (bit-identical rms at every chunk size).
It measures flat today because these shapes are DRAM- or compute-bound, not L1-bound. It is held at
its applied default and is the lever that pays once Refinement 4's regime flip makes L1 the binding
constraint — the freed pages go straight into `BLOCK_HT` / `IN_BUF_DEPTH`.

**F27 (`math_fidelity`) remains the largest unexploited compute lever** and stays `missed`, not
closed: re-measured at the new DEST width it is **1.50×** on `grid_starved` (71 796 → 47 995), up from
Phase 0's 1.45×. It is not applied because `references/precision_convention.md` pins the exported
default and F23 forbids downgrading a caller-supplied value. Note the perf-gated loose case *supplies*
HiFi2 itself, so that cell already gets the win.

### Issues encountered

1. The 16-bit-DEST reduce bias, its two follow-on traps, and the fp32 regression — all above.
2. **A too-narrow golden `-k` filter hid a real regression.** My first slice selected only cells
   matching `False` or `BFLOAT8_B`, which *deselected* the float32 × `w_non_aligned` cells the
   datapath change actually broke. It surfaced only because 8 bf8b-**gamma** cells failed and chasing
   them revealed the gamma dtype was irrelevant. The final verification therefore ran the **whole
   interleaved cartesian surface** (1500 cells, 50 s) rather than a filter — cheap, and it is the
   honest regression net for a change to a shared datapath.
3. `ReduceFp32Mode` lives at global scope, not in `compute_kernel_lib` (free compile error).
4. `Tensor.element_size()` raises outright for bfp8/bfp4/bfp2 — hence `_elem_size()`.

### Tests added

- `tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_precision_matrix.py` — the authoritative
  precision characterization (48 cells) with the PCC **and** row-scale-bias gates.
- `tests/ttnn/unit_tests/operations/rms_norm/precision_matrix_results.md` — the results table.
- `_bench_rms_norm.py`: `run_precision()` (precision corners, bf8b, and the new levers' off-arms) and
  `run_unlocked_recheck()` (re-runs a "possibly unlocked" lever's arm on the regimes that changed),
  plus `RMS_BENCH_MODE=precision|recheck`.
- Probes 008–024 preserved under `probes/` — the full diagnosis trail.
