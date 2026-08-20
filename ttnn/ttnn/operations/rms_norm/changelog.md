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

---

## Perf 1 — fan-out perf tournament (7 ideas measured, 5 graduated, 2 null/regression)

Nothing was added to `SUPPORTED`; `verify_supported` categories are unchanged across this
round. The signal here is device-ns.

**Headline: the perf-gated focus case goes 43,947 → 9,010 ns (4.88×), against a 14,894 ns
gate — 1.65× under it.** Every guard-set arm improved or is flat; none regressed.

### The focus case (mandatory primary target)

`feature_spec.LOOSE_CASES` `_perf_case(32, 7168, 104259, minimum_expected_speedup=7.0)`:
`(1,1,32,7168)`, bf16, TILE, INTERLEAVED, gamma `(1,1,1,7168)` bf16 **TILE**,
`math_fidelity=HiFi2`, `fp32_dest_acc_en=False`, soft PCC gate 0.9995. Device clock measured
at exactly 1350 MHz = the spec's `reference_aiclk_mhz`, so the goal needs no scaling:
`104,259 / 7.0 = 14,894 ns`.

Every axis this case pins is in `SUPPORTED` (bf16 ✓, `fp32_dest_acc_en=False` ✓ — the only
`EXCLUSIONS` entry is fp32×False — TILE ✓, INTERLEAVED ✓, rank 4 ✓, tile_aligned ✓), so there
was no generality gap to report and no proxy was substituted. It was measured at its exact
config throughout: a new `focus` bench shape was added precisely because the pre-existing
`grid_starved` shape differs in one knob that is a **different datapath** — ROW_MAJOR gamma goes
through a staging CB plus a compute-side tilize, TILE gamma is read straight into
`cb_gamma_tiles` — and measuring the goal on that proxy would have measured a different program.

### Instrumentation (PERMANENT)

`ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp` did not exist in this tree; it was created,
carrying `MaybeDeviceZoneScope`, the durability contract, the attribution rules and the
**silent** marker-budget trap. All three kernels are now zoned at every stage boundary, with
each NoC region split into `_reserve` (back-pressure from the consumer) / `_issue` (RISC-serial
transaction issue) / `_barrier` (the real NoC wait) — because a barrier at ≈0 with a hot issue
loop and a hot barrier with a cheap issue loop want opposite fixes. `_zone_report.py` folds
`profile_log_device.csv` into per-(zone, RISC) totals and runs the two integrity checks the
marker budget makes mandatory. This instrumentation is not to be removed; it is free when the
profiler is off.

### Measured breakdown — the ranked bottleneck

Focus case **before**: 43,947 ns on **ONE core of a 13×10 = 130 grid** (Rt=1 → 1 row-block →
1 core), Regime B (**two** DRAM reads of x), `Wt_core=224`, `WT_REDUCE_BLOCK=WT_SCALE_BLOCK=112`,
`BLOCK_HT=1`, `DEST_BLOCK=8`, `IN_BUF_DEPTH=OUT_BUF_DEPTH=1`, working set 1,155,072 of
1,269,888 B. Eight CBs, five of them 112 pages wide: in / squared / gamma / normed / out.

Cumulative ablation (payload stubbed, sync scaffolding intact), peeled cumulatively because the
NoC reads run in parallel with TRISC compute:

| arm | ns | share |
|---|---|---|
| full | 43,983 | — |
| `stub_compute` (DM only) | 24,093 | 55% |
| `stub_dm` (compute only) | 28,221 | 64% |
| `stub_both` (the floor) | 9,230 | 21% |

`full ≈ stub_dm + stub_compute − floor` (43,983 vs 43,084). **The two stages were additive, not
overlapped** — a chunk of 112 tiles with depth-1 CBs is a strict ping-pong. That reading is what
the whole-op-ablated floor licenses; neither single-stub run alone would have said it.

Per-stage zones, the single active core, ns per dispatch:

| zone | RISC | ns | reading |
|---|---|---|---|
| `wr_wait` | BRISC | 35,030 | the writer is **starved** 80% of its 43,855 ns span |
| `cp_scale_mul` | TRISC_2 | 17,137 | occupancy: the helper's own `cb_wait_front` on the reader's 2nd pass is inside it |
| `cp_square` | TRISC_2 | 13,671 | |
| `rd_in_issue` | NCRISC | 13,467 | 4 execs × 112 tiles ≈ **30 ns/tile of issue** |
| `wr_issue` | BRISC | 8,101 | |
| `rd_gamma_issue` | NCRISC | 6,750 | 224 gamma tile transactions |
| `rd_gamma_reserve` | NCRISC | 6,406 | the reader back-pressured on the gamma CB |
| `cp_gamma_mul` | TRISC_2 | 6,324 | |
| `cp_reduce` | TRISC_1 | 3,284 | |
| `rd_in_reserve` | NCRISC | 3,105 | |
| `cp_rms_chain` | TRISC_2 | 1,513 | |
| `rd_in_barrier` | NCRISC | 1,446 | |
| `rd_gamma_barrier` | NCRISC | 835 | |
| `wr_barrier` | BRISC | 427 | |

Marker-cap check: 0 of 5 (core, RISC) at or above 250, and the zones cover the whole kernel span
— so the breakdown is complete, not silently truncated.

**Ranked, roofline-gated:**

1. **No parallelism.** 1 core of 130. Every per-core cost above divides by the core count.
2. **No reader/compute overlap** — proven by the additive ablation, not inferred.
3. **Reader looked issue-bound** — 20.2 µs of issue against 2.3 µs of barriers over 672 tile
   transactions. *This reading was later corrected by measurement — see the null below.*
4. **Compute makes three passes over W** (square, scale-mul, gamma-mul) plus two reduce passes.
5. **gamma is 33% of the focus case** (43,947 with gamma vs 29,403 without) and 224 of the 672
   transactions. A separate discovery from the guard-set baseline: TILE-layout gamma costs
   **+26.7 µs on `(1,1,8192,1024)`** (117,695 vs 91,011 with ROW_MAJOR gamma at the same corner)
   and **+241 µs on `(1,1,8192,7168)`**, because a `(1,1,1,W)` gamma tile-padded to 32 rows is
   read WHOLE — 31/32 of every gamma byte is padding, and every perf-gated case uses TILE gamma.
6. `/perf-ceiling-dm` gating: `(1,1,8192,1024)` was already at 1.03× of its DM target and
   `(1,1,8192,7168)` at 0.99× **of the 2-pass algorithm's** bound — so on the prefill shapes the
   headroom was never in the transfers, it was in moving fewer bytes (drop the 2nd read of x, stop
   replicating gamma). Both were attacked, and both paid.

### The portfolio (overlap and fusion deliberately allowed)

Seeded from `lever_ledger.json`, then filtered by the measured breakdown. Adopted seeds:
**C16** (a "possibly unlocked" row — its on/off arms were the *same program* on a wide Regime B
shape, so its `measured-no-payoff` had never actually been tested there), **B13** and **B8** (the
ranked-remaining rows aimed at the reader's issue loop), **B12** (the reuse-shared broadcast, a
standing portfolio idea), **D20** (the #1 ranked row at a predicted 50%). Rejected seeds: **C15 /
C14 / A2 / A3** (ranked #1-#4 by predicted size but all gated on a sharded `memory_layout` that
is not in `SUPPORTED` — a perf round cannot unlock it), **F27** (largest unexploited compute lever
but it is a precision-contract change, forbidden), **B10 / D18 / D19 / D21 / A4 / C17 / E22**
(≤3% predicted, and the breakdown does not put them on the critical path).

| # | idea | targets | verdict |
|---|---|---|---|
| 1 | `w_split` — cut the DEPENDENT W axis across cores, combine partials cross-core | bottleneck 1 | **WIN 4.72×** |
| 2 | `pipeline_overlap` — subordinate the W-chunk search to CB depth | bottleneck 2 | **WIN 1.36×** |
| 3 | `stateful_noc` — B13 stateful transfers + B8 trid, reader **and** writer twin | bottleneck 3 | **partial WIN / diagnosis correction** |
| 4 | `gamma_row0` — stop reading gamma's 31 padding rows; + read-once; + B12 mcast | bottleneck 5 | **WIN 1.25×** |
| 5 | `fused_sumsq` — fuse Regime B's square into the reduce (Regime A's shape, chunked) | bottleneck 4 | **WIN on L1 + precision** |
| 6 | `fused_scale` — fuse the two scale muls into one DEST window (Lamp L4) | bottleneck 4 | **REGRESSION** |
| 7 | `resident_single_read` — x resident once, scale pass chunked (regime ladder) | bottlenecks 3+4 | **WIN 1.42×, superseded** |

Ideas 5/6/7 deliberately overlap idea 4's stage, and 1 overlaps 2/5/7 on the focus case. That is
what the aggregation step is for.

### Per-idea verdicts, with numbers and domain

**1. `w_split` — WIN, graduated.** Focus 44,314 → **9,386 ns (4.72×)** at group size 32; the
group-size sweep is 1→44,314 · 4→14,645 · 8→10,923 · 14→10,161 · 16→9,838 · 28→10,060 ·
**32→9,386** · 56→10,368, and a payload-stubbed arm prices the combine's own floor at ~1,113 ns
at G=14 rising to ~2,770 at G=56 — which is exactly why the optimum is interior. `mcast_pipe`'s
1→N leg beat a hand-rolled unicast fan-out at every group size (10,161 vs 10,730 at G=14; 10,368
vs 12,016 at G=56). Domain: everywhere on the TILE path, 1.17× to 7.03×. **Two mechanisms pay and
only the first needs an empty grid** — parallelism, and a narrow per-core slice that makes Regime A
fit (one DRAM read of x instead of two, gamma not replicated), which is why `(1,1,8192,7168)`
nearly halves at Rt=256.

**2. `pipeline_overlap` — WIN, graduated.** Focus 43,140 → **31,844 ns (1.36×)** at **less** L1
(1,155,072 → 811,008 B), host-plan only, PCC bit-identical on every arm. Overlap captured
0% → 62%; `rd_gamma_reserve` 6,396 → 146 ns, `rd_in_reserve` 3,100 → 306, `wr_wait` 34,982 →
23,784. The win needs **both** `cb_input_tiles` and `cb_gamma_tiles` at depth 2 — isolated at
chunk 56: all-depth-1 0.94×, output-only 0.94×, input-only 1.02×, gamma-only 1.18×,
input+gamma 1.36× — and `cb_gamma_tiles` was hard-coded to depth 1, so no arm of the old knob
could reach it. The depth-1 chunk curve confirms the ledger's caution that fine chunks *alone*
lose (112→43,186 · 56→46,016 · 8→85,544): the optimum exists only as a joint (chunk, depth)
choice. Exception list empty.

**3. `stateful_noc` — a partial WIN and, more valuably, a CORRECTED DIAGNOSIS.** The graduated
half is `one_packet` (pass a compile-time size bound so the runtime any-length dispatch loop
disappears): 13.7% on the isolated reader, 12.0% on the writer. The **B13 mechanism proper is a
REGRESSION** — state reuse republishes only `TARG/RET_ADDR_LO`, so it requires bank-major issue
order, which serialises DRAM channels: +17% on `(1,1,8192,1024)`, +37% on the widest shape, +47%
on fp32 writes, +9% at `(32,17)`. It *inverts* at bfloat8_b's 1088 B tiles (−36% read / −44%
write) where the transaction rate genuinely is the limit — a real dtype-gated dual path, recorded
and **not** shipped, because a dual path is debt. **B8 trid is NULL**: +5.5 ns/txn of surcharge
against 3.1 ns/txn of barrier it could hide.

> **The correction, and it matters more than the 13%.** Step 1 read "issue 20.2 µs vs barriers
> 2.3 µs" as *RISC-issue-bound*. Wrong. The reader and writer sit at the per-core **NoC/DRAM
> service ceiling of ~29 ns per 2 KB page each way (~70 GB/s)**, and the baseline issued at
> 30.7 ns/txn — just *above* that rate, which is precisely why the zones showed "issue hot,
> barrier ≈ 0". Cutting issue to 16.8 ns/txn puts it *below* the service rate, so the saving
> cannot become throughput: it moves ns-for-ns from `rd_issue` into `rd_barrier`. Measured at
> whole-op scale, not inferred — `(1,1,8192,1024)`: `rd_in_issue` 5,763 → 4,904 (−14.9%) with
> `rd_in_barrier` 14,648 → 15,233 (+585). The win only surfaces where a core is issue-serialised
> rather than DRAM-serialised. The permanent issue/barrier zone split is what made this visible;
> a single fused `reader_noc` zone could not have distinguished the two.

**4. `gamma_row0` — WIN, graduated.** The winning variant is `span`: ONE `noc_async_read` of the
row-0 span (bf16 2048 → **544 B**, fp32 4096 → 1088, bfloat8_b 1088 → 336). The crux answered by
measurement: **the gamma read is transaction-bound, not byte-bound** — the minimal-bytes variant
(64 B/tile in *two* transactions) LOSES 6% on the focus case and loses on every prefill shape too.
A one-packet issue variant of the span measured NULL, confirming the win is bytes. A/B in one
tree: `prefill_1024` 118,112 → **93,542 (1.263×)**, fp32 242,150 → 197,251 (1.228×), bfloat8_b
68,126 → 58,390 (1.167×), `prefill_7168` 622,416 → 602,188 (1.034× — partly subsumed by the split,
which already gives each core a quarter of the gamma tiles). NoC legality is a **residue match**
(`(l1_addr & 63) != (dram_addr & 63)`), not a size alignment, which is why this is legal where the
old ROW_MAJOR-gamma two-face read was not. Exception list **empty** — bfloat8_b gamma is not a
carve-out, because the 64 B per-face-row exponent header sits inside the span. Two sub-options were
measured and rejected: an L1 row-0 cache with local refill (focus 0.556×) and, for B12's gamma
mcast, an `inject_only` ceiling probe bounding it at ≤+5.1% / ≤+8.7% on the two prefill shapes and
**0% on the focus case** *after* `span` — so the ledger's headline gamma-mcast value is not
additive with this idea.

**5. `fused_sumsq` — WIN on L1 and precision; NULL on wall-clock alone. Graduated.** Same-blocking
end-to-end 43,939 → 43,855 (flat), compute-only 28,224 → 25,168 (**1.121×**), and 43,939 →
**40,966 (1.073×)** once the 225,280 B it frees reaches the solver. The honest reading: the
reduce phase was DRAM-starved, so the 222 packs + 224 unpacks + 110 `add_tiles` it deletes were
already hidden. **The precision result is the headline**, and it is an improvement *at* the frozen
contract: row-scale bias goes from −0.32/−0.46/−0.73/−0.76% at Wt = 32/64/128/224 to a flat
−0.17/−0.10/−0.16/−0.08%, i.e. the old bias grew with the reduced width and the fused one does
not. That removes the remaining half of the 16-bit-DEST sum-of-squares trap `AccumulateViaAdd`
only partly fixed at Refinement 1. PCC is blind to it (0.99998 both ways) — which is exactly why
this op keeps a row-scale-bias gate.

**6. `fused_scale` — REGRESSION. Not graduated.** Every fused form is **1.43×–5.7× slower**;
per scale-pass call: baseline 6,641 ns, `baseline_reversed` 6,633 (flat), `fused_rmsfull` 10,448,
`raw_llk` 10,368, `fused_gammafull_amortized` 10,633, `fused_inchain` 17,263, `fused_sfpu` 65,633.
**Lamp L4's premise is false on this hardware**, and the mechanism was isolated: (a) the
`cb_normed` L1 round-trip is **free** — pack (TRISC2) and unpack (TRISC0) overlap math (TRISC1),
measured against a no-gamma control at 29.3 ns/tile for ONE mul vs 29.6 ns per mul-tile for the
baseline's two muls *plus* the whole round-trip; (b) `mul_reuse_dest_tiles` costs ~60 ns/tile
against ~29.6 for `mul_tiles_bcast`, i.e. DEST→src routing is worth about two muls of math-thread
work. A raw-LLK arm matches the chain to 0.8%, so this is not chain or init overhead. Correct
everywhere it ran (9 regimes × 9 arms, PCC ≥ 0.99998) and slower everywhere — **no sub-domain
where it wins.** Its actionable residue (sub-chunk the scale pass, keeping both muls, to free L1)
is subsumed by ideas 2 and 7. *This null is the best-value entry in the round after the win:* it
retires a 1.5×-predicted design lamp with a measured mechanism, so no later round re-buys it.

**7. `resident_single_read` — WIN 1.42×, but SUPERSEDED; not graduated.** Focus 43,119 →
**30,380 (1.419×)**, `prefill_7168` 1,232,827 → **797,803 (1.545×)**, `(1,1,32,5120)` 1.278×,
`(1,1,32,16384)` 1.340×, fp32 1.395×, RM-gamma 1.270×, RM-input 1.124×, with a regime **ladder**
(`A → C1 → C2 → B`) selected by a real property (does this CB set fit) rather than a width list,
and graceful, measured-identical degradation to today's Regime B when x alone does not fit.
Attribution inside it: single-read alone 1.246×, the fused sum-of-squares it enables adds the rest.
It also priced the ledger's open D20 question from both sides: Regime A vs a forced Regime B is
**1.32×** on `(1,1,8192,1024)` and **1.47×** on its ROW_MAJOR twin. Not graduated because
`w_split` attacks the same mechanism (narrow per-core width ⇒ Regime A ⇒ one read, no gamma
replication) and dominates it where they overlap — 4.88× vs 1.42× on the focus case, 2.04× vs
1.545× on `prefill_7168`. Carried to round 2 for the shapes the split cannot take (non-tile-aligned
W, where the ladder would first need the masked variant it does not have).

### Aggregation

`w_split` and `resident_single_read` are both wins that restructure the same thing — the regime/CB
plan — so they cannot both graduate; the split dominates on every shape they share, and it is what
clears the gate. `pipeline_overlap` and `fused_sumsq` survive alongside it because they act on the
residue: shapes where the policy picks G=1 **and** the plan is Regime B (the masked non-aligned
wide shapes, plus the smallest). `gamma_row0` and `one_packet` are orthogonal byte-level and
transaction-level changes that compose with everything. `fused_scale`'s fusion is discarded
outright.

Graduated, in the order they landed: `w_split` → `gamma_row0` → `pipeline_overlap` →
`fused_sumsq` → `one_packet`.

### What graduated, how widely, and every carve-out

Each is the op's **one unqualified path** for everything it is correct on, with the code it
replaced deleted. Carve-outs are written as `if (cannot) { legacy } else { new }` — the exception
shrinks as understanding grows — never as an allow-list around what was benchmarked.

**`w_split` (cross-core W split).** `blocking_plan` → `_choose_group_size` scores every divisor of
`Wt` up to the core count with the **same** `_solve` that produces the shipped plan, so it never
guesses a regime or a block factor. **G = 1 is a value of that policy, not a second code path** —
the row-parallel plan wins by measurement wherever the split does not pay, which is why seven guard
arms are byte-identical. Five properties can send a shape back to G=1:

| # | property | kind | what earned it |
|---|---|---|---|
| P1 | G divides `Wt` | inexpressible | a short trailing slice needs a runtime per-core width (`Wt_core` is a CT arg) + a boundary mask; not built |
| P2 | the slice must solve to Regime A | inexpressible | the combine consumes the pre-collapse accumulator tile only Regime A produces. **This is the entire `W % 32 != 0` carve-out**: on TILE `maskless_w` is false, so every G>1 solves to B and the policy returns G=1 *by construction* — there is no `W % 32` predicate anywhere. On ROW_MAJOR the reader zero-fills each stick's pad tail per core, so a non-aligned W **can** split |
| P3 | the group rect's virtual-coord bbox area == G | incorrect (hang) | the 1→N fan-out is `McastRect::area()`; a non-dense box breaks the handshake count. Worker columns are derived from the device, so Blackhole's skipped virtual columns 8/9 are handled arch-neutrally |
| P4 | ROW_MAJOR per-core stick ≥ 1024 B | **measured regression** | the split shrinks a *stick* but never a *tile page*: 0.59× at 512 B, 0.43× at 256 B. Stated as a stick-byte property, not an RM exclusion — the policy still takes G=4 on `(1,1,1024,7168)` ROW_MAJOR, which wins 1.26× |
| P5 | reader and writer on different NoCs | incorrect (hang) | **found during integration**: `ncrisc_noc_nonposted_writes_flushed` compares a per-NoC hw register against a per-RISC sw counter, so the combine (which makes the reader a writer) deadlocks the writer's barrier when both DM kernels sit on NOC_0. Only the B9 lever off-arm can reach this |

Nothing untested is fenced off — the guard set covers fp32 + `fp32_dest_acc_en=True`, bfloat8_b,
`no_gamma`, `h_non_aligned`, ROW_MAJOR input and ROW_MAJOR gamma, and the split is confirmed taken
on fp32 (both DEST widths), bfloat8_b, LoFi, `dst_full_sync_en=True` and ranks 2/3.

**`gamma_row0` (span read).** Unqualified; `noc_async_read_tile` deleted from the gamma path. **No
carve-out.** Three `HAS_GAMMA`-guarded `static_assert`s turn a future `bfloat4_b` or non-32×32-tile
gamma into a **build error** rather than a silent wrong answer — which is the right shape for an
unbuilt case: it fails loudly instead of being fenced off.

**`pipeline_overlap` (depth-aware chunk search).** Unqualified; the "coarsest chunk at depth 1
first, then double-buffer the remainder" ordering is deleted. **No carve-out** — 12 of 14 guard
arms are byte-identical because they solve to Regime A, which is an honest no-op, not an exception.
The regime-selection predicate deliberately stays on the minimal depth-1 profile, which keeps the
A/B boundary (and with it the split's P2) invariant under a change to how L1 is *spent*: G,
`BLOCK_HT` and `GAMMA_DEPTH` move nowhere.

**`fused_sumsq`.** Unqualified for Regime B; `square` → `cb_squared` → `reduce` and the
`CB_SQUARED` layout entry are deleted (slot 3 retired, not renumbered). **One carve-out:**
`if constexpr (WT_REDUCE_BLOCK == 1)` keeps the pre-fusion `square`, earned by a measured
regression on a supported cell — `(32,17)` went 3,365 → 3,460 because a one-tile chunk has nothing
to fuse. The degenerate fused form *is* `square`, so the carve-out recovers the old shape exactly
and `(32,17)` is back to flat. The non-tile-aligned case stays **on** the fused path (the last
W-tile gets its own accumulator, whose 32 columns map 1:1 onto it, so the existing partial-mask
machinery zeroes exactly the pad columns) — which matters, because masked wide shapes are most of
what is still Regime B after the split.

**`one_packet`.** Unqualified; `noc_async_read_tile` / `noc_async_write_tile` are deleted from both
kernels and the size bound is passed as a template parameter, so the guard *is* the bound and no
live predicate remains on the TILE path. The compile-time fallback survives only on the ROW_MAJOR
stick path (`inexpressible`: the hardware primitive cannot take a larger transfer) — and worth
recording, no shape reachable by this op selects it, because it would need `Wt_core > 256` tiles
while `cb_rm_in` must then hold ≥512 KB. It is correct-by-construction insurance, and it is
**load-bearing** insurance, since `sanitize.h` carries no burst-size rule and an oversized
one-packet transfer would be silent corruption rather than a caught violation.
**Two measured regressions, deliberately NOT carved out:** `prefill_1024` 93,488 → 94,386
(−0.95%) and its fp32 twin 196,177 → 196,927 (−0.38%), both 4/4 reps non-overlapping against a
0.4–0.9% noise floor. They are not added work — on a ≤burst transfer the change is strictly less
RISC work and the NoC transaction is byte-identical — they are a second-order arrival-pattern
effect: issuing a block's reads ~15% faster puts them in flight sooner, and on a DRAM-saturated
arm the burstier arrival costs ~1% (`wr_wait` +708 ns). A measured regression *earns* a guard but
does not compel one, and here the guard is the wrong trade: −0.95% against +6.2% on `w_nonalign`
and +2.8% on `h_nonalign`, paid for with a permanent dual path. A single slightly-sub-optimal path
beats two locally-perfect ones at this size. Recorded with the numbers so a later round revisits it
with evidence rather than rediscovering it.

### Whole-op before → after, and the guard-set no-regression result

One representative per distinct kernel path × layout × placement, measured at the applied
defaults, fresh cache, `DEVICE KERNEL DURATION [ns]`:

| guard-set arm | path | before | after | × |
|---|---|---|---|---|
| **`focus` (1,1,32,7168)** | Regime B→split, TILE, TILE gamma | **43,947** | **9,010** | **4.88** |
| `decode_1024` (1,1,32,1024) | Regime A, 1 core → split | 9,265 | 5,644 | 1.64 |
| `decode_2304` (1,1,32,2304) | Regime A → split | 16,276 | 6,416 | 2.54 |
| `decode_5120` (1,1,32,5120) | A/B boundary → split | 30,636 | 8,180 | 3.75 |
| `prefill_1024` (1,1,8192,1024) | Regime A, full grid, DRAM-bound | 117,695 | 93,934 | 1.25 |
| `prefill_7168` (1,1,8192,7168) | Regime B, full grid → split | 1,229,676 | 603,971 | 2.04 |
| `grid_starved` | Regime B + **ROW_MAJOR gamma** | 49,222 | 8,747 | 5.63 |
| `row_major` | **ROW_MAJOR input** (tilize/untilize) | 91,556 | 92,130 | 0.99 |
| `smallest` (32,17) | B0 per-core-overhead regime | 3,772 | 3,711 | 1.02 |
| `w_nonalign` (1,1,32,4095) | masked reduce / partial scaler | 25,774 | 20,301 | 1.27 |
| `h_nonalign` (1,1,100,736) | phantom-row clamp | 7,909 | 7,393 | 1.07 |
| `focus` `no_gamma` | scale writes straight out | 29,403 | 7,713 | 3.81 |
| `prefill_1024` **bfloat8_b** | block-float datapath | 68,124 | 58,433 | 1.17 |
| `prefill_1024` **float32** | fp32 + fp32 DEST | 242,983 | 197,661 | 1.23 |

**No regression on any supported cell.** The single arm at 0.99 (`row_major`, −0.6%) is inside the
run-to-run band; its plan is byte-identical (the policy picks G=1 there).

Correctness, on the final combined tree: golden **1603 passed / 5202 xfailed / 0 failed**
(210 + 450 + 523 + 420 across the four `test_golden` slices) plus `test_regression.py` 15 passed —
an exact match to the pre-round baseline. Precision matrix **48 passed / 24 skipped**. The full
`eval/golden_tests/rms_norm/` directory exceeds a 10-minute tool timeout, so it is run as five
invocations (`-k` cannot contain `=`): `test_regression.py`, then `test_golden.py` with
`-k "FLOAT32 and not BFLOAT"`, `-k "BFLOAT8_B"`,
`-k "BFLOAT16 and not BFLOAT8_B and not FLOAT32"`, `-k "FLOAT32 and BFLOAT16 and not BFLOAT8_B"`.

### The critical path after this round (round-2 handoff)

The focus case is now 32 cores, and the breakdown has moved. Ablation: full 9,073 /
`stub_dm` 6,477 / `stub_compute` 6,316 / `stub_both` **4,461** — the floor is now **49% of the
wall**, so this shape has become overhead-dominated rather than payload-dominated. Zones, per core
(32 execs, ns/exec): `wr_wait` 7,058 of an 8,176 ns BRISC span (the writer is still starved, now by
compute), `cp_scale_mul` 3,661, **`cp_rms_chain` 3,448**, `cp_sumsq` 2,695, `mcast_recv` 1,765,
`rd_in_barrier` 1,040, `wr_issue` 844, `cb_gather_write` 406, `cp_gamma_mul` 396,
`rd_in_issue` 219, `rd_gamma_issue` 217; the combine's own zones are `cp_combine` 2,414 /
`cb_gather_wait` 1,188 / `mcast_src_wait` 672 / `mcast_send` 476, once per dispatch on the root.
Marker check clean (10–28 markers per RISC).

The loudest new signal: **`cp_rms_chain` runs on all 32 cores** (32 execs) at 3,448 ns each. The
combine broadcasts the summed sum-of-squares and then every core independently applies
`×1/W, +eps, rsqrt` to the same rows — 32× redundant SFPU work on what is now one of the two
largest compute stages. Broadcasting the already-reciprocal-rooted value instead, so only the root
runs the chain, is a round-2 candidate worth ~3.4 µs of a 9 µs kernel. Also carried forward:
`w_split`'s own `no_combine` ablation says ~3,000 ns of parallelism is still behind a cheaper
combine topology (tree or reduce-scatter, per `tensix_all_reduce`'s 4.64–6.48×); `prefill_1024`
gives up a measured 1.17× because the cost model prefers G=1 there (it is DRAM-bandwidth-bound, not
per-core-issue-bound, and no DRAM term is calibrated); the resident-x ladder for the shapes the
split cannot take; aliasing `cb_squared`/`cb_normed`, which are never simultaneously live (now
partly moot — `cb_squared` is gone); the bfloat8_b-gated bank-major inversion (−36%/−44%); and a
chunk-size cost model, since "coarsest divisor that fits" is measurably the wrong objective in a
band (~2.8 µs on the RM-gamma shape alone).

### Levers (`lever_ledger.json`; `python3 -m eval.verify_levers <ledger> --bench <bench> --report`)

`verify_levers` is **clean: 0 blocking**, 4 signal (all staleness flags on earlier phases' rows,
which is the expected consequence of a topology change and never blocks). **17 of 29 levers closed
with evidence** (was 13), by status `applied=11, deferred=11, measured-no-payoff=3, missed=1,
structurally-impossible=3`. Five rows were written back with both arms under phase `Perf 1`:

| lever | new status | on → off | what moved |
|---|---|---|---|
| **B12** multicast instead of N unicasts | `applied` (was `missed`) | 10,161 → 10,730 ns @ G=14 | applied on a *different operand* than phase 0 predicted — the combine's 1→N leg, not gamma. The gamma form's ceiling collapsed to ≤8.7% once the row-0 span landed |
| **C16** double-buffering | `applied` (was `measured-no-payoff`) | 21,559 → 25,233 ns @ `(1,1,32,4095)` | **refuted**: its two arms were the same program on a wide Regime-B shape, so the null had never been tested there. Worth 1.17–1.33× once the chunk search is subordinated to it |
| **D20** layout/regime-specialised plan selection | `applied` (was `deferred` 50%) | 118,095 → 155,792 ns @ `(1,1,8192,1024)` | the `force_regime` arm phase 0 asked for now exists as a shipped lever; plus a third selection axis (`_choose_group_size`) that pays far more than tuning the existing two |
| **B13** stateful set_state/with_state | `measured-no-payoff` (was `deferred` 5%) | 49,607 → 42,420 ns @ `(1,1,8192,1024)` | right about the mechanism, wrong about the sign. Its order-preserving half **is** applied |
| **B8** trid double-issue | `measured-no-payoff` (was `deferred` 8%) | 7,116 → 7,024 ns | measured on the bench gap phase 0 flagged; the surcharge exceeds the barrier it could hide |

`F27` (`math_fidelity`) stays `missed` — it remains the largest unexploited compute lever and it
remains forbidden: the precision contract is the caller's. `C15 / C14 / A2 / A3` stay `deferred` on
a sharded `memory_layout` a perf round cannot add. A new `force_regime` knob was added to
`LEVER_DEFAULTS` (default 0 = the solver decides) so D20's counterfactual is permanently
re-runnable rather than a one-off experiment, and the `stateful_noc` bench now carries B8/B13 arms
in the same `levers=dict(...)` idiom for the same reason.

Note the report flags B8 and B13 as "possibly unlocked" against their own phase. That is a checker
artifact: the derived kernel set now includes the experiment dirs' kernels, so every row looks like
its topology moved. The verdicts are this round's own and are not stale.

Regenerated table, pasted rather than retyped
(`python3 -m eval.verify_levers ttnn/ttnn/operations/rms_norm/lever_ledger.json --report`):

### Completeness ledger — `master.md` Part 2, all 29 levers

| lever | status | evidence | reason |
|---|---|---|---|
| **A0** | applied | on 93,535 / off 109,520 ns — lever saves 14.6% @ `[1, 1, 8192, 1024]`, knob `active_cores` | Core-count sweep on (1,1,8192,1024): full grid 93535 ns, 96 cores 95983, 64 cores 94470, 32 cores 110136, 16 cores 105999. No bandwidth knee below the full grid -> use every core; capping is a 1.17x regression. Re-checke |
| **A1** | measured-no-payoff | on 93,535 / off 92,968 ns — lever COSTS 0.6% @ `[1, 1, 8192, 1024]`, knob `row_wise` | split_work_to_cores(row_wise=True) is applied, but on a 130-core grid the op uses 128 cores, so row-wise and column-wise select almost the same set and the DRAM-facing spread is identical. Delta is inside noise on every  |
| **A2** | deferred | predicted 15% | Launch only on cores that hold data requires a sharded input; Phase 0 SUPPORTED[memory_layout] is INTERLEAVED only, so there are no shard-owning cores to launch on. |
| **A3** | deferred | predicted 10% | Interleaved pages are spread over every DRAM bank by the TensorAccessor, so a core cannot be made adjacent to 'its' bank; only the A1 axis spread is expressible. A3 becomes real once the input is sharded (each core then  |
| **A4** | deferred | predicted 2% | The remainder is handled by split_work_to_cores' two core groups and by padding the final row-block to a full BLOCK_HT (phantom tile-rows clamped to the last valid one), so no core is idle - but there is no specialised c |
| **B0** | applied | on 3,257 / off 3,267 ns — lever saves 0.3% @ `[32, 17]`, knob `barrier_per_block` | Every per-core-overhead lever's counterfactual was measured on the smallest regime the op accepts, (32,17) = 1 tile of real work: A0 -1.1%, A1 -1.4%, B5 -1.7%, B7 +0.3%, B9 -0.2%, C16 -1.2%, compute_block_size -1.6%. Not |
| **B5** | applied | on 3,257 / off 3,201 ns — lever COSTS 1.7% @ `[32, 17]`, knob `coalesce` | Every transfer is a whole tile page (noc_async_read_tile / noc_async_write_tile); the RM path moves whole padded sticks. Splitting each 2048 B tile into two 1024 B halves costs +9.1% on (1,1,8192,1024) and +10.1% on (1,1 |
| **B6** | applied | on 71,273 / off 80,440 ns — lever saves 11.4% @ `[1, 1, 8192, 1024]`, knob `coalesce` | UNLOCKED by refinement-1 and re-measured. Phase 0 closed B6 measured-no-payoff because, at a 2048 B bfloat16 tile, splitting the page bought nothing beyond what B5 already measured. Adding bfloat8_b moved that premise: a |
| **B7** | applied | on 3,257 / off 3,267 ns — lever saves 0.3% @ `[32, 17]`, knob `barrier_per_block` | One noc_async_read_barrier / noc_async_write_barrier per BLOCK (BLOCK_HT x chunk_wt tiles, or one tile-row of sticks on the RM path) instead of per transaction. Largest single dataflow lever on this op: 1.33x on (1,1,819 |
| **B8** | measured-no-payoff | on 7,116 / off 7,024 ns — lever COSTS 1.3% @ `[1, 1, 32, 7168]`, knob `sn_reader_variant` | MEASURED in Perf 1 and refuted, on the bench gap phase-0 flagged (blocks-per-core > 1 now exists: the focus shape issues 4 read groups per dispatch and wide_prefill 2 row-blocks per core). `..._with_trid` costs +5.5 ns/t |
| **B9** | applied | on 93,535 / off 137,506 ns — lever saves 32.0% @ `[1, 1, 8192, 1024]`, knob `noc_split` | Reader on NoC0 (ReaderConfigDescriptor) and writer on NoC1 (WriterConfigDescriptor). This op reads and writes a full tensor, so the two streams are the same size; forcing both onto NOC_0 costs 1.47x on (1,1,8192,1024) an |
| **B10** | deferred | predicted 3% | Per-reader virtual-channel assignment is only meaningful once several readers share a route; with 128 cores each reading interleaved pages from all banks the routes are already spread, and B9 already removed the read/wri |
| **B11** | structurally-impossible | pinned by `tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm.py` | Every DRAM address this op forms is alignment-legal by construction: tile transfers go through TensorAccessor's aligned page stride, and the two byte-offset paths (RM stick W-chunk, RM gamma slice) use offsets that are m |
| **B12** | applied | on 10,161 / off 10,730 ns — lever saves 5.3% @ `[1, 1, 32, 7168]`, knob `w_split` | APPLIED in Perf 1, though not in the form phase 0 predicted. Phase 0 filed B12 as 'read gamma once on an injector core and mcast it', sized at a 5.3% upper bound. Two Perf-1 measurements moved that: (a) the whole-page ga |
| **B13** | measured-no-payoff | on 49,607 / off 42,420 ns — lever COSTS 16.9% @ `[1, 1, 8192, 1024]`, knob `sn_reader_variant` | MEASURED in Perf 1, and the phase-0 prediction of 5% was right about the mechanism and wrong about the sign. Splitting the lever into its two halves is what made it legible. (a) The ACTUAL state-reuse mechanism REGRESSES |
| **C14** | deferred | predicted 25% | Zero-copy aliasing of a CB onto a shard buffer requires a sharded input; Phase 0 is interleaved-only. |
| **C15** | deferred | predicted 30% | Same gate as C14/A2: the op is DRAM-interleaved in Phase 0, so it cannot take the L1-resident path. The measured ablation says this is where the money is on the prefill shapes - stubbing the NoC payload on (1,1,8192,1024 |
| **C16** | applied | on 21,559 / off 25,233 ns — lever saves 14.6% @ `[1, 1, 32, 4095]`, knob `double_buffer` | REFUTED AND RE-CLOSED in Perf 1 with new arms, under this phase. The phase-0/refinement-1 verdict of measured-no-payoff was honest but was measured on the DRAM-saturated (1,1,8192,1024) shape, and on a wide Regime-B shap |
| **C17** | deferred | predicted 2% | rms_norm is not an in-place op on its public contract - it allocates a fresh output tensor. Writing into the input buffer would need a caller opt-in, and the input is still being read in pass B of Regime B. |
| **D18** | deferred | predicted 3% | Already in the applied FORM - every TensorAccessorArgs is emitted through get_compile_time_args() and read back with TensorAccessorArgs<N>() in the kernels, so address generation is unrolled per buffer type. What is miss |
| **D19** | deferred | predicted 2% | Already in the applied FORM - the only runtime args are buffer base addresses plus the per-core (start_row_block, num_row_blocks) pair and the two fp32 scalar bit patterns; every geometry constant is a compile-time arg.  |
| **D20** | applied | on 118,095 / off 155,792 ns — lever saves 24.2% @ `[1, 1, 8192, 1024]`, knob `force_regime` | APPLIED in Perf 1. Phase 0 recorded D20 as 'already in the applied FORM' (two structurally different plans behind a pinned host predicate, plus a compile-time TILE/ROW_MAJOR fork) with the follow-up 'add a force_regime a |
| **D21** | deferred | predicted 1% | Per-core indexing IS precomputed host-side (start_row_block / num_row_blocks per core); the kernels only add. The InterleavedAddrGenFast half does not apply - TensorAccessor is mandatory and already specialises its addre |
| **E22** | deferred | predicted 0% | Metal Trace + multiple command queues is a whole-model lever; this is a single-op deliverable and the measurement here is on-device kernel duration, which host-dispatch overlap does not change. |
| **F23** | structurally-impossible | pinned by `tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm.py::test_rms_norm_maxed_out_compute_config` | RE-CLOSED under the refinement-1 topology, with the wording tightened because this phase added the op's only config-touching code. For every REAL call the caller's ComputeConfigDescriptor is still handed to the compute K |
| **F24** | applied | on 71,113 / off 71,135 ns — lever saves 0.0% @ `[1, 1, 8192, 1024]`, knob `pack_precise` | RE-CLOSED under the refinement-1 topology. The Phase-0 closure was structurally-impossible on the premise that SUPPORTED[dtype] held no block format, so no bfloat8_b pack existed for the knob to govern; refinement 1 adde |
| **F25** | applied | on 71,796 / off 76,080 ns — lever saves 5.6% @ `[1, 1, 32, 7168]`, knob `dest_acc` | Refinement 1 added False to SUPPORTED[fp32_dest_acc_en], which is what Phase 0 deferred F25 on. Attributed ALONE (fidelity held at the default HiFi4, config `dest_off`), the cheap DEST width is 1.06x on the compute-bound |
| **F26** | structurally-impossible | pinned by `tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm.py::test_rms_norm_maxed_out_compute_config` | RE-CLOSED under the refinement-1 topology (which added bfloat8_b, a second reduce datapath and two derived CB formats - none of which creates a lossless-unpack opportunity). A lossless unpack buys nothing downstream of a |
| **F27** | missed | on 47,995 / off 71,796 ns — lever saves 33.2% @ `[1, 1, 32, 7168]` | NOT changed in Phase 0 or Refinement 1, and still MISSED rather than closed. references/precision_convention.md and the op prompt pin the exported default_compute_kernel_config() to HiFi4, and F23 forbids downgrading a c |

**End state:** 17 of 29 levers closed with evidence (12 open). Generated from `rms_norm/lever_ledger.json` by `eval.verify_levers --report`.

### Ranked remaining opportunities

| # | lever | status | predicted | regime / follow-up |
|---|---|---|---|---|
| 1 | **C15** | deferred | 30% | Lamp L3 (HEIGHT_SHARDED) then Lamp L1 (WIDTH/BLOCK_SHARDED with the cross-core combine). |
| 2 | **C14** | deferred | 25% | Lamp L3: ttnn.cb_descriptor_from_sharded_tensor(CB_INPUT_TILES, input_tensor) once HEIGHT_SHARDED is in SUPPORTED - no NoC read at all for the resident rows. |
| 3 | **A2** | deferred | 15% | Lamp L3: add HEIGHT_SHARDED, back cb_input_tiles with ttnn.cb_descriptor_from_sharded_tensor and set CORE_GRID from the shard grid. |
| 4 | **A3** | deferred | 10% | With Lamp L3 sharding, map each shard to its NoC-optimal worker (get_optimal_worker_cores_for_sharded_tensor). |
| 5 | **B10** | deferred | 3% | Add a `vc` arm on the reader's noc_async_read and measure on (1,1,8192,1024) after Lamp L2 removes the gamma re-reads (which currently hide small routing effects). |
| 6 | **D18** | deferred | 3% | Add a `runtime_accessor_args` arm that passes the accessor args as runtime args and measure on (1,1,8192,1024) and (32,17). |
| 7 | **A4** | deferred | 2% | Emit a second kernel variant for the cliff core with its own BLOCK_HT; measure on a shape whose Rt is coprime with the grid (e.g. (1,1,3232,96), Rt=101). |
| 8 | **C17** | deferred | 2% | Add an `output_tensor=` parameter so a caller can alias, and gate it on Regime A (single-read) where the input is fully consumed before the scale pass writes. |
| 9 | **D19** | deferred | 2% | Add an arm that demotes the geometry prefix to runtime args and measure the program-cache re-dispatch cost. |
| 10 | **D21** | deferred | 1% | Add an arm that computes the per-core row range in-kernel from a core index, to price the host-side precompute. |
| 11 | **E22** | deferred | 0% | Out of single-op scope; revisit when rms_norm is benched inside a model. |
| 12 | **F27** | missed | unsized | Compute-bound shapes where the CALLER leaves math_fidelity unset, so the op's own default applies. Measured ladder on (1,1,32,7168) (the ablation says that shape is compute-dominated: stubbing compute |

### Possibly unlocked — negative closures measured under an older topology

These were closed honestly, with evidence, and the op has changed since. Re-running the arm is one bench row; treat them as candidates alongside the open rows above.

| lever | status | closed in | what moved |
|---|---|---|---|
| **A1** | measured-no-payoff | refinement-1 | blocks_per_core {'grid_filling': 1, 'wide_prefill': 2, 'grid_starved': 1, 'smallest': 1, 'row_major': 1}->{'focus': 1, 'grid_filling': 1, 'wide_prefill': 2, 'grid_starved': 1, 'smallest': 1, 'row_major': 1}; kernels +rms |
| **B8** | measured-no-payoff | Perf 1 | kernels +rms_norm_compute.cpp, rms_norm_reader.cpp, rms_norm_writer.cpp, scale_compute.cpp, sn_reader.cpp, sn_writer.cpp, ws_compute.cpp, ws_reader.cpp, ws_writer.cpp |
| **B11** | structurally-impossible | refinement-1 | blocks_per_core {'grid_filling': 1, 'wide_prefill': 2, 'grid_starved': 1, 'smallest': 1, 'row_major': 1}->{'focus': 1, 'grid_filling': 1, 'wide_prefill': 2, 'grid_starved': 1, 'smallest': 1, 'row_major': 1}; kernels +rms |
| **B13** | measured-no-payoff | Perf 1 | kernels +rms_norm_compute.cpp, rms_norm_reader.cpp, rms_norm_writer.cpp, scale_compute.cpp, sn_reader.cpp, sn_writer.cpp, ws_compute.cpp, ws_reader.cpp, ws_writer.cpp |
| **F23** | structurally-impossible | refinement-1 | blocks_per_core {'grid_filling': 1, 'wide_prefill': 2, 'grid_starved': 1, 'smallest': 1, 'row_major': 1}->{'focus': 1, 'grid_filling': 1, 'wide_prefill': 2, 'grid_starved': 1, 'smallest': 1, 'row_major': 1}; kernels +rms |
| **F26** | structurally-impossible | refinement-1 | blocks_per_core {'grid_filling': 1, 'wide_prefill': 2, 'grid_starved': 1, 'smallest': 1, 'row_major': 1}->{'focus': 1, 'grid_filling': 1, 'wide_prefill': 2, 'grid_starved': 1, 'smallest': 1, 'row_major': 1}; kernels +rms |


### Helper bypasses

| helper | kind | what was missing / hard | helper ns | raw ns | site |
|---|---|---|---|---|---|
| `mcast_pipe` `SenderPipe`/`ReceiverPipe` — the **N→1 gather leg** | capability | `SenderPipe::send(src, dst, size)` (`mcast_pipe.hpp:189-197`) is a 1→N multicast of ONE source region to an `McastRect`. The gather is the opposite direction: N→1 unicast fan-in with each sender landing in a **different slot** of one core's CB. `ReceiverPipe::NUM_SENDERS` (`mcast_pipe.hpp:242, 255-261`) governs multi-sender *signalling*, not multi-source *data* fan-in. No constructor expresses it. A `GatherPipe` (N senders → indexed slots of one receiver's CB, one semaphore, `wait(N-1)`) would close it. | n/a — no helper expresses N→1 | 9,540 (whole op) | `kernels/rms_norm_reader.cpp:449-465` |
| none — a `read_bcast_row_tiles(acc, first_page, n, cb, TileFormat)` does not exist; the nearest available is `noc_async_read_tile` / any `kernel_lib` page reader | capability | Every dataflow reader in the library is page-granular: give it a page id, it moves `page_size` bytes. A `BroadcastDim::Row` operand needs 1/32 of that (bf16 544 of 2048 B), and expressing it requires the **face layout of a tile** — face 0 at offset 0, face 1 at `256*elem`, and for block formats the 64 B per-face-row exponent header — which no dataflow helper exposes. A helper owning that arithmetic (keyed on tile format, asserting on formats it does not model) would serve this site and every other Row/Col-broadcast weight-vector read in the library. | 118,112 | 93,542 | `kernels/rms_norm_reader.cpp:351` |
| `noc_async_read_page` / `noc_async_read_tile` | capability | Hard-codes `noc_async_read<NOC_MAX_BURST_SIZE + 1, false>` (`dataflow_api.h:1074`), so **no** page-id call site can reach the one-packet path even when the page size is a compile-time constant (here CT arg 17, `IN_TILE_BYTES`). `TensorAccessor` exposes only per-page `get_noc_addr`, and `PagesAddressIteratorInterleaved` is by its own docs "get_noc_addr for each page without complex optimizations" (`pages_address_iterator.h:267`) — nothing expresses "issue this page set". Fix: a `max_page_size` template parameter on `noc_async_read_page`, or a `pages_affine(acc, first_page, n)` iterator owning both the one-packet dispatch and the constant advance. | `rd_in_issue` 5,763 | 4,904 | `kernels/rms_norm_reader.cpp:452` |
| `noc_async_write_page` / `noc_async_write_tile` | capability | Same gap, write side: hard-codes `noc_async_write<NOC_MAX_BURST_SIZE + 1, false, posted>` (`dataflow_api.h:1258`). | `wr_issue` 498,407 | 485,032 | `kernels/rms_norm_writer.cpp:160` |
| `noc_async_read` / `noc_async_write` default bound (the ROW_MAJOR stick form) | capability | The default `max_page_size = NOC_MAX_BURST_SIZE + 1` forces the any-length loop, and no helper derives the bound from the op's chunk geometry — the op has to compute `RM_CHUNK_MAX_BYTES` itself from `Wt_core` / `WT_*_BLOCK` / `ELEM_SIZE`. | — | — | `kernels/rms_norm_reader.cpp:496`, `kernels/rms_norm_writer.cpp:203` |
| `ckl::sum_of_squares` → its `eltwise_chain(BinaryFpu<Mul,…>, PackTile<…>)` expansion | capability | The non-tile-aligned last chunk must walk a **gapped** column window ("all columns but the last", then "the last"), which needs a caller-supplied `StridedTileRange` per operand. `sum_of_squares` takes no runtime element arguments and has no overload accepting a stride, so for that one window there is no helper form at all. The output spec is spelled out byte-for-byte as `row_output(cb_sumsq_acc)` expands (`PerOuter`/`PerOuter` + `DestAccumulation::PerRow`, `eltwise/api/convenience.inl`). Tile-aligned chunks still call the helper. A `sum_of_squares(shape, StridedTileRange)` overload would return this site to one helper call. | 20,261 / 34,069 / 15,386 / 33,400 / 19,487 | 20,270 / 34,156 / 15,433 / 33,559 / 19,419 | `kernels/rms_norm_compute.cpp:209` (used at `:296`, `:301`) |

The last row's ns pair is **effectively equal** (±1% on all five columns, measured with an arm that
routes the tile-aligned chunk through the raw expansion too). That is reported honestly: the cost
being claimed there is author and maintenance cost, not device time. It is a capability gap in
*kind* — the masked window has no helper form at all — while being free in *cost*.

Every bypass carries its justification in the kernel head at the site, so a later helper-usage
verifier pass has the reasoning in front of it and cannot "fix" the win back.

### Two escaped findings, forwarded rather than buried

1. **`DestReuseBinary` is WRONG at `block_size == DEST_AUTO_LIMIT` on 16-bit DEST.** At
   `fp32_dest_acc_en=False` (limit 8), `block_size=8` corrupts **rows 16–31 of the LAST tile of
   every DEST block** — PCC 0.9834 / max_rel 5.46 against 0.99998 at `block_size ≤ 7`, reproduced
   at `wt=8` and `wt=16` (tiles 7 and 15). At fp32 DEST (limit 4) `block == limit` is clean.
   `chain_max_block_v`'s clamp reserves no headroom for the DEST→src path and there is **no
   `static_assert`**, so this is a silent wrong answer for any future reuse user; the existing
   coverage (`tests/ttnn/unit_tests/kernel_lib/test_chain_elements.py::test_dest_reuse_matrix`)
   only runs `block_size=1`. Found while measuring idea 6, which did not graduate — so nothing in
   this op depends on it, and it would have been lost with the experiment.
2. **`op_design.md`'s Lamp L4 sketch is stale in two ways**: `ckl::unary_bcast<BroadcastDim::Row/Col>`
   materialises a broadcast tile directly from the row-0/col-0 tile, so the `cb_ones` CB the sketch
   names as a blocker does not exist; and the sketch picks the *expensive* operand to pre-broadcast
   (`1/rms` costs `BLOCK_HT` tiles per row-block against gamma's `Wt_core`). Both are moot now that
   L4 is measured a regression, but the doc should not keep asserting a 1.5× that has been refuted.
   The design doc also still describes the Phase-0 "one core owns a full row" work distribution and
   `cb_gamma_tiles` as "resident, depth 1, never popped"; both are superseded by this round.

### Summary

All 7 ideas measured; **5 graduated, 1 null-turned-diagnosis-correction, 1 measured regression,
1 win superseded by a bigger win**. The focus case is **4.88× faster** (43,947 → 9,010 ns, against
a 14,894 ns gate), the op's largest absolute number is **2.04× faster** (1,229,676 → 603,971 ns),
the DRAM-bound prefill shape nobody expected to move is **1.25× faster**, and the masked
non-aligned shape is **1.27× faster**. No regression on any supported cell, golden and the
precision matrix are an exact match to the pre-round baseline, and the sum-of-squares row-scale
bias is now flat in the reduced width instead of growing with it.
