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
