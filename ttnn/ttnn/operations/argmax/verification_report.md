# Verification Report: argmax

Target `quasar` on the ttsim simulator (untimed). Specification `eval/specs/argmax/spec.py` (sha256 `fcff36f2…`). Phase 0 as built by the implementer at `d317dc52cda`.

## Code Review

**Fixed**
- **DRY: cache-line size computed three times.** `max(ttnn.get_l1_alignment(), DM_CACHE_LINE_BYTES)` appeared in `derive_blocks()` (slot spacing and the cap), in the kernel's `line_bytes` CT arg, and in the per-scratchpad pad given to `ScratchpadSpec`. The three must agree, or the cap's pad accounting and the kernel's base alignment drift apart. They now come from one function, `line_bytes()`, carried in `Blocks.line_bytes` to the CT arg and the pads (`argmax_program_artifacts.py`).
- **L1 ledger out of date** (see *L1 Ledger Audit*): brought in line with the allocation.

**Reviewed, no change needed**
- **Specification rules.** R1: the key maps `-0.0 → +0.0`, the scan replaces the best only on strictly greater, and the merge walks segments in column order with strict `>`. The `ties` and `signed_zeros` cases pass. R2: the output is uint32 ROW_MAJOR. R3: the full bf16 word is compared, with no narrowing. R4: `validate()` canonicalises `dim` before it looks at SUPPORTED, and `dim=3` passes (`test_positive_dim_alias`). R5: `validate()` never reads `sub_core_grids`, and the sub-grid tests show equal results on 1×1 and 4×2 sub-grids in both regimes. R6/X2: `output_tensor` that is not None raises `ValueError`. R7: only DRAM is supported so far; L1 is Refinement 2. R8: the output is allocated with `memory_config=input.memory_config()`. R9: `memory_axes()` derives the placement axes. R10: refusals are `UnsupportedAxisValue`, a `NotImplementedError`. X1: a `dim` out of range raises `IndexError`.
- **One dispatch.** Split, scan, gather, merge and store all run in one `ttnn.generic_op`. The host only validates and allocates.
- **Thread plan.** Read from the host program: one DM `KernelSpec` with `num_threads = SCAN_THREADS = 6` (one host constant, passed as CT arg `scan_threads`), and no compute kernel. This is (R, C, W) = (6, 0, 0), every user DM hart. The design states the bound as the scan on the DM harts (a load, a key and a compare per element). It also states why the NEOs cannot help: Quasar's LLK has no index-carrying max (NEO-SCAN, `deferred`). That is a reasoned plan, not a default of 1. The plan's tests are present and pass: every thread ran (a scanner-id probe in ROWS and SEGMENTED, all thread ids 0..5 seen); after another program; before tt-metal's `to_layout`, in one session; and both regimes pinned on sub-grids. The thread plan cannot be measured against alternatives here: ttsim has no timing model, and `pipeline_logs/model_cycles.jsonl` does not exist for this run.
- **Design conformance.** The algorithm is a single-pass running (key, idx) scan, as designed; TWO-PASS and TWO-DISPATCH were rejected. Topology and regimes: ROWS and SEGMENTED, with the predicate exactly as in `select_regime()`. The work split uses `split_work_to_cores(row_wise=True)`, then a contiguous cut over the threads. Every multi-row PHASE0 shape engages all 192 scanners: `(4,8,32,256)`, `(1,1,2048,256)`, the model's `(1,1,2,151936)` (SEGMENTED, spr 96) and `(1,1,32,8192)`. The exception is the model's `(1,1,32,4096)`, with 128 of 192 (Refinement 3). The ≤ 32-row shapes engage one hart per row. The gather is a raw NoC write plus a remote `Semaphore::up` to one finalizer, which does `wait(U)` (an exact count; U increments arrive) and then `set(0)` for re-runs (`test_repeated_calls_reuse_the_semaphore`). The multicast helper does not apply: this is an N→1 gather of distinct records, and a `Scratchpad` cannot be a multicast destination.
- **Blocking-model fidelity.**
  - Knobs: `SCAN_THREADS`, `IN_DEPTH`, `OUT_DEPTH`, `MIN_SEGMENT_BYTES`, `L1_RESERVE_BYTES` and `RECORD_BYTES` are host constants. `block_bytes`, `block_rows`, `block_cols`, `row_pitch` and `in_depth` come from one function, `derive_blocks()`, and reach the kernel as CT/RT args. The kernel's loops are sized from them. No collapsed knob was found.
  - CB sizing: no CB scales unconditionally with an op dimension. The SEGMENTED staging slot scales with `tensor_rows`, which the predicate bounds at ≤ S_total/2.
  - Expression: `load_block` issues one NoC read per row piece, back to back, with one trid barrier per block. `scan_block` holds one cache invalidate per block. `emit_block_results` does one flush, and one write per output page the block touches, with one barrier before a slot is reused. No per-unit handshake inside a block.
  - The streaming path (`block_bytes_cap`) is exercised by `test_streaming_blocks` and `test_streaming_ties_split_across_blocks`: row-block streaming, column-block streaming, and SEGMENTED over 13 column blocks.
- **Correctness.**
  - DM cache: every NoC-filled slot is invalidated (`scoped_lock`) before it is read, and every CPU-written slot is flushed before a NoC write sources it. Per-thread CPU-written slots are line-spaced, so one thread's flush never writes back another thread's bytes.
  - Transaction ids: every transfer carries a user trid (0..4 < 8, asserted on the host). The last two statements hand back trid 0 for reads and for writes.
  - API usage: TensorAccessor, `kernel_main()`, `api/...` include paths.
  - No compute kernel, so the compute-helper and broadcast checks do not apply.

**Advisory (GUIDANCE)**: everything in GUIDANCE is followed: comparison only, the scan order, wide rows in pieces with the running pair carried, and the output allocated through a `TensorSpec` with the input's memory config.

## Registry Conformance
- `INPUT_TAGGERS = AXES`, imported from the specification. The op file defines no axis function.
- `SUPPORTED` equals `PHASE0` on every axis (`validate_contract --phase0`: 0 errors).
- `EXCLUSIONS = []`.
- `validate()` order: X2, then X1, then canonicalise `dim`, then `memory_axes`, then the taggers, then the SUPPORTED per-axis check, then EXCLUSIONS. After those come the uint32 and empty-input mechanism caps.
- The entry point calls `validate()` first.
- The op file declares no `INVALID`.
- No SUPPORTED auto-fix was needed: 0 xpass_drift.
- INVALID audit (specification): `INVALID = []`. Every TARGET cell is buildable. There are no bf8b/bf4b dtypes in TARGET, so the bf8b+ROW_MAJOR canonical entry has nothing to apply to (and Quasar has no bfloat8_b path). There are no weight axes.

## L1 Ledger Audit
- **Ledger currency: fixed.** The implementer's "Implementation deltas" (cache-line spacing) had not been folded into the table's size expressions, the `cap` (259,632 B in the ledger, 259,584 B allocated), the `sp_partials` size (3,168 → 3,456 B) or the worked totals. The table rows, symbol table (a `line` symbol added) and footprint now state what `derive_blocks()` and `ScratchpadSpec` allocate. Worked totals: 23,616 B for `(1,1,2,151936)`, 22,848 B for `(4,8,32,256)`, 38,208 B for `(1,1,2048,256)`, and 3,897,408 B ≤ 3,897,792 B at the cap.
- **Capacity vs live set.** In SEGMENTED, `sp_out_stage` allocates `T × OUT_DEPTH` slots and only the finalizer uses one (≤ 768 B). `sp_partials` is allocated in ROWS too (3,456 B). Both carry a written justification (a uniform layout costing < 5 KB), so neither needs a change. Nothing spans an axis without scaling with it.
- **Page format vs DEST.** Not applicable: no compute kernel and no DEST; the scratchpads carry no format.
- **Disjoint lifetimes.** Every pair has a reason (overlapping lifetimes: records land during the finalizer's own scan, and index writes overlap the next block's scan).
- **Block-size defaults.** Held: interleaved work spread over the whole grid (every user DM hart), and the coarsest block that fits (`block_bytes = min(need, cap)`; every PHASE0 assignment is one block). No departure, so no measurement was needed.
- **Data-movement budget.** Present and consistent with the code. The input crosses DRAM once and the output once. SEGMENTED adds 16·U B of NoC records and U atomics (≤ 3 KB). The cheapest-traffic split (ROWS / INTRA-CLUSTER-W) is implemented wherever `spr < 2`. SEGMENTED is an occupancy choice with a positive reason (INTRA-CLUSTER-W caps the 2-row model call at 12 of 192 scanners). BALANCED-RANGES stays `deferred`, with a reachability reason; it is now Refinement 4, because rank 2 brings in `(128, 8192)`.
- **Per-core footprint:** `T·in_depth·block_bytes + T·OUT_DEPTH·align_up(out_stage_rows·4, line) + align_up(S_total·16, line) + T·line + 3·line`. The first term scales with `block_bytes` (block_rows × block_cols) and `in_depth`, the second with `block_rows` (ROWS) or `tensor_rows` (SEGMENTED, ≤ 96), and the third with the grid only.

## Precision Baseline

`tests/ttnn/unit_tests/operations/argmax/test_argmax_precision_baseline.py`, bf16 ROW_MAJOR DRAM, standard-normal inputs. PCC via `comp_pcc`, abs errors via `comp_allclose`; relative RMS = RMS(got − true) / std(true), as `check_output` defines it. Figures are for the indices.

| Shape | Regime | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | Exact rows | Value err (bf16 ULP) | got/true ratio p5 / median / p95 |
|-------|--------|-----|-------------|--------------|------------------|-----------|----------------------|------------------------------|
| (1,1,32,64) | ROWS | 1.0 | 0 | 0 | 0 | 32/32 | 0 | 1.0 / 1.0 / 1.0 |
| (4,8,32,256) | ROWS | 1.0 | 0 | 0 | 0 | 1024/1024 | 0 | 1.0 / 1.0 / 1.0 |
| (1,1,32,4096) | SEGMENTED (spr 4) | 1.0 | 0 | 0 | 0 | 32/32 | 0 | 1.0 / 1.0 / 1.0 |
| (1,1,2,151936) | SEGMENTED (spr 96) | 1.0 | 0 | 0 | 0 | 2/2 | 0 | 1.0 / 1.0 / 1.0 |

**Assessment**: exact on every shape and every regime, and on the golden `ties` and `signed_zeros` cases. With no rounding there is nothing for the scale-vs-precision triage to separate (the ratio sits at exactly 1.0).
**Recommended tolerances**: exact, as `NUMERIC_RESOLVED` resolves both keys (`bfloat16->uint32`, `float32->uint32`: metric `exact`, ties `lowest_index`).

## Verifier CLI Summary
Results: `pipeline_logs/golden_phase0/verifier_report.json` (`eval/eval_test_runner.sh`, 8 sim workers, forked).
- supported_pass: 15 (12 PHASE0 cells + 3 numeric regression cases)
- xfail_expected: 308
- invalid_skipped: 0 (`INVALID = []`)
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- no_axes_found: 3 (rejections X1 and X2 pass; the empty `LOOSE_CASES` parameter set is skipped)
- queue_gaps: 0 (`--queue op_requirements.md`)
- Acceptance and plan tests: 34/34 pass (`tests/ttnn/unit_tests/operations/argmax/`), including the new precision baseline.

## Recommendations
- **Queue order.** R1 (TILE, the hardest axis) comes first. R2 bundles the knob-turns: float32, L1, keepdim and rank, applied to both paths. R3 is perf: three T1 knobs, of which the segment floor helps the model's `(1,1,32,4096)` call. R4 is perf: the BALANCED-RANGES scheme change, which waits on R2's rank 2. No `LOOSE_CASES` entry is perf-flagged, so I picked the perf regions myself, using the structural counts from a probe of `select_regime` / `split_units` / `derive_blocks` on the 32-cluster grid:
  - `(1,1,32,4096)` engages 128 of 192 scanners, and the most elements any scanner reads is 1024 against 683 if balanced;
  - `(128,8192)` engages 128 of 192, at 8192 against 5,462;
  - every PHASE0 scanner walks 1 block at `in_depth = 1`.
- **TILE occupancy trap** (R1). A unit of one tile-row strands scanners: 32 of 192 on `(4,8,32,256)`. Padded rows of `(1,1,2,151936)` should not cross DRAM, since 30 of the 32 rows are padding. Both points are in R1's notes.
- **Untested on hardware.** In ROWS, neighbouring scanners write disjoint 4-B ranges of the same output page, e.g. 6 writers into one 128-B page for `(4,8,32,256)`. NoC writes are byte-granular on Quasar and ttsim gets them right; whether DRAM merges concurrent partial-word writes correctly on silicon is unverified. If silicon loses one, the fix is a split-only change (op_design.md, Key Risks): deal ROWS ranges in multiples of `dram_alignment/4` rows. It needs other hardware, so it is a finding here, not a refinement.
- **Small-shape dealing** (perf lamp "Dealing"). For ≤ 32 rows only thread 0 of each cluster works, and the shapes are tiny (≤ 128 elements per row), so there is nothing worth queueing.
- **NaN.** Unspecified (`special_values` UNRESOLVED). With the current key, a positive NaN has the largest key and wins, as in torch; a negative NaN gets the smallest key and never wins, unlike torch. If the specification later adopts torch's policy, the key needs a NaN branch.
