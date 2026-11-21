# Verification Report: scatter

Target: Quasar (`quasar`) on the ttsim simulator (functional, no clock). Specification: `eval/specs/scatter/spec.py` (sha256 b563e355fde6). Regime built: `resident_line_block` (op_design.md).

## Code Review

Fixed in place:
- **Stage-depth knob reached only the host.** `SRC_STAGE_DEPTH` sized the `scatter_src_rows` / `scatter_index_rows` scratchpads, but the kernel never received it. Setting it to 2 would have allocated a second stage slot that nothing uses, and nothing would have said so. It is now a CT arg (`src_stage_depth`), and the kernel `static_assert`s it is 1 until the two-slot ping-pong exists (Refinement 5 names it). The host constant stays the single source.
- **`Geometry.index_page_bytes` was computed and never read.** The kernel addresses the index as one page per source row of `W` INT32 positions (page id = row, offset `face_col·64`). `create_program_artifacts` now checks exactly that and raises `ValueError` otherwise, so a future layout or regime that breaks the assumption fails on the host, not in the NoC.

Reviewed, no change needed:
- **Kernel API.** The kernel uses `void kernel_main()`, `api/...` include paths, `TensorAccessor` throughout, Metal 2.0 `get_arg(args::…)`, `Scratchpad<T>` with `scoped_lock`, and `get_my_thread_id()` / `sync_threads()`. It has no DFB, so no push/wait counts to balance. Every NoC transfer is `TXN_ID` trid 0, and the kernel hands trid 0 back as its last two statements.
- **`sync_threads` safety.** Every thread runs the same block and chunk trip counts: they come from the cluster's runtime args and the CT knobs, never from the thread's own share. A thread with no rows still reaches every barrier.
- **R6.** The bound is `(uint32_t)p < H` with the real `H`. Negative INT32 positions read as large unsigned values and are skipped. Tested by `test_positions_outside_the_line_are_skipped`.
- **Helpers.** None applies. The design's API Mapping records why the dataflow and eltwise helpers cannot express a destination row that comes from data. The implementer's kernel-local block operations follow the `tile_column_dataflow.hpp` offset formula. Within the scope boundary there is nothing to flag.
- **Cache protocol.** It matches the design's table. The cache line is confirmed 64 B (`risc_common.h`), so `line_start()` keeps each scratchpad region on lines of its own. `Scratchpad::operator[]` bounds-checks only under `ASSERT` (debug), so the release inner loop carries no check.

Nothing deferred as an architectural issue. The perf observations below are queued as measured refinements.

## Specification Rules

| Rule | Applies at Phase 0 | Status |
|---|---|---|
| R1 move unchanged | yes (bf16) | bitwise exact everywhere measured (precision baseline, model shape probe); `Elem` is a raw `uint16_t`/`uint32_t` copy |
| R2 colliding sources | yes | `index_all_zeros` and `index_with_repeats` loose cases pass; whole 16-bit stores |
| R3 untouched elements keep `input`; new tensor | yes | output from `allocate_tensor_on_device(input.spec)`; slab starts as a copy of `input` |
| R4 `dim` read before support check | yes | `validate()` canonicalizes before tagging; INPUTS give 0 and -2 |
| R5 index INT32 RM, `src` = `input` dtype/layout | yes | `validate()` reads all three tensors' dtype/layout |
| R6 positions not checked, never out of bounds | yes | bound in kernel; test present |
| R7 output where `input` is | yes | `input.spec`'s memory config; HOOKS check passes on every cell |
| R8 placement axes per tensor | yes | `memory_axes` on all three, must agree, then gated |
| R9 `index_form` "other" refused by name | yes | `validate()`; rejections X3/X4 pass |
| R10 NotImplementedError outside SUPPORTED | yes | 223 xfail_expected, 0 wrong mode |

No rule with a `structure` part. The advice in `GUIDANCE` is all followed: columns are the units, a later source row stands within a line, the output starts from a copy of `input`, positions are bounded, and tensors in L1 are read through accessors.

The specification lint (`python3 -m eval.validate_contract scatter --phase0 --text`) passes with 0 errors. It lists 6 UNRESOLVED items, all in the specification (numeric list gap for float32, `governs`, special values, freedoms, PERFORMANCE). None blocks Phase 0. No specification defect.

## Registry Conformance

- `INPUT_TAGGERS = AXES`, imported from the specification. No op-local axis function.
- `SUPPORTED` equals `PHASE0` and lists every axis of `AXES`, plus dtype, layout, memory_config and sharding.
- `EXCLUSIONS = []`.
- `validate()` runs in this order: X2 → X1 → canonical `dim` → placements → taggers → R9 → R5 → SUPPORTED per axis → EXCLUSIONS → regime predicate. Every refusal is a `SupportRefusal` (`NotImplementedError`). `scatter()` calls `validate()` first.
- The op file declares no `INVALID`.
- No XPASS drift, so `SUPPORTED` needed no auto-fix.

**INVALID audit (specification).** `INVALID = []` is correct. TARGET has no block-float dtype, and Quasar has no bfloat8_b path anyway, so the canonical bf8b+ROW_MAJOR entry does not apply. The op has no weights, so there are no canonicalization cells. No cross-tensor entries.

## Design Conformance

- **Algorithm and topology.** The resident output line, the walk in ascending source order, and the face-column width split all match the design. The pipeline is one DM kernel, as the design specifies.
- **Work distribution.** `split_work_to_cores(row_wise=True)` over (image, face-column) units fills `min(32, units)` clusters: the traced call and the model shape engage 32 of 32. `[1024, 64]` engages 4, an under-fill the design accepts (4 face-columns; its "grid synchronization" lamp).
- **Both dataflow halves are batched.** Reads: all of a chunk's reads, then one barrier per thread. Writes: all of a block's writes, then one barrier per thread. The next block's chunk-0 reads are issued under the write drain.
- **Thread plan, read off the host program.** The kernel is `KernelSpec(num_threads=threads)` with `threads = SCATTER_THREADS = 6`, one named constant, passed as the CT arg `threads`. Every dealing loop strides by `kThreads` from `get_my_thread_id()`. There are no buffers between kernels and no ring, so no ring protocol applies.
  - The design gives a plan larger than one thread, (6, 0, 6), and states the op is bound by per-element work on the DM harts, with the reason. NEOs cannot express a data-dependent scatter.
  - The threads-ran test exists: `test_each_thread_moves_its_rows`, 6 threads × {1, 2} chunks. So does the one-session test: `test_after_another_program_and_before_to_layout` runs a tt-metal program before the op and `to_layout` after it. Both pass.
  - The plan alternatives were not measured. This backend has no timing model, and the phase has no `pipeline_logs/model_cycles.jsonl`. The comparison `SCATTER_THREADS` ∈ {1, 2, 4, 6} needs `ttsim-perf` or hardware.
- **Blocking-model fidelity.**
  - `block_width_faces` and `block_src_tile_rows` each come from `choose_blocking`, one CT arg each, and every scratchpad size, trip count and offset derives from them.
  - `block_out_tile_rows` is the whole line by regime: a predicate-guarded resident buffer (`resident_line_fits`, refused by name beyond it), not a collapsed knob.
  - Phase 0's block is the whole per-cluster assignment, the coarsest that fits, so no half-turned split.
  - Expression check: no per-unit acquire/sync/retire in any phase.
  - The one knob not wired to the kernel (stage depth) is fixed above.
- **Implementer departures** (recorded in the changelog, both advisory): the slab lock is one RAII scope with one extra `sync_threads` per block, and each scratchpad region starts on its own cache line. Neither changes traffic or the live set.

## L1 Ledger Audit

- **Ledger currency.** Three rows, one per declared scratchpad. Size expressions match `scratchpad_sizes()` (plus the 64 B per region that the ledger's note documents). No CB/DFB exists. Changed: the symbol table's `src_stage_depth` row now says it reaches the kernel as a CT arg.
- **Capacity vs live set.**
  - *Over:* `scratch_out_column` and `scratch_src_rows` allocate `wf//2 + 1` tile-columns. That is the worst case of a face range that starts on an odd face, so a block that starts on an even face carries one unused tile-column (for example 9 instead of 8 at `wf` = 16). Scratchpad size is per node and uniform across clusters, and the ledger states a mechanism reason, so this was accepted, not fixed. Computing the exact maximum over the split's blocks would save it, but only matters once `choose_blocking` engages, which no INPUTS shape reaches. It is noted for the L1 work.
  - *Under:* none. The slab spans `out_line` and `width`, and its capacity scales with both. The stages span the source chunk and `width`.
- **Page format vs DEST.** Not applicable: no compute, no DEST, raw byte payloads.
- **Disjoint lifetime.** None. All three scratchpads are live together during a chunk's scatter, and the ledger justifies each.
- **Bounds.** Every symbol is bounded in the ledger's symbol table, with its predicate. The total is closed-form.
- **Data-movement budget.** Present and consistent with the code: each tensor crosses DRAM once and there is no cross-core traffic. The cheapest-traffic split, width, is the one implemented. The only costlier split with a use (`banded_output_line`) is a `deferred` regime with a positive reason: it needs H > 60,832 rows on this target, and no shape reaches that.
- **Block-size defaults.** Both held: units spread across the full grid, then the coarsest block that fits (one block and one chunk per cluster on every INPUTS shape and on the model shape, 49.9 % of budget).
- **Per-cluster footprint.** `tensor_out_tile_rows·(wf//2+1)·tile_bytes + sr·(wf//2+1)·tile_bytes·depth + sr·32·wf·64·depth + 3·64`. The first term scales with the line (regime) and the width knob; the stage terms scale with the chunk and width knobs and with depth.

## Precision Baseline

`tests/ttnn/unit_tests/operations/scatter/test_scatter_precision_baseline.py` (bfloat16 TILE DRAM, distinct positions, seed 1234), plus a probe on the model shape:

| Shape (input / Hs, dim) | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | Max ULP | got/true ratio median (p5–p95) |
|---|---|---|---|---|---|---|
| (32, 64) / 32, dim 0 | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0 (1.0–1.0) |
| (70, 512) / 64, dim 0 (traced) | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0 (1.0–1.0) |
| (128, 512) / 17, dim -2 | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0 (1.0–1.0) |
| (1024, 64) / 512, dim 0 | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0 (1.0–1.0) |
| (64, 8192) / 32, dim -2 | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0 (1.0–1.0) |
| (2766, 2560) / 2752, dim 0 (model, probe) | — | — | — | — | 0 (0 of 7.08 M elements differ bitwise) | — |

**Assessment**: bitwise exact, as R1 requires of a pure move. No scale signature: the ratio is exactly 1.
**Recommended tolerances**: bfloat16 cells resolve to the `exact`-equivalent list entry (abs_rel 1e-5 / 1e-8). Float32 cells use `legacy_default` (PCC ≥ 0.9999, relative RMS ≤ 0, i.e. exact). Defaults hold.

## Verifier CLI Summary

Results: `ttnn/ttnn/operations/scatter/golden_results/verifier_report.json` (239 tests, 8 simulator workers).

- supported_pass: 12 (9 PHASE0 cells + 3 loose cases)
- xfail_expected: 223
- invalid_skipped: 0
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- no_axes_found: 4 (the rejection tests X1–X4, all passed)
- queue_gaps: 0 (every value of TARGET − SUPPORTED is named in `op_requirements.md`)

Local tests: 35 pass (4 acceptance, 26 plan/knob/session, 5 precision baseline).

## Structural counts (no clock on this backend)

Probe, bfloat16 TILE DRAM, `dim` 0. Both shapes bitwise exact:

| Count | `[70, 512]` / 64 (traced) | `[2766, 2560]` / 2752 (model) |
|---|---|---|
| clusters engaged | 32 | 32 |
| face-columns per cluster (`wf`), source tile-rows per chunk (`sr`) | 1, 2 | 5, 86 |
| blocks / chunks per cluster | 1 / 1 | 1 / 1 |
| scratchpads per cluster | 14.5 KB | 1.94 MB (49.9 % of budget) |
| 2 KB whole-tile vs 512 B face transfers per tile-row | 0 vs 64 | 64 vs 64 |
| index bytes per transaction, transactions per cluster | 64 B, 64 | 320 B, 2,752 |
| L2 line-maintenance writes per cluster per block (6 threads × acquire + release over slab + stages) | ≈ 2.7 K | ≈ 364 K (against ≈ 220 K element moves) |
| read / write barriers per thread per block; `sync_threads` per block | 1 / 1; 5 | 1 / 1; 5 |

## Recommendations

- **Queue.** The 4 generality values plus 2 perf refinements are ordered hardest first on the 2:1 cadence:
  - R1 `dim` -1 + w_non_aligned
  - R2 `dim` -3/-4 + rank 3/4
  - R3 perf
  - R4 ROW_MAJOR + float32 + L1
  - R5 perf

  No LOOSE_CASE is perf-flagged. The perf target is the real callers (traced call and model shape), already in Phase 0's SUPPORTED, so the anchor rule needs no unlock in R1/R2.
- **Largest structural cost on the model shape: the redundant L2 maintenance.** Every thread invalidates and flushes the whole slab and both stages. That is 6× the necessary line operations, and the stages are flushed even though they are read-only. Refinement 3 lever 1. Its correctness depends on the coherence semantics of an L2 range invalidate with respect to other harts' private L1 D$, which ttsim may not model. Establish it from the source before relying on a green test.
- **Unconfirmed protocol.** The DM cache protocol is correct by construction; no test on this backend shows it. Confirm on `ttsim-perf` or hardware before the op is called done there. That needs another backend, so it is a finding, not a refinement.
- **L1 headroom.** `resident_line_fits` holds to H ≈ 60.8 K rows here. The deferred `banded_output_line` regime is reachable by an outer band loop if a caller ever exceeds it. No INPUTS shape or traced call does.
- **Under-filled grid on narrow-W shapes.** `[1024, 64]` engages 4 clusters (4 face-columns). Splitting the source rows across clusters is the rejected regime, because it needs a cross-cluster merge. Engaging more threads per column is already done. No lever is in scope.
