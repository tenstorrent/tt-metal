# Verification Report: g6_to_layout

Phase 0: TILE bfloat16 → ROW_MAJOR bfloat16, interleaved DRAM/L1 in and out, dataflow-only (reader + writer, no compute kernel). Two regimes: R1 `staged_reorder` (live face-row prefixes → tile window → 32-B self-aimed NoC copies → row block → one write per live row segment) and R2 `face_direct` (one tile column with `W ≤ 16`: faces 0/2 → one write of `2·W` B per row). Blackhole, 11×10 = 110-core compute grid.

Golden results: `generated/g6_verify/` (`verifier_report.json`, `test_results.json`).

## Code Review

State found: the implementation of the current plan (new `face_*` / `staged_*` kernels, rewritten descriptor and entry point, staged deletion of the old compute/reader/writer kernels) was in the working tree, uncommitted. It is reviewed as-is and committed with this pass.

Fixes applied:

| # | Finding | Fix |
|---|---------|-----|
| 1 | DRY: the R2 slot size `2 * FACE_BYTES` was restated as a local `slot_bytes` in both `g6_to_layout_face_reader.cpp` and `g6_to_layout_face_writer.cpp`. The reader and writer must agree on it, and two copies drift | Moved to one `LEFT_SLOT_BYTES` in `kernels/g6_to_layout_block_walk.hpp`; both R2 kernels use it |
| 2 | The `PIPELINE_BLOCKS_PER_CORE = 16` comment cited a sweep this pass could not reproduce, and the ledger did not record why the block departs from the coarsest-fit default | Re-measured {1 (L1 cap), 4, 16, 64} on 308 / 304 / 1978 (table in `l1_ledger.md`, § Block-size default). 16 holds: 5–6% faster than the L1-cap block on R1, within noise on R2. Comment and ledger updated to cite this sweep |

Nothing else needed fixing. Checked and clean:

- `void kernel_main()`, `api/dataflow/dataflow_api.h` includes, `TensorAccessor` everywhere (no `InterleavedAddrGen`).
- CB sync: R1 `cb_row_block` and R2 `cb_left_faces` push exactly one page per block in the reader and wait/pop one page per block in the writer, and both walk the same `BlockWalk` from `g6_to_layout_block_walk.hpp`. `cb_tile_window` is reader-private scratch (no push/pop).
- Trid discipline (R1 reader): DRAM batches on trids `1..tile_window_depth`, reorder copies alternate two more trids per tile (≤ 64 outstanding per id, well under the 255 counter). Every window slot is refilled only after its reorder trids were waited. `static_assert`s on both caps. The kernel resets the trid to 0 and issues a full read barrier at the end (BB-03).
- Writers flush (`noc_async_writes_flushed`) before every `cb_pop_front` and barrier once at kernel end (BB-03).
- No helper gap left open: the design's "helpers considered and rejected" table holds. `untilize` loses the sign of -0.0 and flushes subnormals through MOVA2D. `write_sticks_after_untilize` puts a write barrier on every tile-row and assumes pages are contiguous across image boundaries. The R1 reorder uses the `local_copy_helpers_dataflow` trid and stateful-read helpers.
- Program cache (BB-07): the key covers shape, input dtype/layout/memory layout/buffer, target layout, output dtype, output memory layout/buffer, and grid. Everything the descriptor depends on (geometry, regime, block extents, accessor page sizes, `out_l1_share`) is a function of those. A repeat call only rebinds the two buffer addresses (common RT arg 0). Repeats with different input data (the 308 loose cases) pass bit-exact.
- Plan report (BB-08/21): `select_plan` returns regime, block extents, depth knobs, CB bytes and per-core roles, and dispatches nothing.

Deferred to refinements: none from code review. The perf findings below go to Refinements 3–4.

### Prompt rules (`eval/prompts/g6_to_layout.txt` § Rules)

Every MUST rule whose condition applies to Phase 0 holds:

| Rule | Status |
|------|--------|
| One program; no intermediate device tensor, second DRAM pass or host staging (BB-01) | ✓ one `allocate_tensor_on_device` (the output) + one `generic_op` |
| bf16 output bit-identical for every pattern | ✓ dataflow-only, no value enters Tensix. Specials and poisoned padding are bit-exact in the golden loose cases, the acceptance tests and the precision baseline |
| Native RM / non-aligned handling; no `to_layout`/`tilize`/`untilize`/`pad`/`slice` | ✓ |
| Program built from this op's own kernels | ✓ four kernels under `kernels/` |
| Output never depends on input tile padding | ✓ padding faces/rows never read; segments end at `2·W`; poison 1000.0 / NaN tests pass |
| No OOM / refusal / EXCLUSION / size tag on golden shapes | ✓ 0 OOM, EXCLUSIONS empty, no size tagger |
| Core count is the blocking decision's | ✓ `min(110, tensor_tiles)` cores via `split_work_to_cores(row_wise=True)` |
| L1 output pages + op buffers ≤ 1347776 B/core | ✓ `max_block_tiles` subtracts `out_l1_share`; `test_l1_output_within_model_budget` guards it |
| BB-03 / 05 / 06 / 07 / 08 / 17 / 21 / 22 | ✓ (BB-17: one kernel group on `all_cores`, ≤ 2 ranges; BB-22: all tests and probes use production shapes) |

No soft (`prefer`/`consider`) rules exist in this prompt.

## Registry Conformance

- `INPUT_TAGGERS` (dtype, layout, output_dtype, memory_layout, alignment, rank): all `(inputs, axes)` signatures, projected from the scenario dict exactly as `feature_spec.py` / `axes.py` build it.
- `SUPPORTED` covers every tagged axis. `EXCLUSIONS = []`.
- `validate()` order: argument `ValueError`s, then SUPPORTED (`UnsupportedAxisValue`), then EXCLUSIONS (`ExcludedCell`), then the placement `ValueError`s. The spec requires the sharded-`memory_config` checks to come after the memory_layout refusal. `validate()` is the first line of `g6_to_layout`. Every message contains `g6_to_layout`.
- The op file does **not** declare `INVALID`.
- No SUPPORTED auto-fix was needed (no XPASS).

**INVALID audit** (`eval/golden_tests/g6_to_layout/feature_spec.py`):

1. `{"dtype": bfloat8_b, "layout": ROW_MAJOR}` is the canonical bf8b + RM entry for the activation. It is a single-tensor coupling and structurally impossible. ✓
2. `{"layout": TILE, "output_dtype": bfloat8_b}` couples the input layout with the output dtype, i.e. axes of different tensors. It is still sound: the output layout is by definition the other layout, so this says a ROW_MAJOR output cannot be bf8b, the same block-format impossibility as entry 1. `feature_spec.py` documents it as an op-definition coupling. ✓ No change requested.

Not a norm-like op (no weight canonicalization). `INVALID_FOR_ARCH` is not declared. Every production shard grid (up to x 0..10, y 0..7) fits the 11×10 grid, so every cell is buildable on this board.

## L1 Ledger Audit

- **Currency**: three CBs declared (idx 0 `cb_tile_window`, 1 `cb_row_block`, 2 `cb_left_faces`), three rows. The size expressions match `compute_plan`/`_block_bytes`, and the golden run's `device_l1_peak_bytes` agrees with the ledger's maxima: 688128 B (case 1978, R1) and 153600 B (case 308, R2). ✓
- **Capacity vs live set**:
  - `cb_row_block` spans both block axes, with capacity scaling in `block_rows_tiles·block_cols_tiles`. It is double-buffered by decision. ✓
  - `cb_left_faces` spans `tile_row`; `tile_col` is pinned to 1 by the R2 predicate. ✓
  - `cb_tile_window` is a fixed 32 KB streaming window (depth × batch), never op-scaled. The ledger states that it is mostly unused for `H ≤ 16` and keeps it for the uniform read path. ✓ No over/under finding.
- **Page format vs DEST**: no DEST; Float16_b pages are a dataflow-only payload. n/a.
- **Disjoint lifetime**: none (R1 and R2 are different programs; within each, CBs are concurrently live by the double-buffering decision). ✓
- **Bounds / closed form**: every non-block symbol (`out_l1_share`, `tile_row_stage_rows`, `core_tiles_max`) has a bound and a predicate. The totals are closed-form. ✓
- **Block-size default**: interleaved, so step 1 holds (full grid). Step 2 departs: `block_rows_tiles` is sized by `pipeline_blocks_per_core = 16`, not the L1 cap. That departure is now **measured** (added to `l1_ledger.md`):

  | Case | L1-cap block | 16 (shipped) |
  |------|-------------|--------------|
  | 304 | 3082 µs | 2941 µs |
  | 1978 | 1906 µs | 1784 µs |
  | 308 | — | within its own 4% run-to-run spread of every setting |

  The coarsest block did not fail to fit; the finer block wins on fill/drain.
- **Data-movement budget**: present and consistent with S1, the flattened tile-range split that the code implements. Every split moves identical bytes, so no cheaper-traffic split is deferred. The one byte-for-transactions trade, `same_bank_slot_writes`, is a `deferred` regime row. Its stated reason ("more DRAM bytes, 10.7×") counts logical bytes. DRAM is burst-granular, and the measured cost of R2's 6-B writes (below) shows the transaction count, not the bytes, is what binds. Refinement 3 takes this trade.
- **Per-core footprint**:
  - R1: `2048·tile_window_depth·read_batch_tiles + row_depth·block_rows_tiles·tile_row_stage_rows·block_cols_tiles·64 + out_l1_share`. The window term scales with the depth knobs only, the row term with both block extents and `row_depth`.
  - R2: `row_depth·block_rows_tiles·1024 + out_l1_share`.

Disposition: no ledger finding needs a refinement. The one cheap fix (recording the block-size measurement) is done in place. Refinement 3 replaces `cb_left_faces`' consumer and must update the ledger row.

## Precision Baseline

`tests/ttnn/unit_tests/operations/g6_to_layout/test_g6_to_layout_precision_baseline.py`: randn (seed 1234) and a 1%-specials variant (-0.0, 0x0001, 0x8001, 0x007F, ±inf, NaN 0x7FC1). It uses production shapes covering both regimes. PCC is from `comp_pcc`, abs errors from `comp_allclose` plus the explicit metrics, and the got/true ratio is taken over non-zero finite references.

| Shape (case, regime) | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | Max ULP | Ratio median / p5–p95 | Specials: injected / bit mismatches |
|---|---|---|---|---|---|---|---|
| (1,185) c309, R1 | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0 / (1.0, 1.0) | 7 / 0 |
| (16,1,2560) c1993, R1 | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0 / (1.0, 1.0) | 409 / 0 |
| (1,1024,1024,3) c286, R2 | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0 / (1.0, 1.0) | 31457 / 0 |
| (8,1024,2560) c2020, R1 | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0 / (1.0, 1.0) | 209715 / 0 |

**Assessment**: bit-exact. No value enters the Tensix coprocessor, so there is nothing to round. The scale-vs-precision triage does not apply (zero error).
**Recommended tolerances**: exact (`helpers.TOLERANCES[bfloat16] == ("exact", None)`). bf8b output (Refinement 2) keeps `("block_step", 1)`.

## Verifier CLI Summary

`python3 -m eval.verify_supported generated/g6_verify ttnn.operations.g6_to_layout`:

- supported_pass: 58 (49 of 73 INPUTS cells + 9 of 16 LOOSE_CASES)
- xfail_expected: 31 (24 INPUTS + 7 loose). All are mapped to Refinements 1–2 in `op_requirements.md`:
  - RM bf16 → TILE bf16: 21
  - RM bf16 → TILE bf8b: 2
  - bf8b TILE interleaved: 3
  - bf8b HEIGHT_SHARDED: 4
  - bf8b BLOCK_SHARDED: 1
- invalid_skipped: 0 (no production combination is INVALID)
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- no_axes_found: 2 (`test_regression.py`, not registry-driven; both pass)

Acceptance suite: 31/31 plain and under `--dev` (watcher polling, NoC sanitizer, ebreak asserts).

## Performance (measured)

Golden-run device kernel time (110 cores):

| Case | Regime | µs | Bytes in + out | Effective |
|------|--------|----|----------------|-----------|
| 308 `[1,2048,2048,3]` **perf focus** | R2 | 3331–3432 | 128 MiB face rows + 24 MiB logical (4.19 M writes) | write-transaction-bound |
| 304 `[1,1024,2048,128]` | R1 | 3008 | 512 + 512 MiB | 357 GB/s |
| 1978 `[16,4096,2560]` | R1 | 1762 | 320 + 320 MiB | 381 GB/s |
| 286 `[1,1024,1024,3]` | R2 | 785–794 | 32 MiB + 6 MiB (1.05 M writes) | write-transaction-bound |
| 2031 `[1,4096,2560]` | R1 | 137 | 20 + 20 MiB | 307 GB/s |

**Case 308 decomposition** (verifier ablation; each kernel edited in place, measured, restored):

| Variant | 308 µs | 286 µs |
|---------|--------|--------|
| baseline | 3369–3380 | 781–796 |
| writes removed | 516–518 | 127 |
| reads removed | 2166–2168 | 546 |

- The 4.19 M 6-B DRAM writes alone cost 2.17 ms.
- Reads alone (128 MiB of face rows) take 517 µs, about 260 GB/s. The R2 reader drains every block with a full barrier, so no read stays in flight across blocks.
- Combined, the writes congest the reads. Per-core reader time grades by core row, from 2.57 ms at y = 2 to 0.40 ms at y = 11. The writer's own issue time is uniform (~1.09 ms per core).

**The lead's 147.46 µs goal for 308 is below this op's physical floor.**

- The 109.23 µs roofline counts 48 MiB of logical bytes.
- The storage formats force at least 128 MiB of reads, because a 32-B face row is the read quantum for 3 live values.
- They also force at least one of: 4.19 M sub-slot writes (2.17 ms measured), or 256 MiB of contiguous 64-B slot runs.
- At the 360–400 GB/s this op already sustains on R1, 384 MiB is ~1.0 ms. Even at a 512 GB/s peak it is ~0.8 ms.

The perf refinements aim at that floor; their gate is a measured device-ns win, not the 147 µs figure.

**R1 profile** (304, 1978): bound by issuing the reorder copies on NCRISC.

- The reader's `load_reorder_block` zone is 2.82 ms (304) / 1.63 ms (1978) per core, uniform across cores.
- The writer waits 62% / 71% of its time.
- About 1.1–1.2 µs per tile: 64 self-aimed 32-B copies plus the tile read.

## Recommendations

- **Queue** (`op_requirements.md`): R1 bf8b TILE input + HEIGHT/BLOCK-sharded inputs → R2 ROW_MAJOR input (bf16 + bf8b output) → R3 perf (308 write transactions) → R4 perf (308 read overlap + block co-tune). The perf target (308) is already fully SUPPORTED at Phase 0, so R1/R2 are ordered by difficulty and dependency, not by the perf anchor.
- **R1 regime reorder issue (not queued)**: the only `attention:`-flagged case is 308 (R2), so this R1 finding is not a queue entry. If a later perf phase opens R1, split the 64 copies per tile across NCRISC and BRISC (catalog `split_reader`; BRISC idles 60–70% of R1's run). 1978 would then move toward its DRAM bound (~1.3–1.4 ms at ~480 GB/s from 1.76 ms).
- **bf8b bit identity (R1)**: the golden specials case for a bf8b input (loose c29) grades the device against the host's bf8b rendering bit-for-bit. The `packer_untilize` value path crosses MOVA2D, which zero-substitutes exponent-0 datums under the default Src zero flag (design § Regimes). Probe it before building the full path.
- **L1 pressure**: none near the cap on Phase 0 shapes (max 688128 B). The L1-cap block for R2 reaches 1347584 B (308 at `pipeline_blocks_per_core = 1`), which is why the knob is not the coarsest fit.
- **Untracked pipeline state**: `pipeline_logs/` (including a patch of pre-boundary working-tree changes) is the orchestrator's and is not committed with this pass.
