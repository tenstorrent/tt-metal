# Verification Report: layer_norm_tile

Board: Blackhole p100a, 110-core compute grid (11 x 10). Op: `ttnn.operations.layer_norm_tile`, re-exported as
`ttnn.operations.layer_norm`. Verified commit range: everything after the planner/implementer commits
`371ba4b4ac0` .. `53c6d69912a`.

## Golden suite status (blocker for the mechanical check)

`eval/golden_tests/layer_norm_tile/` does not exist: no `feature_spec.py`, no `test_golden.py`. The design records
this ("feature_spec.py status": `/golden-tests` must author it). As a result:

- `eval/eval_test_runner.sh` and `python3 -m eval.verify_supported` could not run, and **no `verifier_report.json`
  was produced**. Inventing a feature_spec here would be the verifier authoring its own grading surface.
- The SUPPORTED-vs-behaviour check below is manual, driven by the acceptance, extended and precision tests.
- **Action for the user**: run `/golden-tests` for `layer_norm_tile`, then re-run the verifier CLI. Inputs for
  that pass are in "Handoff to /golden-tests".

## Code Review

Fixed in place:

| # | Finding | Fix |
|---|---------|-----|
| 1 | `cb_output_tiles` was always `2 * block_w_tiles` (96 KB). On every R <= 110 shape each core owns one tile-row, so the second slot is never live (ledger check 2, capacity > live set). | `output_depth_rows = min(OUTPUT_DEPTH_ROWS, max_num_blocks * block_rows)` in `select_blocking()`, the same clamp `input_depth_blocks` already used. Saves 48 KB per core on R <= 110. Device time unchanged within noise. Ledger updated. |
| 2 | Face size derived twice on the host (`affine_tile_bytes // 4` for the reader CT arg, `FACE * FACE * element_size` in `_affine_reader_args`). DRY. | One helper `_affine_face_bytes(affine)` feeds both. |
| 3 | Reader used bare `32` / `16` literals for tile width and face width in the RM affine address math. | Named `TILE_WIDTH` / `FACE_WIDTH` constants. |
| 4 | `l1_ledger.md` total table labelled `block_rows = 2` cases as if built; `output_depth_rows` was a constant. | Ledger rows, symbol table and totals updated. Measured DM / compute ablation appended to the data-movement budget. |
| 5 | `op_design.md` Phase 0 values were stale vs the build (`BLOCK_ROWS_MAX` 2 -> 1, P3 DEST accumulate, single 576 B TILE affine read). | Appended a "Post-implementation deltas (verifier)" table. The design body is unchanged. |
| 6 | Perf test covered only 3 shapes at one config. | `test_layer_norm_tile_perf.py` now spans R = 32 / 64 / 128 / 248 x 2 affine forms x 3 compute configs. This is the guard set the perf refinements gate on. |

Checked, no change needed:

- **Prompt contract** (`eval/prompts/layer_norm_tile.txt`, no `## Rules` section; its MUST statements apply):
  - TILE input only; no `to_layout` / `tilize` / `pad` / `slice` on host; non-aligned S handled in kernels.
  - Non-bf16 input raises `ValueError` mentioning `dtype`.
  - Exactly one `ttnn.generic_op` per call.
  - The `layer_norm(input, compute_kernel_config=...)` call pattern in the prompt omits gamma/beta, which are required positional parameters. It is a `TypeError` by signature, as the design notes.
- **Helper usage**: every compute phase is a helper call:
  - P1 `reduce_mean`.
  - P2 `sub`, column broadcast.
  - P3 `eltwise_chain` (`BinaryFpu<Mul>` + `PackTile` with DEST `PerRow` accumulation).
  - P4 `reduce`.
  - P5 / P6 `mul`, in place.
  - P7 `add`.

  The P4 finalize uses raw SFPU calls inside `reduce()`'s `post_reduce_op` slot. That slot takes a DEST-index callable, so no chain element fits. Dataflow is raw `TensorAccessor` page I/O, which the design justifies: no helper covers page reads or the face-row scatter. No multicast or semaphores exist that should use `McastArgs`.
- **CB sync balance** (per block, nominal counts):
  - `cb_input_tiles`: reader pushes 24 per row; P1 waits without popping; P2 pops.
  - `cb_row_stat`: P1 pushes `block_rows`, P2 pops at end; P3 pushes one per row, P4 bulk-pops.
  - `cb_rstd`: P4 push, P5 pop.
  - `cb_centered`: P2 pushes N; P3 waits N without popping; P5 / P6 pop and push N each; P7 pops N.
  - `cb_output_tiles`: P7 pushes N per tile; the writer pops 24 per row.
  - Affine CBs: pushed once, popped once after the last block.
- **API hygiene**: `TensorAccessor` (no `InterleavedAddrGen`); `void kernel_main()` in all three kernels; `api/dataflow/dataflow_api.h` include path.
- **Broadcast efficiency**: COL broadcast for mean and rstd (REDUCE_ROW output, column 0 valid). ROW broadcast for gamma and beta, with only row 0 of faces 0 and 1 filled (64 valid B per tile, not full tiles). No redundant fills, and no reduce-scaler CB.
- **Writer**: `noc_async_writes_flushed()` before each `cb_pop_front`, and one `noc_async_write_barrier()` at the end.

Advisory (no fix possible on this board):

- P3 relies on ELWMUL accumulating into DEST (`mul_init(..., acc_to_dest=1)`). The design flagged that the WH MOP hardcodes `acc_to_dest = 0` for ELWMUL. It is correct on Blackhole: acceptance passes, and the precision baseline matches the fp32 reference to 1.2 mean ULP. It is unverified on Wormhole and needs a WH board to confirm. If WH is ever a target, the packer L1-accumulate realization from the design is the fallback.

## Design Conformance / Blocking Model

- **Algorithm**: two-pass LayerNorm with a resident row (mean, center, sum of squares on the centered values, rstd, scale, gamma, beta), as designed.
- **Work distribution**: `split_work_to_cores(full compute_with_storage_grid, tensor_row_tiles, row_wise=True)`, which fills min(R, 110) cores. R uses the per-image ceil. Kernels are created only on `all_cores`.
- **Knob fidelity**:
  - `select_blocking()` is the single source for `block_rows`, `block_w_tiles`, `input_depth_blocks`, `output_depth_rows` and the core assignment.
  - Those values reach all three kernels as CT args (block shape) or RT args (`num_blocks`, `start_row_tile`, `core_rows`).
  - CB page counts derive from the same values.
  - No CB scales with an unbounded op dimension. `block_w_tiles <= W_TILES_RESIDENT_MAX` is a predicate-guarded resident row; the streaming fallback was rejected as unreachable under the contract (W = 768).
- **Half-turned split**: `BLOCK_ROWS_MAX = 1` while some cores own 2-3 rows (R = 128, 248). The departure from the coarsest-block default is justified by measurement (implementer, overlap lamp: R = 128 39.3 -> ~35 us, R = 248 65.6 -> ~51 us). Refinement 3 re-tests it once overlap levers exist.
- **Expression**:
  - Reader: one handshake per tile-row block, 24 reads, one barrier, one push.
  - Writer: one wait / 24 writes / one flush / one pop per tile-row.
  - Compute: every phase is one helper call over `grid(block_rows, block_w_tiles)`.
  - No unit-at-a-time synchronization at a block boundary.
- **Performance conformance**: the grid is filled up to R; both dataflow halves are batched per tile-row; buffer depth is 2 wherever a core has a next block to prefetch.

  The measured gap is structural: one-tile-row cores serialize read, compute and write (see "Performance baseline"). That is filed as Refinement 3, not a conformance defect. Idle cores at R <= 55 are the design's deferred `w-split-combine` regime (Refinement 1).

## Registry Conformance

- `INPUT_TAGGERS = {"alignment": tag_alignment}` with signature `(inputs, axes)`.
- `SUPPORTED` declares `dtype`, `fp32_dest_acc_en`, `layout`, `alignment`, `affine_layout` and `memory_layout`.
- `EXCLUSIONS = []`.
- `validate()` checks SUPPORTED per axis, then EXCLUSIONS (once per affine layout), raising `UnsupportedAxisValue` / `ExcludedCell`.
- The op file does not declare `INVALID`.
- The entry point resolves the config, raises the prompt-mandated `ValueError` for non-bf16 input, then calls `validate()` before any device work.
- No XPASS-driven SUPPORTED changes (no golden run).

Undeclared refusals (`NotImplementedError` raised outside a SUPPORTED axis):

| Gate | Where | Why not added as an axis now |
|------|-------|------------------------------|
| gamma/beta dtype must be bf16 | `_validate_affine` | `eval.feature_matrix.unsupported_reason` treats a SUPPORTED axis missing from a cell's values as unsupported. An axis the future golden suite does not emit would xfail every golden cell. The golden suite must name the axis first. |
| gamma/beta must be interleaved | `_validate_affine` | same |
| ROW_MAJOR gamma/beta needs `Wg % 32 == 0` and 32-B halves aligned to the buffer's NoC alignment, so RM affine in DRAM is refused on BH (64 B DRAM alignment) | `_validate_affine` | Out of contract: the prompt's RM form is L1. The refusal is arch-dependent (it would pass on a 32 B DRAM alignment), so a static EXCLUSION would be wrong. `test_rm_affine_in_dram` pins the behaviour per arch. |
| `ceil(W/32) <= W_TILES_RESIDENT_MAX = 32` | `validate()` | Regime predicate (resident row). Out of contract (W = 768). |
| W not a multiple of 32 (`alignment = w_non_aligned`) | SUPPORTED["alignment"] | Declared, so already honest. Out of contract (W = 768). |

## INVALID audit

Not possible: `feature_spec.py` is absent. See the handoff below.

## Handoff to /golden-tests

The design's TARGET table names the axes `validate()` gates on:
- `dtype` = [bfloat16].
- `fp32_dest_acc_en` = [True, False].
- `layout` = [TILE].
- `alignment` = [tile_aligned, h_non_aligned].
- `affine_layout` = [TILE, ROW_MAJOR].
- `memory_layout` = [INTERLEAVED].

Recommendations for the suite:
- **Shapes**: build INPUTS from the contract (`(B, 1, S, 768)`, `1024 <= B*S <= 4096`, `S <= 512`, including non-aligned S).
- **Affine pinning**: TILE affine in DRAM and ROW_MAJOR affine in L1. Otherwise the RM-in-DRAM cells become `supported_fail` on BH.
- **INVALID**: nothing structural is needed. TARGET has no bf8b and no RM input, so the canonical bf8b+ROW_MAJOR entry is moot. Add it only if the suite widens dtype or layout.
- **Affine dtype**: if the suite adds an `affine_dtype` axis, the op must add `SUPPORTED["affine_dtype"] = [bfloat16]` in the same change.

## L1 Ledger Audit

1. **Currency**: 7 CBs declared, 7 ledger rows; size expressions match `create_program_descriptor`. Fixed the `cb_output_tiles` depth expression and the totals (Code Review #1, #4).
2. **Capacity vs live set**:
   - Over: `cb_output_tiles` held two rows on single-row cores. Fixed in place.
   - `cb_input_tiles` depth 2 appears only with more than one block per core. Justified.
   - Under: none. `cb_centered` spans `row` x `w` with capacity `block_rows * block_w_tiles`; `cb_row_stat` / `cb_rstd` span `row` with capacity `block_rows`.
3. **Page format vs DEST width**: intermediates are Float32 if and only if `fp32_dest_acc_en`, else Float16_b. I/O and affine CBs are bf16 tensor data that never traverse DEST as an accumulation. Clean both ways.
4. **Disjoint lifetimes**: every pair has a recorded reason:
   - `cb_row_stat` self-shares mean and sum of squares.
   - `cb_rstd` is concurrent with `cb_row_stat` in P4.
   - `cb_input_tiles` / `cb_centered` differ in format and ownership.
   - `cb_centered` / `cb_output_tiles` cannot share: that would give the writer a second consumer.
5. **Bounds**: `block_w_tiles <= 32` (validate), `block_rows <= 1`, depths in {1, 2}, `T_int` in {2, 4} KB. The total is closed-form: 296 KB (R <= 110), 392 KB (R > 110), 784 KB at the knob bound.

**Data-movement budget**: present and consistent with the code (input and output each cross the NoC once; affine row 0 read once per core). The cheapest-traffic split, `row`, is implemented. `w-split-combine` is a `deferred` row with a positive reason: more bytes, occupancy-only, precision-gated, and the gate is now met. Filed as Refinement 1.

`affine-read-once-mcast` was `rejected` in the design. The implementer's and the verifier's measurements reverse that for the TILE-DRAM form: about 15 us of DM at P = 110. Filed as Refinement 2.

**Block-size defaults**: interleaved spreads across the full grid (held). The coarsest block was departed from (`BLOCK_ROWS_MAX` 2 -> 1) on measurement, not on budget pressure. L1 use is 296-392 KB of about 1.5 MB.

## Precision Baseline

`tests/ttnn/unit_tests/operations/layer_norm_tile/test_layer_norm_tile_precision_baseline.py`, seed 1234, `randn` input / gamma / beta, fp32 torch reference, HiFi4. Values are identical for the `tile_dram` and `rm_l1` affine forms (same gamma bits), so one row per shape and config is shown.

| Shape | DEST | PCC | Max Abs Err | Mean Abs Err | Rel RMS Err | Mean ULP (bf16) | p99 ULP | got/true median [p5, p95] |
|-------|------|-----|-------------|--------------|-------------|-----------------|---------|---------------------------|
| (32,1,32,768) | fp32 | 0.9999983 | 0.0446 | 0.00180 | 0.00190 | 1.18 | 6.2 | 0.99988 [0.9958, 1.0040] |
| (32,1,32,768) | bf16 | 0.9999914 | 0.1651 | 0.00434 | 0.00520 | 3.93 | 23.5 | 1.0017 [0.9904, 1.0162] |
| (11,1,100,768) | fp32 | 0.9999983 | 0.0515 | 0.00176 | 0.00190 | 1.05 | 6.0 | 0.99989 [0.9958, 1.0039] |
| (11,1,100,768) | bf16 | 0.9999916 | 0.1391 | 0.00419 | 0.00512 | 3.39 | 22.5 | 1.0017 [0.9906, 1.0157] |
| (8,1,512,768) | fp32 | 0.9999983 | 0.0511 | 0.00176 | 0.00190 | 1.35 | 5.9 | 0.99988 [0.9958, 1.0039] |
| (8,1,512,768) | bf16 | 0.9999918 | 0.1798 | 0.00418 | 0.00510 | 3.27 | 21.9 | 1.0017 [0.9907, 1.0156] |
| (124,1,33,768) | fp32 | 0.9999983 | 0.0440 | 0.00177 | 0.00190 | 1.33 | 6.0 | 0.99988 [0.9958, 1.0039] |
| (124,1,33,768) | bf16 | 0.9999918 | 0.1802 | 0.00416 | 0.00506 | 4.65 | 22.8 | 1.0017 [0.9906, 1.0158] |

**Assessment**: precision is at the bf16 output-rounding floor with fp32 DEST (about 1 mean ULP) and about 3-4x that with bf16 DEST. The error does not depend on shape, padding or the per-core split.

The got/true ratio is tightly centred on 1.0, so there is no scale or structural bug. The +0.17% median bias under bf16 DEST is the bf16-rounded rstd: a tight bias well inside rel-RMS 0.5%, not a defect.

Max ULP (not shown) reaches about 1e6 only at references near zero, where beta cancels the normalized term.

**Recommended tolerances**:
- fp32 DEST: PCC >= 0.99995, rtol = 0.02, atol = 0.06.
- bf16 DEST: PCC >= 0.9999, rtol = 0.05, atol = 0.2.
- The acceptance gate of PCC 0.995 is loose and catches only gross bugs.

## Verifier CLI Summary

Not run: there is no golden suite (see top). Manual equivalent on this board:

| Bucket | Count | Source |
|--------|-------|--------|
| supported cells passing | 41 acceptance + 5 extended (DRAM input, one-stick RM, mixed affine forms) + 16 precision + 24 perf smoke | `scripts/run_safe_pytest.sh` (prod) + `--dev` (acceptance + extended) |
| refusals raising `NotImplementedError` as declared | 2 (`w_non_aligned`; RM affine in DRAM on BH) | `test_layer_norm_tile_extended.py` |
| non-bf16 input -> `ValueError` | 1 | acceptance |
| supported_fail / xpass_drift / xfail_wrong_mode | not measurable without the golden suite; 0 observed in the manual set | - |

## Performance baseline (device kernel ns, default config unless noted)

| Shape (R) | cores | tile_dram | rm_l1 | rm_l1 compute-only | rm_l1 DM-only | tile_dram DM-only |
|-----------|-------|-----------|-------|--------------------|---------------|-------------------|
| (32,1,32,768) R=32 | 32 | 14.1 | 14.2 | 10.2 | 4.5 | 7.3 |
| (16,1,128,768) R=64 | 64 | 21.5-21.8 | 19.1-20.9 | 10.2 | 9.0 | 16.3 |
| (8,1,512,768) R=128 | 110 | 41.6-44.3 | 31.4-34.5 | 19.8 | 17.2 | 32.2 |
| (124,1,33,768) R=248 | 110 | 52.9-54.0 | 46.0-54.5 | 29.4 | 27.0 | 33.3 |

R = 32 under other configs: `fp32_dest_acc_en=False` 13.9 us, LoFi 12.5 us.

Ablation method: kernels temporarily edited, then restored.
- **Compute-only**: the reader and writer skip input/output NoC transfers but keep the CB handshakes and the affine fetch.
- **DM-only**: compute is reduced to CB handshakes.

Readings:
- Compute is about 10.2 us per tile-row on every shape.
- On one-tile-row cores, DM and compute serialize.
- DM grows with participating cores because aggregate NoC throughput saturates near 700-900 GB/s.
- The TILE-DRAM affine fetch costs up to about 15 us at 110 cores.
- Run-to-run spread is 5-10% on the 110-core shapes.

## Recommendations

- **Production affine form**: the prompt asks which form to prefer per shape. RM-L1 is as fast or faster at every measured shape: equal at R = 32, 1-2 us faster at R = 64, 8-12 us faster at R = 128. Production callers should pass `(1, 1, 24, 32)` ROW_MAJOR in L1. Refinement 2 targets closing the TILE-DRAM gap.
- **Queue order**: Refinements 1 -> 4 build structure first:
  1. Cross-core W-split.
  2. Affine multicast, built on the slice groups.
  3. DM overlap and locality, which must keep W-split groups adjacent.
  4. Compute-local trims.
- **Generality**: there is no generality refinement because `TARGET - SUPPORTED` is empty. Documented omissions (out of the prompt contract, so no refinement):
  - `w_non_aligned` (W pinned to 768).
  - ROW_MAJOR input (the prompt forbids it).
  - Sharded placements (TARGET is interleaved). The design records HEIGHT_SHARDED as a knob-turn and WIDTH/BLOCK as the Refinement 1 combine with caller-pinned k.
  - Non-bf16 dtypes, for input or affine.
  - RM affine in DRAM.
- **Golden suite**: author it (`/golden-tests`) and re-run the verifier CLI before or alongside Refinement 1. The queue's perf gates do not depend on it, but the registry check does.
- **Wormhole**: P3's ELWMUL DEST accumulation needs a WH confirmation before the op is claimed for WH (see Code Review advisory).
