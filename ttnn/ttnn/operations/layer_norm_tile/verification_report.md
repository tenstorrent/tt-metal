# Verification Report: layer_norm_tile

Board: Blackhole p100a, 110-core compute grid (11 x 10). Op: `ttnn.operations.layer_norm_tile`, re-exported as
`ttnn.operations.layer_norm` (235 B shim, `ttnn/ttnn/operations/layer_norm/__init__.py`). Verified: the
`row_resident` build (`ab5af465d8f` planner, `489dc153a50` / `a6da58d8aea` implementer).

## Golden suite status (blocks the mechanical check)

`eval/golden_tests/layer_norm_tile/` does not exist: there is no `feature_spec.py` and no `test_golden.py`. The
design records this (Overview, "feature_spec.py"). As a result:

- `eval/eval_test_runner.sh` and `python3 -m eval.verify_supported` cannot run, so **no `verifier_report.json`
  was produced**. The verifier does not author `feature_spec.py`: that would mean grading against a spec it
  wrote itself.
- The SUPPORTED-vs-behaviour check below is manual. It is driven by the acceptance (80), extended (12),
  precision (16) and perf (60) tests.
- **Action for the user**: run `/golden-tests` for `layer_norm_tile`, then re-run the verifier CLI. Inputs
  for that pass are in "Handoff to /golden-tests".

## Code Review

Fixed in place:

| # | Finding | Fix |
|---|---------|-----|
| 1 | `affine_layout` was tagged `"mixed"` when gamma and beta layouts differed, and `validate()` refused it. The writer takes `gamma_is_rm` / `beta_is_rm` as independent CT flags, and the design says beta follows gamma's rules "independently". Mixed is a real, working code path, so the refusal was false (drift: the op under-claimed). | Added `"mixed"` to `SUPPORTED["affine_layout"]`. Covered by `test_affine_forms` (TILE+RM, RM+TILE, RM-DRAM+TILE), prod and `--dev`. |
| 2 | `W % 32 != 0` raised `ValueError` in the universe check, before the `alignment` axis. That made the tagger's `w_non_aligned` value dead code, and a golden `w_non_aligned` cell would land in `xfail_wrong_mode`. Non-aligned W is a kernel capability gap, not a structural impossibility. | Removed the `ValueError`; `SUPPORTED["alignment"]` now refuses it with a typed `UnsupportedAxisValue`. |
| 3 | A ROW_MAJOR affine stick width that is not a multiple of 32 (a writer mechanism cap: a 32-element chunk must sit in one stick) raised `ValueError`. | Moved to `_check_affine_sticks()`, run after the SUPPORTED/EXCLUSIONS gates and raising `UnsupportedAxisValue`. New test `test_rejects_rm_affine_stick_not_tile_multiple`. |
| 4 | The L1-cap refusal (`derive_blocking()` returns `None`) raised a bare `NotImplementedError`. | Now a typed `UnsupportedAxisValue` naming the width and the deferred regime. |
| 5 | DRY: the L1 cap's `_footprint()` restated every CB size as an independent closed-form expression, while `create_program_descriptor()` sized the CBs again on its own. A depth or block change in one would silently drift from the other. | One `_cb_plan()` returns `{allocation: (pages, page_bytes)}`. `_footprint()` sums it, `derive_blocking()` stores it on `Blocking.cb_plan`, and the descriptor builds every CB from it. `_cb` / `_alias_group` merged into one `_cb(indices, ...)`. Dead locals (`acc_bytes`, `block_tiles`, `Blocking.acc_tile_bytes`) removed. The writer's output page-size CT arg comes from the plan. |
| 6 | `PROPERTIES` was declared in the op file but not exported from the package. `eval.dump_op_metadata` reads it from the package module, so it saw `{}`. | Exported from `__init__.py`. |
| 7 | `test_layer_norm_tile_extended.py` was stale against this build: 4/12 failing. It expected RM-in-DRAM affine to be refused and mixed affine to be refused. | Rewritten to the current contract: DRAM-interleaved input (with and without residual); five affine forms incl. mixed and RM in DRAM; typed refusals for `w_non_aligned`, RM stick width, HEIGHT_SHARDED input. |
| 8 | Precision baseline and perf guard set did not cover the residual path or the multi-block regime. | Baseline: 4 shapes x {tile_dram/no residual, rm_l1/residual} x {fp32, bf16 DEST}. Perf: 5 shapes (adds `(256,1,16,768)`, multi-block with residual) x 2 affine x 2 residual x 3 configs, two dispatches per case. |
| 9 | Ledger symbol table bounded `Wt` by "validate: `W == 768`", which `validate()` does not enforce. | Bound restated from the real predicate: the L1 cap, `Wt <= 38` (fp32 + residual) / `<= 47` (fp32, no residual). |

Checked, no change needed:

- **Prompt contract** (`eval/prompts/layer_norm_tile.txt`, no `## Rules` section; its MUST statements apply):
  - TILE input only, with no host `to_layout` / `tilize` / `pad` / `slice`.
  - Non-aligned S is handled in the kernels (padded rows compute finite values and are written as whole tiles).
  - Non-bf16 input raises `ValueError`.
  - The import path `ttnn.operations.layer_norm` works.
  - Exactly one `ttnn.generic_op` per call.
  - The prompt's `layer_norm(input, compute_kernel_config=...)` call pattern omits gamma/beta, which the same prompt makes required. It is a `TypeError` by signature.
- **Helper usage**: every compute phase is one helper call per block:
  - `add`, `reduce_mean`, `sub` (Col bcast).
  - `eltwise_chain` (`BinaryFpu<Mul>` + `PackTile`, DEST `PerRow`).
  - `reduce` with an SFPU post-op.
  - `mul` (Col bcast), `mul` (Row bcast), `add` (Row bcast).

  The post-op uses raw SFPU calls because `reduce()`'s post-op slot is a DEST-index callable (design, "Helpers considered and rejected"). The writer's RM face-1 copy uses `dataflow_kernel_lib::local_addr()`. There is no multicast or semaphore code that should use `McastArgs`.
- **CB sync balance** (per block of `block_rows` tile-rows):

  | CB | Producer side | Consumer side |
  |----|---------------|---------------|
  | `cb_input_tiles` | reader: `Wt` per row | no residual: `reduce_mean` no-pop, then `sub` pops per tile. Residual: `add` pops per tile. |
  | `cb_residual_tiles` | reader | `add` pops per tile |
  | `work` group | `xsum` push (add); `centered` push (sub); `normed` push (mul); `scaled` push (mul) | `xsum` pop (sub); `centered` held by `sumsq`, then pop (mul); `normed` pop (mul); `scaled` pop (add) |
  | `stats` group | `mean` bulk push; `sumsq` push per row; `rstd` push | `mean` pop per row; `sumsq` bulk pop; `rstd` pop per row |
  | `cb_output_tiles` | push per tile | writer pops `Wt` per row |
  | gamma / beta | pushed once | popped once after the last block |

  Each id of an alias group moves `block_rows * Wt` (work) or `block_rows` (stats) per block, lockstep.
- **API hygiene**: `TensorAccessor` everywhere (no `InterleavedAddrGen`); `void kernel_main()`; `api/dataflow/dataflow_api.h`.
- **Broadcast efficiency**:
  - Col broadcast for mean / rstd (REDUCE_ROW output, col 0 valid).
  - Row broadcast for gamma / beta, of which only face-0 row 0 and face-1 row 0 are written: one 576 B read per tile-column for TILE, a 64 B read plus a 32 B local copy for RM.
  - No redundant fills, and no scaler CB (the aligned `AccumulateViaAdd` path never waits on one).

Advisory (needs other hardware, so no fix is possible here):

- `sumsq_block` accumulates ELWMUL into DEST (`DestAccumulation::PerRow`, `mul_init(..., acc_to_dest)`). It is correct on Blackhole: acceptance passes and the fp32 baseline is about 1.2 mean bf16 ULP. It is unverified on Wormhole.

## Design Conformance / Blocking Model

- **Algorithm**: matches the design. Two-pass centered variance on a resident row (`one_pass_moments` rejected), residual folded once into `cb_xsum`, `1/W` from the logical W.
- **Pipeline topology**:
  - The reader streams input and residual.
  - The writer fetches gamma/beta first (stall-shadowed behind block-0 reads and phases add..normalize), then drains output.
  - Compute runs the 8 designed phases.
- **Work distribution**: `split_work_to_cores(grid, R, row_wise=True)` over the full runtime grid. Per-image `ceil(S/32)` row count. Kernels only on `all_cores`.

  At R = 32 / 64 the grid is under-filled (32 / 64 of 110 cores). That is the design's `w_split_combine`, deferred on a precision-baseline gate that is now met, so it is filed as Refinement 1.
- **Knob fidelity**:
  - `block_height`, `input_depth`, `OUTPUT_DEPTH`, `CB_L1_BUDGET_BYTES` and the per-core assignment each have one source (`derive_blocking()` / module constants).
  - They reach the kernels as CT args (`Wt`, `block_height`) and RT args (`start_row`, `num_blocks`, `last_block_extent`).
  - After fix #5, every CB size and the L1 cap read the same `_cb_plan()`.
  - No CB scales with an unbounded op dimension: `Wt` is bounded by the cap predicate (fix #9).
- **Half-turned split**: none. The per-core block is the whole assignment when it fits, otherwise the coarsest block under the cap.
- **Expression**:
  - The reader reserves, reads, barriers and pushes once per tile-row (24 or 48 reads in flight). The CB capacity spans the block, so the block factor sets the live set, not only the trip count. This per-tile-row handshake is the design's planned granularity.
  - The writer waits, issues 24 writes, flushes and pops once per tile-row.
  - Compute is one helper call per phase per block.
  - Not an expression failure.
- **Axis accounting**: every ledger row's span/stream claims hold against the code.

## Registry Conformance

- `INPUT_TAGGERS = {"alignment": tag_alignment}` with the `(inputs, axes)` signature.
- `SUPPORTED`:
  - `dtype` [bf16], `fp32_dest_acc_en` [True, False], `layout` [TILE].
  - `alignment` [tile_aligned, h_non_aligned], `memory_layout` [INTERLEAVED].
  - `residual` [none, present], `affine_layout` [TILE, ROW_MAJOR, mixed].
- `EXCLUSIONS = []`.
- `validate()` order:
  1. The universe check (`ValueError`: dtype, input/residual layout, rank, residual shape, affine volume, TILE affine last dim).
  2. SUPPORTED per axis (`UnsupportedAxisValue`).
  3. EXCLUSIONS (`ExcludedCell`).
  4. The affine stick mechanism cap (`UnsupportedAxisValue`).
- The entry point calls `validate()` before any device work. The op file does not declare `INVALID`.
- Drift fixed in place: `affine_layout="mixed"` added to SUPPORTED (fix #1).
- Refusal typing fixed: `w_non_aligned`, RM stick width, L1 cap (fixes #2-4).
- `ValueError` stays only for structural / prompt-mandated rejections.

## INVALID audit

Not possible: `feature_spec.py` is absent.

## Handoff to /golden-tests

Axes `validate()` builds, with the values the design names as TARGET:
- `dtype` = [bfloat16].
- `fp32_dest_acc_en` = [True, False].
- `layout` = [TILE].
- `alignment` = [tile_aligned, h_non_aligned].
- `memory_layout` = [INTERLEAVED].
- `residual` = [none, present].
- `affine_layout` = [TILE, ROW_MAJOR].

The op also supports `affine_layout = mixed`.

Suite recommendations:
- **Shapes**: build INPUTS from the contract: `(B, 1, S, 768)` with `1024 <= B*S <= 4096`, `S <= 512`, power-of-two and non-aligned S.
- **Affine forms**: `(768,)` TILE in DRAM and `(1,1,24,32)` ROW_MAJOR in L1. RM in DRAM also works.
- **Structural INVALID**: none needed. There is one dtype and one input layout, so the canonical bf8b + ROW_MAJOR entry is moot unless the suite widens dtype / layout.
- **Perf flag**: flag one `LOOSE_CASE` `attention:` at a production shape, e.g. `(8, 1, 512, 768)` with residual, so the perf queue has a pinned target.

## L1 Ledger Audit

1. **Currency**: 7 allocations with residual, 6 without (`input`, `residual`, `gamma`, `beta`, `work` alias group of 4 ids, `stats` alias group of 3 ids, `output`). This matches the code. Every size expression matches `_cb_plan()`. Fixed: the `Wt` bound row (#9).
2. **Capacity vs live set**:
   - *Over*: on the residual path, `cb_input_tiles` / `cb_residual_tiles` hold `input_depth * block_height * Wt` while their consumer-side live set is 1 tile (`add_residual_block` streams them). The ledger gives a reason (whole-block reader run-ahead; the `Wt` window is named as the alternative). The reason is valid, but its value is unmeasured. Folded into **Refinement 2** (co-tune with the NoC levers).
   - `cb_output_tiles` at `OUTPUT_DEPTH = 2` tile-rows is live on one-row cores only as the compute/writer overlap window. Justified.
   - *Under*: none. The `work` group spans `tile_rows x W` with capacity `block_height * Wt`; `stats` spans `tile_rows` with `block_height`.
3. **Page format vs DEST width**: `work` / `stats` are Float32 if and only if `fp32_dest_acc_en`, else Float16_b. I/O and affine CBs are bf16 tensor data (gamma/beta never traverse DEST as accumulators). Clean in both directions.
4. **Disjoint lifetimes**: every non-shared pair has a recorded reason. Reader-produced vs compute-produced (no two producers on the same bytes); kernel-lifetime affine rows; `cb_output_tiles` is writer-consumed and concurrent with the next row's compute.
5. **Bounds / closed form**: total = `block_height * (Wt * (input_depth * T_in * (1 + HAS_RESIDUAL) + T_acc) + T_acc) + Wt * T_in * (2 + OUTPUT_DEPTH)`.
   - It scales with `block_height`, `input_depth`, `HAS_RESIDUAL` and `T_acc`.
   - Worst built case: 598,016 B (fp32 + residual, `block_height = 2`) against the 786,432 B budget.
   - Every symbol is bounded with its predicate.

**Data-movement budget**: present and consistent with the code.
- Input, residual and output each cross the NoC once.
- Gamma/beta row 0 is read once per active core: 13.5 KiB TILE-DRAM or 1.5 KiB RM-L1 per tensor.
- The cheapest-traffic split (`tile_rows`) is the one implemented. `w_split_combine` (more bytes, occupancy-motivated) and `affine_mcast` are `deferred` rows with positive reasons.

**Block-size defaults**: interleaved spreads across the full grid, then takes the coarsest block that fits (the whole assignment). Held. The multi-block path appears only when the cap forces it (`(256,1,16,768)` with residual). Footprint is 400-600 KB of the ~1.4 MB usable L1.

## Precision Baseline

`tests/ttnn/unit_tests/operations/layer_norm_tile/test_layer_norm_tile_precision_baseline.py`. Seed 1234, `randn` input / residual / gamma / beta, fp32 torch reference, HiFi4. Each shape is run as `tile_dram` affine without residual (`nores`) and as `rm_l1` affine with residual (`res`). The affine forms are bit-identical.

| Shape (tile-rows) | Variant | DEST | PCC | Max Abs | Mean Abs | Rel RMS | Mean ULP | p99 ULP | got/true median [p5, p95] |
|-------------------|---------|------|-----|---------|----------|---------|----------|---------|---------------------------|
| (32,1,32,768) R=32 | nores | fp32 | 0.9999983 | 0.0446 | 0.00180 | 0.00190 | 1.18 | 6.2 | 0.99988 [0.9958, 1.0040] |
| | nores | bf16 | 0.9999914 | 0.1651 | 0.00434 | 0.00520 | 3.93 | 23.5 | 1.0017 [0.9904, 1.0162] |
| | res | fp32 | 0.9999983 | 0.0473 | 0.00182 | 0.00190 | 1.21 | 5.8 | 0.99989 [0.9959, 1.0039] |
| | res | bf16 | 0.9999906 | 0.1771 | 0.00443 | 0.00534 | 4.13 | 22.0 | 1.0016 [0.9908, 1.0159] |
| (11,1,100,768) R=44 | nores | fp32 | 0.9999983 | 0.0515 | 0.00176 | 0.00190 | 1.05 | 6.0 | 0.99989 [0.9958, 1.0039] |
| | nores | bf16 | 0.9999916 | 0.1391 | 0.00419 | 0.00512 | 3.39 | 22.5 | 1.0017 [0.9906, 1.0157] |
| | res | fp32 | 0.9999983 | 0.0440 | 0.00171 | 0.00188 | 0.91 | 5.9 | 0.99992 [0.9959, 1.0039] |
| | res | bf16 | 0.9999907 | 0.1619 | 0.00409 | 0.00529 | 3.07 | 22.5 | 1.0015 [0.9906, 1.0161] |
| (8,1,512,768) R=128 | nores | fp32 | 0.9999983 | 0.0511 | 0.00176 | 0.00190 | 1.35 | 5.9 | 0.99988 [0.9958, 1.0039] |
| | nores | bf16 | 0.9999918 | 0.1798 | 0.00418 | 0.00510 | 3.27 | 21.9 | 1.0017 [0.9907, 1.0156] |
| | res | fp32 | 0.9999983 | 0.0518 | 0.00172 | 0.00188 | 3.72 | 6.1 | 0.99991 [0.9958, 1.0040] |
| | res | bf16 | 0.9999907 | 0.1972 | 0.00416 | 0.00527 | 3.41 | 23.1 | 1.0016 [0.9903, 1.0161] |
| (124,1,33,768) R=248 | nores | fp32 | 0.9999983 | 0.0440 | 0.00177 | 0.00190 | 1.33 | 6.0 | 0.99988 [0.9958, 1.0039] |
| | nores | bf16 | 0.9999918 | 0.1802 | 0.00416 | 0.00506 | 4.65 | 22.8 | 1.0017 [0.9906, 1.0158] |
| | res (multi-block) | fp32 | 0.9999983 | 0.0495 | 0.00180 | 0.00189 | 2.48 | 6.1 | 0.99988 [0.9958, 1.0040] |
| | res | bf16 | 0.9999904 | 0.2058 | 0.00451 | 0.00545 | 3.20 | 23.9 | 1.0017 [0.9900, 1.0168] |

**Assessment**:
- fp32 DEST is at the bf16 output rounding floor: rel-RMS 0.19%, p99 6 ULP.
- bf16 DEST costs about 2.7x in RMS (0.51-0.55%).
- Residual, non-aligned S and the multi-block path do not move precision.
- Mean ULP swings (0.9-3.7) come from near-zero references. p99 is the stable figure.
- Scale-bug check: the got/true ratio is centered on 1.0 with a ±0.4% (fp32) / ±1.3% (bf16) spread. That is noise, not a constant offset. The bf16-DEST median of 1.0016-1.0017 is a small systematic bias, about 0.2x the spread. It is consistent with the variance being rounded to bf16 before `rsqrt`, not a structural bug.

**Recommended tolerances**:
- fp32 DEST: PCC >= 0.9999, rel-RMS <= 0.003, atol 0.08.
- bf16 DEST: PCC >= 0.9999, rel-RMS <= 0.008, atol 0.3.
- rtol is not meaningful: references cross zero through beta.

## Performance baseline

`test_layer_norm_tile_perf.py --profile`, `DEVICE KERNEL DURATION`, min of 2 dispatches per case, us. `nores` / `res` = without / with residual.

| Shape (tile-rows, max rows/core) | Affine | nores default | nores fp32 off | nores LoFi | res default | res fp32 off | res LoFi |
|---|---|---|---|---|---|---|---|
| (32,1,32) R=32, 1 | tile_dram | 13.9 | 13.7 | 12.3 | 18.7 | 17.8 | 16.4 |
| | rm_l1 | 14.1 | 13.8 | 12.6 | 17.7 | 17.4 | 15.8 |
| (16,1,128) R=64, 1 | tile_dram | 21.3 | 19.0 | 18.9 | 33.7 | 32.4 | 32.2 |
| | rm_l1 | 21.1 | 20.4 | 18.5 | 29.0 | 32.0 | 31.8 |
| (8,1,512) R=128, 2 | tile_dram | 33.5 | 33.0 | 32.2 | 58.2 | 60.3 | 54.9 |
| | rm_l1 | 33.3 | 33.5 | 31.5 | 59.8 | 60.2 | 54.6 |
| (124,1,33) R=248, 3 | tile_dram | 53.4 | 51.8 | 46.6 | 95.9 | 95.4 | 98.9 |
| | rm_l1 | 54.9 | 55.5 | 50.5 | 93.6 | 99.6 | 93.5 |
| (256,1,16) R=256, 3 (multi-block w/ res) | tile_dram | 60.8 | 65.6 | 60.2 | 110.5 | 117.8 | 96.7 |
| | rm_l1 | 60.2 | 59.3 | 58.5 | 96.5 | 97.5 | 92.9 |

Reading:
- **R = 32 is compute-bound.** The DM-only stub (implementer, affine fetch removed) is 2.8 us of 14 us. 78 of 110 cores are idle, so Refinement 1 (`w_split_combine`) is the lever.
- **R >= 128 and every residual shape are NoC-bound.** The DM-only stub at `(8,1,512)` is 15.5 us without residual and 37.9 us with residual, out of 33.3 / 59.8 us. Residual adds +26 us at R = 128 but +3.6 us at R = 32: one reader RISC on NoC0 carries input plus residual. This is Refinement 2.
- **Affine form is perf-neutral** (tile_dram vs rm_l1 within noise everywhere except a few residual cells at 3-10% spread). The writer's fetch is hidden in the stall shadow, as designed. `affine_mcast` has no measurable headroom today, so it is not queued. Re-check it if Refinement 1 shrinks the pre-affine compute window enough to expose the fetch.
- **Fidelity / DEST width are minor terms**: LoFi saves at most 12%; `fp32_dest_acc_en=False` is within noise.

Verifier ablations, measured and reverted:

| Ablation | Result |
|----------|--------|
| Design "Overlap" lamp: force `block_height = 1`, `input_depth = 2` | With residual: R=128 59.0 -> 57.3, R=248 101.2 -> 93.9, R=256 97.5 -> 94.5 us. Without residual: neutral at R=128/248, **R=256 56.0 -> 66.6 us (-19%)**. Regime-dependent; filed as lever (d) of Refinement 2. |
| Residual reads issued on NoC1 from the NCRISC reader | All 5 cases fail: the default dedicated-NoC mode does not let NCRISC issue on NoC1. NoC balancing needs the second RISC (`split_reader`) or dynamic-NoC mode. Recorded in Refinement 2 lever (a). |

## Verifier CLI Summary

Not run: no golden suite exists (see the top section). Manual equivalent:

| Category | Count | Note |
|----------|-------|------|
| supported_pass | 168 / 168 local tests | acceptance 80 (prod + `--dev`), extended 12 (prod + `--dev`), precision 16, perf 60 |
| xfail_expected | 5 refusal tests | float32 input (`ValueError`, prompt-mandated), RM input, `w_non_aligned`, RM stick width, HEIGHT_SHARDED; all raise the intended type |
| supported_fail | 0 | |
| xpass_drift | 0 after fix #1 (was: `affine_layout = mixed`) | |
| xfail_wrong_mode | 0 after fix #2 (was: `w_non_aligned` raising `ValueError`) | |

## TARGET - SUPPORTED accounting

Every TARGET value (design table) is in SUPPORTED, so the queue holds only perf refinements. Documented omissions, which are outside TARGET and not queued:

- `memory_layout` HEIGHT / WIDTH / BLOCK sharded: the requirement scope is L1 interleaved. The design keeps `height_sharded` (knob-turn) and `width_sharded` / `block_sharded` (`w_split_combine` with a pinned `k`) as deferred regime rows. Promote them if `/golden-tests` adds sharded values to TARGET.
- `dtype` other than bf16, and ROW_MAJOR input: excluded by the prompt (`ValueError`).
- `alignment = w_non_aligned`: outside the contract (W = 768). Refused with a typed refusal.
- `half_tile_rows` regime: needs 16x32 tile handling in the compute helpers (a helper capability, not an op refinement).

## Recommendations

- Run `/golden-tests` first, so the next verifier pass has a mechanical SUPPORTED check and a pinned perf target.
- Refinement order: 1 (`w_split_combine`, T3 scheme-change) before 2 (NoC throughput, T2) before 3 (compute trims, T1/T2). Refinement 1 redefines the per-core block shape that 2 and 3 tune.
- Production callers: the affine forms perform the same, so pick by where gamma/beta already live. The residual path is the expensive one at `B*S = 4096`.
