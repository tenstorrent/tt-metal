# Verification Report: layer_norm_tile

Board: Blackhole p100a, 110-core compute grid (11 x 10). Op: `ttnn.operations.layer_norm_tile`, re-exported as
`ttnn.operations.layer_norm` (shim `ttnn/ttnn/operations/layer_norm/__init__.py`).

This is the second verifier pass. The first pass (`b63b3af6a64`) verified the `row_resident` build. After it,
the implementer landed NoC work that the first queue had filed as Refinement 2 levers (a) and (d), without
updating the queue:
- `block_height` is fixed at 1 tile-row (`BLOCK_HEIGHT_MAX = 1`) with reader run-ahead up to
  `INPUT_DEPTH_MAX = 4` tile-rows.
- From `R >= 64` tile-rows, the residual stream moves to the writer RISC on NoC1.
- Without residual, with gamma/beta in L1, cores that own >= 2 tile-rows alternate x reads over NoC0 / NoC1
  (`DM_DYNAMIC_NOC`).
- A stage-ablation harness (`LAYER_NORM_ABLATE_*` defines, `test_layer_norm_tile_ablation.py`).

This pass re-verifies that state, re-measures, and re-files the queue against the new baseline.

## Golden suite status (blocks the mechanical check)

`eval/golden_tests/layer_norm_tile/` still does not exist: there is no `feature_spec.py` and no
`test_golden.py`. `eval/eval_test_runner.sh` and `python3 -m eval.verify_supported` cannot run, so **no
`verifier_report.json` was produced**. The verifier does not author `feature_spec.py`.

The SUPPORTED-vs-behaviour check below is manual. It is driven by the local suites: acceptance 80, extended
12, precision 16, perf 60, ablation 64.

**Action for the user**: run `/golden-tests` for `layer_norm_tile`, then re-run the verifier CLI. The handoff
notes from the first pass still apply (see "Handoff to /golden-tests").

## Code Review

Fixed in place:

| # | Finding | Fix |
|---|---------|-----|
| 1 | Dead knob `AFFINE_FETCH_ON_NOC0 = False` ("parked off", measured slower). It was still plumbed as writer CT arg 15 and a `noc` template parameter through `issue_affine_reads`. The writer comment described the parked path as live behaviour ("NoC0 ... when NoC1 carries input traffic"). | Removed the constant, the CT arg, the template parameter and the misleading comment. Gamma/beta reads use the writer's `noc_index`, which is what always ran. The measurement stays recorded here: 8x1x512 res tile_dram 41.5 -> 51.4 us with it on. |
| 2 | `create_program_descriptor` recomputed `residual_on_writer = has_residual and blocking.residual_on_writer`. `derive_blocking()` already folds `has_residual` into the field. | Reads `blocking.residual_on_writer` directly (one source). |
| 3 | `Blocking.dynamic_noc` (a `@property`) sat between dataclass fields. | Moved below the fields. |
| 4 | Ledger symbol table: the `Wt` bound (`<= 38` / `<= 47`) and its predicate (`block_height_cap(INPUT_DEPTH_MAX) < 1`) described the planner's formula. The code refuses on `block_height_cap(1) < 1`. | Restated from the code, verified numerically with `_block_height_cap`: `Wt <= 47` (fp32 + residual), `<= 54` (fp32, no residual / bf16 + residual), `<= 63` (bf16, no residual). |
| 5 | Ledger: `BLOCK_HEIGHT_MAX = 1` departs from the design's whole-assignment block default with no measurement recorded. The `affine_mcast` line called the affine fetch "off the critical path", which no longer holds for TILE gamma/beta in DRAM. | Both rows restated with this pass's measurements (sweep and tile_dram vs rm_l1 gap, below). |

After the fixes: 232/232 local tests pass (prod), and acceptance + extended pass 92/92 under `--dev` (watcher,
NoC sanitizer).

Checked, no change needed:

- **Prompt contract** (`eval/prompts/layer_norm_tile.txt`, no `## Rules` section; its MUST statements apply):
  - TILE-only input, with no host-side `to_layout` / `tilize` / `pad` / `slice`.
  - Non-aligned S is handled in-kernel (padded rows compute finite values and are written as whole tiles).
  - Non-bf16 raises `ValueError`.
  - `ttnn.operations.layer_norm` import path.
  - Exactly one `ttnn.generic_op` per call; regime and NoC choices are all inside the one descriptor.
- **Helper usage**: unchanged since pass 1. Every compute phase is one helper call per block, and the
  `rstd_block` post-op uses raw SFPU (a DEST-index callable; the design's "Helpers considered and rejected").
  There is no multicast or semaphore code.
- **CB sync balance**: re-checked with the residual producer now switchable between the reader and the writer.

  | CB | Producer | Consumer |
  |----|----------|----------|
  | `cb_residual_tiles`, writer path | Writer reserves/pushes `Wt` per tile-row. It tops up to `min(r + input_depth * block_height, core_rows)` before waiting on output row `r`. | Compute pops per tile in `add_residual_block`. |
  | `cb_input_tiles`, `cb_output_tiles`, `work` / `stats` alias groups, gamma / beta | unchanged | unchanged |

  Writer-path deadlock check: reserving residual row `r + d - 1` (capacity `d = input_depth * block_height`
  rows) needs row `r - 1` popped. Compute popped it in block `r - 1`'s `add_residual_block`, before it
  produced output row `r - 1`, which the writer has already drained. The writer pushes all residual rows of
  block `b` before waiting on block `b`'s first output row. Holds for any `block_height`, not only 1.
- **Dynamic-NoC path**:
  - The reader issues x reads on `col & 1` when the per-core `split_x` RT arg is set, and waits for both NoCs
    when `dynamic_noc` is on. Cores with one row read on one NoC (measured: the NoC1 half lands late on a
    single latency-bound burst).
  - The writer keeps NoC1 for output and gamma/beta.
  - Clean under `--dev` (NoC sanitizer) on every acceptance shape that takes the path: `(16,1,256)` and
    `(8,1,512)` with rm_l1 affine.
- **API hygiene**: `TensorAccessor` throughout, `void kernel_main()`, `api/dataflow/dataflow_api.h`.
- **Broadcast efficiency**: unchanged. Col broadcast for the stats; Row broadcast for gamma/beta, of which only
  row 0 of faces 0 and 1 is fetched.

Noted, not changed:

- **Ablation hooks in the production kernels.** `#ifndef LAYER_NORM_ABLATE_{READS,WRITES}` and
  `#ifdef LAYER_NORM_ABLATE_COMPUTE` compile out unless the ablation test injects the define. They stay
  because the perf refinements below use the harness to attribute time.
- **Harness pitfall** (implementer, re-confirmed): the `generic_op` program cache key ignores kernel
  `defines`. A cached program from another variant would run silently, so the ablation fixture clears the
  cache per case. This is an infra issue, not an op issue.
- **Unverified off this board** (cannot be checked here): `sumsq_block` accumulates ELWMUL into DEST
  (`DestAccumulation::PerRow`), and the dynamic-NoC split depends on the arch. Both are verified on Blackhole
  only.

## Design Conformance / Blocking Model

- **Algorithm**: matches the design (two-pass centered variance on a resident row; residual folded once into
  `cb_xsum`; `1/W` from the logical W).
- **Pipeline topology**: the design gives the reader both input streams. The build gives the residual stream
  to the writer from `R >= RESIDUAL_SPLIT_MIN_TILE_ROWS (64)`, and splits x over both NoCs without residual
  when the affine tensors are in L1. Each CB still has exactly one producer. This departure is the
  design-sanctioned lever (Dataflow Strategy and the pass-1 queue's NoC levers), and it is measured:

  | Shape (rm_l1, default config, median of 3, us) | Residual on writer / NoC1 | Residual on reader / NoC0 |
  |---|---|---|
  | `(8,1,512)` res | 35.9 | 59.5 |
  | `(256,1,16)` res | 54.6 | 94.5 |

  At R = 32 the residual stays on the reader. The implementer measured the split as slower there (17.4-18.3
  -> 19.2-20.6 us).
- **Work distribution**: `split_work_to_cores(grid, R, row_wise=True)` over the full runtime grid. At R = 32
  / 64 only 32 / 64 of 110 cores are busy. This is the design's `w_split_combine` (Refinement 1).
- **Knob fidelity / DRY**:
  - Each of `block_height`, `input_depth`, `OUTPUT_DEPTH`, `CB_L1_BUDGET_BYTES`,
    `RESIDUAL_SPLIT_MIN_TILE_ROWS`, `X_NOC_SPLIT_MIN_CORE_ROWS` and the per-core assignment has one source
    (`derive_blocking()` / module constants).
  - Every CB size and the L1 cap read `_cb_plan()`.
  - The writer's residual depth CT arg is derived from the plan (`plan["residual"][0] // Wt`), not restated.
  - Fix #2 removed the one redundant derivation.
- **Block size (half-turned split check)**: `block_height = 1` with `tile_rows` spread across cores looks
  like a half-turned knob. It is not:
  - The block is a whole `1 x Wt` row (24 tiles), and each phase is one helper call over it.
  - The departure from the whole-assignment default is measurement-justified, not left at a minimum. The
    verifier re-measured it on the current NoC scheme (`BLOCK_HEIGHT_MAX` 1 vs 2, rm_l1, default, median of
    3):

    | Shape | `block_height = 1` (us) | `block_height = 2` (us) |
    |---|---|---|
    | R=128 nores | 30.7 | 31.6 |
    | R=128 res | 35.9 | 40.0 |
    | R=256 nores | 40.4 | 46.6 |
    | R=256 res | 54.6 | 70.2 |

  - Compute starts on the first landed tile-row. This is the design's own "Overlap" perf lamp.
- **Expression**: every kernel has one reserve/wait, `Wt` NoC transactions, one barrier or flush, and one
  push/pop per tile-row. The block is one tile-row, so the planned granularity and the handshake coincide.
  Not an expression failure.
- **Axis accounting**: ledger span/stream claims hold. `cb_input_tiles` / `cb_residual_tiles` span
  `input_depth` tile-rows as reader run-ahead; their consumer-side live set is smaller (see L1 audit).
- **Design document staleness** (left as the planner's record, not edited):
  - `op_design.md` still states `INPUT_DEPTH_MAX = 2`, a whole-assignment block, and a reader-only residual.
  - Its `affine_mcast` deferral reason ("off the critical path") no longer holds for DRAM gamma/beta.
  - `op_requirements.md` Phase 0 states the current regime precisely and is the reference for the next
    implementer.

## Registry Conformance

- `INPUT_TAGGERS = {"alignment": tag_alignment}`, with the `(inputs, axes)` signature.
- `SUPPORTED`:
  - `dtype` [bf16], `fp32_dest_acc_en` [True, False], `layout` [TILE].
  - `alignment` [tile_aligned, h_non_aligned], `memory_layout` [INTERLEAVED].
  - `residual` [none, present], `affine_layout` [TILE, ROW_MAJOR, mixed].
- `EXCLUSIONS = []`.
- `validate()` order:
  1. The universe check (`ValueError`).
  2. SUPPORTED per axis (`UnsupportedAxisValue`).
  3. EXCLUSIONS (`ExcludedCell`).
  4. The affine stick mechanism cap (`UnsupportedAxisValue`).
- The entry point calls `validate()` first. The op file does not declare `INVALID`.
- No drift found this pass. Every refusal test raises the intended type.

## INVALID audit

Not possible: `feature_spec.py` is absent.

## Handoff to /golden-tests

Unchanged from pass 1:
- Axes: `dtype` [bf16], `fp32_dest_acc_en` [True, False], `layout` [TILE], `alignment` [tile_aligned,
  h_non_aligned], `memory_layout` [INTERLEAVED], `residual` [none, present], `affine_layout` [TILE,
  ROW_MAJOR] (also `mixed`).
- INPUTS from the contract: `(B, 1, S, 768)`, `1024 <= B*S <= 4096`, `S <= 512`, power-of-two and
  non-aligned S.
- Flag one `LOOSE_CASE` with `attention:` as the perf target, e.g. `(8, 1, 512, 768)` with residual.

## L1 Ledger Audit

1. **Currency**: 7 allocations with residual, 6 without, matching `_cb_plan()`. Fixed: the `Wt` bound
   (#4) and the `BLOCK_HEIGHT_MAX` and `affine_mcast` rows (#5). Producer columns are correct: the
   `cb_residual_tiles` row names reader or writer by `RESIDUAL_SPLIT_MIN_TILE_ROWS`.
2. **Capacity vs live set**:
   - *Over*: `cb_input_tiles` / `cb_residual_tiles` hold `input_depth * Wt` (up to 4 tile-rows), while
     compute's live set is 1 row (no residual) or 1 tile (residual). The stated reason is reader run-ahead.
     It is now measured: the run-ahead plus `block_height = 1` is the R=256 res 94.5 -> 54.6 us win, combined
     with the residual split. Justified.
   - `cb_output_tiles` at 2 tile-rows is the compute/writer overlap window. Justified.
   - *Under*: none.
3. **Page format vs DEST width**: `work` / `stats` are Float32 if and only if `fp32_dest_acc_en`, else
   Float16_b. I/O and affine CBs are bf16 tensor data. Clean in both directions.
4. **Disjoint lifetimes**: every non-shared pair has a recorded reason (reader- or writer-produced vs
   compute-produced; kernel-lifetime affine rows; output window concurrent with the next row).
5. **Bounds / closed form**: total = `block_height * (Wt * (input_depth * T_in * (1 + HAS_RESIDUAL) + T_acc) +
   T_acc) + Wt * T_in * (2 + OUTPUT_DEPTH)`.
   - Scales with `input_depth`, `HAS_RESIDUAL` and `T_acc`. `block_height` is 1 today.
   - Worst built case: 692,224 B (fp32 + residual, `input_depth = 4`, R > 330) against the 786,432 B budget.
   - Every symbol is bounded.

**Data-movement budget**: consistent with the code.
- Input, residual and output each cross the NoC once. The NoC assignment section matches the descriptor.
- Cheapest-traffic split (`tile_rows`): implemented.
- Above-minimum term: the per-core affine fetch, queued as Refinement 2 with a positive, measured reason (fix
  #5).

**Block-size defaults**: interleaved spreads across the full grid (held). The coarsest-block default is
departed from by measurement (see Design Conformance).

## Precision Baseline

`tests/ttnn/unit_tests/operations/layer_norm_tile/test_layer_norm_tile_precision_baseline.py`. Seed 1234,
`randn` input / residual / gamma / beta, fp32 torch reference, HiFi4. `nores` = tile_dram affine without
residual; `res` = rm_l1 affine with residual. Re-run this pass: the numbers match pass 1 to 4 digits, as
expected (the NoC work changes no math order).

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
| | res | fp32 | 0.9999983 | 0.0495 | 0.00180 | 0.00189 | 2.48 | 6.1 | 0.99988 [0.9958, 1.0040] |
| | res | bf16 | 0.9999904 | 0.2058 | 0.00451 | 0.00545 | 3.20 | 23.9 | 1.0017 [0.9900, 1.0168] |

**Assessment**:
- fp32 DEST sits at the bf16 output rounding floor: rel-RMS 0.19%, p99 6 ULP.
- bf16 DEST costs about 2.7x in RMS (0.51-0.55%).
- Residual, non-aligned S, the writer-side residual stream and the split-NoC reads do not move precision.
- Scale-bug check: the got/true ratio is centered on 1.0 with a ±0.4% (fp32) / ±1.3% (bf16) spread. That is
  noise, not a constant offset. The bf16-DEST median of 1.0016-1.0017 is a small bias (~0.2x the spread) from
  rounding the variance to bf16 before `rsqrt`.

**Recommended tolerances**:
- fp32 DEST: PCC >= 0.9999, rel-RMS <= 0.003, atol 0.08.
- bf16 DEST: PCC >= 0.9999, rel-RMS <= 0.008, atol 0.3.
- rtol is not meaningful: references cross zero through beta.

## Performance baseline

`test_layer_norm_tile_perf.py --profile`, `DEVICE KERNEL DURATION`. Each case is the min of 2 dispatches;
the table shows the median of 3 file invocations, in us. Pass-1 numbers are in parentheses where they
changed materially.

| Shape (tile-rows, max rows/core) | Affine | nores default | nores fp32 off | nores LoFi | res default | res fp32 off | res LoFi |
|---|---|---|---|---|---|---|---|
| (32,1,32) R=32, 1 | tile_dram | 14.0 | 13.8 | 12.3 | 17.9 | 18.0 | 16.3 |
| | rm_l1 | 14.0 | 13.8 | 12.4 | 17.6 | 17.5 | 16.0 |
| (16,1,128) R=64, 1 | tile_dram | 19.8 | 19.6 | 19.0 | 35.0* | 33.7* | 33.1* |
| | rm_l1 | 19.2 | 20.4 | 16.1 | 32.7* | 32.9* | 31.3* |
| (8,1,512) R=128, 2 | tile_dram | 33.6 | 32.4 | 28.8 | 41.1 (58.2) | 41.1 | 39.1 |
| | rm_l1 | 31.5 | 30.4 | 28.7 | 37.2 (59.8) | 35.8 | 32.8 |
| (124,1,33) R=248, 3 | tile_dram | 47.2 (53.4) | 47.9 | 43.7 | 54.4 (95.9) | 54.8 | 52.1 |
| | rm_l1 | 40.3 (54.9) | 39.1 | 35.1 | 53.1 (93.6) | 51.3 | 48.1 |
| (256,1,16) R=256, 3 | tile_dram | 49.7 (60.8) | 50.0 | 45.1 | 57.3 (110.5) | 55.5 | 57.0 |
| | rm_l1 | 41.0 (60.2) | 38.7 | 35.6 | 59.6 (96.5) | 59.1 | 54.6 |

\* **Bimodal**: R=64 with residual lands at either ~26 us or ~33 us from invocation to invocation, with
unchanged code and knobs (sweep: 22.8 / 25.6 / 26.1 / 31.1-32.5). Treat that cell as a ±15% band.

Stage ablation (`test_layer_norm_tile_ablation.py --profile`, rm_l1, default config, min of 2 invocations,
us):

| Shape | Residual | full | reads_only | compute_only | writes_only | floor |
|---|---|---|---|---|---|---|
| (32,1,32) R=32 | no | 13.9 | 2.6 | 10.2 | 4.7 | 2.6 |
| | yes | 18.6 | 4.5 | 12.0 | 4.7 | 2.6 |
| (16,1,128) R=64 | no | 20.1 | 5.5 | 10.3 | 6.9 | 2.9 |
| | yes | 26.8 | 19.3 | 11.9 | 6.9 | 3.0 |
| (8,1,512) R=128 | no | 30.5 | 10.7 | 19.8 | 13.0 | 5.0 |
| | yes | 36.4 | 18.2 | 23.2 | 12.6 | 4.6 |
| (256,1,16) R=256 | no | 39.4 | 14.4 | 29.3 | 20.8 | 5.0 |
| | yes | 55.6 | 31.6 | 34.3 | 20.4 | 4.8 |

Reading:
- **R = 32 / 64 are compute-bound and under-occupied.** Net compute is about 7.6 us per tile-row without
  residual and 9.4 us with it (`compute_only - floor`). 78 / 46 cores are idle. Lever: Refinement 1
  (`w_split_combine`).
- **Reads and compute still serialize on multi-row cores without residual.**
  - R=128: `full` 30.5 ≈ `reads_only` + `compute_only` (10.7 + 19.8), against an overlapped ideal near 20.
  - R=256: `full` 39.4 vs 14.4 + 29.3.
  - Compute waits on a whole tile-row while every core's first-row reads contend grid-wide. Lever: Refinement 3.
- **DRAM gamma/beta now costs time.** The x NoC split is gated off when the affine tensors are in DRAM,
  because dynamic-NoC mode slows the writer's DRAM fetch. Every core reads the same 24 DRAM pages. Measured
  gap, tile_dram vs rm_l1:
  - R=248 nores: +6.9 us.
  - R=256 nores: +8.7 us.
  - R=128 res: +3.9 us.

  The design's `affine_mcast` deferral ("off the critical path") no longer holds. Lever: Refinement 2.
- **Fidelity / DEST width are minor terms.** LoFi saves 1.7 us at R=32 (12%). `fp32_dest_acc_en=False` is
  within noise.

## Verifier CLI Summary

Not run: no golden suite exists. Manual equivalent:

| Category | Count | Note |
|----------|-------|------|
| supported_pass | 232 / 232 local tests (prod); 92 / 92 acceptance + extended under `--dev` | acceptance 80, extended 12, precision 16, perf 60, ablation 64 |
| xfail_expected | 5 refusal tests | float32 input (`ValueError`, prompt-mandated), RM input, `w_non_aligned`, RM stick width, HEIGHT_SHARDED; each raises the intended type |
| supported_fail | 0 | |
| xpass_drift | 0 | |
| xfail_wrong_mode | 0 | |

## TARGET - SUPPORTED accounting

TARGET is the design's "Expected TARGET axes" table, because `feature_spec.py` is absent. Every TARGET value
is in SUPPORTED, so the queue holds only measured perf refinements.

Documented omissions (outside TARGET, not queued):

| Value | Reason |
|-------|--------|
| `memory_layout` HEIGHT / WIDTH / BLOCK sharded | The requirement scope is L1 interleaved. The design keeps them as deferred regime rows. Promote them if `/golden-tests` adds sharded values to TARGET. |
| dtype other than bf16; ROW_MAJOR input | Excluded by the prompt (`ValueError`). |
| `alignment = w_non_aligned` | Unreachable under the shape contract (`W = 768`). Refused with a typed refusal. |
| `half_tile_rows` regime | Needs 16x32 tile handling in the compute helpers: a helper capability, not an op refinement. |

## Recommendations

- Run `/golden-tests` first, so the next verifier pass has a mechanical SUPPORTED check and a pinned perf
  target.
- Queue order:
  1. `w_split_combine` (T3 scheme-change).
  2. `affine_mcast` (T3 reuse multicast; its slice shape depends on 1).
  3. Read/compute overlap (T2).
  4. Compute trims (T1/T2).

  Refinement 1 redefines the per-core block every later refinement tunes.
- Production callers: with gamma/beta in L1 as ROW_MAJOR, today's build is up to 18% faster on the
  multi-row shapes (the x NoC split engages). It is equal at R <= 64.
