# Verification Report: layer_norm_tile

## Code Review

Reviewed `layer_norm_tile.py`, `layer_norm_tile_program_descriptor.py`, the three kernels, `op_design.md` and `l1_ledger.md`.

Fixed:
- `l1_ledger.md` symbol table named the output-depth constant `OUT_DEPTH_BLOCKS`; the code's constant is `OUT_DEPTH`. Renamed in the ledger so the next stage plans against the real name.

No other defects found:
- **Prompt rules**: `eval/prompts/layer_norm_tile_v2.txt` has no `## Rules` section. Its hard constraints hold: TILE-only input, no host-side `to_layout` / `pad` / `slice`, one `ttnn.generic_op` per call (the residual add runs in-kernel), `memory_config` only selects output placement, and ValueError on non-bf16 / non-TILE input.
- **Kernel hygiene**: `void kernel_main()`, `api/dataflow/dataflow_api.h` includes, TensorAccessor everywhere (no `InterleavedAddrGen`).
- **Helper usage**: every compute phase is a kernel-lib helper (`add` / `sub` / `mul` convenience, `reduce_mean` and `reduce` with `AccumulateViaAdd`, `eltwise_chain` with PerRow DEST accumulation for the sum of squares). The 1/W, +eps, rsqrt post-op is fused into the reduce callback (catalog `compute_fusion`: SFPU consumers fuse). There is no multicast, so the multicast helper does not apply.
- **CB sync**: push count equals wait/pop count for every CB per block (design CB-sync table checked against the kernels). Gamma/beta are pushed once and popped once at kernel end.
- **Broadcast**: Col broadcast for mean / inv_std (REDUCE_ROW col-0 outputs), Row broadcast for gamma/beta (row-0 staged). Only row 0 of each affine tile is read from the tensor (64 B reads), so no tile of repeated data is filled.

## Design Conformance

- **Algorithm**: matches. Two-pass centered variance (mean, then sum of `(z - mean)^2`), whole row resident, no re-read of activations.
- **Pipeline / RISC ownership**: reader on NoC0 owns x, residual, gamma and beta; writer on NoC1 owns output. Matches.
- **Work distribution**: `split_work_to_cores(grid, tensor_row_tiles, row_wise=True)` over `compute_with_storage_grid_size()`, per-image `ceil(H/32)` row count. 110 of 110 cores are active on the headline. Both dataflow halves are batched: reader barrier per C = 8 chunk, writer flush per chunk, one final write barrier.
- **Blocking-model fidelity**: R, C and `in_depth` come from `select_blocking()` only and reach the reader, compute and writer as CT args from that one `Blocking`. CB page counts come from one function, `_cb_pages`, which is used both for the L1-fit solve and for the allocation. No collapsed knob or duplicated block literal. The only whole-op dimension in a capacity expression is `Wt`, which the design makes resident on purpose (the dependent `w` axis). Its bound is the R1 predicate `F(1) <= budget`, checked on the host. R = 1 on the headline comes from the `MIN_BLOCKS_PER_CORE = 2` overlap choice (design lamp), not from collapsed sizing.
- **Expression**: the reader and writer move C-tile chunks per synchronization, and compute phases are single helper calls over `grid(R_b, Wt)`. The per-tile CB credit inside `residual_add_block` / `center_block` is helper-internal streaming against a block-sized live set. It is not a per-unit completion boundary at the schedule level.
- **Inter-core communication**: none (R1). Matches.

## Registry Conformance

- `INPUT_TAGGERS = {"alignment": tag_alignment}`, signature `(inputs, axes)`.
- `SUPPORTED` declares all 10 axes. It equals `feature_spec.TARGET` on every axis.
- `EXCLUSIONS = []`.
- `validate()` checks the ValueError contract first, then SUPPORTED per-axis (`UnsupportedAxisValue`), then EXCLUSIONS (`ExcludedCell`), then the non-axis placement and mechanism gates. The entry point resolves the default config and calls `validate()` before any allocation or kernel work.
- The op file does not declare `INVALID`.
- No drift fixes were needed: zero XPASS, zero wrong-mode.

INVALID audit (`eval/golden_tests/layer_norm_tile/feature_spec.py`): two entries, `{dtype: bf8b, layout: ROW_MAJOR}` and `{affine_dtype: bf8b, affine_layout: ROW_MAJOR}`. Each couples axes of a single tensor and is a structural impossibility (bf8b has no row-major form). The canonical bf8b + ROW_MAJOR activation entry is present. Neither can fire under the bf16-only TARGET; they guard a future dtype expansion. Gamma and beta are required, so the no-weight canonicalization rule does not apply. No issues.

## L1 Ledger Audit

- **Currency**: every declared CB (0-8, 16) has a row, and every row is a live CB. Size expressions match `_cb_pages` / `_cb_page_formats` exactly. Work A/B is one `CBDescriptor` with two format descriptors, `total_size = R * Wt * wb`. Production check: F(1) = 444 KiB (fp32 DEST, residual, `in_depth = 2`), matching the ledger.
- **Capacity vs live set**: `cb_input_tiles` / `cb_residual_tiles` on the residual path hold `in_depth` blocks while the live set is streamed. The ledger states this as the prefetch pipelining decision; accepted. `cb_output_tiles` is one block, with the writer draining C at a time while compute moves to the next block (pipelining, stated). Every CB whose live set spans `row` and `w` (input, work pair, output) scales with both R and Wt, and the stat CBs scale with R only. No collapsed extent and no inflated window.
- **Page format vs DEST**: work and stat CBs are Float32 iff `fp32_dest_acc_en`, otherwise Float16_b. IO CBs are Float16_b (bf16 tensors, not packed intermediates). Correct in both directions.
- **Disjoint lifetimes**: `cb_row_mean` / `cb_row_sumsq` / `cb_inv_std` are disjoint or partially disjoint. Each pair carries a concrete reason for separate ids: a shared FIFO would put the second indexed window behind the first and wrap on the ragged last block. Accepted.
- **Bounds**: `Wt`, R, `in_depth`, C, `has_res` and `wb` are all in the symbol table with their establishing predicates, and the footprint is closed-form.
- **Traffic budget**: present and consistent with the code (x, residual and out each cross once; gamma/beta are sparse re-reads per core). The cheapest-traffic split, R1, is implemented. R3 (tied bytes, 2x transactions) and R2 (more bytes) are deferred regime rows with positive reasons.
- **Block-size defaults**: interleaved path spreads tile-rows over the full grid, then takes the coarsest R that fits under `MIN_BLOCKS_PER_CORE = 2`. The planner flags this departure from "coarsest block" as a measurable lamp. Refinement 2 measures it.
- **Footprint**: `F(R) = R*Wt*(in_depth*tb*(1+has_res) + wb + out_depth*tb) + 3*R*wb + 2*Wt*tb`. Everything except the affine term is linear in R; the first term scales with `in_depth` and the residual; `wb` doubles the work and stat terms under fp32 DEST.

## Precision Baseline

`tests/ttnn/unit_tests/operations/layer_norm_tile/test_layer_norm_tile_precision_baseline.py`: randn inputs, gamma/beta ROW_MAJOR in L1, HiFi4, eps 1e-5, against fp32 torch `layer_norm`.

| Shape | Residual | fp32 DEST | PCC | Max Abs Err | Mean Abs Err | Rel RMS Err | Mean bf16 ULP | got/true median [p5, p95] |
|---|---|---|---|---|---|---|---|---|
| (32,1,32,768) | yes | on | 0.9999983 | 0.043 | 0.0017 | 0.0019 | 0.98 | 0.9999 [0.9959, 1.0039] |
| (32,1,32,768) | yes | off | 0.9999909 | 0.176 | 0.0042 | 0.0052 | 3.99 | 1.0016 [0.9906, 1.0159] |
| (64,1,17,768) | yes | on | 0.9999983 | 0.041 | 0.0018 | 0.0019 | 1.18 | 0.9999 [0.9958, 1.0040] |
| (64,1,17,768) | no | off | 0.9999919 | 0.186 | 0.0041 | 0.0050 | 4.33 | 1.0016 [0.9906, 1.0157] |
| (3,1,500,768) | yes | on | 0.9999983 | 0.046 | 0.0018 | 0.0019 | 1.11 | 1.0000 [0.9960, 1.0040] |
| (3,1,500,768) | no | off | 0.9999918 | 0.191 | 0.0041 | 0.0051 | 10.3 | 1.0016 [0.9906, 1.0154] |
| (8,1,512,768) | yes | on | 0.9999983 | 0.052 | 0.0018 | 0.0019 | 1.28 | 0.9999 [0.9957, 1.0040] |
| (8,1,512,768) | no | on | 0.9999983 | 0.052 | 0.0017 | 0.0019 | 1.79 | 0.9999 [0.9958, 1.0039] |
| (8,1,512,768) | yes | off | 0.9999909 | 0.151 | 0.0043 | 0.0052 | 4.13 | 1.0016 [0.9902, 1.0167] |

(16 cells total in the test; the rest fall in the same bands.) Max ULP is not reported: near-zero reference values give unbounded ULP counts, so max abs is the meaningful worst case.

**Assessment**: fp32 DEST runs at bf16 output quality (mean ~1 ULP, rel-RMS 0.0019). bf16 DEST costs about 2.7x in rel-RMS (0.005) because `z`, `c`, the stats and `inv_std` are rounded to bf16 between phases. Scale-bug check: the fp32-DEST ratio is centered at 1.0000. The bf16-DEST median sits at 1.0016 with a wide p5-p95 spread (+-1.3%), which is consistent with bf16 rounding of the per-row `inv_std` (0.39% half-step), not a uniform scale error. No bug.

**Recommended tolerances**: fp32 DEST PCC >= 0.999, rel-RMS <= 0.02. bf16 DEST PCC >= 0.998, rel-RMS <= 0.05. These are the golden `TOLERANCES`, which are already tight relative to the measurements.

## Verifier CLI Summary

Golden run: `eval/eval_test_runner.sh eval/golden_tests/layer_norm_tile/` gave 545 / 545 passed, 0 hangs. Report: `verifier_results/verifier_report.json`.

- supported_pass: 524
- xfail_expected: 0
- invalid_skipped: 0
- no_axes_found: 21 (`test_regression.py`: magnitude / mean-offset / uniform / cancelling-residual / epsilon / LoFi probes; all passed. They call the op without the `observed()` wrapper, so they carry no axes.)
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- Axis-tagging consistency report: empty (no divergence).

Acceptance suite: 56 / 56 passed.

`xfail_expected = 0` is correct: SUPPORTED equals TARGET, so no cell is out of support.

## Performance Baseline

Tracy device kernel duration, two runs, compared with stock `ttnn.layer_norm`:

| Row | Ours (us) | Stock (us) | Status |
|---|---|---|---|
| 8x512 in L1 / res DRAM / out DRAM (flagged) | 55.8, 55.0 | 55.0 | tie |
| 8x512 all DRAM (flagged) | 66.6, 66.9 | 70.0 | win |
| 8x512 DRAM no residual (flagged) | 41.3, 40.1 | 48.3 | win |
| 8x512 all L1 | 47.5, 49.1 | 43.3 | **loses** |
| 8x512 L1 no residual | 32.5, 33.2 | 34.5 | thin win |
| 8x384 all DRAM | 48.0, 48.4 | 47.0 | **loses** |
| 32x32 all DRAM | 22.3, 22.6 | 24.8 | win |
| 8x512 all DRAM, affine TILE DRAM | 90.2, 89.2 | 125.4 | win |

Diagnostics:
- **Skip-compute ablation** (`CKL_ELTWISE_CHAIN_SKIP_COMPUTE=1`, applied temporarily and reverted): 48.7 / 60.6 / 39.8 / 36.2 / 23.5 / 45.3 / 17.8 / 81.1 us. The DRAM rows are data-movement bound at ~310 GB/s aggregate, against the ~470 GB/s the 40 us floor implies. The L1 rows put 9-11 us of compute on the critical path.
- **Tail probe** (`probes/probe_tail.py`, T = 110 vs T = 128 tile-rows, 110 cores): all-L1 41.4 vs 47.8 / 48.7, in-L1 45.9 vs 54.6 / 54.7, all-DRAM 56.3 vs 66.8 / 66.4. That is 1.17-1.19x time for 1.16x bytes, so the grid is aggregate-throughput bound. The 18 two-row cores do not set the kernel end.

The queue (`op_requirements.md`) targets the DM side first (Refinement 1, flagged DRAM rows, 8x384) and then the compute side (Refinement 2, flagged in-L1 row, all-L1 row).

## TARGET coverage

`TARGET - SUPPORTED` is empty on every axis, so no generality refinement exists. Deferred design regimes not filed as refinements, each with its reason:
- **R3 `half_band_split`** (16-row bands for balance and all-pad skipping): the tail probe shows the op is aggregate-bound, not tail-bound. Packing two bands per compute tile also leaves the max compute per core at 2 tile-rows on the headline. No headroom on the flagged shapes.
- **R2 `w_split_combine`** (cross-core Chan combine): it only fills cores on `tensor_row_tiles < grid` shapes. Every flagged shape has T = 128 >= 110. The one under-filled reference row (32x32) already beats stock (22.3 vs 24.8 us). Candidate if a future flagged case is under-filled.
- **R4 `affine_broadcast`** (multicast gamma/beta): sparse 64 B row-0 reads, about 0.34 MB on the headline (under 3.5% of bytes), issued behind block 0's reads. Production gamma/beta are in L1, so there are no DRAM bytes to save. Not worth a rendezvous on every core.
- **Sharded `memory_layout`**: not in TARGET.

## Recommendations

- The prompt requires beating every reference row. Two rows lose today (all-L1, 8x384) and flagged #1 only ties. Refinements 1 and 2 carry these as Done-when gates, so they are not optional polish.
- Run-to-run noise is about 1-1.5 us on these rows. Perf acceptance should compare multi-run medians, and a stock comparison should clear noise (>= 2 us), not win by a fraction of a microsecond.
- The golden `test_regression.py` tests do not go through `axes.observed`, so they land in `no_axes_found`. This is harmless, but it hides them from per-cell dashboards. Wrapping them would make them visible (eval-suite change, outside the op).
- Precision: no concrete lever is needed. bf16 DEST is 2.7x worse in rel-RMS than fp32 DEST but within its tolerance band.
