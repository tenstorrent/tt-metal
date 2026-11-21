# Verification Report: rms_norm

Target: Quasar (`quasar`) on the ttsim simulator (no clock). Specification: `eval/specs/rms_norm/spec.py` (sha256 1c6520a3…). Specification lint (`python3 -m eval.validate_contract rms_norm --phase0`): **passed**, 0 errors, 6 unresolved items, all in the specification (`tt_regen_spec.governs`, `NUMERIC.special_values` / `freedoms`, `PERFORMANCE.bar` / `measure`, no numeric-list entry for 16-bit-DEST cells: graded by `legacy_tolerances`). None blocks this phase.

## Code Review

### Fixed in place

| # | Finding | Fix |
|---|---------|-----|
| 1 | DRY: the reader restated the gamma ring's entry geometry as bf16 literals (`gamma_page_bytes = 32 * sizeof(uint16_t)`, `face1_row0_offset = 512`) that the host derives from the ring format; the host wrote the bf16 gamma and scaler formats twice each (ring size and ring format). A gamma-format change would have had to land in four places. | One host constant per mechanism-fixed format (`GAMMA_FORMAT`, `SCALER_FORMAT`) feeds both the ring size and the ring format; the reader takes `gamma_page_bytes` (the weight's page size) and `gamma_face_bytes` (entry size / 4) as CT args, with a `static_assert` that a weight page is two face rows of the entry format — a 128-B float32 page now fails at compile time instead of assembling wrong gammas. |
| 2 | DRY: `TILE = 32` defined in both `rms_norm.py` and `rms_norm_program_artifacts.py`. | `rms_norm.py` imports it. |
| 3 | Ledger currency: `row_stat` / `row_sum` capacities stated as `1 · C_eff`; the code allocates `ROWS_PER_BLOCK · C_eff` (equal only at the Phase 0 knob value). Footprint line hard-coded the stat page as 4,096 B. | `l1_ledger.md` corrected (`rows_per_block · C_eff`, `2 · rows_per_block · stat_bytes`). |

All tests (37 unit/plan/baseline, golden suite) re-run green after the changes.

### Checked, no finding

- **Kernel syntax / includes / accessors**: `void kernel_main()`, `api/dataflow/...` includes, `TensorAccessor(tensor::…)` everywhere; no `InterleavedAddrGen`.
- **Ring sync balance** (per NEO per round): input — Wt entries produced (implicit sync), Wt consumed (cumulative wait, no pop in `sum_squares`; per-tile pop in `normalize`); gamma — Wt pushed in `Wt / gamma_block_tiles` blocks, Wt popped per tile; scaler — 1 pushed per NEO, popped once after the last round (`reduce<>` never pops it); `row_stat` — `rows_per_block` pushed by `square`, popped by `reduce`, pushed by the finalize chain, popped `AtEnd` by `normalize`; `row_sum` — pushed by `reduce`, popped by the chain; output — Wt pushed per tile, Wt written (implicit sync). Padded last rounds push/pop the same counts (every NEO takes the same entries).
- **Helper usage**: every compute phase is a kernel-library helper (`square`, `reduce`, `eltwise_chain`); the reduce scaler uses `calculate_and_prepare_reduce_scaler`. The raw dataflow is single tagged transfers with no block loop a helper would replace (design, *Helpers considered and rejected*). The `pack_init` before each chain whose output ring changes is the required call order, not a smell.
- **Broadcasts**: `x · inv_rms` is an FPU **Col** broadcast of a one-tile statistic; gamma is a **Row** broadcast from row 0 of faces 0/1 — the reader fills only those two face rows (two 64-B reads), nothing redundant.
- **Ring protocols**: input/output implicit sync; gamma/scaler explicit, listed in `disable_dfb_implicit_sync_for`; gamma reads carry `{.trid = 0}` with the barrier on the same id; reader and writer hand the tag back as their last statement. Session tests (after `quasar.add` / `quasar.softmax` / itself; before `to_layout`) pass.

## Specification rules

| Rule | Applies in Phase 0 | Status |
|------|--------------------|--------|
| R1 float32 + `fp32_dest_acc_en=False` refused | yes (gate) | `validate()` raises `UnsupportedAxisValue` ahead of the SUPPORTED loop |
| R2 one `default_compute_kernel_config()` | yes | exported; `None` resolves through it only |
| R3 no gate on fidelity / approx | yes | `validate()` reads neither; they flow into `ComputeHardwareConfig` |
| R4 device compute config accepted | yes | read by attribute, re-wrapped as `ComputeConfigDescriptor`; `test_device_compute_config_accepted` passes |
| R5 gamma row-major, `weight.reshape(-1)[j]` | yes | page w → tile column w; random gammas in every cell pass |
| R6 padding out of the mean | no (w_non_aligned not supported) | re-arms in Refinement 2 (`inv_w_bits` already uses logical W) |
| R7 epsilon added to the mean | yes | `·(1/W)` then `+eps` then `rsqrt`; `small_magnitude` regression passes |
| R8 gamma axes read off `weight`; output dtype = input's | yes | `gamma_dtype` / `gamma_layout` from `weight`; output from `input_tensor.spec` |
| R9 / R10 DRAM or L1 input; weight not gated on placement | yes | L1 cells pass; weight read through its accessor wherever it is |
| R11 output in the input's buffer, interleaved | yes (review) | `allocate_tensor_on_device(input_tensor.spec, …)`; `test_l1_input_multi_neo` asserts `BufferType.L1` |
| R12 placement axes via `eval.memory.memory_axes` | yes | used in `validate()` |
| R13 `memory_config` not None → "other", refused | yes | X2 rejection passes |
| R14 `program_config` never read | yes (review) | not referenced after the signature |
| R15 refuse outside SUPPORTED with `NotImplementedError` | yes | `UnsupportedAxisValue` / `ExcludedCell` subclass `NotImplementedError`; 590 `xfail_expected`, 0 wrong-mode |
| X1 malformed weight | yes | `ValueError` before the axis gate; passes |

GUIDANCE advisories: every item that applies is followed (compute-config wrapping, row-major weight into row 0 of a tile + row broadcast, any row split, padding rows computed but not compared, buffers sized from the device, L1 input through its accessor, DEST budget from `DEST_AUTO_LIMIT`, HiFi4). The 16-bit-accumulator advice applies from Refinement 1 and is in its verifier notes.

## Registry Conformance

- `INPUT_TAGGERS = AXES`, imported from the specification; no op-local axis function.
- `SUPPORTED` equals `PHASE0`, axis for axis (lint-checked); every axis the kernel gates on is present.
- `EXCLUSIONS = []`.
- `validate()`: X1 weight form → axes (call, `memory_axes`, weight, resolved config, specification taggers) → R1 → SUPPORTED per axis → EXCLUSIONS → regime predicate. `rms_norm()` calls it first, before any allocation or kernel work.
- The op file declares no `INVALID`.
- No XPASS evidence, so no SUPPORTED auto-fix.

**INVALID audit (specification)**: `INVALID = []`, which is right — nothing in the TARGET rectangle is structurally impossible (float32 + 16-bit DEST is legal-but-lossy and handled by rule R1, not INVALID). The canonical bf8b + ROW_MAJOR entry does not apply: `bfloat8_b` is not in TARGET (no path on Quasar) and `layout` has TILE only. No no-weight canonicalization: the weight is required and has no `gamma_mode` axis (DECISIONS). No cross-tensor entries. No specification defect.

## Design Conformance

- **Algorithm**: Σx² accumulated in DEST over the resident row (`square` with `DestAccumulation::PerRow`), one REDUCE_ROW, `rsqrt(sum · 1/W + eps)` in fp32, then `x · inv_rms (Col) · gamma (Row)` — as designed (`row_resident`, input crosses DRAM once).
- **Implementer departures, all sound**: (a) `rows_per_block` is live (a `grid(rows_per_block, Wt)` walk; tested at 2 incl. a ragged unit); (b) gamma entries are assembled from two whole-page NoC reads instead of read + 32-B RISC copy (write-back DM caches); gamma DRAM bytes double to 1/16 of the input, recorded in the ledger; (c) `1/W` applied by `FillScalar` + `MulBinary` (the scalar-parameter elements are not used).
- **Parallelization**: tile-rows over the full grid (`compute_with_storage_grid_size()`, `split_work_to_cores(row_wise=True)` on whole rounds of `C_eff`), and over the cluster's NEOs. Both dataflow halves are batched: the reader issues tagged reads without per-tile barriers (gamma: one barrier per 16-tile block), the writer issues tagged writes with no barrier at all; round k+1's input reads are interleaved with round k's gamma blocks.
- **Thread plan (read off the host program)**: `THREAD_PLAN = (4, 4, 2)`, one named constant; per call `(R_eff, C_eff, W_eff)` passed as CT args (`reader_threads`, `compute_threads`, `writer_threads`); kernels size their loops from them and `get_my_thread_id()`; every ring's `num_entries` is per-NEO capacity × `C_eff`; protocols as the buffer table. Tests present and passing: threads ran (`test_each_neo_computes_its_own_rows`: NEO t scales its rows by t+1), sessions, regime-pinned at C_eff 1/2/4 and a padded round, alternative plans (2,4,2), (1,4,1), (1,1,1). The design states the op as compute-bound, consistent with C = 4. **No timed backend** in this run: the plan was not measured against (1,1,1) / (2,4,2) in model cycles; that stays a perf lamp for `ttsim-perf`.
- **Occupancy finding (not a defect against the design, but the op's largest perf gap)**: every one-tile-row shape — all four traced model calls — runs on **1 NEO of 128**, since `row_resident` does not split the dependent W axis. The design names this the deferred `w_split` regime with a positive reason (a scheme-change, not a traffic win); it is filed as **Refinement 3**, the first perf slot.
- **Blocking-model fidelity**: `ROWS_PER_BLOCK`, `IN_DEPTH_ROWS`, `GAMMA_BLOCK_MAX_TILES` (→ `gamma_block_tiles`), `GAMMA_DEPTH_BLOCKS`, `OUT_DEPTH_TILES`, `THREAD_PLAN` are each defined once on the host and every ring size, loop bound and CT arg derives from them. The input ring scales with Wt by design: it is the predicate-guarded resident buffer (`c_fit`), and the predicate refuses — rather than mis-sizes — a row over the cap; no Phase 0 cell reaches the refusal (bf16 Wt ≤ 511). For float32 it does (Wt = 256), handled in Refinement 1.
- **Expression**: reader, compute and writer each operate at the block boundary — no per-tile acquire/sync/retire in either dataflow half (implicit sync), compute phases are block-scoped helper calls. No half-turned split: the per-NEO loop's unit is the whole resident row.

## L1 Ledger Audit

- **Currency**: six rows ↔ six `DataflowBufferSpec`s. One mismatch (`row_stat` / `row_sum` capacity) fixed (Code Review #3).
- **Capacity vs live set**: `input_tiles` = live set (whole row, both passes); `gamma_tiles` and `output_tiles` 2× live set, justified as double buffering; `scaler`, `row_stat`, `row_sum` = live set. Under-direction: the input ring spans `w_tiles` and scales with `block_wt`; `row_stat` / `row_sum` span `rows_per_block` and scale with it. No finding.
- **Page format vs DEST width**: `row_stat` / `row_sum` are Float32 with `fp32_dest_acc_en` on and bf16 with it off (`_stat_dtype`); gamma/scaler bf16 by mechanism (row broadcast, scaler format); input/output follow the tensors (ABI). No finding.
- **Disjoint lifetimes**: Σx² and inv_rms already share `row_stat`; `row_sum` vs `row_stat` justified (concurrent within one helper call). Removing `row_sum` is Refinement 4's lever 1.
- **Bounds**: every symbol has a bound and a predicate; the total is closed-form. `L1_RESERVE_BYTES` (512 KiB) is a heuristic headroom for L1-resident tensor banks, not a query — see Recommendations.
- **Data-movement budget**: present and consistent with the code (input 1×, output 1×, gamma 2 × 64 B per gamma tile per tile-row per NEO = 1/16 of the input; padded rows counted). Cheapest-traffic split (`row_resident`) is implemented; `w_split` / `row_streaming` / `gamma_resident` are deferred rows with positive reasons.
- **Block-size defaults**: interleaved — the split spreads tile-rows over the full grid, then takes the coarsest block (whole row, `block_wt = Wt`); held.
- **Footprint**: `C_eff · [IN_DEPTH_ROWS · rows_per_block · Wt · tile_bytes + GAMMA_DEPTH_BLOCKS · gamma_block_tiles · 2,048 + 2,048 + 2 · rows_per_block · stat_bytes + OUT_DEPTH_TILES · tile_bytes]`; the input term scales with `block_wt`, `rows_per_block`, `IN_DEPTH_ROWS`, `C_eff`; gamma with `gamma_block_tiles` and its depth. Worst Phase 0 case ≈ 1.24 MB of a 3.44 MB budget; the ring extent cap binds first.

## Precision Baseline

`tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_precision_baseline.py` (bfloat16 in/weight/out, fp32 DEST, HiFi4, standard-normal inputs; reference in float64 on the stored bf16 values). Results also in `precision_baseline.json` beside the test.

| Shape | eps | Mem | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | got/true median [p5, p95] |
|-------|-----|-----|-----|-------------|--------------|------------------|---------------------------|
| (1,1,32,128) | 1e-6 | DRAM | 0.9999986 | 0.0250 | 0.00083 | 0.00168 | 0.99994 [0.99731, 1.00259] |
| (1,1,32,2048) | 1e-5 | DRAM | 0.9999986 | 0.0310 | 0.00094 | 0.00171 | 1.00011 [0.99741, 1.00280] |
| (2,4,128,512) | 1e-12 | DRAM | 0.9999986 | 0.0331 | 0.00093 | 0.00169 | 1.00001 [0.99732, 1.00272] |
| (1,1,128,4096) | 1e-5 | DRAM | 0.9999986 | 0.0293 | 0.00091 | 0.00169 | 1.00005 [0.99734, 1.00276] |
| (1,1,32,8192) | 1e-6 | DRAM | 0.9999986 | 0.0290 | 0.00090 | 0.00170 | 1.00014 [0.99743, 1.00285] |
| (1,1,8,128) | 1e-6 | L1 | 0.9999986 | 0.0151 | 0.00091 | 0.00169 | 0.99983 [0.99726, 1.00250] |

Golden cells agree: PCC 0.9999985–0.9999987, relative RMS 0.0016–0.0018, ULP p99 = 1 in every PHASE0 cell; regression cases `small_magnitude` / `large_magnitude` / `uniform` pass (rel. RMS ≤ 0.0026).

**Assessment**: the error is one bfloat16 output rounding (±2⁻⁹ ≈ 0.2%: the got/true spread is ±0.27%, centred on 1.000), flat in W from 128 to 8192 — the fp32 DEST accumulation and the tf32 reduce add nothing visible. No scale signature (high PCC with high RMS): ruled out.
**Recommended tolerances**: defaults hold — the bfloat16 `default` entry of `NUMERIC_RESOLVED` (PCC ≥ 0.999, Frobenius ≤ 1%) has a ~6× margin. For 16-bit DEST cells (Refinement 1), `legacy_tolerances[(bfloat16, False)] = (0.99, 0.13)` as specified.

## Verifier CLI Summary

Results dir: `/tmp/rms_norm_golden_v2` (copy: `verifier_report.json` in this directory). Golden: 42 passed / 633, 0 failed, 1 skipped (empty `LOOSE_CASES` parameter set), 0 hangs.

- supported_pass: 40 (34 PHASE0 cells + 6 numeric regression cases)
- xfail_expected: 590 (dtype=float32 312, fp32_dest_acc_en=False 156, gamma_dtype=float32 78, alignment=w_non_aligned 20, rank 2/3 24 — each value named in a refinement)
- invalid_skipped: 0
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- no_axes_found: 3 (rejections X1, X2 — passed; the empty loose-case parameter set — skipped)
- Queue coverage (`--queue op_requirements.md`): 0 gaps.

## Recommendations

- **Queue** (`op_requirements.md`): R1 numeric (float32 in/weight, 16-bit DEST, float32 W=8192 fit) → R2 shape (w_non_aligned mask, ranks 2/3) → R3 perf (`w_split` for few-tile-row shapes: the model calls, 1 → ≥ 16 NEOs) → R4 perf (post-reduce fold dropping `row_sum`; cluster spread). Perf gates are structural counts; no simulator timing is a speedup.
- **Thread-plan measurement** is owed to a timed backend: (1,1,1), (2,4,2), (4,4,2) in model cycles on `ttsim-perf`, on one input large enough for the steady state.
- **`L1_RESERVE_BYTES`** is a fixed 512 KiB headroom for L1-resident input/output/weight banks that `get_max_worker_l1_unreserved_size()` does not subtract. It holds for every INPUTS shape (≤ 64 KiB of tensor per bank), and the ring extent cap binds long before the budget on this part; on a part with less L1, replace it with the allocator's actual low-water mark if one becomes queryable.
- **`gamma_resident` not filed**: every INPUTS shape gives each NEO at most one round, so a cluster-resident gamma saves only the C−1 duplicate reads per cluster (≤ 1/16 of the input bytes), and under R3's `w_split` each NEO reads a distinct gamma chunk — no target shape. Re-arm if tall inputs (the models' [1,1,4096,2560]-class calls) enter INPUTS.
- **Gamma reads**: two 64-B transactions per gamma tile (the second only to place columns 16..31 in face 1). A single read plus a cache-safe local copy would halve gamma transactions; not worth a slot at 1/16 of the input bytes.
- **Precision lamps not needed now**: the SFPU collapse instead of the tf32 FPU reduce and the FPU gamma multiply (design lamps 1 and 3) — Phase 0 is already at one output rounding.
