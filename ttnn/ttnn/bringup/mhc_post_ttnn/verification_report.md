# Verification Report: mhc_post

Phase 0 verified on Blackhole p150 (110-core compute grid), 2026-09-30.
Golden results dir: `/tmp/mhc_golden2` (copy of the CLI output committed as `verifier_report.json` next to this file).

## Code Review

Overall the Phase 0 build is clean and matches the design. Every block knob (`block_col_tiles`, `DEPTH_IN`, `DEPTH_OUT`, `COEF_DEPTH`, `L1_BUDGET_BYTES`, `n`, the grid) is a host parameter defined once, and every CB page count, CT arg and loop bound derives from it. Fixes applied in this pass:

| # | Finding | Fix |
|---|---------|-----|
| 1 | **DRY:** `TILE_HW = 32` was declared twice (`mhc_post.py` and `mhc_post_program_descriptor.py`). The tagger, `C % 32` validation and all host tile arithmetic read the tile width, so the two copies could drift apart. | `mhc_post.py` now imports `TILE_HW` from the program descriptor, so it has a single definition. |
| 2 | **DRY / mechanism cap as a literal:** `MAX_STREAMS = 5` restated the derived cap `n² ≤ TILE_HW` (the comb row must fit one raw tile row) as a separate literal. | `MAX_STREAMS = math.isqrt(TILE_HW)` (= 5), derived from the one tile-width constant. |
| 3 | No precision-baseline test. | Added `ttnn/ttnn/bringup/mhc_post_ttnn/tests/unit/test_mhc_post_precision_baseline.py` (see Precision Baseline). |

Checked and found correct (no change needed):

- **Kernel hygiene:** `void kernel_main()` in all three kernels, `api/dataflow/dataflow_api.h` includes, `TensorAccessor` everywhere (no `InterleavedAddrGen`).
- **CB sync:** push = wait = pop on every CB, using nominal quanta. `cb_sublayer_tiles` moves B, `cb_residual_tiles` and `cb_output_tiles` move n·B, `cb_coef_bcast` moves n+n² per segment, and `cb_coef_raw` moves 2 (reader self-consumer). Each quantum divides its CB capacity exactly, so the ring-wrap invariant holds for ragged last blocks.
- **Deadlock ordering:** the reader reserves the data block, then (at a segment start) the coefficient set. The compute kernel holds at most one coefficient set and one data block, so both reserves always find space at depth 2.
- **Helper usage:**
  - `mix_block` is one `eltwise_chain` per output stream, built from `CopyTile` + `MulBinary` + n× `Addcmul` + `PackTile`. The n terms are unrolled at compile time, and wait/pop/reserve/push are caller-managed once per block.
  - The expansion uses `fill_l1_range<4>`.
  - The face 0→1 / 2→3 duplication uses two raw self-aimed `noc_async_read`s per expanded tile. `local_copy_helpers_dataflow.hpp`'s stateful `set_read_state`/`read_with_state` would save only NoC register setup on 2×(n+n²) = 40 reads per segment, so I kept the raw reads. This is a mechanism choice with no measurable consequence.
- **Broadcast:** none in compute. The column broadcast is materialised losslessly by the reader (the FPU/`UnaryBcast` alternatives truncate fp32 through tf32 `srcB`, as the design explains). There is no redundant fill: each expanded tile is written once per segment.
- **Advisory (not a defect today):** the expansion's RISC L1 stores to face 0/2 are followed directly by the self-aimed NoC read of those faces, with no explicit fence. This relies on NCRISC L1 stores retiring before the NoC read engine reads the face. It is deterministic in every run so far (bitwise determinism test passes, 100% of golden cells pass). If a future arch or refinement reorders this, the symptom would be stale right halves of coefficient tiles.

### Prompt rules (`eval/prompts/mhc_post.txt` § Rules)

Every MUST rule whose condition applies is satisfied:
- **One program per call:** exactly one `ttnn.generic_op`. `allocate_tensor_on_device` is output setup, not a dispatch.
- **Read once / write once:** each F/X element is read once and X' is written once. The coefficients are broadcast in-kernel (per-core expanded tiles in L1); no per-column coefficient tensors are materialised.
- **Work split:** the flattened (token-tile row, column tile) split fills all 110 cores at T=640 (every loose case records `device_num_cores = 110`).
- **Unbiased arithmetic:**
  - The mix runs on the SFPU in fp32 DEST.
  - `UnpackToDestFp32` is derived per fp32 CB from its data format.
  - Measured signed bias is ≤ 3e-10, i.e. within 3σ of the sampling noise.
- **fp32 DEST only:** `fp32_dest_acc_en=False` → `UnsupportedAxisValue`, with the axis named in the message (acceptance test).
- **Config handling:** `math_fidelity` / `math_approx_mode` / `dst_full_sync_en` are forwarded from the caller's config; `None` resolves only through `default_compute_kernel_config()`.
- **No shape specialisation:** split, block and buffering are derived from (T, C, dtypes, grid) at call time. All 48 (T, C) fp32 loose cases pass.
- **Determinism:** outputs are bitwise identical across calls (acceptance test).
- **Padded rows:** T % 32 ≠ 0 padding cannot contaminate real rows, because the math is strictly row-local. The h_non_aligned golden cells pass.

The prompt has no soft (prefer / consider / avoid) rules.

### Design conformance

- **Algorithm:** matches the design: `post_j·F + Σ_i comb[i][j]·X_i` with comb transposed (the cyclic-comb tests pin the orientation), fp32 SFPU only.
- **Topology and RISC ownership:** reader on NCRISC (coefficients + data), compute, writer on BRISC. This matches the design.
- **Parallelisation:**
  - The implemented split is `flat_stream`: `split_work_to_cores(..., row_wise=True)` over the full grid.
  - The reader batches (n+1)·B reads per barrier and the writer batches n·B writes per barrier.
  - All three streaming CBs are double-buffered.
  - The machine is filled.
- **Blocking-model fidelity:**
  - The knobs are carried as parameters. No CB scales with a whole-op dimension (T or C).
  - The per-core loop iterates blocks of B columns, not single tiles.
  - The one extent held at 1, `block_token_tiles`, is an explicit design decision and is `assert`ed.
- **Expression:** at the scheduling boundary each kernel does one reserve / wait, one barrier and one push / pop per block. Tile loops exist only inside a block-scoped phase, with no per-unit handshake.
- **Axis accounting** in `l1_ledger.md` is consistent with the code: the three streaming CBs span c (B) and, where relevant, i or j (n); the coefficient set spans k.

## Registry Conformance

- `INPUT_TAGGERS = {"alignment": tag_alignment}`. The tagger has the `(inputs, axes)` signature and implements exactly the prompt's contract (T % 32 on dim −2).
- `SUPPORTED` declares all five TARGET axes (`dtype`, `sublayer_dtype`, `layout`, `fp32_dest_acc_en`, `alignment`).
- `EXCLUSIONS = []`.
- `validate()` checks SUPPORTED per axis (`UnsupportedAxisValue`), then EXCLUSIONS (`ExcludedCell`), then the shape contract (`ValueError`). The entry point calls it on its first line.
- The op file does **not** declare `INVALID`.
- No XPASS evidence, so no auto-fixes to SUPPORTED.
- **INVALID audit** (`eval/golden_tests/mhc_post/feature_spec.py`): `INVALID = []`, which is correct.
  - There are no structural impossibilities: both float dtypes are valid for both F and X independently.
  - `layout` has only TILE, so the canonical bf8b+ROW_MAJOR entry does not apply (bf8b and ROW_MAJOR are both outside TARGET).
  - There are no weight axes, so the norm-style canonicalisation does not apply.

## L1 Ledger Audit

- **Ledger currency:** there are 5 rows for the 5 declared CBs (0, 1, 2, 3, 16). Each capacity expression matches `create_program_descriptor`: `DEPTH_IN·B`, `DEPTH_IN·n·B`, `ceil(n/32)+ceil(n²/32)`, `COEF_DEPTH·(n+n²)` and `DEPTH_OUT·n·B` pages. The closed-form `B_fit` matches `_block_col_tiles_fit` (checked at fp32/fp32 → 11, bf16/bf16 → 23, bf16 F / fp32 X → 12).
- **Capacity vs live set:**
  - No over-sized buffer: the streaming CBs are exactly 2 blocks, and depth 2 is the overlap live set.
  - No collapsed extent: every axis the live set spans scales with its knob.
  - `cb_coef_bcast` (160 KB at n=4) is the largest fixed term. Its size is justified because the SFPU needs full-tile operands, so the scalar per (row, coefficient) must be materialised as a column-broadcast tile. Depth 2 is justified as an explicit pipelining decision (row boundaries fall mid-range).
  - Disposition: no change now. Refinement 2's DEST-resident-coefficient lever reshapes this buffer's consumer, so any resizing folds into Refinement 2.
- **Page format vs DEST width:**
  - Every page is Float32 at Phase 0 with `fp32_dest_acc_en=True`. No *over* finding: the F/X pages are input pages that never traversed DEST, and the X' page is packed from fp32 DEST.
  - No *under* finding. The planned bf16 X' page is the contract's single output rounding, not truncation of an intermediate.
- **Disjoint lifetime:** every pair carries a stated reason (format independence F vs X, the in-place X→X' impossibility, and the raw→expanded source/destination overlap). There are no blank cells.
- **Bounds / closed form:** all symbols are bounded (`n ≤ 5` by validate, `B ≤ min(B_fit, longest segment)`, dtypes by the registry axes). The total is closed-form, and nothing scales with T or C.
- **Data-movement budget:** present and consistent with the implemented split. F and X cross DRAM once, X' once, and post/comb once per (core, token row) pair (≤ 1.06 MB at T=640 fp32, 0.6%). The cheapest-traffic split, `flat_stream`, is the one implemented, and the ledger states it.
- **Block-size defaults:** held: the full grid first, then the coarsest block that fits (`B = min(B_fit, longest segment)`). The implementer measured the alternative (`MAX_BLOCK_COL_TILES` = 4 / 2 → 665 / 699 µs vs 560 µs at T640 C7168 fp32), so the default stands on measurement.
  - Note: at small C (T640 C1792) a core owns ~10 units, so each segment is a single block and the pipeline never reaches steady-state overlap. This is folded into Refinement 3's block / depth co-tune.
- **Per-core footprint:** `B·[DEPTH_IN·(fB + n·xB) + DEPTH_OUT·n·xB] + COEF_DEPTH·(n+n²)·4096 + (ceil(n/32)+ceil(n²/32))·4096`.
  - The B term scales with `block_col_tiles`, the depths, n and the dtypes. The coefficient term scales with `COEF_DEPTH` and n².
  - At n=4: 983 040 B (fp32/fp32, B=11) and 1 019 904 B (bf16/bf16, B=23), against a 1 MiB CB budget.

## Precision Baseline

`test_mhc_post_precision_baseline.py`, float32 X / float32 F (the only SUPPORTED cell). The reference is float64 computed from the device-rounded inputs, seed 1234, Sinkhorn comb.

| Shape (X) | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | ULP (out) p50 / p99 | ULP (term scale) max / p99 | Signed bias | got/true ratio median [p5, p95] |
|-----------|-----|-------------|--------------|------------------|---------------------|----------------------------|-------------|--------------------------------|
| (32, 128) | 1.0000000000 | 6.16e-07 | 4.07e-08 | 5.33e-08 | 0 / 14 | 1.92 / 1.18 | −9.4e-10 (se 1.1e-9) | 1.000000000 [0.99999988, 1.00000012] |
| (1, 128, 4096) | 1.0000000000 | 9.39e-07 | 4.14e-08 | 5.31e-08 | 0 / 12 | 2.27 / 1.21 | +1.1e-10 (se 9.6e-11) | 1.000000000 [0.99999988, 1.00000012] |
| (1, 100, 4096) (T non-aligned) | 1.0000000000 | 9.78e-07 | 4.24e-08 | 5.34e-08 | 0 / 12 | 2.26 / 1.22 | +2.8e-10 (se 1.1e-10) | 1.000000000 [0.99999988, 1.00000012] |
| (1, 1, 640, 28672) (DSv4, C=7168) | 1.0000000000 | 1.07e-06 | 4.15e-08 | 5.33e-08 | 0 / 12 | 2.36 / 1.21 | +2.9e-11 (se 1.6e-11) | 1.000000000 [0.99999988, 1.00000012] |

- **ULP (out)** is the error in fp32 ULPs of the *output* value. Its p99 of 12–14 comes entirely from cancellation (outputs near 0 in a 5-term signed sum).
- **ULP (term scale)** is the error in ULPs of Σ_k |term_k|, the magnitude the fp32 roundings actually act on. It is ≤ 2.4 everywhere, consistent with 1 multiply + n = 4 fused MADs, each rounding once.

**Assessment:**
- The fp32 path is exact to fp32 arithmetic.
- There is no systematic bias: the signed bias is within 3σ of zero and ≥ 4 orders of magnitude below the 1e-6 gate.
- There is no scale bug: the ratio is centred exactly on 1.0, with a ±1 ULP spread.
- The 122-wrap `test_depth_chain[fp32]` passes.

**Recommended tolerances (fp32 X'):** PCC ≥ 0.9999999, relative RMS ≤ 2e-6, ULP (term scale) ≤ 8, |signed bias| ≤ 1e-6 + 6σ. These are the golden `TOLERANCES` plus the ULP floor asserted in the baseline test.

## Verifier CLI Summary

`eval/eval_test_runner.sh eval/golden_tests/mhc_post/` → PASSED=84 FAILED=0 ERRORS=0 SKIPPED=1 HANGS=0 TOTAL=208 (run twice, before and after the fixes above: identical).

- supported_pass: 84: 25 INPUTS shapes × (fp32, fp32), 48 fp32 loose (perf) cases, and 11 regression tests including the 122-wrap fp32 depth chain.
- xfail_expected: 123: every cell with `dtype=bfloat16` and/or `sublayer_dtype=bfloat16`, i.e. 25 shapes × 3 combos = 75 `test_op` cells + 48 bf16 loose cases. All of them are covered by Refinement 1.
- invalid_skipped: 0 (INVALID is empty)
- no_axes_found: 1: `test_regression.py::test_depth_chain[bf16]`, which the test itself skips while bf16 ∉ SUPPORTED. It un-skips with Refinement 1.
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0

**TARGET − SUPPORTED coverage:** `dtype: bfloat16` and `sublayer_dtype: bfloat16` are the only gaps. Both are in Refinement 1. No gap is covered by INVALID, and there are no documented omissions.

### Perf snapshot (fp32 loose cases, device-kernel ns vs `target_ns` = DRAM roofline at 80% of 512 GB/s)

| T \ C | 1792 | 2560 | 4096 | 5120 | 6144 | 7168 |
|------|------|------|------|------|------|------|
| 256 | 2.39× | 1.96× | 1.72× | 1.63× | 1.51× | 1.40× |
| 640 | 1.65× | 1.43× | 1.34× | 1.22× | 1.29× | 1.33× (537 µs vs 403 µs; composite baseline 3781 µs → **7.0× faster**) |
| 1280 | 1.49× | 1.26× | 1.36× | 1.40× | 1.40× | 1.37× |
| 4096 | 1.39× | 1.36× | 1.39× | 1.40× | 1.39× | 1.39× |

The implementer's ablation at T640 C7168 fp32 attributes the time mostly to compute:

| Variant | Time |
|---------|------|
| Baseline | 556 µs |
| Compute stubbed | 415 µs |
| Compute + expansion stubbed | 398 µs |

That is ~3.3 µs per output tile on the SFPU mix (163 output tiles per core). At bf16 the DRAM roofline halves (T640 C7168 bf16: ~202 µs) but SFPU time does not, so the **perf-focus bf16 cells will be ~2.5–3× off roofline and compute-bound** once Refinement 1 lands. That gap is Refinement 2.

At small C the reader's coefficient expansion and the missing block overlap dominate:

| Variant (T640 C1792 fp32) | Time |
|---------------------------|------|
| Baseline | 223 µs |
| Compute + expansion stubbed | 101 µs |

That gap is Refinement 3.

## Recommendations

- **Queue order:**
  1. Refinement 1: bf16 streams / sublayer, which lands the full perf-focus contract.
  2. Refinement 2: perf, the compute-bound SFPU mix at the flagged bf16 shapes. This is the largest headroom and a structural compute change, so it runs first among the perf phases.
  3. Refinement 3: perf, data movement / overlap / expansion at the flagged bf16 shapes. Once compute is fixed, this becomes the bottleneck.
- **bf16 X' rounding (Refinement 1 risk):** packing fp32 DEST → bf16 must be round-to-nearest-even. A truncating packer would give a systematic −(½ ULP) bias, failing the 1e-5 signed-bias gate and the 1e-3 122-wrap drift bound.
- **Mixed-format chains (bf16 F / fp32 X):** `CopyTile` with `DataFormatReconfig::Enabled` reconfigures the unpacker per element when F and X formats differ. This is correct but adds per-tile reconfigs. Refinement 2's compute restructure should account for it.
- **L1 headroom:** the 1 MiB CB budget leaves ~0.4 MB of Blackhole L1 unused. Raising `L1_BUDGET_BYTES` to the arch's real usable L1 is a Refinement 3 knob, used only if measurement shows a coarser block or a deeper buffer helps.
- **Infrastructure note:** during this pass, two acceptance runs died in the `device` fixture with a bus error at device open, and one with `Query mappings failed on device 2`. Both happened before any op code ran and are shared-box card issues. Re-runs on card 0 passed 100%.
