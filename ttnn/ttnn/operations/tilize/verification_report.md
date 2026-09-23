# Verification Report: tilize

Phase 0 regime: `row_split_interleaved` (op_design.md → Blocking Model). Box: Wormhole B0 (n150), 64 usable Tensix cores (`compute_with_storage_grid_size()` = 8×8), 12 DRAM banks.

Terms, qualified once: **core** = Tensix core; **tile-row** = one row of output tiles (`tile_h` sticks); **quantum** = the unit one CB push / pop covers; **block** = the blocking-model block (`block_height` × `block_width` output tiles), not a bfp8 block.

## Code Review

Every item below was fixed in place.

1. **Collapsed `tile_row` streaming window (blocking-model expression failure).** The reader (NCRISC) and the writer (BRISC) each acquired, synchronized and retired **one tile-row at a time**. Per tile-row that was one `cb_reserve_back`/`cb_push_back`, one read barrier, one `cb_wait_front`/`cb_pop_front` and one write flush. The planned `tile_row` extent is `core_row_tiles` (8 on the perf-focus shape), and the block factor only set the trip count. On the perf-focus shape [1,1,16384,64] a tile-row is `block_width` = 2 tiles (4 KiB in, 4 KiB out), below master.md's granularity floor ("a one-tile-row chunk is poor"). All three conditions of the expression rule held on both dataflow kernels.
   - **Fix.** New host knob `QUANTUM_MIN_TILES` (= 8 tiles) in `tilize_program_descriptor.py`, defined once. From it the host derives `rows_per_quantum = min(ceil(QUANTUM_MIN_TILES / block_width), max_positions // DEPTH_IN, CB budget cap)`, floored at 1, and passes it to both dataflow kernels as a CT arg. The same value sizes both CBs (`DEPTH * rows_per_quantum * block_width` pages).
   - **Kernels.** `StickProducer` (`tilize_stick_reads.hpp`) fills slots of `rows_per_quantum` tile-rows under one NoC transaction id and one barrier. `store_rows` writes a quantum under one flush. Only the kernel's final quantum may be partial, and nothing is pushed after it, so the CB ring-wrap invariant holds. Compute is unchanged (one helper call per kernel, `block_width` pages per helper block).
   - **Guards.** The split reader pins `rows_per_quantum = 1` on the host, backed by a kernel `static_assert`. The `max_positions // DEPTH_IN` cap keeps at least `DEPTH_IN` quanta per core, because one quantum per core measured slower (no overlap).
   - **Measured.** See [Performance](#performance-measured-device-kernel-ns). About −3 % on the perf-focus shape and −5 to −8 % on [1,1,16384,32]. Shapes that resolve to the same program stay within the noise floor.
2. **Runtime `shard_api` mis-tag in `validate()`.** `_side_spec` treated `nd_shard_spec is not None` as ND. A probe showed that an *allocated* legacy HEIGHT-sharded tensor's `memory_config()` also carries a derived `nd_shard_spec`, and that an ND tensor with a 2-D equivalent reports `HEIGHT_SHARDED` + `shard_spec`. So every legacy-sharded input would have been tagged `"nd"` at runtime, while the scenario taggers said `"legacy_2d"`. That goes unseen today because `shard_api` refuses both values, but it breaks the sharding refinement the moment `legacy_2d` enters SUPPORTED.
   - **Fix.** Read the C++ `created_with_nd_shard_spec` flag. It is exposed only via `MemoryConfig.to_json()`.
   - **Test.** `test_tilize_registry.py` covers a legacy input, an ND input and the Phase 0 cell.
3. **DRY.**
   - `tilize.py` restated `TILE_WIDTH` and a second `(R, C)` tile-grid function. It now imports both from the program descriptor, so the tagger and the work split share one `(R, C)`.
   - The reader and writer took an `in_tile_bytes` CT arg that is exactly `tile_h * tile_col_bytes`. Both inputs were already CT args, so the kernel now derives it.
   - The per-column CB byte formula was written twice. It is now one helper, `_per_col_tile_bytes`.
4. **Knob coverage.**
   - `test_tilize_knobs.py` previously had only one shape (8 tile-rows per core) on which any multi-row window could engage. Added [1,1,4320,160]: R = 135, so 3 or 2 tile-rows per core, a partial final quantum, and a ragged column block.
   - Added seven quantum configs. One of them makes the quanta straddle column-block boundaries (`FAST_TILIZE_MAX_BLOCK_WIDTH=2`). Others cross the quantum with read-ahead, depth 3, a tiny CB budget and the split reader.
   - The whole matrix is bit-exact. The full unit-test directory (142 tests) passes under `--dev`.

Checked and left as is (conformant):
- **Compute.** One `compute_kernel_lib::tilize<block_width, …>(core_row_tiles * num_col_blocks)` call per kernel, preceded by `compute_kernel_hw_startup`. Fast-tilize eligible (32×32, half-sync DEST, bf16).
- **Kernel hygiene.** `TensorAccessor` everywhere, `void kernel_main()`, `api/dataflow/dataflow_api.h` includes. Push and wait counts match on every CB (nominal quanta, partial only at kernel end).
- **Custom dataflow block ops.** The reader and writer being custom block operations is the design's documented choice (`read_sticks_for_tilize` rejected with file:line reasons; no TILE-page writer helper exists). There is no broadcast operand and no multicast, so no mcast-pipe or broadcast-efficiency findings.
- **Work distribution.** `split_work_to_cores(grid, R, row_wise=True)` over the runtime grid reaches `min(R, N)` cores; the perf-focus shape runs on 64/64. Reads go on NoC0 (NCRISC) and writes on NoC1 (BRISC). A per-core traversal rotation spreads concurrent requests over the DRAM banks.
- **Parked knobs.** `READ_AHEAD` and the split reader are live, tested knobs, parked at their measured-best values.
- **Validation order.** The entry point runs the malformed-call checks (rules 1–6, `ValueError`) *before* `validate()`. This deviates from the template's "validate() first" and is intentional per op_design.md: a non-aligned input with no pad argument is malformed and must raise `ValueError`, not the `UnsupportedAxisValue` that the alignment gate would raise first.

### Prompt rules (`eval/prompts/tilize.txt`)

| Rule (applies now) | Status |
|---|---|
| MUST declare `output_dtype` in SUPPORTED and gate it in `validate()` | ✓ `SUPPORTED["output_dtype"] = [bfloat16]`, gated |
| ROW_MAJOR input MUST be tilized in-kernel; MUST NOT delegate to `to_layout` / `tilize` / `pad` / … | ✓ one `ttnn.generic_op` dispatch, no layout-converting op |
| MUST NOT hardcode a core count | ✓ `device.compute_with_storage_grid_size()` |
| `SUPPORTED["tile_grid"]` lists only geometries the split reaches | ✓ `single_tile` / `small` / `tall_narrow` (the prompt itself states a row split covers these); `short_wide` / `square_large` refused. **Advisory:** on Blackhole (110–130 cores), `tall_narrow` with `R < N` (e.g. `tall_narrow_grid_scale`, R = 64) leaves cores idle, and the 2-D rule would use them. Refinement 5 is told to apply the rule there too. |
| An L1 misfit on an INPUTS / LOOSE_CASES shape is a defect | ✓ every CB is bounded by `CB_BUDGET_BYTES[low_l1]` (ledger) |
| Report the core count alongside a perf-focus duration | ✓ 64 Tensix cores (below) |

Soft guidance: none unfollowed.

## Registry Conformance

- `INPUT_TAGGERS` has 12 entries. Each is `(inputs, axes)` and reads `inputs[0]` as the scenario dict, per feature_spec's `Expected INPUT_TAGGERS` block. `tile_grid` imports `DOMINANT` rather than restating it.
- `SUPPORTED` covers all 14 gated axes (`dtype`, `output_dtype` and the 12 tagged ones). The 5 "none"-sentinel axes include `"none"`.
- `EXCLUSIONS = []`.
- `validate()` checks SUPPORTED per axis first, then EXCLUSIONS, raising `UnsupportedAxisValue` / `ExcludedCell`.
- The op file does **not** declare `INVALID`.
- No SUPPORTED auto-fixes were needed: `xpass_drift` = 0.

**INVALID audit** (`eval/golden_tests/tilize/feature_spec.py`, not edited):
- All 38 entries couple `dtype` × `output_dtype`, which describe one pipeline (input format → pack format), with a documented contract: tilize is value-preserving, so int↔float and integer width changes are structurally impossible. The universe changes. No capability gaps are dressed as impossibility.
- **bf8b + ROW_MAJOR:** holds structurally. `bfloat8_b` / `bfloat4_b` are not in TARGET `dtype` (inputs), only in `output_dtype`, so no RM block-float input cell exists.
- **Not norm-like:** there are no weight axes.
- **Missing entry (please add via `/golden-tests`):** `{dtype: fp8_e4m3, in_tile_height: 32|16|8|4|2|1}`. fp8_e4m3 exists only in ROW_MAJOR, so there is no TILE input to re-tile (op_design.md "Structural impossibilities"). On Wormhole it is already masked by `INVALID_FOR_ARCH`; on Blackhole these cells would reach `from_torch` and fail.

**Harness-side axis mismatches** (`merge_axes: 143 rows`, not op bugs; please fix in `eval/golden_tests/tilize/axes.py` / `helpers.py`):
- **88 rows, `shard_api` declared `legacy_2d`, captured `nd`.** `axes.py:_spec_of` has the same `nd_shard_spec is not None` bug fixed above in the op. Fix it the same way (read `"created_with_nd_shard_spec"` from `mem_config.to_json()`).
- **55 rows, `low_l1` declared `True`, captured `False`.** `helpers.run_tilize` A/Bs every `low_l1` scenario, calling `low_l1=False` first, so the observed-axes capture records the first call. Capture the `low_l1=True` leg, or record both.

## L1 Ledger Audit

1. **Currency.**
   - `l1_ledger.md` has rows for `cb_input_sticks`, `cb_output_tiles` and the knob-gated `cb_input_sticks_odd`, matching the three `CBDescriptor`s.
   - Size expressions were updated for the new window to `DEPTH * rows_per_quantum * block_width` pages, with a new `rows_per_quantum` symbol-table row and footprint line.
   - The golden run's measured `device_l1_peak_bytes` matches the ledger: 65536 bytes on [1,1,16384,64] (`block_width` 2 × `rows_per_quantum` 4 × 8192 bytes), 16384 bytes on [1,1,2048,64] (one tile-row per core ⇒ `rows_per_quantum` 1).
2. **Capacity vs live set.**
   - *Over:* none. Each CB's capacity is 2 quanta, equal to its live set (one quantum consumed, one filled).
   - *Under:* none. Both CBs span `tile_col` (`block_width`) and `stick_in_tile_row` (`tile_h`, inside the page) with capacity scaling in both. They stream `tile_row` through a fixed window (`rows_per_quantum`) that is not inflated to the block extent.
3. **Page format vs DEST.** Float16_b pages with `fp32_dest_acc_en=False`, both directions clean. The ledger already records the flip to Float32 + `fp32_dest_acc_en=True` + `UnpackToDestFp32` for the fp32 / 32-bit-integer refinement.
4. **Disjoint lifetimes.** There are no disjoint pairs. The input and output CBs are concurrent, as justified in the ledger (pipelining; tilize cannot run in place, `static_assert(input_dfb != output_dfb)`).
5. **Bounds / closed form.**
   - Every capacity symbol is bounded in the symbol table. `L1_cb_total = rows_per_quantum * block_width * per_col_tile_bytes ≤ CB_BUDGET_BYTES[low_l1]`, closed-form.
   - No tensor dimension appears unbounded. `R`, `C`, `core_row_tiles` appear in no capacity expression. `rows_per_quantum` is capped by `ceil(QUANTUM_MIN_TILES / block_width)` ≤ 8 and by the budget.
- **Data-movement budget.** Present and consistent with the code: 1 DRAM read of the input and 1 DRAM write of the output, 0 cross-core bytes. The cheapest-traffic split, the row split, is the one implemented. `grid_2d_split` is a `deferred` regime row with a positive reason (outside the Phase 0 rectangle; adds no bytes).
- **Block-size defaults.** Held.
  - Interleaved: the split spreads `tile_row` over the full grid, then takes the coarsest `block_width` that fits (≤ 64 tiles at bf16, balanced).
  - The `tile_row` window departs from "one tile-row" only by the measurement above, and only where a tile-row is below 8 tiles.
- **Per-core footprint.** `rows_per_quantum · block_width · (DEPTH_IN · in_tile_bytes + DEPTH_OUT · out_tile_bytes)` bytes. `block_width` scales with `CB_BUDGET_BYTES[low_l1]` and C; `rows_per_quantum` with `QUANTUM_MIN_TILES` and the per-core walk length; the depth terms with `DEPTH_IN` / `DEPTH_OUT`.
- **Ledger findings filed.** None. Nothing needed folding into a refinement.

## Precision Baseline

`tests/ttnn/unit_tests/operations/tilize/test_tilize_precision_baseline.py` (bf16 → bf16, DRAM interleaved; PCC via `assert_with_pcc`, deltas via `comp_allclose`):

| Shape | Input | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | Max ULP | Bit mismatches | got/true ratio (median / p5 / p95) |
|-------|-------|-----|-------------|--------------|------------------|---------|----------------|-------------------------|
| (1,1,32,32) | N(0,1) | 1.0 | 0 | 0 | 0 | 0 | 0 | 1.0 / 1.0 / 1.0 |
| (2,3,64,96) | N(0,1) | 1.0 | 0 | 0 | 0 | 0 | 0 | 1.0 / 1.0 / 1.0 |
| (1,1,2048,64) | N(0,1) | 1.0 | 0 | 0 | 0 | 0 | 0 | 1.0 / 1.0 / 1.0 |
| (1,1,16384,64) | N(0,1) | 1.0 | 0 | 0 | 0 | 0 | 0 | 1.0 / 1.0 / 1.0 |
| (1,1,2048,64) | ±[1,2)·2^[-120,120) | 1.0 | 0 | 0 | 0 | 0 | 0 | 1.0 / 1.0 / 1.0 |

**Assessment.** Bit-identical, including values spanning almost the whole bf16 exponent range. Fast tilize (unpack → 16-bit DEST → pack at Float16_b) is a lossless re-lay for bf16. The ratio spread is degenerate at exactly 1.0, so there is no scale signature.

**Recommended tolerances.** `exact` (0 bit mismatches) for bf16 → bf16 and, once they land, fp32 → fp32 and same-width integers. That requires `Fp32Mode::Lossless` for fp32. For lossy casts, use the golden `helpers._transition_tolerance` floors (bf8b PCC ≥ 0.99; bf4b is lossy by design).

## Verifier CLI Summary

`python3 -m eval.verify_supported <results> ttnn.operations.tilize` on the final code. A trimmed copy (summary and category counts; the full per-test report is 3.3 MB) is saved as `ttnn/ttnn/operations/tilize/verifier_report.json`.

- supported_pass: 30
- xfail_expected: 1746
- invalid_skipped: 2310
- no_axes_found: 304. These are rows with no registry axes: the 10 `test_regression.py` rows plus skipped/untagged translated rows.
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0

**The 10 red rows in the raw run are `test_regression.py`,** which is not registry-driven and runs unconditionally. They are `test_integer_passthrough[uint16|int32]` ×4, `test_extreme_magnitudes` (fp32) ×4 and `test_pad_value_extremes` (fp32 + pad) ×2. Each raises `UnsupportedAxisValue` for its dtype. They are tracked failures, closed by Refinement 7 (numeric formats; the pad pair also needs Refinement 4).

**`xfail_expected` by blocking axis** (cells can count under several): output_dtype 869, buffer 848, dtype 748, pad_mode / pad_value 711, rank 671, alignment 550, shard_api 545, orientation 536, out_scheme 473, tile_grid 252, tile_height 140, in_tile_height 85, low_l1 5. Every `(axis, value)` in TARGET − SUPPORTED maps to a refinement in `op_requirements.md`. fp8_e4m3 is INVALID on this box (Wormhole) and is declared in Refinement 7 for Blackhole verification.

## Performance (measured device-kernel ns)

Wormhole B0, 64 Tensix cores, bf16, DRAM interleaved. Medians of 10 dispatches, two independent runs (`test_tilize_perf_knob_sweep.py` under `run_safe_pytest.sh --profile`). `QUANTUM_MIN_TILES=1` reproduces the pre-review one-tile-row schedule.

| Shape | Tensix cores | one tile-row per quantum | default (`QUANTUM_MIN_TILES`=8) | `rows_per_quantum` | LOOSE_CASES `measured_ns_wormhole_b0` |
|-------|-------------:|-------------------------:|--------------------------------:|-------------------:|------------:|
| [1,1,16384,64] **perf focus** | 64 | 26055 / 26896 | 25448 / 26053 | 4 | 25998 |
| [1,1,16384,32] | 64 | 19182 / 19581 | 18331 / 18048 | 4 | 17765 |
| [1,1,32768,64] | 64 | 53332 / 52103 | 51992 / 52163 | 4 | 54010 |
| [1,1,128,64] | 4 | 3139 / 3194 | 3142 / 3121 | 1 | 2294 |
| [1,1,2048,64] | 64 | 4894 / 5010 | 4945 / 4906 | 1 | — |
| [1,1,8192,256] | 64 | 44266 / 44401 | 44225 / 44690 | 1 (`block_width` 8) | — |
| [1,1,16384,512] | 64 | 169879 / 169080 | 170804 / 169818 | 1 | — |
| [4,3,256,96] | 64 | 9356 / 9176 | 9157 / 9186 | 1 | — |

Rows with `rows_per_quantum` = 1 under both settings compile to the identical program, so they show the noise floor (±1–4 %).
- **Headroom evidence for the perf refinements.**
  - The perf-focus shape moves 4 MiB in ~25.5 µs, about 165 GB/s combined DRAM traffic.
  - The same kernel moves 32 MiB on [1,1,16384,512] at about 198 GB/s. There the reads are 1 KiB stick segments instead of 128-byte sticks (at W = 64 bf16 each stick is its own 128-byte DRAM page).
  - The earlier implementer ablation on the perf-focus shape: reads-only ≈ writes-only ≈ 15.6 µs, and they add up.
  - So the flagged shape is bound by small DRAM transactions and read/write serialization, not by compute. That is the lever surface for Refinements 3 and 6.
- **[1,1,128,64].** It runs on 4 cores (R = 4) at ~3.1 µs, versus 2.3 µs in the reference. That is mostly launch plus the fixed cost of a four-core dispatch. It is a perf lamp in op_design.md ("grid synchronization on tiny work"), not the flagged target.

## Recommendations

- **Queue order.** See `op_requirements.md`: sharding → tile geometry → perf → padding → 2-D split + low_l1 → perf → numeric formats → perf. The perf-focus contract (bf16 → bf16, DRAM interleaved, `tall_narrow`, `fp32_dest_acc_en=False`, fast tilize) is already fully in SUPPORTED, so no generality refinement is needed to unlock it. Refinements 1 and 2 are therefore ordered purely by difficulty.
- **Keep the perf-focus path on fast tilize.** When Refinement 7 exposes `fp32_dest_acc_en` / `Fp32Mode`, the bf16 → bf16 default must keep `fp32_dest_acc_en=False` and fast tilize. Otherwise the flagged shape regresses silently while staying bit-exact.
- **`QUANTUM_MIN_TILES` interacts with Refinement 5.** Column groups make `block_width` small, which raises `rows_per_quantum`. The `max_positions // DEPTH_IN` floor keeps overlap, but Refinement 5 should re-measure short_wide, where each core may own a single tile-row.
- **L1 / memory pressure.** No latent OOM. The CB footprint is bounded by `CB_BUDGET_BYTES` (512 KiB, about 1/3 of L1) independently of every tensor dim. That leaves room for L1-interleaved and L1-sharded tensors in Refinement 1.
- **INVALID.** Add the fp8_e4m3 × retile entries (above). Fix `axes.py:_spec_of` and the `low_l1` A/B capture (above).
