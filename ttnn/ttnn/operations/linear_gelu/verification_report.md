# Verification Report: linear_gelu

Board: Blackhole p100a (11x10 = 110 Tensix cores). Phase 0 state: commits `692659b60f3` (R1) and `fad4ed89ba4` (R2 x row multicast, built early by the implementer).

## Code Review

Fixed in place:
- `linear_gelu.py`: the default-config / output-placement resolution was written out twice (in `validate()` and in the entry point). Both now go through `_resolve()`.
- `linear_gelu.py`: the output `memory_layout != INTERLEAVED` refusal ran after the EXCLUSIONS check. It is a model-constant refusal, so it now runs before SUPPORTED, matching the design's check order (structure -> model constants -> SUPPORTED -> EXCLUSIONS).
- `linear_gelu_program_descriptor.py`: the nominal subblock `(sb_h, sb_w)` is loop-invariant and was re-derived per grid row. It is now computed once.

Reviewed, no change needed:
- Kernel syntax (`void kernel_main()`), include paths (`api/dataflow/...`), `TensorAccessor` everywhere (no `InterleavedAddrGen`).
- CB sync: `cb_activation_block` push = pop = nominal `block_h * k_ext_tiles` per row-block, including the tail. `cb_weight_resident` is pushed per chunk and never popped, and compute's cumulative `wait_front` matches. `cb_output_tiles` is pushed per subblock by `matmul_block` (SubblockMajor) and popped per subblock by the writer. `cb_bias_ones_tile` moves no credits (reader-private constant).
- Helper usage: `matmul_block` with the packer-thread `GeluTanhActivation` (additive `GELU_TANH` extension in `sfpu_activation_helpers.hpp`, routed to `gelu_tanh_tile_pack`, not the erf GELU). W and x inter-core traffic uses `ttnn.Mcast` + `McastArgs` (no raw multicast/semaphore code). Bias is folded as the 25th K tile, so there is no separate bias/activation epilogue.
- Broadcast: bias broadcast over rows is a rank-1 K tile (ones column x bias row), so no repeated-data fill.

### Blocking-model fidelity
- Knobs are parameters with one source each: `BLOCK_W`, `ACT_DEPTH`, `OUT_DEPTH`, `CB_BUDGET_BYTES`, `GELU_FAST`, `K_EXT_TILES` are host constants passed as CT args. `block_h` is solved once in `select_blocking()` from the ledger footprint formula (`_footprint_bytes`, shared by the solve and the budget check). The grid comes from `compute_with_storage_grid_size()`. CB page counts derive from these symbols; no duplicated literal found.
- No CB scales unconditionally with a whole-op dimension. `cb_weight_resident` spans the core's n slice (`k_ext_tiles * core_n_tiles_max`, at most 225 tiles), and a host `RuntimeError` guards it when the slice does not fit (unreachable on p100a / 8x8).
- Schedule expression: the reader acquires one row-block per barrier and one W chunk per barrier. Compute issues one `matmul_block` per (row-block, n-block). The writer handshakes per 4-tile subblock (the matmul's push unit), not per tile. Both dataflow halves are batched.
- Work distribution fills the machine: 110 cores at >= 2048 tokens, 88 at 1024 tokens (8 grid rows), which matches the design's table.

### Design deviations (accepted)
- R2 (x row multicast) was deferred in the design for L1-resident x and built anyway. The implementer measured R1's 11x unicast x reads as the binder (8x512 393 -> 195 us). The ledger's data-movement budget was updated accordingly. This is a measured improvement, not a contract violation.
- The x multicast uses a rotating sender per grid row with a per-row round offset (diagonal), where the design had a fixed sender. It is the implementer's choice of realization and stays within the scope boundary.

### Prompt rules
- MUST: one device program per call -> one `ttnn.generic_op`; no composed ttnn ops or host tensor transforms in the entry point. Holds.
- MUST: accurate tanh GELU -> `GELU_FAST = 0`, `gelu_tanh_tile_pack<DST_ACCUM_MODE, false>`. Holds.
- MUST: default HiFi2 + fp32_dest_acc_en, caller config honored -> holds (`test_caller_compute_config_honored`, `test_default_compute_kernel_config`).
- MUST: padded rows exactly 0 when S % 32 != 0 -> not yet applicable (`h_non_aligned` is refused). Re-arms with Refinement 1, which names the writer zero fill and the Inf-padding regression test.

## Registry Conformance
- `INPUT_TAGGERS = {"alignment": tag_alignment}`, signature `(inputs, axes)`. Correct.
- `SUPPORTED` covers every gated axis: dtype, layout, alignment, fp32_dest_acc_en, memory_layout, buffer_type, out_buffer_type.
- `EXCLUSIONS = []`.
- `validate()` checks SUPPORTED per axis, then EXCLUSIONS, and raises `UnsupportedAxisValue` / `ExcludedCell`. It is the entry point's first line.
- The op file declares no `INVALID`.
- No auto-fixes to SUPPORTED: no XPASS drift. A probe with the gates opened showed DRAM x / DRAM output already numerically correct and non-aligned S failing only the padded-row contract. These are queued as Refinement 1 rather than flipped here, because `h_non_aligned` needs the writer W3 zero fill and the DRAM cells need their perf guard measured.

### INVALID audit (`eval/golden_tests/linear_gelu/feature_spec.py`)
- One entry: `{dtype: bfloat8_b, layout: ROW_MAJOR}`, the canonical activation entry. It is single-tensor and structural. It is inert under the current TARGET (no bf8b, no ROW_MAJOR).
- No cross-tensor couplings and no "not yet" entries. No weight-optional canonicalization applies (bias is mandatory).

## L1 Ledger Audit
- **Currency**: four CBs in code (`cb_activation_block` 0, `cb_weight_resident` 1, `cb_output_tiles` 2, `cb_bias_ones_tile` 3), four rows in the ledger, and the size expressions match `create_program_descriptor`. Total `fixed + block_h * per_row` = 921,600 B at `block_h = 4` = `CB_BUDGET_BYTES`.
- **Capacity vs live set**: `cb_output_tiles` holds 2 calls (24 tiles) against a minimal live set of 2 subblocks (8 tiles). The ledger justifies it as call-granularity double buffering. Accepted, but the 32 KB is a candidate for `ACT_DEPTH` headroom; that is folded into Refinement 2's `block_h` / `ACT_DEPTH` co-tune. No under-capacity found: the activation block spans m and k, and the weight CB spans k and the core's n slice, both scaled by their symbols.
- **Page format vs DEST width**: every CB is bf16. With fp32_dest_acc_en on, the only DEST -> L1 boundary is the final output pack (output contract), so there is no phase-boundary truncation. No wide pages with 16-bit DEST.
- **Disjoint lifetimes**: all four CBs are live for the whole kernel concurrently; each row states why it cannot alias.
- **Bounds**: `k_ext_tiles`, `core_n_tiles_max`, `block_h`, `core_m_tiles` are bounded in the symbol table with the predicates that establish them (`validate()` model constants, host solve).
- **Data-movement budget**: present and consistent with the built R1 + R2 (W 1 DRAM crossing + column multicast; x 1 read per row-block + row multicast; y once). The cheapest-traffic split is implemented.
- **Block-size defaults**: `block_h` is the coarsest block that fits the 900 KiB budget (the solve), and `block_w = 3` departs from the coarsest 9 for the first-chunk W wait (stall-shadow reasoning in the design, not a measurement). The `block_w` departure is unmeasured, so it is in Refinement 2's `block_w` co-tune (lamp L2).
- **Per-core footprint**: `(25 * core_n_tiles_max + 1) * 2048 + block_h * (2 * 25 + 2 * 3) * 2048`. The weight term scales with `grid_x` / `block_w`; the activation and output terms scale with `block_h`, `ACT_DEPTH`, `OUT_DEPTH`, `block_w`.

## Precision Baseline

`tests/ttnn/unit_tests/operations/linear_gelu/test_linear_gelu_precision_baseline.py`, production corner (x / output L1, default HiFi2 + fp32 DEST), golden model-distributed inputs (z ~ N(-2.8, 1.7^2)), fp64 reference.

| Shape | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | ULP max / mean (\|ref\| >= 0.1) | got/true ratio p5 / p50 / p95 (\|ref\| >= 0.1) |
|-------|-----|-------------|--------------|------------------|-------------------------------|-----------------------------------------------|
| 1x32 | 0.999970 | 3.50e-2 | 7.06e-4 | 1.184e-2 | 30.2 / 1.55 | 0.9717 / 0.9975 / 1.0115 |
| 2x512 | 0.999970 | 4.28e-2 | 6.91e-4 | 1.174e-2 | 35.0 / 1.55 | 0.9715 / 0.9976 / 1.0114 |
| 8x384 | 0.999969 | 4.31e-2 | 6.88e-4 | 1.178e-2 | 37.2 / 1.54 | 0.9718 / 0.9976 / 1.0114 |
| 8x512 | 0.999970 | 5.46e-2 | 6.91e-4 | 1.175e-2 | 37.1 / 1.55 | 0.9716 / 0.9976 / 1.0114 |

**Assessment**: inside the contract (PCC >= 0.9999, rel RMS <= 1.4e-2) and slightly better than stock (1.21e-2..1.30e-2). Scale-vs-precision triage: the ratio spread is broad (about +/-3% at p5/p95) and centered near 1.0, so this is rounding noise, not a scale bug. The median sits at 0.9976, a small negative bias consistent with bf16 output packing plus HiFi2 operand truncation. The relative RMS is dominated by GELU's negative tail, where |y| is small. The golden `ulp_p99` (about 13k) comes from that tail and is not a useful metric for this op.
**Recommended tolerances**: PCC >= 0.9999, relative RMS <= 1.4e-2 (unchanged; Phase 0 has 16% margin on RMS).

## Verifier CLI Summary

Results: `ttnn/ttnn/operations/linear_gelu/golden_results/verifier_report.json`.

- supported_pass: 23 (15 grid cells + 8 flagged loose cases)
- xfail_expected: 73 (every cell needing `h_non_aligned`, DRAM x or DRAM output -> Refinement 1)
- invalid_skipped: 0 (INVALID is inert under TARGET)
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- no_axes_found: 6 (`test_regression.py`, not registry-driven: 2 pass, 4 refused with `UnsupportedAxisValue` for `h_non_aligned`; they pass once Refinement 1 lands)

Acceptance + precision: 32 passed, 15 skipped (outside SUPPORTED).

### Device time (golden `device_kernel_ns`, production placement)

| (B, S) | cores | us | stock us | target us |
|--------|-------|----|----------|-----------|
| 8x512 | 110 | 195.6 | 247.5 | 90.8 |
| 16x256 | 110 | 195.7 | 247.5 | 90.8 |
| 32x128 | 110 | 195.7 | 247.4 | 90.8 |
| 8x384 | 110 | 155.0 | 191.4 | 68.1 |
| 4x512 | 110 | 114.1 | 135.5 | 45.4 |
| 8x256 | 110 | 114.2 | 135.5 | 45.4 |
| 2x512 | 88 | 73.1 | 83.7 | 22.7 |
| 4x256 | 88 | 73.0 | 83.3 | 22.7 |

Beats stock on every row. Time is linear in tile rows per core: about 41 us per 3 extra rows (about 2050 cycles per output tile, against about 910 FPU cycles per tile at HiFi2), plus about 32 us of fixed cost.

## Recommendations
- Queue: 1 generality refinement (TARGET - SUPPORTED is exactly `alignment = h_non_aligned`, `buffer_type = DRAM`, `out_buffer_type = DRAM`; all knob-turns, bundled), then 3 perf refinements. The perf-1 contract (8x512, L1/L1, default config) is already fully supported at Phase 0, so no generality refinement is needed to unlock it. The 2:1 cadence collapses to "all perf" after Refinement 1.
- Not queued: R6 (K-split + cross-core combine, a scheme change) pays only below 320 tokens (`tensor_m_tiles < grid_y`), which is correctness-only per the prompt. No refinement until a flagged shape needs it.
- Contract-sensitive lever: `GELU_FAST`. The prompt's MUST rule forbids a different or lower-precision activation. Refinement 2 carries the adoption gate. An earlier probe of mine compared fast vs accurate with the program cache on, so both arms ran the same binary; do not trust it. The implementer's probes (`probes/probe_002.py`, `probe_004.py`) clear the cache correctly.
- Advisory (soft): the perf test `test_linear_gelu_perf.py` asserts correctness only. Device ns comes from the golden runner's `device_kernel_ns` (the `--profile` wrapper produced no CSV on this run).
