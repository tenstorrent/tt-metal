# Verification Report: mhc_post

Phase 0 regime: **plain-2d** (op_design.md → Blocking Model → Regimes). Plain mode, fp32 streams + fp32
sublayer, separate post / comb, any T. Hardware: Blackhole, 110-core compute grid.

## Code Review

**Fixed in place**
- `validate()` now refuses **sharded** `input_tensor` / `residual` / `post` / `comb` with a `ValueError`.
  Before the fix, the reader's interleaved `TensorAccessor` page walk would have read a sharded tensor as if it
  were interleaved, and the result would be silently wrong. TARGET has no `memory_layout` axis, and the
  contract says "DRAM interleaved", so this is a structural refusal and not a support refusal. A probe
  confirmed it (`tests/.../mhc_post/probes/probe_001.py`: a HEIGHT-sharded F is refused).

**Reviewed and found correct (no change)**
- *Prompt `## Rules`.* All the MUST / MUST NOT rules that apply to Phase 0 hold:
  - one `generic_op` dispatch, and the Python side issues no device ops.
  - F and X are read once and X' is written once.
  - coefficients are broadcast in-kernel: `UnaryBcast<Col>` reads column-0 expanded tiles, and no per-column
    coefficient tensors are built.
  - `post_col` / `comb_col` are RT args and are read in-kernel.
  - work is split over tokens AND C: `(R, Q) = (5, 20)` gives 100 cores at T=160.
  - SFPU-only fp32 mix with `UnpackToDestFp32` on every fp32 CB, including the bcast CB. The FPU bias is absent:
    measured signed bias ≤ 1e-9.
  - `fp32_dest_acc_en=False` is refused with `UnsupportedAxisValue`.
  - the caller's `math_fidelity` / `math_approx_mode` are honoured.
  - `max_grid` confines the core set, and the bits are identical to the whole-grid run (acceptance test).
  - caller `output_tensor` is checked for shape / dtype / layout / memory config / aliasing.
  - collapse_pre without W raises `ValueError`.
  - split, block and buffering are derived per call. Two calls give the same bits.
  - padded T rows are row-local.

  Rules that are not yet armed (stats / collapse / packed) re-arm in Refinements 1–2.
- *Helpers.* `mix_block` is one `eltwise_chain` over `IterationShape::grid(n, block_c)`, built from
  `UnaryBcast<Col>`, `CopyTile`, `MulBinary`, `Addcmul` ×4 and `PackTile`. No raw compute API is used. Dataflow uses
  `TensorAccessor`, `void kernel_main()` and `api/dataflow/dataflow_api.h` includes.
- *CB sync.* Push and pop counts match:
  - reader pushes `block_c` F and `n·block_c` X per block, and compute pops the same.
  - compute pushes `n·block_c` per block, padding a ragged tail to the nominal window. The writer waits for and
    pops the nominal window.
  - `cb_coef_cols` is pushed and popped `n_coef_cols` per row.
- *Blocking-model fidelity / DRY.*
  - `block_c` has one source, `choose_block_c`, and is passed as a CT arg to all three kernels. It sizes
    `cb_f_tiles`, `cb_x_tiles` and `cb_out_tiles`.
  - `R` and `Q` come from `choose_split` only.
  - `IO_DEPTH`, `OUT_DEPTH` and `COEF_DEPTH` are host constants. Each one feeds both the CB capacity and
    `plain_per_c_bytes` / `plain_fixed_bytes`, so the budget solve and the allocation cannot drift apart.
  - No CB scales with a whole-op dimension.
  - Ragged blocks keep their nominal counts.
- *Expression.*
  - Reader: issues the whole block's reads back-to-back with one barrier and one push per operand per block.
  - Compute: waits for the block once and pops it once.
  - Writer: does a cumulative per-tile wait and issues each write as soon as its tile is pushed, but has one
    barrier and one pop per block. There is no per-tile completion boundary, so the block is honoured on all
    three kernels.
- *Design conformance.*
  - Math matches the design: comb is applied transposed (`n + i·n + j`).
  - Topology matches: reader NCRISC/NoC0, writer BRISC/NoC1.
  - Work distribution matches: the 2-D balanced split from `choose_split`, enumerated row-wise.
  - The grid is filled: 100 / 110 cores at the T=160 focus, 110 at T=640/C=7168, 75 at T=1 decode
    (`Ct = 224` c-tiles, minimum-bytes split).
  - The `(g_r, g_c)` RT args for the fused regimes are already plumbed.
- *Departure from the design, accepted:* `OUT_DEPTH = 2` (the design says 1). It is documented in the ledger
  with a measured −8 % / −10 %.

## Registry Conformance

- `INPUT_TAGGERS = {"alignment": tag_alignment}` with the `(inputs, axes)` signature. It tags on X's dim -2.
- `SUPPORTED` lists every TARGET axis. `EXCLUSIONS = []`. `validate()` runs in this order:
  1. structural `ValueError`s;
  2. per-axis `UnsupportedAxisValue`, in TARGET order;
  3. `ExcludedCell`.

  `validate()` is the first statement of `mhc_post()`.
- The op file does **not** declare `INVALID`.
- No SUPPORTED auto-fixes were needed (xpass_drift = 0).
- **INVALID audit** (`feature_spec.py`): there are 4 entries, which couple `mode` with `dtype` and with
  `sublayer_dtype` to fp32.
  - `mode` is a kwarg-derived axis, not a tensor axis. Each entry couples it to a single tensor (X or F).
  - The coupling is a documented owner contract decision (2026-11-06): the fused modes are defined on the bf16 X'
    that the op writes, so with fp32 streams the "X' they read" would be a different quantity. This is a change of
    universe, not "my kernel can't yet", so the entries are well-formed.
  - The canonical bf8b+ROW_MAJOR entry does not apply: bf8b is not in TARGET and the layout is TILE only.
  - Not a norm op, so no weight canonicalization cells are needed.
  - No entries need removing.

## L1 Ledger Audit

- **Currency.**
  - `l1_ledger.md` Phase 0 rows ↔ the descriptor's five CBs: `cb_coef_src` = `n_src`, `cb_coef_cols` =
    `n_coef_cols·COEF_DEPTH`, `cb_f_tiles` = `block_c·IO_DEPTH`, `cb_x_tiles` = `n·block_c·IO_DEPTH`,
    `cb_out_tiles` = `n·block_c·OUT_DEPTH`. All match the code.
  - Its `block_c_cap` (11 for fp32 at `OUT_DEPTH=2`) matches `choose_block_c`. The design's worked table says 14
    because it was computed at `OUT_DEPTH=1`; the ledger is the current one.
- **Capacity vs live set.**
  - `cb_out_tiles`: 2 windows against a 1-window live set. This is justified by measurement (pack b+1 while the
    writer barriers b).
  - `cb_coef_cols`: 20 × 4 KiB for 20 × 32 meaningful words. This is justified: `UnaryBcast<Col>` consumes
    whole-tile pages, and the excess is the column-0 layout, not a sizing error. A compact layout needs a new
    helper.
  - No under-sized spans: `cb_x_tiles` spans `i` and `c` and scales with both.
- **Page format vs DEST.** `fp32_dest_acc_en` is always on. Every Float32 page carries fp32-defined data (F, X,
  X', coefficients). No over- or under-width pages.
- **Disjoint lifetimes.** None unjustified. Every pair has a "Shares with / why not" entry.
- **Bounds.** `block_c ≤ block_c_cap` is established by `choose_block_c`. `n_src` and `n_coef_cols` are bounded
  constants. Total footprint `= (n_src + n_coef_cols·COEF_DEPTH)·4 KiB + block_c·(IO_DEPTH·tile(F) +
  (IO_DEPTH + OUT_DEPTH)·n·tile(X))`, which is closed-form. At the focus (fp32, `block_c=2`): 232 KiB.
- **Data-movement budget.** Present and consistent with the implemented `(5, 20)` split:
  - F and X are read once and X' is written once.
  - Coefficients are read Q× (+2.7 %).
  - The cheapest-traffic split is `R=5, Q=1`; it is a built point of the `(R, Q)` family. The coefficient
    re-read has a positive-reason `deferred` row (`coef-mcast`, ≤ 3 % of bytes).
- **Block-size defaults.** The interleaved default is "spread across the grid, then take the coarsest block".
  The op departs from it with `MIN_PIPELINE_BLOCKS = 4`, which deliberately takes a finer block for overlap.
  **I measured this departure** (device-kernel ns, plain fp32):

  | MIN_PIPELINE_BLOCKS | T160 C5120 | T640 C1792 | T640 C7168 | T1 C7168 | T4096 C1792 |
  |---|---|---|---|---|---|
  | 1 (coarsest) | 136 408 | 181 924 | 549 572 | 63 800 | 820 261 |
  | 2 | 137 966 | 181 011 | 550 574 | 52 881 | 820 119 |
  | **4 (built)** | **125 904** | **146 338** | **550 035** | **49 332** | **820 608** |
  | 8 | 154 563 | 140 764 | 500 633 | 49 166 | 820 510 |

  - The coarsest block is 8–35 % slower on three of five shapes, so the departure is justified.
  - 8 blocks wins 9 % at T640/C7168, but at T160/C5120 it forces `block_c = 1` and loses 23 %. This points at a
    "≥ 2 c-tiles per block, then as many blocks as possible" rule.
  - This finding is folded into Refinement 3, the knob co-tune. It does not get a refinement of its own.
- **Per-core footprint, Phase 0:** `88 KiB + block_c · 72 KiB` (fp32/fp32), or `+ block_c · 36 KiB` for
  bf16/bf16. Only `cb_f_tiles`, `cb_x_tiles` and `cb_out_tiles` scale with `block_c`.

## Precision Baseline

Source: `tests/ttnn/unit_tests/operations/mhc_post/test_mhc_post_precision_baseline.py`. Inputs are fp32 randn
F and X, `post = 2·sigmoid(randn)`, and a Sinkhorn comb. The reference is float64. ULP is measured against the
float64 reference rounded once to fp32.

| Shape (X) | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | ULP p50 / p99 | Signed bias | got/true ratio median (p5 … p95) |
|---|---|---|---|---|---|---|---|
| (32, 128) | ≥ 0.99999 (≈1.0) | 6.2e-7 | 4.1e-8 | 5.3e-8 | 0 / 14 | −9.4e-10 | 1.000000000 (0.99999988 … 1.00000013) |
| (1, 128, 4096) | ≈1.0 | 9.4e-7 | 4.1e-8 | 5.3e-8 | 0 / 12 | +1.1e-10 | 1.000000000 (0.99999987 … 1.00000013) |
| (1, 1, 150, 20480) ragged | ≈1.0 | 1.06e-6 | 4.2e-8 | 5.3e-8 | 0 / 12 | −1.4e-11 | 1.000000000 (0.99999987 … 1.00000013) |
| (1, 1, 640, 28672) | ≈1.0 | 1.07e-6 | 4.1e-8 | 5.3e-8 | 0 / 12 | +2.9e-11 | 1.000000000 (0.99999987 … 1.00000013) |

The golden suite records PCC = 1.0000 and ulp_p99 13–19 on every passing cell.

**Assessment.**
- The output is fp32-exact up to the rounding of a 5-term fused-MAD sum: median 0 ULP, p99 ≤ 14 ULP.
- The ratio is centred on exactly 1.0 with a ±1.3e-7 spread, so there is **no scale bug**.
- The signed bias is 3–4 orders of magnitude under the 1e-6 floor.
- The large ULP max (up to 2.4e6) is confined to near-zero outputs where the terms cancel. There the absolute
  error is ≤ 1.1e-6, which is ordinary cancellation and not a defect.

**Recommended tolerances (fp32 X').** PCC ≥ 0.99999, rel-RMS ≤ 2e-6, |signed bias| ≤ 1e-6, and `atol = 2e-6`,
`rtol = 0` on the full tensor.

## Verifier CLI Summary

Results directory: `/tmp/mhc_post_verify/golden2` (copied to `verifier_report.json` in this directory).

- supported_pass: 88
- xfail_expected: 379. By refused axis: `dtype=bfloat16` 292, `sublayer_dtype=bfloat16` 58, `coef_source=packed` 29.
  All of them are queued in Refinements 1–2.
- invalid_skipped: 348
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- no_axes_found: 16. These are `test_regression.py` contract tests that skip until bf16 / stats / collapse are
  SUPPORTED: depth_chain[bf16], fused_modes_agree, engram_add, hc_scalars_are_runtime, collapse_pre_controls_y,
  output_buffers, max_grid, collapse_deterministic and projection_hifi4. They are named in Refinements 1–2's
  Done-when.

**TARGET − SUPPORTED gap:** `dtype: bfloat16`, `sublayer_dtype: bfloat16`, `mode: stats, collapse`,
`coef_source: packed`. Every value is covered by Refinement 1 or 2. No omissions.

## Recommendations

- **Perf headroom (plain fp32, measured).**
  - T160/C5120: 126 µs against a 72 µs DRAM-roofline target (57 %).
  - T640/C7168: 550 µs against 403 µs (73 %). That is already 6.9× faster than the 81-op composite (3.78 ms).
  - The perf refinements target the flagged collapse profile, but the block-count co-tune measured above is
    shared across modes.
- **bf16 risk (Refinement 1).** Before claiming bf16, probe the packer's fp32 → bf16 rounding. If it is not
  round-to-nearest-even, round in SFPU before the pack. Otherwise the signed-bias gate (1e-5) fails. The same
  applies to `cb_xprime_mm`, which must be bit-identical to the written X'.
- **SFPU-bound mix with bf16 streams.** Each output tile is 5 bcast unpacks + 5 copies + 5 SFPU ops. With bf16
  the DRAM time per tile halves, so the plain bf16 path will likely become compute-bound. This is Refinement 5.
- **Helper gaps (friction, already logged by the implementer).** An untagged fp32 `UnaryBcast` source silently
  truncates to tf32. Adding a `SfpuReduce` chain element and a lane-block SFPU helper would remove the raw-API
  rows the fused modes need.
- **Unused deferred constants.** `COMBINE_*`, `W_RESIDENT_BUDGET` and `BLOCK_ROWS` sit in the descriptor module
  as single-source placeholders for the fused regimes. They are intentional; Refinement 1 consumes them.
