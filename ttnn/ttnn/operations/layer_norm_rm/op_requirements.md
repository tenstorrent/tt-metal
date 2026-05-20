# Operation Requirements: layer_norm_rm

## Definition

- **Formula**: `out[..., i, j] = gamma[j] · (x[..., i, j] − mean_i) / sqrt(var_i + eps) + beta[j]`
  where `mean_i = mean_j(x[..., i, :])` and `var_i = mean_j((x[..., i, :] − mean_i)²)`.
- **PyTorch Reference**: `eval/golden_tests/layer_norm_rm/helpers.py:33-50` (`pytorch_layer_norm`).
- **Import Path**: `from ttnn.operations.layer_norm import layer_norm`
- **Function Signature**:
  ```python
  layer_norm(
      input_tensor: ttnn.Tensor,
      gamma: ttnn.Tensor | None = None,
      beta: ttnn.Tensor | None = None,
      *,
      epsilon: float = 1e-5,
      compute_kernel_config: ttnn.ComputeConfigDescriptor | None = None,
      memory_config: ttnn.MemoryConfig | None = None,
  ) -> ttnn.Tensor
  ```

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but
> forgot to update SUPPORTED.  The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: `[x]` complete (every named cell passing); `[~]` real
> work landed but at least one named axis value still deferred (surface as
> partial); `[ ]` not started.

### [x] Phase 0 — Core Implementation

- **SUPPORTED["dtype"]**: `[bfloat16, float32]`  (bfloat8_b deferred — Refinement 4)
- **SUPPORTED["layout"]**: `[TILE_LAYOUT, ROW_MAJOR_LAYOUT]`
- **SUPPORTED["alignment"]**: `[tile_aligned, w_non_aligned, h_non_aligned]`
- **SUPPORTED["rank"]**: `[2, 3, 4]`
- **SUPPORTED["affine"]**: `[gamma_beta, gamma_only, no_affine]`
- **SUPPORTED["affine_dtype"]**: `[bfloat16, float32]`
- **SUPPORTED["affine_layout"]**: `[TILE_LAYOUT, ROW_MAJOR_LAYOUT]`  (canonical no_affine cell uses TILE; actual gamma/beta TILE is in EXCLUSIONS)
- **EXCLUSIONS**: 7 entries (see op file).  Most notable:
  - `{dtype: float32, layout: ROW_MAJOR, alignment: w_non_aligned}` — kernel bug (Refinement 3)
  - cross-axis `{dtype, affine_dtype, affine}` mismatches (gamma must match input dtype) — Refinement 5
  - `{affine: gamma_*, affine_layout: TILE_LAYOUT}` — gamma must be RM — Refinement 5
- **Cores**: single (core (0,0)).
- **Compute config**: respects user-supplied `ComputeConfigDescriptor`; defaults to `HiFi4 + fp32_dest_acc_en=False`.  Note: matmul-reduce path force-uses HiFi4 internally (Tensix LLK).
- **Golden baseline**: **292 / 3795 passed**; 1535 xfail_expected; 1855 invalid_skipped; 98 supported_fail (86 OOM, 6 numerical-bug, 6 numerical-precision); 0 xpass_drift; 0 xfail_wrong_mode.  See `verifier_report.json`.

### [ ] Refinement 2 — Wide-W streaming reduce + multi-core distribution

**Goal**: rewrite Pass 1 (mean) and Pass 2 (variance) to use
`compute_kernel_lib::accumulate_reduce<>` / `accumulate_reduce_block<>` from
`streaming_reduce_helpers.hpp` so `cb_input_tiles` is sized at
`2 · BLOCK_SIZE · tile_size` (constant in `Wt`) instead of `2 · Wt · tile_size`.
At the same time, distribute work across `device.compute_with_storage_grid_size()`
via `ttnn.split_work_to_cores(total_tile_rows)`.  Embarrassingly parallel — no
inter-core communication; bundles cleanly with the streaming rewrite because
both touch the program descriptor's per-core slice math (start_tile_row,
Ht_local) and the kernel's outer loop.  No SUPPORTED axis is added —
`shape_size` is not a kernel-level branch.

**Verifier notes**:
- 86 of 98 `supported_fail` cells are `OOM` from CB allocation on
  `W ∈ {4096, 8192}`.  Observed message: "Statically allocated circular
  buffers grow to 1956352 B which is beyond max L1 size of 1572864 B".  See
  `data_transfer.md` § L1 footprint and § Performance characterization.
- `op_design.md:53-55` already plans for this with `BLOCK_SIZE` and
  `NUM_BLOCKS = Wt / BLOCK_SIZE`.  Phase-0 implementer collapsed to
  `BLOCK_SIZE = Wt, NUM_BLOCKS = 1` for simplicity; this refinement promotes
  the documented multi-block path to running code.
- Block-size policy from `toy_variance:27-36`: largest divisor of `Wt` that is
  ≤ 8.  Mind `Wt` not divisible by 8 (use largest valid divisor).
- Pass 2's variance reduce uses the same `cb_centered` chunk; the streaming
  rewrite must thread `accumulate_reduce_block<>` per block (interleaved with
  per-block `sub<COL>` + `square_in_place`).  See
  `streaming_reduce_helpers.hpp:53-61` for the single-block helper.
- After streaming reduce lands, the per-core L1 budget no longer caps `Wt`.
  Multi-core then distributes `total_tile_rows` slices.  Output write per
  core uses the same per-core `start_tile_row` offset.
- `numerical_stability.md` confirms `prepare_partial_reduce_scalers` already
  routes only the last block's scaler — no math change required for partial-W.

**Done when**: every Phase-0 `OOM` cell in `verifier_report.json` passes.
Add a "wide" precision-baseline case (e.g. `(1, 1, 32, 8192)`) to
`test_layer_norm_rm_precision_baseline.py` once the OOM cells pass.

### [ ] Refinement 3 — Non-tile-aligned + multi-batch correctness

**Goal**: fix the tile-id math so `(layout=TILE, alignment in {h_non_aligned,
w_non_aligned}, leading_dim_product > 1)` produces correct results; remove the
`{dtype: float32, layout: ROW_MAJOR_LAYOUT, alignment: w_non_aligned}` EXCLUSION.
No SUPPORTED axis is added — every cell here is already in SUPPORTED ∩ ¬EXCLUSIONS
either via the EXCLUSION or via being mis-categorized as a numerical bug.

**Verifier notes**:
- **6 numerical-bug cells**: all `(4, 8, 47, 256)` × `(layout=TILE,
  alignment=h_non_aligned)` × every affine combo and dtype.  Root cause: the
  reader (TILE branch) and writer compute `tile_id = (start_tile_row + ht) *
  Wt + wt`, which is a flat linear walk over tiles.  But TTNN's TILE-layout
  storage pads each leading-dim slice (each `(b, c)` plane) independently to
  a tile-aligned H — so tile_id `Wt · 47 / 32 = ⌈47/32⌉ · Wt = 2·Wt` is not
  "the first tile of (b=0, c=1)" but "the second tile-row of (b=0, c=0)
  (which contains H rows 32–63 of (b=0,c=0)'s zero-padded plane)".
  - Fix sketch A: replace flat tile-id with per-(b,c) stride math.  Pass
    `tile_rows_per_plane = ⌈H / 32⌉ · Wt` as a runtime arg; compute
    `tile_id = bc_index · tile_rows_per_plane + tile_row_in_plane · Wt + wt`.
  - Fix sketch B: iterate the input tensor at the *logical-row* level
    (RM-style) and rely on the in-kernel tilize step.  Avoids the
    leading-dim stride at the cost of always tilizing — but breaks the
    TILE-fast-path.
- **6 numerical-precision cells**: all `(2, 1, 100, 47)` × `(layout=TILE,
  alignment=w_non_aligned)`.  Same family (`prod(shape[:-1]) > 32` plus
  non-aligned H).  H=100 → 4 padded tile-rows per (b=0, c=0) plane, B=2 →
  the second plane's tile-ids are off by `H_padded·Wt / 32 - H/32·Wt`.  The
  same fix above resolves both.
- **The EXCLUSION** `{dtype: float32, layout: ROW_MAJOR_LAYOUT, alignment:
  w_non_aligned}`: PCC was 0.16–0.80 on `(1, 1, 32, 47)`, `(1, 1, 32, 100)`,
  `(1, 1, 50, 47)` during the acceptance run.  Hypothesis: the RM-output
  write path's `write_sticks_after_untilize` strides interact with the
  partial-W mask layout for fp32 (4 bytes/elem) but not bf16 (2 bytes/elem)
  because the L1 stride alignment changes.  Empirically: bf16 + RM + same
  shapes passes, fp32 + TILE + same shapes passes; only fp32 + RM + W
  non-aligned breaks.  Inspect `tilize_helpers_dataflow.inl:107-160`
  (`write_sticks_after_untilize`) for elem_size-dependent stride bugs.

**Done when**: the 12 named cells move from `numerical-bug` /
`numerical-precision` into passing; the fp32-RM-W-partial EXCLUSION is
removed.

### [ ] Refinement 4 — bfloat8_b activation support

**Goal**: add `ttnn.bfloat8_b` to `SUPPORTED["dtype"]` and to
`SUPPORTED["affine_dtype"]`, with the kernel producing in-range bf8b output
tiles.  This unblocks the bf8b column of the golden matrix (currently entirely
in xfail_expected because bf8b is not in SUPPORTED).

**Verifier notes**:
- Phase-0 implementer left bf8b in the user-facing dtype list of the
  docstring but the kernel produces values that `to_torch` rejects with
  `ValueError: datum for bfp2, bfp4, bfp8 is invalid`.  bf8b is block-shared-
  exponent (1 exponent per 16 mantissas), so out-of-range tile values
  corrupt the exponent during pack.  Likely cause: `mean` and `inv_std` are
  produced in bf8b, but the values can fall outside the block's
  representable range (especially `inv_std` can be very large when var is
  small, even with eps guard).
- The cleanest fix is to override the intermediate-CB dtypes for the bf8b
  input case (`cb_mean`, `cb_inv_std`, `cb_centered`) to bf16 in the
  program descriptor.  Outputs and inputs remain bf8b; intermediates
  upcast.  Cost: ~`Wt + 2` extra bytes per tile in L1 vs. bf8b's compressed
  per-tile size.
- Coupling with Refinement 2: bf8b's L1 scaling is even tighter than fp32's,
  so do Refinement 2 first.
- Once bf8b activation works, the cross-axis EXCLUSIONS for affine_dtype
  also expand to `{dtype: bf8b, affine_dtype: float32}` etc. — those move
  to Refinement 5.
- bf8b + ROW_MAJOR remains INVALID (block-quantized format has no RM
  encoding) — no changes to `feature_spec.py`.
- Tolerance for bf8b is already in `helpers.py:TOLERANCES`: PCC ≥ 0.99,
  rel-RMS ≤ 0.10.

**Done when**: bf8b cells in the golden suite that aren't INVALID and
aren't cross-axis-blocked by Refinement 5 pass.  Add bf8b precision
baseline shapes to `test_layer_norm_rm_precision_baseline.py`.

### [ ] Refinement 5 — Independent gamma/beta dtype and layout

**Goal**: relax two coupling constraints currently in EXCLUSIONS:
1. `affine_dtype != dtype` (mixed precision: e.g. bf16 input + fp32
   weights).  Remove the 4 cross-axis EXCLUSIONS.
2. `affine_layout = TILE_LAYOUT` for actual affine (when `affine in
   {gamma_only, gamma_beta}`).  Remove the 2 `affine_layout=TILE` EXCLUSIONS.

**Verifier notes**:
- The current op rejects mismatched dtypes via the registry EXCLUSION
  `{dtype: X, affine_dtype: Y, affine: gamma_*}` where X≠Y.  The kernel's
  in-kernel tilize for gamma/beta does no dtype conversion; the binary
  `mul_in_place<ROW>` / `add_in_place<ROW>` helpers similarly require A
  and B in the same format.  Fix sketch: insert a per-block format
  reconfig + cast helper.  Most natural place is the pre-Pass-1 gamma/beta
  tilize step — convert from `affine_dtype` to `dtype` during tilize.
- For `affine_layout = TILE_LAYOUT`: skip the in-kernel tilize step (gamma
  arrives already tiled), and adjust the CB sizing accordingly (no
  `cb_gamma_sticks` / `cb_beta_sticks` allocation; `cb_gamma_tiles`
  filled directly by the reader).  This is a program-descriptor
  branching change.
- Coupling: this refinement does NOT depend on Refinements 2/3/4; can be
  done in any order after Refinement 2.  Order it last because the cells
  unlocked here are marginal coverage (most of the value is in 2/3/4).

**Done when**: every cross-axis EXCLUSION (`{dtype, affine_dtype, affine}`
and `{affine, affine_layout=TILE}`) is removed; the affected golden cells
pass.

## Refinements deliberately NOT enqueued

- "Add multi-core" as its own refinement — folded into Refinement 2 (the
  per-core slice math is the same change as the streaming chunk math).
- "Improve precision via fp32 intermediate CBs" — covered by Refinement 6
  if it's ever needed.  No `numerical-precision` cells currently point at
  precision-of-intermediates; the 6 `numerical-precision` cells are about
  the tile-id math (Refinement 3), not arithmetic precision.
- "Reduce DRAM bandwidth by caching input in L1 for small W" — a perf
  optimization with no failing cell to point at; not a refinement queue
  entry.  See `data_transfer.md` § Recommendations item 3.
- "Switch reduce to streaming-reduce in cb_centered" — same primitive
  needed as Refinement 2; bundled.
- bf8b + ROW_MAJOR — already INVALID per `feature_spec.py:54`.
- HiFi4 hardcoded in the reduce path (`reduce_helpers_compute.inl:22`) —
  upstream kernel-lib decision, not op-level.  Mentioned only in the
  numerical-stability report.
