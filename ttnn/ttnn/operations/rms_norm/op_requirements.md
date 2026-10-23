# Operation Requirements: rms_norm

Terminology (qualified once, bare afterwards): *core* = Tensix core; *layout* = `ttnn.Layout`; *placement* =
`TensorMemoryLayout`; *R1 / R2* = the `op_design.md` regimes `row_local` / `w_split_root_combine` (refinements are
always written "Refinement N"); *block* = the blocking-model block; device times are device-kernel **ns per call**.

## Definition
- **Formula**: `out[..., w] = x[..., w] · rsqrt( (1/W) · Σ_{w'<W} x[..., w']² + epsilon ) · (gamma[w] if gamma else 1)`, W = logical `shape[-1]`
- **PyTorch Reference**:
  ```python
  def rms_norm_ref(x, gamma=None, epsilon=1e-6):
      xf = x.float()
      out = xf / torch.sqrt(xf.pow(2).mean(dim=-1, keepdim=True) + epsilon)
      if gamma is not None:
          out = out * gamma.float().reshape(-1)
      return out.to(x.dtype)
  ```
- **Import Path**: `from ttnn.operations.rms_norm import rms_norm`
- **Function Signature**:
  `rms_norm(input_tensor: ttnn.Tensor, *, gamma: Optional[ttnn.Tensor] = None, epsilon: float = 1e-6, compute_kernel_config: Optional[ttnn.ComputeConfigDescriptor] = None, memory_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor`
  (`compute_kernel_config=None` resolves through `default_compute_kernel_config()`: HiFi4, `fp32_dest_acc_en=True`,
  `math_approx_mode=False`. `memory_config` is the output placement and defaults to `ttnn.DRAM_MEMORY_CONFIG`.)

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

### Queue rationale (verifier)

- **No `LOOSE_CASES` entry carries an `attention:` perf flag**, so the perf targets are free-selected by measured
  headroom per class (`verification_report.md` §Performance snapshot).
- **Every perf profile in `feature_spec.py` pins `bf16 · fp32_dest_acc_en=False · HiFi2 · bf16 TILE gamma`.** The
  first perf target is the **sharded LLM profiles**, chosen because they are the largest gap: their references are
  2–4× tighter than any interleaved input can reach.
- **Anchor.** Refinement 1 lands the precision half of that contract (`fp32_dest_acc_en=False`) and Refinement 2
  lands the placement half (WIDTH / BLOCK sharding). So when Refinement 3 fires, its exact config is in `SUPPORTED`.
- **Generality is exhausted after Refinement 2.** Refinements 3–5 are perf, each on a different cell class: sharded
  placement, the interleaved R2 regime, and the ROW_MAJOR path.
- **No perf refinement for interleaved prefill.** It is DRAM-saturated: 8192×7168 runs at ≈ 395 GB/s against
  ~415–430 GB/s achieved, which is under 8 % from the ceiling.
- **Permanent omission, not queued:** `{dtype: float32, fp32_dest_acc_en: False}`. It is a mandated
  precision-convention refusal (prompt: "float32 input with fp32_dest_acc_en=False is rejected natively").

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [float32, bfloat16]
- **SUPPORTED fp32_dest_acc_en**: [True] (any `math_fidelity` / `math_approx_mode`; config passed through as-is)
- **SUPPORTED layout**: [TILE, ROW_MAJOR], both native (in-kernel tilize / untilize)
- **SUPPORTED shape-derived axes**: alignment ∈ {tile_aligned, w_non_aligned, h_non_aligned}; rank ∈ {2, 3, 4}
- **SUPPORTED op-specific axes**: gamma_mode ∈ {gamma, no_gamma}; gamma_dtype ∈ {float32, bfloat16, "none"}; gamma_layout ∈ {TILE, ROW_MAJOR, "none"}
- **SUPPORTED memory_layout**: [INTERLEAVED] (DRAM or L1)
- **EXCLUSIONS**: `{dtype: float32, fp32_dest_acc_en: False}` (mandated, permanent)
- **Cores**: multi-core. R1 splits tile-rows across up to the full grid (120 cores on BH p150b). R2 splits W across
  row-major W-groups with a root gather plus a multicast combine, used when tile-rows under-fill the grid or the
  row's slice does not fit L1.
- **Compute config**: caller-supplied `ComputeConfigDescriptor`, defaulting through `default_compute_kernel_config()`
- **Golden baseline**: 737 `supported_pass`, 0 `supported_fail` / `xpass_drift` / `xfail_wrong_mode`; 6180
  `xfail_expected` (per `verifier_report.json`, after the verifier's `INPUT_TAGGERS` fix)
- **Precision baseline**: bf16 within 1 ULP (p50 = 0); fp32 TF32-class (≈ 1.5e4 fp32 ULP p50, −0.13 % median bias
  from FPU operand truncation). See `verification_report.md`.

### [x] Refinement 1 — Numerical configurability: bfloat8_b (input and gamma) + fp32_dest_acc_en=False

**Goal**: Extend three `SUPPORTED` axes:

- add `ttnn.bfloat8_b` to `SUPPORTED["dtype"]` and to `SUPPORTED["gamma_dtype"]`;
- add `False` to `SUPPORTED["fp32_dest_acc_en"]`, for bfloat16 and bfloat8_b inputs;
- keep the `{float32, False}` EXCLUSION.

Make every DEST-produced intermediate CB's page format follow DEST width:

- the affected CBs are `cb_sumsq`, `cb_rowsum_partial`, `cb_rowsum_gather`, `cb_inv_rms_root`, `cb_inv_rms` and
  `cb_x_gamma`;
- they use `Float32` pages when `fp32_dest_acc_en=True` and 16-bit pages when it is `False`;
- the change goes in the `_FMT_F32` role and `F32_TILE_BYTES` in `_cb_specs`, so the L1 solve and the allocation
  move together.

Pass the caller's `compute_kernel_config` through unchanged, as today. This moves the **922** `xfail_expected` cells
refused only on these values (all INTERLEAVED) to passing. The 6180 `xfail_expected` cells split as:

- 922 need this refinement only;
- 1733 need Refinement 2 only;
- 2160 need both;
- 1365 are the permanent `{float32, False}` EXCLUSION (420 interleaved, 945 sharded) and stay refused.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**:

- **Anchor.** This lands the precision half of Refinement 3's contract: bf16 · `fp32_dest_acc_en=False` · HiFi2 ·
  TILE · bf16 TILE gamma. It also makes all 12 interleaved `_perf_case` profiles runnable at their exact config,
  which Refinement 4 needs.
- **Build the False path on the existing structure.** Keep R1/R2 selection, chunk-batched reads and writes, and the
  grid fill. Do not build a correct-only stub; Refinements 3 and 4 optimize this datapath.
- **Selector shift.** 16-bit intermediates shrink the footprint, so `_select_blocking` will pick coarser blocks.
  Re-check that the regime-pinned acceptance shapes still pin their regime at `fp32_dest_acc_en=False`.
- **bf8b TILE gamma.** The writer's TILE-gamma load reads row 0 of faces 0 and 1 by byte offset
  (`gamma_face_row_read_bytes`, `face1_offset`). That arithmetic is wrong for `bfloat8_b` pages (shared exponents
  precede the mantissas). For a bf8b gamma, read the whole 1088 B page per tile instead; the Row broadcast still
  consumes only row 0.
- **Mirror the bf8b INVALID entries.** Add `{dtype: bfloat8_b, alignment: w_non_aligned}` and
  `{…, alignment: h_non_aligned}` to the op's `EXCLUSIONS`, mirroring the author-scoped INVALID entries.
  `test_translated.py` ignores INVALID: 2 bf8b × `w_non_aligned` WIDTH_SHARDED cells run today as
  `invalid_unexpected` and would start failing once Refinement 2 lands sharding.
- **Precision.** `test_rms_norm_precision_baseline.py` must not regress, and the fp32 rows are TF32-class by design.
  Extend its matrix with bf16 + `fp32_dest_acc_en=False` and with bf8b.

**Done when**:

- every cell whose only refusal is `dtype=bfloat8_b`, `gamma_dtype=bfloat8_b` or `fp32_dest_acc_en=False`
  (INTERLEAVED placement) passes;
- `{float32, False}` still raises `ExcludedCell`;
- the 12 interleaved `_perf_case` profiles pass their soft PCC 0.9995 at their exact config;
- the loud categories stay 0.

### [ ] Refinement 2 — Sharded placements: HEIGHT / WIDTH / BLOCK, native zero-copy

**Goal**: Add `HEIGHT_SHARDED`, `WIDTH_SHARDED` and `BLOCK_SHARDED` to `SUPPORTED["memory_layout"]` natively. Map
each placement onto the already-built regimes, with the geometry pinned by the shard spec (`op_design.md`
§Placement axis):

| Placement | Regime | Geometry |
|---|---|---|
| HEIGHT | R1 | row groups = shard cores; one block = the whole resident shard |
| WIDTH | R2 | one row group; `w_group_size` = shard count; `core_w_tiles` = shard width |
| BLOCK | R2 | row groups = shard-grid rows; W-groups = shard-grid rows of cores spanning the W shards |

Data placement:

- The input shard is consumed **zero-copy**: it backs `cb_x_tiles` (TILE) or the tilize source (RM) via
  `ttnn.cb_descriptor_from_sharded_tensor`.
- The output is written zero-copy into the output shard. The golden passes `memory_config =` the input's shard spec.
- Gamma stays interleaved.

Scope: TILE and RM, every alignment bucket (including the poisoned-padding cases), ragged shards, and every
dtype / `fp32_dest_acc_en` value Refinement 1 landed. This moves the **1733** cells refused only on `memory_layout`,
plus the 2160 that also needed Refinement 1, to passing (cartesian, `_SHARDED`, resilience, pad-poison and
translated sharded tests).

**Verifier notes**:

- **Why it stands alone.**
  - HEIGHT is a local knob-turn (a CB-placement change on R1).
  - WIDTH / BLOCK reuse R2's built combine: root gather, semaphore and `ttnn.Mcast`. But the W-group topology must
    follow the **shard grid** (a row of shard cores, with roots and Mcast groups derived from it) instead of
    consecutive row-major runs of a used-core list. That rewiring is the bulk of the work.
  - No skill covers sharded placement (do **not** use `/memory-layouts`; that skill is RM/TILE).
- **Anchor.** This lands the placement half of Refinement 3's contract: WIDTH 32×{1024, 2304, 5120, 7168} and
  BLOCK {640, 7168, 8192}-row profiles, with pinned `shard_shape` / `core_grid`. Build it performantly:
  - The shard **is** the block; do not re-chunk below it without a stated L1 reason.
  - Reading a core's own shard back through a `TensorAccessor` is **not** an implementation of the placement. It
    gets sent back, regardless of test colour.
- **L1.**
  - `_l1_budget_bytes` already excludes the shard and output buffers (it clips at the lowest allocated L1 block).
    When x / out are zero-copy, drop the `cb_x_tiles` / `cb_out_tiles` terms from `_cb_specs` for that path, or the
    same bytes are counted twice.
  - Mechanism cap M1 still binds: the remote-written CBs (`cb_rowsum_gather`, `cb_inv_rms`) must keep identical L1
    addresses on every core of a W-group. Verify the shard-backed CBs do not perturb them.
- **Placement mismatches.** Input / output placement mismatches (interleaved in → sharded out, or the reverse)
  are not exercised by the golden. Refuse them in `validate()` rather than silently re-reading.
- **Partial-tick.** If WIDTH / BLOCK do not fit in one focused pass, ship HEIGHT, tick `[~]`, and file
  `Refinement 2b` (WIDTH / BLOCK) immediately below. Never ship a `ttnn.to_memory_config` wrapper: it also breaks
  the one-dispatch rule.
- **Not in scope.** The INVALID entries `{RM, *_SHARDED, gamma TILE}` are author-scoped skips, so nothing is
  needed for them here.

**Done when**:

- every cell whose only refusal is `memory_layout ∈ {HEIGHT_SHARDED, WIDTH_SHARDED, BLOCK_SHARDED}` passes;
- `infeasible_skipped` contains only harness-side shard-allocation skips;
- the 9 sharded `_perf_case` profiles pass their soft PCC 0.9995 at their exact config and geometry;
- the loud categories stay 0.

### [ ] Refinement 3 — Speed up the sharded LLM profiles

**Type**: perf

**Goal**: Optimize the 9 sharded `_perf_case` profiles in `feature_spec.LOOSE_CASES` toward their clock-scaled
`achievable_ns` (`scaled = achievable_ns · 1350 / actual_aiclk_mhz`, with the clock from profiler evidence). Run each
at its exact config: bf16, `fp32_dest_acc_en=False`, HiFi2, bf16 TILE gamma, and the pinned `shard_shape` /
`core_grid`.

| Placement | Profiles (rows × W) | Reference |
|---|---|---|
| WIDTH | 32 × {1024, 2304, 5120, 7168} | 4.1–5.5 µs |
| BLOCK | 640 × {6144, 7168}, 7168 × {1024, 2048}, 8192 × 1024 | 18.9–25.9 µs |

With x already resident in L1, DRAM no longer bounds these cells. Expect the time to sit in:

- per-block fixed cost: helper inits, format reconfigs, and the per-chunk `x·γ` / normalize passes;
- the R2 combine round.

Use the relevant patterns in `ttnn/ttnn/operations/examples/master.md`:

- problem types **Cross-core communication** (`tensix_all_reduce`, `tensix_all_reduce_compute`),
  **Compute-bound: fixed cost per call** (`compute_block_size`) and **Compute-bound: wasted SFPU work**
  (`sfpu_tile_scope`);
- the design's perf lamps L4 (column-scoped SFPU finalize), L5 (combine payload = faces 0 / 2 only) and L8
  (DEST-reuse instead of the `cb_x_gamma` round trip);
- "data already resident in L1" is not covered by the catalog, so measure it.

No `SUPPORTED` change.

**Done when**:

- measured device-ns improves on the most-impacted sharded profile(s), moving them toward their scaled reference;
- the soft PCC 0.9995 still holds;
- the golden suite is green;
- there is no regression across the config-spanning guard set: one representative per kernel path × layout ×
  placement (R1 TILE, R1 RM, R2 TILE, R2 RM interleaved; HEIGHT; WIDTH; BLOCK).

### [ ] Refinement 4 — Speed up the interleaved R2-regime cells (decode and 640-row)

**Type**: perf

**Goal**: Optimize the interleaved cells that select R2, at the exact profile config (bf16,
`fp32_dest_acc_en=False`, HiFi2, TILE, interleaved). They have the interleaved headroom; prefill R1 is already near
the DRAM ceiling. Two shapes of cell, measured at the `fp32_dest_acc_en=True` proxy:

| Cell class | Measured | Gap |
|---|---|---|
| Decode 32 × {1024, 2304, 5120, 7168} | 5.8–9.3 µs on 4–28 cores for ≤ 0.9 MB of traffic | ~2 µs of DRAM time |
| 640 × {6144, 7168} | 120 cores, one block per core, no read/compute overlap | ~29 % above the DRAM ceiling |

The decode cells are fixed-cost and combine bound. The levers are knob-turns on the surface the planner exposed:

- `MIN_W_TILES_PER_CORE` (design lamp L2: grid fill vs. one more combine; catalog `width_split`);
- `x_depth_blocks = 2` with `block_rows` split so the reader overlaps compute (lamp L1; catalog `double_buffer`);
- the combine payload size (lamp L5);
- `CHUNK_FLOOR_TILES` (lamp L3).

Each knob lives in `BlockingPlan` / `_cb_specs`, which are single-sourced.

No `SUPPORTED` change. The block-pipelined combine (design regime R6) is a scheme-change and is **not** in this
phase.

**Done when**:

- measured device-ns improves on the most-impacted R2 interleaved cell(s) at the exact profile config;
- the golden suite and the precision baseline are green;
- there is no regression across the guard set named in Refinement 3, including the R1 prefill cells, which must not
  lose their DRAM-ceiling throughput.

### [ ] Refinement 5 — Speed up the ROW_MAJOR path

**Type**: perf

**Goal**: The ROW_MAJOR-layout cells pay a per-call tilize on input and a separate untilize pass on output. Where W
is narrow, that fixed cost dominates:

- `(1,1,8192,64)`: RM 24.5 µs vs TILE 10.8 µs (2.3×);
- RM decode profiles: 10–35 % slower than TILE.

Optimize that class: narrow / medium-W RM cells, plus the RM decode profiles at their exact config. Use the relevant
patterns in `ttnn/ttnn/operations/examples/master.md`:

- **Compute-bound: fixed cost per call** (`compute_block_size`: fewer, larger tilize / untilize calls);
- **Data movement: transfer size** (`tile_reorder`), for the stick writes;
- design lamp L7: pack-untilize straight from the normalize DEST, which removes the RM `cb_out_tiles` pass.

No `SUPPORTED` change.

**Done when**:

- measured device-ns improves on the most-impacted RM cell(s);
- the RM fp32 precision baseline does not regress;
- the golden suite is green;
- there is no regression across the guard set named in Refinement 3.
