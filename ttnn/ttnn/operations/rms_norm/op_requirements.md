# Operation Requirements: rms_norm

## Definition
- **Formula**: `y = x * rsqrt(mean(x^2, dim=-1) + epsilon) * gamma` (gamma optional, broadcast along the last dim; the mean uses the true W, never the tile-padded width)
- **PyTorch Reference**:
  ```python
  def pytorch_rms_norm(x, *, gamma=None, epsilon=1e-6):
      x32 = x.to(torch.float32)
      out = x32 / torch.sqrt(torch.mean(x32 ** 2, dim=-1, keepdim=True) + epsilon)
      if gamma is not None:
          out = out * gamma.to(torch.float32).reshape(-1)
      return out.to(x.dtype)
  ```
- **Import Path**: `from ttnn.operations.rms_norm import rms_norm`
- **Function Signature**:
  ```python
  rms_norm(
      input_tensor: ttnn.Tensor,
      *,
      gamma: Optional[ttnn.Tensor] = None,          # shape (1, 1, 1, W)
      epsilon: float = 1e-6,
      compute_kernel_config: ttnn.ComputeConfigDescriptor = None,  # None -> default_compute_kernel_config()
      memory_config: ttnn.MemoryConfig = None,      # output placement; None -> input's
  ) -> ttnn.Tensor
  ```

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

Perf target (no `LOOSE_CASES` entry carries an `attention:` flag, so the verifier selected it): the
LLM perf loose cases in `feature_spec.py` (`_perf_case`). Their full config contract is `dtype=bfloat16`,
`layout=TILE`, `fp32_dest_acc_en=False`, `math_fidelity=HiFi2` (extras), `gamma_mode=gamma`,
`gamma_dtype=bfloat16`, `gamma_layout=TILE`, soft PCC gate 0.9995, placement INTERLEAVED (Refinement 3) or the
pinned WIDTH/BLOCK shard geometry (Refinement 6). Refinement 1 lands that contract; Refinement 2 builds the
cross-core path the interleaved decode shapes need. `achievable_ns` values were measured on Blackhole p150b at
1350 MHz; this board is a Blackhole p100a, so they are directional references, not acceptance gates here.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [float32, bfloat16]
- **SUPPORTED fp32_dest_acc_en**: [True]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED shape-derived axes**: alignment = {tile_aligned, w_non_aligned, h_non_aligned}; rank = {2, 3, 4}
- **SUPPORTED op-specific axes**: gamma_mode = {gamma, no_gamma}; gamma_dtype = {float32, bfloat16, "none"}; gamma_layout = {ROW_MAJOR, "none"}; memory_layout = {INTERLEAVED}
- **EXCLUSIONS**: `{dtype: float32, fp32_dest_acc_en: False}` (fp32 input requires fp32 accumulation; permanent per the prompt)
- **Regime**: R1 `row_resident` (rows split over the full grid via `split_work_to_cores(..., row_wise=True)`, whole row resident, x read once). R1 predicate failure raises `NotImplementedError` (R2 not built).
- **Cores**: multi-core (up to the full 110-core grid on Blackhole p100a); shapes with one tile-row run on one core
- **Compute config**: `default_compute_kernel_config()` = HiFi4, fp32_dest_acc_en=True; math_fidelity / math_approx_mode passed through ungated
- **Golden baseline**: 226 supported_pass / 0 supported_fail / 0 xpass_drift / 0 xfail_wrong_mode; 6691 xfail_expected (per `verifier_report.json`)

### [ ] Refinement 1 — Numerical configurability + gamma TILE / bfloat8_b

**Goal**: add `False` to `SUPPORTED["fp32_dest_acc_en"]` (bfloat16 and bfloat8_b inputs; the fp32 EXCLUSION stays), `ttnn.bfloat8_b` to `SUPPORTED["dtype"]`, `ttnn.bfloat8_b` to `SUPPORTED["gamma_dtype"]` and `ttnn.TILE_LAYOUT` to `SUPPORTED["gamma_layout"]`. Intermediate CB page formats (`cb_sumsq`, `cb_rinv`, `cb_x_normed`) must follow the DEST width: Float32 pages only when `fp32_dest_acc_en=True` (ledger audit 3; a Float32 page under 16-bit DEST doubles bytes for no precision). Gamma TILE staging is in-kernel: the writer reads gamma tiles (row 0 meaningful) straight into `cb_gamma_tiles` instead of building row-0 tiles from the RM stick; bfloat8_b gamma is TILE-only (INVALID). Cells that fail out of the box (typically `bfloat8_b` with a masked last W tile) go to `EXCLUSIONS`, not a separate refinement.

**Implementation skill**: /numeric-formats-metal, /memory-layouts

**Verifier notes**: lands the full perf-target contract (bf16, fp32_dest_acc_en=False, HiFi2, bf16 TILE gamma), so it goes first. Unblocks 3615 cells on fp32_dest_acc_en, 2839 on gamma_layout=TILE, 962 on dtype=bfloat8_b and 860 on gamma_dtype=bfloat8_b (overlapping counts). Once gamma TILE is in SUPPORTED, the `(1, 1, 32, 32768)` wide loose case becomes a supported cell that fails the R1 L1 predicate (`NotImplementedError`, 2 MiB of resident row). That is an L1-fit failure owned by Refinement 2. Leave it failing; do not put it in EXCLUSIONS or a shape tagger. `(1, 1, 32, 16384)` and `(1, 1, 64, 12288)` bf16 still fit R1 (C = 24 / 46 with gamma on p100a). The perf cases' soft PCC gate is 0.9995 at HiFi2 / 16-bit DEST. If the bf16 DEST sum of squares on W=7168 misses it, that is a precision finding for this refinement's report, not a reason to force fp32 DEST on the perf config. Keep the new paths at the Phase 0 performance bar: same grid fill, batched reader/writer.

**Done when**: every newly supported cell passes except the `(1, 1, 32, 32768)` loose case (R1 predicate, tracked by Refinement 2); verifier loud categories otherwise clean; the perf loose cases at INTERLEAVED pass their 0.9995 PCC gate.

### [ ] Refinement 2 — Cross-core W-split combine (regime R2 `w_split_combine`)

**Goal**: build the deferred R2 regime from `op_design.md` (Regimes table + reserved Tensix-to-Tensix contract): split `w` across a core group of size `G` for shapes where R1's L1 predicate fails or the row split under-fills the grid (`tensor_row_tiles < grid_cores` and `Wt >= 2 * W_SPLIT_MIN_SLICE_TILES`). Each core reduces its slice's sum of squares to col-0 partials, the group combines them (`reduce<SUM, REDUCE_ROW>` with the same finalize post-op), and `rinv` is multicast back with `ttnn.Mcast` / `McastArgs`. Each core then normalizes its own resident slice, so x still crosses DRAM once. Moves the `(1, 1, 32, 32768)` loose case (R1-predicate failure after Refinement 1) to passing. Interleaved decode shapes (`(1, 1, 32, W)`, today one core) and `(1, 1, 64, 12288)` / `(1, 1, 160, 11008)` (2-5 cores) spread across the grid.

**Verifier notes**: scheme-change (dependent `w` axis split cross-core), so it stands alone. It comes before the perf slot because Refinement 3's target shapes (one tile-row decode) are single-core under R1, and the W-split is their only path to the grid. No implementation skill covers cross-core combines yet; required topology: per-core partial `reduce` -> gather of `G-1` partial tiles per row to the combiner (`r mod G` reduce-scatter style, root-only when R == 1) -> finalize -> `McastArgs::sender().send()` of R rinv tiles. Catalog entries `width_split`, `tensix_all_reduce`, `mcast_topology` in `ttnn/ttnn/operations/examples/master.md`. Schedule `row_sumsq_block(b+1)` before `normalize_block(b)` to hide the combine (design stall-shadow row). `G` and the slice width must be `select_blocking()` knobs with a single source. Add one acceptance test pinned to each R2 predicate branch (wide-W, few-rows). Hard dependency: Refinement 4 (WIDTH/BLOCK sharding) reuses this combine.

**Done when**: the `(1, 1, 32, 32768)` loose case passes; regime-pinned acceptance tests pass on both predicate branches; one-tile-row decode shapes (`(1, 1, 32, 7168)` at the perf config) run on more than one core with a measured device-ns win over R1; no device-ns regression across the guard set (one representative per kernel path x layout x placement); golden suite green.

### [ ] Refinement 3 — Speed up the interleaved LLM decode / mid-row perf cases

**Type**: perf

**Goal**: optimize the INTERLEAVED `_perf_case` shapes that are latency-bound on few cores: decode `(1, 1, 32, {1024, 2304, 5120, 7168})` and `(1, 1, 640, {6144, 7168})`, at their exact config (bf16, fp32_dest_acc_en=False, HiFi2, bf16 TILE gamma). Reference `achievable_ns` (p150b, 1350 MHz): 9149 / 17003 / 75825 / 14894 and 101968 / 119307. Clock-scale per the feature_spec header and treat them as directional on this p100a. Phase 0 reference at its own config (fp32 DEST, RM gamma, HiFi4): 32x7168 = 54.5 us on 1 core, 640x7168 = 87.1 us on 20 cores. The levers are knob co-tunes on the R2 / R1 block surface: group size `G` vs slice width, `w_chunk_tiles` vs `in_depth`, `WRITE_BATCH_TILES`. Relevant `master.md` entries: `width_split`, `mcast_topology`, `tensix_all_reduce` (combine topology choice), `sfpu_tile_scope` (finalize post-op on a col-0 rinv tile), `bank_stagger`, `double_buffer`, `compute_block_size`. Also the design's "gamma persistence" lamp: build gamma tiles once per kernel when `Wt * gt` fits. No SUPPORTED change.

**Done when**: measured device-ns improves on the targeted perf shapes at their exact config, their 0.9995 PCC gate still holds, the golden suite is green, and there is no device-ns regression across the config-spanning guard set (one representative per distinct kernel path x layout x placement; include an 8192-row prefill shape, which is at the DRAM copy ceiling and must not move).

### [ ] Refinement 4 — Sharded placements (HEIGHT local, WIDTH / BLOCK cross-core)

**Goal**: add `HEIGHT_SHARDED`, `WIDTH_SHARDED` and `BLOCK_SHARDED` to `SUPPORTED["memory_layout"]` (TILE input, sharded output via `memory_config`). HEIGHT is regime R4: `cb_input_tiles` and `cb_output_tiles` backed on the input / output shards (`ttnn.cb_descriptor_from_sharded_tensor`), block = whole shard, `in_depth = 1`, rows outer / chunks inner so pack order is shard order. WIDTH / BLOCK are R2 realized physically: group = the shard-grid row, slices = the resident shards, the Refinement 2 combine across that row.

**Verifier notes**: complicated sharding (hardest tier), so it comes right after the first perf slot. Hard dependency on Refinement 2's combine. No implementation skill covers placement yet. Native means a core's own shard is consumed in place through the shard-backed CB. Re-reading it through a `TensorAccessor` is not sharding support; send it back even if golden is green. `TensorAccessor` stays only for interleaved I/O, gamma, and a genuinely non-local gather. The resilience cases include ragged shard grids (prime Rt/Wt, `shard_h`/`shard_w` not dividing the tensor) and non-aligned W/H under poisoned padding. Handle padding shards / partial tiles on the last core row and column without folding padding into the reduction. Unblocks ~1600 cells per placement value (210 / 222 / 231 blocked only by placement). L1-infeasible shard geometries are skipped by the harness (`infeasible_skipped`), not the op's fault. A CB-allocation OOM inside the op is a real failure.

**Done when**: all TILE cells at the three sharded placements pass (input + output sharded); the sharded `_perf_case` and sharded resilience / poison loose cases (TILE) pass; no regression on interleaved; golden suite green with loud categories clean.

### [ ] Refinement 5 — ROW_MAJOR input / output layout

**Goal**: add `ttnn.ROW_MAJOR_LAYOUT` to `SUPPORTED["layout"]` via the in-kernel tilize-wrapped path from `op_design.md`: reader `read_sticks_for_tilize` per C-wide chunk into `cb_input_rm`, compute `tilize<C>` into `cb_input_tiles`, normalize unchanged, compute `untilize<C>` into `cb_output_rm`, writer `write_sticks_after_untilize` (only valid sticks / bytes). Applies across every placement in SUPPORTED at that point (interleaved, HEIGHT / WIDTH / BLOCK sharded) and both R1 and R2 regimes. Non-aligned W / H are handled natively: zero-fill `cb_input_rm` once so partial-W lanes are finite, mask with the existing partial-scaler path, and write back only logical rows.

**Implementation skill**: /memory-layouts

**Verifier notes**: layouts tier, after the sharding structure exists, so RM extends it once. RM gamma is already supported; the resilience cases mirror gamma layout to input layout. `bfloat8_b + ROW_MAJOR` is INVALID. RM restricts C to divisors of Wt (helper quanta, design Mechanism caps), so wide prime Wt (37, 53, 89, 127) needs a chunking answer that keeps C coarse. The RM rows already in `l1_ledger.md` become live CBs. Wrapping the op in `ttnn.to_layout` / `tilize` / `untilize` is only a `[~]` partial-tick escape hatch. It must name the kernel blocker, and the in-kernel path must be filed as `Refinement 5b` with the wrapper marked for deletion. One dispatch per call stays mandatory either way. Unblocks 2272 cells (210 blocked only by layout).

**Done when**: all ROW_MAJOR cells (interleaved and sharded, aligned and non-aligned, ranks 2-4, bf16 / fp32) pass; RM resilience loose cases pass; golden suite green with loud categories clean.

### [ ] Refinement 6 — Speed up the sharded LLM perf cases

**Type**: perf

**Goal**: optimize the sharded `_perf_case` shapes at their pinned geometry and exact config (bf16, fp32_dest_acc_en=False, HiFi2, bf16 TILE gamma): WIDTH_SHARDED decode `(1, 1, 32, {1024, 2304, 5120, 7168})` (reference 4110 / 4617 / 5267 / 5481 ns) and BLOCK_SHARDED `(1, 1, 640, {6144, 7168})`, `(1, 1, 7168, {1024, 2048})`, `(1, 1, 8192, 1024)` (reference 19039 / 21669 / 18871 / 25947 / 25640 ns). References are p150b at 1350 MHz: clock-scale them and treat them as directional on this p100a. Relevant `master.md` entries: `tensix_all_reduce`, `tensix_all_reduce_compute`, `mcast_topology` (combine across the shard-grid row), `sfpu_tile_scope` (finalize), `compute_block_size` (whole-shard block), `row_reduce_accumulate`. No SUPPORTED change.

**Done when**: measured device-ns improves on the targeted sharded perf shapes at their pinned geometry, the 0.9995 PCC gate holds, the golden suite is green, and there is no device-ns regression across the config-spanning guard set (one representative per distinct kernel path x layout x placement).
