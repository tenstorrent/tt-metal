# Operation Requirements: rms_norm

## Definition
- **Formula**: `out[..., c] = x[..., c] / sqrt( mean_{c'<W}( x[..., c']² ) + eps ) · gamma[c]` (gamma term omitted when `gamma is None`)
- **PyTorch Reference**:
  ```python
  def rms_norm_ref(x, gamma=None, eps=1e-6):
      xf = x.float()
      out = xf / torch.sqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + eps)
      if gamma is not None:
          out = out * gamma.float().reshape(-1)
      return out.to(x.dtype)
  ```
- **Import Path**: `from ttnn.operations.rms_norm import rms_norm`
- **Function Signature**:
  ```python
  rms_norm(
      input_tensor: ttnn.Tensor,
      *,
      gamma: Optional[ttnn.Tensor] = None,
      epsilon: float = 1e-6,
      compute_kernel_config: Optional[ttnn.ComputeConfigDescriptor] = None,
      memory_config: Optional[ttnn.MemoryConfig] = None,
      program_config=None,
  ) -> ttnn.Tensor
  ```

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

**Perf target selection.** No `feature_spec.LOOSE_CASES` entry carries an `attention:` perf flag, so the
perf regions below are verifier-selected. They are the `_PERF_BASE` loose cases (`_perf_case(...)`),
whose `extras` carry a config-matched `achievable_ns` (Blackhole p150b, 1350 MHz AICLK, clock-scale it
before use: `achievable_ns · 1350 / actual_aiclk_mhz`) and a soft `pcc_threshold = 0.9995`. Their
**full config contract** is: `dtype=bfloat16`, `layout=TILE`, `fp32_dest_acc_en=False`,
`math_fidelity=HiFi2`, `gamma_dtype=bfloat16`, **`gamma_layout=TILE`**, placement INTERLEAVED (or the
pinned WIDTH/BLOCK shard geometry). Refinements 1 and 2 exist to land that contract on the
interleaved path; the sharded perf cases are landed by Refinement 4.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [float32, bfloat16]
- **SUPPORTED fp32_dest_acc_en**: [True] (EXCLUSION: `{dtype: float32, fp32_dest_acc_en: False}` — precision convention, permanent)
- **SUPPORTED layout**: [TILE, ROW_MAJOR] (native: tilize/untilize inside the program)
- **SUPPORTED alignment**: tile_aligned, w_non_aligned, h_non_aligned
- **SUPPORTED rank**: 2, 3, 4
- **SUPPORTED gamma**: gamma_mode ∈ {gamma, no_gamma}; gamma_dtype ∈ {float32, bfloat16, "none"}; gamma_layout ∈ {ROW_MAJOR, "none"}
- **SUPPORTED memory_layout**: [INTERLEAVED] (DRAM or L1)
- **Regime**: R1 row-resident — tile-rows split over the full compute grid (`split_work_to_cores(row_wise=True)`), whole row resident per core (x crosses DRAM once), blocks of `block_rows × row_stride_tiles`, normalize streamed in `w_chunk_tiles` chunks; all knobs host-selected by the block-selection function.
- **Cores**: full grid (active cores = `min(Rt, 120)` on this Blackhole)
- **Compute config**: caller's `ComputeConfigDescriptor` passed as-is (`fp32_dest_acc_en=True` gated; any fidelity / approx)
- **Golden baseline**: 436 supported_pass / 0 supported_fail / 0 xpass_drift / 0 xfail_wrong_mode; 6481 xfail_expected (per `verifier_report.json`)

### [ ] Refinement 1 — TILE-layout gamma + HEIGHT_SHARDED placement (dataflow-boundary refinement)

**Goal**: add `ttnn.TILE_LAYOUT` to `SUPPORTED["gamma_layout"]` and `ttnn.TensorMemoryLayout.HEIGHT_SHARDED`
to `SUPPORTED["memory_layout"]`, both natively in the reader/writer:

- **TILE gamma** (knob-turn): the `(1,1,1,W)` tiled gamma already is the row-0-valid tile the row-broadcast
  multiply wants — the reader reads gamma tile pages `[chunk·c, chunk·c + c)` straight into `cb_gamma`
  (one barrier per chunk, pad columns `≥ Wc` skipped as for x) and skips the row-0 relayout. Applies to
  both input layouts (TILE and ROW_MAJOR x) and both gamma dtypes in SUPPORTED. Gamma padding may be
  poisoned (`extras.poison_padding`): only columns `≥ W` of the last gamma tile are padding and they only
  multiply output padding columns — keep it that way (rows 1–31 are never read by the row broadcast).
- **HEIGHT_SHARDED** (knob-turn, regime R6 of `op_design.md`): each core's shard = whole rows × full W,
  already in L1. Back `cb_x` (TILE) / `cb_x_rm` (RM) with the input shard and the output CB with the
  output shard via `ttnn.cb_descriptor_from_sharded_tensor` (zero-copy); block = whole shard, `K = 1`;
  the work split follows the shard grid (not `split_work_to_cores`); the output is allocated with the
  caller's sharded `memory_config` (the golden runner passes the input's). Ragged last shard
  (`auto_shard_config` / resilience shapes with prime `Rt`) must work.

**Implementation skill**: /memory-layouts

**Verifier notes**:
- Lands half of the perf-1 contract (`gamma_layout=TILE`); Refinement 2 lands the other half. Build the TILE-gamma read batched (≥ c pages per barrier), not per tile — Refinement 3 optimizes this exact path.
- `/memory-layouts` covers the gamma layout leg only. HEIGHT_SHARDED placement has no skill: use the `ttnn.cb_descriptor_from_sharded_tensor` CB-placement pattern. **A `TensorAccessor` read of a core's own local shard is NOT an implementation of HEIGHT_SHARDED** (it merely tolerates it) and will be sent back — the verifier checks the dataflow, not the test colour. The L1 budget query (`get_memory_view(...).largest_contiguous_bytes_free_per_bank`) is already net of the resident shards; the block-selection function must use the shard as the block (no `cb_x` allocation at all on this path).
- `validate()` must also accept a HEIGHT_SHARDED output `memory_config` (the current trailing "output memory_config must be interleaved" check rejects it) and must still reject WIDTH/BLOCK.
- **Known residual failure (do not EXCLUDE)**: once TILE gamma is supported, the `_WIDE` loose case `(1,1,32,32768)` (bf16, Wc = 1024, 2 MiB of x) becomes a SUPPORTED cell that R1 cannot hold resident — the builder raises `NotImplementedError("…needs W-split regime R3")`. It stays failing and is moved to passing by Refinement 4. `(1,1,32,16384)` and `(1,1,64,12288)` fit R1 (probe: `r,c,K = 1,27,19` and `1,48,8`) and must pass here.
- Cells that combine these values with `fp32_dest_acc_en=False` or bf8b stay xfail until Refinement 2.

**Done when**: every golden cell with `gamma_layout=TILE` and/or `memory_layout=HEIGHT_SHARDED` whose other axes are in SUPPORTED passes (except the `(1,1,32,32768)` loose case named above), verifier loud categories at 0, acceptance + precision baseline still green.

### [ ] Refinement 2 — Numerical configurability: `fp32_dest_acc_en=False`, bfloat8_b input and gamma

**Goal**: add `False` to `SUPPORTED["fp32_dest_acc_en"]` (bf16 / bf8b input; `{float32, False}` stays the
permanent EXCLUSION), `ttnn.bfloat8_b` to `SUPPORTED["dtype"]` (TILE only — bf8b+RM and bf8b+non-aligned
are INVALID in `feature_spec.py`) and `ttnn.bfloat8_b` to `SUPPORTED["gamma_dtype"]` (TILE gamma from
Refinement 1). Intermediate CB formats follow the DEST width; `math_fidelity` / `math_approx_mode` keep
passing through unchanged (HiFi2 must work — the perf cases pin it). Cells that fail out of the box land
in `EXCLUSIONS`, not in their own refinement.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**:
- Depends on Refinement 1 (bf8b gamma only exists as TILE gamma). Completes the perf-1 contract: after this, every `_PERF_BASE` interleaved case is a SUPPORTED cell.
- **This is not a descriptor-only change for this op** — the Σx² mechanism is packer L1 accumulation, which is fp32-DEST-only hardware (`op_design.md` mechanism-cap table; catalog `row_reduce_accumulate`). For `fp32_dest_acc_en=False` swap `sumsq_block` to a mechanism that is legal with 16-bit DEST (e.g. square a chunk into a `c`-tile window and fold it with `reduce<SUM, REDUCE_ROW, …>` + `Accumulate` across chunks), selected by a CT flag so the fp32-DEST path keeps L1 accumulation. Keep it chunk-batched — Refinement 3 measures this exact path.
- Page-format audit (L1 ledger check 3): with `fp32_dest_acc_en=False` the Float32 pages of `cb_sumsq`, `cb_rstd`, `cb_normed` (and the Float32 mask, which must match `cb_sumsq`'s format — implementer friction note) are "over" — derive them from DEST width (bf16 when off). That also halves `cb_normed` (`r·c` pages, the largest term after `cb_x` on wide prefill rows) and lets the block selector pick larger `r`/`c`.
- Perf loose cases carry a soft `pcc_threshold = 0.9995` at HiFi2 + bf16 DEST — check it explicitly on the 8192-row cases.

**Done when**: every golden cell with `fp32_dest_acc_en=False` (bf16 / bf8b input), `dtype=bfloat8_b` or `gamma_dtype=bfloat8_b` whose other axes are in SUPPORTED passes or is in a justified EXCLUSION; the interleaved `_PERF_BASE` loose cases pass at their soft PCC gate; loud categories at 0.

### [ ] Refinement 3 — Speed up the interleaved prefill perf profile

**Type**: perf

**Goal**: optimize the interleaved `_PERF_BASE` prefill cases at their **exact** config (bf16, TILE, TILE bf16
gamma, `fp32_dest_acc_en=False`, HiFi2, DRAM interleaved): `(1,1,8192,{1024,2304,5120,7168})`
and `(1,1,7168,{1024,2048})`, toward (and below) their clock-scaled `extras.achievable_ns`. Rt = 224–256
tile-rows fills the 120-core grid under R1, so these are DRAM-bandwidth-bound: Phase 0 measured
8192×1024 at 88.0 µs (~380 GB/s, HiFi4/fp32-DEST/RM-gamma, implementer profile), 8192×4096 at 341 µs.
Relevant patterns in `ttnn/ttnn/operations/examples/master.md`: `double_buffer` / the `x_block_depth`
perf lamp (with `rows_cap = 3` and `r` < `rows_cap` on the widest rows, the reader stalls on the
next block), `read_coalescing`, `bank_stagger`, `trid_double_issue`, `noc_placement` (DRAM side);
`compute_block_size` (reconfig elision; the `r`-first vs `K` block-selection lamp — co-tune
`block_rows`/`w_chunk_tiles`/stream depth), `compute_fusion` (gamma DestReuse instead of the
`cb_normed` round trip), `row_reduce_accumulate` (ReduceTile rstd path). No SUPPORTED change.

**Done when**: measured device-ns improves on the most-impacted prefill shape(s) at the exact config,
each targeted case meets its clock-scaled `achievable_ns` (or the gap is explained against the DRAM
roofline via `/perf-ceiling-dm`), the soft PCC gate (0.9995) holds, the golden suite is green, and no
regression across the config-spanning guard set (one representative per kernel path × layout
{TILE, RM} × placement {DRAM, L1, HEIGHT_SHARDED} × gamma {none, RM, TILE}).

### [ ] Refinement 4 — Cross-core W-split (regime R3): logical W-split on interleaved + WIDTH_SHARDED + BLOCK_SHARDED

**Goal**: add `ttnn.TensorMemoryLayout.WIDTH_SHARDED` and `ttnn.TensorMemoryLayout.BLOCK_SHARDED` to
`SUPPORTED["memory_layout"]`, and move the interleaved `(1,1,32,32768)` `_WIDE` loose case (R1 residency
`NotImplementedError`) to passing, by building the design's R3 scheme: W split across a group of G cores,
each core reducing its slice, partials combined across the group.

**Verifier notes**:
- Scheme-change — the topology is the work, so it stands alone. No implementation skill covers it; the required topology is pinned in `op_design.md` ("R3 Tensix-to-Tensix contract"): per row-group, every core runs `sumsq_block` + a local `reduce<SUM, REDUCE_ROW, AccumulateViaAdd>` (no post-op) into `cb_partial`; contributors write their partial tiles into the root's `cb_gather` slot and signal its semaphore; the root (which also owns a W slice) reduces `(block_rows, G)` with the post-op (·1/W, +eps, rsqrt) and multicasts `cb_rstd` back with the multicast helper (host `ttnn.Mcast` + `attach()`, kernel `McastArgs` `sender().send()` / `receiver().receive()` — not raw `noc_async_write_multicast` + semaphores). Stall shadow: compute `x·γ` for the local slice while waiting for rstd.
- Two placements, one scheme: **WIDTH/BLOCK_SHARDED** — the slice is the resident shard (`cb_descriptor_from_sharded_tensor`, zero-copy, never re-read through a `TensorAccessor`); BLOCK = WIDTH within each grid row. **Interleaved** — the logical variant reads each core's slice from DRAM; its regime predicate (`F(1,K_floor) > B` OR few tile-rows vs the grid) must be pinned exactly and covered by regime-pinned tests (the split must not trigger only on some grid sizes).
- `core_w_tiles` / `w_partial_elems` become per-core quantities (only the last slice carries the partial tile; ragged shard grids like the perf cases' `[64, 704] × (9,10)` make the last slice narrower) — move them from CT to RT or to a second kernel group; host `_n_sq(g, core_w_tiles)` already derives from the knob.
- Hardest remaining refinement (complicated sharding + cross-core combine) → ordered before any further easy axis. If WIDTH + interleaved land but BLOCK (2-D grid, ragged) does not fit the pass, ship `[~]` and file `Refinement 4b` for BLOCK_SHARDED.
- Few-row interleaved shapes (decode `32×W`, the `_WIDE` 64×12288, resilience `Rt ≤ 7` wide shapes) currently run on `Rt` cores; this refinement makes the W-split *available* for them — tuning when to take it is Refinement 5.

**Done when**: every golden cell with `memory_layout ∈ {WIDTH_SHARDED, BLOCK_SHARDED}` whose other axes are in SUPPORTED passes (resilience, pad-poison, translated and sharded `_PERF_BASE` loose cases included; `infeasible_skipped` shard geometries excepted), `(1,1,32,32768)` interleaved passes, regime-pinned tests exist for the interleaved W-split predicate, loud categories at 0.

### [ ] Refinement 5 — Speed up the decode / few-row profile on the W-split scheme

**Type**: perf

**Goal**: the decode and few-row `_PERF_BASE` cases — interleaved `(1,1,32,{1024,2304,5120,7168})`,
`(1,1,640,{6144,7168})` and their WIDTH_SHARDED / BLOCK_SHARDED pinned geometries (`extras.shard_shape` /
`core_grid`) — at their exact config, toward their clock-scaled `achievable_ns` (e.g. interleaved
32×7168 = 14.9 µs, WIDTH_SHARDED 32×7168 on 28 cores = 5.5 µs). Phase 0 runs 32×4096 on **one core**
(33.9 µs); 640-row cases use only 20 cores. Use Refinement 4's W-split for these (pin the interleaved
predicate from measurement) and tune the combine: `tensix_all_reduce`, `tensix_all_reduce_compute`,
`mcast_topology`, `width_split` in `ttnn/ttnn/operations/examples/master.md` (T2/T3 — this phase is
the combine, don't add unrelated knobs). No SUPPORTED change.

**Done when**: measured device-ns improves on the most-impacted decode / few-row shapes at the exact
config, each targeted case moves toward (ideally meets) its clock-scaled `achievable_ns`, the soft PCC
gate holds, the golden suite is green, and no regression across the config-spanning guard set (one
representative per kernel path × layout × placement, incl. R1 prefill shapes that must not flip to the
W-split).

### [ ] Refinement 6 — Speed up the block-sharded prefill profile

**Type**: perf

**Goal**: the BLOCK_SHARDED `_PERF_BASE` prefill cases `(1,1,7168,{1024,2048})` (`[736,128]×(8,10)`,
`[736,192]×(11,10)`, ragged) and `(1,1,8192,1024)` (`[1024,128]×(8,8)`) at their exact config, toward
their clock-scaled `achievable_ns` (18.9–25.9 µs — 4–7× under the interleaved references, so with no
DRAM traffic these become compute/combine-bound). Relevant patterns in
`ttnn/ttnn/operations/examples/master.md`: `compute_block_size` (one block = whole shard, reconfig
elision), `compute_fusion`, `sfpu_tile_scope` (rsqrt / post-op on column 0 only), `reduce_block`,
`mcast_topology` for the per-grid-row rstd broadcast. No SUPPORTED change.

**Done when**: measured device-ns improves on the most-impacted block-sharded shape(s) at the exact
config toward the clock-scaled `achievable_ns`, the soft PCC gate holds, the golden suite is green,
and no regression across the config-spanning guard set (one representative per kernel path × layout ×
placement).
