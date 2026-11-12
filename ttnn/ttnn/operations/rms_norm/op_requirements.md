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

### [x] Refinement 1 — TILE-layout gamma + HEIGHT_SHARDED placement (dataflow-boundary refinement)

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



### [x] Refinement 1b — TILE-layout gamma + HEIGHT_SHARDED placement (dataflow-boundary refinement) (debug: fix gate violations)

**Goal**: fix the hard violation from Refinement 1 so the completion gate's three bullets hold.

**Verifier notes** (mechanical, from the harness completion gate):

```
Bullet 2 FAIL: acceptance/refinement tests failing:
  - tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_sharded_tile_gamma.py::test_height_sharded_auto[nog-fp32-rm-(128, 8192)] - NotImplementedError: rms_norm: row too wide for single-core residency; needs W-split regime R3
```

**Done when**: the gate passes — zero hangs in SUPPORTED, acceptance + refinement tests pass, golden majority with no regression.
**Outcome**: root cause was an L1 leak + a wrong CB budget bound, not the kernel. (1) The refinement test's `oom_guard` skips (infeasible fp32 (128,8192) TILE shards) left the sharded input alive in the pytest traceback, so later cases in the module saw a low leaked L1 buffer; `_run` now deallocates in `finally`. (2) `_cb_budget` now bounds CBs by the lowest allocated L1 address (CB region grows up from the allocator base) instead of `largest_contiguous_bytes_free_per_bank`, which turned a leaked/low buffer into a TT_THROW CB clash. Unit dir 229 passed / 0 failed; full golden 1297 passed, 1 failed (the excepted `(1,1,32,32768)` loose case, R3/Refinement 4), 0 hangs.

### [x] Refinement 2 — Numerical configurability: `fp32_dest_acc_en=False`, bfloat8_b input and gamma

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
**Outcome**: landed with no kernel change. Packer L1 accumulation measured legal and accurate with 16-bit DEST, so the Σx² mechanism swap was not needed. `cb_rstd`/`cb_normed` follow the DEST width; `cb_sumsq` + mask stay Float32 (they are the L1 accumulator). Golden slices: 196/196 fp32-DEST-off loose cells (all 12 interleaved perf cases at PCC ≥ 0.9999), 179 cartesian bf8b / DEST-off cells pass, 0 fail.

### [x] Refinement 3 — Speed up the interleaved prefill perf profile

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
**Outcome**: device-ns at the exact config (Blackhole p150b, 1350 MHz = reference clock, so no scaling), before → after vs `achievable_ns`: 8192×1024 103.1 → 90.2 µs (96.7), 8192×2304 217.9 → 188.4 (211.3), 8192×5120 477.0 → 419.0 (738.3), 8192×7168 681.4 → 569.7 (1032.3), 7168×1024 86.8 → 74.7 (89.4), 7168×2048 172.7 → 146.0 (172.9). Every case now meets its target. The lever: the TILE gamma reader fetched whole 2 KiB pages (31 of 32 rows are padding) once per block, which is as many DRAM bytes as one x tile-row per block. It now reads only row 0 (the 64 B head of face 0 and face 1). DRAM roofline, measured with `ttnn.clone` on the same tensors (same read+write bytes, no compute): 83.2 / 182.8 / 413.4 / 556.7 / 71.8 / 146.8 µs. The no-gamma op runs at the clone time (8192×1024: 83.0, 8192×7168: 557.4), so the op is DRAM-bound. The remaining 0–8% over clone is the gamma multiply pass plus the 16 cores that own 3 tile-rows on 256-row shapes (256 rows over 120 cores). Lever tried and parked: `x_block_depth = 2` with `block_rows = 1` (read/write overlap) is 10–12% slower (8192×1024: 89.3 → 98.2), because serial read-all/write-all phases use DRAM better than mixed traffic at 2–3 rows per core. It stays a live knob (`X_BLOCK_DEPTH`, `BLOCKS_PER_CORE`) at its byte-identical default of 1. Next things to try: keep TILE gamma resident when K = 1 (drops the per-block re-read entirely), and a sub-row W-split of the leftover 16 tile-rows to remove the 3-row tail. Neither was tried here: every target is already met and the remaining gap to the copy roofline is ≤ 8%.

### [x] Refinement 4 — Cross-core W-split (regime R3): logical W-split on interleaved + WIDTH_SHARDED + BLOCK_SHARDED

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

### [x] Refinement 5 — Speed up the decode / few-row profile on the W-split scheme

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

**Outcome**: device-ns at the exact config (Blackhole p150b, 1350 MHz), before → after (`achievable_ns`):
- Interleaved: 32×1024 8 903 → 4 105 (9 149); 32×2304 16 885 → 4 698 (17 003); 32×5120 30 004 → 6 023
  (75 825); 32×7168 39 093 → 6 796 (14 894); 640×6144 55 201 → 45 218 (101 968); 640×7168 60 796 → 48 871
  (119 307).
- WIDTH_SHARDED: 32×1024 3 864 → 3 214 (4 110); 32×2304 4 596 → 3 802 (4 617); 32×5120 5 386 → 4 164
  (5 267); 32×7168 5 543 → 4 374 (5 481).
- BLOCK_SHARDED: 640×6144 11 057 → 9 944 (19 039); 640×7168 13 786 → 12 645 (21 669).

Every targeted case now meets its `achievable_ns`. Prefill is unchanged; it stays on R1 because R3 cannot
exceed R1's core count once Rt ≥ grid.

Remaining bottlenecks:
- **Interleaved decode** is bound by the two DRAM phases around the combine. All G cores read their slice
  at once (~1.5 µs), and all write their 8 tiles at once after rstd. On 32×7168 the far cores' write
  takes ~2 µs, against ~0.75 µs near the root.
- **Sharded decode** is bound by the combine chain. On WS 32×7168: gather incast ~0.75 µs, root combine
  ~1.0 µs (of which the column-scoped rsqrt post-op is 0.66 µs), rstd multicast ~0.5 µs (latency;
  independent of payload size and of the handshake), then the ·rstd pass ~0.45 µs.
- **BLOCK_SHARDED** hides an 8 µs gamma read: 64 row-0 reads per core, re-read by all 10 cores of each
  grid column.

Next levers, not tried here:
- A fused column-0 post-op (`c_skip`: ·1/W, +eps, rsqrt in one 8-vector SFPU walk, ~−0.45 µs on the root).
  It needs raw sfpi internals, while this refinement kept to public `VectorMode` APIs.
- A read-once gamma column multicast for BLOCK grids (`mcast_topology`). This is Refinement 6's
  block-sharded profile.
- A tree / reduce-scatter combine for large G. The measured G sweep is flat at G = 18–44.

### [x] Refinement 6 — Speed up the block-sharded prefill profile

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

**Outcome**: device-ns at the exact config (Blackhole p150b, 1350 MHz = reference clock), before → after (`achievable_ns`): 7168×1024 `[736,128]×(8,10)` 36 039 → 18 784 (18 871); 7168×2048 `[736,192]×(11,10)` 45 507 → 25 318 (25 947); 8192×1024 `[1024,128]×(8,8)` 50 403 → 25 112 (25 640). All three now meet their target. The bottleneck was the root combine: 23 rows × G partials + the rsqrt post-op on one core (17 µs) while the group waited. Levers kept: a row-wise reduce-scatter combine (owners relay rstd rows to the root, which keeps the one helper multicast); the stall shadow for blocks ≥ 4 rows; a half-tile gather sent per owner run as soon as it is packed; the per-core partial on ReduceTile instead of accumulate + SFPU collapse; that partial interleaved into sumsq. The guard set moved the right way too: BLOCK 640-row −4%, WS decode −3 to −5%, interleaved decode −3 to −4%; R1 prefill flat. Now compute-bound per core: sumsq ≈ 5.7 µs + partial ≈ 1.2 (interleaved, ≈ 8 µs together) → combine ≈ 1.3 → x·γ ≈ 4.0 → ·rstd ≈ 3.5 µs, with the gather / relay / multicast mostly hidden. Next: fuse the per-row sumsq L1-accumulate calls (23 chain calls per block) into fewer, wider calls, and the fused column-0 post-op (`c_skip`, raw sfpi) on the owners. Not tried: every target is already met, and both need lower-level work than this refinement's scope.
