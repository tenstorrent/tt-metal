# Operation Requirements: rms_norm

## Definition
- **Formula**: `out[..., h, w] = x[..., h, w] · rsqrt( (1/W) · Σ_{w'<W} x[..., h, w']² + ε ) · γ[w]` (γ ≡ 1 when absent; `W` is the logical last-dimension size in elements, padding never enters the sum or the divisor)
- **PyTorch Reference**:
  ```python
  def rms_norm_ref(x, gamma=None, epsilon=1e-6):
      xf = x.to(torch.float32)
      out = xf / torch.sqrt(torch.mean(xf ** 2, dim=-1, keepdim=True) + epsilon)
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
      gamma: Optional[ttnn.Tensor] = None,          # logical shape (1, 1, 1, W)
      epsilon: float = 1e-6,
      compute_kernel_config: Optional[ttnn.ComputeConfigDescriptor] = None,  # None -> default_compute_kernel_config()
      memory_config: Optional[ttnn.MemoryConfig] = None,                       # output placement; None -> input's
  ) -> ttnn.Tensor
  ```

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

> **Naming collision — read this.** `op_design.md` names its *regimes* R1 / R3 / R1-HS / G-MC. In this file
> those are always written "regime R1", "regime R3", etc. "Refinement N" is a queue entry. They are unrelated
> numbers.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [float32, bfloat16]
- **SUPPORTED fp32_dest_acc_en**: [True] (`{float32, False}` is an EXCLUSION — natively rejected precision combination)
- **SUPPORTED layout**: [TILE, ROW_MAJOR] (native: tilize / untilize inside the kernels, no host-side conversion)
- **SUPPORTED shape-derived axes**: alignment ∈ {tile_aligned, w_non_aligned, h_non_aligned}, rank ∈ {2, 3, 4}
- **SUPPORTED op-specific axes**: gamma_mode ∈ {gamma, no_gamma}; gamma_dtype ∈ {float32, bfloat16, "none"}; gamma_layout ∈ {ROW_MAJOR, "none"}; memory_layout ∈ {INTERLEAVED}
- **Regime**: regime R1 (row-parallel, row-resident): tile-rows split over the full compute grid (`split_work_to_cores(row_wise=True)`, 120 Tensix cores on this Blackhole); a block of `block_rows` tile-rows stays resident in L1 for both the reduction and the scaling pass; x crosses DRAM once. Rows too wide to keep resident are refused with `NotImplementedError` (regime R3 not built).
- **Compute config**: caller's `ComputeConfigDescriptor` passed verbatim (math_fidelity / math_approx_mode honoured, not gated); default HiFi4 + fp32_dest_acc_en.
- **Golden baseline**: 436 supported_pass / 0 supported_fail / 0 xpass_drift / 0 xfail_wrong_mode; 6481 xfail_expected; 33900 invalid_skipped (per `verifier_report.json`).

### [x] Refinement 1 — TILE-layout gamma, with γ off the reader's critical path

**Goal**: add `ttnn.TILE_LAYOUT` to `SUPPORTED["gamma_layout"]`, natively, for both input `Layout`s and every placement in SUPPORTED. A TILE γ of logical shape `(1, 1, 1, W)` is padded to one tile-row of `Wt` tiles whose **row 0 is exactly what the `BroadcastDim::Row` multiply consumes**, so the scaling stages (`scale_by_gamma` and everything downstream) are unchanged. Only the γ supply path changes (today: compact RM row in `cb_gamma_row` → per-(block, chunk) row-0 expansion into `cb_gamma_tiles`).

**Implementation skill**: /memory-layouts

**Verifier notes**:
- *Lands half of the Refinement 3 perf contract* (the interleaved `_perf_case` loose cases pin `gamma_layout=TILE`). Build it to the performance bar, not correct-only: Refinement 3 optimizes this exact path.
- *Performance-conformance finding carried here.* The γ expansion runs on the reader RISC-V once per (block, chunk), **after** the block's x load, and its `cb_gamma_tiles` reserve blocks until compute drains the previous γ chunks — so the reader cannot prefetch block k+1 while block k is scaled. That defeats the design's `x_depth = 2` overlap (perf lamp L1). Measured (bf16, `Layout::TILE` input, `Layout::ROW_MAJOR` γ, HiFi2, fp32 DEST, Blackhole 1350 MHz): decode 32×7168 runs 41.5 µs with γ vs 27.6 µs without; reader NCRISC 32.7 µs vs 7.4 µs. The verifier already replaced word-by-word `volatile` copies with L1→L1 NoC reads (1.15–1.28×). The remaining serialization is design lamp L4. Fix it in the γ rewrite for **both** γ `Layout`s: move the γ supply off the reader (the writer RISC-V is idle until the first output chunk; push γ chunk c+1 before draining output chunk c, to stay deadlock-free and one chunk ahead), or expand once per kernel when the budget allows.
- *No new W-proportional γ buffer without a budget guard.* `_WIDE` loose cases `(1,1,32,16384)` and `(1,1,64,12288)` (bf16, TILE γ) become in-SUPPORTED with this refinement and fit regime R1 only because γ residency is the compact row (`Wt·32·es_g` bytes, 1/32 of `Wt` γ tiles). Resident γ tiles (`Wt·T_g` bytes, e.g. 1 MiB at W = 16384) would push them out of residency. A predicate-guarded resident fast-path with the compact fallback is fine. Re-reading γ tiles from DRAM per (block, chunk) adds up to `1/R` of the x traffic; count it in the `l1_ledger.md` data-movement budget if chosen.
- *Expected residual failing cell, not a defect of this refinement.* `_WIDE (1,1,32,32768)` (bf16, `Wt = 1024` tiles, 2 MiB/row) also becomes in-SUPPORTED. It is refused by the regime-R1 residency predicate (`NotImplementedError: … regime R3 is not built`) and will read as `supported_fail`. That is an L1 residency boundary, the same class as OOM. **Do not** add it to EXCLUSIONS and do not invent a shape tagger. Refinement 4 moves it to passing.
- Poison-padding loose cases fill the TILE γ padding with 1000. Pad columns of γ only multiply x pad columns (already zeroed by the reader) and rows ≥ 1 are never read by `BroadcastDim::Row`, so no masking is needed on γ. Keep it that way.

**Done when**: every golden cell with `gamma_layout=TILE` inside SUPPORTED passes, except `(1,1,32,32768)` (regime-refused, owned by Refinement 4). On the decode γ shapes (32×{1024, 2304, 5120, 7168}), measured with `test_rms_norm_perf_baseline.py`:
- the reader (NCRISC) duration with γ is within ~10 % of its no-γ duration (today 32.7 µs vs 7.4 µs at 32×7168);
- device-ns improves over the Phase 0 γ numbers recorded in `verification_report.md`.

γ still costs one extra FPU pass on compute; that part is not this refinement's target. No regression on the Phase 0 cells.

**Outcome**: TILE γ in SUPPORTED. Golden interleaved: 282/283 TILE-γ cells pass; the one failure is the regime-refused `(1,1,32,32768)`. RM-γ (280) and no-γ (140) cells still pass. γ supply (staging-row fill + row-0 expansion) moved to the writer RISC-V for both γ `Layout`s. Expansion is once per kernel (resident `Wp` γ tiles) when L1 allows, else one chunk ahead of the output drain. Decode 32×7168, bf16, HiFi2, fp32 DEST, Blackhole: reader NCRISC with γ is 7.5 µs (no-γ 7.4 µs; Phase 0 32.7 µs). Device-ns 41.4 → 40.9 µs (RM γ) / 41.0 µs (TILE γ). 32×{1024, 2304, 5120}: 9.20/15.77/30.64 → 8.8/15.3–15.4/30.2 µs. 640×{6144, 7168}: 54.5/63.2 → 53.2/61.9 µs (RM γ) and 51.0/59.3 µs (TILE γ). Decode is now compute-bound on the γ FPU pass (TRISC 39.5 µs vs 26.1 µs no-γ; lamp L6, Refinement 3). Remaining finding: on many-row prefill, TILE γ is 3–10 % slower than RM γ (8192×7168: ~585 vs ~566 µs). Ablating only the TILE row-0 DRAM fill returns it to the no-γ time, so the cost is 120 Tensix cores × 2·Wt tiny γ DRAM reads. The fix is G-MC (read once, multicast), already queued in Refinement 6; it was not attempted here because it is a cross-core topology change.

### [x] Refinement 2 — Numerical configurability: 16-bit DEST, bfloat8_b input and gamma

**Goal**: add `False` to `SUPPORTED["fp32_dest_acc_en"]` (bfloat16 input; `{float32, False}` stays the native EXCLUSION), `ttnn.bfloat8_b` to `SUPPORTED["dtype"]`, and `ttnn.bfloat8_b` to `SUPPORTED["gamma_dtype"]`. Intermediate-CB formats must follow DEST width. Cells that fail for a *structural capability* reason go to `EXCLUSIONS`, not to their own refinement.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**:
- *Depends on Refinement 1*: `gamma_dtype=bfloat8_b` exists only with a TILE γ (`{gamma_dtype: bfloat8_b, gamma_layout: ROW_MAJOR}` is INVALID). Extracting row 0 from a Bfp8_b γ tile must be format-aware: one shared exponent per 16-datum face row in the 64-byte exponent section, then mantissa bytes. A byte copy of the bf16 segment offsets is wrong.
- *Lands the other half of the Refinement 3 perf contract* (`fp32_dest_acc_en=False`, bf16, HiFi2). Precision must hold the perf cases' soft PCC gate of **0.9995** at W up to 7168 elements. Under 16-bit DEST, `square_accumulate` sums `Wt` (up to 224 tiles) x² products in a 16-bit DEST accumulator, which is design lamp L8. See `master.md` → `row_reduce_accumulate` for the accuracy trade.
- Intermediate page format is already single-sourced in `intermediate_format()` (`rms_norm_program_descriptor.py`), used by both the CB declarations and `footprint()`. Do not reintroduce a Float32 page under 16-bit DEST (`l1_ledger.md` audit 2).
- *Bfp8_b traps in the dataflow kernels*: reader and writer derive the element size as `es = tile_bytes / 1024`, which is 1 for Bfp8_b (1088-byte tile). That is harmless only because Bfp8_b is TILE-only (INVALID with ROW_MAJOR) and the W-pad zeroing never runs on tile-aligned shapes. `INVALID` also excludes bf8b with `w_non_aligned` / `h_non_aligned`, but the translated tests still run 2 such cells (`invalid_unexpected`). Make `validate()` refuse `{dtype: bfloat8_b, alignment: w_non_aligned}` via EXCLUSIONS: zeroing pad columns inside a Bfp8_b tile is exponent-section work, a structural capability gap. Pad-*tile* zeroing (`Wp > Wt`) writes all-zero bytes, which is a valid Bfp8_b zero tile.
- Ordering: lighter than Refinement 1 (precision tier) and depends on it, hence second.

**Done when**: every cell in the new SUPPORTED values, at every placement in SUPPORTED, passes, or is a structural-capability EXCLUSION named in the changelog. `numerical-precision` failures stay failing (refinement signal, not EXCLUSIONS). The precision matrix test from the skill is added. No regression.

**Outcome**: All three axis values are in SUPPORTED. One structural-capability EXCLUSION was added: `{bfloat8_b, w_non_aligned}` (W-pad zeroing inside a Bfp8_b tile is exponent-section work). Golden interleaved slices: bf8b 450/450 runnable cells pass; 16-bit DEST 454/454 pass. Precision matrix: 328/328 at PCC ≥ 0.99 (≥ 0.9999 at HiFi2+). Bfp8_b γ row 0 is extracted format-aware (exponent head + two mantissa rows). The compute kernel is unchanged. Exact-config perf baseline for Refinement 3 is recorded in the changelog (32×7168: 40.3 µs; 8192×7168: 589 µs).

### [x] Refinement 3 — Speed up the interleaved perf loose cases at their exact config

**Type**: perf

**Goal**: No loose case carries an `attention:` perf flag, so the region is free-selected. Target the **interleaved** `_perf_case` entries in `feature_spec.LOOSE_CASES` at their **exact** config: bf16, `fp32_dest_acc_en=False`, HiFi2, bf16 TILE γ, `TensorMemoryLayout::INTERLEAVED`. All of it is in SUPPORTED after Refinements 1 and 2. Measure each against its clock-scaled `achievable_ns`.

The verifier's proxy measurements (fp32 DEST, ROW_MAJOR γ — *not* the target config; `test_rms_norm_perf_baseline.py`) show two regimes inside regime R1:
- (a) **Many-row prefill** (7168/8192 × 1024…7168): already under `achievable_ns`, DRAM-bound at ~390–420 GB/s (≈ 80 % of peak). The remaining headroom is data movement. Relevant `master.md` entries: `bank_stagger` (Blackhole interleaved height split: every core starts on the same bank column, lamp L9), `double_buffer` / `trid_double_issue` (writer flushes per C-tile quantum per tile-row), `noc_placement`. Run `/perf-ceiling-dm` first to confirm the headroom is real.
- (b) **Single-Tensix-core decode** (32 × W, one tile-row): compute-bound on TRISC1. Relevant: `compute_block_size` (reconfig removal, lamp L10, is safe where bf16 x / bf16 γ / bf16 intermediates under 16-bit DEST keep the format constant), `sfpu_tile_scope` (lamp L7: rsqrt on column 0 only), `compute_fusion` (lamp L6).

Several T1/T2 knobs fit one phase. Occupancy of the few-row shapes (decode, 640-row) is **not** this phase's lever: that is the regime R3 W-split, Refinement 4. No SUPPORTED change.

**Done when**: measured device-ns improves on the most-impacted interleaved `_perf_case` shape(s) at the exact loose-case config; the soft PCC gate 0.9995 still holds; the golden suite is green; and there is no regression across the config-spanning guard set (one representative per kernel path × `Layout` × γ `Layout` × placement in SUPPORTED). Record every interleaved `_perf_case` vs its clock-scaled `achievable_ns` in the changelog.

**Outcome**: Measured at the exact config (bf16, 16-bit DEST, HiFi2, bf16 TILE γ), Blackhole at 1350 MHz (= the reference clock, so `achievable_ns` needs no scaling):
- **Decode** (the most-impacted cells): 32×{1024, 2304, 5120, 7168} went from 8.61/14.89/28.77/40.26 µs to 7.64/12.36/23.00/30.87 µs (1.13–1.30×). Two levers did it:
  1. The TILE-γ resident supply reads each γ tile's row-0 prefix straight into `cb_gamma_tiles`. Writer γ supply 32.5 k → 11 k cycles; the writer is no longer co-binding.
  2. The two scaling chains run 8-tile DEST windows (`SCALE_BLOCK_TILES`): 67/69 → 37/47 cycles per tile.
- **Other shapes**: 640×{6144, 7168} 50.5/57.3 → 49.9/56.1 µs. Prefill is within ±2 % except 8192×7168, 588.7 → 580.9 µs.
- **Against `achievable_ns`**: every interleaved perf case beats it except 32×7168 (30.9 vs 14.9 µs, was 2.70× over, now 2.07×).
- **What binds now**:
  - Decode is single-Tensix-core compute: `square_accumulate` 84 cycles/tile (two full-tile unpacks) plus the two scaling passes, ~40 k cycles. That is within ~20 % of the unpack floor of this 4-full-unpack-per-tile schedule. The remaining decode headroom is occupancy, i.e. the regime R3 W-split (Refinement 4 / 6).
  - 640-row and prefill are DRAM-bandwidth-bound (~430–500 GB/s). The 640-row shapes serialize a whole-grid read phase and a whole-grid write phase because each core owns one block. Prefill shows NoC0-read / NoC1-write arbitration gradients (east-column writers finish last). A NoC swap (reader NoC1, writer NoC0) measured flat.
- **Measured null levers, kept as knobs or recorded**:
  - bank stagger (L9): `BANK_STAGGER` knob, ships off;
  - `GAMMA_DIRECT_SEGMENTS` (fewer bytes, more requests): 1 % better at 640-row, 2–6 % worse on prefill;
  - L10 reconfig removal: within noise, not built.
- **Next to try**: a G-MC γ multicast to remove the 120-core γ re-read (Refinement 6). Not done here because it is a cross-core topology change.

### [x] Refinement 4 — Cross-core Σx² combine (regime R3): WIDTH/BLOCK-sharded placement + interleaved W-split

**Goal**: add `ttnn.TensorMemoryLayout.WIDTH_SHARDED` and `ttnn.TensorMemoryLayout.BLOCK_SHARDED` to `SUPPORTED["memory_layout"]` (input and output placement), natively. With the same topology, build the design's regime R3 on **interleaved** inputs: a logical W-split, each core reading its W-slice from DRAM once. That moves the regime-refused `_WIDE (1,1,32,32768)` cell (row too wide for L1 residency, in SUPPORTED since Refinement 1) from `supported_fail` to passing, and spreads few-tile-row shapes over the grid: decode 32×W runs on **1** Tensix core today, and the 640-row shapes use 20 of 120. All of this cuts the **dependent** W axis across Tensix cores. It is one scheme-change: the combine topology is the work, so it stands alone.

**Verifier notes** (no skill covers cross-core sharding; required topology stated here):
- *Topology* (`op_design.md` → "Deferred-regime Tensix-to-Tensix contracts", regime R3). Per row-group of G_w Tensix cores sharing a tile-row range:
  - each core runs `square_accumulate` on its resident W-slice (already emits the pre-collapse raw Σx² partial per tile-row, by design);
  - it collapses locally (`reduce<SUM, REDUCE_ROW>`, scaler 1.0, no post-op) and sends the column-0 partial to the row-group root;
  - the root sums G_w partials (`reduce<…, AccumulateViaAdd, …, ReduceWithinTile::Skip>`), applies ×(1/W) +ε rsqrt, and multicasts `inv_rms` back via `ttnn.Mcast` + `McastArgs` (semaphore handshake; the root is also a worker);
  - `scale_by_gamma` and the next block's `square_accumulate` run in the wait (stall-shadow; Phase 0 already orders γ first for this reason).
  - Catalog: `tensix_all_reduce` (root reduce + one multicast back for a 1-D line with a tiny payload; tree reduce for 2-D/busy grids), `mcast_topology`, `width_split`.
- *Sharded placement is native*: back the input CB on the shard with `ttnn.cb_descriptor_from_sharded_tensor` (zero-copy, one block = the whole resident shard, clamped by any mechanism cap) and the output CB on the output shard. **Never** read a core's own shard through a `TensorAccessor`. The reader must not write into the caller's shard, so W-pad masking moves to compute (lamp L5: `prepare_reduce_mask` + masked multiply on the last tile). The poison-padding loose cases include WIDTH/BLOCK cells.
- *Shapes the harness will throw at it*:
  - width shards spanning **multiple grid rows** (`eval/SHARDING.md` §2: derive the slice→core map from the shard spec);
  - ragged shard grids (`_RESILIENCE_SHAPES`, the ragged BLOCK `_perf_case` geometries pinned via `extras.shard_shape` / `core_grid`);
  - ROW_MAJOR sharded inputs (tilize from the shard CB);
  - γ sliced per core along W.
  - Use uniform `max_assigned` push/pop quanta for ragged slices (`tensix_all_reduce` notes).
- *Interleaved regime selection* is a measured trade. The occupancy clause in `op_design.md` (`tensor_tile_rows < grid/2` with `Wt ≥ 2·W_CHUNK_TILES`) is a placeholder "to be fixed by measurement". Pin each branch with a test (as Phase 0 pins `x_depth`). Each W-slice stays resident, so x still crosses DRAM once. The rejected regime R2 (re-read x) stays rejected. Update `l1_ledger.md` and the regime table: regime R3 moves from `deferred` to `built`, with its cross-core traffic row filled in.
- *Hardest generality refinement* (complicated sharding + cross-core restructure), so it precedes the local placement (Refinement 5 reuses its compute-side mask and shard-backed CBs).
- Partial-tick: if the interleaved W-split, ragged BLOCK, or ROW_MAJOR sharded inputs do not fit a focused pass, ship `[~]` and file `Refinement 4b` naming exactly what remains. It runs next.

**Done when**:
- every WIDTH_SHARDED and BLOCK_SHARDED golden cell (cartesian + loose) inside SUPPORTED passes, or is a structural-capability EXCLUSION recorded in the changelog;
- no sharded cell reads its own shard through a `TensorAccessor`;
- `(1,1,32,32768)` passes;
- the interleaved decode `_perf_case` shapes (32 × {1024, 2304, 5120, 7168}) run on more than one Tensix core, with a measured device-ns win over regime R1 at the exact loose-case config;
- no regression on interleaved regime-R1 cells.

**Outcome**: `WIDTH_SHARDED` and `BLOCK_SHARDED` are in SUPPORTED for both input `Layout`s; regime R3 is built (root gather + `ttnn.Mcast` inv_rms broadcast per row-group).
- Golden: every WIDTH/BLOCK cell passes (cartesian TILE 1,740 + ROW_MAJOR 630, loose 192; 0 failures; 3 harness `INFEASIBLE_L1` skips). `(1,1,32,32768)` passes. Interleaved TILE + ROW_MAJOR cells: 0 failures.
- Native placement: TILE shards are zero-copy CBs (no NoC read of the input); ROW_MAJOR shards are re-laid from local L1 (no TensorAccessor). The W-pad mask is in compute (partial reduce scaler).
- Decode, exact config: 32×{1024, 2304, 5120, 7168} run on 8 / 18 / 23 / 23 Tensix cores, 7.64 / 12.36 / 23.00 / 30.87 → 4.57 / 5.25 / 6.34 / 7.30 µs (1.67–4.23×). 32×7168 is now under `achievable_ns` (0.49×; was 2.07×). 640-row: 1.15×. R1 prefill is A/B-identical.
- What binds now: the tall BLOCK-sharded geometries (7168/8192 × 1024/2048) are 1.8–2.2× over `achievable_ns`. Their flat root reduce serializes the sum and the full-tile finalize of all R rows on one core per group.
- Next to try: reduce-scatter the combine across the group, and column-0 scoping of the finalize (lamps L2 / L7). Both belong to Refinement 6, which owns sharded perf.

### [x] Refinement 5 — HEIGHT-sharded placement (local reduction)

**Goal**: add `ttnn.TensorMemoryLayout.HEIGHT_SHARDED` to `SUPPORTED["memory_layout"]` (input and output). Each Tensix core holds full-W rows, so the reduction stays local: this is a placement (knob-turn) change, the design's regime R1-HS.

**Verifier notes**:
- Native path only. Back `cb_x_tiles` (TILE) / the RM stick CB on the input shard with `ttnn.cb_descriptor_from_sharded_tensor` (zero-copy), and `cb_output` on the output shard. One block = the whole resident shard (sharded block-size default) unless L1 forces a split, which must be justified in the ledger.
- Work distribution comes from `shard_spec()`, not `split_work_to_cores`.
- The W-pad mask is the compute-side one from Refinement 4 (the caller's shard is read-only).
- An accessor read of a core's own shard means this value was **not implemented**, even if the cells pass.
- Ragged H shards (`_RESILIENCE_SHAPES`, e.g. `(99991, 64)`, `(1,1,333,544)`) leave a partial last shard.
- Lighter than Refinement 4 and reuses its compute-side mask and shard-backed CBs, so it comes after it.

**Done when**: every HEIGHT_SHARDED golden cell (cartesian + loose) inside SUPPORTED passes, with zero NoC reads of the input on the sharded path. No regression.

**Outcome**: `HEIGHT_SHARDED` is in SUPPORTED (input and output, both `Layout`s, ROW_MAJOR and COL_MAJOR shard orientation). It is regime R1-HS: the R3 sharded path with G = 1 (no combine). TILE shards are zero-copy; ROW_MAJOR shards are re-laid from local L1. The work split comes from the shard spec. Golden HEIGHT: 1,167 cartesian pass / 0 fail (315 expected xfail), 93/93 loose. W = 11008 TILE γ needed a new L1 fallback: streaming γ read directly, with no staging row. Measured (bf16, HiFi2, 16-bit DEST, TILE γ), HEIGHT vs INTERLEAVED device ns: 1024×1024 6.4 vs 13.7 µs; 640×6144 19.1 vs 43.3 µs; 7168×1024 14.3 vs 76.3 µs; 8192×1024 14.4 vs 85.8 µs.

### [x] Refinement 6 — Speed up the decode and sharded perf loose cases

**Type**: perf

**Goal**: free-selected region (no loose case is flagged). Target the `_perf_case` entries that regime R3 serves, at their exact config (bf16, `fp32_dest_acc_en=False`, HiFi2, bf16 TILE γ):
- the interleaved decode shapes (32 × {1024, 2304, 5120, 7168}): today 9.2–41.5 µs on one Tensix core vs references of 9.1–75.8 µs (32×7168: 41.5 µs vs 14.9 µs);
- the WIDTH/BLOCK-sharded geometries pinned via `extras.shard_shape` / `core_grid`. These references are aggressive, e.g. 8192×1024 block-sharded at 25.6 µs.

Co-tune the knobs Refinement 4 exposes:
- G_w (W-slice cores per row-group) and block rows per combine round (lamp L2: fewer combine rounds vs. fewer tiles per core);
- combine topology (`tensix_all_reduce`: root + multicast vs tree reduce under a busy grid);
- γ read-once-and-multicast (design regime G-MC, `shared_input_reuse` / `mcast_topology`) — validate by building it and measuring device-ns, not by ablation;
- `trid_double_issue` where the shard caps the read block.

One T3 restructure is a whole phase; do not pack it with unrelated knobs. No SUPPORTED change.

**Done when**: measured device-ns improves on the most-impacted decode / sharded `_perf_case` shape(s) toward their clock-scaled `achievable_ns`; the soft PCC gate 0.9995 holds; the golden suite is green; and there is no regression across the config-spanning guard set (one representative per kernel path × `Layout` × placement).

**Outcome**: Measured at the exact config (bf16, 16-bit DEST, HiFi2, bf16 TILE γ), Blackhole at 1350 MHz (= the reference clock). Before/after is an A/B against the Refinement 5 kernels in the same session.
- **Tall BLOCK shards** (most impacted): 7168×1024 40.9 → 16.7 µs, 7168×2048 47.1 → 22.3 µs, 8192×1024 56.4 → 22.6 µs (2.11–2.49×). All three are now under `achievable_ns` (18.9 / 25.9 / 25.6 µs; they were 1.8–2.2× over).
- **WIDTH decode**: 32×{1024, 2304, 5120, 7168} 3.5–4.8 → 2.7–4.0 µs (1.20–1.30×).
- **Interleaved decode**: 4.6–7.3 → 3.7–6.5 µs (1.13–1.22×).
- **Other shapes**: 640-row (BLOCK and interleaved), prefill and HEIGHT are at parity (medians of 3 within ±1 %). The guard set is 0.98–1.15×.
- **Two levers did it**:
  1. `FINALIZE_COL0` (lamp L7): ×1/W, +ε and rsqrt fused into one SFPU pass over the 8 column-0 vectors, instead of three full-tile passes (96 vectors, ~1.8 k cycles per tile-row). On the root this was 38 k of 51 k combine cycles.
  2. `R3_SCATTER_ROWS_PER_OWNER` (lamp L2, `tensix_all_reduce` reduce-scatter): each of the G members sums and finalizes ⌈R/owners⌉ tile-rows, then unicasts them back. The gather is contributor-major, so each contributor sends one write per owner.
- **Measured and dropped**: concurrent rotating-sender multicast for the all-gather. It was 1.0–2.5× slower and erratic across cores.
- **Kept but gated**: the γ(0)-before-combine hoist applies only with more than one owner. With a single root it delayed every member's multicast.
- **What binds now** (8192×1024 BLOCK, ~30 k cycles per Tensix core):
  - square+collapse 7.1 k;
  - γ pass 5 k;
  - the NoC exchanges: the gather wait still 2.9 k after the γ pass, plus ~7.7 k inv_rms wait including the owners' combine.
  The exchanges are link-bandwidth-bound all-to-all traffic inside an 8-core row on the NoC0 torus (~64 KB per Tensix core each way); the 2 KB-per-tile issue cost does not bind (one write per owner measured the same).
- **Next to try**: halve the exchange bytes, by sending only faces 0/2 of each column-0 partial or by using a half-tile (32×16) intermediate format. Also route west-bound flows on NoC1.
- **Not attempted here**:
  - G-MC γ multicast: a second T3 restructure, and the writer's γ supply finishes in ~1.1 µs, off the critical path on every decode / sharded target shape. Its remaining opportunity is the interleaved-prefill TILE-γ re-read (3–10 %, Refinement 1 finding), which is outside this refinement's targets.
  - `trid_double_issue`: zero-copy shards have no reads to pipeline.
