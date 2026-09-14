# Operation Requirements: rms_norm

## Definition
- **Formula**: `output[..., r, w] = input[..., r, w] * rsqrt( (1/W) * Σ_w input[..., r, w]² + epsilon ) * gamma[w]` (gamma term omitted when absent)
- **PyTorch Reference**:
  ```python
  def rms_norm_ref(x, gamma=None, epsilon=1e-6):
      xf = x.to(torch.float32)
      y = xf * torch.rsqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + epsilon)
      if gamma is not None:
          y = y * gamma.to(torch.float32).reshape(-1)
      return y.to(x.dtype)
  ```
- **Import Path**: `from ttnn.operations.rms_norm import rms_norm`
- **Function Signature**:
  ```python
  rms_norm(
      input_tensor: ttnn.Tensor,
      *,
      gamma: Optional[ttnn.Tensor] = None,
      epsilon: float = 1e-6,
      compute_kernel_config: ttnn.ComputeConfigDescriptor = None,
      memory_config: Optional[ttnn.MemoryConfig] = None,
      program_config=None,  # accepted for signature compatibility; ignored
  ) -> ttnn.Tensor
  ```

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.
> **Perf refinements**: entries marked `**Type**: perf` add nothing to SUPPORTED; they are accepted on a **measured device-ns win** (Tracy, `scripts/run_safe_pytest.sh --profile <explicit ::node ids>` — never `-k`, it silently runs zero tests under tracy) with **no regression on the config-spanning guard set** (one representative per distinct kernel path × layout × placement: TILE/R1, TILE/R2, ROW_MAJOR/R1, WIDTH_SHARDED/R3, gamma and no-gamma, bf16 and fp32). Every `achievable_ns` in `feature_spec.LOOSE_CASES` is stamped at 1350 MHz on blackhole_p150b — scale it by `reference_aiclk_mhz / actual_aiclk_mhz` (read the clock from the profiler, not a board default) before comparing.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16, float32]
- **SUPPORTED fp32_dest_acc_en**: [True] (the maxed-out precision corner; `{float32, False}` is a permanent EXCLUSION)
- **SUPPORTED layout**: [TILE_LAYOUT, ROW_MAJOR_LAYOUT] (output layout == input layout; native tilize/untilize in-kernel)
- **SUPPORTED shape-derived axes**: rank ∈ {2, 3, 4}; alignment = tile_aligned only (verifier-added tagger axis — the kernel gates on it)
- **SUPPORTED op-specific axes**: gamma_mode ∈ {gamma, no_gamma}; gamma_dtype ∈ {bfloat16, float32, "none"}; gamma_layout ∈ {TILE, ROW_MAJOR, "none"}; memory_layout ∈ {INTERLEAVED, WIDTH_SHARDED} (WIDTH_SHARDED incl. non-rectangular shard grids and a ragged/padded last shard)
- **Cores**: multi-core from Phase 0 — rows split over `num_row_groups` groups × `num_w_splits` W-split cores per group (R1 row split, R2 W-split with root gather + rstd multicast, R3 zero-copy resident shards); 130-core grid filled for prefill, W-split occupancy term for decode
- **Compute config**: `default_compute_kernel_config()` = HiFi4 + fp32_dest_acc_en=True + math_approx_mode=False; `math_fidelity` / `math_approx_mode` pass through ungated
- **Golden baseline**: see `verification_report.md` → Verifier CLI Summary (all loud categories 0)

### [x] Refinement 1 — Numerical configurability: 16-bit DEST accumulation + bfloat8_b

**Goal**: add `False` to `SUPPORTED["fp32_dest_acc_en"]` (for bfloat16 and bfloat8_b input — `{float32, False}` stays in `EXCLUSIONS`, natively refused forever), add `ttnn.bfloat8_b` to `SUPPORTED["dtype"]` and to `SUPPORTED["gamma_dtype"]`. `compute_kernel_config` is already exposed and passed as-is; the work is making every accumulated-intermediate page format and every DEST-capacity-dependent quantity follow the config instead of assuming fp32 DEST. Cells that fail out of the box for a structural reason land in `EXCLUSIONS`, not in their own refinement (`bfloat8_b + ROW_MAJOR` on either tensor is already INVALID, so no exclusion is needed there).

**Implementation skill**: /numeric-formats-metal

**Verifier notes**:
- **This refinement alone lands the perf-1 contract.** `feature_spec.LOOSE_CASES` pins every perf case (`_PERF_BASE`) to bf16 / TILE / **`fp32_dest_acc_en=False`** / HiFi2 / bf16 TILE gamma / INTERLEAVED (and the sharded variants); the mandatory ≥7× decode case `(1, 1, 32, 7168)` is one of them. Refinement 2 (perf) measures and optimizes *that exact config*, so build this path to the Phase 0 performance bar (grid filled by the W-split, batched reader/writer, depth-2 CBs) — a correct-only stub is wasted work R2 will rewrite.
- **Single source of truth for the accumulation width.** `rms_norm_program_descriptor.py` currently hard-codes `P32 = tile_size(float32)` as the page size of `cb_sumsq_partial`, `cb_partial_collapsed`, `cb_gather`, `cb_rstd_handoff`, `cb_rstd`, `cb_normed`, in `Footprint.per_block`, and in the `P32_BYTES` named CT the writer uses as the gather/mcast payload stride. Derive one `acc_tile_bytes` / `acc_format` from `compute_kernel_config.fp32_dest_acc_en` and route all of those through it (16-bit → `Float16_b` pages, 2048 B, halving the collective payload). The ledger's `Page format` column (`l1_ledger.md`) must be updated to state the rule "follows the DEST width" per row — a `Float32` page under 16-bit DEST is an audit-3 finding. The formats of `cb_x_tiles` / `cb_gamma_tiles` / `cb_output_tiles` stay the tensor dtypes (relayout only); for bf8b they become `Bfp8_b` with `tile_size(bfloat8_b)` pages (1088 B).
- **DEST capacity.** `DEST_AUTO_LIMIT` in the helpers already follows the config (8 tiles half-sync at 16-bit vs 4 at fp32) — nothing to hand-tune, but re-check `Dst::D0`-only chains and the reduce's DST usage compile in both modes.
- **Precision risk under 16-bit DEST (the real work).** Σx² per tile-row accumulates `core_w_tiles` tiles in DEST (bf16 rounding per step), the collapse rounds once, and the root combine sums up to `num_partials` (≤ 26 at decode 7168) collapsed partials — the catalog (`row_reduce_accumulate`) shows bf16 *accumulation* error grows with width. Gates: hard rel-RMS ≤ 0.04 (bf16) / 0.10 (bf8b), soft PCC 0.9995 on the perf cases. Levers if it does not clear: keep the collapse on `ReduceTile` (FPU matmul-with-ones accumulates internally in fp32 regardless of DEST width) rather than `AccumulateViaAdd`; consider `ReduceFp32Mode::Accurate`; the `rsqrt` post-op is SFPU and unaffected.
- **The unlock activates ~200 additional loose cells at once**: 12 `_PERF_BASE` cases (4 of them WIDTH_SHARDED with explicit shard specs, all exact covers) and 24 `_RESILIENCE_SHAPES` × (TILE, ROW_MAJOR) × (INTERLEAVED, WIDTH_SHARDED). Phase 0 already handles the awkward geometries these produce (`auto_shard_config(PAD)`: prime tile counts → single-tile shards over up to 127 cores; `W=11008` → 115 × 3 tiles with a padded last shard and a non-rectangular bounding box — regime-pinned in `tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_ragged_width_shard.py`); `(100000, 64)` WIDTH_SHARDED is an `infeasible_skipped` S1 OOM, uncharged. Run the whole golden directory before ticking — the ROW_MAJOR resilience shapes at 16-bit DEST are the largest untested surface.
- bf8b activations: the RM path is INVALID for bf8b, so only the TILE reader/writer and the R3 shard path need the new format; bf8b gamma likewise TILE only. `auto_shard_config` is dtype-aware (bf8b granule), so sharded bf8b cells appear automatically.
- Dependency: none. Ordering: first, because R2 (perf) cannot measure the flagged config until this lands.

**Done when**: every golden cell with `fp32_dest_acc_en=False` (bf16, bf8b) and every `bfloat8_b` / `gamma_dtype=bfloat8_b` cell passes; `{float32, fp32_dest_acc_en=False}` is still refused with `ExcludedCell`; all `_PERF_BASE` loose cases pass their soft PCC gate; `python3 -m eval.verify_supported` reports `supported_fail = xpass_drift = xfail_wrong_mode = 0`; the acceptance directory (`tests/ttnn/unit_tests/operations/rms_norm/`) is green.

**Landed** (2026-09-14): SUPPORTED `dtype`/`gamma_dtype` += bfloat8_b, `fp32_dest_acc_en` += False; `{float32, False}` still `ExcludedCell`. Whole golden directory 1708 passed / 388 xfailed / 0 failed / 0 xpass (Phase 0: 656 passing). Post-R1 device baseline for Refinement 2 at the flagged config (`test_rms_norm_perf_shape_flagged_config[decode_7168]`): **9208 ns**, 26 cores (Phase 0 HiFi4/fp32 DEST: 10288 ns). No `EXCLUSIONS` were needed — every named cell passed out of the box (bf16 / 16-bit DEST rel-RMS ≤ 0.0072 at HiFi2, bf8b ≤ 0.011 at HiFi4; the LoFi corner is the only place the error approaches the gates and it is fidelity-, not DEST-width-, bound).

### [ ] Refinement 2 — Speed up the perf-flagged decode profile (32 × 7168, interleaved)

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES` flags `(1, 1, 32, 7168)` bf16 / TILE / INTERLEAVED / HiFi2 / `fp32_dest_acc_en=False` / bf16 TILE gamma as the mandatory perf target: reference `achievable_ns = 104259` at 1350 MHz with `minimum_expected_speedup = 7.0` ⇒ goal **≤ 14894 ns** (clock-scaled), soft PCC gate 0.9995. Measure this exact config (available after Refinement 1) and optimize it toward the goal using the relevant `ttnn/ttnn/operations/examples/master.md` patterns. No SUPPORTED change.

**Verifier notes**:
- Phase 0 baseline at the *supported* corner (HiFi4, fp32 DEST, 26 cores as a 13 × 2 rectangle, `W_TILES_PER_CORE_TARGET = 16` → Wc = 8–9 tiles): **10288 ns** (Tracy, single fresh run) — already under the 14894 ns goal at the wrong config. Re-measure at the flagged config first; the 16-bit DEST build halves the collective payload, so expect it to be at or below this number before any lever is turned.
- This shape is one block per core (`Rt = 1`), so the whole kernel is the collective's latency chain: gather (25 unicasts of one 4 KiB → 2 KiB tile into the root + 25 semaphore incs) → root combine → rstd multicast to a 13 × 2 rectangle → normalize → store. The Blocking Model's lamps that apply, all knob-turns on the planner's surface: **grid-synchronization** (`W_TILES_PER_CORE_TARGET` 8 / 32 / full-grid `Cw = min(Wt, grid)` — fewer W-splits shrink the collective, more shrink per-core DRAM bytes; catalog `width_split`, `tensix_all_reduce`), **`partial_payload`** (only faces 0 and 2 of a collapsed tile carry column 0 — two 512 B writes per sender halve gather and mcast bytes; a `mcast_pipe` payload-size change, not a topology change), **`sfpu_scope`** (catalog `sfpu_tile_scope`: col-0 `c_skip` body for the mul/add/rsqrt finalize, ~2–4× on the finalize), **`format_reconfig`** (`ReduceDataFormatReconfigMode::NONE` / `DataFormatReconfig::Disabled` where the host proves both formats equal — under 16-bit DEST with bf16 x, *every* boundary is bf16 → the reconfigs are pure MMIO waste; catalog `compute_block_size` second lever, up to 1.19×), and the 13 × 2 rectangle's NoC placement (catalog `noc_placement`, `tensix_all_reduce_ring_transport`: Blackhole rows vs columns).
- All of the above are ⭐/⭐⭐ levers; pack several into this phase. Keep the one-round `cb_gather` / `cb_rstd` mechanism cap intact (slot addresses must not move) — `software_pipelined_blocks` is not applicable at one block per core.
- The `l1_ledger.md` finding folded in here: `cb_gather` is over-allocated on the 25 non-root cores by design (uniform descriptors); it stays, but if the payload lever lands, its page size shrinks with it — keep the ledger row current.

**Done when**: measured device-ns on `(1, 1, 32, 7168)` at the flagged config improves versus the post-R1 baseline and meets the clock-scaled ≤ 7× goal; the soft PCC gate still holds; the golden suite is green; no regression across the config-spanning guard set (TILE/R1 `(2, 4, 128, 512)`, TILE/R2 `(1, 1, 64, 12288)`, ROW_MAJOR/R1, WIDTH_SHARDED/R3 `(1, 1, 32, 2048)` on 8 cores, no-gamma and fp32 representatives).

### [ ] Refinement 3 — Prefill: intra-core overlap and reader placement (8192 × {1024, 2304, 5120, 7168})

**Type**: perf

**Goal**: the interleaved prefill perf cases (`_PERF_BASE`, bf16 / HiFi2 / 16-bit DEST) carry `achievable_ns` of 96744 / 211345 / 738307 / 1032281 at 1350 MHz. Phase 0 (HiFi4 / fp32 DEST) measured **124832 ns on 8192 × 1024 — 1.29× slower than the reference** — and 669646 ns on 8192 × 7168 (1.54× faster). Bring every prefill case to or under its clock-scaled reference using the knob-turn levers below; no SUPPORTED change.

**Verifier notes**:
- Root cause on 8192 × 1024 (R1, 130 row-groups of ~2 tile-rows): `block_rows = min(core_row_tiles, block_rows_max_l1)` gives **one block per core**, so the reader's whole assignment is read before compute starts and the writer starts after compute ends — no intra-core overlap despite `DEPTH_X = DEPTH_OUT = 2` (the depth buys nothing with a single block). This is the planner's **overlap lamp**: measure `block_rows = ceil(core_row_tiles / 2)` (two blocks, depth 2) against the current coarsest block; the knob is already a host parameter (`derive_blocking`), so this is a one-line policy in §H3 plus measurement. Catalog: `double_buffer`, `compute_block_size` (the trade is fixed per-block cost vs overlap — measure, don't assume).
- Second lever, same shape class: the 130 row-groups are laid out with `a = 1` (Cw = 1) so groups tile the grid in x-major order — check the reader line orientation against catalog `noc_placement` (row/diagonal vs column, reads on NoC0 / writes on NoC1 — the descriptor already uses the reader/writer defaults).
- Third lever: `format_reconfig` elision (as in R2) — five phase boundaries per block, seven blocks per core on 8192 × 7168.
- **Not in this refinement**: the gamma re-read (every one of the 130 cores reads the full 64 KiB gamma at W = 1024 → 8.3 MB of DRAM traffic on top of 16 MB of x, i.e. +52 % of the input bytes — the design's "≤ 14 %" figure holds only for W = 7168). That is regime R4 (a new mcast topology) and stands alone as Refinement 4. Land the knob-turns first so R4 is measured against a tuned baseline.

**Done when**: measured device-ns improves on 8192 × 1024 (the most-impacted shape) and no prefill perf case regresses; each prefill case is at or under its clock-scaled `achievable_ns`, or the report states the measured ceiling (`/perf-ceiling-dm`) that prevents it; golden green; no regression on the guard set.

### [ ] Refinement 4 — Gamma column broadcast (regime R4: read gamma once per W-slice, multicast down the grid column)

**Type**: perf

**Goal**: implement the deferred regime **R4 `gamma_column_broadcast`** from `op_design.md`: one injector core per W-slice reads its `core_w_tiles` gamma tiles from DRAM and multicasts them (`mcast_pipe` `SenderPipe` / `ReceiverPipe`, host `Mcast1D(PerColumn)` / a second `Mcast2D` family at a disjoint `base_sem_id`) into `cb_gamma_tiles` on every other row-group's core holding the same slice, instead of every active core re-reading its slice. Target shapes: the interleaved prefill perf cases (`(1, 1, 8192, {1024, 2304, 5120, 7168})`) where gamma traffic is 52 % → 15 % of the input bytes. No SUPPORTED change.

**Verifier notes**:
- **Scheme-change, stands alone**: the operand is reuse-shared along `row` (constant across row-groups) and the design built the stepping stone (every core lands gamma in the same `cb_gamma_tiles`, so R4 only changes who writes it). Validate by **building it and measuring device-ns**, not by a remove-gamma ablation — a broadcast's benefit is under-shown by such an ablation. Keep the RM-gamma path in mind: the injector must tilize (or the receivers must each tilize the multicast stick block) — the cheapest correct choice is to multicast the *tilized* `Wc` tiles after the injector's own `tilize`, so receivers land finished tiles.
- Catalog: `shared_input_reuse` (⭐⭐⭐, 1.71× on a 22-core re-read at 2.4 MB), `mcast_topology` (a 2-D work split needs 1-D mcasts along the axis the operand does *not* vary with — here `PerColumn` for gamma). Decode shapes have one row-group and gain nothing; guard them for no regression (the extra semaphore family must be inert when `num_row_groups == 1`).
- Deadlock check: the gamma phase must complete on every core before any core blocks in the per-block rstd collective (`SEM_MCAST_READY/CONSUMED` are already taken — use fresh semaphore ids).
- Dependency: order after Refinement 3 so the win is measured against the tuned prefill baseline.

**Done when**: measured device-ns improves on 8192 × 1024 and 8192 × 2304 with the golden suite green, decode and sharded cases unchanged (guard set), and the `l1_ledger.md` data-movement budget updated (gamma DRAM crossings: `num_w_splits` per program instead of `active_cores`).

### [ ] Refinement 5 — Speed up the sharded decode geometries (R3 collective latency)

**Type**: perf

**Goal**: the four WIDTH_SHARDED perf cases (`[32, 128]` on 8 × 1 for W = 1024 → 4110 ns; `[32, 256]` on 9 × 1 for 2304 → 4617 ns; `[32, 160]` on 8 × 4 for 5120 → 5267 ns; `[32, 256]` on 7 × 4 for 7168 → 5481 ns; all at 1350 MHz, bf16 / HiFi2 / 16-bit DEST) are pure collective-latency chains on resident shards (x and out never cross the NoC). Phase 0 measured **5761 ns** on the acceptance geometry `(1, 1, 32, 2048)` over 8 × 1 (HiFi4 / fp32 DEST) — ~1.4× the 8-core reference. Bring each sharded case to or under its clock-scaled reference. No SUPPORTED change.

**Verifier notes**:
- Same lever family as R2 (`partial_payload`, `sfpu_scope`, `format_reconfig`, rectangle placement) — whatever R2 landed applies here for free; re-measure first. Additional R3-specific candidates: the root's own-slot copy is a NoC self-read (`noc_async_read(my_x, my_y, …)` + barrier) before the gather wait — it could be issued *before* the peers' writes are awaited (already is) but the barrier could be deferred to after `gather_sem.wait`; the writer's `receive()` pre-handshake costs one round trip per block per core (with `Rt = 1` there is one block — the handshake is the whole cost; `PRE_HANDSHAKE=false` is only legal if round-k+1 delivery can never overtake round-k consumption, which the one-round `cb_rstd` cap forbids in general but is trivially true at one block — gate the choice on `num_blocks == 1` from the host).
- `collapse_algorithm` lamp: for `ReduceInputBlockShape::of(rows, 1)` the single `ReduceTile` is the catalog's fastest choice at 1 tile — leave it.
- Guard set must include a multi-block sharded case (`(1, 1, 1024, 512)` over 16 cores, `B = 16`, 4 blocks — the geometry that found the passive-core bug) so the one-round mechanism cap is exercised.

**Done when**: measured device-ns improves on the most-impacted sharded perf geometry and each of the four is at or under its clock-scaled `achievable_ns` (or the measured ceiling is stated); golden green; no regression on the guard set including the multi-block sharded case.

## Documented omissions (not queued)

- `alignment = non_tile_aligned` — refused as an unsupported axis value (`UnsupportedAxisValue`), but **not in `feature_spec.TARGET`** and explicitly out of contract (`eval/prompts/rms_norm.txt`: "Non-tile-aligned shapes are not supported"). The translated production tests with H = 24 / W = 42 / a rank-2 `(1, 4096)` input xfail leniently through the harness hook. If the contract changes, add `alignment` to TARGET (+ non-aligned INPUTS) and queue an in-kernel edge-tile mask refinement (`/memory-layouts`).
- `HEIGHT_SHARDED` / `BLOCK_SHARDED` — not in TARGET (design notes both are placement knob-turns on the built scheme).
- `ROW_MAJOR + WIDTH_SHARDED` — INVALID in `feature_spec.py` (author-scoped), refused with `ValueError` by `validate()`.
