# Operation Requirements: rms_norm

## Definition
- **Formula**: `out[..., h, w] = x[..., h, w] * rsqrt( (1/W) * Σ_{w'<W} x[..., h, w']² + eps ) * gamma[w]` (gamma term omitted when `gamma is None`; W = logical last dim — padding never enters Σ or 1/W)
- **PyTorch Reference**:
  ```python
  def rms_norm_ref(x, gamma=None, epsilon=1e-6):
      xf = x.float()
      y = xf / torch.sqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + epsilon)
      return y * gamma.float().reshape(-1) if gamma is not None else y
  ```
- **Import Path**: `from ttnn.operations.rms_norm import rms_norm`
- **Function Signature**: `rms_norm(input_tensor: ttnn.Tensor, *, gamma: Optional[ttnn.Tensor] = None, epsilon: float = 1e-6, compute_kernel_config: Optional[ttnn.ComputeConfigDescriptor] = None, memory_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor` — one native `ttnn.generic_op` dispatch per call.

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [float32, bfloat16]
- **SUPPORTED fp32_dest_acc_en**: [True] (EXCLUSION `{dtype: float32, fp32_dest_acc_en: False}` — precision convention, permanent)
- **SUPPORTED layout**: [TILE, ROW_MAJOR] (both native: tilize/untilize inside the program)
- **SUPPORTED shape-derived axes**: alignment ∈ {tile_aligned, w_non_aligned, h_non_aligned}; rank ∈ {2, 3, 4}
- **SUPPORTED op-specific axes**: gamma_mode ∈ {gamma, no_gamma}; gamma_dtype ∈ {float32, bfloat16, "none"}; gamma_layout ∈ {TILE, ROW_MAJOR, "none"}
- **SUPPORTED memory_layout**: [INTERLEAVED] (DRAM or L1 `BufferType`)
- **Cores**: multi-core — row-groups over the whole Tensix grid (R1); W split across k ≤ 8 Tensix cores of one grid row with a symmetric unicast all-gather of per-row Σx² partials (R2) when tile-rows ≤ grid/8 or W does not fit one core's L1
- **Compute config**: caller's `compute_kernel_config` passed as-is (default HiFi4 + fp32_dest_acc_en + exact SFPU); `math_fidelity` / `math_approx_mode` honored
- **Golden baseline**: 737 supported_pass / 0 supported_fail / 0 xpass_drift / 0 xfail_wrong_mode; 6180 xfail_expected (all refinement candidates below), 33900 invalid_skipped (per verifier CLI, BH p150b 13×10)

### [x] Refinement 1 — Numerical configurability: bfloat8_b + fp32_dest_acc_en=False

**Goal**: add `ttnn.bfloat8_b` to `SUPPORTED["dtype"]` and to `SUPPORTED["gamma_dtype"]`, and `False` to `SUPPORTED["fp32_dest_acc_en"]` (for bfloat16 and bfloat8_b inputs; `{float32, False}` stays an EXCLUSION). Intermediate CB formats follow the DEST width. This unlocks the 822 cells blocked only by `fp32_dest_acc_en=False`, the 240 blocked only by bfloat8_b, and every LLM perf profile in `feature_spec.LOOSE_CASES` (`_perf_case`: bf16, `fp32_dest_acc_en=False`, HiFi2, TILE input, TILE gamma, soft PCC 0.9995) and resilience case at INTERLEAVED — the cells Refinement 3 measures and optimizes.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**:
- Lands the full config contract of the Refinement 3 perf targets (bf16 + `fp32_dest_acc_en=False` + HiFi2 + TILE + TILE gamma, interleaved). Do not stop at the dtype list — the `False` DEST path is the one the perf phases optimize, so build it at the same performance bar as Phase 0 (grid-filling, batched reader/writer), not as a correct-only stub.
- L1 ledger (check 3, folded here): `cb_sumsq_tiles`, `cb_partial_sumsq`, `cb_gathered_sumsq`, `cb_inv_rms` are Float32 pages because Phase 0 DEST is fp32. Under `fp32_dest_acc_en=False` 16-bit DEST has already rounded the value before `pack_tile`, so Float32 pages would carry no extra precision and double the packer/unpacker bytes. Derive their format (and `cb_plan()`'s `T32` page size → a DEST-width tile size) from the flag. Also derive the gathered-partial NoC payload from it (the writer reads `get_tile_size(cb_partial_sumsq)` — it follows automatically).
- bfloat8_b TILE gamma: the reader extracts gamma row 0 with two 64 B face-row reads per tile, which assumes a plain per-face datum layout. A bfp8 tile stores a shared-exponent section ahead of the mantissas, so that read is wrong for bf8b gamma. Make the gamma load format-aware (e.g. read whole gamma tiles and expand row 0 with format-aware offsets), keeping the "every row of tile j = gamma[32j..32j+31]" invariant pass 2 relies on.
- bfloat8_b input is TILE-only and tile-aligned only (feature_spec INVALID: bf8b×ROW_MAJOR, bf8b×{w,h}_non_aligned), so the reader's W-tail zero fill never runs on bf8b data. Mirror those two INVALID entries as op EXCLUSIONS too: translated tests do not consult INVALID (the CLI already reports 2 `invalid_unexpected` bf8b×w_non_aligned cells), and a native refusal keeps them xfail instead of running an unsupported path.
- Precision: under 16-bit DEST, Σx² accumulates in bf16 DEST across `core_w_tiles` tiles. If the soft 0.9995 PCC gate of the perf profiles or the golden bf16 tolerance fails on wide W, those cells stay failing (`numerical-precision`) and become a `Refinement 1b` follow-up (e.g. keep only the pass-1 accumulation + `cb_sumsq_tiles` in fp32 via the skill's intermediate-CB / `UnpackToDestFp32` levers). Never move them to EXCLUSIONS.

**Done when**: `bfloat8_b` ∈ `SUPPORTED["dtype"]` and `SUPPORTED["gamma_dtype"]`, `False` ∈ `SUPPORTED["fp32_dest_acc_en"]`; every golden cell whose only gap was one of these values passes (INTERLEAVED); no supported_fail / xpass_drift / xfail_wrong_mode; Phase 0 acceptance + debug + precision-baseline tests still green.

### [x] Refinement 2 — WIDTH_SHARDED + BLOCK_SHARDED: shard-resident blocks + a cross-core combine that scales with the shard grid

**Goal**: add `ttnn.TensorMemoryLayout.WIDTH_SHARDED` and `ttnn.TensorMemoryLayout.BLOCK_SHARDED` to `SUPPORTED["memory_layout"]` (input and output sharded with the same spec; both `Layout`s, all alignments) natively: each Tensix core consumes its resident shard in place and the per-row Σx² partials are combined across every Tensix core holding a slice of the same rows. Unlocks the 586 + 586 cells blocked only by these values (1172 more once Refinement 1 lands) plus the sharded perf profiles and resilience cases.

**Verifier notes**:
- Scheme-change, stands alone (no skill covers cross-core sharded reductions). This is the design's deferred **R3 wide_combine** row made real, plus shard placement:
  - **Member → slice map from the shard spec, not "k consecutive Tensix cores in one grid row".** `eval.sharding.auto_shard_config` WIDTH-shards W tiles over the whole grid row-major (a 4096-wide row → up to 128 one-tile slices spanning ~10 grid rows); BLOCK uses up to the full grid width (13 on BH p150b). The pinned perf geometries need k = 9 (WIDTH 9×1), 32 (WIDTH 8×4), 28 (WIDTH 7×4), and 9 / 11 shard columns (BLOCK 9×10, 11×10). Build members and their NoC coordinates from the shard grid (orientation-aware) — the RT-arg peer list is already per member, so groups spanning grid rows are a host change.
  - **Combine topology by k** (catalog `tensix_all_reduce`, per-row-vector rows): k ≤ 8 keeps the built symmetric unicast all-gather; beyond that a per-row owner gathers the partials and multicasts inv_rms (or the combined Σx²) back — row partials (128 B per member per tile-row) from K ≥ 32, faces 0/2 or row at K = 12–16, multicast return from K ≥ 12. The symmetric all-gather does not scale: `cb_gathered_sumsq` is `2·k·block_row_tiles` fp32 pages (k = 128 → 1 MB per core per tile-row of block).
  - **Zero-copy placement**: back `cb_x` on the input shard and `cb_out` on the output shard with `ttnn.cb_descriptor_from_sharded_tensor`. Re-reading a core's own shard through a `TensorAccessor` does not count as implementing the value — the refinement goes back. Block = the whole shard (sub-blocked only under L1 pressure, justified in `l1_ledger.md`). The W-tail / shard-padding zero fill must not mutate the caller's shard: copy the last W tile per tile-row into a 1-tile scratch and zero it there (design: Placement axis table).
  - **Ragged shards** (`Split.PAD`): the last shard column/row is part padding; the padding columns are beyond logical W and must be zeroed before Σx², exactly like the W tail.
  - **ROW_MAJOR shards**: stick shards are resident in L1; tilize locally from the resident sticks (catalog `resident_shard_locality`: balanced ≤ 32-tile chunks, never gather sticks across Tensix cores).
  - **L1**: shards occupy L1 top-down; `cb_l1_budget()` (verifier fix) already measures the free region after the input and output are allocated — keep the CB solve on it.
  - Gamma stays INTERLEAVED; each member reads only its own W slice (as in R2 today).
  - `validate()` currently rejects any non-INTERLEAVED output `memory_config`; output follows the input's shard spec.
- Ordered before HEIGHT_SHARDED (Refinement 4) deliberately: it builds the shard-resident CB plumbing and the scalable combine; HEIGHT is then the k = 1 degenerate case of this structure.
- Golden `infeasible_skipped` cells (shard geometry vs device L1, harness-side) are uncharged and need nothing.

**Done when**: `WIDTH_SHARDED` and `BLOCK_SHARDED` ∈ `SUPPORTED["memory_layout"]`; every golden cell (cartesian, pinned-geometry perf profiles, resilience and padding-poison cases) whose only gap was one of these values passes, for TILE and ROW_MAJOR; the dataflow reads each core's own shard zero-copy (no accessor re-read); no loud verifier category; prior phases green.

**Landed**: both values in SUPPORTED (TILE + ROW_MAJOR, all alignments). Golden WIDTH/BLOCK slice: 642 pass / 126 xfail (exclusions) / 0 fail; all 192 sharded loose cells pass (+3 harness `INFEASIBLE_L1`). TILE cb_x / cb_out are backed on the shards; RM stick shards are copied locally into the tilize staging (16 B-granular shard widths are not a tile-page layout). The combine is the R3 owner topology for every sharded k > 1 (measured 2.2× over the all-gather on tall BLOCK shards). Remaining sharded headroom vs `achievable_ns` (tall BLOCK 32.3 vs 18.9 µs, decode WIDTH 5.7–9.6 vs 4.1–5.5 µs) is per-tile-row SFPU work (full-tile rsqrt finalize + within-tile collapse) — see changelog; next levers: finalize SFPU scope, row-partial + multicast return.

### [x] Refinement 3 — Speed up the large interleaved cells: fill the grid at mid tile-row counts, overlap the one-block prefill

**Type**: perf

**Goal**: no loose case carries an `attention:` perf flag, so this phase targets the large supported INTERLEAVED cells with the largest measured headroom, at their own config (bf16, `fp32_dest_acc_en=False`, HiFi2, TILE, TILE gamma — landed by Refinement 1; re-measure there, the numbers below are Phase 0 at fp32 DEST / HiFi4). Two classes, both knob-turns on the planner's block surface:
- **(a) Mid tile-row counts under-fill the grid.** The occupancy predicate `8·tensor_row_tiles ≤ G` leaves `G/8 < tile-rows < G` on k = 1: (1,1,640,6144) and (1,1,640,7168) run on 20 of 130 Tensix cores at 116 / 135 µs (DRAM ceiling ≈ 37 / 44 µs; `achievable_ns` 102 / 119 µs); (1,1,1024,1024) on 32 cores at 25.5 µs (ceiling ≈ 10 µs). Turn `k` (`w_split_cores`) so row-groups × k fills the grid — the R2 combine is already built. Catalog `width_split` (its R_t ≤ ~1/8-grid band was measured on one reduce+combine shape; re-measure the boundary here).
- **(b) Full-grid prefill runs one block per core with no overlap.** `block_row_tiles = core_row_tiles`, `x_block_depth = 1` ⇒ read, compute and write serialize: (1,1,8192,7168) 789 µs vs ≈ 559 µs DRAM ceiling; (1,1,7168,1024) 96.6 µs vs ≈ 70 µs; (1,1,8192,2304) 246 µs vs ≈ 180 µs. Co-tune `block_row_tiles` × `x_block_depth` (design perf lamps "Overlap (one-block assignment)" / "Overlap at the fit boundary") and the writer's unit size — catalog `double_buffer` (BH still +7–9% at the full grid), `compute_block_size`.
Use the relevant entries of `ttnn/ttnn/operations/examples/master.md`. No SUPPORTED change.

**Done when**: measured device-ns improves on the most-impacted shape of each class at its own config, the soft PCC 0.9995 gate of the perf profiles still holds, the golden suite is green, and no regression across the config-spanning guard set (one representative per distinct kernel path × layout × placement).

**Outcome**: (BH p150b, perf config, device ns) (a) per-core-cost W split `_choose_k`: (640,6144) 99 683 → 45 884, (640,7168) 116 090 → 52 039, (1024,1024) 22 710 → 15 226. (b) ablation showed the binding stage was the reader's serial per-core gamma load (131 µs of 771 µs at (8192,7168)), not overlap. Gamma moved to the writer and multicast once per W slice (`GAMMA_MCAST`): (8192,7168) 771 298 → 628 756, (7168,1024) 90 816 → 77 611, (8192,2304) 235 847 → 194 893. The overlap knob `OVERLAP_BLOCKS` is built and parked at 1 (2 measured 630 → 673 µs). Writer unit / depth measured flat. Every guard shape (interleaved TILE/RM, sharded WIDTH/BLOCK) is faster. Now: k = 1 prefill is DRAM-write-bound: compute stubbed = same time; write-only 438 µs (268 GB/s) vs read-only 313 µs at (8192,7168). Next: split output writes across both NoCs (needs a second output CB and a compute pack split, a scheme change). Mid tile-row cells serialize read → pass 1 → write inside each Tensix core (45.9 vs ≈ 37 µs ceiling). Next: stream pass 1 per W chunk. Neither was attempted here: both are restructurings beyond this knob-turn heading.

### [x] Refinement 4 — HEIGHT_SHARDED: consume the resident row shard in place

**Goal**: add `ttnn.TensorMemoryLayout.HEIGHT_SHARDED` to `SUPPORTED["memory_layout"]` (input and output sharded with the same spec; TILE and ROW_MAJOR; all alignments), natively on the shard. Unlocks the 561 cells blocked only by this value (+652 with Refinement 1) and the HEIGHT resilience / padding-poison cases.

**Verifier notes**:
- Local sharding: knob-turn on R1 (design Placement table). Core assignment = the shard spec (row-major 2-D fill from `auto_shard_config`, ragged last shard), block = the whole shard, k = 1, no cross-core traffic. Reuse Refinement 2's plumbing: `cb_x` / `cb_out` backed on the shards via `ttnn.cb_descriptor_from_sharded_tensor`, W-tail zero fill into a 1-tile scratch (never mutate the caller's shard), RM stick shards tilized locally.
- An accessor re-read of a core's own shard is not "HEIGHT_SHARDED implemented" — the refinement goes back.
- Comes after Refinement 2 by the hardest-first rule; it is not a true dependency, but building it second avoids reworking the shard plumbing under the cross-core structure.

**Done when**: `HEIGHT_SHARDED` ∈ `SUPPORTED["memory_layout"]`; every golden cell whose only gap was this value passes (TILE and ROW_MAJOR); shards consumed zero-copy; no loud verifier category; prior phases green.

### [x] Refinement 5 — Speed up the few-row wide-W interleaved cells: widen the W split past 8 Tensix cores

**Type**: perf

**Goal**: the decode / few-tile-row INTERLEAVED cells cap the W split at `K_CAP = 8`, so one tile-row of a wide row runs on 8 of 130 Tensix cores: (1,1,32,7168) 21.8 µs vs `achievable_ns` 14.9 µs and a ≈ 2.2 µs DRAM ceiling; (1,1,32,32768) 73.5 µs; (1,1,32,16384) 40.8 µs (Phase 0 numbers, re-measure at the perf config). With Refinement 2's scalable combine built, raise k for interleaved few-row shapes (knob-turn: `K_CAP`, the `k_occ` formula and its ≥ 4-tiles-per-slice floor) and tune the combine on these shapes — catalog `tensix_all_reduce` (per-row vector: never ship the whole 4 KB partial; faces 0/2 or row partial by K; multicast return from K ≥ 12; flush instead of a write barrier before the semaphore increment) and `width_split`. Design perf lamps: "Partial payload", "Signal fence", "Finalize SFPU scope" (`sfpu_tile_scope`). Use the relevant entries of `ttnn/ttnn/operations/examples/master.md`. No SUPPORTED change.

**Verifier notes**: depends on Refinement 2 (the k > 8 combine topology) — a true dependency. Large cells first (perf_rules: cells > 32 tiles; rank by size): W ≥ 16384 single tile-row, the 2–5 tile-row wide rows of the resilience set (e.g. (1,1,160,11008), (1,1,96,6144)), then the decode profiles (1,1,32,{5120,7168}). Re-check the k_occ boundary against Refinement 3's retuned predicate so the two phases do not fight over the same shapes.

**Done when**: measured device-ns improves on the most-impacted few-row wide-W shape(s) at their own config, golden suite green, and no regression across the config-spanning guard set (one representative per distinct kernel path × layout × placement).

**Outcome**: (BH p150b, perf config, device ns) interleaved k now goes past 8 (owner combine; groups span grid rows for k > grid_x; `_occ_cost` prices the owner at `OWNER_MEMBER_COST` = 0.4 tile per member, fitted on a forced-k sweep). TILE (32,32768) 50 598 → 20 876 (k 49), (32,16384) 27 710 → 14 585, (32,7168) 14 497 → 9 962, (32,5120) 11 531 → 9 083, (96,6144) 16 491 → 13 308, (160,11008) 28 982 → 24 270; RM 1.11–2.30× on the same shapes. Two combine levers were added. `GAMMA_AFTER_SEND` issues the gamma reads at kernel start but finishes them after the exchange; it applies to per-core gamma with blocks of ≤ 8 tile-rows, and an ablation showed the combine had been waiting on the gamma load. `SIGNAL_AFTER_FLUSH` (flush before the semaphore increment) gains 2–4% on every k > 1 shape. Guards (interleaved k ≤ 8, sharded WIDTH / BLOCK) are flat or faster, except RM (32,1024), a ≤ 32-tile cell, which is +2–3%. Now the owner's per-member combine (k tile-adds + 2k face writes, ~0.15 µs per member) is about a third of the time at the optimum, and gamma still costs 1.6–3.4 µs. Next: multicast inv_rms return, which needs rectangular groups; row-vector partials; a Row-broadcast gamma multiply with no expand. Not attempted: each is a topology / compute rewrite beyond this knob-turn.
