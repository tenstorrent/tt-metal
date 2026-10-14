# Operation Requirements: distributed_rms_norm

## Definition
- **Formula**: for each collective group of G devices along `cluster_axis`, with device p holding `x[p]` of shape `(..., M, W_local)`:
  `ss[r] = Σ_g Σ_w x[g][r,w]²`, `out[p][r,w] = x[p][r,w] · rsqrt(ss[r] / (G·W_local) + ε) · gamma[p][w]` (gamma ≡ 1 when absent)
- **PyTorch Reference**:
  ```python
  def ref(x_stacked, gamma_stacked, cluster_axis, eps=1e-6):  # stacked: (R, C, ..., M, W_local)
      out = torch.empty_like(x_stacked, dtype=torch.float32)
      R, C, Wl = x_stacked.shape[0], x_stacked.shape[1], x_stacked.shape[-1]
      for o in range(C if cluster_axis == 0 else R):
          members = [(r, o) for r in range(R)] if cluster_axis == 0 else [(o, c) for c in range(C)]
          full = torch.cat([x_stacked[m].float() for m in members], dim=-1)
          w = None if gamma_stacked is None else torch.cat([gamma_stacked[m].float().reshape(-1) for m in members])
          y = torch.nn.functional.rms_norm(full, (full.shape[-1],), weight=w, eps=eps)
          for p, m in enumerate(members):
              out[m] = y[..., p * Wl:(p + 1) * Wl]
      return out.to(x_stacked.dtype)
  ```
- **Import Path**: `from ttnn.operations.distributed_rms_norm import distributed_rms_norm`
- **Function Signature**:
  ```python
  distributed_rms_norm(input_tensor: ttnn.Tensor, *, cluster_axis: int, gamma: ttnn.Tensor | None = None,
                       epsilon: float = 1e-6, topology=Topology.Linear, num_links: int | None = None,
                       compute_kernel_config: ttnn.ComputeConfigDescriptor = None,
                       memory_config: ttnn.MemoryConfig | None = None) -> ttnn.Tensor
  ```

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.
> **Test mode (this box)**: `scripts/run_safe_pytest.sh --dev` cannot run more than one fabric test per session: later mesh re-opens time out on Ethernet cores. Run multi-test suites non-dev, and use `--dev` only for a single test during hang triage.
> **Perf measurement**: `scripts/run_safe_pytest.sh --profile tests/ttnn/unit_tests/operations/distributed_rms_norm/test_distributed_rms_norm_perf_bench.py`, then read DEVICE KERNEL DURATION (ops CSV **column 20**), second call. Report the **max over the 4 devices**: the exchange makes the group wait for the slowest.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [float32, bfloat16]
- **SUPPORTED fp32_dest_acc_en**: [True] (EXCLUSION `{float32, fp32_dest_acc_en=False}` kept for Refinement 4)
- **SUPPORTED layout**: [TILE]
- **SUPPORTED shape-derived axes**: alignment ∈ {tile_aligned, h_non_aligned}; rank ∈ {2, 3, 4}
- **SUPPORTED op-specific axes**:
  - gamma_mode ∈ {gamma, no_gamma}
  - gamma_dtype ∈ {float32, bfloat16, "none"}
  - memory_layout ∈ {INTERLEAVED}
  - cluster_axis ∈ {0, 1}
  - topology ∈ {Linear, Ring}
  - num_links ∈ {1, 2, None}
  - The verifier promoted `h_non_aligned`, `Ring` and `num_links=None` on evidence.
- **Cores**: full worker grid. Rows × W-chunks over all `grid_x·(grid_y−1)` workers; 2·L port cores in logical row 0. One `generic_op` per call.
- **Compute config**: `compute_kernel_config` exposed. `math_fidelity` and `math_approx_mode` are free; `fp32_dest_acc_en` is gated to True.
- **Golden baseline**: 931 supported_pass, 0 supported_fail, 0 xpass_drift, 1 xfail_wrong_mode. The wrong-mode cell is a harness artifact (L1-profiling `peak_total=0` on a correct refusal; see verification_report.md).
- **Device time (bf16, gamma, axis 1, L=2; max over devices)**: 640×1792 24.9 µs · 32×1792 10.9 µs · 2048×1280 44.6 µs.

### [x] Refinement 1 — Native sharded placement (HEIGHT_SHARDED + WIDTH_SHARDED)

**Goal**:
- Add `ttnn.TensorMemoryLayout.HEIGHT_SHARDED` and `ttnn.TensorMemoryLayout.WIDTH_SHARDED` to `SUPPORTED["memory_layout"]`, with **native zero-copy** I/O. `cb_x_resident` is backed on the input shard and `cb_output_tiles` on the output shard, both via `ttnn.cb_descriptor_from_sharded_tensor`. The output follows `memory_config` (the golden suite passes the input's own sharded config).
- Both placements are the design's knob-turns on the built R1 regime (op_design.md R9 / R10):
  - **HEIGHT** pins `S_w = 1` and `core_rows = shard_h/32`, so the block is the shard.
  - **WIDTH** pins `S_w` = shard cores, `core_w_tiles = shard_w/32` and `S_r = 1`. The record combine across W chunks already exists for any `S_w`.

**Verifier notes**:
- **Ordered first:** it is the hardest remaining generality item, and Refinements 2 and 4 must extend the sharded path rather than be reworked under it. No `memory_layout` skill exists; the CB-placement pattern is `ttnn.cb_descriptor_from_sharded_tensor`.
- **Zero-copy is the contract.**
  - A worker must never `TensorAccessor`-read its own shard (that would be tolerated, not implemented).
  - The writer's DRAM store disappears: compute packs straight into the output shard.
  - The block = the whole resident shard (the sharded block-size default). Do not re-chunk below it without an L1 reason.
- **Port collision:** `eval.sharding.auto_shard_config` may use logical row 0, which today holds the 2·L port cores. Either:
  - place ports on cores the shard grid leaves free; or
  - (design R9 note) move the record publish onto the worker's NCRISC and run the port BRISC kernel on a shard core whose writer is idle, since a sharded output has no DRAM write.
- **Worker set:**
  - The worker set is the shard grid, not the rectangle `(0,1)..(gx−1,y_last)`.
  - The stats-ready multicast must target only cores of this program. Use per-rectangle multicasts, or unicast incs, when the shard grid is not one rectangle.
  - Idle cores inside a multicast rectangle must still run the zero-row kernels.
- **Footprint:** `P_resident` / `GAMMA_PRESCALE` still apply. The x block is the shard (already in L1, so it costs no CB bytes); `cb_xg`, the combine CBs and the gamma tiles still count.
- **Infeasible shards:** sharded cells whose shard spec does not fit L1 are `infeasible_skipped` by the harness (uncharged).
- **Out of scope here:** `w_non_aligned` × sharded cells belong to Refinement 2, and `bfloat8_b` / `fp32_dest_acc_en=False` × sharded cells to Refinement 4.

**Done when**:
- Every golden cell with `memory_layout ∈ {HEIGHT_SHARDED, WIDTH_SHARDED}` and otherwise-SUPPORTED axes (float32/bfloat16, fp32_dest_acc_en=True, tile_aligned/h_non_aligned) is `supported_pass` (or `infeasible_skipped`).
- The verifier CLI shows 0 supported_fail / 0 xpass_drift.
- The dataflow shows no TensorAccessor read of a core's own shard.
- The interleaved cells and the ring mock stay green.

**Outcome**:
- **Landed.** HEIGHT and WIDTH (L1) are in SUPPORTED with zero-copy x / output CBs. The sharded golden slice is 1404 passed / 0 failed (36 harness `INFEASIBLE_L1` skips). Interleaved 720/720, translated, and ring mock are green.
- **WIDTH needed more than the knob-turn.** It also needed the design's R5 device-local pre-combine (a pinned S_w ≤ 86 makes flat records quadratic), stats CBs streamed in `stat_rows` batches, and HEIGHT gamma streamed in `gamma_chunk` chunks; see the l1_ledger "Refinement 1 deltas".
- **Perf finding.**
  - HEIGHT 640×1792 runs 19.8 µs, against 24.5 µs interleaved.
  - WIDTH is compute-bound per tile row (~2.6 µs/row: 640×1792 ~81 µs, 2048×2048 ~171 µs), because every W chunk runs the per-row reduce/transpose/finalize for all Ht rows.
  - Levers for a later perf pass: finalize on the combiners plus an rstd multicast, and combiners on both links.

### [x] Refinement 2 — W_local non-tile-aligned (`w_non_aligned`)

**Goal**: add `"w_non_aligned"` to `SUPPORTED["alignment"]` natively, with no host-side pad or slice, on every placement Refinement 1 landed (interleaved, HEIGHT_SHARDED, WIDTH_SHARDED):
- The tile padding of the last W tile must not enter Σx².
- The mean already divides by the true `G·W_local`.
- gamma reads only the valid `W_local mod 32` elements of its last tile.
- This includes `W_local < 32`: golden has `(32, 17)`, `(1,1,32,50)`, `(4,128,47)`.

**Implementation skill**: /memory-layouts

**Verifier notes**:
- **Two in-kernel options** (the implementer picks, by measurement if it matters):
  - **(a)** the reader zeroes the padding columns of the last W tile of each tile row in its own CB page before `cb_push_back`. Then sumsq, normalize and the record count are unchanged, and padding outputs become 0.
  - **(b)** the design's plan (op_design.md risk table): reduce the last tile with a partial scaler (`prepare_partial_reduce_scalers<…, W_local mod 32>`) and publish one extra record, so the row group has S_w+1 sources.
  - Option (a) keeps the exchange layout untouched. The `.claude/skills/partial-scaler-reduce` skill documents (b)'s helper contract.
- **Gamma:** the stick read is a single contiguous read of `cw·32` elements. With non-aligned W, the last segment must not read past the stick's valid bytes into undefined memory (or its garbage must provably never reach a valid output element). Keep the 64 B DRAM-source alignment rule from the design.
- **Ragged chunks:** `_choose_split` and the per-chunk-group CB sizing already handle a ragged last chunk. Only the in-tile column mask is new.
- **Ordered after Refinement 1:** the mask must work on the sharded path too, where the WIDTH shard's last shard holds the partial tile.

**Done when**:
- Every `alignment=w_non_aligned` golden cell with otherwise-SUPPORTED axes passes, on all placements in SUPPORTED.
- 0 supported_fail / 0 xpass_drift.
- Prior phases stay green.

### [x] Refinement 3 — Speed up the perf-flagged profile (overlap the exchange with DRAM)

**Type**: perf

**Goal**:
- **Target:** `feature_spec.LOOSE_CASES` flags `(1, 1, 640, 1792)` as the **PERF FOCUS**, the mandatory target, at its full contract: bf16 x, bf16 gamma, `fp32_dest_acc_en=True`, HiFi4, TILE, DRAM interleaved, `cluster_axis=1`, Linear, `num_links=2`. All of it is in SUPPORTED since Phase 0.
- **Numbers:** measured 21.7–24.9 µs device kernel (max **24.9 µs**) against a device-side budget of ~15.3 µs (`target_us` 24.1 minus the 8.8 µs program floor). The DRAM floor is ~9 µs.
- **Why:** the single-round R1 schedule serializes all x reads → exchange (fence, Fabric, relay, stats-ready) → normalize + writes. DRAM idles for the whole exchange (the design's "Overlap" perf lamp).
- **Lever:** this phase's lever is the design's deferred **R2 multi-round regime**, `block_rows < core_rows`, so that round b's exchange runs under round b+1's reads. The design's example is `S_r = Ht/2` with doubled `S_w` and `block_rows = 1`. It is a ⭐⭐⭐ T3 overlap restructure (see `ttnn/ttnn/operations/examples/master.md`), so it is the whole phase.
- **Already in place:** the worker kernels already loop `num_blocks` and wait for stats-ready ≥ `2L·(round+1)`.
- **Still missing:**
  - the port kernel needs per-round local-arrival thresholds and a per-round stats-ready multicast;
  - the host must drop its `num_blocks == 1` assert.
- **Also measure:** whether the gamma prescale still sits in the stall, or whether the fused-gamma normalize (the Phase 0 fallback path) wins once the stall shrinks.
- No SUPPORTED change.

**Done when**:
- Measured device-ns (max over the 4 devices) improves on 640×1792 in the flagged config.
- The golden suite stays green.
- No regression across the guard set (one representative per distinct kernel path × layout × placement): 32×1792 decode, 2048×1280, a fp32 + no-gamma interleaved shape, the fused-gamma fallback shape (2, 2048, 4096), one HEIGHT_SHARDED and one WIDTH_SHARDED shape, and one ring-mock Ring shape.
- The multi-round regime also lifts the remaining `P_resident` ceiling. Record in `l1_ledger.md` the R2 row's new status and any CB whose capacity changes with `block_rows`.

**Outcome**:
- **Built:** R2 multi-round with a lookahead-1 schedule on all three worker kernels.
  - The port loops rounds, with rotating per-round local-arrival counters and one stats-ready counter per port.
  - The split doubles `S_w`: 640×1792 → `S_r` 10 × `S_w` 8, 2 rounds of one tile row, 14 tiles per worker as before.
  - An L1-fit round search lifts `P_resident` on the prescale path.
- **Flagged config, 640×1792** (median over 20 calls of the max over 4 devices): **25.1 → 21.9–22.5 µs**. The min device went 21.7 → 18.8 µs.
- **Guard set (median-max):**
  - 32×1792: 10.8 → 10.5 (one round, same path).
  - 2048×1280: 45.2 → 37.3–40.7.
  - fp32 no-gamma 640×1792: 38.8 → 34.3–35.4.
  - (2, 2048, 4096): 346.5 → 196–204. It now takes the prescale path over 4 rounds instead of the fused fallback.
  - HEIGHT 640×1792: 19.8 = 19.8. WIDTH 640×1792: 70.9 → 70.9–71.7 (noise; sharded stays one round).
  - Ring mock 640×1792 L2: 54.6/55.3 → 51.7/54.3 (single calls).
- **Gamma prescale vs fused** (also measured): fused is ~2× slower at 640×1792 (52 vs 23 µs) because the reader replicates gamma rows with RISC stores, so prescale stays the default (`GAMMA_PRESCALE` knob).
- **What binds now:** cross-device launch skew. In 20 calls the max device sits 2–5 µs above the min device, with occasional 10+ µs spikes. Each device's own work is ~18.8 µs against the ~9 µs DRAM floor plus the 8.8 µs program floor.
- **Next lever:** Refinement 5's per-link stats-ready and port placement. Also: more rounds at 640×1792 would need `S_r` < 10, which raises the tiles per worker. That trade is not taken; `_rounds_split` rejects any split that raises the per-worker tile count.

### [x] Refinement 4 — Numerical configurability (bfloat8_b + fp32_dest_acc_en=False)

**Goal**:
- Add `ttnn.bfloat8_b` to `SUPPORTED["dtype"]` and `False` to `SUPPORTED["fp32_dest_acc_en"]`. `compute_kernel_config` is already exposed and `math_fidelity` is not gated.
- Derive intermediate-CB formats from the config:
  - With `fp32_dest_acc_en=False`, `cb_xg` becomes Float16_b.
  - The stats CBs (`cb_sq_acc`, `cb_partial`, `cb_stats_combine`, `cb_rstd`) and the wire records **stay Float32**. The spec requires the exchanged partials and their sum in fp32; that is the recorded mechanism reason for the page-format audit.
  - Apply `UnpackToDestFp32` tagging where it applies.
- Cells that fail out of the box (typically `bfloat8_b + w_non_aligned`) go to `EXCLUSIONS`, not a new refinement.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**:
- **Ordered last of the generality work** (cheapest tier). It must cover every placement and alignment Refinements 1–2 landed.
- Keep the `{float32, fp32_dest_acc_en=False}` EXCLUSION (precision convention).
- **bf8b inputs** pass through the reader as tiles. gamma stays fp32/bf16 (TARGET has no bf8b gamma).
- **fp32 precision lever (in scope, optional):** the precision baseline shows fp32 outputs with a systematic −0.1 % bias (median got/true 0.9988; ~15 k fp32 ULP mean). The cause is FPU operands unpacked as tf32 by truncation on the x·gamma and ·rstd multiplies. It passes the gates today. If this refinement touches the float32 path, unpack-to-DEST fp32 or SFPU multiplies on that path are the lever; re-measure with `test_distributed_rms_norm_precision_baseline.py`.

**Done when**:
- Every golden cell with `dtype=bfloat8_b` and/or `fp32_dest_acc_en=False` is `supported_pass` or a documented EXCLUSION, on all placements and alignments in SUPPORTED. The golden tolerances are bf8b 0.995/0.04–0.05, bf16+False 0.999/0.015.
- 0 supported_fail / 0 xpass_drift.
- The precision baseline is re-run and recorded in the changelog.
- **Outcome**: landed bf8b + fp32_dest_acc_en=False with no EXCLUSIONS beyond {float32, False}; 1221 golden target cells run (incl. bf8b x w_non_aligned / HEIGHT / WIDTH), all pass. UnpackToDestFp32 on the copy_tile-only stats CBs is implemented but parked off (measured worse end-to-end: fp32 mean ULP 15k -> 19k), see changelog.

### [x] Refinement 5 — Speed up the perf-flagged profile (synchronization and block knobs)

**Type**: perf

**Goal**:
- **Target:** the same mandatory target, `(1, 1, 640, 1792)` at the flagged config (see Refinement 3), measured after Refinement 3.
- **Levers:** this phase co-tunes the cheaper ⭐/⭐⭐ levers from `ttnn/ttnn/operations/examples/master.md` and the design's remaining perf lamps:
  - **Stats-ready granularity:** per-link multicast rectangles, so a worker waits only for its link's 2 ports instead of all 2L.
  - **Port placement:** ports under each link's Ethernet column, derived on the host from the SoC descriptor and `get_forwarding_link_indices`; no probe dispatch.
  - **Block knobs:** `MIN_CORE_W_TILES` (S_w), `out_batch_tiles` and `OUT_DEPTH`, co-tuned under the master.md chunk floor (whole tiles minimum; coarser amortizes).
- **Device skew:** check whether the 1–3 µs device-to-device skew is placement-driven.
- No SUPPORTED change.
- If the decode shape (32×1792, 10.9 µs vs a ~4 µs device budget) moves with the same levers, record it, but the gate is the flagged shape.

**Done when**:
- Measured device-ns (max over devices) improves on 640×1792 in the flagged config relative to the post-Refinement-3 number.
- The golden suite stays green.
- No regression across the Refinement 3 guard set.
- **Outcome**: 640×1792 flagged config **22.98 → 19.32 µs** (median over 60 calls of the max over 4 devices, same-session A/B; min device 19.15 → 16.16 µs). Guard set: 32×1792 10.40 → 8.06, 2048×1280 41.3 → 39.0, fp32 no-gamma 34–35 → 30.1, (2,2048,4096) ~200 → 192.7, HEIGHT 19.8 → 19.1, WIDTH 71 → 48.8. Levers kept: per-link stats-ready multicast rectangles (contiguous link runs), port kernels on NoC0, port header setup off the fence path (route once, copy after the ready packet, only as many headers as packets), early gamma read, per-port gather phases, finalize SFPU on row 0 only, common runtime args. Measured and left at default: `MIN_CORE_W_TILES` (8; 4 is +5 µs, 16 is the same split at 640×1792 and +0.9 µs at decode), `out_batch_tiles` (whole row; 1 tile no better), `OUT_DEPTH` (2; 7 no better), port column offset (0/4/7 within noise: the skew is not placement-driven). **What binds now**: (a) cross-device launch skew from host dispatch. The bench loop is host-bound (op-to-op ~14 ms), and the mesh enqueue writes the 4 per-device programs one after another from an unordered map. Calls are bimodal: a ~2.7 µs or a ~7 µs late start of the p=0 devices. The early device's port waits in the ready fence for it. (b) On the late device, the DRAM write tail: compute ends ~10.7 µs, but the writes of the far grid corner (NoC1 congestion) run to ~16.4 µs, ~295 GB/s aggregate. Next: split the post-exchange writes over both NoCs (a second output CB drained by the idle reader on NoC0, with the port multicast kept off that NoC's write load), or push the round's records from the ports by multicast instead of 80 pulls. Not done here: the first is a dataflow restructure across all three worker kernels plus a CB, and the second a new landing buffer and handshake.
