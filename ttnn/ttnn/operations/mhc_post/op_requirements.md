# Operation Requirements: mhc_post

## Definition
- **Formula**: `X'_j = post_j·F + Σ_i comb[i][j]·X_i` (fp32 arithmetic, one rounding to X.dtype). Optional on the same pass: `stats[:,0:24] = X'@W`, `stats[:,24] = Σ X'²` (stats mode); the next hc step's coefficients `hc` (DeepSeek-V4 Sinkhorn on the stats row) and `y = Σ_i P_i·X'_i` (bf16) (collapse mode); `y = bf16((y_b·rsqrt(Σy_b²/C + eps))·w)`, `stats[:,25] = Σ y_b²` (norm); y column-split all-to-all over the fabric into `feed_output` (feed). Full statement: `eval/golden_tests/mhc_post/feature_spec.py` docstring, `op_design.md` Overview.
- **PyTorch Reference**: `models/demos/deepseek_v3_d_p/reference/mhc/mhc_reference.py`; the golden gates in `eval/golden_tests/mhc_post/helpers.py`. Plain mode:
  ```python
  def mhc_post_ref(f, x, post, comb, n=4):
      C = f.shape[-1]
      f3 = f.double().reshape(-1, 1, C); x3 = x.double().reshape(-1, n, C)
      return (post.double().reshape(-1, n, 1) * f3
              + torch.einsum("tij,tic->tjc", comb.double().reshape(-1, n, n), x3)).reshape(x.shape)
  ```
- **Import Path**: `from ttnn.operations.mhc_post import mhc_post`
- **Function Signature**:
  ```python
  mhc_post(input_tensor, residual, post, comb, *, compute_kernel_config=None, post_col=0, comb_col=0,
           stats_weight=None, collapse_pre=None, hc_scale=None, hc_base=None, hc_norm_width=None,
           hc_norm_eps=1e-6, hc_eps=1e-6, hc_sinkhorn_iters=20, output_tensor=None, stats_output=None,
           hc_output=None, y_output=None, max_grid=None, norm_weight=None, norm_eps=1e-6,
           feed_output=None, feed_semaphores=None, feed_num_links=1, feed_cluster_axis=1)
  ```
  Returns X' (plain); `(X', stats)` (stats); `(X', stats, hc, y)` (collapse).

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

**Perf-1 contract (anchor).** `feature_spec.LOOSE_CASES` flags two PERF FOCUS cases, DeepSeek-V4.1's two hot calls on the Galaxy 8x4: T=160, C=5120 (X `(1,1,160,20480)`), `dtype = sublayer_dtype = bfloat16`, TILE, `fp32_dest_acc_en=True`, `coef_source=packed` (post_col=4, comb_col=8, the packed mixes also serve as `collapse_pre`), `math_fidelity=HiFi2`, `mode=collapse`, `norm=on`, and **(a)** `feed=on` on the 1x4 mesh under `FABRIC_2D_TORUS_XY` with `feed_num_links=2`, **(b)** `feed=off`. Both: `target_ns = 46 575` (DRAM roofline 37.3 µs at 80 %); (a) also `roofline_us = 37.3`, `links_us = 8.4`; (b) `reference_ns = 131 300` (the bring-up op). Refinements 1 and 2 together land every axis value and knob of that contract; Refinement 3 is the first perf slot.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [float32]; **sublayer_dtype**: [float32]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED shape-derived axes**: alignment ∈ {tile_aligned, h_non_aligned}; feed_split ∈ {block_aligned, not_block_aligned}
- **SUPPORTED op-specific axes**: mode = plain; coef_source = separate; norm = off; feed = off; fp32_dest_acc_en = True (the whole TARGET)
- **Regime**: R1 plain-2D (op_design.md): R row groups × Q column members over the max_grid rectangle (110 cores on BH p150), block = 1 token-tile row × `block_cols` column tiles × n streams, fp32 SFPU MAD mix chain
- **Compute config**: `compute_kernel_config` exposed (math_fidelity / approx honoured; fp32_dest_acc_en must be True), DEST full-sync
- **Golden baseline**: 92 / 92 supported cells pass; 625 xfail_expected, 2578 INVALID skipped, 0 loud (per `verifier_report.json`)

### [ ] Refinement 1 — bf16 streams + packed coefficients + the fused gather (stats and collapse modes)

**Goal**: add `ttnn.bfloat16` to `SUPPORTED["dtype"]` and `SUPPORTED["sublayer_dtype"]` (all four plain dtype pairs, incl. the mixed fp32/bf16 ones), `"packed"` to `SUPPORTED["coef_source"]`, and `"stats"`, `"collapse"` to `SUPPORTED["mode"]`. This builds regime **R2 fused-gather** from `op_design.md` — a **scheme-change** (the C axis is *dependent* for stats / hc: each core's per-row partial `X'@W` + `Σ X'²` is combined across the row group by the fixed-order 2-level tree with rotating leaders / roots), plus `project_block` (matmul_block on the resident bf16 X' × resident W slice; X' is never re-read from DRAM — regime X2 is forbidden), `finalize_row`, `combine_row`, the custom SFPU `hc_step_row` block op on `root(t)` in lane layout, and `collapse_block` (y = Σ P_i·X'_i, bf16). Includes the R2-W-streamed predicate fallback, `stats_output` / `hc_output` / `y_output` caller buffers, and the hc scalars as runtime args (re-sent every call; `custom_program_hash` covers the core layout only).

**Implementation skill**: /numeric-formats-metal

**Verifier notes**:
- Ordered first because it is the hardest structural change and everything after it (norm, feed, every perf lever) extends its combine topology; the bf16 / packed axes ride along because the fused modes are INVALID on fp32 (feature_spec contract scope), so stats / collapse cannot be tested without bf16.
- The skill covers the dtype / CB-format / `UnpackToDestFp32` half only (fp32_dest_acc_en is fixed True by TARGET — no DEST-width knob to add). No skill covers the cross-core combine: build it from `op_design.md` §Split chooser / Block schedule / Block Operation Realization and the `l1_ledger.md` R2 rows (fanin ≤ 16 slots, `cb_partial_slots` slot-indexed for determinism).
- Design risk to settle on the first bf16 run: the packer's fp32→bf16 rounding mode (op_design Risks: if it truncates, round in DEST with an SFPU RNE typecast before the pack). The X' bf16 bias gate catches it.
- The split chooser's score must see only `(T, C, X.dtype, F.dtype, has_W, |L|)` — never mode / norm / feed — so stats is bit-identical between stats and collapse calls (test_regression `test_fused_modes_agree`, `test_collapse_deterministic`).
- Build it performant, not correct-only (perf 1 optimizes this path): the combine must not serialize the grid, the reader / writer stay block-batched, the W slice is read once per core. The ragged last column pass currently runs as 1-unit blocks (a full templated remainder overflows the kernel-config buffer — see verification_report.md); keep the bf16 path's DEST / L1 `block_cols` caps derived from the one `choose_block_cols()` source.
- Gates that come alive with this refinement (currently skipped): `test_regression.py` `test_fused_modes_agree`, `test_collapse_deterministic`, `test_projection_hifi4`, `test_output_buffers`, `test_hc_scalars_are_runtime`, `test_max_grid` (incl. the 3×2 W-streamed pin), `test_depth_chain[bf16]`, `test_engram_add[plain|collapse]`, `test_collapse_pre_controls_y`.

**Done when**: every golden cell with `mode ∈ {plain, stats, collapse}`, `norm=off`, `feed=off`, any TARGET dtype pair and either coef_source passes; the newly enabled test_regression gates pass; `verify_supported` reports 0 in every loud category.

### [ ] Refinement 2 — norm + feed (y delivery: rstd return and the fabric all-to-all)

**Goal**: add `"on"` to `SUPPORTED["norm"]` and `"on"` to `SUPPORTED["feed"]`. **Norm** = regime R2-norm: `Σ y_b²` in partial column 25 through the same combine tree, rstd computed on `root(t)` inside the hc block op and unicast back to the row group's members (`cb_rstd_col`), y_b held in `cb_y_hold` with the `norm_rows_in_flight` stall shadow, then `y = (y_b·rstd)·w`. **Feed** = regime **R4 feed-DRAM-staged**: reserve the 2·MAX_FEED_LINKS port cores out of the worker list whenever the fabric is active (feed or not — bit-identity of X'/stats/hc between feed and no-feed calls on one mesh), workers publish remote-destined y tiles per row to `y_output` scratch + a port-local landed counter, the ports send fused write + atomic-inc packets to the peer `feed_output`, ready / arrival fences on the caller's two semaphores left at 0 after every call (trace-safe), own block written locally.

**Verifier notes**:
- Must follow Refinement 1 (true dependency: both act on y and the combine tree it builds). Lands the full perf-1 contract: after it, both PERF FOCUS cases run supported.
- TRANSPORT is taken from the examples, not re-derived: `fabric_all_gather` (balanced ring, far block split per direction, 7-tile chunks = `payload // 2048`, arrival inc every 8 chunks — measured 8–17 % cost for more frequent incs, AG report; ~18 µs at 2 links for 3 × 200 tiles) and `fabric_ring_exchange` (direct 2-D routing, `PacketHeaderPool` ≤ 12 per RISC → `feed_chunks_in_flight ≤ 8`). Cite them in the changelog. OVERLAP in this refinement is the design's baseline: per-row publication after `norm_block(t)` so the ports stream row t while workers compute row t+1; the ready fence only blocks ports.
- No skill covers fabric work; the design (§Feed, §Split chooser "worker set L") and the three fabric example dirs are the reference. Keep the kernel core sets as merged rectangles even with the 4 ports carved out (DISPATCH-BOUND check in the perf harness).
- R4 is the deliberate stepping stone (op_design Regimes): do not build R5 here; Refinement 3 does, against the per-chip breakdown this one makes measurable.
- Gates that come alive: `test_regression.py` `test_norm_matches_no_norm`, `test_norm_needs_collapse`, `test_norm_runtime_and_buffer`, `test_engram_add_norm`; `test_feed.py` `test_feed_matches_no_feed`, `test_feed_raw_y`, `test_feed_refusals`, `test_feed_repeated_calls`, `test_feed_fabric_2d`.

**Done when**: every golden cell of the TARGET universe passes (norm=on and feed=on cells included, on the 1x4 mesh for feed); the feed / norm regression gates pass; `verify_supported` reports 0 loud; the semaphores read 0 after repeated traced calls.

### [ ] Refinement 3 — Feed OVERLAP: hide the exchange behind the local work (perf-flagged feed case)

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES[0]` flags collapse + norm + feed, T=160, C=5120, bf16, packed, HiFi2, 1x4 mesh, `FABRIC_2D_TORUS_XY`, 2 links as the mandatory perf target (`target_us = 46.6`, `roofline_us = 37.3`, `links_us = 8.4`). First measure the **per-chip breakdown** with the harness: the feed call vs the same call without the feed on the same mesh (`LOOSE_CASES[1]`) — their fastest-chip difference is the transport the op leaves exposed per call (**OVERLAP**), and the R4 staging's DRAM cost is part of it. Then cut it, label every lever **OVERLAP** or **TRANSPORT**: the design's successor **R5 feed-L1-direct** (OVERLAP + DRAM: workers NoC-write remote y tiles straight into port L1 staging slots under credit semaphores, removing ≈ 2.4 MB = 2·(G−1)/G·T·C·2 B of DRAM staging traffic, ~13 % of the call's bytes), earlier publication / look-ahead so the last row's chain (rstd → norm → publish → send → arrival) is the only exposed part, and launch-skew tolerance. TRANSPORT parameters (links, chunk size, chunks in flight, far-block split) stay at the `fabric_all_gather` / `fabric_ring_exchange` measured settings unless the breakdown shows the links, not the overlap, bound the exposed time. Patterns: `ttnn/ttnn/operations/examples/master.md` (`fabric_all_gather`, `fabric_ring_exchange`, `double_buffer`). No SUPPORTED change.

**Verifier notes**:
- Cross-device perf is filed before the local perf (Refinement 4) per the fused-CCL rule. Expected order of magnitude: DRAM roofline 37.3 µs vs busiest link 8.4 µs, so after R5 the local part probably dominates; if the measured exposed transport is already < ~5 % of the feed call after the R4 baseline, implement R5 for its DRAM saving and close the phase.
- **Harness precondition (blocker, not the op's):** `eval/golden_tests/mhc_post/perf_measure.py` currently cannot measure any case — `prepare_mhc_post`'s `call()` re-uploads every input with `from_torch` on each call, which raises `TT_FATAL: Writes are not supported during trace capture` inside `ccl_perf.measure`'s trace capture (and the broken teardown hung the device once). Reported in verification_report.md; the eval owner must fix it (upload once in `prepare_mhc_post`, `call()` dispatches only) before this slot can produce numbers.

**Done when**: measured with `CCL_PERF_EXPERIMENTS=5 … perf_measure.py` and kept by `python -m eval.ccl_perf compare`: the fastest-chip mean of the flagged feed case improves beyond noise with the slowest chip not worse; the per-chip breakdown (exposed transport before / after) is recorded in the changelog; no DISPATCH-BOUND flag; golden suite (incl. `test_feed.py`) green; no regression on the config-spanning guard set (one representative per kernel path: plain fp32, plain bf16, stats, collapse, collapse + norm, collapse + norm + feed).

### [ ] Refinement 4 — Local pass toward the DRAM roofline (perf-flagged collapse + norm case)

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES[1]` flags collapse + norm, T=160, C=5120, bf16, packed, HiFi2 (`target_ns = 46 575`, `reference_ns = 131 300`); optimize THIS config (on the feed mesh, as the harness runs it). Phase 0's plain fp32 shapes sit at 49–62 % of DRAM peak (T=160 C=5120: 117.5 µs vs 57.6 µs roofline), so the local pass has headroom. Relevant levers from `ttnn/ttnn/operations/examples/master.md` and the design's Perf lamps: the **block-size / buffer-depth co-tune** (`block_cols`, `IN_BLOCKS` / `OUT_BLOCKS`, `in_depth` 3–4 or 2 units per transfer — `double_buffer`, `compute_block_size`), the **ragged last column pass** (runs as 1-unit blocks today: at T=160 every core does a 5- or 6-unit pass plus a 2–3-unit remainder — balance the passes or bound the template instantiations to fit the kernel-config buffer), **reader issue rate** (`split_reader`), the **NoC1 write hotspot** the Phase 0 implementer measured on the writer (`noc_placement`), the mix chain's SFPU cost (Perf lamp: 9 SFPU tile-ops per output tile), and the **combine fan-in** (`fanin` 8 / 11 / 16; R=1 vs R=5 at T=160). No SUPPORTED change.

**Verifier notes**: filed after the cross-device phase (fused-CCL rule) — move it ahead only if Refinement 3's breakdown shows the local part dominating with headroom. These are knob-turns on the block surface the planner exposed (T1/T2); several fit one phase.

**Done when**: the fastest-chip mean of the flagged collapse + norm case improves beyond noise (`eval.ccl_perf compare`) with the slowest chip not worse, moving toward `target_ns`; golden suite green; no regression across the config-spanning guard set (one representative per kernel path × dtype: plain fp32, plain bf16, stats, collapse, collapse + norm, collapse + norm + feed).

### [ ] Refinement 5 — Read the reuse-shared coefficient row once per row group and multicast it

**Type**: perf

**Goal**: the coefficient tile row (post / comb / P, one packed tile at the perf-flagged config) is **reuse-shared** along C (op_design Blocking Model) and today every one of the Q cores sharing a token row reads it from DRAM — at R=1, T=160 that is Q·Ht ≈ 530 reads of the same 5 tiles, concentrated on few DRAM banks at the same moment. Read it once per row group and broadcast it to the group with `mcast_pipe` (`SenderPipe` / `ReceiverPipe`; host `Mcast1D`) — patterns `mcast_topology`, `shared_input_reuse` in `ttnn/ttnn/operations/examples/master.md`. Target: the perf-flagged collapse + norm (and feed) case. No SUPPORTED change.

**Verifier notes**: a scheme-change (new one-injector + broadcast dataflow topology), so it stands alone (T3, one per phase). Validate by building the variant and measuring device time with the harness, not by a remove-the-coefficient-read ablation. If Refinement 4 already removed the hotspot by other means and the measured gain is within noise, close it as no-win with the measurement recorded.

**Done when**: the flagged collapse + norm case's fastest-chip mean improves beyond noise with the slowest chip not worse; golden green; no regression on the config-spanning guard set (plain fp32 / bf16, stats, collapse, collapse + norm, collapse + norm + feed).
