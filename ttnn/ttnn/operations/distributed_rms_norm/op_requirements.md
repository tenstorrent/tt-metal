# Operation Requirements: distributed_rms_norm

## Definition
- **Formula**: for each collective group of `G` devices along `cluster_axis` (group position `p`),
  `ss[r] = Σ_{g∈group} Σ_w x[g][r,w]²`, `out[p][r,w] = x[p][r,w] · rsqrt(ss[r] / W + ε) · gamma[p][w]`,
  with `W = G · W_local` (true width, never padded); the gamma factor is dropped when gamma is absent.
- **PyTorch Reference**:
  ```python
  def ref(x_stacked, gamma_stacked, cluster_axis, eps):          # x_stacked (R, C, ..., M, W_local)
      out = torch.empty_like(x_stacked, dtype=torch.float32)
      R, C = x_stacked.shape[:2]
      for other in range(C if cluster_axis == 0 else R):
          members = [(r, other) for r in range(R)] if cluster_axis == 0 else [(other, c) for c in range(C)]
          full = torch.cat([x_stacked[m].float() for m in members], dim=-1)
          w = None if gamma_stacked is None else torch.cat([gamma_stacked[m].float().reshape(-1) for m in members])
          y = torch.nn.functional.rms_norm(full, (full.shape[-1],), weight=w, eps=eps)
          for p, m in enumerate(members):
              out[m] = y[..., p * x_stacked.shape[-1]:(p + 1) * x_stacked.shape[-1]]
      return out
  ```
- **Import Path**: `from ttnn.operations.distributed_rms_norm import distributed_rms_norm`
- **Function Signature**:
  `distributed_rms_norm(input_tensor: ttnn.Tensor, *, cluster_axis: int, gamma: ttnn.Tensor | None = None,
  epsilon: float = 1e-6, topology: ttnn.Topology = ttnn.Topology.Linear, num_links: int | None = None,
  compute_kernel_config: ttnn.ComputeConfigDescriptor = None, memory_config: ttnn.MemoryConfig | None = None)
  -> ttnn.Tensor`

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.
> **Golden-run note (harness)**: run the golden suite with `EVAL_CAPTURE_L1=0` until the harness L1 capture is fixed. With the capture on, every cell fails with `L1ProfilingError: peak_total=0`, because graph capture records no CB allocation for any op on the fabric mesh. Native `ttnn.add` reproduces it, so it is not this op (see `verification_report.md`).
> **Device note**: under `--dev`, run one mesh shape per pytest invocation. A fabric mesh close under the watcher leaves ERISCs stuck, and reopening the mesh at another shape in the same session then errors.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [float32, bfloat16]
- **SUPPORTED fp32_dest_acc_en**: [True] (`{float32, False}` stays in EXCLUSIONS)
- **SUPPORTED layout**: [TILE]
- **SUPPORTED shape-derived axes**: alignment ∈ {tile_aligned, h_non_aligned}; rank ∈ {2, 3, 4} — h_non_aligned was promoted by the verifier after the probe recorded in verification_report.md
- **SUPPORTED op-specific axes**: gamma_mode ∈ {gamma, no_gamma}; gamma_dtype ∈ {float32, bfloat16, "none"}; gamma_layout ∈ {TILE, "none"}; memory_layout ∈ {INTERLEAVED}; cluster_axis ∈ {0, 1}; topology ∈ {Linear, Ring}; num_links ∈ {1, 2, None}
- **Cores**: multi-core per chip — `num_row_groups × num_w_chunks` workers from the host solve (100 workers at 640×1792), plus 2 port cores per link. One fused `generic_op` (MeshProgramDescriptor) per call.
- **Compute config**: `default_compute_kernel_config()` = HiFi4, fp32_dest_acc_en=True, no approx. math_fidelity and math_approx_mode are passed through and not gated.
- **Golden baseline**: 1283 supported_pass, 8 supported_fail (the refinement-1 OOM cells), 11790 xfail_expected, 33696 invalid_skipped (verifier CLI, `EVAL_CAPTURE_L1=0`). Runner: 1320 passed / 8 failed / 0 hangs

### [x] Refinement 1 — Multi-block per-core schedule (L1 fit for tall × wide shapes)

**Goal**: move the supported cells that fail with the op's own L1 refusal (`RuntimeError: working set exceeds L1 for every block`) to passing. These are the 8 `test_translated.py::test_distributed_norm_allclose[*-8192-2048-2]` cells: per device `(2, 1, 2048, 4096)` bf16 with bf16 gamma, `tensor_h_tiles = 128 × tensor_w_tiles = 128`. They are in SUPPORTED (bf16, tile_aligned, INTERLEAVED). The plan solve only accepts `num_row_groups ≤ num_workers`, so each core can own exactly one block. When rows outnumber the workers, `block_h` grows until the resident block (x + fp32 `cb_xg` + out) exceeds L1, and no feasible pair is left. No SUPPORTED axis is added.

**Implementation skill**: /memory-budget-metal

**Verifier notes**:
- This is the collapsed **exchange_round / blocks-per-core** knob. `op_design.md` writes the schedule as `for block_idx < num_blocks_this_core`, with `rows_per_round` as a host knob. The host exposes `ROWS_PER_ROUND`, but it raises `NotImplementedError` for anything other than `block_h`, and none of the five kernels has a block loop. Re-expose it as a real parameter:
  - the reader, compute, writer and port-sender kernels loop over `num_blocks_this_core` blocks of `block_h` rows;
  - each block gets its own exchange round (board slot region per round, or per-round arrival / board-ready / inv-ready counts);
  - the solve accepts `num_row_groups > num_workers` by assigning several row-blocks per core.
- Keep x resident per block, not per core. That keeps the per-core footprint bounded by one block, so the solve's `fits_l1` predicate holds for any `tensor_h_tiles`.
- The fp32 `cb_xg` stall-shadow buffer is 2× the bf16 x block. It is the largest term on these shapes (`l1_ledger.md`). If a predicate-guarded fallback that skips the stall-shadow (normalize as one `x · inv · gamma` pass, no `cb_xg`) is the cheaper fit for some shapes, record it as a regime row in the ledger. Do not silently drop the reorder for shapes that fit today.
- Mandatory: this lands first, because every later refinement extends the per-core schedule it introduces. It also gives refinement 6 its overlap lever: round `k+1`'s exchange overlaps round `k`'s normalize/store.
- Update `l1_ledger.md` (exchange_round axis accounting, board bytes per round) and the regime table in `op_design.md`.

**Done when**: the 8 `*-8192-2048-2` cells pass. Also:
- an ad-hoc check passes at per-device `(1, 1, 8192, 4096)` bf16 and at `(1, 1, 4096, 4096)` fp32 with gamma, on the system mesh;
- the golden suite (`EVAL_CAPTURE_L1=0`) shows `supported_fail = 0`;
- the acceptance suite and the bit-identical back-to-back test stay green;
- no regression on the three LOOSE perf cases on the 1×4 ring. Measure device-ns before and after; one block per core must stay the solve's choice whenever it fits.

**Outcome**: the 8 `*-8192-2048-2` cells pass (golden slice, `EVAL_CAPTURE_L1=0`); ad-hoc (1,1,8192,4096) bf16 ± gamma and (1,1,4096,4096) fp32 + gamma pass (`test_distributed_rms_norm_multiround.py`). Solve picks 2 rounds of 5×16 tiles for (2,1,2048,4096) on 106 workers. LOOSE cases still solve to one round; last-launched chip, untraced, 1×4 ring 2 links: 640×1792 22.2 → 21.1 µs, 32×1792 10.5 → 10.1 µs, 2048×1280 48.7 → 46.7 µs (no regression). `MIN_ROUNDS` is the live overlap knob for refinement 6 (gamma path overlaps round r+1's exchange with round r's normalize/store); not measured as a perf lever here.

### [x] Refinement 2 — Sharded input/output placement (HEIGHT_SHARDED + WIDTH_SHARDED)

**Goal**: add `ttnn.TensorMemoryLayout.HEIGHT_SHARDED` and `ttnn.TensorMemoryLayout.WIDTH_SHARDED` to `SUPPORTED["memory_layout"]`, with a sharded output that follows the input's shard spec (the golden suite passes `memory_config=x.memory_config()`). Both use native in-kernel consumption.
- **HEIGHT**: the worker set is the shard grid, `num_w_chunks = 1`, and `block_h` = the shard's tile rows (design regime R9).
- **WIDTH**: `num_w_chunks` = the number of shards, `num_row_groups = 1` (design regime R10). The existing leader gather plus fan-out is the cross-chunk combine.
- **Both**: x is consumed in place through a CB backed on the shard (`ttnn.cb_descriptor_from_sharded_tensor` as `cb_x_block`), and `cb_out_tiles` is backed on the output shard. Never re-read a core's own shard through a `TensorAccessor`.

**Verifier notes**:
- No implementation skill covers `memory_layout` placement yet; do not use `/memory-layouts` (that is RM/TILE layout).
- This is the hardest generality item (complicated sharding), so it goes first among the axis unlocks:
  - **Port-core collision.** `eval.sharding.auto_shard_config` lays shards row-major from core (0, 0) across the whole grid, while Phase 0 hard-places the `2 · num_links` port cores on logical row 0, `x = 0..2L-1`. Every sharded cell would put a port's sender kernel (RISCV_0) on a shard core whose writer also needs RISCV_0. Place the ports on cores outside the shard grid. Every `INPUTS` shard grid leaves free cores, at most 86 of 110 used; refuse with a clear error if none are free.
  - **Workers come from the shard spec.** The worker list and row-group / chunk mapping must be derived from the shard spec (ROW_MAJOR orientation, ragged last grid row), not from the Phase 0 row-major solve. Note that `auto_shard_config` uses `Split.PAD`, so the last shard can carry padding granules.
  - **WIDTH shards pin the block at whole shards**, often 1 tile wide (1792 wide → 56 one-tile shards). That forces `num_w_chunks` up to 86 and `n_gather_tiles = ceil(G · num_w_chunks / 32)` up to 6 at G = 2. Check the board cap (`BOARD_CAP_BYTES`) and the leader's fan-out width; the fan-out lamp (mcast to the contiguous row-group) becomes relevant here.
  - **Sharded × non-aligned cells**: if they need refinement 4's masking, they may stay in EXCLUSIONS until refinement 4 lands. List them explicitly.
- If WIDTH_SHARDED turns out to be more than one focused pass, ship HEIGHT as `[~]` and file `Refinement 2b — WIDTH_SHARDED`. Do not wrap the op in `ttnn.to_memory_config` / interleaved conversions as the "implementation": that is a dishonest SUPPORTED and only acceptable as a named partial-tick escape hatch with the in-kernel follow-up filed.

**Done when**: every golden `test_op` cell with `memory_layout ∈ {HEIGHT_SHARDED, WIDTH_SHARDED}` inside SUPPORTED passes. Cells may be skipped `infeasible_skipped` only for harness shard-geometry OOM. Also:
- the translated sharded tests (`test_sharded_distributed_layernorm`, `test_rms_fuse*`) leave `xfail_expected`;
- a dataflow review confirms both x and out use sharded CB placement and no `TensorAccessor` reads of a core's local shard;
- the golden suite shows `supported_fail = 0`.

**Outcome**:
- Golden slice: the 2160 in-SUPPORTED `test_op` cells with `memory_layout ∈ {HEIGHT_SHARDED, WIDTH_SHARDED}` (TILE gamma or none, fp32 DEST) were run in collection order. 2136 pass and 24 skip `INFEASIBLE_L1`. The skips are fp32 per-device 32×8192 HEIGHT shards: one core would hold a 1 MiB x shard plus a 1 MiB out shard, and the output allocation OOMs, which is harness-side.
- `test_sharded_distributed_layernorm` passes. The `test_rms_fuse*` cells stay `xfail_expected` only because they use ROW_MAJOR gamma (refinement 4).
- Dataflow: x is consumed through `cb_x_block` backed on the shard, and a sharded out is packed into `cb_out_tiles` backed on its shard. No `TensorAccessor` touches a core's own shard.
- Sharded × `h_non_aligned` cells pass; `w_non_aligned` stays out of SUPPORTED for every layout until refinement 4. No EXCLUSIONS were added.

### [x] Refinement 3 — Speed up the perf-flagged profile (640 × 1792, G = 4 ring)

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES` flags the per-device `(1, 1, 640, 1792)` case as the mandatory perf target (PERF FOCUS). Its contract, all of it already in Phase 0 SUPPORTED:
- bf16 x, bf16 TILE gamma, `fp32_dest_acc_en=True`, HiFi4;
- DRAM interleaved, `cluster_axis=1`, Ring, `num_links=2`;
- mesh `(1, 4)` under FABRIC_2D_TORUS_XY.

Its extras set `roofline_us = 16.7` and `target_us = 22.2`, against the 92.1 µs composite baseline. Measured today: the harness `device_kernel_ns` is 189.8 µs, but that is ≈ the sum of the 4 chips' untraced kernel times and is dominated by host dispatch skew at the Fabric fence. The last-launched chip measures 22.2 µs (implementer, untraced); the targets are traced numbers. Optimize THIS config, not a proxy, using the relevant patterns in `ttnn/ttnn/operations/examples/master.md`. The planner's perf lamps (`op_design.md` § Perf lamps) list the cheap candidates:
- **block-size co-tune** of the `num_w_chunks` / `MIN_BLOCK_W` / `block_h` solve: 5 chunks × 12 tiles today. Fewer chunks mean a shorter leader gather and fan-out; more mean more DRAM-parallel workers (`width_split`, `compute_block_size`). Whole tiles is the floor; coarser blocks amortize.
- **fan-out of inv_rms**: `num_w_chunks` unicast rounds today. Try an `mcast_pipe` fan-out when a row-group is a contiguous grid-row segment (`mcast_topology`).
- **port placement**: logical row 0 today ("simple"). Try ports next to each link's Ethernet core, via a host-side lookup and **not** a probe program, because of the one-dispatch rule.
- **board-ready granularity**: per-source waits instead of a whole-board wait.
- **`OUT_DEPTH`** (writer/compute overlap).

These are T1/T2 knobs; take several in one phase. No SUPPORTED change.

**Done when**:
- measured device-ns improves on the flagged 640×1792 ring case, moving it toward its 22.2 µs target. Measure the last-launched chip, or a traced call, so host dispatch skew between chips is excluded, and record the method;
- the golden suite stays green, with the case still meeting its precision gate;
- no regression across the config-spanning guard set: one representative per distinct kernel path (gamma / no gamma, fp32 / bf16, `num_w_chunks = 1` vs > 1, single vs multi row-group) × placement (Linear / Ring, 1 / 2 links). This includes the other two LOOSE cases (32×1792 decode, 2048×1280).

**Outcome**: 640×1792 ring, 2 links, last-launched chip (realtime profiler, untraced, median of 3) **21.65 → 18.87 µs** (target 22.2). The two LOOSE siblings: 32×1792 10.26 → 8.13 µs and 2048×1280 47.2 → 37.9 µs. No guard case is slower beyond ±1%.

Levers that landed:
- the ready credit in place of the start-of-call fence;
- an auto-rounds solve on the gamma path (2 rounds of 1×6 at the perf focus);
- a single-write fan-out;
- the DRAM bank stagger;
- a fix for a latent `cb_out_tiles` straddle-pop bug that the new plans exposed.

Measured flat and parked as knobs: `OUT_DEPTH`, `STORE_FLUSH` and `COMBINE_AHEAD`; the last one also made 2048×1280 slower.

What binds now:
- DRAM: x reads take ~7.5 µs at ~310 GB/s plus a ~3 µs store tail. `ttnn.clone` of the same tensor floors at 12.4 µs.
- The last round's chain: round-1 arrivals ~2.5 µs, because the ring ports send round r+1 only after relaying round r. Then combine ~1, fan-out ~1.1 (shares NoC1 with the stores) and normalize ~0.7 µs.

Next to try: a dynamic port schedule together with a leader-side combine that can run ahead, and an mcast fan-out over a grid-row segment. Each piece was tried alone (port scheduler, reader scheduler, combine-ahead) and measured flat, because the other serializations still bound the chain. A fused port-side combine (the ports hold the whole board) would remove the gather/fan-out hops altogether, but it is a scheme change beyond this round.

### [x] Refinement 4 — Non-tile-aligned W_local + ROW_MAJOR gamma

**Goal**: add `"w_non_aligned"` to `SUPPORTED["alignment"]` and `ttnn.ROW_MAJOR_LAYOUT` to `SUPPORTED["gamma_layout"]`, both natively in the kernels:
- **w_non_aligned**: the tile padding of the last tile column of each device's W_local must NOT enter the sum of squares (a MUST rule in the prompt). Mask it in the reader (zero-fill the padded columns of the last x tile of the last chunk before the push) or with a mask tile in `sumsq_block`. The mean already divides by the true `W_global = G · W_local` (`1/W_global` on the host), so do not use a padded width anywhere. The normalize output's padded columns are don't-care.
- **ROW_MAJOR gamma**: the per-device "sticks" `(1, 1, ceil(W_local/32), 32)` have one 32-element stick per tile column, and that stick is exactly the row 0 the Row-broadcast reads. The reader drops stick `c` into `cb_gamma` page `c` as face0-row0 (elements 0–15) and face1-row0 (elements 16–31). DRAM reads must stay 64 B aligned: read the whole stick, then relocate the second half locally. The tail of a non-aligned W_local is zero-padded by the caller.

**Implementation skill**: /memory-layouts

**Verifier notes**:
- Bundled because both are reader-side edge handling of the same W-chunk tiles: the last chunk's last tile, and the gamma slivers.
- `h_non_aligned` is already SUPPORTED (verifier probe). The padded rows of the last tile row are row-independent end to end, so they never reach a valid row. Keep that property when adding the W mask.
- If refinement 2 left sharded × non-aligned cells in EXCLUSIONS, lift them here.
- No `ttnn.to_layout` / `ttnn.tilize` / `ttnn.fill_implicit_tile_padding` wrapper on the input or gamma. Only the partial-tick escape hatch may use one, and then the in-kernel follow-up must be filed as `Refinement 4b`.

**Done when**:
- every golden `test_op` cell with `alignment = w_non_aligned` and every cell with `gamma_layout = ROW_MAJOR` passes, on the SUPPORTED dtypes and memory layouts;
- the translated non-aligned / RM-weight tests (`test_distributed_fused_rmsnorm_odd_hidden_dim`, `test_simulated_distributed_norm_*_non_tile_aligned_width`, `test_distributed_rmsnorm_memory_layouts`) leave `xfail_expected`;
- `supported_fail = 0`.

### [ ] Refinement 5 — Numerical configurability: bfloat8_b + fp32_dest_acc_en=False

**Goal**: add `ttnn.bfloat8_b` to `SUPPORTED["dtype"]` and `False` to `SUPPORTED["fp32_dest_acc_en"]`. Keep the `{float32, fp32_dest_acc_en=False}` EXCLUSION (precision-convention MUST). Make the intermediate CB formats follow `fp32_dest_acc_en`, and set the `UnpackToDestFp32` tagging per mode. Cells that fail out of the box go to EXCLUSIONS, not to their own refinement.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**:
- The prompt requires the exchanged partials to stay fp32 "on the wire and when summing, whatever the input dtype". `cb_partial_row`, the board, `cb_gather` and `cb_inv_row*` keep Float32 pages under both DEST modes; record that format and API interoperability reason in the ledger.
- `cb_xg` (Float32 today) should drop to a 16-bit page when `fp32_dest_acc_en=False`, the over-wide case of ledger audit 3: 16-bit DEST already rounded it.
- `UnpackToDestFp32` on `cb_sumsq*`, `cb_gather` and `cb_inv_row` assumes a 32-bit DEST. Gate it on the mode.
- bfloat8_b x/out need only the dtype-derived CB formats the descriptor already uses (`tile.get_tile_size(dtype)`). The verifier's probe of bf8b with fp32 DEST is recorded in verification_report.md.
- Precision context, not a gate: the fp32 corner carries a −0.12% got/true bias from TF32 truncation of the FPU multiply operands (verification_report.md, probe-confirmed). If the fp32 tolerance ever tightens, the lever is to run the post-stats multiply on the SFPU in fp32 for `dtype=float32` only. It is measured slower in `compute_fusion`, so do not apply it to bf16 / bf8b.
- Lands after the sharding and alignment refinements on purpose: it is the cheapest axis and extends the kernels they reshape.

**Done when**:
- every golden `test_op` cell with `dtype = bfloat8_b` or `fp32_dest_acc_en = False` that is not EXCLUDED passes within the harness tolerances;
- `{float32, False}` still raises `ExcludedCell`;
- `supported_fail = 0`;
- the precision baseline test is extended to the new corners, and its numbers are recorded in the changelog.

### [x] Refinement 6 — Second perf pass on the perf-flagged profile (structural levers)

**Type**: perf

**Goal**: same target as refinement 3, the flagged per-device `(1, 1, 640, 1792)` bf16 case on the G = 4 ring with 2 links, `fp32_dest_acc_en=True`, against its `target_us = 22.2` / `roofline_us = 16.7`. This pass takes the structural (T2/T3) levers the first pass left, using `ttnn/ttnn/operations/examples/master.md` (`mcast_topology`, `shared_input_reuse`, `tensix_all_reduce`):
- **exchange/compute overlap**: two or more exchange rounds per core with the refinement-1 block loop, so round `k+1`'s Fabric exchange hides under round `k`'s normalize/store. The solve must be allowed to trade `block_h = 1` × 100 workers for multi-round blocks if it measures faster.
- **gamma broadcast**: design regime R4 (`deferred`). One reader per W-chunk column reads the gamma slivers once and multicasts them down the column (`mcast_pipe`). Validate it by building it and measuring device-ns, not by inferring from an ablation.
- **local pre-combine before the exchange**: regime R3, if the first pass shows the leader's board read on the critical path.

Take one T3 lever at a time; each must show its own device-ns win. No SUPPORTED change.

**Done when**:
- measured device-ns on the flagged case improves over the refinement-3 result, by the same measurement method;
- the golden suite stays green and the case still meets its precision gate;
- no regression across the config-spanning guard set (as in refinement 3, plus one sharded representative from refinement 2 and one bf8b representative from refinement 5).

**Outcome**: 640×1792 ring, 2 links, last-launched chip (realtime profiler, untraced, median of 3) **18.87 → 18.06 µs**. Same-session baseline 18.79; target 22.2, roofline 16.7. The 1-link case went 21.4 → 20.3 µs; bf8b 15.9 → 15.2 µs.

The win comes from the overlap lever's solve:
- `TILE_SLACK` lets a plan with one gather tile per row win (8 chunks × 7 tiles over 2 rounds; G·C = 32) over the strict tile minimum (10 × 6, two gather tiles).
- Balanced row-slots remove under-filled last rounds.
- 640×1536 and 2048×2048 also gain 8%. No guard case is slower beyond noise.

The other two levers, each built and measured:
- **R4 gamma broadcast**: `mcast_pipe` per W-chunk column with a 2-D column placement. Flat to 1% slower (19.05 vs 18.88 µs), so it is parked at `GAMMA_MCAST = False`, with a pinned test. The gamma slivers are a 6%-of-x read that never sat on the critical path once x(0) is first.
- **R3 pre-combine**: 18.8 → 26–27 µs, kept at auto (board cap only).

What binds now:
- x DRAM reads (~7.5 µs) plus the store tail.
- The last round's exchange chain: round-1 arrivals behind the ports' in-order relay, then gather, combine, fan-out and normalize.

Next to try: a port-side fused combine (the ports hold the whole board), which removes the gather and fan-out hops and the relay ordering together. Not tried here: it is a new topology, beyond this refinement's levers.
