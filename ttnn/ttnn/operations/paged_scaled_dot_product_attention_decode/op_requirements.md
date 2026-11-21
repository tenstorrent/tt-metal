# Operation Requirements: paged_scaled_dot_product_attention_decode

## Definition
- **Formula**: for every user u and query head h, with g = NH // NKV and key(u,c,p) = K[page_table[u, p // block_size], c, p % block_size, :]:
  `out[0,u,h,:] = Σ_p softmax_p(scale · q[0,u,h,:]·key(u,h//g,p) + mask[u,p]) · value(u,h//g,p)`, mask = 0 for p ≤ cur_pos[u], −∞ after it; `scale=None` → 1/√D.
- **PyTorch Reference**: `reference(...)` in `eval/specs/paged_scaled_dot_product_attention_decode/spec.py` (float64, user by user: gather K/V through the page table, `repeat_interleave` the heads by g, scores · scale + position mask, softmax, · V) — per user equal to `torch.nn.functional.scaled_dot_product_attention(q[0,u].unsqueeze(1), K, V, scale=scale).squeeze(1)` over positions 0..cur_pos[u].
- **Import Path**: `from ttnn.operations.paged_scaled_dot_product_attention_decode import paged_scaled_dot_product_attention_decode`
- **Function Signature**:
  `paged_scaled_dot_product_attention_decode(input_tensor_q, input_tensor_k, input_tensor_v, page_table_tensor, *, cur_pos_tensor, scale=None, sliding_window_size=None, memory_config=None, program_config=None, compute_kernel_config=None) -> ttnn.Tensor`

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.
> **Backend**: this run is Quasar on ttsim (no clock). Every perf gate below is a **structural count** read off the op by a test or probe (`.claude/references/backends/ttsim.md`, *Performance measurement*); the gain in time is unconfirmed until a timed backend (`ttsim-perf`) or hardware measures it. Never report a simulator timing as a speedup.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED placement**: q `memory_config` = L1, `sharding` = height; `output_memory` = DRAM
- **SUPPORTED shape-derived axes**: alignment = tile_aligned, kv_heads_mode = grouped, batch = single
- **SUPPORTED precision**: fp32_dest_acc_en ∈ {True, False}; any `math_fidelity` / `math_approx_mode` (R9)
- **Cores**: flash-decode over a virtual key sequence split across `G_u` clusters × 4 NEOs (√N worker rule), semaphore-gated combine on the group's rank-0 cluster; thread plan (R, C, W) = (4, 4, 2)
- **Compute config**: `compute_kernel_config` exposed, default `default_compute_kernel_config()` = HiFi4, fp32_dest_acc_en=True, math_approx_mode=False; device configs (`ttnn.WormholeComputeKernelConfig`) read by attribute (R10)
- **Golden baseline**: 23 / 23 supported cells pass (14 PHASE0 cells + 9 loose cases), 178 xfail_expected, 64 invalid_skipped, 10 rejections pass; 0 loud (per verifier CLI)

**Probe evidence for the queue (verifier, this phase).** With `SUPPORTED` widened to `TARGET` in-process only and every cell graded by the specification's own check, 135 of the 144 TARGET cells (16 INPUTS × 3 query placements × {bf16 + both accumulators, float32 + fp32 accumulator}) already pass on the Phase 0 kernels: float32 (PCC ≥ 0.999999, rel-RMS ≤ 0.0026 against 0.01), DRAM- and L1-interleaved queries, several users (up to 32), a key/value head for every query head, and 8 / 16 / 20 query heads. The 9 failures are the one entry with D = 4096 (`((1, 2, 32, 4096), (2, 1, 32, 4096), …)`), refused on the host with `NotImplementedError` ("the head dimension is too wide for one cluster's L1"): the per-NEO rings that scale with `tensor_d_tiles` (q, K, V, q', O state, step, partial) do not fit 4 NEOs at `tensor_d_tiles` = 128. So each generality refinement below is a SUPPORTED widening plus its golden evidence, except the wide-D regime, which is real kernel work.

### [ ] Refinement 1 — Several users, every head geometry, and the wide-D regime

**Goal**: add `batched` to `SUPPORTED["batch"]`, `equal` to `SUPPORTED["kv_heads_mode"]` and `h_non_aligned` to `SUPPORTED["alignment"]`, and build the wide-D regime so the batched D = 4096 entry fits a cluster's L1 instead of raising `NotImplementedError` on the host. Every cell those three values open passes, at every query placement already supported, at both accumulators.

**Implementation skill**: /memory-budget-metal

**Verifier notes**:
- Hardest first: the wide-D regime is the one piece of new structure in the generality queue (the design's deferred `R-wide-D` regime row: O state blocked along D, `o_block_tiles < tensor_d_tiles`, which the design already names as a knob). The rings that scale with `tensor_d_tiles` today are `q_tiles` (Dt), `k_tiles` / `v_tiles` (2·Dt each), `q_scaled` (Dt), `out_acc` (Dt), `step` (Dt+1), `partial_out` (Dt+2) — at Dt = 128 with fp32 intermediates that is ≈ 3 MiB per NEO against a 3.9 MB cluster budget for 4 NEOs (`l1_ledger.md`, Footprint). Shrink the streaming windows to fixed depths first (q, K, V, partial-out are streams — their capacity must not span D; see the ledger's axis accounting), then block the resident O state / q' along D. Keep the predicate-guarded fast path: Phase 0's residency where it fits, the blocked regime only where the L1 predicate says otherwise (single host function, no second copy of the footprint arithmetic). Update `l1_ledger.md` (new rows, symbol bounds, the regime table) in the same change.
- The other two values need no kernel change on the probe's evidence (row masks for NH < 32 and group size 1 already exist: `has_row_groups`); they ride here because the D = 4096 entry is a batched cell and the three shape axes cross in INPUTS (entries with 2 users and a key/value head per query head, 2 users and 20 heads).
- Several users must keep the grid filled: `_group_assignment` gives each of B users `G_u` clusters when `B·G_u ≤ grid`, else one cluster per user dealt by `split_work_to_cores(row_wise=True)`. The thread test (`test_every_worker_wrote_its_own_partial`) gets one batched case (e.g. the 32-user entry) so every user's workers are seen to write their slots.

**Done when**: `batched`, `equal`, `h_non_aligned` are in SUPPORTED; every golden cell they open passes, the D = 4096 entry included; `verify_supported` shows 0 `supported_fail`, 0 `xpass_drift`, 0 `xfail_wrong_mode`; the thread and session tests pass; the ledger matches the new rings.

### [ ] Refinement 2 — Interleaved queries and float32

**Goal**: add `DRAM` to `SUPPORTED["memory_config"]` and `interleaved` to `SUPPORTED["sharding"]` (queries interleaved in DRAM or in L1, beside the height-sharded L1 of Phase 0), and add `ttnn.float32` to `SUPPORTED["dtype"]` (inputs, caches and output float32; `fp32_dest_acc_en=False` stays refused for it, rule R7). Every cell these values open passes, across every value Refinement 1 added.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**:
- Both are light (precision knobs and a placement served by one read path) and the probe shows them passing on the Phase 0 kernels, so this is SUPPORTED + `validate()` + golden evidence. Order after Refinement 1 so float32 is checked on the wide-D rings too (a float32 D = 4096 cell doubles every B ring: the regime predicate must see the dtype).
- Placement has no implementation skill. q is read through its `TensorAccessor` in every case (rule R12 forbids depending on the shard grid; spec GUIDANCE), so interleaved DRAM / L1 and height-sharded L1 are the same reader path. This is not the "local shard re-read" anti-pattern: q is reuse-shared by every worker of a user and the workers are not the shard's core (design regime `R-q-in-place`, deferred with its reason). The zero-copy / read-once work on q is Refinement 4.
- float32 with fp32 DEST: `cb_q_scaled` and `cb_probs` follow the input format (matmul sources must match K/V, `matmul.h:99-102`), so they become Float32 here — re-check the L1 predicate and the per-ring `unpack_modes` (`UnpackToSrc` for Float32 rings; `UnpackToDest` is not used, design Unavailable (10)).

**Done when**: `DRAM`, `interleaved`, `ttnn.float32` are in SUPPORTED; every golden cell passes (float32 graded at (0.999, 0.01)); `verify_supported` shows 0 loud; the precision baseline test gains float32 rows.

### [ ] Refinement 3 — K/V stream: more reads in flight per barrier on the long-context call

**Type**: perf

**Goal**: speed up the largest supported call, `LLAMA_CAPTURED` (q [1, 1, 32, 64], caches [128, 8, 32, 64], cur_pos near 4095: 1024 units over 8 clusters × 4 NEOs, 32 position tiles per worker), whose cost is the K/V read stream. Today every reader hart reads K one tile-row per barrier (`tensor_d_tiles` = 2 reads in flight at D = 64) and V one d-sub-block per barrier (`pv_subblock_tiles` = 2), into rings of `KV_DEPTH_ROWS` = 2 tile-rows. Catalog: `double_buffer` (block of 4–8 reads before one barrier, ring = 2 × block) and `trid_double_issue` (several blocks outstanding, barrier on the oldest trid) in `ttnn/ttnn/operations/examples/master.md`. Co-tune a reads-per-barrier block knob (in position tile-rows, independent of D) with `KV_DEPTH_ROWS`; both stay single-source host constants passed as CT args. No SUPPORTED change.

**Structural gate (ttsim)**: K and V reads issued per barrier on a reader hart at `LLAMA_CAPTURED`: **2 now → ≥ 8**, read off the op (CT args exposed to a test, or a probe counting barriers), with the L1 predicate still met at every D ≤ 128.

**Done when**: the structural count reaches ≥ 8 reads per barrier for K and for V at `LLAMA_CAPTURED` (recorded before / after in the changelog, gain in time marked unconfirmed); the golden suite is green; no regression on the config-spanning guard set (one representative per kernel path × placement: one cluster and several clusters; height-sharded L1, interleaved DRAM; bf16 at both accumulators and float32; one batched, one wide-D cell), its structural counts not worse and its cells passing.

### [ ] Refinement 4 — Read q once per cluster

**Type**: perf

**Goal**: q is reuse-shared: every worker (NEO) of a user reads all `tensor_d_tiles` q tiles from the shard core over the NoC, so a cluster reads q C = 4 times. The design's deferred regime `R-q-reuse` names the fix: make the q ring an ALL-pattern consumer ring (DataflowBuffer.md, "ALL": every NEO sees every entry) fed once per cluster — the reader harts can deal q's tiles among themselves STRIDED, each NEO consuming all of them — or a multicast to the group. It is a new dataflow topology (one read, a fan-out), so it stands alone. Catalog: `shared_input_reuse`, `mcast_topology` in `ttnn/ttnn/operations/examples/master.md`. No SUPPORTED change.

**Structural gate (ttsim)**: q tiles read over the NoC per engaged cluster per user: **C · `tensor_d_tiles` (8 at D = 64) → `tensor_d_tiles` (2)**, read off the op by a test.

**Verifier notes**: a worker with no unit (the neutral partial) skips q today; with an ALL ring every NEO must consume what is pushed, so either every NEO of an active cluster pops q or the ring is filled only for clusters whose four NEOs all have units — decide by the push = pop rule, not by the neutral case. `ttnn.Mcast` does not take a multi-threaded kernel (design Unavailable (17)); the ALL ring does not have that limit. Validate by building it and reading the count, not by a remove-one-stage ablation.

**Done when**: the count drops to `tensor_d_tiles` per cluster per user at `LLAMA_CAPTURED` and on the guard set; golden suite green; the thread test still shows every worker writing its own partial; no regression on the guard set of Refinement 3.

### [ ] Refinement 5 — Cheaper combine on the long-context call

**Type**: perf

**Goal**: at `LLAMA_CAPTURED` the combiner cluster merges W_active = 32 partials per user, and each merge is a stats DEST section (2 copies, then max, 2 sub, 2 exp on the SFPU, on whole tiles whose meaning is column 0), a β product over `d_share_tiles + 1` tiles and the α merge chain; with `tensor_d_tiles` = 2 < C = 4, two of the four combiner NEOs recompute a padded d-tile. Levers (`ttnn/ttnn/operations/examples/master.md`): `sfpu_tile_scope` (run the column-vector stats ops on column 0 only), `eltwise_l1_vs_dest_accumulate` (fold α·O into DEST instead of re-reading O — also the design's perf lamp "α·O folded into DEST before PV" for phase 1), and the padded d-share (deal the combine to `min(C, tensor_d_tiles)` NEOs' worth of real work). Several T2 levers: one phase. No SUPPORTED change.

**Structural gate (ttsim)**: on the combiner at `LLAMA_CAPTURED`, per partial merged: SFPU vector ops in the stats section (**5 SFPU ops — max, 2 sub, 2 exp — × 32 vectors now → × 8**) and L1 round-trips of the β-scaled step (**pack + unpack of `d_share_tiles + 1` `step` tiles now → 0**: β·O_k folded into the merge's DEST section); and redundant padded d-tile merges per user (**2 NEOs now → 0**). Read off the op by a test or by kernel CT/structure assertions.

**Done when**: those counts reach their targets; precision baseline unchanged within its tolerances (the fold reorders sums); golden suite green; no regression on Refinement 3's guard set.

## Omissions

None. Every `(axis, value)` of `TARGET − SUPPORTED` is named above: `SUPPORTED["batch"]` batched, `SUPPORTED["kv_heads_mode"]` equal, `SUPPORTED["alignment"]` h_non_aligned (Refinement 1); `SUPPORTED["memory_config"]` DRAM, `SUPPORTED["sharding"]` interleaved, `SUPPORTED["dtype"]` ttnn.float32 (Refinement 2). `SUPPORTED["memory_config"]` L1 is already supported (with height sharding; Refinement 2's interleaved adds L1-interleaved).
