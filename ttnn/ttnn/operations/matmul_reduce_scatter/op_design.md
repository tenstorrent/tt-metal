# Operation Design: matmul_reduce_scatter

## Overview

| Field | Value |
|-------|-------|
| Classification | fused (compute + CCL): per-device tiled matmul + fabric reduce-scatter (SUM) in **one** `ttnn.generic_op` dispatch |
| Goal | The tensor-parallel row-parallel linear: every device computes the partial `A[g] @ W[g]`, the group along `cluster_axis` sums the partials and scatters the sum. The transport of a finished output block starts while later blocks are still being computed. |
| Math | `P = Σ_{g ∈ group} A[g] @ W[g]`; device at group position `p` keeps block `p` of `P` along `scatter_dim` (`-2`: rows `p·M/G ..`, `-1`: columns `p·N/G ..`) |
| Mode | Derivative (fused) — transport derived from `fabric_reduce_scatter`, matmul from the 2D dual-multicast matmul (`eval/prompts/matmul.txt`) |
| References | `ttnn/ttnn/operations/examples/fabric_reduce_scatter/program_descriptor_with_inline_kernels.py` (ports, relays, final cores, arrival counters, ready fence, `(G+1)`-slot scratch, port reader/sender/add/final kernels at :46-446, host wiring :462-628); `ttnn/ttnn/operations/examples/fabric_all_gather/program_descriptor_with_inline_kernels.py` (`_CHUNK_WALK` :40-87, `build_groups` :346, `_schedule` :359, `_needs_ready` :382, `plan()` :660); `references/blackhole-fabric.md` (§5 worker API, §6 rules 1-10); `eval/prompts/matmul.txt` (2D dual multicast); `ttnn/ttnn/operations/examples/master.md` entries `mcast_topology`, `shared_input_reuse`, `double_buffer`, `matmul_output_subblock`, `eltwise_l1_vs_dest_accumulate`, `compute_block_size`, `noc_placement` |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` (A) | mesh `ttnn.Tensor` | yes | per device `(..., M, K)`, rank 2-4, leading dims 1, TILE, DRAM interleaved | — | RT (address) |
| `weight` (W) | mesh `ttnn.Tensor` | yes | per device `(K, N)`, TILE, DRAM interleaved; `K` equals A's `K` | — | RT (address) |
| `cluster_axis` | int (kw-only) | yes | `0` or `1`; group size `G = mesh_shape[cluster_axis] ≥ 2` | — | host (selects groups, schedule) |
| `scatter_dim` | int | no | `-2` or `-1` | `-2` | host (selects block origin per block; same kernels) |
| `topology` | `Topology` | no | `Linear` (Phase 0); `Ring` (TARGET) | `Linear` | host (schedule) |
| `num_links` | int \| None | no | explicit: `1 ≤ L ≤` usable links on every hop of the group; `None` = all usable | `None` | host (number of ports/finals, bank-set stride) |
| `compute_kernel_config` | `ComputeConfigDescriptor` \| None | no | any fidelity; `fp32_dest_acc_en` any (a precision **floor**) | `default_compute_kernel_config()` = HiFi2, `fp32_dest_acc_en=True` | CT of the matmul compute kernel only |
| `memory_config` | `MemoryConfig` \| None | no | Phase 0: `None` or DRAM interleaved | DRAM interleaved | host |

`default_compute_kernel_config()` is one exported function; `None` resolves through it and nothing else hard-codes the default. The transport add kernels **always** run with `fp32_dest_acc_en=True` (every cross-device addition accumulates in fp32 — a requirement, independent of the user's config).

`Topology` is imported at module level as `from ttnn.operations.ccl import Topology` (not `ttnn.Topology`).

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | A: `(..., M, K)`, rank 2-4, every leading dim 1; W: `(K, N)`; A's K == W's K; identical on every device (mesh tensors) |
| Dtype | A: `bfloat16` (Phase 0), `bfloat8_b` (TARGET). W: `bfloat16`, `bfloat8_b` (Phase 0), `bfloat4_b` (TARGET) |
| Layout | TILE |
| Memory | DRAM interleaved |
| Divisibility | `scatter_dim=-2`: `M % (32·G) == 0`; `scatter_dim=-1`: `N % (32·G) == 0` (else `ValueError`); M, K, N tile-aligned (`alignment` axis) |

### Output

| Property | Value |
|----------|-------|
| Shape | A's leading dims + `(M/G, N)` for `scatter_dim=-2`, `(M, N/G)` for `scatter_dim=-1` |
| Dtype | `bfloat16` always (reduced sum feeding the residual stream) |
| Layout | TILE |
| Memory | DRAM interleaved (or `memory_config` if DRAM interleaved) |

### Op-internal persistent buffers (cached per plan key, never per call)

| Buffer | Shape / layout | Where | Purpose |
|--------|----------------|-------|---------|
| `relay_scratch` | ROW_MAJOR bf16 `[1, 1, (G+1)·segs_per_block, seg_tiles·1024]` → **one page = one segment = `seg_tiles` bf16 tiles** (`seg_tiles·2048` B) | DRAM interleaved | landing zone of every fabric packet (slot `j` = block `j` arriving forward or backward; slot `G` = the receiver's own block arriving backward) — the reference's `(G+1)`-slot layout with a segment page instead of a tile page |
| `handoff_l1` | HEIGHT_SHARDED L1, one shard per compute core of `handoff_depth·core_m_tiles·core_n_tiles` bf16 tiles | L1 of the compute cores | backs `cb_partial_handoff` (`cb_descriptor_from_sharded_tensor`) so its L1 address is known on the host and is passed to the transport readers |
| 5 global semaphores | `ttnn.create_global_semaphore` over transport ∪ compute cores, initial 0 | L1 | `sem_arrival_fwd`, `sem_arrival_bwd`, `sem_ready_fence` (as the reference) + `sem_block_ready` (transport cores) + `sem_block_ack` (compute cores). Every kernel re-arms what it consumed (subtract) before exiting. |

## Blocking Model

Everything below is downstream of this section. Semantics: `.claude/references/blocking-model.md`.

**The op in one sentence:** every chip walks the `G` **scatter blocks** of its partial product in transport order; each scatter block is computed by the whole compute grid in one 2D-multicast matmul pass (per-core block `core_m_tiles × core_n_tiles`, K-blocked), parked in an L1 hand-off buffer, and drained segment-by-segment by the transport cores (ports: gather + relay-add + fabric send; finals: gather + 3-way add + output store), which follow `fabric_reduce_scatter` exactly except that "my partial" comes from compute-core L1 instead of DRAM.

Notation: `Mt = M/32`, `Kt = K/32`, `Nt = N/32`. The scatter block shape is
`(blk_m_tiles, blk_n_tiles) = (Mt/G, Nt)` for `scatter_dim=-2` and `(Mt, Nt/G)` for `scatter_dim=-1`;
block `j`'s origin in `P` is `(j·blk_m_tiles, 0)` resp. `(0, j·blk_n_tiles)`. **This is the only place
`scatter_dim` enters the kernels** (a per-block origin runtime arg): the transport, the scratch, the
output and the per-core blocking are identical for both scatter dims, because the output and every
scratch slot are shaped `(blk_m_tiles, blk_n_tiles)`.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `group` (device g, G of them) | **dependent** — the result is a sum over it | — (one device = one partial) | whole partial per device | mesh shape (caller) | across devices by construction; combined over Fabric by the reduce-scatter transport | Ring = host schedule change (see Regimes) |
| `scatter_block` j ∈ [0, G) | **independent** for the matmul (blocks never interact on a chip); **reuse-shared carrier** — the operand that does not vary along it (A for `-1`, W for `-2`) is re-needed by every block | `blocks_in_flight` (scatter blocks one compute pass covers) | 1 | host constant in `_plan_blocking()` | **not** spread across cores: walked sequentially by every chip in `compute_order` (transport order) — spreading it would finish all blocks at once and kill the overlap | knob-turn (`blocks_in_flight=2` trades fill for occupancy, lamp L3) |
| `m` (block rows, `blk_m_tiles`) | **independent** — output rows never interact | `core_m_tiles` | `ceil(blk_m_tiles / m_lines)` from the grid factorization (below) | host `_plan_blocking()` → CT arg | split across the grid's **m-lines** (one grid dimension, chosen by the factorization) | knob-turn |
| `n` (block cols, `blk_n_tiles`) | **independent** — output columns never interact | `core_n_tiles` | `ceil(blk_n_tiles / n_lines)` | host `_plan_blocking()` → CT arg | split across the grid's **n-lines** (the other grid dimension) | knob-turn |
| `k` (Kt) | **dependent** — every output tile is a sum over K | `k_block_tiles` (K tiles per K-block) | largest divisor of `Kt` with the streamed K-block CBs (`operand_depth`-buffered) ≤ `min(STREAM_BUDGET, L1_CB_BUDGET − resident − accum − handoff)`, `STREAM_BUDGET` = 384 KiB | host `_plan_blocking()` → CT arg | **not split across cores** — sequential accumulate inside the core (packer L1 accumulation) | scheme-change (K-split, deferred row R6) |
| A rows×K (operand) | **reuse-shared** along `n` (every n-line needs the same A rows) **and** along `scatter_block` for `scatter_dim=-1` | resident iff predicate `RESIDENT` (Regimes) | resident for the focus case | host predicate | read once per m-line by an injector and **multicast along the n-direction line** (mcast_pipe) | — |
| W K×cols (operand) | **reuse-shared** along `m` (every m-line needs the same W columns) **and** along `scatter_block` for `scatter_dim=-2` | resident iff predicate `RESIDENT` | streamed for `-1`; resident when it fits for `-2` | host predicate | read once per n-line by an injector and **multicast along the m-direction line** | — |
| `segment` (transport stage data: `seg_tiles` consecutive tiles of one block row) | **independent** — each segment is added and sent on its own | `seg_tiles` | `min(run_pages, blk_n_tiles)`, `run_pages = max_payload // 2048` | host, from `ttnn.get_tt_fabric_max_payload_size_bytes()` | **spread over transport cores**: the `L` ports of a direction split a block's segments by **scratch bank set** (bank `l`, stride `L`); the `2L` finals split the own block by bank `l + h·L`, stride `2L` (as the reference) | knob-turn |
| `link` l ∈ [0, L) | **independent** — disjoint bank sets, one routing plane each | `num_links` | caller's value or all usable | `num_links` → host | one port per (link, direction), two finals per link | knob-turn |
| `direction` (fwd toward p+1 / bwd toward p−1) | **independent** — disjoint block lists | — | both | schedule | one port core per (direction, link) | — |

**Grid factorization (pinned; `_plan_blocking()`).** The compute rectangle is `R_c × C_c` cores (the
grid minus the transport row(s), below). Two orientations are evaluated: (A) grid rows = m-lines,
grid cols = n-lines; (B) grid cols = m-lines, grid rows = n-lines. For each,
`core_m_tiles = ceil(blk_m_tiles / m_avail)`, `core_n_tiles = ceil(blk_n_tiles / n_avail)`, then
`m_lines = ceil(blk_m_tiles / core_m_tiles)`, `n_lines = ceil(blk_n_tiles / core_n_tiles)` (idle
lines dropped). Pick the orientation minimizing `core_m_tiles·core_n_tiles` (the per-core critical
path per block); ties → larger `min(core_m_tiles, core_n_tiles)` (bigger subblocks), then fewer cores.
Example on an 11-wide × 9-tall compute rectangle (LoudBox, transport row removed): FOCUS
`(20, 56)` → orientation B, `core 2×7`, 10×8 = 80 cores, 14 tiles/core/block; GLM `(20, 48)` → B,
`2×6`, 80 cores; MiMo `(16, 128)` → A, `2×12`, 8×11 = 88 cores.

**Operand-reuse check (mechanical, per (operand, split)).**

| Split | Operand | Varies along the split? | Consequence |
|-------|---------|-------------------------|-------------|
| `m` across m-lines | W | no | reuse-shared → one W injector per n-line, multicast along the line (`mcast_pipe`) |
| `m` across m-lines | A | yes | — |
| `n` across n-lines | A | no | reuse-shared → one A injector per m-line, multicast along the line |
| `n` across n-lines | W | yes | — |
| `scatter_block` in time | A (`-1`) / W (`-2`) | no | reuse-shared across blocks → **resident** (load once, replay) when `RESIDENT` holds, else re-streamed per block (regime S) |
| `scatter_block` in time | W (`-1`) / A (`-2`) | yes | streamed once in total (each block needs a disjoint slice) |

**Stall-shadow check.**

| Waiting stage | Waits on | What runs in the window | Why legal |
|---------------|----------|-------------------------|-----------|
| port reader (relay block j) | upstream chip's arrival of block j (fabric) | the compute grid keeps computing blocks j+1.. into the other hand-off slot(s) | the relay add is on the transport cores, not in the matmul loop; hand-off depth decouples them (the matmul never waits on a remote chip) |
| compute core (slot reuse) | acks of block `idx − handoff_depth` | nothing — this wait only happens when transport is the binding resource (link-bound shapes), where idling compute is the correct outcome | — |
| final cores (own block p) | both arrivals of block p (last in the neighbours' lists) | the matmul computes block p **last** (`compute_order` ends with `p`) so its own partial is ready exactly when the arrivals land | addition is commutative/associative: own + fwd + bwd can be summed in fixed order after both land; order is fixed ⇒ deterministic |
| transport cores at call start | first finished block (pipeline fill) | nothing independent exists (transport has no data before the first block); fill = one block of matmul, the term the roofline already budgets | — |
| port sender at call start | ready fence of the downstream chip | compute starts immediately; the fence gates only fabric writes | as the reference |

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_weight_kblock` / `cb_act_kblock` (streamed operand K-blocks) | `operand_depth` | 2 | injector prefetches K-block k+1 (DRAM read + multicast) while compute multiplies k |
| `cb_partial_handoff` | `handoff_depth` | 2 | compute packs block idx+1 while the transport cores still drain block idx; ≥ 2 decouples the matmul from remote arrivals (stall-shadow) |
| transport CBs (`cb_xport_partial`, `cb_xport_arrival_*`, `cb_xport_sum`) | `xport_group` (segments per handshake) × 2 | `xport_group = max(1, min(8, XPORT_CB_BYTES // (2·seg_bytes)))`, `XPORT_CB_BYTES = 112 KiB` (reference) | reader batches `xport_group` segment reads per barrier; sender flushes once per group (blackhole-fabric rule 3) |
| `cb_act_resident` / `cb_weight_resident` | none (capacity = one full K pass) | — | residency, not overlap |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Fabric max payload (`ttnn.get_tt_fabric_max_payload_size_bytes()`, per config) | `seg_tiles` (one segment = one packet) | `seg_tiles = min(max_payload // 2048, blk_n_tiles)` | packet larger than a router slot → hang / corrupted landing |
| Resident replay (CB ring wrap: pointers reset only on an exact `fifo_limit` hit) | `k_block_tiles` | must divide `Kt` (`k_block_tiles` chosen as a divisor) so `num_k_blocks·k_block_tiles·core_x_tiles` = capacity exactly | replayed pages misaligned by one K-block on the second scatter block → silently wrong partials |
| `cb_partial_handoff` slot addressing by transport readers | `handoff_depth`, per-core block | slot of compute-order index `idx` is `idx mod handoff_depth` at byte offset `(idx mod handoff_depth)·core_m_tiles·core_n_tiles·2048`; every block pushes **exactly** `core_m_tiles·core_n_tiles` pages (nominal counts on ragged cores) | ragged push count shifts the ring → transport reads another block's tiles |
| `OutputCBLayout::TileRowMajor` (matmul_block_helpers.hpp:47) | hand-off tile order | required: tile `(r, c)` of the per-core block at page `r·core_n_tiles + c` so a segment piece is one contiguous NoC read | SubblockMajor scatters a row → segment gathers become per-tile and wrong |
| DEST capacity | subblock (helper-owned), not the block | `matmul_block` derives subblocks from `core_m_tiles × core_n_tiles` and `fp32_dest_acc_en` (`DEST_AUTO_LIMIT`, dest_helpers.hpp:103) | — (helper-owned) |
| Arrival-increment cadence (blackhole-fabric rule 4) | none on extent; counter = `ceil(segments/inc_every)` | `inc_every = 8`; increment on every 8th segment of a stream and on its last | — |
| Router channel: one producer per (link, direction) (blackhole-fabric §1) | transport cores per link | exactly one port core per (link, direction) | two producers on one channel → undefined / hang |
| Transport-row capacity | `4·L` transport cores | transport rows = `ceil(4·L / grid_cols)` rows adjacent to the Ethernet row; compute rectangle = remaining rows | transport core inside the compute rectangle breaks the rectangular multicast |
| L1 per compute core | `core_m_tiles`, `core_n_tiles`, `k_block_tiles`, residency | regime predicate `RESIDENT` (below) picks R1/R2; the per-core block is bounded by `CORE_BLOCK_MAX = 64` tiles (`core_m_tiles·core_n_tiles`; every INPUTS shape on G ≥ 2 with a ≥ 8×8 compute rectangle is ≤ 48); a plan exceeding it raises `ValueError` naming R4 (waves) — no INPUTS shape reaches it | OOM at program creation |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum (minimum: A, W cross DRAM once; output crosses DRAM once; each chip's busiest link direction carries `(G−1)·B` on a line) | What a bigger block buys |
|--------|--------|-----------|-------|----------------------------------------------------------------------------|--------------------------|
| **R1 — resident invariant operand, DRAM relay scratch, Linear** | **built** | `RESIDENT`: a divisor `k ≥ min(K_MIN_RESIDENT=4, Kt)` of `Kt` exists with `core_x_tiles·Kt·tile_bytes(X) + operand_depth·k·core_y_tiles·tile_bytes(Y) + core_m_tiles·core_n_tiles·acc_bytes ≤ L1_CB_BUDGET` (residency never forces a degenerate K-block), where X = block-invariant operand (A for `-1`, W for `-2`), Y = the other, `L1_CB_BUDGET` = per-core L1 available to CBs (device query) minus the `handoff_l1` shard | scatter block `(blk_m_tiles, blk_n_tiles)` per compute pass; per core `core_m_tiles × core_n_tiles × Kt` in K-blocks of `k_block_tiles` | A, W: minimum (once). Output: once. **Above minimum:** relay scratch: each chip writes and reads back the `≤ G−1` blocks it receives (FOCUS: +13.8 MB DRAM) — structural to the DRAM landing (R5 removes it). Links: `(G−1)·B` busiest (minimum for a line). NoC: every partial tile read once by one transport core (9.2 MB FOCUS); W multicast to `m_lines` receivers, A to `n_lines` | per scatter block: one fill/drain of the transport pipeline, one ready/ack round (`core count` incs + `L` or `2L` multicast acks); per K-block: one multicast handshake per operand + one CB handshake; matmul init once per kernel (formats constant); bigger `k_block_tiles` → fewer multicast rounds |
| **R2 — streamed operands, DRAM relay scratch, Linear** | **built** | `not RESIDENT` (e.g. Kimi K3 dense down, `Kt = 264`) | as R1, the invariant operand streamed per K-block like the other | invariant operand crosses DRAM **G times** (re-streamed per scatter block): A for `-1` (+`(G−1)·M·K·2` B), W for `-2` (+`(G−1)·K·N·w_bytes`). Everything else as R1 | as R1 |
| R3 — Ring (both directions, per-block virtual line) | **built (Refinement 1)**: per-entry `has_upstream` carried to the port reader and the add kernel (copy-through of the upstream-less first entry); arrival increments never deferred across a block (a ring has no line end to bottom out a deferred-increment wait). Original note: **deferred** — Phase 0 SUPPORTED is Linear (spec); positive reason it needs no new structure: the ring schedule (`_schedule_mmrs(..., ring=True)`, Work Distribution) emits the same per-port block lists (`block, slot, has_upstream`) and neighbour coordinates the kernels already take as runtime args; a block's reduction chain is a line centred on its owner, so ports/relays/finals/counters/scratch slots are unchanged. Adds: wrap-hop link discovery and the torus-config checks (`ValueError`). | `topology == Ring` and the fabric config wraps `cluster_axis` and the wrap hop has links | as R1/R2 | busiest link direction carries `ceil((G−1)/2)` blocks instead of `G−1`; balanced variant (far block split by bank-set halves, `fabric_all_gather` `balance`) → `(G−1)/2` | as R1 |
| R4 — sub-block sends (`waves_per_block > 1`) | **deferred** — positive reason: the FUSED ROOF used for every target already budgets a one-block fill (FOCUS 17.4 of 88.3 us; GLM 29.7 of 148.6; MiMo 31.7 of 161.4), so the 75% targets are reachable at `waves_per_block = 1`; reachable because the block walk already takes a wave index (Phase 0 trip count 1) and segments are row-ordered, so a row-wave is a contiguous segment range of every scratch slot and of the output | `waves_per_block > 1` | a scatter block split into `waves_per_block` row-waves, each its own K pass on the same grid | for `-1` with A resident: unchanged; the streamed operand (W) must be re-streamed per wave unless the W slice `Kt·core_n_tiles` is also resident | shrinks the fill term toward `T_block / waves_per_block`; adds one ready/ack round per wave |
| R5 — L1 relay landing | **deferred** — positive reason: in all three LOOSE_CASES the DRAM-scratch fused DRAM bound stays below the binding term (FOCUS 66.9 < links 70.9 us; GLM 89.3 < compute 118.9; MiMo 91.1 < links 129.7), so R1 does not move the roofline; reachable because the landing address is a per-stream (base, page mapping) runtime arg and the arrival counters already gate every read | when L1 landing rings fit on the receiving transport cores | arrival segments land in an L1 ring of the consuming port/final core; credits return over the **reverse** direction's port (the only legal producer on that router channel) | removes `2·(G−1)·B` DRAM per chip (FOCUS −13.8 MB → fused DRAM ≈ 36 us); adds one credit packet per `inc_every` segments on the reverse direction | — |
| R6 — K-split across cores, combine folded into the transport add | **deferred** — positive reason: the built independent split covers every shape (all INPUTS fill ≥ 80 of ~99 compute cores per block); reachable because the port/final add already sums 2-3 inputs per segment and can take one more | `occupancy(R1) < threshold` (not reached by any INPUTS shape) | per core `core_m_tiles × core_n_tiles × Kt/2` | doubles hand-off NoC bytes into the transport cores (FOCUS 9.2 → 18.4 MB), whose ingress already runs at link rate | uses idle cores |
| R7 — relay add on the compute cores | **deferred** — superseded for every shape by the port-side add (R1): a compute-side add puts a remote-chip wait inside the matmul loop (stall-shadow: the wait is a whole block transfer, ~23 us FOCUS); reachable because the hand-off slot is already addressable and an add phase over a parked slot is additive | — | — | moves `(G−1)·B` of fp32 adds off the transport cores | — |
| R8 — unfused (matmul → DRAM P → reduce-scatter) | **rejected** — forbidden (materializes P in DRAM before the first send) and superseded by R1 everywhere | — | — | +2·M·N·2 B DRAM, no overlap | — |
| R9 — compute cores send over Fabric directly | **rejected** — one producer per router channel (blackhole-fabric §1); superseded by ports gathering from compute L1 | — | — | — | — |
| R10 — gather tile-by-tile in output-page order (reference chunk = pages b, b+B, ...) | **rejected** — superseded by segment pages: a chunk of the reference's tile-interleaved scratch spans up to `seg_tiles` different compute cores (one NoC read per tile); the segment-paged scratch makes one packet = one or two contiguous NoC reads | — | — | same bytes, `seg_tiles×` the NoC read count | — |

**Regime-selection function (pinned, host):**

```
mm_regime = "R1" if RESIDENT(core_m_tiles, core_n_tiles, Kt, k_block_tiles, dtypes, scatter_dim) else "R2"
transport  = "line"  (Phase 0; "ring" only when topology == Ring, refinement)
```

The predicate is evaluated **after** the grid factorization and `k_block_tiles` are fixed, with
`k_block_tiles` re-chosen for R2 (both operands streamed) when R1 fails. Regime-pinned tests are
required: one INPUTS shape that lands in R1 (FOCUS `640×2048×7168`) and one that lands in R2
(`640×8448×7168`, K3 dense down) must both run in the golden suite on every cluster.

### Traffic ranking

Per chip, FOCUS case (`M640 K2048 N7168`, G=4 line, L=2, bf8b W), bytes per tier.

| Rank | Candidate split | DRAM | Cross-core NoC | Links (busiest dir.) | Verdict |
|------|-----------------|------|----------------|----------------------|---------|
| 1 | R5: R1 + L1 landing | 15.6 W + 2.6 A + 2.3 out = **20.5 MB** | hand-off 9.2 MB + arrivals local | 6.9 MB | cheapest; deferred (R5 row) |
| 2 | **R1: scatter blocks in time, m×n across grid, invariant operand resident, DRAM scratch** | 20.5 + 13.8 scratch = **34.3 MB** | hand-off 9.2 MB; W mcast 15.6 MB injected; A mcast 2.6 MB injected once | 6.9 MB | **built** |
| 3 | R2: as R1, invariant operand re-streamed | 34.3 + 3·2.6 = 42.1 MB | + A mcast ×4 | 6.9 MB | built for shapes where R1 does not fit |
| 4 | R6: K split across cores | 34.3 MB | hand-off 18.4 MB | 6.9 MB | deferred |
| 5 | scatter blocks across cores (all blocks concurrently) | 34.3 MB | 9.2 MB | 6.9 MB | rejected as a split: same bytes, **no overlap** (every block finishes at T) — fails the overlap requirement |
| 6 | R8 unfused | 34.3 + 18.4 MB | — | 6.9 MB | rejected |

Ranking by traffic picks rank 1; it is deferred with the positive reason in R5. Among built
regimes R1 is minimal at every tier except the scratch round trip.

### Block schedule

Logical schedule per chip (reader, compute, writer and transport kernels run asynchronously; adjacent blocks pipeline):

```cpp
// compute cores (every core of the compute rectangle)
load_resident_operand();                                  // R1 only: once per call (mcast along the line)
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core /* = G */; ++block_idx) {
    stream_operand_block(block_idx);                      // K-blocks of the streamed operand(s), mcast
    matmul_block(block_idx);                              // core_m_tiles x core_n_tiles x Kt, packer-L1 K-accumulation
    handoff_block(block_idx);                             // partial parked in cb_partial_handoff slot, ready -> consumers
    release_block(block_idx - handoff_depth + 1);         // ack from consumers -> pop slot
}
// port core (link l, direction d), per block in its list (farthest first)
for (uint32_t block_idx = 0; block_idx < num_blocks_this_port; ++block_idx) {
    gather_partial_block(block_idx);                      // my bank set's segments from compute-core L1
    load_arrival_block(block_idx);                        // relay only: same segments from relay_scratch (arrival-gated)
    relay_add_block(block_idx);                           // relay only: partial + arrival (fp32 DEST) -> bf16
    send_block(block_idx);                                // fabric: into the downstream chip's relay_scratch slot
    ack_block(block_idx);                                 // multicast ack to the compute rectangle
}
// final core (link l, half h): the own block p
gather_partial_block(p); load_arrival_block(p /* fwd slot p, bwd slot G */);
final_add_block(p);                                      // own + fwd + bwd (fp32 DEST) -> bf16
store_output_block(p); ack_block(p);
```

| Block operation | Block shape (in extent knobs) | Resident across it | Intended frequency of fixed costs |
|-----------------|-------------------------------|--------------------|-----------------------------------|
| `load_resident_operand` | `core_x_tiles × Kt` (X = A for `-1`, W for `-2`) | stays resident for all G blocks | once per call: one DRAM read + mcast per line injector |
| `stream_operand_block` | K-blocks `k_block_tiles × core_y_tiles` (and, R2, `core_x_tiles × k_block_tiles`) | — | per K-block: one injector DRAM read (bank-round-robin), one mcast + semaphore handshake per operand; resident operand: one credit-only replay `reserve/push` per K-block (no data movement) |
| `matmul_block` | `core_m_tiles × core_n_tiles`, `Kt/k_block_tiles` K-blocks | partial accumulator in `cb_partial_accum` (packer L1 acc) | `compute_kernel_hw_startup<SrcOrder::Reverse>` once per kernel; matmul init once per kernel (formats never change on compute cores); no per-tile init |
| `handoff_block` | `core_m_tiles × core_n_tiles` bf16 tiles in slot `block_idx mod handoff_depth` | slot stays fronted until acked | per block: one non-blocking front check; one `noc_semaphore_inc` per consumer (`L` ports or `2L` finals) |
| `release_block` | one slot | — | per block: wait `sem_block_ack ≥ cumulative expected`, one `cb_pop_front` |
| `gather_partial_block` | the port's segments of block j (bank set `l`, stride `L`), `seg_tiles` each | — | per block: one wait on `sem_block_ready ≥ compute_cores·(k+1)`; per segment: 1-2 contiguous NoC reads (pieces of one block row on ≤ 2 compute cores, `ceil(seg_tiles/core_n_tiles)+1` worst case); one read barrier per `xport_group` segments |
| `load_arrival_block` | same segments, from `relay_scratch` slot | — | per segment: one DRAM read of one page, gated by `noc_semaphore_wait_min(arrival, idx/inc_every + 1)` |
| `relay_add_block` / `final_add_block` | `xport_group·seg_tiles` tiles per handshake | — | eltwise add init once per kernel (2-input) / once per DEST batch for the 3-input dest-reuse add (as the reference) |
| `send_block` | segments of block j | packet-header ring of `xport_group` headers | per segment: one packet (`seg_tiles·2048` B); flush once per `xport_group`; fused write+inc every `inc_every` segments and on each stream's last |
| `store_output_block` | own-block segments of this final's bank set | — | per segment: `valid` tile writes to consecutive output pages; one write barrier per `xport_group` |
| `ack_block` | — | — | per block per consumer: one `noc_semaphore_inc_multicast` over the compute rectangle |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| L1 overlap — `handoff_depth = 2` | link-bound shapes (shared-expert down, MiMo) may need more slack so a temporary arrival stall does not back-pressure the matmul | `handoff_depth = 3` (L1 permitting) |
| L2 overlap — `k_block_tiles` coarsest-that-fits | a large K-block makes the line injector serialize one big read+mcast before compute can start a block | half the K-block with `operand_depth = 3` |
| L3 grid synchronization — one scatter block per pass over ≤ 99 cores | per block only `blk_m_tiles·blk_n_tiles` tiles (FOCUS 1120) → 14 tiles/core; per-block ready/ack rounds + mcast handshakes are paid G times | `blocks_in_flight = 2` (two blocks per pass, fewer fixed rounds, double fill) or the other grid orientation |
| L4 port reader issue rate | a port reader issues ~1-2 NoC reads + 1 DRAM read per 14 KiB segment at link rate; at small `core_n_tiles` the gather becomes issue-bound (`split_reader` catalog entry) | split the gather between the port's NCRISC and its compute-idle path, or larger `seg_tiles` where the payload allows |
| L5 transport placement | **Built (Refinement 3)**: per chip, each port sits on the transport-row core with the fewest NoC1 hops to its (direction, link) Ethernet core. The channel is the first word of `setup_fabric_connection`'s RT args, and the physical Ethernet / worker-column maps come from the cluster + SoC descriptors, so there is no probe dispatch. Finals take the remaining row cores. Each sender addresses its *peer chip's* placement. Measured at 14 KiB packets: FOCUS 208 → 167 us, steady-state send ≈ 36 GB/s per link (box link ceiling) | `XPORT_PLACEMENT="simple"` restores the fixed chip-independent layout (A/B knob) |
| L6 NoC assignment on compute cores | the W injector on BRISC reads DRAM on NoC1 (`noc_placement`: reads on NoC0 are 2.5-4.8× better for spread lines) | swap operand roles between NCRISC/BRISC per orientation |
| L7 fp32 accumulation under `fp32_dest_acc_en=False` | the floor allows fp32; it halves DEST subblocks and doubles `cb_partial_accum` | fp32 DEST on the FOCUS case; keep only if within noise |
| L8 compute config of transport adds | fp32 DEST for 2-3 input adds is required; packing relay sums bf16 is allowed | — (requirement, no alternative) |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| A, W in DRAM | A bf16 tiles (2048 B), W bf16 / Bfp8_b (1088 B) tiles | interleaved, read by line injectors with `TensorAccessor`, bank-round-robin runs | each operand crosses DRAM once (R1) |
| injector → line cores | same | `mcast_pipe` `SenderPipe`/`ReceiverPipe` (dataflow); host `Mcast1D` families: one per m-line (A), one per n-line (W) at disjoint semaphore ids (`mcast_topology` entry) | sender-in-line self-excludes; the 2D work split needs **two 1-D** mcast families |
| resident operand replay | same | CB credit-only `cb_reserve_back` + `cb_push_back` of the same pages per K-block (capacity = one K pass exactly) | zero data movement after block 0 |
| matmul | in0 bf16, in1 bf16/Bfp8_b → DEST (user fidelity / `fp32_dest_acc_en`) | `matmul_block` helper, packer L1 accumulation in `cb_partial_accum` | output packed bf16 into `cb_partial_handoff`, TileRowMajor |
| compute L1 → transport L1 | bf16 tiles | **pull**: transport readers `noc_async_read` from `handoff_l1` base (RT arg) + slot + `(r·core_n_tiles + c)·2048`; gated by `sem_block_ready`; released by multicast `sem_block_ack` | many-to-few; the compute core's NCRISC is the CB consumer (front check, ready incs, ack wait, pop) |
| relay arrival | bf16 segment pages | `relay_scratch` DRAM page = segment; `noc_semaphore_wait_min` on `sem_arrival_{fwd,bwd}` | as the reference with segment pages |
| relay add / final add | fp32 DEST → bf16 | `eltwise_chain` BinaryFpu Add (+ DestReuseBinary for the third input) | transport compute config: `fp32_dest_acc_en=True` |
| fabric | bf16 segment = one packet | raw fabric worker API copied from the reference port sender (route by 2D node or 1D hop count under `ROUTING_MODE`), `send_current_slot_non_blocking`, header ring, fused write+inc every `inc_every` | into the downstream chip's `relay_scratch` slot (`j`, or `G` for the receiver's own block arriving backward) |
| output | bf16 tiles | final writer, `TensorAccessor` page `row·blk_n_tiles + col` | DRAM interleaved |

**Ready fence and counter re-arm** are the reference's, unchanged: each port's sender first sends one
ready increment to the peer port that writes into this chip and waits for its own peer's ready before
the first data packet; every consumer of `sem_arrival_*`, `sem_block_ready`, `sem_block_ack` subtracts
exactly what it consumed before exiting, so every semaphore is 0 between calls (no host sync, trace-safe).

**Hand-off contract (compute core ↔ transport core), exact:**

| Step | Who | Action |
|------|-----|--------|
| 1 | compute (TRISC) | packs block `idx` into `cb_partial_handoff` (`core_m_tiles·core_n_tiles` pages, TileRowMajor, nominal count on ragged cores) |
| 2 | compute core NCRISC | non-blocking poll (`cb_pages_available_at_front`, dataflow_api.h:443) between K-block transfers; when block `idx` is fronted: `noc_semaphore_inc(consumer.sem_block_ready, 1)` for each consumer core of block `idx` (host list) |
| 3 | transport reader | waits `sem_block_ready ≥ compute_cores·(k+1)` for its k-th block; reads its segments' pieces from every owning compute core |
| 4 | transport reader | after its last piece read of block `idx` has completed (read barrier): `noc_semaphore_inc_multicast(compute rectangle, sem_block_ack, 1)` |
| 5 | compute core NCRISC | when `sem_block_ack ≥ Σ consumers(blocks ≤ idx)`: `cb_pop_front(cb_partial_handoff, core_m_tiles·core_n_tiles)` |

**TARGET axes against this structure.** `dtype=bfloat8_b` (activation) and `weight_dtype=bfloat4_b`:
knob-turns (operand CB formats + tile sizes in the L1 predicate; partials stay bf16). `topology=Ring`:
host schedule change (R3) + validation. `num_links`: knob-turn (number of ports/finals and bank-set
stride). `scatter_dim`, `cluster_axis`: already one code path (block origin, groups).

**Structural impossibilities:** none beyond `feature_spec.py`'s (empty) `INVALID`.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | compute: one per-core block `core_m_tiles × core_n_tiles` of one scatter block (full K); transport: one segment (`seg_tiles` tiles of one block row) |
| Grid | transport rows: the `ceil(4L / grid_cols)` grid rows adjacent to the Ethernet row (Blackhole: logical row 0 upward); compute rectangle: all remaining rows × all columns, factorized per the Blocking Model (`m_lines × n_lines`, idle lines dropped). The grid and the transport-row count are host-derived from `compute_with_storage_grid_size()` — never inlined |
| Per-core work | compute core: G per-core blocks in `compute_order`; port `(d, l)`: segments of bank set `{l, l+L, ...}` (scratch page index mod `num_banks`) of each block in its list; final `(l, h)`: own-block segments of bank set `{l + h·L, stride 2L}` |
| Remainder | `m_lines = ceil(blk_m_tiles / core_m_tiles)`, `n_lines = ceil(blk_n_tiles / core_n_tiles)`; last line ragged: it keeps the **nominal** CB counts and its injector reads only the valid rows/cols (padding pages are never sent or stored: transport moves `valid` tiles of each piece, finals store only valid tiles). Segments: `segs_per_row = ceil(blk_n_tiles / seg_tiles)`, the last segment of a row has `blk_n_tiles − (segs_per_row−1)·seg_tiles` valid tiles. All tile counts are `ceil` and per-image (the leading dims are 1) |

**Segment indexing (pinned).** Within a block: `seg = row·segs_per_row + s`, `row ∈ [0, blk_m_tiles)`,
tiles `[s·seg_tiles, min((s+1)·seg_tiles, blk_n_tiles))` of that row. Scratch page of block slot `q`:
`q·segs_per_block + seg`, `segs_per_block = blk_m_tiles·segs_per_row`. Owner of tile `(row, col)`: m-line
`row / core_m_tiles`, n-line `col / core_n_tiles`, local page `(row mod core_m_tiles)·core_n_tiles + col mod core_n_tiles`.
Per-port bank sets and `_chunks_per_shard`-style counts are computed on the host and passed as
runtime args (blackhole-fabric rule 5: no on-device counting loops).

**Schedule function (pinned; `_schedule_mmrs(p, G, ring)` on the host):**

| Topology | fwd list (toward p+1, in send order) | bwd list (toward p−1) | `has_upstream` | `compute_order` |
|----------|--------------------------------------|------------------------|----------------|-----------------|
| Linear | `G−1, G−2, …, p+1` | `0, 1, …, p−1` | fwd: `p > 0`; bwd: `p < G−1` | interleave(fwd, bwd) one-by-one starting with fwd, then `p` |
| Ring (R3) | `p+hf, …, p+1` (mod G), `hf = ceil((G−1)/2)` | `p−hb, …, p−1` (mod G), `hb = G−1−hf` | every entry but the first of each list | same rule |

Landing slot at the receiver: forward-sent block `j` → slot `j`; backward-sent block `j` → slot `j`,
except the receiver's own block → slot `G` (reference `_blocks`). The last entry of each list is the
downstream chip's own block and is counted on the downstream **final** core owning the segment's bank
(`sem_arrival_fwd` / `sem_arrival_bwd`); the others on the downstream port of the same (direction, link).

**Validation and caller errors (entry point order).**

| # | Check | Raises |
|---|-------|--------|
| 1 | `validate()` — first line: axes `dtype, weight_dtype, layout, alignment, cluster_axis, scatter_dim, topology, num_links` (num_links checked only when explicit; `None` = default) against SUPPORTED, then EXCLUSIONS | `UnsupportedAxisValue` / `ExcludedCell` |
| 2 | A rank 2-4, leading dims 1, `A.K == W.K`, W rank 2, TILE + DRAM interleaved inputs, `memory_config` DRAM interleaved or None | `ValueError` |
| 3 | `G = mesh_shape[cluster_axis] ≥ 2`; scattered extent `% (32·G) == 0` | `ValueError` |
| 4 | every hop of every group has `≥ num_links` links (`ttnn.get_forwarding_link_indices`); `None` → min over hops | `ValueError` |
| 5 | Ring (refinement): fabric config wraps `cluster_axis` (TORUS_Y: 0, TORUS_X: 1, TORUS_XY/1D_RING: both) and the wrap hop has links | `ValueError` |

INPUT_TAGGERS: exactly `alignment(inputs, axes)` → `"tile_aligned"` iff `M, K, N % 32 == 0`
(`M = inputs[0][-2]`, `K = inputs[0][-1]`, `N = inputs[1][-1]`), else `"non_aligned"`.
SUPPORTED (Phase 0): dtype `[bfloat16]`, weight_dtype `[bfloat16, bfloat8_b]`, layout `[TILE]`,
alignment `["tile_aligned"]`, cluster_axis `[0, 1]`, scatter_dim `[-1, -2]`, topology `[Linear]`,
num_links `[1, 2]`. EXCLUSIONS: `[]`.

**Caching (no per-call host sync).** Plan key `(mesh id, cluster_axis, topology, num_links, M, K, N,
scatter_dim, dtypes, fp32_dest_acc_en, math_fidelity)` → placement, factorization, schedule,
`relay_scratch`, `handoff_l1`, global semaphores (created once; one `ttnn.synchronize_device` at creation,
as the reference). A call builds the `MeshProgramDescriptor` from the cached plan with the current
tensor addresses only and issues **one** `ttnn.generic_op([A, W, relay_scratch, handoff_l1, output], desc)`.
Placement must not dispatch any probe program (one dispatch per invocation).

## Circular Buffers

Compute cores (`X` = block-invariant operand: A for `scatter_dim=-1`, W for `-2`; `Y` = the other; `core_x_tiles`/`core_y_tiles` = `core_m_tiles`/`core_n_tiles` accordingly):

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_act_operand` (in0) | 0 | A tile (2048) | R1, X=A: `core_m_tiles·Kt` (resident, replay); otherwise `operand_depth·core_m_tiles·k_block_tiles` | spans m (core) and, when resident, all K; streams over n, scatter_block | Float16_b | reader (NCRISC) | compute | call |
| `cb_weight_operand` (in1) | 1 | W tile (2048 / 1088) | R1, X=W: `Kt·core_n_tiles` (resident, replay); otherwise `operand_depth·k_block_tiles·core_n_tiles` | spans n (core) and K-block (or all K resident); streams over m, scatter_block | Float16_b / Bfp8_b | writer (BRISC) | compute | call |
| `cb_partial_accum` | 2 | 4096 if `fp32_dest_acc_en` else 2048 | `core_m_tiles·core_n_tiles` | spans m, n (core); streams over K (accumulated in place by the packer), scatter_block | Float32 if `fp32_dest_acc_en` else Float16_b | compute | compute | one matmul_block |
| `cb_partial_handoff` | 3 | 2048 | `handoff_depth·core_m_tiles·core_n_tiles` — backed by `handoff_l1` (globally allocated) | spans m, n (core) × depth; streams over scatter_block | Float16_b | compute | reader (NCRISC) | call |

Transport cores (ports and finals; `seg_bytes = seg_tiles·2048`, `xport_group` per the depth knob):

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_xport_partial` | 0 | 2048 | `2·xport_group·seg_tiles` | spans segment × group; streams over blocks | Float16_b | reader | compute (relay/final); unused on a line-end port (reader writes `cb_xport_sum` directly) | call |
| `cb_xport_arrival_a` | 1 | 2048 | `2·xport_group·seg_tiles` | as above | Float16_b | reader | compute | call (relay ports, finals with a forward upstream) |
| `cb_xport_arrival_b` | 2 | 2048 | `2·xport_group·seg_tiles` | as above | Float16_b | reader | compute | call (finals only: the backward arrival) |
| `cb_xport_sum` | 16 | 2048 | `2·xport_group·seg_tiles` | as above | Float16_b | compute (relay/final) **or** reader (line-end port: no add) — one producer per port, fixed by the port's role | sender (port) / writer (final) | call |

`cb_xport_sum` has exactly one producer on any given core: the host compiles the line-end port's reader
to target it and that port's compute kernel to do nothing (reference `_PORT_READER` `cb_own = relay ? 0 : 16`).

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_resident_operand` | `core_x_tiles × Kt` | `mcast_pipe` | DRAM X slice (injector) | `cb_act_operand` or `cb_weight_operand`, full pass | resident; replayed by credit-only push per K-block |
| 2 | `stream_operand_block` | `k_block_tiles × core_y_tiles` per K-block | `mcast_pipe` | DRAM Y slice (injector) | `cb_*_operand`, `k_block_tiles·core_y_tiles` per push | consumed per K-block |
| 3 | `matmul_block` | `core_m_tiles × core_n_tiles`, `Kt/k_block_tiles` K-blocks | `matmul_block` | `cb_act_operand`, `cb_weight_operand` (per K-block) | `cb_partial_accum` (L1 acc), `cb_partial_handoff` (`core_m_tiles·core_n_tiles`, TileRowMajor) | block fronted in its hand-off slot |
| 4 | `handoff_block` / `release_block` | one hand-off slot | raw (semaphores) | `cb_partial_handoff` fronted | — | popped after acks |
| 5 | `gather_partial_block` | port's segments of block j | raw (NoC reads; walk adapted from `_CHUNK_WALK`) | remote `handoff_l1` | `cb_xport_partial` (or `cb_xport_sum` on a line end), `xport_group·seg_tiles` per push | — |
| 6 | `load_arrival_block` | same segments | raw (`TensorAccessor` + `noc_semaphore_wait_min`) | `relay_scratch` | `cb_xport_arrival_a` / `_b` | — |
| 7 | `relay_add_block` | `xport_group·seg_tiles` tiles | `eltwise_chain` | `cb_xport_partial`, `cb_xport_arrival_a` | `cb_xport_sum` | — |
| 8 | `send_block` | segments | raw fabric API | `cb_xport_sum` | downstream `relay_scratch` | popped after flush |
| 9 | `final_add_block` | `xport_group·seg_tiles` tiles | `eltwise_chain` | `cb_xport_partial`, `cb_xport_arrival_a` (fwd, if p>0), `cb_xport_arrival_b` (bwd, if p<G−1) | `cb_xport_sum` | — |
| 10 | `store_output_block` | final's own-block segments | raw (`TensorAccessor`) | `cb_xport_sum` | output DRAM | — |
| 11 | `ack_block` | — | raw | — | `sem_block_ack` on the compute rectangle | — |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| boot (compute cores) | helper prerequisite | `compute_kernel_hw_startup<SrcOrder::Reverse>(in0, in1, out)` | ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp:100 | once at top of kernel_main | — | — | — |
| `matmul_block` | helper | `compute_kernel_lib::matmul_block` | ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp:339-372 | `transpose=false, packer_l1_acc=true, LastBlockTarget::Out (:62), OutputCBLayout::TileRowMajor (:47)`; `MatmulBlockShape::of(in0_num_subblocks, in1_num_subblocks, out_subblock_h, out_subblock_w, k_block_tiles, Kt/k_block_tiles)` (:139) | `cb_act_operand`, `cb_weight_operand` | `cb_partial_handoff`, interm `cb_partial_accum` | block = `core_m_tiles × core_n_tiles` (the product of the subblock counts and sizes), `k_block_tiles`, `num_k_blocks`; subblock factors derived by the tuner from block size + `fp32_dest_acc_en` (not design knobs); `batch` = 1 (per-block phase work between calls) |
| relay add | helper | `eltwise_chain(IterationShape::tiles(n), BinaryFpu<Add, input(cb_xport_partial), input(cb_xport_arrival_a)>{}, PackTile<output(cb_xport_sum)>{})` | ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp:491 (BinaryFpu), :504 (PackTile) | `n = xport_group·seg_tiles` (ragged last group: actual count) | `cb_xport_partial`, `cb_xport_arrival_a` | `cb_xport_sum` | `n` |
| final add (3 inputs) | helper | `eltwise_chain(..., BinaryFpu<Add, partial, arrival_a>, DestReuseBinary<Add, input(cb_xport_arrival_b), DEST_TO_SRCA>, PackTile<output(cb_xport_sum)>)` | chain.hpp:491, :499-501 (DestReuseBinary), :504 | 2-input form when only one arrival exists | `cb_xport_partial`, `cb_xport_arrival_a`, `cb_xport_arrival_b` | `cb_xport_sum` | `n` |
| operand multicast | helper | `SenderPipe::send` / `ReceiverPipe::receive`, `McastArgs` | ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_pipe.hpp:55, :74, :126, :138; mcast_args.hpp:15 | one pipe per operand line; host `Mcast1D` (mcast_host.hpp:303) per family, disjoint semaphore ids | DRAM → injector L1 | line receivers' `cb_*_operand` | payload = one K-block |
| `handoff_block` ready | raw_api | `noc_semaphore_inc` | tt_metal/hw/inc/api/dataflow/dataflow_api.h:2279 | 1 per consumer | — | — | — |
| `ack_block` | raw_api | `noc_semaphore_inc_multicast` | dataflow_api.h:2324 | compute rectangle | — | — | — |
| front poll | raw_api | `cb_pages_available_at_front` | dataflow_api.h:443 | `core_m_tiles·core_n_tiles` | — | — | — |
| `gather_partial_block`, `load_arrival_block`, `store_output_block` | raw_api | `noc_async_read` / `noc_async_write`, `TensorAccessor`, `noc_semaphore_wait_min` | reference port/final reader and writer: fabric_reduce_scatter/program_descriptor_with_inline_kernels.py:46-189, :251-302 | segment walk (bank set, stride) | — | — | `seg_tiles`, `xport_group` |
| `send_block` | raw_api | `WorkerToFabricEdmSender`, `fabric_set_unicast_route` (2D) / `<false>` hop count (1D), `to_noc_unicast_write`, `to_noc_fused_unicast_write_atomic_inc`, `send_current_slot_non_blocking` | fabric_reduce_scatter/program_descriptor_with_inline_kernels.py:305-446; references/blackhole-fabric.md §5 | header ring of `xport_group` | `cb_xport_sum` | remote scratch | `seg_tiles`, `inc_every` |

**Helpers considered and rejected (raw_api rows):**

| Raw block op | Candidate helper | Mismatch (file:line) | Reason |
|--------------|------------------|----------------------|--------|
| `send_block` | `ccl_helpers_dataflow.hpp` `FabricStreamSender` / `FabricDuplexSender` | ttnn/cpp/ttnn/kernel_lib/ccl/ccl_helpers_dataflow.hpp:12 ("1-D route programming") | the op must run under the FABRIC_2D family, where routes are set by destination fabric node (`fabric_set_unicast_route(hdr, chip, mesh)`); the helper programs 1-D hop routes only. Gap to close: a 2-D unicast route type on `open(route)`. |
| semaphore hand-off (`handoff_block`, `ack_block`) | `mcast_pipe` `SenderPipe::send_signal` / `ReceiverPipe::receive_signal` | mcast_pipe.hpp:55-74, :126-138 | the pipe is one sender → one receiver rectangle with a per-round handshake; the hand-off is many producers (all compute cores) → a few consumers (ports/finals) with cumulative counters and a multicast release — a different topology. |
| `gather_partial_block` | `local_copy_helpers_dataflow.hpp` (`set_read_state` / `read_with_state`) | ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:7 ("Local L1 -> L1 copy helpers: self-aimed NoC reads") | the source endpoint is self-aimed (`local_addr()`); the gather reads **remote** compute cores' L1. Gap to close: a remote-source variant of the stateful read (it is the natural fix for lamp L4). |

## Broadcast Verification

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast Dim |
|-------|-----|-------------------|-------------------|---------------|
| relay add | Add | `cb_xport_partial`: All | `cb_xport_arrival_a`: All | None |
| final add | Add (+ dest-reuse Add) | `cb_xport_partial`: All | `cb_xport_arrival_a` / `_b`: All | None |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Hand-off ring offset drift | transport readers compute slot addresses from the compute-order index; a ragged core pushing fewer pages shifts every later block | nominal push counts on every core; capacity = `handoff_depth·core_m_tiles·core_n_tiles` exactly; mechanism-cap row |
| Resident replay misalignment | credit-only re-push relies on the CB ring wrapping exactly once per K pass | `k_block_tiles` divides `Kt`; capacity = one pass exactly |
| Compute NCRISC blocking on the hand-off | a blocking `cb_wait_front` / ack wait in the operand-streaming kernel stalls the next block's operand stream → compute idles | front check and ack check are non-blocking polls interleaved with K-block transfers; only slot reuse (`idx − handoff_depth`) may block |
| Deadlock between directions | a compute core blocks on a slot held by a block whose port waits on an upstream chain | `compute_order` interleaves fwd/bwd one-by-one so slot reuse at index i waits on a block of the same direction, whose chain bottoms out at a line end that needs only local compute; `handoff_depth ≥ 2` |
| Counter re-arm / stale scratch across calls | global semaphores persist; a neighbour's next call can write scratch before this chip's relay finished reading | reference ready fence (one ready per sending port per call) + every consumer subtracts what it consumed; regression test runs the op repeatedly bit-exact |
| Exactly-once contribution | a missed or doubled partial passes PCC on random data | `test_rank_identity`-style exact integer test (golden regression + acceptance) |
| fp32 cross-device add | user may pass `fp32_dest_acc_en=False` | transport add kernels hard-wired to `fp32_dest_acc_en=True`; only the matmul kernel honours the user's config |
| Fabric payload differs per config | `seg_tiles` derived from the live config's max payload | `seg_tiles = min(max_payload // 2048, blk_n_tiles)` per plan |
| Second dispatch on first call | `fabric_all_gather.plan()` with `placement="auto"` runs a probe program | placement is host-only (`placement="simple"`-style rules: transport row, ports in distinct columns); lamp L5 |
| Line ends | `p = 0` has no forward upstream, `p = G−1` no backward; a line-end port's reader feeds `cb_xport_sum` directly | `has_upstream` per list entry; host compiles the port role (relay / line end) |
| Transport row vs mcast rectangle | a transport core inside the compute rectangle breaks both mcast families | compute rectangle = rows not used by transport; all transport cores in the row(s) adjacent to the Ethernet row |
| Group size up to 8 (ring mock, Galaxy axis 0) | Linear on 8 devices: scratch `(G+1)` slots, lists up to 7 blocks | nothing is specialized on G; all lists/slots are host-derived |
