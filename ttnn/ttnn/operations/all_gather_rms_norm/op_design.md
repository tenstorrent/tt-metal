# Operation Design: all_gather_rms_norm

## Overview

| Field | Value |
|-------|-------|
| Classification | fused CCL: a Fabric 2D all-gather of the hidden dim plus an RMSNorm over the gathered row, in **one** `ttnn.generic_op` (one `MeshProgramDescriptor`) per call |
| Goal | Every device of a `cluster_axis` group starts with its `(..., M, W_local)` slice and ends with the same RMS-normalized `(..., M, W)`, `W = G·W_local`. The exchange starts at t = 0 with no data dependency, and the norm runs on the landed rows **while later rows are still in flight**. |
| Math | `full[q] = concat_p x[p]` along the last dim (p = group position). `out[q][..., m, w] = full[..., m, w] · rsqrt((1/W)·Σ_{w'<W} full[..., m, w']² + eps) · gamma[w]` for every member q. `W` is the true logical width, never a padded one. |
| Mode | Hybrid. Local half = the base op `rms_norm` (kernel-lib helpers + its raw pass-1 / finalize / gamma-row-broadcast pieces, ported). Cross-device half = the `fabric_all_gather` example's port kernels (ready fence, arrival counters, relay, balanced ring), adapted so remote slices land in the **consumers' L1** instead of DRAM. |
| References | `ttnn/ttnn/operations/base_ops/rms_norm/{op_design.md, changelog.md, rms_norm_program_descriptor.py, kernels/}`; `ttnn/ttnn/operations/examples/{master.md, fabric_all_gather/, fabric_ring_exchange/, fabric_reduce_scatter/}`; `.claude/references/{blocking-model.md, l1-footprint-discipline.md, blackhole-fabric.md, blackhole-ccl-kernels.md, precision_convention.md}`; `eval/golden_tests/all_gather_rms_norm/{feature_spec.py, roofline.py, helpers.py, topology.py}`; `ttnn/cpp/ttnn/kernel_lib/{eltwise/api/chain.hpp, reduce_helpers_compute.hpp, reduce_helpers_dataflow.hpp, l1_helpers.hpp, ccl/ccl_helpers_dataflow.hpp}` |

### The scheme in one paragraph

Every chip reads its own slice from DRAM **once**, straight into the L1 of the Tensix cores that will normalize those columns. The slices then travel between chips as raw input tiles. Each slice moves as a store-and-forward relay along a balanced ring (or a line), with the far slice split between the two directions. Each packet is written by the receiving router **directly into the L1 slot of the core that owns those output columns** on the destination chip, so no DRAM round trip happens between the gather and the norm. Because every chip receives every raw slice, every chip computes the full norm itself, and no partial sum ever crosses the fabric.

On each chip the output tile-rows are dealt **round-robin** to row groups. Each group has `k` cores, and each core owns a W range nested inside one slice. Rows travel in ascending order, so each core runs Σx², the in-chip combine, the finalize, the normalize and the output write **one tile-row at a time, as that row lands**. Only the last row's work is exposed after the last packet.

### Catalog / example entries → the decisions they set

| Source | Measured fact used | Decision it set |
|---|---|---|
| `fabric_all_gather` report | balanced ring beats line 1.25–1.8×; balance pays from ~64 tiles/chip; 2 links 0.53–0.80×; fit t = 8.4 µs + bytes/77.2 GB/s (ring, 1 link), 9.8 µs + bytes/42.1 GB/s (line) | TRANSPORT: balanced ring schedule, one port per (direction, link); `T_xfer_est` in the split rule |
| `fabric_all_gather` report | an increment every 2 chunks costs 1.14×, every 4 costs 1.01–1.03×, every 8 is the safe default; a 112 KiB port CB (4 chunks per batch) is best or within 5% everywhere | TRANSPORT: `item_min_chunks = 4` (one fused increment per item) and `port_cb_bytes = 112 KiB` |
| `fabric_ring_exchange` | the ready fence alone costs ~5 µs; direct 2-hop sends are 1.1–1.6 µs slower than relay (their fence waits on the 2-hop peer); a second link does not help payloads ≤ 32 KiB | TRANSPORT: relay rather than direct sends; the local half runs inside the fence (stall shadow) |
| `blackhole-fabric.md` rules 1, 4, 6, 7, 8, 10 | send full packets (≤ 14 KiB payload); fused write+inc stalls the receiving router; place each port under its Ethernet core; send on NoC1; keep other work off the sending core; one producer per (link, direction) | port placement, sender NoC, chunk sizing, the increment frequency, no local copies on the sender |
| base op `changelog.md` Ref 2/5, Perf 1/2 | symmetric unicast all-gather of faces-0/2 partials for k ≤ 8; per-row owner combine beyond (`K_SYM_MAX_INTERLEAVED = 8`); FPU `reduce_tile` row collapse; column-0 SFPU finalize; gamma consumed row-broadcast from compact faces-0/1 pages | in-chip combine and the compute pipeline (ported) |
| base op `rms_norm_program_descriptor.py:534-575` (`_occ_cost`, `_choose_k`) | `COMBINE_TILE_COST = 3`, `OWNER_MEMBER_COST = 0.4` x-tile equivalents; `OCC_TILES_PER_MEMBER = 4`; `TILIZED_WTILE_NS = 310` | the constants of the fused split rule (re-run on the fused grid, see Work Distribution) |
| catalog `shared_input_reuse` | broadcasting a resident window measured null for r ≤ 0.125 | gamma multicast stays `deferred` (r ≤ 1/32) |
| catalog `double_buffer`, `compute_block_size` | ~1.4 µs per extra helper pass; reads batched per barrier | one chain call per block; batched member reads |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| input_tensor | ttnn.Tensor (mesh) | yes | per device `(..., M, W_local)`, rank ≥ 2, TILE; identical shape / dtype / placement on every device | — | accessor CT args; address RT |
| cluster_axis | int (kw-only) | yes | 0 or 1; that mesh axis must have size ≥ 2 | — | host only (groups, positions) |
| gamma | ttnn.Tensor (mesh) or None | no | full-W weight, same on every device: TILE `(1, 1, 1, W)` or ROW_MAJOR sticks `(1, 1, ceil(W/32), 32)`; dtype float32 / bfloat16 | None | `has_gamma`, `gamma_is_sticks` CT; address RT |
| epsilon | float | no | finite, ≥ 0 | 1e-6 | RT (fp32 bits) |
| topology | `ccl.Topology` | no | Linear, Ring (a Ring of G = 2 runs as Linear, which uses the same links) | Linear | host only (schedule) |
| num_links | int or None | no | explicit: 1 ≤ L ≤ links discovered on every hop of every group; None = that discovered minimum | None | host only (ports per direction) |
| compute_kernel_config | ttnn.ComputeConfigDescriptor | no | `fp32_dest_acc_en` gated; `math_fidelity` / `math_approx_mode` accepted and honored | `default_compute_kernel_config()` (HiFi4, fp32_dest_acc_en=True, approx False) | compute KernelDescriptor `config` |
| memory_config | ttnn.MemoryConfig or None | no | INTERLEAVED (DRAM/L1) or L1 HEIGHT/WIDTH_SHARDED for the gathered shape | DRAM interleaved | host (output allocation); output accessor CT |
| output_tensor | ttnn.Tensor (mesh) or None | no | per device `(..., M, W)`, input dtype, TILE | None | address RT; replaces `memory_config` |

Entry-point contract:

| Item | Decision |
|------|----------|
| `default_compute_kernel_config()` | exported from `ttnn/operations/all_gather_rms_norm/__init__.py`; a factory (a fresh descriptor per call); `None` resolves through it and nowhere else (`eval/golden_tests/all_gather_rms_norm/axes.py` imports it) |
| Topology enum at import time | `from ttnn.operations.ccl import Topology` (`ttnn.Topology` is not bound while `ttnn.operations` imports op packages) |
| First line | `validate(...)` |
| Single dispatch | validate → derive geometry → `plan()` (groups, schedules, ports, worker set, split) → allocate the output (or take `output_tensor`) → build one `MeshProgramDescriptor` → **one** `ttnn.generic_op([input, (gamma), output], mesh_desc)` → return the output. No other ttnn op touches the tensors. No probe program (see Port placement). No `synchronize_device` per call. |
| One-time setup (per mesh, cached) | The `ready` global semaphore (initial 0) on the union of port cores. It must hold 0 on every chip before **any** chip's first program is enqueued: a blocking initial write, or one synchronize at creation (the example's `_SEM_CACHE` path). Done on the first call per mesh, never inside a trace. |
| Validation errors (`ValueError`) | input rank < 2. gamma shape not `(1,1,1,W)` for TILE or `(1,1,ceil(W/32),32)` for ROW_MAJOR. `output_tensor` per-device shape / dtype / layout ≠ the output's. `cluster_axis` size 1. num_links ≤ 0 or above the links discovered on a hop. Ring along an axis with no wrap link (`ttnn.get_forwarding_link_indices(last, first)` empty, or the active config does not wrap that axis: TORUS_X wraps axis 1, TORUS_Y axis 0, TORUS_XY both). Not enough cores for the 2·L ports per chip plus ≥ 1 worker. |
| Capacity error (`RuntimeError`) | no admissible split fits L1 (see Mechanism caps). No INPUTS shape at G ≤ 8 reaches it (l1_ledger.md). |
| Support gate (`UnsupportedAxisValue` / `ExcludedCell`) | SUPPORTED per axis, then EXCLUSIONS. Axis values are read from the call: dtype, layout, `memory_config().memory_layout`, per-device shape (taggers), `fp32_dest_acc_en` of the resolved config, gamma dtype / layout or `"none"`, cluster_axis, topology, num_links. An explicit num_links is checked against SUPPORTED. None is resolved to the discovered count and **not refused** (the design is generic in L). Sharded input / output must be L1. |
| INPUT_TAGGERS | exactly two: `alignment` (`"w_non_aligned"` if `W_local % 32`; else `"h_non_aligned"` if `M % 32`; else `"tile_aligned"`) and `rank` (`len(shape)`), both on the per-device input shape `inputs[0]`. gamma_dtype / gamma_layout are **not** taggers: validate() reads `gamma.dtype` / `gamma.layout`, or `"none"`. |
| Recommended Phase 0 SUPPORTED (implementer owns it) | dtype {float32, bfloat16, bfloat8_b}; fp32_dest_acc_en {True, False}; layout {TILE}; alignment {tile_aligned, w_non_aligned, h_non_aligned}; rank {2, 3, 4}; gamma_mode {gamma, no_gamma}; gamma_dtype {float32, bfloat16, "none"}; gamma_layout {TILE, ROW_MAJOR, "none"}; memory_layout {INTERLEAVED, HEIGHT_SHARDED, WIDTH_SHARDED}; cluster_axis {0, 1}; topology {Linear, Ring}; num_links {1, 2} |
| Recommended EXCLUSIONS | `{dtype: float32, fp32_dest_acc_en: False}` (precision convention). ~~`{dtype: bfloat8_b, alignment: w_non_aligned}`~~ lifted by Refinement 2: the compose cannot shift datums inside packed bfp8 shared-exponent groups, so compute first unpacks each landed row into a bf16 staging row (`cb_x_stage`) and the reader composes from that (see the regime N row). The base op's `bfloat8_b × h_non_aligned` exclusion is **not** carried over: H padding stays row-local here, and the transport moves whole bfp8 tiles unchanged. |
| Index axes | none (the gathered / reduced dim is fixed at -1) |

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | per device `(..., M, W_local)`, rank 2–4; leading dims fold into the row axis |
| Dtype | float32, bfloat16, bfloat8_b |
| Layout | TILE |
| Memory | DRAM or L1 INTERLEAVED; L1 HEIGHT_SHARDED / WIDTH_SHARDED (native: consumed in place, see `sharded_zero_copy`; else read through a sharded `TensorAccessor`) |
| Gamma | full W on every device; TILE `(1,1,1,W)`: row 0 of tile t holds `gamma[32t .. 32t+31]`. ROW_MAJOR sticks `(1,1,ceil(W/32),32)`: page t is the 32-element stick `gamma[32t .. 32t+31]`, tail zero-padded. float32 / bfloat16, independent of the input dtype |

### Output

| Property | Value |
|----------|-------|
| Shape | per device `(..., M, G·W_local)`; group position p at columns `[p·W_local, (p+1)·W_local)`, dense (no padding between slices) |
| Dtype | input dtype |
| Layout | TILE |
| Memory | `output_tensor`'s config, else `memory_config`, else DRAM interleaved. Written through a `TensorAccessor`, except on the native sharded path, where compute packs into the resident output shard. |

## Blocking Model

### Geometry (single source: host `derive_geometry()`; per image, `ceil`)

| Symbol | Formula | Scope |
|---|---|---|
| `G` | `mesh_shape[cluster_axis]` | group |
| `p_self` | group position of this chip (`c` for axis 1, `r` for axis 0) | chip |
| `num_images` | `prod(shape[:-2])` | tensor |
| `tensor_row_tiles` (`R_t`) | `num_images · ceil(M/32)` (never `floor(num_images·M/32)`) | tensor |
| `slice_w_tiles` (`Wt_lp`) | `ceil(W_local/32)` | one slice |
| `land_w_tiles` | `G · Wt_lp`: the **landing tile space**; slice p's tile j is landing tile `p·Wt_lp + j` | gathered row, slice-major |
| `out_w_tiles` | `ceil(G·W_local/32)`; equals `land_w_tiles` iff `W_local % 32 == 0` | gathered row, dense |
| `w_valid_in_slice_tail` | `W_local − 32·(Wt_lp − 1)` ∈ [1, 32] | slice |
| input page of (row r, slice tile j) | `r·Wt_lp + j` | per chip |
| output page of (row r, out tile t) | `r·out_w_tiles + t` | per chip |

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| **device / slice (p)**: which chip holds which W range | **dependent**: Σx² spans every slice. The caller pins the cut (one slice per chip). This op must not and does not re-cut it. | `G` (fixed by the mesh) | `mesh_shape[cluster_axis]` | host `plan()` | the caller's: chip p holds slice p. The **combine across devices is replaced by replication**: every chip receives every raw slice and reduces locally, so no partial crosses the fabric. | — (input contract) |
| **row** (tile-rows, folded leading dims) | **independent**: an output row depends only on its own input row | `block_row_tiles` | **1**. Rows land one after another (the transport is row-ascending), so a block spanning two rows waits for the later one and the one before it stays unprocessed. The fixed per-block cost (≈ one combine round + chain inits, ~1 µs) is paid while later rows are still in flight, for every block but the last. Same bar as an op-sized CB: the alternative is a perf lamp. | host `select_split()` → RT `num_blocks_this_core`; CT `block_row_tiles` | **round-robin** over row groups: row r → group `r mod n_active_groups`, block index `r div n_active_groups`, which spreads the latest-landing rows over different groups. Groups = consecutive `k` cores of the worker list. | knob-turn |
| **W within a chip** (landing tiles) | **dependent**: Σx² spans the row | `core_w_tiles` (member width; pitch `w_pitch` = max member width) | from `select_split()`: `k` members per row group whose ranges are **nested in slices** (`k = G·k_s` with each slice cut into `k_s` balanced parts, or `k = G/m`, `m ∣ G`, each member holding m whole slices). Focus: k = 8, 6 tiles. | host `select_split()` → RT `w_tile_start`, `core_w_tiles`, `member_idx`; CT `w_split_cores` (k), `combine_owner` | the k members of a row group share every row of the group | knob-turn for k ∈ nested set (both combine topologies built); non-nested k is a perf lamp |
| **transport item rows** (`xfer_rows`) | — (a transport quantum along the row axis, not a block extent: the unit one arrival increment covers) | `xfer_rows` | `max(1, ceil(item_min_chunks / chunks_per_row_on_link))`, `item_min_chunks = 4`; focus = 4 | host `plan()` → port RT | n/a (per port) | knob-turn |
| **transport chunk** (tiles per packet) | — (a mechanism quantum) | `run_pages` | `floor(fabric_max_payload / T_in)`: 7 bf16, 3 fp32, 13 bfp8 at the 14 KiB + 64 B payload; a per-row run is split into `ceil(len/run_pages)` **balanced** chunks | host → port CT | n/a | fabric-config knob |
| **link** (L parallel planes) | independent: the column range of every transport item is cut into L sub-ranges, one per link | `num_links` | resolved L | validate → `plan()` | one port core per (direction, link) | knob-turn |
| **far-slice direction split** (balanced ring) | independent: the far slice is split between the two directions by **tile-row parity** (part 1 = even rows, part 2 = odd rows, each over the link's whole column range; Refinement 3), so its packets carry the link's full run | `FAR_ROW_SPLIT` | on (off = the Phase 0 column split, A/B) | host `link_cols()` → entry row filter `rf` | n/a | knob-turn |

Operand-reuse check (operand × split):

| Operand | Split across | Varies along it? | Consequence |
|---|---|---|---|
| x (local slice) | row groups / members | yes | each byte read from DRAM once, by the member that owns it |
| x (remote slices) | chips (the gather itself) | yes, by slice | each chip receives each remote slice once; relays re-send landed bytes from L1 (the ring's store-and-forward) |
| x (local slice) | 2·L ports (both directions send it) | **no**: both directions send the same bytes | reuse-shared by the ring: ports read it from the **members' L1** (one DRAM read feeds the 2 directions), not from DRAM |
| gamma | row groups | **no** → reuse-shared | each member reads its `core_w_tiles` columns once per program. Broadcast across groups is the `deferred` row `gamma_mcast`. |
| inv_rms (stage output) | the k members of a group | **no** | the in-chip combine (symmetric all-gather k ≤ 8, owner combine beyond) |

Step-1 table, one row per axis of every data object, no blanks:

| Data | Axis | Character | Block factor (Phase 0) | Core assignment | Buffer depth | Why |
|---|---|---|---|---|---|---|
| x local slice | row | independent | 1 tile-row per block, all blocks resident | round-robin row groups | resident (whole assignment) | read once from DRAM into the landing slot; the ports re-send it from there |
| x local slice | W | dependent | member range ∩ slice `p_self` | the members whose range lies in slice `p_self` | same CB | |
| x remote slices | row | independent | 1 tile-row per block | the same round-robin groups on every chip, so a landing address is computable by the sender | resident: **no flow control from receiver to sender** (one-shot landing per call) | landing into L1 is only legal for slots that are never reused within a call |
| x remote slices | W | dependent | member range ∩ slice p | members nested in slice p | same CB | |
| x remote slices | device | dependent (pinned) | whole slice | every chip receives all G−1 | — | replication instead of a cross-device combine |
| sumsq / partial | row | independent | 1 tile-row | the member | 1 block | |
| sumsq / partial | W | reduced away | — | — | — | DEST accumulation + row collapse |
| gathered partials | row × member | row independent, member dependent | `k` per row | every member (symmetric) or the row owner (owner combine) | 2 blocks (lockstep cap) | base op mechanism; the combine has nothing to spread: k tile-adds per row |
| inv_rms | row | independent | 1 tile-row | every member of the group | 1 block | consumed by normalize of the same block |
| out | row, W | as x (aligned); dense tiles `[0, out_w_tiles)` (regime N) | 1 tile-row block, streamed in `out_unit` tiles | the member (regime N: the single row owner) | `out_depth = 2` | written as soon as normalized: overlaps the transport of later rows |
| port stream | item × chunk | independent | `xfer_rows × link-part` items, `run_pages` chunks | one port per (direction, link) | `2·group` chunk slots | example's batching |

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_x_landing` | — (resident: `num_blocks_this_core` blocks) | all blocks | **correctness**: remote chips write into it without credit, so no slot may be reused within a call (Mechanism caps) |
| `cb_out_tiles` | `out_depth` × `out_unit` | 2 × `min(core_w_tiles, 16)` | the writer drains one unit while compute packs the next |
| `cb_x_aligned` (regime N) | `compose_depth` × `out_unit` | 2 | reader composes the next unit while compute normalizes this one |
| `cb_gathered_sumsq` | fixed | 2 blocks if `num_blocks_this_core > 1`, else 1 | **correctness** (lockstep slot reuse, base op cap) |
| `cb_port_chunks` | `port_cb_bytes` → `group` | 112 KiB → `group = min(8, port_cb_bytes / (2·run_pages·T_in))` (4 at bf16) | reader fills one batch while the sender sends the other (example baseline) |
| `cb_sumsq_tiles`, `cb_partial_sumsq`, `cb_inv_rms` | 1 block | 1 | handoff within one block |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Fabric writes into a peer's L1 **without credit** | `cb_x_landing` capacity | holds the core's whole assignment (`num_blocks_this_core · block_row_tiles · w_pitch` tiles). `select_split()` admits only splits whose footprint fits; none fits → `RuntimeError` (the `dram_landing` row is the deferred fallback) | a reused slot is overwritten by a peer's packet before compute read it → silently wrong rows |
| Landing address computed by the **sender** | `cb_x_landing` base address | `cb_x_landing` is CB index 0, **declared first, on the union of worker and port cores, identically on every chip** (same CB list and order) → its base is identical everywhere. The port kernel reads it with `get_write_ptr(cb_x_landing)` at kernel start (it never reserves or pushes that CB). Worker set and member layout are identical on every chip. | packets land at the wrong L1 address on the peer |
| Identical worker set on every chip | which logical cores are workers | worker list = allowed cores, row-major, minus the **union over all chips** of the port cores | a member's (x, y) differs per chip → a packet lands on the wrong core |
| Arrival-counter semantics | items per port stream | the upstream increments the downstream port's `sem_arrival` **exactly once per item, on the item's last chunk** (fused write+inc). Item `i` (in the upstream's send order) has landed iff `sem_arrival ≥ i + 1`. | a relay or a notification reads data that has not landed |
| `ready` fence (global semaphore) | sends per call | each port with a peer that writes into this chip sends exactly 1 ready increment per call; each sending port consumes exactly 1 (`wait ≥ 1`, then `−1`). Back at 0 after every call. | a peer writes into my L1 while the previous program (possibly another op) still owns it, or a ready is double-counted |
| Program semaphores re-initialized per launch | `sem_arrival`, `sem_landed[*]`, `sem_local_loaded`, `sem_partials`, `sem_inv` | monotonic targets within a call, never reset. Every remote increment is issued only after this chip's `ready` went out, which is after this program's config (semaphore init) was written. | — (the `ready` counter is the only one that can be hit before launch, hence global) |
| Per-stream landing notifications | `sem_landed[s]` targets | one semaphore per incoming stream s = (direction, link), at most 2·L. Within a stream the port notifies in its own (row-ascending) order, so block b of a member is complete iff `sem_landed[s] ≥ (b+1)·u[s]` for every s (`u[s]` ∈ {0, 1}: whether stream s touches the member at all, host flag; since Refinement 3 the per-row entry count varies with the far slice's row parity, a flag does not) | a shared counter across streams could reach a block's target with one stream's unit of that block still missing |
| Semaphore count per core | ids | ≤ 2·L + 5 ≤ 9 program semaphores (incl. the per-parity partials pair), all declared on the union core range (`NUM_SEMAPHORES = 16`, `tt_metal/impl/buffers/semaphore.hpp:16`) | — |
| Fabric packet payload | `run_pages` | `floor(get_tt_fabric_max_payload_size_bytes() / T_in)` | the router truncates or faults |
| Fused write + atomic inc stalls the receiving router (rule 4) | `xfer_rows` | ≥ `item_min_chunks = 4` chunks per item where the shape allows | the link rate drops (1.14× at an increment every 2 chunks) |
| One producer per router channel (rule 10) | ports | exactly one sender core per (direction, link). Only the port's BRISC opens a connection. | undefined router behavior |
| Sender on NoC1, under its Ethernet core (rules 6, 7) | placement | port core = free worker with the fewest NoC1 hops to that link's Ethernet core (example `_worker_nearest_noc1`, `program_descriptor_with_inline_kernels.py:581`) | 40.6 / 30.5 GB/s instead of 48.5, or a hang (NoC0 sender, both directions active) |
| Symmetric all-gather of partials | `k` | symmetric for `k ≤ K_SYM_MAX = 8`; owner combine beyond, `k ≤ K_CAP = min((341 − 13)/2, n_workers)` (base op `rms_norm_program_descriptor.py:66-73`) | owner/peer RT args exceed 341 |
| Lockstep slot reuse in `cb_gathered_sumsq` | its depth | 2 blocks when a core runs > 1 block (base op) | a peer's partial(b+1) overwrites slot(b) |
| Partials arrival counter (symmetric and owner combine) | which block an increment counts for | **one counter per block parity** (`sem_partials`, `sem_partials_odd`), wait ≥ k·(blocks of that parity so far). A peer runs at most one block ahead (its partial(b+2) needs this member's partial(b+1), produced only after combine(b)). (verifier fix; the base op's single cumulative counter was ported first) | on one cumulative counter a fast peer's block-(b+1) increment completes a slow peer's still-in-flight block-b count: combine(b) reads a stale slot → wrong per-row inv_rms (golden: intermittent inf / precision misses on 2048-row shapes after a program switch) |
| Members nested in slices | `k` candidates | `k ∈ {G/m : m ∣ G} ∪ {G·k_s : k_s ≥ 1}`, member width ≥ `OCC_TILES_PER_MEMBER = 4` unless k is the smallest candidate | a transport chunk straddles two members → packets fragment (fewer bytes per packet on a packet-rate-bound link) |
| Regime N compose needs the whole dense row on one core | `k` in regime N | `k = 1` | an output tile whose columns come from two members cannot be composed without a cross-core exchange |
| W padding must not enter Σx² | slice tail tile (regime N), last output tile | reader zero-fills columns `≥ w_valid_in_slice_tail` of each slice's last landing tile in L1 (base op `zero_w_tail_of_tile`, `kernels/rms_norm_reader.cpp:52`; `fill_l1_range`, `l1_helpers.hpp:90`) after the block is notified and before its credit is pushed. The compose writes zeros for output columns ≥ W. | poisoned padding (golden `poison_padding`) leaks into Σx² |
| BH DRAM read alignment 64 B | gamma stick / row-0 reads | whole 64 B (bf16) / 128 B (fp32) stick read, then a local copy of datums 16..31 into face 1 row 0 (base op `GammaSliceLoad`, `kernels/rms_norm_writer.cpp:287`) | wrong gamma bytes |
| DEST capacity (fp32 DEST: 4 tiles half-sync) | none on the block; the helpers clamp their own walk (`DEST_AUTO_LIMIT`, `dest_helpers.hpp:103`) | — | — |
| RT args per kernel ≤ 341 | per-core args | the worker-NoC table (≤ n_workers packed `x | y<<16` words) goes in **common** runtime args of the port kernels; per-core args stay O(G + 2L + k) | compile / launch failure |

### Regimes

Named boundaries: **DRAM** (per chip: x read once = S, out written once = G·S) and **links** (busiest directed link: (G−1)/2·S on a balanced ring, (G−1)·S on a line).

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **A — aligned, slice-nested W members** (k = 1 / symmetric k ≤ 8 / owner k > 8 are one code path, `combine_owner` CT) | **built** | `W_local % 32 == 0` | `block_row_tiles × core_w_tiles` (1 × member width) | DRAM at minimum (x once, out once). Links at minimum (balanced ring). Adds on-chip NoC: ports read 2·S (own slice, both directions) + relayed bytes from member L1. Partials: symmetric k²·2 KiB per row / owner 2k·2 KiB per row. Gamma: n_active_groups · W · g_bytes. | per block: one landing wait, one chain call for Σx² (inits once), one collapse `reduce`, one combine round (k writes + k semaphore incs + one wait) and one combine `reduce`, one normalize chain call (mul-bcast + dest-reuse inits once), one writer barrier per `out_unit`. A bigger block pays these fewer times but waits for its latest row. |
| **N — W_local non-aligned, one core per row, compose at pass 2** | **built** | `W_local % 32 != 0` | `block_row_tiles × land_w_tiles` (pass 1) / `× out_w_tiles` (pass 2) | DRAM at minimum. Links: slices travel as padded tiles (`Wt_lp·32 ≥ W_local` columns, ≤ 1 tile per slice-row over the minimum). No combine. | same fixed costs minus the combine. The compose (RISC-V datum shift) is per tile and does not amortize. |
| `n_split` (regime N with W split across members: members own dense output-tile ranges and fetch neighbour landing tiles) | **deferred**: every INPUTS non-aligned shape fits regime N at k = 1 for G ≤ 8 (largest: `(1,1,64,1000)` fp32 G = 8, landing 8·32·4 KiB = 1 MiB < budget, see l1_ledger.md), and none is perf-flagged. Reachable: the landing layout is already slice-major per member; only the compose source changes from local L1 to a peer's L1 (+ one notification). | — | — | same DRAM / links; adds cross-core landing-tile reads at member boundaries | — |
| `dram_landing` (remote slices land in the output DRAM buffer at their final pages; members read back, normalize, overwrite in place) | **deferred**: the capacity fallback for an assignment that does not fit L1. No INPUTS shape at G ≤ 8 needs it (l1_ledger.md symbol bounds). Reachable: the port kernels already address destinations per chunk; the destination changes from a member L1 slot to an output page. | footprint of every admissible split > budget | — | +(G−1)·S DRAM writes, +G·S DRAM reads | — |
| `stats_then_normalized_gather` (exchange per-row Σx² partials first (fp32), normalize own slice, then gather normalized slices into the peers' output DRAM) | **rejected**, superseded by A everywhere. Same link bytes (+M·4 B) but a serial ~5 µs stats exchange (`fabric_ring_exchange`) before the first output byte may cross. Fabric writes into W-gathered output pages are runs of `Wt_lp` tiles per row whose same-bank runs are 1–2 tiles, so packets fragment. | — | — | same | — |
| `direct_multihop` (send each slice straight to its 2-hop destination instead of relaying) | **rejected** (TRANSPORT): +1.1–1.6 µs fence (`fabric_ring_exchange`), and the busiest link carries 2 slices instead of 1.5 on a 4-ring | — | — | links +33% | — |
| `fabric_multicast` (one packet delivered to p+1 and p+2 along a direction, no relay turnaround) | **deferred** (TRANSPORT): not measured by any example on this fabric (the examples cover relay and direct). Reachable: replaces the relay entries of a port's schedule only; landing layout, notifications and members are unchanged. | — | — | links at minimum, no relay L1 reads | — |
| `port_dram_sourcing` (ports read the own slice from DRAM, as the example does) | **rejected**: +2·S DRAM reads per chip, and per-tile DRAM page reads at port rate (the example needed same-bank runs to feed a link, rule 9). Sourcing from member L1 is both fewer bytes and contiguous runs. | — | — | +2·S DRAM | — |
| `contiguous_rows` (base op: a contiguous tile-row range per group) | **rejected**: the last-landing rows all fall on the last group, whose whole assignment is then exposed after the final packet. Round-robin is the same traffic with a one-block tail. | — | — | same | — |
| `gamma_mcast` (one reader per W range multicasts gamma to that range's members across groups) | **deferred**: r = gamma bytes / x bytes per member ≤ 1/32. Catalog `shared_input_reuse` measured null for r ≤ 0.125. Reachable: `cb_gamma` is a program-lifetime CB with one producer (the writer), and the base op's `GAMMA_MCAST` channel only replaces its DRAM read. | — | — | gamma read once per W range instead of once per group | — |
| `sharded_zero_copy` (consume the input shard in place and pack into the output shard: CBs backed on the shards) | **implemented (Refinement 1)**, gated: `native_shard_split()` + `native_keeps_up()`. HEIGHT (input and output with the same row partition, full-width shards): one group per shard, k = 1, contiguous rows (`rows_per_group`). The own-slice columns are read in place from `cb_x_in` (backed on the input shard) by compute, by the regime-N compose and by the ports (no load). WIDTH (output full-height shards, regime A): one group, k = output shard columns, member m = output shard core m; its own tiles are gathered from the input shards (cross-core reads; the input partition differs from the output's). Both: remote slices still land in `cb_x_landing` (not in the output shard: the ports relay from the landing, so normalizing it in place would forward normalized data), and compute packs straight into `cb_out_tiles` backed on the output shard (the writer only waits / pops). The ports are placed off the member cores (`plan(avoid=...)`: the shard grid starts at core (0,0), where Phase 0 put ports). | TILE L1 shards, ROW_MAJOR orientation, shard partition as above, the CBs fit L1, >= 2L free cores per chip off the shard grid, and the shard-fixed split keeps up with the transport: per-core work (cost model) <= `NATIVE_KEEP_UP_FRACTION` (1.0) x T_xfer. Otherwise the sharded `TensorAccessor` fallback (Phase 0 path). Measured: HEIGHT 640x384 34.9 vs 41.4 us, 2048x384 68.2 vs 93.4 us (native wins). HEIGHT 128x512 (4 shards x 64 tiles) and WIDTH 64x512 (64 one-tile members: one-tile packets, 64-way combine) / 32x4096 (k = 103) lose 7-18 us natively, and the gate keeps them on the fallback. | shard | x own slice (HEIGHT) / out never cross the NoC | — |
| `probe_placement` (example: one probe program reports each connection's Ethernet core) | **rejected**: a second device dispatch on the first call violates one dispatch per invocation. Placement is derived on the host instead (Work Distribution). | — | — | — | — |
| `single_core` | **rejected**: Phase 0 is multi-core | — | — | — | — |

### Regime / split selection (exact, host-side, `select_split()`)

```
regime = "N" if W_local % 32 else "A"
n_workers = |allowed − ∪_chips ports|                     # worker list: row-major logical order
budget = cb_l1_budget(device) (base op :888), minus nothing else; port footprint checked separately
cands  = {1} if regime == "N" else nested_k(G, Wt_lp)    # {G/m : m | G} ∪ {G·k_s}, member width >= 4
                                                          # (the smallest candidate is always kept)
for k in cands, k <= K_CAP:
    member ranges: k <= G -> m = G/k whole slices; k > G -> each slice cut in k_s balanced parts
    w_pitch      = max member width;  C(k) = 0 if k == 1 else 3 if k <= 8 else 3 + 0.4·k
    n_groups     = min(n_workers // k, R_t);  nb = ceil(R_t / n_groups)
    fits(k)      = worker_footprint(k, nb) <= budget and port_footprint(nb, w_pitch) <= budget
    total(k)     = nb · (w_pitch + C(k))             # x-tile equivalents per core
    tail(k)      = w_pitch + C(k)                     # what remains after the last landing
T_xfer(k) = t0 + (1 + 0.55·(pk(k)/pk_ref − 1)) · (G−1)·S / (B_chip · L)   # ring (8.4 µs, 77.2 GB/s); line (9.8 µs, 42.1 GB/s)
            # pk(k) = packets per gathered row (every link part of every slice cut at the member boundaries into runs of
            # <= run_pages tiles); pk_ref = the unsplit link runs. PACKET_TIME_SHARE = 0.55 (Refinement 4, calibrated)
keep_up(k) = total(k) · 310 ns <= KEEP_UP_FRACTION · T_xfer(k)      # KEEP_UP_FRACTION = 0.5
k* = argmin [T_xfer(k) + tail(k) · 310 ns] over {fits ∧ keep_up}; ties → fewer blocks (nb).
     (Refinement 4: the estimated end of the call. With equal packet counts it reduces to the Phase 0 argmin tail.)
     If no candidate keeps up: argmin total(k) over {fits}. If none fits: RuntimeError.
```

Perf-flagged shapes on the fused grid (QB2, 11×10 = 110 cores; union of port cores |P| = 4 → 106 workers, |P| = 8 → 102):

| Case | R_t | Wt_lp / Wt | k* (member w) | combine | n_groups | blocks/core | busy cores | tail (x-tile eq.) | total (µs est.) vs T_xfer_est |
|---|---|---|---|---|---|---|---|---|---|
| 640×384 [FOCUS] | 20 | 12 / 48 | 8 (6) | symmetric | 13 (12) | 2 | 104 (96) | 9 | 5.6 vs 18.0 |
| 640×512 | 20 | 16 / 64 | 8 (8) | symmetric | 13 (12) | 2 | 104 (96) | 11 | 6.8 vs 21.1 |
| 640×1792 | 20 | 56 / 224 | ~~12 (19, 19, 18) owner~~ → **8 (28)** since Refinement 4 | symmetric | 13 (12) | 2 | 104 (96) | 31 | 19.2 vs 53.0 |

At |P| = 8 every case keeps its k, blocks per core and regime. No row is idle and no group is ragged beyond one block. This is the base op's split **rule** re-run on the fused grid with one deliberate deviation: the objective is the tail, not the total. That deviation applies because rows land progressively, so all but the last block hide under the transport (the base op had no exchange to hide under).

Regime-pinned tests: the acceptance test pins regime A at k = 1 (`(1,1,32,64)`: Wt = 8, smallest candidate), A symmetric (focus), A owner (`(1,1,640,1792)`), and regime N (`(1,1,17,50)`, `(1,1,64,1000)`). Every pin depends only on `n_workers ≥ 96`, which holds on BH (110 or 130 cores) for any |P| ≤ 8.

### Traffic ranking

Per chip, per call; S = M·W_local·elem_bytes. Counted as crossings per tensor: DRAM first, then links, then on-chip NoC.

| Candidate | x DRAM | out DRAM | busiest link | on-chip NoC | Verdict |
|---|---|---|---|---|---|
| **A/N: raw slices land in member L1, relay from L1, every chip normalizes everything** | 1·S | G·S | (G−1)/2·S ring, (G−1)·S line | 2·S own-slice port reads + relayed reads + (G−1)·S landing writes + partials | **chosen**: minimum on DRAM and links; on-chip traffic is the cheapest tier |
| ports source the own slice from DRAM (example) | 3·S | G·S | same | −2·S | rejected: +2·S on the most expensive tier |
| stats exchange, then gather normalized slices into peers' output DRAM | 1·S | G·S (1 local + (G−1) by fabric) | same + M·4 B | less (1/G of the norm) | rejected: same bytes, serial dependency, fragmented DRAM-landing packets |
| DRAM landing | 1·S + (G−1)·S write + G·S read-back | G·S | same | less | deferred capacity fallback: +(2G−1)·S DRAM |
| rows only (k = 1) for aligned W | 1·S | G·S | same | no partials | same traffic; rejected on the tail (a whole row per core is exposed after the last landing) |

The traffic ranking and occupancy agree here. The chosen split is the cheapest on every tier but the on-chip NoC, and that tier is spent to make the tail short.

### Block schedule

Per worker (member) core; reader (NCRISC), compute (TRISCs) and writer (BRISC) realize their parts asynchronously:

```cpp
load_gamma_slice();                                       // writer, once (base op GammaSliceLoad)
load_local_blocks();                                      // reader, once: own-slice columns of ALL blocks, DRAM -> cb_x_landing
signal_local_loaded();                                    // reader, once: +1 on each port that sends these columns
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    await_landing_block(block_idx);                       // reader: per-stream sem_landed targets; regime N: zero slice tails; push credit
    sum_squares_block(block_idx);                         // compute: Σ x² per row (DEST-accumulated) -> cb_sumsq_tiles
    collapse_sumsq_block(block_idx);                      // compute: row collapse; k == 1: + finalize -> cb_inv_rms; else -> cb_partial_sumsq
    exchange_partials_block(block_idx);                   // writer, k > 1: symmetric all-gather / owner gather + inv return
    combine_partials_block(block_idx);                    // compute, k > 1: Σ over k + finalize -> cb_inv_rms
    compose_aligned_block(block_idx);                     // reader, regime N: dense output tiles -> cb_x_aligned (streamed)
    normalize_block(block_idx);                           // compute: x · inv_rms (bcast col) · gamma (row bcast) -> cb_out_tiles
    store_block(block_idx);                               // writer: cb_out_tiles -> output pages
}
```

Per port core (one per (direction, link); reader NCRISC/NoC0, sender BRISC/NoC1):

```cpp
open_and_fence();                                         // sender: open connection, send ready to the peer that writes into me, wait own ready (−1)
await_local_loaded();                                     // reader: sem_local_loaded >= host-counted members
for (uint32_t step = 0; step < num_steps; ++step) {
    stream_own_item(step);                                // reader: own entry item `step` from member L1 -> cb_port_chunks; sender: packets, fused inc at item end
    handle_incoming_items(step);                          // reader: upstream items scheduled at step − relay_lag: wait sem_arrival, notify members,
                                                          //         relay ones -> cb_port_chunks; sender forwards them
}
drain_and_close();                                        // sender: flush, close. reader: wait the final sem_arrival target, notify the rest
```

Item order: the upstream sends entry e's item xb at step `xb + e·relay_lag` (`relay_lag = 1`), with entries in the example's order (`_schedule`, `program_descriptor_with_inline_kernels.py:384`; own slice first, then entries relayed from upstream). Relaying entry e of mine is the upstream's entry e−1 (or its half), so its item xb was sent by the upstream one step earlier.

| Block operation | Block shape it acts on | Resident across it | Intended frequency of fixed costs |
|---|---|---|---|
| `load_gamma_slice` | `core_w_tiles` compact gamma pages (regime N: `out_w_tiles`) | — | once per program; stick page t (ROW_MAJOR) or row 0 of faces 0/1 of tile t (TILE) per output tile column; one barrier |
| `load_local_blocks` | all blocks' own-slice columns (`num_blocks · |member ∩ slice p_self|` tiles) | cb_x_landing | once per program; reads batched, one barrier |
| `signal_local_loaded` | — | — | ≤ 2·L NoC semaphore increments per member, once |
| `await_landing_block` | one block's landing tiles | all landed blocks | ≤ 2·L semaphore waits per block; one `cb_push_back` per block; regime N: ≤ G tail zero-fills per block |
| `sum_squares_block` | `block_row_tiles × core_w_tiles` (N: `× land_w_tiles`) | cb_x_landing (not popped), cb_gamma | one call per block (mul init once); one DEST acquire + pack per tile-row |
| `collapse_sumsq_block` | `ReduceInputBlockShape::of(block_row_tiles, 1)` | same | one `reduce` per block; k == 1: finalize post-op |
| `exchange_partials_block` | `block_row_tiles` partials → `block_row_tiles` row-form gather tiles (symmetric, Refinement 5) / owner rows | same | one round per block: per peer two 64 B face-row writes into row m of the row's gather tile (Refinement 5; was faces 0/2, 2 KiB) + flush + k semaphore incs + one wait. Owner: k partials in + k inv back per owned row |
| `combine_partials_block` | `of(block_row_tiles, k)` with `ReduceWithinTile::Skip` / owner row combine | same | one `reduce` per block + finalize |
| `compose_aligned_block` (N) | `out_w_tiles` dense tiles per row, in `out_unit` units | cb_x_landing | one reserve/push per unit; per tile a RISC-V datum copy from ≤ ⌈32/W_local⌉+1 landing tiles |
| `normalize_block` | `grid(block_row_tiles, core_w_tiles)` (N: `out_w_tiles`, streamed from cb_x_aligned) | cb_x_landing, cb_inv_rms (popped at end), cb_gamma | one chain call per block: mul-bcast-col init + dest-reuse gamma init once |
| `store_block` | `core_w_tiles` output pages per row (N: `out_w_tiles`) | — | one wait / flush / pop per `out_unit` |
| `open_and_fence` | — | — | once per call per port: connection open, 1 ready sent, 1 consumed |
| `stream_own_item` / `handle_incoming_items` | one item = `xfer_rows` rows × link-part columns, `run_pages` chunks | cb_port_chunks (2 batches) | per item: one fused write+inc (last chunk); per batch of `group` chunks: one read barrier + push (reader), one flush + pop (sender); per incoming item: one counter wait + ≤ (xfer_rows · members) local notifications |
| `drain_and_close` | — | — | once per call |

**Stall-shadow check.**

| Waiting stage | Waits for | What runs in the window (designed) | Not filled, and why |
|---|---|---|---|
| port `open_and_fence` (~5 µs, `fabric_ring_exchange`) | the peer's program start | members: `load_local_blocks`, `load_gamma_slice`, and Σx² + collapse of every local-slice block. Ports: `await_local_loaded`, then the reader pre-fills `cb_port_chunks` (2 batches) with own items, so the first packets go out the moment the fence clears. | — |
| members `await_landing_block` | remote rows | the member's earlier blocks (normalize + store of rows already landed): the round-robin + row-ascending transport keeps a backlog of landed rows | a member holding only the far slice has nothing before its first row lands: no work is independent of its data |
| relay reads (port reader) | upstream item landed | the next own item (`relay_lag = 1` puts own item xb+1 before relay item xb) | — |
| compute `combine_partials_block(b)` | peers' partials of block b | nothing in Phase 0: pass 1 of block b+1 is independent but **not** landed yet in general, and a blocking lookahead would hold normalize(b) hostage to the landing of b+1. The base op's `combine_lookahead` is therefore **off** here. | lamps: a non-blocking landed-check of b+1; pre-scaling x·γ (commutativity) during the wait |
| last block | the final packet | nothing: every remaining step depends on it. Bounded by `tail(k*)` (≤ 9 x-tile eq. on the focus) | — |

Algebraic permission used: none that changes floating-point order. The combine sums partials in fixed member order and pass 1 in fixed tile order, so results are bit-identical run to run regardless of arrival order.

### Perf lamps

| Lamp | Kind | Why the default may be wrong here | Nearby alternative to measure |
|------|------|-----------------------------------|-------------------------------|
| ~~Overlap: `block_row_tiles = 1`~~ | OVERLAP | **measured (Refinement 4)**: 2 is worse on the focus (24.57 → 28.84 µs) and on 640×1792 (78.38 → 89.57 µs): a 2-row block waits for its later row. 1 stays (env `AGRMS_BLOCK_ROW_TILES`, live knob) | — |
| Grid sync: symmetric k = 8 on the focus | OVERLAP | 64 partial messages per row per group; at 6 tiles per member the combine is comparable to the compute | k = 4 (12 tiles, 1 block) and k = 12 owner (4 tiles) on the focus |
| ~~`KEEP_UP_FRACTION = 0.5`, cost constants~~ | OVERLAP | **done (Refinement 4)**: the k sweep showed the transport itself depends on k (640×1792 TRANSPORT-only 76.2 µs at k = 12 vs 70.0 at k = 8: 10 vs 8 packets per slice-row). The objective is now the packet-aware estimated end (see Regime / split selection): 640×1792 L2 k = 12 owner → 8 (86.1 → 81.5 µs before the prescale), L1 k = 28 → 16 (145.0 → 108.0 µs). Focus sweep k = 4 / 8 / 12: 26.0 / 25.1 / 34.4 µs (k = 8 kept) | — |
| `relay_lag = 1` | OVERLAP / TRANSPORT | a relay item may still be in flight one step later (fence skew between chips) | `relay_lag` 0 and 2 |
| `xfer_rows` (`item_min_chunks = 4`) | TRANSPORT | larger items mean fewer router stalls but coarser relay and notification latency | `item_min_chunks` 2 and 8 |
| ~~Far-half link split by columns~~ | TRANSPORT | **done (Refinement 3)**: the far half is split between the directions by tile-row parity, 6-tile packets on the focus (31.47 → 29.38 µs). Scatter packets are moot after it: a packet is one member's 6-tile run, two runs (12 tiles) do not fit one 7-tile payload, and the item's fused increment ends a packet anyway | — |
| Port-mediated notifications | OVERLAP | a landed row is announced only when the port reader reaches that item | fused inc straight to the member's semaphore when a unit is ≥ 4 chunks |
| ~~Own slice sent only after **all** local members loaded~~ | OVERLAP | **measured null (Refinement 4)**: on every chip the ports' `AG_LOCAL_WAIT` ends (4.2–5.1 µs) before their ready fence `AG_FENCE` (4.6–10.3 µs): the fence binds the first packet, not the load. Not built | — |
| Members nested in slices | OVERLAP | on 1792, k = 12 nested gives 19 tiles per member. A non-nested k (e.g. 5 on 512: 13 tiles) shortens the tail at the cost of split packets at slice edges. | non-nested k on 512 / 1792 |
| `port_cb_bytes = 112 KiB` | TRANSPORT | items here are short (4–7 chunks), not the multi-MiB shards the example tuned on | re-measured in Refinement 3 (`AGRMS_PORT_CB_KIB`): 56 KiB 31.3, 224 KiB 26.3 / 30.0 vs 112 KiB 25.1 / 29.4 µs: 112 stays |
| Compute lookahead off | OVERLAP | the combine wait of block b is unused | non-blocking `cb_pages_available_at_front` check of block b+1 |
| ~~x·γ pre-scale in the combine wait~~ | OVERLAP | **done (Refinement 4)**: `cb_x_gamma` (T_acc, one block), compute CT `prescale` (regime A, k > 1, gamma, round-robin split). Focus 25.09 → 24.57 µs, 640×1792 k = 8 81.47 → 78.41 µs, precision baseline unchanged | — |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| own slice: input → member `cb_x_landing` | input-dtype tiles | member reader, `TensorAccessor(input)` (DRAM, L1 interleaved or L1 sharded alike); page `r·Wt_lp + j` → slot `(blk(r)·w_pitch + (landing_tile − w_tile_start))·T_in` | then `+1` on `sem_local_loaded` of every port carrying these columns |
| own / relayed item: member L1 → `cb_port_chunks` | input-dtype tiles | port reader, NoC0 `noc_async_read` of contiguous runs (one run = ≤ `run_pages` tiles of one row of one member), batches of `group` chunks per barrier | relayed items only after `sem_arrival ≥ item_idx + 1` |
| `cb_port_chunks` → fabric → peer member `cb_x_landing` | input-dtype tiles, 1 run per packet | port sender, NoC1, the example's raw sender: pre-routed header ring, `send_current_slot_non_blocking`, flush per batch, `fabric_set_unicast_route` (2D) / hop count (1D); destination `get_noc_addr(member_x, member_y, landing_base + offset)` on the peer | last chunk of each item: `to_noc_fused_unicast_write_atomic_inc` to the **peer port's** `sem_arrival` |
| arrival → member notification | — | receiving port reader: after `sem_arrival ≥ i+1`, `noc_semaphore_inc(member, sem_landed[stream], 1)` for each member row in item i | per-stream monotonic targets |
| member compute | T_acc intermediates (Float32 / Float16_b per `fp32_dest_acc_en`) | base op pipeline (API Mapping) | |
| in-chip combine | T_acc tiles, faces 0/2 | base op writer exchange (`exchange_send` / `exchange_recv_*`, `kernels/rms_norm_writer.cpp:477-680`) | peers' NoC coords as RT args |
| regime N compose | input-dtype tiles | member reader: RISC-V datum copy, dense column x ← slice `x div W_local`, column `x mod W_local` of the landing tiles; zeros for x ≥ W | face-aware addressing (row r: face `2·(r≥16) + (c≥16)`) |
| `cb_out_tiles` → output | output-dtype tiles | member writer, `TensorAccessor(output)`, per `out_unit`: wait, writes, flush, pop | aligned: page `r·out_w_tiles + landing_tile`; N: `r·out_w_tiles + t` |

Port placement and wiring (host, `plan()`, adapted from `fabric_all_gather.plan()`, `program_descriptor_with_inline_kernels.py:717`):

| Item | Decision |
|---|---|
| Groups / schedules | `build_groups(mesh_shape, cluster_axis)` (`:371`); `_schedule(p, G, ring, balance=True)` (`:384`): balanced ring for even G ≥ 4, plain ring for odd G, line otherwise. Ring with G = 2 → line. |
| Ports | one per (direction ∈ {fwd, bwd}, link l < L) on every chip, including receive-only line ends (their reader still notifies, their sender only sends `ready`) |
| Placement (host only) | the connection's Ethernet channel is the **first runtime arg** `ttnn.setup_fabric_connection(...)` returns for a Tensix VC0 connection (`edm_fabric_worker_adapters.hpp:110-118` reads it as `eth_channel`). Map it through the arch SoC descriptor's `eth` list and the cluster descriptor's `eth_harvesting_mask` (example `_eth_physical_lists`, `:514`), then `_worker_nearest_noc1` (`:581`). If the mapping is unavailable, use simple placement (`_worker_below`). **Never** the probe program. |
| Links | `ttnn.get_forwarding_link_indices(src, dst)` per hop; `num_links` ≤ its length on every hop |
| Connection RT args | `ttnn.setup_fabric_connection(src, dst, link, program, core)` appended **last** to the sender's per-core args |
| Ready semaphore | one cached global semaphore on the union of port cores (per mesh) |
| Per-chip programs | one `ProgramDescriptor` per mesh coordinate, identical kernels, CT args and CB / semaphore lists; only RT args differ. `custom_program_hash` from the plan key (see Key Risks). |

Placement axis (`memory_layout`) seen as TARGET:

| `TensorMemoryLayout` | Axis it cuts | Character | Unlock class | Realization |
|---|---|---|---|---|
| INTERLEAVED | — | — | built | as above |
| HEIGHT_SHARDED (input, and the gathered output) | row | independent | **scheme-change (Refinement 1)**: the shard grid fixes the split (one group per shard, k = 1) when `native_keeps_up`; else round-robin over the accessor fallback | `cb_x_in` / `cb_out_tiles` backed on the shards; ports avoid the shard cores |
| WIDTH_SHARDED | W_local (input) / W (output) | dependent **within the chip** | **scheme-change (Refinement 1)**: k = output shard columns, one group (regime A), when `native_keeps_up`; else the accessor fallback | `cb_out_tiles` backed on the output shard; own tiles gathered from the input shards |

Structural impossibilities: none beyond the INVALID entries already in `feature_spec.py`.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | a block: `block_row_tiles` (= 1) tile-row × the member's landing-tile range |
| Grid | `compute_with_storage_grid_size()` (a runtime value, never a literal). Ports = 2·L per chip. Workers = all cores minus the union of port cores over all chips, row-major. |
| Row groups | `n_active_groups = min(n_workers // k, R_t)`; group g = workers `[g·k, (g+1)·k)`; row r → group `r mod n_active_groups`, block `r div n_active_groups` |
| Per-core work | rows `{g, g + n_active_groups, …} < R_t`; `num_blocks_this_core = ceil((R_t − g) / n_active_groups)`; landing tiles `[w_tile_start, w_tile_start + core_w_tiles)` |
| Remainder | rows: groups differ by ≤ 1 block. W: balanced member widths, `w_pitch` = max, the CB is sized on the pitch. Geometry is per image with `ceil`. |
| Idle cores | not in any kernel's core range |
| Core ranges | worker set and port set built as `CoreRangeSet`s **merged into rectangles** (one CoreRange per core makes the program DISPATCH-BOUND) |
| Common RT args | worker NoC table (packed x, y), `G`, `k`, `n_active_groups`, `R_t`, `Wt_lp`, `out_w_tiles`, `xfer_rows`, `relay_lag`, `run_pages`, `landing tile pitch`; per-core RT args only what differs (member index, group, w_tile_start, blocks, stream unit counts `u[s]`, port roles, schedule entries, connection args) |

## Circular Buffers

T_in = T_out = input tile bytes; T_acc = 4096 (`fp32_dest_acc_en=True`) / 2048 (False); P_g = compact gamma page stride (base op `_gamma_page_bytes`).

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_x_landing` | 0 | T_in | `num_blocks_max · block_row_tiles · w_pitch` (declared first, on workers ∪ ports) | spans the row (all blocks) × W (member range): resident landing | input dtype | reader (credit; bytes written by the reader, by fabric routers and by nobody else) | compute | program |
| `cb_gamma` | 1 | P_g | `w_pitch_out` + base op tail pages (has_gamma) | spans W; row absent | gamma dtype | writer | compute | program |
| `cb_ones` | 2 | bf16 tile | 1 | constant (1.0 reduce scaler for the FPU row collapse) | Float16_b | reader | compute | program |
| `cb_sumsq_tiles` | 3 | T_acc | `block_row_tiles` | spans row of one block | T_acc | compute | compute | pass 1 → collapse |
| `cb_partial_sumsq` (k > 1) | 4 | T_acc | `block_row_tiles` | spans row | T_acc | compute | writer | collapse → exchange |
| `cb_gathered_sumsq` (k > 1) | 5 | T_acc | symmetric: `depth_g · k · block_row_tiles`; owner: base op `cb_plan()` row-form gather tiles | spans row × member | T_acc | writer | compute | exchange → combine |
| `cb_inv_owned` (owner) | 6 | T_acc | base op `cb_plan()` | owned rows | T_acc | compute | writer | owner combine → return |
| `cb_inv_rms` | 7 | T_acc | `block_row_tiles` | spans row | T_acc | compute (k = 1 / symmetric) or writer (owner return) | compute | finalize → end of normalize |
| `cb_x_aligned` (regime N) | 8 | T_in (bf16 when staged) | `compose_depth · out_unit` | streams W | input dtype (Float16_b when staged) | reader | compute | per unit |
| `cb_x_stage` (regime N, bfp8) | 12 | 2048 | `STAGE_ROWS · land_w_tiles` (one landed row) | spans W | Float16_b | compute (copy of the row's landing tiles) | reader (compose) | per row |
| `cb_x_gamma` (regime A, k > 1, gamma, round-robin; Refinement 4) | 13 | T_acc | `block_row_tiles · w_pitch` | spans the block: x·γ packed during the combine wait | T_acc | compute (prescale) | compute (normalize) | one block |
| `cb_out_tiles` | 9 | T_out | `out_depth · out_unit` | streams W | output dtype | compute | writer | per unit |
| `cb_port_chunks` (ports only) | 10 | `run_pages · T_in` | `2 · group` | streams items | input dtype | port reader | port sender | per batch |

`cb_x_landing` exception to "one producer": the bytes arrive by DMA (the reader's own NoC reads and peers' fabric writes). The **credit** has exactly one producer (reader `cb_push_back`) and one consumer (compute). Nobody else reads or writes that CB's pointers. The port core's reader only reads member L1 by address, and declaring the CB on the port core exists only to fix its address.

Push / wait parity: cb_x_landing pushes `block_row_tiles·w_pitch` pages per block (nominal, including the pitch gap of a narrower member), compute waits cumulative per block and pops the block after normalize. Its capacity is exactly `num_blocks_max · block` and every core pushes `num_blocks_this_core` blocks (≤ max), so the ring never wraps. cb_out pushes/pops `out_unit`. cb_gathered per base op. cb_port_chunks: the reader pushes a batch ≤ `group` chunks (`run_pages` pages per chunk slot); the sender pops the same count.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 0 | `compute_kernel_hw_startup` | — | API | cb_x_landing, cb_x_landing, cb_out_tiles | — | first statement of the compute MAIN |
| 1 | `load_gamma_slice` (writer) | `w_pitch_out` pages | port (base op) | gamma DRAM | cb_gamma, pushed once | resident |
| 2 | `load_local_blocks` (reader) | own-slice columns × all blocks | raw dataflow | input | cb_x_landing (bytes only; no credit yet) | — |
| 3 | `await_landing_block` (reader) | one block | raw dataflow | sem_landed[*] | cb_x_landing `+block` credit | regime N: tails zeroed |
| 4 | `sum_squares_block` | `rows × core_w_tiles` (N: `land_w_tiles`) | port of base op raw pass 1 | cb_x_landing, Strided `{base = blk·block·w_pitch, row_stride = w_pitch}`, not popped | cb_sumsq_tiles, `rows` | — |
| 5 | `collapse_sumsq_block` | `of(rows, 1)` | yes (`reduce`) | cb_sumsq_tiles, BulkWaitBulkPop | cb_inv_rms (k = 1) / cb_partial_sumsq | — |
| 6 | `exchange_partials_block` (writer) | `rows` → `k·rows` | raw dataflow (base op) | cb_partial_sumsq (popped) | cb_gathered_sumsq | — |
| 7 | `combine_partials_block` | `of(rows, k)` Skip / owner row combine | yes (`reduce`) / base op owner path | cb_gathered_sumsq | cb_inv_rms | — |
| 8 | `compose_aligned_block` (reader, N) | `out_w_tiles` per row, `out_unit` units | raw dataflow | cb_x_landing (by address) | cb_x_aligned | — |
| 9 | `normalize_block` | `grid(rows, core_w_tiles)` (N: streamed `grid(1, out_unit)` windows) | yes (`eltwise_chain`) + base op `MulGammaRowBcast` | cb_x_landing (None/None, Strided) or cb_x_aligned (PerTile); cb_inv_rms (`BroadcastDim::Col`, None/None, `InputTileMapping::Col`); cb_gamma (row broadcast, Offset by tile column) | cb_out_tiles (PerTile) | caller pops cb_x_landing block and cb_inv_rms |
| 10 | `store_block` (writer) | `out_unit` units | raw dataflow | cb_out_tiles | output | — |

Finalize post-op (in 5 and 7): base op `FinalizeInvRms` (`kernels/rms_norm_compute.cpp:125`): column-0-scoped `x·(1/W) + eps → rsqrt`, with `1/W` from the **logical** `W = G·W_local` and `eps` as fp32-bit RT args.

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| startup | raw_api (boot) | `compute_kernel_hw_startup(icb0, icb1, ocb)` | `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:60` | (cb_x_landing, cb_x_landing, cb_out_tiles) | — | — | — |
| `sum_squares_block` | raw_api (ported) | base op `sum_squares_block` (x·x DEST-accumulated per tile-row, `DEST_AUTO_LIMIT` rows per acquire) | `ttnn/ttnn/operations/base_ops/rms_norm/kernels/rms_norm_compute.cpp:140-193` | `rows`, `full_w = core_w_tiles`, `row_base` | cb_x_landing | cb_sumsq_tiles | `rows` (`block_row_tiles`), `core_w_tiles` |
| `collapse_sumsq_block` | helper | `compute_kernel_lib::reduce<SUM, REDUCE_ROW, cb_sumsq_tiles, cb_ones, out, BulkWaitBulkPop, INPUT_AND_OUTPUT, Fast, ReduceTile>` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp:632`; base op wrapper `rms_norm_compute.cpp:195` | `ReduceInputBlockShape::of(rows, 1)`, post-op = finalize (k = 1) / NoOp | cb_sumsq_tiles | cb_inv_rms / cb_partial_sumsq | `rows` |
| `combine_partials_block` (symmetric) | helper | `reduce<SUM, REDUCE_ROW, cb_gathered_sumsq, cb_ones, cb_inv_rms, BulkWaitBulkPop, …, AccumulateViaAdd, ReduceWithinTile::Skip>` | `reduce_helpers_compute.hpp:632`, `:156`, `:191`; base op `rms_norm_compute.cpp:286` | `of(rows, k)`, finalize | cb_gathered_sumsq | cb_inv_rms | `rows`, `k` |
| `combine_partials_block` (owner) | raw_api (ported) | base op `collapse_rows_to_row0` + `combine_rows_block` | `rms_norm_compute.cpp:226`, `:254` | owned rows | cb_gathered_sumsq | cb_inv_owned | `k` |
| finalize | raw_api inside the reduce hook | base op `FinalizeInvRms` | `rms_norm_compute.cpp:125` | 1/W, eps bits | DEST | DEST | — |
| `normalize_block` | helper (+ raw element) | `eltwise_chain(grid(rows, w), BinaryFpu<Mul, x, input(cb_inv_rms, BroadcastDim::Col, …)>, MulGammaRowBcast<cb_gamma>, PackTile<cb_out_tiles>)` | `ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp:533` (`eltwise_chain`), `:485` (`BinaryFpu`), `:503` (`PackTile`), `:261-266` (`TileAddressing::Strided`, `StridedTileRange`); base op `MulGammaRowBcast` `rms_norm_compute.cpp:322`, `normalize_window` `:361` | x Strided (resident) / PerTile (cb_x_aligned) | cb_x_landing / cb_x_aligned, cb_inv_rms, cb_gamma | cb_out_tiles | shape `grid(block_row_tiles, core_w_tiles)` |
| `cb_ones` fill | helper | `dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<cb_ones, PoolType::SUM, ReduceDim::REDUCE_ROW>()` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp:98-99` | pool-type-aware overload, 1.0 | — | cb_ones | — |
| tail zero-fill (N) | helper | `fill_l1_range` (via base op `zero_w_tail_of_tile`) | `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp:90`; `kernels/rms_norm_reader.cpp:52` | tail byte ranges per face | cb_x_landing | same | — |
| `load_gamma_slice` | raw_api (ported) | base op `GammaSliceLoad` (stick page t instead of a single-row offset) | `kernels/rms_norm_writer.cpp:287` | `w_tile_start`, `core_w_tiles` | gamma DRAM | cb_gamma | `core_w_tiles` |
| `exchange_partials_block` | raw_api (ported) | base op `exchange_send` / `owner_wait_partials` / `owner_return_inv` / `exchange_recv_partials` / `exchange_recv_inv` | `kernels/rms_norm_writer.cpp:477`, `:545`, `:556`, `:620`, `:636` | peers' NoC table | cb_partial_sumsq | cb_gathered_sumsq / cb_inv_rms | `k`, `block_row_tiles` |
| member reads / writes | raw_api (dataflow) | `TensorAccessor`, `noc_async_read` / `noc_async_write`, barriers | `tech_reports/tensor_accessor/tensor_accessor.md` | — | input / cb_out_tiles | cb_x_landing / output | — |
| landing waits, notifications | raw_api | `noc_semaphore_wait_min`, `noc_semaphore_inc` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:1980` | — | — | — | — |
| port reader | raw_api (example, adapted) | example `_READER_SOURCE`: batch / barrier / push discipline, arrival waits | `ttnn/ttnn/operations/examples/fabric_all_gather/program_descriptor_with_inline_kernels.py:89-170` | `run_pages`, `group`; own source = member L1 instead of DRAM | member L1 | cb_port_chunks | `xfer_rows`, `run_pages`, `group` |
| port sender | raw_api (example, adapted) | example `_SENDER_SOURCE`: header ring, `route_one_hop`, ready fence, `to_noc_fused_unicast_write_atomic_inc`, `send_current_slot_non_blocking` | `program_descriptor_with_inline_kernels.py:172-305` | destinations = member L1 on the peer | cb_port_chunks | fabric | `run_pages`, `group` |

Helpers considered and rejected (raw_api entries):

| raw_api entry | Candidate helper | Mismatch (file:line) | Reason |
|---|---|---|---|
| port sender | `dataflow_kernel_lib::ccl::FabricStream` (`arm_unicast_write` / `arm_fused_write_inc` / `signal`) | `ttnn/cpp/ttnn/kernel_lib/ccl/ccl_helpers_dataflow.inl:220-240` (`arm_unicast_write(page_size)` fixes the payload size, `write()` issues the armed size); `ccl_helpers_dataflow.hpp:302` (`unicast_route(num_hops)`: 1-D hop-count routes); `:532` (`signal()` opens, sends and **closes**: terminal) | Our chunks vary in size per packet (balanced row runs, link parts), so a fixed-size armed channel cannot express them. The ready must go out on the same open connection that then carries the data, and `signal()` closes it. The measured transport (73–79% of the link) is the example's 2D route-by-node header ring. |
| port reader | `ccl` helpers | `ccl_helpers_dataflow.hpp:135-147` (“does NOT own: store-and-forward relay, … page<->packet coalescing, … the local barrier wait”) | out of the helper's declared scope |
| `sum_squares_block` | `eltwise_chain` with `DestAccumulation::PerRow` (`chain.hpp:283`) | base op header, `rms_norm_compute.cpp:24-26` | the base op's measured raw path batches `DEST_AUTO_LIMIT` rows per acquire; ported as is |
| `combine_partials_block` (owner), finalize, `MulGammaRowBcast` | `reduce` / `DestReuseBinary` (`chain.hpp:499`) | base op header, `rms_norm_compute.cpp:18-23`, `:213-218` | `DestReuseBinary` hard-codes `BroadcastType::NONE`. `reduce` has no row-form output and no column-scoped SFPU. Both are capability gaps documented and measured in the base op. |
| `exchange_partials_block` | `mcast_pipe` / `ttnn.Mcast` | base op Refinement 2 / Perf 1 | the symmetric k ≤ 8 unicast round is the measured choice. Owner return already uses one multicast where it measured a win (ported as is). |
| `compose_aligned_block` | `tilize` / `untilize` / `reblock_and_untilize` (`untilize_helpers.hpp`, `reblock_untilize_helpers.hpp`) | they move whole tiles / sticks at tile-column granularity | the dense concat shifts datums by `(p·W_local) mod 32` inside a tile row; no helper offsets a tile by a sub-tile column count |

## Broadcast Verification

| Phase | Op | CB_A (semantic name) Valid Region | CB_B (semantic name) Valid Region | Broadcast Dim |
|-------|-----|-----------------------------------|-----------------------------------|---------------|
| sum_squares | Mul (acc to DEST) | cb_x_landing: All (W tails zeroed) | cb_x_landing: All | None |
| normalize step 1 | Mul | cb_x_landing / cb_x_aligned: All | cb_inv_rms: Col0 (REDUCE_ROW output) | `BroadcastDim::Col` |
| normalize step 2 | DestReuse Mul (row bcast) | DEST: All | cb_gamma: Row0 (faces 0/1 row 0) | Row |
| combine (symmetric) | AccumulateViaAdd + Skip | cb_gathered_sumsq: Col0 | — | — |

## Precision

| Item | Decision |
|---|---|
| What crosses the fabric | raw input tiles at the input dtype (≥ the input's own precision). **No partial sum ever crosses the fabric.** |
| Σx² | x·x DEST-accumulated: fp32 when `fp32_dest_acc_en=True`; T_acc pages Float32 |
| In-chip partials (`cb_partial_sumsq`, `cb_gathered_sumsq`, `cb_inv_rms`) | T_acc. These follow the DEST width as in the base op (Refinement 1): Float32 under fp32 DEST. Under `fp32_dest_acc_en=False` the caller asked for 16-bit accumulation; a Float32 page would carry no extra bits (ledger audit 2). The prompt's "fp32 on the wire" rule concerns partials crossing devices, and none do. |
| `float32` + `fp32_dest_acc_en=False` | EXCLUSION (precision convention) |
| Mean divisor | `1/W` with `W = G·W_local` logical |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| A peer writes into my L1 before my program owns it | landing is into worker CBs, not into an output buffer | the per-port `ready` fence (global semaphore) gates every send. My ports send ready only once my program runs, and the dispatcher launches the next program only after **all** cores of this one (ports included) finish, so no relay read of member L1 outlives the program. |
| Landing address mismatch across chips | sender computes the peer's address from its own CB base | `cb_x_landing` first, on workers ∪ ports, identical CB / semaphore lists on every chip; identical worker list (union of port cores excluded); a debug test asserting `get_write_ptr(cb_x_landing)` equals one host-reported value on every core of every chip |
| Program-cache aliasing | `generic_op` keys on kernel source, core ranges, CT args and the RT grid size, not per-core arg counts (`fabric_ring_exchange` gotcha) | `custom_program_hash` from the plan key (G, schedule, L, k, regime, xfer_rows, port placement, shapes, dtypes) |
| Notification race across streams | links / directions notify the same member independently | per-stream `sem_landed[s]` with `(b+1)·u[s]` targets |
| Deadlock in the relay order | a relay waits on upstream items; batches must not hold back an item | flush a batch at every item end (example). Dependencies decrease in (step, entry), so they are acyclic. The final `sem_arrival` wait and its notifications live in the **reader** (the example's sender-side final wait cannot see the reader's later waits). |
| Port placement without a probe | the example's probe is a second dispatch | host-side channel → Ethernet core mapping. A debug test (not the op) compares it once with the example's probe on the QB2 and on a Galaxy if available. |
| Ring on a box without wrap links | QB2 2×2 has no wrap per axis | Ring raises `ValueError` there; the 1×4 / 4×1 snake mocks (`topology.ring_mock_meshes`) exercise it |
| Regime only on some grids | k* depends on n_workers | pinned shapes need only n_workers ≥ 96 (any BH box, |P| ≤ 8) |
| Non-aligned slices landing at non-tile offsets | dense output concat | regime N compose; per-slice tail zero-fill before Σx²; gamma and output stay dense-aligned |
| bfp8 tiles in transport | 1088 B pages | `run_pages = floor(payload / 1088) = 13`; tile bytes move unchanged; regime N on bfp8: the slice tails are zeroed on the bfp8 mantissa bytes (exact zeros), compute unpacks the landed row into a bf16 staging row, the reader composes 2-byte datums from it into bf16 `cb_x_aligned` tiles, and the normalize packs bfp8 (Refinement 2) |
| First call sets up the ready semaphore | a remote ready can race its zero-initialization | blocking initial write or one synchronize at creation, cached per mesh, outside traces |
| `combine_lookahead` from the base op | a lookahead waits on the next block's landing | off in this op (Stall-shadow check) |
| Determinism | regression suite checks bit-identical repeats | fixed reduction orders; no arrival-order-dependent arithmetic |

## Phase 0 implementation notes (implementer, measured on QB2 1x4 ring, FABRIC_2D_TORUS_XY, 14 KiB + 64 B payload)

Built as designed: ports (TRANSPORT) with balanced ring / line schedules, ready fence, relay from member L1, one fused
write + increment per item; slice-nested members, round-robin row groups, per-stream landing notifications; symmetric
combine (k <= K_SYM_MAX = 8) and the owner combine beyond (base op row-form partial, rotated per block); regime N
compose. Host placement maps each connection's Ethernet channel to its core (no probe program): on QB2 the union of
port cores is 8 cores in logical row 0, leaving 102 workers. The split rule picks k = 8 (12 groups, 2 rows per core) for
640x384 / 640x512 and k = 12 owner (8 groups, 3 rows per core) for 640x1792, as in the table above.

Knob turned from the design's Phase 0 value (both values measured, the knob stays live):

| Knob | Design | Now | Measured (fastest chip, eval harness) |
|---|---|---|---|
| `relay_lag` | 1 | **2** | focus 35.25 -> 31.35 us, 640x512 46.13 -> 41.55 us (`ccl_perf compare`: win). Transport-only ablation: lag 1 31.0, 2 26.2, 3 25.9 us |

Breakdown of the focus case at `relay_lag = 2` (ablation hook `AGRMS_ABLATE`, perf probe
`tests/ttnn/unit_tests/operations/all_gather_rms_norm/perf_ablation_all_gather_rms_norm.py`): full 31.4 us, transport
only (members load, signal, wait their landings) 25.5-26.2 us, local work only (no transport) 14.7 us, empty program
0.74 us. The transport binds.

Findings for the refinement queue:

1. TRANSPORT, top item: the ports are packet-rate bound. Device zones show every 4-chunk item taking ~1.6-1.9 us
   on both port RISCs whether it is 48 KB (own slice, 6-tile packets) or 24 KB (far half, 3-tile packets): about
   0.45 us per packet, 40 packets per port on the focus case. The lever is fuller packets: the design's lamp "far half
   split across links by rows" (6-tile packets, 30 packets per port instead of 40), or scatter packets.
2. OVERLAP: on 640x1792 the design's k = 12 (owner) measures 99.8 us against 87.4 us at k = 8 (symmetric); k = 12
   symmetric is 100.5 us and k = 16 is 97.1 us, so the cost is k itself (8 groups give 3 rows per core instead of 2, and
   19-tile members split the 28-tile link ranges into partial packets), not the owner topology. This is the design's
   lamp "KEEP_UP_FRACTION / cost constants: sweep k* +- one candidate". The rule is kept as designed in Phase 0.
3. Test infrastructure: under `--dev` (kernel asserts) Blackhole leaves ERISC0 halted at mesh teardown, so every later
   fabric mesh open in the same process fails (the wrapper's own note; the fabric_all_gather example shows the same).
   Multi-test runs therefore use production mode; single tests pass under `--dev`.

Sharded inputs / outputs take the native shard-resident path when `native_shard_split()` places them and `native_keeps_up()` holds (Refinement 1, see `sharded_zero_copy`); otherwise they go through sharded `TensorAccessor`s (the Phase 0 path).

## Refinement 3 — TRANSPORT: fill the packets (implementer, measured on QB2 1x4 ring, FABRIC_2D_TORUS_XY)

The ports were bound by the port **sender's RISC work per packet**, not by the link. Per-packet device zones on the
focus case: the send call (payload + header NoC writes + credit) takes 0.10 µs, the fabric slot wait 0.03 µs, but
~0.25 µs more went to the port walk's address math before the next packet (per-row div / mod of the round-robin
placement, a per-row scan of all k member boundaries with two runtime-arg reads each, per-member chunk-split
divisions), and ~0.5 µs per item boundary. Kernels build at -Os. Three changes, each a measured win
(`eval.ccl_perf compare`, fastest chip on the focus case):

| Change | Focus fastest chip (µs) |
|---|---|
| baseline (Refinement 2 tree) | 31.47 (TRANSPORT-only 26.37) |
| far slice split between directions by tile-row parity: 6-tile instead of 3-tile packets, 30 instead of 40 packets per port | 29.38 (TRANSPORT-only 24.01) |
| port walk: members outer, rows inner, incremental (group, local row) placement, one chunk split per member per item | 26.98 |
| host-precomputed member segments per send entry in the port runtime args (`_segment_args`: member, landing offset, run length): no per-item scan of the k member boundaries; `ws[0..k]` no longer passed to the ports | **25.08** (TRANSPORT-only 19.81, LOCAL-only 14.72) |

Row-filtered entries: an entry carries `rf` (0 every row, 1 / 2 even / odd tile-rows). An item with no row of its
parity sends nothing and carries no arrival increment; the receiver's counter targets count only non-empty items
(`Geom::items_before`, closed form). Landing notifications became one per row per stream that touches the member at
all (a flag, `n_s ∈ {0, 1}`), since the per-row entry count now varies with parity. The host coverage check runs per row
parity.

Knobs re-swept after the change (all live; defaults kept): `relay_lag` 1 / 2 / 3 → 29.29 / **25.08** / 25.82;
`item_min_chunks` 2 / 4 / 8 → 30.99 / **25.08** / 26.37; `port_cb_bytes` 56 / 112 / 224 KiB → 31.29 (earlier tree) / **25.08** / 26.28.

Transport against bytes: each chip receives 3 × 480 KiB = 1.47 MB; TRANSPORT-only 19.8 µs (members load, signal, wait
their landings), 74 GB/s received per chip. The `fabric_all_gather` example moves 256 tiles per chip on 2 links in
~18.3 µs (29.1 µs × 0.63) on this ring; the gap to it went from ~8 µs to ~1.5 µs.

What still binds the transport (focus, per-packet zones): the sender's per-packet RISC work (~0.25 µs above the
0.10 µs send, 30 packets), the relay tail (the last relayed item waits ~2 µs on its upstream: steps 5–6 carry only
relay items), and the start: the own-slice DRAM load before the first packet (`AG_LOCAL_WAIT` ~1.6 µs, the OVERLAP
lamp "per-row-group readiness", Refinement 4's scope) plus the first batch read (~0.6 µs).

## Refinement 4 — OVERLAP: the exposed tail and the split around the ports (implementer, QB2 1×4 ring, FABRIC_2D_TORUS_XY)

Member zones (`AG_C_SUMSQ` / `AG_C_COMBINE` / `AG_C_NORM` on compute, `AG_W_XCHG` / `AG_W_STORE` on the writer, gated
by `AGRMS_PROFILE_ZONES`) put the exposed tail after the **last landing**, not in a delayed transport (TRANSPORT-only
and full runs have the same last-landing time):
- focus (k = 8, 6-tile members): ~5 µs = ~1.5 µs land → partial exchange complete (the row's latest member gates it) +
  ~0.5 µs combine / finalize + ~1.3 µs normalize (two FPU multiplies per tile) + ~1 µs store;
- 640×1792 (k = 8, 28-tile members): ~11 µs = ~1.6 µs Σx² + ~2.3 µs exchange + ~5.7 µs normalize + ~1 µs store.

What landed:
1. **x·γ pre-scale** in the combine wait (design lamp): the normalize after the combine is one bcast-col multiply of
   `cb_x_gamma`. x·γ is packed at T_acc, so the rounding count equals the fused chain's (x·inv_rms stays in DEST
   there). Regime N streams composed tiles, native shards normalize in segments, and k = 1 has no combine wait, so all
   three keep the fused chain.
2. **Packet-aware split objective** (`select_split`, `packets_per_row`, `PACKET_TIME_SHARE = 0.55`): the estimated
   end of the call replaces the tail alone. Shapes whose candidates cut the link runs into the same number of packets
   select as before (focus, 640×512 L2: k = 8). The changed selections were checked on device where they matter most:
   640×1792 L2 (k = 12 owner → 8) and L1 (k = 28 → 16), 640×512 L1 (k = 8 → 12 owner: 47.30 → 47.02 µs).

Measured and not kept as defaults: `block_row_tiles = 2` (worse, see lamps), `out_unit` 1 / 2 (25.00 / 25.20 vs 25.09
µs: within ~0.1 µs, cap 16 kept), symmetric k = 12 / 16 on 640×1792 (87.2 / 85.2 µs vs 81.5 at k = 8).
Not built: member-direct landing notifications (fusing the increment into each member's last packet puts an atomic on
every packet, against rule 4's router stall; for the tail they would only save the port's notify hop, and they would
un-gate earlier rows, which are not on the critical path here), the compute lookahead (no effect on the last block),
and the Linear line-end notify loop (not on the flagged cases, untouched).

### Refinement 5 deviation — symmetric combine on row-form partials

The base op's symmetric combine ships each member's column-0 partial as faces 0/2 (2 KiB at fp32) to every peer. On the
fused grid every member core sends at the same moment, so the 64 messages per row-group-block (k = 8) were NoC-bound:
zones showed 1.2–2.9 µs per send round. The symmetric path now reuses the owner combine's row form: compute collapses
with `collapse_rows_to_row0` (ones-row × sumsq^T, row 0 = the 32 row sums), the writer ships two 64 B face rows into
row `member_idx` of the row's gather tile on every peer, and every member runs `combine_rows_block` (REDUCE_COL +
transpose_dest + finalize) straight into `cb_inv_rms`. Same k² messages, 1/16 of the bytes, `cb_gathered_sumsq` k× smaller.
Knob `ROW_PARTIALS` (env `AGRMS_ROW_PARTIALS=0`: column form, A/B only). Group-rectangle multicast was not built: the
harvested worker columns (virtual x {1–3}, {5–7}, {11–15} on this QB2) differ per chip, and the CB addresses require
one logical core set on every chip, so a logical rectangle is not a valid multicast rectangle everywhere.

### Perf 1 deviation — pre-send critical path
- Port sender: route header 0 only → `conn.open()` → ready out → (chunk count + headers 1.. as byte copies of header 0,
  in the shadow of the peer's ready) → wait own ready. The ready used to wait behind 1.46 µs of per-header routing.
- Port reader: own items `xb < PORT_INPUT_ITEMS` (2, the port CB's two batches) are read from the input tensor
  (`TensorAccessor`, CT 15+) with the same chunk walk (`for_each_chunk_of_item_src`). `sem_local_loaded` gates only
  the first own item read from the members' landing slots. Native HEIGHT (`own_from_input`) keeps the in-place shard
  read and uses no input items. The own slice is read from DRAM twice for those two items (once by the member, once
  by the port: ~2 × 6 × 4 tiles per port). That costs less than waiting for the member launch skew.
- Fabric multicast of the far slice (instead of relay) does not work on the QB2 ring mock: the snake turns a corner
  every hop. See changelog Perf 1.

### Perf 2 deviation — landing notifications
The port reader's landing notifications cost ~100 ns each, mostly loop overhead: a per-row scan of k member flags
through runtime args and the incremental placement. The NoC atomic itself costs ~24 ns per issue. The reader now walks
a once-built list of the stream's members and places each row directly. It notifies each remaining row-block as soon as
that block's own last item lands, rather than after the whole stream. RELAY_FIRST (relay reads before the step's
notifications) won on the old, expensive fan-out but regresses on the cheap one, so notifications stay before the
relay reads. See changelog "Perf 2".
