# Operation Design: distributed_rms_norm

## Overview

| Field | Value |
|-------|-------|
| Classification | fused CCL op: the row-wise RMSNorm of the base op `rms_norm` over a per-device hidden slice, plus a cross-device all-gather of one fp32 Σx² partial per row over Fabric 2D. **One** `ttnn.generic_op` dispatch per call (one `MeshProgramDescriptor`, one program per chip). |
| Goal | Every device `p` of a group of `G` devices along `cluster_axis` holds `x[p]` of shape `(..., M, W_local)`. It writes `out[p] = x[p] · rsqrt(Σ_g Σ_w x[g]² / W + ε) · γ[p]`, with `W = G · W_local`. The result equals `torch.nn.functional.rms_norm(concat_p x[p], (W,), concat_p γ[p], ε)` sliced back per device. |
| Math | `ss[n,h] = Σ_{q=0}^{G-1} part_q[n,h]` where `part_q[n,h] = Σ_{w<W_local} x[q][n,h,w]²` (the logical W_local only; tile padding never enters). `out[p][n,h,w] = x[p][n,h,w] · γ[p][w] · rsqrt(ss[n,h] · (1/W) + ε)`, γ ≡ 1 when absent. `W` is the true global width `G · W_local`, never a padded one. |
| Mode | Hybrid. The local half is ported from the base op (`ttnn/ttnn/operations/base_ops/rms_norm/`): kernel-lib helpers for every compute phase, custom reader/writer dataflow. The cross-device half is a port-core kernel adapted from `ttnn/ttnn/operations/examples/fabric_ring_exchange/kernels/port.cpp` plus a new root-side exchange client. |
| References | Base op: `ttnn/ttnn/operations/base_ops/rms_norm/{op_design.md, changelog.md, rms_norm_program_descriptor.py, kernels/}`. Transport: `ttnn/ttnn/operations/examples/{fabric_ring_exchange, fabric_all_gather, fabric_reduce_scatter}/` (README, report, code), `.claude/references/blackhole-fabric.md`, `.claude/references/blackhole-ccl-kernels.md`. Blocking: `.claude/references/blocking-model.md`, `.claude/references/l1-footprint-discipline.md`. Precision: `.claude/references/precision_convention.md`. Helpers: `ttnn/cpp/ttnn/kernel_lib/{eltwise/api/chain.hpp, eltwise/unary/{scalar,math}.hpp, eltwise/binary/sfpu/basic.hpp, reduce_helpers_compute.hpp, reduce_helpers_dataflow.hpp, dest_helpers.hpp}`. Catalog (`ttnn/ttnn/operations/examples/master.md`): `fabric_ring_exchange`, `fabric_all_gather`, `fabric_reduce_scatter`, `tensix_all_reduce`, `width_split`, `compute_block_size`, `double_buffer`, `noc_placement`. Golden: `eval/golden_tests/distributed_rms_norm/{feature_spec.py, roofline.py, helpers.py, topology.py, perf_measure.py}`. |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` (mesh) | yes | rank ≥ 2. Per-device shape `(..., M, W_local)` identical on every device. `dtype ∈ {float32, bfloat16, bfloat8_b}`, `Layout::TILE`. DRAM/L1 interleaved, or L1 `HEIGHT_SHARDED` / `WIDTH_SHARDED`. | — | tensor: accessor args CT, address RT |
| `cluster_axis` (kw-only) | `int` | yes | 0 or 1; the mesh size along it is ≥ 2 | — | host: groups, G, p (RT) |
| `gamma` (kw-only) | `Optional[ttnn.Tensor]` (mesh) | no | TILE `(1, 1, 1, W_local)`, or ROW_MAJOR sticks `(1, 1, ceil(W_local/32), 32)`; `dtype ∈ {float32, bfloat16}` (may differ from x); interleaved | `None` | presence CT `has_gamma`; format CT; address RT |
| `epsilon` (kw-only) | `float` | no | finite, ≥ 0 | `1e-6` | RT (fp32 bit pattern) |
| `topology` (kw-only) | `ttnn.Topology` | no | `Linear`, `Ring` | `Linear` | host: exchange plan (RT) |
| `num_links` (kw-only) | `Optional[int]` | no | `1 ≤ num_links ≤ L_usable`. `L_usable` is the minimum over every hop of the group of `len(ttnn.get_forwarding_link_indices(src, dst))`. | `None` (= `L_usable`) | host: port count `2L` |
| `compute_kernel_config` (kw-only) | `ttnn.ComputeConfigDescriptor` | no | any `math_fidelity` / `math_approx_mode`; `fp32_dest_acc_en ∈ {True, False}`, except `{float32, False}` (EXCLUSION) | `default_compute_kernel_config()` | the compute kernel's `config=` (copied, plus `unpack_to_dest_mode[cb_stats_gather]`, see API Mapping) |
| `memory_config` (kw-only) | `Optional[ttnn.MemoryConfig]` | no | interleaved (DRAM/L1), or the input's own shard spec for a sharded input | DRAM interleaved | host-only (output allocation) |

### Entry-point contract

| Item | Decision |
|------|----------|
| Module | `ttnn/ttnn/operations/distributed_rms_norm/`. `__init__.py` exports `distributed_rms_norm`, `default_compute_kernel_config`, `INPUT_TAGGERS`, `SUPPORTED`, `EXCLUSIONS`, `validate`. The `Topology` enum comes from `from ttnn.operations.ccl import Topology` at import time (`ttnn.Topology` is not bound yet). |
| Default config | One exported factory, `default_compute_kernel_config()`. It returns a **fresh** `ttnn.ComputeConfigDescriptor(math_fidelity=HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)`. The entry point resolves `None` through it. Nothing else restates the default. |
| First line | `validate(input_tensor, cluster_axis=…, gamma=…, topology=…, num_links=…, compute_kernel_config=…, memory_config=…)`. |
| Hard validation (ValueError, inside `validate()`, before the support gate) | (1) `len(per-device shape) < 2`. (2) γ shape does not match its layout's format for this W_local: TILE must be `(1,1,1,W_local)`, ROW_MAJOR must be `(1,1,ceil(W_local/32),32)`. (3) `cluster_axis ∉ {0,1}`, or the mesh size along it is 1. (4) `num_links` given and `≤ 0` or `> L_usable`. (5) `topology == Ring` with `G ≥ 3` when **either** the active fabric config does not wrap `cluster_axis` (axis 1 needs `FABRIC_2D_TORUS_X` / `_XY`, axis 0 needs `FABRIC_2D_TORUS_Y` / `_XY`; the 1D ring config wraps both) **or** some group's wrap hop `(last → first)` has no forwarding link. A `Ring` with `G = 2` runs as `Linear` (a 2-chip ring is the line). The fabric config is read with `ttnn.get_fabric_config()` and never changed. |
| Support gate | Builds the axes dict exactly as `eval/golden_tests/distributed_rms_norm/axes.py:classify_call` does: `dtype`, `fp32_dest_acc_en` (from the resolved config), `layout`, `alignment`, `rank`, `gamma_mode`, `gamma_dtype` / `gamma_layout` (`"none"` when γ is absent), `memory_layout`, `cluster_axis`, `topology`, `num_links` (`None` resolved to `L_usable` before the check). Raises `UnsupportedAxisValue` / `ExcludedCell` (`ttnn/ttnn/operations/_op_contract.py`). `"none"` is always accepted. |
| INPUT_TAGGERS | `alignment(inputs, axes)` takes `shape = inputs[0]`, `H = shape[-2]`, `W = shape[-1]` and returns `"tile_aligned"` if `H%32==0 and W%32==0`, `"w_non_aligned"` if `W%32≠0`, else `"h_non_aligned"`. `rank(inputs, axes)` returns `len(inputs[0])`. Exactly these two. |
| EXCLUSIONS | `{dtype: float32, fp32_dest_acc_en: False}` — natively rejected (`precision_convention.md`). `{dtype: bfloat8_b, alignment: w_non_aligned}` — carried from the base op: W-pad columns of a bfp8 tile cannot be zeroed element-wise (the shared exponent), the same mechanism limit the base op hit. |
| Regime refusals (NotImplementedError, not an axis) | (a) A sharded input or output whose shard grid leaves fewer than `2L` free cores for the ports (Work Distribution → Port placement). (b) A sharded input with an output `memory_config` that is not the input's own shard spec, or a sharded output with an interleaved input. (c) The L1 predicate of the selected local regime fails (the base op's refusal). Each one names its deferred regime row. |
| Dispatch | Allocate the output (per-device shape, dtype and `Layout::TILE` as the input; placement per `memory_config`). Fetch or create the persistent exchange resources (Dataflow Strategy → Persistent resources). Build one `MeshProgramDescriptor` (one `ProgramDescriptor` per chip, `custom_program_hash` set per chip). Call `ttnn.generic_op` once. No host padding, slicing, synchronization or second dispatch. |

### Phase 0 support target (TARGET → Phase 0 / later)

The Phase 0 scope is "from-base": TARGET ∩ the base op's SUPPORTED, plus the prompt's Phase 0 values for the three cross-device axes.

| TARGET axis | Phase 0 values | Later values and their cost class |
|-------------|----------------|-----------------------------------|
| `dtype` | `float32`, `bfloat16`, `bfloat8_b` | — |
| `fp32_dest_acc_en` | `True`, `False` (minus the EXCLUSION) | — |
| `layout` | `TILE` | — |
| `alignment` | `tile_aligned`, `w_non_aligned`, `h_non_aligned` (bf8b × `w_non_aligned` excluded) | — |
| `rank` | 2, 3, 4 | — |
| `gamma_mode` | `gamma`, `no_gamma` | — |
| `gamma_dtype` | `float32`, `bfloat16`, `"none"` | — |
| `gamma_layout` | `TILE`, `ROW_MAJOR` (sticks), `"none"` | — |
| `memory_layout` | `INTERLEAVED`; `HEIGHT_SHARDED` (base R1-HS: the cut is along `tile_rows`, independent, so it is a placement knob-turn); `WIDTH_SHARDED` (base R3-sharded: the cut is along `width`, dependent inside the device, so it is a scheme-change, already built in the base and ported here). | — |
| `cluster_axis` | 0, 1 | — |
| `topology` | `Linear`, `Ring` | — |
| `num_links` | 1, 2 | — |

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | per device rank ≥ 2, `(..., M, W_local)`, any `M, W_local ≥ 1`, identical on every device |
| Dtype | `float32`, `bfloat16`, `bfloat8_b` (identical on every device) |
| Layout | `Layout::TILE` |
| Memory | `INTERLEAVED` (DRAM or L1), or L1 `HEIGHT_SHARDED` / `WIDTH_SHARDED` (identical spec on every device) |

### Gamma (optional)

| Property | Requirement |
|----------|-------------|
| Shape | TILE: `(1, 1, 1, W_local)`. ROW_MAJOR: `(1, 1, ceil(W_local/32), 32)` sticks. In the sticks format page `j` holds γ tile-column `j`'s 32 values, with the tail zero-padded. |
| Dtype | `float32` or `bfloat16`, independent of x |
| Memory | interleaved |

### Output

| Property | Value |
|----------|-------|
| Shape | identical per-device shape to the input |
| Dtype | input dtype |
| Layout | `Layout::TILE` |
| Memory | `memory_config` if given, else DRAM interleaved |

## Deviations from the base op

| Base op mechanism | In this op | Why |
|-------------------|------------|-----|
| Grid = the full compute grid | Grid = the compute grid minus `2L` port cores | The cores that drive the fabric do not run the local work (`blackhole-fabric.md` rule 8). |
| R3 owners: `R3_SCATTER_ROWS_PER_OWNER = 1` (reduce-scatter when a group has > 1 row) | `OWNERS_PER_GROUP = 1`: one root per row-group. The root's block rows are cut into waves instead. | The port kernel counts publications as `waves × n_roots` (`port.cpp` `own_stats ≥ (w+1)·n_roots`), so every root must publish in every wave. One root per group with rows cut into waves satisfies that. The base measured the two combine shapes at parity on the 640-row interleaved shapes (`changelog.md` Refinement 6: "640×6144 / 640×7168 interleaved … parity"). Owners > 1 is perf lamp L5. |
| `rms_finalize`: reduce + post-op (×1/W, +ε, rsqrt) in one call, root-local | Split in two. `collapse` (reduce, no post-op) produces the **device partial**. `finalize` runs after the exchange: an SFPU fp32 sum of the G slot tiles plus the same post-op with `1/W_global`. | The cross-device sum sits between the reduction and the post-op. |
| R3 owners = 1: inv_rms multicast (`McastArgs`) to the row-group | Unicast of the inv_rms tiles (faces 0/2) from the root to each member, counted by `SEM_INV` (the base's owners > 1 distribution mechanism) | With the port hole in logical row 0, a row-group of consecutive `(y, x)` cores is not always a rectangle. Unicast works for any core set. Perf lamp L6. |
| γ multicast groups (`GAMMA_MCAST = True`, Perf 1) | Off: every active core reads its own γ window | The base's γ-window groups assume rectangular groups (the port hole breaks that). The base measured γ multicast at noise on the 640-row interleaved shape (`changelog.md` Perf 1: "640×6144 interleaved 42,936 → 43,828 … noise"). This is regime row G-MC (deferred). |
| RM γ: one stick of W elements `(1,1,1,W)` | RM γ **sticks** `(1,1,ceil(W_local/32),32)`: page `j` = γ tile-column `j` | This is the models' weight format. Page `j` is exactly the base's `cb_gamma_row` page `j` (`32·es_g` B), so the reader issues one page read per tile-column of its window instead of one stick read. Everything downstream is unchanged. |
| RM input / RM output paths (`tilize`, `untilize`, `cb_x_rm_sticks`, `cb_y_tiles`) | Not ported | `layout` TARGET is TILE only. |
| R3 all-gather on the writer (`R3_ALLGATHER_ON_WRITER`) | The root's exchange client and the distribution run on the root's reader | The reader is already the base's combine hub (partials in, `cb_gather` out). Its x reads are all issued before the exchange (x resident), so it is idle exactly when the exchange runs. |

## Blocking Model

The first design section. Everything below it is downstream of these decisions.
Semantics: `.claude/references/blocking-model.md`.

### Axes

Definitions (host, once, all `ceil`, per image; per device):

| Symbol | Value |
|--------|-------|
| `W_local` | `shape[-1]` |
| `tensor_width_tiles` (Wt) | `ceil(W_local / 32)` |
| `tensor_tile_rows` (T) | `prod(shape[:-2]) · ceil(shape[-2] / 32)` (each image tile-padded independently) |
| `W_global` | `G · W_local` (the divisor; the RT arg `inv_W_bits = fp32(1 / W_global)`) |
| `G` | devices in the group: `mesh_shape[cluster_axis]` |
| `p` | this device's group position: `c` for `cluster_axis = 1`, `r` for 0 |
| `L` | links per hop: `num_links` or `L_usable` |
| `N_fused` | compute cores left after the ports: `grid.x · grid.y − 2L` (interleaved), or the available set (sharded, Work Distribution) |

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `batch` (dims before H) | independent: the math never mixes leading indices | folded into `tile_rows` | folded | — (leading dims are treated identically and are contiguous in tile-page order) | via `tile_rows` | — |
| `tile_rows` | independent: a row's result depends only on its own W elements across the group | `block_rows` (R) | coarsest R ≤ the core's assignment that fits L1: the base op's `choose_blocking` closed form with this op's extra terms (`l1_ledger.md`) | host `choose_blocking` → CT arg `block_rows`; every R-scaled CB capacity and trip count is derived from it | R1: contiguous tile-row ranges over the fused grid (`split_work_to_cores(…, row_wise=True)` on the available-core list). R3: row-groups of `G_w` cores, `G_h` groups (the base `_r3_interleaved_layout` on `N_fused`). Sharded: the shard spec. | knob-turn (R, grid) |
| `wave` (sub-block of `tile_rows`) | independent: each wave's rows finalize independently once all G partials of those rows landed | `wave_rows` (Rw) | `Rw = ceil(R / num_waves_per_block)`, with `num_waves_per_block = min(R, NUM_WAVES)` and `NUM_WAVES = 2` | host constant `NUM_WAVES` → CT args `wave_rows`, `num_waves_per_block`; the exchange's total `num_waves = num_blocks_max · num_waves_per_block` is derived (port CT arg) | every core of a row-group walks the same waves in the same order. The root publishes one record run per wave. | knob-turn (L1) |
| `width` (W_local), **reduction stage** (`square_accumulate`, `collapse`) | **dependent** (inside the device): Σx² spans the row | `slice_tiles` (S) | R1 / HEIGHT: `S = Wt`, not split. R3-interleaved: `S` from the base `_r3_interleaved_layout(Wt, T, N_fused)`. WIDTH-sharded: the shard width. | host plan (`select_plan`) → CT `S`, RT `w0` | R3: `G_w = ceil(Wt / S)` cores per row-group, combined in-device by the base's root gather (partials → root's `cb_gather`) | scheme-change (built in base, ported) |
| `width`, **streaming stages** (`scale_by_gamma`, `scale_by_inv_rms`, `store`) | independent: once inv_rms exists every output tile depends only on its x tile, its γ tile and its row's inv_rms | `w_chunk_tiles` (C) | `min(S, W_CHUNK_TILES)`, `W_CHUNK_TILES = 32` (the base's final value) | host constant `W_CHUNK_TILES` → CT `w_chunk_tiles`; `num_w_chunks`, `Wp` derived | within the core | knob-turn |
| `device_slot` (q ∈ [0, G)) | **dependent** (across devices): `ss` sums all G slots | `slots_per_root` | **whole extent G**: every root gathers all G slots of its wave rows (G column-0 tiles per tile-row) | CT `G` | one slot per device (fixed by the op's contract). The slots cross the fabric by relay all-gather. | — (the contract fixes it; the transport alternatives are regime rows) |
| `link` (l ∈ [0, L)) | independent: the roots' records are disjoint row sets | `num_links` | `L_usable` or the caller's value | host `num_links` → port count `2L`, per-root `link` RT arg | roots round-robin over links: row-group root `g` uses link `g mod L` | knob-turn |
| `gamma` operand (W_local elements) | **reuse-shared**: γ does not vary along `tile_rows`, the split axis, so every core in a column of slices reads identical bytes | the core's γ window (`S` tiles) | the whole window resident per core (base `GAMMA_RESIDENT`) | derived from S | every active core reads its window once from DRAM (stepping stone; multicast = row G-MC, deferred) | refinement (G-MC) |
| within-tile (32×32) | — (the tile is the unit) | — | — | — | — | partial tiles: the base op's padding rules |

**Per-stage re-run of the table (each stage's own data):**

| Stage output | Its axes | Block it is produced in | Assignment | Note |
|--------------|----------|------------------------|------------|------|
| `partial_col` (member's collapsed Σx² over its slice, column 0) | `tile_rows` (independent) | Rw tiles per wave | each member, sent to its root's `cb_gather` | as in the base R3 |
| `device_partial` (Σ over the G_w slices, column 0) | `tile_rows` | Rw tiles per wave | the row-group root | R1 / HEIGHT: every core is its own root (`G_w = 1`, no gather) |
| `record` (the wire form: 32 fp32 per tile-row) | `tile_rows` × `device_slot` | Rw records per wave per slot | the root publishes slot p; ports move slots; the root collects all G | the only cross-device data |
| `inv_rms` (column-0-valid tile per tile-row) | `tile_rows` | Rw tiles per wave | the root computes it for its group's rows and unicasts it to every member | the root does this **once per wave for its own rows only**. Every group has its own root, so finalize work is spread over `G_h` roots in parallel. No core computes another group's rows. |
| `x·γ`, `out` | `tile_rows` × `width` (both independent) | Rw (wave) × C per chunk | the core that owns the slice | — |
| port-core work (fence, sends, relays, notifies) | `device_slot` × `wave` | one item per (wave, slot) | `2L` port cores, a **communication role and not a work assignment**: ports run no local compute and no DRAM traffic | The ports are data-movement only. Their work is the exchange itself; nothing of the local half is left on them. |

**Scheme committed for Phase 0: X1, the "local base split + per-group roots + relay all-gather of per-row partials, in waves".**

1. Port placement fixes the fused grid.
2. The base op's regime rule (R1 / R3-interleaved / R1-HS / R3-sharded) is re-run on that grid.
3. Every core reads its block of x once and keeps it resident in L1 for both passes.
4. Per wave: Σx² → member partials → the group root's device partial. The root publishes it as a record run into its link's port landing. The ports relay the records over the ring (or line). The root collects all G slots, sums them in slot order in fp32, finalizes, and unicasts inv_rms to its members. Every core scales and writes that wave.

**Every knob (R, Rw / NUM_WAVES, S, C, x_depth, L, the grid and its port carve-out, OWNERS_PER_GROUP) is a parameter with one source. None is an inlined literal.**

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_x_tiles` | `x_depth` | base rule: 2 when `rows_g1 > block_rows` and it fits, else 1 (`x_depth = 1` when every core walks a single block, base Refinement 1) | the reader fetches block k+1 while block k waits on its exchange and scales |
| `cb_gamma_tiles` | `stream_depth` | base: resident window (`GAMMA_RESIDENT`) when it fits, else 2 chunks | — |
| `cb_output` | `stream_depth` | 2 (in C-tile quanta) | compute packs chunk j+1 while the writer drains chunk j |
| `cb_dev_partial` | `partial_depth` | 2 waves (`2·Rw` pages) | compute packs wave w+1's device partial while the reader still publishes wave w |
| `cb_stats_gather` | `stats_depth` | 1 wave (`G·Rw` pages) | — compute consumes a wave's slots immediately (it is the only thing finalize waits on) |
| `cb_partial_col`, `cb_gather`, `cb_rms_root`, `cb_inv_rms`, `cb_sumsq_partial`, `cb_x_gamma` | base sizing (block-wide: R or G_w·R or R·C) | base | inherited: base combine slot safety (`changelog.md` Refinement 4) |
| port landing (L1 tensor, not a CB) | `num_waves` | all waves of the call (every slot written once per call) | no credit protocol: a slot is never reused within a call (`blackhole-ccl-kernels.md` §7) |

`stream_depth` is one host constant (2), as in the base.

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Port kernel publication count `own_stats ≥ (w+1)·n_roots_l` (`fabric_ring_exchange/kernels/port.cpp`) | every root of link l must publish **every** wave | nominal waves: a root whose block has fewer rows (ragged split) or fewer blocks still publishes a (don't-care) record run for every wave index `w < num_waves` | the port never reaches the count → hang. Or, with counts uneven across waves, a fast root's wave-(w+1) increment counts toward wave w before a slow root wrote its wave-w record → stale data sent. The `SEM_OWN_COUNTED` release (Dataflow Strategy) closes the second case. |
| Fabric payload `ttnn.get_tt_fabric_max_payload_size_bytes()` (14 KiB + 64 B on the perf box) | slot bytes per (wave, link) `seg = n_roots_l · Rw · 128 B` | the port splits a slot into `ceil(seg / payload)` packets, the last fused with the arrival increment (`port.cpp` `send_item`) | — (handled) |
| Packet header pool: ≤ 12 headers per RISC (`port.cpp` static_assert, `blackhole-ccl-kernels.md` §14) | distinct destinations per port | relay sends only to the one-hop neighbour: 1 header per port | pool exhaustion |
| Program-local semaphores: ≤ 16 ids per core, and `setup_fabric_connection` takes the lowest free ids (`blackhole-ccl-kernels.md` §2) | semaphore inventory on port and root cores | ids fixed and declared **before** the connection setup: `SEM_GATHER`, `SEM_INV`, `SEM_OWN_STATS`, `SEM_LANDED_FWD`, `SEM_LANDED_BWD`, `SEM_OWN_COUNTED` (6) + 2 per connection on ports | program build fails ("Semaphore id N already in use") |
| Port landing in L1: `num_waves · G · seg_stride` bytes per port core, `seg_stride = align_up(seg, 64)` | `T · G` (total records per port ≈ `T/L · G`) | refuse (NotImplementedError) when `num_waves · G · seg_stride > port L1 budget` (≈ 1.3 MB). With `T ≤ 128` (largest INPUTS) and `G ≤ 8` it is ≤ 128 KiB. | allocation failure |
| fp32 DEST: 4 tiles in half-sync (`dest_helpers.hpp:103` `DEST_AUTO_LIMIT`) | `finalize` holds 2 DEST tiles live (running sum D0 + incoming slot D1) | 2 ≤ 4: the sum is a left fold over slots, never all G in DEST at once | — |
| Lossless fp32 copy into DEST needs `UnpackToDestMode::UnpackToDestFp32` on the source CB (`tilize_helpers.inl:103-113`, `reduce_helpers_compute.hpp:240`) | `cb_stats_gather` → DEST | the op sets `unpack_to_dest_mode[cb_stats_gather] = UnpackToDestFp32` when `fp32_dest_acc_en` | the partials are truncated to tf32 in SrcA (the "fp32 on the wire and when summing" contract is broken) |
| Record wire unit = 32 fp32 = one tile-row's column 0 | rows per record | fixed: one 128 B record per tile-row | — |
| NoC / L1 alignment: 16 B L1, 64 B DRAM (Blackhole) | record writes / reads (128 B, at 128 B-aligned offsets) and `seg_stride` (64 B-aligned) | records are 128 B and slots are 64 B-aligned | misaligned NoC transfers |
| Base op caps (`tilize`/`untilize` CT width, W-pad zeroing, L1 budget, NoC alignment) | as in the base op | as in the base op (`base_ops/rms_norm/op_design.md` → Mechanism caps) | as in the base |

### Regimes

The table enumerates the local regimes (ported) × the cross-device scheme and its alternatives.

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **X1 / R3-i**: interleaved, W-split row-groups + per-group root + relay all-gather | **built** | `memory_layout == INTERLEAVED` and the base occupancy predicate on `N_fused`: `G_w ≥ R3_MIN_GROUP_CORES (4)` and `G_w·G_h ≥ R3_OCCUPANCY_GAIN (2) · R1_active_cores`, or R1's L1 predicate fails | per core `R × S` resident; waves of `Rw × S` | DRAM: x read once, out written once (minimum). γ: each active core reads its S-tile window once. Cross-core (per tile-row): `G_w − 1` faces-0/2 partials to the root + `G_w − 1` faces-0/2 inv_rms tiles back. Fabric: per device, `(G−1)·T` records of 128 B received (the minimum: one fp32 per row per peer). | per block: one combine round + one `square_accumulate` init. Per wave: one publish, one collect, one finalize, one distribution. Larger Rw → fewer of these per row, but less exchange/compute overlap (lamp L1). |
| **X1 / R1**: interleaved, row-parallel, every core its own root (`G_w = 1`) | **built** | `INTERLEAVED` and not R3-i (the base rule) | `R × Wp` resident | DRAM minimum. No cross-core partial traffic. Fabric as above. | as R3-i, minus the combine |
| **X1 / R1-HS**: HEIGHT_SHARDED | **built** (port of the base's Refinement 5) | `HEIGHT_SHARDED` and ≥ 2L free cores outside the shard grids | one block = the whole resident shard (zero-copy `cb_x_tiles`) | x: 0 DRAM crossings (already in L1); output into its own shard | one block: one set of inits per shard |
| **X1 / R3-s**: WIDTH_SHARDED | **built** (port of the base's Refinement 4) | `WIDTH_SHARDED` and ≥ 2L free cores | per core: its shard (`R × S`), `G_w` = shard columns | 0 DRAM for x. The in-device combine as R3-i. | as R3-i |
| **Transport: relay** (one-hop store-and-forward over the ring / line) | **built** | every topology | one item per (wave, slot) | the minimum bytes. Ring: busiest link `ceil((G−1)/2)` slots per wave, farthest slot `G/2` hops. Line: `G−1` slots, `G−1` hops. | — |
| Transport: direct multi-hop unicast | **rejected**: superseded by relay. Measured 1.1–1.6 µs slower on the 4-ring: its fence waits on a 2-hop ready (`fabric_ring_exchange/README.md` result 2). | — | — | same bytes | — |
| Transport: fabric chip-multicast of each slot | deferred: positive reason, relay is the fastest **measured** transport for this payload and multicast is `[unmeasured]` for small payloads (`blackhole-ccl-kernels.md` §4). The port's send items are the only thing that changes, so the structure keeps it reachable. | — | — | same bytes, fewer sender-side packets | — |
| Transport: sum on the wire (reduce-scatter style; each relay adds its partial before forwarding) | **rejected**: superseded by relay all-gather. It puts a compute step on every relay hop of a latency-bound chain, and it makes each device's sum order differ, so chips would not compute bit-identical `ss`. | — | — | `(G−1)` fewer records per relay | — |
| All-gather x itself, then a local rms_norm | **rejected** (dead end): moves `(G−1)·M·W_local·es` over the links (focus: 6.9 MB per chip ≈ 70 µs at 2 links) against 2.5 KiB of partials | — | — | 2700× the link bytes | — |
| **G-MC**: γ read once + multicast per window group | deferred: stepping stone over per-core γ reads (the base's Perf 1 mechanism layers on). Positive reason: measured noise on the 640-row interleaved shape in the base. Its groups must first exclude the port hole. | interleaved with γ | — | removes `(cores per window − 1) · S` γ tile reads | — |
| Sharded input whose grid leaves < 2L free cores | deferred: no INPUTS shape reaches it. `auto_shard_config` uses ≤ 86 of 110 cores across INPUTS (Work Distribution). Reaching it means moving a shard's local work to a neighbour core (one NoC read of that shard), with the rest unchanged. | free cores < 2L | — | +1 NoC read of the displaced shards | — |
| Sharded in / interleaved out (or the reverse) | deferred: the golden suite passes the input's own memory_config for sharded inputs. Reachable by keeping the shard-backed `cb_x_tiles` and swapping `cb_output` for the interleaved writer path, which both exist. | placement mismatch | — | — | — |

**Regime-selection function (host, exact).** `plan = select_plan(input, gamma, fp32_acc, available_cores)` is the base `select_plan` with the grid replaced by the available-core list:
- `HEIGHT_SHARDED` → R1-HS.
- `WIDTH_SHARDED` → R3-s.
- `INTERLEAVED` → R3-i if `R1 is None`, or if `G_w ≥ 4 and G_w·G_h ≥ 2·R1_cores` (TILE); else R1.

`_r3_interleaved_layout` and `_r3_interleaved_work` take `num_cores = N_fused` and the available list instead of `grid.x·grid.y` and the row-major prefix of the full grid. `OWNERS_PER_GROUP = 1` overrides the base's owner count.

Regime-pinned tests are required: the acceptance test pins
- R3-i: `(1,1,32,1792)`, `(1,1,640,1792)`;
- R1: `(1,1,2048,1280)`, `(4,1,512,512)`;
- R1-HS and R3-s: sharded cells;
- multi-wave: 640 rows (2 rows per group, 2 waves);
- multi-block: `(1,1,4096,1024)` interleaved fp32 (more rows than one block holds).

### Traffic ranking

Per device, bytes per tier for `T·Wt` tiles (`T_in` = input tile bytes).

| Candidate split | DRAM crossings (x / out / γ) | Cross-core traffic (on chip) | Cross-device (fabric) | Rank | Decision |
|-----------------|------------------------------|------------------------------|------------------------|------|----------|
| `tile_rows` over cores, rows resident (R1) + record all-gather | 1 / 1 / one row per active core | none | `(G−1)·T·128 B` received | 1 when `T ≥ N_fused` (cheapest on chip) | built (R1) |
| `tile_rows` × `width` (R3-i) + record all-gather | 1 / 1 / one S-window per core | per tile-row: `2·(G_w−1)` half-tiles (2 KiB fp32) | same | 2: strictly more NoC bytes than R1. It wins on occupancy when `T < N_fused`; the base measured +15 % at 640 rows. | built (R3-i), chosen by the base predicate |
| `tile_rows` only, two-pass streaming (re-read x for the scale pass) | 2 / 1 / per core | none | same | 3 | rejected (base R2: dead end) |
| Full-row all-gather of x, local norm | 1 + (G−1) remote / 1 / per core | none | `(G−1)·M·W_local·es` | 4 | rejected (dead end) |

**Occupancy is a separate question** (`blocking-model.md` §4). Decode leaves cores idle: `(1,1,32,1792)` uses 14 of 106. This shape is latency-bound by the exchange (~5.4 µs measured floor) against a 0.45 µs DRAM roofline, and the base measured ~20–30 slice cores as the decode optimum. **The cheapest-traffic split is R1**, and it is implemented where the base predicate picks it. R3-i is implemented where occupancy pays for its extra NoC bytes.

### Block schedule

Logical schedule per compute core. `num_blocks_this_core = ceil(rows_this_core / block_rows)`. `R_b` is the rows in this block (ragged last block). Each block has `num_waves_per_block` waves of `Rw` rows (ragged last wave; nominal publishing).

```cpp
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    load_block(block_idx);                      // reader: R_b x S tiles of x -> cb_x_tiles (resident), wave order
    for (uint32_t wave = 0; wave < num_waves_per_block; ++wave) {
        square_accumulate_wave(block_idx, wave); // Rw x S -> Rw raw sum-of-squares tiles
        collapse_wave(block_idx, wave);          // Rw tiles -> Rw column-0 partials (member) / device partial (R1 root)
        combine_wave(block_idx, wave);           // R3: members' partials -> root's cb_gather -> root sums G_w -> device partial
        publish_wave(block_idx, wave);           // root: device partial -> fp32 records -> its link's port landing (slot p)
    }                                           //   ports, concurrently: ready fence (once per call), relay sends, notify roots
    scale_by_gamma_hoisted(block_idx);          // x*gamma for chunk 0 of the block: needs no remote data (stall shadow)
    for (uint32_t wave = 0; wave < num_waves_per_block; ++wave) {
        collect_wave(block_idx, wave);          // root: wait landed(wave), read G slots' records -> cb_stats_gather (column 0)
        finalize_wave(block_idx, wave);         // root: fp32 sum over slots q = 0..G-1, x(1/W_global), +eps, rsqrt -> cb_rms_root
        distribute_wave(block_idx, wave);       // root -> every member's cb_inv_rms (unicast, SEM_INV)
        normalize_wave(block_idx, wave);        // every core: per chunk [x*gamma] * inv_rms -> cb_output
        store_wave(block_idx, wave);            // writer: the wave's output tiles -> DRAM / output shard
    }
    release_block(block_idx);                   // pop cb_x_tiles (R_b x WS), cb_inv_rms (R_b)
}
```

The ports run independently of this loop. At kernel start each port opens its fabric connection and runs the ready fence: one header-only increment to the peer port that writes into this chip, then a wait for one ready per destination chip. **It needs no local data, so it overlaps `load_block` and `square_accumulate`.** After the fence each port, per wave, sends item 0 (its own slot) once all of its link's roots published, then relays entries as they land. It notifies its roots once per wave per direction when all of that wave's entries on that port have landed. At the end it re-arms its arrival counters.

| Block operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------------|-------------|--------------------|-----------------------------------|
| `load_block` | `R_b × S` tiles into `cb_x_tiles` in C-tile quanta per tile-row, wave 0's rows first | the previous block in the other `x_depth` slot | one read barrier per C-tile quantum. W-pad zeroing once per tile-row (interleaved). Pad tiles zeroed once per kernel. Sharded: one push of the whole shard, no reads. |
| `square_accumulate_wave` | `grid(Rw, Wp)` | `cb_x_tiles` (not popped) | one chain call per wave (one init); DEST acquire/pack once per tile-row; cumulative wait so math starts on the first landed tile |
| `collapse_wave` | `Rw × 1` | `cb_x_tiles` | one `reduce<SUM, REDUCE_ROW>` call per wave (scaler 1.0, no post-op) |
| `combine_wave` (R3) | `Rw` partials per member → root `cb_gather` (`G_w·Rw` pages), root sums → `Rw` device partials | `cb_x_tiles` | per wave: one write + one semaphore increment per member; one root reduce call (`AccumulateViaAdd`, `ReduceWithinTile::Skip`). R1: absent. |
| `publish_wave` (root) | `Rw` records (128 B each) | — | per wave: extract column 0 of `Rw` tiles (RISC-V word copies) → `cb_stats_record`; **one** NoC write of `Rw·128 B` to each of its link's sending ports (≤ 2); one `SEM_OWN_STATS` increment per port (for wave ≥ 1, after `SEM_OWN_COUNTED ≥ n_send_ports·w`) |
| `scale_by_gamma_hoisted` | `grid(R_b, C)` (chunk 0) | `cb_x_tiles` | one chain call per block. It runs before the first inv_rms wait of the block. |
| `collect_wave` (root) | `G·Rw` records → `G·Rw` column-0 tiles | — | per wave: one wait per landed semaphore (one per sending direction); `G−1` NoC reads of `Rw·128 B` (one per peer slot, from the port where it lands); one push of `G·Rw` pages |
| `finalize_wave` (root) | `Rw` tile-rows × `G` slots | — | one chain call per wave: per tile-row a left fold of `G` copies + `G−1` SFPU adds, then `MulUnary`, `AddUnary`, `Rsqrt` on D0, packed to `cb_rms_root` |
| `distribute_wave` (root) | `Rw` inv_rms tiles (faces 0/2) → `G_w` members | — | per wave: `G_w − 1` unicast writes (one per member, all `Rw` tiles in one write) + one `SEM_INV` increment each; a local copy for the root itself. R1: local copy only. |
| `normalize_wave` | `num_w_chunks` sub-blocks of `Rw × C` | `cb_x_tiles`, `cb_inv_rms` (waited cumulatively per wave) | per chunk: `scale_by_gamma` (chunks ≥ 1) and `scale_by_inv_rms` (one chain call each, `SCALE_BLOCK_TILES` DEST blocking from the base) |
| `store_wave` | `Rw × S` tiles | — | one write barrier per C-tile quantum |
| `release_block` | — | — | 2 pops per block |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| L1 **Overlap: wave count** (`NUM_WAVES = 2`) | More waves start the first exchange earlier and let wave 0's writes overlap wave 1's exchange. Each wave adds one publish/collect/finalize round and one port slot. At the focus shape (2 rows per group) 2 is the maximum. At 2048×1280 (R1, 1 row per core) there is only 1 wave and the exchange tail is exposed once. | `NUM_WAVES ∈ {1, 2, 4}`. At 2048×1280: R1 at 32 cores × 2 rows (2 waves) against 64 × 1. Measure the exposed transport time, i.e. op time minus op time with `DRMS_ABLATE_EXCHANGE` (Measurement hooks), on the fastest chip. |
| L2 **Grid synchronization: slice width** | The base `R3_*` constants were tuned on a single chip. Here every group's root also waits on the exchange, so fewer, larger groups (fewer roots, bigger records per root) may be cheaper. | `G_w ∈ {7, 10, 14}` at 640×1792 (`R3_ROWS_PER_GROUP`, `R3_MIN_SLICE_TILES`) |
| L3 **Occupancy at 2048×1280** | The base rule picks R1 (64 of 106 cores, 1 row each). Whether 64 cores saturate DRAM is unmeasured on this grid. | R3-i with `G_w = 3` (`S = 14`, 105 cores, ragged 2/1 rows) |
| L4 **Notify quantum** (TRANSPORT-side notify, OVERLAP effect) | Phase 0 notifies per wave per direction: one `noc_semaphore_inc` per root, `n_roots_l` per wave. At R1 shapes that is 32 atomics per wave per port on the critical path. | notify per entry (`notify_quantum = 1`) so the root reads slot q while q+1 relays; or one multicast semaphore increment over the roots' bounding rectangle |
| L5 **Owners per group** | One root per group serializes the group's sum + finalize + distribution on one core per wave | `OWNERS_PER_GROUP = rows per wave` (the base reduce-scatter); every owner publishes its rows of every wave |
| L6 **inv_rms distribution** | unicast to `G_w − 1` members per wave (≤ 13 writes of 2 KiB at decode) | multicast (`McastArgs`) for groups whose cores form a rectangle |
| L7 **Links** (TRANSPORT) | `num_links = None` uses every usable link. The ring-exchange example measured 2 links equal or slower than 1 for 128 B–8 KiB slots (each link pays its own fence). | the perf case pins `num_links = 2`. Compare against `num_links = 1` on the perf case and report it; the contract keeps `None` = all. |
| L8 **Port placement** (TRANSPORT) | Ports sit on logical row 0, centred (the placement the ring-exchange numbers were measured with), not under their link's Ethernet core. That costs bandwidth (`blackhole-fabric.md` rule 6), which this payload does not use. | probe-based placement (`fabric_all_gather.probe_ethernet_cores`). Its one-time probe is a second dispatch on the first call, so it must move to resource creation outside the op's call path. |
| L9 **Precision under 16-bit DEST** | With `fp32_dest_acc_en = False` the device partial is packed from 16-bit DEST, so the fp32 record carries a 16-bit value. The cross-device sum then folds in 16-bit DEST. | keep an fp32 DEST for `collapse` / `finalize` only. Measure the precision matrix at W_global = 8192 (base L8 found 16-bit Σx² not binding up to W = 8192 on one chip). |

**Stall-shadow check.**
- **Ports.** The ready fence (≈ 5 µs measured, `fabric_ring_exchange/README.md` result 1) waits on the peer chips' launch. Everything local is independent of it: the reads of x, Σx², the in-device combine and the publication of wave 0 all run on the compute cores while the ports fence. A chip's launch skew is absorbed by its ports and never stalls its local work.
- **Root waiting for its slots.** Independent work is scheduled into the window:
  1. the next waves' `square_accumulate` / `collapse` / `combine` / `publish`. The root reader is a non-blocking event loop, so wave w+1's gather and publish never wait behind wave w's collect (Dataflow Strategy).
  2. `scale_by_gamma` of chunk 0 of the block. This is legal because the two scalings commute, `x·γ·s = (x·γ)·s`. The base already orders γ first, so floating-point order is unchanged from the base.
  3. with `x_depth = 2`, the next block's reads.
- **Members waiting for inv_rms.** The same items 1 to 3 apply.
- **Nothing on the write side is independent of inv_rms**, so the last wave's chain is the exposed tail: collect → finalize → distribute → scale → store.

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| DRAM → `cb_x_tiles` (interleaved) | input-dtype tiles | reader, `TensorAccessor` page reads, page id `tile_row · Wt + w0 + j`, C reads per barrier, nominal C-quantum pushes (base) | W-pad zeroing as in the base. Rows of wave 0 are read first. |
| shard → `cb_x_tiles` (sharded) | input-dtype tiles | `ttnn.cb_descriptor_from_sharded_tensor` (zero-copy); reader pushes the shard once | W-pad handled by the compute-side mask (`cb_sumsq_last`, `cb_sumsq_acc`, base R3-s) |
| `cb_x_tiles` → `cb_sumsq_partial` | interm-format tile per tile-row | compute chain FPU `Mul(x,x)` with DEST accumulation per row (base) | lanes not yet collapsed |
| `cb_sumsq_partial` → `cb_partial_col` (member) / `cb_dev_partial` (R1 root) | interm-format tile, column 0 valid | `reduce<SUM, REDUCE_ROW>`, scaler 1.0, **no post-op** | raw Σx² over the core's slice |
| `cb_partial_col` → root `cb_gather` (R3) | faces 0/2 of interm tiles | reader NoC write into the root's gather slot `index·R + r` + `SEM_GATHER` increment (base `rms_norm_combine.hpp`) | the root copies its own partial locally |
| `cb_gather` → `cb_dev_partial` (R3 root) | interm tile, column 0 | `reduce<SUM, REDUCE_COL, …, AccumulateViaAdd, ReduceWithinTile::Skip>(ReduceInputBlockShape::of(G_w, Rw))`: the base root sum, which is an element-wise add of the G_w partial tiles, **without** the finalize post-op | device partial: Σ over the device's W_local |
| `cb_dev_partial` → `cb_stats_record` → port landing | **fp32 records**, 32 words per tile-row (128 B) | root reader. It reads column 0 of each tile: element `(r, 0)` sits at face 0 row r (r < 16) / face 2 row r−16, i.e. word offsets `r·16` / `512 + (r−16)·16` for Float32, half that in 16-bit for Float16_b. It writes 32 contiguous fp32 words. Then one NoC write per sending port of its link into `landing[w][p][root_idx_l][0..Rw)`, a write barrier, and `noc_semaphore_inc(port SEM_OWN_STATS, 1)` per port. | Float16_b page (16-bit DEST): the fp32 word is the bf16 bits `<< 16` (exact). Wire is fp32 for every dtype. |
| port landing → peers (TRANSPORT) | fp32 records | port kernel (`port.cpp`, adapted): relay items, `ceil(seg/payload)` packets per item, the last one fused with the destination entry's arrival increment | Contract: next table |
| port landing → `cb_stats_record` → `cb_stats_gather` | fp32 records → interm tiles (column 0 = slot q's record) | root reader: after `SEM_LANDED_{dir} ≥ (w+1)·n_in_dir` for each sending direction, one NoC read per peer slot q from the port where q lands. Own slot from its local record. Each record is scattered into column 0 of tile `q` of the tile-row's G-tile group (inverse of the extract above; Float16_b: `>> 16`). Push `G·Rw`. | Only column 0 is written. Other lanes are don't-care: the finalize is lane-wise and the scale reads column 0 only (`BroadcastDim::Col`). |
| `cb_stats_gather` → `cb_rms_root` | interm tile, column 0 = inv_rms | compute chain: `CopyTile(slot 0)→D0`, then for `q = 1..G−1`: `CopyTile(slot q)→D1`, `AddBinary(D0,D1→D0)`; then `MulUnary(D0, inv_W_global)`, `AddUnary(D0, ε)`, `Rsqrt(D0)`, pack | **Fixed slot order 0..G−1 on every device**, with identical fp32 inputs, so every device computes a bit-identical `ss` and inv_rms (deterministic run to run). `UnpackToDestFp32` keeps the copy lossless under fp32 DEST. |
| `cb_rms_root` → members' `cb_inv_rms` | faces 0/2 of interm tiles | root reader: one unicast write of the wave's `Rw` tiles into each member's `cb_inv_rms` row offset, then `SEM_INV` increment. The member reader reserves `cb_inv_rms`, waits `SEM_INV ≥ wave count`, pushes `Rw`. | base owners > 1 distribution mechanism |
| γ DRAM → `cb_gamma_tiles` | γ dtype, row 0 valid | writer. TILE γ: base direct row-0 prefix read (`GAMMA_DIRECT_READ`). RM sticks: one page read per tile-column `j` of the window into `cb_gamma_row` page `j`, then the base row-0 expansion. | `BroadcastDim::Row` reads tile row 0 only |
| `cb_x_tiles` × `cb_gamma_tiles` → `cb_x_gamma` | interm | chain FPU `Mul`, Row broadcast (base) | chunk 0 hoisted ahead of the inv_rms wait |
| (`cb_x_gamma` / `cb_x_tiles`) × `cb_inv_rms` → `cb_output` | output dtype | chain FPU `Mul`, Col broadcast (base; `SCALE_BLOCK_TILES` blocking, srcB hold) | the final rounding to the output dtype happens here |
| `cb_output` → DRAM / output shard | tiles | writer (base): valid tiles only, one barrier per quantum. Sharded: `cb_output` backed on the output shard. | — |

### Cross-device exchange contract (TRANSPORT)

Taken from `fabric_ring_exchange` (relay, measured) and `fabric_all_gather` (groups, Linear schedule, link discovery). Labelled TRANSPORT. The surrounding OVERLAP decisions are in the Block schedule and Stall-shadow check.

| Item | Decision |
|------|----------|
| Groups | `build_groups(mesh_shape, cluster_axis)` (`fabric_all_gather/program_descriptor_with_inline_kernels.py:371`). `p` = index in the group. All groups run concurrently in the same dispatch. |
| Ports | `2L` port cores per chip, one per (direction ∈ {fwd, bwd}, link l), **the same logical cores on every chip** (a sender addresses the peer port at its own coordinates). RISCV_0 / NoC1 (`blackhole-fabric.md` rule 7). Line ends keep all `2L` ports; one with nothing to send or receive has `has_conn = 0`. Connection: `ttnn.setup_fabric_connection(src, neighbour, ttnn.get_forwarding_link_indices(src, neighbour)[l], program, core)`, appended **last** in the port's RT args. |
| Schedule: Ring (G ≥ 3) | relay (`fabric_ring_exchange/program_descriptor.py:exchange_plan`, `direct=False`): `kf = G//2`, `kb = G−1−kf`. fwd items `[p, p−1, …, p−kf+1]`, bwd items `[p, p+1, …, p+kb−1]`. Item 0 is the own slot, item i relays entry i−1. Entries landing on chip p's fwd port: `[p−1, …, p−kf]`; on its bwd port: `[p+1, …, p+kb]`. |
| Schedule: Linear (and Ring with G = 2) | relay along the line (`fabric_all_gather` `_schedule` line form). fwd items `[p, p−1, …, 0]` (none at `p = G−1`), bwd items `[p, p+1, …, G−1]` (none at `p = 0`). Entries on fwd: `[p−1, …, 0]`; on bwd: `[p+1, …, G−1]`. |
| Slot layout (landing, per port, per link l) | `landing[w][q]` at offset `(w·G + q)·seg_stride`, with `seg = n_roots_l · Rw · 128 B` and `seg_stride = align_up(seg, 64)`. Inside slot q: root `r_l`'s record run at `r_l · Rw · 128 B`. The plan is identical on every chip (identical shapes and grid), so root `r_l` of chip q owns the same rows as root `r_l` here. |
| Own-slot publication | root `r_l` writes slot `p` on each sending port of link l, then increments that port's `SEM_OWN_STATS` (program-local; `get_semaphore(id)` is the same offset on every core). A port sends item 0 of wave w once `own_stats ≥ (w+1)·n_roots_l`. **Ordering guard (`port.cpp` `n_counted`):** when a port counts wave w it increments every root's `SEM_OWN_COUNTED` by 1. A root increments `own_stats` for wave `w ≥ 1` only after `SEM_OWN_COUNTED ≥ n_send_ports_l · w`. The record write itself does not wait. |
| Arrival | each entry i of a port has its own arrival counter (a **global** semaphore on the port cores, the same address on every chip). The last packet of an item carries a fused `write + atomic_inc` (`port.cpp` `send_item`). A relay of entry j for wave w waits `arrived[j] ≥ w+1`. |
| Notify | port `d` of link l increments `SEM_LANDED_d` on each root of link l by `n_in_d` once per wave (`notify_quantum = n_in_d`), when all of the wave's entries on it have landed. A root waits `SEM_LANDED_d ≥ (w+1)·n_in_d` for each direction with entries. |
| Ready fence | once per call per port with a connection: a header-only `unicast_atomic_inc` to the ready counter of the peer port that writes into this chip, then a wait for one ready per destination chip and a decrement by that count (`port.cpp`; `blackhole-ccl-kernels.md` §6.3). It runs at kernel start, independent of local work. |
| Re-arm | each port decrements every arrival counter by exactly what it consumed (`0u − num_waves`) and ends with `noc_async_atomic_barrier()`. Every global semaphore is back at 0 after every call (trace-safe, `blackhole-ccl-kernels.md` §5.2). |
| Links | link l carries the roots `g` with `g mod L == l`. `L_usable = min over the group's hops of len(get_forwarding_link_indices)`. |
| Persistent resources | Created once per `(mesh id, port core set, G, landing bytes)` and cached in a module-level dict that holds the Python handles (`generic_op` keeps only addresses). They are: the ready global semaphore and `max_entries = G−1` arrival global semaphores over the port cores; the landing scratch, an L1 `HEIGHT_SHARDED` uint32 tensor over the port cores with one shard per port of `num_waves·G·seg_stride` bytes. A single `ttnn.synchronize_device(mesh)` runs **at creation only**, so every chip's counters are 0 before any peer's first increment. Steady-state and traced calls never synchronize. |
| Program cache | `program.custom_program_hash` per chip from every field that changes a kernel's RT-arg layout: per-device shape, dtypes, layouts, memory configs, γ format, plan (regime, S, G_w, G_h, R, Rw, waves), G, p, L, topology, cluster_axis, the port/root coordinates and the landing / semaphore addresses (the `fabric_ring_exchange/README.md` gotcha: `generic_op` keys only the outer size of the RT-arg grid). |

### Root reader as an event loop (OVERLAP)

The root's reader advances five in-order cursors in one non-blocking poll loop. Each takes a step whenever its condition holds, so no wave's step waits behind an earlier wave's exchange:

| Cursor | Condition to step wave w | Action |
|--------|--------------------------|--------|
| gather (R3) | `SEM_GATHER ≥ (block_waves_done + w + 1)·(G_w − 1)` | push `cb_gather` (`G_w·Rw` pages) |
| publish | `cb_dev_partial` has wave w (`cb_pages_available_at_front`), and for `w ≥ 1` `SEM_OWN_COUNTED ≥ n_send_ports·w` | extract the records, write them to the ports, barrier, increment `SEM_OWN_STATS`, pop |
| collect | every direction's `SEM_LANDED_d ≥ (w+1)·n_in_d`, and wave w published, and `cb_stats_gather` has space | read the peer records, scatter them, push `G·Rw` |
| distribute | `cb_rms_root` has wave w | unicast to the members, increment `SEM_INV`, local copy, push `cb_inv_rms` |
| x reads (multi-block) | `cb_x_tiles` has space | the next block's reads |

Members' readers keep the base order: send the partials of every wave of the block, then receive the inv_rms of each wave.

### Measurement hooks (OVERLAP is measured in the op only)

| Hook | Effect |
|------|--------|
| `DRMS_ABLATE_EXCHANGE` (host constant → define on ports and roots; 0 in production) | Ports skip the connection, fence and sends; each port copies its own slot into every peer slot locally and runs arrival/notify/re-arm as usual. Output is wrong, timing is valid. The exposed transport time is `t(op) − t(op, ablated)` on the fastest chip (`perf_measure.py`). |
| Device zones | ports: `prt_conn`, `prt_ready_wait`, `prt_exchange`, `prt_close` (from `port.cpp`); roots: `rt_publish`, `rt_landed_wait`, `rt_collect`, `rt_distribute`; plus the base's per-stage zones. |

## Work Distribution

| Field | Value |
|-------|-------|
| Port placement | Candidates are the cores holding no shard of input or output (interleaved: every core). Take the `2L` candidates with the lowest logical y (closest to the Ethernet row), then nearest the grid's x-centre: interleaved `(x0 + i, 0)` with `x0 = grid.x//2 − L` (the ring-exchange placement). Fewer than `2L` candidates → regime refusal (a). The ports are the same logical cores on every chip. |
| Fused grid | `available = [c for c in (y, x) row-major order over the compute grid if c not in ports]`, `N_fused = len(available)` |
| Work unit | one tile-row (R1, R1-HS) or one tile-row × one W-slice (R3); blocks of `block_rows`, waves of `wave_rows` |
| Split | R1: `split_work_to_cores` semantics over `available` (contiguous tile-row ranges, two groups differing by one row). R3-i: `_r3_interleaved_layout(Wt, T, N_fused)` → `(S, G_w, G_h)`; row-groups are consecutive runs of `G_w` cores from `available`. Sharded: from `shard_spec()` (base `_sharded_work`). |
| Roots | R3: member index 0 of each row-group. R1 / R1-HS: every active core. Root `g` (in group order) → link `g mod L`, index `r_l = g // L` within its link. |
| Per-core work | RT args (base): `start_tile_row`, `rows_this_core`, `w0`, slice tiles, group member coordinates. Fused additions (roots): link ports' coordinates, `r_l`, `n_send_ports`, `n_in_fwd`, `n_in_bwd`, slot→port map for the G slots, landing address, `seg_stride`. Derived on device: `num_blocks_this_core`, `R_b`, waves. |
| Remainder | rows: two `split_work_to_cores` groups / ragged row-groups (base). Ragged last block and last wave at nominal push counts. Ragged last slice (R3). Every root publishes every wave (nominal records). Tile geometry is always `ceil`, per image. |
| Core ranges | every kernel's core set is merged into rectangles (`CoreRangeSet.merge`): interleaved = row 0 left of the ports, row 0 right of the ports, rows 1.. (no per-core `CoreRange` lists; DISPATCH-BOUND guard). |

**The base split rule re-run on the fused grid.** Blackhole QuietBox 2: 11 × 10 = 110 cores, so `N_fused` = 106 at L = 2 and 108 at L = 1.

| Perf case | T, Wt | Regime | (S, G_w, G_h) | Active cores | Rows per group / core | Roots (per link at L=2) | Waves | Seg per (wave, link) | Base on the full grid |
|-----------|-------|--------|---------------|--------------|-----------------------|-------------------------|-------|----------------------|-----------------------|
| 640 × 1792 (focus) | 20, 56 | R3-i | (6, 10, 10); last slice 2 tiles | 100 | 2 | 10 (5) | 2 × 1 row | 640 B | identical (110 // 10 = 11 → S = 6, G_w = 10, G_h = 10) |
| 32 × 1792 (decode) | 1, 56 | R3-i | (4, 14, 1) | 14 | 1 | 1 (1 on link 0; link 1 ports exchange empty slots) | 1 | 128 B | identical |
| 2048 × 1280 | 64, 40 | R1 (G_w = 3 < 4) | — | 64 | 1 | 64 (32) | 1 | 4 KiB | identical (110 // 32 = 3) |

The rule does not fall off a cliff for any perf case: removing 4 (or 2) cores changes none of the three splits. Lamps L2 and L3 name the nearby alternatives. Per-device record bytes at the focus case: 20 × 128 B = **2.5 KiB, the prompt's minimum (one fp32 per row)**.

## Circular Buffers

Symbols: `R` = `block_rows`, `Rw` = `wave_rows`, `S` = slice tiles, `C` = `w_chunk_tiles`, `Wp` = `num_w_chunks · C`, `WS` = `cb_x_tiles` row stride (`Wp` interleaved, `S` shard-backed), `G` = devices per group, `G_w` = cores per row-group. `T_in` / `T_out` / `T_g` = tile bytes of input / output / γ dtype. `T_i` = interm tile bytes (4096 Float32 under fp32 DEST, 2048 Float16_b under 16-bit DEST).

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_x_tiles` | 1 | `T_in` | `x_depth · R · WS` (sharded: shard pages, backed on the shard) | spans the block (R × padded slice): read once, used by both passes | input dtype | reader | compute | load → release, per block |
| `cb_reduce_scaler` | 2 | 2048 | 1 (2 with the sharded W-pad mask) | constant | bfloat16 | reader (once) | compute | kernel |
| `cb_sumsq_partial` | 3 | `T_i` | `R` | one raw Σx² tile per tile-row | interm | compute | compute | square → collapse |
| `cb_inv_rms` | 4 | `T_i` | `R` | one column-0 tile per tile-row; live through every chunk of its wave | interm | reader | compute | distribute → release |
| `cb_gamma_row` (RM γ only) | 5 | `32 · es_g` | `S` | the window's γ sticks (one page per tile-column) | γ dtype | writer | writer (scratch, never pushed) | kernel |
| `cb_gamma_tiles` (γ only) | 6 | `T_g` | resident: `S`; streaming: `stream_depth · C` | the γ window's row-0 tiles | γ dtype | writer | compute | kernel / per chunk |
| `cb_x_gamma` (γ only) | 7 | `T_i` | `R · C` | x·γ of one chunk of the block (chunk 0 hoisted) | interm | compute | compute | per (block, chunk) |
| `cb_partial_col` (R3) | 9 | `T_i` | `R` | this core's collapsed partial per tile-row | interm | compute | reader | collapse → combine |
| `cb_gather` (R3) | 10 | `T_i` | `G_w · R` | the block's partials from every member (contributor-major slots) | interm | reader | compute | per block |
| `cb_rms_root` (root) | 11 | `T_i` | `R` | the block's finalized inv_rms rows before distribution | interm | compute | reader | finalize → distribute |
| `cb_sumsq_last`, `cb_sumsq_acc` (sharded, W % 32 ≠ 0) | 12, 13 | `T_i` | base | base W-pad mask | interm | compute | compute | per tile-row |
| `cb_output` | 16 | `T_out` | `stream_depth · C` (sharded: backed on the output shard) | one C chunk in flight | output dtype | compute | writer | per chunk |
| `cb_dev_partial` (root) | 17 | `T_i` | `2 · Rw` | the wave's device partials (column 0) | interm | compute | reader | collapse/combine → publish |
| `cb_stats_gather` (root) | 18 | `T_i` | `G · Rw` | the wave's G slots, one column-0 tile per (tile-row, slot), slot-major per tile-row | interm (`UnpackToDestFp32` under fp32 DEST) | reader | compute | collect → finalize |
| `cb_stats_record` (root) | 19 | 128 | `(G + 1) · Rw` | the wave's records: G received + 1 own staging run; reader scratch | uint32 (fp32 words) | reader | reader (scratch, never pushed) | per wave |

`cb_dev_partial`, `cb_stats_gather` and `cb_stats_record` are declared on every compute core (uniform CB layout; ≤ `(3G + 2)·Rw·4 KiB` worst case). Full ledger: `l1_ledger.md`.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 0 | kernel prologue | — | yes (`calculate_and_prepare_reduce_scaler`) | — | `cb_reduce_scaler` (reader); pad tiles of `cb_x_tiles` zeroed (interleaved); γ supply (writer) | scaler resident |
| 1 | `load_block` | `R_b × S` | no (base raw reads) | DRAM / shard | `cb_x_tiles`, C per quantum | pushed |
| 2 | `square_accumulate_wave` | `grid(Rw, Wp)` | yes (chain) | `cb_x_tiles` cumulative, not popped | `cb_sumsq_partial` `Rw` | x resident |
| 3 | `collapse_wave` | `Rw × 1` | yes (reduce) | `cb_sumsq_partial` `Rw`; `cb_reduce_scaler` | `cb_partial_col` (R3) / `cb_dev_partial` (R1, R1-HS) `Rw` | — |
| 4 | `combine_wave` (R3) | `G_w · Rw` → `Rw` | yes (reduce, root) + raw NoC (reader) | `cb_partial_col` → root `cb_gather` `G_w·Rw` | `cb_dev_partial` `Rw` | gather drained |
| 5 | `publish_wave` (root) | `Rw` records | no (raw) | `cb_dev_partial` `Rw`, popped | port landing slot p | — |
| 6 | `scale_by_gamma_hoisted` | `grid(R_b, C)` chunk 0 | yes (chain) | `cb_x_tiles` strided; `cb_gamma_tiles` | `cb_x_gamma` `R_b·C` | — |
| 7 | `collect_wave` (root) | `G·Rw` | no (raw) | port landings + own record (`cb_stats_record`) | `cb_stats_gather` `G·Rw` | pushed |
| 8 | `finalize_wave` (root) | `Rw` × G slots | yes (chain) | `cb_stats_gather` `G·Rw`, popped | `cb_rms_root` `Rw` | — |
| 9 | `distribute_wave` (root) | `Rw` tiles → `G_w` members | no (raw) | `cb_rms_root` `Rw`, popped | members' `cb_inv_rms` `Rw` | inv_rms resident until release |
| 10 | `normalize_wave` | chunks of `Rw × C` | yes (chain ×2) | `cb_x_tiles` strided, `cb_x_gamma`, `cb_inv_rms` (Col, caller-managed) | `cb_output` C per quantum | — |
| 11 | `store_wave` | `Rw × S` | no (base raw writes) | `cb_output` | DRAM / shard | popped |
| 12 | `release_block` | — | raw | `cb_x_tiles` `R_b·WS`, `cb_inv_rms` `R_b` | — | empty |
| P | port: fence / sends / relays / notify / re-arm | per (wave, slot) | no (fabric raw API, adapted `port.cpp`) | port landing | peer landing, root `SEM_LANDED_d` | counters at 0 |

## API Mapping

Rows of the base op (`base_ops/rms_norm/op_design.md` → API Mapping) are unchanged for `square_accumulate`, the γ supply, the two scaling chains, `load_block` and `store_wave`, and are not repeated. This table lists the fused-op rows and the rows whose arguments change.

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| compute boot | raw_api (mandatory) | `compute_kernel_hw_startup(cb_x_tiles, cb_x_tiles, cb_sumsq_partial)` | `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:60` | first statement | — | — | — |
| prologue | helper | `dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<cb_reduce_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>()` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp:98` | pool-type-aware; SUM → 1.0, bfloat16 | — | `cb_reduce_scaler` | — |
| `collapse_wave` | helper | `compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, cb_sumsq_partial, cb_reduce_scaler, cb_out, ReduceInputPolicy::BulkWaitBulkPop>(ReduceInputBlockShape::of(Rw, 1), …)`, **no post-op**. `cb_out` = `cb_partial_col` (R3) or `cb_dev_partial` (R1). | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp:632` (reduce), `:290` (shape), `:112` (policy) | — | `cb_sumsq_partial`, `cb_reduce_scaler` | `cb_partial_col` / `cb_dev_partial` | `Rw` |
| `combine_wave` root sum | helper | `reduce<SUM, REDUCE_COL, cb_gather, cb_reduce_scaler, cb_dev_partial, BulkWaitBulkPop, INPUT_AND_OUTPUT, ReduceFp32Mode::Fast, AccumulateViaAdd, ReduceWithinTile::Skip>(ReduceInputBlockShape::of(G_w, Rw), ReduceInputMemoryLayout::with_row_stride(Rw), NoAccumulation{})` (the base owner path) with the finalize post-op **removed**, packing to `cb_dev_partial` | `reduce_helpers_compute.hpp:632`; base `kernels/rms_norm_compute.cpp:848-862` | `G_w` partials per row | `cb_gather` | `cb_dev_partial` | `Rw`, `G_w` |
| `finalize_wave` | helper | `compute_kernel_lib::eltwise_chain(IterationShape::tiles(Rw), CopyTile<input(cb_stats_gather, …)>{} /*slot 0 → D0*/, [CopyTile<…, Dst::D1>{} /*slot q*/, AddBinary<Dst::D0, Dst::D1, Dst::D0>{}] × (G−1), MulUnary<Dst::D0>{inv_W_global_bits}, AddUnary<Dst::D0>{eps_bits}, Rsqrt<Approx::Exact /*follows math_approx_mode*/, Dst::D0>{}, PackTile<output(cb_rms_root)>{})` with `G` compile-time. If the chain cannot index the G slots of one tile-row (slot `q` = tile `t·G + q` of the waited window), the implementer builds a thin block operation with the same contract: G-slot left fold, lossless copy, one init per wave. | `ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp:483` (CopyTile), `:533` (eltwise_chain); `eltwise/binary/sfpu/basic.hpp:20` (AddBinary); `eltwise/unary/scalar.hpp:26,32` (AddUnary, MulUnary); `eltwise/unary/math.hpp:38` (Rsqrt) | `unpack_to_dest_mode[cb_stats_gather] = UnpackToDestFp32` when `fp32_dest_acc_en` (`ComputeConfigDescriptor.unpack_to_dest_mode`, `ttnn/cpp/ttnn-nanobind/program_descriptors.cpp:679`) | `cb_stats_gather` | `cb_rms_root` | `Rw`, `G` |
| `scale_by_gamma_hoisted`, `normalize_wave` | helper | base `scale_by_gamma` / `scale_by_inv_rms` chains over `grid(Rw or R_b, C)` | base API Mapping (`chain.hpp:232-237, 261-266, 305-310`) | `SCALE_BLOCK_TILES` (base) | base | base | `Rw`, `C` |
| `combine_wave` member send, `distribute_wave` | raw_api | `noc_async_write`, `noc_async_write_barrier`, `noc_semaphore_inc`, `noc_semaphore_wait_min`, `cb_reserve_back` / `cb_push_back` (base `rms_norm_combine.hpp` faces-0/2 writes) | `tt_metal/hw/inc/api/dataflow/dataflow_api.h`; base `kernels/rms_norm_combine.hpp` | one write per member per wave | `cb_partial_col` / `cb_rms_root` | root `cb_gather` / members' `cb_inv_rms` | `Rw` |
| `publish_wave`, `collect_wave` | raw_api | L1 word loads/stores (column-0 extract/scatter), `noc_async_write` / `noc_async_read` of `Rw·128 B` runs, `noc_semaphore_inc(port SEM_OWN_STATS)`, `noc_semaphore_wait_min(SEM_LANDED_d / SEM_OWN_COUNTED)` with `invalidate_l1_cache()` in the poll loop | `dataflow_api.h`; `blackhole-ccl-kernels.md` §5.2 | — | `cb_dev_partial`, `cb_stats_record` | port landing / `cb_stats_gather` | `Rw` |
| port | raw_api | `WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>`, `PacketHeaderPool::allocate_header`, `fabric_set_unicast_route`, `to_noc_unicast_write`, `to_noc_fused_unicast_write_atomic_inc`, `to_noc_unicast_atomic_inc`, `send_current_slot_non_blocking`, `send_payload_flush_blocking_from_address`, `noc_async_writes_flushed` | `ttnn/ttnn/operations/examples/fabric_ring_exchange/kernels/port.cpp` (whole file); `tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp`; `tt_metal/fabric/hw/inc/packet_header_pool.h` | `G`, `num_waves`, payload, `MAX_HEADERS` CT | port landing | peer landing | `num_waves`, seg |
| host fabric | raw_api | `mesh.get_fabric_node_id`, `ttnn.get_forwarding_link_indices`, `ttnn.setup_fabric_connection`, `ttnn.get_tt_fabric_max_payload_size_bytes`, `ttnn.get_fabric_config`, `ttnn.create_global_semaphore`, `ttnn.get_global_semaphore_address` | `fabric_ring_exchange/program_descriptor.py`; `fabric_all_gather/program_descriptor_with_inline_kernels.py:371,717` | — | — | — | — |

**Helpers considered and rejected (raw_api rows):**

| Raw use | Candidate helper | Mismatch (file:line) | Why it cannot express the shape |
|---------|------------------|----------------------|---------------------------------|
| port sends / relays | `kernel_lib/ccl/ccl_helpers_dataflow.hpp` `FabricStreamSender` | `.claude/references/blackhole-ccl-kernels.md` §12 (status `[unmeasured]`, 1D routes by hop count) | The measured transport this design cites is `port.cpp`'s raw sender. The CCL helper is unmeasured and has no relay-with-arrival-counter item model. Swapping it in would make the cited numbers inapplicable. |
| publish / collect (column-0 extract/scatter + small NoC runs) | `read_sticks_for_tilize` / `write_sticks_after_untilize`; `prepare_zero_tile` | `tilize_helpers_dataflow.hpp:46-93`; `l1_helpers.hpp:62` | stick helpers move row-major pages. No helper moves a face-strided column of a tile into a contiguous run. |
| inv_rms distribution | `mcast_pipe.hpp` `SenderPipe` / `McastArgs` | `ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_pipe.hpp:55` | multicast needs a rectangle of receivers. Row-groups around the port hole are not rectangles (lamp L6 keeps it for rectangular groups). |
| member partial send | base `rms_norm_combine.hpp` (already a raw helper of the base) | base `kernels/rms_norm_combine.hpp` | inherited unchanged |

## Broadcast Verification

| Phase | Op | CB_A (semantic name) Valid Region | CB_B (semantic name) Valid Region | Broadcast Dim |
|-------|-----|-----------------------------------|-----------------------------------|---------------|
| square_accumulate | Mul (DEST-accumulate) | `cb_x_tiles`: All (W-pad zeroed / masked) | `cb_x_tiles`: All | None |
| collapse | reduce SUM REDUCE_ROW | `cb_sumsq_partial`: All | `cb_reduce_scaler`: Row0 | — (output Col0) |
| finalize | SFPU AddBinary (DEST) | `cb_stats_gather` slot q: Col0 | `cb_stats_gather` slot q': Col0 | None (lane-wise; only Col0 consumed) |
| scale_by_gamma | Mul | `cb_x_tiles`: All | `cb_gamma_tiles`: Row0 | Row |
| scale_by_inv_rms | Mul | `cb_x_gamma` / `cb_x_tiles`: All | `cb_inv_rms`: Col0 | Col |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Wrong divisor | `W_local` padded, or the per-device W used instead of the global one | `inv_W_global_bits = fp32(1/(G·W_local))`; W-pad zeroed/masked (base) |
| Chips disagree on `ss` | each chip summing its own partial first gives different rounding per chip | every root sums slots in the fixed order 0..G−1 from identical fp32 records, including its own slot, so `ss` is bit-identical on every device |
| Publication race across waves | `own_stats` is cumulative | `SEM_OWN_COUNTED` guard (Cross-device exchange contract) |
| Global semaphores not at 0 | a later call reads early or hangs; traced calls never reset them | re-arm by decrement (`port.cpp`), one-time creation sync, handles cached so the L1 stays alive |
| Program cache aliasing | `generic_op` keys only the outer RT-arg grid size; shapes/plans with the same core grid but a different per-core layout would share a program | `custom_program_hash` per chip covering the plan (Cross-device exchange contract) |
| Port cores colliding with shards | `auto_shard_config` fills from (0,0), i.e. logical row 0 first | ports are chosen among cores outside both shard grids; refusal otherwise |
| Port landing clobbered by CBs | ports are inside the compute grid | no CBs on port cores; the landing is an allocator-owned L1 buffer (allocation validates against CB ranges) |
| Root event loop deadlock | a blocking wait on wave w's landing while compute waits on wave w+1's gather | non-blocking poll loop with independent cursors (Root reader event loop) |
| Line end chips | p = 0 has no bwd sends; p = G−1 no fwd sends | `n_send_ports`, `n_in_d` per chip from the host plan; idle ports run `has_conn = 0` |
| `num_links = None` with heterogeneous hops | links can differ per hop | `L_usable = min over every hop of every group` |
| Ring on a non-wrapping config / axis | the wrap hop has no route | ValueError before dispatch (Entry-point contract) |
| fp32 partials truncated | FPU reads fp32 through tf32 | finalize on SFPU with `UnpackToDestFp32` |
| 16-bit DEST cells | the partial is 16-bit before the wire | lamp L9; the wire stays fp32 (exact bf16 → fp32 widening) |
| Garbage H-pad rows | their partials travel and finalize too | lane/row-local math; never stored to logical positions; NaN in pad lanes cannot reach valid rows |
| First call includes resource creation | creation does a one-time `synchronize_device` | documented: the first call per (mesh, shape class) is a warm-up; traces are captured after it (as `perf_measure.py` does) |

### Structural impossibilities

None missing from `feature_spec.py` INVALID.


## Implementation notes (Phase 0, ttnn-implementer)

What the implementation does differently from the text above, and why. Everything binding (algorithm, regimes,
port/relay topology, root event loop, one dispatch) is as designed.

| Item | Design | Implemented | Why |
|------|--------|-------------|-----|
| `MIN_BLOCKS_PER_CORE` | base rule (2: interleaved cores with ≥ 2 rows walk ≥ 2 blocks) | **1** (knob kept), `x_depth = 1` for single-block cores | Each block is one exchange round that has to finish before the next block's Σx² starts, so blocks serialize the exchange; the block's waves already overlap reads, exchange and scale. Measured (1×4 ring, 2 links, fastest chip): 640×1792 30.92 → 23.93 µs (slowest 31.46 → 25.78), the other perf cases unchanged. |
| Waves per block | `nwpb = min(R, NUM_WAVES)`, `Rw = ceil(R / nwpb)` | `nwpb` = the largest divisor of `R` ≤ `NUM_WAVES` (≤ 8), `Rw = R / nwpb` | A full block's waves are then all `Rw` rows, so every per-wave CB quantum divides its capacity (a ragged wave in a full block would push across the ring end). |
| Gather semaphore | `SEM_GATHER` (one) | one `SEM_GATHER_k` per wave of a block | `G_w − 1` members send each wave. On one cumulative counter a fast member's wave-(k+1) increment satisfied the wave-k threshold before a slow member's wave-k partial landed: stale wave-0 rows (`test_distributed_rms_norm_debug.py`). |
| Own slot in `collect` | from the root's own record | read back from its own port landing slot `p` | The event loop may publish wave w+1 before it collects wave w, so a single own-staging run is overwritten. The landing slot is per wave and holds the same bytes the peers receive. |
| R1 finalize target | `cb_rms_root` → local copy → `cb_inv_rms` | packs straight into `cb_inv_rms` | No members to distribute to; one copy saved. |
| x loads | per C-quantum, wave 0's rows first | one barrier per tile-row inside the reader's poll loop (row order = wave order) | Rows are the wave unit; one barrier per row is the base's `READ_BATCH_CHUNKS = None`. |
| Interleaved pad tiles | zeroed once per ring slot | not zeroed; Σx² over the `V` valid tiles only | Pad output tiles are never written; the W-pad columns of the last valid tile are still zeroed. |
| γ supply | base resident / streaming / staging row | resident only; TILE: row-0 prefix read; RM sticks: stick read into the page + 16-word face-1 move | No `cb_gamma_row`. Streaming γ is not built; the L1 solve includes the resident window. |
| `DRMS_ABLATE_EXCHANGE` | ports copy their own slot locally | ports skip the fabric and bump each arrival counter by `num_waves` up front | Same purpose (timing only). The harness copy `tests/.../perf_ablate.py` skips the golden check. |

Measured Phase 0 (1×4 ring, 2 links, `perf_measure.py`, 5 experiments, fastest / slowest chip, µs):
640×1792 23.93 / 25.78 (target 11.9); 32×1792 12.13 / 13.51 (target 0.6); 2048×1280 44.10 / 47.30 (target 27.3).
Exposed transport, i.e. op time minus `DRMS_ABLATE_EXCHANGE` time on the fastest chip, measured at
`MIN_BLOCKS_PER_CORE = 1` before the gather-semaphore fix: 640×1792 ≈ 3.0, 32×1792 ≈ 1.7, 2048×1280 ≈ 9.6. The local
half dominates every perf case. 2048×1280 runs R1 with one row per core (no overlap of read, exchange and write);
lamp L3 (R3-i, `G_w = 3`, 105 cores) is the nearby split to measure.
