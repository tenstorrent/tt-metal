# Operation Design: distributed_rms_norm

## Overview

| Field | Value |
|-------|-------|
| Classification | fused CCL: a row-wise normalization (one row reduction, two element-wise scalings) whose reduced axis, the hidden dim W, is split across the G devices of a mesh group. Runs as **one** `ttnn.generic_op` (one `MeshProgramDescriptor`, one `ProgramDescriptor` per chip) per call |
| Goal | `out[p] = x[p] / sqrt(Σ_g Σ_w x[g]² / (G·W_local) + ε) · γ[p]` on every device p of every group. The only cross-device data is each device's per-row partial Σx²: fp32 on the wire and when summed. Local work is spread over the whole Tensix grid. x crosses DRAM once (read) and the output once (write) |
| Math | `ss[..., m] = Σ_{g<G} Σ_{w<W_local} x[g][..., m, w]²`; `inv[..., m] = rsqrt(ss[..., m] · (1/(G·W_local)) + ε)`; `out[p][..., m, w] = x[p][..., m, w] · γ[p][w] · inv[..., m]` (γ ≡ 1 when absent). The divisor is the true global W = G·W_local, never a padded width |
| Mode | Hybrid. Every compute phase uses kernel-lib helpers, adapted from the base op `rms_norm`. Dataflow (DRAM reads/writes, in-chip combine transport, Fabric exchange) is custom kernels adapted from the base op and the `fabric_all_gather` example |
| References | Base op `ttnn/ttnn/operations/base_ops/rms_norm/` (`op_design.md`, `changelog.md`, `rms_norm_program_descriptor.py`, `kernels/`): regimes R1 and R3, the combine transport, the γ direct read, `FINALIZE_COL0`. `ttnn/ttnn/operations/examples/fabric_all_gather/` (`plan()`, `_schedule()`, `_needs_ready()`, the port sender, ready fence). `ttnn/ttnn/operations/examples/fabric_reduce_scatter/` (arrival counters, ready fence). `references/blackhole-fabric.md`, `references/blackhole-ccl-kernels.md`, `references/precision_convention.md`. `.claude/references/blocking-model.md`, `l1-footprint-discipline.md`, `op-design-template.md`. Catalog `ttnn/ttnn/operations/examples/master.md` entries `tensix_all_reduce`, `compute_block_size`, `double_buffer`, `noc_placement`, `row_reduce_accumulate`, `sfpu_tile_scope`, `compute_fusion`, `shared_input_reuse`, `width_split`. Feature spec: `eval/golden_tests/distributed_rms_norm/feature_spec.py`, `roofline.py` |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | mesh `ttnn.Tensor` | yes | per device `(..., M, W_local)`, rank ≥ 2; identical per-device shape / dtype / placement on every device. Phase 0: `dtype ∈ {float32, bfloat16}`, `Layout::TILE`, `TensorMemoryLayout::INTERLEAVED` (DRAM or L1), tile-aligned | — | accessor args CT, base address RT |
| `cluster_axis` (kw) | `int` | yes | `0` or `1`; the mesh axis must have size ≥ 2 | — | host only (selects the group; the group size G is a CT arg) |
| `gamma` (kw) | `Optional[ttnn.Tensor]` | no | per device: TILE `(1, 1, 1, W_local)`, or ROW_MAJOR sticks `(1, 1, ceil(W_local/32), 32)`. dtype `float32` / `bfloat16` (may differ from the input). Phase 0: TILE only | `None` | presence CT (`has_gamma`), address RT |
| `epsilon` (kw) | `float` | no | finite, ≥ 0 | `1e-6` | RT (fp32 bit pattern) |
| `topology` (kw) | `ttnn.Topology` (import `from ttnn.operations.ccl import Topology`) | no | `Linear`, `Ring` | `Linear` | host only (selects the port schedule) |
| `num_links` (kw) | `Optional[int]` | no | `None`, or `1 ≤ n ≤ L_avail` (links discovered on every hop of the group) | `None` (= `L_avail`) | host only |
| `compute_kernel_config` (kw) | `ttnn.ComputeConfigDescriptor` | no | Phase 0: `fp32_dest_acc_en=True`. `math_fidelity` / `math_approx_mode` are not gated | `default_compute_kernel_config()` | the compute kernel `config=`, plus the op's `unpack_to_dest_mode` entry for `cb_stats_slots` |
| `memory_config` (kw) | `Optional[ttnn.MemoryConfig]` | no | Phase 0: interleaved (DRAM or L1) | `ttnn.DRAM_MEMORY_CONFIG` | host only (output allocation) |

### Entry-point contract

| Item | Decision |
|------|----------|
| Files | `ttnn/ttnn/operations/distributed_rms_norm/__init__.py` exports `distributed_rms_norm`, `default_compute_kernel_config`, `INPUT_TAGGERS`, `SUPPORTED`, `EXCLUSIONS`, `validate`. `distributed_rms_norm.py` holds the registry and the entry point. `distributed_rms_norm_program_descriptor.py` holds the plan, the knobs (module constants) and the `MeshProgramDescriptor` build. Kernels live in `kernels/`. |
| Default config | One exported factory `default_compute_kernel_config()` returns a fresh `ComputeConfigDescriptor(math_fidelity=HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)`. Resolve with `cfg = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()`. Nothing else restates the default. |
| Config the kernel gets | A **new** descriptor with the caller's `math_fidelity`, `fp32_dest_acc_en`, `math_approx_mode` (and `dst_full_sync_en` if set). `unpack_to_dest_mode[cb_stats_slots] = UnpackToDestFp32`, so the cross-device sum reads exact fp32. The caller's object is never mutated. |
| Order inside `validate()` (first line of the entry point) | **1. Caller errors (`ValueError`)**: rank < 2; `cluster_axis ∉ {0, 1}`; mesh axis size 1; gamma shape not equal to its layout's format; `num_links ≤ 0` or `num_links > L_avail`; `Ring` when G ≥ 3 and the fabric config does not wrap `cluster_axis` or a wrap hop has no links. **2. Support gate**: build the axes dict exactly as `eval/golden_tests/distributed_rms_norm/axes.py:classify_call` does: `dtype`, `fp32_dest_acc_en`, `layout`, `alignment`, `rank`, `gamma_mode`, `gamma_dtype`, `gamma_layout`, `memory_layout`, `cluster_axis`, `topology`, `num_links`. `gamma_dtype` / `gamma_layout` are `"none"` without gamma. `num_links` is gated only when given explicitly; `None` resolves to `L_avail`. Raise `UnsupportedAxisValue` / `ExcludedCell` (`ttnn/ttnn/operations/_op_contract.py`). `"none"` is always accepted. |
| `L_avail` | `min` over every hop the topology uses (both directions; the wrap hop for Ring) of `len(ttnn.get_forwarding_link_indices(src, dst))` under the active fabric config. Never hard-coded. |
| Ring wrap check | 2D family: `FABRIC_2D_TORUS_XY` wraps both axes, `FABRIC_2D_TORUS_X` wraps axis 1, `FABRIC_2D_TORUS_Y` wraps axis 0 (`blackhole-fabric.md` §4). 1D family: `FABRIC_1D_RING`. A Ring with G = 2 runs the line schedule; the hop pair is identical, so there is no wrap check. |
| INPUT_TAGGERS | Exactly two, both on `inputs[0]` (the per-device shape). `alignment` returns `"tile_aligned"` (H % 32 == 0 and W_local % 32 == 0), `"w_non_aligned"` (W_local % 32 != 0), or `"h_non_aligned"` (only H misaligned), with H = shape[-2] and W_local = shape[-1]. `rank` returns `len(inputs[0])`. `gamma_dtype` / `gamma_layout` are **not** taggers: they are TARGET axes read from the gamma tensor. |
| EXCLUSIONS | `{dtype: float32, fp32_dest_acc_en: False}` (`precision_convention.md`) |
| Regime refusal | If `block_rows = 1` does not fit L1 (Work Distribution), raise `NotImplementedError` naming the L1 predicate. No INPUTS shape reaches it. |
| Output `memory_config` | A sharded `memory_config` raises `NotImplementedError` in Phase 0 (it arrives only with sharded inputs, which the `memory_layout` gate refuses first). |
| Dispatch | Allocate the output: same per-device logical shape, dtype and `Layout` as the input, placed per `memory_config`. Build one `MeshProgramDescriptor` and call `ttnn.generic_op` **once**. No `to_layout` / `pad` / `slice` / host round trip, no other device program. Persistent resources (global semaphores, the landing scratch) are created on the first call for a (mesh, port-core set, size) key and cached for the process. They are never freed, so traced programs keep valid addresses. Creating them is a host buffer write plus one `ttnn.synchronize_device` on that first call only, not a device-program dispatch. No Ethernet-core **probe program** runs: placement is host-derived (Work Distribution). |

### Phase 0 support target (TARGET → Phase 0 / later)

| TARGET axis | Phase 0 values | Later values and their cost class |
|-------------|----------------|-----------------------------------|
| `dtype` | `float32`, `bfloat16` | `bfloat8_b`: knob-turn (CB page format of `cb_x_tiles` / `cb_output`; intermediates stay Float32) |
| `fp32_dest_acc_en` | `True` | `False` (bf16 / bf8b only): knob-turn. Every intermediate page format is derived from this flag (ledger audit 2). The cross-device sum stays fp32 on the wire whatever the flag |
| `layout` | `TILE` | — |
| `alignment` | `tile_aligned` | `h_non_aligned`: knob-turn. Rows are independent; pad rows are computed and never stored. The tile-row count is already `ceil` per image. `w_non_aligned`: knob-turn. The reader zeroes the W-pad columns of tile `Wt−1` (base op padding rules); the divisor is already `G·W_local` |
| `rank` | 2, 3, 4 | — |
| `gamma_mode` | `gamma`, `no_gamma` | — |
| `gamma_dtype` | `float32`, `bfloat16`, `"none"` | — |
| `gamma_layout` | `TILE`, `"none"` | `ROW_MAJOR` sticks: knob-turn. Stick r is exactly row 0 of γ tile r. The writer reads it as two 16-element segments into face 0 / face 1 row 0 of `cb_gamma_tiles` page r (base op `GAMMA_DIRECT_READ = SEGMENTS`) |
| `memory_layout` | `INTERLEAVED` | `HEIGHT_SHARDED`: knob-turn (placement). Each shard is whole local rows, so the local partial stays per core: regime R1-X with `cb_x_tiles` / `cb_output` zero-copy over the shards (base R1-HS). `WIDTH_SHARDED`: knob-turn. It splits W_local across cores, and that is exactly the in-chip combine R3-X already builds; row-groups are read off the shard grid (base R3 sharded) |
| `cluster_axis` | 0, 1 | — |
| `topology` | `Linear`, `Ring` | — |
| `num_links` | 1, 2 | — (the plan is written for any L) |

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | per device `(..., M, W_local)`, rank ≥ 2. Phase 0: M and W_local multiples of 32 |
| Dtype | `float32` / `bfloat16` |
| Layout | `Layout::TILE` |
| Memory | `TensorMemoryLayout::INTERLEAVED`, DRAM or L1 |

### Gamma (optional)

| Property | Requirement |
|----------|-------------|
| Shape | per device `(1, 1, 1, W_local)` (Phase 0 TILE). Device p's slice of the full-W scale |
| Dtype | `float32` / `bfloat16`, independent of the input |
| Layout | `Layout::TILE`: `Wt_local` tiles, only tile row 0 meaningful |
| Memory | `TensorMemoryLayout::INTERLEAVED` |

### Output

| Property | Value |
|----------|-------|
| Shape | identical per-device logical shape |
| Dtype | input dtype |
| Layout | `Layout::TILE` |
| Memory | `memory_config` if given (Phase 0: interleaved), else `ttnn.DRAM_MEMORY_CONFIG` |

## Blocking Model

The first design section; everything below realizes it. Semantics: `.claude/references/blocking-model.md`.

### Symbols (host-computed once per call; every tile count uses `ceil` and per-image tile padding)

| Symbol | Definition |
|--------|------------|
| `G` | group size `mesh_shape[cluster_axis]`; `p` = this device's group position (`c` for axis 1, `r` for axis 0) |
| `tensor_tile_rows` (M_t) | `prod(shape[:-2]) · ceil(shape[-2] / 32)` |
| `tensor_width_tiles` (Wt) | `ceil(W_local / 32)` |
| `L` | links in use: `num_links` or `L_avail`; `L_act = min(L, num_row_groups)` links carry data |
| `compute_cores` | `device.compute_with_storage_grid_size()` grid minus the `2·L_act` port cores, in row-major (y, x) order |
| `G_w` | Tensix cores per row-group: 1 in R1-X, from the R3 rule in R3-X |
| `S` (`core_width_tiles`) | a core's W-slice: `Wt` in R1-X, `ceil(Wt / G_w)` in R3-X (the last core is ragged, `V ≤ S` valid tiles) |
| `R` (`block_rows`) | tile-rows per block (the extent knob of `tile_rows`) |
| `num_waves` | `ceil(max_group_rows / R)`; global, identical on every core and every chip. Wave w is block w on every core and one exchange round |

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `batch` (dims before M) | independent: the math never mixes leading indices | folded into `tile_rows` | folded | folding rule: contiguous, treated identically (`blocking-model.md` §1) | via `tile_rows` | — |
| `tile_rows` (M_t) | independent: each row's result depends only on its own row, across all devices | `block_rows` (R); `num_waves` derived | **coarsest R ≤ the row-group's rows that fits L1**. Every INPUTS shape gives R = all of the core's rows and `num_waves = 1` | host constant `BLOCK_ROWS_CAP` (`None` = uncapped) feeds the closed-form solve (Work Distribution) → CT/RT `block_rows`, `num_waves`. CB pages, loop trips, landing layout and port counts all derive from these two | row-groups across the **whole compute grid**. R1-X: `ttnn.split_work_to_cores(compute_cores, M_t, row_wise=True)`. R3-X: groups of `R3_ROWS_PER_GROUP` tile-rows × `G_w` cores | knob-turn (R, waves; the grid is an RT parameter) |
| `width_local` (Wt), **reduction stages** (`square_accumulate`, `collapse`, `core_combine`) | **dependent**: Σx² spans the row | `core_width_tiles` S via `G_w` | R1-X: S = Wt (not split). R3-X: `G_w = min(R3_MAX_GROUP_CORES, floor(cores / groups), floor(Wt / R3_MIN_SLICE_TILES))`, S = `ceil(Wt / G_w)` | host constants `R3_ROWS_PER_GROUP = 2`, `R3_MAX_GROUP_CORES = 24`, `R3_MIN_SLICE_TILES = 4`, `R3_OCCUPANCY_GAIN = 2`, `R3_MIN_GROUP_CORES = 4` (base op values) | R3-X: `G_w` consecutive compute cores per row-group, root = first core | knob-turn: the in-chip combine exists, and `G_w` is a parameter |
| `width_local`, **streaming stages** (`scale_by_gamma`, `scale_by_inv_rms`, `store`) | independent: once `inv` exists every output tile depends only on its x tile, γ tile and row | `w_chunk_tiles` C | `min(S, W_CHUNK_TILES)`, `W_CHUNK_TILES = 64` (C = S for every INPUTS shape) | host constant `W_CHUNK_TILES` → CT `w_chunk_tiles` | within the core | knob-turn |
| **device axis** (group position, G devices; the global W = concat over devices) | **dependent**: the sum spans all G slices, and the **caller** fixed this split (each device owns W_local) | the exchanged block = one **segment**: the stats of one (wave, link), `rows(w, l) × 128 B` (32 fp32 per tile-row) | one segment per (wave, link) per device | derived from R, waves, link split | **port cores**, one per (direction, link): a communication role, no arithmetic | transport: perf lamp L6 |
| `links` (L) | independent: the stats rows of different links travel on disjoint routing planes | rows per link | contiguous runs of row-groups, balanced by rows over `L_act` links | user `num_links` / `L_avail` | link l's two ports (fwd / bwd) carry its rows | — |
| `gamma` operand (Wt tiles, row 0) | **reuse-shared**: γ does not vary along `tile_rows`, the split axis, so every core with the same W-slice needs the same bytes | resident: the core's S γ tiles | whole slice resident, read once per call | derived from S | each active core reads its slice's row-0 prefixes from DRAM (stepping stone; multicast = regime G-MC, deferred) | refinement (G-MC) |
| within-tile (32×32) | — (the tile is the unit) | — | — | — | — | partial tiles: padding rules (alignment refinements) |

**Per-stage re-run of the table (each stage's own data):**

| Stage output | Its axes | Block it is produced in | Assignment | Note |
|--------------|----------|------------------------|------------|------|
| `sumsq_partial` (1 tile per tile-row, pre-collapse) | `tile_rows` independent | R tiles per block | the core owning the slice | — |
| core partial (col-0 tile per tile-row) | `tile_rows` independent | R per block | each core | R3-X: the input of the in-chip combine |
| device partial D_p (col-0 tile per tile-row) | `tile_rows` independent | R per block | **row-group root**: every root sums only its own rows, so the work is spread over all roots | the root is a collector for G_w partials. Its arithmetic (G_w−1 adds per row) is its own rows' share, not a whole-device stage |
| stats segment (compact fp32, 128 B per tile-row) | `tile_rows`, `links` independent | one segment per (wave, link) | port cores **only move bytes** | no port computes anything, so no Tensix core computes all rows while others wait |
| `ss` = Σ_g D_g, `inv` | `tile_rows` independent | R per block | the row-group root of those rows (all roots in parallel) | each device computes it redundantly for its own copy. The sum order g = 0..G−1 is identical on every device, so all G devices produce **bit-identical** `inv` |
| `x·γ`, `out` | `tile_rows` × `width` independent | R × C | every core of the row-group | — |

**Scheme committed for Phase 0:** the base op's local regimes (R1 row-parallel / R3 W-slice row-groups, rows resident) feed a **stats all-gather over Fabric**. Each device sends its compact per-row partial Σx² (fp32) to every peer of its group with a one-hop store-and-forward relay along the topology. Every device then sums all G partials in device order and finalizes locally. Phase 0 is multi-core: rows (and, in R3-X, W-slices) cover the whole compute grid. **Every knob (R, waves, G_w and the R3 constants, C, x_depth, gather_depth, stream_depth, L, the port placement, the grid) is a parameter with one source. None is an inlined literal.**

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_x_tiles` | `x_depth = min(2, num_waves)` | 1 (one wave) | with waves > 1 the reader loads block w+1 while block w waits on its exchange (overlap lamp L1) |
| `cb_gather` | `gather_depth = min(2, num_waves)` | 1 | a contributor may send wave w+1's partial before the root consumed wave w |
| `cb_inv_rms` | `inv_depth = min(2, num_waves)` | 1 | the next wave's inv may land while the current one scales |
| `cb_output` | `stream_depth` | 2 | compute packs chunk j+1 while the writer drains chunk j |
| `cb_sumsq_partial`, `cb_partial_col`, `cb_device_partial`, `cb_stats_slots`, `cb_inv_root`, `cb_x_gamma` | none (1 block) | 1 | hand-offs between sequential phases of one block |
| `cb_reduce_scaler`, `cb_gamma_tiles` | none | 1 | constant / resident for the kernel |
| landing scratch (port cores) | per-wave slots | `num_waves · G` slots | every (wave, source device) has its own slot. No credit loop inside a call; the ready fence protects across calls |

`stream_depth = 2` is one host constant; every `min(2, num_waves)` depth reads one host constant `MAX_PIPELINE_DEPTH = 2`.

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Fabric max payload, `ttnn.get_tt_fabric_max_payload_size_bytes()` (14,336 B on the production config) | one segment's bytes per packet | the segment is sent as `ceil(seg_bytes / payload)` packets; the fused increment goes on the last packet of each entry only | an oversized packet asserts in the header code / corrupts the router buffer |
| One producer per router channel (`blackhole-fabric.md` §1, §6 rule 10) | senders per (link, direction) | exactly one port core per (direction, link), and only it opens that connection | two producers on one channel: silent corruption / hang |
| Packet header pool: 12 headers per data-movement RISC-V (`blackhole-ccl-kernels.md` §3.3) | packets in flight per port | headers allocated once, a ring of ≤ 8, `noc_async_writes_flushed()` before reuse | `PACKET HEADER POOL EXHAUSTION` hang |
| Semaphore ids ≤ 16 per core; `setup_fabric_connection` takes the lowest free ids | port-core semaphores | the op's own semaphores are declared **before** any `setup_fabric_connection` call | "Semaphore id N already in use" at build |
| Arrival counter ordering (`blackhole-ccl-kernels.md` §6.1) | senders per counter | one upstream sender per port counter; thresholds are cumulative entry counts | a relay reads a slot that has not landed |
| L1 NoC alignment (16 B; DRAM 64 B on Blackhole) | stats row (128 B), segment stride, γ prefix reads (544 B bf16 / 1088 B fp32) | segment stride `align_up(seg_bytes, 64)`; rows are 128 B; γ prefix reads go page-start to page-start | misaligned NoC transfer → corrupt data |
| DEST capacity, fp32 DEST: 4 tiles half-sync (`dest_helpers.hpp:103` `DEST_AUTO_LIMIT`) | `device_sum_finalize` uses D0, D1 only; chains / reduce clamp internally | 2 slots per tile-row step | — (no design extent depends on DEST) |
| Runtime-arg count per kernel | per-port root list (notify targets), per-root slot sources | pack `(x, y)` into one u32; ≤ 128 roots per port (106 compute cores max) | arg overflow at program build |
| L1 budget (`ttnn.get_memory_view(device, ttnn.BufferType.L1).largest_contiguous_bytes_free_per_bank`, measured **after** the landing scratch exists) | R and the depth knobs | closed-form R (Work Distribution), then refuse at R = 1 | CB clash / OOM at program creation |
| Mesh view may bend the axis (QuietBox 1×4 snake) | chip-level multicast routes (`MeshMcastRange` hop counts) | not used: every packet is a one-hop unicast to a logical neighbour, routed by destination node | `all_gather_multicast_factory.cpp:96-99` TT_FATALs on a bent axis; the 2D multicast route would leave the group |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **R1-X**: row-parallel local work, rows resident, stats all-gather | **built** | `memory_layout == INTERLEAVED` and `select_local(...) == R1` (the base op rule below) and `footprint(R = 1) ≤ L1_budget` | `R × Wt` per core (whole rows) | **Minimum at DRAM**: x read once, out written once. Above it: γ, each active core reads its slice's row-0 prefixes once. That is ≤ 544 / (2048 · rows_this_core) ≈ 27 % / rows of x bytes for bf16 (≤ 13 % when a core owns ≥ 2 rows). **Fabric**: per device `(G − 1) · M_t · 128 B` received, the minimum for an all-gather of the partials. **NoC**: each root writes `R · 128 B` to ≤ 2 ports and reads `(G − 1) · R · 128 B` back | per block: one square chain init, one collapse reduce init, one γ chain, one finalize chain, one inv chain. **Per wave: one exchange round (fence + G−1 relayed entries + notify)**. That round is the dominant fixed cost, which is why R defaults to all of the core's rows (one wave) |
| **R3-X**: W-slice row-groups + in-chip root combine + stats all-gather | **built** | `memory_layout == INTERLEAVED` and `select_local(...) == R3` and `footprint(R = 1) ≤ L1_budget` | `R × S` per core, `G_w` cores per row-group | DRAM as R1-X (γ per core is its slice only). **Added NoC**: per tile-row, `G_w − 1` partial tiles (faces 0/2, 2 KiB fp32) to the root, and `G_w − 1` inv tiles (faces 0/2) back. Fabric as R1-X | per block: one gather + one inv distribution instead of one per row (grid-sync lamp L2). Per wave: one exchange round |
| **WAVES** (`num_waves > 1`) | **built** as the same structure. It is not a separate code path: R below the core's rows sets `num_waves`, and every loop is already per wave | `footprint(R = core rows) > L1_budget`, or `BLOCK_ROWS_CAP` set. No INPUTS shape | `R × S` | same DRAM bytes; one extra exchange round per wave (fixed cost) | each wave halving removes one exchange round. Pinned by test `test_waves_forced` |
| **G-MC**: γ read once per sharing set + NoC multicast | **deferred**, the successor of R1-X / R3-X's per-core γ reads (stepping stone). Positive reason: multicast needs hole-free destination rectangles, and Phase 0's compute grid has the port cores cut out of row 0. γ is ≤ 27 % / rows of x bytes (above). Structure kept: only the source of `cb_gamma_tiles` changes | `has_gamma` and a sharing set ≥ 2 cores | `cb_gamma_tiles` S pages | removes `(sharing_cores − 1) · S · prefix` DRAM bytes per set; adds one `S · T_g` multicast per set (base op Perf 1 `gamma_mcast`: 1.11× on DRAM-bound prefill) | — |
| **R3-RS**: in-chip reduce-scatter owners (each member finalizes `R/G_w` rows) | **built for WIDTH_SHARDED (Refinement 1)**: owners = the largest divisor of R ≤ G_w (`R3_RS_OWNERS`); measured 171 → 26.3 µs at 640×1792 WIDTH. Interleaved: still deferred. Positive reason: interleaved R3-X row-groups hold `R3_ROWS_PER_GROUP = 2` tile-rows, so a root finalizes ≤ 2 rows. The scatter's measured win (base Refinement 6: 2.1–2.5×) came from tall shards with ≥ 23 rows per group, which no Phase 0 path produces. Reachable: the gather already exists, and owners only change which core sums which rows | WIDTH_SHARDED tall shards (refinement) | — | same bytes, split over owners | — |
| **FLAT**: core partials straight to the ports (no in-chip pre-sum), every root sums `G · G_w` partials | **deferred**. Positive reason: it removes the in-chip gather round, but makes the exchange segment `G_w`× larger and puts `G · G_w` column-0 expansions (decode: 56 tiles × 32 RISC-V stores) and adds on the root's critical path after the exchange. That is structurally more serial work than the round it removes. Reachable: ports already carry arbitrary segments | decode (`M_t = 1`) | — | Fabric `× G_w`; NoC −(G_w − 1) partials | — |
| **RS+AG**: reduce-scatter the stats, then all-gather `inv` | **rejected**: superseded by the stats all-gather. Two serial latency-bound exchange rounds of the same size, where the all-gather needs one and each device's redundant sum is ≤ G − 1 adds per tile-row | — | — | 2 exchange rounds | — |
| **XG**: all-gather x, then a local rms_norm | **rejected** (dead end): moves `(G − 1) · M · W_local · es` over Fabric (448× the stats at 640×1792) and holds G× x in L1; superseded everywhere | — | — | Fabric (G−1)·|x| | — |
| **2P**: two-pass streaming (re-read x after the exchange) | **rejected** (dead end): x crosses DRAM twice. Superseded by resident blocks + WAVES, which keeps one crossing at every size | — | — | x 2× | — |
| **COMPOSITE**: pre / all_gather / post as three programs | **rejected** by the requirements (one native dispatch) and by the measurement: 92 µs vs a 14 µs target | — | — | stats round trip through DRAM, 3 launches | — |
| **HEIGHT_SHARDED / WIDTH_SHARDED** placements | **deferred** (TARGET refinements; Phase 0's SUPPORTED rectangle is fixed by the requirements to INTERLEAVED). Reachable as knob-turns: HEIGHT = R1-X over zero-copy shard CBs, WIDTH = R3-X with `G_w` = shard columns. The output stays in the shard (`cb_output` over the output shard) | `memory_layout` | whole shard = one block | x / out 0 DRAM crossings | — |

**Local regime selection `select_local` (host, exact; the base op's `select_plan()` rule for interleaved TILE, over `compute_cores` instead of the full grid):**

```
groups_R3 = ceil(M_t / R3_ROWS_PER_GROUP)
G_w      = min(R3_MAX_GROUP_CORES, floor(len(compute_cores) / groups_R3), floor(Wt / R3_MIN_SLICE_TILES))
R1_cores = min(len(compute_cores), M_t)
R3 if  G_w >= R3_MIN_GROUP_CORES  and  G_w * groups_R3 >= R3_OCCUPANCY_GAIN * R1_cores
      or footprint_R1(R = 1) > L1_budget >= footprint_R3(R = 1)
else R1
```

| Shape (perf cases, QB2, L = 2 → 106 compute cores) | Regime | Grid use |
|---|---|---|
| 640×1792 (M_t 20, Wt 56) | R3-X: 10 groups × G_w 10, S = 6 | 100 cores |
| 32×1792 (M_t 1, Wt 56) | R3-X: 1 group × G_w 14, S = 4 | 14 cores |
| 2048×1280 (M_t 64, Wt 40) | R1-X (G_w = 3 < 4) | 64 cores, 1 tile-row each |

Regime-pinned tests (the selection depends on the grid and L): `test_regime_r1`, `test_regime_r3` (acceptance test) assert the shapes above give correct output, and `test_waves_forced` pins WAVES.

### Traffic ranking

Bytes per tier for one device (`T_in` = input tile bytes; `seg` = 128 B per tile-row). The cross-device split of W is **fixed by the caller**, so the cross-device combine is mandatory. The candidates differ in what is exchanged and how the local work is cut.

| Candidate | DRAM crossings (x / out / γ) | In-chip NoC | Fabric per device | Rank | Decision |
|-----------|------------------------------|-------------|-------------------|------|----------|
| rows across cores, rows resident, exchange stats (R1-X) | 1 / 1 / per-core slice prefixes | ≤ 2·R·seg out + (G−1)·R·seg in per root | (G−1)·M_t·seg received | **1** | built |
| W-slices + in-chip combine, exchange stats (R3-X) | 1 / 1 / per-core slice prefixes | + 2·(G_w−1) half-tiles per tile-row | same | 2: more NoC bytes, same DRAM; wins on occupancy when M_t < grid | built (selected by `select_local`) |
| reduce-scatter stats + all-gather inv (RS+AG) | 1 / 1 | same | same bytes, 2 rounds | 3 | rejected |
| all-gather x (XG) | 1 / 1 | — | (G−1)·M_t·Wt·T_in | 4 | rejected |
| re-read x (2P) | 2 / 1 | — | stats | 5 | rejected |

Occupancy is a separate question (`blocking-model.md` §4). R1-X leaves cores idle when `M_t < 106`; `select_local` moves to R3-X when that buys ≥ 2× the cores. The **residency** argument for cutting the dependent W axis is also live here: a core's slice is resident for both passes in both regimes, so x crosses DRAM once at every INPUTS shape.

### Block schedule

```cpp
// Compute core (reader / compute / writer); num_blocks_this_core == num_waves; R_b = this block's rows (ragged last).
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    load_block(block_idx);                 // reader: R_b x S tiles of x from DRAM -> cb_x_tiles
    square_accumulate_block(block_idx);    // compute: grid(R_b, S) -> R_b raw Σx² tiles (cb_sumsq_partial)
    collapse_block(block_idx);             // compute: REDUCE_ROW -> R1-X: cb_device_partial; R3-X: cb_partial_col
    core_combine_block(block_idx);         // R3-X: contributors -> root's cb_gather; root sums G_w partials -> cb_device_partial
    publish_stats_block(block_idx);        // root reader: compact col 0 of R_b tiles, write the rows to its link's ports
    scale_by_gamma_block(block_idx);       // compute, in the exchange's stall shadow: x * γ -> cb_x_gamma; x released
    gather_slots_block(block_idx);         // root reader: wait for "landed", read G-1 peer rows, expand -> cb_stats_slots
    device_sum_finalize_block(block_idx);  // root compute: Σ_g (fp32, g = 0..G-1), x(1/(G·W_local)), +ε, rsqrt -> inv
    distribute_inv_block(block_idx);       // R3-X: root -> every member's cb_inv_rms (R1-X: compute writes it directly)
    scale_by_inv_rms_block(block_idx);     // compute: (x·γ) * inv -> cb_output, in C-tile chunks
    store_block(block_idx);                // writer: cb_output -> DRAM
    release_block(block_idx);              // pop cb_inv_rms (and cb_x_tiles when there is no γ)
}
// Port core (one kernel, RISCV_0 / NoC1), per (direction d, link l):
//   send_ready();                         // once per call: the ready fence to the upstream port that writes into my landing
for (uint32_t wave = 0; wave < num_waves; ++wave) {
    exchange_wave(wave);                   // own segment out, relay received entries, collect G-1 slots, notify roots
}
//   drain();                              // re-arm the global counters by what was consumed (back to 0)
```

| Block operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------------|-------------|--------------------|-----------------------------------|
| `load_block` | `R_b × S` tiles | previous block (when `x_depth = 2`) | reads issued per tile-row, one barrier per tile-row (S reads in flight; catalog `double_buffer`); push per tile-row so squaring starts on the first landed row |
| `square_accumulate_block` | `grid(R_b, S)` | `cb_x_tiles` (not popped) | one chain call per block (one init + reconfig); DEST acquire / pack per tile-row; cumulative input wait |
| `collapse_block` | `R_b × 1` | `cb_x_tiles` | one `reduce` call per block (one `reduce_init`) |
| `core_combine_block` (R3-X) | `G_w × R_b` partial tiles | `cb_x_tiles` | per block: one gather round. Each contributor writes all its R_b partials as **one** NoC write burst (faces 0/2) plus **one** semaphore increment. The root does one `reduce` call over `(G_w, R_b)` |
| `publish_stats_block` | `R_b` rows × 32 fp32 | `cb_device_partial` until compacted | per block: R_b row writes of 128 B to each of ≤ 2 ports and one semaphore increment per port |
| `exchange_wave` (port) | one segment per entry | the landing slots of every wave | per call: one ready packet + one ready wait. Per wave: one own send, `n_relay` relays (`ceil(seg / payload)` packets each, a fused increment on each entry's last packet), one notify per root of the link (local NoC semaphore increment) |
| `scale_by_gamma_block` | `grid(R_b, C)` per chunk; C = S → one call | `cb_gamma_tiles` (resident), `cb_x_tiles` until the call ends | one chain init per chunk. It runs **before** `cb_stats_slots` is waited, so it fills the exchange wait |
| `gather_slots_block` | `G·R_b` slot tiles | — | per block: one wait on `SEM_STATS_LANDED`, `(G − 1) · R_b` NoC reads of 128 B (one barrier), `(G − 1) · R_b` in-place column-0 expansions, one push of `G·R_b` pages |
| `device_sum_finalize_block` | `R_b` tile-rows × G slots | — | one chain call per block (G−1 SFPU adds + 3 SFPU unary ops per tile-row) |
| `distribute_inv_block` (R3-X) | `R_b` inv tiles | — | per block: one faces-0/2 write burst + one semaphore increment per member (G_w − 1 members, rotated order; base `combine_write_tiles` / `rotated`) |
| `scale_by_inv_rms_block` | `grid(R_b, C)` per chunk | `cb_inv_rms` (waited once per block) | one chain init per chunk; output reserved / pushed per tile-row × C |
| `store_block` | `R_b × S` tiles in `R_b · ceil(S/C)` quanta | — | one write barrier per quantum |
| `release_block` | — | — | ≤ 2 pops per block |

### Stall-shadow check

| Wait | Who waits | Work independent of it, scheduled into it | Legal because |
|------|-----------|-------------------------------------------|---------------|
| the Fabric exchange (`gather_slots_block` → `device_sum_finalize_block`) | root compute, and through `cb_inv_rms` every member's compute | `scale_by_gamma_block` of the same block, for the **whole** block (C = S), so the full x·γ pass runs during the exchange. With `num_waves > 1`, also `load_block` / `square` / `collapse` of block w+1 (`x_depth = 2`) | the two scalings commute and associate: `x·γ·inv = (x·γ)·inv`. γ-first is also the base op's floating-point order, so precision does not move |
| the in-chip gather (`core_combine_block`) | root compute | nothing on the root (its own partial is already in). Members do `scale_by_gamma_block` during it | — |
| `distribute_inv_block` | members | as above (γ pass) | as above |
| DRAM reads | compute | squaring of landed rows (cumulative wait) | — |

What the exchange window **cannot** fill: with one wave per core, DRAM is idle between the last read and the first write. That idle window is overlap lamp L1, and `num_waves > 1` is its knob.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| L1 Overlap (waves) | One wave per core (R = all of the core's rows) serializes all reads → exchange → all writes, and DRAM idles for the exchange round (~1.6–3 µs on the 4-ring). At 640×1792 that is 15–30 % of the 10.6 µs roofline | `BLOCK_ROWS_CAP` so `num_waves ∈ {2, 4}` on 640×1792 and 2048×1280: wave w's exchange overlaps wave w+1's reads (the roofline's "pipelining row blocks") |
| L2 Grid synchronization (R3-X group size) | Maximal `G_w` gives 4–6 tiles per core; the in-chip gather + inv distribution (two NoC rounds) may dominate on decode, where the exchange is already the long pole | `R3_MAX_GROUP_CORES ∈ {8, 14, 24}`; `R3_ROWS_PER_GROUP ∈ {1, 2, 4}` at 640×1792 (catalog `tensix_all_reduce`; base sweep: decode best at 20–30 slice cores) |
| L3 Port placement | Phase 0 puts the `2·L_act` port cores in logical row 0 (nearest the Ethernet row), middle columns, chosen on the host with no probe program. The ports may sit far from their link's Ethernet core | Ethernet-adjacent placement derived on the host from the cluster descriptor + SoC descriptor (`fabric_all_gather._eth_physical_lists`, `_worker_nearest_noc1`), **without** the AG probe program (a probe would be a second dispatch) |
| L4 Notify granularity | Ports notify roots once **all** G−1 slots of a wave have landed, so the root's adds start after the farthest slot | notify per landed entry; the root sums slots as they arrive (fp32 order still fixed: accumulate in g order and wait for slot g before adding it). **Built (Refinement 5, `NOTIFY_PER_ENTRY`)**: per-entry arrival counters; the root reads each slot as it lands, and the compute sum keeps its fixed g order. |
| L5 inv distribution | R3-X distributes `inv` by unicast faces 0/2 from the root to each member (base combine transport). Multicast was ruled out because the port cores sit inside the grid | `ttnn.Mcast` over hole-free row-group rectangles (port cores excluded by grid construction). **Built (Refinement 5, `INV_MCAST`)**: interleaved R3-X row-groups are single rectangles (`_rect_groups`), and the root multicasts through kernel_lib `McastFamily` (Counter). |
| L6 Exchange transport | One-hop store-and-forward relay: the farthest slot pays `floor(G/2)` router hops plus `floor(G/2) − 1` relay-kernel turnarounds on the ring, and G−1 hops on a line | direct multi-hop unicast per peer, routed by destination node with the first hop's direction from `ttnn.get_eth_forwarding_direction` (Ring only; Linear must stay on line neighbours). It removes the relay turnaround. 2D chip-level multicast stays excluded (bent mesh views, Mechanism caps). **Built (Refinement 5, `DIRECT_RING_SENDS`)**: `_exchange_plan` (direct on Ring, relay on Linear) |
| L7 Finalize SFPU scope | `device_sum_finalize` runs G−1 adds + 3 unary ops on all 32×32 lanes, though only column 0 is consumed | column-0 scoped SFPU (base `FINALIZE_COL0` body extended with the fp32 adds; catalog `sfpu_tile_scope`, base: finalize 1.8 k → ~0.5 k cycles per tile-row) |
| L8 srcB hold / NoC balance | Phase 0 uses the stock scaling chains and the default NoC pairing | base Perf 2 `bcast_srcb_reuse` (1.06× compute-bound) and `NOC_SWAP_COLUMNS = 3` (1.03–1.16× interleaved) |
| L9 Links | `L_act = min(L, row_groups)` splits rows over links. The payload is latency-bound (M_t·128 B), so a second link may buy nothing (composite measured 1 ≈ 2 links) | L = 1 vs 2 on the perf cases. With 2 links each port's segment halves but the hop count does not |

### Perf 1 amendments (supersede the rows below where they disagree; changelog `## Perf 1`)

- **Slot forward replaces the inv distribution on single-root R3-X (FWD_INV, `replicated_finalize`).** The root no
  longer finalizes for its row-group and distributes inv (`cb_inv_root` → copy → McastFamily multicast of the tail →
  members expand). Instead it forwards the RAW compact slot tails (own slot at publish, each peer slot as soon as its
  read from the port completes) to the same `cb_stats_slots` page tails of every member: raw linked multicasts per member
  rectangle, each chain closed by a multicast semaphore SET (`SEM_OWN_FWD` = blocks published, `SEM_SLOT_FWD` =
  `(block + 1) << 24 | slot mask`). Every member expands its slots, pushes `G·Q`, and its compute runs the same
  `device_sum_finalize` into its own `cb_inv_rms` (exactly the R1-X path), so inv is bit-identical group-wide. On
  single-root R3-X, `cb_inv_root`'s page is now only the 16 B staging-word ring of the semaphore sets. R3-RS (several
  owners) keeps the owners' unicast inv all-gather. Every root's slot gather is now pipelined: reads are issued as entries
  land, and a batch is expanded once its reads complete. `INV_MCAST` is gone; `RECT_GROUPS` keeps the single-rectangle
  group shape.
- **DRAM NoC balance (`noc_balanced_writes`).** The compute cores of the 3 westmost logical columns, plus the east-most
  column when it holds ≥ 2 compute cores, run reader on NoC1 and writer on NoC0. This is a second reader/writer
  descriptor pair, with the same sources and args. FWD_INV roots (multicast senders) are excluded. Nothing swaps when x
  and the output are both shard-resident.

### Perf 2 amendments (supersede the rows below where they disagree; changelog `## Perf 2`)

- **Stats slots stay in their wire form (`no_col0_expand`).** A `cb_stats_slots` page now holds its slot as 32
  contiguous fp32 words at the page base (tile face 0 rows 0-1), not the face-3 tail plus a column-0 copy. This holds
  for the own slot (publish, `col0_to_wire`), the slots read from the ports, and the forwarded slots. No RISC-V
  expansion runs after a slot lands: `tail_to_col0` and the `rd_slot_expand` / `rd_mbr_expand` stages are gone from
  the stats path. `tail_to_col0` / `col0_to_page` remain only for the R3-RS inv all-gather.
- Compute sums and finalizes the two wire-form SFPU vectors (`AddRow01`, `FinalizeRow01`), in the same fixed g order
  with the same per-lane fp32 ops. `Row01ToCol0` (`transpose_dest`, then one SFPU lane move of face 0 column 1 →
  face 2 column 0) puts inv into column 0 for the `BroadcastDim::Col` scale. inv is bit-identical to the column-0
  path. The landing and wire formats on the Fabric are unchanged (128 B per tile-row).

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| DRAM → `cb_x_tiles` | input-dtype tiles | reader (NCRISC, NoC0): `TensorAccessor` reads, page id `tile_row · Wt + w` for `w ∈ [w0, w0 + V)`, pushed per tile-row (S pages; a ragged slice pushes the nominal S with only V read) | Phase 0 is tile-aligned: no pad zeroing |
| `cb_x_tiles` → `cb_sumsq_partial` | Float32 | compute chain: FPU `Mul(x, x)`, `DestAccumulation::PerRow` | base op `square_accumulate` |
| `cb_sumsq_partial` → `cb_partial_col` (R3-X) / `cb_device_partial` (R1-X) | Float32, column 0 valid | `reduce<SUM, REDUCE_ROW>` (scaler 1.0, no post-op) | base op collapse |
| `cb_partial_col` → root `cb_gather` (R3-X) | Float32 faces 0/2 | contributor reader: one NoC write burst into the root's `cb_gather` slots `j·R + r` (same CB address on every compute core), barrier, `noc_semaphore_inc(SEM_GATHER)`. The root's reader copies its own partial locally | base `combine_write_tiles` (faces 0/2 suffice: every consumer reads column 0) |
| `cb_gather` → `cb_device_partial` (root) | Float32 col 0 | `reduce<SUM, REDUCE_COL, AccumulateViaAdd, ReduceWithinTile::Skip>` over `(G_w × R_b)` | base op root combine (lane-wise pairwise adds) |
| `cb_device_partial` → ports (own segment) | **compact fp32**: 32 values per tile-row, 128 B | root reader: copy col 0 (rows 0–15: face 0, word `16·r`; rows 16–31: face 2) into the own-slot page `r·G + p` of `cb_stats_slots`, compact at the page head (ascending r, in place). Then NoC-write those 128 B to `landing[w][p][offset(r)]` on each port of the root's link, barrier, `noc_semaphore_inc(SEM_OWN_STATS)` on each port. Then expand the own slot back in place (descending r) | the own slot thus holds D_p in tile form for the sum. Compaction and expansion are pure 4 B moves: bit-exact |
| port → peer port (Fabric) | compact fp32 segment | port kernel (RISCV_0, NoC1): `WorkerToFabricEdmSender` to its neighbour in direction d on link l. The route is one hop: `fabric_set_unicast_route(hdr, peer_chip, peer_mesh)` on 2D, hop count 1 on 1D (the AG `route_one_hop`). The packet writes into the **peer's same-direction port** landing slot (address uniform on every chip), with a fused atomic increment of that port's `arrival` global semaphore on the entry's last packet | Ring schedule (AG `_schedule`): fwd sends slots `p, p−1, …` (`floor(G/2)` entries), bwd sends `p, p+1, …` (`G−1−floor(G/2)`). Linear: fwd sends `p, p−1, …, 0` if `p < G−1`, bwd sends `p, …, G−1` if `p > 0`. Entry i ≥ 1 is relayed from the landing once `arrival ≥ w·n_in + i` |
| ready fence (per call) | header-only packet | each port that **receives** sends one `unicast_atomic_inc` to the `ready` semaphore of the upstream port that writes into it (over this chip's connection toward that peer, AG `_needs_ready`). A sender waits `ready ≥ 1` before its first data packet, then decrements by 1 | guarantees the peer finished the previous call before its landing is overwritten |
| port → roots (landed) | local semaphore | after all of a wave's expected entries landed (`arrival ≥ (w+1)·n_in`) and its own sends are flushed: `noc_semaphore_inc` of `SEM_STATS_LANDED` on every root of link l | roots wait `≥ (w+1) · n_notifying_ports(l)` |
| ports' landing → root `cb_stats_slots` | compact fp32 → tile column 0 | root reader: for every q ≠ p, `R_b` NoC reads of 128 B from `landing[w][q][offset]` on the port holding slot q (host-computed per q: the fwd port for the slots arriving on the fwd stream, the bwd port otherwise) into the head of page `r·G + q`, one barrier, then expansion in place (descending r: word r → face offset of row r, column 0) | lanes other than column 0 are don't-care (base: faces 1/3 never matter; no consumer reads non-column-0 lanes) |
| `cb_stats_slots` → `cb_inv_root` (R3-X) / `cb_inv_rms` (R1-X) | Float32 col 0 | compute chain (`UnpackToDestFp32` on `cb_stats_slots`): `D0 = slot 0`; for g = 1..G−1: `D1 = slot g`, `D0 = D0 + D1` (SFPU, fp32); `D0 ·= 1/(G·W_local)`; `D0 += ε`; `D0 = rsqrt(D0)`; pack | fixed g order on every device → bit-identical `inv` group-wide; fp32 exact through the sum |
| `cb_inv_root` → members' `cb_inv_rms` (R3-X) | Float32 faces 0/2 | root reader: one write burst per member + `noc_semaphore_inc(SEM_INV)`; local copy for itself. Members' reader reserves `R_b`, waits `SEM_INV ≥ block_idx + 1`, pushes | base owners-path transport |
| γ DRAM → `cb_gamma_tiles` | γ-dtype tiles, row 0 valid | writer (BRISC, NoC1), once per kernel: per slice tile, one read of the tile's row-0 prefix (`gamma_row0_prefix_bytes`: face 0 + 16 elements of face 1 = 544 B bf16 / 1088 B fp32) into page j, then push S | base `GAMMA_DIRECT_READ = PREFIX` |
| `cb_x_tiles` × `cb_gamma_tiles` → `cb_x_gamma` | Float32 | chain `Mul`, `BroadcastDim::Row` | γ only |
| (`cb_x_gamma` or `cb_x_tiles`) × `cb_inv_rms` → `cb_output` | output dtype | chain `Mul`, `BroadcastDim::Col` | final rounding at this pack |
| `cb_output` → DRAM | tiles | writer: page id `tile_row · Wt + w`, one barrier per quantum, pop | skips ragged-slice pad tiles |

### Tensix-to-Tensix and chip-to-chip contracts

| Contract | Definition |
|----------|------------|
| Landing scratch | one persistent L1 mesh tensor, HEIGHT_SHARDED over the port-core set (the same cores on every chip: placement is chip-independent), shard = `num_waves · G · seg_stride_max` bytes. It is allocated by the mesh, so the address is identical on every chip, and it goes to every port kernel as an RT arg. Slot address = `base + ((w · G) + q) · seg_stride_max`. Cache key `(id(mesh), port cores, shard bytes)`; never freed |
| Global semaphores | `arrival`, `ready` per port core (`ttnn.create_global_semaphore(mesh, port_core_set, 0)`), cached with the scratch; `ttnn.synchronize_device` once at creation. Each call consumes exactly what it receives and re-arms by decrement (`noc_semaphore_inc(addr, 0u − n)` + `noc_async_atomic_barrier()`), so both return to 0 every call (trace-safe) |
| Program-local semaphores | compute cores: `SEM_GATHER` (root), `SEM_INV` (members), `SEM_STATS_LANDED` (roots), `SEM_OWN_COUNTED` (roots; verifier amendment). Port cores: `SEM_OWN_STATS`. All are declared over compute ∪ port cores **before** any `setup_fabric_connection`, so the ids (and addresses) match on every core of the chip |
| Wave gating of `SEM_OWN_STATS` (verifier amendment) | `SEM_OWN_STATS` counts publishes cumulatively over waves, so a root whose next block is empty must not increment for wave w+1 before every sending port of its link has counted wave w: each sending port increments `SEM_OWN_COUNTED` on its link's roots once `own_stats ≥ (w+1)·n_roots` (waves w < num_waves − 1 only), and a root waits `SEM_OWN_COUNTED ≥ w·n_pub` before its wave-w increment. Without it, R1 with `num_waves = 2` shipped ~40 of 108 rows stale per call |
| Ordering | one upstream sender per `arrival` counter; entries arrive in schedule order; per-wave thresholds are cumulative |
| Determinism | every sum has a fixed order (chain walk, `reduce` order, g = 0..G−1). No arrival-order-dependent arithmetic: bit-identical repeats |

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | a block: `R` tile-rows × the core's W-slice (`S` tiles) |
| Grid | `device.compute_with_storage_grid_size()` minus the port cores. The core list is an RT parameter, never a constant |
| Port cores | `2·L_act` per chip, one per (direction ∈ {fwd, bwd}, link < L_act). This includes receive-only ports at line ends, which hold landings and send the ready. Phase 0 placement (lamp L3): logical row 0, consecutive columns centred on `grid.x // 2`. Every chip uses the same cores. Kernel: one data-movement kernel on RISCV_0, NoC1 (`blackhole-fabric.md` rule 7) |
| Groups | AG `build_groups(mesh_shape, cluster_axis)`. Position p, `next` / `prev` (Ring: modulo G; Linear: `None` at the ends) |
| Fabric connections | for each port (d, l) with a peer in direction d: `ttnn.setup_fabric_connection(me, peer, ttnn.get_forwarding_link_indices(me, peer)[l], program, port_core)`, appended **last** to that port's RT args |
| R1-X split | `ttnn.split_work_to_cores(compute_cores, M_t, row_wise=True)`: each active core is a row-group with `G_w = 1`; groups differ by ≤ 1 row |
| R3-X split | `groups = ceil(M_t / R3_ROWS_PER_GROUP)` contiguous runs of tile-rows (last ragged), each on `G_w` consecutive compute cores. Core j of a group owns `[j·S, min((j+1)·S, Wt))` |
| Link split | row-groups split into `L_act` contiguous runs, balanced by rows; link l carries its groups' rows |
| `block_rows` R / `num_waves` | `BLOCK_ROWS_CAP` is a module constant of `distributed_rms_norm_program_descriptor.py`, read **at call time** (the acceptance test `test_waves_forced` monkeypatches it to 1). `per_row(R)` / `fixed` from the ledger's footprint expression. `R = min(max_group_rows, BLOCK_ROWS_CAP or ∞, R_fit)`, where `R_fit` is the largest R with `footprint(R) ≤ L1_budget` (monotone in R; `x_depth` / `gather_depth` / `inv_depth` re-derived from `num_waves = ceil(max_group_rows / R)`). If `R_fit = 0` → `NotImplementedError`. No safety fraction |
| Stats layout | segment (w, l) = the rows of every group of link l in wave w, in group order. `offset(g, w)` = Σ over earlier groups of link l of their wave-w row counts × 128 B. `seg_bytes(w, l)`; `seg_stride_max = align_up(max_{w,l} seg_bytes, 64)` |
| Remainder | rows: ragged last group / last block (`R_b < R`), same block operations at a shorter runtime extent. CB push / pop counts stay nominal where a ring is shared (`blocking-model.md` §2). W: ragged last slice (V < S): nominal S pushes, pad tiles never read or stored. Geometry is `ceil` per image (`tensor_tile_rows`) |
| Per-chip programs | one `ProgramDescriptor` per chip, differing only in RT args (p, peers, fabric connection args, root lists). Compiled kernels are identical across chips; G, `has_gamma`, R, S, C, `num_waves` are CT |
| Regime pinning | `select_local` depends on the grid and L, so the acceptance test pins R1-X (2048×1280), R3-X (640×1792, 32×1792) and WAVES (`BLOCK_ROWS_CAP`) |

## Circular Buffers

Symbols: `R` block_rows, `S` core_width_tiles, `C` w_chunk_tiles, `G` group size, `G_w` row-group cores, `xd` / `gd` / `id` = `min(2, num_waves)`, `sd` = 2. `T_in` / `T_out` / `T_g` = tile bytes of input / output / γ dtype; `T32` = 4096 B.

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_x_tiles` | 0 | `T_in` | `xd · R · S` | spans the block (both passes read it) | input dtype | reader | compute | load → γ pass (γ) / release (no γ) |
| `cb_reduce_scaler` | 1 | 2048 | 1 | constant 1.0 tile | bfloat16 | reader (once) | compute | kernel |
| `cb_sumsq_partial` | 2 | `T32` | `R` | one raw Σx² tile per tile-row | Float32 | compute | compute | square → collapse |
| `cb_partial_col` (R3-X) | 3 | `T32` | `R` | this core's collapsed partial per tile-row | Float32 | compute | reader | collapse → gather send |
| `cb_gather` (R3-X; allocated on every compute core so the root's address is uniform) | 4 | `T32` | `gd · G_w · R` | G_w contributor slots per tile-row | Float32 | reader (root) | compute (root) | gather → core combine |
| `cb_device_partial` | 5 | `T32` | `R` | D_p per tile-row (col 0) | Float32 | compute | reader (root) | combine → publish |
| `cb_stats_slots` | 6 | `T32` | `G · R` | G device slots per tile-row; also the compaction / expansion staging | Float32 (`UnpackToDestFp32`) | reader (root) | compute (root) | publish → device sum |
| `cb_inv_root` (R3-X) | 7 | `T32` | `R` | finalized inv before distribution | Float32 | compute (root) | reader (root) | finalize → distribute |
| `cb_inv_rms` | 8 | `T32` | `id · R` | inv per tile-row of the block, read by every chunk | Float32 | R1-X: compute; R3-X: reader | compute | finalize / distribute → release |
| `cb_gamma_tiles` (γ) | 9 | `T_g` | `S` | resident slice of γ tiles (row 0 valid) | γ dtype | writer | compute | kernel (waited once, popped at end) |
| `cb_x_gamma` (γ) | 10 | `T32` | `R · C` | x·γ of one (block, chunk), the stall-shadow pass | Float32 | compute | compute | γ pass → inv pass |
| `cb_output` | 16 | `T_out` | `sd · C` | one tile-row × C chunk in flight | output dtype | compute | writer | per chunk |

Port cores carry **no** CBs. Their only L1 is the landing scratch (a mesh tensor) and the fabric header pool. The full ledger (live set, axis accounting, sharing) is in `l1_ledger.md`.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 0 | prologue | — | yes (`calculate_and_prepare_reduce_scaler`) | — | `cb_reduce_scaler` 1 (reader); `cb_gamma_tiles` S (writer) | resident |
| 1 | `load_block` | `R_b × S` | no (raw reads) | DRAM | `cb_x_tiles`, S per tile-row | pushed |
| 2 | `square_accumulate_block` | `grid(R_b, S)` | yes (chain) | `cb_x_tiles` `R_b·S`, cumulative wait, not popped | `cb_sumsq_partial` `R_b` | x resident |
| 3 | `collapse_block` | `R_b × 1` | yes (reduce) | `cb_sumsq_partial` `R_b` (bulk wait/pop); `cb_reduce_scaler` (not popped) | R1-X `cb_device_partial`; R3-X `cb_partial_col` | — |
| 4 | `core_combine_block` (R3-X) | `(G_w, R_b)` | yes (reduce) + raw transport | `cb_partial_col` → NoC → root `cb_gather` `G_w·R_b` | root `cb_device_partial` `R_b` | gather drained |
| 5 | `publish_stats_block` (root) | `R_b` rows | no (raw 4 B moves + NoC writes) | `cb_device_partial` `R_b` (popped after) | own-slot pages of `cb_stats_slots` (reserved `G·R_b`, not yet pushed); port landings | — |
| 6 | `scale_by_gamma_block` | `grid(R_b, C)` per chunk | yes (chain) | `cb_x_tiles` strided `{c·C, S}` caller-managed; `cb_gamma_tiles` Row mapping (waited once per kernel) | `cb_x_gamma` `R_b·C` | `cb_x_tiles` popped after the last chunk (γ) |
| 7 | `gather_slots_block` (root) | `G·R_b` pages | no (raw NoC reads + 4 B moves) | port landings | `cb_stats_slots` push `G·R_b` | slots full |
| 8 | `device_sum_finalize_block` (root) | `grid(R_b, 1)` × G slots | yes (chain: `CopyTile`, `AddBinary`, `MulUnary`, `AddUnary`, `Rsqrt`, `PackTile`) | `cb_stats_slots` `G·R_b` (caller-managed wait, popped after) | R1-X `cb_inv_rms`; R3-X `cb_inv_root` `R_b` | — |
| 9 | `distribute_inv_block` (R3-X) | `R_b` tiles | no (raw transport, base) | `cb_inv_root` | every member's `cb_inv_rms` `R_b` | — |
| 10 | `scale_by_inv_rms_block` | `grid(R_b, C)` per chunk | yes (chain) | `cb_x_gamma` (γ) or `cb_x_tiles` strided (no γ); `cb_inv_rms` Col caller-managed | `cb_output` C per tile-row | `cb_x_gamma` drained |
| 11 | `store_block` | `R_b × S` | no (raw writes) | `cb_output` | DRAM | popped |
| 12 | `release_block` | — | raw pops | `cb_inv_rms` `R_b`; `cb_x_tiles` (no γ) | — | empty |
| P | `exchange_wave` (port) | segment entries | no (fabric API) | landing scratch | peer landing scratch; roots' `SEM_STATS_LANDED` | — |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| compute boot | raw_api (mandatory) | `compute_kernel_hw_startup(cb_x_tiles, cb_x_tiles, cb_sumsq_partial)` | `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:60` | first statement of the compute kernel | — | — | — |
| prologue scaler | helper | `dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<cb_reduce_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>()` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp:98` | pool-type-aware; SUM → 1.0 | — | `cb_reduce_scaler` | — |
| `square_accumulate_block` | helper | `eltwise_chain(IterationShape::grid(R_b, S), BinaryFpu<Mul, input(cb_x_tiles, Cumulative, None, Block), input(cb_x_tiles, Cumulative, None, Block), D0, DestAccumulation::PerRow>{}, PackTile<output(cb_sumsq_partial, PerOuter, PerOuter, …, DestAccumulation::PerRow)>{})` | `ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp:491` (BinaryFpu), `:507` (PackTile), `:532` (eltwise_chain); as base op `kernels/rms_norm_compute.cpp` | same CB both operands (identical policies) | `cb_x_tiles` | `cb_sumsq_partial` | shape `(R_b, S)` |
| `collapse_block` | helper | `compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, cb_sumsq_partial, cb_reduce_scaler, cb_out, ReduceInputPolicy::BulkWaitBulkPop>(ReduceInputBlockShape::of(R_b, 1))` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp:632` (reduce), `:112` (policy), `:290` (shape) | no post-op | `cb_sumsq_partial` | `cb_device_partial` / `cb_partial_col` | shape `(R_b, 1)` |
| `core_combine_block` (sum) | helper | `reduce<PoolType::SUM, ReduceDim::REDUCE_COL, cb_gather, cb_reduce_scaler, cb_device_partial, BulkWaitBulkPop, …, ReduceAlgorithm::AccumulateViaAdd, ReduceWithinTile::Skip>(ReduceInputBlockShape::of(G_w, R_b))` | `reduce_helpers_compute.hpp:156` (algorithm), `:191` (Skip), `:632` | lane-wise pairwise adds; the base op's root combine | `cb_gather` | `cb_device_partial` | `(G_w, R_b)` |
| `core_combine_block` (transport) | raw_api | `rms_norm_combine::combine_write_tiles<T32>` (copied), `noc_semaphore_inc`, `noc_async_write_barrier` | base `kernels/rms_norm_combine.hpp:28` | faces 0/2 | `cb_partial_col` | root `cb_gather` | `R_b` |
| `publish_stats_block`, `gather_slots_block` | raw_api | `noc_async_write` / `noc_async_read` (128 B), barriers, `noc_semaphore_inc`, `noc_semaphore_wait_min`, L1 word moves | `tt_metal/hw/inc/api/dataflow/dataflow_api.h` | — | `cb_device_partial`, landings | landings, `cb_stats_slots` | `R_b`, G |
| `scale_by_gamma_block` | helper | `eltwise_chain(grid(R_b, C), BinaryFpu<Mul, input(cb_x_tiles, None, None, Block, Enabled, Strided), input(cb_gamma_tiles, BroadcastDim::Row, Upfront/None, …, Row)>{StridedTileRange{c·C, S}}, PackTile<output(cb_x_gamma)>{})` | `chain.hpp:263` (StridedTileRange), `:331` (BroadcastInputSpec), `:491` | as base op scaling chain | `cb_x_tiles`, `cb_gamma_tiles` | `cb_x_gamma` | `(R_b, C)` |
| `device_sum_finalize_block` | helper | `eltwise_chain(grid(R_b, 1), CopyTile<input(cb_stats_slots, None, None, Block, Enabled, Strided), D0>{{0, G}}, [CopyTile<…, D1>{{g, G}}, AddBinary<D0, D1, D0>{}] for g = 1..G−1 (pack expansion over CT G), MulUnary<D0>{inv_W_bits}, AddUnary<D0>{eps_bits}, Rsqrt<Approx::Exact, D0>{}, PackTile<output(cb_inv)>{})` | `chain.hpp:483` (CopyTile), `:240-262` (Strided addressing), `eltwise/binary/sfpu/basic.hpp:20` (AddBinary), `eltwise/unary/scalar.hpp:26,32` (AddUnary, MulUnary), `eltwise/unary/math.hpp:38` (Rsqrt) | caller waits `G·R_b` once, pops after. `unpack_to_dest_mode[cb_stats_slots] = UnpackToDestFp32` | `cb_stats_slots` | `cb_inv_rms` / `cb_inv_root` | `R_b`; G (CT) |
| `distribute_inv_block` | raw_api | `combine_write_tiles<T32>`, `rotated`, `noc_semaphore_inc` | base `kernels/rms_norm_combine.hpp:28,43` | faces 0/2, rotated member order | `cb_inv_root` | members' `cb_inv_rms` | `R_b` |
| `scale_by_inv_rms_block` | helper | `eltwise_chain(grid(R_b, C), BinaryFpu<Mul, A, input(cb_inv_rms, BroadcastDim::Col, None, None, Col)>{…}, PackTile<output(cb_output, PerOuter, PerOuter)>{})`; A = `cb_x_gamma` (γ) or `cb_x_tiles` strided (no γ) | `chain.hpp:491`, `:331` | — | `cb_x_gamma` / `cb_x_tiles`, `cb_inv_rms` | `cb_output` | `(R_b, C)` |
| `load_block` / `store_block` / γ prefix reads | raw_api | `TensorAccessor::get_noc_addr`, `noc_async_read` / `noc_async_write`, barriers, CB reserve / push / wait / pop | `tech_reports/tensor_accessor/tensor_accessor.md`; base `kernels/rms_norm_reader.cpp`, `rms_norm_writer.cpp` | — | DRAM / `cb_output` | `cb_x_tiles`, `cb_gamma_tiles` / DRAM | `R_b`, S, C |
| `exchange_wave` | raw_api | `WorkerToFabricEdmSender::build_from_args<TENSIX>`, `.open()` / `.close()`, `PacketHeaderPool::allocate_header`, `fabric_set_unicast_route`, `to_noc_unicast_write` / `to_noc_fused_unicast_write_atomic_inc` / `to_noc_unicast_atomic_inc`, `send_current_slot_non_blocking`, `noc_async_writes_flushed`, `noc_semaphore_wait_min`, `noc_semaphore_inc` | `references/blackhole-fabric.md` §5; `tt_metal/fabric/hw/inc/tt_fabric_api.h:229`; AG `program_descriptor_with_inline_kernels.py` `_SENDER_SOURCE` (`route_one_hop` :184-196) | one connection per port | landing | peer landing | segment bytes, `num_waves` |
| compute ↔ release | raw_api | `cb_wait_front` / `cb_pop_front` | `tt_metal/hw/inc/api/compute/cb_api.h` | — | — | — | `R_b` |

**Helpers considered and rejected (raw_api rows):**

| Raw use | Candidate helper | Mismatch (file:line) | Why it cannot express the shape |
|---------|------------------|----------------------|---------------------------------|
| x / γ / output tile reads and writes | `read_sticks_for_tilize` / `write_sticks_after_untilize` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp:87-93,129-135` | stick (row-major) helpers; TILE pages are tiles, and `kernel_lib` has no tile-page reader helper |
| in-chip gather / inv distribution | `mcast_pipe` `McastArgs` / `SenderPipe` | `ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp:90` | multicast targets rectangles, and the compute grid has the port cores inside row 0 (lamp L5). The gather is many-to-one, which is not a multicast |
| stats compaction / expansion | `prepare_zero_tile` (`l1_helpers.hpp:62`), `tilize` | `l1_helpers.hpp:62`; `tilize_helpers.hpp:187` | neither moves 32 column-0 words between a tile and a 128 B vector; tilize consumes 32 rows of sticks per tile |
| Fabric send / relay / fence | `kernel_lib/ccl` `FabricStreamSender` | `ttnn/cpp/ttnn/kernel_lib/ccl/ccl_helpers_dataflow.hpp` (routes are 1D hop counts; `blackhole-ccl-kernels.md` §12: unmeasured, "not on main") | the measured path on this box is the AG raw sender; the typed layer is unmeasured. The implementer may adopt it if it expresses one-hop 2D routes |

## Broadcast Verification

| Phase | Op | CB_A (semantic name) Valid Region | CB_B (semantic name) Valid Region | Broadcast Dim |
|-------|-----|-----------------------------------|-----------------------------------|---------------|
| square_accumulate | Mul (DEST-accumulate) | `cb_x_tiles`: All | `cb_x_tiles`: All | None |
| collapse | reduce SUM REDUCE_ROW | `cb_sumsq_partial`: All | `cb_reduce_scaler`: Row0 | — (output Col0) |
| core_combine | reduce SUM REDUCE_COL AccumulateViaAdd Skip | `cb_gather`: Col0 (faces 0/2) | — | — (lane-wise; output Col0) |
| device_sum | SFPU add | `cb_stats_slots` slot g: Col0 | slot g' : Col0 | — (lane-wise; output Col0) |
| scale_by_gamma | Mul | `cb_x_tiles`: All | `cb_gamma_tiles`: Row0 | Row |
| scale_by_inv_rms | Mul | `cb_x_gamma` / `cb_x_tiles`: All | `cb_inv_rms`: Col0 | Col |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Landing overwritten by a faster chip's next call | the landing slots are reused every call with no host sync (trace) | ready fence per receiving port per call (AG `_needs_ready`), and counters re-armed by decrement |
| Global counters left non-zero | a traced or repeated call then reads early or hangs | every port consumes exactly its host-computed `n_in · num_waves` arrivals and 1 ready, then decrements by those counts; `noc_async_atomic_barrier()` at kernel end (base Perf 1 latent race) |
| Wrong denominator / missed or doubled contribution | the golden check scores the **worst** device | divisor `1/(G·W_local)` as fp32 bits RT arg; host-computed relay schedules give each device exactly G−1 peer slots + its own (asserted on host: each q ≠ p appears once in the receipts) |
| Different `inv` per device | per-device sum orders would differ in the last bit | slots indexed by device position, summed g = 0..G−1 on every device |
| fp32 lost in the exchange | FPU paths truncate fp32 to tf32 | the exchange and sum are 4 B moves + SFPU fp32 adds with `UnpackToDestFp32`; only the in-chip collapse / combine use the FPU (base op precision: fp32 PCC 1.000000) |
| Port-core semaphore id collision | `setup_fabric_connection` takes the lowest free ids | declare `SEM_*` first (Mechanism caps) |
| Bent mesh view (QuietBox 1×4 snake) | 2D multicast / direction-based routes leave the group | one-hop unicast by destination node only; Ring mock and Galaxy rings use the same code (never special-case the mock) |
| Line-end receive-only ports | they have no outgoing connection but must send the ready for their landing | the ready goes over the chip's other-direction connection (AG `_needs_ready`). Receive-only ports run the same kernel with `n_out = 0` |
| Program cache keyed without semaphore ids | two plans differing only in ids can collide (`blackhole-ccl-kernels.md` §5.1) | semaphore ids are fixed constants per role; plan-dependent values are CT args / RT args |
| Persistent handle lifetime | `generic_op` keeps only the address | module cache holds the scratch tensor and semaphores for the process |
| Regime depends on grid and L | port cores change the compute-core count | regime-pinned acceptance tests (Work Distribution) |
| Ragged ring drift (base Refinement 6) | ragged blocks pushing fewer pages than capacity cycles | nominal push / pop quanta on every shared ring (`blocking-model.md` §2) |

### Structural impossibilities

None missing from `feature_spec.py` INVALID.
