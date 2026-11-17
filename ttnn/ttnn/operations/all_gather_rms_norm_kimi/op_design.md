# Operation Design: all_gather_rms_norm_kimi

## Overview

| Field | Value |
|-------|-------|
| Classification | fused CCL: a Fabric 2D ring all-gather and an RMSNorm in **one** `ttnn.generic_op` (one `MeshProgramDescriptor`, one program per chip) |
| Goal | Every device of a `cluster_axis=1` ring group of G devices starts with its `(1,1,M,W_local)` slice of the hidden dim. Every device ends with the same RMS-normalized full `(1,1,M,W)` row block, `W = G·W_local`. Remote raw slices land in compute-core **L1**, never in DRAM. Fp32 per-row partial sums of squares go out over the same links **ahead of the bulk**. Each landed chunk is normalized on arrival with a scale tile that was precomputed before the chunk landed. |
| Math | `full = concat_p x[p]` (last dim, p = c). `ss[r] = Σ_q P_q[r]`, where `P_q[r] = Σ_{w∈slice q} x[r,w]²` in fp32 and the sum runs in fixed order q = 0..G−1. `rstd[r] = rsqrt(ss[r]·(1/W) + ε)`. `out[r,w] = x[r,w] · (rstd[r] · γ[w])` on every member. |
| Production call | MLA q_a latent of Kimi K3 / DeepSeek-V3: per device in `(1,1,640,384)`, out `(1,1,640,1536)`, G = 4 ring, 2 links, bf16, `fp32_dest_acc_en=True` |
| References | `ttnn/ttnn/operations/examples/fabric_all_gather/` (ports, `plan()`, probe placement, balanced ring, arrival counters, ready fence: copy and adapt). `examples/fabric_ring_exchange/README.md` (fence cost ≈ 5 µs; relay beats direct 2-hop). `examples/fabric_reduce_scatter/`. `.claude/references/blackhole-fabric.md`, `blackhole-ccl-kernels.md`, `blocking-model.md`, `l1-footprint-discipline.md`, `precision_convention.md`. Catalog: `examples/master.md` entries `compute_fusion`, `compute_block_size`, `sfpu_tile_scope`, `noc_placement`, `fabric_ring_exchange`. |
| Companion artifact | `l1_ledger.md`, in this directory: CB ledger and data-movement budget |

Every cross-device item below is labelled **TRANSPORT** (how bytes cross the fabric; taken from the examples' measurements) or **OVERLAP** (hiding the transport behind this op's own work; designed here, measured only in the op).

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` (mesh) | yes | per device `(1,1,M,W_local)`, M and W_local multiples of 32, bf16, TILE, DRAM interleaved, same on every device | — | addresses → common RT; shape → host-derived CT |
| `gamma` | `ttnn.Tensor` (mesh) | yes (keyword) | per device `(1,1,ceil(W/32),32)`, bf16, ROW_MAJOR, DRAM interleaved, same on every device; page = one 32-element stick = 64 B | — | address → common RT |
| `cluster_axis` | int | yes (keyword) | `1` (SUPPORTED); size of that axis ≥ 3 for Ring | — | host only |
| `epsilon` | float | no | > 0 | `1e-6` | fp32 bit pattern, CT arg of compute |
| `topology` | `Topology` | no | `Ring` (from `ttnn.operations.ccl import Topology` at import time) | `Ring` | host only |
| `num_links` | int | no | `1 ≤ num_links ≤` links that `ttnn.get_forwarding_link_indices` discovers on **every** hop of the group (including the wrap hop) | `2` | host (L, number of port pairs) |
| `compute_kernel_config` | `ttnn.ComputeConfigDescriptor` | no | `fp32_dest_acc_en=True` (gated); fidelity / approx not gated and honoured | `default_compute_kernel_config()` (HiFi4, fp32 acc, approx off) | compute config |

**Validation (caller errors, raise `ValueError`/`RuntimeError`, NOT support refusals):**

| Condition | Error |
|-----------|-------|
| `gamma` padded shape ≠ `(1,1,ceil(W/32),32)`, with W from the input and the mesh | `ValueError` whose message contains "gamma" |
| `num_links` > links discovered on any hop of a group (query every `p → p±1` hop, including the wrap) | `ValueError` |
| Ring requested but the wrap hop `(r, G−1) → (r, 0)` has no forwarding link under the active fabric config (no wrap link, or a config that does not wrap axis 1) | `ValueError` |
| `mesh_shape[cluster_axis] == 1` | `ValueError` |
| Grid cannot hold ports + receivers + hub + compute cores for the derived block (Work Distribution) | `ValueError` |

The op MUST NOT call `ttnn.set_fabric_config`. It reads `ttnn.get_fabric_config()` only for error messages. Never special-case the 1×4 snake mock: it is just the group the mesh gives.

**Registry gate:** the public entry point calls `validate()` first. `validate()` checks, in this order: SUPPORTED per axis (`dtype`, `fp32_dest_acc_en`, `layout`, `alignment`, `rank`, `gamma_dtype`, `gamma_layout`, `memory_layout`, `cluster_axis`, `topology`, `num_links`), then EXCLUSIONS. The axis names are the ones `feature_spec.py` TARGET uses. INPUT_TAGGERS hold exactly two taggers: `alignment` ("tile_aligned" / "w_non_aligned" / "h_non_aligned") and `rank`. `default_compute_kernel_config()` is the single exported factory; `None` resolves through it.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | per device `(1, 1, M, W_local)`; `Mt = M/32` tile rows, `Wl_t = W_local/32` tile columns |
| Dtype | bfloat16 |
| Layout | TILE |
| Memory | DRAM interleaved |
| gamma | `(1,1,Wt,32)` ROW_MAJOR sticks, `Wt = G·Wl_t`; stick `n` = γ[32n .. 32n+31] = the scale of output tile column n |

### Output

| Property | Value |
|----------|-------|
| Shape | per device `(1, 1, M, W)`, `W = G·W_local`. Group position p's slice is columns `[p·W_local, (p+1)·W_local)`; tile `(m, n)` is page `m·Wt + n` |
| Dtype | bfloat16 |
| Layout | TILE |
| Memory | DRAM interleaved, allocated per call with `ttnn.allocate_tensor_on_device` (identical address on every chip) |

### Persistent scratch (op-owned; allocated on first call, cached per (mesh id, geometry), same address on every chip)

| Name | Kind | Placement | Per-core size | Purpose |
|------|------|-----------|---------------|---------|
| `landing` | L1 sharded bf16 TILE tensor | compute cores, one shard each | `block_h·block_w` tiles | The core's x block. Own slice: loaded from DRAM. Remote slice: written by the fabric. Backs `cb_x_block` (`ttnn.cb_descriptor_from_sharded_tensor`). |
| `stats_scratch` | L1 sharded uint32 tensor | every op core (compute, ports, receivers, hub) | `G·Mt·128 B` (slot q, row m → offset `(q·Mt + m)·128`) | Fp32 partial vectors. The hub uses it as the collection point, ports as their fabric source staging, compute cores as their receive buffer. |
| global semaphores | `ttnn.create_global_semaphore` | ports, receivers, hub | — | Counters a remote chip increments: port `ready`, receiver `arrival`, hub `stats_fwd` / `stats_bwd`. Created once, followed by one `ttnn.synchronize_device`, then cached. Kept alive by the module cache. |

## Blocking Model

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| group position / slice `q` (G = 4 devices) | **dependent**: `ss[r]` sums over every slice | `slices_per_block` | 1 (mechanism cap: a block never straddles slices, because each slice arrives from a different chip and lands in a different stream) | host constant `SLICES_PER_BLOCK = 1` | The caller pre-shards this axis across chips (a physical shard of the dependent axis). On each chip, every slice q gets its own set of compute cores. The cross-chip combine is the fp32 partial exchange. | none: fixed by the contract |
| hidden within a slice, `Wl_t` = 12 tile columns | **dependent**: part of the row sum | `block_w` | `Wl_t` = 12, the whole slice width | host `block_w = Wl_t` (Phase 0), passed as CT arg | Not split across cores. The whole slice row of a block sits on one core; the combine across slices of a row is the stats path. | knob-turn (a divisor of `Wl_t`; adds same-slice partial cores) |
| rows, `Mt` = 20 tile rows | **independent**: every row normalizes alone | `block_h` | 1, derived: `block_h = ceil(Mt / floor(n_compute_avail / (G·Wl_t/block_w)))` with `n_compute_avail = grid − 2·2L − 1` = 101 gives 1. 1 is the core's **whole** assignment, so the block equals the assignment. | host-derived CT arg `block_h` | Row block `m` goes to its own core for each slice: `G·ceil(Mt/block_h)` = 80 compute cores. Links split row blocks contiguously (link l owns `[l·ceil(nb/L), (l+1)·ceil(nb/L))`). | knob-turn (block_h 2 halves the compute cores) |
| batch dims `(1,1)` | independent | folded | 1 (the contract pins them to 1) | contract | folded into rows | none |
| gamma's W (`Wt` sticks) | **reuse-shared**: the same γ scales every row | `block_w` (shared) | 12 sticks per core | `block_w` | Each core reads its 12 sticks from DRAM: 20 cores per slice re-read 768 B. | deferred broadcast (regime table) |
| **stage data: partial `P_q`** (rows × 1 col tile) | independent over rows | `block_h` | 1 | — | The 20 own-slice cores (m, p) each compute their row's partial. Spread over the grid, not centralized. | — |
| **stage data: rstd** (rows × 1) | independent over rows | `block_h` | 1 | — | Each compute core computes its own row's rstd from the G partials. 4× redundant per row, and it costs 1 tile of work. No core waits on another core's compute. | — |
| **stage data: scale S = rstd ⊗ γ** | independent | `block_h × block_w` | 1 × 12 | — | Every compute core computes its own block. | — |
| **communication roles** | — | — | — | — | **Hub** (1 core): collects own partials, forwards P_{p−1} to the forward port, multicasts the stats. It does communication only and has no compute stage. **Receivers** (2L cores): turn arrival counters into per-core landing notifications. Communication only. **Ports** (2L cores): fabric reader/sender. | — |

The scheme Phase 0 realizes (multi-core from day 1, every knob a parameter):

1. **Raw slices cross the fabric.** They go from L1 to L1: the own slice is read from the own-slice compute cores' `landing`, and remote slices land in the target compute cores' `landing`.
2. **Fp32 partials cross the same links ahead of the bulk.** Each is 128 B per tile row per slice.
3. **rstd and the scale `S = rstd ⊗ γ` are computed while the bulk is in flight** (the stall shadow).
4. **One FPU multiply per landed tile:** `out = x ⊙ S`. The writer then stores the chunk to DRAM.

Every extent (`block_h`, `block_w`, `chunk_tiles`, `port_batch_chunks`, `inc_every`, `num_links`, the grid) is a host constant or a derived CT/RT arg, defined once. CB sizes, loop trip counts and RT args are derived from it.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_port_chunks` (port cores) | `port_batch_chunks` (chunks per batch); capacity `2·port_batch_chunks` chunks | 4 (CB = 8 chunks = 48 tiles = 96 KiB) | The reader fills one batch while the sender drains the other; one barrier / flush per batch. `fabric_all_gather` measured 4 per batch as best or within 5% at every size; 2 starves, 8 is a batch short entries cannot fill. **TRANSPORT** |
| `cb_out` (compute) | `out_depth_chunks` | 2 chunks (= 12 tiles = `block_w` here) | Compute packs chunk k+1 while the writer stores chunk k |
| `cb_x_block` | none, block-resident | 1 block (`block_h·block_w`) | Residency: x is needed by the partial phase, the port reads and the final multiply. Backed on `landing`. |
| arrival increments | `inc_every` | 8 chunks, plus every entry's last chunk | Fused write+inc stalls the receiving router; the `fabric_all_gather` default. Notification lateness ≤ 7 chunks mid-stream, 0 at entry ends (the tail). **TRANSPORT** |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Router max packet payload (`ttnn.get_tt_fabric_max_payload_size_bytes()`, 14,336 B) | `chunk_tiles·tile_bytes` | `chunk_tiles` = largest divisor of `block_h·block_w` that is ≤ `floor(max_payload / tile_bytes)`. That is 6 here (12,288 B). | Packet rejected or split, and the landing offset arithmetic breaks |
| One packet lands at one NoC address, i.e. one core's contiguous `landing` | chunk must not straddle a block (core) | `chunk_tiles` divides `block_h·block_w`; tiles in `landing` are row-major within the block | Bytes land on the wrong core / overwrite |
| Slice origin | `block_w` divides `Wl_t`; `slices_per_block = 1` | host assert | A block whose tiles come from two chips' streams has no single notifier |
| Packet header pool: 12 per DM RISC | port sender headers = `port_batch_chunks` + 1 (ready) + stats entries (≤ 2) | `port_batch_chunks ≤ 8` (the example's cap) | `PACKET HEADER POOL EXHAUSTION` hang |
| DEST, fp32 half-sync = 4 tiles (`DEST_AUTO_LIMIT`) | live accumulators | partial chain: 1 DEST tile per row (WholeShape / PerRow); combine: 2 (D0, D1); multiplies: helper-chunked | Silent corruption |
| Core grid (`compute_with_storage_grid_size()`, 110 on QB2) | `G·ceil(Mt/block_h)·(Wl_t/block_w) + 2·2L + 1 ≤ grid` | `block_h` derived upward. If even `block_h = Mt` cannot fit, raise `ValueError`. | Placement fails |
| `stats_scratch` holds the whole `G·M·4 B` vector on every op core | `M` | predicate `G·M·4 B ≤ 64 KiB` (M ≤ 4096 at G = 4); else `ValueError` (outside the one cell) | L1 overflow |
| One ordered sender per counter | every counter whose threshold is computed (arrival, relay_ready, stats_fwd, stats_bwd, landed) | each counter has exactly one incrementing sender; unordered fan-in counters (`own_loaded`, `own_partials`) are waited only for their **total** | Early reads of data that has not landed |
| 16 semaphore ids per core; fabric setup takes the lowest free ids | ports | declare the op's own semaphores **before** `setup_fabric_connection` | "Semaphore id N already in use" |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **R1 L1-landing ring gather + fp32 stats-ahead, normalize-on-landing** | **built** | Ring, G even ≥ 4 (balanced) or G odd (unbalanced schedule from `_schedule`). Tile-aligned. Core-grid and stats caps above hold. Covers the one cell and every tile-aligned shape the caps admit. | `block_h × block_w` tiles of one slice on one core: 1 × 12 | **DRAM:** minimum. Own slice read once (240 tiles), output written once (960 tiles), plus γ re-reads (61 KiB, the deferred broadcast row below). **Fabric:** busiest link direction carries (G−1)/2 = 1.5 slices / L links, plus stats `2·M·4 B` = 5 KiB on link 0 forward. That is the roofline's link term to within M·4 B. **On-chip NoC:** own slice L1 → port (1.5 slices of reads, since each port re-reads its share), relays L1 → port (0.5 slice), hub stats multicast 10 KiB. | Fewer per-block fixed costs on each core: partial chain init + reduce init + SFPU inits (combine/rsqrt) + the γ / scale setup are paid **once per block**. Larger `block_h` also halves the compute-core count, so fewer landing notifications and fewer multicast targets. The cost is a longer tail (more tiles per landing notification, see the grid-sync lamp). |
| R2 gather into the output DRAM, then a norm pass reading it back | **rejected**: dead end, superseded by R1 | — | — | DRAM 2640 tiles (land 720 + re-read 960 on top of the minimum 1200) = 5.4 MB ≈ 12 µs at 447 GB/s, above the 7.6 µs link term. The norm is serialized after the last landing. No good scheme passes through it. | — |
| R3 stats round first, then gather **normalized** slices straight into peers' output DRAM | **rejected**: superseded by R1 | — | own slice 1 × 12 | Same fabric bytes. DRAM +240 tiles (relays re-read from DRAM) +240 (own normalized write). The stats round (own partial → one relay hop → rstd → normalize) becomes a **serial prefix before the first bulk byte**. R1 moves the same bytes with the stats overlapped and leaves only one multiply per tile on the tail. | — |
| R4 no fabric stats: each chip square-reduces remote slices itself and combines on-chip among the G cores of a row | **rejected**: superseded by R1 | — | 1 × 12 | Fabric −M·4 B per slice (−0.5%). It puts square + reduce + an on-chip combine round **on the tail after the last landing**. R1 moves that work before the landing. | — |
| R5 rows-only split (one core per tile row, the whole W on it) | **rejected**: superseded by R1's W-split | — | 1 × 48 | Identical DRAM and fabric bytes. Only 20 compute cores. Each core does 4× the tile work (≈ 96 FPU tile-ops), which exceeds the ~10 µs transport window, and its last landing still needs all of its stats. Occupancy decides here, at equal traffic (step 2 of the ordered rule). | — |
| R6 γ read once per chip + on-chip multicast | **deferred**: R1 re-reads 61 KiB of γ against 2.5 MB of DRAM traffic (≈ 0.1 µs). It layers onto R1 unchanged: `cb_gamma` keeps its shape and only its producer changes from a DRAM read to a multicast receive. | — | — | −58 KiB DRAM | — |
| R7 far slice (p+2) by direct 2-hop send or fabric multicast instead of a relay (**TRANSPORT**) | **deferred**: `fabric_ring_exchange` measured relay 1.1–1.6 µs cheaper than direct, because direct needs a fence with the 2-hop chip. It layers onto R1: only the port sender's destination set and the fence change, not the landing / notification structure. | — | — | −0.5 slice of relay L1 reads | — |
| R8 ports read the own slice from DRAM (as `fabric_all_gather` does) | **rejected**: superseded by R1's L1 source | — | — | +2 reads of the own slice from DRAM per link half (+480 tiles). The own cores load it within ~1 µs, inside the ≈ 5 µs fence. | — |

**Regime-selection function:** only R1 is built, so there is no selection. The host computes `block_h` (a knob-turn within R1) and raises `ValueError` outside the caps. A regime-pinned test covers `block_h > 1` (`M = 2048`, `Mt = 64`: `4·64 = 256` > 101 free cores, so `block_h = ceil(64/25) = 3` and `ceil(64/3) = 22` row blocks); see the acceptance test.

### Traffic ranking

Per device, G = 4, 640 × 384 bf16, slice S = 480 KiB. "Crossing" means the whole tensor moves once across that tier.

| Candidate split / scheme | DRAM | Fabric (busiest link direction) | On-chip NoC | Rank |
|--------------------------|------|---------------------------------|-------------|------|
| R1: W-split by slice (dependent axis) × rows, fp32 stats combine, L1 landing | own x ×1, out ×1 (minimum) | 1.5 S / L, plus 5 KiB stats | own x ×1.5 + relay 0.5 S (L1 → port), 10 KiB stats multicast | **1 (chosen)** |
| R5: rows only (independent axis only) | same | same | same minus 0 | 1 (tie on traffic; loses on occupancy and per-core work, see R5) |
| R3: normalize-then-gather into peers' DRAM | +2 × 0.5 S | same | less | 3 |
| R4: local partials, on-chip combine | same | −0.5% | +G·M·4 B per row group | 2 on traffic; rejected on tail latency |
| R2: gather to DRAM, then norm | +1.5 × (G−1)·S | same | — | 5 |

Tiebreak: R1 and R5 move equal bytes. R1 wins on occupancy (80 vs 20 cores) and leaves one FPU op per tile on the tail. The cheapest-traffic split is the one implemented.

### Block schedule

The logical schedule per chip p. Reader, compute, writer, port, receiver and hub kernels are separate and asynchronous.

```cpp
// ---- every compute core (m_blk, q) ----
prepare_gamma_block();                       // t0: γ sticks → row 0 of block_w tiles (cb_gamma)
if (q == p) {                                // own-slice cores (20)
    load_own_block();                        // DRAM → landing (cb_x_block), then own_loaded++ on its link's 2 ports
    partial_block();                         // Σ_w x² over block_w tiles in DEST (fp32) → collapse the row → cb_partial
    export_partial();                        // col 0 (32 fp32 per tile row) → hub stats slot p, own_partials++ on the hub
}
wait_stats();                                // hub multicast landed: stats_ready
combine_stats_block();                       // ss = P_0+P_1+P_2+P_3 (fixed order, SFPU fp32) ·(1/W) +ε → rsqrt → cb_rstd
expand_rstd_block();                         // col-broadcast rstd → full tile (cb_rstd_full)
scale_block();                               // S = rstd_full ⊙ γ(row-bcast) over block_h × block_w → cb_scale (fp32)
normalize_block();                           // out = x ⊙ S, consuming cb_x_block in landed-chunk quanta → cb_out
store_block();                               // cb_out → DRAM output tiles (m, q·Wl_t + j), per chunk

// ---- every port core (link l, dir d ∈ {fwd, bwd}) ----
fence();                                     // send ready to the peer's opposite port; wait own ready ≥ 1; consume it
send_entries();                              // entries [own slice rows of link l, relay half] in batches;
                                             //   at each batch boundary, send newly-ready stats entries (link 0 only)

// ---- every receiver core (link l, dir d) ----
notify_landed_stream();                      // walk the upstream's chunk stream: arrival ≥ threshold →
                                             //   landed++ on the chunk's core; relay_ready++ on the local port per relay block
drain_arrival();                             // wait arrival ≥ expect_in, decrement by expect_in

// ---- hub core ----
forward_stats();                             // own_partials = nb: slot p → both link-0 ports' stats_scratch, stats_tx++;
                                             //   stats_fwd ≥ 1: slot p−1 → fwd port, stats_tx++
broadcast_stats();                           // stats_fwd ≥ 2 and stats_bwd ≥ 1: multicast stats_scratch to compute cores, stats_ready++
drain_stats();                               // stats_fwd −= 2, stats_bwd −= 1
```

| Block operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------------|-------------|--------------------|-----------------------------------|
| `prepare_gamma_block` | `block_w` sticks → `block_w` tiles (row 0) | `cb_gamma` stays until `scale_block` | 1 DRAM barrier per block |
| `load_own_block` | `block_h·block_w` tiles | `cb_x_block` (landing) for the rest of the call | 1 barrier, 1 push, 2 semaphore incs per block |
| `partial_block` | `block_h × block_w` → `block_h` fp32 tiles → `block_h` col tiles | x not popped | 1 chain init + 1 reduce init per block |
| `export_partial` | `block_h` × 128 B | — | 1 NoC write + 1 inc per block |
| `combine_stats_block` | G·block_h fp32 col tiles → block_h | — | SFPU inits (add, mul-scalar, add-scalar, rsqrt) once per block |
| `expand_rstd_block` | block_h tiles | — | 1 unary-bcast init per block |
| `scale_block` | `block_h × block_w` | `cb_scale` until `normalize_block` ends | 1 binary-bcast init per block |
| `normalize_block` | `block_h × block_w`, waited in `chunk_tiles` quanta | — | 1 binary init per block; 1 CB wait per landed chunk |
| `store_block` | `chunk_tiles` per write batch | — | 1 write barrier per chunk |
| port `send_entries` | `chunk_tiles` per packet, `port_batch_chunks` per batch | — | 1 read barrier / 1 flush per batch; fused increment every `inc_every` chunks + at each entry's last chunk |
| receiver `notify_landed_stream` | per covered chunk | — | 1 on-chip inc per landed chunk, 1 per relay block |
| hub `broadcast_stats` | `G·Mt·128 B` | — | **once per call**: 1 multicast write + 1 multicast semaphore inc per compute-core rectangle |

**Stall-shadow decisions (OVERLAP):**

| Stage that waits | Waits for | Work scheduled into the wait | Why it is legal |
|------------------|-----------|------------------------------|-----------------|
| Port sender: the ready fence (≈ 5 µs, `fabric_ring_exchange`) | peer chip started | Own cores load x, compute and export partials. The hub forwards P_p to the ports. Every core prepares γ. Port readers stage the first batch. | None of this depends on any peer |
| Compute core: remote landing | that core's chunks from the fabric | `combine_stats` → `expand_rstd` → `scale_block` (precompute `S = rstd ⊗ γ`) | **Associativity of the product**: `x·rstd·γ = x·(rstd⊗γ)`. This changes the fp order (one tf32 rounding of S instead of two of the intermediates) and is gated on the golden tolerance (`helpers.TOLERANCES[(bf16, True)]`). It is ≥ as accurate as the two-multiply order. |
| Port sender: between batches | — | Sends stats entries as soon as the hub marks them ready (dynamic position), so the stats never wait behind the bulk | Stats increments go to the hub's own counters, never to the arrival counter, so bulk thresholds do not depend on where a stats packet sits |
| Port reader: relay chunk | relay_ready | Nothing. The reader has already sent the whole own entry, which in a balanced ring takes as long as the upstream's matching entry. | — |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Arrival increment interval `inc_every = 8` (**TRANSPORT**) | Measured with DRAM landing. L1 landing acks faster, so the fused-inc stall is cheaper, and finer notification shortens the mid-stream lateness. | `inc_every` 2, 4 (one increment per block end) |
| Chunk 6 tiles = 12 KiB (**TRANSPORT**) | 14 KiB packets (7 tiles) amortize the per-packet cost 7/6 better, but 7 does not divide a 12-tile block | `block_w` choices whose block tile count is a multiple of 7 (e.g. `block_h = 7` once Mt allows), or scatter-write packets (unmeasured) |
| Grid synchronization: 80 cores × 12 tiles each | Per-core work is small. Notification and multicast fan-out (80 targets) may cost more than the parallelism buys. | `block_h = 2`: 40 compute cores, 24 tiles each |
| Overlap: `normalize_block` waits per landed chunk | One multiply per tile plus a DRAM write per chunk is the exposed tail | Finer last-entry chunks (shorter tail) vs a whole-block wait (fewer handshakes) |
| Stats on link 0 only, dynamic insertion | The relayed P_{p−1} reaches p+1 one hop after it lands. If it lands late, S precompute starts late. | Send P_{p−1} on link 1's forward port, or by direct 2-hop (R7) |
| Port batch `port_batch_chunks = 4` (**TRANSPORT**) | 30 chunks per port is a short stream (the example's best was for larger ones) | 2 and 3 per batch |
| Common RT args on compute cores (dispatch) | Per-core RT args cost a flat ~3 µs of dispatch | Already common. Watch DISPATCH-BOUND in the harness summary. |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| γ DRAM → compute core | bf16 64 B sticks | Compute reader: `TensorAccessor` page n read into a 64 B-aligned spot inside the target tile, then a RISC copy to face 0 row 0 (elements 0–15) and face 1 row 0 (elements 16–31) | Rows 1–31 are never read (row broadcast); no zero fill |
| own x DRAM → `landing` (own cores) | bf16 tiles | Compute reader `noc_async_read` of tile pages `m·Wl_t + j` into `cb_x_block`; one barrier; `noc_semaphore_inc` of `own_loaded` on the 2 ports of link l(m) | 20 cores × 24 KiB, round-robin banks |
| partial export (own cores) | fp32, 32 values per tile row | Writer reads col 0 of `cb_partial` (face 0 rows 0–15 word 0, face 2 rows 0–15 word 0), packs 128 B into its own `stats_scratch[p][m]`, writes it to the hub's `stats_scratch[p][m]`, incs the hub's `own_partials` | fp32 kept bit-exact |
| hub → link-0 ports (**OVERLAP**) | fp32 vectors (`Mt·128 B`) | Hub writes slot p into both link-0 ports' `stats_scratch[p]` (and slot p−1 into the forward port's on arrival); `stats_tx++` on each port (one ordered sender: the hub) | Port senders send only from their own L1 |
| fabric stats (**TRANSPORT**) | fp32 | Port sender (link 0): fused write+inc of `Mt·128 B` into the peer hub's `stats_scratch[q]`; increments the peer hub's `stats_fwd` (fwd port) or `stats_bwd` (bwd port). Forward port sends P_p, then P_{p−1} (relay); backward port sends P_p. | Each chip receives P_{p−1}, P_{p−2} from forward and P_{p+1} from backward |
| fabric bulk (**TRANSPORT**) | bf16 tiles, `chunk_tiles` per packet | Port reader (NCRISC, NoC0) reads chunks from compute cores' `landing` into `cb_port_chunks`. Port sender (BRISC, NoC1) sends each chunk into the **same `landing` offset of the same logical core** on the peer (NoC coords from `mesh_device.worker_core_from_logical_core`, uniform across chips). It attaches a fused increment of the peer receiver's `arrival` every `inc_every`-th chunk and on each entry's last chunk. | Balanced ring from `_schedule`. Forward (l): [own slice rows R_l, relay of slice p−1 rows R_l first half]. Backward (l): [own rows R_l, relay of slice p+1 rows R_l second half]. |
| landing notification (**OVERLAP**) | — | Receiver (l, d) walks the upstream's static chunk stream: when `arrival` ≥ the host-computed threshold of chunk k, it does `noc_semaphore_inc(landed@core(k), 1)`. For relay blocks it also does `relay_ready@port(l,d) += 1` once both chunks are covered. | Each compute core's `landed` has exactly one receiver as sender |
| stats broadcast (**OVERLAP**) | fp32 | Hub multicasts its whole `stats_scratch` (G·Mt·128 B = 10 KiB) to the compute-core rectangle(s), then a multicast `stats_ready` inc. Mechanism: the multicast helper (`McastArgs` sender `send()` + `send_signal()`), or `noc_async_write_multicast` + `noc_semaphore_inc_multicast` when the core set is not equal groups. | Every compute core can then build its own G partial tiles |
| compute reader: stats → tiles | fp32 tiles, col 0 valid | Writes the 32 values of `stats_scratch[q][m]` into col 0 of tile q of `cb_partials` (G tiles), push G | Unused lanes are garbage; only col 0 is read later (col broadcast) |
| compute reader: landed x → `cb_x_block` | bf16 | remote: `cb_reserve_back(block)`; per chunk k: wait `landed ≥ k+1`, `cb_push_back(chunk_tiles)`. Zero-copy: the bytes are already there. | |
| compute → writer → DRAM | bf16 | `cb_out` per chunk; writer `noc_async_write` to output pages `m·Wt + q·Wl_t + j`; barrier per chunk | 80 cores, NoC1 |

**Fence and counter discipline (TRANSPORT, copied from `fabric_all_gather`):**
- Each port opens by sending one `unicast_atomic_inc` to the **ready** counter of the peer port that writes into this chip. It then waits for its own ready ≥ 1 and decrements by 1.
- Fabric bulk, stats and the peer hub writes all start only after the fence. This protects every peer `landing` and `stats_scratch` from call N+1 writes while call N still reads them.
- Every global counter is re-armed by **decrementing what was consumed**, never by setting 0:
  - receiver `arrival` −= `expect_in`;
  - hub `stats_fwd` −= 2, `stats_bwd` −= 1;
  - port `ready` −= 1.
- The program-local semaphores (`own_loaded`, `own_partials`, `stats_tx`, `relay_ready`, `landed`, `stats_ready`) are reset by dispatch at every launch.

**Placement (memory_layout axis):** TARGET has `INTERLEAVED` only, so there are no physical shard variants. If `*_SHARDED` inputs ever enter TARGET:
- **WIDTH_SHARDED (cutting W):** a scheme-change only in the input path. Each core's slice block would already be resident; consume it zero-copy through a CB backed on the shard instead of `load_own_block`. The dependent axis is already combined by the stats path.
- **HEIGHT_SHARDED (rows):** a knob-turn. Rows are independent.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one block = `block_h` tile rows × `block_w` tile columns of one slice q, on one core |
| Grid | `mesh_device.compute_with_storage_grid_size()` (11 × 10 = 110 on QB2), the same logical layout on every chip |
| Port cores | `2·L` = 4. One per (direction, link), placed by the example's probe + `_worker_nearest_noc1` (fewest NoC1 hops to its link's Ethernet core). Sender on NoC1. |
| Receiver cores | `2·L` = 4, one per incoming (direction, link). Any free cores. |
| Hub | 1 free core |
| Compute cores | `n_compute = G·nb·(Wl_t/block_w)`, with `nb = ceil(Mt/block_h)` row blocks. This is 80 = 4·20·1. Chosen as the first `n_compute` cores, row-major, of the grid rows that contain no port core, so they form rectangles (one 10 × 8 rectangle on QB2 with ports in row 0). Core index `i = q·nb + m` (or any fixed bijection), recoverable on device from the logical coordinate, so compute kernels take **common** RT args only. |
| Per-core work | own cores (q = p): load + partial + export + combine + scale + normalize/store `block_h·block_w` tiles. Remote cores: combine + scale + normalize/store. |
| Rows per link | link l owns row blocks `[l·ceil(nb/L), min(nb, (l+1)·ceil(nb/L)))`. A relay half = first `floor(n_l/2)` (part 1, forward) / the rest (part 2, backward), as `_schedule` with `balance=True`. |
| Tile geometry | `Mt = ceil(M/32)`, `Wl_t = ceil(W_local/32)`, `Wt = G·Wl_t`, `nb = ceil(Mt/block_h)`. The last row block may be short (`last_block_h = Mt − (nb−1)·block_h`): it keeps the nominal CB push/wait counts, and only the NoC transfers and compute use the actual count. Chunks per block = `block_h·block_w/chunk_tiles`. |
| Idle cores | 110 − 89 = 21 (Grid-sync lamp) |
| Host-counted args | Per-port chunk lists, per-receiver thresholds, `expect_in`, relay flags: all computed on the host (blackhole-fabric rule 5). They are cached with the program (only tensor addresses change per call). |

## Circular Buffers

Compute cores (all 80; own-only CBs exist only on own cores but are declared uniformly):

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|----------|
| `cb_x_block` | 0 | tile bf16 (2048) | `block_h·block_w` | spans rows and width of the block; backed on `landing` (sharded, zero-copy) | Float16_b | reader | compute | whole call |
| `cb_gamma` | 1 | tile bf16 | `block_w` | spans width; row 0 valid | Float16_b | reader | compute | until `scale_block` ends |
| `cb_out` | 2 | tile bf16 | `out_depth_chunks·chunk_tiles` (= 12) | streams width/rows in chunk quanta; **shares the allocation with `cb_gamma`** (disjoint lifetimes, same format) | Float16_b | compute | writer | `normalize_block` … end |
| `cb_partials` | 3 | tile fp32 (4096) | `G·block_h` | spans slices × rows; col 0 valid; `UnpackToDestFp32` | Float32 | reader | compute | `combine_stats_block` |
| `cb_xsq_sum` | 4 | tile fp32 | `block_h` | per row: Σ x² over the width in fp32 | Float32 | compute | compute | `partial_block` (own cores) |
| `cb_rstd` | 5 | tile fp32 | `block_h` | **shares with `cb_xsq_sum`** | Float32 | compute | compute | `combine` → `expand` |
| `cb_partial` | 6 | tile fp32 | `block_h` | reduced col tile | Float32 | compute | writer | `partial_block` → `export_partial` |
| `cb_rstd_full` | 7 | tile fp32 | `block_h` | **shares with `cb_partial`** | Float32 | compute | compute | `expand` → `scale_block` |
| `cb_scale` | 8 | tile fp32 | `block_h·block_w` | spans the block; precomputed before the landing | Float32 | compute | compute | `scale_block` → end |
| `cb_scaler` | 9 | tile bf16 | 1 | constant 1.0 (exact) for SUM; `prepare_reduce_scaler<cb_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>(1.0f)` | Float16_b | reader | compute | own cores, whole call |

Port cores:

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|----------|
| `cb_port_chunks` | 0 | tile bf16 | `2·port_batch_chunks·chunk_tiles` (= 48) | streams the entry; two batches in flight; a batch ends at a multiple of `port_batch_chunks` and at each entry end, so a reservation never wraps | Float16_b | port reader | port sender | whole call |

Receivers and the hub have no CBs. They use `stats_scratch` and semaphores only.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `prepare_gamma_block` | `block_w` | dataflow (raw) | DRAM γ sticks | `cb_gamma`, `block_w` | pushed, held |
| 2 | `load_own_block` | `block_h·block_w` | dataflow (raw) | DRAM x | `cb_x_block`, `block_h·block_w` | pushed, held (landing contents persist after pops) |
| 3 | `partial_block` (a) | `grid(block_h, block_w)` | `eltwise_chain` `BinaryFpu<Mul>` x·x, `DestAccumulation::PerRow` | `cb_x_block` (wait upfront, **no pop**) | `cb_xsq_sum`, `block_h` | x still waited |
| 3 | `partial_block` (b) | `ReduceInputBlockShape::of(block_h, 1)` | `reduce<SUM, REDUCE_ROW, …, ReduceFp32Mode::Accurate>` | `cb_xsq_sum` (bulk wait/pop), `cb_scaler` (held) | `cb_partial`, `block_h` | — |
| 4 | `export_partial` | `block_h` × 32 fp32 | dataflow (raw) | `cb_partial` | hub `stats_scratch` | popped |
| 5 | `combine_stats_block` | `block_h` rows, G tiles each | `eltwise_chain`: `CopyTile`(q=0→D0), then per q = 1..G−1 `CopyTile`(→D1) + `AddBinary`(D0,D1→D0); `MulUnary`(1/W); `AddUnary`(ε); `Rsqrt`; `PackTile`(D0) | `cb_partials` (G·block_h, wait upfront, pop at end) | `cb_rstd`, `block_h` | — |
| 6 | `expand_rstd_block` | `block_h` | `unary_bcast<BroadcastDim::Col>` | `cb_rstd` | `cb_rstd_full`, `block_h` | — |
| 7 | `scale_block` | `grid(block_h, block_w)` | `mul<A = cb_rstd_full (Col mapping, upfront, pop at end), B = cb_gamma (BroadcastDim::Row, Row mapping, upfront, pop at end)>` | `cb_rstd_full`, `cb_gamma` | `cb_scale`, `block_h·block_w` | `cb_gamma` memory free → `cb_out` |
| 8 | `normalize_block` | `grid(block_h, block_w)`, waited per chunk | `mul<A = cb_x_block (waited in chunk quanta, popped), B = cb_scale (Block mapping, upfront, pop at end)>` | `cb_x_block`, `cb_scale` | `cb_out`, chunk quanta | — |
| 9 | `store_block` | chunk | dataflow (raw) | `cb_out` | DRAM output | popped |

## API Mapping

Paths are relative to `ttnn/cpp/ttnn/kernel_lib/`.

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| compute boot | raw_api | `compute_kernel_hw_startup(cb_x_block, cb_x_block, cb_out)` | compute API (every helper's file-level prerequisite) | once, before any helper | — | — | — |
| `partial_block` (a) | helper | `eltwise_chain(IterationShape::grid(block_h, block_w), BinaryFpu<input(cb_x_block, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Block), same, BinaryFpuOp::Mul, BroadcastDim::None, Dst::D0, DestAccumulation::PerRow>{}, PackTile<output(cb_xsq_sum, …, DestAccumulation::PerRow)>{})` | `eltwise/api/chain.hpp:533` (`eltwise_chain`), `:491` (`BinaryFpu`), `:504` (`PackTile`), `:283` (`DestAccumulation`), `:356` (`input`), `:383` (`output`) | same CB on both operands: one wait, x·x exact in fp32 DEST, summed in DEST per row | `cb_x_block` | `cb_xsq_sum` | `block_h`, `block_w` (the `grid` shape) |
| `partial_block` (b) | helper | `reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, cb_xsq_sum, cb_scaler, cb_partial, ReduceInputPolicy::BulkWaitBulkPop, ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT, ReduceFp32Mode::Accurate>(ReduceInputBlockShape::of(block_h, 1))` | `reduce_helpers_compute.hpp:632`; policy `:112`; `ReduceFp32Mode` `reduce_helpers_common.hpp:18` (Accurate = fp32 through the SFPU, no tf32 truncation) | scaler 1.0 | `cb_xsq_sum`, `cb_scaler` | `cb_partial` | `block_h` |
| scaler | helper | `dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>(1.0f)` | `reduce_helpers_dataflow.hpp:59` | pool-type-aware overload; bf16 packed tile | — | `cb_scaler` | — |
| `combine_stats_block` | helper | `eltwise_chain(IterationShape::tiles(block_h) …, CopyTile<…, D0>, CopyTile<…, D1>, AddBinary<D0,D1,D0>, …, MulUnary{bits(1/W)}, AddUnary{bits(ε)}, Rsqrt<>, PackTile<output(cb_rstd)>)` | `CopyTile` `eltwise/api/chain.hpp:483`; `TileAddressing::Offset` `:261`; `AddBinary` `eltwise/binary/sfpu/basic.hpp:20`; `AddUnary`/`MulUnary` `eltwise/unary/scalar.hpp:26/32`; `Rsqrt` `eltwise/unary/math.hpp:38` | the G partials are addressed by tile offset (q·block_h + row); fixed order q = 0..G−1; every add is an SFPU fp32 add | `cb_partials` | `cb_rstd` | `block_h` |
| `expand_rstd_block` | helper | `unary_bcast<BroadcastDim::Col, input(cb_rstd), output(cb_rstd_full)>(IterationShape::tiles(block_h))` | `eltwise/broadcast/bcast.hpp:25` (`UnaryBcast`), `:28` (`unary_bcast`) | — | `cb_rstd` | `cb_rstd_full` | `block_h` |
| `scale_block` | helper | `mul<input(cb_rstd_full, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Col), input(cb_gamma, BroadcastDim::Row, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Row), output(cb_scale)>(IterationShape::grid(block_h, block_w))` | `eltwise/api/convenience.hpp:51`; `BroadcastDim` `eltwise/api/chain.hpp:305` | broadcast on srcB only (γ row 0 → all rows) | `cb_rstd_full`, `cb_gamma` | `cb_scale` | `block_h`, `block_w` |
| `normalize_block` | helper | `mul<input(cb_x_block, …block-size quanta = chunk_tiles…), input(cb_scale, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Block), output(cb_out)>(IterationShape::grid(block_h, block_w).block_size(chunk_tiles))` | `eltwise/api/convenience.hpp:51`; `block_size` `eltwise/api/chain.hpp:121-148` | `.block_size` is clamped to `DEST_AUTO_LIMIT` (`dest_helpers.hpp:103`). The CB wait quantum is what matters: x is waited cumulatively in `chunk_tiles` steps. | `cb_x_block`, `cb_scale` | `cb_out` | `chunk_tiles`, `block_h`, `block_w` |
| port reader / sender, fence, counters | raw_api | `WorkerToFabricEdmSender::build_from_args`, `to_noc_fused_unicast_write_atomic_inc`, `to_noc_unicast_write`, `to_noc_unicast_atomic_inc`, `send_current_slot_non_blocking`, `noc_semaphore_wait_min`, `noc_semaphore_inc` | `ttnn/ttnn/operations/examples/fabric_all_gather/program_descriptor_with_inline_kernels.py:118-297` (reader 85-170, sender 172-297) | Chunk walk replaced by the block-ordered walk over `landing` sources (L1, not DRAM) | — | — | `chunk_tiles`, `port_batch_chunks`, `inc_every` |
| γ sticks, x load, export, stats → tiles, store | raw_api | `TensorAccessor`, `noc_async_read/write`, barriers, RISC word copies | `tech_reports/tensor_accessor/tensor_accessor.md` | — | — | — | `block_w`, `block_h` |
| hub stats multicast | helper (preferred) | `McastArgs<>` `sender(noc).send()` / `send_signal()`; receivers `receive_signal()`; host `ttnn.Mcast` | `mcast/kernel/mcast_args.hpp:238`, `mcast/kernel/mcast_pipe.hpp:74,82,149`, host `mcast/host/mcast_host.hpp:131` | — | — | — | — |

**Helpers considered and rejected (`raw_api` rows):**

| Raw use | Candidate helper | Mismatch (file:line) | What would close it |
|---------|------------------|----------------------|---------------------|
| fabric port reader / sender | `ccl/ccl_helpers_dataflow.hpp` `FabricStreamSender` (`:486`), `FabricStream` (`:409`), `FusedWriteIncChannel` (`:389`) | Usable for the sender. `blackhole-ccl-kernels.md` §12 marks the CCL helper library **[unmeasured]**, while the raw form in `fabric_all_gather` is the measured transport this design cites (TRANSPORT numbers come from the examples). The implementer **may** use `FabricStreamSender` with `DirectConn` (`:190`) if it reproduces the example's per-batch flush and dynamic stats insertion. | Measuring the CCL helper against the example |
| γ sticks → tile row 0 | `tilize<…>` `tilize_helpers.hpp:186-196`; `read_sticks_for_tilize` `tilize_helpers_dataflow.hpp:87-93` | `read_sticks_for_tilize` assumes one accessor page = one full row (`tilize_helpers_dataflow.hpp:87`), but γ's page is a 32-element stick. `tilize` asymmetric mode treats each page as a row of width `block_width_tiles·32` (`tilize_helpers.inl:195-240`). A row broadcast needs only row 0, so no tilize is required at all. | — |
| partial export / stats → col-0 tiles | none (no helper moves a column of a tile) | — | — |
| x load / output store | `TensorAccessor` is the helper; there is no block-load helper for interleaved tiles | — | — |

## Broadcast Verification

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast Dim |
|-------|-----|-------------------|-------------------|---------------|
| `partial_block` | Mul (x·x) | `cb_x_block`: All | `cb_x_block`: All | None |
| `scale_block` | Mul | `cb_rstd_full`: All (column broadcast already expanded) | `cb_gamma`: Row0 | Row |
| `normalize_block` | Mul | `cb_x_block`: All | `cb_scale`: All | None |
| `combine_stats_block` | SFPU Add on DEST | `cb_partials` tiles: Col0 | Col0 | — (garbage lanes never read downstream: `expand_rstd` broadcasts col 0) |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Fp32 partial truncated to tf32 | FPU unpack of an fp32 CB into srcA/srcB keeps 19 bits; the prompt requires fp32 on the wire and in the sum | Partial reduce uses `ReduceFp32Mode::Accurate` (SFPU). The combine is `CopyTile` from a `UnpackToDestFp32` CB plus SFPU `AddBinary`. 1/W is applied in fp32 by `MulUnary` (a bf16 scaler of 1/1536 would be inexact). |
| `x²` packed to bf16 | 16-bit page under fp32 DEST truncates (audit 2) | Σx² is accumulated in DEST; `cb_xsq_sum` is Float32 |
| Determinism across calls and devices | the regression suite checks bit-identical repeats | Fixed sum order q = 0..G−1, independent of arrival order. Every P_q is computed only on device q. No arrival-order-dependent arithmetic anywhere. |
| Relay or normalize reading `landing` before it lands | counters with several senders, or dynamic packet order | One ordered sender per thresholded counter. Stats increments go to the hub's counters, never to `arrival`. Fan-in counters are waited for their totals only. |
| `landing` overwritten by the next call | calls run back to back in traces with no host sync | The ready fence (one per port per call) precedes every fabric write. Output is not reused between calls, and landing / stats are consumed inside the call. |
| In-place normalize over `landing` | ports read x from `landing` (own sends and relays) while compute runs | Output goes to `cb_out` (aliased with dead `cb_gamma`), never back into `landing` |
| Program-cache collision | `generic_op` hashes kernel source, core ranges, CT args and only the outer RT grid size (`fabric_ring_exchange` gotcha) | Set `custom_program_hash` from (G, M, W_local, num_links, block knobs, mesh shape). Per call, patch only the tensor-address RT args. |
| Global semaphores / scratch lifetime | `generic_op` keeps only addresses | Module-level cache keyed by `id(mesh_device)` and geometry holds the handles. Create once, followed by one `ttnn.synchronize_device`; the first call happens outside trace capture. |
| Semaphore ids collide with fabric setup | `setup_fabric_connection` takes the lowest free ids | Declare the op's `SemaphoreDescriptor`s before calling it |
| γ stick 32 B DRAM reads | Blackhole DRAM read alignment is 64 B | Read the whole 64 B stick to a 64 B-aligned L1 spot, then copy with the RISC |
| DISPATCH-BOUND | per-core CoreRanges and per-core RT args are written core by core | Compute cores form merged rectangles and use common RT args only (derive (m, q) from the logical coordinate). Only the 9 port / receiver / hub cores carry per-core args. |
| Mock vs Galaxy | QB2 1×4 snake vs Galaxy 8×4 (8 groups) | `plan()` builds groups from `mesh_shape` and `cluster_axis`. One program per chip, the chip's group position p = c. Link discovery covers every hop. |
| `rsqrt` garbage lanes | Unused lanes of the partial tiles hold whatever was in L1 | They are never read: the col broadcast reads only column 0 |

### Structural impossibilities

None beyond `feature_spec.py`'s empty INVALID.

## Refinement 2 realization (OVERLAP: stats-critical chain and tail)

The schedule above is as planned for Phase 0. Refinement 2 changed it as follows (measured on the PERF FOCUS cell,
fastest chip 24.72 → 18.95 µs; see `changelog.md`):

| Change | Replaces | Why |
|--------|----------|-----|
| Own x is loaded before γ (`load_own_block` first) | γ first | the own partial heads the stats chain; γ is needed only after the stats |
| γ read once per chip by the hub and multicast into the stats-scratch tail, then `gamma_go` (regime R6, `GAMMA_HUB_MCAST`) | per-core γ reads | 20 cores per slice read the same 12 pages: ~4.5 µs of small DRAM reads that also slowed the own loads |
| Hub publishes each stats slot as it arrives: linked multicast of the slot + multicast set of `stats_ready` = cumulative slot mask, no barrier between (same NoC, same static VC); readers copy each slot as its bit appears | one multicast of all slots + barrier + flag after the last slot | only the last slot is left on the chain |
| `gamma_block`: xg = x ⊙ γ (row broadcast, fp32) per chunk, before the combine for the own block and for the remote chunks the host's look-ahead table puts ahead (`XG_LATE_STREAM_CHUNKS`: all but the last 8 chunks of each upstream stream); `normalize_block`: out = xg ⊙ rstd_full per chunk | `scale_block` (S = rstd ⊗ γ over the whole block after the stats) + out = x ⊙ S | after the stats only the combine and one multiply per tile remain, so the DRAM stores (≈ 3.8 µs SOL for the output) start right after the combine and spread over the landings instead of bunching at the end |
| `STATS_BULK_WINDOW = 2`: a port that still owes a stats slot keeps at most 2 bulk packets queued in its router channel | stats inserted between batches only | the relayed slot no longer waits behind a full channel of bulk packets |

Every member computes a slice with the same arithmetic (xg through `cb_xg`, then ⊙ rstd_full), so replicas stay
bit-identical. `FUSE_LATE_CHUNKS` (x ⊙ γ ⊙ rstd_full in one DEST-reuse chain) is a parked knob: its DEST-reuse
rounding differs from the `cb_xg` path, so it must apply to every core of a slice at once, and it measured slower.
