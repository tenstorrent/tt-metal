# Operation Design: distributed_rms_norm

## Overview

| Field | Value |
|-------|-------|
| Classification | fused (compute + on-chip collective + Fabric CCL) |
| Goal | One device program per call that computes RMSNorm over a hidden dim sharded across one mesh axis. It reads x from DRAM once and writes the output once. The only thing exchanged over Fabric is per-row fp32 partial sums of squares. |
| Math | `ss[r] = Σ_{g∈group} Σ_{w<W_local} x[g][r,w]²`; `out[p][r,w] = x[p][r,w] · rsqrt(ss[r] / (G·W_local) + ε) · gamma[p][w]` (gamma optional, ≡ 1 when absent) |
| Mode | Hybrid. Compute comes from kernel-lib helpers. Transport is adapted from `fabric_all_gather`: port cores, ready fence, arrival counters, one-hop relay. |
| References | `ttnn/ttnn/operations/examples/fabric_all_gather/` (plan, port placement, ready fence, relay, `_schedule`) · `ttnn/ttnn/operations/examples/fabric_reduce_scatter/` (arrival counters, fence) · `.claude/references/blackhole-fabric.md` · `.claude/references/blackhole-ccl-kernels.md` · `.claude/references/precision_convention.md` · `.claude/references/blocking-model.md` · `.claude/references/l1-footprint-discipline.md` · catalog `ttnn/ttnn/operations/examples/master.md` (entries cited per knob below) · golden `eval/golden_tests/distributed_rms_norm/` |

**Single dispatch.** Every call issues exactly one `ttnn.generic_op(tensors, MeshProgramDescriptor)`. The MeshProgramDescriptor holds one ProgramDescriptor per device; the programs differ only in runtime args.

Some host-side setup happens outside that dispatch. None of it is an op dispatch:
- per-call allocation of the output and the exchange scratch tensor;
- plan caching;
- one-time creation of global semaphores, cached per (mesh, port-core set).

There is **no placement probe program**. `fabric_all_gather.probe_ethernet_cores` would be a second dispatch, so it is not used. See the *Port placement* perf lamp.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| input_tensor | ttnn.Tensor (mesh) | yes | TILE, rank ≥ 2, identical per-device shape / dtype / placement | — | RT (address) |
| cluster_axis | int | yes (kw) | {0, 1}; mesh_shape[cluster_axis] ≥ 2 | — | host (plan) |
| gamma | ttnn.Tensor (mesh) or None | no | ROW_MAJOR, per-device (1,1,1,W_local), dtype ∈ {float32, bfloat16} | None | CT define `HAS_GAMMA` + RT address |
| epsilon | float | no | > 0 | 1e-6 | RT (fp32 bit pattern) |
| topology | `ttnn.operations.ccl.Topology` | no | Linear, Ring | Linear | host (plan) |
| num_links | int or None | no | 1 ≤ L ≤ min over group hops of `len(get_forwarding_link_indices)`; None = that minimum | None | host (plan) |
| compute_kernel_config | ttnn.ComputeConfigDescriptor or None | no | any `math_fidelity` / `math_approx_mode`; `fp32_dest_acc_en` gated (see validate) | `default_compute_kernel_config()` | compute kernel config |
| memory_config | ttnn.MemoryConfig or None | no | Phase 0: interleaved (DRAM or L1) | `ttnn.DRAM_MEMORY_CONFIG` | output allocation |

**`default_compute_kernel_config()`.** This is an exported factory that returns `ComputeConfigDescriptor(math_fidelity=HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)`. The entry point resolves `None` through it, and so does `eval/golden_tests/distributed_rms_norm/axes.py`.

**Imports.** `Topology` is imported via `from ttnn.operations.ccl import Topology` for SUPPORTED and the signature default, because `ttnn.Topology` is unbound at op-package import.

### Registry (op file)

| Item | Phase 0 value |
|------|---------------|
| INPUT_TAGGERS | `alignment`: "tile_aligned" (H%32==0 and W_local%32==0), "w_non_aligned" (W_local%32≠0), "h_non_aligned" (otherwise). `rank`: `len(inputs[0])`. Both read `inputs[0]` = per-device shape. |
| SUPPORTED | dtype {float32, bfloat16}; fp32_dest_acc_en {True}; layout {TILE}; alignment {tile_aligned}; rank {2,3,4}; gamma_mode {gamma, no_gamma}; gamma_dtype {float32, bfloat16, "none"}; memory_layout {INTERLEAVED}; cluster_axis {0,1}; topology {Linear}; num_links {1,2} |
| EXCLUSIONS | `{dtype: float32, fp32_dest_acc_en: False}` (precision convention; outside SUPPORTED in Phase 0 anyway, kept so it survives the fp32_dest_acc_en=False refinement) |
| validate() axis read | dtype = `input_tensor.dtype`; layout; memory_layout = `input_tensor.memory_config().memory_layout`; per-device shape `tuple(input_tensor.shape)`; gamma_mode / gamma_dtype (`gamma.dtype` or "none"); fp32_dest_acc_en from the resolved config; cluster_axis; topology; num_links. **`num_links=None` is "auto" and is always legal.** The support gate checks only an explicit value: None resolves to all usable links, and the design is generic in L. "none" is never refused for gamma_dtype. |
| Caller errors (ValueError, raised before the support gate) | rank < 2; gamma last dim ≠ W_local; cluster_axis ∉ {0,1} or mesh_shape[cluster_axis] == 1; explicit num_links ≤ 0 or > min links discovered on any hop of any group; Ring when the active fabric config does not wrap cluster_axis (TORUS_Y wraps axis 0, TORUS_X wraps axis 1, TORUS_XY wraps both) or `get_forwarding_link_indices(last, first)` is empty |
| Capacity error (RuntimeError) | the per-core working-set predicate `P_resident` (Regimes) fails. No TARGET INPUTS shape reaches it. |

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | per device `(..., M, W_local)`, rank 2–4. Leading dims fold into `batch = prod(shape[:-2])`. |
| Dtype | Phase 0: float32, bfloat16 (TARGET adds bfloat8_b) |
| Layout | TILE |
| Memory | Phase 0: interleaved (DRAM or L1). TARGET adds L1 HEIGHT_SHARDED / WIDTH_SHARDED. |

### gamma (optional)

| Property | Requirement |
|----------|-------------|
| Shape | per device (1, 1, 1, W_local): device p's slice of the full-W scale |
| Dtype | float32 or bfloat16; may differ from x |
| Layout / Memory | ROW_MAJOR, DRAM interleaved: one page = the whole stick |

### Output

| Property | Value |
|----------|-------|
| Shape | equal to the input's per-device shape |
| Dtype | the input's dtype |
| Layout | TILE |
| Memory | `memory_config` or DRAM interleaved |

### Exchange scratch (internal, allocated per call)

| Property | Value |
|----------|-------|
| What | the per-port L1 landing zone for stats records (see Dataflow) |
| Allocation | `ttnn.allocate_tensor_on_device`, uint32 ROW_MAJOR, L1 HEIGHT_SHARDED over the port-core set (the same logical cores on every device), one shard of `G · slot_words` words per port core. It is passed in the generic_op tensor list, so the mesh allocates it at the **same address on every device**. |
| Size | `slot_bytes = 128 · max_l recs_link[l]` per source device q; `G · slot_bytes` per port core |

## Blocking Model

Semantics are defined in `.claude/references/blocking-model.md`. Tile geometry is always per image and alignment-aware:
- `tensor_row_tiles = batch · ceil(H/32)`
- `tensor_w_tiles = ceil(W_local/32)`

Global tile-row index: `R = b · ceil(H/32) + th`. Tile id: `R · tensor_w_tiles + tw`.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `rows` (tile rows, batch folded) | independent: each row's statistic and output depend only on that row | `block_rows` | `core_rows` (the core's whole row assignment, one block) | host `_plan_blocks()`; RT arg to reader/compute/writer, CT arg for CB sizing | contiguous balanced ranges over `S_r` row groups (first `tensor_row_tiles mod S_r` groups get `ceil`) | knob-turn (`block_rows < core_rows` = multi-round regime, deferred) |
| `w_local` (tiles of the device's hidden slice) | dependent: the sum of squares spans it, a cross-core combine is needed when it is split. It is independent for the normalize. | `core_w_tiles` = `ceil(tensor_w_tiles / S_w)` (the block's W extent = the core's whole chunk) | whole chunk | host `_choose_split()` (`S_w`), floor knob `min_core_w_tiles` = 4 | `S_w` contiguous W chunks per row group; the combine is the stats-record exchange (built) | `S_w` is a knob-turn: the record combine already exists for any `S_w` ≥ 1 |
| `group` (devices along cluster_axis, G) | dependent: ss sums over all G devices; the caller pre-splits it physically (mesh) | `G` (not a blocking choice: mesh_shape[cluster_axis]) | given by the mesh | `plan()` | one device per group position p; fixed by the caller | — (fixed by the mesh) |
| `gamma` operand along `rows` | reuse-shared by construction of the row split: gamma does not vary with `rows`, so every row group re-reads the same W segment | `core_w_tiles` (gamma block = the core's W chunk, row 0 valid) | whole chunk, read once per call per core | same as `w_local` | every worker reads its own segment from DRAM (stepping stone) | broadcast (multicast down the S_r workers sharing a W chunk) is a deferred regime |
| `sources` (stats records per tile row: G·S_w) | dependent: rstd needs their sum. Stage-produced data. | `n_src_tiles = ceil(G·S_w / 32)` (32 sources per combine tile, one per tile row) | derived | host `_plan_blocks()` | every worker of a row group computes the same combine for its own rows. Computation is spread (S_r·S_w workers in parallel), never left on a designated core. | — |
| `records` on the wire (stage data: one 128 B record per (tile row, w-chunk)) | independent per record; the port only moves them | `recs_link[l]` | all of link l's records in one entry per source device | host `_plan_blocks()` | link l carries row groups with `r_g mod L == l` | `L` is a knob-turn (num_links) |

**Step-1 table (one row per axis, no blanks): the three knobs.**

| Axis | Character | Block factor (Phase 0) | Core assignment | Buffer depth | Why |
|---|---|---|---|---|---|
| rows | independent | `block_rows = core_rows` (whole assignment) | balanced contiguous ranges over S_r row groups, spread across the full worker grid | cb_x_resident: 1 block (residency, not pipelining); cb_out: 2 | Residency is what lets x cross DRAM once. One block means one exchange round per call. |
| w_local | dependent (sum) / independent (normalize) | `core_w_tiles` (whole chunk) | S_w chunks per row group, chosen to fill the grid (`_choose_split`) | reader pushes one tile-row chunk (`core_w_tiles` tiles) per barrier into the resident CB | M is too small to fill the grid at every decode and most prefill shapes (Ht = 1 / 20 / 64 vs ~117 workers). Splitting W is the available parallelism, and its combine is the same record exchange the cross-device step needs anyway. |
| group (G) | dependent | fixed by the caller | one per device | — | pre-split by the mesh; combined by the record all-gather |
| gamma along rows | reuse-shared | `core_w_tiles` | each worker reads its segment | cb_gamma_tiles: 1 (persistent) | It is ≤ 1/(32·core_rows) of the core's x bytes, so the stepping stone is built and the broadcast is deferred. |
| sources (stage data) | dependent | `n_src_tiles × block_rows` combine tiles | **every worker computes the combine for its own rows.** The port core only moves bytes. It is a communication role, not a work assignment. | cb_stats_combine: 1 block | Spreading the finalize (sum, ×1/W, +ε, rsqrt) over all workers avoids the "root computes while the group waits" stage. Workers in one row group compute an identical result redundantly, in parallel. |
| records (stage data) | independent | one entry per (source device, link) | link l ↔ row groups `r_g mod L == l`; one port per (link, direction) | exchange scratch: all G slots resident per call (no credits) | One slot per source device for the whole call means no flow-control round trips. The ready fence protects reuse across calls. |

### Split selection: `_choose_split(tensor_row_tiles, tensor_w_tiles, N)` (host, deterministic)

`N = grid_x · (grid_y − 1)` workers (logical row 0 is reserved for port cores; see Work Distribution).

```
best = None
for S_w in 1 .. min(N, ceil(tensor_w_tiles / min_core_w_tiles)):
    S_r = min(tensor_row_tiles, N // S_w)
    cost = ceil(tensor_row_tiles / S_r) * ceil(tensor_w_tiles / S_w)   # max tiles per worker
    keep (cost, S_w) lexicographically smallest                         # tie -> fewer W splits
```

Examples on a 13×10 Blackhole grid (N = 117):

| Shape | Ht | Wt | S_w | S_r | Workers | Tiles per worker | Note |
|-------|----|----|-----|-----|---------|------------------|------|
| perf focus 640×1792 | 20 | 56 | 5 | 20 | 100 | 12 | |
| decode 32×1792 | 1 | 56 | 14 | 1 | 14 | 4 | |
| 2048×1280 | 64 | 40 | 5 | 23 | 115 | 24 | |
| 32×4096 | 1 | 128 | 32 | 1 | 32 | 4 | G·S_w = 64 → n_src_tiles = 2 |
| (4,1,512,512) | 64 | 16 | 1 | 64 | 64 | 16 | |
| (1,1,32,64) | 1 | 2 | 1 | 1 | 1 | 2 | |

`min_core_w_tiles = 4` comes from catalog `double_buffer` (≥ 4 tiles per read barrier is the bf16 sweet spot) and `width_split` (W split gains up to 64 cores at H = 32). Smaller chunks only add records.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| cb_x_resident | `x_blocks_in_flight` | 1 block (`block_rows·core_w_tiles` pages) | Residency across the exchange. It is not overlap: the reader fills it in `core_w_tiles`-tile batches, and phase A consumes cumulatively while the reads land. |
| cb_out | `out_depth` | 2 × `out_batch_tiles` (`out_batch_tiles = core_w_tiles`) | normalize of tile-row j+1 overlaps the DRAM write of tile-row j (catalog `double_buffer`) |
| cb_xg, cb_sq_acc, cb_partial, cb_rstd, cb_stats_combine | — | 1 block | intermediates between sequential helpers on the same TRISCs; they cannot pipeline |
| cb_gamma_tiles, cb_scaler | — | 1 | persistent for the call |
| exchange scratch | `G` slots | all G | landing zone needs no credits (`blackhole-ccl-kernels.md` §7) |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Combine tile holds one source per tile row (32 rows) | sources per combine tile ≤ 32 | `n_src_tiles = ceil(G·S_w/32)`; source s → tile `s/32`, row `s mod 32`; unused rows zero-filled every round | sources overwrite each other or garbage rows enter the sum (wrong ss) |
| Router max payload (`ttnn.get_tt_fabric_max_payload_size_bytes()`, 14 336 B on BH) | records per packet | `recs_per_packet = payload // 128`; an entry is split into `ceil(recs_link/recs_per_packet)` packets; fused inc on the entry's last packet only | oversized packet: router assert or hang |
| Packet header pool (12 headers per DM RISC) | packets in flight per port | header ring of `min(8, 12)`; flush before reuse | `PACKET HEADER POOL EXHAUSTION` hang |
| Port cores live in logical row 0 | `2·L ≤ grid_x` | validate L against `grid_x // 2` (L ≤ 4 on BH 13-wide) | ports overlap the worker rectangle |
| L1 working set per worker | `block_rows · core_w_tiles` | predicate `P_resident` (l1_ledger.md). Phase 0 refuses (RuntimeError) when it fails. `block_rows < core_rows` is the deferred multi-round regime. | CB allocation failure |
| Exchange scratch per port | `G · recs_link · 128 B` | predicate `P_scratch`: ≤ 1 MiB. The largest INPUTS case (4096×1024 rows, G = 8, L = 1) is 512 KiB. | L1 allocation failure |
| DEST capacity under fp32 accumulation (4 tiles, half-sync) | none on blocks; subblocking is the helpers' (`DEST_AUTO_LIMIT`, `dest_helpers.hpp:103`) | — | — |
| fp32 FPU sources are tf32 (srcA/srcB 19-bit) | precision of the FPU squares and multiplies for float32 x | the stats sums (phase A reduce, combine reduce) use `ReduceFp32Mode::Accurate` (SFPU, full fp32). Squares and normalize multiplies stay on the FPU. | float32 rel-RMS rises toward the 0.005 gate; see Risks |

### Regimes

The memory boundary is DRAM. The minimum is: x crosses once, out crosses once, gamma crosses once per device.

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **R1 interleaved, resident, flat record all-gather** | **built** | `memory_layout == INTERLEAVED ∧ P_resident ∧ P_scratch` | `core_rows × core_w_tiles` (whole worker assignment) | x 1× read, out 1× write: the minimum. gamma is read S_r× (each worker reads its W segment). That is above the minimum, and it is ≤ 1/(32·core_rows) of x bytes per worker. Stats: G·Ht·S_w·128 B per device through port L1 plus Fabric. | One block means one exchange round per call. This amortizes: the ready fence (1 header packet per port per call); the local-arrival wait; the stats-ready mcast; the combine reduce and post-op init (once per call); the gamma-prescale and normalize chain inits (once per block); `compute_kernel_hw_startup` and the scaler (once per kernel). |
| R2 multi-round (`block_rows < core_rows`) | deferred | `¬P_resident` with `block_rows = core_rows`, or the overlap lamp says it wins | `block_rows × core_w_tiles` | same DRAM bytes as R1 (never a re-read); stats bytes unchanged | Rounds pay the fence once but the stats-ready mcast and combine once per round. **Positive reason:** every TARGET INPUTS shape fits one round. The largest resident set is ~0.5 MiB (2048×2048 fp32, 39 tiles/worker) against the 1 MiB budget. **Kept reachable:** record addresses are a function of (q, R, w) for the call, not of the round. Workers already loop `num_blocks_this_core` and wait stats-ready ≥ `2L·(round+1)`. The port loops rounds with per-round local-arrival thresholds. |
| R3 streaming re-read (normalize re-reads x from DRAM) | rejected | — | — | x crosses DRAM 2× | Superseded everywhere by R1 + R2, which never re-read. A dead end. |
| R4 root-per-row-group combine (W-chunk 0 sums S_w partials, then mcasts rstd back) | rejected | — | — | the same DRAM bytes; Fabric bytes ÷S_w | It adds a serial gather stage and an rstd mcast stage, and it idles S_w−1 workers during the root's finalize. Superseded by R1's flat records, where each worker sums G·S_w sources itself. |
| R5 device-local pre-combine before Fabric (sum the S_w records per row on chip, ship one record per row) | deferred | G·Ht·S_w·128 B per call exceeds about one packet per link | — | Fabric bytes ÷S_w. It adds an on-chip gather plus combine stage before the send. | **Positive reason:** on every measured case (QB G = 2) the per-link entry is ≤ 3 packets (2048×1280: 320 records = 40 KB per device, 20 KB per link at L = 2). The extra serial stage costs more than the bytes saved. It matters only for Galaxy G = 8 with M ≥ 2048 (≤ 512 KiB). **Reachable:** a pre-combine worker is just another record producer writing slot p. |
| R6 single stats core per device finalizes all rows | rejected | — | — | — | It serializes `tensor_row_tiles` finalize tiles on one core (128 at M = 4096) while every worker waits. Superseded by R1. |
| R7 composite pre / all_gather / post | rejected | — | — | stats round-trip DRAM; three program floors | Forbidden (one dispatch). This is today's 87–95 µs baseline. |
| R8 gamma broadcast (one reader per W chunk, multicast down the S_r workers sharing it) | deferred | `S_r > 1 ∧ HAS_GAMMA` | — | gamma DRAM bytes ÷S_r | **Positive reason:** gamma is ≤ 3 % of a worker's DRAM bytes at the perf-focus shape (12 gamma rows' worth vs 12 tiles). The cheapest-traffic gain is below noise. **Reachable:** cb_gamma_tiles keeps the same contents; only its producer changes (`mcast_pipe`). |
| R9 HEIGHT_SHARDED input | deferred (TARGET `memory_layout`) | `memory_layout == HEIGHT_SHARDED` | `block = shard` (`core_rows = shard_h/32`, `S_w = 1`) | x 0× DRAM: cb_x_resident is backed on the input shard via `ttnn.cb_descriptor_from_sharded_tensor` (zero-copy, no NoC re-read). out: cb_out is backed on the output shard (pack into destination, no writer DRAM traffic). | **Knob-turn:** S_w = 1, the core-assignment is pinned by the shard, and the record exchange is unchanged. If ports collide with shard cores, the record send moves to the worker's NCRISC and the port BRISC kernel runs on a shard core whose writer is idle (no DRAM output). |
| R10 WIDTH_SHARDED input | deferred (TARGET `memory_layout`) | `memory_layout == WIDTH_SHARDED` | `block = shard` (`S_r = 1`, `S_w` = shard cores, `core_w_tiles = shard_w/32`, `core_rows = tensor_row_tiles`) | same zero-copy CBs as R9 | **Knob-turn:** this is exactly R1's W split with S_w pinned by the caller. The cross-core combine (records) already exists. It is not a scheme-change, despite cutting a dependent axis. |

**Regime-selection function (Phase 0).** There is a single built regime, R1. `_choose_split` picks `S_w` per shape and grid, so pinned tests must cover each of:
- `S_w = 1` (`(4,1,512,512)` / `(1,1,32,64)`);
- `S_w > 1`, one combine tile (decode 32×1792);
- `n_src_tiles > 1` (32×4096, G·S_w = 64);
- multi-row groups (2048×1280).

The acceptance test pins all four.

### Traffic ranking

Bytes are per device and per call. B = x bytes.

| Candidate split | DRAM | On-chip NoC | Fabric | Fills grid? | Verdict |
|---|---|---|---|---|---|
| rows only (S_w = 1), resident | x 1×B read, out 1×B write | records → 2 ports; pulls G·128 B per tile row per worker | G−1 entries of Ht·128 B (per link share) | no: 1 core at decode, 20 at the perf focus | dominated by the next row on occupancy at equal bytes |
| **rows × W chunks, resident, flat records (chosen)** | **x 1×, out 1×** (minimum) | records Ht·S_w·128 B to the ports; pulls G·S_w·128 B per tile row per worker | (G−1)·Ht·S_w·128 B / L per link direction (perf focus: 12.8 KB total) | yes (100–117 workers) | **chosen: cheapest bytes, full grid** |
| rows × W, root combine (R4) | same | + S_w·4 KB per tile row into each root, + rstd mcast | ÷S_w | yes | rejected: same DRAM, two extra serial stages |
| any split without residency (R3) | x 2× | — | — | — | rejected |
| split W and pre-combine on chip (R5) | same | + gather stage | ÷S_w | yes | deferred (see R5) |

The traffic is the minimum at the DRAM boundary. The chosen split is also the cheapest-traffic split.

### Block schedule

```cpp
// per worker (r_g, w): one round in Phase 0 (num_blocks_this_core == 1)
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    load_block(block_idx);           // reader: x block -> cb_x_resident (tile-row batches); gamma once per call
    sumsq_block(block_idx);          // compute: per-row Σ x² -> cb_partial (row-0 record layout)
    publish_records_block(block_idx);// writer: cb_partial rows -> own-slot records at the link's ports (+ local-arrival inc)
    gamma_prescale_block(block_idx); // compute (HAS_GAMMA): x·gamma -> cb_xg   [runs in the exchange stall]
    exchange_round(block_idx);       // ports: ready fence, own entry + relays over Fabric, stats-ready mcast
    gather_sources_block(block_idx); // reader: wait stats-ready, pull G·S_w records per tile row -> cb_stats_combine
    combine_finalize_block(block_idx);// compute: Σ sources -> transpose -> ×1/W -> +ε -> rsqrt -> cb_rstd
    normalize_block(block_idx);      // compute: (xg | x) · rstd(col bcast) -> cb_out
    store_block(block_idx);          // writer: cb_out -> DRAM, tile-row batches
}
```

| Block operation | Block shape | Resident across it | Intended frequency of fixed costs |
|---|---|---|---|
| load_block | `block_rows × core_w_tiles` x tiles; gamma `core_w_tiles` | x stays in cb_x_resident until normalize / prescale pops it | one barrier per `core_w_tiles` tiles; gamma once per call; scaler once per kernel |
| sumsq_block | `block_rows × core_w_tiles` → `block_rows` | cb_x_resident is waited, not popped | one chain init per block; one reduce init per block; x² accumulates in DEST per tile row (one pack per row) |
| publish_records_block | `block_rows` records × (ports of link: 1–2) | — | 2 NoC writes of 64 B per record per port; one barrier, then one sem inc per port per block |
| gamma_prescale_block | `block_rows × core_w_tiles` | gamma persistent; pops cb_x_resident at block end | one chain init per block |
| exchange_round (port) | one entry per source device per (link, direction) | the whole call's records sit in scratch | ready fence once per call; ≤ `ceil(recs_link/recs_per_packet)` packets per entry with one fused inc per entry; one stats-ready mcast per port per round |
| gather_sources_block | `n_src_tiles × block_rows` combine tiles | — | zero-fill once per round; 2 NoC reads (64 B) per (source, tile row); one barrier |
| combine_finalize_block | `n_src_tiles × block_rows` → `block_rows` | — | one reduce plus post-op init per block |
| normalize_block | `block_rows × core_w_tiles` | rstd window (`block_rows` tiles) held with Upfront/AtEnd | one chain init per block |
| store_block | `block_rows × core_w_tiles` | — | one barrier per `out_batch_tiles` |

**Stall-shadow (design-time reorder).** The wait is `gather_sources_block` blocking on the stats-ready semaphore. That covers the on-chip publish, the Fabric hop(s) and the relays.

Work that does not depend on it is `x ⊙ gamma`. rstd is a per-row scalar and gamma is a per-column vector, so `x·rstd·gamma = (x·gamma)·rstd`: multiplication commutes and associates. That algebraic property is what licenses lifting `gamma_prescale_block` ahead of the wait.

- **Precision gate.** cb_xg pages are Float32 (DEST width), so the reorder changes only the rounding order inside fp32. That is far below the bf16 / fp32 gates (rel-RMS 0.01 / 0.005). It must still be checked against the golden fp32 baseline before any perf pass removes it.
- **Without gamma** nothing is independent of the wait. All x is already resident, and normalize needs rstd. The remaining stall is the overlap lamp's subject (R2).

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Overlap (whole-assignment block) | One round serializes: all DRAM reads → exchange (fence, Fabric, relays, mcast) → normalize and writes. DRAM idles for the whole exchange, likely several µs against an 18 µs roofline at the perf focus. | R2 with 2 rounds. At the perf focus that means `_choose_split` with `S_r = Ht/2`, `S_w` doubled, `block_rows = 1`: the same 12 tiles per worker, 2 rounds, round 0's exchange under round 1's reads. |
| Grid synchronization (max S_w) | At decode the cost of S_w = 14 workers is G·S_w = 28 record pulls per worker plus 14 local-arrival incs at each port. That may exceed the 4-tile DRAM saving. | `min_core_w_tiles` 8 or 16 (S_w = 7 / 4) at 32×1792 and 32×4096 |
| Gamma prescale in the stall | It costs a full fp32 block buffer and an extra pack plus unpack pass. If the exchange is shorter than the prescale, it lands on the critical path. | Fused normalize after the wait: `BinaryFpu<Mul, x, rstd Col>` + `DestReuseBinary<Mul, gamma Row>` → cb_out, no cb_xg |
| Port placement | Ports sit at logical row 0, columns 0..2L−1, not under each link's Ethernet core (blackhole-fabric rule 6). Payloads are tiny, so latency, not bandwidth, is at stake. | Host-derived Ethernet column (SoC descriptor + `get_forwarding_link_indices`, no probe dispatch) → port in that column |
| Relay vs direct multi-hop routing (Linear G > 2, Ring) | Store-and-forward through port cores adds a worker hop per device. 2D routing by destination could forward in the router. | Direct `fabric_set_unicast_route` to each destination device (Galaxy only; G = 2 on QB is one hop either way) |
| Record pull mechanism | 2 × 64 B NoC reads per (source, tile row): 56 reads per worker at decode | One contiguous read of `S_w·128 B` per (q, tile row) into a staging area, then RISC copies into face rows |
| Stats-ready granularity | Every worker waits for all 2L ports, even ports that carry none of its rows | Per-link mcast rectangles: wait only for the link's 2 ports |
| out batch | `out_batch_tiles = core_w_tiles` (4–13). The catalog flat region is 4–64. | 2× `core_w_tiles` (cb_out depth 2 already) |

Catalog entries that informed the knobs:
- `double_buffer` (min batch ≥ 4, depth 2)
- `width_split` (split W when H = 32)
- `compute_block_size` (whole block per helper call)
- `row_reduce_accumulate` / `reduce_block` (DEST-pair accumulation over W, then one reduce)
- `tensix_all_reduce` (flat vs root at 1 tile per core)
- `eltwise_l1_vs_dest_accumulate` (accumulate in DEST)
- `compute_fusion` (rsqrt fused as a reduce post-op; multiplies on the FPU via broadcast)
- `noc_placement` (reads on NoC0, writes and sends on NoC1)

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| x DRAM → worker L1 | x dtype tiles | reader (NCRISC, NoC0), TensorAccessor; tile ids `R·tensor_w_tiles + tw` for its rows × chunk; batch = `core_w_tiles` tiles per barrier; push per tile row into cb_x_resident | x crosses DRAM once |
| gamma DRAM → worker L1 | gamma dtype; tile with **row 0 valid** | reader, once per call. Per gamma tile t: one 64 B (bf16) / 128 B (fp32) DRAM read of elements `[32·(w0+t), +32)` (64 B-aligned) into face 0. Elements 0–15 land in face 0 row 0; elements 16–31 land in face 0 row 1 and are moved by a local L1 copy to face 1 row 0. | Avoids a tilize phase. Rows ≠ 0 are never read under `BroadcastDim::Row`. A direct DRAM read into face 1 would violate the BH 64 B DRAM source alignment. |
| sumsq → record | fp32 tile, row 0 = per-row Σx² (rows 0–15 in face 0 row 0, bytes [0,64); rows 16–31 in face 1 row 0, bytes [1024,1088)) | compute: x² accumulated in DEST per tile row → Accurate REDUCE_ROW → `transpose_dest` post-op | A **record** is 128 B: half 0 = rows 0–15, half 1 = rows 16–31, 32 fp32 values |
| worker → port (own slot p) | records | writer (BRISC, NoC1): 2 NoC writes (64 B) per record to each **sending** port of its link, at `scratch + p·slot_bytes + (link_rec_base + j·S_w + w)·128`; `noc_async_write_barrier`; `noc_semaphore_inc` of each port's local-arrival semaphore (program-local) | link_rec_base and S_w are RT args |
| port → peer port (Fabric) | records | port BRISC (NoC1), adapted from `fabric_all_gather` `_SENDER_SOURCE`: open the connection; send ready (header-only atomic inc) to the peer that writes into this device; wait own ready, then decrement; wait local arrivals == `workers_in_link`; send own entry (slot p) into the **same-direction port** of the next device at the same scratch offset; fused inc of that port's arrival counter on the entry's last packet; then relay each inbound entry once the arrival counter ≥ k | Linear: toward p+1 send p, p−1, …, 0; toward p−1 send p, p+1, …, G−1 (`fabric_all_gather._schedule(p, G, ring=False)`). Ring (G ≥ 3): toward p+1 G//2 entries, toward p−1 the rest (`_schedule(..., ring=True, balance=False)`). Inbound +direction traffic lands at the **fwd** port (which relays it); −direction traffic lands at the **bwd** port. |
| port → workers | stats-ready | each of the 2L ports (sending, receive-only or idle) waits arrival ≥ `expect_in` (and its own local arrivals if it sends), then `noc_semaphore_inc_multicast` +1 to the worker rectangle (rows 1..grid_y−1). Then it re-arms: arrival −= expect_in, ready −= 1 consumed, `noc_async_atomic_barrier`. | Workers wait stats-ready ≥ 2L (program-local semaphore, reset each launch). On NoC1 the multicast rectangle is given end→start. |
| port L1 → worker combine | records → fp32 combine tiles, source `s = q·S_w + w` in row `s mod 32` of tile `s/32` | reader: after stats-ready, reserve `n_src_tiles·block_rows`, zero-fill, then for each tile row j and each source: half 0 → face `2·(r/16)` row `r mod 16`, half 1 → face `2·(r/16)+1` row `r mod 16` (`r = s mod 32`), via NoC reads from the port that holds q; barrier; push | Source-port table (G entries of NoC x/y) is an RT arg. Linear: q < p → fwd port, q > p → bwd port, q = p → the fwd port if p < G−1 else the bwd port. Ring: by `_schedule` membership. |
| combine → rstd | fp32 tile, col 0 = rstd | compute: Accurate REDUCE_COL over `n_src_tiles` rows (Σ sources → row 0) → post-op `transpose_dest` → `mul_unary(1/(G·W_local))` → `add_unary(ε)` → `rsqrt` | Every worker of row group r_g, on every device, sums the same sources in the same order, so results are bit-identical across workers and devices and deterministic run to run |
| normalize → DRAM | x dtype tiles | compute packs to cb_out; writer (NoC1) TensorAccessor writes `core_w_tiles` per barrier | out crosses DRAM once |

**Ready fence and counters.**
- Global semaphores, created once per (mesh id, port-core set), cached, synchronized once at creation (`ttnn._ttnn.ccl_host.make_ccl_semaphore` style), and kept alive by the module cache:
  - `arrival` — fused-inc target of the upstream same-direction port;
  - `ready` — fence.
- Program-local semaphores, declared **before** `setup_fabric_connection` is called:
  - `local_arrivals` — on ports;
  - `stats_ready` — on workers.
- Every call leaves every global counter at 0.

**Physical shards (TARGET).** R9 and R10 are the same logical scheme with the extents pinned by the shard. cb_x_resident and cb_out become `cb_descriptor_from_sharded_tensor` CBs over the input and output shards: zero-copy, no TensorAccessor re-read. HEIGHT and WIDTH are both **knob-turns** for this op. HEIGHT keeps rows self-contained. WIDTH cuts the dependent W axis, whose combine (records) is already built.

**Structural impossibilities.** No additions to `feature_spec.py` INVALID.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | a block = `block_rows × core_w_tiles` tiles of one row group's W chunk |
| Grid | `grid = mesh_device.compute_with_storage_grid_size()` (a runtime parameter, never a literal). **Ports:** logical row 0, columns `0..2L−1`; ordered (link l, dir) → column `2l + (0 fwd, 1 bwd)`. **Workers:** rectangle `(0,1)..(grid_x−1, grid_y−1)`, `N = grid_x·(grid_y−1)`, row-major worker index `k = r_g·S_w + w`. Columns `2L..grid_x−1` of row 0 stay idle (≤ 11 cores). |
| Per-core work | worker (r_g, w): rows `[row_start(r_g), +core_rows(r_g))`, W tiles `[w·cw, min((w+1)·cw, tensor_w_tiles))` with `cw = ceil(tensor_w_tiles/S_w)` |
| Remainder | rows are balanced (`split_work_to_cores` semantics): first `tensor_row_tiles mod S_r` groups take `ceil`. The last W chunk is shorter (`tensor_w_tiles − (S_w−1)·cw`). The ragged extent goes to NoC transfer counts and chain shapes. CB capacities stay nominal. All tile geometry uses `ceil` per image. |
| Links | row group r_g → link `r_g mod L`. `recs_link[l] = S_w · Σ_{r_g ≡ l} core_rows(r_g)`; `link_rec_base(r_g)` = prefix sum within the link. |
| Plan cache | key: (mesh id, mesh shape, cluster_axis, topology, L, fabric config, grid). Value: groups, schedules, port cores, links per hop. Shape-dependent quantities are recomputed per call (cheap host math). |

## Circular Buffers

Worker cores (all on the worker rectangle):

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| cb_x_resident | 0 | x tile | `block_rows · core_w_tiles` | spans rows and w_local (resident block) | x dtype | reader | compute | load → prescale (gamma) / normalize (no gamma) |
| cb_gamma_tiles | 1 | gamma tile | `core_w_tiles` (only `HAS_GAMMA`) | spans w_local, streams over rows | gamma dtype | reader | compute | whole call |
| cb_scaler | 2 | bf16 tile | 1 | constant (SUM scaler 1.0) | Float16_b | reader | compute | whole call |
| cb_sq_acc | 3 | fp32 tile | `block_rows` | spans rows; w_local collapsed in DEST | Float32 | compute | compute | sumsq |
| cb_partial | 4 | fp32 tile | `block_rows` | spans rows | Float32 | compute | writer | sumsq → publish |
| cb_xg | 5 | fp32 tile | `block_rows · core_w_tiles` (only `HAS_GAMMA`) | spans rows and w_local | Float32 | compute | compute | prescale → normalize |
| cb_stats_combine | 6 | fp32 tile | `n_src_tiles · block_rows` | spans rows and sources | Float32 | reader | compute | gather → finalize |
| cb_rstd | 7 | fp32 tile | `block_rows` | spans rows | Float32 | compute | compute | finalize → normalize |
| cb_output_tiles | 8 | x tile | `2 · out_batch_tiles` | streams rows and w_local | x dtype | compute | writer | normalize → store |

Port cores: **no CBs**. They send from, and receive into, the exchange scratch tensor shard. Full ledger: `l1_ledger.md`.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | load_block | `block_rows × core_w_tiles` | dataflow (TensorAccessor) + `calculate_and_prepare_reduce_scaler` | DRAM | cb_x_resident (block), cb_gamma_tiles (`core_w_tiles`, once), cb_scaler (1, once) | gamma and scaler persist for the call |
| 2 | sumsq_block | grid(`block_rows`, `core_w_tiles`) → `block_rows` | helper: `eltwise_chain` (square, DEST-accumulated per row) then `reduce` | cb_x_resident (Cumulative wait, **no pop**) → cb_sq_acc (`block_rows`, BulkWaitBulkPop) | cb_sq_acc → cb_partial (`block_rows`) | cb_x_resident is still resident |
| 3 | publish_records_block | `block_rows` records | dataflow (writer) | cb_partial (`block_rows`, popped) | port scratch slot p | — |
| 4 | gamma_prescale_block | grid(`block_rows`, `core_w_tiles`) | helper: `eltwise_chain` BinaryFpu Mul, Row bcast | cb_x_resident (Upfront, pop AtEnd), cb_gamma_tiles (Upfront, no pop, Row mapping) | cb_xg (block) | cb_x_resident is freed |
| 5 | exchange_round | records of the link | port BRISC kernel (Fabric) | scratch | scratch (peer devices) + stats-ready | counters re-armed |
| 6 | gather_sources_block | `n_src_tiles × block_rows` | dataflow (reader) | port scratch | cb_stats_combine (block) | — |
| 7 | combine_finalize_block | `ReduceInputBlockShape::of(n_src_tiles, block_rows)` | helper: `reduce` REDUCE_COL Accurate + post-op | cb_stats_combine (BulkWaitBulkPop) | cb_rstd (`block_rows`) | — |
| 8 | normalize_block | grid(`block_rows`, `core_w_tiles`) | helper: `eltwise_chain` BinaryFpu Mul, Col bcast | cb_xg (gamma) or cb_x_resident (no gamma) (PerTile/PerTile); cb_rstd (Upfront, pop AtEnd, Col mapping) | cb_output_tiles (PerTile push) | all intermediates are empty |
| 9 | store_block | `core_w_tiles` per batch | dataflow (writer) | cb_output_tiles | DRAM | — |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| all compute | helper | `compute_kernel_hw_startup(cb_x_resident, cb_scaler, cb_output_tiles)` | called once at kernel start (required by `reduce_helpers_compute.hpp:500-503`) | — | — | — | — |
| load_block (scaler) | helper | `dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<cb_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>()` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp:97-99` | pool-type-aware; full tile. The full-tile fill is the same row-0 pattern for REDUCE_COL (`reduce_helpers_dataflow.inl:191-211`; only partial fills differ), so one scaler serves both reduces. | — | cb_scaler | — |
| sumsq_block (square + accumulate) | helper | `eltwise_chain(IterationShape::grid(block_rows, core_w_tiles), BinaryFpu<BinaryFpuOp::Mul, input(cb_x_resident, WaitPolicy::Cumulative, PopPolicy::None, InputTileMapping::Block), input(input(cb_x_resident, WaitPolicy::Cumulative, PopPolicy::None, InputTileMapping::Block), BroadcastDim::None), Dst::D0, DestAccumulation::PerRow>{}, PackTile<output(cb_sq_acc, ReservePolicy::PerOuter, PushPolicy::PerOuter, DataFormatReconfig::Enabled, TileAddressing::Direct, DestAccumulation::PerRow)>{})` | `eltwise/api/chain.hpp:533` (entry), `:485-494` (BinaryFpu), `:283-287` (DestAccumulation), `:503-504` (PackTile); CbA == CbB allowed with identical policies (`eltwise/core/chain.inl:1102-1112`); PerRow pack lifecycle (`chain.inl:971-976`); usage `ttnn/cpp/ttnn/kernel_lib/tests/eltwise/chain/accumulation.cpp:32-44` | — | cb_x_resident | cb_sq_acc | `block_rows`, `core_w_tiles` (IterationShape) |
| sumsq_block (row reduce) | helper | `reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, cb_sq_acc, cb_scaler, cb_partial, ReduceInputPolicy::BulkWaitBulkPop, ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT, ReduceFp32Mode::Accurate>(ReduceInputBlockShape::of(block_rows, 1), {}, NoAccumulation{}, post_op)`, where `post_op(dst) = { transpose_dest_init<...>(); transpose_dest(dst); }` | `reduce_helpers_compute.hpp:616-637`; `ReduceInputBlockShape` `:290-301`; Accurate = SFPU fp32 (`reduce_helpers_common.hpp:11-18`); post-op call site for ROW `reduce_helpers_compute.inl:1133`; `transpose_dest` `tt_metal/hw/inc/api/compute/transpose_dest.h:29,52` | — | cb_sq_acc | cb_partial | `block_rows` |
| gamma_prescale_block | helper | `eltwise_chain(grid(block_rows, core_w_tiles), BinaryFpu<Mul, input(cb_x_resident, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Block), input(input(cb_gamma_tiles, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Row), BroadcastDim::Row)>{}, PackTile<output(cb_xg)>{})` | `chain.hpp:356-377` (input specs), `:305` (BroadcastDim), Row mapping legal pairs `chain.inl:29-75` | — | cb_x_resident, cb_gamma_tiles | cb_xg | `block_rows`, `core_w_tiles` |
| combine_finalize_block | helper (+ raw post-op body) | `reduce<SUM, REDUCE_COL, cb_stats_combine, cb_scaler, cb_rstd, BulkWaitBulkPop, INPUT_AND_OUTPUT, ReduceFp32Mode::Accurate>(ReduceInputBlockShape::of(n_src_tiles, block_rows), {}, NoAccumulation{}, finalize)`, where `finalize(dst) = transpose_dest(dst); mul_unary_tile(dst, bits(1/(G·W_local))); add_unary_tile(dst, bits(ε)); rsqrt_tile(dst);` (each preceded by its `*_init`) | `reduce_helpers_compute.hpp:616-637`; REDUCE_COL post-op semantics `:514-516`, call site `.inl:1254`; `transpose_dest.h:29,52`; SFPU scalar ops wrapped by `eltwise/unary/scalar.hpp:26,32` (`AddUnary` / `MulUnary` → `add_unary_tile` / `mul_unary_tile`); `Rsqrt` `eltwise/unary/math.hpp:38` | — | cb_stats_combine | cb_rstd | `n_src_tiles`, `block_rows` |
| normalize_block | helper | `eltwise_chain(grid(block_rows, core_w_tiles), BinaryFpu<Mul, input(cb_src, WaitPolicy::PerTile, PopPolicy::PerTile, InputTileMapping::Block), input(input(cb_rstd, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Col), BroadcastDim::Col)>{}, PackTile<output(cb_output_tiles)>{})`, where `cb_src = HAS_GAMMA ? cb_xg : cb_x_resident` (compile-time) | `chain.hpp:356-391`, Col mapping `chain.hpp:232-247` / legal pairs `chain.inl:29-75` | — | cb_xg / cb_x_resident, cb_rstd | cb_output_tiles | `block_rows`, `core_w_tiles` |
| load_block / store_block / publish / gather | raw_api (dataflow) | `TensorAccessor`, `noc_async_read/write`, `noc_async_*_barrier`, `noc_semaphore_inc`, `noc_semaphore_wait_min`, `noc_semaphore_inc_multicast` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:1980` (wait_min), `:2279` (inc), `:2324` (inc_multicast); `tech_reports/tensor_accessor/tensor_accessor.md` | — | — | — | batch sizes `core_w_tiles`, `out_batch_tiles` |
| exchange_round | raw_api (dataflow + Fabric) | `WorkerToFabricEdmSender::build_from_args`, `PacketHeaderPool::allocate_header`, `fabric_set_unicast_route`, `to_noc_unicast_write` / `to_noc_fused_unicast_write_atomic_inc` / `to_noc_unicast_atomic_inc`, `send_current_slot_non_blocking`, `noc_async_writes_flushed` | `.claude/references/blackhole-fabric.md` §5; `ttnn/ttnn/operations/examples/fabric_all_gather/program_descriptor_with_inline_kernels.py:200-305` (sender with ready fence, to adapt); host `ttnn.setup_fabric_connection` (`ttnn/cpp/ttnn-nanobind/fabric.cpp:151`), `ttnn.get_forwarding_link_indices` (`:381`) | — | scratch | peer scratch | `recs_per_packet` (derived from payload) |

**Helpers considered and rejected, for raw_api entries:**
- **Gamma row-0 assembly (load_block).** Considered `tilize<core_w_tiles, cb_gamma_stick, cb_gamma_tiles>(1, 1)` (`tilize_helpers.hpp:187-197`). Rejected: it needs an extra RM staging CB and a compute tilize init/uninit pair, only to produce a tile whose rows 1–31 are never read under `BroadcastDim::Row`. The dataflow placement costs 2 L1 moves per tile and no compute.
- **Fabric exchange.** `mcast_pipe.hpp` (`dataflow_kernel_lib`) is on-chip NoC multicast only and has no Fabric route. The CCL helper library (`ttnn/cpp/ttnn/kernel_lib/ccl/`, `blackhole-ccl-kernels.md` §12) is **[unmeasured]** and 1D-route-typed. The raw sender pattern is the measured one in both reference examples. A port relay of fixed records is not a block compute phase, so there is no compute helper to use.
- **Stats-ready multicast.** `mcast_pipe` `SenderPipe::send_signal` was considered. Rejected: it expects a sender/receiver pipe pair with handshake semaphores both ways, whereas this is a one-way counter increment from up to 2L independent ports into one counter. `noc_semaphore_inc_multicast` is the one-transaction primitive.
- **Combine post-op body.** The chain's `MulUnary` / `AddUnary` / `Rsqrt` elements need a CB input (`chain.hpp:356`). The post-op acts on a DEST tile inside `reduce` (`reduce_helpers_compute.hpp:514-516`), and `transpose_dest` has no chain element. Packing the reduced tile out and re-reading it through a chain would add a CB plus a pack and unpack per tile row, only to express four SFPU calls.

## Broadcast Verification

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast Dim |
|-------|-----|-------------------|-------------------|---------------|
| sumsq | Mul (x·x) | cb_x_resident: All | cb_x_resident: All | None |
| gamma_prescale | Mul | cb_x_resident: All | cb_gamma_tiles: Row0 (1D [W]) | Row |
| normalize | Mul | cb_xg / cb_x_resident: All | cb_rstd: Col0 (REDUCE_COL output transposed by `transpose_dest`) | Col |

The combine reduce consumes cb_stats_combine as REDUCE_COL: rows = sources, output in Row0 before the transpose.

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| FPU `ELWMUL` with `acc_to_dest` (`DestAccumulation::PerRow` on Mul) | The accumulation test exercises Add (`tests/eltwise/chain/accumulation.cpp:32`). Mul+acc goes through `mul_init(…, acc_to_dest)` (`chain.inl:~1185`), which no in-tree test covers. | The acceptance test checks sumsq through the whole op. Fallback if wrong, same CB count: per tile row, `eltwise_chain(tiles(core_w_tiles), BinaryFpu<Mul,x,x>, PackTile<output(cb_sq_acc, OneUpfront, OneAtEnd, …, L1Accumulation::Enabled)>)` (packer L1 accumulate into one tile per row; catalog `eltwise_l1_vs_dest_accumulate`). |
| tf32 FPU sources for float32 x | The squares and the normalize multiplies truncate fp32 to tf32 in srcA/srcB | The stats sums are Accurate (SFPU fp32), which is the error-amplifying part. Squares carry ~2⁻¹¹ relative error each, averaged over W. Watch float32 rel-RMS (gate 0.005). Fallback: SFPU `Square<>` in the sumsq chain. |
| In-place / cross-phase credit on cb_x_resident | sumsq waits without popping, prescale or normalize pops later. Each phase has real `wait_front`, so no in-place write ever happens on cb_x_resident. | Only the reader produces cb_x_resident and only compute consumes it. No phase writes back into it (`l1-footprint-discipline.md` Rule 3). |
| Global semaphores never reset by dispatch | A count left non-zero hangs the next call or reads too early | Re-arm by decrement (`0u − consumed`) followed by `noc_async_atomic_barrier()`. Each call sends and consumes exactly one ready per connected port and `expect_in` arrivals. Bit-identical repeat test. |
| Remote writes into a device that has not started this call | A peer port could land records in L1 still owned by the previous op | Ready fence (`fabric_all_gather` sender): no data before the peer's ready. Scratch is a per-call mesh tensor, so it has the same address on every device. |
| Semaphore ids vs fabric setup | `setup_fabric_connection` takes the lowest free ids | Declare `local_arrivals` and `stats_ready` SemaphoreDescriptors over **all** cores before any `setup_fabric_connection` (`blackhole-ccl-kernels.md` §2). The generic_op hash ignores semaphore ids (§5.1). |
| Program-local semaphore addresses across cores | Workers inc the port's `local_arrivals`; ports mcast-inc the workers' `stats_ready` | Define both descriptors over the union core set, so `get_semaphore(id)` resolves to the same address on every core of the program. |
| NoC1 multicast rectangle orientation | On NoC1, mcast start and end coordinates are swapped relative to NoC0 | Port BRISC is on NoC1 (fabric rule 7). Pass the worker rectangle in NoC1 order (or issue the mcast with an explicit `noc=0` index). |
| Records arrive per entry in order | Relay waits `arrival ≥ k` for its k-th inbound entry. Assumes one ordered sender per counter. | Exactly one upstream same-direction port writes each port's arrival counter, with one fused inc on the last packet of each entry. |
| Combine rows beyond G·S_w | Garbage rows would enter the column sum | Zero-fill the reserved combine pages every round before placing records |
| Worker vs device ordering of sources | The bit-identical requirement plus identical rstd across the group | Source index `s = q·S_w + w` is fixed. The reduce order is fixed (SFPU Accurate path). The same sources reach every worker of the row group on every device. |
| Gamma partial last tile (refinement `w_non_aligned`) | The last gamma tile reads past W_local | Phase 0 is tile-aligned. The refinement reads `min(32, W_local − 32·t)` elements. |
| `w_non_aligned` refinement | Tile padding of the last W tile must not enter Σx² | The core owning the last W tile accumulates its first `core_w_tiles−1` tiles as now. The last tile is squared and reduced alone with `prepare_partial_reduce_scalers<…, W_local mod 32>` (`reduce_helpers_dataflow.hpp:131`), and its result is published as **one extra record** (S_w+1 sources for that row group). 1/W already uses the true `G·W_local`. |
| `h_non_aligned` refinement | Padding rows produce garbage ss | Rows are independent, so garbage stays in the padding rows of the output and nothing else changes. Tile geometry is already `ceil` per image. |
| `fp32_dest_acc_en=False` refinement | The spec requires stats on the wire and in the sum to stay fp32 | Keep the stats CBs at Float32. The page-format audit's "Over" finding is then justified by the requirement, not by DEST width. Bulk CBs (cb_xg) become Float16_b. |
| Ring on wrap-less boxes | Golden prunes Ring there; `test_ring_mock.py` opens 1×N / N×1 snake meshes | The Ring path is generic in G via `_schedule` (no special case for the mock). Validate wrap by fabric config plus `get_forwarding_link_indices(last, first)`. |
| `num_links=None` resolves to 4 on the QuietBox | The SUPPORTED list is {1,2} | validate treats None as auto (always legal). Port columns `2L ≤ grid_x` hold. |
| First call creates global semaphores (host sync once) | Not allowed inside trace capture | Create on first use and cache. The first call (the model's compile run) happens before capture. Later calls only patch runtime args. |
