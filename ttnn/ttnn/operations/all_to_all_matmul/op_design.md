# Operation Design: all_to_all_matmul

## Overview

| Field | Value |
|-------|-------|
| Classification | fused (CCL all-to-all over Fabric 2D + per-head batched matmul), one `ttnn.generic_op` on a `MeshProgramDescriptor` |
| Goal | Within each group of G devices along `cluster_axis` (a mesh row), move the query tensor from head-sharded to sequence-sharded and multiply every head by its own (K, N) matrix of a replicated weight, in **one** device program per call. The remote rows land directly in the L1 of the core that multiplies them, and the weight is read from DRAM while those rows are still in flight. |
| Math | `chunks[p] = chunk(x[p], G, dim=2)`; `a2a[q] = cat_p(chunks[p][q], dim=1)`; `out[q][b, h, m, :] = a2a[q][b, h, m, :] @ w[0, h]`, with `h = p*Hl + j` = source p's local head j and `m` in rows `[q*S/G, (q+1)*S/G)` of the chunk |
| References | `ttnn/ttnn/operations/examples/fabric_all_gather/` (ports, chunk/batch walk, arrival counters, ready fence, `plan()`), `fabric_ring_exchange/` (`_first_hop_dir`, direct vs relay, fence cost), `fabric_reduce_scatter/` (different slice per destination), `.claude/references/blackhole-fabric.md`, `.claude/references/blackhole-ccl-kernels.md`, `ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp`, `eval/golden_tests/all_to_all_matmul/{feature_spec,roofline,helpers}.py`, `tt_metal/third_party/tt_ops_code_gen/references/precision_convention.md` |

### Cross-device work, by kind

Every cross-device item in this design is labelled **TRANSPORT** (how bytes cross the fabric, measured in isolation, taken from the examples) or **OVERLAP** (hiding the transport behind this op's own work, measured only in the op).

| Item | Kind | Decision | Evidence |
|------|------|----------|----------|
| What crosses the fabric | TRANSPORT | The **input rows** (bit-exact tiles), never outputs: output per head is N/K = 4x the input at every perf case | roofline `links` term; output-exchange is a `deferred` regime row |
| Far quarter (ring distance 2) | TRANSPORT | **Direct** 2-hop send on the routing table's own route (`fabric_set_unicast_route`); the port whose direction matches the route's first hop sends it | prompt: 4-ring all-to-all at 7232 B took 53-56 us direct or relayed alike; `fabric_ring_exchange` `_first_hop_dir` |
| Links | TRANSPORT | `num_links` (2) ports per direction, one producer per router channel, heads split across links | `blackhole-fabric.md` rules 6, 10; `fabric_all_gather` 2 links 0.53-0.58x time at size |
| Packet | TRANSPORT | `packet_tiles = floor(max_payload / tile_bytes)` tiles per packet from `ttnn.get_tt_fabric_max_payload_size_bytes()` (3 at 7232 B, 7 at 14400 B, 2 at 4160 B); never hard-coded | `blackhole-fabric.md` rule 1-2 |
| Arrival signal | TRANSPORT | One fused write+atomic-inc on the **last packet of each entry** (entry = one head's rows for one destination); every other packet a plain write | `blackhole-ccl-kernels.md` §6.1: fused inc on every packet halves the rate; 1 per ~8 is the measured sweet spot (an entry is 7 packets at 7232 B) |
| Ready fence | TRANSPORT | Each sending port waits for one *ready* from every destination chip it writes into before its first data packet | `blackhole-ccl-kernels.md` §6.3; `fabric_ring_exchange`: fence_direct 6.35 us |
| Landing place | OVERLAP | Remote rows land **in the consuming compute core's L1** (persistent L1 scratch, same address on every chip), not in DRAM | removes 2 x 2.95 MB of DRAM traffic per chip (Traffic ranking) |
| Work before the data lands | OVERLAP | Every compute core issues its **weight** reads (the dominant DRAM traffic) before it waits for its rows; heads whose rows are local are processed first | Stall-shadow check, Block schedule |
| Per-entry notification | OVERLAP | A head is computed as soon as its own entry's increment arrives; no core waits for the whole exchange | arrival counter per head slot |
| Last landing to output | OVERLAP | After the last entry lands, the chain is one head's matmul + its output write on one core | Perf lamp "landing granularity" |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` (mesh) | yes | rank 4 `(B, Hl, S, K)` per device, TILE, DRAM interleaved, same shape/dtype on every device; `S % G == 0`; `K % 32 == 0` | — | RT (address) |
| `weight` | `ttnn.Tensor` (mesh) | yes | rank 4 `(1, H, K, N)`, `H == G*Hl`, same K as input, TILE, DRAM interleaved, same on every device; `N % 32 == 0` | — | RT (address) |
| `cluster_axis` | int | yes (kw) | `1` (SUPPORTED); mesh extent along it ≥ 2 | — | host |
| `topology` | `Topology` | no | `Ring` | `Ring` (import `from ttnn.operations.ccl import Topology`) | host |
| `num_links` | int | no | `1 ≤ num_links ≤ min over the group's hops of len(get_forwarding_link_indices)`; SUPPORTED `{2}` | 2 | host (sets port count) |
| `memory_config` | `ttnn.MemoryConfig` | no | interleaved; buffer DRAM (Phase 0) or L1 (refinement) | `ttnn.DRAM_MEMORY_CONFIG` | host |
| `compute_kernel_config` | `ttnn.ComputeConfigDescriptor` | no | any fidelity / approx; `fp32_dest_acc_en` gated | `default_compute_kernel_config()` = HiFi4, `fp32_dest_acc_en=True`, `math_approx_mode=False` | CT (compute config) |

`default_compute_kernel_config()` is exported from the op module and is the only place the default is spelled; `None` resolves through it (the golden tagger `axes.py` calls the same factory).

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | per device `(B, Hl, S, K)`; weight `(1, H, K, N)`, `H = G*Hl` |
| Dtype | input bfloat16; weight bfloat16 (Phase 0), bfloat8_b (refinement) |
| Layout | TILE |
| Memory | DRAM interleaved (both) |

### Output

| Property | Value |
|----------|-------|
| Shape | per device `(B, H, S/G, N)` |
| Dtype | input dtype (bfloat16) |
| Layout | TILE |
| Memory | interleaved, in `memory_config.buffer_type` (DRAM default; L1 refinement) |

### Op-owned persistent resources (created once per mesh, cached at module level, never per call)

| Resource | What | Same address on every chip because | Lifetime / reset |
|----------|------|-----------------------------------|------------------|
| `landing_scratch` | L1 HEIGHT_SHARDED bf16 TILE tensor over the **compute core set** (all non-port cores), one shard per core of `scratch_slot_capacity_tiles` tiles | mesh-allocated (`ReplicateTensorToMesh` / mesh buffer) | grow-only: reallocated only when a call needs a larger shard; data needs no reset (every slot is fully overwritten before it is read) |
| `arrival_sem[i]`, `i < max_heads_per_core` | global semaphores (`ttnn.create_global_semaphore`) over the compute core set, initial 0 | global semaphore | grow-only in count; every call leaves each at 0 (consumer decrements what it waited for) |
| `ready_sem` | one global semaphore over the port core set, initial 0 | global semaphore | every call leaves it at 0 |

Creation does one `ttnn.synchronize_device` (so every chip holds 0 before any increment can arrive). It happens on the first call for a mesh (and on a grow), never on a call that reuses the cache — so a traced or back-to-back call performs no host synchronization. Port placement is computed on the host only (see Work Distribution) — **no probe program is dispatched**: a probe would be a second device dispatch inside an invocation.

## Blocking Model

The op's index space per device q: batch `b`, output head `h` (`= p*Hl + j`: source position p, source-local head j), output row `m` (one of the quarter's rows), output column `n`, contraction `k`. Batch folds into the row axis (the weight does not vary with b, and the rows of one head across all b are one block).

Derived geometry (all alignment-aware, per image): `Kt = K/32`, `Nt = N/32`, `Mq = S/G`, `Mq_t = ceil(Mq/32)`, `St = ceil(S/32)`, `tensor_rows_tiles = B * Mq_t` per head, `slot_tiles = B * Mq_t * Kt` (one head's A block).

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `head` (h; within a chip the output heads, across chips the source position p) | **independent** — each head has its own weight and its own rows; no result spans heads | `block_heads` | 1 — a head shares no operand with another head, so a 2-head block amortizes only one matmul short-init per head while doubling `cb_weight`; the per-block fixed cost is already amortized over a whole head (`B*Mq_t x Nt` output tiles) | host constant `BLOCK_HEADS = 1` | **spread across the compute grid**: `ttnn.split_work_to_cores(compute_cores, H)` → `heads_per_core` 1 or 2 at the perf cases; same head→core map on every chip | knob-turn |
| `batch` (b) | **independent**, but the weight is **reuse-shared** along it (one `w[h]` for every b) | folded into `block_m` | all B | host: `block_m_tiles = B * Mq_t` | not split (a head's batches stay on its core so the weight is read once) | knob-turn |
| `rows` (m, the quarter's rows) | **independent**, weight **reuse-shared** along it | `block_m_tiles` | `B * Mq_t` — the whole head's rows in one block (the core's full assignment; resident A slot) | host-derived from the shape; one source, passed as CT arg | not split across cores (splitting re-reads `w[h]` once per row split, or needs a weight multicast: regime `head x row split`, rejected) | scheme-change |
| `n` (output columns) | **independent**, A **reuse-shared** along it (every column block multiplies the same rows) | `block_n_tiles` | largest divisor of `Nt` with `Kt * block_n_tiles * weight_tile_bytes ≤ WEIGHT_BLOCK_BYTES` (= `Nt` at every perf case: one block per head) | host-derived from `WEIGHT_BLOCK_BYTES` (one host constant, 256 KiB); CT arg | not split across cores in Phase 0 (regime `head x N split`, deferred); within a core the n-blocks of a head run in sequence on the retained A slot | scheme-change (needs an A fan-out) |
| `k` (contraction) | **dependent** — every output element sums over all of K | `block_k_tiles` | `Kt` — the whole reduction in one K-block (A slot holds all K; one weight column block holds all K); legal while `Kt * weight_tile_bytes ≤ WEIGHT_BLOCK_BYTES` | host CT arg (`= Kt` in the built regime) | within a core only (sequential accumulate in DEST); cross-core split is regime `K split + combine`, deferred | scheme-change |
| `source → destination` (the exchange: dest q takes rows quarter q of every source p) | **exchange axis** — the rows a destination needs are fixed by its position; this is the all-to-all, not a split choice | `entry` = one (destination chip, head) block of `slot_tiles` tiles; TRANSPORT unit `packet_tiles` | entry = the whole head's quarter; `packet_tiles = floor(max_payload / tile_bytes)` | host: `ttnn.get_tt_fabric_max_payload_size_bytes()` read per call, passed as CT arg | ports: per chip `2 * num_links` port cores (one per (direction, link)); a destination's entries go to the port whose direction is the route's first hop; heads split across links | knob-turn (links), TRANSPORT variants are regime rows |
| `group` (the other mesh axis) | **independent** — groups run concurrently and never exchange | — (one group per mesh row) | each mesh row is its own group | mesh shape | each chip's program serves its own group | — |

**Every stage's own data, re-tabled.** The scheme has three stages with their own data: (1) the port stage's entries — independent along (destination, head), spread over the `2*num_links` ports (all ports busy: each sends ≥ 1 quarter's worth); (2) the landed A slots — the same head axis as above, consumed where they land (no designated collector: every compute core computes its own heads; no stage runs on one core while others wait); (3) output row groups — independent along rows/columns, written by the core that computed them. No core collects, coordinates or hands out work; the only coordination is the per-port ready fence, which is a communication role, not a work assignment.

**Phase 0 scheme (committed):** head-split across the full compute grid (grid minus port cores), one head per block, the head's rows (all batches) and all of K resident, N in `block_n_tiles` column blocks (one at every perf case), remote rows landing zero-copy in the consuming core's L1. Every knob above is a parameter with the single source named; none is an inlined constant, including the grid (read from `mesh_device.compute_with_storage_grid_size()`) and the port count (`2 * num_links`).

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_weight` | `WEIGHT_DEPTH` | 2 blocks | the next (head, n-block) weight block loads while the current one multiplies; on a 1-block core it lets the reader finish all weight reads before it blocks on the A arrival |
| `cb_out` | `OUT_DEPTH` | 2 row groups | packer fills one row group while the writer drains the other |
| `cb_port_chunks` | `PORT_BATCH_CHUNKS` (batch) x 2 | 4 chunks per batch, 2 batches | reader issues a batch of DRAM reads per barrier while the sender sends the previous batch (measured best-or-within-10% at every size, `blackhole-ccl-kernels.md` §6.2) |
| `cb_a_landed` | none (whole resident) | all slots of the core | the landed rows are written by the remote router at any time during the call; the buffer cannot recycle a slot within a call |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Fabric packet max payload | `packet_tiles` | `packet_tiles = floor(max_payload / tile_bytes)`; validate `packet_tiles ≥ 1` (raise if the router payload is smaller than one tile) | a packet larger than the payload is rejected / corrupts the router slot |
| Packet header pool | headers per data-movement RISC ≤ 12 | sender's header ring = `PORT_BATCH_CHUNKS` (4) + 1 ready/route headers per destination ≤ 12 | the pool hangs with `PACKET HEADER POOL EXHAUSTION` |
| Packet contiguity | one packet lands at one contiguous L1 range on one core | a packet never crosses an entry boundary (the last packet of an entry may be short) | a packet straddling two heads would write the second head's tiles into the first core's slot |
| Arrival counters | one ordered sender per counter | one counter per **head slot** (`arrival_sem[slot]`), exactly one entry (one source port) per remote slot | with one counter per core and two sources, the threshold cannot tell which head landed — computes on unlanded rows |
| Semaphore count | `max_heads_per_core` global semaphores | `max_heads_per_core = ceil(H / num_compute_cores) ≤ 8` (validate); grow-only cache | the 9th slot has no counter |
| L1 residency of the A slots | `heads_per_core * slot_tiles * tile_bytes` | predicate `A_RESIDENT`: `≤ A_SLOT_BUDGET_BYTES` (384 KiB, one host constant); a call outside it is regime R3 (deferred) and the host raises `RuntimeError` naming the predicate (it holds for every INPUTS shape: max 256 KiB) | landing overwrites other CBs |
| Weight block | `Kt * block_n_tiles * weight_tile_bytes ≤ WEIGHT_BLOCK_BYTES` | `block_n_tiles` = largest divisor of Nt meeting it; requires `Kt * weight_tile_bytes ≤ WEIGHT_BLOCK_BYTES` (K ≤ 4096 at bf16) — else regime `K-blocked` (deferred) | `cb_weight` exceeds L1 |
| DEST capacity | `out_subblock_h * out_subblock_w ≤ DEST_AUTO_LIMIT` (4 at fp32 DEST, 8 at 16-bit) | host derives `(out_subblock_h, out_subblock_w)` from `(block_m_tiles, block_n_tiles, fp32_dest_acc_en)`: largest product within the limit, `out_subblock_h | block_m_tiles`, `out_subblock_w | block_n_tiles`, prefer wide | the helper `ASSERT`s (`matmul_block_helpers.inl:158`) |
| Fabric route | a destination two hops away | route only with the routing table (`fabric_set_unicast_route`) on the port whose direction equals `ttnn.get_eth_forwarding_direction(src, dst)` | an explicit "2 hops in my first direction" route leaves the snake board and hangs (prompt, Topology semantics) |

### Regimes

Selection is host-side, per call. Predicates use the symbols above with `C = num_compute_cores = grid_cores - 2*num_links`, `heads_per_core = ceil(H / C)`.

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **R1 head-resident** (input exchange, L1 landing, head split) | **built** | `A_RESIDENT` (`heads_per_core * slot_tiles * 2048 ≤ 384 KiB`) and `Kt * weight_tile_bytes ≤ WEIGHT_BLOCK_BYTES` and `alignment == tile_aligned`. Holds for every INPUTS shape (max A residency: `(1,4,128,4096)` → 128 tiles = 256 KiB; max `Kt` = 128 → one bf16 column block = 256 KiB) | `(1 head) x (block_m_tiles = B*Mq_t) x (Kt) x (block_n_tiles)` | **minimum** against DRAM and links: input read from DRAM once (3/4 by ports, 1/4 by compute cores), weight read once, output written once (DRAM out) or 0 DRAM (L1 out); each remote quarter crosses each link of its route once (busiest directed link = input/2 when the routes' first hops alternate, as on the QuietBox snake). Above minimum only in core-local traffic: none (landing is zero-copy) | per block: one `matmul_block_init` + one data-format reconfig, one A wait, `block_m/out_subblock_h` row-group handshakes, one weight-block handshake; per head: one arrival wait + one counter re-arm. Bigger blocks (whole head, `block_n = Nt`) pay these once per head |
| R1' head-resident, **m_non_aligned** | **deferred** — refinement on R1's structure: the port reader (and the local-quarter reader) re-tiles the quarter's rows (row shift `(q*Mq) mod 32` across two source tiles, 32 B face-row copies in L1 after a whole-tile DRAM read; padding rows zeroed) before the packet is built; landing, counters and compute unchanged | `alignment == m_non_aligned` | same as R1 (`Mq_t = ceil(Mq/32)`) | same as R1, plus one core-local L1→L1 copy of each re-tiled entry | same as R1 |
| R2 K-blocked weight | **deferred** — no INPUTS shape has `K > 4096` (bf16) / `K > 7680` (bfp8); R1 serves all of them | `Kt * weight_tile_bytes > WEIGHT_BLOCK_BYTES` | `(1 head) x block_m x block_k_tiles x Nt`, A slot landed in **k-block-major** order so the helper pops it per K-block; spill/reload through an fp32 `cb_partials` | same DRAM/link minimum; adds an interm spill of `block_m x Nt` fp32 tiles per K-block (core-local) | the K-loop's spill/reload per K-block; packer-L1-acc is its perf lamp |
| R3 DRAM landing | **deferred** — every INPUTS shape satisfies `A_RESIDENT` (max 256 KiB) | `not A_RESIDENT` | as R1, A read from a DRAM scratch `(B, H, S/G, K)` per K-block | +2 x remote input DRAM crossings (remote rows written to DRAM by the router, read back by compute): +5.9 MB per chip at Kimi | — |
| R4 head x N split (A fanned out to the N-split cores over the NoC) | **deferred** — at every perf case compute is 6.6-13 us of a ≥ 32 us bound and the weight is already read by ≥ 64 cores at once (GLM: 64 of 106 cores busy, ~200 KB each), so the extra cores buy nothing the roofline can see; it pays where `H < C/2` and compute dominates (the wide-N / wide-K golden cells, not perf cases). Reachable: the landing core becomes the head's injector — after its arrival wait it multicasts the slot to its N-partners (`ttnn.Mcast` + `McastArgs`, `mcast/kernel/mcast_args.hpp`) and each partner runs R1's compute on its column range; R1's per-core loop is unchanged | `H * 2 ≤ C` (and compute-bound shapes) | `(1 head) x block_m x Kt x (Nt / n_split)` per core | +A per head x (n_split - 1) cross-core (40 KB per partner at Kimi); weight unchanged (columns partition it) | as R1 |
| R5 head x row split (weight multicast to the row-split cores) | **rejected** — superseded by R4: per head the weight (`Kt*Nt` = 64 tiles at Kimi) is 3.2x the A slot (20 tiles), so fanning A out (R4) moves fewer bytes than fanning the weight out; without the multicast it re-reads the weight from DRAM per split (dead end in a DRAM-bound op) | — | — | +weight x (splits - 1) cross-core, or DRAM | — |
| R6 output exchange (compute at the source on its own heads, send output quarters) | **deferred** — reads only `1/G` of the weight (12.6 → 3.1 MB DRAM per chip at Kimi) but puts `N/K = 4x` the bytes on the links (busiest link 7.9 MB vs 1.97 MB: 81 us vs 20 us at full link rate); wins only when `N < K`. The one INPUTS shape with `N < K` (`(1,4,128,4096)x(1,16,4096,128)`) is a correctness cell that R1 serves correctly. Reachable: same ports, counters, fence and landing; the entry payload becomes output tiles and the landing target the output buffer | `N < K` | `(1 local head) x (all S rows) x Kt x block_n` | DRAM weight 1/G; links x N/K | as R1 |
| R7 K split across cores + combine | **deferred** — K ≤ 192 at every perf case (Kt ≤ 6: a split leaves < 3 tiles per core while the combine moves `block_m x Nt` fp32 partials, 4x the A slot); for the deep-K golden cell (Kt=128, H=16 → 16 busy cores) it would raise occupancy and shrink the 256 KiB A slot (residency), but that cell is not a perf case and R1 serves it | `Kt ≥ 32 and H * 4 ≤ C` | `(1 head) x block_m x (Kt / k_split) x Nt` + a cross-core reduce of partials | + `(k_split - 1) * block_m * Nt` fp32 tiles cross-core | — |
| T1 far quarter direct (2-hop on the routing table's route) | **built** (TRANSPORT) | always (Ring, G ≥ 3) | — | far quarter crosses 2 links, router-forwarded (no Tensix relay) | — |
| T2 far quarter relayed through the neighbour, split half each direction | **deferred** (TRANSPORT) — measured equal to direct at the perf payload (53-56 us both at 7232 B); it wins only at 14400 B (44 vs 49 us) and from ~15 MiB per chip, and it forces balance where a routing table sends every far route the same way (a real 4-ring torus may: 3 quarters on one direction instead of 2). Reachable: the relay lands in the neighbour's `landing_scratch` exactly like a local entry plus a forward list on that chip's port reader, counters and fence unchanged | routing-table first hops for the group's far routes not alternating, or payload ≥ 14400 B | — | balanced 2 quarters per directed link | — |

**Regime-pinned tests:** Phase 0 builds one blocking regime (R1) and one transport regime (T1); the only shape-dependent selection inside R1 is `block_n_tiles` (one n-block at every perf case; 8 n-blocks at `N=8192`; 4 one-column n-blocks at `K=4096`) and `heads_per_core` (1 or 2). The acceptance test pins one shape per case: single n-block (Kimi), multi n-block (`(1,4,128,128)x(1,16,128,4096)` exercises 4 blocks — in the golden suite), 2 heads per core (DeepSeek shape in the golden suite), and `heads_per_core = 1` with idle cores (small shapes).

### Traffic ranking

Per device, Kimi focus shape (input 3.93 MB, weight 12.58 MB, output 15.73 MB), counting crossings per tier.

| Rank | Candidate split | DRAM crossings | Link bytes (busiest directed link) | Cross-core NoC | Verdict |
|------|-----------------|----------------|------------------------------------|----------------|---------|
| 1 | **Heads across cores, input exchanged, L1 landing (R1)** | input 1, weight 1, output 1 (0 if L1) | input/2 = 1.97 MB | 0 beyond the router's landing writes | **chosen**: minimum on every tier that binds |
| 2 | Heads x N across cores (R4) | same | same | + A per head x (splits-1) | deferred: same DRAM/link bytes, more NoC; occupancy not binding |
| 3 | Heads, DRAM landing (R3) | input 1 + remote quarters 2 more (write + read) = +5.9 MB | same | 0 | deferred (residency predicate never fails on INPUTS) |
| 4 | K split + combine (R7) | same | same | + fp32 partials ≫ A | deferred |
| 5 | Heads x rows (R5) | weight x splits (or + weight x splits on NoC with multicast) | same | + weight | rejected |
| — | Output exchange (R6) | weight 1/G (−9.4 MB) | 4x (7.9 MB) | 0 | DRAM-cheapest, link-dearest: at the perf cases the links (194 GB/s egress per chip) bind before DRAM (512 GB/s): max(13.7, 81) vs max(32.3, 20.3) us — deferred for `N < K` |

Occupancy is kept out of this ranking: R1 fills 96 of 106 compute cores at Kimi, 106 (22 with 2 heads) at DeepSeek, 64 at GLM — recorded under R4.

**Operand-reuse check (chosen split = heads across cores):** input A varies with h → not reuse-shared across cores; weight varies with h → not reuse-shared across cores (each head's weight is read by exactly one core per chip). Within a core: the weight is reuse-shared along batch and rows → folded into `block_m` (read once per head); A is reuse-shared along n → retained across the head's n-blocks (`InputPolicy::WaitAndRetainOnLastBlock`). Across chips: every chip reads the full weight (replicated by contract; the alternative is R6).

**Stall-shadow check.** Three stages wait on something they do not produce:

| Waiting stage | Waits for | Work that does not depend on it, scheduled into the wait |
|---------------|-----------|----------------------------------------------------------|
| Compute core reader, remote head | its entry's arrival increment | the head's **weight block(s)** — the dominant DRAM read (12.6 MB per chip) — are issued before the arrival wait (up to `WEIGHT_DEPTH` blocks); the core's **local-source heads** are ordered first in its slot list, remote heads after, in their expected arrival order (source-local head index j ascending) |
| Port sender | ready from each destination chip | the port reader keeps prefetching its entries from DRAM into `cb_port_chunks` (it never waits on the fence) |
| Compute after the last landing | — (tail) | nothing independent remains on that core; the tail is one head's matmul + output write (perf lamp "landing granularity") |

No reorder changes floating-point order (each output element is one K-sum in DEST, same order as Phase 0), so no precision gate applies.

### Block schedule

```cpp
// per chip, per call (logical schedule; ports and compute cores are separate kernels running concurrently)

// ---- port core (direction d, link l): one per (d, l) ----
send_readies();                                   // header-only incs to every source port that writes into this chip
for (uint32_t entry_idx = 0; entry_idx < num_entries_this_port; ++entry_idx) {
    load_entry(entry_idx);                        // reader: slot_tiles input tiles (b, j, quarter rows of dest, all K) DRAM -> cb_port_chunks, in dest-slot order
    send_entry(entry_idx);                        // sender: first time only: wait_readies(); then ceil(slot_tiles/packet_tiles) packets into
}                                                 //   dest chip's landing_scratch slot on the dest core; last packet fused-incs arrival_sem[slot]

// ---- compute core (owns heads_this_core heads, slot order: local first, then remote by j) ----
for (uint32_t slot_idx = 0; slot_idx < heads_this_core; ++slot_idx) {
    load_weight_block(slot_idx, 0);               // issued BEFORE land_block: fills the fabric wait (up to WEIGHT_DEPTH blocks ahead)
    land_block(slot_idx);                         // local head: DRAM -> slot; remote head: wait arrival_sem[slot] >= 1, re-arm (-1); push slot
    for (uint32_t n_block_idx = 0; n_block_idx < num_n_blocks; ++n_block_idx) {
        load_weight_block(slot_idx, n_block_idx); // (block 0 already issued)
        matmul_block(slot_idx, n_block_idx);      // block_m x Kt x block_n, A retained across n-blocks
        store_block(slot_idx, n_block_idx);       // block_m/out_subblock_h row groups -> output pages
    }
    release_slot(slot_idx);                       // pop the slot (compute)
}
```

| Block operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------------|-------------|--------------------|-----------------------------------|
| `send_readies` / `wait_readies` | header-only packets | — | once per port per call (1-2 readies sent and awaited per port) |
| `load_entry` | `slot_tiles` input tiles = `B*Mq_t*Kt` | — | one DRAM read barrier per batch of `PORT_BATCH_CHUNKS` chunks; batch also ends at every entry end |
| `send_entry` | `ceil(slot_tiles / packet_tiles)` packets | header ring | one `noc_async_writes_flushed` + pop per batch; one fused inc per entry (≈ 1 per 7 packets at 7232 B) |
| `load_weight_block` | `Kt x block_n_tiles` weight tiles | — | one read barrier per block (runs of `block_n_tiles` consecutive pages per k row) |
| `land_block` | `slot_tiles` | the slot (until `release_slot`) | one arrival wait + one counter decrement (+ atomic barrier) per remote head; one DRAM read barrier per local head |
| `matmul_block` | `block_m_tiles x Kt x block_n_tiles` | A slot, weight block | one `matmul_block_init` + reconfig per block (helper `InitMode::Short`); one in0 wait per block (no pop), one in1 wait/pop per block |
| `store_block` | `out_subblock_h x block_n_tiles` per row group, `block_m/out_subblock_h` groups | — | one write barrier per row group |
| `release_slot` | `slot_tiles` | — | one pop per head |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| **OVERLAP — landing granularity (tail)** | One arrival increment per (destination, head) entry means the last-landing head is multiplied whole after the exchange ends: a one-head tail of ~8-15 us (HiFi2/HiFi4 at Kimi) behind the transport | entries of `rows_per_entry < block_m` row tiles with an increment each and the matmul started per landed row block (rows are independent); measure tail vs the router stall of more frequent fused incs (every 2 packets: 1.05-1.17x slower) |
| **OVERLAP — send order** | Head-major send order (j ascending, round-robin over a port's destinations) lands heads at the same pace on every destination; the cores holding two heads (DeepSeek) and the cores of the last-sent heads set the tail | order entries by destination core's remaining work (2-head cores first), or reverse the order on link 1 |
| **OVERLAP — idle and early-finishing cores** | Kimi leaves 10 compute cores idle and the 24 local-head cores finish before any remote data lands | assign the last-landing heads' columns to idle cores (R4 restricted to the tail heads) |
| **TRANSPORT — port placement** | Phase 0 places each port in the NoC column of its link's Ethernet core from host-side data; if that map is unavailable it falls back to a fixed row and 2 links drop to 40.6 / 30.5 GB/s (rule 6) | host map vs simple placement, per-link utilization |
| **TRANSPORT — partial packets** | An entry of 20 tiles at 3 tiles per packet is 7 packets (6 full + 1 of 2): ~5% more packets than full packing at Kimi 7232 B | land A k-major per (b, row) so entries can be cut at packet multiples; or accept |
| **TRANSPORT — port batch depth** | `PORT_BATCH_CHUNKS = 4` was measured at 14 KiB chunks; at 7232 B a chunk is 6 KiB | 8 chunks per batch at 7232 B |
| **Overlap (block) — whole-head weight block** | A local-head core reads its whole 128 KiB weight block before its first multiply | `block_n_tiles = Nt/2` or `Nt/4` on local-head cores |
| **Grid synchronization — the ready fence** | ~5-6 us per call of pure synchronization (`fabric_ring_exchange`: fence 5.04 us, fence_direct 6.35 us); direct routing makes the far chip's ready a 2-hop wait | T2 (relay) makes every fence wait 1 hop |
| **matmul subblock** | the host-derived `(out_subblock_h, out_subblock_w)` follows `fp32_dest_acc_en`; `block_m = 5` admits only `out_subblock_h ∈ {1, 5}` | `packer_l1_acc` is irrelevant (one K-block); measure 1x4 vs 1x8 (16-bit DEST) |

Catalog entries that informed the knobs: `double_buffer` (batch reads + double-buffered CBs → `WEIGHT_DEPTH = 2`, `OUT_DEPTH = 2`), `matmul_output_subblock` (subblock as large as DEST allows; fp32 DEST halves it), `compute_block_size` (one pass over the whole block, no tile-row loop → `block_m = B*Mq_t`), `noc_placement` (spread DRAM readers row-wise: compute cores enumerated row-major), `fabric_all_gather` / `fabric_reduce_scatter` / `fabric_ring_exchange` (all TRANSPORT knobs), `shared_input_reuse` / `mcast_topology` (R4 / R5 costing).

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| Input → port core | bf16 tiles, `tile_bytes` = 2048 | port reader (NCRISC, NoC0) `TensorAccessor` reads of the entry's pages (consecutive page ids → round-robin over DRAM banks) into `cb_port_chunks`, one CB page = one packet (`packet_tiles` tiles) | input page id `((b*Hl + j)*St + q*Mq_t + mt)*Kt + k`; read in dest-slot order `(b, mt, k)` |
| Port core → remote chip | raw tile bytes, bit-exact | port sender (BRISC, NoC1) on its own fabric connection (`setup_fabric_connection(src, first_hop_neighbour, link)`); packet route `fabric_set_unicast_route(hdr, dst_chip, dst_mesh)` (1 or 2 hops); payload written by the destination router to `get_noc_addr(dest_core_x, dest_core_y, landing_scratch_addr + slot*slot_bytes + chunk_offset)`; last packet of the entry: fused inc of `arrival_sem[slot]` on that core | no receiver kernel; no credits (every entry has its own slot for the whole call) |
| Local quarter → compute core | bf16 tiles | compute reader (NCRISC, NoC0) reads its local-source head's quarter from the input via `TensorAccessor` straight into its slot (the slot's address via `cb_reserve_back` / `get_write_ptr` on `cb_a_landed`) | the local quarter never touches a port or the fabric |
| Landed slot → compute | bf16 tiles in `cb_a_landed` (backed on `landing_scratch`, zero-copy via `ttnn.cb_descriptor_from_sharded_tensor`) | reader `cb_reserve_back(slot_tiles)` → (DRAM read, or `noc_semaphore_wait_min(arrival_sem[slot], 1)`) → `cb_push_back(slot_tiles)` | slot order on a core = processing order, so `get_write_ptr` at reserve time equals `landing_scratch_addr + slot*slot_bytes` (host uses the same formula for the senders) |
| Weight → compute | weight dtype tiles | compute reader `TensorAccessor` reads `Kt` runs of `block_n_tiles` consecutive pages per block into `cb_weight` | weight page id `(h*Kt + k)*Nt + n` |
| Compute → output | output dtype tiles (bf16), TileRowMajor | `matmul_block` packs row groups into `cb_out`; writer (BRISC, NoC1) writes each tile to output page `((b*H + h)*Mq_t + mt)*Nt + n` via `TensorAccessor` | DRAM (Phase 0) or L1 interleaved (refinement) — same code, the accessor follows the buffer |

### TARGET axes as realizations of the scheme

| TARGET axis / value | Kind | What changes |
|---------------------|------|--------------|
| `weight_dtype = bfloat8_b` | knob-turn | `cb_weight` page = bfp8 tile (1088 B); `WEIGHT_BLOCK_BYTES` is in bytes, so `block_n_tiles` grows on its own; unpacker format from the CB |
| `fp32_dest_acc_en = False` | knob-turn | DEST limit 8 → host subblock rule picks larger subblocks; CB formats unchanged (no intermediate CB) |
| `output_buffer = L1` | knob-turn | writer's `TensorAccessor` on the L1 output buffer; output page count per core unchanged; L1 budget check includes the output's per-bank share |
| `alignment = m_non_aligned` | refinement on R1 (row re-tiling in the port reader and the local reader) | regime R1' |
| `memory_layout` | only INTERLEAVED in TARGET | — |
| `num_links`, `topology`, `cluster_axis` | single TARGET value each | — |

Index axes: none (`cluster_axis` is validated, not canonicalized; TARGET uses `1`).

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one head-block: `(head h) x (B*Mq_t rows) x Kt x (block_n_tiles)`; a core owns whole heads |
| Grid | `grid = mesh_device.compute_with_storage_grid_size()` (11 x 10 = 110 on QuietBox 2). Port cores: `2 * num_links` (4), one per (direction, link). Compute cores: the remaining `C = grid - 2*num_links` (106), enumerated row-major (DRAM readers spread along rows), the core set merged into rectangles (`CoreRangeSet.merge`) |
| Per-core work | `ttnn.split_work_to_cores(compute_core_set, H)` → group 1: `ceil(H/C)` heads, group 2: `floor(H/C)`; cores with 0 heads get no kernel. Kimi: 96 cores x 1 head; DeepSeek: 22 x 2 + 84 x 1; GLM: 64 x 1. The head→core map is identical on every chip; the **slot order per core is per chip** (local-source heads, i.e. `h // Hl == p`, first; then remote heads by `h % Hl` ascending), so the host computes `(core, slot)` per (destination chip, head) for the senders |
| Remainder | heads: ragged groups from `split_work_to_cores`; rows: `Mq_t = ceil(Mq/32)` per image, `St = ceil(S/32)`, tile-row index per image (never `floor(B*S/32)`) |
| Ports | per chip, per direction `d ∈ {fwd, bwd}` and link `l`: neighbour = ring position `p±1`; link index `get_forwarding_link_indices(me, neighbour)[l]`. Destination `t ≠ p` is served by direction `_first_hop_dir(me, t)` (`ttnn.get_eth_forwarding_direction`, as `fabric_ring_exchange/program_descriptor.py:55`); its entries (one per source-local head j) go to link `l = j mod num_links`. Send order per port: j ascending, round-robin over the port's destinations. Readies: chip t sends one ready to every (source chip s, port (d, l)) that writes into t, from its own port of direction `_first_hop_dir(t, s)` and link l; each port awaits as many readies as it has destination chips |
| Port placement | host-only: the first word of each connection's runtime args (`setup_fabric_connection(...)`) is the Ethernet channel; map it to the Ethernet core's NoC column through the cluster descriptor + SoC descriptor (`fabric_all_gather` `_eth_physical_lists`, `_worker_nearest_noc1`); fallback when unmappable: `2*num_links` cores of logical row 0, centred (`fabric_ring_exchange` `port_cores`). No probe program |
| Program cache | `ProgramDescriptor.custom_program_hash` over: per-port entry counts and destinations, per-core head counts, `block_n_tiles`, `packet_tiles`, subblock dims, `num_links`, port placement, slot orders (everything that changes a per-core argument layout) |

Regime selection: one built regime (R1). The host raises `RuntimeError` naming the predicate if a call reaches R2 or R3 (no INPUTS shape does). Required tests: the regime-pinned list under Regimes.

### Registry contract (op file)

| Item | Value |
|------|-------|
| `INPUT_TAGGERS` | exactly `{"alignment": tag_alignment}`; `inputs = ((B, Hl, S, K), (1, H, K, N), G)`; `"m_non_aligned"` if `S % 32 or (S // G) % 32`, else `"tile_aligned"`. In `validate()`, G = `mesh_device.shape[cluster_axis]` |
| `SUPPORTED` (Phase 0) | dtype `[bfloat16]`, weight_dtype `[bfloat16]`, layout `[TILE]`, fp32_dest_acc_en `[True]`, alignment `["tile_aligned"]`, output_buffer `[DRAM]`, memory_layout `[INTERLEAVED]`, cluster_axis `[1]`, topology `[Topology.Ring]`, num_links `[2]` |
| `EXCLUSIONS` | `[]` |
| Caller errors (`ValueError` / `RuntimeError`, raised by `validate()` before the support check) | rank ≠ 4 (input or weight); `H != G*Hl`; weight K ≠ input K; `S % G`; `K % 32` or `N % 32`; non-interleaved `memory_config`; `cluster_axis` extent 1; `num_links` > discovered links on any hop; Ring with no wrap link along the axis or a fabric config that does not wrap it; `packet_tiles < 1` |
| Support refusal | `UnsupportedAxisValue` / `ExcludedCell` from `ttnn.operations._op_contract` (NotImplementedError subclasses) |
| Entry point | `validate()` is its first line; `compute_kernel_config=None` → `default_compute_kernel_config()` |

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_a_landed` | 0 | input tile (2048) | `scratch_slot_capacity_tiles` (≥ `heads_per_core_max * slot_tiles`) — globally allocated on `landing_scratch` | spans head-slots x rows x K of the core's assignment (whole resident: written remotely at any time in the call) | input dtype (Float16_b) | reader (compute core) | compute | whole call |
| `cb_weight` | 1 | weight tile (2048 bf16 / 1088 bfp8) | `WEIGHT_DEPTH * Kt * block_n_tiles` | spans K x n-block of one block; streams over heads and n-blocks | weight dtype | reader (compute core) | compute | whole call |
| `cb_out` | 2 | output tile (2048) | `OUT_DEPTH * out_subblock_h * block_n_tiles` | spans one row group of one n-block; streams over rows, n-blocks, heads | output dtype (Float16_b) | compute | writer (compute core) | whole call |
| `cb_port_chunks` | 3 (port cores only) | `packet_tiles * tile_bytes` (one packet) | `2 * PORT_BATCH_CHUNKS` | one batch in flight + one being sent; streams over entries | raw input bytes (input dtype) | reader (port core) | writer = port sender (port core) | whole call |

No interm CB: one K-block, so `matmul_block` receives `cb_out` as the interm placeholder (`matmul_block_helpers.hpp:247-249`).

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `send_readies` / `wait_readies` | header-only | no (fabric raw) | — | — (remote `ready_sem`) | `ready_sem` decremented by readies consumed → 0 |
| 2 | `load_entry` | `slot_tiles` → `ceil(slot_tiles/packet_tiles)` pages | no (dataflow) | input DRAM | `cb_port_chunks`, pages per batch | pushed |
| 3 | `send_entry` | same packets | no (fabric raw) | `cb_port_chunks`, popped per batch | remote `landing_scratch` slot + `arrival_sem[slot]` | empty after last batch |
| 4 | `load_weight_block` | `Kt x block_n_tiles` | no (dataflow) | weight DRAM | `cb_weight`, `Kt*block_n_tiles` | pushed |
| 5 | `land_block` | `slot_tiles` | no (dataflow) | input DRAM (local) / remote writes (remote) | `cb_a_landed`, `slot_tiles` | pushed, retained by compute across n-blocks |
| 6 | `matmul_block` | `block_m_tiles x Kt x block_n_tiles` | **yes** `matmul_block` | `cb_a_landed` (`slot_tiles`, waited, not popped), `cb_weight` (`Kt*block_n_tiles`, waited+popped) | `cb_out`, `out_subblock_h*block_n_tiles` per row group | A still fronted |
| 7 | `store_block` | row groups | no (dataflow) | `cb_out` | output pages | drained |
| 8 | `release_slot` | `slot_tiles` | no (one `cb_pop_front`) | `cb_a_landed` | — | slot popped |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| compute boot | raw_api | `compute_kernel_hw_startup<SrcOrder::Reverse>(cb_a_landed, cb_weight, cb_out)` once at top | `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:60`; Reverse required per `matmul_block_helpers.hpp:100-104` | — | — | — | — |
| `matmul_block` | helper | `compute_kernel_lib::matmul_block` | `ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp:358` (impl `.inl`) | `<transpose=false, packer_l1_acc=false, LastBlockTarget::Out, OutputCBLayout::TileRowMajor, InitMode::Short, in0_policy=InputPolicy::WaitAndRetainOnLastBlock, in1_policy=InputPolicy::WaitAndPopPerKBlock>`; `MatmulBlockShape::of(block_m/out_subblock_h, block_n/out_subblock_w, out_subblock_h, out_subblock_w, Kt, 1, 1)` (`:160`); `interm_buf = cb_out` (`:247-249`, `:307-314`) | `cb_a_landed`, `cb_weight` | `cb_out` | `block_m_tiles`, `block_n_tiles` (and `in0_block_k = Kt`, `num_k_blocks = 1` in R1); subblocks from the host rule, not knobs |
| `matmul_block` lifecycle facts | helper | in0 waits `out_subblock_h*Kt*in0_num_subblocks = slot_tiles` per call | `.inl:194, 265`; retained (no pop) on the last K-block `.inl:539-545`; in1 wait/pop `block_n*Kt` `.inl:207, 269, 549` | — | — | — | — |
| `release_slot` | raw_api | `cb_pop_front(cb_a_landed, slot_tiles)` after the head's last n-block | `api/compute/cb_api.h` | — | `cb_a_landed` | — | — |
| `land_block`, `load_weight_block`, `load_entry` | raw_api | `TensorAccessor::get_noc_addr(page)` + `noc_async_read` + barrier; `cb_reserve_back/push_back` | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:149` | — | DRAM | `cb_a_landed` / `cb_weight` / `cb_port_chunks` | — |
| arrival wait / re-arm | raw_api | `noc_semaphore_wait_min(arrival, 1)`; `noc_semaphore_inc(get_noc_addr(arrival), 0u - 1)`; `noc_async_atomic_barrier()` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:1980, 2279, 1871` | — | — | — | — |
| `send_entry` / `send_readies` | raw_api | `WorkerToFabricEdmSender::build_from_args<TENSIX>` → `open()`; `PacketHeaderPool::allocate_header()`; `fabric_unicast_noc_unicast_write`, `fabric_unicast_noc_fused_unicast_with_atomic_inc`, `fabric_unicast_noc_unicast_atomic_inc` (route via `fabric_set_unicast_route`); `noc_async_writes_flushed()` per batch; `close()` | `edm_fabric_worker_adapters.hpp:91, 558, 623`; `packet_header_pool.h:50`; `tt_metal/fabric/hw/inc/mesh/api.h:57, 949, 237`; `tt_metal/fabric/hw/inc/tt_fabric_api.h:137`; `dataflow_api.h:1817` | `PORT_BATCH_CHUNKS` header ring | `cb_port_chunks` | remote L1 | `packet_tiles`, `PORT_BATCH_CHUNKS` |
| `store_block` | raw_api | `TensorAccessor` + `noc_async_write_tile`/`noc_async_write` + barrier per row group | `tensor_accessor.h:149` | — | `cb_out` | output | — |

**Helpers considered and rejected (dataflow raw entries):**

| Raw entry | Helper considered | Why it cannot express this | Citation |
|-----------|-------------------|----------------------------|----------|
| `send_entry` / fence | `ttnn/cpp/ttnn/kernel_lib/ccl/` `FabricStreamSender` (§12) | linear-API layer: routes by 1D hop count and its route is bound once per stream; this op needs per-packet 2D destination routing to 1-hop and 2-hop chips on one connection, and the layer is **[unmeasured]**; the measured path is the raw sender of `fabric_all_gather` | `.claude/references/blackhole-ccl-kernels.md` §12 ("Routes are 1D (hop counts)", "[unmeasured]: no example here uses it") |
| `land_block` (A broadcast) | `mcast/kernel/mcast_args.hpp` `McastArgs` | no operand is shared across cores in R1 (operand-reuse check); it is R4's mechanism | Traffic ranking |
| `load_*` | `tilize_helpers_dataflow.hpp`, `local_copy_helpers_dataflow.hpp` | input is already TILE; the copies are DRAM page reads by computed page id, not tilize / local copy | — |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Landing before the destination is in this call | a remote write into `landing_scratch` / `arrival_sem` while the destination still runs the previous call's consumer corrupts it | per-port ready fence; ports send readies first, then wait; counters return to 0 every call |
| Wrong first-hop port for the 2-hop destination | an explicit 2-hop route leaves the snake board and hangs | route with `fabric_set_unicast_route` and assign the far destination to `_first_hop_dir`'s port; assert every destination maps to fwd or bwd |
| `cb_a_landed` write pointer ≠ host slot offset | the sender computes the slot address on the host; the reader reserves in slot order | slot order = processing order on every core, no wrap within a call (`scratch_slot_capacity_tiles ≥ heads_per_core_max * slot_tiles`), CB pointers reset per launch |
| Grow-only scratch reallocation | a larger shape reallocates `landing_scratch`; a trace captured before the grow would replay the freed address | the address is a runtime arg (cache hits re-patch it); document that a trace must be captured after the largest call shape on that mesh |
| Program-cache aliasing | two calls whose per-core arg layouts differ (entry counts, heads per core, ports) share a cached program and hang | `custom_program_hash` over everything in Work Distribution's last row |
| Global semaphore lifetime | `generic_op` keeps only the address | module-level cache holds the handles for the process lifetime |
| DISPATCH-BOUND programs | ~110 cores with per-core args; core sets built one range per core | merge compute core sets into rectangles; per-core RT args only where they differ (head list, slot list); common args for geometry |
| Fused inc on every packet | halves the router rate | one fused inc per entry (last packet) |
| Two senders on one counter | thresholds assume one ordered sender | one counter per head slot |
| HiFi4 + fp32 DEST with bf16 | the helper notes K-accumulator corruption on Wormhole B0 (#38306) | Blackhole only here, one K-block; if the golden PCC fails at HiFi4, compare HiFi3 first |
| Bit-exactness of the exchange | the rows are inputs, not partials | packets copy the stored bytes; no CB or packer touches them before the matmul unpack |
| Header pool (12 per RISC) | a port with 2 destinations and a ring of batch headers | header ring = `PORT_BATCH_CHUNKS`, route rewritten per entry; total ≤ 12 |

### Structural impossibilities

None beyond `feature_spec.py`'s (empty) INVALID.
