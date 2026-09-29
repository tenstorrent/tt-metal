# L1 Ledger: high_bw_all_reduce

Schema: `.claude/references/l1-footprint-discipline.md`.

**Axis naming.**
- Block axes: `tile` (extent `chunk_tiles`), `member` (1 per device) and `group` (1 per device).
- Knobs: `RECV_DEPTH_CHUNKS`, `INPUT_DEPTH_CHUNKS` and `REDUCED_DEPTH_CHUNKS` (reducer); `STAGING_DEPTH_PER_REDUCER` and `FINAL_DEPTH_PER_REDUCER` (port).
- `fp32_dest_acc_en = True` on the reducer compute kernel.

**Inventory decisions (made before sizing):**
1. There is **no** DRAM or L1 scratch between "add" and "forward". Compute packs straight into `cb_reduced`, which is the forward source.
2. Received partials land **in place** in the compute-input CB (`cb_remote_partial` is backed on the fabric landing tensor). There is no landing→CB copy.
3. The relay of finals lands once, in the port's `final_landing`, and both the DRAM write and the next-hop forward read that one slot. There is no second copy.
4. Port cores have **no CBs**.

The only buffer above the theoretical minimum is `port_staging` (the reducer→port on-chip hop). It exists because a fabric send must source the sender's own L1 (`edm_fabric_worker_adapters.hpp:319-349` → `edm_fabric_utils.hpp:68-75`).

## Reducer core (every reducer of every lane, identical on every device)

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_remote_partial` (idx 0; backed on `reducer_landing` shard) | `RECV_DEPTH_CHUNKS · chunk_tiles` | `RECV_DEPTH_CHUNKS · chunk_tiles`: one chunk being consumed plus one granted, in flight from upstream | `{tile: spans → chunk_tiles, member: streams (one upstream partial per chunk), group: streams (—)}` | input dtype: Float16_b / Float32 (wire format, never downcast; fp32: tagged UnpackToDestFp32, consumed only by copy_tile into the SFPU add; audit-2 "under" is deliberate: the fp32 per-step accumulation is in DEST, and the spec allows input-dtype partials) | reader (credit-only `push_back`; the bytes are written by the upstream `port_fwd` over Fabric) | compute | whole kernel (middle/tail); allocated but unused on the head | Cannot alias `cb_local_input`: both are live at the same time as the two add operands. Cannot alias `cb_reduced`: concurrent, because the pipeline writes chunk `k+1` while chunk `k` is read. Its address must be host-known and uniform across devices (a sharded tensor), so it cannot be a program-allocated CB |
| `cb_local_input` (idx 1) | `INPUT_DEPTH_CHUNKS · chunk_tiles` | `2 · chunk_tiles`: chunk `k` consumed while `k+1` is prefetched (stall-shadow fill) | `{tile: spans → chunk_tiles, member: streams (own contribution), group: streams (—)}` | input dtype: Float16_b / Float32 (fp32: tagged UnpackToDestFp32, copy_tile-only consumer) | reader | compute | whole kernel | Concurrent with both other CBs (operand of the same add). A depth of 1 would serialize the DRAM read latency behind the partial wait |
| `cb_reduced` (idx 2) | `REDUCED_DEPTH_CHUNKS · chunk_tiles` | `2 · chunk_tiles`: compute packs `k+1` while the writer copies `k` to the port | `{tile: spans → chunk_tiles, member: streams, group: streams}` | input dtype: Float16_b / Float32 (wire format of the forwarded partial; see `cb_remote_partial`) | compute | writer | whole kernel | Cannot be packed in place into `cb_remote_partial`: compute's input and output are concurrent, the landing slot is credit-returned upstream only after it is popped, and in-place would stall the upstream grant for the whole chunk. Cannot be the port's staging slot directly: pack writes local L1 only |

Reducer total = `(RECV_DEPTH_CHUNKS + INPUT_DEPTH_CHUNKS + REDUCED_DEPTH_CHUNKS) · chunk_tiles · tile_bytes`, all three terms `∝ chunk_bytes`. bf16 as implemented (`CHUNK_BYTES_TARGET = 64 KiB`, `chunk_tiles = 32`): `(2+2+2) · 32 · 2048 = 384 KiB` (landing 128 KiB is a lockstep L1 tensor; 256 KiB program CBs).

## Port core (one per lane; `port_fwd` = RISCV_0, `port_bwd` = RISCV_1). Raw L1 in the per-call `port_scratch` tensor, no CBs

| Buffer | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|--------|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `port_control` (4 arrays × `W` words, `CONTROL_WORD_STRIDE` = 16 B stride, padded up to one tile) | `4 · W` 16-byte words, rounded up to `tile_bytes` (constant in blocks; scales with `W`) | all `4W` words are live together (one counter per reducer per stream) | `{tile: streams (counts chunks), member: streams, group: streams}` | uint32 counters | `staged[r]` / `final_ready[r]`: reducer writer; `partial_granted[r]`: reducer reader; `final_freed[r]` (R1 refinement): `port_bwd` (plain L1 store, same core). Each word has exactly one writer. Zeroed by `port_bwd` before `go` | `port_fwd` (`staged`, `final_freed`), `port_bwd` (`final_ready`, `partial_granted`) | whole invocation | Cannot live in program semaphores: `3W` counters (12 at `W = 4`, 24 at the design's `W = 8`) plus the 3 program semaphores and the fabric connection's own semaphores exceed the 16-per-core cap (`semaphore.hpp:16`). Cannot be `GlobalSemaphore`s: only on-chip peers touch them, so per-call memory is safe and avoids `3W` persistent allocations |
| `port_staging` ring | `W · STAGING_DEPTH_PER_REDUCER · chunk_tiles` | up to `W · chunk_tiles`: each reducer has at most one chunk staged | `{tile: spans → chunk_tiles × W slots, member: streams (partial), group: streams}` | input dtype (Float16_b / Float32) | reducer writers (slot `c mod (W·depth)` is written only by reducer `c mod W`) | `port_fwd` | whole invocation; unused on the tail | Cannot alias `port_final_landing`: both streams are concurrent (partials flow toward `p+1` while finals flow toward `p−1`). This buffer is the single above-minimum buffer, forced by the rule that the fabric source must be local L1 |
| `port_final_landing` ring | `W · FINAL_DEPTH_PER_REDUCER · chunk_tiles` | up to `W · chunk_tiles` granted slots in flight | `{tile: spans → chunk_tiles × W slots, member: streams (final), group: streams}` | input dtype (Float16_b / Float32) | the downstream `port_bwd` via Fabric (tail: local reducer writers) | `port_bwd` (DRAM write + forward from the same slot) | whole invocation | Concurrent with staging (above). A single slot serves both the DRAM write and the forward, so there is no second relay copy |
| Fabric packet headers | `PacketHeaderPool` (firmware-reserved) | — | `{tile: streams, member: streams, group: streams}` | header | `port_fwd` / `port_bwd` | EDM | kernel | Not allocated by the op |

Port total = `align_up(3W · 16 B, tile_bytes) + W · (STAGING_DEPTH_PER_REDUCER + FINAL_DEPTH_PER_REDUCER) · chunk_bytes`. The ring terms scale with both `W` and `chunk_bytes`. bf16 as implemented: `2 KiB + 4 · 2 · 64 KiB = 514 KiB`.

## Persistent (cross-invocation) L1

The op also keeps `1 + 3 · REDUCERS_PER_LANE` `GlobalSemaphore`s (13 at `REDUCERS_PER_LANE = 4`): `gsem_partial_arrival` (on reducer cores) plus, **per reducer index**, `gsem_partial_credit_r`, `gsem_final_arrival_r` and `gsem_final_credit_r` (Refinement 1: the port kernels serve reducers independently, so every port-to-port counter is per reducer). Each is one 16 B-aligned word per core, cached per `(mesh_device, cluster_axis, route kind line/ring, num_lanes)`. They are negligible and constant. The `final_freed` program semaphore was replaced by the `final_freed[r]` control words; a new `final_egress` program semaphore on reducers gates tail writes into the final-landing ring.

## Symbol table

| Symbol | Bound | Predicate / source |
|--------|-------|--------------------|
| `tile_bytes` | 2048 (bf16), 4096 (fp32) | dtype ∈ SUPPORTED |
| `packet_tiles` | `max_payload // tile_bytes`, ≥ 1 | `ttnn.get_tt_fabric_max_payload_size_bytes()` (4352 by default, caller-controlled) |
| `chunk_tiles` | `max(packet_tiles, align_down(CHUNK_BYTES_TARGET // tile_bytes, packet_tiles))`: 32 (bf16), 16 (fp32) | host constant `CHUNK_BYTES_TARGET = 65536` |
| `chunk_bytes` | `chunk_tiles · tile_bytes` ≤ 64 KiB | same |
| `W` | `1 ≤ W ≤ REDUCERS_PER_LANE = 4` | `W = min(4, max_ℓ num_blocks_this_lane)` |
| `num_lanes` | `1 ≤ num_lanes ≤ usable_links` | validated; `ValueError` otherwise |
| depths | 2, 2, 2, 1, 1 | host constants |
| `MAX_DATA_HEADERS` | 8 (+1 credit header) per port connection | host constant, CT arg to both port kernels; ≤ the per-RISC `PacketHeaderPool` budget (firmware L1, not op L1) |
| `tensor_tiles` | unbounded, but **never in a capacity expression** | only in trip counts |

**Per-core footprint (bf16, as implemented):** reducer 384 KiB, port about 514 KiB. Both stay well inside Wormhole's (~1.3 MiB usable) and Blackhole's L1.

**Total per device:** `num_lanes · (W · reducer_total + port_total)`. That is 2 lanes × (4 × 384 KiB + 514 KiB), spread over 10 cores. No capacity depends on the tensor size.

## Data-movement budget (R1 `chain_line`, per device, `S` = per-device tensor bytes)

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| input | 1 (read by the owning reducer) | each tile belongs to exactly one (lane, reducer); the own contribution is read once and consumed from L1 | — |
| partial stream | 0 | lands in the reducer's L1 in place (credit-gated ring) | Fabric: `S` per device toward `p+1` (tail: 0). On-chip: `S` router→reducer landing, and `S` reducer→port staging |
| final stream | 0 (relay) | lands in the port's L1; one slot feeds both the DRAM write and the forward | Fabric: `S` per device toward `p−1` (head: 0). On-chip: `S` router→port landing |
| output | 1 (written by the port) | each final tile is written once, from the relay slot | — |
| credits | 0 | 4 B atomics, one per chunk per hop | negligible (`S / 64 KiB` atomics) |

**Totals per device (middle position):**
- DRAM: `2S` (the minimum).
- Fabric egress: `S` toward `p+1` + `S` toward `p−1`. Every link direction of every hop carries `S / num_lanes` per lane, which is the line lower bound.
- On-chip NoC L1↔L1: `3S`.

> Cheapest-traffic split considered: R3 `rotated_chain_ring` — `(G−1)/G · S` per link direction instead of `S`, with the same DRAM (`2S`). It applies only to `topology=Ring`. Implemented: R1 `chain_line`, which is the cheapest split for `topology=Linear`: it meets both the DRAM minimum and the line's Fabric lower bound. R3 is deferred because Ring is outside Phase 0 SUPPORTED. The structure keeps it reachable: every kernel iterates `num_slices` with a per-slice role and chunk range (Phase 0 `num_slices = 1`), and the port kernels already own both neighbour connections.

## Implementation deltas (ttnn-implementer)

Knob values as implemented (single source: `high_bw_all_reduce_program_descriptor.py`):

| Knob | Design Phase 0 | Implemented | Why (measured, BH 2x2, G=2) |
|------|----------------|-------------|------------------------------|
| `CHUNK_BYTES_TARGET` | 32 KiB | **64 KiB** (`chunk_tiles` = 32 bf16) | port send loop is per-chunk-handshake/issue bound; +7-9% at 2 links |
| `REDUCERS_PER_LANE` (W cap) | 8 | **4** | keeps the port rings at `W·(SD+FD)·chunk_bytes = 4·2·64 KiB = 512 KiB`; 64 KiB × W=8 (1 MiB rings, lockstep-allocated on every core) collided with the reducers' static CB region |
| depths | 2,2,2,1,1 | unchanged | `FINAL_DEPTH_PER_REDUCER=2` measured neutral |

Resulting per-core footprint (bf16): reducer = `(2+2+2)·32·2048 = 384 KiB` (landing 128 KiB is a
lockstep L1 tensor + 256 KiB of program CBs); port = `4W·16 B` control (padded to one tile) +
`W·(1+1)·64 KiB = 512 KiB`. No capacity depends on the tensor size.

Counter units: `gsem_partial_arrival` and `gsem_final_arrival` count **chunks**, not packets —
packets `0..ppc-2` are plain unicast writes and only the last packet of a chunk is a flushed fused
write+atomic-inc (perf lamp "per-packet fused atomic inc": BH golden shows fused packets at
~5.9 GB/s vs ~38 GB/s plain). Per-invocation totals are `num_blocks_this_core` / `num_blocks_this_lane`.

Data-movement delta: on the **tail** (p = G−1) the reducer writer writes its chunk's valid output
pages to DRAM straight from `cb_reduced` (the chunk is already final); the tail port only relays
over Fabric. Non-tail devices write output from the port's final landing slot as designed. DRAM
crossings per tensor are unchanged (input 1, output 1); the tail's port-side relay DRAM read of
the final landing is removed from the port core.

## Verifier notes (ledger currency pass)

- Main-body sizes/symbols above were updated to the implemented knob values (64 KiB chunk, `W ≤ 4`); the
  design's 32 KiB / `W = 8` numbers survive only in `op_design.md` and the deltas table.
- The compute kernel now consumes each CB in whole-chunk windows (upfront wait / pop-at-end, one
  reserve/push of `chunk_tiles` on `cb_reduced`). No capacity changed: every CB was already an exact
  multiple of `chunk_tiles`, and the live set of each is still `depth · chunk_tiles`.
- `CONTROL_WORD_STRIDE` and `MAX_DATA_HEADERS` are now single-sourced on the host and passed as CT args
  (previously restated as literals in both port kernels).

## Refinement 1 deltas (ring + snake)

- No CB or ring capacity changed. The R2 snake line and the R3 ring reuse every buffer at the same
  size; per-chunk roles choose which ring a block uses (non-tail → staging, tail → final landing).
- `port_control` grows from 3 to 4 arrays (`final_freed[r]`): +`W · 16 B`, still inside one tile.
- Program semaphores per core: `go`, `egress_credit`, `final_egress` (3) plus the fabric connections'.
- Data-movement budget: DRAM crossings unchanged (input read once by reducers, output written once per
  device by the tail reducer or the port). Fabric per link direction: R2 `S` on each snake edge; R3
  `(G−1)/G · S` on every ring edge, including the closing edge.

### Refinement 2 delta (float32)

Every CB and L1 ring carries the input dtype. For fp32, `tile_bytes = 4096` and `chunk_tiles = 16`, so
`chunk_bytes = 64 KiB` is unchanged and so is every capacity (reducer 384 KiB, port ~514 KiB). No
buffer was added. `cb_remote_partial` and `cb_local_input` are tagged `UnpackToDestFp32` on the fp32
path only. That is legal because their sole consumer is the chain's `copy_tile` into DEST, feeding
the SFPU `AddBinary` or the head copy. DRAM crossings per tensor are unchanged: one read of the input
and one write of the output per device.

### Refinement 3 delta (credit coalescing + one shared per-call scratch)

- **One lockstep scratch tensor** (`op_scratch`, HEIGHT_SHARDED over every op core) replaces
  `reducer_landing` + `port_scratch` + the reducers' two program CBs. An L1 allocation reserves its
  address range on every core, so the old layout cost `landing + port + program CBs` on every
  core; now reducers overlay `cb_remote_partial | cb_local_input | cb_reduced` (via
  `cb_descriptor_from_sharded_tensor(address_offset=…)`) and ports overlay
  `control | staging | final_landing` on the same shard. The footprint is
  `max(reducer_total, port_total)`, not their sum. This resolves the collision noted above by
  re-placing the port scratch, not by shrinking blocks: chunks stay 64 KiB and W stays 4.
- **Depths derive from `credit_batch`** (host `CREDIT_BATCH_CHUNKS = 4`, path-gated). The landing ring
  is `RECV_EFFECTIVE_DEPTH (2) + credit_batch − 1` chunks per reducer; the final-landing ring is
  `FINAL_EFFECTIVE_DEPTH (1) + credit_batch − 1` chunks per reducer.
  - At batch 4: reducer `(5 + 2 + 2) · 64 KiB = 576 KiB`; port `2 KiB + 4 · (1 + 4) · 64 KiB = 1282 KiB`.
    The shard is 1 282 KiB of the 1 427 KiB BH bank.
  - At batch 1 (the 2-link G > 2 chains): reducer 384 KiB, port 514 KiB, shard 514 KiB (less than the
    Phase 0 footprint of 642 KiB lockstep + 256 KiB CBs).
- **L1-aware clamp**: the host lowers `credit_batch` until the shard fits
  `get_memory_view(L1).largest_contiguous_bytes_free_per_bank`, so a caller with resident L1 tensors
  gets shallower rings, not an OOM. The global semaphores are created before the scratch, so they
  never split the free block the scratch needs; before this fix the unit suites OOM'd at 870 KiB
  largest-free.
- **Data-movement budget**: unchanged. DRAM input is read once and output written once per device;
  Fabric carries `S` per link direction. Credit packets per link direction drop from
  `S / 64 KiB` to `S / 256 KiB`.

### Refinement 4 delta (split ports + lane-per-row placement)

- **Split ports** (`SPLIT_PORTS = 1`, path-gated to `G ≥ 3`, i.e. the None snake and ring): each lane
  gets a second port core, so there is one `port_fwd` core and one `port_bwd` core per lane (+1 core per
  lane: 6 per lane at `W = 4`). Both port cores overlay the same `control | staging | final_landing`
  layout on the shared scratch shard (lockstep tensor: max, not sum).
  - The `port_fwd` core uses `staged[]`, `final_freed[]` and the staging ring.
  - The `port_bwd` core uses `final_ready[]`, `partial_granted[]`, the new `drained[]` and the
    final-landing ring.
  - No capacity changed. The unused half of each port core's shard is the price of the uniform
    address. It is still inside the 1 282 KiB shard, because the port rings already set the shard size.
- **`port_control` grows from 4 to 5 arrays** (`drained[r]`, written by the new `port_drain` kernel on
  the `port_bwd` core's RISCV_0): +`W · 16 B`, still inside one tile. `port_bwd` zeroes both port cores'
  arrays before `go` (remote inline writes for the `port_fwd` core's words).
- **The final slot has two readers**, but no second copy: `port_bwd` forwards it over Fabric and
  `port_drain` writes it to output DRAM, both from the same slot. The slot is freed once both are done
  (`freed_k[r] < relayed` and `drained[r] > freed_k[r]`).
- **Data-movement budget**: unchanged in bytes. DRAM input is read once and output written once per
  device, and Fabric carries `S` (line) or `(G−1)/G · S` (ring) per link direction. The output DRAM
  write now leaves the `port_bwd` core on NoC1 (`port_drain`) instead of NoC0, next to the Fabric sends.
- **Placement** (`LANE_ROW_STRIDE = 1`): lane `l` occupies grid row `l`. No L1 change.
- **Bank-run chunk layout** (`BANK_RUN_LAYOUT`, `kernels/high_bw_all_reduce_chunk_io.hpp`): a live
  knob, parked at 0. At 1 it permutes tiles inside a chunk (bank-major) and makes every DRAM transfer
  one `run_tiles`-page burst per bank. No capacity change.
