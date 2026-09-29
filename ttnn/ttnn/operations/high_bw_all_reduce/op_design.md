# Operation Design: high_bw_all_reduce

## Overview

| Field | Value |
|-------|-------|
| Classification | CCL (multi-device, Fabric 2D) + fused eltwise reduction |
| Goal | Every device of a collective group receives the elementwise SUM of all group members' tensors. Target: 8–128 MB per device, Fabric-bandwidth-bound. Every requested link must run near link rate in **both** directions, and the adds must overlap the transport. |
| Math | `out[d] = Σ_{d' ∈ group(d)} in[d']` elementwise, over physical tile pages, with fp32 accumulation per step |
| Mode | Derivative (fabric wiring from `tests/ttnn/unit_tests/operations/debug/test_generic_op.py:137-439`; the chain scheme is first-principles) |
| References | `.claude/references/blocking-model.md`, `.claude/references/l1-footprint-discipline.md`, `ttnn/ttnn/operations/examples/master.md` (`double_buffer`, `tensix_all_reduce_compute`, `eltwise_l1_vs_dest_accumulate`, `noc_placement`, `tensix_all_reduce`), `ttnn/cpp/ttnn-nanobind/fabric.cpp`, `tt_metal/fabric/hw/inc/mesh/api.h`, `tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp`, `eval/golden_tests/high_bw_all_reduce/feature_spec.py` |

**Dispatch contract.** Each call makes exactly **one** `ttnn.generic_op(io_tensors, MeshProgramDescriptor)` dispatch. The descriptor holds one `ProgramDescriptor` per mesh coordinate, covering every device of every group. There is no second op and no host round-trip. Host-side allocation of the output and the per-call L1 scratch tensors is output setup, not a dispatch.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` (mesh) | yes | TILE, DRAM interleaved, rank ≥ 2; identical per-device shape and dtype | — | RT: buffer address |
| `cluster_axis` | `int \| None` | yes (keyword) | `0`, `1`, `None` (Phase 0: `0`, `1`) | — | host: selects group and path → per-device role CT args + neighbour RT args |
| `topology` | `Topology` (`from ttnn.operations.ccl import Topology`) | no | `Linear`, `Ring` (Phase 0: `Linear`) | `Topology.Linear` | host: selects regime |
| `num_links` | `int \| None` | no | `1 ≤ n ≤ usable_links`, where `usable_links = min` over every hop and direction of the route of `len(ttnn.get_forwarding_link_indices(src, dst))` | `None` → `usable_links` | host: `num_lanes` |
| `memory_config` | `ttnn.MemoryConfig \| None` | no | DRAM interleaved (Phase 0) | `ttnn.DRAM_MEMORY_CONFIG` | host: output allocation |

**Validation order.** `validate()` is the first line of the entry point.

1. **Registry gate** (`UnsupportedAxisValue` / `ExcludedCell`). Checks the axes `dtype`, `layout`, `alignment`, `cluster_axis`, `topology` and `num_links`. A `num_links` value of `None` is the default: it is always accepted and skips the SUPPORTED check. It resolves later to `usable_links`.
2. **Caller errors** (`ValueError`, not a support refusal):
   - Input is not DRAM interleaved, or rank < 2.
   - Explicit `num_links` ≤ 0 or > `usable_links`.
   - `ttnn.get_fabric_config()` is not a `FABRIC_2D*` config.
   - Ring along an axis whose fabric config does not wrap that axis (`TORUS_Y` wraps axis 0, `TORUS_X` wraps axis 1, `TORUS_XY` wraps both).
   - Ring over `None` when neither mesh dimension is even.
3. **Unsupported `memory_config`**: anything other than DRAM interleaved raises `UnsupportedAxisValue("memory_config", …)`.

The op reads the fabric config. It never sets it.

**Alignment tagger** (the only entry in `INPUT_TAGGERS`). It reads the per-device `inputs[0]`, with `H, W = shape[-2], shape[-1]`:
- `W % 32 != 0` → `"w_non_aligned"`
- else `H % 32 != 0` → `"h_non_aligned"`
- else → `"tile_aligned"`

**Phase 0 SUPPORTED**

| Axis | Values |
|------|--------|
| dtype | `bfloat16` |
| layout | `TILE` |
| alignment | `tile_aligned` |
| cluster_axis | `0`, `1` |
| topology | `Linear` |
| num_links | `1`, `2` |

EXCLUSIONS is `[]`.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | Any rank ≥ 2, identical on every device. `tensor_tiles = prod(shape[:-2]) · ceil(H/32) · ceil(W/32)` (per image, `ceil`) |
| Dtype | bfloat16 (Phase 0). TARGET adds float32 |
| Layout | TILE |
| Memory | DRAM interleaved |

### Output

| Property | Value |
|----------|-------|
| Shape | The input's TensorSpec: same logical shape and same padded shape. It is allocated with `ttnn.allocate_tensor_on_device(input.spec-with-memory_config, mesh_device)` |
| Dtype | Same as input |
| Layout | TILE |
| Memory | `memory_config` (default DRAM interleaved), on every mesh device |

The op sums physical tile pages, pad lanes included. Pad lanes of the output hold the sum of the input pads and are never part of the logical view, so no padding leaks into the logical output. Non-aligned shapes therefore need no masking. Widening `alignment` is a SUPPORTED change plus tests (see Regimes).

---

## Blocking Model

**The scheme in one sentence.** Each group is an ordered line of one-hop neighbours, positions `p = 0..G-1`.
- The per-device tile stream is cut into chunks. Chunks are spread over `num_lanes` links, and within a lane over `W` reducer cores.
- Every chunk is **reduced hop by hop toward `p = G-1`**: each device adds its own contribution in fp32 DEST and forwards a bf16 partial.
- The completed sum is then **relayed hop by hop back toward `p = 0`**. Every device writes it to its output DRAM on the way.
- Chunks are pipelined, so each link direction carries the stream at once. The adds for chunk `k` overlap the transport of chunks `k±1..`.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `group` (the mesh axis ≠ `cluster_axis`; for `None`, a single group) | independent: groups are disjoint device sets with no data dependence | `groups_per_device` | 1. Each device belongs to exactly one group; this is fixed by the mesh and not a choice | mesh shape (`mesh_device.shape`) | one group per device set; all groups run concurrently in the same dispatch | — (fixed by mesh) |
| `member` (position `p` in the group, extent `G`) | **dependent**: the sum spans all `G` contributions | `contributions_per_device` | 1. Each device physically holds exactly one contribution; the combine is the Fabric chain, not a blocking choice | group path builder (host `_group_path(cluster_axis, topology)`) | one member per device; combine = hop-by-hop chain (reduce toward `p=G-1`, relay toward `p=0`) | scheme-change: Ring = rotated chains (R3); snake order (R2) is a path change |
| `tile` (flattened physical tile pages `0..tensor_tiles-1`; leading dims, H-tiles and W-tiles fold into it because the op is elementwise and page order is irrelevant) | independent: elementwise, no cross-tile dependence | `chunk_tiles` (block along `tile`; the unit of DRAM read, add, forward, relay and credit) | `chunk_tiles = align_down(CHUNK_BYTES_TARGET // tile_bytes, packet_tiles)`. bf16: 32 KiB / 2 KiB = **16 tiles** (8 packets of 2 tiles) | host constant `CHUNK_BYTES_TARGET = 32768`. `chunk_tiles` is derived once on host and passed as a CT arg to every kernel | two-level split: **lane** `ℓ ∈ [0, num_lanes)` owns a contiguous `ceil` range of tiles; within a lane, lane-local chunk `c` is owned by reducer `r = c mod W` | knob-turn (bigger/smaller chunk, more reducers, more lanes) |
| `tile` → `lane` (sub-assignment) | independent | `num_lanes` | `num_links` resolved (1 or 2) | host `num_lanes = num_links or usable_links` | lane ℓ = one link per hop in each direction, with its own port core and reducer set | knob-turn |
| `tile` → `reducer` (sub-assignment) | independent | `W = min(REDUCERS_PER_LANE, num_chunks_lane_max)` | `REDUCERS_PER_LANE = 8` | host constant `REDUCERS_PER_LANE`. `W` is derived once and is identical on every device, because senders address receivers by `c mod W` | chunk `c` of lane ℓ → reducer `c mod W` of lane ℓ, **on every device** | knob-turn |
| `tile` → `packet` (inside a chunk) | independent | `packet_tiles` | `ttnn.get_tt_fabric_max_payload_size_bytes() // tile_bytes`. bf16 with the default 4352 B payload → **2 tiles** | derived on host from the runtime fabric payload size; the op never sets it | the transport quantum of a fabric write. Not a block | — (mechanism cap) |

**Why 32 KiB and not "the whole per-core assignment".**
- L1 would allow chunks several times larger: see `l1_ledger.md`, reducer total 192 KiB at Phase 0.
- The bound is structural. A hop forwards a chunk only after that chunk is reduced, so the pipeline fill costs about `2(G-1) × chunk_time`.
- A chunk equal to the whole assignment degenerates the chain into store-and-forward: `G×` slower.
- 32 KiB takes 16 tiles per DRAM barrier, which is well past the catalog's 4–8-tile batching sweet spot (`double_buffer`). It is 8 fabric packets per credit, and 1 credit message per 32 KiB.
- This is an overlap-lamp decision (see Perf lamps), not an L1 limit.

**Stage re-run of the table (the data the scheme itself produces).**

| Stage data | Axes | Assignment | Why |
|------------|------|------------|-----|
| partial stream (bf16 partial sums moving toward `p=G-1`) | `tile` (independent) | same lane × reducer split as the input. Reducer `r` on device `p+1` receives exactly the chunks reducer `r` on device `p` produced | the reduction work is spread over `num_lanes × W` cores per device; no single core computes for the group |
| final stream (bf16 sums moving toward `p=0`) | `tile` (independent) | one **port core per lane** relays the lane's finals: DRAM write plus forward | this stage computes nothing; it is pure data movement at one link's rate. The single-core-per-lane assignment is forced by the EDM **one worker channel per link per direction** cap (Mechanism caps). It is a communication role, and it idles nobody: reducers keep reducing while the port relays |
| credit stream (slot grants moving against the data) | `chunk` index | forwarded by the port kernel that owns the connection toward the grant's recipient | 4 B atomic incs, interleaved on existing connections |

### Buffer-depth knobs

| CB / buffer | Depth knob | Phase 0 value | What the depth buys |
|-------------|------------|---------------|---------------------|
| `cb_remote_partial` (reducer landing ring, fabric-written) | `RECV_DEPTH_CHUNKS` | 2 | Upstream may send chunk `k+1` to this reducer while compute consumes chunk `k`. With `W` reducers per lane, the lane has `W·2` chunks (512 KiB) in flight, far above one link's bandwidth-delay product |
| `cb_local_input` | `INPUT_DEPTH_CHUNKS` | 2 | DRAM prefetch of the reducer's next own-input chunk while it waits for the partial (stall-shadow fill) |
| `cb_reduced` | `REDUCED_DEPTH_CHUNKS` | 2 | Compute packs chunk `k+1` while the writer copies chunk `k` to the port |
| `port_staging` ring (port L1, reducer-written) | `STAGING_DEPTH_PER_REDUCER` | 1 → ring = `W` chunks | Each reducer can have one chunk staged while the port sends others. The reducer emits one chunk every `W` chunk-times, so 1 suffices; `cb_reduced` provides that reducer's own double buffer |
| `port_final_landing` ring (port L1, fabric- or reducer-written) | `FINAL_DEPTH_PER_REDUCER` | 1 → ring = `W` chunks | `W` chunks of finals in flight per lane per hop; enough to cover the credit round trip |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Fabric packet payload (`ttnn.get_tt_fabric_max_payload_size_bytes()`, default 4352 B, set by the caller's fabric config) | `packet_tiles · tile_bytes ≤ max_payload` | `packet_tiles = max_payload // tile_bytes` (assert ≥ 1). `chunk_tiles = max(packet_tiles, align_down(chunk_tiles, packet_tiles))` | An oversized packet overruns the EDM channel slot, corrupting the neighbour's traffic or hanging the router |
| EDM worker sender channel: exactly **one** worker connection per (link, direction) per device (`tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp:127-133`, "likely hang") | fabric-connected kernels per (device, link, direction) | exactly one port kernel per (lane, direction): `port_fwd` on RISCV_0 and `port_bwd` on RISCV_1 of one port core per lane. `num_lanes ≤ usable_links`, else `ValueError` | Two workers on one channel is undefined behaviour and hangs |
| Receiver addressing: senders address reducer `c mod W` and slot `(c div W) mod RECV_DEPTH_CHUNKS` on the neighbour | `W`, `chunk_tiles`, depths, lane ranges | all are derived on host from the same inputs and are identical on every device of the mesh | Mismatched `W` or ranges on two devices → chunks land in the wrong reducer or slot; wrong sums with no hang |
| 32-bit NoC atomic counters | per-invocation cumulative packet count per reducer | ≤ `128 MiB / 4 KiB = 32768 ≪ 2³²`; no clamp needed | — |
| Program semaphores (16 per core, `tt_metal/impl/buffers/semaphore.hpp:16`) | per-reducer local counters | per-reducer counters live in the port's per-call L1 control array, not in program semaphores. Program semaphores are used only for `go`, `egress_credit` and `final_freed` | More than 16 semaphores fails program build |
| Port-core L1 | `W · (STAGING_DEPTH_PER_REDUCER + FINAL_DEPTH_PER_REDUCER) · chunk_bytes` | host assert against the core's free L1 (bf16 Phase 0: 512 KiB) | Allocation failure at `allocate_tensor_on_device` |
| FPU binary add reads SrcA/SrcB as tf32 (`tilize_helpers.hpp:50-52`) | dtype of the add path | bf16: FPU `add` is exact into fp32 DEST. fp32 refinement: SFPU `AddBinary` with `UnpackToDestFp32` on both inputs | fp32 inputs silently truncated to tf32, which violates "fp32 end-to-end" |

### Regimes

Named boundaries used below: **DRAM**, where the minimum is that each input crosses once and each output crosses once per device; and **Fabric**, where the line lower bound is that each cut must carry the sum of one side in each direction, i.e. `S` bytes per link direction for a line and `(G-1)/G · S` for a bidirectional ring.

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **R1 `chain_line`** — reduce hop by hop toward `p=G-1`, relay finals toward `p=0`, pipelined per chunk | **built** | `topology == Linear and cluster_axis in (0, 1) and G ≥ 2` | `chunk_tiles × 1 × 1` over `tile`; per reducer, `ceil((num_chunks_lane − r)/W)` chunks | DRAM: minimum (input read once by reducers, output written once by the port). Fabric: minimum, `S` per link direction on every hop, split across `num_lanes`. Partials travel in bf16, as the spec allows. Above minimum: one on-chip L1→L1 hop per chunk (reducer → port staging), forced by "fabric source must be the sender's own L1" | Per chunk: one DRAM barrier (reader), one credit round trip, one staging handoff (2 local incs), and one relay DRAM-write barrier. Compute init is once per kernel, not per chunk |
| **R1-solo** (degenerate, `G == 1`) | **built** | `G == 1` | same | DRAM minimum, no Fabric. Reducers copy, deliver to the local port, and the port writes. Pruned by the golden suite, but reachable by callers on 1-wide axes | same, minus the fabric terms |
| **R2 `chain_snake_line`** (`cluster_axis=None`, Linear) | **built (Refinement 1)**. **Knob-turn**: the kernels already take the path as an ordered neighbour list, and only `_group_path` changes to a row- or column-snake whose every edge is one hop | `topology == Linear and cluster_axis is None` | same | Fabric: `S` per direction on the snake's edges; the other mesh links are idle (spec-mandated snake). DRAM minimum | same. Fill grows as `2(G_mesh − 1)` hops (the fill lamp; see R5) |
| **R3 `rotated_chain_ring`** (Ring on an axis with a wrap link, or a snake cycle for `None`) | **built (Refinement 1)** for the `None` snake cycle; per-axis torus rings are in `EXCLUSIONS` (no torus cluster to verify). **Scheme-change** (role becomes per slice). Reachable because every kernel iterates `num_slices` with a per-slice role (`head`/`middle`/`tail`) and a per-slice chunk range, and Phase 0 is `num_slices = 1`. Ring = `G` slices; slice `j`'s line starts at position `j` and ends at `j−1 (mod G)`, so every link, including the wrap, carries partials one way and finals the other | `topology == Ring` (axis with wrap, or `None` with an even mesh dim) | `chunk_tiles` over each slice's tile range | Fabric: `(G−1)/G · S` per link direction. This equals the bidirectional reduce-scatter + all-gather bound and uses **both** ring directions (the spec's preference). DRAM minimum | same, plus slice boundaries (≤ `G` per lane) |
| R4 split-halves bidirectional chain (half the tensor reduces toward each end) | rejected; superseded by R1 | — | — | Same Fabric bytes (`S` per direction) and same DRAM as R1, and the same per-device add load (max `S`). It costs two data streams per port kernel with arbitration, for no traffic gain | — |
| R5 line reduce-scatter + all-gather (bidirectional) | deferred: its only advantage over R1 is a pipeline fill of about `G` hops instead of `2G` hops at equal Fabric and DRAM bytes. That matters only for long snake lines (Galaxy `None`, `G = 32`), which R2 does not yet serve. Reachable because it is the same per-hop primitives (credit-gated neighbour send, add-or-forward) driven by a different per-slice role table, which is the R3 generalisation | `cluster_axis is None and G ≥ 16` (when built) | `chunk_tiles` per slice | Equal to R1 | same |
| R6 Fabric-native line multicast of finals (routers write every downstream device) | deferred: R1's relay already meets the Fabric minimum (`S` per direction), so multicast would save only the port's on-chip L1 relay and final credits, not link bytes. It is inapplicable to snake corners (R2/R3). Reachable because it replaces `port_bwd`'s relay loop only | `cluster_axis in (0,1)` | `chunk_tiles` | Fabric equal; on-chip −1 L1 hop per chunk per device | fewer multicast headers |
| R7 all-gather then local reduce | rejected; superseded by R1 | — | — | Fabric `(G−1)·S` per direction, about `(G−1)×` R1; DRAM `G·S` read | — |
| R8 DRAM landing: bulk transfer into a DRAM scratch, then reduce | rejected (dead end); superseded by R1's credit-gated L1 landing | — | — | DRAM `+2S` per device (landing write + read), and the relay adds `+S` more. Transfer-then-reduce does not overlap; no good scheme passes through it | — |
| R9 fp32 partials on the wire for bf16 inputs | rejected; superseded by R1 (bf16 wire, fp32 per-step accumulation, as the spec allows) | — | — | Fabric `2S` per direction, which halves the achievable rate | — |
| R10 one core per lane doing port + reduce (no on-chip hop) | rejected; superseded by R1's `W` reducers per lane | — | — | Saves the on-chip L1 hop, but puts DRAM read, FPU add and fabric send for a whole link on one core. That contradicts "spread the payload over many worker cores per link", and one core's add rate caps the link | — |

**Regime selection (host, exact):**

```
groups, is_ring = route(mesh_shape, cluster_axis, topology)   # host, program descriptor
G = len(groups[0])
if G == 1                                          -> R1-solo
elif not is_ring and cluster_axis in (0,1)         -> R1 (axis line)
elif not is_ring                                   -> R2 (row-snake line; 1-D mesh: the axis line)
else                                               -> R3 (num_slices = G; a 2-device ring runs as the line)
```

**Refinement 1 implementation notes (R2/R3).**
- Roles are per block, not per device: `kernels/high_bw_all_reduce_roles.hpp` (single source).
  Block `k` of reducer `r` belongs to slice `(r + W·seg) mod G`, where reducer `r`'s block stream
  is cut into `G / gcd(G, W)` contiguous segments. At `W = G = 4` (2×2 snake ring) every reducer
  serves exactly one slice.
- Both port kernels serve reducers **independently** (round-robin by readiness, in order per
  reducer), and every port-to-port counter is per reducer (`gsem_partial_credit_r`,
  `gsem_final_arrival_r`, `gsem_final_credit_r`; units = that reducer's block / receive ordinal).
  A single global lane order couples the rotated chains around the ring. Measured before this
  change: 32 MB None-Ring took 6.6 ms, against 1.5 ms after. On the partial side each hop waits on
  the previous device's previous chunk. On the final side a local tail's slot reuse queues behind
  finals still being relayed from far tails.
- Staging and landing slots are keyed by per-reducer ordinals (staging ordinal = forwarded count =
  downstream receive ordinal), so head/tail blocks that skip a ring leave no holes.
- `Ring` over `None` with no even mesh dimension, or `Ring` along an axis the fabric config does not
  wrap, raises `ValueError` from `route()`.

Required regime-pinned tests: R1 with `G = 2` (head+tail only, on a 2x2) and R1-solo. R1 with `G ≥ 3` (middle role) runs on T3K / Galaxy; the acceptance test runs it wherever the mesh axis has ≥ 3 devices.

### Traffic ranking

`S` is the per-device tensor size. Link loads are per link direction, summed over lanes.

| Rank | Candidate | Fabric per link direction | DRAM per device | On-chip | Verdict |
|------|-----------|---------------------------|-----------------|---------|---------|
| 1 | R3 rotated-chain ring (Ring only) | `(G−1)/G · S` | `2S` (minimum) | `3S` | cheapest where a ring exists; deferred (not Phase 0) |
| 2 | **R1 chain line** / R4 halves / R5 RS+AG line | `S` (line lower bound) | `2S` (minimum) | R1 `3S` | **R1 chosen**. For a line, R1, R4 and R5 tie on bytes; R1 has the simplest port arbitration, one data stream per port kernel |
| 3 | R9 fp32 wire | `2S` | `2S` | `6S` | rejected |
| 4 | R8 DRAM landing | `S` | `4S–5S` | — | rejected |
| 5 | R7 all-gather + reduce | `(G−1)S` | `(G+1)S` | — | rejected |

The dependent axis (`member`) is split across devices by definition; its combine *is* the Fabric chain.

Inside a device, the independent `tile` axis is spread over `num_lanes × W` reducers. No operand is reused across that split: each reducer reads only its own chunks. The operand-reuse check therefore finds no broadcast candidate, and no multicast row is needed for inputs.

**Stall-shadow check**

| Waiting stage | Waits on | Work scheduled into the stall |
|---------------|----------|-------------------------------|
| Reducer | the remote partial | The reader issues the DRAM read of this reducer's own input for chunk `k` (and `k+1`, depth 2) **before** it waits on the arrival counter. The own contribution is independent of the upstream partial, so it is always resident when the partial lands |
| Chain fill on device `p` | the first partial, `p` hops away | The same prefetch fills `cb_local_input` to depth |
| `port_bwd` on the head | finals (a full round trip) | Nothing: its only work depends on the finals |
| `port_fwd` / `port_bwd` waiting on anything | — | Both service **credit forwarding** inside every wait loop. This is also the deadlock-freedom rule: a port blocked on a remote credit must still forward the credits its neighbour is waiting for |
| Floating-point order | — | No reorder changes FP order beyond R1's fixed chain order |

### Block schedule

The logical schedule for lane ℓ, reducer `r`, on the device at position `p` of a line of `G`. Roles:
- `head` = `p == 0`: no upstream partial; compute copies.
- `tail` = `p == G-1`: the sum is final; the reducer delivers to the local port's final landing.
- `middle` = otherwise.

```cpp
// reducer core (lane l, reducer r) — reader / compute / writer
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {   // chunk c = block_idx*W + r
    load_local_block(block_idx);        // DRAM -> cb_local_input (issued ahead, depth 2)
    grant_landing_slot(block_idx);      // credit the upstream port_fwd once the landing slot is free (not head)
    receive_partial_block(block_idx);   // fabric-written landing slot -> cb_remote_partial (not head)
    reduce_block(block_idx);            // cb_remote_partial + cb_local_input -> cb_reduced, fp32 DEST (head: copy)
    stage_block(block_idx);             // cb_reduced -> port staging slot (tail: -> port final landing slot)
}

// port core (lane l) — port_fwd (RISCV_0, connection toward p+1) / port_bwd (RISCV_1, connection toward p-1)
for (uint32_t block_idx = 0; block_idx < num_blocks_this_lane; ++block_idx) {    // chunk c = block_idx, in order
    forward_partial_block(block_idx);   // port_fwd: staging slot -> reducer (c mod W) on p+1        (not tail)
    relay_final_block(block_idx);       // port_bwd: final landing slot -> output DRAM, and -> port_bwd on p-1 (not head)
}
```

| Block operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------------|-------------|--------------------|-----------------------------------|
| `load_local_block` | `chunk_tiles` tiles, pages `lane_start + c·chunk_tiles …` (last chunk of a lane: `last_chunk_tiles` valid) | `cb_local_input` depth 2 | one DRAM read barrier per chunk; TensorAccessor built once per kernel |
| `grant_landing_slot` | 1 slot of `chunk_tiles` | — | one local noc inc per chunk into the port's control array (`partial_granted[r]`). At kernel start, `min(RECV_DEPTH_CHUNKS, num_blocks_this_core)` grants. Grants are capped at `num_blocks_this_core` in total |
| `receive_partial_block` | `chunk_tiles` = `packets_per_chunk` packets | the landing ring | one arrival-counter poll per packet, `cb_push_back(packet_tiles)` per packet. No init |
| `reduce_block` | `chunk_tiles` | fp32 DEST per tile batch; nothing across chunks | `compute_kernel_hw_startup` once; one helper call per slice, streaming across chunk boundaries with per-tile wait/pop. Inits and reconfig once per kernel |
| `stage_block` | `chunk_tiles` | `cb_reduced` depth 2 | one NoC write burst plus one write barrier plus one local inc per chunk; wait on the `egress_credit` program semaphore per chunk |
| `forward_partial_block` | `packets_per_chunk` fused write+atomic-inc packets | staging ring | one staging-ready check, one remote-credit check, and `packets_per_chunk` EDM slot waits per chunk; free the staging slot with one local inc to reducer `c mod W`. Connection opened and closed once per kernel |
| `relay_final_block` | `packets_per_chunk` packets + `last/chunk_tiles` DRAM pages | final landing ring | one arrival check, one DRAM write barrier and one remote-credit check per chunk; free the slot with one local inc (`final_freed`, or at the tail the reducer's `egress_credit`) |

**Cross-device counters.** (Refinement 1: the three port-to-port counters below are now one per reducer index — see the R2/R3 notes under Regime selection.) These are persistent `GlobalSemaphore`s, cached per `(mesh_device, cluster_axis, route kind, num_lanes)` and created over the maximal core set of that key. Every counter is **cumulative within an invocation**. Its owner subtracts, atomically, exactly this invocation's total at the end: `noc_semaphore_inc(own, -total)`. It is never reset to 0, so increments from a neighbour that has already started the next invocation are preserved.

| Counter | Lives on | Incremented by | Waited on by | Unit | Per-invocation total |
|---------|----------|----------------|--------------|------|----------------------|
| `gsem_partial_arrival` | reducer cores | upstream `port_fwd` (fused write+inc, 1 per packet) | reducer reader | packets | `num_blocks_this_core · packets_per_chunk` |
| `gsem_partial_credit` | port cores | downstream `port_bwd` (atomic inc of the prefix delta) | `port_fwd` | chunks (in-order prefix) | `num_blocks_this_lane` |
| `gsem_final_arrival` | port cores | downstream `port_bwd` (fused write+inc, 1 per packet) | `port_bwd` (non-tail) | packets | `num_blocks_this_lane · packets_per_chunk` |
| `gsem_final_credit` | port cores | upstream `port_fwd` (atomic inc of the delta of upstream `port_bwd`'s `final_freed`) | `port_bwd` (non-head) | chunks | `num_blocks_this_lane` |

**Credit semantics.**
- The in-order prefix of `gsem_partial_credit` is computed by the downstream `port_bwd`. Chunk `c` is granted iff `partial_granted[c mod W] > c div W`, which gives `prefix = min_r(partial_granted[r]·W + r)`, capped at `num_blocks_this_lane`.
- `port_fwd` may send chunk `c` iff `gsem_partial_credit > c`.
- Final landing grants are in chunk order already, because `port_bwd` consumes in order. The initial grant is `min(W·FINAL_DEPTH_PER_REDUCER, num_blocks_this_lane)`.

**Why no stale state crosses invocations:**
1. No data is written remotely without a grant, and a grant is only issued by the receiver while it runs this invocation.
2. A receiver finishes only after it has received every chunk the sender sends, so the sender has no sends left when the receiver's next-invocation grants arrive.
3. The per-config gsem key keeps a different config's early grants off this config's counters.

**On-chip counters.** These live in per-call L1 or in program semaphores; they are only touched while both peers run this program.

| Counter | Lives on | Written by | Read by |
|---------|----------|-----------|---------|
| control array `staged[r]` (W words, 16 B stride) | port core (per-call `port_scratch` tensor) | reducer `r` writer (local inc after write barrier) | `port_fwd` |
| control array `final_ready[r]` (W words; tail only) | port core | reducer `r` writer at the tail | `port_bwd` |
| control array `partial_granted[r]` (W words) | port core | reducer `r` reader | `port_bwd` (prefix → upstream credit) |
| `go` (program semaphore) | reducer cores and port core | `port_bwd` after zeroing the control array | reducers and `port_fwd`, before their first control-array access |
| `egress_credit` (program semaphore) | reducer cores | `port_fwd` (staging slot freed) or `port_bwd` (tail final slot freed) | reducer writer |
| `final_freed` (program semaphore) | port core | `port_bwd` (landing slot freed) | `port_fwd` (forwards the delta as `gsem_final_credit` to `p+1`) |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Overlap / fill: `CHUNK_BYTES_TARGET = 32 KiB` | Chain fill costs about `2(G−1)·chunk_time`. That is small on a 2x2 but about 5% of an 8 MB transfer on an 8-long line. Larger chunks cut per-chunk handshakes | 16 / 64 / 128 KiB at `G = 2, 4, 8` and 8 MB / 64 MB |
| Grid synchronization: `REDUCERS_PER_LANE = 8` (18 cores per device at 2 lanes, not the full grid) | The op is capped by link rate, not by grid size. More reducers add per-chunk round-robin polling on the port and cut per-reducer stream length. Fewer reducers may be compute- or DRAM-bound, especially on Blackhole links | 4 / 8 / 16 / `floor(grid/num_lanes) − 1` |
| Port co-location: `port_fwd` and `port_bwd` share one core (two RISCs) | The port core's NoC carries staging-in, EDM-out, final-landing-in, DRAM-out and EDM-out per chunk, about 5 streams at link rate | `port_fwd` and `port_bwd` on separate cores (+1 core per lane) |
| Placement / NoC selection (catalog `noc_placement`) | Port cores should sit near the ethernet cores serving their link; reducers are row-major by lane. Ports use the RISC-default NoCs | Ports on the eth-adjacent row; reducers row-wise vs column-wise; swapping NoCs on the port kernels |
| Per-packet fused atomic inc | One inc per 4 KiB packet lets the reducer start the add before the whole chunk lands, at the cost of an atomic per packet | plain writes plus one fused inc on the last packet of each chunk |
| Relay DRAM-write barrier per chunk in `port_bwd` | A blocking barrier serializes DRAM write latency against the forward | defer the barrier by one chunk (free slot `c−1` after issuing chunk `c`) |
| Staging and final depth of 1 per reducer | Enough by argument (`W`-chunk rings), but unmeasured | 2 per reducer (port L1 1 MiB at bf16) |
| Fabric payload size (caller's `FabricRouterConfig`) | A larger payload means fewer packets per chunk; the op reads it at runtime and never sets it | re-measure under a caller-raised `max_packet_payload_size_bytes` |

**Refinement 3 measurements (BH 2x2, FABRIC_2D).**
- **Fabric ceiling**: a send-only port loop (`tests/.../probes/perf_ceiling_fabric.py`) gives the
  real ceiling. One worker channel moves about 237 cycles per 4 KiB packet, i.e. 23.3 GB/s per link
  direction, in either direction and with 4 or 8 headers. The 38 GB/s golden is the 1-D
  NeighborExchange figure, which is not reachable on FABRIC_2D.
- **Credit packets**: a header-only credit costs the EDM a full slot. One credit per 16 data
  packets costs +5.5%, which was exactly the op's whole gap.
- **Lever: credit coalescing.** `CREDIT_BATCH_CHUNKS = 4` sends one credit packet per 4 blocks of a
  reducer, on both streams. Ring depths are `effective + batch − 1`, and one shared per-call
  scratch keeps them inside L1. The axis lines now run at 98% of the send-only ceiling.
- **Path gate**: 2-link chains with middle devices keep per-chunk credits, because the deeper
  rings cost them 2–7% (see `_credit_batch`).

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| input DRAM → reducer `cb_local_input` | bf16 tiles | `TensorAccessor` `noc_async_read_page` × `chunk_tiles`, one barrier per chunk | reads only the valid pages of the last chunk; the push count stays nominal |
| upstream `port_fwd` → reducer `cb_remote_partial` (fabric) | bf16 tiles, `packet_tiles` per packet | `fabric_unicast_noc_fused_unicast_with_atomic_inc` into the neighbour reducer's landing slot, incrementing its `gsem_partial_arrival` | The landing tensor address is identical on every device (lockstep mesh allocation) and is passed as an RT arg. The CB is bound to it with `cb_descriptor_from_sharded_tensor` |
| reducer compute | fp32 DEST, bf16 packs | FPU `add` (middle/tail) or `copy` (head), `fp32_dest_acc_en=True` | per-step fp32 accumulation, as required |
| reducer `cb_reduced` → port staging slot (on-chip) | bf16 | `noc_async_write` burst + barrier + `noc_semaphore_inc(staged[r])` | the tail targets the port's final landing slot and `final_ready[r]` |
| `port_fwd` → downstream reducer (fabric) | bf16 | as above | `W` distinct destination cores, round-robin by chunk |
| downstream `port_bwd` → `port_bwd` final landing (fabric) | bf16 | fused write+inc into `gsem_final_arrival` | chunk-order ring |
| `port_bwd` final landing → output DRAM | bf16 | `TensorAccessor` `noc_async_write_page` of the valid pages | every device writes its own output; **only here** does the output cross DRAM |
| credits | 4 B atomics | `fabric_unicast_noc_unicast_atomic_inc` (cross-device) / `noc_semaphore_inc` (on-chip) | sent by the port kernel owning the connection toward the recipient |

**Placement axis.** TARGET has no sharded `memory_layout` axis: input and output are DRAM interleaved only. The lane × reducer split is a logical shard of the `tile` axis, and the `tile` axis is independent. A future sharded input would therefore be a knob-turn: reducers read their chunks from the local shard through a CB backed on the sharded buffer, with no NoC re-read.

**TARGET axis → cost class**

| TARGET value beyond Phase 0 | Class | What changes |
|-----------------------------|-------|--------------|
| `alignment ∈ {w_non_aligned, h_non_aligned}` | knob-turn | SUPPORTED plus tests only. The op already sums physical pages, `tensor_tiles` uses `ceil` per image, and the output reuses the input's TensorSpec |
| `dtype = float32` | knob-turn in compute plus a derived-constant change | `tile_bytes = 4096` → `packet_tiles = 1`, `chunk_tiles = 8`. The compute path switches to `binary_sfpu<AddBinary<>, …>` with `UnpackToDestFp32` on `cb_remote_partial` and `cb_local_input` (`ComputeConfigDescriptor.unpack_to_dest_mode`, `ttnn/cpp/ttnn-nanobind/program_descriptors.cpp:679`), and `copy` for the head. Pages are fp32 end to end; the wire carries fp32 |
| `cluster_axis = None`, Linear | knob-turn (R2) | the group-path builder emits the snake |
| `topology = Ring` | scheme-change (R3) | `num_slices = G`, per-slice roles, wrap-link connections |
| `num_links` beyond 2 | knob-turn | `num_lanes` |

**Structural impossibilities**: none beyond the (empty) `INVALID`.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one chunk of `chunk_tiles` tile pages (the block) |
| Grid | per device: `num_lanes × (W + 1)` Tensix cores. Lane ℓ uses a row-major run of `W` reducer cores plus 1 port core, taken from the compute grid. The **same logical cores and the same roles-by-index on every device**, because senders address neighbour cores by index |
| Per-lane work | `lane_tiles_nominal = ceil(tensor_tiles / num_lanes)`; `lane_start[ℓ] = ℓ·lane_tiles_nominal`; `lane_tiles[ℓ] = clamp(tensor_tiles − lane_start[ℓ], 0, lane_tiles_nominal)`; `num_blocks_this_lane[ℓ] = ceil(lane_tiles[ℓ] / chunk_tiles)` |
| Per-reducer work | `W = min(REDUCERS_PER_LANE, max_ℓ num_blocks_this_lane[ℓ])`, at least 1. Reducer `r` of lane ℓ: `num_blocks_this_core = ceil((num_blocks_this_lane[ℓ] − r) / W)`, clamped at 0; chunks `c = r, r+W, …` |
| Remainder | `tensor_tiles = prod(shape[:-2]) · ceil(H/32) · ceil(W/32)` (per image, `ceil`). The last chunk of a lane holds `last_chunk_tiles = lane_tiles − (num_blocks_this_lane − 1)·chunk_tiles` valid tiles. CB push/wait/pop counts and fabric packets stay at **nominal** `chunk_tiles` / `packets_per_chunk`; only DRAM reads and writes narrow to the valid pages. Garbage in the tail pages is summed and never stored. Idle lanes and reducers (`num_blocks = 0`) run no loop iterations but still perform the `go`/zeroing and end-of-invocation protocol, with totals of 0 |
| Group roles | `p` = the device's index in its group path (axis 0: row index; axis 1: column index). Roles: `head` (`p == 0`), `tail` (`p == G−1`), `middle`, or `solo` (`G == 1`: head copy plus tail delivery). CT args `is_head` and `is_tail` per device program |
| Links | lane ℓ: `port_fwd` = `ttnn.setup_fabric_connection(me, next, link_idx=fwd_links[ℓ], …)` with `fwd_links = ttnn.get_forwarding_link_indices(me, next)`. `port_bwd` does the same toward `prev`. `usable_links = min` over every hop in every group, in both directions |
| Regime pinning | `G == 1` → R1-solo; otherwise R1 (Phase 0). Tests pin `G = 2` (every 2x2 cell) and `G ≥ 3` where the mesh permits |

## Circular Buffers

Reducer cores only. Port cores have no CBs: their rings are raw L1 in the per-call `port_scratch` tensor, and packet headers come from `PacketHeaderPool` (firmware-reserved L1). Full accounting is in `l1_ledger.md`.

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_remote_partial` | 0 | `tile_bytes` | `RECV_DEPTH_CHUNKS · chunk_tiles` | spans `tile` → `chunk_tiles`, times the depth. Streams over `member`. Backed on the per-call `reducer_landing` L1 sharded tensor (`cb_descriptor_from_sharded_tensor`); the bytes are written by the upstream device's `port_fwd` over Fabric | Float16_b (wire format) | reader (credit-gated `push_back` per arrived packet; the bytes come from Fabric) | compute | whole kernel (middle/tail); unused on the head |
| `cb_local_input` | 1 | `tile_bytes` | `INPUT_DEPTH_CHUNKS · chunk_tiles` | spans `tile` → `chunk_tiles` × depth | Float16_b | reader | compute | whole kernel |
| `cb_reduced` | 2 | `tile_bytes` | `REDUCED_DEPTH_CHUNKS · chunk_tiles` | spans `tile` → `chunk_tiles` × depth | Float16_b (the partial's wire format; fp32 lives in DEST only, per the spec's "partials MAY travel in the input dtype") | compute | writer | whole kernel |

CB sync (per reducer, per chunk):
- `cb_local_input`: the reader pushes `chunk_tiles`; compute waits/pops `chunk_tiles` in total, per tile.
- `cb_remote_partial`: the reader pushes `packets_per_chunk × packet_tiles = chunk_tiles`; compute waits/pops per tile, `chunk_tiles` in total.
- `cb_reduced`: compute pushes `chunk_tiles`, per tile; the writer waits and pops `chunk_tiles` once.
- On the head, `cb_remote_partial` has no pushes and no pops.

Every push and pop is nominal, so each ring cycle closes: capacity is an exact multiple of `chunk_tiles`.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_local_block` | `chunk_tiles` | raw dataflow (TensorAccessor) | input DRAM, valid pages | `cb_local_input`, `chunk_tiles` pushed | — |
| 2 | `grant_landing_slot` | 1 slot | raw dataflow | `cb_remote_partial` free space (`cb_pages_reservable_at_back`) | control word `partial_granted[r]` +1 on the port core | — |
| 3 | `receive_partial_block` | `chunk_tiles` in `packet_tiles` steps | raw dataflow | `gsem_partial_arrival` ≥ cumulative packets | `cb_remote_partial`: reserve `chunk_tiles`, push `packet_tiles` per arrived packet | bytes were written in place by Fabric; the push only moves credit |
| 4 | `reduce_block` | `IterationShape::tiles(num_blocks_this_core · chunk_tiles)` per slice (streams per tile) | **helper** `add` (middle/tail) / `copy` (head) | `cb_remote_partial`, `cb_local_input` (PerTile wait/pop) | `cb_reduced` (PerTile reserve/push) | both inputs fully popped |
| 5 | `stage_block` | `chunk_tiles` | raw dataflow | `cb_reduced`: wait/pop `chunk_tiles` | port staging slot `c mod (W·STAGING_DEPTH_PER_REDUCER)` (tail: final landing slot) + `staged[r]` / `final_ready[r]` +1 | — |
| 6 | `forward_partial_block` | `packets_per_chunk` packets | raw fabric API | staging slot, `staged[c mod W]`, `gsem_partial_credit` | neighbour's `reducer_landing` slot + its `gsem_partial_arrival` | staging slot freed → `egress_credit` of reducer `c mod W` |
| 7 | `relay_final_block` | `packets_per_chunk` packets + valid DRAM pages | raw fabric + raw dataflow | final landing slot, `gsem_final_arrival` (tail: `final_ready[c mod W]`), `gsem_final_credit` | output DRAM pages, and the upstream `port_bwd` landing slot + its `gsem_final_arrival` | slot freed → `final_freed` (tail: `egress_credit` of reducer `c mod W`) |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| compute boot | raw_api | `compute_kernel_hw_startup(icb0, icb1, ocb)` | `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:59` | `(cb_remote_partial, cb_local_input, cb_reduced)`; head: the 2-arg form at `:106` `(cb_local_input, cb_reduced)` | — | — | — |
| `reduce_block` (middle/tail) | helper | `compute_kernel_lib::add<input(cb_remote_partial), input(cb_local_input), output(cb_reduced)>(IterationShape::tiles(n))` | `ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp:44-45` (body `convenience.inl:8-12`, `BinaryFpu` `eltwise/api/chain.hpp:485-494`) | default `WaitPolicy::PerTile` / `PopPolicy::PerTile` (`chain.hpp:208`, `:211`, `input()` defaults `:356-363`); `IterationShape` `chain.hpp:121` | `cb_remote_partial`, `cb_local_input` | `cb_reduced` | `n = num_blocks_this_core · chunk_tiles` (from `chunk_tiles`) |
| `reduce_block` (head) | helper | `compute_kernel_lib::copy<input(cb_local_input), output(cb_reduced)>(IterationShape::tiles(n))` | `ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp:86-87` | as above | `cb_local_input` | `cb_reduced` | same |
| `reduce_block` (fp32 refinement) | helper | `binary_sfpu<AddBinary<>, input(a), input(b), output(o)>` | `convenience.hpp:79-80`; `AddBinary` `ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp:15-20` | with `unpack_to_dest_mode` Fp32 on both inputs | same | same | same |
| compute config | host | `ComputeConfigDescriptor(fp32_dest_acc_en=True)` | `ttnn/cpp/ttnn-nanobind/program_descriptors.cpp:641-662` | — | — | — | — |
| 1, 7 DRAM | raw_api | `TensorAccessor` page read/write + barriers | `tech_reports/tensor_accessor/tensor_accessor.md` | CT `TensorAccessorArgs` | — | — | page count = valid tiles of the chunk |
| 2 | raw_api | `cb_pages_reservable_at_back`, `cb_reserve_back`, `cb_push_back` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:374`, `:403`, `:205` | `chunk_tiles`, `packet_tiles` | `cb_remote_partial` | — | `chunk_tiles` |
| 2, 5, 6, 7 local incs | raw_api | `noc_semaphore_inc` (negative value = atomic subtract at invocation end) | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:2279` | — | — | — | — |
| 6, 7 connection | raw_api | `WorkerToFabricEdmSender::build_from_args`, `open()`, `close()` | `tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp:91`, `:558`, `:623` | RT args from `ttnn.setup_fabric_connection` | — | — | — |
| 6, 7 headers | raw_api | `PacketHeaderPool::allocate_header` | `tt_metal/fabric/hw/inc/packet_header_pool.h:50` | — | — | — | — |
| 6, 7 route | raw_api | `fabric_set_unicast_route(hdr, dst_dev_id, dst_mesh_id)` (2D, the neighbour chip is one hop away) | `tt_metal/fabric/hw/inc/tt_fabric_api.h:229` | neighbour `FabricNodeId` from RT args | — | — | — |
| 6, 7 data | raw_api | `fabric_unicast_noc_fused_unicast_with_atomic_inc` + `NocUnicastAtomicIncFusedCommandHeader{noc_addr, sem_noc_addr, 1, flush}` | `tt_metal/fabric/hw/inc/mesh/api.h:949`; `tt_metal/fabric/fabric_edm_packet_header.hpp:297` | `packet_tiles · tile_bytes` bytes per packet | — | — | `packet_tiles` |
| credits | raw_api | `fabric_unicast_noc_unicast_atomic_inc` + `NocUnicastAtomicIncCommandHeader` | `tt_metal/fabric/hw/inc/mesh/api.h:237`; `fabric_edm_packet_header.hpp:289` | delta value | — | — | — |
| host: program | host | `ttnn.MeshProgramDescriptor`, `desc[MeshCoordinateRange(c, c)] = program`; `ttnn.generic_op(io_tensors, mesh_desc)` | `ttnn/cpp/ttnn-nanobind/program_descriptors.cpp:1198-1275`; `ttnn/cpp/ttnn/operations/generic/generic_op_nanobind.cpp:56-83` | io = `[input, reducer_landing, port_scratch, output]` | — | — | — |
| host: fabric | host | `ttnn.setup_fabric_connection(src, dst, link_idx, program, core)`, `ttnn.get_forwarding_link_indices`, `ttnn.get_tt_fabric_max_payload_size_bytes`, `ttnn.get_fabric_config`, `ttnn.get_fabric_kernel_defines`, `mesh_device.get_fabric_node_id(coord)` | `ttnn/cpp/ttnn-nanobind/fabric.cpp:151`, `:381`, `:334`, `:144`, `:231`; `ttnn/core/distributed/distributed_nanobind.cpp:296` | — | — | — | `num_lanes` |
| host: counters | host | `ttnn.create_global_semaphore(mesh_device, cores, 0)`, `ttnn.get_global_semaphore_address` | `ttnn/cpp/ttnn-nanobind/global_semaphore.cpp:21-47` | cached per config key | — | — | — |
| host: landing CB | host | `ttnn.cb_descriptor_from_sharded_tensor` | `ttnn/cpp/ttnn-nanobind/program_descriptors.cpp:518` | `reducer_landing` (HEIGHT_SHARDED L1, one shard of `RECV_DEPTH_CHUNKS·chunk_tiles` pages per reducer core) | — | `cb_remote_partial` | `RECV_DEPTH_CHUNKS`, `chunk_tiles` |

**Helpers considered and rejected**

| Raw API entries | Candidate | Mismatch |
|-----------------|-----------|----------|
| dataflow entries (1, 2, 3, 5) | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:7-30` | It covers L1→L1 *self-aimed reads* into a typed buffer only. Our on-chip hop is a write into **another core's** raw L1 slot followed by a semaphore inc, and the DRAM moves are TensorAccessor page I/O. Neither is expressible there |
| dataflow entries | `dfb_helpers_dataflow.hpp` | it exposes only tile-geometry getters (`:15-18`); no transfer primitive |
| cross-core transport | `mcast_pipe.hpp:6` (`SenderPipe` / `ReceiverPipe`) | an on-chip NoC **multicast** with a semaphore handshake. This op has no multicast, and every hop is a Fabric unicast to another chip |
| fabric entries | none | The kernel library has no Fabric helper. Recorded gap: a credit-gated Fabric send-ring helper would close it |
| compute boot | none | the caller owns startup (`chain.hpp:24-40`) |

## Broadcast Verification

| Phase | Op | CB_A (semantic name) Valid Region | CB_B (semantic name) Valid Region | Broadcast Dim |
|-------|-----|-----------------------------------|-----------------------------------|---------------|
| `reduce_block` | FPU add | `cb_remote_partial`: All | `cb_local_input`: All | None |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Two kernels on one EDM channel | Only one worker may connect per (link, direction). A second connection is undefined behaviour and hangs | exactly one `port_fwd` and one `port_bwd` per lane. `num_lanes ≤ usable_links`, else `ValueError`. No reducer ever opens a connection |
| Port deadlock | `port_fwd(p)` waits on credits that `port_bwd(p+1)` must forward, while `port_bwd(p+1)` may itself be waiting on credits from `port_fwd(p)` | every wait loop in both port kernels polls **and** forwards pending credits (the rule is stated in the Block schedule). No wait may be a bare blocking call |
| Out-of-order local completions | Reducers finish chunks out of order, so a single shared counter would falsely signal chunk `c` | per-reducer words (`staged[r]`, `final_ready[r]`, `partial_granted[r]`), with prefix/min logic on the port |
| Stale counters across invocations | A neighbour may already run invocation `n+1` and grant credits to a device still ending `n` | persistent per-config `GlobalSemaphore`s; atomic subtract of the exact per-invocation total, never a reset; zero-then-`go` for the per-call control array |
| Remote writes into a device still running a previous op | Its L1 or DRAM may be in use | every remote data write is credit-gated, and credits are only issued by the running receiver |
| Symmetric addressing | Senders compute neighbour L1 addresses and core coordinates locally | `reducer_landing` and `port_scratch` are mesh tensors (uniform address). Core layout, `W`, `chunk_tiles` and lane ranges are derived identically for every device. Reducer NoC coordinates are passed as RT args |
| fp32 DEST vs bf16 pages | `cb_reduced` is 16-bit under `fp32_dest_acc_en` (L1 audit 2 "under") | deliberate: it is the wire format. The spec permits input-dtype partials, and every step accumulates in fp32. The fp32 dtype refinement makes all pages Float32 |
| Ragged last chunk | Nominal packet and page counts carry garbage tiles | DRAM reads and writes narrow to valid pages; garbage is summed and never stored. Push/pop counts stay nominal, so the ring closes |
| Exactness tests | The rank-identity and single-contributor tests require bit-exact sums | FPU bf16+bf16 is exact in fp32 DEST, and small integer sums are exact in bf16. Every contribution is added exactly once by chain construction |
| `num_links` silently clamped | Forbidden by the spec | the explicit value is used as-is or rejected with `ValueError` |
| `Topology` import cycle | `ttnn.Topology` is unbound during `ttnn.operations` import | `from ttnn.operations.ccl import Topology` in SUPPORTED and defaults |
