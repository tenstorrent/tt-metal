# Operation Design: all_gather

## Overview

| Field | Value |
|-------|-------|
| Classification | CCL (multi-device collective — pure data movement, no arithmetic) |
| Goal | Gather every device's shard of a mesh-sharded tensor and concatenate all shards along `gather_dim`, so that AFTER the op every participating device on the 1-D line holds the full concatenated tensor (identical on every device). |
| Math | `output[d] = concat_{c=0..N-1}( input_shard[c], axis=gather_dim )` for every device `d` (identity gather; PCC ~1.0). |
| Mode | Hybrid — newly authored ring/line dataflow kernels under `ttnn/ttnn/operations/all_gather/kernels/`, assembled by a Python `ttnn.generic_op` + `ttnn.MeshProgramDescriptor`. Does NOT wrap, import, call, or dispatch to any existing `all_gather`/`all_gather_async` op. |
| References | Helper: `ttnn/cpp/ttnn/kernel_lib/ccl_helpers_dataflow.hpp` (+ `.inl`). Structural template (Python generic_op CCL): `ttnn/ttnn/operations/point_to_point/point_to_point.py` + `..._program_descriptor.py`. Correctness reference for the ring slice-walk / store-and-forward / barrier / counting (READ-ONLY): `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/minimal_default_writer.cpp` + `minimal_default_reader.cpp`. Semaphore-ownership gold standard: `ttnn/cpp/ttnn/operations/experimental/ccl/deepseek_moe_reduce_scatter/device/*`. Routing structs: `ttnn/cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp`. |

This is a multi-device op: it builds a `ttnn.MeshProgramDescriptor` with **one `(MeshCoordinateRange, ProgramDescriptor)` entry per participating device** on the 1-D line. Each device runs the same ring role (seed its own shard, receive from the prev hop, store-and-forward to the next hop). Distribution is op-owned.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | mesh-sharded along `gather_dim` across an N-device line; interleaved DRAM or L1; TILE (primary) or ROW_MAJOR; bfloat16 (primary) or float32 | — | — |
| `gather_dim` | `int` | yes | a dim of `input_tensor`; canonicalized to negative (`gd = gather_dim if gather_dim < 0 else gather_dim - rank`); **`gather_dim == 0` is the proven primary case** | — | host → per-device writer/reader CT arg (`gather_dim` normalized to 0..3) |
| `topology` | `ttnn.Topology` | no | `Linear` (primary) or `Ring` | `ttnn.Topology.Linear` | host (route computation) + writer/reader CT arg |
| `output_tensor` | `ttnn.Tensor \| None` | no | if given, spec must equal the resolved output spec | `None` (op allocates) | — |

`gather_dim` canonicalization: the public entry point MUST canonicalize `gather_dim` to a single sign convention (negative) **before** the support check, so positive aliases (`gather_dim=0` ≡ `-rank`) are not rejected by a literal membership test. The kernels consume the normalized 0..3 form (the host maps the canonical dim into the kernel's 4-D `gather_dim` slot, mirroring the reference factory's `normalized_dim`).

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | per-device shard `(B, C, H, W)` (rank ≤ 4). For `gather_dim=0` the shard's outer dim is what concatenates. |
| Dtype | bfloat16 (primary), float32 |
| Layout | TILE (primary) or ROW_MAJOR |
| Memory | interleaved, DRAM or L1 |
| Device | `ttnn.MeshDevice` line view of shape `(1, N)` — one shard per device |

### Output

| Property | Value |
|----------|-------|
| Shape | input shape with `shape[gather_dim] *= N` (N = number of devices on the line). For `gather_dim=0`: `(N·B, C, H, W)`. |
| Dtype | = input |
| Layout | = input |
| Memory | = input (interleaved, same buffer_type) |
| Value | identical on every device = host-side concat of all N input shards along `gather_dim` |

The output is **fully overwritten** by the op (every output page is produced — block `c` by device `c`'s self-copy, the rest by fabric writes), so no input-seeding/clone is required; the op only `allocate_tensor_on_device`s the output spec (or validates a supplied `output_tensor`). The output buffer is persistent and exists before dispatch, so fabric writes that land "early" are correct (persistent DRAM target).

## Dataflow Strategy

### Page-contiguous concat model (`gather_dim = 0`, primary)

The op is **pure byte movement** — like `point_to_point`, it never tilizes/untilizes and is format-agnostic: it copies physical pages (`buffer_page_size()` bytes, `buffer_num_pages()` pages) verbatim. For `gather_dim = 0` (outermost), each device's shard maps to a **contiguous block of output pages**:

```
out_page(chip_c, local_page p) = c * pages_per_shard + p          # pages_per_shard = input.buffer_num_pages()
```

The output tensor on every device is identical (block `c` always at `[c*pages_per_shard, (c+1)*pages_per_shard)`), so a write of chip-c's block always targets the same output page range on any device. This makes the neighbor's destination address the local output address routed +1 hop. (For `gather_dim != 0` the block is a strided page set computed exactly as the reference writer does — `tile_id_start = position * stride` with row wrapping by `output_tensor_Wt`; see `minimal_default_writer.cpp:247-256, 449-457`. Primary scope is `gather_dim=0` page-contiguous.)

### Ring store-and-forward on a line of N devices

Each device `i` runs **two worker cores** — a **forward worker** (rightward flow, → neighbor `i+1`) and a **backward worker** (leftward flow, → neighbor `i-1`). Bidirectional flow is required on a *line*: forward flow carries low-index shards rightward, backward flow carries high-index shards leftward. (For a *Ring*, a single direction with wraparound suffices; Ring is a noted extension — the kernel `gather_dim`/topology CT arg selects it via the reference's slice-walk modulo math.)

Per device, define (host-computed, per device coordinate):

| Symbol | Meaning | Formula (line, chip id `i`, ring size `N`) |
|--------|---------|--------|
| `num_targets_forward` | devices reachable forward (toward `i+1..N-1`) | `N - 1 - i` |
| `num_targets_backward` | devices reachable backward (toward `0..i-1`) | `i` |

**Forward worker (direction = 0, fabric connection → `i+1`):**
1. **Reader (NCRISC):** reads device `i`'s local input shard; (a) writes it to its OWN output block `i` (the **self-copy**, local NoC, always); (b) pushes the seed pages into `cb_relay_pages` only if `num_targets_forward > 0`. Then, for each forward-flow block arriving from the left (chips `i-1, i-2, …, 0`), waits on the counting semaphore and reads the landed block back from local output DRAM into `cb_relay_pages` (store-and-forward) — only if this device forwards (`num_targets_forward > 0`); otherwise it just waits on the counting semaphore to confirm arrival.
2. **Writer (BRISC):** fabric-writes the seed block (chip `i`) to `i+1`'s output block `i`, then relays the left-arrived blocks (chips `i-1..0`) to `i+1`'s output blocks. Increments `i+1`'s forward counting semaphore per `chunks_per_sync` chunks. Gated entirely on `num_targets_forward > 0` (the line-end device `N-1` opens no forward connection and sends nothing forward; its forward reader is a pure receiver).

**Backward worker (direction = 1, fabric connection → `i-1`):** mirror image. Reader reads the local input shard and pushes the seed (no self-copy — the forward reader already did it); receives right-arrived blocks (chips `i+1..N-1`), reads them back, and the writer relays them to `i-1`. Gated on `num_targets_backward > 0` (device `0` opens no backward connection).

**Store-and-forward contract (op-owned, NOT the helper):** a writer's fabric `write_page` lands a block **directly into the downstream device's persistent output DRAM** at the block's canonical page range. The downstream device's reader detects arrival via the counting semaphore, reads the landed block **back out of its own output DRAM** into `cb_relay_pages`, and its writer forwards it one more hop. There is **no FabricStreamReceiver** — the receive ingress is a local `noc_async_read` the op owns (helper banner `ccl_helpers_dataflow.hpp:66-67`).

### Tensix-to-Tensix contract (cross-device synchronization)

ONE op-internal `GlobalSemaphore` (parked on the descriptor) is reserved on both worker cores of every device. Per `(device, core)` it is used in **two strictly-ordered temporal phases** on the same per-core counter — exactly the helper's documented "barrier then counting" pattern (`ccl_helpers_dataflow.hpp:70-74`):

1. **Phase 1 — N-party startup barrier (`arm_multicast_inc`).** Each writer, before any payload write, multicast-increments the barrier semaphore on its same-direction peer cores along its direction's line-multicast route (`start_distance_in_hops = 1`, `range_hops = num_targets_<dir>`), then locally waits `noc_semaphore_wait_min(sem, barrier_threshold)` and resets `noc_semaphore_set(sem, 0)`. The `barrier_threshold` is the count of upstream same-direction peers that target this core: `i` for the forward core, `N-1-i` for the backward core (host-computed; the line ends wait for `0`). The `MulticastIncChannel` is **block-scoped** so it is destroyed before `arm_inc` re-arms the shared pooled header (`ccl_helpers_dataflow.hpp:70-74, 320-322`).
2. **Phase 2 — incremental/counting (`arm_inc`).** After the barrier reset, the upstream writer increments this device's same-direction counting semaphore once per `chunks_per_sync` chunks (`AtomicIncChannel::inc`); the local reader waits on cumulative targets (`noc_semaphore_wait_min(sem, running_target)`) before reading each landed chunk back. The reader resets `noc_semaphore_set(sem, 0)` at the very end (cache-reuse re-arm; helper banner `ccl_helpers_dataflow.hpp:67-69`).

The barrier resets to 0 *before* any counting `inc` (counting incs are issued only after the barrier reset and the seed-write loop begins), so counting starts cleanly from 0. Each device's own counter is reset by its own writer's barrier and then incremented only by its upstream neighbor's counting — see Key Risks for the cache-reuse ordering.

Teardown: every writer ends with `stream.drain()` (NoC write barrier + atomic barrier) then `stream.close()`.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one device's shard (all `pages_per_shard` pages), per direction |
| Grid | per device: 2 worker cores — `forward = CoreCoord(0,0)`, `backward = CoreCoord(0,1)` (uniform across all devices, so peer core virtual coords are identical mesh-wide) |
| Per-core work | forward core processes the seed block (chip `i`) + `num_targets_backward` relayed blocks; backward core processes the seed block + `num_targets_forward` relayed blocks; one link, one worker per direction (no per-link tile split) |
| Remainder | none at the core level (single worker per direction). Within a slice, the last chunk carries `pages % chunks_per_sync` and triggers a final counting `inc` (`minimal_default_writer.cpp:335-340`). |

`chunks_per_sync` (host-computed flow-control granularity) defaults to the reference heuristic `min(max(pages_per_shard / pages_per_packet, 1), 160)` (`all_gather_async_default_program_factory.cpp` heuristic). A single per-page packet uses `pages_per_packet = 1`.

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|--------|----------|----------|----------|
| `cb_relay_pages` | 16 | `round_up(input.buffer_page_size(), l1_alignment)` | `2 * pages_per_packet` (double-buffered chunk; `pages_per_packet = 1` for the per-page primary path → 2) | input dtype | reader (NCRISC): seed pages from input DRAM + relayed pages read back from output DRAM | writer (BRISC): fabric `write_page` | whole kernel |
| `cb_self_copy` | 24 | `round_up(input.buffer_page_size(), l1_alignment)` | 2 (allocated double-buffered; the forward reader `cb_reserve_back(.., 1)` once and reuses the one write ptr across all `P` pages) | input dtype | forward reader (NCRISC): stages one input page for the local self-copy | forward reader (NCRISC): same kernel `noc_async_write`s it to its own output block `i` (no cross-kernel consumer — intra-kernel scratch) | whole kernel |

Notes:
- One `cb_relay_pages` instance per worker core (same index 16 on both the forward and backward cores; each core has its own program-local CB). Likewise one `cb_self_copy` per core (index 24), but it is exercised ONLY by the **forward reader** (`direction == 0`); the descriptor allocates it on both cores for a uniform CB set, and the backward reader never touches it.
- **CB sync invariant (`cb_relay_pages`):** every page the reader `cb_push_back`es is `cb_wait_front`+`cb_pop_front`ed by the writer. The reader must NOT push the seed pages on a line-end device whose writer does not forward in that direction (gate the seed push on `num_targets_<dir> > 0`), or the CB fills and the reader blocks. Push count == pop count per direction.
- **`cb_self_copy` is intra-kernel scratch, deliberately NOT push/pop balanced.** The forward reader reserves the slot once, then loops `read input page → write to own output block i` reusing the same L1 write pointer; there is no producer/consumer split across kernels, so it is never `cb_push_back`ed. It exists solely because on the generic_op path every kernel-local L1 working buffer is a CB (there is no unmanaged scratch region for the `noc_async_read`→`noc_async_write` staging).
- No tilize/untilize, no compute CBs, no scaler — pure data movement.

## API Mapping

Every mechanism — helper or raw — has an exact file:line reference. The CCL fabric **dataflow kernel helper** is the primary building block for the writer's fabric egress; the raw NoC/semaphore calls are exactly the pieces the helper banner states it does NOT own.

### Kernel-side helper (fabric egress — writer)

| Phase | Type | Function | File:Line | Args / Template | Reads CB | Writes (target) | Manages own CB? |
|-------|------|----------|-----------|-----------------|----------|-----------------|------------------|
| Build egress | helper | `FabricStreamSender<>(conn_arg_idx, is_forward, alignment)` | `ccl_helpers_dataflow.hpp:425` | `is_forward = (direction==0)`; `alignment=1` (per-page payload already L1-aligned) | — | — | no |
| Open | helper | `FabricStreamSender::open()` → `FabricStream` | `ccl_helpers_dataflow.hpp:436` | — | — | — | no |
| Barrier arm | helper | `FabricStream::arm_multicast_inc(mcast_route, 1)` → `MulticastIncChannel` | `ccl_helpers_dataflow.hpp:387`, impl `.inl:133` | `mcast_route` built from `(start=1, range=num_targets_<dir>)` | — | — | no |
| Barrier issue | helper | `MulticastIncChannel::multicast_inc(peer_sem_noc_addr)` | `ccl_helpers_dataflow.hpp:328`, impl `.inl:151` | per same-direction peer-core sem noc addr | — | remote barrier sem (multicast) | no |
| Counting arm | helper | `FabricStream::arm_inc(route, 1)` → `AtomicIncChannel` | `ccl_helpers_dataflow.hpp:379`, impl `.inl:105` | `route = unicast_route(1)` | — | — | no (shares pooled sem header with the barrier — block-scope the multicast handle first) |
| Counting issue | helper | `AtomicIncChannel::inc(downstream_sem_noc_addr)` | `ccl_helpers_dataflow.hpp:310`, impl `.inl:122` | downstream same-direction counting sem | — | remote counting sem | no |
| Seed/relay write arm | helper | `FabricStream::arm_unicast_write(route, page_size)` → `UnicastWriteChannel` | `ccl_helpers_dataflow.hpp:365`, impl `.inl:23` | `route = unicast_route(1)`, `page_size = aligned page bytes` | — | — | no |
| Seed/relay write issue | helper | `UnicastWriteChannel::write_page(src_l1, out_page_idx, output_accessor)` | `ccl_helpers_dataflow.hpp:276`, impl `.inl:49` | `out_page_idx = c*pages_per_shard + p`; `output_accessor` = `TensorAccessor(output)` (uniform mesh address → routes to neighbor) | `cb_relay_pages` (read ptr) | neighbor output DRAM page | no |
| (optional) coalesced write | helper | `arm_scatter_write` / `ScatterWriteChannel::write_scatter` (≤4 pages/packet) | `ccl_helpers_dataflow.hpp:373/293`, impl `.inl:59/82` | optimization over per-page unicast | `cb_relay_pages` | neighbor output DRAM pages | no |
| Drain | helper | `FabricStream::drain()` | `ccl_helpers_dataflow.hpp:393`, impl `.inl:161` | NoC write barrier + atomic barrier | — | — | n/a |
| Close | helper | `FabricStream::close()` | `ccl_helpers_dataflow.hpp:395`, impl `.inl:167` | idempotent | — | — | n/a |
| Route build | helper | `unicast_route(num_hops)` | `ccl_helpers_dataflow.hpp:253` | `num_hops = 1` (immediate neighbor) | — | — | n/a |

The multicast route struct `line_multicast_route_info_t` (fields `start_distance_in_hops`, `range_hops`, …) is defined in `worker_routing_utils.hpp:23-36`; the kernel builds it from the host-passed `(start_distance, range_hops)` runtime args, zeroing the 2-D mesh fields (valid on the 1-D LowLatency path, consistent with how `unicast_route()` zeroes `dst_mesh_id` — `ccl_helpers_dataflow.hpp:253-258`). `num_line_unicast_args = 2`, `num_line_multicast_args = 6` (`worker_routing_utils.hpp:38-39`).

### Raw APIs (op-owned data movement — the helper explicitly does NOT own these)

| Phase | Raw API | File:Line | Purpose | Helpers considered and rejected |
|-------|---------|-----------|---------|---------------------------------|
| Seed read + relay read-back (receive ingress) | `noc_async_read` / `noc_async_read_barrier` | `api/dataflow/dataflow_api.h` | read local input shard for seed; read landed forward/backward blocks back out of local output DRAM for store-and-forward | **`FabricStream`/`FabricStreamReceiver`** — rejected: the helper has NO receive type; `ccl_helpers_dataflow.hpp:66-67` states "The receive INGRESS is likewise a local NoC read the op owns; there is no FabricStreamReceiver." |
| Self-copy (own shard → own output block) | `noc_async_write` / `noc_async_write_barrier` | `api/dataflow/dataflow_api.h` | place device `i`'s own block into its own output block `i` (no fabric, local) | **`UnicastWriteChannel::write`** — rejected: that is a *fabric* (cross-device) write; the self-copy is intra-device. The helper is "PURE DATA MOVEMENT ... fabric egress" (`ccl_helpers_dataflow.hpp:7-18`), not local copies. |
| Barrier/counting WAIT half | `noc_semaphore_wait_min`, `noc_semaphore_set` | `api/dataflow/dataflow_api.h` | wait for barrier threshold / cumulative counting target; reset for cache re-arm | **None** — the helper owns only the *sending* half (atomic-inc). `ccl_helpers_dataflow.hpp:63-69`: "The WAITING half is a plain local `noc_semaphore_wait_min(sem, threshold)` the op calls directly ... each side must `noc_semaphore_set(sem, 0)` to re-arm." |
| Ring slice-walk + concat addressing | plain index arithmetic + `TensorAccessor` (`get_noc_addr`) | `minimal_default_writer.cpp:404-457` (reference), `minimal_default_reader.cpp:200-252` (reference) | compute `actual_slice_chip_id = i ± k (mod N)` and `out_page = c*pages_per_shard + p` | **None** — `ccl_helpers_dataflow.hpp:82-86`: "What the helper does NOT own (the op composes it): ring slice-walk (chip_id +/- k mod ring_size), store-and-forward relay, concat-by-gather_dim output addressing, address generation (TensorAccessor ... is consumed, never re-wrapped)." |
| Output/input address generation | `TensorAccessor(TensorAccessorArgs<...>, addr)` | `tech_reports/tensor_accessor/tensor_accessor.md` | per-page DRAM/L1 NoC addresses for input read, output read-back, output write, and the fabric `write_page` dst | **None** — TensorAccessor is the address-gen primitive the helper *consumes* (`write_page`'s `AddrGen` template, `.inl:49-51`); building it is op-owned. |

### Host-side helpers (program-descriptor assembly, mirror `point_to_point_program_descriptor.py`)

| Phase | Function | File:Line | Purpose |
|-------|----------|-----------|---------|
| 1-D route (per direction, owns fwd/bwd sign + ring short-way) | `ttnn._ttnn.fabric.ccl_dm_route(mesh_device, src_coord, dst_coord, topology)` → `{num_hops, is_forward, neighbor_id}` | `ttnn/cpp/ttnn-nanobind/fabric.cpp:254-266` | forward route `coord_i → coord_{i+1}` and backward route `coord_i → coord_{i-1}`; gives `num_hops=1`, `is_forward`, `neighbor_id` |
| Fabric connection RT args | `ttnn.setup_fabric_connection(src_fabric_id, neighbor_fabric_id, link_idx, program, core)` → `list[uint32]` (mutates `program`: appends `SemaphoreDescriptor`s) | `ttnn/cpp/ttnn-nanobind/fabric.cpp:142-177` | per-writer fabric arg block, laid out `[has_forward][fwd args][has_backward][bwd args]` exactly as `append_ccl_fabric_rt_args` (mirror `point_to_point_program_descriptor.py:64-78`) |
| Op-internal cross-device semaphore | `ttnn.create_global_semaphore(mesh_device, worker_cores, 0)` + `ttnn.get_global_semaphore_address(sem)` | `ttnn/cpp/ttnn-nanobind/global_semaphore.cpp:41-43, 59-60` | created ONCE per mesh_device, `ttnn.synchronize_device` ONCE after, cached, parked via `mpd.semaphores = [sem]`, address baked into RT args (mirror `point_to_point.py:88-98, 197-210`) |
| (optional) packet framing | `ttnn._ttnn.fabric.ccl_packet_dims(dtype, page_size, num_pages, l1_alignment)` | `ttnn/cpp/ttnn-nanobind/fabric.cpp:246-247` | available for multi-page coalescing (owns the bf16 `bit_floor`); the per-page primary path uses 1:1 page↔packet framing and the L1-aligned page size as on-wire payload, so it is not required |
| Mesh assembly | `ttnn.MeshProgramDescriptor()`, `mpd[ttnn.MeshCoordinateRange(coord, coord)] = program`, `mpd.semaphores = [sem]`, `ttnn.generic_op([input, output], mpd)` | `ttnn/cpp/ttnn-nanobind/program_descriptors.cpp` (MeshProgramDescriptor + `.semaphores`); `point_to_point.py:200-213` | one `ProgramDescriptor` per device coordinate on the line |

## Dataflow Phases (per device `i`, per direction worker)

| # | Phase | Owner | Consumes | Produces | State after |
|---|-------|-------|----------|----------|-------------|
| 0 | Allocate output, create/cache + park ONE GlobalSemaphore, compute per-device routes (`ccl_dm_route`) & `num_targets_{fwd,bwd}` & barrier thresholds | host | input spec | `MeshProgramDescriptor` (one program/device), `sem` parked | workload built |
| 1 | Startup barrier | writer | barrier sem (multicast incs from peers) | barrier sem reaches `barrier_threshold`, then reset to 0 | sem == 0; `MulticastIncChannel` destroyed (pooled header freed) |
| 2 | Seed read + self-copy | reader | local input shard pages | self-copy → own output block `i`; seed pages → `cb_relay_pages` (if forwarding) | own output block `i` correct; `cb_relay_pages` holds seed |
| 3 | Seed fabric-write | writer | `cb_relay_pages` seed pages | block `i` → neighbor output block `i`; counting `inc` to neighbor | neighbor output block `i` landed; `cb_relay_pages` drained of seed |
| 4 | Relay receive (store) | reader | counting sem (incs from upstream) + landed blocks in local output DRAM | landed upstream blocks → `cb_relay_pages` | upstream blocks in local output + re-staged in CB |
| 5 | Relay fabric-write (forward) | writer | `cb_relay_pages` relay pages | upstream blocks → neighbor output blocks; counting `inc`s | neighbor has all relayed blocks |
| 6 | Drain + close + final reset | writer / reader | — | `drain()` + `close()`; reader `noc_semaphore_set(sem,0)` | fabric quiesced; sem re-armed for next program-cache run |

After phase 6 on all devices: every device's output == concat of all N shards along `gather_dim`.

## Key Risks and Gotchas

- **ONE op-internal GlobalSemaphore, created once, parked (mandate W4 slot).** Create it once keyed on `id(mesh_device)`, `ttnn.synchronize_device` ONCE after creation, cache it, and `mpd.semaphores = [sem]` so the framework holds its L1 alive across program-cache hits. Do NOT re-create per call; do NOT add a per-call post-dispatch `synchronize_device` barrier. Mirror `point_to_point.py:88-98, 197-210`.
- **Shared pooled packet header — block-scope the barrier.** `arm_multicast_inc` (barrier) and `arm_inc` (counting) reuse ONE pooled sem header (`ccl_helpers_dataflow.hpp:70-74`). The `MulticastIncChannel` MUST be destroyed (block-scoped to the barrier phase) before `arm_inc` re-arms the header for counting — otherwise the pool exhausts and hangs. Mirror the reference's block-scoped `barrier` (`minimal_default_writer.cpp:196-221`).
- **Cache-reuse semaphore re-arm (footgun).** Programs are cached and the GlobalSemaphore reused. The barrier writer resets its own sem to 0 *after its wait*; the counting reader resets to 0 *after its last wait* (`ccl_helpers_dataflow.hpp:67-69`). Without these resets the first call passes and the second hangs/corrupts. The acceptance test's two-call program-cache case exercises this.
- **Barrier→counting ordering on the shared per-core counter.** Both phases increment the same per-`(device,core)` counter. Counting incs are issued only after the barrier reset and the seed-write loop begins (long after the barrier), and the barrier resets to 0 immediately after its wait, so counting always starts from 0. The barrier's `noc_semaphore_wait_min` gives the cross-device ordering. If the shared-counter coupling ever proves fragile, the mandate-compliant fallback is a second op-internal GlobalSemaphore — *also created once and parked in `mpd.semaphores`* (the gold standard `deepseek_moe_reduce_scatter` parks two).
- **CB push/pop balance at line ends.** Gate the seed push and the relay read-back on `num_targets_<dir> > 0` (the line-end device in a given direction does not forward, so its reader must NOT push pages its writer will never pop — else `cb_relay_pages` fills and the reader blocks). Push count MUST equal pop count per direction.
- **Edge-device fabric connections.** Device `0` opens NO backward connection; device `N-1` opens NO forward connection. The host calls `setup_fabric_connection` and the writer opens the `FabricStreamSender` only when `num_targets_<dir> > 0` (mirror the `valid_targets` gating in `minimal_default_writer.cpp:76-99, 311, 360-387`). The line-end worker in the missing direction is a pure receiver (reader waits on the counting sem; no writer egress). The self-copy is done by the **forward reader on every device** so it never depends on a missing connection.
- **Uniform mesh output address.** The fabric `write_page` uses the local `TensorAccessor(output)` base address as the neighbor's destination; this is correct only because mesh-allocated interleaved tensors share a buffer address across devices and the fabric route (1 hop) directs the write to the neighbor. Verify the output is mesh-allocated interleaved with uniform addressing (true for `allocate_tensor_on_device` on a MeshDevice).
- **Per-page payload must fit the fabric packet.** The per-page primary path sends one page (e.g. a bf16 tile = 2 KB) as one fabric packet (`arm_unicast_write(route, aligned_page_size)`). Typical tiles/sticks fit; for very large RM rows use the `ccl_packet_dims` coalescing/segmentation path.
- **Verification topology is fixed.** The op is verified on a simulated Wormhole T3K **line mesh `(1, 8)` with `fabric_config = ttnn.FabricConfig.FABRIC_1D`** via `scripts/run_multidevice_sim_pytest.py --op all_gather`. The acceptance test MUST open exactly `(1, 8)` with `FABRIC_1D`; a different mesh shape hangs fabric init (`Fabric Router Sync: Timeout`). Proven first case: `gather_dim=0`, bfloat16, TILE_LAYOUT, Linear.

## Structural impossibilities (INVALID candidates for feature_spec.py)

- `{dtype: bfloat8_b, layout: ROW_MAJOR_LAYOUT}` — bfloat8_b is a tiled block-float format with no row-major representation (single-tensor coupling: both axes describe the input tensor; the data-format definition would have to change). This is the canonical INVALID cell, authored in `eval/golden_tests/all_gather/feature_spec.py`.
