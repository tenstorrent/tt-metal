# Operation Design: point_to_point

## Overview

| Field | Value |
|-------|-------|
| Classification | CCL (collective communication) — pure cross-chip **data movement**, NO arithmetic |
| Goal | Copy the input tensor's shard living on `sender_coord` to `receiver_coord` over the Tenstorrent fabric. The receiver device's output shard becomes bit-for-bit equal to the sender's input shard; every other device's shard is left unchanged. |
| Math | `output[receiver_coord] = input[sender_coord]`; `output[d] = input[d]` for every other device `d` (identity copy) |
| Mode | Derivative — assembled from `ttnn.generic_op` + `ttnn.MeshProgramDescriptor`, the bound CCL host helpers (`ccl_packet_dims`, `ccl_dm_route`, `setup_fabric_connection`), and the kernel-side fabric egress helper `dataflow_kernel_lib::ccl`. NEWLY-authored dataflow kernels; the bound C++ `ttnn.point_to_point` is NOT wrapped. |
| References | Kernel helper: `ttnn/cpp/ttnn/kernel_lib/ccl_helpers_dataflow.hpp` (+ `.inl`). Host helpers: `ttnn/cpp/ttnn/operations/ccl/common/host/ccl_helpers_dataflow_host.hpp`. Pybind: `ttnn/cpp/ttnn-nanobind/fabric.cpp:235-266`. Correctness reference (read-only): C++ `ttnn/cpp/ttnn/operations/point_to_point/device/*`; `all_gather_async` writer `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/minimal_default_writer.cpp`. Framework example: `tests/ttnn/unit_tests/base_functionality/test_generic_op_internal_semaphore.py`. |

This op is **dataflow-only**: there is no compute kernel. All work is done by two dataflow kernels per participating device (reader + writer). Only two mesh devices run programs; every other mesh device runs nothing.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | interleaved, on a `ttnn.MeshDevice` (≥2 devices along the connecting row/col), rank ≥ 2 | — | RT (buffer address, page count, page size) |
| `sender_coord` | `ttnn.MeshCoordinate` | yes | inside the mesh; shares a row or column with `receiver_coord` | — | host-only (selects the SEND program's mesh range + fabric route) |
| `receiver_coord` | `ttnn.MeshCoordinate` | yes | inside the mesh; `!= sender_coord` | — | host-only (selects the RECEIVE program's mesh range + fabric route) |
| `topology` | `ttnn.Topology` | no | `Linear`, `Ring` | `Linear` | host-only (feeds `ccl_dm_route` hop/direction math; `Ring` may route the short way around) |
| `output_tensor` | `ttnn.Tensor \| None` | no | spec must equal the resolved output spec (== input spec) | `None` | RT (buffer address) |
| `intermediate_tensor` | `ttnn.Tensor \| None` | no | spec must equal `resolve_intermediate_spec(input_tensor)` | `None` | RT (buffer address of the fabric landing buffer) |

There are no compute-kernel template params. Op-specific values reach the kernels as runtime args (packet dims, hop count, semaphore address, fabric-connection block); CB indices and alignment reach them as compile-time args.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | per-device shard shape, rank ≥ 2 |
| Dtype | `bfloat16` (primary), `float32`, `bfloat8_b` (TILE only), `uint16`, `int32`, `uint32` |
| Layout | `TILE_LAYOUT` (primary) or `ROW_MAJOR_LAYOUT` |
| Memory | interleaved, DRAM or L1 (per-shard buffer **page size must be 16-byte aligned**) |

### Output

| Property | Value |
|----------|-------|
| Shape | identical to input's per-device shard shape |
| Dtype | same as input |
| Layout | same as input |
| Memory | same memory config as input |
| Content | `output[receiver_coord]` == `input[sender_coord]` (bit-identical); `output[d]` == `input[d]` for every other `d`, including the sender's own output shard |

### Intermediate (op-internal staging / fabric landing buffer)

| Property | Value |
|----------|-------|
| Shape | `[total_packets, packet_size_bytes / 4]` |
| Dtype | `uint32` (sidesteps `element_size` being undefined for block-float dtypes such as `bfloat8_b`) |
| Layout | `ROW_MAJOR_LAYOUT` |
| Memory | interleaved, same `buffer_type` as input |
| Rationale | Addressed **per-packet**: page index = `packet_idx`, page size overridden to `packet_size_bytes`. Holds exactly `total_packets` on-wire packets for every dtype. `packet_size_bytes` is always a multiple of the L1 alignment (≥16), hence of 4. Resolved by `resolve_intermediate_spec()`; allocated on the mesh if not supplied. |

## Multi-Device Assembly

The op builds ONE `ttnn.MeshProgramDescriptor` with exactly **two** `(MeshCoordinateRange, ProgramDescriptor)` entries, then dispatches `ttnn.generic_op([input, intermediate, output], mesh_pd)` (output tensor **last** in the IO list).

| Mesh range | Program | Kernels (index) | Core |
|------------|---------|-----------------|------|
| `MeshCoordinateRange(sender_coord, sender_coord)` | SEND | `point_to_point_sender_reader.cpp` (0), `point_to_point_sender_writer.cpp` (1) | `(0,0)` |
| `MeshCoordinateRange(receiver_coord, receiver_coord)` | RECEIVE | `point_to_point_receiver_reader.cpp` (0), `point_to_point_receiver_writer.cpp` (1) | `(0,0)` |

No entry is created for any other mesh coordinate — those devices run no kernels.

### Output seeding — the "all other shards unchanged" contract

The fabric transfer writes only the receiver device's output shard. To guarantee `output[d] == input[d]` on every non-receiver device (including the sender's own output shard, which no kernel writes), the op seeds the output tensor from the input on EVERY device **before** dispatch:

- `output_tensor is None`: `output = ttnn.clone(input_tensor, memory_config=input.memory_config())` (seeded output).
- `output_tensor` provided: `ttnn.copy(input_tensor, output_tensor)` (specs already validated equal).

The seed is a per-device data-movement op enqueued before the `generic_op` on the same mesh queue, so it completes first; the receiver's writer then overwrites only the receiver shard.

### Cross-device GlobalSemaphore (op-internal, created once)

| Concern | Decision |
|---------|----------|
| Creation | ONE `ttnn.create_global_semaphore(mesh_device, worker_cores, 0)` per `mesh_device`, cached in a module-level dict keyed on `id(mesh_device)`. Never recreated per call. `worker_cores` from `compute_with_storage_grid_size()` via `ttnn.num_cores_to_corerangeset(...)`. |
| Cross-device alloc consistency | `ttnn.synchronize_device(mesh_device)` exactly ONCE, immediately after creation. |
| Lifetime across program-cache hits | `mesh_pd.semaphores = [sem]` — the framework holds the semaphore's L1 allocation alive for the cached workload's lifetime (`program_descriptors.cpp:1077-1087`). |
| Kernel access | `ttnn.get_global_semaphore_address(sem)` (a `uint32` absolute address, same on every device) baked into BOTH programs' runtime args at the slot each kernel reads. |
| Post-dispatch barrier | NONE. Do NOT add a per-call `synchronize_device` — the framework owns the lifetime once parked. |

### Fabric route + connection (host-side)

| Mechanism | Use |
|-----------|-----|
| `ccl_dm_route(mesh_device, sender_coord, receiver_coord, topology)` | SEND route (payload direction). Returns `{num_hops, is_forward, neighbor_id}`. Owns the fabric fwd/bwd **sign reversal** and Ring short-way. |
| `ccl_dm_route(mesh_device, receiver_coord, sender_coord, topology)` | RECEIVE (ack) route — the receiver's "ready" travels back toward the sender (coords swapped). |
| `setup_fabric_connection(src_fabric_id, neighbor_id, link_idx=0, program, core)` | Returns the fabric-connection runtime-arg block AND mutates the program (appends fabric `SemaphoreDescriptor`s). Called **after** the `ProgramDescriptor` is constructed. |

The fabric-connection runtime block is laid out exactly as `append_ccl_fabric_rt_args` (`ccl_helpers_dataflow_host.hpp:219-237`):

```
[ has_forward = int(is_forward) ]                 # kernel peeks this word at conn_arg_idx for direction
[ <forward conn args from setup_fabric_connection> ]   # only if is_forward
[ has_backward = int(not is_forward) ]
[ <backward conn args from setup_fabric_connection> ]  # only if not is_forward
```

It lives on the SEND writer (kernel 1) and the RECEIVE reader (kernel 0), starting at rt-arg index 9 (right after the 9 scalar args each kernel reads first).

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | the entire per-device shard (`num_pages` pages → `total_packets` packets), streamed by one core |
| Grid | single core `(0,0)` — `CoreRangeSet([CoreRange((0,0),(0,0))])` — on each of the two endpoint programs |
| Per-core work | SEND core streams and coalesces all `num_pages` into `total_packets` fabric writes; RECEIVE core de-coalesces all `total_packets` back into `num_pages` output pages |
| Remainder | none — a single core owns the whole shard. The last packet may carry fewer than `pages_per_packet` pages: `curr_pages_per_packet = min(max_pages_per_packet, pages_left)`. The segmented regime always uses `pages_per_packet == 1`. |

Single-core is intentional: the transfer is fabric-bandwidth bound, the shard is one interleaved buffer, and one core streaming pages keeps the fabric egress and the semaphore handshake trivial. The reference C++ factory is likewise single-core. Multi-core/multi-link striping is a future refinement, not part of this design.

## Circular Buffers

### SEND program

| Semantic Name | Index | Page Size | Num Pages | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|--------|----------|----------|----------|
| `cb_input_pages` | 0 | `aligned_input_page_size` = `round_up(input_page_size, l1_alignment)` | 2 (streaming double-buffer) | input dtype | `sender_reader` | `sender_writer` | one page per shard page; produced then consumed |
| `cb_packet_scratch` | 24 | `packet_size_bytes` | 1 (single scratch packet) | input dtype | `sender_writer` (self) | `sender_writer` (self) | private L1 working buffer for one coalesced packet; reserved once, held for the whole kernel |

### RECEIVE program

| Semantic Name | Index | Page Size | Num Pages | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|--------|----------|----------|----------|
| `cb_packet_scratch` | 24 | `packet_size_bytes` | 1 (single scratch packet) | input dtype | `receiver_reader` (self) | `receiver_reader` (self) | private L1 landing buffer for one packet read from the intermediate; reserved once, held for the whole kernel |
| `cb_output_pages` | 16 | `output_page_size` | `3 * pages_per_packet` (de-coalesce pipeline slack) | input dtype | `receiver_reader` | `receiver_writer` | one page per shard page; produced then consumed |

CB index convention honored: `0` input, `16` output, `24` intermediate/scratch. No packet-header CB is declared — `FabricStreamSender` draws headers from the fabric-L1 `PacketHeaderPool` (`tt_metal/fabric/hw/inc/packet_header_pool.h`), a fixed reserved region, so no program-side CB is needed.

**CB sync verification (producer push count == consumer wait/pop count):**

| CB | Producer pushes | Consumer waits/pops | Balanced |
|----|-----------------|---------------------|----------|
| `cb_input_pages` (SEND) | reader: `num_pages` × `cb_push_back(…,1)` | writer: `num_pages` × `cb_wait_front/cb_pop_front(…,1)` | ✓ |
| `cb_packet_scratch` (SEND) | writer: 1 × `reserve+push` at start | none (used as raw L1 scratch) | ✓ (scratch, single allocation) |
| `cb_packet_scratch` (RECV) | reader: 1 × `reserve` at start, `push` at end | none (used as raw L1 scratch) | ✓ (scratch, single allocation) |
| `cb_output_pages` (RECV) | reader: `num_pages` × `cb_push_back(…,1)` | writer: `num_pages` × `cb_wait_front/cb_pop_front(…,1)` | ✓ |

## Dataflow Strategy

The op never tilizes/untilizes — it moves raw physical pages (padded tiles for TILE, last-dim rows for ROW_MAJOR), so it is format-agnostic. A single shard page may be larger or smaller than one fabric transfer unit; `ccl_packet_dims` frames this into two regimes and the kernels coalesce/segment accordingly.

```
SENDER DEVICE (sender_coord)                          RECEIVER DEVICE (receiver_coord)
──────────────────────────                            ────────────────────────────────
input shard (DRAM/L1, interleaved)                    intermediate (DRAM/L1) ── fabric landing zone
   │ reader: noc_async_read page-by-page                   ▲
   ▼                                                        │ fabric unicast write (per packet)
cb_input_pages (L1, double-buffered)                        │
   │ writer: tt_memmove coalesce N pages ───────────────────┘
   ▼                                                   intermediate (DRAM/L1)
cb_packet_scratch (L1, one packet) ── FABRIC ──▶            │ reader: noc_async_read one packet
                                                            ▼
                                                       cb_packet_scratch (L1, one packet)
                                                            │ reader: tt_memmove de-coalesce into pages
                                                            ▼
                                                       cb_output_pages (L1)
                                                            │ writer: noc_async_write page-by-page
                                                            ▼
                                                       output shard (DRAM/L1, interleaved)
```

### Packet framing (page ↔ packet)

Computed once on the host by `ccl_packet_dims(dtype, page_size_bytes, num_pages, l1_alignment)` and passed to the kernels. Two regimes (owned by the helper, incl. the bf16 `std::bit_floor` max-packet rule):

| Regime | Condition | Behavior |
|--------|-----------|----------|
| Coalescing | `aligned_page ≤ max_packet` | `pages_per_packet = min(max_packet/aligned_page, num_pages)`, `page_segments = 1`, `packet_size = aligned_page * pages_per_packet`, `total_packets = div_up(num_pages, pages_per_packet)` |
| Segmentation | `aligned_page > max_packet` | `pages_per_packet = 1`, `page_segments = div_up(aligned_page, max_packet)`, `packet_size = max_packet`, `total_packets = page_segments * num_pages` |

Both kernels handle both regimes: the SEND writer runs an inner `page_segments` loop and coalesces up to `curr_pages_per_packet` pages before issuing one fabric write; the RECEIVE reader mirrors it, reading a whole landed packet and scattering its pages/segments out.

### Cross-device coordination contract (ordering guarantees)

One op-internal `GlobalSemaphore` (same absolute address on both devices) drives a **ready → payload → done** handshake. This is the load-bearing correctness invariant. The **inc** (sending) half is helper-owned; the **wait/reset** (waiting) half is op-owned, as the helper banner mandates.

| Step | Device | Action | Owner |
|------|--------|--------|-------|
| 1 | RECEIVE reader | signal "ready" to the sender: `FabricStreamSender::signal(sender_num_hops, sender_sem_noc_addr)` — one-shot fabric atomic-inc (open→arm_inc→inc→close) | helper (`signal`) |
| 2 | SEND writer | `noc_semaphore_wait_min(sem, 1)` for "ready", then `noc_semaphore_set(sem, 0)` — reset **BEFORE** its own outgoing inc (cache re-arm) | op (raw) |
| 3 | SEND writer | stream all packets into the receiver's intermediate (`UnicastWriteChannel::write_page`) | helper (fabric write) |
| 4 | SEND writer | signal "done" on the receiver's semaphore (`AtomicIncChannel::inc`), then `stream.close()` (drains) | helper (`inc`/`close`) |
| 5 | RECEIVE reader | `noc_semaphore_wait_min(sem, 1)` for "done" — payload has fully landed | op (raw) |
| 6 | RECEIVE reader | de-coalesce packets → `cb_output_pages`, then `noc_semaphore_set(sem, 0)` — reset **AFTER** its wait (cache re-arm) | op (raw) |

Guarantees: the sender never transmits before the receiver is ready (step 2 blocks on step 1); the receiver never consumes before the payload fully lands (step 5 blocks on step 4). The single shared semaphore is reused for both signals because the payload stream separates them; each device's local copy goes `0 → 1 → 0`, and the reset convention (sender resets **before** its inc, receiver resets **after** its wait) leaves it clean for the next program-cache-hit invocation. `noc_semaphore_wait_min` is order-insensitive — a counting inc that arrives early persists, so no signal is lost.

## API Mapping

Every mechanism has a verified file:line reference. Kernel-helper paths are relative to `ttnn/cpp/ttnn/kernel_lib/`; host-helper paths to `ttnn/cpp/ttnn/operations/ccl/common/host/`; pybind paths to `ttnn/cpp/ttnn-nanobind/`.

### Kernel-side helpers (fabric egress) — `dataflow_kernel_lib::ccl`

| Phase | Type | Function | File:Line | Args / Template | In CB | Out | Manages own CB ops? |
|-------|------|----------|-----------|-----------------|-------|-----|---------------------|
| SEND: open fabric egress | helper | `FabricStreamSender<>(conn_arg_idx, is_forward, alignment)` + `.open(unicast_route(hops))` | `ccl_helpers_dataflow.hpp:436`, `:448`; `.inl:22` | reads fabric conn rt-args at `conn_arg_idx`; binds route from hops | — | — | no (fabric only) |
| SEND: build route | helper | `unicast_route(num_hops)` | `ccl_helpers_dataflow.hpp:260-265` | `num_hops` | — | — | n/a |
| SEND: arm payload write | helper | `FabricStream::arm_unicast_write(payload_size_bytes)` → `UnicastWriteChannel` | `ccl_helpers_dataflow.hpp:374`; `.inl:22-37` | draws a pooled packet header (set_state PayloadSize) | — | — | no |
| SEND: issue packet write | helper | `UnicastWriteChannel::write_page(src_l1, packet_idx, intermediate_addrgen)` | `ccl_helpers_dataflow.hpp:283-284`; `.inl:46-51` | `cb_packet_scratch` L1 (src) | fabric → receiver intermediate page `packet_idx` | no (op owns the CB) |
| SEND: arm "done" inc | helper | `FabricStream::arm_inc(1)` → `AtomicIncChannel` | `ccl_helpers_dataflow.hpp:386`; `.inl:103-118` | — | — | no |
| SEND: issue "done" inc | helper | `AtomicIncChannel::inc(receive_sem_noc_addr)` | `ccl_helpers_dataflow.hpp:317`; `.inl:120-125` | — | fabric atomic-inc receiver sem | no |
| SEND: teardown | helper | `FabricStream::close()` (drains write + atomic barriers; idempotent RAII) | `ccl_helpers_dataflow.hpp:402`; `.inl:166-176` | — | — | no |
| RECV: "ready" one-shot | helper | `FabricStreamSender<>::signal(sender_num_hops, sender_sem_noc_addr)` | `ccl_helpers_dataflow.hpp:459`; `.inl:182-189` | reads fabric conn rt-args | fabric atomic-inc sender sem | yes (self-contained) |

### Host-side helpers (program-descriptor assembly)

| Phase | Type | Function | File:Line | Purpose |
|-------|------|----------|-----------|---------|
| Packet framing | helper | `ttnn._ttnn.fabric.ccl_packet_dims(dtype, page_size, num_pages, alignment)` | pybind `fabric.cpp:245-252`; impl `ccl_helpers_dataflow_host.hpp:74-96` | `packet_size_bytes / pages_per_packet / page_segments / total_packets`; owns bf16 `std::bit_floor` + both regimes |
| Route | helper | `ttnn._ttnn.fabric.ccl_dm_route(mesh_device, src, dst, topology)` | pybind `fabric.cpp:253-266`; impl `ccl_helpers_dataflow_host.hpp:102-166` | `num_hops / is_forward / neighbor_id`; owns fwd/bwd sign reversal + Ring short-way |
| Fabric conn rt-args | helper | `ttnn.setup_fabric_connection(src_id, neighbor_id, link_idx, program, core)` | pybind `fabric.cpp:141-178` | returns conn rt-args; mutates program (appends fabric semaphores) |
| Fabric rt-arg layout | contract | `append_ccl_fabric_rt_args` layout `[has_forward][fwd?][has_backward][bwd?]` | `ccl_helpers_dataflow_host.hpp:219-237` | reproduced in Python `_append_fabric_rt_args`; kernel reads `has_forward` at `conn_arg_idx` |
| Cross-device sem | helper | `ttnn.create_global_semaphore(mesh_device, cores, 0)` | pybind `global_semaphore.cpp:40-56` | op-internal semaphore over the worker grid |
| Sem address | helper | `ttnn.get_global_semaphore_address(sem)` | pybind `global_semaphore.cpp:58-67` | absolute `uint32` address baked into both kernels' rt-args |
| Alloc barrier | helper | `ttnn.synchronize_device(mesh_device)` | pybind `device.cpp:548-559` | ONCE after semaphore creation |
| Sem lifetime | helper | `mesh_pd.semaphores = [sem]` | pybind `program_descriptors.cpp:1077-1087` | framework keeps L1 alive across program-cache hits |
| Fabric node id | helper | `mesh_device.get_fabric_node_id(coord)` | `ttnn/core/distributed/distributed_nanobind.cpp` | src ids for `setup_fabric_connection` |
| Descriptor | helper | `ttnn.MeshProgramDescriptor()`, `mesh_pd[MeshCoordinateRange(c,c)] = program` | pybind `program_descriptors.cpp:990-1087` | two-program mesh workload |
| Dispatch | helper | `ttnn.generic_op([input, intermediate, output], mesh_pd)` | pybind `generic_op_nanobind.cpp:47-60` | runs the mesh workload; output tensor last |

### Raw APIs (fallbacks) — each with helpers considered and rejected

The kernel helper `ccl_helpers_dataflow.hpp` is a **fabric-egress-only** surface. Its banner explicitly enumerates what it does NOT own and hands back to the op. Each raw fallback below is one of those explicitly-excluded responsibilities.

| Phase | Type | Function | File:Line | What it does | In → Out |
|-------|------|----------|-----------|--------------|----------|
| SEND reader / RECV writer: interleaved page streaming | raw_api | `TensorAccessor` + `noc_async_read` / `noc_async_write` + barriers | kernels `point_to_point_sender_reader.cpp:34-36`, `point_to_point_receiver_writer.cpp:34-36` | stream shard pages DRAM/L1 ↔ L1 CB | input → `cb_input_pages` / `cb_output_pages` → output |
| RECV reader: receive ingress | raw_api | `noc_async_read` + `noc_async_read_barrier` | `point_to_point_receiver_reader.cpp:73-75` | local NoC read of a landed packet | intermediate → `cb_packet_scratch` |
| Both: page↔packet (de)coalescing | raw_api | `tt::data_movement::common::tt_memmove` | `.../data_movement/common/kernels/common.hpp`; used `point_to_point_sender_writer.cpp:83`, `point_to_point_receiver_reader.cpp:87` | copy page/segment into/out of the packet buffer | L1 ↔ L1 |
| Both: WAITING half + cache re-arm | raw_api | `noc_semaphore_wait_min` + `noc_semaphore_set(sem, 0)` | `point_to_point_sender_writer.cpp:64-65`, `point_to_point_receiver_reader.cpp:60,95` | wait on peer's inc; reset for cache reuse | — |
| Both: CB protocol | raw_api | `cb_reserve_back` / `cb_push_back` / `cb_wait_front` / `cb_pop_front` | all four kernels | inter-kernel L1 handoff | — |

**Helpers considered and rejected (mandatory justification):**

- **Fabric write / inc → could a helper own the whole send loop?** No. `UnicastWriteChannel::write_page` (`ccl_helpers_dataflow.hpp:283`) owns exactly the single armed fabric write from an *already-populated* L1 source address to a page index; it does not iterate pages, coalesce, or manage the source CB. The banner states the helper does NOT own "page<->packet coalescing/segmentation" (`ccl_helpers_dataflow.hpp:90-92`). Therefore the coalescing loop + `tt_memmove` + the `cb_input_pages` lifecycle are necessarily op code.
- **`noc_semaphore_wait_min` / `noc_semaphore_set` → is there a helper "wait/reset"?** No, by design. The banner: "The WAITING half is a plain local `noc_semaphore_wait_min(sem, threshold)` the op calls directly … a stock dataflow call, not renamed" and "CACHE-REUSE FOOTGUN: … each side must `noc_semaphore_set(sem, 0)` to re-arm." The helper deliberately owns only the SENDING half (`inc`/`multicast_inc`). Using `arm_inc` here would be wrong — it increments, it does not wait/reset. **Verified:** `ccl_helpers_dataflow.hpp:69-77`.
- **Receive ingress `noc_async_read` → is there a `FabricStreamReceiver`?** No, by design: "The receive INGRESS is likewise a local NoC read the op owns; there is no FabricStreamReceiver." **Verified:** `ccl_helpers_dataflow.hpp:74`.
- **Interleaved DRAM↔L1 page streaming (plain reader/writer) → any CCL helper?** No. `ccl_helpers_dataflow.hpp` is fabric egress only ("This is PURE DATA MOVEMENT … No compute … appears here"); local interleaved page streaming is the standard `reader/writer_unary_interleaved_start_id_gen` pattern with `TensorAccessor`, which the banner states is "consumed, never re-wrapped." **Verified:** `ccl_helpers_dataflow.hpp:92`.
- **CB reserve/push/wait/pop → any helper?** No. The CCL kernel helper touches no CBs (it takes/returns L1 addresses only); CB sync is intrinsic to the reader→writer pipeline and is the op's responsibility. **Verified:** `ccl_helpers_dataflow.hpp:89-93` (list of non-owned responsibilities).

## Dataflow Phases

Sequential execution across the two programs. "CB / sem state after" is noted where non-obvious.

| # | Phase | Device / Kernel | Consumes | Produces | State after |
|---|-------|-----------------|----------|----------|-------------|
| 0 | seed output from input | host, all devices | `input_tensor` | `output_tensor` == input on every device | output seeded; receiver shard overwritten in phase 6 |
| 1 | "ready" handshake | RECV reader | fabric conn rt-args | fabric atomic-inc → `sem@sender` | `sem@sender` will reach 1 |
| 2 | wait "ready" + reset | SEND writer | `sem@sender` | — | `sem@sender` reset to 0 (before its own inc) |
| 3 | read input shard | SEND reader | `input_tensor` | `cb_input_pages` (num_pages pushes) | up to 2 input pages buffered |
| 4 | coalesce + fabric write | SEND writer | `cb_input_pages` | `cb_packet_scratch` → `intermediate@receiver` | all `total_packets` written; `cb_input_pages` consumed |
| 5 | "done" handshake | SEND writer | — | fabric atomic-inc → `sem@receiver`; `close()` | sender complete; `sem@receiver` will reach 1 |
| 6 | wait "done" + local read + de-coalesce | RECV reader | `sem@receiver`, `intermediate` (local) | `cb_packet_scratch`, `cb_output_pages` (num_pages pushes) | de-coalesced pages streaming; `sem@receiver` reset to 0 (after its wait) |
| 7 | write output shard | RECV writer | `cb_output_pages` | `output@receiver` (overwrites seed) | `output[receiver]` == `input[sender]`; `cb_output_pages` consumed |

Phases 3 & 4 (SEND) pipeline through `cb_input_pages` and run concurrently with the RECV reader blocked at phase 6's wait; phase 5's "done" inc unblocks it. After phase 7 every device's `sem` copy is back to 0 (clean for the next cache hit).

## Validation / Error Cases

`validate()` raises before any dispatch (typed `UnsupportedAxisValue` / `ExcludedCell` from `ttnn.operations._op_contract` for axis refusals, `ValueError` for structural input errors):

| Condition | Error |
|-----------|-------|
| `input_tensor` not on a `ttnn.MeshDevice` | `ValueError` |
| mesh view not 2-D | `ValueError` |
| `sender_coord == receiver_coord` | `ValueError` (cannot send to self) |
| `sender_coord` or `receiver_coord` outside the mesh | `ValueError` |
| coords share neither row nor column (not a 1-D fabric route) | `ValueError` |
| `input_tensor.is_sharded()` (non-interleaved) | `ValueError` (not yet supported) |
| per-shard page size not 16-byte aligned | `ValueError` |
| `output_tensor` spec ≠ resolved output spec (shape/dtype/layout/buffer_type) | `ValueError` |
| `intermediate_tensor` spec ≠ `resolve_intermediate_spec(input_tensor)` | `ValueError` |
| axis value outside `SUPPORTED` (dtype/layout/topology/alignment) | `UnsupportedAxisValue` |
| combination in `EXCLUSIONS` | `ExcludedCell` |

`topology` is a finite op axis; `SUPPORTED["topology"] = [Linear, Ring]` and is taken verbatim (no index canonicalization applies). `alignment` is a shape-derived axis tagged by `tag_alignment` (both of the shard's last two dims divisible by 32 → `tile_aligned`); the op is byte-movement and alignment-agnostic, so both values are supported.

## Structural impossibilities (already captured in feature_spec.py — pipeline mode)

`eval/golden_tests/point_to_point/feature_spec.py` already exists with `INVALID` populated by `/golden-tests` (pipeline mode → authoritative; not edited by this design):

| INVALID cell | Rationale |
|--------------|-----------|
| `{dtype: bfloat8_b, layout: ROW_MAJOR_LAYOUT}` | `bfloat8_b` is a tiled block-float format with no row-major representation — a data-format-definition impossibility (single-tensor coupling: dtype + layout both describe the input). `ttnn` cannot construct this tensor. |

No additional op-specific structural impossibilities are needed. `topology` is orthogonal to `dtype`/`layout` (no coupling). The 16-byte page-size constraint is a shape×dtype `validate()` gate kept satisfiable by every INPUTS shard (last dim a multiple of 8), not modeled as an axis.

## Key Risks and Gotchas

- **Semaphore cache-reuse re-arm (highest risk).** The SEND writer must `noc_semaphore_set(sem, 0)` *before* its outgoing "done" inc; the RECV reader must reset *after* its "done" wait. Miss either → first call green, second call (program-cache hit) hangs or corrupts. Documented at `ccl_helpers_dataflow.hpp:74-77`.
- **GlobalSemaphore created ONCE, parked on the descriptor.** Create per `mesh_device` (cached), `synchronize_device` once, `mesh_pd.semaphores = [sem]`. Do NOT recreate per call and do NOT add a post-dispatch `synchronize_device` barrier — the framework owns the lifetime once parked.
- **Do NOT reimplement route math.** `ccl_dm_route` owns the fabric forward/backward **sign reversal** (`!is_forward`) and the Ring shorter-way choice. Call the bound helper on both endpoints (SEND uses `src=sender`, RECV uses `src=receiver`); do NOT re-derive direction.
- **Do NOT reimplement packet framing.** `ccl_packet_dims` owns the bf16 `std::bit_floor` max-packet rule and both coalesce/segment regimes. Never hardcode `4096`/`4416` — always call it and consume its outputs verbatim.
- **Intermediate is addressed PER-PACKET**, not per-shard-page: page index = `packet_idx`, page size overridden to `packet_size_bytes`. Its logical shape is `[total_packets, packet_size_bytes/4]` `uint32` RM (the `uint32` sidesteps `element_size` being undefined for `bfloat8_b`).
- **Output must be seeded == input on every device** (host `clone`/`copy`) before dispatch, so non-participating shards remain unchanged and the sender's own output shard is untouched (the SEND program never writes its output shard).
- **`cb_packet_scratch` is a private scratch buffer**, not a streaming CB: reserved once, held for the whole kernel, single page of `packet_size_bytes`. Its `total_size` must be exactly `packet_size_bytes` — do not add a balancing consumer wait.
- **`cb_input_pages` / `cb_output_pages` page size is the *aligned* page size** (`round_up(page_size, l1_alignment)`); the runtime page-size arg overrides the compile-time `AlignedPageSize` (which can be stale on program-cache hits).
- **Fabric-connection RT block layout is exact.** The word at `conn_arg_idx` (index 9) must be `int(is_forward)` (the kernel peeks it for direction); the rest follows the `[has_forward][fwd][has_backward][bwd]` layout. `setup_fabric_connection` must run after `ProgramDescriptor` construction (it mutates the program). No packet-header CB: `FabricStreamSender` uses `PacketHeaderPool`.
- **Verification topology is fixed.** The acceptance test MUST open a `(2, 4)` mesh with `fabric_config = ttnn.FabricConfig.FABRIC_1D`. A different mesh shape (e.g. `(1,2)`) hangs fabric init with `Fabric Router Sync: Timeout` — a test/topology mismatch, not an op defect. `topology=Ring` with adjacent 1-hop coords routes the line way (short-way is not shorter), so it runs correctly on `FABRIC_1D`.
- **1-D fabric routing only.** `sender_coord` and `receiver_coord` must share a row or column; a diagonal pair is rejected in `validate()`.
