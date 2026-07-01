# Operation Design: point_to_point

## Overview

| Field | Value |
|-------|-------|
| Classification | CCL (collective communication) — pure cross-chip data movement, NO arithmetic |
| Goal | Send one device's interleaved shard of a mesh-sharded tensor to another device over the Tenstorrent fabric, returning a tensor whose receiver-device shard now equals the sender's input shard. |
| Math | `output[d] = input[d]` for every device `d`, **except** `output[receiver_coord] = input[sender_coord]`. Element values unchanged end to end. |
| Mode | Derivative — newly-authored Python `generic_op` + `MeshProgramDescriptor` op with newly-authored sender/receiver dataflow kernels. The bound C++ `ttnn.point_to_point` is a correctness reference only and is NOT called, imported, or wrapped. |
| Devices participating | Exactly two: the program at `sender_coord` and the program at `receiver_coord`. Every other mesh device runs no program for the fabric transfer. |
| References | `ttnn/cpp/ttnn/kernel_lib/ccl_helpers_dataflow.hpp` (+`.inl`); `ttnn/cpp/ttnn/operations/ccl/common/host/ccl_helpers_dataflow_host.hpp`; `ttnn/cpp/ttnn-nanobind/fabric.cpp:235-266` (Python-bound `ccl_packet_dims`/`ccl_dm_route`); reference kernels `ttnn/cpp/ttnn/operations/point_to_point/device/kernels/dataflow/{writer_send,reader_receive,reader_unary_interleaved_start_id_gen,writer_unary_interleaved_start_id_gen}.cpp`; reference host `point_to_point_device_op.cpp:60-144`; Python assembly blueprint `tests/ttnn/unit_tests/operations/debug/test_generic_op.py:139-438`; ownership gold standard `ttnn/cpp/ttnn/operations/experimental/ccl/deepseek_moe_reduce_scatter/device/*`. |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | On a `ttnn.MeshDevice` (≥2 devices along the row/col connecting the coords); interleaved (DRAM or L1); rank ≥ 2; per-shard page size 16-byte aligned | — | host |
| `sender_coord` | `ttnn.MeshCoordinate` | yes | Inside the mesh; same row OR same column as `receiver_coord` (1-D fabric route) | — | host → routing RT args |
| `receiver_coord` | `ttnn.MeshCoordinate` | yes | Inside the mesh; `!= sender_coord` | — | host → routing RT args |
| `topology` | `ttnn.Topology` | no | `Linear`, `Ring` | `ttnn.Topology.Linear` | host → routing (`ccl_dm_route`) |
| `output_tensor` | `ttnn.Tensor \| None` | no | spec must equal the resolved output spec (= input spec) | `None` | host |
| `intermediate_tensor` | `ttnn.Tensor \| None` | no | spec must equal the resolved intermediate spec | `None` | host |

There are no compute-kernel template params. All op-specific values reach the kernels as runtime args (packet dims, hop count, semaphore address, fabric-connection block); the CB indices and alignment reach them as compile-time args.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | per-device logical shard shape, rank ≥ 2 |
| Dtype | `bfloat16` (primary), `float32`, `bfloat8_b` (TILE only), `uint16`, `int32`, `uint32` |
| Layout | `TILE_LAYOUT` (primary) or `ROW_MAJOR_LAYOUT` |
| Memory | interleaved (DRAM or L1); sharded/non-interleaved rejected |
| Distribution | sharded across a `MeshDevice`; `sender_coord` and `receiver_coord` resolve to two distinct mesh devices on a common row/column |

### Output

| Property | Value |
|----------|-------|
| Shape | identical to input's per-device shard shape |
| Dtype | same as input |
| Layout | same as input |
| Memory | same memory config as input (interleaved) |
| Content | `output[receiver_coord]` == `input[sender_coord]` bit-for-bit; `output[d]` == `input[d]` (unchanged) for every other device, including the sender |

### Intermediate (op-internal staging tensor, lands fabric packets on the receiver)

| Property | Value |
|----------|-------|
| Shape | `[total_packets, packet_page_dim]` (2-D), where `packet_page_dim = packet_size_bytes // element_size(dtype)` |
| Dtype | same as input |
| Layout | same `TensorLayout` (layout + dtype + interleaved memory of input's `buffer_type`) as the output — matches `point_to_point_device_op.cpp:77-82` |
| Memory | interleaved, same `buffer_type` (DRAM/L1) as input |
| Role | A raw landing buffer addressed **per-packet** (page index = `packet_idx`, page size **overridden** to `packet_size_bytes` in both endpoints' `TensorAccessor`). Allocated by the op or supplied via `intermediate_tensor`. Holds `total_packets` packets of `packet_size_bytes` bytes. |

## Multi-Device Distribution

The op dispatches a single `ttnn.generic_op([input_tensor, intermediate_tensor, output_tensor], mesh_program_descriptor)` where `mesh_program_descriptor` is a `ttnn.MeshProgramDescriptor` holding **exactly two** `(MeshCoordinateRange, ProgramDescriptor)` entries:

| Mesh coordinate | Program | Kernels | Role |
|-----------------|---------|---------|------|
| `MeshCoordinateRange(sender_coord, sender_coord)` | send program | `sender_reader` (NCRISC), `sender_writer` (BRISC) | read input shard → coalesce → fabric-write packets to receiver's intermediate → "done" inc |
| `MeshCoordinateRange(receiver_coord, receiver_coord)` | receive program | `receiver_reader` (NCRISC), `receiver_writer` (BRISC) | "ready" inc → wait "done" → read intermediate locally → de-coalesce → write output shard |

No `ProgramDescriptor` is added for any other coordinate, so only the two endpoint devices execute kernels (mandate honored).

### Output-content seeding (the "all other shards unchanged" contract)

The fabric transfer only writes the receiver device's output shard. To guarantee `output[d] == input[d]` on every non-receiver device (including the sender's own output shard, which no kernel writes), the op **seeds the output tensor from the input** before dispatching the two-device program:

- `output_tensor is None`: allocate output (spec == input spec), then device-copy `input → output` on every device (e.g. `ttnn.clone(input_tensor)` produces the seeded output directly).
- `output_tensor` provided: device-copy `input → output_tensor` (specs already validated equal), then dispatch.

The seed is a standard per-device data-movement op enqueued **before** the `generic_op` on the same mesh command queue, so it completes first; the receiver's writer then overwrites only the receiver shard. This is the only way to make criterion "non-participating shards unchanged" hold for both call paths. (Documented in Key Risks — callers that only need the receiver shard may skip the seed, leaving non-receiver shards undefined.)

## Cross-Device Coordination

One op-internal `GlobalSemaphore` (`sem`), created over the worker grid, drives the full handshake. Its **absolute L1 address** (same on every device) is baked into both programs' runtime args. Each device owns its local copy at that address; fabric atomic-incs cross the chip boundary.

Lifecycle (mirrors `point_to_point_device_op.cpp:102-128` and the `ccl_helpers_dataflow.hpp:61-74` banner):

1. Create `sem = ttnn.create_global_semaphore(mesh_device, worker_cores, 0)` **once per `mesh_device`** (cache it on the module), then `ttnn.synchronize_device(mesh_device)` **once** right after creation. Reuse the same `sem` (stable address) on every later call.
2. `sem_addr = ttnn.get_global_semaphore_address(sem)` → runtime arg to both programs.
3. Park it: `mesh_program_descriptor.semaphores = [sem]`. The framework keeps its L1 allocation alive across program-cache hits. **No** per-call post-dispatch `synchronize_device` barrier is added.

Handshake ordering contract (both directions use the same `sem` address; the **waiting** half is a plain `noc_semaphore_wait_min` the op owns; the **inc** half is `AtomicIncChannel::inc` over fabric):

| Step | Device | Action |
|------|--------|--------|
| 1 | receiver | open fabric stream toward sender; `arm_inc(unicast_route(hops_recv), 1)`; `inc(sem@sender)` — signals "ready"; close ack stream |
| 2 | sender | `noc_semaphore_wait_min(sem_local, 1)` — waits for "ready" |
| 3 | sender | `noc_semaphore_set(sem_local, 0)` — **reset BEFORE its own outgoing inc** (cache-reuse footgun) |
| 4 | sender | open stream toward receiver; `arm_unicast_write(route, packet_size)`; write all packets; `arm_inc(route,1)` |
| 5 | sender | `inc(sem@receiver)` — signals "done"; `close()` |
| 6 | receiver | `noc_semaphore_wait_min(sem_local, 1)` — waits for "done" (payload fully landed in intermediate) |
| 7 | receiver | de-coalesce intermediate → output CB |
| 8 | receiver | `noc_semaphore_set(sem_local, 0)` — **reset AFTER its wait** (cache-reuse footgun) |

Each device's local `sem` copy goes `0 → 1 → 0`, so a program-cache hit re-arms cleanly. `noc_semaphore_wait_min` is order-insensitive (a counting inc that arrives early persists), so neither signal is ever lost.

## Dataflow Strategy

The shard travels `sender DRAM/L1 → sender L1 CBs → fabric → receiver intermediate (DRAM/L1) → receiver L1 CBs → receiver DRAM/L1`. Format is preserved end to end (the op never tilizes/untilizes — it copies bytes). A single shard page may be larger or smaller than one fabric transfer unit; `ccl_packet_dims` frames this into two regimes (multiple pages per packet, or one page split into segments) and the kernels coalesce/segment accordingly.

### Send program

| Kernel | RISC / config | Reads | Writes | Helpers |
|--------|---------------|-------|--------|---------|
| `sender_reader` | NCRISC / `ReaderConfigDescriptor` | `input_tensor` (interleaved DRAM/L1) via `TensorAccessor` | `cb_input_pages` | `TensorAccessor` (raw); pattern of `reader_unary_interleaved_start_id_gen.cpp:9-37` |
| `sender_writer` | BRISC / `WriterConfigDescriptor` | `cb_input_pages` | `cb_packet_scratch` (L1), then `intermediate_tensor` on receiver via fabric | `FabricStreamSender`/`FabricStream`/`UnicastWriteChannel`/`AtomicIncChannel`, `unicast_route`, `TensorAccessor`; `tt_memmove` (raw coalesce); `noc_semaphore_wait_min`/`set` (raw); pattern of `writer_send.cpp:13-94` |

`sender_writer` algorithm: handshake steps 2–3 → `open()` → `arm_unicast_write(unicast_route(hops_send), packet_size_bytes)` + `arm_inc(route, 1)` → loop over the `num_pages` input pages: `cb_wait_front(cb_input_pages,1)`, for each page segment `tt_memmove` the source page (or segment) into `cb_packet_scratch` at the running page offset, and when the packet fills (`pages_per_packet`, or every segment when segmented) issue `UnicastWriteChannel::write_page(packet_base, packet_idx, intermediate_addrgen)`; `cb_pop_front`. After the loop, handshake step 5: `AtomicIncChannel::inc(get_noc_addr(sem_addr))` then `close()`.

### Receive program

| Kernel | RISC / config | Reads | Writes | Helpers |
|--------|---------------|-------|--------|---------|
| `receiver_reader` | NCRISC / `ReaderConfigDescriptor` | `intermediate_tensor` (local, after "done") via `TensorAccessor`; fabric ack | `cb_packet_scratch` (L1 landing), `cb_output_pages` | `FabricStreamSender`/`arm_inc`/`inc`, `unicast_route`; `noc_async_read`/`noc_async_read_barrier` (raw ingress); `tt_memmove` (raw de-coalesce); `noc_semaphore_wait_min`/`set` (raw); pattern of `reader_receive.cpp:12-86` |
| `receiver_writer` | BRISC / `WriterConfigDescriptor` | `cb_output_pages` | `output_tensor` (interleaved DRAM/L1) via `TensorAccessor` | `TensorAccessor` (raw); pattern of `writer_unary_interleaved_start_id_gen.cpp:7-34` |

`receiver_reader` algorithm: handshake step 1 (open ack stream toward sender, `arm_inc(unicast_route(hops_recv),1)`, `inc(get_noc_addr(sem_addr))`, close) → reserve `cb_packet_scratch` → handshake step 6 (`noc_semaphore_wait_min(sem_local,1)`) → loop over `num_pages` output pages: when a fresh packet is needed (`packet_idx` advances), `noc_async_read(intermediate.get_noc_addr(packet_idx,0,0), packet_l1, packet_size_bytes)` + barrier; `tt_memmove` the page (or segment) out of the packet buffer into `cb_output_pages`; `cb_push_back`. After the loop, handshake step 8 (`noc_semaphore_set(sem_local,0)`).

### Kernel-side fabric-connection arg contract (non-derivable — pin it)

Both fabric-using kernels build their connection with `FabricStreamSender<>(conn_arg_idx, is_forward, alignment)`, where `conn_arg_idx` points at the start of the fabric-connection runtime block and the **leading word of that block is `has_forward` = `int(is_forward)`** (the kernel peeks it for the direction, then the ctor consumes the whole block). The host must lay the block out exactly as `append_ccl_fabric_rt_args` does (`ccl_helpers_dataflow_host.hpp:219-237`):

```
[ has_forward = int(is_forward) ]
[ <forward connection args from setup_fabric_connection> ]   # only if is_forward
[ has_backward = int(not is_forward) ]
[ <backward connection args from setup_fabric_connection> ]  # only if not is_forward
```

In Python (per `test_generic_op.py:308-422`): write `int(is_forward)` as the word at `conn_arg_idx`, then append `int(not is_forward)` and `extend(ttnn.setup_fabric_connection(src_fabric_id, neighbor_fabric_id, link_idx, program, core))` in the order dictated by which flag is set. `setup_fabric_connection` also mutates the `ProgramDescriptor` (appends `SemaphoreDescriptor`s), so it must be called after the `ProgramDescriptor` is constructed.

## Host-Side Assembly

| Step | API | Source / Formula |
|------|-----|------------------|
| page metrics | `input_tensor.buffer_page_size()`, `input_tensor.buffer_num_pages()` | per-device shard page size & count |
| packet framing | `ttnn._ttnn.fabric.ccl_packet_dims(input_tensor.dtype, page_size_bytes, num_pages, 16)` | → `.packet_size_bytes`, `.pages_per_packet`, `.page_segments`, `.total_packets` (owns bf16 `bit_floor` + both regimes) |
| intermediate spec | `packet_page_dim = packet_size_bytes // ttnn.element_size(dtype)`; shape `[total_packets, packet_page_dim]`; `ttnn.TensorSpec(shape, dtype, layout, buffer_type=input.memory_config().buffer_type)` | interleaved staging tensor |
| send route | `ttnn._ttnn.fabric.ccl_dm_route(mesh_device, sender_coord, receiver_coord, topology)` | → `.num_hops`(send), `.is_forward`(send dir), `.neighbor_id` (toward receiver). Owns fwd/bwd sign reversal + ring short-way. |
| recv route | `ttnn._ttnn.fabric.ccl_dm_route(mesh_device, receiver_coord, sender_coord, topology)` | → `.num_hops`(ack), `.is_forward`(ack dir), `.neighbor_id` (toward sender) |
| fabric node ids | `mesh_device.get_fabric_node_id(sender_coord)`, `...(receiver_coord)` | src ids for `setup_fabric_connection` |
| semaphore | `ttnn.create_global_semaphore(mesh_device, worker_cores, 0)` once + `ttnn.synchronize_device` once; `ttnn.get_global_semaphore_address(sem)` | op-internal, cached, parked on `mesh_program_descriptor.semaphores` |
| connection args | `ttnn.setup_fabric_connection(src_id, neighbor_id, link_idx=0, program, core)` | appends conn RT args + mutates program; laid out per the arg contract above |
| dispatch | `ttnn.generic_op([input_tensor, intermediate_tensor, output_tensor], mesh_program_descriptor)` | output tensor last in the IO list |

`worker_cores`: build from `mesh_device.compute_with_storage_grid_size()` via `ttnn.num_cores_to_corerangeset(...)` (matches `test_generic_op.py:195-198`). The transfer itself uses a single core (`CoreCoord(0,0)`) on each endpoint (see Work Distribution).

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | the entire per-device shard (`num_pages` pages → `total_packets` packets) |
| Grid | single core `CoreCoord(0,0)` on each endpoint (`sender_core_set`, `receiver_core_set` each a 1×1 `CoreRangeSet`) |
| Per-core work | sender core: all `num_pages` input pages and all `total_packets` fabric writes; receiver core: all `total_packets` local reads and `num_pages` output pages |
| Remainder | none — one core owns all packets. The last packet may carry fewer than `pages_per_packet` pages (`curr_pages_per_packet = min(pages_per_packet, pages_remaining)`); the segmented regime always uses `pages_per_packet == 1`. |

Multi-link / multi-core fan-out is out of scope for this op (the reference C++ factory is likewise single-core); the design leaves room for it (per-core packet partitioning) but does not implement it.

## Circular Buffers

### Send program (on `sender_core`)

| Semantic Name | Index | Page Size | Num Pages | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|--------|----------|----------|----------|
| `cb_input_pages` | 0 | `align(buffer_page_size(), 16)` | 2 | input dtype | `sender_reader` | `sender_writer` | streaming double-buffer (one input page in flight) |
| `cb_packet_scratch` | 24 | `packet_size_bytes` | 1 | input dtype | `sender_writer` (reserve+push once) | none (working L1 for coalesced packet) | whole-op scratch buffer |

### Receive program (on `receiver_core`)

| Semantic Name | Index | Page Size | Num Pages | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|--------|----------|----------|----------|
| `cb_packet_scratch` | 24 | `packet_size_bytes` | 1 | input dtype | `receiver_reader` (reserve once, `noc_async_read` lands packets here) | none (working L1) | whole-op scratch buffer |
| `cb_output_pages` | 16 | `align(buffer_page_size(), 16)` | `3 * pages_per_packet` | input dtype | `receiver_reader` | `receiver_writer` | streaming buffer (de-coalesced output pages) |

No packet-header CB is declared: `FabricStreamSender` draws headers from the fabric-L1 `PacketHeaderPool` (`tt_metal/fabric/hw/inc/packet_header_pool.h`), a fixed reserved region — confirmed self-provisioning, no program-side CB needed (`send_program_factory.cpp:63-65`, `receive_program_factory.cpp:48-50`).

CB sync check (push == wait):
- `cb_input_pages`: `sender_reader` pushes `num_pages`; `sender_writer` waits/pops `num_pages`. ✔
- `cb_output_pages`: `receiver_reader` pushes `num_pages`; `receiver_writer` waits/pops `num_pages`. ✔
- `cb_packet_scratch` (both): used as a scratch L1 region (reserve once, no cross-kernel consumer); not a producer/consumer queue, so no wait to balance. ✔

## API Mapping

Every mechanism — helper or raw — with an exact file:line reference.

| Phase | Type | Function | File:Line | Args / Notes | In CB | Out CB |
|-------|------|----------|-----------|--------------|-------|--------|
| host: packet framing | helper (Py-bound) | `ttnn._ttnn.fabric.ccl_packet_dims` | `ttnn/cpp/ttnn-nanobind/fabric.cpp:245-252`; impl `ccl_helpers_dataflow_host.hpp:74-96` | `(dtype, page_size_bytes, num_pages, 16)` | — | — |
| host: routing | helper (Py-bound) | `ttnn._ttnn.fabric.ccl_dm_route` | `ttnn/cpp/ttnn-nanobind/fabric.cpp:253-266`; impl `ccl_helpers_dataflow_host.hpp:137-166` | `(mesh_device, src_coord, dst_coord, topology)` | — | — |
| host: fabric conn args | helper (Py-bound) | `ttnn.setup_fabric_connection` | `ttnn/cpp/ttnn-nanobind/fabric.cpp:141-178` | `(src_id, dst_id, link_idx, program, core)`; mutates program, returns RT arg vector | — | — |
| host: semaphore | helper (Py-bound) | `ttnn.create_global_semaphore` / `get_global_semaphore_address` / `synchronize_device` | C++ host equivalent `ccl_helpers_dataflow_host.hpp:250-256` (`make_ccl_semaphore`) | once-per-mesh, parked on `mesh_program_descriptor.semaphores` | — | — |
| host: fabric node id | helper (Py-bound) | `mesh_device.get_fabric_node_id` | `ttnn/core/distributed/distributed_nanobind.cpp:272` | `(coord)` | — | — |
| host: dispatch | helper (Py-bound) | `ttnn.generic_op` over `ttnn.MeshProgramDescriptor` | `ttnn/cpp/ttnn/operations/generic/generic_op_nanobind.cpp`; descriptors `ttnn/cpp/ttnn-nanobind/program_descriptors.cpp` | `([input, intermediate, output], mesh_pd)` | — | — |
| kernel: fabric connection | helper | `FabricStreamSender<>(conn_arg_idx, is_forward, alignment)` → `.open()` | `ccl_helpers_dataflow.hpp:425-439` | reads the conn RT block; binds direction | — | — |
| kernel: route build | helper | `unicast_route(num_hops)` | `ccl_helpers_dataflow.hpp:253-258` | builds `line_unicast_route_info_t` | — | — |
| kernel: arm write | helper | `FabricStream::arm_unicast_write(route, packet_size_bytes)` | `ccl_helpers_dataflow.hpp:365-366`; impl `.inl:23-38` | sets invariant payload size + route | — | — |
| kernel: issue write | helper | `UnicastWriteChannel::write_page(src_l1, packet_idx, intermediate_addrgen)` | `ccl_helpers_dataflow.hpp:277`; impl `.inl:47-52` | per-packet fabric write to intermediate | `cb_packet_scratch` | `intermediate_tensor` |
| kernel: arm inc | helper | `FabricStream::arm_inc(route, 1)` | `ccl_helpers_dataflow.hpp:379-380`; impl `.inl:105-119` | shared sem header | — | — |
| kernel: issue inc | helper | `AtomicIncChannel::inc(remote_sem_noc_addr)` | `ccl_helpers_dataflow.hpp:310`; impl `.inl:121-126` | cross-device semaphore inc | — | — |
| kernel: drain/close | helper | `FabricStream::drain()` / `close()` | impl `.inl:160-172` | teardown; `close()` idempotent | — | — |
| kernel: addressing | raw_api | `TensorAccessor(args, base_addr, page_size_override)` / `.get_noc_addr(idx,0,0)` | used `writer_send.cpp:39`, `reader_receive.cpp:43,63` | page-size override = `packet_size_bytes` for intermediate | — | — |
| kernel: handshake wait/reset | raw_api | `noc_semaphore_wait_min` / `noc_semaphore_set` | `writer_send.cpp:53-54`, `reader_receive.cpp:50,85` | the WAITING half + cache-reuse reset | — | — |
| kernel: receive ingress | raw_api | `noc_async_read` / `noc_async_read_barrier` | `reader_receive.cpp:64-65` | local read of landed packet | `intermediate_tensor` | `cb_packet_scratch` |
| kernel: page↔packet (de)coalesce | raw_api | `tt_memmove` | `writer_send.cpp:73`, `reader_receive.cpp:77` | byte move within L1 | — | — |
| kernel: input read | raw_api | `noc_async_read` + `TensorAccessor` | `reader_unary_interleaved_start_id_gen.cpp:31-33` | input shard → CB | `input_tensor` | `cb_input_pages` |
| kernel: output write | raw_api | `noc_async_write` + `TensorAccessor` | `writer_unary_interleaved_start_id_gen.cpp:28-31` | CB → output shard | `cb_output_pages` | `output_tensor` |

### Helpers considered and rejected (for each raw-API fallback)

- **`noc_semaphore_wait_min` / `noc_semaphore_set` (handshake WAITING half + re-arm).** Candidate: `ccl_helpers_dataflow.hpp` (`AtomicIncChannel` / `MulticastIncChannel`). Rejected because the helper **owns only the SENDING half** of a cross-device sync; the banner at `ccl_helpers_dataflow.hpp:61-69` states the WAITING half is "a plain local `noc_semaphore_wait_min(sem, threshold)` the op calls directly" and the cache-reuse `noc_semaphore_set(sem, 0)` re-arm is explicitly an op responsibility. No helper API exists for either; using `arm_inc` here would be wrong (it increments, it does not wait/reset).
- **`noc_async_read` / `noc_async_read_barrier` (receive ingress).** Candidate: a fabric receive helper. Rejected because there is no `FabricStreamReceiver` — `ccl_helpers_dataflow.hpp:66-67` and `:82-86` state "the receive INGRESS is likewise a local NoC read the op owns; there is no FabricStreamReceiver." The helper deliberately does not own page↔packet coalescing/segmentation.
- **`tt_memmove` (page↔packet coalesce / de-coalesce).** Candidate: a fabric coalescing helper. Rejected because `ccl_helpers_dataflow.hpp:82-84` lists "page<->packet coalescing/segmentation" among "What the helper does NOT own (the op composes it)." It is an L1→L1 byte move with no fabric-helper equivalent.
- **`TensorAccessor` (DRAM/L1 addressing).** Candidate: an addrgen helper. Rejected because `ccl_helpers_dataflow.hpp:84-85` states "address generation (TensorAccessor/ShardedAddrGen) is consumed, never re-wrapped" — `TensorAccessor` IS the intended primitive; the fabric helper consumes it (`write_page` takes an addrgen).
- **`noc_async_read`/`noc_async_write` for input read & output write (`sender_reader`, `receiver_writer`).** These kernels are pure interleaved DRAM↔L1 streaming with no fabric and no compute; no kernel-library helper covers a plain interleaved page reader/writer — the established idiom is the `reader_unary_interleaved_start_id_gen.cpp` / `writer_unary_interleaved_start_id_gen.cpp` pattern, used verbatim by the reference op.

## Dataflow Phases

| # | Phase | Device | Consumes | Produces | CB / sem state after |
|---|-------|--------|----------|----------|----------------------|
| 0 | seed output from input | all | `input_tensor` | `output_tensor` == input on every device | output seeded; receiver shard will be overwritten in phase 6 |
| 1 | ready handshake | receiver | — | fabric inc → `sem@sender` | `sem@sender` will reach 1 |
| 2 | wait ready + reset | sender | `sem@sender` | — | `sem@sender` reset to 0 |
| 3 | read input shard | sender | `input_tensor` | `cb_input_pages` (streaming) | up to 2 input pages buffered |
| 4 | coalesce + fabric write | sender | `cb_input_pages` | `cb_packet_scratch` → `intermediate_tensor`@receiver | all `total_packets` written to receiver intermediate |
| 5 | done handshake | sender | — | fabric inc → `sem@receiver`; `close()` | sender complete; `sem@receiver` will reach 1 |
| 6 | wait done + local read + de-coalesce | receiver | `sem@receiver`, `intermediate_tensor` (local) | `cb_packet_scratch`, `cb_output_pages` | de-coalesced pages streaming to writer; `sem@receiver` reset to 0 |
| 7 | write output shard | receiver | `cb_output_pages` | `output_tensor`@receiver (overwrites seed) | `output[receiver]` == `input[sender]` |

After phase 7, every device's `sem` copy is back to 0 (clean for the next cache hit); `output[receiver] == input[sender]`, `output[d] == input[d]` elsewhere.

## Validation

`validate()` raises (typed `UnsupportedAxisValue` / `ExcludedCell` from `ttnn.operations._op_contract`, or `ValueError`/`RuntimeError` for structural input errors) for:

| Condition | Error |
|-----------|-------|
| `input_tensor` not on a `MeshDevice` | reject ("input must be on a MeshDevice") |
| `sender_coord == receiver_coord` | reject ("cannot send to self") |
| `sender_coord` or `receiver_coord` outside the mesh view | reject ("coordinate outside mesh") |
| `sender_coord` / `receiver_coord` not on a common row or column | reject (1-D fabric route undefined — `ccl_dm_route` also throws) |
| input sharded / non-interleaved | reject ("sharded input not yet supported") |
| per-shard page size not 16-byte aligned (`buffer_page_size() % 16 != 0`, unless `== 16`) | reject ("page size must be 16-byte aligned") |
| `output_tensor` provided with shape/dtype/layout/memory_config != resolved output spec | reject ("output spec mismatch") |
| `intermediate_tensor` provided with spec != resolved intermediate spec | reject ("intermediate spec mismatch") |

`dim`-style index canonicalization is not applicable (no index axis). `topology` is taken verbatim (`Linear`/`Ring`).

## Verification Topology

The op is verified on a simulated multi-chip mesh via `scripts/run_multidevice_sim_pytest.py`. The graded (`grade_primary`, `required`) entry is `bh_8xP150_p2p` in `scripts/multidevice_sim_topologies.yaml`:

| Field | Value |
|-------|-------|
| arch | Blackhole (8× P150, all-MMIO) |
| mesh shape | **(2, 4)** — 8 chips |
| mesh-graph descriptor | `blackhole_8xP150_torus_x_mesh_graph_descriptor.textproto` (torus wraparound in x/columns) |
| fabric_config | **`ttnn.FabricConfig.FABRIC_1D`** (set by the test, not the descriptor) |

**The acceptance test's `mesh_device` fixture MUST open exactly `(2, 4)` with `FABRIC_1D`.** The mesh-graph descriptor fixes the mesh shape; opening any other shape (e.g. `(1, 2)`) hangs fabric init with `Fabric Router Sync: Timeout` — a test/topology mismatch, not a sim or op defect. Sender/receiver coords are picked inside the mesh: `(0, 0) → (0, 1)` (adjacent, row 0, a valid 1-D route).

Both Linear and Ring op-topologies are exercised under the SAME `FABRIC_1D` fabric config. For an adjacent coord pair, `ccl_dm_route(..., Ring)` computes `line_hops = 1` and `ring_hops = 1 + mesh_shape[dim] = 5`; since `|5| > |1|` it falls through to the 1-hop line route (`ccl_helpers_dataflow_host.hpp:156-165`). So Ring degenerates to the Linear route for adjacent coords and is safely routable under `FABRIC_1D`. A genuine ring wraparound (short-way over the torus, e.g. `(0,0) → (0,3)` = 1 backward hop) requires `FABRIC_1D_RING` and is covered by throwaway sim confirmation tests, not the graded acceptance suite.

Per-device shards are produced with `ttnn.ShardTensorToMesh(mesh_device, dim=0)` and read back per-device with `ttnn.get_device_tensors(...)` → `ttnn.to_torch(...)`; the linear (row-major) index of a `MeshCoordinate` is `coord[0] * mesh_shape[1] + coord[1]`. The oracle is identity: `to_torch(out_shard[receiver]) == to_torch(in_shard[sender])` (bit-exact for integer dtypes, PCC for float / block-float), and every non-receiver shard equals its own input.

## Key Risks and Gotchas

- **Semaphore reset order is load-bearing.** Sender resets BEFORE its outgoing "done" inc; receiver resets AFTER its "done" wait (`ccl_helpers_dataflow.hpp:67-69`). Missing or mis-ordered resets pass run 1 and hang/corrupt run 2 (program-cache reuse).
- **Create the GlobalSemaphore once.** Create + `synchronize_device` exactly once per mesh, cache the handle on the module, and park it in `mesh_program_descriptor.semaphores`. Do NOT recreate per call and do NOT add a per-call post-dispatch `synchronize_device`.
- **Fabric-connection RT block layout is exact.** The word at `conn_arg_idx` must be `int(is_forward)` (the kernel peeks it for direction); the rest follows the `[has_forward][fwd][has_backward][bwd]` layout of `append_ccl_fabric_rt_args`. `setup_fabric_connection` must run after `ProgramDescriptor` construction (it mutates the program).
- **`ccl_dm_route` already applies the fwd/bwd sign reversal and ring short-way.** Use `.is_forward` / `.num_hops` / `.neighbor_id` directly; do NOT re-derive direction. Compute the ack route with the coords swapped (`receiver_coord, sender_coord`).
- **Intermediate is addressed per-packet with an overridden page size.** Both endpoints construct `TensorAccessor` with the third arg = `packet_size_bytes`, and index by `packet_idx`. The intermediate's natural layout (TILE/RM) only governs allocation size, which must be ≥ `total_packets * packet_size_bytes`; following the reference (same `TensorLayout` as output) guarantees this.
- **No packet-header CB.** `FabricStreamSender` uses `PacketHeaderPool` (fixed fabric L1). The vestigial `packet_header_cb` in `test_generic_op.py` is not used by the kernels and must not be carried over.
- **`cb_packet_scratch` is a working buffer, not a queue.** Sized to one packet; reserved once. Do not add a balancing consumer wait.
- **bf16 packet sizing uses `bit_floor`.** `ccl_packet_dims` shrinks the max packet to a power of two for bf16; never hardcode `4096`/`4416` — always call `ccl_packet_dims`.
- **Output seeding cost.** The `input → output` device copy runs every call. It is required to satisfy "non-participating shards unchanged"; a caller needing only the receiver shard could skip it (non-receiver shards then undefined).

## Structural impossibilities (for feature_spec.py INVALID)

- `{dtype: bfloat8_b, layout: ROW_MAJOR_LAYOUT}` — `bfloat8_b` is a tiled block-float format with no row-major representation (single-tensor coupling; universe-must-change → INVALID). This is the only structural impossibility; `topology` is orthogonal to dtype/layout, and the 16-byte page-size constraint is a shape×dtype validate() gate (kept satisfiable by INPUTS, not modeled as an axis).
