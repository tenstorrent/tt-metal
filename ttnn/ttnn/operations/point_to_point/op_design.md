# Operation Design: point_to_point

> Self-contained Python op built on `ttnn.generic_op` + `ttnn.MeshProgramDescriptor`, with
> newly-authored sender/receiver dataflow kernels under
> `ttnn/ttnn/operations/point_to_point/kernels/dataflow/`. It does **not** wrap, import, call, or
> dispatch to the bound C++ op `ttnn.point_to_point` / `ttnn._ttnn.operations.point_to_point`. The
> bound C++ op and its kernels are read as a **correctness reference only**.

## Overview

| Field | Value |
|-------|-------|
| Classification | CCL / data_movement (pure cross-chip copy — NO arithmetic) |
| Goal | Copy the shard living on `sender_coord` to `receiver_coord` over the Tenstorrent fabric. The returned tensor's receiver-device shard becomes bit-identical to the sender-device input shard; every other device's shard is untouched. |
| Math | `output[receiver_coord] = input[sender_coord]` (identity copy); `output[d] = undefined/preserved` for `d != receiver_coord` |
| Mode | Derivative (mirrors the proven C++ `point_to_point` send/receive program structure on the generic-op surface) |
| References | C++ reference op: `ttnn/cpp/ttnn/operations/point_to_point/` (kernels `device/kernels/dataflow/writer_send.cpp`, `reader_receive.cpp`; factories `device/host/{send,receive}_program_factory.cpp`; `device/host/point_to_point_device_op.cpp`). Kernel helper: `ttnn/cpp/ttnn/kernel_lib/ccl_helpers_dataflow.hpp`. Host helpers: `ttnn/cpp/ttnn/operations/ccl/common/host/ccl_helpers_dataflow_host.hpp`. Generic-op template: `tt_metal/third_party/tt_ops_code_gen/references/generic_op_template/`. |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | sharded across a `MeshDevice`, interleaved (DRAM or L1), rank ≥ 2 | — | tensor |
| `sender_coord` | `ttnn.MeshCoordinate` | yes | inside mesh; shares a row or column with `receiver_coord` | — | host (route) |
| `receiver_coord` | `ttnn.MeshCoordinate` | yes | inside mesh; `!= sender_coord` | — | host (route) |
| `topology` | `ttnn.Topology` | no | `Linear` (primary) or `Ring` | `ttnn.Topology.Linear` | host (route) |
| `output_tensor` | `ttnn.Tensor \| None` | no | spec must equal resolved output spec | `None` (allocate) | tensor |
| `intermediate_tensor` | `ttnn.Tensor \| None` | no | spec must equal resolved intermediate spec | `None` (allocate) | tensor |

The function signature is exactly:

```python
from ttnn.operations.point_to_point import point_to_point

point_to_point(
    input_tensor: ttnn.Tensor,
    sender_coord: ttnn.MeshCoordinate,
    receiver_coord: ttnn.MeshCoordinate,
    topology: ttnn.Topology = ttnn.Topology.Linear,
    output_tensor: ttnn.Tensor | None = None,
    intermediate_tensor: ttnn.Tensor | None = None,
) -> ttnn.Tensor
```

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | rank ≥ 2; this is the per-device (logical) shard shape |
| Dtype | bfloat16 (primary), float32, bfloat8_b (TILE only), uint16, int32, uint32 |
| Layout | TILE (primary) or ROW_MAJOR |
| Memory | interleaved, DRAM or L1 (sharded memory layout rejected) |
| Mesh | lives on a `ttnn.MeshDevice` with ≥ 2 devices along the row/column joining sender and receiver |

### Output

| Property | Value |
|----------|-------|
| Shape | `== input_tensor.shape` (per-device shard shape; identical on every device) |
| Dtype | `== input_tensor.dtype` |
| Layout | `== input_tensor.layout` |
| Memory | `== input_tensor.memory_config()` |
| Contents | `output@receiver_coord == input@sender_coord` (bit-for-bit); `output@d` for other `d` is uninitialized (fresh alloc) or preserved (when `output_tensor` is supplied) |

### Intermediate (fabric landing buffer — transient, never returned to the caller)

| Property | Value |
|----------|-------|
| Shape | `(total_packets, packet_size_bytes / element_size_bytes)` — 2-D, mirrors the C++ reference (`point_to_point_device_op.cpp` compute_output_specs) |
| Dtype | `== input_tensor.dtype` |
| Layout | `== input_tensor.layout` |
| Memory | `== input_tensor.memory_config()` |
| Allocation | allocated mesh-wide so every device's copy sits at the same address; only the receiver device's copy is written (by the sender, over the fabric) and read (locally, by the receiver) |

`total_packets`, `packet_size_bytes`, `pages_per_packet`, `page_segments` come from
`ttnn._ttnn.fabric.ccl_packet_dims(dtype, page_size_bytes, num_pages, l1_alignment)` — the same host
helper the kernels' framing assumes, so host and kernel agree on the framing exactly.

## Validation Rules (raised before any device work, in `validate()`)

| # | Condition | Error |
|---|-----------|-------|
| 1 | `input_tensor` not on a `MeshDevice` | input must be on a MeshDevice |
| 2 | `sender_coord == receiver_coord` | cannot send to self |
| 3 | `sender_coord` or `receiver_coord` not contained in the mesh view | coordinate outside mesh |
| 4 | sender/receiver share neither a row nor a column (not 1-D routable) | coordinates not 1-D-fabric routable |
| 5 | input is sharded (non-interleaved memory layout) | sharded configs not yet supported |
| 6 | `page_size_bytes % 16 != 0` and `page_size_bytes != 16` (`l1_alignment = ttnn` HAL L1 alignment = 16) | page size must be 16-byte aligned |
| 7 | `output_tensor` provided and its shape/dtype/layout/memory_config ≠ resolved output spec | output_tensor spec mismatch |
| 8 | rank < 2 | rank must be ≥ 2 |

`page_size_bytes = input_tensor.tensor_spec().compute_page_size_bytes()` (the per-page byte size:
tile bytes for TILE, last-dim×element bytes for ROW_MAJOR). `dim` canonicalization is N/A (no `dim`
parameter).

## Dataflow Strategy

This is a **dataflow-only** op (no compute/unpack/math/pack kernels). Exactly two devices run a
program; every other mesh device runs nothing. The `MeshProgramDescriptor` holds two
`(MeshCoordinateRange, ProgramDescriptor)` entries:

| Mesh entry | Range | Program |
|------------|-------|---------|
| send | `MeshCoordinateRange(sender_coord, sender_coord)` | reader_send + writer_send |
| receive | `MeshCoordinateRange(receiver_coord, receiver_coord)` | reader_receive + writer_receive |

### Per-Tensix data path (reader → CB → writer)

**Sender Tensix (1 worker core):**
1. `reader_send` walks the input shard's pages via a `TensorAccessor` over `input_tensor` and streams them into `cb_sender_pages` (NoC0 reads from DRAM/L1 → L1 CB).
2. `writer_send` is the fabric egress. It waits for the receiver's "ready" inc on the shared semaphore, resets the semaphore to 0 (cache-reuse re-arm — **before** its own inc), opens a one-direction fabric connection, programs the unicast route, arms the write + inc channels, then loops: pop pages from `cb_sender_pages`, **coalesce** them into `cb_packet_scratch` (op-owned page→packet framing), and `FabricStreamSender::write_page(...)` each full packet into the receiver's **intermediate** buffer. After the last packet it `inc`s the receiver's semaphore ("done") and closes.

**Receiver Tensix (1 worker core):**
1. `reader_receive` first signals "ready" to the sender (a brief fabric `arm_inc`+`inc`, then close), then `noc_semaphore_wait_min(sem, 1)` for the sender's "done". Once the payload has fully landed in the local intermediate buffer, it reads each packet locally (`noc_async_read` from the intermediate `TensorAccessor`) and **de-coalesces** packet pages into `cb_receiver_pages`. It resets the semaphore to 0 **after** the wait (cache-reuse re-arm).
2. `writer_receive` pops pages from `cb_receiver_pages` and writes them to the `output_tensor` via a `TensorAccessor` (L1 CB → DRAM/L1).

### Tensix-to-Tensix (cross-device) contract

| Aspect | Decision |
|--------|----------|
| Transport | TT fabric, 1-D unicast. Route (`num_hops`, `is_forward`, `neighbor_id`) computed host-side by `ccl_dm_route(mesh_device, sender, receiver, topology)`; the receiver computes the reverse route (`ccl_dm_route(mesh_device, receiver, sender, topology)`) for its ready-inc. The helper owns the fabric forward/backward sign reversal and the Ring short-way choice. |
| Payload target | The sender's fabric writes land in the **receiver's intermediate buffer** (same mesh address on both devices). Receive ingress is a **local** `noc_async_read`, intentionally op-owned (helper banner `ccl_helpers_dataflow.hpp:54-60`). |
| Coordination | One `GlobalSemaphore`, created fresh each call, addressed identically on both devices. Receiver `inc`s the sender's copy → "ready"; sender `inc`s the receiver's copy → "done". Both via `FabricStreamSender::inc` over the fabric; each side waits on its **local** copy with `noc_semaphore_wait_min(sem, 1)`. |
| Ordering guarantee | Sender does not transmit payload until it has seen "ready"; receiver does not consume the intermediate until it has seen "done". This is a strict 2-party handshake (threshold = 1). |
| Cache-reuse reset | Sender resets the semaphore to 0 **before** its outgoing "done" inc; receiver resets **after** its wait. (Footgun: missing reset = first run green, second run hangs/corrupts — `ccl_helpers_dataflow.hpp:61-64`.) |
| Semaphore lifetime | The Python op holds the `GlobalSemaphore` Python object alive across the `generic_op` call and runs `ttnn.synchronize_device(mesh_device)` **before returning**, so the device finishes consuming the semaphore before its L1 allocation is freed at function return — safe across program-cache hits. |

### Host↔kernel runtime-arg contract (fabric block framing)

The fabric connection runtime args are appended with `ttnn.setup_fabric_connection(...)`, wrapped in
the `has_forward / has_backward` framing that the kernel-side `FabricStreamSender` consumes (the
Python equivalent of the C++ `append_ccl_fabric_rt_args`, `ccl_helpers_dataflow_host.hpp:219-237`):

```
[ ...op-specific RT args... ][ has_forward ][ fwd conn args if fwd ][ has_backward ][ bwd conn args if bwd ]
                              ^ conn_arg_idx (the kernel records this index; its leading flag == send direction)
```

The op-specific RT-arg prefixes (mirroring the proven reference kernels) are:

| Kernel | RT args (in order), then fabric block at the next index |
|--------|---------------------------------------------------------|
| `reader_send` | `input_buffer_addr` (Buffer*), `num_pages`, `page_idx_start`, `input_page_size_bytes` — no fabric block |
| `writer_send` | `intermediate_buffer_addr` (Buffer*), `page_idx_start`, `page_idx_end`, `dst_num_hops`, `page_size_bytes`, `packet_size_bytes`, `pages_per_packet`, `page_segments`, `sem_addr`, **then fabric block at index 9** |
| `reader_receive` | `page_idx_start`, `page_idx_end`, `pages_per_packet`, `intermediate_buffer_addr` (Buffer*, idx 3), `packet_size_bytes`, `output_page_size_bytes`, `page_segments`, `sem_addr`, `sender_num_hops`, **then fabric block at index 9** |
| `writer_receive` | `output_buffer_addr` (Buffer*), `num_pages`, `page_idx_start`, `output_page_size_bytes` — no fabric block |

Compile-time args (scalars first, `TensorAccessorArgs` last per convention):

| Kernel | CT args |
|--------|---------|
| `reader_send` | `TensorAccessorArgs(input_tensor)` |
| `writer_send` | `cb_sender_pages`, `cb_packet_scratch`, `l1_alignment`, then `TensorAccessorArgs(intermediate_tensor)` |
| `reader_receive` | `cb_packet_landing`, `cb_receiver_pages`, `l1_alignment`, then `TensorAccessorArgs(intermediate_tensor)` |
| `writer_receive` | `cb_receiver_pages`, then `TensorAccessorArgs(output_tensor)` |

`ttnn.setup_fabric_connection(src_fabric_node_id, neighbor_fabric_node_id, link_idx=0, program_descriptor, worker_core)`
both returns the connection RT-arg `uint32_t` list **and** appends the connection `SemaphoreDescriptor`s
to the program descriptor — so it must be called on the same `ProgramDescriptor` the kernel belongs to.
`src_fabric_node_id = mesh_device.get_fabric_node_id(local_coord)`; `neighbor_fabric_node_id` is the
`.neighbor_id` returned by `ccl_dm_route`.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one fabric packet (a coalesced group of `pages_per_packet` pages, or one page split into `page_segments`) |
| Grid | a single worker core `{1, 1}` (logical core `(0,0)`) per program — single-link implementation, matching the C++ reference (`send_program_factory.cpp:39`) |
| Per-core work | all `total_packets` packets on the one core (`split_work_to_cores({1,1}, total_packets)` → group 1 gets everything) |
| Remainder | the last packet carries `min(pages_per_packet, pages_remaining)` pages; `page_idx_end` is clamped to `input_num_pages`. The `GlobalSemaphore` is allocated over the full mesh worker grid (`make_ccl_semaphore` pattern) so its address is valid on the worker core. |

Multi-link / multi-core fan-out is a future refinement; the design fixes a single link (`link_idx = 0`).

## Circular Buffers

### Send program (at `sender_coord`)

| Semantic Name | Index | Page Size | Num Pages | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|--------|----------|----------|----------|
| `cb_sender_pages` | c_0 | `round_up(input_page_size_bytes, 16)` | 2 (double-buffer streaming) | input dtype | `reader_send` (NoC read of input shard) | `writer_send` (coalesce + fabric write) | per call |
| `cb_packet_scratch` | c_1 | `packet_size_bytes` | 1 (scratch staging) | input dtype | `writer_send` (reserved once for a stable L1 packet-staging address) | `writer_send` | per call |

### Receive program (at `receiver_coord`)

| Semantic Name | Index | Page Size | Num Pages | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|--------|----------|----------|----------|
| `cb_packet_landing` | c_0 | `packet_size_bytes` | 1 (scratch staging) | input dtype | `reader_receive` (local NoC read of one landed packet) | `reader_receive` | per call |
| `cb_receiver_pages` | c_1 | `output_page_size_bytes` | `3 * pages_per_packet` | output dtype | `reader_receive` (de-coalesce) | `writer_receive` (NoC write to output) | per call |

CB sync ledger:
- `cb_sender_pages`: `reader_send` pushes `num_pages`; `writer_send` waits/pops `num_pages` → equal.
- `cb_receiver_pages`: `reader_receive` pushes `num_pages`; `writer_receive` waits/pops `num_pages` → equal.
- `cb_packet_scratch` / `cb_packet_landing`: single-owner scratch (reserved once for a stable L1 base address; no cross-kernel producer/consumer pairing). No sync obligation beyond the single owner.

## API Mapping

Every mechanism — helper or raw — has an exact file:line reference.

| Phase | Type | Function | File:Line | Args / Notes | Reads | Writes | Owns CB ops? |
|-------|------|----------|-----------|--------------|-------|--------|--------------|
| Host: packet framing | helper | `ttnn._ttnn.fabric.ccl_packet_dims(dtype, page_size_bytes, num_pages, alignment)` | pybind `ttnn/cpp/ttnn-nanobind/fabric.cpp:245-252`; impl `ccl_helpers_dataflow_host.hpp:74-96` | returns `.packet_size_bytes / .pages_per_packet / .page_segments / .total_packets`; owns bf16 `bit_floor` + both packing regimes | — | — | n/a (host) |
| Host: route | helper | `ttnn._ttnn.fabric.ccl_dm_route(mesh_device, sender, receiver, topology)` | pybind `ttnn/cpp/ttnn-nanobind/fabric.cpp:253-266`; impl `ccl_helpers_dataflow_host.hpp:137-166` | returns `.num_hops / .is_forward / .neighbor_id`; owns fwd/bwd sign reversal + Ring short-way | — | — | n/a (host) |
| Host: fabric RT args | helper | `ttnn.setup_fabric_connection(src_id, dst_id, link_idx, program_descriptor, worker_core, core_type=WORKER)` | pybind `ttnn/cpp/ttnn-nanobind/fabric.cpp:142-178` | returns conn RT-arg list; appends conn `SemaphoreDescriptor`s to the descriptor; wrapped in `has_forward/has_backward` framing | — | mutates `ProgramDescriptor` | n/a (host) |
| Host: semaphore | helper | `ttnn.create_global_semaphore(mesh_device, cores, 0)` / `ttnn.get_global_semaphore_address(sem)` | pybind `ttnn/cpp/ttnn-nanobind/global_semaphore.cpp:40-56` / `:58-67` | fresh each call; `cores` = `CoreRangeSet` over the worker grid; address passed as RT arg to both programs | — | — | n/a (host) |
| Host: barrier | helper | `ttnn.synchronize_device(mesh_device)` | pybind `ttnn/cpp/ttnn-nanobind/device.cpp:548-559` | post-dispatch barrier keeping the sem alive across cache hits | — | — | n/a (host) |
| Host: fabric node id | helper | `mesh_device.get_fabric_node_id(coord)` | pybind `ttnn/core/distributed/distributed_nanobind.cpp:272` | source FabricNodeId for `setup_fabric_connection` | — | — | n/a (host) |
| Host: dispatch | helper | `ttnn.generic_op(io_tensors, mesh_program_descriptor)` | pybind `ttnn/cpp/ttnn/operations/generic/generic_op_nanobind.cpp:46-60`; returns `io_tensors.back()` (`generic_op_device_operation.cpp:124-136`) | `io_tensors = [input, intermediate, output]`; op returns `output_tensor` explicitly | — | — | n/a (host) |
| Host: descriptors | helper | `MeshProgramDescriptor` / `ProgramDescriptor` / `KernelDescriptor` / `CBDescriptor` / `CBFormatDescriptor` | `tt_metal/api/tt-metalium/experimental/mesh_program_descriptor.hpp:15-22`; `program_descriptors.hpp:{212-223,126-210,69-82,61-67}` | two `(MeshCoordinateRange, ProgramDescriptor)` entries | — | — | n/a (host) |
| Host: addrgen CT args | helper | `ttnn.TensorAccessorArgs(tensor).get_compile_time_args()` | `tt_metal/api/tt-metalium/tensor_accessor_args.hpp:18-68` | appended LAST in each kernel's CT args | — | — | n/a (host) |
| Sender fabric egress | helper | `dataflow_kernel_lib::ccl::FabricStreamSender<>` — `open`/`set_route_unicast(num_hops)`/`arm_unicast_write(payload)`/`arm_inc(1)`/`write_page(...)`/`inc(noc_addr)`/`close` | `ccl_helpers_dataflow.hpp:279-421` (`open:299`, `close:302`, `set_route_unicast:309`, `arm_unicast_write:324`, `write:331`, `write_page:339`, `arm_inc:348`, `inc:355`) | armed-channel model; helper owns header alloc + `UpdateMask`; consumes a `TensorAccessor` for `write_page` | `cb_sender_pages` (via op coalesce) | receiver intermediate buffer + receiver sem | No — op owns CB pop and coalescing |
| Receiver ready-inc | helper | `FabricStreamSender<>` — `open`/`set_route_unicast`/`arm_inc(1)`/`inc(noc_addr)`/`close` | same as above | brief egress only | — | sender sem | No |
| Input shard read | raw_api | `TensorAccessor` + `noc_async_read` + `noc_async_read_barrier` (`reader_send`) | `tech_reports/tensor_accessor/tensor_accessor.md`; `api/dataflow/dataflow_api.h` | DRAM/L1 page → `cb_sender_pages` | input tensor | `cb_sender_pages` | op owns `cb_reserve_back`/`cb_push_back` |
| Packet landing read | raw_api | `TensorAccessor::get_noc_addr` + `noc_async_read` + `noc_async_read_barrier` (`reader_receive`) | same | local read of one landed packet | intermediate tensor | `cb_packet_landing` | op-owned scratch |
| Page↔packet (de)coalesce | raw_api | `tt::data_movement::common::tt_memmove` | `ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp` | L1→L1 copy with offset; framing math is op-owned | scratch/CB | scratch/CB | op-owned |
| Output shard write | raw_api | `TensorAccessor` + `noc_async_write` + `noc_async_write_barrier` (`writer_receive`) | `api/dataflow/dataflow_api.h` | `cb_receiver_pages` page → DRAM/L1 | `cb_receiver_pages` | output tensor | op owns `cb_wait_front`/`cb_pop_front` |
| Cross-device wait/reset | raw_api | `noc_semaphore_wait_min(sem, 1)` + `noc_semaphore_set(sem, 0)` | `api/dataflow/dataflow_api.h` | 2-party handshake threshold = 1; reset per cache-reuse rule | local sem | local sem | op-owned |

### Helpers considered and rejected (for each raw-API fallback)

- **Input shard read / Packet landing read / Output shard write (raw NoC + TensorAccessor).**
  Considered: `FabricStreamSender` and the compute-side `kernel_lib` helpers (tilize/reduce/eltwise).
  Rejected — `ccl_helpers_dataflow.hpp:72-76` states explicitly that "address generation
  (TensorAccessor/ShardedAddrGen is **consumed, never re-wrapped**)" and that page↔packet
  coalescing/segmentation is **not** owned by the helper; and `ccl_helpers_dataflow.hpp:54-60` states
  the receive **ingress is a local NoC read the op owns** (there is intentionally no
  `FabricStreamReceiver`). The compute `kernel_lib` helpers do not apply — this op performs no
  compute (`ccl_helpers_dataflow.hpp:20`: "PURE DATA MOVEMENT. No compute/unpack/math/pack").
- **Page↔packet (de)coalesce (`tt_memmove`).** Considered: `FabricStreamSender::write_page`. It frames
  and sends one packet but does not assemble multiple pages into a packet or split a page across
  packets — `ccl_helpers_dataflow.hpp:73` lists "page<->packet coalescing/segmentation" as op-owned.
  `tt_memmove` is the L1→L1 staging copy the framing loop requires.
- **Cross-device wait/reset (`noc_semaphore_wait_min` / `noc_semaphore_set`).** Considered: folding the
  wait into `FabricStreamSender`. Rejected — `ccl_helpers_dataflow.hpp:54-64` says the local wait/reset
  "stay op-owned … stock dataflow-API calls, so the op calls them directly rather than through a
  renamed wrapper," and documents the cache-reuse reset placement (sender before its inc, receiver
  after its wait) as the op's responsibility.

## Dataflow Phases

| # | Program / Kernel | Consumes | Produces | Connecting CB | State after |
|---|------------------|----------|----------|---------------|-------------|
| 0 | host | input spec | route (`ccl_dm_route`), packet dims (`ccl_packet_dims`), fresh `GlobalSemaphore`, intermediate + output tensors, two-program `MeshProgramDescriptor` | — | descriptors built; sem at 0 on both devices |
| 1 | receive / `reader_receive` (egress) | — | "ready" inc on sender's sem | — | sender's sem == 1 |
| 2 | send / `writer_send` | sender's sem == 1 | resets sem→0; arms fabric channels | — | sender re-armed; fabric open |
| 3 | send / `reader_send` | input shard pages | streamed pages | `cb_sender_pages` | pages flowing to writer |
| 4 | send / `writer_send` | `cb_sender_pages` | packets coalesced via `cb_packet_scratch`; fabric-written into receiver intermediate | `cb_packet_scratch` | intermediate filling on receiver |
| 5 | send / `writer_send` | (after last packet) | "done" inc on receiver's sem; close | — | receiver's sem == 1 |
| 6 | receive / `reader_receive` (ingress) | receiver's sem == 1 | local reads of landed packets, de-coalesced pages; resets sem→0 | `cb_packet_landing` → `cb_receiver_pages` | pages flowing to writer |
| 7 | receive / `writer_receive` | `cb_receiver_pages` | output shard pages written to `output_tensor` | `cb_receiver_pages` | receiver output shard complete |
| 8 | host | — | `ttnn.synchronize_device` barrier; return `output_tensor` | — | sem object safely freed after barrier |

## Program-Cache & Semaphore Lifecycle

- The `GlobalSemaphore` is created **fresh on every call** (internally — never a caller argument),
  its absolute address (`get_global_semaphore_address`) passed as an RT arg to both programs.
- Custom program hashing must key the cached program on shape/dtype/layout/coords/topology so the
  second call with identical parameters is a **cache hit**. The semaphore address changes per call;
  bind it as a runtime arg (not compile-time) so cache hits patch it cleanly.
- `ttnn.synchronize_device(mesh_device)` runs **before** returning, guaranteeing the device finished
  consuming the semaphore before the Python `GlobalSemaphore` object (and its L1 allocation) is freed
  at function return. This is what makes the fresh-sem-per-call pattern safe across cache hits.
- In-kernel semaphore reset (sender before its "done" inc; receiver after its wait) re-arms the
  reused semaphore for the next cached invocation.

## Key Risks and Gotchas

- **Cache-reuse semaphore reset (the central footgun).** Missing or mis-placed `noc_semaphore_set(sem,0)`
  passes on the first run and hangs/corrupts on the second. Sender resets **before** its outgoing inc;
  receiver resets **after** its wait (`ccl_helpers_dataflow.hpp:61-64`).
- **Semaphore object lifetime.** The Python op must keep the `GlobalSemaphore` alive until after the
  `synchronize_device` barrier. Returning before the barrier frees the L1 allocation mid-flight.
- **Intermediate buffer must be mesh-allocated** so its address matches on sender and receiver; the
  fabric write targets that shared address. Allocate it (or validate a supplied one) against the
  resolved intermediate spec `(total_packets, packet_size_bytes/element_size)`.
- **Page↔packet framing is op-owned** — handle both regimes: many pages per packet (small pages) and
  one page split across `page_segments` packets (large pages). The host `ccl_packet_dims` and the
  kernel loops must use the same `packet_size_bytes` / `pages_per_packet` / `page_segments`.
- **`packet_size_bytes` page override on the intermediate `TensorAccessor`.** Both sender (write) and
  receiver (read) construct the intermediate `TensorAccessor` with the page size **overridden to
  `packet_size_bytes`** so they compute identical NoC addresses; the override also dodges a stale
  `AlignedPageSize` on program-cache hits (reference kernels: `writer_send.cpp:39`,
  `reader_receive.cpp:44`).
- **Route direction is reversed per endpoint.** Sender uses `ccl_dm_route(mesh, sender, receiver, topo)`;
  receiver uses `ccl_dm_route(mesh, receiver, sender, topo)` for its ready-inc. Both must agree on the
  route; the helper owns the fwd/bwd sign reversal and the Ring short-way.
- **bfloat16 packet sizing differs** (`std::bit_floor` of the fabric channel buffer) — do not hand-size
  packets; always go through `ccl_packet_dims`.
- **1-D routability.** Sender and receiver must share a row or column; reject otherwise (the 1-D
  routing vector throws on a diagonal pair).

## Structural impossibilities (feature_spec `INVALID`)

This op is authored in **local mode**, so `eval/golden_tests/point_to_point/feature_spec.py` declares
`INVALID` directly. Candidates (single-tensor coupling, universe-must-change):

- `{dtype: bfloat8_b, layout: ROW_MAJOR_LAYOUT}` — bfloat8_b is a block-quantized format defined only
  over 32×32 tiles; ROW_MAJOR has no blocks. The data-format definition itself would have to change →
  INVALID, not a refinement candidate.

## Hardware Constraints

- [x] CB sync: push count = wait count for `cb_sender_pages` and `cb_receiver_pages`; the two
  packet-scratch CBs are single-owner staging buffers (reserved once for a stable L1 base).
- [x] No reduce — scaler-format constraints N/A.
- [x] No compute — DEST register limits N/A (no unpack/math/pack).
- [x] Page sizes aligned: tile CBs use tile bytes; RM/packet CBs use `round_up(page_size, 16)` /
  `packet_size_bytes`; input page size validated 16-byte-aligned.
- [x] All `cb_wait_front` on a given CB use the same page count (1 page per iteration on the streaming CBs).
- [x] Helpers are not wrapped with extra CB operations — `FabricStreamSender` owns its packet headers
  and `UpdateMask`; the op owns only the page→packet coalescing and CB pop/push around it.
- [x] Cross-device handshake uses a fresh `GlobalSemaphore`, reset per the cache-reuse rule, kept alive
  by a post-dispatch `synchronize_device` barrier.
- [x] Fabric RT-arg block framed as `[has_forward][fwd][has_backward][bwd]` at the documented
  `conn_arg_idx`, matching `FabricStreamSender`'s connection-build contract.
