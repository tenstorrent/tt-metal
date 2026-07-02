# Operation Design: point_to_point

## Overview

| Field | Value |
|---|---|
| Classification | **CCL** (collective communication — multi-device, pure data movement, NO arithmetic) |
| Goal | Copy one mesh device's interleaved shard of a mesh-sharded tensor to another device over the Tenstorrent fabric. After the op the **receiver** device's shard equals the **sender** device's input shard bit-for-bit; every other device's shard is unchanged. |
| Math | `out[receiver] = in[sender]` ; `out[d] = in[d]` for every `d != receiver` (identity / byte copy, PCC ~1.0) |
| Mode | **Hybrid** — a self-contained Python op built on `ttnn.generic_op` + `ttnn.MeshProgramDescriptor` with newly authored sender/receiver dataflow kernels under `ttnn/ttnn/operations/point_to_point/kernels/`. Does NOT wrap / import / call the bound C++ `ttnn.point_to_point`. |
| References | Kernel helper: `ttnn/cpp/ttnn/kernel_lib/ccl_helpers_dataflow.hpp` (+`.inl`). Host helpers: `ttnn/cpp/ttnn/operations/ccl/common/host/ccl_helpers_dataflow_host.hpp`, Python fabric bindings `ttnn/cpp/ttnn-nanobind/fabric.cpp`. Python template op: `ttnn/ttnn/operations/all_gather/all_gather.py` + `all_gather_program_descriptor.py`. Correctness reference (read-only): C++ op `ttnn/cpp/ttnn/operations/point_to_point/device/*`. Semaphore-ownership gold standard: `ttnn/cpp/ttnn/operations/experimental/ccl/deepseek_moe_reduce_scatter/device/*`. |

This op is pure data movement: **no compute kernel**. Both endpoints run only dataflow (reader/writer) kernels. Cross-chip coordination uses ONE op-internal `GlobalSemaphore`, created once, parked on `MeshProgramDescriptor.semaphores`.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|---|---|---|---|---|---|
| `input_tensor` | `ttnn.Tensor` | yes | interleaved, rank ≥ 2, on a `ttnn.MeshDevice` (≥2 devices on the row/col connecting the coords) | — | tensor |
| `sender_coord` | `ttnn.MeshCoordinate` | yes | inside the mesh; shares a row or a column with `receiver_coord` | — | host (route + program placement) |
| `receiver_coord` | `ttnn.MeshCoordinate` | yes | inside the mesh; `!= sender_coord` | — | host (route + program placement) |
| `topology` | `ttnn.Topology` | no | `Linear` (primary) or `Ring` (routes short-way) | `ttnn.Topology.Linear` | host (route computation) |
| `output_tensor` | `ttnn.Tensor \| None` | no | spec must equal the resolved output spec (== input spec) | `None` → **in-place alias of `input_tensor`** | tensor |
| `intermediate_tensor` | `ttnn.Tensor \| None` | no | spec must equal the resolved intermediate spec | `None` → freshly allocated | tensor |

## Tensors

### Input

| Property | Requirement |
|---|---|
| Shape | per-device shard shape, rank ≥ 2 |
| Dtype | `bfloat16` (primary), `float32`, `bfloat8_b` (TILE only), `uint16`, `int32`, `uint32` |
| Layout | `TILE_LAYOUT` (primary) or `ROW_MAJOR_LAYOUT` |
| Memory | interleaved, DRAM or L1. Physical page size (`buffer_page_size()`) MUST be 16-byte (L1) aligned |

### Output (final, returned)

| Property | Value |
|---|---|
| Shape | identical to the input's per-device shard shape |
| Dtype | same as input |
| Layout | same as input |
| Memory | same memory config as input |
| Semantics | `output_tensor is None` → **the op aliases `input_tensor` (in-place)**: only the receiver device's shard is overwritten (with the sender's data); the sender's own shard and every non-participating device's shard are never written and stay bit-identical to the input. `output_tensor` supplied → the receiver device's shard is written with the sender's data and the handle is returned. |

### Intermediate (packet-framed staging, internal)

| Property | Value |
|---|---|
| Shape | `{total_packets, packet_page_dim}` where `packet_page_dim = packet_size_bytes / datum_size(dtype)` |
| Dtype / Layout | same as input |
| Memory | same buffer type as input; mesh tensor (allocated across the whole mesh → **same device-local address on both endpoints**) |
| Role | The fabric landing zone. The sender fabric-writes packet-framed data into the **receiver device's** copy of this buffer; the receiver reads its **local** copy back and de-frames it into the output shard. Decouples "fabric transfer in packet units" from "output in page units". |

`total_packets`, `packet_size_bytes`, `pages_per_packet`, `page_segments` all come from `ccl_packet_dims(dtype, buffer_page_size, buffer_num_pages, l1_alignment)` (see API Mapping). Two regimes:
- **page ≤ packet**: `pages_per_packet = min(max_packet/aligned_page, num_pages)`, `page_segments = 1` — multiple shard pages coalesced per fabric transfer.
- **page > packet**: `pages_per_packet = 1`, `page_segments = ceil(aligned_page/max_packet)`, `packet_size = max_packet` — one shard page split across several fabric transfers.

## Validation (raise before dispatch)

`validate()` runs on the entry path (and on program-cache miss). Each check raises `ValueError` (structural) or the registry-model `UnsupportedAxisValue` / `ExcludedCell` (axis refusal). Order the structural checks first.

| # | Condition | Error |
|---|---|---|
| 1 | `input_tensor.device()` is not a `ttnn.MeshDevice` | `ValueError` — "input must be on a MeshDevice" |
| 2 | `sender_coord == receiver_coord` | `ValueError` — "cannot send to self" |
| 3 | `sender_coord` not inside the mesh view | `ValueError` — "sender_coord outside mesh" |
| 4 | `receiver_coord` not inside the mesh view | `ValueError` — "receiver_coord outside mesh" |
| 5 | coords do not share a row or a column (diagonal — invalid for 1-D fabric) | `ValueError` — "coords must share a row or column" |
| 6 | `input_tensor.is_sharded()` (non-interleaved) | `ValueError` — "sharded input not yet supported" |
| 7 | `input_tensor.buffer_page_size() % 16 != 0` | `ValueError` — "page size must be 16-byte aligned" |
| 8 | `output_tensor` supplied and its shape / dtype / layout / memory_config != resolved output spec (== input spec) | `ValueError` — "output_tensor spec must equal input spec" |
| 9 | `intermediate_tensor` supplied and its spec != resolved intermediate spec | `ValueError` — "intermediate_tensor spec mismatch" |
| 10 | axis gate: `dtype` / `layout` / `topology` not in `SUPPORTED` | `UnsupportedAxisValue` |

Registry-model op-file declarations (implementer authors these in `point_to_point.py`, paired with the authoritative `eval/golden_tests/point_to_point/feature_spec.py`):
- `INPUT_TAGGERS = {"alignment": tag_alignment}` where `tag_alignment(inputs, axes)` returns `"tile_aligned"` iff the shard's last two dims are both divisible by 32, else `"non_tile_aligned"`.
- `SUPPORTED` — the Phase-0 proven subset of TARGET (e.g. `dtype: [bfloat16, float32]`, `layout: [TILE, ROW_MAJOR]`, `topology: [Linear]`, `alignment: [tile_aligned, non_tile_aligned]`); values in `TARGET - SUPPORTED` are refinement candidates.
- `EXCLUSIONS = []` (no in-SUPPORTED not-yet-implemented cells expected).
- `topology` has no index/sign convention to canonicalize; the coords are mesh objects, not indices.

## Dataflow Strategy

Two `ProgramDescriptor`s on ONE `MeshProgramDescriptor`, keyed to exactly two single-coordinate `MeshCoordinateRange`s:

- **Send program** at `sender_coord` — one worker core `(0,0)`, two dataflow kernels.
- **Receive program** at `receiver_coord` — one worker core `(0,0)`, two dataflow kernels.

No program runs on any other mesh coordinate.

Data path (DRAM/L1 → Tensix → fabric → Tensix → DRAM/L1):

```
SENDER device (sender_coord)                         RECEIVER device (receiver_coord)
------------------------------                       ---------------------------------
input shard (interleaved pages)                      output shard  <-- (in-place = input shard)
   | sender_reader (NCRISC): TensorAccessor              ^ receiver_writer (BRISC): TensorAccessor
   |   noc_async_read page -> cb_shard_pages             | cb_shard_out -> noc_async_write page
   v                                                     |
cb_shard_pages (L1, double-buffered)                 cb_shard_out (L1, pipelined)
   | sender_writer (BRISC):                              ^ receiver_reader (NCRISC): de-coalesce
   |   tt_memmove coalesce pages -> cb_packet_send        | tt_memmove split packet -> output pages
   |   FabricStream.write_page(packet, addrgen) =========>| noc_async_read local intermediate page
   v          (fabric unicast, num_hops)               cb_packet_recv (L1 scratch, 1 packet)
cb_packet_send (L1 scratch, 1 packet)                    ^
                                                     receiver's INTERMEDIATE buffer copy
                                                     (fabric write lands here)
```

The fabric egress on the sender (`FabricStream.write_page`) computes the destination NoC address from the intermediate tensor's `TensorAccessor` at the intermediate's base address (identical on all devices) and the fabric routes it `num_hops` to the receiver device. The receiver's ingress is a plain local `noc_async_read` of its own intermediate copy.

### Tensix-to-Tensix (cross-chip) contract

| Aspect | Value |
|---|---|
| Who sends to whom | sender core `(0,0)` fabric-unicasts to receiver core `(0,0)`, `num_hops` away along the shared row/column |
| What is transferred | packet-framed shard bytes into the receiver's **intermediate** buffer (payload), plus two 1-count fabric atomic-incs on the shared `GlobalSemaphore` (control) |
| Synchronization | one shared cross-device `GlobalSemaphore` carries a two-phase handshake: (1) receiver → sender "ready", (2) sender → receiver "done" |
| Ordering guarantee | receiver must not read the payload until the sender's "done" inc arrives (payload writes complete before the inc via `drain`/`close`); sender must not transmit until the receiver's "ready" inc arrives (so the intermediate buffer is allocated/landable) |
| Route agreement | both endpoints derive the route from `ccl_dm_route`; sender uses `(sender→receiver)`, receiver uses `(receiver→sender)` for its ack. `ccl_dm_route` owns the fwd/bwd sign reversal and the Ring short-way choice |

## Work Distribution

| Aspect | Value |
|---|---|
| Work unit | one packet (a coalesced group of shard pages, or one segment of a large shard page) |
| Grid | single worker core `(0,0)` on each of the two participating devices; single fabric link (`link_idx = 0`) |
| Per-core work | the full transfer: `page_idx_start = 0`, `page_idx_end = buffer_num_pages`; iterates all `total_packets` packets |
| Remainder | the last packet of a run may carry fewer pages than `pages_per_packet` — the sender recomputes `curr_pages_per_packet = min(max_pages_per_packet, remaining_pages)` per packet; the receiver mirrors it |

A single core/single link is sufficient and matches the reference; multi-link/multi-core is a future refinement (not in scope).

## Circular Buffers

CBs are per-program (each device has its own L1). Index convention: 0–7 input, 16–23 output, 24–31 intermediate.

### Send program (at `sender_coord`)

| Semantic Name | Index | Page Size | Num Pages | Format | Producer | Consumer | Lifetime |
|---|---|---|---|---|---|---|---|
| `cb_shard_pages` | 0 | `aligned_page = round_up(buffer_page_size, 16)` | 2 (double-buffer streaming) | input dtype | sender_reader (NCRISC) | sender_writer (BRISC) | streamed page-by-page; producer pushes `buffer_num_pages`, consumer waits/pops `buffer_num_pages` |
| `cb_packet_send` | 24 | `packet_size_bytes` | 1 | input dtype | sender_writer (self, scratch) | sender_writer (self) | L1 scratch to assemble one coalesced packet; reserved+pushed once at start, reused for every packet |

### Receive program (at `receiver_coord`)

| Semantic Name | Index | Page Size | Num Pages | Format | Producer | Consumer | Lifetime |
|---|---|---|---|---|---|---|---|
| `cb_packet_recv` | 24 | `packet_size_bytes` | 1 | input dtype | receiver_reader (self, scratch) | receiver_reader (self) | L1 scratch that holds one landed packet read from the local intermediate; reserved once, pushed once at end |
| `cb_shard_out` | 16 | `aligned_page = round_up(buffer_page_size, 16)` | `3 * pages_per_packet` (pipelined) | input dtype | receiver_reader (NCRISC) | receiver_writer (BRISC) | streamed; producer pushes `buffer_num_pages`, consumer waits/pops `buffer_num_pages` |

**CB sync check** — every push count equals its wait/pop count:
- `cb_shard_pages`: sender_reader pushes `buffer_num_pages` == sender_writer pops `buffer_num_pages`. ✓
- `cb_shard_out`: receiver_reader pushes `buffer_num_pages` == receiver_writer pops `buffer_num_pages`. ✓
- `cb_packet_send` / `cb_packet_recv`: pure single-owner L1 scratch (one reserve/one push, no cross-kernel producer/consumer contract). ✓

## API Mapping

Type: `helper` (use as-is, owns its own resource management) or `raw_api` (with a "Helpers considered and rejected" sub-entry).

### Host-side (Python program-descriptor assembly)

| Phase | Type | Function | File:Line | Args / Returns | Notes |
|---|---|---|---|---|---|
| packet framing | helper | `ttnn._ttnn.fabric.ccl_packet_dims(dtype, page_size_bytes, num_pages, alignment)` | binding `ttnn/cpp/ttnn-nanobind/fabric.cpp:245`; impl `ccl_helpers_dataflow_host.hpp:74` | returns `.packet_size_bytes / .pages_per_packet / .page_segments / .total_packets` | owns the bf16 `bit_floor` clamp and both packing regimes. Drives the intermediate shape and all packet RT args. |
| route | helper | `ttnn._ttnn.fabric.ccl_dm_route(mesh_device, sender_coord, receiver_coord, topology)` | binding `fabric.cpp:253`; impl `ccl_helpers_dataflow_host.hpp:137` | returns `.num_hops / .is_forward / .neighbor_id` | owns fwd/bwd sign reversal + Ring short-way. Sender: `(sender→receiver)`. Receiver ack: `(receiver→sender)`. |
| fabric conn rt args | helper | `ttnn.setup_fabric_connection(src_fabric_node_id, dst_fabric_node_id, link_idx, program_descriptor, worker_core)` | binding `fabric.cpp:141` | returns `list[int]` fabric connection rt args; also appends `SemaphoreDescriptor`s to the program | wrap with the `[has_forward][fwd args][has_backward][bwd args]` layout the kernel's `FabricStreamSender` reads (mirror `append_ccl_fabric_rt_args`, `ccl_helpers_dataflow_host.hpp:219`). `src/dst_fabric_node_id = mesh_device.get_fabric_node_id(coord)`. |
| op-internal semaphore | helper | `ttnn.create_global_semaphore(mesh_device, cores, 0)` → `ttnn.get_global_semaphore_address(sem)` | binding `ttnn/cpp/ttnn-nanobind/global_semaphore.cpp:40` / `:58` | create ONCE per mesh_device (module-level cache), `ttnn.synchronize_device(mesh_device)` ONCE after creation | park on `mpd.semaphores = [sem]` (`program_descriptors.cpp:1077`) so its L1 survives program-cache hits. Bake `sem_addr` into both programs' rt args. |
| descriptor assembly | helper | `ttnn.MeshProgramDescriptor()` / `mpd[range] = program` / `.semaphores` | `program_descriptors.cpp:990`,`:1039`,`:1077` | one `(MeshCoordinateRange, ProgramDescriptor)` per participating coord | `ttnn.ProgramDescriptor(kernels=[...], semaphores=[], cbs=[...])`. |
| dispatch | helper | `ttnn.generic_op(io_tensors, mesh_program_descriptor)` | binding `ttnn/cpp/ttnn/operations/generic/generic_op_nanobind.cpp:47` | `io_tensors = [input_tensor, intermediate_tensor, output_tensor]` (output last) | when `output_tensor is None` it aliases `input_tensor` (in-place). See Risks re. the aliased-buffer binding. |

### Kernel-side (fabric egress — the CCL dataflow kernel helper)

All from `ttnn/cpp/ttnn/kernel_lib/ccl_helpers_dataflow.hpp` (+`.inl`), namespace `dataflow_kernel_lib::ccl`. Typestate: `FabricStreamSender<>` → `.open(route)` → `FabricStream<>` → armed channel.

| Phase | Type | Function | File:Line | Reads/Writes | Manages own CB ops? |
|---|---|---|---|---|---|
| build sender | helper | `FabricStreamSender<>(conn_arg_idx, is_forward, alignment)` | hpp:436 | reads fabric conn rt-arg block (advances `conn_arg_idx`) | n/a |
| open stream | helper | `.open(unicast_route(num_hops))` | hpp:448 (`unicast_route` hpp:257) | binds the route once | n/a |
| arm write | helper | `.arm_unicast_write(packet_size_bytes)` → `UnicastWriteChannel` | hpp:374 / inl:22 | programs invariant payload size (`align(page_size, alignment)`) | n/a |
| write packet | helper | `writer.write_page(src_l1_addr, packet_idx, dst_intermediate_accessor)` | hpp:283 / inl:46 | reads `cb_packet_send` L1; fabric-writes to receiver intermediate page `packet_idx` | no (op owns the L1 coalescing + `packet_idx`) |
| arm done inc | helper | `.arm_inc(1)` → `AtomicIncChannel` | hpp:386 / inl:103 | programs invariant inc value | n/a |
| signal done | helper | `done.inc(get_noc_addr(receive_semaphore_addr))` | hpp:317 / inl:120 | fabric atomic-inc on receiver's semaphore copy | n/a |
| receiver ready (one-shot) | helper | `ack_sender.signal(sender_num_hops, get_noc_addr(sender_semaphore_addr))` | hpp:456 / inl:182 | collapses open→arm_inc→inc→close for the "ready" ack | n/a |
| flush + teardown | helper | `stream.close()` (and `.drain()` if needed) | hpp:402 / inl:166 (drain hpp:399/inl:160) | drains outstanding writes + incs, then closes; idempotent (dtor backstops) | n/a |

### Kernel-side (raw APIs — each with justification)

| Phase | Type | Function | File:Line (proof) | Reads/Writes |
|---|---|---|---|---|
| wait handshake | raw_api | `noc_semaphore_wait_min(sem_ptr, 1)` | — | blocks on the local semaphore copy |
| re-arm semaphore | raw_api | `noc_semaphore_set(sem_ptr, 0)` | — | resets the semaphore (cache-reuse) |
| local ingress read | raw_api | `noc_async_read(...)` + `noc_async_read_barrier()` | — | receiver reads its local intermediate packet into `cb_packet_recv` |
| shard page load | raw_api | `noc_async_read` + `noc_async_read_barrier` via `TensorAccessor` | tensor_accessor.md | sender_reader: input page → `cb_shard_pages` |
| shard page store | raw_api | `noc_async_write` + `noc_async_write_barrier` via `TensorAccessor` | tensor_accessor.md | receiver_writer: `cb_shard_out` → output page |
| L1 page↔packet | raw_api | `tt_memmove(dst, src, n)` | `data_movement/common/kernels/common.hpp` | coalesce pages into a packet (sender) / split a packet into pages (receiver) |

**Helpers considered and rejected** (single sub-entry — the same helper header covers all fabric phases, so it is evaluated once against every raw phase above):

- **Candidate: `ccl_helpers_dataflow.hpp` (the CCL dataflow kernel helper).**
  - *`noc_semaphore_wait_min` (the WAITING half of the handshake):* the helper deliberately owns ONLY the SENDING half (remote atomic-inc via `AtomicIncChannel::inc` / `signal()`). Its banner states the waiting half is a plain op-owned `noc_semaphore_wait_min` and that **there is no `FabricStreamReceiver`** — `ccl_helpers_dataflow.hpp:69–74`. Using the helper here is impossible: no receive/wait entry point exists.
  - *`noc_semaphore_set(sem, 0)` (cache-reuse re-arm):* the helper explicitly delegates the reset to the op — banner `ccl_helpers_dataflow.hpp:75–77`: "each side must `noc_semaphore_set(sem, 0)` to re-arm — a SENDER resets BEFORE its outgoing inc, a RECEIVER after its wait." The helper does not perform the reset; skipping it green-first-run/hang-second-run.
  - *`noc_async_read` local ingress:* the banner (`ccl_helpers_dataflow.hpp:71–73`) states "the receive ingress is a local NoC read the op owns." No helper API performs the local read-back.
  - *`tt_memmove` L1 coalescing:* the helper's only multi-payload primitive is `arm_scatter_write` (hpp:379–381), which packs multiple **destination NoC addresses** into a fabric packet — it does NOT gather/scatter within local L1. Assembling a coalesced packet in L1 (multiple pages → one contiguous packet buffer) and splitting a landed packet back into pages is a local-memory copy the helper does not cover; `tt_memmove` is the correct primitive.
  - *`noc_async_read`/`noc_async_write` shard load/store via `TensorAccessor`:* this is plain interleaved page movement with no arithmetic. `ccl_helpers_dataflow.hpp` has no page-load/page-store entry, and the compute helper library (tilize/untilize/reduce/eltwise/matmul) is inapplicable — there is no compute phase in this op. `TensorAccessor` (tech_reports/tensor_accessor/tensor_accessor.md) is the standard addressing primitive.

## Dataflow Phases

Sequential cross-device execution (the two programs run concurrently on their devices; the semaphore enforces ordering):

| # | Phase | Kernel (device) | Consumes | Produces | State after |
|---|---|---|---|---|---|
| 1 | signal ready | receiver_reader (receiver) | — | fabric atomic-inc → sender's semaphore | sender's sem ≥ 1 |
| 2 | load shard | sender_reader (sender) | input shard pages (TensorAccessor) | `cb_shard_pages` (push `buffer_num_pages`) | pages streaming to writer |
| 3 | wait ready + reset | sender_writer (sender) | sender's semaphore | resets sender's sem to 0 (**before** its own inc) | sem re-armed for next cache hit |
| 4 | coalesce + fabric-write | sender_writer (sender) | `cb_shard_pages` (pop `buffer_num_pages`), `cb_packet_send` scratch | fabric writes `total_packets` packets → receiver intermediate | payload landing in receiver intermediate |
| 5 | signal done | sender_writer (sender) | — | fabric atomic-inc → receiver's semaphore; `stream.close()` drains writes+inc | receiver's sem ≥ 1 after all writes complete |
| 6 | wait done | receiver_reader (receiver) | receiver's semaphore | — | payload fully landed locally |
| 7 | read + de-coalesce | receiver_reader (receiver) | local intermediate (`noc_async_read` → `cb_packet_recv`) | `cb_shard_out` (push `buffer_num_pages`) | output pages streaming to writer |
| 8 | store shard | receiver_writer (receiver) | `cb_shard_out` (pop `buffer_num_pages`) | output shard pages (TensorAccessor) | receiver output shard = sender data |
| 9 | reset | receiver_reader (receiver) | receiver's semaphore | resets receiver's sem to 0 (**after** its wait) | sem re-armed for next cache hit |

Non-participating devices execute nothing; their output shards are the (aliased) input shards, untouched.

## Key Risks and Gotchas

| Risk | Mitigation |
|---|---|
| **Cache-reuse semaphore footgun** — the `GlobalSemaphore` is reused across program-cache hits; without a reset the first run passes and the second hangs/corrupts. | SENDER resets `noc_semaphore_set(sem,0)` **before** its outgoing "done" inc; RECEIVER resets **after** its "done" wait. (`ccl_helpers_dataflow.hpp:75–77`.) |
| **Semaphore lifetime** — a recreated-per-call semaphore or a lost L1 allocation breaks the cache hit. | Create ONCE per `mesh_device` (module-level cache) with ONE `synchronize_device`; park on `mpd.semaphores` so the framework keeps its L1 alive for the cached workload. |
| **In-place aliasing** — non-participating and sender-own output shards must equal the input, but only 2 devices run programs. | `output_tensor` defaults to `input_tensor` (in-place). Only the receiver's shard is written. `io_tensors = [input, intermediate, output]`; when aliased, `output is input`. Implementer must confirm the generic_op mesh adapter binds the aliased buffer once (the C++ path enumerates only the input buffer to avoid the aliasing guard — issue #45422); if the Python adapter rejects a repeated tensor, bind the input once and mark it as the in-place output. |
| **Intermediate must be a mesh tensor at the same device-local address** on both endpoints. | Allocate the intermediate as a mesh tensor (`allocate_tensor_on_device(intermediate_spec, mesh_device)`); the sender fabric-writes to `intermediate.buffer_address()` (identical across devices), routed `num_hops` to the receiver. |
| **bfloat16 packet clamp** — bf16 clamps the max packet to the largest power-of-two ≤ the fabric channel buffer. | Delegate entirely to `ccl_packet_dims` (owns the `bit_floor`). Never hand-compute packet sizing. |
| **Fabric fwd/bwd sign reversal + Ring short-way** — the fabric's forward/backward is inverted vs the coordinate-delta sign. | Delegate to `ccl_dm_route` (returns the already-negated `is_forward` and the Ring shorter-path `num_hops`). Feed `route.is_forward` to `FabricStreamSender` and `route.num_hops` to `unicast_route`. |
| **1-D routing only** — sender/receiver must share a row or a column; diagonal is illegal. | `ccl_dm_route` throws on non-aligned coords; validate that the two coords share a row or column and raise a clear error. |
| **Page-size 16-byte alignment** — the fabric writer sends `align(page_size, l1_alignment)` bytes; a non-16B page would overrun the next output page. | Validate `buffer_page_size() % 16 == 0` and raise `ValueError` otherwise. Every acceptance/golden shard shape keeps the last dim a multiple of 8 to satisfy this for all dtypes. |
| **Packet framing both regimes** — a shard page may be smaller (coalesce N pages/packet) or larger (split 1 page across `page_segments`) than a fabric transfer. | The kernels iterate `page_segments` inside the page loop and recompute `curr_pages_per_packet` per packet, exactly mirroring `ccl_packet_dims`' two regimes. |
| **`dim`/index canonicalization** — n/a: point_to_point has no dim/axis parameter (coords are mesh objects, not indices). | — |

## Structural impossibilities (feature_spec INVALID — authoritative, pipeline mode)

`eval/golden_tests/point_to_point/feature_spec.py` already declares `INVALID = [{"dtype": ttnn.bfloat8_b, "layout": ttnn.ROW_MAJOR_LAYOUT}]` — single-tensor coupling (dtype + layout both describe the input tensor); `bfloat8_b` is a block-quantized tiled format with no row-major representation (a data-format-definition impossibility). `topology` is orthogonal to dtype/layout (no coupling). The 16-byte page-size constraint is a shape×dtype `validate()` gate, not an axis. No additional structural impossibilities are needed. (This design is in pipeline mode; `feature_spec.py` is read as authoritative and not edited here.)

## Notes for the implementer

- No `Broadcast Verification` / `Reduce Direction Verification` sections: this op has no binary or reduce compute.
- `dim` support check canonicalization is n/a (no index parameter).
- Kernel arguments (CT/RT layouts) are intentionally omitted — derive them from the CB layout, the helper signatures above, and the fabric-arg-block layout (`[has_forward][fwd conn args][has_backward][bwd conn args]` beginning at the `conn_arg_idx` the kernel records). The RT `page_size` MUST be passed to the `TensorAccessor` to override a possibly-stale `AlignedPageSize` on cache hits.
