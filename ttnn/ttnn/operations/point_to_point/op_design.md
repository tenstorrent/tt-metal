# Operation Design: point_to_point

> Self-contained Python CCL op built on `ttnn.generic_op` + `ttnn.MeshProgramDescriptor`.
> Authored FROM SCRATCH — it does **not** wrap / import / dispatch to the bound C++ op
> `ttnn.point_to_point` / `ttnn._ttnn.operations.point_to_point`. The bound C++ op and the
> `all_gather_async` kernels were read as a **correctness reference only**.

## Overview

| Field | Value |
|-------|-------|
| Classification | CCL (multi-device fabric data movement — pure byte copy, NO arithmetic, NO compute kernel) |
| Goal | Copy one mesh device's interleaved shard of a mesh-sharded tensor to another device over the Tenstorrent fabric; return a tensor whose receiver-device shard equals the sender-device input shard bit-for-bit, all other shards unchanged. |
| Math | `output_shard[receiver_coord] = input_shard[sender_coord]` (identity); `output_shard[c] = input_shard[c]` for every `c != receiver_coord` |
| Mode | Derivative (mirrors the vetted `point_to_point` C++ handshake + framing; re-expressed on the `generic_op` / `MeshProgramDescriptor` path with the `ccl_helpers_dataflow.hpp` kernel helper) |
| References | `ttnn/cpp/ttnn/kernel_lib/ccl_helpers_dataflow.hpp` (kernel fabric egress helper), `ttnn/cpp/ttnn/operations/ccl/common/host/ccl_helpers_dataflow_host.hpp` (host route/framing), `ttnn/cpp/ttnn/operations/point_to_point/device/*` (C++ reference), `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/*` (fabric egress reference), `.claude/references/generic_op_template/` (descriptor API) |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | interleaved, DRAM or L1, rank ≥ 2, on a `ttnn.MeshDevice` (≥2 devices along the row/column joining the two coords) | — | tensor |
| `sender_coord` | `ttnn.MeshCoordinate` | yes | inside the mesh; shares a row or column with `receiver_coord`; `!= receiver_coord` | — | host (→ hops/dir baked into RT args) |
| `receiver_coord` | `ttnn.MeshCoordinate` | yes | inside the mesh; shares a row or column with `sender_coord`; `!= sender_coord` | — | host (→ hops/dir baked into RT args) |
| `topology` | `ttnn.Topology` | no | `Linear` (primary) or `Ring` (may route the short way around) | `ttnn.Topology.Linear` | host (→ `ccl_dm_route` route choice) |
| `output_tensor` | `ttnn.Tensor \| None` | no | must match resolved output spec (shape == input shard shape, same dtype, same memory_config) | `None` (allocate) | tensor (output, last) |
| `intermediate_tensor` | `ttnn.Tensor \| None` | no | fabric landing buffer; if `None`, allocated internally from `ccl_packet_dims` | `None` (allocate) | tensor (staging) |

### Validation (op `validate()` — raises before dispatch)

| Condition | Error |
|-----------|-------|
| input not on a `ttnn.MeshDevice` | reject — requires a mesh |
| `sender_coord == receiver_coord` | reject — cannot send to self |
| `sender_coord` or `receiver_coord` outside the mesh extents | reject — out of range |
| input is sharded (non-interleaved) memory layout | reject — not yet supported |
| input per-device page size not 16-byte aligned | reject — fabric requires 16B-aligned pages |
| `output_tensor` supplied with shape/dtype/memory_config ≠ resolved output spec | reject — spec mismatch |

`topology` is canonicalized as an enum (`Linear` / `Ring`); there is no signed-index axis to normalize.

## Dataflow Strategy

Pure data movement across exactly **two** mesh devices; every other device runs no program.
The op is a `ttnn.generic_op` over a `ttnn.MeshProgramDescriptor` holding exactly two
`(MeshCoordinateRange, ProgramDescriptor)` entries: a **send program** pinned to
`MeshCoordinateRange(sender_coord, sender_coord)` and a **receive program** pinned to
`MeshCoordinateRange(receiver_coord, receiver_coord)`.

Three mesh tensors flow into `ttnn.generic_op([input_tensor, intermediate_tensor, output_tensor], mpd)`
(output tensor last, per the generic-op convention at `ttnn/cpp/ttnn/operations/generic/generic_op_nanobind.cpp:46-60`).

End-to-end data path (per participating device, one Tensix core each):

```
SENDER device                                              RECEIVER device
  DRAM input shard                                           DRAM output shard
       │ (sender_reader: noc_async_read via TensorAccessor)       ▲ (receiver_writer: noc_async_write via TensorAccessor)
       ▼                                                          │
  cb_send_pages (L1, page-size pages)                        cb_recv_pages (L1, page-size pages)
       │ (sender_writer: coalesce pages → packet)                 ▲ (receiver_reader: de-coalesce packet → pages)
       ▼                                                          │
  cb_send_packet (L1, packet-size scratch)                   cb_recv_packet (L1, packet-size scratch)
       │ (sender_writer: FabricStream unicast write)              ▲ (receiver_reader: noc_async_read, LOCAL)
       └────────────── fabric (num_hops, direction) ─────────────►│
                    intermediate_tensor page i  ───────────────► intermediate_tensor page i
                    (receiver-device copy, at same address)
```

- **Format is preserved end to end** — the op moves bytes only. Input layout (TILE or ROW_MAJOR) and
  dtype are unchanged; there is no tilize/untilize and no compute thread is used.
- The **intermediate tensor** is the fabric landing buffer. Its per-device buffer holds `total_packets`
  pages of `packet_size_bytes` each (from `ccl_packet_dims`). The sender fabric-writes packet `i` into
  the **receiver device's copy** of `intermediate_tensor` page `i` (the fabric route delivers the write
  to the remote copy at the shared address). The receiver then reads its own copy **locally** and
  de-coalesces packets back into shard pages.
- **Packet framing handles both regimes** (owned by `ccl_packet_dims`,
  `ccl_helpers_dataflow_host.hpp:74-96`): when a page is smaller than a fabric transfer unit, N whole
  pages are packed per packet (`pages_per_packet > 1`, `page_segments == 1`); when a page is larger,
  each page is split across `page_segments` packets (`pages_per_packet == 1`). The sender writer
  coalesces/segments; the receiver reader reverses it.

### Multi-Device Coordination (Tensix-to-Tensix contract)

One **op-internal cross-device `GlobalSemaphore`**, created ONCE per `mesh_device` (cached on the op
module), `synchronize_device` ONCE right after creation, its absolute address baked into both
programs' runtime args, and the semaphore object parked in `mpd.semaphores` so the framework keeps its
L1 alive across program-cache hits (`MeshProgramDescriptor.semaphores`,
`ttnn/cpp/ttnn-nanobind/program_descriptors.cpp:1077-1087`; struct
`tt_metal/api/tt-metalium/experimental/mesh_program_descriptor.hpp:16-34`). No per-call
`synchronize_device` barrier is added — the framework owns lifetime once parked.

A `GlobalSemaphore` has the **same L1 address on every device** but an independent physical cell per
device. "Sender's cell" and "receiver's cell" are two cells at one address on two devices. Each fabric
atomic-inc lands on the **remote** device's cell along the opened route; each `noc_semaphore_wait_min`
reads the **local** cell.

Ordering contract (exactly this sequence — matches the `ccl_helpers_dataflow.hpp:69-82` cache-reuse
footgun rule: *sender resets BEFORE its outgoing inc, receiver resets AFTER its wait*):

| Step | Actor | Action | Cell touched |
|------|-------|--------|--------------|
| 1 | receiver_reader | `signal(hops_to_sender, sender_sem_noc_addr)` — one-shot fabric "ready" inc | sender's cell += 1 |
| 2 | sender_writer | `noc_semaphore_wait_min(local_sem, 1)` — block until receiver ready | reads sender's cell |
| 3 | sender_writer | `noc_semaphore_set(local_sem, 0)` — reset **before** its own inc (re-arm for cache reuse) | sender's cell ← 0 |
| 4 | sender_writer | fabric-write every payload packet → receiver's `intermediate_tensor` | receiver DRAM |
| 5 | sender_writer | `done.inc(receiver_sem_noc_addr)` — fabric "done" inc, then `stream.close()` (drains) | receiver's cell += 1 |
| 6 | receiver_reader | `noc_semaphore_wait_min(local_sem, 1)` — block until payload fully landed | reads receiver's cell |
| 7 | receiver_reader | local `noc_async_read` intermediate → de-coalesce → `cb_recv_pages` | receiver DRAM/L1 |
| 8 | receiver_reader | `noc_semaphore_set(local_sem, 0)` — reset **after** its wait (re-arm for cache reuse) | receiver's cell ← 0 |

Step 1 (ready) also serves as flow control: the sender does not begin payload writes until the
receiver program is live and has acked, guaranteeing the receiver's intermediate buffer is allocated
and ready to land data.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | rank ≥ 2; the per-device shard shape (unchanged end to end) |
| Dtype | bfloat16 (primary), float32, bfloat8_b (TILE only), uint16, int32, uint32 |
| Layout | TILE (primary) or ROW_MAJOR |
| Memory | interleaved, DRAM or L1 |

### Output

| Property | Value |
|----------|-------|
| Shape | identical to the input per-device shard shape |
| Dtype | same as input |
| Layout | same as input |
| Memory | same `memory_config` as input |

### Intermediate (fabric landing buffer — internal, allocated if not supplied)

| Property | Value |
|----------|-------|
| Shape | `(total_packets, packet_size_bytes // element_size)` — one page/row per fabric packet (mirror `point_to_point_device_op.cpp:60` `compute_output_specs`) |
| Dtype | same as input |
| Layout | ROW_MAJOR (raw packet bytes; layout-agnostic staging) |
| Memory | DRAM interleaved (replicated across the mesh; only sender→receiver copies are touched) |
| Sizing source | `ttnn._ttnn.fabric.ccl_packet_dims(dtype, page_size_bytes, num_pages, alignment)` → `.total_packets`, `.packet_size_bytes` |

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one whole per-device shard, processed page-by-page |
| Grid | single Tensix core per participating device (`CoreCoord{0,0}`); send program on sender device, receive program on receiver device |
| Per-core work | sender core: read all `num_pages` shard pages, coalesce into `total_packets` packets, fabric-write all; receiver core: read all `total_packets` packets locally, de-coalesce into `num_pages` pages, write to output DRAM |
| Remainder | last packet may carry fewer pages (`num_pages % pages_per_packet`) or a partial page segment (`aligned_page % max_packet`); `ccl_packet_dims` provides the counts, kernels loop `total_packets` and clamp the final packet |

Single-core is the correct, simplest design for a pure copy and matches the C++ reference's per-link
worker. Multi-link / multi-core parallelism is a future refinement, not required for correctness.

## Circular Buffers

### Send program (sender device)

| Semantic Name | Index | Page Size | Num Pages | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|--------|----------|----------|----------|
| `cb_send_pages` | 0 | `page_size_bytes` (input buffer aligned page size) | 2 | input dtype | sender_reader (DRAM→CB) | sender_writer (CB→packet) | streaming (double-buffered) |
| `cb_send_packet` | 24 | `packet_size_bytes` | 2 | input dtype | sender_writer (coalesce scratch) | sender_writer (fabric-write source) | transient per packet (double-buffered scratch) |

### Receive program (receiver device)

| Semantic Name | Index | Page Size | Num Pages | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|--------|----------|----------|----------|
| `cb_recv_packet` | 24 | `packet_size_bytes` | 2 | input dtype | receiver_reader (local NoC read from intermediate) | receiver_reader (de-coalesce source) | transient per packet (double-buffered scratch) |
| `cb_recv_pages` | 16 | `page_size_bytes` | `3 * pages_per_packet` (≥2) | input dtype | receiver_reader (de-coalesce→CB) | receiver_writer (CB→DRAM) | streaming (pipelined) |

No packet-header CB — fabric packet headers are drawn from the `PacketHeaderPool` inside the kernel by
each `arm_*` call (`ccl_helpers_dataflow.hpp:78-81`).

**CB sync (push == wait):**
- `cb_send_pages`: sender_reader pushes `num_pages`; sender_writer waits/pops `num_pages`. ✔
- `cb_send_packet`: sender_writer reserves/pushes then waits/pops `total_packets` (self-recycled scratch). ✔
- `cb_recv_packet`: receiver_reader reserves/pushes then waits/pops `total_packets` (self-recycled scratch). ✔
- `cb_recv_pages`: receiver_reader pushes `num_pages`; receiver_writer waits/pops `num_pages`. ✔

## API Mapping

Every mechanism has a verified file:line reference. Base: `ttnn/cpp/ttnn/kernel_lib/ccl_helpers_dataflow.hpp`
(decls) + `ccl_helpers_dataflow.inl` (defs); host in `ttnn/cpp/ttnn/operations/ccl/common/host/ccl_helpers_dataflow_host.hpp`.

### Kernel-side (dataflow)

| Phase | Type | Function | File:Line | Args | Reads CB | Writes | Requirements |
|-------|------|----------|-----------|------|----------|--------|--------------|
| build sender fabric conn | helper | `FabricStreamSender<DirectConn>(conn_arg_idx, is_forward, alignment)` | `ccl_helpers_dataflow.hpp:436-437` | connection RT-arg block index, forward flag, L1 alignment | — | — | `conn_arg_idx` = start of `append_ccl_fabric_rt_args` block; sender declared before the stream (must outlive it) |
| open route | helper | `.open(route)` → `FabricStream` | `ccl_helpers_dataflow.hpp:448-451` | `line_unicast_route_info_t` route | — | — | route bound ONCE here, reused by every `arm_*` |
| construct route | helper | `unicast_route(num_hops)` | `ccl_helpers_dataflow.hpp:260-265` | `num_hops` (RT arg) | — | — | 1-D linear route; `line_unicast_route_info_t` (`worker_routing_utils.hpp:15-21`) |
| arm payload write | helper | `.arm_unicast_write(page_size_bytes)` → `UnicastWriteChannel` | decl `:374`, def `.inl:22-37` | `packet_size_bytes` (armed payload) | — | — | route implicit (from `open`); NOT a param |
| issue payload write | helper | `UnicastWriteChannel::write_page(src_l1_addr, page_idx, addrgen)` | decl `:283-284`, def `.inl:46-51` | packet scratch addr, packet index, intermediate TensorAccessor | `cb_send_packet` | receiver intermediate | one call per packet; arg order `(src, idx, addrgen)` |
| arm done inc | helper | `.arm_inc(val=1)` → `AtomicIncChannel` | decl `:386`, def `.inl:103-118` | val=1 | — | — | route implicit |
| issue done inc | helper | `AtomicIncChannel::inc(remote_sem_noc_addr)` | decl `:317`, def `.inl:120-125` | receiver sem NoC addr | — | receiver's sem cell | fabric atomic-inc |
| drain + teardown | helper | `.close()` (idempotent; RAII backstop) | decl `:402`, def `.inl:166-176` | — | — | — | drains write+atomic barriers before closing |
| receiver "ready" ack | helper | `FabricStreamSender::signal(num_hops, remote_sem_noc_addr, val=1)` | decl `:456-461`, def `.inl:182-189` | hops-to-sender, sender sem NoC addr | — | sender's sem cell | one-shot open→arm_inc→inc→close |

### Host-side (program-descriptor assembly)

| Phase | Type | Function | File:Line | Purpose |
|-------|------|----------|-----------|---------|
| route + direction | helper | `ttnn._ttnn.fabric.ccl_dm_route(mesh_device, src, dst, topology)` → `.num_hops/.is_forward/.neighbor_id` | binding `ttnn/cpp/ttnn-nanobind/fabric.cpp:253-266`; impl `ccl_helpers_dataflow_host.hpp:137-166` | owns fwd/bwd **sign reversal** + Ring short-way; sender uses `(sender,receiver)`, receiver uses `(receiver,sender)` |
| packet framing | helper | `ttnn._ttnn.fabric.ccl_packet_dims(dtype, page_size_bytes, num_pages, alignment)` → `.packet_size_bytes/.pages_per_packet/.page_segments/.total_packets` | binding `fabric.cpp:245-252`; impl `ccl_helpers_dataflow_host.hpp:74-96` | owns bf16 `bit_floor`; sizes intermediate + kernel loop bounds |
| fabric conn RT args | helper | `ttnn.setup_fabric_connection(src_node, dst_node, link, program_descriptor, worker_core)` → `List[int]` | `fabric.cpp:141-178` | emits conn RT-arg block `[has_forward][fwd args][has_backward][bwd args]` (`ccl_helpers_dataflow_host.hpp:219-237`); also appends `SemaphoreDescriptor`s to the program |
| cross-device sem | helper | `ttnn.create_global_semaphore(mesh_device, cores, 0)` | `ttnn/cpp/ttnn-nanobind/global_semaphore.cpp:40-56` | created ONCE per mesh_device, cached on op module |
| sem sync | helper | `ttnn.synchronize_device(mesh_device)` | `ttnn/cpp/ttnn-nanobind/device.cpp:548-559` | ONCE right after semaphore creation |
| sem address | helper | `ttnn.get_global_semaphore_address(sem)` | `global_semaphore.cpp:58-67` | baked into both programs' RT args |
| descriptors | helper | `ttnn.MeshProgramDescriptor` (`.semaphores`), `ttnn.ProgramDescriptor`, `ttnn.KernelDescriptor`, `ttnn.CBDescriptor`, `ttnn.RuntimeArgs`, `ttnn.MeshCoordinate/Range` | `program_descriptors.cpp:990-1087,930-961,694-737,398-419,167-241`; `distributed_nanobind.cpp:183-241` | two per-coord programs + parked semaphore |
| dispatch | helper | `ttnn.generic_op([input, intermediate, output], mpd)` | `generic_op_nanobind.cpp:46-60` | output tensor last |

### Raw APIs (op-owned; helper explicitly does NOT own these)

Every raw call below is a documented **non-goal** of the fabric helper — verified against its banner.

| Phase | Raw API | Helpers considered & rejected (file:line) | Concrete reason |
|-------|---------|--------------------------------------------|-----------------|
| wait for handshake (both sides) | `noc_semaphore_wait_min(local_sem, 1)` | `ccl_helpers_dataflow.hpp` — banner `:69-74` states "The WAITING half of a cross-device sync is a plain local `noc_semaphore_wait_min(sem, threshold)` the op calls directly … there is no FabricStreamReceiver." | The helper owns only the *sending* (atomic-inc) half; no helper exists for the waiting half. |
| re-arm semaphore (cache reuse) | `noc_semaphore_set(local_sem, 0)` | `ccl_helpers_dataflow.hpp` — banner `:74-77` "each side must `noc_semaphore_set(sem, 0)` to re-arm — a SENDER resets BEFORE its outgoing inc, a RECEIVER after its wait." | Explicitly the op's responsibility; the helper deliberately does not reset. |
| receiver payload ingress | `noc_async_read` + `noc_async_read_barrier` (LOCAL, intermediate→`cb_recv_packet`) | `ccl_helpers_dataflow.hpp` — banner `:69-72` "the receive INGRESS is a local NoC read the op owns." | Fabric egress is remote-write only; the local read of the landed buffer has no helper. |
| page↔packet coalesce / segment | `tt_memmove` (sender pages→packet; receiver packet→pages) | `ccl_helpers_dataflow.hpp` — banner `:89-91` lists "page↔packet coalescing/segmentation" under "What the helper does NOT own." | Explicit non-goal of the helper. |
| tensor addressing | `TensorAccessor` / `get_noc_addr` (input read, intermediate write, output write, sem NoC addr) | `ccl_helpers_dataflow.hpp` — banner `:89-93` "address generation (TensorAccessor/ShardedAddrGen consumed, never re-wrapped)." | Helper consumes an addrgen (e.g. `write_page`'s `addrgen`); it does not construct one. |

## Dataflow Phases

Two independent kernel pipelines per device, synchronized only through the cross-device semaphore.

| # | Program / Kernel | Operation | Input | Output | State After |
|---|------------------|-----------|-------|--------|-------------|
| S0 | send / sender_reader | `noc_async_read` each shard page (TensorAccessor over `input_tensor`) → `cb_send_pages` | `input_tensor` DRAM | `cb_send_pages` (push `num_pages`) | pages streaming to writer |
| S1 | send / sender_writer | wait ready (`wait_min 1`) → reset (`set 0`) → open route → arm write + inc | sender's sem cell | fabric stream armed | route + channels live |
| S2 | send / sender_writer | pop pages from `cb_send_pages`, `tt_memmove` into `cb_send_packet`, `write_page` each full packet → receiver intermediate | `cb_send_pages`, `cb_send_packet` | receiver `intermediate_tensor` | all `total_packets` written |
| S3 | send / sender_writer | `done.inc(receiver_sem_noc_addr)` → `stream.close()` (drain) | — | receiver's sem cell += 1 | sender done, sem re-armed |
| R0 | receive / receiver_reader | `signal(hops_to_sender, sender_sem_noc_addr)` — fabric ready ack | — | sender's sem cell += 1 | handshake opened |
| R1 | receive / receiver_reader | `noc_semaphore_wait_min(local_sem, 1)` — block until payload landed | receiver's sem cell | — | payload guaranteed present |
| R2 | receive / receiver_reader | local `noc_async_read` intermediate page → `cb_recv_packet`, `tt_memmove` de-coalesce → `cb_recv_pages` | `intermediate_tensor`, `cb_recv_packet` | `cb_recv_pages` (push `num_pages`) | pages streaming to writer |
| R3 | receive / receiver_writer | `noc_async_write` each page (TensorAccessor over `output_tensor`) | `cb_recv_pages` (wait/pop `num_pages`) | `output_tensor` DRAM | receiver shard == sender shard |
| R4 | receive / receiver_reader | `noc_semaphore_set(local_sem, 0)` — reset after wait | — | receiver's sem cell ← 0 | sem re-armed for cache reuse |

## Structural impossibilities (feature_spec.py — pipeline mode)

`eval/golden_tests/point_to_point/feature_spec.py` already exists and is authoritative (INVALID
populated by `/golden-tests`). Its single INVALID cell is correct and complete for this op:

- `{dtype: bfloat8_b, layout: ROW_MAJOR_LAYOUT}` — bfloat8_b is a block-quantized tiled format with
  no row-major representation (single-tensor coupling: both axes describe the input tensor; a
  data-format-definition impossibility, not a kernel refinement). No additional op-specific INVALID
  cells are needed: `sender_coord`/`receiver_coord` are mesh-dependent and are not enumerable
  cartesian axes (fixed by the harness, per the feature_spec docstring), and the 16-byte page-size
  gate is a shape×dtype `validate()` check kept satisfiable by every INPUTS shard (last dim multiple
  of 8), not modeled as an axis. `topology` is orthogonal to dtype/layout (no coupling).

## Key Risks and Gotchas

- **Semaphore re-arm (cache-reuse footgun).** Sender resets its cell **before** its outgoing `done`
  inc; receiver resets its cell **after** its `wait`. Missing/misordered reset ⇒ first run green,
  second run hangs or corrupts (`ccl_helpers_dataflow.hpp:74-77`). This is the single most important
  correctness rule.
- **Semaphore lifetime.** Create the `GlobalSemaphore` ONCE per `mesh_device` (cache it on the op
  module), `synchronize_device` ONCE right after, and park it in `mpd.semaphores`. Do NOT recreate it
  per call and do NOT add a per-call post-dispatch barrier — the framework holds its L1 alive for the
  cached workload (`mesh_program_descriptor.hpp:16-34`).
- **`ccl_dm_route(...).is_forward` is sign-reversed** relative to geometry (`ccl_helpers_dataflow_host.hpp:161,165`).
  Pass its `is_forward` straight into `append_ccl_fabric_rt_args` / the `FabricStreamSender` forward
  flag — do not re-derive the sign by hand.
- **Route is bound at `open(route)`, not at `arm_*`.** `arm_unicast_write` / `arm_inc` take NO route
  argument (`:374`, `:386`) — only `arm_multicast_inc` carries its own route (unused here). Passing a
  route to a unicast `arm_*` will not compile; forgetting `open`'s route silently corrupts packets.
- **Sender lifetime vs. stream.** The `FabricStreamSender` must be declared before (and outlive) the
  `FabricStream` returned by `.open()` — the stream borrows the sender's connection
  (`ccl_helpers_dataflow.hpp:448-451`).
- **Route direction differs per side.** Sender computes `ccl_dm_route(sender→receiver)`; receiver
  computes `ccl_dm_route(receiver→sender)` for its ready ack. Both must agree on hops so the atomic-inc
  lands on the correct remote cell.
- **Both regimes of framing.** Do not assume one page == one packet. `pages_per_packet` may be >1
  (coalesce) or the page may span `page_segments` packets (split). Loop `total_packets` from
  `ccl_packet_dims`, clamp the final packet, and never hard-code page/packet equality.
- **Intermediate is a landing buffer, not the output.** The sender writes into the receiver's copy of
  `intermediate_tensor`; the receiver reads its own copy locally and only then writes `output_tensor`.
  Sizing must come from `ccl_packet_dims` (`.total_packets` × `.packet_size_bytes`), not from the
  input shape directly.
- **`close()` must run (or the RAII destructor).** It drains write + atomic barriers before tearing
  down the connection so a trailing `done` inc / final packet is never lost (`.inl:166-176`).

## Hardware Constraints

- [x] CB sync: push == wait for every CB (`cb_send_pages`, `cb_send_packet`, `cb_recv_packet`, `cb_recv_pages`) — verified above
- [ ] Reduce scaler — N/A (no reduction, no compute kernel)
- [ ] DEST register — N/A (no compute thread)
- [x] Page sizes 16-byte aligned (op `validate()` rejects non-aligned pages; every INPUTS shard has last dim multiple of 8)
- [x] RM CBs count pages in sticks, tile CBs count in tiles — all four CBs count in `page_size_bytes`/`packet_size_bytes` pages consistently
- [x] All `cb_wait_front` on the same CB use the same page count
- [x] No compute helper wrapped with extra CB ops — fabric helper (`FabricStreamSender`) owns its own header/connection lifecycle; op owns only the wait/reset/local-read/coalesce it is documented to own
