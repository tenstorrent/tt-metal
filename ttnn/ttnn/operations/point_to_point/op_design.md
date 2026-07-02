# Operation Design: point_to_point

## Overview

| Field | Value |
|-------|-------|
| Classification | CCL (multi-device collective communication) — pure data movement, NO arithmetic |
| Goal | Copy one mesh device's interleaved shard of a mesh-sharded tensor to another device over the Tenstorrent fabric. The receiver device's output shard becomes a bit-for-bit copy of the sender device's input shard; every other device's shard is untouched. |
| Math | `output[receiver_coord] = input[sender_coord]`; `output[d] = input[d]` for every other device `d`. Identity oracle (element values unchanged end to end). |
| Mode | Derivative (self-contained Python op on `ttnn.generic_op` + `ttnn.MeshProgramDescriptor`; NEWLY authored sender/receiver dataflow kernels). Does NOT wrap the bound C++ `ttnn.point_to_point`. |
| References | `ttnn/cpp/ttnn/kernel_lib/ccl_helpers_dataflow.hpp` (+`.inl`) — fabric egress typestate; `ttnn/cpp/ttnn/operations/ccl/common/host/ccl_helpers_dataflow_host.hpp` — host framing/route helpers, Python-bound as `ttnn._ttnn.fabric.ccl_packet_dims` / `ccl_dm_route`; `tech_reports/tensor_accessor/tensor_accessor.md`; `ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp:87` (`tt_memmove`). Verification topology: `scripts/run_multidevice_sim_pytest.py --list` → `bh_8xP150_p2p` mesh `[2,4]` / `FABRIC_1D`. |

### Multi-device workload shape

A two-program `ttnn.MeshProgramDescriptor` — exactly one `ProgramDescriptor` per participating coordinate; no program runs on any other mesh device.

| Coordinate | Program | Kernels (core `(0,0)`) |
|------------|---------|------------------------|
| `sender_coord` | SEND | `point_to_point_sender_reader.cpp` (NCRISC), `point_to_point_sender_writer.cpp` (BRISC) |
| `receiver_coord` | RECEIVE | `point_to_point_receiver_reader.cpp` (NCRISC), `point_to_point_receiver_writer.cpp` (BRISC) |
| all other coords | none | — |

Cross-device coordination uses ONE op-internal cross-device `GlobalSemaphore`, created ONCE per `mesh_device`, `synchronize_device`'d once at creation, cached (stable address), and parked on `mesh_program_descriptor.semaphores` so the framework keeps its L1 alive across program-cache hits. No per-call re-creation and no per-call post-dispatch barrier.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | on a `ttnn.MeshDevice` (2-D view), interleaved (DRAM or L1), rank ≥ 2, per-shard page size 16-B aligned | — | host |
| `sender_coord` | `ttnn.MeshCoordinate` | yes | inside the mesh; shares a row or column with `receiver_coord`; `!= receiver_coord` | — | host route → kernel RT args |
| `receiver_coord` | `ttnn.MeshCoordinate` | yes | inside the mesh; shares a row or column with `sender_coord`; `!= sender_coord` | — | host route → kernel RT args |
| `topology` | `ttnn.Topology` | no | `Linear` or `Ring` | `Linear` | host-side route selection only (`ccl_dm_route`) |
| `output_tensor` | `ttnn.Tensor \| None` | no | shape/dtype/layout/buffer-type == resolved output spec (== input spec) | `None` | host (write-into path) |
| `intermediate_tensor` | `ttnn.Tensor \| None` | no | shape/dtype/layout == resolved intermediate spec | `None` | host (op-internal staging) |

There are no compute-kernel template params. Op-specific values reach the kernels as runtime args (packet dims, hop count, semaphore address, fabric-connection block); CB indices and alignment reach them as compile-time args.

### Validation (raises `ValueError` unless noted)

| Condition | Error |
|-----------|-------|
| input not on a `ttnn.MeshDevice` | "must be on a MeshDevice" |
| mesh view not 2-D | "expected a 2-D mesh view" |
| `sender_coord == receiver_coord` | "cannot send to self" |
| `sender_coord` or `receiver_coord` outside mesh | "outside the mesh" |
| coords share neither row nor column | "must share a row or column (1-D fabric route)" |
| input is sharded (non-interleaved) | "sharded input not yet supported (interleaved only)" |
| per-shard page size not 16-B aligned | "page size must be 16-byte aligned" |
| `output_tensor` spec != input spec | "output_tensor spec must equal the resolved output spec" |
| `intermediate_tensor` spec != resolved intermediate spec | "intermediate_tensor spec must equal the resolved intermediate spec" |
| dtype / layout / topology / alignment axis outside SUPPORTED | `UnsupportedAxisValue` |

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | per-device shard shape, rank ≥ 2 (mesh-sharded across the 2-D `MeshDevice`) |
| Dtype | `bfloat16` (primary), `float32`, `bfloat8_b` (TILE only), `uint16`, `int32`, `uint32` |
| Layout | `TILE_LAYOUT` (primary) or `ROW_MAJOR_LAYOUT` |
| Memory | interleaved, DRAM or L1 |

### Output

| Property | Value |
|----------|-------|
| Shape | identical to input's per-device shard shape |
| Dtype | same as input |
| Layout | same as input |
| Memory | same memory config as input |
| Semantics | `output[receiver_coord]` == `input[sender_coord]` (bit-for-bit); every other device's shard == its own input shard |

### Intermediate (op-internal staging)

Resolved by `resolve_intermediate_spec(input_tensor)`. Addressed PER-PACKET (page index = `packet_idx`, page size overridden to `packet_size_bytes`), carrying raw bytes only.

| Property | Value |
|----------|-------|
| Shape | `[total_packets, packet_size_bytes / 4]` |
| Dtype | `uint32` (sidesteps `element_size` being undefined for block-float; `packet_size_bytes` is a multiple of the L1 alignment ≥ 16, hence of 4) |
| Layout | `ROW_MAJOR_LAYOUT` |
| Memory | input's `buffer_type` |

## Dataflow Strategy

Pure cross-chip data movement. No compute kernel, no tilize/untilize — the op copies physical pages verbatim (padded tiles for TILE, last-dim rows for ROW_MAJOR), so it is layout- and alignment-agnostic. Data path:

```
SENDER device:                              fabric              RECEIVER device:
DRAM/L1 input shard                                             DRAM/L1 output shard (pre-seeded == input)
  │ reader (NCRISC): page-at-a-time                               ▲ writer (BRISC): page-at-a-time
  ▼ noc_async_read                                                │ noc_async_write
cb_input_pages (double-buffer)                                  cb_output_pages (de-coalesced stream)
  │ writer (BRISC): coalesce pages → 1 packet (tt_memmove)        ▲ reader (NCRISC): scatter packet → pages (tt_memmove)
  ▼ FabricStream unicast write_page  ──────────────────────►    cb_packet_scratch (one landed packet)
                                     receiver intermediate        ▲ noc_async_read (local ingress) from intermediate
                                     (per-packet buffer)  ────────┘
```

### Format at each stage

| Stage | Format |
|-------|--------|
| Sender DRAM/L1 → `cb_input_pages` | one physical page per CB slot (tile page for TILE, stick page for RM) |
| `cb_input_pages` → `cb_packet_scratch` | N pages coalesced into one fabric packet (or one page split across `page_segments`) |
| fabric → receiver intermediate | one packet per `packet_idx` |
| intermediate → `cb_packet_scratch` (receiver) | one landed packet |
| `cb_packet_scratch` → `cb_output_pages` | packet de-coalesced back into physical pages |
| `cb_output_pages` → receiver DRAM/L1 | one physical page per CB slot |

### Page ↔ packet framing

`ttnn._ttnn.fabric.ccl_packet_dims(dtype, page_size_bytes, num_pages, l1_alignment)` frames the shard, owning the bf16 `bit_floor` and both regimes:
- **aligned page ≤ max packet** → pack `pages_per_packet` pages per packet, `page_segments == 1`.
- **aligned page > max packet** → split each page into `page_segments` segments, one segment per packet.

The sender writer coalesces (`page → packet`, running `packet_idx`); the receiver reader de-coalesces (`packet → pages`). Both use the SAME `ccl_packet_dims` result, so their loops agree. The per-packet count `curr_pages_per_packet = min(max_pages_per_packet, remaining)` is recomputed identically on both sides.

### Tensix-to-Tensix (cross-device) contract

Exactly two Tensix cores participate: sender core `(0,0)` on `sender_coord`, receiver core `(0,0)` on `receiver_coord`. They coordinate through ONE shared `GlobalSemaphore` (same absolute address baked into both programs' RT args) using a ready/done handshake:

| Step | Actor | Action |
|------|-------|--------|
| 1 | RECEIVER reader | Fabric atomic-inc "ready" to sender (`ack_sender.signal(sender_num_hops, sender_sem_noc_addr)` — one-shot `open→arm_inc→inc→close`). Route = receiver→sender (`recv_route`). |
| 2 | SENDER writer | `noc_semaphore_wait_min(sem, 1)` for "ready", then `noc_semaphore_set(sem, 0)` — reset **BEFORE** its own outgoing inc (cache-reuse re-arm). |
| 3 | SENDER writer | Open fabric stream (route sender→receiver, `send_route`), `arm_unicast_write` + `arm_inc`; coalesce + `write_page` each packet into the receiver's intermediate buffer. |
| 4 | SENDER writer | After all packets: `done.inc(receive_sem_noc_addr)` (fabric atomic-inc "done"), then `stream.close()` (drains). |
| 5 | RECEIVER reader | `noc_semaphore_wait_min(sem, 1)` for "done"; then locally `noc_async_read` each landed packet from the intermediate and de-coalesce into `cb_output_pages`. |
| 6 | RECEIVER reader | `noc_semaphore_set(sem, 0)` — reset **AFTER** its wait (cache-reuse re-arm). |

Ordering guarantee: the sender does not transmit payload before the receiver's "ready"; the receiver does not consume before the sender's "done". Both sides reset the shared semaphore at the correct moment so a program-cache hit re-arms cleanly (sender before its inc, receiver after its wait) — the footgun documented at `ccl_helpers_dataflow.hpp:75-77`.

### Route derivation

`ttnn._ttnn.fabric.ccl_dm_route(mesh_device, from_coord, to_coord, topology)` returns `{num_hops, is_forward, neighbor_id}`, owning the fabric forward/backward sign reversal and the Ring shorter-way choice (`ccl_helpers_dataflow_host.hpp:137-166`; sign reversal at `:161`/`:165`). The SEND program uses `send_route = route(sender→receiver)`; the RECEIVE program uses `recv_route = route(receiver→sender)` for the ack. Kernels build the 1-D route from `num_hops` via `unicast_route(num_hops)` and pick the fwd/bwd connection from the leading `has_forward` flag of the fabric arg block.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one fabric packet (`total_packets` packets frame the whole shard) |
| Grid | single core `(0,0)` on each of the two participating devices; single fabric link (`_LINK_IDX = 0`) |
| Per-core work | sender core streams all `num_pages`, coalesces into `total_packets` packets, transmits all; receiver core lands and de-coalesces all `total_packets` |
| Remainder | last packet may hold fewer pages: `curr_pages_per_packet = min(max_pages_per_packet, pages_remaining)`; both sender coalescing and receiver de-coalescing recompute it per packet so counts agree |

Single-core / single-link is the design point: the shard is small relative to fabric bandwidth for the target shapes, and coalesce/de-coalesce is inherently sequential per link. Multi-core/multi-link striping is a future refinement (split `total_packets` across cores with `ttnn.split_work_to_cores`).

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|--------|----------|----------|----------|
| `cb_input_pages` | 0 | `aligned_input_page_size` = `round_up(input_page_size, l1_align)` | 2 (double-buffer) | input dtype | SEND reader (NCRISC) | SEND writer (BRISC) | streaming; one page in flight, popped after coalesce |
| `cb_output_pages` | 16 | `output_page_size` | `3 * pages_per_packet` | input dtype | RECEIVE reader (NCRISC) | RECEIVE writer (BRISC) | streaming; de-coalesced pages drained to output |
| `cb_packet_scratch` | 24 | `packet_size_bytes` | 1 | input dtype | reserved once as scratch (both programs) | working L1 for one coalesced/landed packet | whole op (single-slot scratch) |

CB band convention respected: `cb_input_pages` in the input band (0–7), `cb_output_pages` in the output band (16–23), `cb_packet_scratch` in the intermediate band (24–31).

### CB sync verification

| CB | Producer push | Consumer wait | Balanced? |
|----|---------------|---------------|-----------|
| `cb_input_pages` | SEND reader pushes 1 per page × `num_pages` | SEND writer waits 1 per page × `num_pages`, pops 1 each | ✅ |
| `cb_output_pages` | RECEIVE reader pushes 1 per page × `num_pages` | RECEIVE writer waits 1 per page × `num_pages`, pops 1 each | ✅ |
| `cb_packet_scratch` | reserved as raw scratch (reserve/push bookkeeping only; addressed directly by `write_page` / `noc_async_read`) | same kernel | ✅ (single slot, no cross-kernel handoff) |

Sizing rationale: `cb_input_pages` = 2 pages so the reader prefetches the next page while the writer coalesces. `cb_packet_scratch` = 1 packet — transient working L1 for exactly one packet at a time. `cb_output_pages` = `3 * pages_per_packet` gives the receiver reader headroom to de-coalesce a full packet's pages plus slack while the writer drains, without stalling per packet.

## API Mapping

Every mechanism has an exact file:line reference. This op is dataflow-only (no compute helpers).

| Phase | Type | Function | File:Line | Args / Template | Input CB | Output CB | Manages CB? |
|-------|------|----------|-----------|-----------------|----------|-----------|-------------|
| Host: packet framing | helper | `ttnn._ttnn.fabric.ccl_packet_dims` | `ccl_helpers_dataflow_host.hpp:74-96` (bf16 `bit_floor` at `:78`) | `(dtype, page_size_bytes, num_pages, l1_alignment)` → `packet_size_bytes / pages_per_packet / page_segments / total_packets` | — | — | n/a |
| Host: 1-D route | helper | `ttnn._ttnn.fabric.ccl_dm_route` | `ccl_helpers_dataflow_host.hpp:137-166` (sign reversal `:161`,`:165`) | `(mesh_device, from, to, topology)` → `num_hops / is_forward / neighbor_id`; owns fwd/bwd sign reversal + Ring short-way | — | — | n/a |
| Host: fabric RT args | helper | `ttnn.setup_fabric_connection` (mirrors `append_ccl_fabric_rt_args`) | `ccl_helpers_dataflow_host.hpp:219-237` (layout doc `:208-218`) | appends `[has_forward][fwd conn args?][has_backward][bwd conn args?]`; also mutates the `ProgramDescriptor` (adds SemaphoreDescriptors) | — | — | n/a |
| Host: cross-device sem | helper | `ttnn.create_global_semaphore` + `ttnn.synchronize_device` + `ttnn.get_global_semaphore_address` (Python analog of `make_ccl_semaphore`) | `ccl_helpers_dataflow_host.hpp:250-256` | create ONCE over worker grid, `synchronize_device` ONCE, cache, park on `mesh_program_descriptor.semaphores` | — | — | n/a |
| Sender: connection open | helper | `FabricStreamSender<>(conn_arg_idx, is_forward, alignment)` → `.open(unicast_route(num_hops))` | `ccl_helpers_dataflow.hpp:436-437`, `:448-451`; `unicast_route` `:260-265` | DirectConn ctor reads fabric RT block from `conn_arg_idx`; `open` binds the unicast route once | — | — | n/a |
| Sender: arm write | helper | `FabricStream::arm_unicast_write(payload_size_bytes)` | `ccl_helpers_dataflow.hpp:374` (impl `ccl_helpers_dataflow.inl:22-37`) | `set_state`s invariant per-packet payload size once; returns `UnicastWriteChannel` | — | — | no (op owns coalescing) |
| Sender: write packet | helper | `UnicastWriteChannel::write_page(packet_base_addr, packet_idx, intermediate)` | `ccl_helpers_dataflow.hpp:283-284` (impl `.inl:46-51`) | flow-controlled fabric write of one packet to receiver's per-packet intermediate slot | `cb_packet_scratch` (read) | receiver intermediate | no |
| Sender: arm/send "done" | helper | `FabricStream::arm_inc(1)` → `AtomicIncChannel::inc(receive_sem_noc_addr)` | `ccl_helpers_dataflow.hpp:386`, `:317` (impl `.inl:103-125`) | cross-device atomic-inc "done" | — | — | n/a |
| Sender: teardown | helper | `FabricStream::close()` (drains) | `ccl_helpers_dataflow.hpp:402` (impl `.inl:166-176`); `drain` `:399` | `noc_async_write_barrier` + `noc_async_atomic_barrier` then close; idempotent | — | — | n/a |
| Receiver: "ready" signal | helper | `FabricStreamSender<>::signal(sender_num_hops, sender_sem_noc_addr)` | `ccl_helpers_dataflow.hpp:459-461` (impl `.inl:182-189`) | one-shot `open→arm_inc→inc→close` "ready" handshake | — | — | n/a |
| Sender: input page read | raw_api | `TensorAccessor` + `noc_async_read` / `noc_async_read_barrier` | `tech_reports/tensor_accessor/tensor_accessor.md` | interleaved page read into `cb_input_pages`; page size from RT arg | input DRAM/L1 | `cb_input_pages` | reader owns CB reserve/push |
| Receiver: packet ingress | raw_api | `TensorAccessor::get_noc_addr` + `noc_async_read` / barrier | `tech_reports/tensor_accessor/tensor_accessor.md` | LOCAL read of a landed packet from the intermediate (no `FabricStreamReceiver`) | receiver intermediate | `cb_packet_scratch` | reader owns |
| Both: page↔packet copy | raw_api | `tt::data_movement::common::tt_memmove` | `ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp:87` | L1→L1 copy of one page/segment during coalesce (sender) / de-coalesce (receiver) | — | — | n/a |
| Both: wait / re-arm | raw_api | `noc_semaphore_wait_min`, `noc_semaphore_set` | `dataflow_api.h` | the WAITING half + cache-reuse reset — the helper does NOT own these by design | — | — | n/a |
| Receiver: output page write | raw_api | `TensorAccessor` + `noc_async_write` / `noc_async_write_barrier` | `tech_reports/tensor_accessor/tensor_accessor.md` | interleaved page write from `cb_output_pages` to output shard | `cb_output_pages` | output DRAM/L1 | writer owns CB wait/pop |

### Helpers considered and rejected (raw-API fallbacks)

- **Packet ingress `noc_async_read` (receiver reader).** Considered: `ccl_helpers_dataflow.hpp` receive API. Rejected — there is no `FabricStreamReceiver`; the banner states the receive ingress is a local NoC read the op owns (`ccl_helpers_dataflow.hpp:74`, `:89-93`). Concrete reason: the helper is fabric-*egress*-only; landing a packet from the local intermediate buffer is a plain local `noc_async_read` that no helper covers.
- **`noc_semaphore_wait_min` / `noc_semaphore_set` (both kernels).** Considered: `AtomicIncChannel::inc` and the `signal()` convenience. Rejected for the *waiting/reset* half — `inc`/`signal` own only the SENDING half of a cross-device sync; the banner explicitly assigns the WAITING half (`noc_semaphore_wait_min`) and the cache-reuse reset (`noc_semaphore_set(sem,0)`) to the op (`ccl_helpers_dataflow.hpp:69-77`, `:89-93`). Concrete reason: wrapping these would contradict the helper's documented ownership split.
- **`tt_memmove` page↔packet coalescing (both kernels).** Considered: `arm_scatter_write` / `write_scatter`. Rejected — scatter-write handles multi-chunk *fabric* writes to distinct destination NoC addresses, not L1→L1 *coalescing of contiguous shard pages into one packet buffer* (`ccl_helpers_dataflow.inl:57-97`). Concrete reason: coalesce/de-coalesce is local L1 packing the banner lists as op-owned ("page<->packet coalescing/segmentation" — `ccl_helpers_dataflow.hpp:89-93`); `tt_memmove` is the established primitive for it.
- **Interleaved page read/write (sender reader, receiver writer).** Considered: none applicable — this is stock `reader_unary_interleaved` / `writer_unary_interleaved` DRAM/L1 page streaming via `TensorAccessor`. No helper covers plain interleaved page movement.

## Data Movement Phases

Sequential contract (no compute; "CB State After" shows persistent/freed CBs).

### SEND program

| # | Operation | Input (CB / state) | Output (CB) | CB State After |
|---|-----------|--------------------|-------------|-----------------|
| S0 | Reserve `cb_packet_scratch` scratch slot (writer) | — | `cb_packet_scratch` (1 slot) | scratch reserved |
| S1 | Wait "ready", reset sem (writer, BEFORE its inc) | shared sem | — | sem == 0 |
| S2 | Open fabric stream, `arm_unicast_write`, `arm_inc` (writer) | `send_route` | armed channels | route bound |
| S3 | Stream input pages → `cb_input_pages` (reader) | input shard | `cb_input_pages` (1/page) | reader ahead by ≤2 pages |
| S4 | Coalesce pages → packet in `cb_packet_scratch`; `write_page` per full packet (writer) | `cb_input_pages` | receiver intermediate | `cb_input_pages` popped as consumed |
| S5 | `done.inc`, `stream.close()` (writer) | shared sem | receiver sem | payload + done delivered |

### RECEIVE program

| # | Operation | Input (CB / state) | Output (CB) | CB State After |
|---|-----------|--------------------|-------------|-----------------|
| R0 | `signal` "ready" to sender (reader) | `recv_route` | sender sem | handshake armed |
| R1 | Reserve `cb_packet_scratch` slot (reader) | — | `cb_packet_scratch` | scratch reserved |
| R2 | Wait "done" (reader) | shared sem | — | payload fully landed in intermediate |
| R3 | Local `noc_async_read` a landed packet → `cb_packet_scratch`; de-coalesce → `cb_output_pages` (reader) | intermediate | `cb_output_pages` (1/page) | packet scattered to pages |
| R4 | Stream `cb_output_pages` → output shard (writer) | `cb_output_pages` | output DRAM/L1 | receiver shard overwritten |
| R5 | Reset sem (reader, AFTER its wait) | shared sem | — | sem == 0 |

Output seeding: before dispatch, the op seeds `output == input` on **every** device (`ttnn.clone` when `output_tensor is None`, else `ttnn.copy(input, output_tensor)`). The RECEIVE writer then overwrites only the receiver shard, guaranteeing all non-participating shards remain equal to their input.

## Key Risks and Gotchas

| Risk | Mitigation |
|------|------------|
| **Cache-reuse semaphore corruption** — `GlobalSemaphore` reused across program-cache hits; missing re-arm = first run green, second hangs/corrupts (`ccl_helpers_dataflow.hpp:75-77`) | SENDER resets `noc_semaphore_set(sem,0)` BEFORE its outgoing inc; RECEIVER resets AFTER its wait. Both baked into the kernels. |
| **Semaphore lifetime across cache** | Create ONCE per `mesh_device` (module cache), `synchronize_device` ONCE at creation, park on `mesh_program_descriptor.semaphores`. No per-call re-create, no per-call post-dispatch barrier. |
| **Route direction sign** — fabric fwd/bwd is sign-reversed vs. coordinate delta | Always derive via `ccl_dm_route` (owns the reversal at `ccl_helpers_dataflow_host.hpp:161`,`:165`). Never hand-compute direction. |
| **bf16 packet sizing** — bf16 needs a `bit_floor` on the max packet size | Always frame via `ccl_packet_dims` (owns it at `ccl_helpers_dataflow_host.hpp:78`). Sender and receiver MUST use the same result so coalesce/de-coalesce loops agree. |
| **Page > packet vs. page < packet** — both regimes must be handled | `page_segments` (split one page across packets) and `pages_per_packet` (pack many pages) both come from `ccl_packet_dims`; per-packet `curr_pages_per_packet` recomputed identically on both sides. |
| **Non-participating shards must stay unchanged** | Seed `output == input` on every device before dispatch; only the receiver program writes. |
| **Fabric arg block position** — kernels read the connection block at a fixed RT index | SEND writer and RECEIVE reader place the `[has_forward][fwd?][has_backward][bwd?]` block starting at RT index 9 (after 9 scalar args). The leading `has_forward` flag doubles as the kernel's `is_forward`. |
| **Intermediate `element_size` undefined for block-float** | Stage the intermediate as `uint32` `[total_packets, packet_size_bytes/4]` row-major; `packet_size_bytes` is a multiple of L1 alignment (≥16), hence of 4. |
| **Page size stale on cache hit** — compile-time `AlignedPageSize` may be stale | Interleaved reader/writer take the page size as a RUNTIME arg (index 3) and override the `TensorAccessor` page size. |
| **Verification topology mismatch = fabric-init hang** | The acceptance test MUST open exactly mesh `(2,4)` with `FABRIC_1D` (per `run_multidevice_sim_pytest.py --list` → `bh_8xP150_p2p`). A `(1,2)`/`(1,4)` mesh hangs fabric init with `Fabric Router Sync: Timeout` — a test/topology mismatch, not an op defect. |

## Structural impossibilities (feature_spec.py INVALID)

`eval/golden_tests/point_to_point/feature_spec.py` already exists (pipeline mode — authoritative, not edited here). Its single `INVALID` entry is consistent with this design:

| INVALID cell | Rationale |
|--------------|-----------|
| `{"dtype": ttnn.bfloat8_b, "layout": ttnn.ROW_MAJOR_LAYOUT}` | Single-tensor coupling (dtype + layout both describe the input). `bfloat8_b` is a block-quantized *tiled* format with no ROW_MAJOR representation — a data-format-definition impossibility, not a not-yet-implemented EXCLUSION. |

No additional op-specific INVALID candidates: `topology` is orthogonal to `dtype`/`layout`; the 16-byte page-size constraint is a shape×dtype `validate()` gate (kept satisfiable by every INPUTS shard — last dim a multiple of 8), not modeled as an axis.

## Registry-model axes (mirror in the op file)

| Axis | TARGET (feature_spec) | Notes |
|------|-----------------------|-------|
| `dtype` | `bfloat16, float32, bfloat8_b, uint16, int32, uint32` | pure byte movement — every fixed-width dtype is correct |
| `layout` | `TILE_LAYOUT, ROW_MAJOR_LAYOUT` | copied verbatim; layout preserved end to end |
| `topology` | `Linear, Ring` | op kwarg; route-only effect |
| `alignment` (shape-derived) | `tile_aligned, non_tile_aligned` | tagged from the shard's last two dims; op is alignment-agnostic (copies physical pages) |
