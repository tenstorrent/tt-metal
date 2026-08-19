# Operation Design: point_to_point

## Overview

| Field | Value |
|-------|-------|
| Classification | CCL (multi-device collective communication), dataflow-only |
| Goal | Copy one mesh device's interleaved shard of a mesh-sharded tensor to another mesh device across the TT-Fabric, leaving every other device's shard untouched. |
| Math | `output_shard[receiver_coord][i] = input_shard[sender_coord][i]` for every element `i`; `output_shard[c] = output_shard_on_entry[c]` for every `c != receiver_coord`. No arithmetic. |
| Mode | Derivative (newly authored from scratch on `ttnn.generic_op` + `ttnn.MeshProgramDescriptor`) |
| Compute kernel | **None.** The op runs only dataflow kernels (NCRISC reader + BRISC writer) on one worker core of each of the two participating devices. |
| References | `ttnn/cpp/ttnn/kernel_lib/ccl_helpers_dataflow.hpp` (+`.inl`), `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp`, `ttnn/cpp/ttnn/operations/ccl/common/host/ccl_helpers_dataflow_host.hpp`, `ttnn/ttnn/operations/all_gather/` (proven Python `generic_op` CCL op), `ttnn/cpp/ttnn/operations/point_to_point/device/kernels/dataflow/{writer_send,reader_receive}.cpp` (correctness reference **only** — not wrapped, not imported, not dispatched to) |

### Generation mandate

This op is generated **from scratch**. It must NOT re-export, import, call, wrap, or dispatch to
`ttnn.point_to_point` / `ttnn._ttnn.operations.point_to_point`. The C++ op and the `all_gather_async`
kernels may be *read* as a correctness reference. Everything shipped by this design is newly authored:
four `.cpp` dataflow kernels under `ttnn/ttnn/operations/point_to_point/kernels/` plus three Python
modules under `ttnn/ttnn/operations/point_to_point/`.

### Files to produce

| Path | Purpose |
|------|---------|
| `ttnn/ttnn/operations/point_to_point/__init__.py` | Re-exports `point_to_point`, `INPUT_TAGGERS`, `SUPPORTED`, `EXCLUSIONS`, `validate` |
| `ttnn/ttnn/operations/point_to_point/point_to_point.py` | Registry-model declarations, `validate()`, public entry point, semaphore cache |
| `ttnn/ttnn/operations/point_to_point/point_to_point_program_descriptor.py` | `MeshProgramDescriptor` assembly (two per-coordinate programs) |
| `ttnn/ttnn/operations/point_to_point/kernels/point_to_point_sender_reader.cpp` | Sender NCRISC: input DRAM → `cb_shard_pages` |
| `ttnn/ttnn/operations/point_to_point/kernels/point_to_point_sender_writer.cpp` | Sender BRISC: handshake, packet framing, fabric egress |
| `ttnn/ttnn/operations/point_to_point/kernels/point_to_point_receiver_reader.cpp` | Receiver NCRISC: ack, wait, intermediate read-back, de-framing |
| `ttnn/ttnn/operations/point_to_point/kernels/point_to_point_receiver_writer.cpp` | Receiver BRISC: `cb_output_pages` → output DRAM |

### Public signature (exact)

```python
from ttnn.operations.point_to_point import point_to_point

point_to_point(
    input_tensor: ttnn.Tensor,                       # sharded across a MeshDevice
    sender_coord: ttnn.MeshCoordinate,               # device holding the shard to send
    receiver_coord: ttnn.MeshCoordinate,             # device that receives the shard
    topology: ttnn.Topology = ttnn.Topology.Linear,  # Linear or Ring
    output_tensor: ttnn.Tensor | None = None,        # write into existing tensor
    intermediate_tensor: ttnn.Tensor | None = None,  # optional staging tensor
) -> ttnn.Tensor
```

All six parameters are positional-or-keyword, in exactly this order. Note the coordinate order is
`(sender, receiver)` — the **opposite** of the bound C++ op's `(receiver, sender)`; do not copy that
ordering.

`ttnn.Topology` binds late at import time; the module must reference
`from ttnn._ttnn.operations.ccl import Topology as _Topology` for the default argument value
(same pattern as `ttnn/ttnn/operations/all_gather/all_gather.py:26`).

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | rank ≥ 2, interleaved, DRAM or L1, on a `ttnn.MeshDevice`, page size a multiple of `ttnn.get_l1_alignment()` | — | host (buffer address → RT; `TensorAccessorArgs` → CT) |
| `sender_coord` | `ttnn.MeshCoordinate` | yes | inside `mesh_device.shape`; `!= receiver_coord`; same mesh row or column as `receiver_coord` | — | host only (selects the `MeshCoordinateRange` for the send program) |
| `receiver_coord` | `ttnn.MeshCoordinate` | yes | inside `mesh_device.shape`; `!= sender_coord` | — | host only |
| `topology` | `ttnn.Topology` | no | `Linear`, `Ring` | `Linear` | host only (feeds `ccl_dm_route`) |
| `output_tensor` | `ttnn.Tensor \| None` | no | shape/dtype/layout/memory_config == input's | `None` | host (buffer address → RT) |
| `intermediate_tensor` | `ttnn.Tensor \| None` | no | must match the resolved staging spec below | `None` | host (buffer address → RT; `TensorAccessorArgs` → CT) |

### Registry-model axes (`SUPPORTED` gate)

`TARGET`/`INPUTS`/`INVALID` already exist and are authoritative at
`eval/golden_tests/point_to_point/feature_spec.py`. The op file declares the matching axis names:

| Axis | Source | Values in `TARGET` |
|------|--------|--------------------|
| `dtype` | `input_tensor.dtype` | `bfloat16, float32, bfloat8_b, uint16, int32, uint32` |
| `layout` | `input_tensor.layout` | `TILE_LAYOUT, ROW_MAJOR_LAYOUT` |
| `topology` | kwarg | `Topology.Linear, Topology.Ring` |
| `alignment` | `INPUT_TAGGERS["alignment"]` — both of the shard's last two dims divisible by 32 → `"tile_aligned"`, else `"non_tile_aligned"` | `tile_aligned, non_tile_aligned` |

**Recommended Phase-0 `SUPPORTED` = all of `TARGET`.** This op is a pure byte copy: nothing in the
data path inspects the element type or the tile grid (see *Format agnosticism* under Key Risks), so
every dtype × layout × alignment cell is reachable with a single code path. `EXCLUSIONS` starts empty.
The one dependency that can force an exclusion is the output-seeding copy (see *Output contract*).

### Structural impossibilities

`INVALID` in `eval/golden_tests/point_to_point/feature_spec.py` is `[{dtype: bfloat8_b, layout: ROW_MAJOR_LAYOUT}]`.
That is complete for this op's axis set: `bfloat8_b` is a block-quantized tiled format with no
row-major representation (single-tensor coupling — both axes describe `input_tensor`). `topology` is
orthogonal to `dtype`/`layout`. The 16-byte page-size rule is a shape × dtype `validate()` gate, not an
axis. **No additions requested.** This design does not edit `feature_spec.py`.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | Per-device shard shape, rank ≥ 2 |
| Dtype | `bfloat16` (primary), `float32`, `bfloat8_b` (TILE only), `uint16`, `int32`, `uint32` |
| Layout | `TILE_LAYOUT` (primary) or `ROW_MAJOR_LAYOUT` |
| Memory | Interleaved, `DRAM` or `L1`. Sharded input is rejected. |
| Device | `ttnn.MeshDevice` with ≥ 2 devices on the row/column joining `sender_coord` and `receiver_coord` |
| Page | `input_tensor.buffer_page_size()` must be a multiple of `ttnn.get_l1_alignment()` (16 B) |

### Output

| Property | Value |
|----------|-------|
| Shape | identical to `input_tensor.shape` |
| Dtype | identical to `input_tensor.dtype` |
| Layout | identical to `input_tensor.layout` |
| Memory | identical to `input_tensor.memory_config()` |
| Contents | `receiver_coord` shard == `sender_coord` input shard, bit-for-bit; every other coordinate's shard is byte-for-byte what it was on entry |

### Intermediate (staging) tensor — resolved spec

The staging tensor is a **raw-byte packet buffer**, deliberately decoupled from the payload dtype and
layout so that a single code path serves every `TARGET` cell:

| Property | Value |
|----------|-------|
| Shape | `ttnn.Shape([total_packets, packet_size_bytes // 4])` |
| Dtype | `ttnn.uint32` |
| Layout | `ttnn.ROW_MAJOR_LAYOUT` |
| Memory | `ttnn.DRAM_MEMORY_CONFIG` (interleaved) |
| Resulting `buffer_page_size()` | exactly `packet_size_bytes` (one row = one packet) |
| Resulting `buffer_num_pages()` | exactly `total_packets` |

`packet_size_bytes` is always a multiple of 16 (see *Packet framing*), so `packet_size_bytes // 4` is
exact. Allocated with `ttnn.allocate_tensor_on_device(shape, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh_device, ttnn.DRAM_MEMORY_CONFIG)`.
Because it is a mesh tensor, its device address is identical on every mesh device — which is what lets
the **sender** address the **receiver's** staging buffer using its own local base address.

If the caller supplies `intermediate_tensor`, `validate()` compares shape, dtype, layout and
memory-config against this resolved spec and raises `ValueError` on any mismatch.

### Output contract and how the default output is allocated

The op's programs write **only** the receiver device's shard. Therefore:

| Path | Output allocation | Non-receiver shards after the call |
|------|-------------------|------------------------------------|
| `output_tensor` supplied | used as-is | exactly the caller's prior contents |
| `output_tensor is None` | `output_tensor = ttnn.clone(input_tensor)` | equal to the corresponding **input** shards |

`ttnn.clone` with no dtype/memory-config override is a same-dtype, same-layout, same-memory device
copy; its only layout restriction applies to dtype *conversion*
(`ttnn/cpp/ttnn/operations/data_movement/clone/device/clone_device_operation.cpp:15`), which this call
never requests. Seeding is what makes "all other devices' shards are untouched" a *total* statement
rather than an undefined one. If a `TARGET` (dtype, layout) cell turns out not to be clonable, that
cell moves to `EXCLUSIONS` (a refinement candidate) — the contract itself does not fork.

## Dataflow Strategy

### Mesh-level shape

```
        ┌──────────────── mesh_device (1, N), FABRIC_1D ────────────────┐
        │                                                               │
   coord (0,0)          (0,1)              (0,2)            (0,3)       │
   ┌───────────┐   ┌───────────┐     ┌───────────┐    ┌───────────┐     │
   │ SEND prog │══>│ (no prog) │════>│ RECV prog │    │ (no prog) │     │
   └───────────┘   └───────────┘     └───────────┘    └───────────┘     │
       core(0,0)     fabric relay        core(0,0)                      │
```

Exactly two `(MeshCoordinateRange, ProgramDescriptor)` entries are installed in the
`ttnn.MeshProgramDescriptor` — one covering only `sender_coord`, one covering only `receiver_coord`.
Every other mesh coordinate receives an **empty** `ProgramDescriptor` from the generic-op factory
(`ttnn/cpp/ttnn/operations/generic/device/generic_op_program_factory.cpp:12-35`), i.e. runs no program.
Intermediate hops are pure fabric routing — no Tensix on a relay chip participates.

### Intra-Tensix path (both participating devices use logical core `(0, 0)`)

```
SENDER device, core (0,0)
  NCRISC (reader)                       BRISC (writer)
  input DRAM ─noc_async_read─> cb_shard_pages ─tt_memmove─> cb_packet_staging
                                                                   │
                                                    FabricStream::write_page (fabric)
                                                                   ▼
RECEIVER device, core (0,0)                        receiver's INTERMEDIATE DRAM (page == packet)
  NCRISC (reader)                                                  │
        <──────────────── noc_async_read (local) ──────────────────┘
        ─tt_memmove─> cb_output_pages
  BRISC (writer)
        cb_output_pages ─noc_async_write─> output DRAM
```

The **only** cross-chip traffic is (a) the payload packets, sender → receiver intermediate, and (b)
two fabric atomic-increments (the handshake). The receiver has no fabric ingress kernel: the fabric
lands the payload directly into the receiver's DRAM, and the receiver's read-back is a plain local
`noc_async_read` the op owns (`ccl_helpers_dataflow.hpp:109-110` — "the receive INGRESS is likewise a
local NoC read the op owns; there is no FabricStreamReceiver").

### Tensix-to-Tensix (cross-chip) contract

One op-internal `GlobalSemaphore` is shared by both endpoints. It is a mesh-wide L1 allocation, so its
**absolute address is identical on every device**, and both endpoints use logical core `(0, 0)`, so the
NoC `(x, y)` is identical too. `get_noc_addr(sem_addr)` computed on either core therefore names "the
same semaphore, on the chip the packet is routed to".

| Step | Actor | Action | Why |
|------|-------|--------|-----|
| 1 | Receiver NCRISC | `FabricStreamSender::signal(sender_num_hops, get_noc_addr(sem_addr))` — one-shot fabric atomic-inc of the **sender's** semaphore | "I am launched and my local semaphore is clean" |
| 2 | Sender BRISC | `noc_semaphore_wait_min(local_sem, 1)` | do not transmit before the receiver of *this* invocation is ready |
| 3 | Sender BRISC | `noc_semaphore_set(local_sem, 0)` | **re-arm before the outgoing inc** — cache-reuse rule, `ccl_helpers_dataflow.hpp:111-113` |
| 4 | Sender BRISC | open stream, stream all `total_packets` fabric writes into the receiver's intermediate | payload |
| 5 | Sender BRISC | `AtomicIncChannel::inc(get_noc_addr(sem_addr))` on the **receiver's** semaphore, then `stream.close()` | "payload fully sent". `close()` drains write + atomic barriers before tearing down (`ccl_helpers_dataflow.inl:186-195`), so the inc can never overtake or be lost |
| 6 | Receiver NCRISC | `noc_semaphore_wait_min(local_sem, 1)` | do not consume before the payload has fully landed |
| 7 | Receiver NCRISC | read back, de-frame, push `cb_output_pages` | |
| 8 | Receiver NCRISC | `noc_semaphore_set(local_sem, 0)` | **re-arm after the wait** — cache-reuse rule, `ccl_helpers_dataflow.hpp:111-113` |

**Ordering guarantee that step 1 buys.** The two devices' command queues are independent, so without
the ready-handshake the sender of invocation *k+1* could increment the receiver's semaphore while the
receiver is still inside invocation *k*; the step-8 reset of invocation *k* would then erase invocation
*k+1*'s "done" and the receiver would hang. Gating the sender on the receiver's own ready-inc — which
the receiver can only issue after its previous program on that device retired — makes that impossible.

**Fabric ordering guarantee that step 5 relies on.** All payload writes and the trailing atomic-inc are
issued on the *same* fabric connection with the same route, so they are delivered in issue order; the
receiver observing `sem >= 1` implies every payload byte has landed.

### Packet framing

The op never invents packet sizing — it calls the bound host helper
`ttnn._ttnn.fabric.ccl_packet_dims(dtype, page_size_bytes, num_pages, alignment)`
(`ttnn/cpp/ttnn-nanobind/fabric.cpp:236-266`, impl
`ttnn/cpp/ttnn/operations/ccl/common/host/ccl_helpers_dataflow_host.hpp:74-96`), which owns the
`bfloat16` `std::bit_floor` special case on the fabric channel buffer size.

```
l1_align      = ttnn.get_l1_alignment()                      # 16 B on Wormhole and Blackhole
page_size     = input_tensor.buffer_page_size()              # logical page: tile bytes, or RM stick bytes
num_pages     = input_tensor.buffer_num_pages()
aligned_page  = round_up(page_size, l1_align)                # per-page stride INSIDE a packet (L1)
dims          = ccl_packet_dims(input_tensor.dtype, page_size, num_pages, l1_align)
packet_size   = dims.packet_size_bytes
pages_per_pkt = dims.pages_per_packet
page_segments = dims.page_segments
total_packets = dims.total_packets
```

Two disjoint regimes, both of which this op must implement:

| Regime | Condition | `page_segments` | `pages_per_pkt` | `packet_size` | `total_packets` | Meaning |
|--------|-----------|-----------------|-----------------|---------------|-----------------|---------|
| **A — coalesce** | `aligned_page <= max_packet` | `1` | `min(max_packet // aligned_page, num_pages)` | `aligned_page * pages_per_pkt` | `ceil(num_pages / pages_per_pkt)` | several shard pages ride in one fabric transfer |
| **B — segment** | `aligned_page > max_packet` | `ceil(aligned_page / max_packet)` | `1` | `max_packet` | `page_segments * num_pages` | one shard page is split across several fabric transfers |

`packet_size` is a multiple of 16 in both regimes (`aligned_page` is 16-aligned; the fabric channel
buffer size and its `bit_floor` are 16-aligned), so `arm_unicast_write(packet_size)`'s internal
`align(payload, l1_align)` (`ccl_helpers_dataflow.inl:53`) is a no-op and the on-wire size is exactly
`packet_size`.

**Sender framing (spec — the implementer writes the loop):**

```
staging = get_write_ptr(cb_packet_staging)          # reserved once, used as raw L1 scratch

if page_segments == 1:                               # regime A
    for pkt in 0 .. total_packets-1:
        n = min(pages_per_pkt, num_pages - pkt * pages_per_pkt)
        for k in 0 .. n-1:
            cb_wait_front(cb_shard_pages, 1)
            tt_memmove(staging + k * aligned_page, get_read_ptr(cb_shard_pages), page_size)
            cb_pop_front(cb_shard_pages, 1)
        writer.write_page(staging, pkt, intermediate_acc)
else:                                                # regime B
    pkt = 0
    for p in 0 .. num_pages-1:
        cb_wait_front(cb_shard_pages, 1); src = get_read_ptr(cb_shard_pages)
        for s in 0 .. page_segments-1:
            off = s * packet_size
            tt_memmove(staging, src + off, min(page_size - off, packet_size))
            writer.write_page(staging, pkt, intermediate_acc); pkt += 1
        cb_pop_front(cb_shard_pages, 1)
```

**Receiver de-framing (exact mirror):**

```
landing = get_write_ptr(cb_packet_landing)

if page_segments == 1:                               # regime A
    for pkt in 0 .. total_packets-1:
        noc_async_read(intermediate_acc.get_noc_addr(pkt), landing, packet_size); noc_async_read_barrier()
        n = min(pages_per_pkt, num_pages - pkt * pages_per_pkt)
        for k in 0 .. n-1:
            cb_reserve_back(cb_output_pages, 1)
            tt_memmove(get_write_ptr(cb_output_pages), landing + k * aligned_page, page_size)
            cb_push_back(cb_output_pages, 1)
else:                                                # regime B
    pkt = 0
    for p in 0 .. num_pages-1:
        cb_reserve_back(cb_output_pages, 1); dst = get_write_ptr(cb_output_pages)
        for s in 0 .. page_segments-1:
            noc_async_read(intermediate_acc.get_noc_addr(pkt), landing, packet_size); noc_async_read_barrier()
            pkt += 1
            off = s * packet_size
            tt_memmove(dst + off, landing, min(page_size - off, packet_size))
        cb_push_back(cb_output_pages, 1)
```

The final packet of regime A carries a full `packet_size` payload even when `n < pages_per_pkt`; the
trailing bytes are stale staging content that lands in the intermediate's tail and is never read back.
This is intentional — the armed payload size is a per-stream invariant
(`ccl_helpers_dataflow.hpp:486`, `.inl:49-53`) and the intermediate is sized for it.

### The alignment rule that makes non-tile-aligned row-major shards correct

Blackhole `DRAM_ALIGNMENT = 2^6 = 64 B`, `L1_ALIGNMENT = 2^4 = 16 B`
(`tt_metal/hw/inc/internal/tt-1xx/blackhole/core_config.h:41-42`). An interleaved DRAM buffer therefore
strides pages within a bank by `align(page_size, 64)`, not by `page_size`. A row-major shard whose row
is e.g. 96 B (`(1,1,32,48)` bfloat16) has a **128 B** per-bank stride.

`TensorAccessor` resolves `addr = bank_start + base + bank_page_offset * aligned_page_size + offset`
(`tt_metal/hw/inc/api/tensor/tensor_accessor.h:315`), where `aligned_page_size` comes from the
constructor's third argument, whose default is the compile-time-baked
`TensorAccessorArgs::AlignedPageSize` (`tensor_accessor.h:83-87`, `tensor_accessor_args.h:44`) — and the
host puts `buffer.aligned_page_size()` there
(`tt_metal/impl/buffers/tensor_accessor_args.cpp:46-52`).

> **MANDATORY: every `TensorAccessor` in this op is constructed with exactly two arguments —
> `TensorAccessor(args, base_address)`. Never pass a runtime page-size override.**
> The NoC transfer size (`page_size` / `packet_size`) is a *separate* runtime argument and is used only
> as the `noc_async_read` / `noc_async_write` byte count, never as the accessor's stride.

Passing the raw logical page size as the third argument forces `aligned_page_size = page_size`, which
mis-addresses every page beyond the first bank row whenever `page_size % 64 != 0`. That is the exact
failure mode of the reference C++ kernels
(`ttnn/cpp/ttnn/operations/point_to_point/device/kernels/dataflow/reader_unary_interleaved_start_id_gen.cpp:23`
and `writer_unary_interleaved_start_id_gen.cpp:21`), and it is why row-major shards such as
`(1,1,24,24)`, `(1,1,32,48)` bf16/uint16, and `(1,1,56,88)` corrupt while every TILE shard (tile page
sizes 1088 / 2048 / 4096 are all multiples of 64) is fine. This design does not inherit that bug.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | One shard page on the reader/writer side; one fabric packet on the framing side |
| Grid | **One worker core, `ttnn.CoreCoord(0, 0)`, on each of the two participating devices.** No cores anywhere else. |
| Per-core work | All `num_pages` shard pages and all `total_packets` fabric packets |
| Remainder | None — a single core owns the whole range |
| Rationale | The transfer uses a single fabric link (`link_idx = 0`). Sharing one link between several worker cores requires the mux connection policy (`MuxConn<N>`, `ccl_helpers_dataflow.hpp:202-295`), which is out of scope for this op. Splitting across links would also need a per-link route and per-link staging region. The reference factory makes the same choice (`send_program_factory.cpp:39-42`, `use_cores = {1,1}`, with the comment "eventually add more cores for multi-link"). |
| Scaling note | The natural multi-core extension is `ttnn.split_work_to_cores(grid, total_packets)` with one `MuxConn` client per core; deferred as a refinement. |

## Circular Buffers

Two independent CB sets — the two devices run different programs, so indices do not collide.
All four CBs use `data_format = ttnn.uint32`: there is no compute kernel, the CB format is inert for
pure byte movement, and a `uint32` format keeps `bfloat8_b` payloads (whose CB format would otherwise
demand tile-shaped pages) on the same code path.

`aligned_page = round_up(page_size, ttnn.get_l1_alignment())`.

### Sender device, core (0, 0)

| Semantic Name | Index | Page Size | Num Pages | Total Size | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------|--------|----------|----------|----------|
| `cb_shard_pages` | 0 | `aligned_page` | 2 | `2 * aligned_page` | `uint32` | sender reader (NCRISC) | sender writer (BRISC) | streaming, whole kernel |
| `cb_packet_staging` | 24 | `packet_size` | 1 | `packet_size` | `uint32` | sender writer (scratch) | sender writer | scratch, whole kernel |

### Receiver device, core (0, 0)

| Semantic Name | Index | Page Size | Num Pages | Total Size | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------|--------|----------|----------|----------|
| `cb_packet_landing` | 24 | `packet_size` | 1 | `packet_size` | `uint32` | receiver reader (scratch) | receiver reader | scratch, whole kernel |
| `cb_output_pages` | 16 | `aligned_page` | 2 | `2 * aligned_page` | `uint32` | receiver reader (NCRISC) | receiver writer (BRISC) | streaming, whole kernel |

### Sizing rationale

| CB | Why this size |
|----|---------------|
| `cb_shard_pages` | Double-buffered streaming: the NCRISC reads page *p+1* from DRAM while the BRISC memmoves page *p* into the packet. Depth 2 is sufficient because the framing loop consumes strictly in FIFO order, one page at a time. Page size is `aligned_page`, not `page_size`, so every slot starts 16-B aligned (a `tt_memmove` source must be L1-aligned) and matches the intra-packet stride exactly. |
| `cb_packet_staging` | Exactly one packet. Regime A needs `pages_per_pkt * aligned_page == packet_size`; regime B needs `packet_size` (`pages_per_pkt == 1`, one segment at a time at offset 0). One slot suffices because `write_page` → `fabric_unicast_noc_unicast_write_with_state` copies the payload into the fabric channel buffer before returning (`ccl_helpers_dataflow.inl:59-63`), so the buffer is free for the next packet immediately. |
| `cb_packet_landing` | Exactly one packet; the local `noc_async_read` + `noc_async_read_barrier` completes before the de-framing memmoves touch it. |
| `cb_output_pages` | Double-buffered streaming: the NCRISC assembles page *p+1* while the BRISC writes page *p* to DRAM. |

### CB synchronisation ledger

| CB | Producer pushes | Consumer waits/pops | Balanced |
|----|-----------------|---------------------|----------|
| `cb_shard_pages` | `num_pages` (`cb_push_back(...,1)` per page) | `num_pages` (`cb_wait_front(...,1)` / `cb_pop_front(...,1)` per page) | ✅ |
| `cb_output_pages` | `num_pages` | `num_pages` | ✅ |
| `cb_packet_staging` | 0 pushes | 0 waits | ✅ (reserve-once scratch: `cb_reserve_back(...,1)` at kernel start to claim the L1 region, then raw `get_write_ptr` addressing; producer and consumer are the same kernel, so no CB handshake exists to balance) |
| `cb_packet_landing` | 0 pushes | 0 waits | ✅ (same reserve-once scratch pattern) |

Every `cb_wait_front` on a given CB uses page count `1`, uniformly.

## API Mapping

### Kernel-side helpers (used)

| Phase | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Owns CB ops? | Notes |
|-------|------|----------|-----------|------------------------|----------|-----------|--------------|-------|
| Sender fabric connection | helper | `dataflow_kernel_lib::ccl::FabricStreamSender<>` ctor | `ccl_helpers_dataflow.hpp:577-578` | `(size_t& conn_arg_idx, bool is_forward, uint32_t alignment)` | — | — | no | Consumes the whole fabric RT-arg block by reference and advances `conn_arg_idx` (`DirectConn` ctor, `:166-169`). Declare the sender **before** the stream — the stream borrows the connection (`:453-457`). |
| Sender route bind | helper | `FabricStreamSender::open` | `ccl_helpers_dataflow.hpp:589-592` | `(const line_unicast_route_info_t& route)` | — | — | no | Route bound ONCE here; every `arm_*` reuses it. |
| Sender route value | helper | `dataflow_kernel_lib::ccl::unicast_route` | `ccl_helpers_dataflow.hpp:307-312` | `(uint32_t num_hops)` | — | — | no | `num_hops` from `ccl_dm_route(...).num_hops`. |
| Sender payload arm | helper | `FabricStream::arm_unicast_write` | `ccl_helpers_dataflow.hpp:486` / `.inl:42-56` | `(uint32_t page_size_bytes = packet_size)` | — | — | no | Invariant on-wire size `align(packet_size, alignment)`; draws its own pooled packet header. |
| Sender inc arm | helper | `FabricStream::arm_inc` | `ccl_helpers_dataflow.hpp:498` / `.inl:123-137` | `(uint32_t val = 1)` | — | — | no | Independent pooled header from the write channel — both may be live at once (`:113-118`). |
| Sender packet issue | helper | `UnicastWriteChannel::write_page` | `ccl_helpers_dataflow.hpp:330-331` / `.inl:67-70` | `(uint32_t src_l1_addr, uint32_t page_idx, const AddrGen& dst)` | `cb_packet_staging` (raw L1) | remote intermediate DRAM | no | Resolves the destination via `addrgen_detail::get_noc_address(dst, page_idx, 0)` — feed it the **2-arg** `TensorAccessor`. |
| Sender done signal | helper | `AtomicIncChannel::inc` | `ccl_helpers_dataflow.hpp:364` / `.inl:140-144` | `(uint64_t remote_sem_noc_addr)` | — | — | no | `remote_sem_noc_addr = get_noc_addr(sem_addr)`. |
| Sender teardown | helper | `FabricStream::close` | `ccl_helpers_dataflow.hpp:540` / `.inl:186-195` | `()` | — | — | no | Drains write + atomic barriers, then closes. Idempotent; the destructor closes too. Call it explicitly. |
| Receiver ack | helper | `FabricStreamSender::signal` | `ccl_helpers_dataflow.hpp:600-602` / `.inl:202-208` | `(uint32_t num_hops, uint64_t remote_sem_noc_addr, uint32_t val = 1)` | — | — | no | One-shot open → `arm_inc` → `inc` → `close`. **Terminal** — do not also call `open()` on that sender. |
| Semaphore pointer (both endpoints) | helper | `dataflow_kernel_lib::addr_to_l1_ptr` | `l1_helpers.hpp:25-27` | `(uint32_t addr) -> volatile tt_l1_ptr uint32_t*` | — | — | no | Use instead of a hand-rolled `reinterpret_cast<volatile tt_l1_ptr uint32_t*>`. |

### Kernel-side raw APIs (with mandatory justification)

| Phase | Raw API | File:Line | Why raw |
|-------|---------|-----------|---------|
| Wait halves of the handshake | `noc_semaphore_wait_min(ptr, 1)` | `tt_metal` dataflow API | **Helpers considered and rejected:** `ccl_helpers_dataflow.hpp` — the banner states at `:104-108` that "The SENDING half of a cross-device sync — a remote atomic-inc — is owned here … The WAITING half is a plain local `noc_semaphore_wait_min(sem, threshold)` the op calls directly … a stock dataflow call, not renamed." There is **no** wait helper in the header (`grep -n "wait" ccl_helpers_dataflow.hpp` yields only doc prose). |
| Semaphore re-arm | `noc_semaphore_set(ptr, 0)` | `tt_metal` dataflow API | **Helpers considered and rejected:** `ccl_helpers_dataflow.hpp:111-113` explicitly assigns the reset to the op: "each side must `noc_semaphore_set(sem, 0)` to re-arm — a SENDER resets BEFORE its outgoing inc, a RECEIVER after its wait." No helper exists. |
| Remote semaphore NoC address | `get_noc_addr(sem_addr)` | `tt_metal` dataflow API | **Helpers considered and rejected:** `l1_helpers.hpp:36-40` `local_noc_addr()` returns a `noc_traits_t<UnicastEndpoint>::src_args_type` struct for the new `Noc` object API, **not** the `uint64_t` NoC address that `AtomicIncChannel::inc` (`ccl_helpers_dataflow.hpp:364`) and `noc_semaphore_wait_min` require. Type mismatch, not preference. |
| Local DRAM ↔ L1 page moves (sender reader, receiver reader read-back, receiver writer) | `TensorAccessor::get_noc_addr`, `noc_async_read`, `noc_async_read_barrier`, `noc_async_write`, `noc_async_write_barrier` | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:83-87,315` | **Helpers considered and rejected:** (1) `ccl_helpers_dataflow.hpp:130-140` — the banner lists what the helper does **not** own, naming "address generation (TensorAccessor/ShardedAddrGen is consumed, never re-wrapped)" and, at `:109-110`, "The receive INGRESS is likewise a local NoC read the op owns; there is no FabricStreamReceiver." (2) `dfb_helpers_dataflow.hpp:14-19` — its entire surface is `get_tile_r_dim()` / `get_tile_c_dim()`, tile-dimension queries for `DataflowBuffer`s; this op moves opaque pages and never needs tile dimensions. (3) `tilize_helpers.hpp` / `untilize_helpers.hpp` — compute-thread (unpack/math/pack) APIs; this op has no compute kernel and must preserve layout byte-for-byte, so tilizing would be a semantic error, not just overhead. |
| Page ↔ packet coalescing / segmentation | `tt::data_movement::common::tt_memmove<false,false,false,0>` | `ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp:87` | **Helpers considered and rejected:** `ccl_helpers_dataflow.hpp:130-133` — the banner's "does NOT own" list names "page<->packet coalescing/segmentation" first. `ScatterWriteChannel` (`:341-356`) is the nearest fabric-side alternative but caps at 4 chunks per packet ("the NocUnicastScatter limit", `:341`) and scatters to ≤ 4 *destinations*; this op needs up to `pages_per_pkt` (85 for a 48-byte row-major page) contiguous pages in one payload with a **single** destination. |
| Intra-packet stride rounding | `tt::data_movement::common::round_up(a, b)` | `ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp:196` | Arithmetic utility from the same header as `tt_memmove`; no helper-library equivalent exists. |

### Host-side helpers (used)

| Phase | Type | Function | File:Line | Args / Returns |
|-------|------|----------|-----------|----------------|
| Packet sizing | helper | `ttnn._ttnn.fabric.ccl_packet_dims` | pybind `ttnn/cpp/ttnn-nanobind/fabric.cpp:236-266`, impl `ccl_helpers_dataflow_host.hpp:74-96` | `(dtype, page_size_bytes, num_pages, alignment)` → `.packet_size_bytes`, `.pages_per_packet`, `.page_segments`, `.total_packets`. **Owns the `bfloat16` `bit_floor` rule — never reimplement.** |
| Routing | helper | `ttnn._ttnn.fabric.ccl_dm_route` | pybind `fabric.cpp:236-266`, impl `ccl_helpers_dataflow_host.hpp:109-166` | `(mesh_device, src_coord, dst_coord, topology)` → `.num_hops`, `.is_forward`, `.neighbor_id`. **Owns the fwd/bwd sign reversal and the Ring short-way choice — pass `.is_forward` straight through, never recompute it from the coordinate delta.** `.neighbor_id` is the *next hop*, which is what `setup_fabric_connection` wants. |
| Fabric connection args | helper | `ttnn.setup_fabric_connection` | `ttnn/cpp/ttnn-nanobind/fabric.cpp:141-178` | `(src_fabric_node_id, dst_fabric_node_id, link_idx, program_descriptor, worker_core, core_type=WORKER)` → `list[int]` (3 values for WORKER). **Side effect: appends two `SemaphoreDescriptor`s to `program_descriptor.semaphores`** (`tt_metal/fabric/fabric.cpp:181-203`), so it must be called *after* the `ttnn.ProgramDescriptor` object exists and be handed that same object. |
| Alignment | helper | `ttnn.get_l1_alignment()` | `ttnn/cpp/ttnn-nanobind/bfp_utils.cpp:127` | 16 B; used for the intra-packet stride and for the kernels' `alignment` CT arg |
| Buffer geometry | helper | `Tensor.buffer_page_size()`, `Tensor.buffer_num_pages()` | `ttnn/cpp/ttnn-nanobind/pytensor.cpp:1399,1417` | logical page bytes and page count |
| Accessor CT args | helper | `ttnn.TensorAccessorArgs(tensor).get_compile_time_args()` | `tt_metal/impl/buffers/tensor_accessor_args.cpp:46-52` | CT arg index 1 of the block is `buffer.aligned_page_size()` — the correct stride |
| Cross-device semaphore | helper | `ttnn.create_global_semaphore` / `ttnn.get_global_semaphore_address` / `ttnn.synchronize_device` | `ttnn/cpp/ttnn-nanobind/global_semaphore.cpp:40-56,58-67`; `ttnn/cpp/ttnn-nanobind/device.cpp:548-559` | created ONCE per mesh device, one `synchronize_device` immediately after |
| Descriptor / dispatch | helper | `ttnn.MeshProgramDescriptor` (+ `.semaphores`), `ttnn.ProgramDescriptor`, `ttnn.KernelDescriptor`, `ttnn.CBDescriptor`, `ttnn.CBFormatDescriptor`, `ttnn.RuntimeArgs`, `ttnn.generic_op` | `ttnn/cpp/ttnn-nanobind/program_descriptors.cpp:1077-1087` (`semaphores`), `:930-961`, `:694-907`, `:398-515`, `:325-396`, `:167-242`; `ttnn/cpp/ttnn/operations/generic/generic_op_nanobind.cpp:23-60` | `mpd[MeshCoordinateRange(c, c)] = program` is **append-only** (`:1039-1048`) — insert each coordinate exactly once |

### Host-side raw / hand-written (with justification)

| Phase | What | Why not a helper |
|-------|------|------------------|
| Fabric RT-arg block layout | A local `_append_fabric_rt_args(rt_ref, src_id, neighbor_id, program, core, is_forward)` that writes `[has_forward][fwd args if fwd][has_backward][bwd args if bwd]` | **Helpers considered and rejected:** `ttnn::ccl::dataflow::append_ccl_fabric_rt_args` (`ccl_helpers_dataflow_host.hpp:219-237`) implements exactly this, but it is a **C++-only** host function with no pybind binding (`grep -rn "append_ccl_fabric_rt_args" ttnn/cpp/ttnn-nanobind/` → no match; `fabric.cpp:236-266` binds only `ccl_packet_dims` and `ccl_dm_route`). The Python op must therefore mirror the six-line layout, exactly as the proven `ttnn/ttnn/operations/all_gather/all_gather_program_descriptor.py:52-66` does. |
| Output seeding | `ttnn.clone(input_tensor)` | Not a kernel-library concern; it is a host-level tensor op. See *Output contract*. |

## Compute Phases

This op has **no compute kernel** — the table below is the dataflow phase sequence. Phases 1–3 and 4–7
run concurrently on their respective devices; the ordering constraints across devices are exactly the
semaphore edges from the *Tensix-to-Tensix contract* table.

| # | Device | RISC | Operation | Helper? | Consumes | Produces | CB state after |
|---|--------|------|-----------|---------|----------|----------|----------------|
| 0 | receiver | NCRISC | Fabric-ack the sender ("ready") | ✅ `FabricStreamSender::signal` | — | remote sender semaphore += 1 | — |
| 1 | sender | BRISC | Wait for "ready", then `noc_semaphore_set(sem, 0)` | raw (op-owned per `:104-113`) | local semaphore | re-armed semaphore | — |
| 2 | sender | NCRISC | Stream `num_pages` input pages DRAM → `cb_shard_pages` | raw (`TensorAccessor` + `noc_async_read`, 2-arg ctor, transfer `page_size`) | input DRAM | `cb_shard_pages` × `num_pages` | `cb_shard_pages` drained in FIFO by phase 3 |
| 3 | sender | BRISC | Open stream, arm write + inc, frame pages into packets, issue `total_packets` fabric writes | ✅ `open` / `arm_unicast_write` / `arm_inc` / `write_page`; raw `tt_memmove` for framing | `cb_shard_pages` × `num_pages` | receiver's intermediate DRAM × `total_packets` | `cb_shard_pages` empty; `cb_packet_staging` holds the last packet (dead) |
| 4 | sender | BRISC | `inc` the receiver's semaphore ("done"), then `stream.close()` | ✅ `AtomicIncChannel::inc`, `FabricStream::close` | — | remote receiver semaphore += 1 | — |
| 5 | receiver | NCRISC | Wait for "done" | raw (op-owned) | local semaphore | — | — |
| 6 | receiver | NCRISC | Read back each packet locally, de-frame into `cb_output_pages` | raw (`TensorAccessor` + `noc_async_read` + `tt_memmove`) | intermediate DRAM × `total_packets` | `cb_output_pages` × `num_pages` | `cb_packet_landing` holds the last packet (dead) |
| 7 | receiver | BRISC | Drain `cb_output_pages` → output DRAM | raw (`TensorAccessor` + `noc_async_write`, 2-arg ctor, transfer `page_size`) | `cb_output_pages` × `num_pages` | output DRAM shard | `cb_output_pages` empty |
| 8 | receiver | NCRISC | `noc_semaphore_set(sem, 0)` (re-arm for the next cache hit) | raw (op-owned per `:111-113`) | — | clean semaphore | — |

## Host ↔ Kernel value contract

The implementer chooses the argument indices; these are the *values* each kernel must receive, and the
two placement rules that are not derivable from the CB layout.

| Kernel | Compile-time values | Runtime values |
|--------|--------------------|----------------|
| `point_to_point_sender_reader.cpp` | `cb_shard_pages`, then `TensorAccessorArgs(input_tensor)` | `input_tensor.buffer_address()`, `num_pages`, `page_size` |
| `point_to_point_sender_writer.cpp` | `cb_shard_pages`, `cb_packet_staging`, `l1_alignment`, `page_segments`, then `TensorAccessorArgs(intermediate_tensor)` | `intermediate_tensor.buffer_address()`, `num_pages`, `total_packets`, `page_size`, `packet_size`, `pages_per_packet`, `sem_addr`, `dst_num_hops`, **then the fabric connection block** |
| `point_to_point_receiver_reader.cpp` | `cb_packet_landing`, `cb_output_pages`, `l1_alignment`, `page_segments`, then `TensorAccessorArgs(intermediate_tensor)` | `intermediate_tensor.buffer_address()`, `num_pages`, `total_packets`, `page_size`, `packet_size`, `pages_per_packet`, `sem_addr`, `sender_num_hops`, **then the fabric connection block** |
| `point_to_point_receiver_writer.cpp` | `cb_output_pages`, then `TensorAccessorArgs(output_tensor)` | `output_tensor.buffer_address()`, `num_pages`, `page_size` |

**Rule 1 — accessor CT block last.** Each kernel binds exactly one `TensorAccessor`, so all scalar CT
args come first and `TensorAccessorArgs<N>()` (with `N` = number of scalar CT args) closes the block.

**Rule 2 — fabric block last, index recorded.** The fabric connection block is appended to the *end* of
the fabric-owning kernel's runtime args on its single core. The kernel records that start position as
`size_t conn_arg_idx` and peeks `get_arg_val<uint32_t>(conn_arg_idx)` as `is_forward` (the leading
`has_forward` flag doubles as the direction bit for a unidirectional sender), then hands
`conn_arg_idx` **by reference** to `FabricStreamSender<>` which consumes the whole block. The fabric
owner is the **writer (index 1)** on the sender program and the **reader (index 0)** on the receiver
program.

### Route computation on the host

| Program | Route call | Used for |
|---------|-----------|----------|
| send | `ccl_dm_route(mesh_device, sender_coord, receiver_coord, topology)` | `dst_num_hops = .num_hops`; `is_forward = .is_forward`; `setup_fabric_connection(fabric_id(sender_coord), .neighbor_id, 0, send_program, CoreCoord(0,0))` |
| receive | `ccl_dm_route(mesh_device, receiver_coord, sender_coord, topology)` | `sender_num_hops = .num_hops`; `is_forward = .is_forward`; `setup_fabric_connection(fabric_id(receiver_coord), .neighbor_id, 0, recv_program, CoreCoord(0,0))` |

## Semaphore lifecycle (host)

```python
_SEMAPHORE_CACHE: dict = {}          # module level, keyed by id(mesh_device)

def _get_or_create_semaphore(mesh_device):
    sem = _SEMAPHORE_CACHE.get(id(mesh_device))
    if sem is None:
        grid  = mesh_device.compute_with_storage_grid_size()
        cores = ttnn.num_cores_to_corerangeset(grid.x * grid.y, grid, row_wise=True)
        sem   = ttnn.create_global_semaphore(mesh_device, cores, 0)
        ttnn.synchronize_device(mesh_device)     # ONCE, right after creation
        _SEMAPHORE_CACHE[id(mesh_device)] = sem
    return sem
```

Then, per call:

```python
sem      = _get_or_create_semaphore(mesh_device)
sem_addr = ttnn.get_global_semaphore_address(sem)
mpd      = create_mesh_program_descriptor(..., sem_addr=sem_addr)
mpd.semaphores = [sem]                    # framework keeps its L1 alive across cache hits
ttnn.generic_op([input_tensor, intermediate_tensor, output_tensor], mpd)
return output_tensor                      # NO post-dispatch synchronize_device
```

- Created **once** per mesh device, never per call. `ttnn.synchronize_device` runs exactly once, right
  after creation — never as a per-call barrier.
- `MeshProgramDescriptor.semaphores` (`program_descriptors.cpp:1077-1087`) is excluded from the
  program-cache hash, so parking the semaphore there does not defeat caching while it does keep the
  GlobalSemaphore's L1 allocation alive for the cached workload's lifetime.
- The absolute address is baked into both kernels' runtime args and is stable across cache hits.

## Validation (`validate()`)

Structural errors raise `ValueError`; axis refusals raise the registry-model
`UnsupportedAxisValue` / `ExcludedCell` from `ttnn.operations._op_contract`.

| # | Check | Exception |
|---|-------|-----------|
| 1 | `isinstance(input_tensor.device(), ttnn.MeshDevice)` | `ValueError` — "input_tensor must be on a MeshDevice" |
| 2 | `tuple(sender_coord) != tuple(receiver_coord)` | `ValueError` — "cannot send to self" |
| 3 | every component of both coords is within `mesh_device.shape` | `ValueError` — "coordinate outside the mesh" |
| 4 | `not input_tensor.is_sharded()` | `ValueError` — "sharded input not yet supported (interleaved only)" |
| 5 | `len(input_tensor.shape) >= 2` | `ValueError` |
| 6 | `input_tensor.buffer_page_size() % ttnn.get_l1_alignment() == 0` | `ValueError` — "page size must be 16-byte aligned" |
| 7 | `output_tensor` (if given): shape, dtype, layout and `memory_config()` all equal the input's | `ValueError` |
| 8 | `intermediate_tensor` (if given): shape, dtype, layout, memory_config equal the resolved staging spec | `ValueError` |
| 9 | axis gate: `dtype`, `layout`, `topology`, `alignment` ∈ `SUPPORTED`; cell ∉ `EXCLUSIONS` | `UnsupportedAxisValue` / `ExcludedCell` |

Checks 1–8 run **before** the axis gate so that structural misuse is reported as a `ValueError`, not
mistaken for an unsupported-feature refusal by the golden harness.

Check 3 must accept coordinates on the same mesh row *or* column; `ccl_dm_route` raises if the two
coordinates share neither (`ccl_helpers_dataflow_host.hpp:109-166`, `detail::fabric_1d_routing_vector`).
Surface that as a `ValueError` rather than letting the C++ throw escape.

## Key Risks and Gotchas

| # | Risk | Mitigation |
|---|------|------------|
| 1 | **TensorAccessor page-size override corrupts non-64-B-aligned row-major pages.** Blackhole DRAM alignment is 64 B (`blackhole/core_config.h:41`); passing the raw logical page size as the accessor's third argument sets the per-bank stride to the *unaligned* value and mis-addresses every page after the first bank row. This is the single highest-impact defect in this op class — it silently corrupts `(1,1,24,24)`, `(1,1,32,48)` bf16/uint16, `(1,1,56,88)`, and every other row-major shard whose row bytes are not a multiple of 64, while leaving all TILE shards green. | **Construct every `TensorAccessor` with two arguments only.** Transfer sizes (`page_size`, `packet_size`) are separate runtime args used purely as `noc_async_read`/`noc_async_write` byte counts. |
| 2 | **Cache-reuse semaphore footgun** (`ccl_helpers_dataflow.hpp:104-113`). Missing or misplaced resets give "first run green, second hangs". | Sender resets **before** its outgoing inc; receiver resets **after** its wait. Both are mandatory and both appear in the phase table (phases 1 and 8). |
| 3 | **Ring topology on a `FABRIC_1D` mesh can route through a wraparound link that does not exist.** `ccl_dm_route` picks the short way when `|line_hops ± mesh_dim| < |line_hops|` (`ccl_helpers_dataflow_host.hpp:109-166`) and uses `BoundaryMode::WRAP` for it. On a `(1,4)` mesh, `(0,0)→(0,3)` under `Topology.Ring` resolves to a 1-hop wrap, which `FABRIC_1D` (non-ring) cannot deliver → fabric hang. | The op is correct for both topologies; the *test* topology is the constraint. Acceptance and golden tests exercise `Ring` only with coordinate pairs whose short way is the line way (hop distance ≤ ⌊N/2⌋ on an N-device line, i.e. ≤ 2 on the `(1,4)` verification mesh). A true wraparound requires `FabricConfig.FABRIC_1D_RING`. |
| 4 | **`is_forward` sign is reversed relative to the coordinate delta.** `ccl_dm_route` returns `!line_is_forward` because "fabrics' forward/backward concept is reversed" (`ccl_helpers_dataflow_host.hpp:109-166`). | Pass `route.is_forward` straight into the `has_forward`/`has_backward` block and into the `FabricStreamSender` constructor. Never derive it from `receiver_coord[d] - sender_coord[d]`. |
| 5 | **`setup_fabric_connection` mutates the `ProgramDescriptor`.** It appends two `SemaphoreDescriptor`s (`tt_metal/fabric/fabric.cpp:181-203`). Calling it before the descriptor exists, or on a different object, loses them. | Build the `ttnn.ProgramDescriptor` first, then splice the fabric args into `program.kernels[i].runtime_args[core.x][core.y]` (which returns a mutable `VectorUInt32` reference), passing that same `program` object. Mirror `all_gather_program_descriptor.py:52-66,228-236`. |
| 6 | **`neighbor_id`, not the destination coordinate.** For a multi-hop transfer the fabric connection is to the *adjacent* chip. | Use `route.neighbor_id` as `dst_fabric_node_id`. `all_gather` gets away with `get_fabric_node_id(coord±1)` only because it is always 1 hop; this op is not. |
| 7 | **Both endpoints must agree on the intermediate's page geometry.** The sender writes page `k` using its own local staging address; the receiver reads page `k` from its own. | Both build `TensorAccessor` from `TensorAccessorArgs(intermediate_tensor)` (identical CT args, since it is one mesh tensor) with the default page size. Symmetric mesh-buffer addressing does the rest. |
| 8 | **`MeshProgramDescriptor.__setitem__` is append-only** (`program_descriptors.cpp:1039-1048`) — it does not replace an existing key. | Insert `sender_coord` and `receiver_coord` exactly once each. `MeshCoordinateRange(c, c)` is the single-coordinate range. |
| 9 | **Format agnosticism.** No CB in this op is ever consumed by a compute thread; all four carry opaque bytes. Declaring a `bfloat8_b` CB format would impose tile-shaped page constraints for no benefit. | All CBs declare `data_format = ttnn.uint32`. The payload dtype only ever reaches `ccl_packet_dims` (which needs it for the `bfloat16` `bit_floor` rule). |
| 10 | **Tail packet over-send.** Regime A's last packet always carries a full `packet_size` payload even when it holds fewer than `pages_per_pkt` live pages. | The intermediate is sized `total_packets` pages of `packet_size`, so the over-send lands inside the staging buffer and is never read back. Do *not* try to shrink the last write — the armed payload size is a stream invariant (`ccl_helpers_dataflow.hpp:486`). |
| 11 | **Regime B (`page_segments > 1`) is unexercised by the current `INPUTS`** (every listed shard has a page ≤ 4096 B, and the fabric channel buffer is larger). It is still required by the specification. | Implement and keep both branches. A row-major shard with a last dim of, e.g., 4096 float32 elements (16 KB page) reaches it. |
| 12 | **Sender staging buffer reuse.** The framing loop overwrites `cb_packet_staging` immediately after `write_page`. | Safe: `write_page` → `fabric_unicast_noc_unicast_write_with_state` (`ccl_helpers_dataflow.inl:59-63`) copies the payload into the fabric channel buffer under flow control before returning. No extra barrier is needed, and none is present in the reference kernel (`writer_send.cpp:76-83`). |
| 13 | **Do not add a post-dispatch `synchronize_device`.** | The parked `mpd.semaphores` is what keeps the GlobalSemaphore alive across cache hits; a per-call barrier is both unnecessary and a throughput regression. `synchronize_device` appears exactly once, at semaphore creation. |
| 14 | **Fabric header pool budget.** The sender arms two channels (write + inc) and the receiver's `signal()` arms one. | Well inside the per-RISC pool of 8 (`ccl_helpers_dataflow.hpp:113-118`). No `reserved_packet_header_cb` is needed — the pool owns header storage. |

## Hardware Constraints checklist

- [x] CB sync: push count == wait count for every CB (see the ledger; the two packet CBs are reserve-once scratch with 0 pushes and 0 waits)
- [x] No reduce, no scaler CB — not applicable
- [x] No DEST usage — no compute kernel
- [x] No sequential-helper intermediates — no compute kernel
- [x] Page sizes are 16-B aligned: `aligned_page = round_up(page_size, 16)`; `packet_size` is a multiple of 16 by construction
- [x] Row-major CBs count pages in sticks; tile CBs count in tiles — here every CB counts opaque pages of a fixed byte size
- [x] All `cb_wait_front` calls on a given CB use the same page count (`1`)
- [x] `compute_kernel_hw_startup()` — not applicable, no compute kernel
- [x] Every `TensorAccessor` uses the 2-argument constructor (no page-size override)
