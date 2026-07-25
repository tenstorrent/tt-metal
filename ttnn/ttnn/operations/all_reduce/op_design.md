# Operation Design: all_reduce

## Overview

| Field | Value |
|-------|-------|
| Classification | **CCL + compute** (collective communication WITH a local arithmetic reduction) |
| Goal | Sum each device's shard element-wise across all N devices of a 1-D MeshDevice line and leave the identical sum on every device. |
| Math | `output[d][i] = Σ_{k=0..N-1} input[k][i]` for every device `d` and every element index `i`. Reduction op is **SUM**. Output shape == a single input shard's shape; output is bit-identical across devices. |
| Mode | **From scratch.** A self-contained Python op on `ttnn.generic_op` + `ttnn.MeshProgramDescriptor` with newly authored reader / compute / writer kernels. It does NOT import, wrap, re-export or dispatch to any existing `all_reduce` / `reduce_scatter` / `all_gather`. |
| Algorithm | **Broadcast-all then local N-way sum.** Every device chip-level-multicasts its own shard to every peer on the line (both fabric directions from one worker), landing in slot `sender_id` of an op-internal gathered buffer. Each device then locally sums its own shard (read from the input tensor) plus the N-1 received slots. |
| References (read as correctness references ONLY) | `ttnn/cpp/ttnn/kernel_lib/ccl_helpers_dataflow.hpp` (+ `.inl`) — the fabric dataflow helper; `ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/kernels/dataflow/worker_writer.cpp:32-35,88-99,127-135` — the only shipped duplex-multicast tier user; `.../all_reduce_async/device/kernels/compute/reduction.cpp:24-67` and `.../reduce_scatter_minimal_async/device/kernels/ring_reduction.cpp:91-118` and `.../llama_reduce_scatter/device/kernels/compute/reduction.cpp:19-62` — the shipped N-tile-sum compute idiom; `ttnn/ttnn/operations/all_gather/` and `ttnn/ttnn/operations/point_to_point/` — the Python `generic_op` + `MeshProgramDescriptor` scaffolding precedent |

### Why the algorithm choice

| Candidate | Verdict |
|---|---|
| Broadcast-all (chip-level multicast) + local N-way sum | **Chosen.** One `open()` per worker, N-1 peers covered by exactly two multicast routes, O(1) fabric phases, and the reduction is a single streaming compute pass. The helper header itself designates the multicast duplex route pair as "all_reduce's shape" (`ccl_helpers_dataflow.hpp:875`). |
| Ring reduce-scatter + all-gather | Rejected for Phase 0: `2(N-1)` sequenced fabric phases with per-step slice-walk and store-and-forward relaying, i.e. 2(N-1) cross-device sync points instead of 1. Bandwidth-optimal but a much larger correctness surface. Noted as a refinement. |
| Sequential line reduce to centre + broadcast back | Rejected: two dependent multi-hop phases and per-device asymmetric roles (leaf / interior / root), with no benefit over multicast on N ≤ 8. |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | on a `ttnn.MeshDevice` line view `(1, N)`, `N >= 2`; TILE_LAYOUT; interleaved; rank >= 2; each device holds a shard of the SAME shape | — | derives every CT/RT arg below |
| `topology` | `ttnn.Topology` | no | `Linear` (Phase 0). `Ring` is a refinement candidate — see Risks. | `ttnn.Topology.Linear` | host-only (selects the target counts) |
| `output_tensor` | `ttnn.Tensor \| None` | no | must equal the input shard spec exactly (shape, dtype, layout, buffer_type) | `None` → op allocates | `buffer_address()` → writer RT |

Exact public signature (import path `from ttnn.operations.all_reduce import all_reduce`):

```python
def all_reduce(
    input_tensor: ttnn.Tensor,
    topology: ttnn.Topology = ttnn.Topology.Linear,
    output_tensor: ttnn.Tensor | None = None,
) -> ttnn.Tensor
```

`topology` / `output_tensor` are positional-or-keyword (no `*`), matching the mandate. The acceptance test passes both by keyword.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | Per-device shard, rank >= 2. Last two dims both multiples of 32 (`alignment == tile_aligned`, Phase 0). Every device's shard has the same shape. |
| Dtype | `bfloat16` (primary), `float32` |
| Layout | `TILE` (the reduction is a tile compute) |
| Memory | interleaved, DRAM or L1 |
| Page | one 32x32 tile — `page_size = input_tensor.buffer_page_size()` (2048 B bf16 / 4096 B fp32); `P = input_tensor.buffer_num_pages()` = tiles per shard |

### Output

| Property | Value |
|----------|-------|
| Shape | identical to the input shard shape |
| Dtype | `input_tensor.dtype` |
| Layout | `TILE` |
| Memory | `input_tensor.memory_config()` |
| Value | element-wise SUM of all N devices' input shards — identical on every device |

### Op-internal tensors

| Name | Spec | Purpose | Lifetime |
|---|---|---|---|
| `gathered_tensor` | shape `(N * shard_shape[0], *shard_shape[1:])`, same dtype / layout / `buffer_type` as input; `ttnn.TensorSpec(...)` → `ttnn.allocate_tensor_on_device(spec, mesh_device)` | Landing buffer for the fabric multicast. Slot `k` = pages `[k*P, (k+1)*P)` = device `k`'s shard. Slot `my_id` is written by nobody and read by nobody (the local shard is read straight from `input_tensor`). | allocated per call; passed to `generic_op` as operand 1 so the framework keeps it alive for the dispatch |

**Why scaling dim 0 by N yields exactly slot-`k`-at-page-`k*P`:** for a TILE interleaved tensor the page order is row-major over `(..., Ht, Wt)`. Multiplying the leading dim by N prepends `N` independent copies of the trailing block, so device `k`'s shard is the contiguous page run `[k*P, (k+1)*P)`. This holds for rank >= 3 (dim 0 is a batch dim) and for rank 2 when `H % 32 == 0` (dim 0 is the tile-height dim; tile-aligned so the tile-row count scales exactly). Both are inside Phase-0 `SUPPORTED`. See Risks for why `non_tile_aligned` rank-2 breaks this and is therefore gated out.

## Dataflow Strategy

### Stage overview

```
                       ┌──────────────── device j (worker core (0,0)) ───────────────┐
 input DRAM  ──read──► │ cb_broadcast_pages ─► WRITER: duplex MULTICAST fabric write │──► peers' gathered DRAM
   (shard j)           │                                    + fused sem inc (last)   │    (slot j) + their sem
                       │                                                             │
 peers' fabric  ──────►│ gathered DRAM (slots k != j)                                │
                       │            │                                               │
                       │   READER waits sem >= N-1, resets sem                       │
                       │            │                                               │
 input DRAM  ──read──► │   READER interleaves N tiles ─► cb_shard_tiles              │
   (shard j, slot j)   │            │                                               │
                       │   COMPUTE: pairwise add_tiles folded in DEST ─► cb_output_tiles
                       │            │                                               │
                       │   WRITER ──────────────────────────────────────────────────►│──► output DRAM
                       └─────────────────────────────────────────────────────────────┘
```

No tilize / untilize: input, gathered buffer, and output are all TILE layout, so a page IS a tile end to end. The compute kernel consumes tiles and produces tiles.

### Intra-Tensix (RISC-to-RISC) contract

| Producer | CB | Consumer | Format | Unit |
|---|---|---|---|---|
| Reader (NCRISC) phase 1 | `cb_broadcast_pages` | Writer (BRISC) phase 1 | tile | 1 page per push |
| Reader (NCRISC) phase 3 | `cb_shard_tiles` | Compute (TRISC) | tile | N pages per push (one block = the N devices' contribution to one output tile) |
| Compute (TRISC) | `cb_output_tiles` | Writer (BRISC) phase 2 | tile | 1 page per push |

Kernel phase ordering per core (no circular wait — see Deadlock analysis):

| RISC | Phase 1 | Phase 2 | Phase 3 |
|---|---|---|---|
| Reader (NCRISC) | feed `cb_broadcast_pages` with the P input pages | `noc_semaphore_wait_min(sem, N-1)` then `noc_semaphore_set(sem, 0)` | feed `cb_shard_tiles` with P blocks of N tiles |
| Compute (TRISC) | — | — | drain `cb_shard_tiles`, fold, push `cb_output_tiles` |
| Writer (BRISC) | open duplex stream, multicast P pages, `close()` | — | drain `cb_output_tiles` → output DRAM |

### Tensix-to-Tensix / chip-to-chip contract

| Aspect | Contract |
|---|---|
| Participants | One worker core, logical `(0, 0)`, on every device of the `(1, N)` line. Exactly one `ProgramDescriptor` per mesh coordinate. |
| Egress | One `FabricDuplexSender` per device drives BOTH fabric directions from that single core. Route pair = two **multicast** routes → `Cast::Multicast` stream (`ccl_helpers_dataflow.hpp:876-884`). Each issue fans out to every CONNECTED direction; an end-of-line worker has exactly one (`DuplexConn::has(dir)`, `:668-670`). |
| What is multicast | Device `i` multicasts its own shard, page by page, to `dst_noc_addr = gathered.get_noc_addr(i*P + p)`. Because the gathered tensor is ONE mesh allocation it sits at the identical address on every device, so the same noc0-encoded address resolves to the correct DRAM bank on every receiving chip. This is the all_gather mechanism (`ttnn/ttnn/operations/all_gather/kernels/all_gather_writer.cpp:83`). |
| Multicast coverage | Forward route `{start_distance_in_hops = 1, range_hops = n_fwd}`, backward `{1, n_bwd}`. On Linear, `n_fwd + n_bwd == N - 1`, so the two multicasts cover every peer **exactly once** (`ttnn/cpp/ttnn/operations/ccl/ccl_common.cpp:1778-1797`). `range_hops = k` means **k chips**, and `start_distance_in_hops = 1` means "starting at the immediate neighbour" (`tt_metal/fabric/fabric_edm_packet_header.hpp:130-133`; derivation in `tt_metal/hostdevcommon/api/hostdevcommon/fabric_common.h:282-336`). |
| Per-hop delivery | Every chip in the range delivers: `encode_1d_multicast` sets `WRITE_AND_FORWARD` on the interior hops and `WRITE_ONLY` on the last (`fabric_common.h:291-336`), and the router branches accordingly (`tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp:869-896`, `WRITE_AND_FORWARD` at `:884-887`). |
| Sync primitive | ONE op-internal cross-device `GlobalSemaphore`, used as a pure **receive counter**. |
| Sync protocol | Device `i` sends P packets per connected direction. The LAST packet of each direction is a **fused write+atomic-inc** carrying `val = 1` and `flush = true`; the preceding `P-1` packets are plain multicast writes. Because the fused packet is multicast, EVERY chip in that direction's range performs both the payload write and the increment in one delivery (`tt_metal/fabric/hw/inc/edm_fabric/fabric_edm_packet_transmission.hpp:190-211` — `NOC_FUSED_UNICAST_ATOMIC_INC` does the write then the inc, and the chip-level routing field is orthogonal to `noc_send_type`). Therefore each device receives exactly `N-1` increments, one per peer. |
| Ordering guarantee | Within one direction the connection is in-order, so a peer's fused inc is issued after its `P-1` payload writes. `flush = true` makes the receiving fabric endpoint flush its NoC write pipeline before performing the inc (`fabric_edm_packet_transmission.hpp:203-205`), which covers the earlier packets on the same receive channel. Consequence: when `sem >= N-1`, every peer's full shard has landed in DRAM. |
| Waiting half | Op-owned, as the helper mandates (`ccl_helpers_dataflow.hpp:104-112`): `noc_semaphore_wait_min(sem_ptr, N-1)` in the reader. |
| Cache-reuse re-arm | Op-owned. This semaphore is a pure receive counter, so the **receiver resets after its wait** — `noc_semaphore_set(sem_ptr, 0)` immediately after `wait_min` in reader phase 2 (`ccl_helpers_dataflow.hpp:109-112`; same placement as `ttnn/ttnn/operations/all_gather/kernels/all_gather_reader.cpp:101`). The sender never touches its own semaphore, so no sender-side reset applies. |
| Receive ingress | Op-owned local NoC reads (`noc_async_read` through a `TensorAccessor` over `gathered_tensor`). There is no `FabricStreamReceiver` (`ccl_helpers_dataflow.hpp:104-112`). |

### Host-side route + framing contract

| Item | Source |
|---|---|
| Neighbour route + fabric direction | `ttnn._ttnn.fabric.ccl_dm_route(mesh_device, coord_i, coord_neighbour, topology)` → `.num_hops`, `.is_forward`, `.neighbor_id` (`ttnn/cpp/ttnn-nanobind/fabric.cpp:253-266`). **`is_forward` owns the fabric fwd/bwd sign reversal** — the op must NOT assume `i+1` is the fabric-forward neighbour. |
| Direction slotting | Query both neighbours (`i+1` if it exists, `i-1` if it exists). The one reporting `is_forward == True` fills the FORWARD slot, the other the BACKWARD slot. Host asserts the two do not report the same direction. |
| Multicast range per slot | Linear: chips beyond the `i+1` neighbour = `N - 1 - i`; chips beyond the `i-1` neighbour = `i` (`ccl_common.cpp:242-247`, `:1778-1785`). The range assigned to a slot is the count for whichever neighbour landed in that slot. |
| Packet framing sanity | `ttnn._ttnn.fabric.ccl_packet_dims(dtype, page_size, num_pages, l1_alignment)` (`ttnn/cpp/ttnn-nanobind/fabric.cpp:245-252`, impl `ttnn/cpp/ttnn/operations/ccl/common/host/ccl_helpers_dataflow_host.hpp:74-96`). Used as a **gate only**: `validate()` requires `page_segments == 1` (one tile page fits in one fabric packet). It holds for every supported dtype — bf16 2048 B against `bit_floor(4400) = 4096`, fp32 4096 B against 4400. The op deliberately sends ONE page per packet and does not coalesce, so `pages_per_packet` is not consumed. |
| Fabric connection RT args | `ttnn.setup_fabric_connection(src_fabric_node_id, dst_fabric_node_id, link_idx=0, program_descriptor, worker_core)` (`ttnn/cpp/ttnn-nanobind/fabric.cpp:141-178`) — returns the arg vector AND mutates the `ProgramDescriptor` by appending `SemaphoreDescriptor`s, so it must be called on an already-constructed program with the RT args appended through a live reference `program.kernels[writer_idx].runtime_args[core.x][core.y]`. |

**Duplex RT-arg block (no Python precedent — this is the contract).** `DuplexConn` consumes the block documented at `ccl_helpers_dataflow_host.hpp:213-237`:

```
[has_forward][<forward connection args> if has_forward][has_backward][<backward connection args> if has_backward]
```

Unlike the unidirectional mirrors in `all_gather_program_descriptor.py:52-66` / `point_to_point_program_descriptor.py:64-78`, **both flags may be 1** and both blocks present. Interior devices emit `[1][fwd][1][bwd]`; device 0 and device N-1 emit exactly one populated slot. The kernel records the block's start index as `conn_arg_idx` and hands it to `FabricDuplexSender` by reference, which advances it past the whole block (`ccl_helpers_dataflow.hpp:853,662-664`).

**Multicast route CT-arg block (12 uint32, pure-multicast layout).** Appended to the WRITER's compile-time args; do NOT use `append_ccl_line_route_ct_args`, which forces an unwanted unicast interleave.

| Offset | Value |
|---|---|
| `+0` | forward `start_distance_in_hops` = 1 (0 if no forward slot) |
| `+1` | forward `range_hops` = `n_fwd` (0 if no forward slot) |
| `+2..+5` | forward `e/w/n/s_num_hops` = 0 (2-D-only fields, ignored on the 1-D LowLatency path) |
| `+6` | backward `start_distance_in_hops` = 1 (0 if no backward slot) |
| `+7` | backward `range_hops` = `n_bwd` (0 if no backward slot) |
| `+8..+11` | backward `e/w/n/s_num_hops` = 0 |

Read in the kernel as `ccl_routing_utils::get_line_multicast_route_info_from_args<idx>()` and `<idx + ccl_routing_utils::num_line_multicast_args>` (`ttnn/cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp:46-56`, `num_line_multicast_args = 6` at `:39`), exactly as `all_reduce_async/.../worker_writer.cpp:32-35` does. The two designated initializers `.dst_mesh_id` / `.dst_chip_id` are the first members of the two anonymous unions, i.e. `start_distance_in_hops` / `range_hops` on the 1-D path (`worker_routing_utils.hpp:23-36`). An absent direction's six words are zeros; they are never programmed because `arm_*` allocates a header only for connected directions (`ccl_helpers_dataflow.inl:314-321`) and every issue gates on `has(d)` (`.inl:415-418`).

### GlobalSemaphore ownership

| Rule | Implementation |
|---|---|
| Created ONCE per `mesh_device` | module-level `_SEMAPHORE_CACHE: dict` keyed on `id(mesh_device)`, mirroring `ttnn/ttnn/operations/all_gather/all_gather.py:83-98` verbatim |
| Over the worker grid | `ttnn.create_global_semaphore(mesh_device, ttnn.num_cores_to_corerangeset(grid.x*grid.y, grid, row_wise=True), 0)` |
| Sync once at creation | `ttnn.synchronize_device(mesh_device)` inside the cache-miss branch only |
| Parked on the descriptor | `mesh_program_descriptor.semaphores = [sem]` — the framework copies it into the cached workload's `shared_variables`, keeping its L1 alive across cache hits |
| Address into kernels | `ttnn.get_global_semaphore_address(sem)` as a uint32 RT arg on reader and writer |
| No per-call barrier | Do NOT add a post-dispatch `ttnn.synchronize_device` to keep it alive — the framework owns its lifetime |

Per-device `ProgramDescriptor.semaphores` stays `[]`; that list is reserved for the `SemaphoreDescriptor`s `setup_fabric_connection` appends.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one output tile (one page). The reduce loop runs `P = input_tensor.buffer_num_pages()` iterations. |
| Grid | **one worker core per device**, logical `(0, 0)`; `core_set = ttnn.CoreRangeSet([ttnn.CoreRange(CoreCoord(0,0), CoreCoord(0,0))])`. Programs are emitted for every mesh coordinate `(0, i)`, `i in [0, N)`, each as `mesh_pd[ttnn.MeshCoordinateRange(coord_i, coord_i)] = program`. |
| Per-core work | the whole shard: broadcast P pages, then reduce P output tiles |
| Remainder | none — a single core takes all P pages, so `ttnn.split_work_to_cores()` is not used |
| Rationale | `FabricDuplexSender` is defined as ONE worker driving both directions of ONE `FabricConnectionManager` (`ccl_helpers_dataflow.hpp:614-620`). Sharding pages across multiple cores requires either one fabric link per core or worker-mux (`MuxConn<N>`), and `MuxConn` cannot back the duplex tier because it exposes `sender()` with no direction while the duplex channels call `conn_->has(d)` / `conn_->sender(d)` (`ccl_helpers_dataflow.hpp:282` vs `.inl:415-422`). Multi-core is a refinement, not Phase 0. |

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|--------|----------|----------|----------|
| `cb_broadcast_pages` | 0 | `page_size` (= tile size) | 2 | `input_tensor.dtype` | Reader phase 1 | Writer phase 1 | phase 1 only; empty afterwards |
| `cb_shard_tiles` | 1 | `page_size` | `2 * N` | `input_tensor.dtype` | Reader phase 3 | Compute | phase 3 only; empty afterwards |
| `cb_output_tiles` | 16 | `page_size` | 2 | `input_tensor.dtype` | Compute | Writer phase 2 | phase 3 only; empty afterwards |

`total_size = page_size * num_pages` for each. All three live on the single core set.

### Sizing rationale

| CB | Rationale |
|---|---|
| `cb_broadcast_pages` = 2 | Streaming double-buffer: the reader prefetches page `p+1` while the writer's fabric write of page `p` is in flight. |
| `cb_shard_tiles` = `2 * N` | The compute kernel needs all N contributions to one output tile addressable **simultaneously and contiguously** (`add_tiles(cb, cb, d, d+1, 0)` indexes the same CB at two offsets). One block is N pages; two blocks give the reader a prefetch slot. **The total MUST be an integer multiple of N and every push/pop MUST be exactly N pages** — that invariant is what guarantees the write pointer is always at page offset `0` or `N`, so a single `cb_reserve_back(cb_shard_tiles, N)` always yields N contiguous pages and `get_write_ptr() + k*page_size` never wraps. |
| `cb_output_tiles` = 2 | Streaming double-buffer: the compute kernel folds output tile `p+1` while the writer drains tile `p`. |

### Sync verification

| CB | Producer pushes | Consumer waits | Consumer pops | Balanced |
|---|---|---|---|---|
| `cb_broadcast_pages` | `P` x `cb_push_back(...,1)` | `P` x `cb_wait_front(...,1)` | `P` x `cb_pop_front(...,1)` | yes |
| `cb_shard_tiles` | `P` x `cb_push_back(...,N)` | `P` x `cb_wait_front(...,N)` | `P` x `cb_pop_front(...,N)` | yes — every wait uses the same count `N` |
| `cb_output_tiles` | `P` x `cb_push_back(...,1)` | `P` x `cb_wait_front(...,1)` | `P` x `cb_pop_front(...,1)` | yes |

### Deadlock analysis

| Potential wait | Resolved by |
|---|---|
| Reader phase 2 spins on the semaphore while the writer still needs `cb_broadcast_pages` | Reader phase 1 completes *before* phase 2 begins, so all P pages are already pushed. Reader and writer are different RISCs. |
| Writer phase 1 blocks on fabric flow control while peers' workers spin on their semaphores | Fabric routers are separate ERISC cores; a peer's payload lands in DRAM via its router, independent of its worker's state. |
| Writer phase 2 waits on `cb_output_tiles` that depends on the barrier | Writer phase 2 is strictly after phase 1, which depends only on reader phase 1. No cycle: reader1 → writer1 → (fabric) → reader2 → reader3 → compute → writer2. |
| `cb_shard_tiles` reserve of N pages blocks forever | Capacity is `2N >= N`; the compute kernel pops a full block before the reader needs a third. |

## API Mapping

Every mechanism — helper or raw — with a verified file:line reference. Helper paths are relative to the repo root.

### Fabric dataflow (writer kernel) — helper-owned

| Phase | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Requirements |
|-------|------|----------|-----------|----------------------|----------|-----------|--------------|
| broadcast: connection | helper | `FabricDuplexSender<>::FabricDuplexSender(size_t& conn_arg_idx, uint32_t alignment)` | `ttnn/cpp/ttnn/kernel_lib/ccl_helpers_dataflow.hpp:856-857` | `ConnT = DuplexConn` (default), `alignment = 1` | — | — | No `is_forward` — a duplex sender uses whichever directions the host wired (`:660-661`). Advances `conn_arg_idx` past the whole block. **Declare the sender before the stream — the stream borrows its connection** (`:781-784`). |
| broadcast: open | helper | `FabricDuplexSender::open(const line_multicast_route_info_t& fwd, const line_multicast_route_info_t& bwd)` | `:876-884` | returns `FabricDuplexStream<Cast::Multicast, DuplexConn>` | — | — | The MULTICAST route pair selects `Cast::Multicast` at compile time (`:621-629`); routes are bound ONCE here, never per-`arm_*`. |
| broadcast: arm payload | helper | `FabricDuplexStream::arm_write(uint32_t page_size_bytes)` | `:811` (impl `ccl_helpers_dataflow.inl:311-338`) | `page_size_bytes = page_size` | — | — | Programs each CONNECTED direction's route + invariant payload size via `set_state`; allocates a header only for connected directions (`.inl:314-321`). |
| broadcast: arm fused | helper | `FabricDuplexStream::arm_fused_write_inc(uint32_t page_size_bytes, uint32_t val = 1, bool flush = false)` | `:813-814` (impl `.inl:341-368`) | `page_size, val = 1, flush = true` | — | — | `flush = true` makes the receiving endpoint flush its NoC write pipeline before the inc (`tt_metal/fabric/hw/inc/edm_fabric/fabric_edm_packet_transmission.hpp:203-205`). This is the payload→semaphore ordering guarantee the design depends on. |
| broadcast: issue pages `0..P-2` | helper | `DuplexWriteChannel::write(uint64_t dst_noc_addr, uint32_t src_l1_addr)` | `:696` (impl `.inl:440-442`) | armed-size overload | `cb_broadcast_pages` | — | Fans out to every connected direction, gating on `has(d)` (`.inl:415-418`). Armed size suffices — every packet is exactly one tile page, so the explicit-size overload (`:698`, documented at `:686-690`) is unnecessary. |
| broadcast: issue page `P-1` | helper | `DuplexFusedWriteIncChannel::write_fused(uint64_t dst_noc_addr, uint32_t src_l1_addr, uint64_t remote_sem_noc_addr)` | `:733` (impl `.inl:486-489`) | armed-size overload | `cb_broadcast_pages` | — | One packet carries the payload AND bumps the peer semaphore (`:723-728`). The receiver-side wait stays op-owned (`:399-400`). |
| broadcast: teardown | helper | `FabricDuplexStream::close()` | `:821` (impl `.inl:544-552`) | — | — | — | Drains (write + atomic barriers) then closes; idempotent, destructor is the backstop (`:807`). |

**`*_with_local_copy` forms considered and rejected.** `DuplexWriteChannel::write_with_local_copy` (`:704-705`) and `DuplexFusedWriteIncChannel::write_fused_with_local_copy` (`:741-744`) would additionally mirror the payload into the local chip's slot `my_id`. Rejected for a concrete reason: the local shard is already available in the input tensor at the identical page indices, so mirroring is pure duplicated DRAM traffic, and it would require a **new writer→reader ordering handshake** — the fused mirror explicitly does NOT flush local writes (`:737-740`: "Unlike the write-only mirror this does NOT flush local writes"), so the reader would need a second local semaphore (or a local `noc_semaphore_inc` after an explicit `noc_async_write_barrier`) before it could trust slot `my_id`, and the receive counter's target would have to change from `N-1` to `N`. The chosen design reads slot `my_id` from `input_tensor` instead, which removes that handshake entirely. The unidirectional tier (`FabricStreamSender`, `:566-607`) is likewise rejected: it drives ONE direction per sender, so covering both sides of the line would need two senders / two cores and re-derive the per-send direction gating this tier exists to remove (`:614-620`).

### Compute (N-way element-wise tile sum) — raw API, helper non-use justified

**Helpers considered and rejected.**

1. **`compute_kernel_lib::reduce` — `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp:378-392`.** Cannot express this reduction. `ReduceDim` offers only `REDUCE_ROW`, `REDUCE_COL`, `REDUCE_SCALAR`, all of which reduce *within* a tile's 32x32 grid — the documented output sizes are `rows x batches`, `cols x batches`, and `batches` tiles respectively (`reduce_helpers_compute.hpp:129-133`, restated at `:266-268` and `:290`). all_reduce needs an element-wise sum **across a stack of N tiles that preserves the full 32x32 shape**, which is not one of the three dims and produces `1` tile from `N` tiles with no dimensional collapse. `Accumulate` (`:184-198`) accumulates *across `reduce()` calls* but each call still performs an intra-tile dimensional reduction, so composing it cannot yield an element-wise result either.
2. **`eltwise_convenience.hpp` / `eltwise_chain.hpp` (`add`, `BinaryFpu`, `AddBinary`).** These headers **do not exist on this branch.** `ttnn/cpp/ttnn/kernel_lib/` contains only `ccl_helpers_dataflow.{hpp,inl}`, `dest_helpers.hpp`, `dfb_helpers_compute.{hpp,inl}`, `dfb_helpers_dataflow.{hpp,inl}`, `l1_helpers.hpp`, `reduce_helpers_{common,compute,dataflow}`, `tilize_helpers.{hpp,inl}`, `untilize_helpers.{hpp,inl}`. `git ls-files ttnn/cpp/ttnn/kernel_lib/` confirms no eltwise / binary / matmul_block / bias_add / sfpu_activation / reblock_untilize header is tracked at HEAD. **An `#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"` will not compile.** The implementer must not add one.
3. **`tilize_helpers` / `untilize_helpers`.** Not applicable — input, gathered buffer, and output are all TILE layout, so there is no row-major boundary to cross.
4. **`compute_kernel_lib::DEST_AUTO_LIMIT` (`dest_helpers.hpp:89-103`) IS used** — as a `static_assert` guard that the design's DEST footprint (1 register) fits whatever the host configured.

**Chosen raw APIs.** This mirrors the shipped, silicon-verified idiom: pairwise `add_tiles` folded into a single DEST accumulator (`all_reduce_async/.../reduction.cpp:32-54`, `llama_reduce_scatter/.../reduction.cpp:26-50`, `ring_reduction.cpp:91-118`).

| Phase | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Requirements |
|-------|------|----------|-----------|----------------------|----------|-----------|--------------|
| boot (once) | raw_api | `compute_kernel_hw_startup(uint32_t icb0, uint32_t icb1, uint32_t ocb)` | `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:70-71` | `(cb_shard_tiles, cb_shard_tiles, cb_output_tiles)` | — | — | **Exactly once, at the top of the kernel, before any other compute API.** It performs MMIO writes and is unsafe mid-kernel (`:54-58`). Do NOT also call `binary_op_init_common`. |
| seed (odd N only) | raw_api | `copy_tile_init(uint32_t cbid)` / `copy_tile(uint32_t in_cb_id, uint32_t in_tile_index, uint32_t dst_tile_index)` | `tt_metal/hw/inc/api/compute/tile_move_copy.h:57` / `:103` | `copy_tile(cb_shard_tiles, 0, 0)` | `cb_shard_tiles` | DEST[0] | Short init — safe inside the DEST window (precedent `ring_reduction.cpp:93-97`). Requires `#include "api/compute/tile_move_copy.h"` (NOT pulled in transitively by `eltwise_binary.h`). |
| fold | raw_api | `add_tiles_init(uint32_t icb0, uint32_t icb1, bool acc_to_dest = false)` | `tt_metal/hw/inc/api/compute/eltwise_binary.h:128-129` | `(cb_shard_tiles, cb_shard_tiles, acc_to_dest)` | — | — | `acc_to_dest = true` ⇒ `DEST[idst] += A + B` (`:125`). Short init, re-issuable inside the DEST window (precedent `ring_reduction.cpp:97,99`). |
| fold | raw_api | `add_tiles(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)` | `tt_metal/hw/inc/api/compute/eltwise_binary.h:206-214` | `icb0 == icb1 == cb_shard_tiles`, `idst = 0` | `cb_shard_tiles` | DEST[0] | Both FPU operands may come from the same CB at different tile indices — this is the shipped pattern (`llama_reduce_scatter/.../reduction.cpp:48`, `all_reduce_async/.../reduction.cpp:46-51`). `add_tiles` pins `MathFidelity::LoFi` internally (`:212`), so `math_fidelity` in the ComputeConfig does not affect it; precision is governed by the CB formats and `fp32_dest_acc_en`. |
| DEST protocol | raw_api | `tile_regs_acquire()` / `tile_regs_commit()` / `tile_regs_wait()` / `tile_regs_release()` | `tt_metal/hw/inc/api/compute/reg_api.h:45` / `:82` / `:54` / `:87` | — | — | — | Standard MATH/PACK handshake. Do not use the deprecated `acquire_dst()` / `release_dst()` (`:31-32`, `:71-72`). |
| pack | raw_api | `pack_tile(uint32_t ifrom_dst, uint32_t icb, uint32_t output_tile_index = 0)` | `tt_metal/hw/inc/api/compute/pack.h:85-86` | `pack_tile(0, cb_output_tiles)` — default `out_of_order_output = false` | DEST[0] | `cb_output_tiles` | With `out_of_order_output = false` the `output_tile_index` argument is IGNORED and packing is sequential from 0 within the reserved region (`:59-67`). We pack exactly one tile per reserve, so the default is correct — do NOT pass an index. |
| DEST guard | helper | `compute_kernel_lib::DEST_AUTO_LIMIT` | `ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp:103` (computed at `:89-100`) | — | — | — | `static_assert(1 <= compute_kernel_lib::DEST_AUTO_LIMIT)`. Auto-detects 8 (bf16, half-sync) / 4 (fp32, half-sync) from the JIT-generated `DST_ACCUM_MODE` / `DST_SYNC_MODE` (`:22-26`), so the kernel can never desync from the host's `fp32_dest_acc_en`. |

Required includes for the compute kernel (the only correct paths on this branch — there is no `tt_metal/include/compute_kernel_api/` and no `tt_metal/api/compute/`):

```cpp
#include <cstdint>
#include "api/compute/eltwise_binary.h"                  // pulls in common.h -> reg_api, pack, cb_api, hw_startup
#include "api/compute/tile_move_copy.h"                  // copy_tile_init / copy_tile
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"     // DEST_AUTO_LIMIT
```

### Dataflow (reader / writer) — raw API, explicitly op-owned

`ccl_helpers_dataflow.hpp:104-112` and `:130-140` state that the receive ingress, the waiting half of a sync, the local barrier wait/reset, and address generation are **op-owned and deliberately not wrapped** ("there is no FabricStreamReceiver"; "address generation (TensorAccessor/ShardedAddrGen) is consumed, never re-wrapped"). These are therefore not helper non-uses.

| Phase | Type | Function | File:Line | Args | Input CB | Output CB | Requirements |
|-------|------|----------|-----------|------|----------|-----------|--------------|
| addressing | raw_api | `TensorAccessor(TensorAccessorArgs<idx>(), base_addr, page_size)` + `.get_noc_addr(page_idx)` | consumed per `ccl_helpers_dataflow.hpp:135-136`; precedent `ttnn/ttnn/operations/all_gather/kernels/all_gather_writer.cpp:64` | one accessor each over `input_tensor`, `gathered_tensor`, `output_tensor` | — | — | Host supplies CT args via `ttnn.TensorAccessorArgs(t).get_compile_time_args()`, appended AFTER all scalar CT args. |
| read | raw_api | `noc_async_read` + `noc_async_read_barrier` | `api/dataflow/dataflow_api.h` (precedent `all_gather_reader.cpp:88-90`) | — | — | `cb_broadcast_pages`, `cb_shard_tiles` | — |
| barrier wait | raw_api | `noc_semaphore_wait_min(volatile tt_l1_ptr uint32_t*, uint32_t)` | op-owned per `ccl_helpers_dataflow.hpp:104-108`; precedent `all_gather_reader.cpp:85` | threshold `N - 1` | — | — | Local L1 spin. Threshold is the counting-semaphore form named in the header (`:107-108`). |
| re-arm | raw_api | `noc_semaphore_set(volatile tt_l1_ptr uint32_t*, 0)` | mandated by `ccl_helpers_dataflow.hpp:109-112`; precedent `all_gather_reader.cpp:101` | — | — | — | **Receiver resets AFTER its wait.** Missing reset = first run green, second hangs. |
| sem address | raw_api | `safe_get_noc_addr(x, y, sem_addr, 0)` | precedent `all_gather_writer.cpp:70` | peer worker NoC coords + `sem_addr` | — | — | Same logical core `(0,0)` on every chip ⇒ identical NoC coords, from `mesh_device.worker_core_from_logical_core`. |
| CB-slot reuse | raw_api | `noc_async_writes_flushed()` | precedent `all_gather_writer.cpp:86` | — | `cb_broadcast_pages` | — | Guarantees the fabric sender has read the page out of the CB slot before `cb_pop_front`. `drain()` does NOT cover this (`.inl:180-183` is write+atomic barriers only). |
| output write | raw_api | `noc_async_write` + `noc_async_write_barrier` | `api/dataflow/dataflow_api.h` | — | `cb_output_tiles` | output DRAM | — |

### Host-side program assembly

| Item | API | File:Line / precedent |
|---|---|---|
| Per-device programs | `ttnn.MeshProgramDescriptor()`; `mesh_pd[ttnn.MeshCoordinateRange(coord_i, coord_i)] = program` | `all_gather_program_descriptor.py:241-289` |
| Program | `ttnn.ProgramDescriptor(kernels=[reader, compute, writer], semaphores=[], cbs=[...])` | `all_gather_program_descriptor.py:222-226` |
| Kernels | `ttnn.KernelDescriptor(kernel_source=str(KERNEL_DIR/"..."), core_ranges=core_set, compile_time_args=[...], runtime_args=rt, config=...)` with `ttnn.ReaderConfigDescriptor()` / `ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=..., math_fidelity=...)` / `ttnn.WriterConfigDescriptor()` | `all_gather_program_descriptor.py:140-170`; `ComputeConfigDescriptor` fields per `tt_metal/api/tt-metalium/program_descriptors.hpp:98-107` |
| CBs | `ttnn.CBDescriptor(total_size=..., core_ranges=core_set, format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=..., data_format=input_tensor.dtype, page_size=page_size)])` | `all_gather_program_descriptor.py:100-114` |
| Runtime args | `rt = ttnn.RuntimeArgs(); rt[core.x][core.y] = [...]` | `all_gather_program_descriptor.py:135-138` |
| Tensors into dispatch | `ttnn.generic_op([input_tensor, gathered_tensor, output_tensor], mesh_program_descriptor)` | `all_gather.py:205` |

`ComputeConfigDescriptor` settings: `fp32_dest_acc_en = (input_tensor.dtype == ttnn.float32)` (the shipped one-liner, `llama_reduce_scatter_program_factory.cpp:729`), `dst_full_sync_en = False` (default), `math_fidelity = ttnn.MathFidelity.HiFi4` for bf16 and `HiFi3` for float32 (Wormhole HiFi4 + fp32-dest-acc hardware bug #38306, `ttnn/cpp/ttnn/operations/core/compute_kernel/compute_kernel_config.cpp:43-67`; `add_tiles` pins LoFi internally so this only matters as future-proofing).

## Compute Phases

| # | Operation | Helper? | Input CB (semantic name, tiles, state) | Output CB (semantic name, tiles) | CB State After |
|---|-----------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | Reader streams the P input-shard pages | no (raw `noc_async_read`) | input DRAM | `cb_broadcast_pages` (1 tile per push, P pushes) | `cb_broadcast_pages` drained by the writer as it goes |
| 2 | Writer multicasts the shard to every peer: `arm_write` for pages `0..P-2`, `arm_fused_write_inc` for page `P-1`, then `close()` | **yes** — `FabricDuplexStream` (`ccl_helpers_dataflow.hpp:811,813-814,696,733,821`) | `cb_broadcast_pages` (1 tile per wait) | peers' `gathered_tensor` slot `my_id` + peers' semaphore | `cb_broadcast_pages` empty; fabric drained by `close()`; `input_tensor` still live (re-read in phase 4) |
| 3 | Reader barriers on arrivals and re-arms | no (op-owned per `:104-112`) | local semaphore | — | `sem == 0`; every peer's shard is in `gathered_tensor` (guaranteed by `flush = true` + in-order connection) |
| 4 | Reader interleaves the N contributions to output tile `p`: for `k in [0,N)` read `input.get_noc_addr(p)` if `k == my_id` else `gathered.get_noc_addr(k*P + p)` into `get_write_ptr(cb_shard_tiles) + k*page_size` | no (raw `noc_async_read`) | `input_tensor` + `gathered_tensor` | `cb_shard_tiles` (**N tiles per push**, P pushes) | one N-tile block per output tile; block order is device 0..N-1 |
| 5 | Compute folds the N tiles into one: `tile_regs_acquire`; if N odd `copy_tile(cb_shard_tiles,0,0)` and start at `d=1`, else start at `d=0`; for `d = start; d < N; d += 2` → `add_tiles_init(cb, cb, acc_to_dest = (d != start))` then `add_tiles(cb, cb, d, d+1, 0)`; `tile_regs_commit`; pop N; reserve 1; `tile_regs_wait`; `pack_tile(0, cb_output_tiles)`; `tile_regs_release`; push 1 | no (raw — see justification above) | `cb_shard_tiles` (N tiles, `cb_wait_front(...,N)`) | `cb_output_tiles` (1 tile) | `cb_shard_tiles` block popped; DEST released |
| 6 | Writer drains the reduced tiles to output DRAM | no (raw `noc_async_write`) | `cb_output_tiles` (1 tile per wait) | output DRAM | all CBs empty |

### The N-way fold, verified for every supported N

`acc_to_dest` is `false` on the FIRST pair and `true` thereafter. That makes the fold independent of whether `tile_regs_acquire()` zeroes DEST — the assumption the shipped `llama_reduce_scatter/.../reduction.cpp:26` and `all_reduce_async/.../reduction.cpp:33` both rely on implicitly (they arm `acc_to_dest = true` for every pair including the first).

| N | Seed | Pairs | DEST[0] |
|---|---|---|---|
| 2 | none, `start = 0` | `(0,1)` with `acc=false` | `t0 + t1` |
| 3 | `copy_tile(...,0,0)`, `start = 1` | `(1,2)` with `acc=true` | `t0 + t1 + t2` |
| 4 | none, `start = 0` | `(0,1)` acc=false, `(2,3)` acc=true | `Σ t0..t3` |
| 5 | `copy_tile`, `start = 1` | `(1,2)` acc=true, `(3,4)` acc=true | `Σ t0..t4` |
| 7 | `copy_tile`, `start = 1` | `(1,2),(3,4),(5,6)` all acc=true | `Σ t0..t6` |
| 8 | none, `start = 0` | `(0,1)` acc=false, `(2,3),(4,5),(6,7)` acc=true | `Σ t0..t7` |

`N` is a compile-time arg, so the odd/even branch is `if constexpr`. Cost is `ceil(N/2)` FPU ops per output tile, one DEST register, no intermediate CB. **This closes a real defect in the shipped C++ reference:** `all_reduce_async/.../reduction.cpp:22,42-43` computes `copy_first_block = num_blocks % 2 != 0` and then leaves the odd branch as an empty `// TODO: Future support`, silently DROPPING slice 0 for odd N.

## Broadcast Verification

The op uses one binary op (`add_tiles`). It is a full-tile element-wise add with no broadcast in any dimension.

| Phase | Op | CB_A (semantic name) Valid Region | CB_B (semantic name) Valid Region | Broadcast Dim |
|-------|-----|-----------------------------------|-----------------------------------|---------------|
| 5 (fold) | `add_tiles` (`ELWADD`, `BroadcastType::NONE` — `eltwise_binary.h:208-213`) | `cb_shard_tiles[d]`: 2D `[H,W]` → **All** | `cb_shard_tiles[d+1]`: 2D `[H,W]` → **All** | **None** — both operands are full 32x32 tiles from the same CB at different indices |
| 5 (seed, odd N) | `copy_tile` (unary move, not a binary op) | `cb_shard_tiles[0]`: 2D `[H,W]` → **All** | — | n/a |

No reduce-produced operand is ever fed to the binary op, so no `Row0` / `Col0` valid-region restriction applies.

## Registry Contract

Exported from `ttnn/ttnn/operations/all_reduce/__init__.py` as `all_reduce`, `SUPPORTED`, `EXCLUSIONS`, `INPUT_TAGGERS`.

```python
INPUT_TAGGERS = {"alignment": tag_alignment}   # last two dims both % 32 == 0 -> "tile_aligned"

SUPPORTED = {
    "dtype":     [ttnn.bfloat16, ttnn.float32],
    "layout":    [ttnn.TILE_LAYOUT],
    "topology":  [_Topology.Linear],
    "alignment": ["tile_aligned"],
}

EXCLUSIONS = []
```

`_Topology` must be imported as `from ttnn._ttnn.operations.ccl import Topology as _Topology` — the top-level `ttnn.Topology` alias only binds after `ttnn.operations` is auto-imported (`all_gather.py:27-29`).

`validate()` split, mirroring `all_gather.py:101-168`:

| Check | Raises |
|---|---|
| not a `ttnn.MeshDevice`; mesh view not `(1, N)`; `N < 2`; sharded input; rank < 2; `output_tensor` spec mismatch; `ccl_packet_dims(...).page_segments != 1`; both neighbour routes report the same `is_forward` | `ValueError` (structural) |
| axis value outside `SUPPORTED` | `UnsupportedAxisValue` (a `NotImplementedError` subclass, from `ttnn.operations._op_contract`) — required so the golden harness' `xfail(strict=True, raises=NotImplementedError)` on refinement cells works |
| cell matches an `EXCLUSIONS` entry | `ExcludedCell` |

There is no index axis (no `dim` / `axis` parameter), so no sign-convention canonicalization is needed.

### TARGET / INPUTS / INVALID (pipeline mode — authoritative, do not edit)

`eval/golden_tests/all_reduce/feature_spec.py` already exists and is read as authoritative:

```python
TARGET = {"dtype": [ttnn.bfloat16, ttnn.float32], "layout": [ttnn.TILE_LAYOUT], "topology": [ttnn.Topology.Linear]}
INPUTS = [((1, 1, 32, 32),), ((1, 1, 64, 128),), ((1, 1, 128, 64),)]
INVALID = []
```

`TARGET[axis] - SUPPORTED[axis]` is empty on every axis, so Phase 0 targets the full declared universe. `alignment` is a tagger-only axis (absent from TARGET, present in SUPPORTED) — the harness fills it from `INPUT_TAGGERS` and all three INPUTS are tile-aligned, so every cell is in-support.

**Structural impossibilities (candidates for a future `/golden-tests` pass; NOT edited here).** `INVALID = []` is correct for the current TARGET because `layout` is pinned to TILE and both dtypes are constructible. If TARGET is ever widened:

| Cell | Why INVALID (universe-must-change), not EXCLUSIONS |
|---|---|
| `{"dtype": ttnn.bfloat8_b, "layout": ttnn.ROW_MAJOR_LAYOUT}` | `bfloat8_b` is a tiled block-float format with no row-major representation — the data-format *definition* would have to change. Single-tensor coupling (both axes describe the input). This is the canonical entry carried by both sibling CCL specs. Required the moment `bfloat8_b` and `ROW_MAJOR_LAYOUT` both enter TARGET. |

Everything else currently out of reach is a **kernel improvement**, hence `EXCLUSIONS`/refinement territory, not `INVALID`: `ROW_MAJOR_LAYOUT` (needs tilize/untilize around the fold), `Ring` topology (needs the alternating target-count math and the `range_hops == 0` guard), `non_tile_aligned` (see Risks), `bfloat8_b` on TILE (needs `bfp8_pack_precise` tuning).

## Key Risks and Gotchas

| # | Risk | Mitigation / contract |
|---|------|------------------------|
| 1 | **`eltwise_convenience.hpp` / `eltwise_chain.hpp` do not exist on this branch.** They are referenced by the shared implementer/planner prompts and by `tt_metal/third_party/tt_ops_code_gen/references/ttnn-cb-memory-fundamentals.md:193-199`, but `git ls-files ttnn/cpp/ttnn/kernel_lib/` shows they are untracked at HEAD. Including one will fail to compile. | The compute kernel uses raw `add_tiles` / `copy_tile` per the API Mapping. Do not attempt the helper include. |
| 2 | **`cb_shard_tiles` contiguity.** `add_tiles(cb, cb, d, d+1, 0)` requires the N contributions to one output tile to be N *contiguous* pages. A multi-page `cb_reserve_back` does not guarantee contiguity in general. | `num_pages = 2 * N` (an integer multiple of N) and **every** push/pop is exactly N pages, so the write pointer is always at page offset 0 or N ⇒ N contiguous pages, and `get_write_ptr() + k*page_size` never wraps. Any change to `num_pages` MUST keep it a multiple of N. |
| 3 | **`range_hops == 0` traps the router.** A zero-range multicast header encodes an all-`NOOP` routing field and hits `default: ASSERT(false)` (`tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp:892-894`). | On **Linear** a zero-target direction is always also an unconnected direction, so `DuplexConn::has(dir)` suppresses both the arm and the issue (`ccl_helpers_dataflow.hpp:668-670`, `.inl:314-321,415-418`). **This invariant does NOT hold on Ring** — `get_forward_backward_line_mcast_distance`'s `static_alternate` swap can yield `num_targets_forward == 0` while the forward connection exists (`ccl_common.cpp:1786-1794`). Ring is therefore outside Phase-0 `SUPPORTED`; enabling it requires the explicit zero-range guard that `broadcast_tile_writer.cpp:95-98` implements. |
| 4 | **`is_forward` is NOT "toward increasing index".** `ccl_dm_route` deliberately owns a fwd/bwd sign reversal. Assuming `i+1` is the fabric-forward neighbour will put the wrong multicast range on the wrong connection and silently reduce the wrong subset of devices (wrong values, no hang). | Always slot the two neighbours by their reported `route.is_forward`, and pair each slot's multicast `range_hops` with the neighbour that landed in it. `validate()` asserts the two neighbours do not report the same direction. |
| 5 | **Semaphore cross-call re-arm race.** The counter is reset by the receiver after `wait_min(sem, N-1)`. If a peer's *next* call's inc arrives before this device resets, the reset wipes it and the next call hangs. | Exactly `N-1` incs arrive per call, so the reset is safe *within* a call. Across calls the window is bounded by the peer's `stream.close()` drain. This is precisely the pattern the sibling Python op ships with (`all_gather_reader.cpp:79-101`), and the acceptance test's `ttnn.synchronize_device(mesh_device)` between calls closes the window. **Known limitation:** back-to-back calls with no intervening device completion are not guaranteed. The refinement is a dedicated cross-device barrier semaphore at kernel entry (the `all_gather_async` shape, `llama_shapes_sharded_writer.cpp:96-118`). Record it in `op_requirements.md`. |
| 6 | **`flush = true` on the fused channel is load-bearing.** The payload lands in DRAM while the semaphore lives in L1 — different destinations, so without a flush the inc can overtake the write and the reader would read stale DRAM (wrong values, no hang). | `arm_fused_write_inc(page_size, 1, /*flush=*/true)`. The receiving endpoint then flushes its NoC write pipeline before the inc (`fabric_edm_packet_transmission.hpp:203-205`), which also covers the earlier in-order packets on that channel. This is a deliberate deviation from `all_reduce_async/.../worker_writer.cpp:88-99`, which uses `flush = false` because its payload targets L1 shards. |
| 7 | **The gathered buffer must be at the same address on every device.** The fabric carries a noc0-encoded address; if the buffer were allocated at different offsets per device, pages would land in the wrong place. | It is ONE mesh allocation (`ttnn.allocate_tensor_on_device(spec, mesh_device)`), so the address is uniform by construction — the same property `all_gather` relies on when it writes into a peer's output DRAM. Do not allocate it per-device. |
| 8 | **Slot `my_id` of the gathered buffer is never written.** N slots (not N-1) are required because a *multicast* sender cannot know which receiver it is talking to, so all senders must agree on "slot = sender id". | Deliberate: `1/N` of the intermediate is unused. The reader special-cases `k == my_id` and reads that contribution from `input_tensor` instead. Do not "optimise" to N-1 slots. |
| 9 | **`fp32_dest_acc_en` must track the dtype.** With fp32 CBs and 16-bit DEST, every accumulation step is silently rounded to bf16 — with N=8 that is up to 4 chained roundings and a blown 0.999 PCC gate. | `ComputeConfigDescriptor(fp32_dest_acc_en = (dtype == ttnn.float32))` (`llama_reduce_scatter_program_factory.cpp:729`). DEST capacity drops to 4 tiles, which is fine — the design uses 1. `DEST_AUTO_LIMIT` (`dest_helpers.hpp:89-103`) auto-detects the host setting so a `static_assert` in the kernel cannot desync. |
| 10 | **`pack_tile`'s `output_tile_index` is ignored by default.** With `out_of_order_output = false` packing is sequential from 0 within the reserved region (`pack.h:59-67`), so a passed index silently does nothing. | Reserve 1 page and call `pack_tile(0, cb_output_tiles)` with no index. |
| 11 | **`compute_kernel_hw_startup` exactly once, at the top.** It performs MMIO writes and is unsafe mid-kernel (`compute_kernel_hw_startup.h:54-58`). Calling it and `binary_op_init_common` both is redundant/unsafe. | One `compute_kernel_hw_startup(cb_shard_tiles, cb_shard_tiles, cb_output_tiles)` at the top; only the short inits (`add_tiles_init`, `copy_tile_init`) appear inside the loop. |
| 12 | **Sender must outlive its stream.** `FabricDuplexStream` borrows the sender's `ConnT*` (`ccl_helpers_dataflow.hpp:781-784,827`). | Declare `FabricDuplexSender` first, in the same scope, above `open()`. Streams are move-only with move-assign deleted; `close()` is idempotent and the destructor is the backstop. |
| 13 | **`non_tile_aligned` breaks the gathered-buffer page mapping for rank 2.** Scaling dim 0 by N only preserves `slot k == pages [k*P,(k+1)*P)` when each shard occupies whole tile-rows. For rank 2 with `H % 32 != 0` (e.g. `(48,64)`: `P = 4`, but `(384,64)` has 24 pages, not `N*P = 32`), the tile-padding does not survive concatenation. | `SUPPORTED["alignment"] = ["tile_aligned"]` and `validate()` requires rank >= 2. Rank >= 3 non-tile-aligned would actually work (dim 0 is a batch dim) — enabling it is a rank-conditional refinement, recorded in `op_requirements.md`. |
| 14 | **`page_segments > 1` would silently corrupt.** A page larger than one fabric packet must be segmented; the design sends one whole page per packet. | `validate()` gates `ccl_packet_dims(...).page_segments == 1`. Holds for every supported dtype (bf16 2048 <= 4096; fp32 4096 <= 4400). |
| 15 | **`alignment = 1` on the duplex sender.** `align(page_size, 1) == page_size`, so the on-wire size can never round up past the destination page. | Matches the only shipped duplex user (`all_reduce_async/.../worker_writer.cpp:88`). Tile page sizes (2048 / 4096 B) are already 32-B aligned for the DRAM write, so no rounding is needed. |
| 16 | **`noc_async_writes_flushed()` before `cb_pop_front`.** `stream.drain()` / `close()` only issue write + atomic barriers (`.inl:180-183`); they do not guarantee the fabric sender has finished reading the CB slot. | Flush after every issue, before popping — the `all_gather_writer.cpp:86` pattern. |
| 17 | **Untracked `tests/ttnn/unit_tests/operations/all_reduce/conftest.py` adds `use_module_device` to every item.** The marker is honoured only by the single-device `device` fixture; for `mesh_device` tests it is inert. It becomes a hard `ValueError` if a test in that directory ever requests `device` while `device_params` is parametrized. | Harmless as written — both tests in the directory use `mesh_device` exclusively. Do NOT add a single-device test to this directory, and do NOT define a local `device` fixture (it would shadow the root one). |
| 18 | **Single core is the Phase-0 grid.** All P pages of the broadcast and the whole reduction run on one Tensix, so large shards are latency-bound. | Deliberate: `MuxConn<N>` cannot back the duplex tier (it exposes `sender()` with no direction, `ccl_helpers_dataflow.hpp:282`, while duplex channels call `conn_->has(d)` / `conn_->sender(d)`, `.inl:415-422`), so multi-core needs one fabric link per core. Refinement, recorded in `op_requirements.md`. |

## Hardware Constraints

- [x] CB sync: push count = wait count = pop count for every CB (see Sync verification table — `P`x1, `P`xN, `P`x1)
- [x] Reduce scaler CB is bfloat16 — **n/a**: no `reduce_helpers` scaler CB exists in this op (see the helper rejection justification)
- [x] Reduce scaler uses the pool-type-aware API — **n/a**: no reduce scaler
- [x] DEST: 1 tile used; `DEST_AUTO_LIMIT` is 8 (bf16) / 4 (fp32) — guarded by `static_assert(1 <= compute_kernel_lib::DEST_AUTO_LIMIT)`
- [x] Sequential helper intermediates sized to full block — `cb_shard_tiles` holds a full N-tile block (and is a multiple of N pages, Risk 2)
- [x] Page sizes aligned to tile size — every CB page is `input_tensor.buffer_page_size()` (one tile)
- [x] RM CBs count pages in sticks, tile CBs count in tiles — all three CBs are tile CBs; no RM CB in this op
- [x] All `cb_wait_front` calls on the same CB use the same page count — `cb_broadcast_pages` 1, `cb_shard_tiles` N, `cb_output_tiles` 1, invariant across all iterations
- [x] `compute_kernel_hw_startup()` called before any helper/compute usage — exactly once at the top of the compute kernel (Risk 11)
- [x] Helpers are not wrapped with extra CB operations — the duplex channels own their packet-header state; the op owns only `cb_wait_front` / `noc_async_writes_flushed` / `cb_pop_front` around the issue, which the header designates as op-owned (`ccl_helpers_dataflow.hpp:130-140`)
