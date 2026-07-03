# Operation Design: all_gather

## Overview

| Field | Value |
|-------|-------|
| Classification | CCL (collective communication) — pure cross-chip data movement, NO arithmetic |
| Goal | Gather every device's shard of a mesh-sharded tensor and concatenate all shards along `gather_dim`, so AFTER the op every participating device holds the full concatenated tensor. |
| Math | For a line of `N` devices where device `j` holds shard `S_j` (sharded along `gather_dim`): `output[d] = concat(S_0, S_1, …, S_{N-1}, dim=gather_dim)` for EVERY device `d`. Identity gather (element values unchanged). |
| Mode | Derivative — newly-authored Python `generic_op` + `MeshProgramDescriptor` op with newly-authored ring dataflow kernels. The bound C++ `ttnn.all_gather` / `all_gather_async` is NEVER imported, called, wrapped, or dispatched to. The experimental `all_gather_async` writer and the `point_to_point` op are correctness references only. |
| Topology | `Linear` (primary, the sim's `FABRIC_1D` line). `Ring` is a declared target (a Ring wrap would let a single direction suffice; the Linear design below is bidirectional and also runs on a Ring). |
| Devices participating | All `N` devices on the 1-D mesh line. One `ProgramDescriptor` per device (per `MeshCoordinate`). |
| References | KERNEL egress helper `ttnn/cpp/ttnn/kernel_lib/ccl_helpers_dataflow.hpp` (+`.inl`); HOST helpers `ttnn/cpp/ttnn/operations/ccl/common/host/ccl_helpers_dataflow_host.hpp`; Python bindings `ttnn/cpp/ttnn-nanobind/fabric.cpp:141-266`; ring algorithm reference `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/minimal_default_writer.cpp` (slice-walk `:406-415`, barrier `:190-224`, phase-1 own-slice `:236-357`, chunks_per_sync `:325-340`, relay `:389-545`); route args `ttnn/cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp:15-97`; structural template `ttnn/ttnn/operations/point_to_point/` (op + program descriptor + kernels); GlobalSemaphore ownership gold standard `ttnn/cpp/ttnn/operations/experimental/ccl/deepseek_moe_reduce_scatter/device/*`. |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | On a `ttnn.MeshDevice` line of `N ≥ 2` devices; interleaved (DRAM or L1); one shard per device (sharded along `gather_dim`); per-shard page size 16-byte aligned | — | host |
| `gather_dim` | `int` | yes | `-rank … rank-1`; canonicalized to the NEGATIVE convention (`gd = gather_dim if gather_dim < 0 else gather_dim - rank`) before the support check — matches `feature_spec.py` TARGET `[-4,-3,-2,-1]` | — | host → concat stride (CT args) |
| `topology` | `ttnn.Topology` | no | `Linear`, `Ring` | `ttnn.Topology.Linear` | host → routing (`ccl_dm_route`) |
| `output_tensor` | `ttnn.Tensor \| None` | no | spec must equal the resolved output spec | `None` | host |

There are no compute-kernel template params (pure data movement — no unpack/math/pack). All op-specific values reach the kernels as compile-time args (CB indices, alignment, ring_size, ring_index, gather stride table, packet dims, route infos) and runtime args (buffer addresses, semaphore addresses, per-device relay counts, fabric-connection block).

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | per-device shard shape, rank ≥ 2; `input.shape[gather_dim]` is this device's slice size along the gather dim |
| Dtype | `bfloat16` (primary), `float32`, `bfloat8_b` (TILE only) |
| Layout | `TILE_LAYOUT` (primary) or `ROW_MAJOR_LAYOUT` |
| Memory | interleaved (DRAM or L1); sharded/non-interleaved rejected |
| Distribution | sharded across a `MeshDevice` line along `gather_dim` (`ShardTensorToMesh(mesh_device, dim=gather_dim)`); device at column `c` holds shard `S_c` |

### Output

| Property | Value |
|----------|-------|
| Shape | input shape with `gather_dim` scaled by `N`: `output.shape[gather_dim] = input.shape[gather_dim] * N`; all other dims identical |
| Dtype | same as input |
| Layout | same as input |
| Memory | same memory config as input (interleaved DRAM/L1) |
| Content | **identical on every device** — the full `concat(S_0…S_{N-1}, dim=gather_dim)`. Bit-for-bit identity gather (PCC ~1.0). |
| Allocation | replicated: `ttnn.allocate_tensor_on_device(output_spec, mesh_device)` gives each device a full-shape buffer at the same L1/DRAM address (mirrored), which the fabric writes target on the neighbor chip. |

No host-side output seeding is needed (unlike point_to_point): every output slot is written by a kernel — the own slot by a local NoC write, the remote slots by fabric writes — and the counting semaphores gate the op's completion until all slots have landed.

## Multi-Device Distribution

The op dispatches a single `ttnn.generic_op([input_tensor, output_tensor], mesh_program_descriptor)` where `mesh_program_descriptor` is a `ttnn.MeshProgramDescriptor` holding **`N` entries** — one `(MeshCoordinateRange(coord, coord), ProgramDescriptor)` per device coordinate on the line (`program_descriptors.cpp:1039-1048` `__setitem__`). Every device runs the same ring role parameterized by its `ring_index` (its column on the `(1, N)` line).

| Concept | Value |
|---------|-------|
| `ring_size` `N` | `prod(mesh_device.shape)` (= 8 on the sim's `(1, 8)` line) |
| `ring_index` `d` | device column: `coord[1]` for a `(1, N)` line (`0 … N-1`) |
| forward neighbor | `MeshCoordinate(0, d+1)` if `d < N-1` |
| backward neighbor | `MeshCoordinate(0, d-1)` if `d > 0` |

Each device's program runs **two worker cores**:

| Core | CoreCoord | Direction | Fabric connection |
|------|-----------|-----------|-------------------|
| `core_fwd` | `(0, 0)` | forward — sends toward higher `ring_index` (to `d+1`) | forward, 1 hop (only if `d < N-1`) |
| `core_bwd` | `(0, 1)` | backward — sends toward lower `ring_index` (to `d-1`) | backward, 1 hop (only if `d > 0`) |

Each core has a reader (NCRISC) + writer (BRISC). The two cores run concurrently and are independent (they touch disjoint output slots and disjoint semaphores), which is why the single op-internal counting semaphore per direction is unambiguous (single producer per device per direction — see Cross-Device Coordination).

### Ring algorithm (bidirectional store-and-forward, single-hop relay)

Store-and-forward: every fabric write is **one hop** to the immediate neighbor's OUTPUT buffer at the concat offset for the slice's origin. A slice reaches a far device by being re-read from the intermediate device's output and re-pushed one hop at a time. This keeps all routes at `distance_in_hops = 1` (matching `minimal_default_writer.cpp` — a single-hop store-and-forward, `ccl_common.cpp` line-unicast `distance_in_hops = 1`).

Slice delivery invariant (verified correct):
- **Forward channel**: device `d` delivers slices `{0 … d}` to `d+1`. It sends its own `S_d` first, then relays the forward slices it received from `d-1` in arrival order `S_{d-1}, S_{d-2}, …, S_0`. Device `d` **receives** `d` forward slices (`{0 … d-1}`) from `d-1`.
- **Backward channel**: device `d` delivers slices `{d … N-1}` to `d-1`. It sends its own `S_d` first, then relays the backward slices it received from `d+1` in arrival order `S_{d+1}, S_{d+2}, …, S_{N-1}`. Device `d` **receives** `N-1-d` backward slices (`{d+1 … N-1}`) from `d+1`.

After both channels drain, device `d` holds: slot `d` (own, local write) + slots `{0 … d-1}` (forward, written by `d-1`'s forward writer) + slots `{d+1 … N-1}` (backward, written by `d+1`'s backward writer) = all `N` slots. ✔

Pseudocode per device `d` (forward channel shown; backward is the mirror with `d±1`, `d`↔`N-1-d` swapped):

```text
# core_fwd
BARRIER:  core_fwd issues the forward-direction barrier multicast (range N-1-d)   # + core_bwd issues the backward multicast (range d)
          core_fwd waits barrier_sem >= N-1; resets barrier_sem = 0                # genuine N-party rendezvous, forward-core-owned
OWN SLICE (r = 0):
          read S_d from input -> cb_fwd_relay
          LOCAL noc_async_write S_d -> output slot d          # own slot, once, local
          if d < N-1: fabric-write S_d -> (d+1) output slot d ; arm_inc (d+1).forward_sem += 1
RELAYS (r = 1 … d), origin j = d-r:
          reader waits forward_sem >= r ; re-reads output slot j -> cb_fwd_relay    # slice j landed (data-before-inc ordering)
          if d < N-1: fabric-write slice j -> (d+1) output slot j ; arm_inc (d+1).forward_sem += 1
FINAL:    reader waits forward_sem >= d (all d forward slices landed) ; resets forward_sem = 0
          writer: drain() + close()
```

Concat-by-`gather_dim` output addressing: slice from origin chip `j` occupies output pages starting at `tile_id_start = j * stride(gather_dim)`. Stride table (TILE layout, tile counts; mirrors `minimal_default_writer.cpp:247-256, 449-457`):

| gather_dim | `stride` (pages per slice step) | Output page walk within a slice |
|------------|--------------------------------|---------------------------------|
| `0` (outermost) — **primary/proven** | `pages_per_shard` (whole shard) | contiguous: pages `[j*pages_per_shard, (j+1)*pages_per_shard)` |
| `1` | `C_in * Ht * Wt` | contiguous block per slice |
| `2` (H) | `Ht_in * Wt` | rows stride by `output_Wt`, read `input_Wt` per row |
| `3` (W, innermost) | `input_Wt` | rows stride by `output_Wt`, read `input_Wt` per row |

For `gather_dim = 0` the slice is a contiguous page range — the reader/writer walk sequential tile ids `j*pages_per_shard + [0 … pages_per_shard)`. Non-zero gather dims use the interleaved row-stride walk (a declared refinement; the acceptance test proves `gather_dim = 0`).

## Cross-Device Coordination

**Three** op-internal `GlobalSemaphore`s, created ONCE per `mesh_device`, cached on the module, and parked on `mesh_program_descriptor.semaphores` (`program_descriptors.cpp:1077-1087` — a `std::vector<GlobalSemaphore>`, matching the experimental op's `semaphore.at(dir)` per-direction pattern). The framework holds their L1 alive across program-cache hits; NO per-call post-dispatch `synchronize_device` barrier is added.

| Semaphore | Role | Producer(s) | Owner (waits + resets) | Mechanism |
|-----------|------|-------------|------------------------|-----------|
| `barrier_sem` | N-party readiness rendezvous | every device's `core_fwd` (forward multicast) + `core_bwd` (backward multicast) inc all peers | each device's `core_fwd` waits `>= N-1`, resets `0` | `arm_multicast_inc` + `noc_semaphore_wait_min(N-1)` + `noc_semaphore_set(0)` |
| `forward_sem` | forward-channel counting (store-and-forward flow control) | ONLY `d-1`'s `core_fwd` writer (`arm_inc += 1` per delivered slice) | device `d`'s `core_fwd` reader waits `>= r`, resets `0` | `arm_inc` + `noc_semaphore_wait_min(r)` + `noc_semaphore_set(0)` |
| `backward_sem` | backward-channel counting | ONLY `d+1`'s `core_bwd` writer (`arm_inc += 1` per delivered slice) | device `d`'s `core_bwd` reader waits `>= r`, resets `0` | `arm_inc` + `noc_semaphore_wait_min(r)` + `noc_semaphore_set(0)` |

**Why three semaphores, single-owner each (the correctness argument).** A Linear all_gather is fundamentally bidirectional (a line cannot wrap), so on every device the forward channel (fed by `d-1`) and the backward channel (fed by `d+1`) are two INDEPENDENT counting streams running concurrently. A counting semaphore is unambiguous only with a **single producer**: merging the two directions onto one semaphore makes device `d` unable to tell whether an increment signalled a forward slice or a backward slice, so it cannot know which output slot has landed. Hence one counting semaphore per direction. The barrier is a third semaphore with pure barrier semantics so it never shares a reset window with either counting stream (a shared barrier↔counting semaphore has a reset race: a lagging peer's barrier inc can land on an already-reset counter, or a counting inc can be wiped by a barrier reset). Each semaphore has exactly one owning core per device that waits and resets it, so there is **no cross-core reset race**.

**Load-bearing ordering (data-before-inc).** The counting `arm_inc` from `d-1` to `d` travels the SAME 1-hop fabric route as the slice data that precedes it, so fabric ordering guarantees the slice has landed in `d`'s output before `d` observes the increment. Device `d` therefore safely re-reads output slot `j` once `forward_sem >= r` (the `r`-th slice). This is why counting (not a coarse multicast barrier, whose route differs from the data route and gives no landing guarantee) is the correctness mechanism for the relay dependency.

**Cache-reuse re-arm.** Every device drives each semaphore `0 → … → 0`: the counting owner resets AFTER its final wait (all incoming incs already counted — no more coming), and `core_fwd` resets `barrier_sem` AFTER seeing `N-1` (all peers already inc'd — no more coming). Single command-queue ordering guarantees a launch's resets complete before the next cached launch's incs. Missing/mis-ordered resets = run 1 green, run 2 hangs.

## Dataflow Strategy

Data path (per slice): `origin DRAM/L1 → origin core CB → fabric (1 hop) → neighbor OUTPUT DRAM/L1 → neighbor core CB (re-read) → fabric (1 hop) → …`. Format is preserved end to end (the op copies tiles/pages, never tilizes/untilizes). Fabric writes land **directly in the neighbor's output tensor** at the concat offset — no separate intermediate landing buffer (unlike point_to_point), because the concat addressing places each slice in its final output slot immediately, and the relay re-reads it from there.

### Forward core (`core_fwd`) kernels

| Kernel | RISC / config | Reads | Writes | Helpers |
|--------|---------------|-------|--------|---------|
| `all_gather_fwd_reader` | NCRISC / `ReaderConfigDescriptor` | `input_tensor` (own shard) + `output_tensor` (re-read landed forward slices, local) via `TensorAccessor` | `cb_fwd_relay` | `TensorAccessor` (raw); `noc_async_read`/barrier (raw ingress); `noc_semaphore_wait_min`/`set` (raw, the WAITING half + re-arm) |
| `all_gather_fwd_writer` | BRISC / `WriterConfigDescriptor` | `cb_fwd_relay` | own output slot (local `noc_async_write`) + neighbor `output_tensor` via fabric | `FabricStreamSender`/`FabricStream`/`arm_multicast_inc`/`arm_scatter_write`/`arm_unicast_write`/`arm_inc`/`multicast_inc`/`write_scatter`/`write_page`/`inc`/`drain`/`close`; `unicast_route`; `get_line_multicast_route_info_from_args`; `TensorAccessor` (raw); `noc_semaphore_wait_min`/`set` (raw barrier wait+reset) |

### Backward core (`core_bwd`) kernels

Mirror of the forward core with `d+1 → d-1`, `forward_sem → backward_sem`, relay count `d → N-1-d`, and the backward barrier multicast (range `d`). The backward writer does **not** perform the own-slot local write (the forward writer owns it) and does **not** wait on `barrier_sem` (it only issues its backward barrier multicast, then proceeds — correctness is from `backward_sem` counting).

### Kernel-side fabric-connection arg contract (non-derivable — pin it)

Each fabric-using writer builds its connection with `FabricStreamSender<>(conn_arg_idx, is_forward, alignment)` where the word at `conn_arg_idx` is `has_forward = int(is_forward)` (the kernel peeks it for direction, then the ctor consumes the whole block). The host lays the block out exactly as `append_ccl_fabric_rt_args` does (`ccl_helpers_dataflow_host.hpp:219-237`), which is what `ttnn.setup_fabric_connection` (`fabric.cpp:141-178`) produces:

```
[ has_forward = int(is_forward) ]
[ <forward connection args from setup_fabric_connection> ]   # only if is_forward
[ has_backward = int(not is_forward) ]
[ <backward connection args from setup_fabric_connection> ]  # only if not is_forward
```

End devices (`d = N-1` forward core; `d = 0` backward core) have **no neighbor in that direction**, so the host emits NO fabric-connection block and the writer constructs NO `FabricStreamSender` for that direction — it only does the local own-slot write (`core_fwd`, `d=N-1`) and the local semaphore waits/resets. This matches the experimental op's `valid_targets` gating.

### Route + barrier CT-arg layout

The four route infos are compile-time args in the fixed order the writer reads them (`worker_routing_utils.hpp:46-56`; `minimal_default_writer.cpp:58-72`): `[fwd_unicast(2)][fwd_barrier_multicast(6)][bwd_unicast(2)][bwd_barrier_multicast(6)]`. For 1-D fabric the values are trivial (`ccl_common.cpp` line configs): unicast = `{dst_mesh_id=0, distance_in_hops=1}`; forward barrier multicast = `{start_distance_in_hops=1, range_hops=(N-1-d), 0,0,0,0}`; backward barrier multicast = `{start_distance_in_hops=1, range_hops=d, 0,0,0,0}`. The Python builder packs these directly (the C++ `append_ccl_line_route_ct_args` is not Python-bound; the 1-D packing is the literal 6-uint layout above).

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one output slot's worth of pages (`pages_per_shard = input_tensor.buffer_num_pages()` tiles), coalesced into `num_tiles_per_packet`-tile fabric packets |
| Grid | two cores per device: `core_fwd = CoreCoord(0,0)`, `core_bwd = CoreCoord(0,1)` (each a 1×1 `CoreRangeSet`) |
| Per-core work | `core_fwd`: own slice (1) + `d` forward relays = `d+1` slices sent to `d+1` (0 if `d=N-1`), plus the own-slot local write and `d` gated re-reads. `core_bwd`: own slice (1) + `N-1-d` backward relays = `N-d` slices sent to `d-1` (0 if `d=0`), plus `N-1-d` gated re-reads. |
| Remainder | last packet of a slice carries `min(num_tiles_per_packet, tiles_remaining)` tiles; single-tile tail uses `arm_unicast_write`/`write_page`, multi-tile packets use `arm_scatter_write`/`write_scatter` (≤4 dsts, the `NocUnicastScatter` limit) |

`num_tiles_per_packet = min(4, max(1, packet_size_bytes // tile_size_bytes))` where `packet_size_bytes` comes from `ccl_packet_dims` (owns the bf16 `bit_floor`). Multi-link / multi-worker-per-direction page fan-out is out of scope (single worker per direction, mirroring the minimal single-core reference); the design leaves room for it (per-worker page partitioning) but does not implement it.

## Circular Buffers

### Forward core (`core_fwd`)

| Semantic Name | Index | Page Size | Num Pages | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|--------|----------|----------|----------|
| `cb_fwd_relay` | 0 | `tile_size(dtype)` (TILE) / `align(buffer_page_size(),16)` (RM) | `3 * num_tiles_per_packet` | input dtype | `all_gather_fwd_reader` | `all_gather_fwd_writer` | streaming triple-buffer (own shard pages, then re-read forward-slice pages) |

### Backward core (`core_bwd`)

| Semantic Name | Index | Page Size | Num Pages | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|--------|----------|----------|----------|
| `cb_bwd_relay` | 0 | `tile_size(dtype)` (TILE) / `align(buffer_page_size(),16)` (RM) | `3 * num_tiles_per_packet` | input dtype | `all_gather_bwd_reader` | `all_gather_bwd_writer` | streaming triple-buffer |

No packet-scratch CB and no packet-header CB. The scatter/unicast source is the contiguous CB tiles directly (each tile is written to its own output tile position, so no L1 coalescing buffer is needed). `FabricStreamSender` draws packet headers from the fabric-L1 `PacketHeaderPool` (a fixed reserved region, self-provisioning — no program-side CB).

CB sync (push == wait), per core:
- `cb_fwd_relay`: reader pushes `pages_per_shard` (own) + `d * pages_per_shard` (relays) tiles = `(d+1) * pages_per_shard`; writer waits/pops the same. For `d = N-1` the writer still consumes the own-slice pages (local write) but pushes/consumes no relay pages (reader waits on `forward_sem` without re-reading). ✔
- `cb_bwd_relay`: reader pushes `(N-1-d+1) * pages_per_shard` (or `0` own-read for `d=0`); writer waits/pops the same. ✔

## API Mapping

Every mechanism — helper or raw — with an exact file:line reference.

| Phase | Type | Function | File:Line | Args / Notes | In CB | Out CB |
|-------|------|----------|-----------|--------------|-------|--------|
| host: packet framing | helper (Py-bound) | `ttnn._ttnn.fabric.ccl_packet_dims` | `fabric.cpp:245-252`; impl `ccl_helpers_dataflow_host.hpp:74` | `(dtype, page_size_bytes, num_pages, l1_alignment)` → `.packet_size_bytes/.pages_per_packet/.page_segments/.total_packets`; owns bf16 `bit_floor` | — | — |
| host: routing (unicast) | helper (Py-bound) | `ttnn._ttnn.fabric.ccl_dm_route` | `fabric.cpp:253-266`; impl `ccl_helpers_dataflow_host.hpp:137` | `(mesh_device, my_coord, neighbor_coord, topology)` → `.num_hops(=1)/.is_forward/.neighbor_id`; owns fwd/bwd sign reversal + ring short-way | — | — |
| host: fabric conn args | helper (Py-bound) | `ttnn.setup_fabric_connection` | `fabric.cpp:141-178` | `(src_id, neighbor_id, link_idx=0, program, core)`; mutates program (appends SemaphoreDescriptors), returns RT arg vector; laid out per the arg contract | — | — |
| host: fabric node id | helper (Py-bound) | `mesh_device.get_fabric_node_id` | `distributed_nanobind.cpp:272` | `(coord)` → `FabricNodeId` | — | — |
| host: mesh geometry | helper (Py-bound) | `mesh_device.shape`, `ttnn.MeshCoordinate`, `ttnn.MeshCoordinateRange` | `distributed_nanobind.cpp:362-371, 183-200, 221-227` | enumerate `[MeshCoordinate(0,i) for i in range(N)]` | — | — |
| host: semaphores | helper (Py-bound) | `ttnn.create_global_semaphore` / `get_global_semaphore_address` / `synchronize_device` | Python equivalent of `make_ccl_semaphore` (`ccl_helpers_dataflow_host.hpp:250-256`) | create ×3 once per mesh + `synchronize_device` once; park on `mesh_program_descriptor.semaphores` | — | — |
| host: descriptor | helper (Py-bound) | `ttnn.MeshProgramDescriptor` `__setitem__` / `.semaphores` | `program_descriptors.cpp:1039-1048, 1077-1087` | one `ProgramDescriptor` per `MeshCoordinateRange(coord,coord)`; `.semaphores = [barrier_sem, forward_sem, backward_sem]` | — | — |
| host: dispatch | helper (Py-bound) | `ttnn.generic_op` over `MeshProgramDescriptor` | `ttnn/cpp/ttnn/operations/generic/generic_op_nanobind.cpp` | `([input_tensor, output_tensor], mesh_pd)` | — | — |
| kernel: connection | helper | `FabricStreamSender<>(conn_arg_idx, is_forward, alignment)` → `.open(route)` | `ccl_helpers_dataflow.hpp:436-451` | reads conn RT block; binds direction + 1-hop unicast route | — | — |
| kernel: route build | helper | `unicast_route(1)` / `get_line_multicast_route_info_from_args<idx>()` | `ccl_helpers_dataflow.hpp:260-265`; `worker_routing_utils.hpp:46-56` | 1-hop unicast route; barrier multicast route from CT args | — | — |
| kernel: arm barrier | helper | `FabricStream::arm_multicast_inc(mcast_route, 1)` | `ccl_helpers_dataflow.hpp:392-393`; impl `.inl` | dedicated mcast header; **block-scope before `arm_inc`** (footgun) | — | — |
| kernel: issue barrier | helper | `MulticastIncChannel::multicast_inc(barrier_sem_noc_addr)` | `ccl_helpers_dataflow.hpp:335` | N-party barrier inc to all peers on the route | — | — |
| kernel: arm scatter | helper | `FabricStream::arm_scatter_write(tile_size_bytes, num_tiles_per_packet)` | `ccl_helpers_dataflow.hpp:381` | ≤4 tile dsts per packet | — | — |
| kernel: issue scatter | helper | `ScatterWriteChannel::write_scatter(dst_noc_addrs, n, src_l1)` | `ccl_helpers_dataflow.hpp:300` | multi-tile slice segment → neighbor output tiles | `cb_*_relay` | neighbor `output_tensor` |
| kernel: arm unicast | helper | `FabricStream::arm_unicast_write(tile_size_bytes)` | `ccl_helpers_dataflow.hpp:374` | single-tile tail | — | — |
| kernel: issue unicast | helper | `UnicastWriteChannel::write_page(src_l1, tile_id, output_addrgen)` | `ccl_helpers_dataflow.hpp:284` | single tile → neighbor output tile | `cb_*_relay` | neighbor `output_tensor` |
| kernel: arm inc | helper | `FabricStream::arm_inc(1)` | `ccl_helpers_dataflow.hpp:386` | counting inc value | — | — |
| kernel: issue inc | helper | `AtomicIncChannel::inc(neighbor_sem_noc_addr)` | `ccl_helpers_dataflow.hpp:317` | `+1` per delivered slice (chunks_per_sync = whole slice) | — | — |
| kernel: teardown | helper | `FabricStream::drain()` / `close()` | `ccl_helpers_dataflow.hpp:399, 402` | flush write + atomic barriers, close connection (idempotent) | — | — |
| kernel: own-slot local write | raw_api | `noc_async_write` + `TensorAccessor` | pattern `point_to_point_receiver_writer.cpp:35`; addrgen `writer_unary_interleaved_start_id_gen` | own `S_d` → own output slot `d` (local, `core_fwd` only) | `cb_fwd_relay` | own `output_tensor` |
| kernel: input read | raw_api | `noc_async_read` + `noc_async_read_barrier` + `TensorAccessor` | pattern `point_to_point_sender_reader.cpp:34-36` | own shard → CB | `input_tensor` | `cb_*_relay` |
| kernel: relay re-read | raw_api | `noc_async_read` + `noc_async_read_barrier` + `TensorAccessor` | pattern `point_to_point_receiver_reader.cpp:73-75` | landed slice (local output) → CB, after counting wait | `output_tensor` | `cb_*_relay` |
| kernel: addressing | raw_api | `TensorAccessor(args, base_addr, page_size)` / `.get_noc_addr(tile_id)` | `point_to_point_*_reader.cpp` | tile_id = `origin*stride + offset`; consumed by `write_page`/`write_scatter` | — | — |
| kernel: sem wait/reset | raw_api | `noc_semaphore_wait_min` / `noc_semaphore_set` | `point_to_point_sender_writer.cpp:64-65`, `receiver_reader.cpp:60,95` | WAITING half + cache-reuse re-arm (all three sems) | — | — |

### Helpers considered and rejected (for each raw-API fallback)

- **`noc_semaphore_wait_min` / `noc_semaphore_set` (barrier wait + all counting waits + all re-arms).** Candidate: `ccl_helpers_dataflow.hpp` channels. Rejected — the banner at `ccl_helpers_dataflow.hpp:69-77` states the SENDING half (`inc`/`multicast_inc`) is owned by the helper but the **WAITING half is "a plain local `noc_semaphore_wait_min(sem, threshold)` the op calls directly (1 = handshake, ring_size-1 = N-party barrier, sem_target = counting)"** and the cache-reuse `noc_semaphore_set(sem,0)` re-arm is explicitly op-owned. No helper API waits or resets; `arm_inc` increments, it does not wait.
- **`noc_async_read` / `noc_async_read_barrier` (own-shard read + relay re-read of landed slices).** Candidate: a fabric receive helper. Rejected — `ccl_helpers_dataflow.hpp:66-67, 89-92` state "the receive INGRESS is likewise a local NoC read the op owns; there is no FabricStreamReceiver," and lists "store-and-forward relay (re-read the landed slice)" among what the op composes. The re-read of a landed slice from local output DRAM is exactly this op-owned ingress.
- **`noc_async_write` (own-slot local write).** Candidate: `UnicastWriteChannel::write`. Rejected — the own-slot write is a purely LOCAL DRAM/L1 write on the same chip (no fabric hop); the fabric helper's `write`/`write_page` (`ccl_helpers_dataflow.hpp:280,284`) issue over the fabric connection and would route off-chip. A same-chip write is a stock `noc_async_write`.
- **`TensorAccessor` (DRAM/L1 addressing).** Candidate: an addrgen helper. Rejected — `ccl_helpers_dataflow.hpp:84-85, 92` state "address generation (TensorAccessor/ShardedAddrGen) is consumed, never re-wrapped"; `TensorAccessor` IS the intended primitive and `write_page` consumes it.
- **Python-side 1-D route packing / semaphore creation.** Candidate: C++ `append_ccl_line_route_ct_args` / `make_ccl_semaphore`. Rejected — neither is Python-bound (`fabric.cpp` binds only `ccl_packet_dims`/`ccl_dm_route`/`setup_fabric_connection`; verified no binding for `append_ccl_line_route_ct_args`/`get_forward_backward_line_mcast_configuration`). The 1-D multicast packing is the literal 6-uint `[1, range, 0,0,0,0]` layout, and `create_global_semaphore` + `synchronize_device` is the Python equivalent of `make_ccl_semaphore`.

## Dataflow Phases

| # | Phase | Core | Consumes | Produces | Sem state after |
|---|-------|------|----------|----------|-----------------|
| 0 | barrier | both cores | — | `barrier_sem` inc'd on all peers | `core_fwd` waits `barrier_sem >= N-1`, resets `0` |
| 1 | own slice send | `core_fwd` | `input_tensor` (`S_d`) | own output slot `d` (local) + `d+1` output slot `d` (fabric) | `(d+1).forward_sem += 1` |
| 1b | own slice send | `core_bwd` | `input_tensor` (`S_d`) | `d-1` output slot `d` (fabric) | `(d-1).backward_sem += 1` |
| 2 | forward relays | `core_fwd` | local `output` slots `{d-1…0}` (after `forward_sem` waits) | `d+1` output slots `{d-1…0}` (fabric) | `(d+1).forward_sem` counts up |
| 2b | backward relays | `core_bwd` | local `output` slots `{d+1…N-1}` (after `backward_sem` waits) | `d-1` output slots `{d+1…N-1}` (fabric) | `(d-1).backward_sem` counts up |
| 3 | drain/close + final wait + reset | both | `forward_sem`/`backward_sem` | — | `core_fwd` reader resets `forward_sem=0` after `>= d`; `core_bwd` reader resets `backward_sem=0` after `>= N-1-d` |

After phase 3 every device's output holds all `N` slots (identical full tensor) and all three semaphores are back to `0` (clean for the next cache hit).

## Validation

`validate()` raises typed `UnsupportedAxisValue` / `ExcludedCell` (from `ttnn.operations._op_contract`) for axis refusals, or `ValueError` for structural input errors:

| Condition | Error |
|-----------|-------|
| `input_tensor` not on a `MeshDevice` (or `< 2` devices) | reject ("input must be on a MeshDevice line of ≥2 devices") |
| mesh view not 2-D / not a `(1, N)` line | reject ("expected a 1-D mesh line") |
| `gather_dim` out of `-rank … rank-1` | reject ("gather_dim out of range") |
| input sharded / non-interleaved | reject ("sharded input not yet supported (interleaved only)") |
| per-shard page size not 16-byte aligned | reject ("page size must be 16-byte aligned") |
| `output_tensor` provided with spec != resolved output spec | reject ("output spec mismatch") |
| axis value outside SUPPORTED (dtype/layout/topology/gather_dim/alignment) | `UnsupportedAxisValue` |

**`gather_dim` canonicalization:** `validate()` canonicalizes to the NEGATIVE convention (`gd = gather_dim if gather_dim < 0 else gather_dim - rank`) BEFORE the SUPPORTED membership check, so a positive alias (`gather_dim = 0` ≡ `-rank`, the proven primary case) is not rejected by a literal membership test. `feature_spec.py` TARGET declares `gather_dim` as `[-4, -3, -2, -1]` (rank-4 shards: `-4` ≡ dim 0). The internal concat-stride computation converts the canonical negative dim back to a positive axis index for the stride table.

## Key Risks and Gotchas

- **Three semaphores, single-owner each — do not merge.** The two directions are independent counting streams; one counter per direction is required for unambiguous "which slot landed." The barrier is a third, pure-barrier semaphore to avoid the barrier↔counting reset race. Each is owned (waited + reset) by exactly one core per device.
- **Data-before-inc ordering is the correctness mechanism.** The counting `arm_inc` follows the slice data on the SAME 1-hop route, guaranteeing the slice landed before the increment is observed. The multicast barrier (different route) gives NO landing guarantee and must NOT be used to gate relays.
- **Semaphore reset order (cache-reuse footgun, `ccl_helpers_dataflow.hpp:74-77`).** Counting owner resets AFTER its final wait; `core_fwd` resets `barrier_sem` AFTER seeing `N-1`. Single-CQ ordering makes a launch's resets precede the next launch's incs. Missing resets = run 1 green, run 2 hangs.
- **Block-scope the barrier `MulticastIncChannel` before `arm_inc`.** Per the mandate footgun, destroy the barrier channel (end its block scope) before arming the counting `AtomicIncChannel`, so the pooled header is free — barrier phase strictly precedes the counting phase.
- **Create all three GlobalSemaphores ONCE.** Create + one `synchronize_device` per mesh, cache on the module keyed by `id(mesh_device)`, park on `mesh_program_descriptor.semaphores`. Never recreate per call; never add a per-call post-dispatch `synchronize_device`.
- **End devices have no fabric connection in one direction.** `d = N-1` (forward) and `d = 0` (backward) emit NO `setup_fabric_connection` block and construct NO `FabricStreamSender` in that direction — they only do local work + local sem waits. The host must gate the connection block on neighbor existence.
- **Fabric writes target the neighbor's OUTPUT at the mirrored address.** The output buffer address is identical across devices; the writer computes the local NOC address for `tile_id` and the fabric delivers it to the same NOC address on the neighbor chip. Requires the output `TensorAccessorArgs` (CT) + output base address (RT).
- **`ccl_packet_dims` owns bf16 sizing.** Never hardcode packet bytes; `num_tiles_per_packet = min(4, packet_size_bytes // tile_size_bytes)` (4 = `NocUnicastScatter` limit).
- **Acceptance topology is fixed.** The mesh MUST be opened as `(1, 8)` with `FabricConfig.FABRIC_1D`; any other shape hangs fabric init (`Fabric Router Sync: Timeout`) — a test/topology mismatch, not an op defect.
- **`gather_dim != 0` uses the interleaved row-stride walk.** Only `gather_dim = 0` (contiguous slice) is proven by the acceptance test; other dims are a refinement using the stride table above.

## Structural impossibilities (for feature_spec.py INVALID)

- `{dtype: bfloat8_b, layout: ROW_MAJOR_LAYOUT}` — `bfloat8_b` is a tiled block-float format with no row-major representation (single-tensor coupling; universe-must-change → INVALID). `topology`, `gather_dim`, and `alignment` are orthogonal to dtype/layout; the 16-byte page-size constraint is a shape×dtype `validate()` gate kept satisfiable by INPUTS, not an axis. This is the only structural impossibility.
