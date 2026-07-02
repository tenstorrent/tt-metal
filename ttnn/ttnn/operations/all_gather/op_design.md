# Operation Design: all_gather

## Overview

| Field | Value |
|-------|-------|
| Classification | CCL (multi-device collective — pure data movement, no arithmetic) |
| Goal | Gather every device's shard of a mesh-sharded tensor and concatenate all shards along `gather_dim`, so that AFTER the op every participating device on the 1-D line holds the full concatenated tensor (identical on every device). |
| Math | `output[d] = concat_{c=0..N-1}( input_shard[c], axis=gather_dim )` for every device `d` (identity gather; PCC ~1.0, no math). |
| Mode | Hybrid — newly authored ring/line dataflow kernels under `ttnn/ttnn/operations/all_gather/kernels/`, assembled by a Python `ttnn.generic_op` + `ttnn.MeshProgramDescriptor`. Does **NOT** wrap, import, call, re-export, or dispatch to any existing `all_gather`/`all_gather_async` op. |
| References | **Kernel helper (primary building block):** `ttnn/cpp/ttnn/kernel_lib/ccl_helpers_dataflow.hpp` (+ `.inl`). **Structural template (Python generic_op CCL):** `ttnn/ttnn/operations/point_to_point/point_to_point.py` + `point_to_point_program_descriptor.py`. **Correctness reference for the ring slice-walk / store-and-forward / barrier / counting (READ-ONLY, do not wrap):** `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/minimal_default_writer.cpp`. **Semaphore-ownership gold standard (parks two semaphores):** `ttnn/cpp/ttnn/operations/experimental/ccl/deepseek_moe_reduce_scatter/device/*`. **Routing structs:** `ttnn/cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp`. **Host CCL helpers:** `ttnn/cpp/ttnn/operations/ccl/common/host/ccl_helpers_dataflow_host.hpp`. |

This is a multi-device op: it builds a `ttnn.MeshProgramDescriptor` with **one `(MeshCoordinateRange, ProgramDescriptor)` entry per participating device** on the 1-D line. Every device runs the same bidirectional store-and-forward ring role (seed its own shard, receive from a neighbour, forward to the next hop). Distribution is op-owned.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | mesh-sharded along `gather_dim` across an N-device line; interleaved DRAM or L1; TILE (primary) or ROW_MAJOR; bfloat16 (primary) or float32 | — | tensor |
| `gather_dim` | `int` | yes | a dim of `input_tensor`; **canonicalized to negative** (`gd = gather_dim if gather_dim < 0 else gather_dim - rank`) BEFORE the support check; **`gather_dim == 0` is the proven primary case** | — | host → per-device kernel CT arg (canonical dim mapped to the kernel's 4-D slot) |
| `topology` | `ttnn.Topology` | no | `Linear` (primary) or `Ring` | `ttnn.Topology.Linear` | host (route computation) + kernel CT arg |
| `output_tensor` | `ttnn.Tensor \| None` | no | if given, spec must equal the resolved output spec | `None` (op allocates) | tensor |

**`gather_dim` canonicalization (mandatory):** the public entry point MUST canonicalize `gather_dim` to a single sign convention (negative) **before** the `SUPPORTED`/axis check, so a positive alias (`gather_dim=0 ≡ -rank` for a rank-4 shard) is not rejected by a literal membership test. `feature_spec.py` declares the axis negative (`gather_dim: [-4,-3,-2,-1]`); `-4 == gather_dim 0` for the rank-4 shards in `INPUTS` (the proven primary case). The kernels consume the normalized `0..3` form.

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
| Memory | = input (interleaved, same `buffer_type`) |
| Value | identical on every device = host-side concat of all N input shards along `gather_dim` |

The output is **fully overwritten** by the op — every output page is produced (block `c` by device `c`'s local self-copy, the other blocks by fabric writes), so **no input-seeding/`clone` is required** (unlike `point_to_point`, which seeds `output == input`). The op only `allocate_tensor_on_device`s the output spec (or validates a supplied `output_tensor`). The output buffer is persistent and exists before dispatch, so fabric writes that land "early" are correct (they land in allocated persistent DRAM).

## Dataflow Strategy

### Page-contiguous concat model (`gather_dim = 0`, primary)

The op is **pure byte movement** — like `point_to_point` it never tilizes/untilizes and is format-agnostic: it copies physical pages (`buffer_page_size()` bytes, `buffer_num_pages()` pages) verbatim. For `gather_dim = 0` (outermost), each device's shard maps to a **contiguous block of output pages**:

```
pages_per_shard = input.buffer_num_pages()
out_page(chip c, local page p) = c * pages_per_shard + p
```

The output tensor is identical on every device (block `c` always occupies `[c*pages_per_shard, (c+1)*pages_per_shard)`), so a write of chip-`c`'s block targets the **same output page range on any device**. This makes the neighbour's destination address the local output address routed +1 hop. (For `gather_dim != 0` the block is a strided page set computed exactly as the reference writer does — `tile_id_start = position * stride` with row wrapping by `output_tensor_Wt`; see `minimal_default_writer.cpp:247-256, 449-457`. Primary scope is `gather_dim=0` page-contiguous; the higher dims are the refinement path.)

### Bidirectional store-and-forward ring on a line of N devices

On a **line** (not a ring), a single direction cannot gather everything: rightward flow carries low-index shards to higher devices, but the highest shard never reaches lower devices. **Bidirectional flow is required for `Linear`.** (For `Ring`, one direction with wraparound suffices — Ring is the noted extension, selected via the topology CT arg and the reference's modulo slice-walk.)

Each device `i` runs **two worker cores**:

- **forward worker** — `CoreCoord(0,0)`, direction `0`, fabric connection → neighbour `i+1` (rightward).
- **backward worker** — `CoreCoord(0,1)`, direction `1`, fabric connection → neighbour `i-1` (leftward).

Host-computed per device coordinate `i` on a line of `N`:

| Symbol | Meaning | Formula |
|--------|---------|---------|
| `num_targets_forward` | devices reachable rightward (`i+1 .. N-1`) | `N - 1 - i` |
| `num_targets_backward` | devices reachable leftward (`0 .. i-1`) | `i` |

**Forward worker (direction 0, → `i+1`):**
1. **Reader (NCRISC):** reads device `i`'s local input shard into `cb_relay_pages` (the **seed**, always). If `num_targets_forward > 0`, then for each block arriving from the left (chips `i-1, i-2, …, 0`, in that order) it waits on the forward counting semaphore, reads the landed block **back out of its own output DRAM** into `cb_relay_pages` (store-and-forward), and pushes it.
2. **Writer (BRISC):** consumes the seed page and does the **local self-copy** into its own output block `i` (`noc_async_write`, always — even on the line end `i=N-1` that forwards nothing). If `num_targets_forward > 0`, it also fabric-`write_page`s the seed block to `i+1`'s output block `i`, then relays the left-arrived blocks (`i-1 … 0`) to `i+1`, issuing a counting `inc` to `i+1` every `chunks_per_sync` chunks.

**Backward worker (direction 1, → `i-1`):** mirror image, but **no self-copy** (the forward writer already did it). Reader reads the seed (only if `num_targets_backward > 0`) and receives right-arrived blocks (`i+1 … N-1`); writer relays them to `i-1` with counting incs. Device `0` has `num_targets_backward = 0` and its backward worker only participates in the barrier.

**Completeness:** shard `c` reaches device `d>c` via the forward chain `c→c+1→…→d`; reaches `d<c` via the backward chain `c→c-1→…→d`; and is placed on device `c` by the self-copy. Every device therefore holds every shard. ∎

**Store-and-forward contract (op-owned, NOT the helper):** a writer's fabric `write_page` lands a block **directly into the downstream device's persistent output DRAM** at the block's canonical page range. The downstream reader detects arrival via the counting semaphore, reads the landed block **back out of its own output DRAM** into `cb_relay_pages`, and its writer forwards it one more hop. There is **no `FabricStreamReceiver`** — the receive ingress is a local `noc_async_read` the op owns (helper banner `ccl_helpers_dataflow.hpp:69-74`).

### Tensix-to-Tensix contract (cross-device synchronization)

**TWO op-internal `GlobalSemaphore`s** (both parked on the descriptor), each reserved on both worker cores of every device:

- **`barrier_sem`** — the N-party startup barrier (`arm_multicast_inc` / `multicast_inc`).
- **`counting_sem`** — store-and-forward flow control (`arm_inc` / `inc`).

**Why two, not one (design decision).** The barrier and the counting phase both hit a per-`(device,core)` counter, and they overlap in time across devices: device `i-1` may issue a counting `inc` to device `i` (right after `i-1` clears its barrier and starts writing) **while device `i` is still in its barrier wait**. If both phases shared one counter, that early counting `inc` would push `i`'s counter past the `ring_size-1` barrier threshold, letting `i` fall through the barrier prematurely and then reset (`set 0`), losing the counting `inc` → hang/corruption. Separating the counters removes the race entirely. This mirrors the reference writer, which carries a distinct `barrier_sem` (RT 5) and `out_ready_sem` (RT 3) — `minimal_default_writer.cpp:105-134, 190-224` — and the gold-standard `deepseek_moe_reduce_scatter`, which parks **two** semaphores. Both semaphores satisfy the mandate's "created ONCE, parked, survive the cache" contract (created once per `mesh_device`, one `synchronize_device`, cached, `mpd.semaphores = [barrier_sem, counting_sem]`).

**Phase 1 — N-party startup barrier (`arm_multicast_inc`).** Each worker that has targets in its direction, before any payload write, multicast-increments `barrier_sem` on the peers it reaches (`line_multicast_route_info_t{ start_distance_in_hops = 1, range_hops = num_targets_<dir> }`), issuing **two** `multicast_inc`s — one to the forward-core sem NOC addr and one to the backward-core sem NOC addr of every reachable device (so every core of every reachable peer is signalled). Then **every** worker (targets or not) locally `noc_semaphore_wait_min(barrier_sem, ring_size - 1)` and `noc_semaphore_set(barrier_sem, 0)`. Each core receives exactly `N-1` incs: one from each `j<i` (via `j`'s forward worker) and one from each `j>i` (via `j`'s backward worker) → `i + (N-1-i) = N-1`. The `MulticastIncChannel` is **block-scoped to the barrier phase** (structural hygiene matching the reference; not strictly required for header-pool reasons in the current helper — see Key Risks).

**Phase 2 — incremental/counting (`arm_inc` / `inc`).** After the barrier reset, an upstream writer increments this device's same-direction `counting_sem` once per `chunks_per_sync` chunks; the local reader `noc_semaphore_wait_min(counting_sem, running_target)` before reading each landed block back, and `noc_semaphore_set(counting_sem, 0)` after its **last** wait (cache-reuse re-arm; helper banner `ccl_helpers_dataflow.hpp:75-77`).

**Teardown:** every writer ends with `stream.drain()` then `stream.close()` (both drain the NoC-write + atomic barriers; `close()` is idempotent — `ccl_helpers_dataflow.hpp:399-402`).

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one device's shard (all `pages_per_shard` pages), per direction |
| Grid | per device: **2 worker cores** — `forward = CoreCoord(0,0)`, `backward = CoreCoord(0,1)` (uniform across all devices, so peer-core virtual coords are identical mesh-wide) |
| Per-core work | forward core: seed block `i` (self-copy always; fabric-forward if `i<N-1`) + relay of the `num_targets_backward = i` left-arrived blocks. backward core: seed block `i` (if `i>0`) + relay of the `num_targets_forward = N-1-i` right-arrived blocks. One link, one worker per direction (no per-link tile split). |
| Remainder | none at the core level (single worker per direction). Within a slice the final chunk carries `pages_per_shard % chunks_per_sync` pages and triggers a final counting `inc` (`minimal_default_writer.cpp:335-340`). |

`chunks_per_sync` (host-computed flow-control granularity) defaults to the reference heuristic `min(max(pages_per_shard / pages_per_packet, 1), 160)`. The per-page primary path uses `pages_per_packet = 1` (one page = one fabric packet).

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|--------|----------|----------|----------|
| `cb_relay_pages` | 16 | `round_up(input.buffer_page_size(), l1_alignment)` | `2 * pages_per_packet` (double-buffered chunk; `= 2` for the per-page primary path) | input dtype | reader (NCRISC): seed pages from input DRAM + relayed pages read back from output DRAM | writer (BRISC): self-copy (forward core) + fabric `write_page` | whole kernel |

Notes:
- One `cb_relay_pages` instance **per worker core** (same index 16 on both the forward and backward cores; each core has its own program-local CB). Declare with `core_ranges = {forward, backward}`; the framework instantiates it per core.
- **CB sync invariant (push == pop, per direction):** the forward reader always pushes the seed (the forward writer always pops it for the self-copy, even when `num_targets_forward = 0`). Relay read-backs are pushed **only** when the worker forwards (`num_targets_<dir> > 0`), and popped by the writer's fabric relay. A line-end worker must **not** push relay pages its writer will never pop, or `cb_relay_pages` fills and the reader blocks. Verified balance: forward core pushes `1 + (num_targets_backward if i<N-1 else 0)`; writer pops the same.
- No tilize/untilize, no compute CBs, no scaler — pure data movement. There is no compute kernel; the two RISCs used are NCRISC (reader) and BRISC (writer) per core.

## API Mapping

Every mechanism — helper or raw — has an exact file:line reference. The CCL fabric **dataflow kernel helper** is the primary building block for the writer's fabric egress; the raw NoC/semaphore calls are exactly the pieces the helper banner states it does **NOT** own.

> **Current-header signature note (do not use the mandate's stale forms).** After the "simplify the dataflow API" refactor (HEAD `0897dd6f1d`), the stream's unicast route is bound **once at `open(route)`**, so `arm_unicast_write`, `arm_scatter_write`, and `arm_inc` take **no route** — only `arm_multicast_inc` carries its own multicast route (`ccl_helpers_dataflow.hpp:374,381,386,392`). Also, **each `arm_*` draws its own independent pooled header** (`payload_hdr_ / scatter_hdr_ / sem_hdr_ / mcast_hdr_`, `ccl_helpers_dataflow.hpp:413-416`; banner `@note` lines 78-81), so the barrier and counting channels do not share a header.

### Kernel-side helper (fabric egress — writer)

| Phase | Type | Function | File:Line | Args / Template | Reads CB | Writes (target) | Manages own CB? |
|-------|------|----------|-----------|-----------------|----------|-----------------|------------------|
| Build egress | helper | `FabricStreamSender<>(conn_arg_idx, is_forward, alignment)` | `ccl_helpers_dataflow.hpp:436` | `is_forward = (direction == 0)`; `alignment = l1_alignment` | — | — | no |
| Open (bind unicast route) | helper | `FabricStreamSender::open(route)` → `FabricStream` | `ccl_helpers_dataflow.hpp:448` | `route = unicast_route(1)` (immediate neighbour) | — | — | no |
| Barrier arm | helper | `FabricStream::arm_multicast_inc(mcast_route, 1)` → `MulticastIncChannel` | `ccl_helpers_dataflow.hpp:392` | `mcast_route = {start_distance_in_hops = 1, range_hops = num_targets_<dir>}` | — | — | no |
| Barrier issue | helper | `MulticastIncChannel::multicast_inc(peer_sem_noc_addr)` | `ccl_helpers_dataflow.hpp:335` | two calls: fwd-core `barrier_sem` NOC addr + bwd-core `barrier_sem` NOC addr | — | remote `barrier_sem` (multicast) | no |
| Counting arm | helper | `FabricStream::arm_inc(1)` → `AtomicIncChannel` | `ccl_helpers_dataflow.hpp:386` | route already bound at `open()` (1 hop); own pooled header (independent of the barrier's) | — | — | no |
| Counting issue | helper | `AtomicIncChannel::inc(downstream_counting_sem_noc_addr)` | `ccl_helpers_dataflow.hpp:317` | downstream same-direction `counting_sem` | — | remote `counting_sem` | no |
| Seed/relay write arm | helper | `FabricStream::arm_unicast_write(aligned_page_size)` → `UnicastWriteChannel` | `ccl_helpers_dataflow.hpp:374` | payload = aligned page bytes; route bound at `open()` | — | — | no |
| Seed/relay write issue | helper | `UnicastWriteChannel::write_page(src_l1, out_page_idx, output_accessor)` | `ccl_helpers_dataflow.hpp:284` | `out_page_idx = c*pages_per_shard + p`; `output_accessor = TensorAccessor(output)` (uniform mesh addr → routes +1 hop to neighbour) | `cb_relay_pages` (read ptr) | neighbour output DRAM page | no |
| (optional) coalesced write | helper | `FabricStream::arm_scatter_write(chunk_size, n≤4)` → `ScatterWriteChannel::write_scatter(dst[], n, src)` | `ccl_helpers_dataflow.hpp:381 / 300` | optimization over per-page unicast (≤4 pages/packet) | `cb_relay_pages` | neighbour output DRAM pages | no |
| Drain | helper | `FabricStream::drain()` | `ccl_helpers_dataflow.hpp:399` | NoC write barrier + atomic barrier | — | — | n/a |
| Close | helper | `FabricStream::close()` | `ccl_helpers_dataflow.hpp:402` | idempotent (RAII backstop at `:369`) | — | — | n/a |
| Route build | helper | `unicast_route(num_hops)` | `ccl_helpers_dataflow.hpp:260` | `num_hops = 1` | — | — | n/a |

The multicast route struct `line_multicast_route_info_t` (fields `start_distance_in_hops` / `range_hops` via unions, plus `e/w/n/s_num_hops`) is defined in `worker_routing_utils.hpp:23-31`; the kernel builds it from host-passed `(start_distance, range_hops)` runtime args, zeroing the 2-D mesh fields (valid on the 1-D LowLatency path, as `unicast_route()` zeroes `dst_mesh_id` — `ccl_helpers_dataflow.hpp:260-265`). `num_line_unicast_args = 2`, `num_line_multicast_args = 6` (`worker_routing_utils.hpp:37-38`).

### Raw APIs (op-owned data movement — the helper banner states it does NOT own these)

| Phase | Raw API | File:Line | Purpose | Helpers considered and rejected |
|-------|---------|-----------|---------|---------------------------------|
| Seed read + relay read-back (receive ingress) | `noc_async_read` / `noc_async_read_barrier` | `api/dataflow/dataflow_api.h` | read local input shard for the seed; read landed forward/backward blocks back out of local output DRAM for store-and-forward | **`FabricStream` / a receive channel** — rejected: the helper has NO receive type. `ccl_helpers_dataflow.hpp:74`: "The receive INGRESS is likewise a local NoC read the op owns; there is no FabricStreamReceiver." |
| Self-copy (own shard → own output block `i`) | `noc_async_write` / `noc_async_write_barrier` | `api/dataflow/dataflow_api.h` | place device `i`'s own block into its own output block `i` (intra-device, local NoC, no fabric) | **`UnicastWriteChannel::write`** — rejected: that is a *fabric* (cross-device) egress; the self-copy is intra-device. The helper is "PURE DATA MOVEMENT … fabric egress" (`ccl_helpers_dataflow.hpp:17-18`), not local copies. |
| Barrier / counting WAIT half + resets | `noc_semaphore_wait_min`, `noc_semaphore_set` | `api/dataflow/dataflow_api.h` | wait for barrier threshold (`ring_size-1`) / cumulative counting target; reset for cache re-arm | **None** — the helper owns only the *sending* half (atomic-inc). `ccl_helpers_dataflow.hpp:69-77`: "The WAITING half is a plain local `noc_semaphore_wait_min(sem, threshold)` the op calls directly … each side must `noc_semaphore_set(sem, 0)` to re-arm." |
| Ring slice-walk + concat addressing | index arithmetic + `TensorAccessor::get_noc_addr` | `minimal_default_writer.cpp:404-415` (reference), `worker_routing_utils.hpp` | compute `actual_slice_chip_id = i ± k (mod N)` and `out_page = c*pages_per_shard + p` | **None** — `ccl_helpers_dataflow.hpp:89-93`: "What the helper does NOT own (the op composes it): ring slice-walk (chip_id +/- k mod ring_size), store-and-forward relay, concat-by-gather_dim output addressing, address generation (TensorAccessor … is consumed, never re-wrapped)." |
| Input / output address generation | `TensorAccessor(TensorAccessorArgs<...>, base_addr, page_size)` | `tech_reports/tensor_accessor/tensor_accessor.md` | per-page NoC addresses for input read, output read-back, output self-copy write, and the fabric `write_page` dst | **None** — `TensorAccessor` is the address-gen primitive the helper *consumes* (`write_page`'s `AddrGen` template, `ccl_helpers_dataflow.hpp:283-284`); building it is op-owned. |

### Host-side helpers (program-descriptor assembly, mirror `point_to_point_program_descriptor.py`)

| Phase | Function | File:Line | Purpose |
|-------|----------|-----------|---------|
| 1-D route (per direction; owns fwd/bwd sign + ring short-way) | `ttnn._ttnn.fabric.ccl_dm_route(mesh_device, src_coord, dst_coord, topology)` → `{num_hops, is_forward, neighbor_id}` | `ttnn/cpp/ttnn-nanobind/fabric.cpp:253-266` | forward route `coord_i → coord_{i+1}` and backward route `coord_i → coord_{i-1}`; gives `num_hops = 1`, `is_forward`, `neighbor_id` |
| Fabric-connection RT args | `ttnn.setup_fabric_connection(src_fabric_id, neighbor_fabric_id, link_idx, program, core)` → `list[uint32]` (mutates `program`: appends `SemaphoreDescriptor`s) | `ttnn/cpp/ttnn-nanobind/fabric.cpp:141-178` | per-writer fabric arg block, laid out `[has_forward][fwd args][has_backward][bwd args]` exactly as `append_ccl_fabric_rt_args` (mirror `point_to_point_program_descriptor.py:64-78`) |
| Op-internal cross-device semaphores (×2) | `ttnn.create_global_semaphore(mesh_device, worker_cores, 0)` + `ttnn.get_global_semaphore_address(sem)` | `ttnn/cpp/ttnn-nanobind/global_semaphore.cpp:40-56, 58-67` | `barrier_sem` and `counting_sem`, each created ONCE per `mesh_device`, ONE `ttnn.synchronize_device` after both are created, cached, parked via `mpd.semaphores = [barrier_sem, counting_sem]`, addresses baked into RT args (mirror `point_to_point.py:88-98, 197-210`) |
| Mesh assembly | `ttnn.MeshProgramDescriptor()`, `mpd[ttnn.MeshCoordinateRange(coord, coord)] = program`, `mpd.semaphores = [...]`, `ttnn.generic_op([input, output], mpd)` | `ttnn/cpp/ttnn-nanobind/program_descriptors.cpp:990-1087` (MeshProgramDescriptor + `.semaphores` at 1077-1087); `point_to_point.py:200-213` | one `ProgramDescriptor` per device coordinate on the line |
| (optional) packet framing | `ttnn._ttnn.fabric.ccl_packet_dims(dtype, page_size, num_pages, l1_alignment)` | `ttnn/cpp/ttnn-nanobind/fabric.cpp:245-252` | available for multi-page coalescing (owns the bf16 `bit_floor`); the per-page primary path uses 1:1 page↔packet framing and the L1-aligned page size as on-wire payload, so it is not required |

## Dataflow Phases (per device `i`, per direction worker)

| # | Phase | Owner | Consumes | Produces | State after |
|---|-------|-------|----------|----------|-------------|
| 0 | Allocate output; create/cache + park `barrier_sem` & `counting_sem` (one `synchronize_device`); compute per-device routes (`ccl_dm_route`), `num_targets_{fwd,bwd}`, `chunks_per_sync` | host | input spec | `MeshProgramDescriptor` (one program/device), 2 sems parked | workload built |
| 1 | Startup barrier | writer | `barrier_sem` (multicast incs from all peers) | `barrier_sem` reaches `ring_size-1`, then reset to 0 | `barrier_sem == 0`; `MulticastIncChannel` destroyed |
| 2 | Seed read | reader | local input shard pages | seed pages → `cb_relay_pages` | `cb_relay_pages` holds seed |
| 3 | Seed self-copy + fabric-write | writer | `cb_relay_pages` seed pages | self-copy → own output block `i` (always); if `num_targets_<dir> > 0`: block `i` → neighbour output block `i` + counting `inc` | own output block `i` correct; neighbour block `i` landed; seed page popped |
| 4 | Relay receive (store) | reader | `counting_sem` (incs from upstream) + landed blocks in local output DRAM | landed upstream blocks → `cb_relay_pages` | upstream blocks in local output + re-staged in CB |
| 5 | Relay fabric-write (forward) | writer | `cb_relay_pages` relay pages | upstream blocks → neighbour output blocks + counting `inc`s | neighbour has all relayed blocks |
| 6 | Drain + close + final reset | writer / reader | — | writer `drain()` + `close()`; reader `noc_semaphore_set(counting_sem, 0)` | fabric quiesced; sems re-armed for the next program-cache run |

After phase 6 on all devices: every device's output == concat of all N shards along `gather_dim`.

## Key Risks and Gotchas

- **TWO op-internal `GlobalSemaphore`s, each created once, parked (mandate W4 slot).** Create `barrier_sem` and `counting_sem` once keyed on `id(mesh_device)`, `ttnn.synchronize_device` ONCE after both are created, cache them, and `mpd.semaphores = [barrier_sem, counting_sem]` so the framework holds their L1 alive across program-cache hits. Do NOT re-create per call; do NOT add a per-call post-dispatch `synchronize_device` barrier. Two are required to break the barrier↔counting race (see Dataflow Strategy); mandate-compliant per the `deepseek_moe_reduce_scatter` gold standard, which parks two.
- **Current-header API — no route on `arm_unicast_write`/`arm_inc`, independent pooled headers.** The route is bound once at `open(route)`; the barrier (`mcast_hdr_`) and counting (`sem_hdr_`) channels use independent pooled headers (`ccl_helpers_dataflow.hpp:413-416`, banner 78-81). Block-scoping the `MulticastIncChannel` to the barrier phase is therefore **structural hygiene** (matches the reference `minimal_default_writer.cpp:190-224`), **not** a header-sharing requirement as older docs stated. Still block-scope it — it costs nothing and keeps the barrier/counting phases textually separate.
- **Cache-reuse semaphore re-arm (footgun).** Programs are cached and the GlobalSemaphores reused. Each barrier writer resets `barrier_sem` to 0 *after its wait*; each counting reader resets `counting_sem` to 0 *after its last wait* (`ccl_helpers_dataflow.hpp:75-77`: "a SENDER resets BEFORE its outgoing inc, a RECEIVER after its wait"). Without these resets the first call passes and the second hangs/corrupts. The acceptance test's `test_all_gather_program_cache` (two calls) exercises this.
- **CB push/pop balance at line ends.** The forward reader always pushes the seed (the forward writer always pops it for the self-copy). Relay read-backs are pushed **only** when `num_targets_<dir> > 0`. A line-end worker must NOT push relay pages its writer will never pop, or `cb_relay_pages` fills and the reader blocks. Push count MUST equal pop count per direction.
- **Edge-device fabric connections.** Device `0` opens **no backward** connection; device `N-1` opens **no forward** connection (`num_targets_<dir> = 0`). The host calls `setup_fabric_connection` and the writer opens the `FabricStreamSender` / arms channels only when `num_targets_<dir> > 0` (mirror the `valid_targets` gating in `minimal_default_writer.cpp:190-224, 264-266`). A line-end worker in its missing direction is a pure receiver that still runs the local barrier wait+reset. The self-copy is done by the **forward writer on every device**, so it never depends on a missing connection.
- **Uniform mesh output address.** The fabric `write_page` uses the local `TensorAccessor(output)` base address as the neighbour's destination; correct only because mesh-allocated interleaved tensors share a buffer address across devices and the 1-hop fabric route directs the write to the neighbour. Verify the output is mesh-allocated interleaved with uniform addressing (true for `allocate_tensor_on_device` on a MeshDevice, and for a supplied replicated `output_tensor`).
- **Per-page payload must fit the fabric packet.** The per-page primary path sends one page (e.g. a bf16 tile = 2 KB) as one fabric packet via `arm_unicast_write(aligned_page_size)`; typical tiles/sticks fit. For very large RM rows use the `ccl_packet_dims` coalescing/segmentation path (optional `arm_scatter_write`, ≤4 pages/packet).
- **Verification topology is fixed.** Verified on a simulated Wormhole T3K **line mesh `(1, 8)` with `fabric_config = ttnn.FabricConfig.FABRIC_1D`** via `scripts/run_multidevice_sim_pytest.py --op all_gather`. The acceptance test MUST open exactly `(1, 8)` with `FABRIC_1D`; a different mesh shape hangs fabric init (`Fabric Router Sync: Timeout`) — a test/topology mismatch, not an op defect. Proven first case: `gather_dim=0`, bfloat16, TILE_LAYOUT, Linear.

## Structural impossibilities (INVALID candidates for feature_spec.py)

- `{dtype: bfloat8_b, layout: ROW_MAJOR_LAYOUT}` — bfloat8_b is a tiled block-float format with no row-major representation (single-tensor coupling: both axes describe the input tensor; the data-format definition would have to change). This is the canonical INVALID cell, authored in `eval/golden_tests/all_gather/feature_spec.py`.
