# Operation Design: all_gather

## Overview

| Field | Value |
|-------|-------|
| Classification | CCL (multi-device collective — pure data movement, no arithmetic) |
| Goal | Gather every device's shard of a mesh-sharded tensor and concatenate all shards along `gather_dim`, so that AFTER the op EVERY participating device on the 1-D line holds the full concatenated tensor (identical on every device). |
| Math | `output[d] = concat_{c=0..N-1}( input_shard[c], axis=gather_dim )` for every device `d` (identity gather; PCC ~1.0, bit-for-bit copy). |
| Mode | Hybrid — newly authored ring/line dataflow kernels under `ttnn/ttnn/operations/all_gather/kernels/`, assembled by a Python `ttnn.generic_op` + `ttnn.MeshProgramDescriptor`. Does NOT wrap, import, call, or dispatch to any existing `all_gather` / `all_gather_async` op. |
| References | Fabric egress helper: `ttnn/cpp/ttnn/kernel_lib/ccl_helpers_dataflow.hpp` (+ `.inl`). Python generic_op CCL structural template: `ttnn/ttnn/operations/point_to_point/point_to_point.py` + `..._program_descriptor.py`. Correctness reference for ring slice-walk / store-and-forward / concat addressing (READ-ONLY): `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/minimal_default_writer.cpp` + `minimal_default_reader.cpp`. Host CCL helpers: `ttnn/cpp/ttnn/operations/ccl/common/host/ccl_helpers_dataflow_host.hpp`. Semaphore-ownership gold standard: `ttnn/cpp/ttnn/operations/experimental/ccl/deepseek_moe_reduce_scatter/device/*`. |

This is a **multi-device** op: it builds a `ttnn.MeshProgramDescriptor` with **one `(MeshCoordinateRange, ProgramDescriptor)` entry per participating device** on the 1-D line. Every device runs the same bidirectional store-and-forward ring role (self-copy its own shard, receive from a neighbour, forward to the next hop). Distribution is op-owned.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | mesh-sharded along `gather_dim` across an N-device line; interleaved DRAM or L1; TILE (primary) or ROW_MAJOR; bfloat16 (primary) or float32 | — | — |
| `gather_dim` | `int` | yes | a dim of `input_tensor`; **canonicalized to NEGATIVE** (`gd = gather_dim if gather_dim < 0 else gather_dim - rank`) BEFORE the support check; **`gather_dim == 0` (≡ `-rank`, page-contiguous concat) is the proven primary case** | — | host-computed page addressing (kernels are dim-agnostic — see below) |
| `topology` | `ttnn.Topology` | no | `Linear` (primary) or `Ring` (noted extension) | `ttnn.Topology.Linear` | host route computation (`ccl_dm_route`) |
| `output_tensor` | `ttnn.Tensor \| None` | no | if given, spec must equal the resolved output spec (`shape[gd] *= N`, same dtype/layout/buffer_type) | `None` (op allocates) | — |

**`gather_dim` canonicalization (load-bearing).** The public entry point MUST canonicalize `gather_dim` to a single sign convention (negative) *before* the `SUPPORTED` membership test, so positive aliases (`gather_dim=0` ≡ `-rank`) are not rejected by a literal membership test. Implemented in `all_gather.py:101-103` (`_canonical_gather_dim`). `SUPPORTED["gather_dim"] = [-4]` today (primary path); TARGET is `[-4, -3, -2, -1]`.

**Kernels do not read `gather_dim`.** For the primary `gather_dim=0` path, the concat is page-contiguous, so the only place the dim enters is the host's output-page formula (`out_page(c, p) = c*P + p`). The kernels receive `pages_per_shard` and per-block page ranges; strided-concat (`gather_dim != 0`) is a host-addressing refinement (compute `tile_id_start = position * stride` with row wrapping, exactly as the reference writer `minimal_default_writer.cpp:247-256, 449-457`) that does not change the kernel structure.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | per-device shard `(B, C, H, W)` (rank ≤ 4). For `gather_dim=0` the shard's outermost dim concatenates. |
| Dtype | bfloat16 (primary), float32 |
| Layout | TILE (primary) or ROW_MAJOR |
| Memory | interleaved, DRAM or L1 |
| Device | `ttnn.MeshDevice` line view of shape `(1, N)`, `N ≥ 2` — one shard per device |

### Output

| Property | Value |
|----------|-------|
| Shape | input shape with `shape[gather_dim] *= N` (N = devices on the line). For `gather_dim=0`: `(N·B, C, H, W)`. |
| Dtype | = input |
| Layout | = input |
| Memory | = input (interleaved, same `buffer_type`) |
| Value | identical on every device = host-side concat of all N input shards along `gather_dim` |

The output is **fully overwritten** by the op (every output page is produced — block `c` by device `c`'s self-copy, all other blocks by fabric writes), so **no input-seeding / clone is required**; the op only `allocate_tensor_on_device`s the output spec (or validates a supplied `output_tensor`). The output buffer is persistent and exists before dispatch, so fabric writes that land "early" are correct (they target persistent DRAM).

## Dataflow Strategy

### Page-contiguous concat model (`gather_dim = 0`, primary)

The op is **pure byte movement** — like `point_to_point`, it never tilizes/untilizes and is format-agnostic: it copies physical pages (`buffer_page_size()` bytes, `buffer_num_pages()` pages) verbatim. For `gather_dim = 0` each device's shard maps to a **contiguous block of output pages**:

```
out_page(chip_c, local_page p) = c * P + p          P = pages_per_shard = input.buffer_num_pages()
```

The output tensor on every device is identical (block `c` always at `[c*P, (c+1)*P)`), and mesh-allocated interleaved tensors share a buffer address across devices, so a write of chip-`c`'s block targets the **same output page range on any device**. This makes the neighbour's destination address the local output address routed +1 hop.

### Ring store-and-forward on a line of N devices

Each device `i` runs **two worker cores**:

| Core | Logical coord | Direction | Fabric connection | Flow |
|------|---------------|-----------|-------------------|------|
| forward worker | `CoreCoord(0, 0)` | `direction = 0` | → neighbour `i+1` | carries low-index shards (`0..i`) rightward |
| backward worker | `CoreCoord(0, 1)` | `direction = 1` | → neighbour `i-1` | carries high-index shards (`i..N-1`) leftward |

Each core runs a **reader (NCRISC)** + **writer (BRISC)**. Bidirectional flow is required on a *line*: forward flow propagates shards rightward, backward flow propagates them leftward; together they deliver all N shards to every device. (For a *Ring*, one direction with wraparound suffices — a noted extension; the `ring_size` CT arg is a placeholder for the Ring modulo slice-walk.)

Per device (host-computed from the device coordinate `i`, `all_gather_program_descriptor.py:88-89`):

| Symbol | Meaning | Formula (line, chip id `i`, ring size `N`) |
|--------|---------|---------|
| `num_targets_fwd` | devices reachable forward (`i+1..N-1`) | `N - 1 - i` |
| `num_targets_bwd` | devices reachable backward (`0..i-1`) | `i` |
| `my_num_targets` | blocks this direction FORWARDS (seed + relays) | `direction==0 ? num_targets_fwd : num_targets_bwd` |
| `num_relay_blocks` | blocks this direction RECEIVES from its upstream | `direction==0 ? num_targets_bwd : num_targets_fwd` |

**Forward worker (direction 0, → `i+1`):**
1. **Reader (NCRISC):**
   - **Self-copy (ALWAYS, forward reader only):** read device `i`'s own input shard and write it verbatim into its OWN output block `i` (local NoC, via `cb_self_copy` scratch). This is the one write that never depends on a fabric connection, so it is placed on the forward reader which every device runs.
   - **Seed (only if `my_num_targets > 0`):** stage the P input pages into `cb_relay_pages` for the writer to fabric-forward one hop.
   - **Relay / store-and-forward (only if `my_num_targets > 0`):** for each of `num_relay_blocks` blocks arriving from the left (chips `i-1, i-2, …`), `noc_semaphore_wait_min` on the counting semaphore, then read the **landed block back out of local output DRAM** into `cb_relay_pages`. There is **no fabric receive** — the receive ingress is this local `noc_async_read` (the block was written directly into this device's persistent output DRAM by the upstream writer).
   - **Line-end (`my_num_targets == 0`):** pure receiver — just `noc_semaphore_wait_min(sem, num_relay_blocks)` to confirm all upstream blocks landed.
2. **Writer (BRISC):** `if (my_num_targets == 0) return;` (no fabric egress). Else fabric-`write_page` the seed block (chip `i`) to `i+1`'s output block `i`, then relay blocks `i-1, i-2, …, 0`, in FIFO order matching the reader's push order. One counting `inc` to `i+1`'s forward-core semaphore **per block** (after the block's P page-writes).

**Backward worker (direction 1, → `i-1`):** mirror image. Reader stages the seed and receives right-arrived blocks (`i+1..N-1`); writer relays them to `i-1`. No self-copy (the forward reader already did it). Gated on `num_targets_bwd > 0` (device `0` has no backward target).

**Store-and-forward contract (op-owned, NOT the helper).** A writer's fabric `write_page` lands a block **directly into the downstream device's persistent output DRAM** at the block's canonical page range. The downstream reader detects arrival via one counting `inc` per block (fabric in-order delivery guarantees the payload lands before the inc), reads the landed block back out of its own output DRAM, and its writer forwards it one more hop. The receive ingress, the ring slice-walk, and the concat addressing are all op-owned (helper banner `ccl_helpers_dataflow.hpp:89-93`).

### Tensix-to-Tensix contract (cross-device synchronization) — counting semaphore ONLY

There is **ONE op-internal `GlobalSemaphore`** (created once, parked on the descriptor). It is reserved on both worker cores of every device; each `(device, core)` has its own L1 counter instance at the shared address. Synchronization is **counting-only** — there is deliberately **no startup barrier** (see Key Risks for why this is correct and preferred):

1. **Sending half (helper-owned).** After each block's P page-writes, the upstream writer issues ONE `AtomicIncChannel::inc(neighbour_sem)` targeting the SAME-role core on the downstream neighbour (forward writer → neighbour's forward core; backward writer → neighbour's backward core). Fabric in-order delivery guarantees the block's payload lands before its inc.
2. **Waiting half (op-owned).** The downstream reader `noc_semaphore_wait_min(sem, running)` (incrementing `running` by 1 per expected block) before reading each landed block back. `running` maps 1:1 onto block arrival order.
3. **Cache-reuse re-arm (op-owned).** The reader `noc_semaphore_set(sem, 0)` after its last wait. The `GlobalSemaphore` is created with initial value 0 and reused across program-cache hits; the end-of-kernel reset is what makes call #2 correct.

**Per-`(device,core)` isolation.** A core's counter is incremented only by its immediate same-direction upstream (the forward core's sem only by forward flow, the backward core's only by backward flow — the writer's `target_noc_x/y` picks the neighbour's same-role core). Increment counts match exactly: device `i`'s forward writer issues `num_targets_bwd(i)+1 = i+1` incs; device `i+1`'s forward reader waits for `num_targets_bwd(i+1) = i+1`.

**Teardown:** every forwarding writer ends with `FabricStream::close()` (drains write + atomic barriers, then closes the connection); line-end writers early-return before opening any connection.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one shard block (P pages) per direction per ring step |
| Grid | per device: 2 worker cores — `forward = CoreCoord(0,0)`, `backward = CoreCoord(0,1)` (uniform across all devices, so peer-core NoC coords are identical mesh-wide; the counting inc targets the same logical core on the neighbour). One link, one worker per direction. |
| Per-core work | forward core: seed block `i` + `num_targets_bwd` relayed blocks; backward core: seed block `i` + `num_targets_fwd` relayed blocks. No per-link tile split. |
| Remainder | none at the core level (single worker per direction). Pages within a block loop 1-by-1; one counting `inc` per block (not per page). |

`num_targets_fwd`/`num_targets_bwd` are host-computed per device coordinate (`all_gather_program_descriptor.py:88-89`). Line ends have one direction with `my_num_targets == 0` (that direction's writer early-returns; that direction's reader is a pure receiver).

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|--------|----------|----------|----------|
| `cb_relay_pages` | 16 | `round_up(input.buffer_page_size(), l1_alignment)` | 2 (double-buffered streaming page; `total_size = 2 * aligned_page_size`) | input dtype | reader (NCRISC): seed pages read from input DRAM + relay pages read back from output DRAM | writer (BRISC): fabric `write_page` | whole kernel |
| `cb_self_copy` | 24 | `round_up(input.buffer_page_size(), l1_alignment)` | 2 (`total_size = 2 * aligned_page_size`) | input dtype | forward reader (NCRISC): input page → scratch | forward reader itself (scratch → local output write); no cross-kernel consumer | whole kernel |

Notes:
- One instance of each CB per worker core (same index on the forward and backward cores; each core has its own program-local CB). Declared on `both_set = CoreRange((0,0),(0,1))` (`all_gather_program_descriptor.py:101-114`).
- **`cb_relay_pages` sync invariant (push == pop).** When `my_num_targets > 0`, the reader pushes `P` seed pages + `num_relay_blocks × P` relay pages; the writer pops `(num_relay_blocks + 1) × P` — equal. When `my_num_targets == 0`, the reader takes the pure-receiver branch (pushes nothing) and the writer early-returns (pops nothing) — also balanced. **The seed push and relay read-back MUST be gated on `my_num_targets > 0`** (a line-end direction must not push pages its early-returning writer will never pop, or `cb_relay_pages` fills and the reader blocks).
- **`cb_self_copy` is scratch, deliberately unbalanced.** The forward reader `cb_reserve_back(cb_self_copy, 1)` once and reuses the write pointer across all P pages (read input page → local self-copy write). No cross-kernel consumer, so it is not push/pop balanced (never `cb_push_back`ed / `cb_wait_front`ed).
- No tilize/untilize, no compute CBs, no scaler — pure data movement. The compute (TRISC) kernel is intentionally absent (`all_gather_compute.cpp` documents the absence and is not wired into the descriptor).

## API Mapping

Every mechanism — helper or raw — has an exact file:line reference. The CCL fabric **dataflow kernel helper** (`ccl_helpers_dataflow.hpp` + `.inl`) is the primary building block for the writer's fabric egress; the raw NoC/semaphore calls are exactly the pieces the helper banner states it does NOT own.

### Kernel-side helper (fabric egress — writer)

| Phase | Type | Function | File:Line | Args / Template | Reads CB | Writes (target) | Manages own CB? |
|-------|------|----------|-----------|-----------------|----------|-----------------|------------------|
| Build egress | helper | `FabricStreamSender<>(conn_arg_idx, is_forward, alignment)` | hdr `ccl_helpers_dataflow.hpp:436` | `is_forward` = leading `has_forward` flag peeked at `conn_arg_idx`; `alignment` = `l1_alignment` CT arg | — | — | no |
| Open (bind route) | helper | `FabricStreamSender::open(unicast_route(num_hops))` → `FabricStream` | hdr `:448`, route `:260` | `num_hops` from `ccl_dm_route` (`= 1` for the immediate neighbour on a line) | — | — | no |
| Arm write | helper | `FabricStream::arm_unicast_write(page_size)` → `UnicastWriteChannel` | hdr `:374`, impl `.inl:23` | `page_size` (raw bytes; helper rounds up to `align(page_size, alignment)` on-wire — `.inl:34`) | — | — | no |
| Issue write | helper | `UnicastWriteChannel::write_page(src_l1, out_page_idx, output_accessor)` | hdr `:284`, impl `.inl:48` | `out_page_idx = c*P + p`; `output_accessor = TensorAccessor(output_args, output_addr, page_size)` (uniform mesh address → routes to neighbour) | `cb_relay_pages` (read ptr) | neighbour output DRAM page | no |
| Arm counting inc | helper | `FabricStream::arm_inc(1)` → `AtomicIncChannel` | hdr `:386`, impl `.inl:104` | invariant increment value `1` | — | — | no |
| Issue counting inc | helper | `AtomicIncChannel::inc(neighbour_sem_noc_addr)` | hdr `:317`, impl `.inl:121` | one per block: `neighbour_sem = safe_get_noc_addr(target_noc_x, target_noc_y, counting_sem_addr, 0)` | — | neighbour same-role counting sem | no |
| Close (drain + teardown) | helper | `FabricStream::close()` | hdr `:402`, impl `.inl:167` | drains NoC-write + atomic barriers, then closes; idempotent | — | — | n/a |
| (optional) coalesced write | helper | `arm_scatter_write(chunk_size, n)` / `ScatterWriteChannel::write_scatter` (≤4 pages/packet) | hdr `:381/300`, impl `.inl:58/81` | throughput optimization over per-page unicast | `cb_relay_pages` | neighbour output DRAM pages | no |

The op deliberately uses **no barrier channel** (`arm_multicast_inc`), so there is no shared-pooled-header block-scoping hazard (`arm_inc` and `arm_multicast_inc` would otherwise share one header — `ccl_helpers_dataflow.hpp:78-81`). See Key Risks for why the counting semaphore alone is sufficient.

### Raw APIs (op-owned data movement — the helper explicitly does NOT own these)

| Phase | Raw API | File:Line | Purpose | Helpers considered and rejected |
|-------|---------|-----------|---------|---------------------------------|
| Seed read + relay read-back (receive ingress) | `noc_async_read` / `noc_async_read_barrier` | `api/dataflow/dataflow_api.h`; kernel `all_gather_reader.cpp:72-77, 85-91` | read local input shard for the seed; read landed forward/backward blocks back out of local output DRAM for store-and-forward | **`FabricStream` / a receiver type** — rejected: the helper has NO receive type. `ccl_helpers_dataflow.hpp:71-73`: "The receive INGRESS is likewise a local NoC read the op owns; there is no FabricStreamReceiver." |
| Self-copy (own shard → own output block) | `noc_async_read` + `noc_async_write` / barriers | `api/dataflow/dataflow_api.h`; kernel `all_gather_reader.cpp:59-68` | place device `i`'s own block into its own output block `i` (intra-device, no fabric) | **`UnicastWriteChannel::write`** — rejected: that is a *fabric* (cross-device) write; the self-copy is intra-device. The helper is "PURE DATA MOVEMENT ... fabric egress" (`ccl_helpers_dataflow.hpp:17-18`), not local copies. |
| Counting WAIT half + reset | `noc_semaphore_wait_min`, `noc_semaphore_set` | `api/dataflow/dataflow_api.h`; kernel `all_gather_reader.cpp:84, 96, 101` | wait for cumulative counting target before each read-back; reset to 0 for cache re-arm | **None** — the helper owns only the *sending* half (atomic-inc). `ccl_helpers_dataflow.hpp:69-77`: "The WAITING half is a plain local `noc_semaphore_wait_min(sem, threshold)` the op calls directly ... each side must `noc_semaphore_set(sem, 0)` to re-arm." |
| Ring slice-walk + concat addressing | plain index arithmetic + `TensorAccessor::get_noc_addr` | kernel `all_gather_reader.cpp:82, 88` / `all_gather_writer.cpp:78-87`; reference `minimal_default_writer.cpp:404-457` | compute `c = i ± k` and `out_page = c*P + p` | **None** — `ccl_helpers_dataflow.hpp:89-93`: the helper does NOT own "ring slice-walk (chip_id +/- k mod ring_size), store-and-forward relay, concat-by-gather_dim output addressing, address generation (TensorAccessor ... is consumed, never re-wrapped)." |
| Input / output address generation | `TensorAccessor(TensorAccessorArgs<...>, addr, page_size)` | `tech_reports/tensor_accessor/tensor_accessor.md`; kernel `all_gather_reader.cpp:53-54`, `all_gather_writer.cpp:64` | per-page DRAM/L1 NoC addresses for input read, output read-back/write, and the fabric `write_page` dst | **None** — `TensorAccessor` is the address-gen primitive the helper *consumes* (`write_page`'s `AddrGen` template, `.inl:48-51`); building it is op-owned. |

### Host-side helpers (program-descriptor assembly, mirror `point_to_point_program_descriptor.py`)

| Phase | Function | File:Line | Purpose |
|-------|----------|-----------|---------|
| 1-D route (per direction; owns fwd/bwd sign + Ring short-way) | `ttnn._ttnn.fabric.ccl_dm_route(mesh_device, src_coord, dst_coord, topology)` → `{num_hops, is_forward, neighbor_id}` | host helper `ccl_helpers_dataflow_host.hpp:102-166`; call sites `all_gather_program_descriptor.py:179, 194` | forward route `coord_i → coord_{i+1}` and backward route `coord_i → coord_{i-1}`; gives `num_hops = 1`, `is_forward`, `neighbor_id` |
| Fabric connection RT args | `ttnn.setup_fabric_connection(src_fabric_id, neighbor_fabric_id, link_idx, program, core)` → `list[uint32]` (mutates `program`: appends `SemaphoreDescriptor`s) | host helper `ccl_helpers_dataflow_host.hpp:219-237`; wrapper `all_gather_program_descriptor.py:52-66` | per-writer fabric arg block, laid out `[has_forward][fwd args][has_backward][bwd args]`; appended ONLY on a forwarding writer (`my_num_targets > 0`) — `all_gather_program_descriptor.py:228-236` |
| Op-internal cross-device semaphore | `ttnn.create_global_semaphore(mesh_device, worker_cores, 0)` + `ttnn.get_global_semaphore_address(sem)` | host gold-standard `ccl_helpers_dataflow_host.hpp:250-256`; op `all_gather.py:88-98, 199-206` | created ONCE per `id(mesh_device)`, `ttnn.synchronize_device` ONCE after, cached, parked via `mpd.semaphores = [sem]`, address baked into RT args |
| (optional) packet framing | `ttnn._ttnn.fabric.ccl_packet_dims(dtype, page_size, num_pages, l1_alignment)` | host helper `ccl_helpers_dataflow_host.hpp:59-96` | available for multi-page coalescing (owns the bf16 `bit_floor`); the per-page primary path uses 1:1 page↔packet framing and the L1-aligned page size as on-wire payload, so it is not required today |
| Mesh assembly | `ttnn.MeshProgramDescriptor()`, `mpd[ttnn.MeshCoordinateRange(coord, coord)] = program`, `mpd.semaphores = [sem]`, `ttnn.generic_op([input, output], mpd)` | op `all_gather.py:199-207`; assembly `all_gather_program_descriptor.py:266-289` | one `ProgramDescriptor` per device coordinate on the line |

### Fabric-arg contract (load-bearing — the implementer must honour this offset)

The writer reads **7 scalar RT args** (`output_addr, pages_per_shard, page_size, num_hops, counting_sem_addr, target_noc_x, target_noc_y`), then the `[has_forward][fwd?][has_backward][bwd?]` fabric block starting at `conn_arg_idx = 7`. The leading `has_forward` flag is peeked as `dst_is_forward` and consumed by the `FabricStreamSender` ctor. This block is appended by `_append_fabric_rt_args` ONLY when `my_num_targets > 0` (a line-end writer early-returns before reading any RT arg). Mirrors `point_to_point_sender_writer.cpp:43-47`.

## Dataflow Phases (per device `i`, per direction worker)

| # | Phase | Owner | Consumes | Produces | State after |
|---|-------|-------|----------|----------|-------------|
| 0 | Allocate output; create/cache + park ONE GlobalSemaphore (+ one `synchronize_device`); compute per-device routes (`ccl_dm_route`), `num_targets_{fwd,bwd}`, peer-core NoC coords | host | input spec | `MeshProgramDescriptor` (one program per device), `sem` parked in `mpd.semaphores` | workload built; `sem == 0` on every core |
| 1 | Self-copy (forward reader only, ALL devices) | reader | local input shard pages | own output block `i` written (local NoC via `cb_self_copy`) | own output block `i` correct |
| 2 | Seed read (if `my_num_targets > 0`) | reader | local input shard pages | seed pages → `cb_relay_pages` | `cb_relay_pages` holds seed block `i` |
| 3 | Seed fabric-write (if `my_num_targets > 0`) | writer | `cb_relay_pages` seed pages | block `i` → neighbour output block `i`; one counting `inc` to neighbour | neighbour output block `i` landed; seed drained from CB |
| 4 | Relay receive/store (if `my_num_targets > 0`) | reader | counting sem incs + landed blocks in local output DRAM | for each upstream block: `wait_min(sem, running)` then read-back → `cb_relay_pages` | upstream blocks in local output + re-staged in CB |
| 5 | Relay fabric-write/forward (if `my_num_targets > 0`) | writer | `cb_relay_pages` relay pages | upstream blocks → neighbour output blocks; one counting `inc` per block | neighbour has all relayed blocks |
| 6 | Close + final reset | writer / reader | — | writer `stream.close()` (drain + close); reader `noc_semaphore_set(sem, 0)` | fabric quiesced; sem re-armed for the next program-cache run |
| 6′ | Line-end (`my_num_targets == 0`) | reader / writer | counting sem incs | reader `wait_min(sem, num_relay_blocks)` then reset; writer early-returns | all upstream blocks confirmed landed |

After phase 6 on all devices: every device's output == concat of all N shards along `gather_dim` (bit-for-bit; PCC = 1.0).

## Key Risks and Gotchas

- **ONE op-internal GlobalSemaphore, created once, parked (mandate W4 slot).** Create it once keyed on `id(mesh_device)`, `ttnn.synchronize_device` ONCE after creation, cache it, and `mpd.semaphores = [sem]` so the framework holds its L1 alive across program-cache hits. Do NOT re-create per call; do NOT add a per-call post-dispatch `synchronize_device` barrier. (`all_gather.py:88-98, 199-206`.)
- **No startup barrier is needed — do NOT add one.** Cross-device ordering is provided entirely by the counting semaphore: (1) fabric in-order delivery lands a block's payload before its inc; (2) the reader `wait_min` gives happens-before ordering; (3) the pre-allocated persistent output DRAM makes "early" fabric writes correct; (4) the `GlobalSemaphore` starts at 0 and the reader resets it at end-of-kernel, so program-cache call #2 starts clean. Omitting the barrier also SIDESTEPS the helper's shared-pooled-header footgun (`arm_multicast_inc` and `arm_inc` share one header and cannot be live at once — `ccl_helpers_dataflow.hpp:78-81`). This is verified: all 30 sim cases pass, **including the two-call program-cache test**, which is exactly the scenario a barrier would exist to protect. If a future refinement ever needs a hard startup fence, the mandate-compliant fallback is a *second* op-internal `GlobalSemaphore`, also created once and parked (never the shared-header multicast barrier).
- **Cache-reuse semaphore re-arm (footgun).** The reader MUST `noc_semaphore_set(sem, 0)` after its last wait (`all_gather_reader.cpp:101`). Without it call #1 passes and call #2 hangs/corrupts. The acceptance test's two-call program-cache case exercises this.
- **CB push/pop balance at line ends.** Gate the seed push and the relay read-back on `my_num_targets > 0` (the line-end direction does not forward, so its reader must NOT push pages its early-returning writer will never pop — else `cb_relay_pages` fills and the reader blocks). Push count MUST equal pop count per direction.
- **Edge-device fabric connections.** Device `0` opens NO backward connection; device `N-1` opens NO forward connection. The host calls `setup_fabric_connection` (and appends the fabric RT block) only when `my_num_targets > 0`; the writer early-returns before opening the `FabricStreamSender` when `my_num_targets == 0`. The line-end worker in the missing direction is a pure receiver. The **self-copy is done by the forward reader on every device** so it never depends on a missing connection.
- **Uniform mesh output address.** The fabric `write_page` uses the local `TensorAccessor(output)` base address as the neighbour's destination; this is correct only because mesh-allocated interleaved tensors share a buffer address across devices and the 1-hop fabric route directs the write to the neighbour. Verify the output is mesh-allocated interleaved with uniform addressing (true for `allocate_tensor_on_device` on a `MeshDevice`).
- **16-byte page alignment is load-bearing.** The fabric writer sends `align(page_size, l1_alignment)` bytes per page (`ccl_helpers_dataflow.inl:34`); the output `TensorAccessor` spaces pages by the raw `page_size`. A non-16-aligned `page_size` would let the rounded-up on-wire payload overrun into the next output page. `validate()` requires `page % 16 == 0` (`all_gather.py:138-140`), making the round-up a no-op.
- **Per-page payload must fit the fabric packet.** The per-page primary path sends one page (e.g. a bf16 tile = 2 KB) as one fabric packet (`arm_unicast_write(page_size)`). Typical tiles / RM sticks fit; for very large RM rows use the `ccl_packet_dims` coalescing/segmentation path.
- **Verification topology is fixed.** The op is verified on a simulated Wormhole T3K **line mesh `(1, 8)` with `fabric_config = ttnn.FabricConfig.FABRIC_1D`** via `scripts/run_multidevice_sim_pytest.py --op all_gather`. The acceptance test MUST open exactly `(1, 8)` with `FABRIC_1D`; a different mesh shape hangs fabric init (`Fabric Router Sync: Timeout`) — a test/topology mismatch, not an op defect. Proven first case: `gather_dim=0`, bfloat16, TILE_LAYOUT, Linear.

## Structural impossibilities (INVALID candidates for feature_spec.py)

- `{dtype: bfloat8_b, layout: ROW_MAJOR_LAYOUT}` — bfloat8_b is a tiled block-float format with no row-major representation (single-tensor coupling: both axes describe the input tensor; the data-format definition would have to change). This is the canonical INVALID cell, authored in `eval/golden_tests/all_gather/feature_spec.py` (already present).
