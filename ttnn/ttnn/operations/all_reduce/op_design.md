# Operation Design: all_reduce

## Overview

| Field | Value |
|-------|-------|
| Classification | CCL (multi-device collective) **WITH a compute stage** — cross-device data movement + element-wise arithmetic reduction (SUM). |
| Goal | Sum every device's shard element-wise across all N devices on a 1-D MeshDevice line, leaving the IDENTICAL sum on every device (same shape/dtype/layout as one input shard). |
| Math | `output[d][idx] = Σ_{c=0..N-1} input_shard[c][idx]` for **every** device `d` (the sum is identical on all devices). SUM reduction; values change (unlike the pure-movement CCLs). |
| Mode | Hybrid — newly authored fabric-dataflow **and** compute (TRISC) kernels under `ttnn/ttnn/operations/all_reduce/kernels/`, assembled by Python `ttnn.generic_op` + `ttnn.MeshProgramDescriptor`. Does NOT wrap, import, call, or dispatch to any existing `all_reduce`/`reduce_scatter`/`all_gather` op. |
| Algorithm | **Gather-then-reduce.** Phase A (fabric): a line store-and-forward gather lands all N shards into an op-internal `gather_buffer` (block `c` at pages `[c·P, (c+1)·P)`), identical on every device. Phase B (compute): a local element-wise N-way tile sum reduces the N blocks → the output shard. The two phases are two ordered `ttnn.generic_op` dispatches (same command queue → Phase A completes on each device before Phase B reads its `gather_buffer`). |
| References | Fabric egress helper: `ttnn/cpp/ttnn/kernel_lib/ccl_helpers_dataflow.hpp` (+ `.inl`). Structural template (Python generic_op CCL, line store-and-forward): `ttnn/ttnn/operations/all_gather/all_gather.py` + `all_gather_program_descriptor.py` + `kernels/all_gather_{reader,writer}.cpp` (READ-ONLY correctness reference). Host CCL helpers: `ttnn/cpp/ttnn/operations/ccl/common/host/ccl_helpers_dataflow_host.hpp`. Compute reduction reference (seed-then-accumulate into one DST): `models/demos/deepseek_v3_b1/unified_kernels/reduce_to_one_b1.hpp:600-617`, `ttnn/cpp/ttnn/operations/experimental/deepseek/mla/matmul_wo/device/kernels/compute_collector.cpp:39-50`. Semaphore-ownership gold standard: `ttnn/cpp/ttnn/operations/experimental/ccl/deepseek_moe_reduce_scatter/device/*`. |

This is a multi-device op: **each phase builds a `ttnn.MeshProgramDescriptor` with one `(MeshCoordinateRange, ProgramDescriptor)` entry per participating device** on the 1-D line. Distribution is op-owned.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | mesh-sharded across an N-device line `(1, N)`; each device holds one SAME-shape shard; interleaved DRAM or L1; TILE_LAYOUT; bfloat16 (primary) or float32 | — | — |
| `topology` | `ttnn.Topology` | no | `Linear` (primary) | `ttnn.Topology.Linear` | host (route computation) + gather kernels' CT arg |
| `output_tensor` | `ttnn.Tensor \| None` | no | if given, spec must equal the input shard spec (same shape/dtype/layout/buffer_type) | `None` (op allocates) | — |

There is **no reduce-dim parameter** — the reduction is always the full element-wise sum across devices; the output has the same shape as one input shard. No `dim` to canonicalize.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | per-device shard `(B, C, H, W)` (rank ≤ 4), SAME shape on every device |
| Dtype | bfloat16 (primary), float32 |
| Layout | TILE_LAYOUT (the reduction is a tile compute) |
| Memory | interleaved, DRAM or L1 |
| Device | `ttnn.MeshDevice` line view of shape `(1, N)` — one shard per device |

### Output

| Property | Value |
|----------|-------|
| Shape | = input shard shape (unchanged) |
| Dtype | = input |
| Layout | = input (TILE_LAYOUT) |
| Memory | = input (interleaved, same buffer_type) |
| Value | identical on every device = host-side element-wise SUM of all N input shards |

### Intermediate — `gather_buffer` (op-internal)

| Property | Value |
|----------|-------|
| Shape | `(N·B, C, H, W)` — N shard-blocks stacked on the outer dim (block `c` = pages `[c·P, (c+1)·P)`, `P = input.buffer_num_pages()`) |
| Dtype / Layout / Memory | = input (TILE_LAYOUT, interleaved, same buffer_type) |
| Lifetime | allocated per call via `allocate_tensor_on_device` on the mesh (replicated shape — each device owns an `N·P`-page buffer); written by Phase A, read by Phase B |

`gather_buffer` is mesh-allocated interleaved, so its buffer address is uniform across devices — that is what lets the Phase-A fabric `write_page` target a neighbor's block using the LOCAL accessor base address routed one hop (see Dataflow Strategy).

## Dataflow Strategy

Two ordered phases, each a `ttnn.MeshProgramDescriptor` dispatched via `ttnn.generic_op`. Because both dispatches share the device command queue, Phase A completes on device `i` before Phase B runs on device `i`; and Phase A's store-and-forward readers block (`noc_semaphore_wait_min`) until every block destined for device `i` has landed, so `gather_buffer` on device `i` is fully populated before Phase B reads it. No extra cross-device barrier is needed between the phases.

### Phase A — line store-and-forward gather (fabric, no arithmetic)

Page-contiguous gather (the all_gather `gather_dim=0` pattern) into `gather_buffer`. Pure byte movement — never tilizes/untilizes; copies physical pages verbatim. Block mapping (identical on every device):

```
gb_page(block c, local page p) = c * P + p           # P = input.buffer_num_pages()
```

Each device `i` runs **two worker cores** — a **forward worker** (rightward flow → neighbor `i+1`) and a **backward worker** (leftward flow → neighbor `i-1`). Bidirectional flow is required on a line: forward flow carries low-index blocks rightward, backward flow carries high-index blocks leftward.

Per device `i` (host-computed):

| Symbol | Meaning | Formula (line, chip id `i`, size `N`) |
|--------|---------|---------------------------------------|
| `num_targets_fwd` | devices reachable rightward (`i+1..N-1`) | `N - 1 - i` |
| `num_targets_bwd` | devices reachable leftward (`0..i-1`) | `i` |

**Forward worker (direction=0, fabric connection → `i+1`):**
1. **Reader (NCRISC):** (a) reads device `i`'s local input shard and writes it verbatim into its OWN `gather_buffer` block `i` (the **self-copy**, local NoC, always); (b) if `num_targets_fwd > 0`, stages the seed pages into `cb_relay_pages`; then, for each forward-flow block arriving from the left (`i-1, i-2, …`), waits on the counting semaphore and reads the landed block back out of local `gather_buffer` into `cb_relay_pages` (store-and-forward). If `num_targets_fwd == 0` (line end `N-1`), it is a pure receiver: it only waits on the counting semaphore to confirm arrivals.
2. **Writer (BRISC):** fabric-writes the seed block (`i`) to `i+1`'s `gather_buffer` block `i`, then relays the left-arrived blocks to `i+1`'s blocks; increments `i+1`'s forward counting semaphore once per landed block. Gated on `num_targets_fwd > 0`.

**Backward worker (direction=1, fabric connection → `i-1`):** mirror image; reader stages the seed (no self-copy — the forward reader already did it), receives right-arrived blocks (`i+1..N-1`), reads them back, and the writer relays them to `i-1`. Gated on `num_targets_bwd > 0`.

**Store-and-forward contract (op-owned, NOT the helper):** a writer's fabric `write_page` lands a block directly into the downstream device's persistent `gather_buffer` at the block's canonical page range. The downstream reader detects arrival via the counting semaphore, reads the landed block back out of its own `gather_buffer`, and its writer forwards it one more hop. There is **no FabricStreamReceiver** — the receive ingress is a local `noc_async_read` the op owns (`ccl_helpers_dataflow.hpp:74`, `:89-93`).

**Cross-device synchronization (Tensix-to-Tensix contract):** ONE op-internal `GlobalSemaphore` (parked on the Phase-A descriptor) is the per-`(device, core)` counting semaphore. The **sending** half is owned by the helper (`AtomicIncChannel::inc` — the upstream writer increments this device's same-direction counter once per landed block). The **waiting** half is op-owned: the local reader `noc_semaphore_wait_min(sem, running_target)` before reading each landed block back, and `noc_semaphore_set(sem, 0)` at the very end to re-arm for the next (cached) program run. Increment-after-write ordering on the fabric connection guarantees the block's data has landed before its counting `inc` arrives (`ccl_helpers_dataflow.hpp:66-77`). This mirrors the proven `all_gather_{reader,writer}.cpp` — a barrier phase is NOT used (the persistent buffer + counting semaphore + cache re-arm suffice).

### Phase B — local element-wise N-way tile sum (compute, no fabric)

Pure local compute per device; no fabric, no cross-device semaphore. The output tile at position `i` is the element-wise sum of the N blocks' tile `i`:

```
output_tile[i] = Σ_{c=0..N-1} gather_buffer_tile[c * P + i]      for i in [0, P)
```

Each core owns a contiguous range of the `P` output-tile positions (`split_work_to_cores`). For each owned position `i`:
- **Reader (NCRISC):** reads the N tiles `gather_buffer[c·P + i]` (c=0..N-1) into `cb_gathered_shards` (pushes N tiles in block order).
- **Compute (TRISC):** seeds `DST[0]` with block 0's tile (`copy_tile`), then accumulates blocks 1..N-1 into `DST[0]` one tile at a time (`binary_dest_reuse_tiles<ELWADD, DEST_TO_SRCA>`), packs the single result tile to `cb_reduced`. Uses only `DST[0]` — DST-capacity-safe for any N (independent of the f32 4-tile / bf16 8-tile DST limit).
- **Writer (BRISC):** writes the reduced tile to `output` page `i`.

`gather_buffer` and `output` are TILE_LAYOUT, so Phase B reads/writes tiles directly — no tilize/untilize.

## Work Distribution

### Phase A (gather)

| Field | Value |
|-------|-------|
| Work unit | one device's shard (all `P` pages), per direction |
| Grid | per device: 2 worker cores — `forward = CoreCoord(0,0)`, `backward = CoreCoord(0,1)` (uniform across all devices → peer core NoC coords identical mesh-wide) |
| Per-core work | forward core: seed block `i` + `num_targets_bwd` relayed blocks; backward core: seed block `i` + `num_targets_fwd` relayed blocks (one link, one worker per direction) |
| Remainder | none at the core level (single worker per direction); line ends (`0` backward, `N-1` forward) open no connection and are pure receivers |

### Phase B (reduce)

| Field | Value |
|-------|-------|
| Work unit | one output tile position `i` (sum of N gather blocks' tile `i`) |
| Grid | `ttnn.split_work_to_cores(mesh_device.compute_with_storage_grid_size(), P)` → `(num_cores, all_cores, core_group_1, core_group_2, tiles_per_core_g1, tiles_per_core_g2)` |
| Per-core work | `core_group_1` cores handle `tiles_per_core_g1` positions each; `core_group_2` cores handle `tiles_per_core_g2` (the two-group split absorbs the remainder when `P` does not divide evenly) |
| Remainder | handled by the standard two-core-group split from `split_work_to_cores`; each core is given its `[start_tile, start_tile + n)` range via runtime args |

## Circular Buffers

### Phase A (gather) — per worker core

| Semantic Name | Index | Page Size | Num Pages | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|--------|----------|----------|----------|
| `cb_relay_pages` | 16 | `round_up(input.buffer_page_size(), l1_alignment)` | `2` (double-buffered single-page chunk) | input dtype | reader (NCRISC): seed pages from input DRAM + relayed pages read back from `gather_buffer` | writer (BRISC): fabric `write_page` | whole Phase-A kernel |
| `cb_self_copy` | 24 | `round_up(input.buffer_page_size(), l1_alignment)` | `2` (double buffer) | input dtype | forward reader (NCRISC): local self-copy scratch (input → own `gather_buffer` block `i`) | forward reader (same core, local `noc_async_write`) | whole Phase-A kernel |

### Phase B (reduce) — per compute core

| Semantic Name | Index | Page Size | Num Pages | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|--------|----------|----------|----------|
| `cb_gathered_shards` | 0 | `tile_size(input.dtype)` | `2 * N` (double-buffered block of the N shard tiles for one position) | input dtype | reduce reader (NCRISC): N tiles `gather_buffer[c·P+i]` | reduce compute (TRISC) | whole Phase-B kernel |
| `cb_reduced` | 16 | `tile_size(output.dtype)` | `2` (double buffer) | output dtype | reduce compute (TRISC): `pack_tile` of the summed tile | reduce writer (BRISC) | whole Phase-B kernel |

**CB sync invariants:**
- Phase A: every page the reader `cb_push_back`es to `cb_relay_pages` is `cb_wait_front`+`cb_pop_front`ed by the writer. Gate the seed push and relay read-back on `num_targets_<dir> > 0` — the line-end device in a direction does not forward, so its reader must NOT push pages its writer will never pop (else the CB fills and the reader blocks). Push count == pop count per direction.
- Phase B: reader pushes `N` tiles per position; compute `cb_wait_front(cb_gathered_shards, N)` then `cb_pop_front(..., N)` (balanced). Compute pushes `1` to `cb_reduced`; writer waits `1`, pops `1` (balanced). All `cb_wait_front` on `cb_gathered_shards` use the same count `N`.

## API Mapping

Every mechanism — helper or raw — has an exact file:line reference.

### Phase A — kernel-side fabric egress helper (writer)

| Phase | Type | Function | File:Line | Args / Template | Reads CB | Writes (target) | Manages own CB? |
|-------|------|----------|-----------|-----------------|----------|-----------------|------------------|
| Build egress | helper | `FabricStreamSender<>(conn_arg_idx, is_forward, alignment)` | `ccl_helpers_dataflow.hpp:436` (class `:425`) | `is_forward` = leading `has_forward` flag of the fabric arg block; `alignment = l1_alignment` | — | — | no |
| Open | helper | `FabricStreamSender::open(route)` → `FabricStream` | `ccl_helpers_dataflow.hpp:448` | `route = unicast_route(num_hops)` (num_hops from `ccl_dm_route`, =1 to immediate neighbor) | — | — | no |
| Route build | helper | `unicast_route(num_hops)` | `ccl_helpers_dataflow.hpp:260` | `num_hops = 1` | — | — | n/a |
| Write arm | helper | `FabricStream::arm_unicast_write(page_size)` → `UnicastWriteChannel` | `ccl_helpers_dataflow.hpp:374` | `page_size` = aligned page bytes (invariant per-page payload) | — | — | no |
| Write issue | helper | `UnicastWriteChannel::write_page(src_l1, out_page_idx, output_accessor)` | `ccl_helpers_dataflow.hpp:283` (class `:276`) | `out_page_idx = c·P + p`; `output_accessor = TensorAccessor(gather_buffer)` (uniform mesh addr → routes to neighbor) | `cb_relay_pages` (read ptr) | neighbor `gather_buffer` DRAM page | no |
| Counting arm | helper | `FabricStream::arm_inc(1)` → `AtomicIncChannel` | `ccl_helpers_dataflow.hpp:386` | invariant inc value = 1 | — | — | no |
| Counting issue | helper | `AtomicIncChannel::inc(downstream_sem_noc_addr)` | `ccl_helpers_dataflow.hpp:317` (class `:311`) | downstream same-direction counting sem NoC addr (`safe_get_noc_addr(peer_noc_x, peer_noc_y, sem_addr, 0)`) | — | remote counting sem | no |
| Drain | helper | `FabricStream::drain()` | `ccl_helpers_dataflow.hpp:399` | NoC write barrier + atomic barrier | — | — | n/a |
| Close | helper | `FabricStream::close()` | `ccl_helpers_dataflow.hpp:402` | idempotent (RAII backstop) | — | — | n/a |

### Phase A — raw APIs (op-owned; the helper explicitly does NOT own these)

| Phase | Raw API | File:Line | Purpose | Helpers considered and rejected |
|-------|---------|-----------|---------|---------------------------------|
| Receive ingress + relay read-back + seed read | `noc_async_read` / `noc_async_read_barrier` | `api/dataflow/dataflow_api.h` | read local input shard; read landed blocks back out of local `gather_buffer` for store-and-forward | **`FabricStream`/`FabricStreamReceiver`** — rejected: the helper has NO receive type; `ccl_helpers_dataflow.hpp:74` "there is no FabricStreamReceiver ... the receive INGRESS is likewise a local NoC read the op owns." |
| Self-copy (own shard → own `gather_buffer` block) | `noc_async_write` / `noc_async_write_barrier` | `api/dataflow/dataflow_api.h` | place device `i`'s own block into its own `gather_buffer` block `i` (intra-device, no fabric) | **`UnicastWriteChannel::write`** — rejected: that is a *fabric* cross-device write; the self-copy is intra-device. Helper is "PURE DATA MOVEMENT ... fabric egress" (`ccl_helpers_dataflow.hpp:7-18`). |
| Counting WAIT half + re-arm | `noc_semaphore_wait_min`, `noc_semaphore_set` | `api/dataflow/dataflow_api.h` | wait cumulative counting target; reset for cache re-arm | **None** — helper owns only the *sending* half. `ccl_helpers_dataflow.hpp:70-77`: "The WAITING half is a plain local `noc_semaphore_wait_min` ... each side must `noc_semaphore_set(sem, 0)` to re-arm." |
| Ring slice-walk + block addressing | index arithmetic + `TensorAccessor(...).get_noc_addr(page)` | `tech_reports/tensor_accessor/tensor_accessor.md`; ref `all_gather_reader.cpp` | `c = i ∓ k` block walk; `gb_page = c·P + p` | **None** — `ccl_helpers_dataflow.hpp:89-93`: "What the helper does NOT own: ring slice-walk ... concat-by-gather_dim output addressing, address generation (TensorAccessor is consumed, never re-wrapped)." |

### Phase B — compute reduction (RAW compute API — justified below)

| Phase | Type | Function | File:Line | Args / Template | Reads CB | Writes | Notes |
|-------|------|----------|-----------|-----------------|----------|--------|-------|
| Boot init | raw | `binary_op_init_common(icb0, icb1, ocb)` | `api/compute/eltwise_binary.h:31` | `(cb_gathered_shards, cb_gathered_shards, cb_reduced)` | — | — | once at kernel start |
| Seed init | raw | `copy_tile_to_dst_init_short(cb)` | `api/compute/tile_move_copy.h:32` | `cb_gathered_shards` | — | — | before each seed |
| Seed | raw | `copy_tile(in_cb, in_tile, dst)` | `api/compute/tile_move_copy.h:103` | `(cb_gathered_shards, 0, 0)` → `DST[0] = block0 tile` | `cb_gathered_shards` | DST[0] | — |
| Accumulate init | raw | `binary_dest_reuse_tiles_init<ELWADD, DEST_TO_SRCA>(cb)` | `api/compute/eltwise_binary.h:250` | `cb_gathered_shards` | — | — | once after the seed, before the accumulate loop |
| Accumulate | raw | `binary_dest_reuse_tiles<ELWADD, DEST_TO_SRCA>(in_cb, in_tile, dst)` | `api/compute/eltwise_binary.h:289` | `(cb_gathered_shards, c, 0)` for c=1..N-1 → `DST[0] += block_c tile` | `cb_gathered_shards` | DST[0] | DST_TO_SRCA pulls DST[0] into SrcA so the add accumulates |
| DST mgmt | raw | `tile_regs_acquire` / `tile_regs_commit` / `tile_regs_wait` / `tile_regs_release` | `api/compute/reg_api.h` | — | — | — | one DST reg (DST[0]) |
| Pack | raw | `pack_tile(dst, cb)` | `api/compute/pack.h` | `(0, cb_reduced)` | — | `cb_reduced` | one tile per position |

**Helpers considered and rejected (Phase B compute reduction):**
1. **`compute_kernel_lib::reduce()` + `Accumulate`** (`ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp:378`). Rejected: `reduce()` COLLAPSES the within-tile 32×32 dims — REDUCE_ROW reduces W, REDUCE_COL reduces H, REDUCE_SCALAR reduces both (`reduce_helpers_compute.hpp:15-18`, `:264-268`), producing FEWER tiles. all_reduce needs an element-wise sum of N tiles at matching positions that PRESERVES the 32×32 tile structure — a fundamentally different operation. The `Accumulate` wrapper (`:151-198`) accumulates already-collapsed reduce RESULTS across successive `reduce()` calls (it reloads a collapsed accumulator via `copy_tile`), so it also cannot produce `out[r][c] = Σ a_k[r][c]` for full tiles.
2. **`eltwise_convenience.hpp` / `eltwise_chain.hpp` one-liners (`add<...>`, `binary_sfpu<...>`, `eltwise_chain(...)`).** Rejected: these headers **do not exist in this tree** — a repo-wide search finds them only referenced in agent planning docs (`tt_metal/third_party/tt_ops_code_gen/agents/*.md`), never as real headers or symbols. No `eltwise_chain(...)`/`add<...>()` helper definition exists anywhere in the source. Therefore the element-wise add MUST use the raw compute API. (Verified: `find`/`grep` across the whole repo — the only `sfpu_eltwise_chain` hit is an unrelated standalone programming example.)
3. **`tilize`/`untilize` helpers.** N/A — input and output are already TILE_LAYOUT; no format conversion.

The raw seed-then-accumulate-into-one-DST idiom is proven in production: `models/demos/deepseek_v3_b1/unified_kernels/reduce_to_one_b1.hpp:600-617` (explicit `copy_tile` seed + `binary_dest_reuse_tiles<ELWADD, DEST_TO_SRCA>` accumulate loop) and `.../mla/matmul_wo/device/kernels/compute_collector.cpp:39-50` (single-`DST[0]` accumulator). Using one DST reg is DST-capacity-safe for any N and for float32 (DST=4 tiles).

### Phase B — reader/writer raw APIs

| Phase | Raw API | File:Line | Purpose | Helpers considered and rejected |
|-------|---------|-----------|---------|---------------------------------|
| Read N shard tiles | `noc_async_read` + `TensorAccessor(gather_buffer).get_noc_addr(c·P+i)` | `api/dataflow/dataflow_api.h`; `tensor_accessor.md` | stream the N blocks' tile `i` into `cb_gathered_shards` | **`tilize_helpers` (dataflow)** — N/A: `gather_buffer` is already TILE_LAYOUT; a straight tiled page read suffices. |
| Write reduced tile | `noc_async_write` + `TensorAccessor(output).get_noc_addr(i)` | `api/dataflow/dataflow_api.h`; `tensor_accessor.md` | write `cb_reduced` tile to output page `i` | **`untilize_helpers`** — N/A: output is TILE_LAYOUT; a straight tiled page write suffices. |

### Host-side helpers (program-descriptor assembly, mirror `all_gather_program_descriptor.py`)

| Phase | Function | File:Line | Purpose |
|-------|----------|-----------|---------|
| 1-D route (per direction; owns fwd/bwd sign reversal + ring short-way) | `ttnn._ttnn.fabric.ccl_dm_route(mesh_device, src_coord, dst_coord, topology)` → `{num_hops, is_forward, neighbor_id}` | `ttnn/cpp/ttnn-nanobind/fabric.cpp:254-266` (impl `ccl_helpers_dataflow_host.hpp:137-166`) | forward route `i→i+1`, backward route `i→i-1`; gives `num_hops=1`, `is_forward`, `neighbor_id` |
| Fabric connection RT args (mutates `program`: appends `SemaphoreDescriptor`s) | `ttnn.setup_fabric_connection(src_fabric_id, neighbor_fabric_id, link_idx, program, core)` → `list[uint32]` | `ttnn/cpp/ttnn-nanobind/fabric.cpp:142-177` | per-writer fabric arg block; the op wraps it to lay out `[has_forward][fwd args][has_backward][bwd args]` exactly as `append_ccl_fabric_rt_args` (`ccl_helpers_dataflow_host.hpp:219-237`; mirror `all_gather_program_descriptor.py:52-66`) |
| Op-internal cross-device semaphore (Phase A only) | `ttnn.create_global_semaphore(mesh_device, worker_cores, 0)` + `ttnn.get_global_semaphore_address(sem)` | `ttnn/cpp/ttnn-nanobind/global_semaphore.cpp` | created ONCE per `id(mesh_device)`, `ttnn.synchronize_device` ONCE after, cached, parked via `gather_mpd.semaphores = [sem]`, address baked into Phase-A RT args (mirror `all_gather.py:83-98, 199-204`) |
| Reduce work split | `ttnn.split_work_to_cores(grid_size, P)` | `ttnn/ttnn/core.py:19` | Phase-B per-core output-tile ranges |
| Mesh assembly | `ttnn.MeshProgramDescriptor()`, `mpd[ttnn.MeshCoordinateRange(coord, coord)] = program`, `mpd.semaphores = [sem]`, `ttnn.generic_op([tensors], mpd)` | `ttnn/cpp/ttnn-nanobind/program_descriptors.cpp`; `all_gather.py:200-206` | one `ProgramDescriptor` per device coordinate, per phase |
| Packet framing (not required) | `ttnn._ttnn.fabric.ccl_packet_dims(...)` | `ttnn/cpp/ttnn-nanobind/fabric.cpp:246` | available for multi-page coalescing; the per-page primary path uses 1:1 page↔packet framing (like all_gather), so it is unused |

## Compute Phases (Phase B, per output-tile position `i`)

| # | Operation | Helper? | Input CB (semantic name, tiles, state) | Output CB (semantic name, tiles) | CB State After |
|---|-----------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | Wait for N shard tiles | — | `cb_gathered_shards` (wait `N`) | — | N tiles present, not popped |
| 2 | Seed `DST[0]` = block 0 tile | raw `copy_tile` | `cb_gathered_shards[0]` | DST[0] | DST[0] = block0 |
| 3 | Accumulate blocks 1..N-1 into `DST[0]` | raw `binary_dest_reuse_tiles<ELWADD, DEST_TO_SRCA>` (loop c=1..N-1) | `cb_gathered_shards[c]` | DST[0] | DST[0] = Σ blocks |
| 4 | Pack + pop | raw `pack_tile` + `cb_pop_front(N)` | — | `cb_reduced` (push 1) | `cb_gathered_shards` freed of these N; `cb_reduced` holds 1 |

After the per-core loop over all owned positions: `output` holds the element-wise sum shard. After Phase B on all devices: every device's `output` == element-wise sum of all N input shards.

## Broadcast Verification

Phase B uses a binary add, but there is **no broadcast** — every operand is a full `[32,32]` tile at the identical `(H,W)` position; the sum accumulates matching full tiles.

| Phase | Op | CB_A (`cb_gathered_shards` tile `c`) Valid Region | CB_B (accumulator `DST[0]`) Valid Region | Broadcast Dim |
|-------|-----|--------------------------------------------------|------------------------------------------|---------------|
| B.3 accumulate | ELWADD (DEST_TO_SRCA) | All `[H,W]` | All `[H,W]` | None |

## Key Risks and Gotchas

- **ONE op-internal GlobalSemaphore, created once, parked (mandate W2 slot).** Create it once keyed on `id(mesh_device)`, `ttnn.synchronize_device` ONCE after creation, cache it, and set `gather_mpd.semaphores = [sem]` so the framework holds its L1 alive across program-cache hits. Do NOT re-create per call; do NOT add a per-call post-dispatch `synchronize_device` barrier. Only Phase A uses it (Phase B has no cross-device sync). Mirror `all_gather.py:83-98, 199-204`.
- **Cache-reuse semaphore re-arm (footgun).** Programs are cached and the GlobalSemaphore reused. The Phase-A counting reader MUST `noc_semaphore_set(sem, 0)` after its last wait (`ccl_helpers_dataflow.hpp:74-77`). Without it the first call passes and the second hangs/corrupts. The acceptance test's two-call program-cache case exercises this.
- **Phase ordering is queue-ordered, not barrier-ordered.** Phase A and Phase B are two `generic_op` dispatches on the same command queue → Phase A completes on device `i` before Phase B runs on device `i`. Phase A's readers block (`noc_semaphore_wait_min`) until every block destined for device `i` has landed, so `gather_buffer` is fully populated before Phase B reads it. Do NOT reorder the dispatches and do NOT drop the Phase-A read-side waits.
- **CB push/pop balance at line ends (Phase A).** Gate the seed push and relay read-back on `num_targets_<dir> > 0`; the line-end device in a direction does not forward, so its reader must NOT push pages its writer will never pop (else `cb_relay_pages` fills and the reader blocks). Push count MUST equal pop count per direction.
- **Edge-device fabric connections.** Device `0` opens NO backward connection; device `N-1` opens NO forward connection. The host calls `setup_fabric_connection` and the writer opens the `FabricStreamSender` only when `num_targets_<dir> > 0`. The line-end worker in the missing direction is a pure receiver. The self-copy is done by the **forward reader on every device** so it never depends on a missing connection.
- **Uniform mesh buffer address.** Phase-A fabric `write_page` uses the LOCAL `TensorAccessor(gather_buffer)` base address as the neighbor's destination; correct only because mesh-allocated interleaved tensors share a buffer address across devices and the 1-hop route directs the write to the neighbor. `gather_buffer` (and `output`) MUST be `allocate_tensor_on_device` on the MeshDevice (uniform addressing) — do not use a per-device ad-hoc address.
- **DST capacity — sum into ONE reg.** Phase B accumulates into `DST[0]` only (seed + DEST_TO_SRCA accumulate), so it is safe for any N and for float32 (DST holds 4 tiles) and bfloat16 (8 tiles). Do NOT load all N tiles into DST at once (that breaks for f32 + N=8). Reference: `reduce_to_one_b1.hpp:600-617`, `compute_collector.cpp:39-50`.
- **`binary_dest_reuse_tiles` accumulate template must be `DEST_TO_SRCA`, not `NONE`.** With `NONE` the op is a plain two-CB add and does NOT accumulate DST; only `DEST_TO_SRCA`/`DEST_TO_SRCB` pull `DST[dst]` into a Src so the single unpacked tile is added back (`eltwise_binary.h:262-284`). Every real accumulation usage in the tree uses `DEST_TO_SRCA`. Call `binary_dest_reuse_tiles_init<ELWADD, DEST_TO_SRCA>` once after the `copy_tile` seed and before the loop (it reconfigures unpacker/math state).
- **bf16 sum precision.** A bf16 sum of N terms accumulates rounding, so the acceptance/golden PCC threshold for bf16 is 0.99 (not the pure-movement 0.995). The oracle accumulates in fp32 then casts. Verified case: bfloat16, TILE_LAYOUT, Linear, `(1, 8)` FABRIC_1D.
- **Verification topology is fixed.** Verified on a simulated Wormhole T3K **line mesh `(1, 8)` with `fabric_config = ttnn.FabricConfig.FABRIC_1D`** via `scripts/run_multidevice_sim_pytest.py --op all_reduce`. The acceptance test MUST open exactly `(1, 8)` with `FABRIC_1D`; a different mesh shape hangs fabric init (`Fabric Router Sync: Timeout`).

## Structural impossibilities (INVALID candidates for feature_spec.py)

The golden `feature_spec.py` already exists (pipeline mode) with `TARGET = {dtype:[bf16,f32], layout:[TILE_LAYOUT], topology:[Linear]}` and `INVALID = []`. Since the Phase-0 TARGET is TILE-only over float dtypes, every cell is constructible — there are **no** structural impossibilities to add (the canonical `{bfloat8_b, ROW_MAJOR}` cell is out of scope for this TILE-only float TARGET). No changes to `feature_spec.py` are proposed.

## Registry contract notes (for the implementer's op file)

The implementer authors `INPUT_TAGGERS`, `SUPPORTED`, `EXCLUSIONS`, and `validate()` in `ttnn/ttnn/operations/all_reduce/all_reduce.py` (matching the golden harness, which imports them). `validate()` must raise `UnsupportedAxisValue` (a `NotImplementedError` subclass from `ttnn.operations._op_contract`) for out-of-`SUPPORTED` axis values. Suggested Phase-0 contract (aligned with `TARGET` and the golden driver's `all_reduce(input, topology=...)` call):

```
SUPPORTED = {
    "dtype":    [ttnn.bfloat16, ttnn.float32],
    "layout":   [ttnn.TILE_LAYOUT],
    "topology": [Topology.Linear],
}
EXCLUSIONS = []
INPUT_TAGGERS = {}   # no shape-derived axis gated by TARGET (all INPUTS are tile-aligned)
```

`validate()` also enforces the structural requirements (raise `ValueError`): input on a `ttnn.MeshDevice` line view `(1, N)` with `N ≥ 2`; interleaved (not sharded); and, if `output_tensor` is supplied, its spec equals the input shard spec.
