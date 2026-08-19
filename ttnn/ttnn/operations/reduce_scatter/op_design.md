# Operation Design: reduce_scatter

## Overview

| Field | Value |
|-------|-------|
| Classification | CCL (multi-device) + compute (tile reduction) — the compute-CCL probe |
| Goal | Sum every device's same-shape shard element-wise across all N devices on a 1-D MeshDevice line, then SCATTER the sum: device i's output is the i-th of N equal slices of the summed tensor along `dim`. Per-device DISTINCT outputs (unlike all_reduce). |
| Math | `output_i[b, c, h, w] = Σ_{j=0..N-1} shard_j[b, c, h, i·(W/N) + w]` for device i, `dim = 3` |
| Mode | Derivative — gather-then-reduce-local-slice, structurally the proven Python `all_reduce` two-dispatch pattern with the scatter folded into Phase-B source addressing and the compute swapped onto `sum_blocks` |
| Algorithm | **GATHER-THEN-REDUCE-LOCAL-SLICE** (the mandate's "simplest" blessed option). Phase A: full-shard line store-and-forward gather into an op-internal `gather_buffer` (fabric dataflow). Phase B: per output-tile position, sum the N gathered blocks' tiles at the slice-i source index (`sum_blocks`), write to output. Two ordered `ttnn.generic_op` dispatches on the same command queue — Phase A completes on device i before Phase B reads its `gather_buffer`; no extra cross-device barrier. |
| Generation mandate | Generated FROM SCRATCH: `ttnn.generic_op` + `ttnn.MeshProgramDescriptor`, newly authored kernels under `ttnn/ttnn/operations/reduce_scatter/kernels/`. Does NOT import/wrap/dispatch to any existing reduce_scatter / all_reduce / all_gather op. |
| Correctness references (READ ONLY) | `ttnn/ttnn/operations/all_reduce/` (proven Python two-dispatch compute-CCL: Phase-A gather kernels, semaphore lifecycle, descriptor assembly); `ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/kernels/compute/reduction.cpp` (the 2-statement `sum_blocks` compute model); `ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/` (silicon-verified ring/line reduce-scatter — the reduce-en-route alternative NOT chosen for Phase-0, see "Algorithm choice") |

### Algorithm choice (decision, not deliberation)

Ring/line reduce-en-route (`reduce_scatter_minimal_async` style) is the bandwidth-optimal classic but requires an N-1-step receive+reduce+forward schedule agreed across three kernels (`RingRsSchedule` / `LineChannelWalk` + `SyncCadence` + `BlockAccumulate`), a double-height intermediate tensor, and a per-direction semaphore protocol. Gather-then-reduce-local-slice is chosen for Phase-0 because it is the algorithm that can be made correct on the first pass: Phase A reuses the silicon-proven `all_reduce` gather structure verbatim (1-hop unicast + store-and-forward relay + one counting semaphore), and the entire "which slice does device i keep" logic collapses into pure local addressing in Phase B — there is no multi-step cross-kernel schedule to drift. The cost is gather-level traffic (each shard traverses the line whole instead of shedding 1/N per hop); recorded as a refinement below.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | rank-4, TILE_LAYOUT, interleaved, on a `(1, N)` MeshDevice line, N ≥ 2, one same-shape shard per device | — | RT (addresses) |
| `dim` | `int` | no | `3` or `-1` (canonicalized to `3` BEFORE the SUPPORTED membership test — a literal test on the raw value would reject the legal `-1` alias) | `3` | CT (structural) |
| `topology` | `ttnn.Topology` | no | `Topology.Linear` (Phase-0) | `Topology.Linear` | CT via route construction |
| `output_tensor` | `ttnn.Tensor \| None` | no | spec must equal the derived output spec (shape/dtype/layout/buffer type); written into and returned | `None` | RT (address) |

Signature (pinned by `eval/golden_tests/reduce_scatter/helpers.py:89` — `dim` is the second positional):

```python
from ttnn.operations.reduce_scatter import reduce_scatter
reduce_scatter(input_tensor, dim=3, topology=ttnn.Topology.Linear, output_tensor=None) -> ttnn.Tensor
```

## Tensors

### Input (per-device shard; every device holds one shard of the SAME shape)

| Property | Requirement |
|----------|-------------|
| Shape | rank-4 `(B, C, H, W)` with `W % (N · 32) == 0` (whole tiles per output slice) — **rejected loudly with `ValueError` otherwise, never padded** |
| Dtype | `bfloat16` (primary), `float32` |
| Layout | TILE_LAYOUT (the reduction is a tile compute) |
| Memory | interleaved, DRAM or L1 |
| Device | `ttnn.MeshDevice` with shape `(1, N)`, N ≥ 2, `FABRIC_1D` |

### Output (device i)

| Property | Value |
|----------|-------|
| Shape | `(B, C, H, W / N)` — slice i of the N-way sum along dim 3 |
| Dtype | same as input |
| Layout | TILE_LAYOUT |
| Memory | same buffer type as input, interleaved; allocated via `ttnn.allocate_tensor_on_device` on the MeshDevice (uniform address) when `output_tensor is None` |

### Op-internal `gather_buffer` (per call, per device)

| Property | Value |
|----------|-------|
| Shape | `(N · B, C, H, W)` — N full-shard blocks stacked on dim 0; block c occupies pages `[c · P_shard, (c+1) · P_shard)` |
| Dtype / layout / memory | same as input; `ttnn.allocate_tensor_on_device` on the MeshDevice — **uniform mesh address is load-bearing**: the fabric `write_page` targets the neighbour's block through the LOCAL accessor base address routed one hop |

### Page arithmetic (host, per call)

| Symbol | Formula |
|--------|---------|
| `P_shard` | `input_tensor.buffer_num_pages()` |
| `page_size` | `input_tensor.buffer_page_size()` (bf16 tile = 2048 B, f32 tile = 4096 B; validate guards `page_size % 16 == 0`) |
| `Wt` | `W / 32` (tiles per full row) |
| `slice_Wt` | `Wt / N` (tiles per row of one output slice) |
| `P_out` | `P_shard / N` (output pages per device) |
| src page for output position `t` on device i | `(t / slice_Wt) · Wt + i · slice_Wt + (t % slice_Wt)` — emitted by `SliceRowWalker`, base `slice_tile_offset(3, i, ·, ·, slice_Wt) = i · slice_Wt` |

## Dataflow Strategy

Two ordered `ttnn.generic_op` dispatches per call, each a `ttnn.MeshProgramDescriptor` with one `(MeshCoordinateRange, ProgramDescriptor)` entry per device on the line (all N devices participate).

### Phase A — gather (fabric): identical structure to the proven `all_reduce` Phase A

Per device i, two worker cores, each running a reader (NCRISC) + writer (BRISC):

| Core | Role | Fabric connection |
|------|------|-------------------|
| forward core `(0, 0)` | flow rightward | → chip i+1 (`num_targets_fwd = N-1-i` downstream; none for i = N-1) |
| backward core `(0, 1)` | flow leftward | → chip i-1 (`num_targets_bwd = i` downstream; none for i = 0) |

Data path per direction (store-and-forward, 1-hop unicast):

1. **Self-copy** (forward reader only, every device): read own input shard, plain-NoC write it into own `gather_buffer` block i. Uses `cb_self_copy_scratch` as reserve-only scratch (never pushed).
2. **Seed**: reader stages the own shard's `P_shard` pages into `cb_relay_pages`; writer drains them via `write_page` into the neighbour's `gather_buffer` at pages `i · P_shard + p`, then one counting `inc` on the neighbour's semaphore (in-order on the connection ⇒ the inc lands after the block's data).
3. **Relay**: for each upstream block that lands locally (forward core: blocks i-1, i-2, …, 0 arriving from device i-1; backward core: blocks i+1, …, N-1 from device i+1), the reader waits `noc_semaphore_wait_min(sem, k)` (incremental), reads the block BACK out of local `gather_buffer` into `cb_relay_pages` (**this local `noc_async_read` IS the receive ingress — there is no FabricStreamReceiver**, per the helper banner `ccl_helpers_dataflow.hpp:112-118`), and the writer forwards it one more hop + `inc`.
4. **Line end in a direction** (`my_num_targets == 0`): writer opens no connection and returns; reader is a pure receiver — one `noc_semaphore_wait_min(sem, num_relay_blocks)`.
5. **Cache-reuse re-arm**: every reader ends with `noc_semaphore_set(sem, 0)` after its last wait (RECEIVER resets AFTER its wait — `ccl_helpers_dataflow.hpp:118-121`). Without it: first call green, second call hangs.

Semaphore accounting (per device i — waits consumed == incs received, both directions independent per-core L1 words of ONE GlobalSemaphore):

| Core | Blocks received | Sender of each inc | Final wait value before re-arm |
|------|-----------------|--------------------|--------------------------------|
| forward `(0,0)` | `i` (blocks i-1 … 0, in that order) | device i-1's forward writer | `i` |
| backward `(0,1)` | `N-1-i` (blocks i+1 … N-1) | device i+1's backward writer | `N-1-i` |

After Phase A, device i's `gather_buffer` holds all N blocks: block i by self-copy, blocks < i via forward-direction arrivals, blocks > i via backward-direction arrivals.

### Phase B — scatter-reduce (local compute; the scatter IS the addressing)

Pure local, no fabric, no cross-device sync. `P_out` output-tile positions split across the compute grid. Per owned position `t`:

- **Reader (NCRISC)**: one `SliceRowWalker::next()` gives the in-shard source page `src` for slice `my_chip_id`; reads the N gathered tiles `gather_buffer[c · P_shard + src]`, c = 0…N-1, into `cb_gathered_slices` (block order c ascending), one barrier, one `cb_push_back(cb_gathered_slices, N)`.
- **Compute (TRISC)**: `compute_kernel_lib::sum_blocks(cb_gathered_slices, cb_reduced_slice, N, 1, /*pop_input=*/true)` — waits the N tiles, sums them (DST-chunked internally, odd-N seeded, even-N acc_to_dest from DST-zero), pops the input, pushes 1 reduced tile.
- **Writer (BRISC)**: waits 1 tile, `noc_async_write` to output page `start_tile + t`, barrier, pop.

Format at every stage: TILE (32×32 tiles end to end; no tilize/untilize).

### Inter-Tensix / inter-device contract summary

| Link | Mechanism | Ordering guarantee |
|------|-----------|--------------------|
| device j → j±1 payload | fabric 1-hop unicast `write_page` into persistent `gather_buffer` | in-order on one fabric connection |
| device j → j±1 "block landed" | fabric `AtomicIncChannel::inc` on the receiver core's GlobalSemaphore word | issued after the block's last page on the same connection ⇒ lands after the data |
| Phase A → Phase B on one device | same command queue, two ordered `generic_op` dispatches | dispatch order |
| Phase-B cores | none (independent tile ranges) | — |

## Work Distribution

### Phase A

| Field | Value |
|-------|-------|
| Work unit | one shard block (`P_shard` pages) relayed per direction |
| Grid | 2 fixed worker cores per device: forward `(0,0)`, backward `(0,1)`; single link (`link_idx = 0`) |
| Per-core work | forward core: 1 seed + `num_targets_bwd` relays (i blocks in), self-copy; backward core: 1 seed + `num_targets_fwd` relays |
| Remainder | none — block counts are exact functions of the line position |

### Phase B

| Field | Value |
|-------|-------|
| Work unit | one output-tile position (N input tiles → 1 output tile) |
| Grid | `mesh_device.compute_with_storage_grid_size()` via `ttnn.split_work_to_cores(grid, P_out)` |
| Per-core work | contiguous `[start_tile, start_tile + n)`; `n = tiles_per_core_g1` or `g2` |
| Remainder | handled by split_work_to_cores' two core groups; cores beyond `num_cores` get no kernel args |

## Circular Buffers

### Phase A (both worker cores)

| Semantic Name | Index | Page Size | Num Pages | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|--------|----------|----------|----------|
| `cb_relay_pages` | 16 | `align(page_size, l1_alignment)` | 2 | input dtype | gather reader | gather writer | streaming double-buffer, one page per push/pop |
| `cb_self_copy_scratch` | 24 | `align(page_size, l1_alignment)` | 2 | input dtype | forward gather reader (reserve-only) | — | scratch: `cb_reserve_back(·, 1)` once, NEVER pushed/popped (proven all_reduce idiom — do not "fix" into a push/pop CB) |

### Phase B (all work cores)

| Semantic Name | Index | Page Size | Num Pages | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|--------|----------|----------|----------|
| `cb_gathered_slices` | 0 | `page_size` (tile) | `2 · N` | input dtype | reduce reader | compute (`sum_blocks`) | double-buffered group of N tiles; push/wait/pop count = N per position |
| `cb_reduced_slice` | 16 | `page_size` (tile) | 2 | output dtype | compute (`sum_blocks`) | reduce writer | double buffer; push/wait/pop count = 1 per position |

Sizing rationale: `cb_gathered_slices` capacity `2·N` tiles is independent of tensor shape (N ≤ 8 ⇒ ≤ 64 KB even at f32) — `sum_blocks` waits its whole input (`N·1` tiles) up front (`accumulate_helpers_compute.hpp:199-201`), so capacity must be ≥ N; 2× double-buffers the reader against the compute. Per-position granularity (`block_num_tiles = 1`) is chosen over multi-position chunks: chunking multiplies CB capacity by the chunk size for a throughput-only gain — recorded as a refinement.

CB sync verification (push count == wait count == pop count per CB per position/page):

| CB | Producer pushes | Consumer waits/pops |
|----|-----------------|---------------------|
| `cb_relay_pages` | 1 page × `P_shard` × (1 + relays) | 1 page × `P_shard` × (1 + relays) |
| `cb_self_copy_scratch` | 0 (reserve-only) | 0 |
| `cb_gathered_slices` | N × n positions | N × n positions (`sum_blocks` waits N, pops N via `pop_input=true`) |
| `cb_reduced_slice` | 1 × n positions | 1 × n positions |

## API Mapping

Every mechanism with verified file:line. Paths abbreviated: `kernel_lib/` = `ttnn/cpp/ttnn/kernel_lib/`, `schedule.hpp` = `ttnn/cpp/ttnn/operations/ccl/shared_with_host/ccl_helpers_schedule.hpp` (kernel include path: `"ttnn/operations/ccl/shared_with_host/ccl_helpers_schedule.hpp"`).

### Kernel-side

| Phase | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Requirements |
|-------|------|----------|-----------|------------------------|----------|-----------|--------------|
| A writer | helper | `FabricStreamSender<>` ctor (RT-cursor form) | `kernel_lib/ccl_helpers_dataflow.hpp:481,492` | `(size_t& conn_arg_idx, bool is_forward, uint32_t alignment)`; `is_forward` peeked from the conn block's leading `has_forward` flag | — | — | conn block laid out by host `setup_fabric_connection` |
| A writer | helper | `sender.open(unicast_route(num_hops))` | `:503` (open), `:302` (`unicast_route`) | route bound ONCE; `num_hops = 1` (neighbour) from host `ccl_dm_route` | — | — | typestate: arm/issue only on the returned `FabricStream` |
| A writer | helper | `stream.arm_unicast_write(page_size)` | `:423` | invariant per-page payload size | — | — | one pooled header per arm |
| A writer | helper | `stream.arm_inc(1)` | `:435` | unicast, stream's route, invariant inc value | — | — | coexists with the write channel |
| A writer | helper | `writer.write_page(l1, c·P_shard + p, gather_buffer_accessor)` | `:327` | destination page via the LOCAL TensorAccessor, routed 1 hop | `cb_relay_pages` (read ptr) | remote `gather_buffer` | `noc_async_writes_flushed()` after each `write_page`, BEFORE `cb_pop_front` (CB slot reuse guard — proven all_reduce writer idiom) |
| A writer | helper | `counter.inc(neighbor_sem_noc_addr)` | `:368` | once per completed block | — | remote sem | in-order after the block's data |
| A writer | helper | `stream.close()` | `:461` | drains write + atomic barriers, disconnects | — | — | idempotent with dtor |
| A reader | raw_api | `noc_semaphore_wait_min(sem_ptr, k)` / `noc_semaphore_set(sem_ptr, 0)` | dataflow_api; ownership: `ccl_helpers_dataflow.hpp:112-121` | incremental counting waits; terminal re-arm to 0 | — | — | **op-owned by design** — the helper owns only the SENDING half (`inc`); "The WAITING half is a plain local noc_semaphore_wait_min the op calls directly … The receive INGRESS is likewise a local NoC read the op owns; there is no FabricStreamReceiver" (`:112-118`) |
| A reader | raw_api | `noc_async_read` / `noc_async_write` (+ barriers) | dataflow_api | self-copy, seed staging, relay read-back | → `cb_relay_pages` / `cb_self_copy_scratch` | — | receive ingress is op-owned (see above); `TensorAccessor` addressing |
| A+B | helper | `TensorAccessor(args, addr, page_size)` | `tech_reports/tensor_accessor/tensor_accessor.md`; CT args via `ttnn.TensorAccessorArgs` | interleaved page → bank NoC address | — | — | CT `TensorAccessorArgs` placed after scalar CT args |
| B reader | helper | `ttnn::ccl::schedule::slice_tile_offset(3, my_chip_id, slice_C, slice_Ht, slice_Wt)` | `schedule.hpp:466-478` | = `my_chip_id · slice_Wt` for dim 3 | — | — | gate `static_assert(is_supported_scatter_dim(dim))` (`schedule.hpp:460`) |
| B reader | helper | `ttnn::ccl::schedule::SliceRowWalker` | `schedule.hpp:491-540` (ctor `:498`, `set_base` `:502`, `reset_offsets` `:510`, `next` `:516`) | `SliceRowWalker(slice_Wt, Wt)`; `set_base(my_chip_id · slice_Wt)`; `reset_offsets(start_tile % slice_Wt, (start_tile / slice_Wt) · Wt)` — the same start-offset formula the silicon-verified host helper uses (`reduce_scatter_program_utils.cpp:162-165`); ONE `next()` per output position | — | — | plain C++17, host-unit-tested (`tests/ttnn/unit_tests/gtests/ccl/test_ccl_helpers_schedule.cpp`) |
| B compute | raw_api | `binary_op_init_common(cb_gathered_slices, cb_gathered_slices, cb_reduced_slice)` | `api/compute/eltwise_binary.h`; required by `accumulate_helpers_compute.hpp:211` (`@pre`) | hardware startup, once per kernel, before the first `sum_blocks` | — | — | deliberately NOT owned by the helper (`hpp:70-77`) |
| B compute | **helper** | `compute_kernel_lib::sum_blocks(cb_gathered_slices, cb_reduced_slice, N, 1, /*pop_input=*/true)` | decl `kernel_lib/accumulate_helpers_compute.hpp:221-222`, impl `.inl:106-157` | `num_blocks = N` (CT), `block_num_tiles = 1`, `pop_input = true` (streaming producer/consumer CB — the llama_reduce_scatter pattern, `hpp:201-204`); called once per owned position | `cb_gathered_slices` (N tiles) | `cb_reduced_slice` (1 tile) | owns wait-whole-input / reserve / seed-or-pair / DST-chunk vs `DEST_AUTO_LIMIT` / pack / pop / push (`hpp:199-209`); kernel includes `"api/compute/eltwise_binary.h"` + `"ttnn/cpp/ttnn/kernel_lib/accumulate_helpers_compute.hpp"` (model: `all_reduce_async/device/kernels/compute/reduction.cpp:6-7`) |
| B writer | raw_api | `noc_async_write(l1, output.get_noc_addr(start_tile + t), page_size)` + barrier | dataflow_api | dense sequential output pages | `cb_reduced_slice` | output tensor | pure local writes |

### Host-side (Python)

| Concern | Binding | File:Line | Usage |
|---------|---------|-----------|-------|
| Route (1-hop, sign reversal owned) | `ttnn._ttnn.fabric.ccl_dm_route(mesh_device, coord_i, coord_i±1, topology)` | `ttnn/cpp/ttnn-nanobind/fabric.cpp:253-266`; impl `ccl_helpers_dataflow_host.hpp:151-180` | `.num_hops` → writer RT arg; `.is_forward` → `_append_fabric_rt_args` |
| Fabric connection | `ttnn.setup_fabric_connection(src_id, dst_id, 0, program, core)` | `fabric.cpp:141-178` | returns conn RT-arg list AND mutates `program` (appends SemaphoreDescriptors). Appended AFTER `ProgramDescriptor` construction by mutating `program.kernels[k].runtime_args[x][y]` in place, wrapped `[has_forward][fwd args][has_backward][bwd args]` (proven `all_reduce_program_descriptor.py:78-92,257-265` idiom) |
| GlobalSemaphore | `ttnn.create_global_semaphore(mesh_device, worker_cores, 0)` + `ttnn.synchronize_device` ONCE + `ttnn.get_global_semaphore_address` | `ttnn/cpp/ttnn-nanobind/global_semaphore.cpp:40-67` | module-level `_SEMAPHORE_CACHE` keyed on `id(mesh_device)`; created once, NEVER per call; NO per-call post-dispatch barrier |
| Semaphore parking | `gather_mpd.semaphores = [sem]` | `ttnn/cpp/ttnn-nanobind/program_descriptors.cpp:1077-1087` | keeps the L1 allocation alive across program-cache hits; excluded from the cache hash (`:1086`) |
| Dispatch | `ttnn.generic_op([input, gather_buffer], gather_mpd)` then `ttnn.generic_op([gather_buffer, output], reduce_mpd)` | `generic_op_nanobind.cpp:23-60` | output tensor pre-allocated, LAST in io_tensors |
| Work split (Phase B) | `ttnn.split_work_to_cores(grid, P_out)` | re-export `ttnn/ttnn/core.py:19` | two core groups + `ttnn.corerange_to_cores` |
| Allocation | `ttnn.allocate_tensor_on_device(Shape, dtype, layout, mesh_device, memory_config)` | `ttnn/cpp/ttnn-nanobind/operations/core.cpp:287-305` | output (when None) and `gather_buffer` |
| Program cache | hash = kernel sources + CT args + core ranges + CB descriptors per `(range, program)` | `generic_op_device_operation.cpp:48-118` | addresses / page counts / sem_addr / num_hops are RT args (excluded from hash) ⇒ 2nd same-shape call hits cache |

### Helpers considered and rejected (mandatory justifications)

| Candidate | File:Line of mismatch | Concrete reason |
|-----------|------------------------|-----------------|
| `BlockAccumulate::arm/run` | `kernel_lib/accumulate_helpers_compute.hpp:119-125` | `arm(cb_a, cb_b, cb_out, granularity)` adds tiles across TWO separate CBs (`out[i] = a[i] + b[i]`) — the ring receive+reduce shape where each step has exactly two operand streams. Here all N partials land block-major in ONE CB; that is verbatim "the all_reduce pattern, where the gathered per-device partials land as contiguous blocks of one input CB" that `sum_blocks` was built for (`hpp:196-198`). Summing N > 2 blocks with BlockAccumulate would need a second CB plus an intermediate ping-pong and N-1 passes vs one DST-accumulated pass. |
| `RingRsSchedule` / `LineChannelWalk` / `LineSliceCursor` / `SyncCadence` | `schedule.hpp:223-383, 690-753, 627-645, 588-607` | These encode the multi-step receive-reduce-forward schedules (per-step flags, chunk parity, chunks-per-sync reader-wait/writer-inc pairing) of the reduce-en-route algorithms. Gather-then-reduce has exactly ONE reduction step per device with no reader/compute/writer chunk cadence to agree on — the only cross-kernel protocol is "N tiles per position", a host-fixed constant. Constructing a schedule object would manufacture agreement where no multi-step schedule exists. The slice-walk members that DO apply (`slice_tile_offset`, `SliceRowWalker`) ARE used. |
| `SequentialTileWalker` | `schedule.hpp:543-563` | Designed for the ring kernels' per-step `set_base` / per-channel `bump_base`/`reset_offsets` cadence. The Phase-B writer's output pages are the flat range `[start_tile, start_tile + n)` with no step/channel structure — `start_tile + t` is the entire walk; wrapping it adds state with zero shared-definition value (compute and reader don't consume output tile ids). |
| `reduce_helpers_compute.hpp` `reduce()` | `kernel_lib/accumulate_helpers_compute.hpp:12-20` | Wrong operation: `reduce_tile` reduces WITHIN a tensor along a dim (collapses the 32×32 tile dims over ROW/COL/SCALAR). This op needs whole tile-blocks added TOGETHER across CB blocks — "Different LLK op, different shape, no overlap" (quoted from the accumulate header's own banner). |
| `eltwise_convenience.hpp` `add<cb_a, cb_b, cb_out>(n)` | `kernel_lib/accumulate_helpers_compute.hpp:8-10, 194-198` | Two-operand streaming add — same two-CB shape mismatch as BlockAccumulate for an N-way sum resident in one CB. A pairwise tree of `add` calls needs extra intermediate CBs and log₂N passes; `sum_blocks` is the purpose-built single-pass primitive ("the compute-side primitive every reduction collective's compute kernel is built from"). |
| `FabricDuplexSender` | `kernel_lib/ccl_helpers_dataflow.hpp:49-58` | Duplex fans ONE issue out to every CONNECTED direction (its use case: all_reduce_async's identical bidirectional broadcast, `worker_writer.cpp:88-99`). Phase A's two directions carry DIFFERENT block sequences (forward relays blocks < i, backward relays blocks > i) from two different cores — two independent `FabricStreamSender`s on two cores is the proven shape. |
| `ttnn._ttnn.fabric.ccl_packet_dims` | `fabric.cpp:245-252`; impl `ccl_helpers_dataflow_host.hpp:89-111` | Packet framing is for splitting/packing pages against the fabric max packet. This op uses 1:1 page↔packet framing via `arm_unicast_write(page_size)` + `write_page` (tile pages of 2048/4096 B fit a fabric channel buffer), the same deliberate non-use as the proven all_gather/all_reduce Python ops. The bf16 `bit_floor` special case never arises. `validate()` keeps the load-bearing `page_size % 16 == 0` guard: the fabric writer sends `align(page_size, l1_alignment)` bytes per page while the accessor spaces pages by raw `page_size` (`ccl_helpers_dataflow.inl:35`). |
| `mcast_pipe.hpp` `SenderPipe`/`ReceiverPipe` | header scope: intra-device NoC multicast + semaphore handshake | No intra-device multicast exists: Phase-A cores are independent per-direction workers; Phase-B cores share nothing. |

## Compute Phases

Phase B compute kernel, per owned output position (n iterations; Phase A has no compute kernel):

| # | Operation | Helper? | Input CB (tiles, state) | Output CB (tiles) | CB State After |
|---|-----------|---------|--------------------------|-------------------|----------------|
| 0 | hardware startup `binary_op_init_common(cb_gathered_slices, cb_gathered_slices, cb_reduced_slice)` — once, before the loop | raw (required `@pre`, `accumulate_helpers_compute.hpp:211`) | — | — | — |
| 1 | N-way tile sum: `sum_blocks(cb_gathered_slices, cb_reduced_slice, N, 1, true)` | **yes** | `cb_gathered_slices` (N tiles, block order c = 0…N-1 matching gather block indices) | `cb_reduced_slice` (1 tile) | `cb_gathered_slices` empty (popped by `pop_input=true`); `cb_reduced_slice` +1 pending for writer |

Compute config: `ttnn.ComputeConfigDescriptor(math_fidelity=HiFi4, fp32_dest_acc_en=True, math_approx_mode=False, dst_full_sync_en=False)` — fp32 DST accumulation covers both the bf16 sum-of-N rounding budget and the float32 dtype; `sum_blocks`' internal `DEST_AUTO_LIMIT` chunking (`hpp:204-206`) derives the matching 4-tile f32 capacity kernel-side, so `block_num_tiles = 1` is trivially safe.

## Broadcast Verification

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast Dim |
|-------|----|--------------------|--------------------|---------------|
| B.1 | `sum_blocks` (elementwise `add_tiles` across equal-shaped blocks) | `cb_gathered_slices` full tiles (All) | `cb_gathered_slices` full tiles (All) | None (no broadcast anywhere in this op) |

## Module Layout (implementer contract)

```
ttnn/ttnn/operations/reduce_scatter/
  __init__.py                          # exports reduce_scatter, SUPPORTED, EXCLUSIONS, INPUT_TAGGERS (harness reads them at package level)
  reduce_scatter.py                    # registry contract + validate() + entry point + _SEMAPHORE_CACHE
  reduce_scatter_program_descriptor.py # Phase-A (gather) + Phase-B (scatter-reduce) MeshProgramDescriptor builders
  kernels/
    reduce_scatter_gather_reader.cpp   # NCRISC, both direction cores (direction = CT arg): self-copy / seed / relay read-back / sem waits / re-arm
    reduce_scatter_gather_writer.cpp   # BRISC, FabricStreamSender egress: write_page + counting inc, per direction
    reduce_scatter_reduce_reader.cpp   # NCRISC, SliceRowWalker slice-i source walk over the N gather blocks
    reduce_scatter_compute.cpp         # TRISC, binary_op_init_common + sum_blocks loop
    reduce_scatter_reduce_writer.cpp   # BRISC, dense output page writes
  op_design.md
```

Deliberate divergence from `all_reduce`'s shared-source-with-phase-CT-arg pattern: five single-purpose kernel sources instead of two dual-phase sources. This removes the uniform-CT-superset zero-padding footgun (`get_compile_time_arg_val` static-asserts on the index even in a discarded `if constexpr` branch) at the cost of three extra small files. CT-arg layouts are then free per kernel: all scalar CT args first, `TensorAccessorArgs` last.

Registry contract (Phase-0):

```python
INPUT_TAGGERS: dict = {}
SUPPORTED = {
    "dtype": [ttnn.bfloat16, ttnn.float32],
    "layout": [ttnn.TILE_LAYOUT],
    "topology": [_Topology.Linear],
    # Index axis, canonicalized to POSITIVE (rank 4) before the membership test:
    # dim=-1 ≡ 3. Not swept by the golden harness (absent from TARGET); op-level gate only.
    "dim": [3],
}
EXCLUSIONS: list = []
```

`validate()` shape (mirrors the proven `all_reduce.py:106-157`): structural errors → `ValueError` (not on MeshDevice; mesh not `(1, N)`; N < 2; sharded input; rank ≠ 4; `W % (N·32) != 0` — loud, no padding; `page_size % 16 != 0`; `output_tensor` spec ≠ derived output spec). Axis refusals → `UnsupportedAxisValue` / `ExcludedCell` from `ttnn.operations._op_contract` (`_op_contract.py:23-32`, with the ImportError fallback idiom). `dim` is canonicalized (`dim if dim >= 0 else dim + rank`) BEFORE the `SUPPORTED["dim"]` membership test. Import `Topology` via `from ttnn._ttnn.operations.ccl import Topology as _Topology` (eager-import safety).

Entry point flow: `validate` → allocate output if `None` (`(B, C, H, W//N)`, input dtype/layout/memory_config — every output page is overwritten, no seeding) → allocate `gather_buffer` → get/park semaphore → dispatch Phase A → dispatch Phase B → `return output_tensor` (the SUPPLIED handle when given).

## Verification Topology

| Field | Value |
|-------|-------|
| Runner | `scripts/run_multidevice_sim_pytest.py --runtime hardware --op reduce_scatter` |
| Hardware topology | `bh_quietbox_1x4_hw`: 4-chip Blackhole mesh `(1, 4)`, `fabric_config = ttnn.FabricConfig.FABRIC_1D` (`scripts/multidevice_sim_topologies.yaml:175-187`) |
| Acceptance test fixture | MUST open exactly `mesh_device (1, 4)` with `FABRIC_1D` — any other shape hangs fabric init (`Fabric Router Sync: Timeout`) or fails `system_mesh.cpp: requested_size <= system_size` |
| Golden suite | already present at `eval/golden_tests/reduce_scatter/` (pipeline mode — `feature_spec.py` with TARGET/INPUTS/INVALID is authoritative; `INVALID = []` and no additional structural-impossibility candidates exist: the TILE-only layout universe never constructs a `bfloat8_b × ROW_MAJOR` cell) |
| Test shapes | `W % (N·32) == 0` — multiples of 128 on the `(1, 4)` box |

## Key Risks and Gotchas

| # | Risk | Mitigation |
|---|------|-----------|
| 1 | **Semaphore re-arm on cache reuse** — programs are cached and the GlobalSemaphore reused; missing reset ⇒ first call green, second hangs | Every Phase-A reader ends with `noc_semaphore_set(sem, 0)` AFTER its last wait (`ccl_helpers_dataflow.hpp:118-121`: RECEIVER resets after its wait). The acceptance test's two-call program-cache case exercises exactly this. |
| 2 | **Semaphore lifecycle** — re-creating per call re-syncs the mesh and can race in-flight waits | Module-level `_SEMAPHORE_CACHE` keyed `id(mesh_device)`; ONE `create_global_semaphore` + ONE `synchronize_device` at creation; parked on `gather_mpd.semaphores`; NO per-call post-dispatch barrier. |
| 3 | **Uniform mesh addressing** — the fabric `write_page` computes the destination through the LOCAL accessor base address | `gather_buffer` and `output` MUST be `ttnn.allocate_tensor_on_device` on the MeshDevice; never per-device ad-hoc addresses. |
| 4 | **Block order into `sum_blocks`** — the reader must push exactly N tiles per position in gather-block order c = 0…N-1; a swapped or short push deadlocks the wait-whole-input (`hpp:199-201`) or silently mis-sums | Single reader loop `for c in 0..N-1` over `c · P_shard + src`; CB protocol table above (push N == wait N == pop N). |
| 5 | **Slice addressing** — the ONLY place the scatter exists; an off-by-one in `src = row·Wt + i·slice_Wt + w` produces a valid-looking wrong slice (all_reduce's identical-everywhere oracle would mask it; reduce_scatter's per-device-distinct oracle catches it) | `SliceRowWalker` + `slice_tile_offset` from the shared schedule header (host-unit-tested), not hand-rolled arithmetic; kernel derives `reset_offsets(start_tile % slice_Wt, (start_tile / slice_Wt) · Wt)` from the single host-passed `start_tile`. |
| 6 | `cb_self_copy_scratch` is reserve-only (never pushed/popped) | Documented in the CB table; implementer must not "repair" it. |
| 7 | **CB slot reuse vs in-flight fabric write** | `noc_async_writes_flushed()` after every `write_page`, BEFORE `cb_pop_front(cb_relay_pages, 1)` (proven all_reduce writer ordering). |
| 8 | **Page overrun on the wire** — fabric sends `align(page_size, 16)` bytes; accessor spaces pages by raw `page_size` | `validate()` guards `page_size % 16 == 0` (a no-op for TILE pages, kept explicit). |
| 9 | **Program-cache correctness** — anything per-call must be RT | RT: all buffer addresses, `P_shard`, `page_size`, `num_hops`, `sem_addr`, NoC coords, `start_tile`/`n`, `slice_Wt`/`Wt`. CT: CB indices, `direction`, `my_chip_id`, `N`, `dim`, `l1_alignment`, TensorAccessorArgs. |
| 10 | **`dim` sign aliasing** — `dim=-1` is legal and equals 3 | Canonicalize to positive before the `SUPPORTED["dim"]` membership test (the all_gather `_canonical_gather_dim` discipline). |
| 11 | **DEST capacity** | `fp32_dest_acc_en=True` ⇒ 4 f32 tiles; `sum_blocks` chunks internally against the kernel-derived `DEST_AUTO_LIMIT` (`hpp:204-206`) — no host clamp needed at `block_num_tiles = 1`. |
| 12 | **bf16 precision** — a bf16 sum of N terms accumulates rounding | fp32 DST accumulation + PCC threshold 0.99 for bf16 (matches the golden suite tolerance `(0.99, 0.05)`), 0.999 for f32. |
| 13 | **Line-end early return** — device 0's backward writer / device N-1's forward writer have no targets | `my_num_targets == 0` ⇒ writer opens no connection, returns; its RT arg list is empty; reader in that direction is a pure receiver. Kernel must guard ALL RT reads behind the early return. |

## Refinement candidates (for op_requirements.md)

| Candidate | Note |
|-----------|------|
| Traffic-optimal Phase A | Relay only slices still needed downstream (each hop sheds 1/N), or full ring/line reduce-en-route (`reduce_scatter_minimal_async` model with `RingRsSchedule`/`LineChannelWalk` + `BlockAccumulate` + `SyncCadence`). Cuts hop-bytes by ~N and removes the N×-shard `gather_buffer`. |
| Multi-position `sum_blocks` chunks | `block_num_tiles = g > 1` amortizes per-call overhead; CB grows to `2·N·g` tiles. |
| `topology = Ring` | `FABRIC_1D_RING` + `ccl_dm_route`'s short-way; TARGET currently Linear-only. |
| `dim ∈ {0, 1, 2}` | dim 1/2 via `slice_tile_offset` (`schedule.hpp:466-478` already covers them); dim 0 is a batch-block slice (dense pages — simpler than dim 3). |
| Multi-link / multi-worker Phase A | `reduce_scatter_default_workers` heuristic (`reduce_scatter_program_utils.cpp:31-97`). |
