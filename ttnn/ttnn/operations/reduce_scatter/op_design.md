# Operation Design: reduce_scatter

## Overview

| Field | Value |
|-------|-------|
| Classification | CCL (multi-device, fabric) + compute (TRISC reduction) |
| Goal | Sum every device's shard element-wise across the N devices of a 1-D MeshDevice line, then scatter: device `i` keeps only slice `i` (of N equal slices along `dim`) of the sum. Per-device-DISTINCT output, unlike all_reduce. |
| Math | `output_i[...] = (Σ_{c=0}^{N-1} shard_c[...])[slice i along dim]`, `output.shape[dim] = input.shape[dim] / N` |
| Algorithm | **GATHER-THEN-REDUCE-LOCAL-SLICE** — two ordered `ttnn.generic_op` dispatches on the same command queue. Phase A: line store-and-forward gather of full shards into an op-internal `gather_buffer` (proven all_reduce Phase-A structure). Phase B: local N-way tile sum over ONLY the tile positions of device `i`'s slice, written to the `[dim]/N` output. |
| Mode | Derivative — structural clone of the in-tree Python `all_reduce` generic_op, plus slice addressing from the shared schedule header |
| References | `ttnn/ttnn/operations/all_reduce/` (host + kernels, the proven two-phase model); `ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/` (silicon-verified ring/line reference — read-only correctness reference); `ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/kernels/compute/reduction.cpp` (the 2-statement `sum_blocks` compute model) |

**Generation mandate**: the op is generated from scratch as a self-contained Python op on
`ttnn.generic_op` + `ttnn.MeshProgramDescriptor` with newly authored kernels under
`ttnn/ttnn/operations/reduce_scatter/kernels/`. It must NOT import/call/wrap any existing
reduce_scatter / all_reduce / all_gather op.

### Why gather-then-reduce (algorithm decision)

| Consideration | Decision |
|---|---|
| Correctness risk | Ring reduce-scatter needs the full `RingRsSchedule` step machine agreed across 3 kernels. Gather-then-reduce reduces the cross-kernel agreement to a per-core `(start_tile, num_tiles)` contract + a fixed N-tiles-per-position CB protocol — far smaller drift surface. Both algorithms are blessed; this is the one we can make correct first. |
| Helper coverage | Still composes all three helper families: fabric egress (`FabricStreamSender`, Phase A writer), compute accumulation (`sum_blocks`, Phase B compute), schedule header (`SliceRowWalker` + `slice_tile_offset` + `is_supported_scatter_dim`, Phase B reader — the ONE definition of slice addressing). |
| Cost | Phase A moves full shards (all_gather traffic) instead of only needed slices. Perf refinement candidate (slice-only relay, or true ring RS); recorded in Key Risks §R12. |

## Parameters

| Name | Type | Required | Valid Range | Default | Notes |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | on a `ttnn.MeshDevice` `(1, N)` line, N ≥ 2; one same-shape shard per device | — | positional |
| `dim` | `int` | no | Phase-0: canonical `3` (negative alias `-1` accepted); TARGET adds `2` | `3` | **Canonicalize to POSITIVE before the SUPPORTED membership test**: `dim if dim >= 0 else dim + 4` (rank pinned to 4). The golden feature_spec pins the positive convention (`eval/golden_tests/reduce_scatter/feature_spec.py:41-52`). |
| `topology` | `ttnn.Topology` | no | Phase-0: `Linear` | `Linear` | Import as `from ttnn._ttnn.operations.ccl import Topology as _Topology` (the top-level alias binds too late at eager-import time — `all_reduce.py:35-37`). |
| `output_tensor` | `ttnn.Tensor \| None` | no | spec must equal the derived output spec | `None` | Written into and returned when supplied. |

## Tensors

### Input (per-device shard; every device holds the SAME shape, distinct values)

| Property | Requirement |
|----------|-------------|
| Shape | rank 4 `(B, C, H, W)`; `H % 32 == 0`, `W % 32 == 0` (no padded sub-tile dims); `shape[dim] % (N * 32) == 0` (slice divisible AND tile-aligned — reject loudly with ValueError, never pad) |
| Dtype | `bfloat16` (primary), `float32` |
| Layout | `TILE_LAYOUT` only (the reduction is a tile compute) |
| Memory | interleaved, DRAM or L1 (`is_sharded()` → ValueError) |
| Page size | `buffer_page_size() % 16 == 0` (fabric payload is rounded up to l1_alignment; TILE pages always satisfy this — keep the explicit guard, mirroring `all_reduce.py:124-131`) |

### Output (per-device, DISTINCT)

| Property | Value |
|----------|-------|
| Shape | input shard shape with `shape[dim] //= N` |
| Dtype | = input dtype |
| Layout | `TILE_LAYOUT` |
| Memory | = input memory config |
| Allocation | `ttnn.allocate_tensor_on_device(ttnn.Shape(out_shape), dtype, layout, mesh_device, input.memory_config())` when not supplied. Every output page is written — no seeding. |

### Op-internal gather_buffer (allocated per call, never returned)

| Property | Value |
|----------|-------|
| Shape | `[B * N, C, H, W]` — N shard-blocks stacked on dim 0, so block `c` occupies the contiguous page range `[c*P, (c+1)*P)` where `P = input.buffer_num_pages()` |
| Dtype/Layout/Memory | = input |
| Why mesh-allocated | Uniform buffer address across devices lets the Phase-A fabric `write_page` target the NEIGHBOUR's gather_buffer through the LOCAL accessor base address routed one hop (`all_reduce.py:186-197`). |

## Dataflow Strategy

Two ordered `ttnn.generic_op` dispatches share the device command queue, so Phase A completes on
device `i` before Phase B reads its gather_buffer — **no cross-device barrier between phases**.

```
Phase A (fabric gather — 2 worker cores per device):
  DRAM input shard ──reader(NCRISC)──> cb_relay_pages ──writer(BRISC)──> fabric 1 hop ──> neighbour's gather_buffer (DRAM)
  neighbour-landed blocks ──reader reads back out of local gather_buffer──> cb_relay_pages ──writer──> next hop (store-and-forward)
  own shard ──forward reader self-copy (local NoC via cb_self_copy scratch)──> own gather_buffer block i

Phase B (local slice reduce — full compute grid, no fabric):
  gather_buffer ──reader(NCRISC): per owned output position, N tiles (one per block) at the SLICE
  tile id from SliceRowWalker──> cb_gathered_slices ──compute(TRISC): sum_blocks(N→1)──>
  cb_summed_slice ──writer(BRISC): dense output page──> DRAM output (shape[dim]/N)
```

### Phase A — Tensix-to-Tensix / device-to-device contract (identical to all_reduce Phase A)

| Item | Contract |
|---|---|
| Worker cores | `forward_core = (0, 0)` (flow rightward, fabric conn → chip i+1), `backward_core = (0, 1)` (flow leftward, → chip i−1). Each runs a reader (NCRISC) + writer (BRISC). No compute kernel in Phase A. |
| Direction convention | `direction` CT arg: 0 = forward, 1 = backward (Python CCL convention; note the C++ ring writer uses the OPPOSITE — do not copy its `is_forward` handling). The fabric-connection rt-arg block's leading `has_forward` flag doubles as the send direction the kernel peeks. |
| Block flow (device i, direction d) | Forward writer sends block `i` (seed), then relays blocks `i−1 … 0` to chip i+1. Backward writer sends block `i`, then relays `i+1 … N−1` to chip i−1. Reader pushes the SAME block order into `cb_relay_pages`, so a single FIFO drain matches. |
| Self-copy | Forward reader only, every device: own input shard → own gather_buffer block `i` via `cb_self_copy` scratch (local NoC read + write per page). |
| Landing address | Every fabric write lands DIRECTLY in the downstream device's gather_buffer at the block's canonical range: `gb_page(c, p) = c*P + p`, addressed via the local `TensorAccessor` + `write_page` (uniform mesh address). |
| Sync | ONE op-internal GlobalSemaphore (counting). After each full block lands, the sending writer's `arm_inc(1)` channel incs the semaphore ON THE RECEIVER's same-role core (`(0,0)` for forward flow, `(0,1)` for backward). Receiver's reader does `noc_semaphore_wait_min(sem_ptr, k)` before reading block k back out for relay; a line-end reader (no targets) waits for all `num_relay_blocks` then stops. |
| Ordering guarantee | The counting inc is issued on the SAME fabric connection after the block's pages — in-order on the connection, so the inc lands after the data. |
| Cache re-arm | Reader executes `noc_semaphore_set(sem_ptr, 0)` after its LAST wait (both directions, including line ends). See Key Risks §R1. |
| Line ends | A direction with `my_num_targets == 0` opens NO fabric connection, and its writer gets an EMPTY rt-arg list (the `else: []` branch) — a different rt-arg COUNT is a distinct program hash, which is correct (different device role). |

### Phase B — per-device local pipeline (per work core)

| Stage | Contract |
|---|---|
| Reader | For each owned output position `t ∈ [start, start+n)`: compute the slice tile id `id = walker.next()` ONCE, then read the N gathered tiles `gather_buffer[c*P + id]` for `c = 0..N−1` in block order into one `cb_gathered_slices` reservation of N pages; one `noc_async_read_barrier`; push N. |
| Compute | `binary_op_init_common(cb_gathered_slices, cb_gathered_slices, cb_summed_slice)` once, then per position: `compute_kernel_lib::sum_blocks(cb_gathered_slices, cb_summed_slice, N, /*block_num_tiles=*/1, /*pop_input=*/true)`. The helper owns wait(N)/pop(N)/reserve(1)/push(1), the tile_regs lifecycle, DEST chunking, and the odd-N seed path. |
| Writer | Per position: wait 1 on `cb_summed_slice`, `noc_async_write` to output page `start + t` (dense — the output slice is dense by construction), barrier, pop 1. |
| Shared schedule | The Phase-B collective schedule degenerates to the per-core `(start_tile, num_tiles)` pair, computed ONCE on host by `ttnn.split_work_to_cores` and passed to all three kernels. All CB counts derive from the same `n`; there is nothing else to drift. |

### Slice addressing — ONE definition, from the shared schedule header

The Phase-B reader includes `ttnn/operations/ccl/shared_with_host/ccl_helpers_schedule.hpp`
(kernel include path verbatim from `ring_reduce_scatter_minimal_async_reader.cpp:9`) and walks the
slice with `ttnn::ccl::schedule::SliceRowWalker` — the same type the silicon-verified C++
reduce_scatter kernels use for slice tile ids. The host passes three per-device quantities that
generalize the walker across scatter dims (all in tiles; `Ht = H/32`, `Wt = W/32`):

| Quantity | dim = 3 (Phase-0) | dim = 2 (refinement) | dim = 1 (refinement) | Meaning |
|---|---|---|---|---|
| `slice_base` | `i * (Wt / N)` | `i * (Ht / N) * Wt` | `i * (C / N) * Ht * Wt` | first tile id of device i's slice = `slice_tile_offset(dim, i, slice_C, slice_Ht, slice_Wt)` (schedule hpp:466-478) |
| `slice_run_len` | `Wt / N` | `(Ht / N) * Wt` | `(C / N) * Ht * Wt` | contiguous tile run inside the slice (walker ctor param 1, "slice_Wt") |
| `slice_stride` | `Wt` | `Ht * Wt` | `C * Ht * Wt` | tile-id jump between consecutive runs (walker ctor param 2, "tensor_Wt") |

Per-core seeding for a core owning `[start, start+n)` (walker API: ctor schedule hpp:498,
`set_base` :502, `reset_offsets(pages_read_in_row, row_offset)` :510, `next()` :516):

```cpp
sched::SliceRowWalker walker(slice_run_len, slice_stride);
walker.set_base(slice_base);
walker.reset_offsets(start % slice_run_len, (start / slice_run_len) * slice_stride);
// per position: const uint32_t id = walker.next();   // ONCE per position — see Key Risks §R5
```

`static_assert(ttnn::ccl::schedule::is_supported_scatter_dim(dim))` (schedule hpp:460) in the
Phase-B reader turns the host predicate into a compile-time gate. Batches need NO special
handling for dim 3: the walker's run/stride wrap walks all `B*C*Ht` tile rows seamlessly
(there is no per-batch cursor in this algorithm — the ring header's per-batch-restart trap does
not exist here).

Dim-3 correctness argument (implementer sanity check): flattened input tile id
`= ((b*C + c)*Ht + h)*Wt + w`; device i keeps `w ∈ [i*sWt, (i+1)*sWt)` with `sWt = Wt/N`; dense
output id `= ((b*C + c)*Ht + h)*sWt + (w − i*sWt)`. Walker at position `t`:
`id = i*sWt + (t / sWt)*Wt + (t % sWt)` — exactly the inverse map. Output page = `start + t`.

## Kernel Sources & Phase Selection

| File | RISC | Phases | Notes |
|---|---|---|---|
| `kernels/reduce_scatter_reader.cpp` | NCRISC (`ReaderConfigDescriptor`) | A + B, selected by leading `phase` CT arg (`if constexpr`) | Phase A = gather reader (self-copy / seed / relay read-back / counting waits / sem re-arm). Phase B = slice reader (SliceRowWalker + N reads per position). |
| `kernels/reduce_scatter_writer.cpp` | BRISC (`WriterConfigDescriptor`) | A + B, phase CT arg | Phase A = fabric egress via `FabricStreamSender`. Phase B = dense output page writes. |
| `kernels/reduce_scatter_compute.cpp` | TRISC (`ComputeConfigDescriptor`) | B only | `binary_op_init_common` + `sum_blocks` loop. |

Shared-source rule (from `all_reduce_program_descriptor.py:22-30`): both phases of a shared source
use a UNIFORM compile-time-arg superset — a fixed count of scalar CT args after `phase`
(zero-padded where a phase needs fewer), then a fixed number of `TensorAccessorArgs` (reader: 2 —
input+gather_buffer for A, gather_buffer+output for B with the 2nd `[[maybe_unused]]`; writer: 1 —
gather_buffer for A, output for B). `get_compile_time_arg_val` static-asserts on the index even in
the discarded `if constexpr` branch, so the superset is load-bearing. Phase B's reader needs 7
scalars (`cb_gathered_slices, num_devices, pages_per_shard, dim, slice_base, slice_run_len,
slice_stride`), which matches Phase A's 7 (`cb_relay_pages, cb_self_copy, direction, my_chip_id,
ring_size, num_targets_fwd, num_targets_bwd`) — keep the superset at **7 scalars** on both kernels.

Compute config (Phase B): `math_fidelity=HiFi4, fp32_dest_acc_en=True, math_approx_mode=False,
dst_full_sync_en=False` — fp32 DEST accumulation covers both the bf16 sum-of-N rounding budget and
the float32 dtype (mirrors `all_reduce_program_descriptor.py:436-443`).

## Work Distribution

| Field | Phase A | Phase B |
|-------|---------|---------|
| Work unit | one shard block (P pages) per direction | one output tile position (N input tiles → 1 output tile) |
| Grid | 2 fixed cores: `(0,0)` forward, `(0,1)` backward | `ttnn.split_work_to_cores(compute_with_storage_grid_size(), S)` where `S = P / N` output pages |
| Per-core work | forward: seed + `num_targets_bwd` relays (+ self-copy); backward: seed + `num_targets_fwd` relays | group-1 cores get `ceil`, group-2 `floor`; core k owns `[start_k, start_k + n_k)` with `start` accumulated in `ttnn.corerange_to_cores(all_cores, num_cores, row_wise=True)` order |
| Remainder | n/a (block counts exact) | handled by the two-group split; cores beyond `num_cores` get no kernel |
| Descriptor granularity | one `ProgramDescriptor` per `MeshCoordinate(0, i)` (append via `mpd[MeshCoordinateRange(coord, coord)] = program`) for BOTH phases — per-device CT args (`my_chip_id`, `slice_base`) make each device's program distinct | same |

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Format | Producer | Consumer | Lifetime / Rationale |
|---------------|-------|-----------|-----------|--------|----------|----------|----------|
| `cb_relay_pages` | 16 | `aligned_page_size = round_up(page_size, l1_alignment)` | 2 (double buffer) | input dtype | Phase-A reader (seed + relay read-backs) | Phase-A writer (fabric `write_page`) | Phase A only, on both worker cores. Streaming one page at a time; writer does `noc_async_writes_flushed()` before `cb_pop_front` so the slot isn't reused before the fabric read completes. |
| `cb_self_copy` | 24 | `aligned_page_size` | 2 | input dtype | Phase-A forward reader (reserve-only SCRATCH) | — (never pushed/popped) | Phase A only. One `cb_reserve_back(cb_self_copy, 1)` then the write-ptr is reused per page for the local self-copy bounce. Intentionally never pushed — document, don't "fix". |
| `cb_gathered_slices` | 0 | `tile_size = output.buffer_page_size()` | `2 * N` (double-buffered block of N slice tiles) | input dtype | Phase-B reader (pushes N per position, block order c=0..N−1) | Phase-B compute (`sum_blocks` waits N, pops N) | Phase B only, on all work cores. Sized so the reader can stage position t+1 while compute reduces t. |
| `cb_summed_slice` | 16 | `tile_size` | 2 (double buffer) | output dtype | Phase-B compute (pushes 1 per position) | Phase-B writer (waits/pops 1) | Phase B only. |

CB sync ledger (push == wait/pop for every CB, per core, per launch):

| CB | Producer total | Consumer total |
|---|---|---|
| `cb_relay_pages` | `(1 + num_relay_blocks) * P` pages when `my_num_targets > 0`, else 0 | identical expression in the writer (same `if constexpr` guard) |
| `cb_self_copy` | 0 pushes (scratch) | 0 waits |
| `cb_gathered_slices` | `n * N` | `n` sum_blocks calls × wait/pop N |
| `cb_summed_slice` | `n` | `n` |

## API Mapping

Every mechanism — helper or raw — with verified file:line. Paths relative to repo root; the two
kernel-lib headers are `ttnn/cpp/ttnn/kernel_lib/ccl_helpers_dataflow.hpp` (= `dataflow.hpp` below)
and `ttnn/cpp/ttnn/kernel_lib/accumulate_helpers_compute.hpp` (= `accum.hpp`); the schedule header
is `ttnn/cpp/ttnn/operations/ccl/shared_with_host/ccl_helpers_schedule.hpp` (= `sched.hpp`,
kernel include spelling `ttnn/operations/ccl/shared_with_host/ccl_helpers_schedule.hpp`).

| Phase | Type | Function | File:Line | Args / Template Params | Input CB | Output CB | Requirements |
|-------|------|----------|-----------|------------------------|----------|-----------|--------------|
| A writer | helper | `FabricStreamSender<>` ctor | dataflow.hpp:492 | `(size_t& conn_arg_idx, bool is_forward, uint32_t alignment)`; `ConnT = DirectConn` (default) | — | — | rt-arg block from `ttnn.setup_fabric_connection` FIRST at `conn_arg_idx`; leading `has_forward` flag peeked as the direction. Sender must OUTLIVE the stream (banner :81-84). |
| A writer | helper | `open(unicast_route(num_hops))` | dataflow.hpp:503 (open), :302-307 (`unicast_route`) | route bound ONCE at open | — | — | `num_hops` rt-arg from `ccl_dm_route(...).num_hops` (always 1 here — neighbour hop). |
| A writer | helper | `arm_unicast_write(page_size)` → `write_page(l1, c*P + p, gather_buffer)` | dataflow.hpp:423 (arm), :326-327 (write_page) | invariant per-page payload | `cb_relay_pages` (read ptr) | remote gather_buffer | `noc_async_writes_flushed()` before `cb_pop_front` (CB slot reuse guard). |
| A writer | helper | `arm_inc(1)` → `inc(neighbor_sem_noc_addr)` | dataflow.hpp:435 (arm), :368 (inc) | counting value invariant = 1 | — | remote semaphore | One inc per landed block, in-order after the block's pages on the same connection. |
| A writer | helper | `stream.close()` | dataflow.hpp:461 | — | — | — | Drains write + atomic barriers, idempotent; destructor closes if forgotten. |
| A reader | raw_api | `noc_async_read` / `noc_async_write` / `noc_async_read_barrier` / `noc_async_write_barrier` | dataflow_api.h | self-copy, seed staging, relay read-back | input / gather_buffer via `TensorAccessor` | `cb_self_copy`, `cb_relay_pages` | **Op-owned by the helper's documented split** — see "Helpers considered" below. |
| A reader | raw_api | `noc_semaphore_wait_min(sem_ptr, k)` + `noc_semaphore_set(sem_ptr, 0)` | dataflow_api.h | counting wait per relay block; re-arm after last wait | — | — | Re-arm is MANDATORY (dataflow.hpp:118-120 warning). |
| B reader | helper | `ttnn::ccl::schedule::SliceRowWalker` | sched.hpp:491 (class), :498 (ctor), :502 (`set_base`), :510 (`reset_offsets`), :516 (`next`) | `(slice_run_len, slice_stride)`; base = `slice_base` | — | — | ONE `next()` per output position, id reused for all N block reads. `shared_with_host` → legal in dataflow kernels (plain C++17). |
| B reader | helper | `slice_tile_offset` / `is_supported_scatter_dim` | sched.hpp:466-478 / :460 | `slice_base` formula lives on host; kernel `static_assert`s the dim | — | — | Host is the single evaluation site of the offset; kernel re-checks dim validity at compile time. |
| B reader | raw_api | `noc_async_read(gather_buffer.get_noc_addr(c*P + id), l1, page_size)` | dataflow_api.h | N reads per position, block-major | gather_buffer | `cb_gathered_slices` | One barrier + one `cb_push_back(cb_gathered_slices, N)` per position. |
| B compute | raw_api | `binary_op_init_common(cb_gathered_slices, cb_gathered_slices, cb_summed_slice)` | tt_metal compute API (`api/compute/eltwise_binary.h`) | hardware startup, once | — | — | Explicitly NOT owned by the helper (accum.hpp:70-77: `compute_kernel_hw_startup` and `binary_op_init_common` are NOT interchangeable; the C++ `sum_blocks` model kernel uses `binary_op_init_common` — `all_reduce_async/.../reduction.cpp:27`). |
| B compute | helper | `compute_kernel_lib::sum_blocks(cb_gathered_slices, cb_summed_slice, N, 1, /*pop_input=*/true)` | accum.hpp:221-222 (decl), accumulate_helpers_compute.inl:106-157 (impl) | `num_blocks = N` (runtime, mesh-derived), `block_num_tiles = 1`, `pop_input = true` | `cb_gathered_slices` (N tiles, block-major) | `cb_summed_slice` (1 tile) | Owns wait(N·1)/pop/reserve/push, tile_regs lifecycle, DEST chunking vs `DEST_AUTO_LIMIT` (inl:121-123), odd-N copy_tile seed. `pop_input=true` because this is a real producer/consumer CB (accum.hpp:200-205 — the llama_reduce_scatter mode, NOT all_reduce's resident-shell mode). Called in a loop, `n` times per core; each call re-inits, and no `BlockAccumulate` coexists, so the `@post` acc_to_dest note (accum.hpp:212-213) is moot. |
| B writer | raw_api | `noc_async_write(l1, output.get_noc_addr(start + t), page_size)` + barrier | dataflow_api.h | dense output pages | `cb_summed_slice` | output tensor | wait 1 / pop 1 per position. |
| host | helper | `ttnn._ttnn.fabric.ccl_dm_route(mesh_device, coord_i, coord_neighbour, topology)` | binding `ttnn/cpp/ttnn-nanobind/fabric.cpp:262-269`; impl `ttnn/cpp/ttnn/operations/ccl/common/host/ccl_helpers_dataflow_host.hpp:151` | per direction, per device | — | — | Owns the fabric fwd/bwd sign reversal + ring short-way. `route.num_hops` → writer rt-arg; `route.is_forward` → `has_forward` flag. |
| host | helper | `ttnn.setup_fabric_connection(src_fabric_id, dst_fabric_id, link_idx=0, program, core)` | binding fabric.cpp:142-167 | appended AFTER `ProgramDescriptor` construction (it mutates it — appends SemaphoreDescriptors) via the `_append_fabric_rt_args` idiom (`all_reduce_program_descriptor.py:78-92`) | — | — | Block layout `[has_forward][fwd args…][has_backward][bwd args…]`, placed at the END of the writer's rt args; kernel consumes with a cursor. |
| host | helper | `ttnn.create_global_semaphore` / `ttnn.get_global_semaphore_address` / `ttnn.synchronize_device` | `all_reduce.py:85-95` (idiom) | created on the full worker grid, initial 0 | — | — | Module-level cache keyed `id(mesh_device)`; `synchronize_device` ONCE inside the miss branch only. |
| host | helper | `ttnn.split_work_to_cores` / `ttnn.corerange_to_cores` | `all_reduce_program_descriptor.py:338-384` (idiom) | Phase-B split over `S` output pages | — | — | Same `(start, n)` pair feeds reader/compute/writer rt args. |

### Helpers considered and rejected (mandatory justifications)

| Candidate | Where it would apply | Rejection (concrete, cited) |
|---|---|---|
| `compute_kernel_lib::BlockAccumulate` (accum.hpp:111, `arm(cb_a, cb_b, cb_out, granularity)` :125) | Phase B sum | `arm` takes exactly TWO input CBs and each `run` adds `a + b` from two streams (inl:46-70). Here the N operands land block-major in ONE CB and N is runtime (mesh-derived, 2..8): the N-blocks-resident-in-one-CB shape is exactly what the free function `sum_blocks` exists for (accum.hpp:194-205 "out = the sum of num_blocks equal-shaped tile blocks RESIDENT in one CB — the all_reduce pattern"). `sum_blocks` IS the helper used; this is a helper-vs-helper selection, not a raw fallback. |
| `compute_kernel_lib::reduce()` (`reduce_helpers_compute.hpp:263-301`) | Phase B sum | `reduce()` performs WITHIN-TILE pooling — REDUCE_ROW/COL/SCALAR collapse the 32×32 tile dims and require a scaler CB (`reduce_helpers_compute.hpp:266-268, 274-276`). This op needs an element-wise N-way sum that PRESERVES tile shape; the tile dims must not be collapsed. Wrong operation class. |
| `FabricStreamSender::signal()` (dataflow.hpp:527) | Phase A per-block sync | `signal()` is TERMINAL — one inc then close ("do not also call open() on this sender afterwards", hpp:526). The Phase-A writer issues MANY packets + MANY incs across the relay loop, which is the documented staged `open → arm → issue → close` case (banner :86-91). The staged path IS used. |
| `FabricDuplexSender` (dataflow.hpp:799-824) | Phase A egress | Duplex fans every issue out to ALL connected directions (banner :44-61). Phase A's two directions send DIFFERENT block sequences from DIFFERENT cores (fwd: `i, i−1…0`; bwd: `i, i+1…N−1`), so a shared-issue duplex stream cannot express it; two unidirectional senders on two cores (the proven all_reduce shape) are correct. |
| `ttnn._ttnn.fabric.ccl_packet_dims` (fabric.cpp:264-265) | Phase A packet framing | Deliberately unused: the primary path uses 1:1 page↔packet framing with `aligned_page_size` as the on-wire payload, same as all_gather/all_reduce (`all_reduce/op_design.md:206`). Available for multi-page coalescing as a perf refinement. |
| `RingRsSchedule` / `ring_rs_step_flags` / `RingSliceCursor` / `LineChannelWalk` / `SyncCadence` (sched.hpp:223-383, :588-607, :627-753) | whole-op schedule | These model the N-1-step ring/line transfer walks with per-step reduce/forward flags (banner :58-76). The gather-then-reduce algorithm HAS no multi-step transfer schedule — the prompt's blessed simplest form. The schedule header is still the single source of slice addressing (`SliceRowWalker`, `slice_tile_offset`, `is_supported_scatter_dim`). A future ring implementation is the documented refinement (§R12). |
| `SequentialTileWalker` (sched.hpp:543, `next()` = `base_ + offset_++` :556) | Phase B writer output ids | The output slice is dense by construction; the walker's `next()` is literally `start + t`. Plain dense indexing in the writer is the same arithmetic with no cross-kernel agreement at stake (the shared contract is the per-core `n`, enforced by CB counts). Either is acceptable; the design specifies plain `start + t` to match the proven all_reduce Phase-B writer. |
| A `FabricStreamReceiver` | Phase A ingress | Does not exist by design — "The receive INGRESS is likewise a local NoC read the op owns; there is no FabricStreamReceiver" (dataflow.hpp:112-120). The relay read-back `noc_async_read` and the `noc_semaphore_wait_min` are the documented op-owned halves. |

## Compute Phases (Phase B compute kernel, per work core)

| # | Operation | Helper? | Input CB (tiles, state) | Output CB (tiles) | CB State After |
|---|-----------|---------|--------------------------|-------------------|----------------|
| 0 | `binary_op_init_common(cb_gathered_slices, cb_gathered_slices, cb_summed_slice)` | raw (hardware startup, kernel-owned per accum.hpp:70-77) | — | — | — |
| 1..n | `sum_blocks(cb_gathered_slices, cb_summed_slice, N, 1, true)` — one call per owned output position | helper | `cb_gathered_slices`: N tiles (block-major, pushed by reader), fully consumed | `cb_summed_slice`: 1 tile | `cb_gathered_slices` empty (popped before the output push); `cb_summed_slice` drained by the writer |

## Broadcast Verification

The only binary op is `add_tiles` inside `sum_blocks` — full-tile element-wise, no broadcast form.

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast Dim |
|-------|-----|-------------------|-------------------|---------------|
| B compute | `add_tiles` (via `sum_blocks`) | `cb_gathered_slices` full tiles | `cb_gathered_slices` full tiles (other block offset) | None |

## Registry / Entry-Point Contract (module layout)

```
ttnn/ttnn/operations/reduce_scatter/
  __init__.py                            # re-export: reduce_scatter, SUPPORTED, EXCLUSIONS, INPUT_TAGGERS
  reduce_scatter.py                      # entry point, validate(), registry, semaphore cache
  reduce_scatter_program_descriptor.py   # Phase A + Phase B MeshProgramDescriptor builders
  kernels/
    reduce_scatter_reader.cpp
    reduce_scatter_writer.cpp
    reduce_scatter_compute.cpp
  op_design.md                           # this file
```

| Item | Decision |
|---|---|
| `SUPPORTED` | `{"dtype": [ttnn.bfloat16, ttnn.float32], "layout": [ttnn.TILE_LAYOUT], "topology": [_Topology.Linear], "dim": [3]}`. **`"dim"` MUST be a key even though Phase-0 has one value** — the golden harness derives xfail marks by iterating SUPPORTED (feature_spec.py:44-52); a missing axis surfaces unimplemented `dim=2` as a hard failure instead of the expected `UnsupportedAxisValue`. |
| `EXCLUSIONS` | `[]` |
| `INPUT_TAGGERS` | `{}` (golden INPUTS are chosen valid for every TARGET dim; no shape-derived axis needed) |
| Refusal types | `from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue` with the ImportError fallback subclassing `NotImplementedError` (`all_reduce.py:39-47` verbatim pattern) |
| `validate()` order | 1) structural → `ValueError`: MeshDevice, `(1, N)` line view, N ≥ 2, rank 4, not sharded, `H % 32`/`W % 32`, canonical-dim divisibility `shape[dim] % (N*32) == 0`, 16-byte page guard, output_tensor spec (shape with `[dim]//N`, dtype, layout, buffer_type). 2) axis gate → `UnsupportedAxisValue` per axis (`dtype, layout, topology, dim`), then `ExcludedCell` scan. Returns `(num_devices, canonical_dim)`. |
| dim canonicalization | `dim if dim >= 0 else dim + 4` BEFORE the divisibility check and the membership test (positive convention pinned by feature_spec; `-1 ≡ 3`, `-2 ≡ 2`). |
| Semaphore | Module-level `_SEMAPHORE_CACHE` keyed `id(mesh_device)`; miss branch: create on full worker grid (initial 0) + ONE `ttnn.synchronize_device(mesh_device)`; park on the Phase-A descriptor via `gather_mpd.semaphores = [sem]` (guard with `hasattr` for older bindings). Excluded from the program-cache hash; copied into the cached workload's `shared_variables` on miss. NO per-call post-dispatch barrier. |
| Dispatches | `ttnn.generic_op([input_tensor, gather_buffer], gather_mpd)` then `ttnn.generic_op([gather_buffer, output_tensor], reduce_mpd)` — the LAST tensor in `io_tensors` is the output. Queue order IS the phase barrier. |

## Feature spec (pipeline mode)

`eval/golden_tests/reduce_scatter/feature_spec.py` **already exists and is authoritative** — do not
edit it. TARGET: `dtype [bf16, fp32] × layout [TILE] × topology [Linear, Ring] × dim [3, 2]`;
INPUTS: `(1,1,256,256), (1,1,256,512), (2,1,256,256)` (widths/heights multiples of 256 = lcm(tile,
8 devices) so every TARGET dim stays tile-aligned on both the (1,8) sim mesh and the (1,4) hardware
box); `INVALID = []`.

**Structural impossibilities**: none beyond the existing empty INVALID — every Phase-0 axis
combination (float dtypes × TILE × topology × dim) is constructible; `Ring` topology and `dim=2`
are kernel/host refinements (EXCLUSIONS-class, already outside SUPPORTED), not universe changes.

## Program Cache & Semaphore Lifecycle

| Item | Behaviour |
|---|---|
| Hash inputs | Per kernel: source path, CT args, core ranges, config, rt-arg COUNTS (not values); per CB: total_size, ranges, formats (`generic_op_device_operation.cpp:48-108`). Same shape/dtype/topology/dim → both dispatches hit. |
| gather_buffer per call | Allocated fresh each call → new addresses on a cache hit. Safe: rt-arg VALUES are re-pushed by `override_runtime_arguments`; sizes/CT args unchanged. |
| Semaphore survival | GlobalSemaphore created once (module cache) + parked on the descriptor (excluded from hash, held in `shared_variables`). The acceptance test's cache-hit case fails/hangs if it is re-created per call or if the kernel re-arm (§R1) is missing. |
| Distinct hash per device role | Line-end writers have rt-arg count 0 vs interior 7+conn-block — intentional distinct entries. |

## Key Risks and Gotchas

| # | Risk | Mitigation |
|---|------|------------|
| R1 | **Cache-reuse semaphore re-arm.** Programs are cached and the GlobalSemaphore reused: first run green, second hangs/corrupts. | Phase-A reader executes `noc_semaphore_set(sem_ptr, 0)` after its LAST wait, in BOTH the relay and the pure-receiver (line-end) paths (dataflow.hpp:118-120; `all_reduce_reader.cpp:115-116`). The acceptance test's program-cache case exists to catch this. |
| R2 | **Phase ordering is queue-ordered, not barrier-ordered.** | Two `generic_op` dispatches on the same CQ; Phase A completes on device i before Phase B runs there. Do NOT reorder the dispatches; do NOT drop Phase A's read-side counting waits (they are what guarantees the DATA landed before relay/read). |
| R3 | **Uniform mesh buffer address assumption.** Fabric `write_page` targets the neighbour's gather_buffer through the LOCAL accessor + 1-hop route. | gather_buffer/output MUST be mesh-allocated (`allocate_tensor_on_device` on the mesh) — never per-device allocations. |
| R4 | **Block-major push order into `cb_gathered_slices`.** `sum_blocks` reads block c at CB index `c * block_num_tiles`; with `block_num_tiles = 1`, tile c must be the c-th page of the reservation. | Reader loops `c = 0..N−1` within one N-page reservation, single barrier, single push of N. |
| R5 | **One `next()` per output position.** `SliceRowWalker::next()` returns AND advances; calling it per block read (N times per position) silently mis-addresses with no hang. | `const uint32_t id = walker.next();` hoisted above the block loop. |
| R6 | **Reader-walk ↔ writer-dense agreement.** The reader's `t`-th pushed group must correspond to output page `start + t`. | Both derive from the SAME per-core `(start, n)` rt args; the walker seed formula `reset_offsets(start % run, (start / run) * stride)` is the only nontrivial piece — get it from this doc, not re-derived. |
| R7 | **`pop_input=true` on `sum_blocks`.** `cb_gathered_slices` is a real producer/consumer CB; the default `false` (all_reduce's resident-shell mode) deadlocks the reader on `cb_reserve_back` after the CB fills. | Explicit `true` (accum.hpp:200-205). |
| R8 | **CB slot reuse before fabric read.** `write_page` is async; popping the relay page immediately can recycle the L1 slot mid-read. | `noc_async_writes_flushed()` between `write_page` and `cb_pop_front` (`all_reduce_writer.cpp:95`). |
| R9 | **`cb_self_copy` is reserve-only scratch.** It is never pushed; a well-meaning "fix" adding push/pop desyncs nothing but wastes nothing either — leave as scratch per the proven kernel. | Documented here and in the kernel comment. |
| R10 | **Line-end writers must read NO rt args.** Their rt-arg list is `[]`; any unconditional `get_arg_val` before the `my_num_targets > 0` guard reads garbage. | The entire Phase-A writer body sits inside `if constexpr (my_num_targets > 0)` (`all_reduce_writer.cpp:57`). |
| R11 | **fp32_dest_acc_en halves DEST capacity** (DEST_AUTO_LIMIT 8 → 4, `dest_helpers.hpp:89-103`). | Irrelevant at `block_num_tiles = 1` (`sum_blocks` chunks internally, inl:121-123), but binding for any future granularity > 1 refinement. |
| R12 | **Perf refinements (recorded, not Phase-0):** slice-only gather (each device only needs S·N tiles, not P·N), packet coalescing via `ccl_packet_dims`, granularity > 1 `sum_blocks` blocks, true ring reduce-scatter on `RingRsSchedule` + `BlockAccumulate` (the C++ `ring_reduction.cpp` model), Ring topology, dim ∈ {1, 2} (host-only change: new `slice_base/run/stride` row). | op_requirements.md candidates for the verifier. |
| R13 | **Mesh/fabric contract in every test.** Verification topology is `bh_quietbox_1x4_hw`: mesh `(1, 4)`, `FABRIC_1D`, real Blackhole silicon via `scripts/run_multidevice_sim_pytest.py --runtime hardware --op reduce_scatter`. A different mesh shape hangs fabric init (`Fabric Router Sync: Timeout`) or fails `system_mesh.cpp: requested_size <= system_size`. The runner exports `MULTIDEV_SIM_MESH_SHAPE` per topology; the acceptance test reads it with default `(1, 4)`. |
| R14 | **Negative-dim aliases.** `dim=-1` must behave exactly as `dim=3` (positive-convention canonicalization BEFORE the membership test) — a literal `dim in SUPPORTED["dim"]` check rejects legal aliases. |
| R15 | **Topology import at module import time.** Use `from ttnn._ttnn.operations.ccl import Topology as _Topology`; the `ttnn.Topology` alias binds only after `ttnn.operations` auto-import (`all_reduce.py:35-37`). |

## Hardware Constraints checklist

- [x] CB sync: push == wait/pop for every CB (ledger above)
- [x] No reduce scaler (no `reduce()` usage — `sum_blocks` needs none)
- [x] DEST: `sum_blocks` chunks internally against `DEST_AUTO_LIMIT` (= 4 under fp32_dest_acc_en); `block_num_tiles = 1` uses one DST tile
- [x] No sequential-helper intermediate CBs (single compute helper per position)
- [x] Page sizes: tile CBs use `buffer_page_size()` of TILE tensors; relay CB uses l1-aligned page size (16-byte guard in validate)
- [x] All `cb_wait_front` calls on a given CB use one page count (`cb_relay_pages`: 1; `cb_gathered_slices`: N; `cb_summed_slice`: 1)
- [x] Helpers not wrapped with extra CB ops (`sum_blocks` and the fabric channels own their protocols end-to-end)
- [x] Hardware startup (`binary_op_init_common`) before the first helper call, kernel-owned
