# Operation Design: reduce_scatter_average

## Overview

| Field | Value |
|---|---|
| Classification | CCL (multi-device collective) + compute (reduction + scaling epilogue) |
| Goal | Device i's output = slice i (of N equal slices along `dim`) of `(shard_0 + … + shard_{N-1}) / N` |
| Math | `out_i = (1/N) · Σ_j shard_j[slice_i]`, element-wise over tiles; scaling is part of the op |
| Algorithm | **One-dispatch bidirectional LINE reduce-scatter** (partial-sum store-and-forward chain), modeled on the hardware-green C++ `line_reduce_scatter_minimal_async` kernel triple, with the 1/N scale fused into the final reduction chunk. NOT the reference Python `reduce_scatter`'s two-dispatch gather-then-reduce (its named weakness). |
| Dispatches | **ONE** `ttnn.generic_op([input, intermediate, output], mesh_pd)` per invocation. Compute overlaps fabric arrival via the `out_ready` semaphore + `SyncCadence` chunk protocol. |
| Mode | Derivative — kernels are NEWLY AUTHORED under `ttnn/ttnn/operations/reduce_scatter_average/kernels/`, modeled kernel-for-kernel on `ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/line_reduce_scatter_minimal_async_{reader,writer}.cpp` + `line_reduction.cpp`, simplified (DirectConn instead of MuxConn, 1 link, 1 worker/direction, no sharded paths, no fused-matmul path) and extended with the averaging epilogue and a cross-call readiness handshake. Host assembly follows the four reference Python ops' generic_op idioms. |
| References | `reduce_scatter_minimal_async_program.cpp:972-1556` (line factory), `ttnn/ttnn/operations/reduce_scatter/reduce_scatter_program_descriptor.py` (generic_op host idioms), `ttnn/ttnn/operations/point_to_point/` (reader-side signal / ready-handshake precedent), `strided_reduce_scatter_async/device/kernels/minimal_ring_reduction.cpp:129-231` (fused scalar epilogue + `rearm()` precedent) |

### Why this algorithm (decision, not deliberation)

| Consideration | Decision |
|---|---|
| Mandate | One `generic_op`, compute overlapping fabric arrival. The reference Python `reduce_scatter`'s two-dispatch split is explicitly forbidden; its own `op_requirements.md:91-100` names the line/ring schedule-driven algorithm as the fix. |
| Correctness risk | Contained by construction: the reader, compute, and writer each construct the SAME `LineChannelWalk` + per-batch `LineSliceCursor` from `ccl_helpers_schedule.hpp` (host-unit-tested in `tests/ttnn/unit_tests/gtests/ccl/test_ccl_helpers_schedule.cpp`), so chunk boundaries — and therefore the CB protocol — cannot drift. A hardware-green C++ reference exists for every code path. |
| Traffic / memory | O(1) slices per link per step (vs the reference's ~N× full-shard gather traffic); intermediate buffer is 2× one shard (vs N× `gather_buffer`). |
| Averaging placement | Exactly ONE final reduction per device is the LAST write to the output; scale that chunk by 1/N inside its DEST window (see "Averaging Placement Verification"). |

## Parameters

| Name | Type | Required | Valid Range | Default | CT-RT |
|---|---|---|---|---|---|
| `input_tensor` | `ttnn.Tensor` | yes | rank-4, TILE, interleaved DRAM/L1, bf16/f32, on a MeshDevice line of N≥2 | — | RT (addresses) |
| `dim` | `int` | no | Phase-0: canonical 3 (accepts `-1`); TARGET adds 2 | `3` | CT |
| `topology` | `ttnn.Topology` | no | Phase-0: `Linear`; TARGET adds `Ring` | `Topology.Linear` | host-only |
| `output_tensor` | `ttnn.Tensor \| None` | no | must match derived output spec (shape/dtype/layout/buffer_type) | `None` | RT (address) |

**Import note (§R15 of the reference ops):** the op module must import `from ttnn._ttnn.operations.ccl import Topology as _Topology` and default `topology=_Topology.Linear`. Referencing `ttnn.Topology` at module scope raises during `ttnn.operations` auto-import (`pkgutil.walk_packages` ordering).

**dim canonicalization (POSITIVE convention, per `feature_spec.py`):** `canonical_dim = dim if dim >= 0 else dim + 4` (rank pinned to 4 by a structural check first), applied BEFORE the SUPPORTED membership test — a literal `dim in SUPPORTED["dim"]` check would wrongly reject the legal alias `-1`.

## Tensors

### Input (per device)

| Property | Requirement |
|---|---|
| Shape | `(B, C, H, W)`, H and W multiples of 32; `shape[canonical_dim] % (N·32) == 0` (reject loudly with ValueError, never pad) |
| Dtype | `bfloat16` (primary), `float32` |
| Layout | TILE_LAYOUT |
| Memory | Interleaved, DRAM or L1 |
| Distribution | One SAME-shape shard per device on a `(1, N)` MeshDevice line, `FABRIC_1D` |

### Output (per device)

| Property | Value |
|---|---|
| Shape | input shard shape with `shape[canonical_dim] //= N` (dim 3 → `(B, C, H, W/N)`) |
| Dtype / Layout / Memory | same as input |
| Content | device i holds slice i of the averaged tensor |
| Allocation | `output_tensor` if supplied (validated, returned by handle), else `ttnn.allocate_tensor_on_device(...)` with the input's memory config |

### Intermediate (op-internal, allocated fresh per call)

| Property | Value |
|---|---|
| Shape | `(2·B, C, H, W)` — input shard shape with dim0 doubled. FWD partials land at pages `[0, P)`, BWD at `[P, 2P)` where `P = input_num_pages`; kernels compute `intermediate_full_offset = is_forward ? 0 : P` (mirrors `line_reduce_scatter_minimal_async` reader/writer) |
| Dtype / Layout / Memory | same as input |
| Allocation | `ttnn.allocate_tensor_on_device(...)` on the **mesh** — the uniform-across-devices buffer address is load-bearing: the fabric writer targets the neighbour's buffer through the LOCAL accessor's base address routed one hop (same invariant as reference reduce_scatter §R3) |
| generic_op position | `ttnn.generic_op([input_tensor, intermediate_tensor, output_tensor], mesh_pd)` — output LAST |

## Dataflow Strategy

### Algorithm — bidirectional line partial-sum chain (per device i, N devices, `dim=3`)

Geometry (host-derived, all in tiles): `Ht=H/32`, `Wt=W/32`, `slice_Wt=Wt/N`, `slice_C=C`, `slice_Ht=Ht`; `P = B·C·Ht·Wt`; `input_batch_num_pages = P/B`; `input_channel_num_pages = Ht·Wt`; `output_channel_num_pages = Ht·slice_Wt`; `output_batch_num_pages = C·Ht·slice_Wt`.

Two worker cores per device, each running reader (NCRISC) + compute (TRISC) + writer (BRISC):

| Core | Direction | Slice walk (`LineSliceCursor`) | Forwards | Final reduction |
|---|---|---|---|---|
| `(0,0)` | FORWARD (sends → device i+1) | slices `N-1, N-2, …, i+1` (`num_targets_fwd = N-1-i`) | partial(slice s) = incoming_fwd(s) + input(s), fabric-forwarded to i+1's FWD interm region; device 0 forwards raw input (bypass) | slice i, unless `is_first_device_in_direction` (device 0) |
| `(0,1)` | BACKWARD (sends → device i−1) | slices `0, 1, …, i-1` (`num_targets_bwd = i`) | mirrored, into i−1's BWD interm region; device N−1 forwards raw input (bypass) | slice i, unless device N−1 |

Per-device role table (`sync_with_other_direction = !(i==0 || i==N-1)`, a CT arg on reader+writer):

| device | dir | `is_first_device_in_direction` | `num_targets` | `do_final_reduction` | `num_total_reduction_steps` | final mode | scaled? |
|---|---|---|---|---|---|---|---|
| 0 | fwd | **true** | N−1 | false | **0** (compute idle; bypass CB) | — | — |
| 0 | bwd | false | 0 | true | 1 | `out = input(slice 0) + interm_bwd` | **yes** |
| mid k | fwd | false | N−1−k | true | N−k | `out = input(slice k) + interm_fwd` (= Σ shards 0..k), then **hands off** per chunk | no |
| mid k | bwd | false | k | true | k+1 | `out += interm_bwd` — reads the OUTPUT tensor as its local operand (`line_rs_accumulate_output`) | **yes** |
| N−1 | fwd | false | 0 | true | 1 | `out = input(slice N−1) + interm_fwd` | **yes** |
| N−1 | bwd | **true** | N−1 | false | **0** | — | — |

`scale_output = do_final_reduction && !line_rs_forward_hands_off(sync_with_other_direction, is_forward)` — exactly one direction per device scales, and it is always the direction performing the LAST write to the output.

### Per-chunk pipeline (the overlap)

```
device i-1 FWD writer ──fabric write_page──▶ device i interm FWD region (DRAM)
                      ──same-stream inc────▶ device i FWD core out_ready_sem (L1)   [in-order after payload]
device i FWD reader:  if (cadence.wait_due()) noc_semaphore_wait_min(out_ready, ++target); cadence.advance();
                      read input chunk → cb_local_operand; read interm chunk → cb_partial_in
device i FWD compute: BlockAccumulate::run(n)  — already draining chunk k−1 while the reader blocks on chunk k
device i FWD writer:  cb_wait_front(cb_reduced) → fabric-forward (non-final) or local NoC write to output (final)
```

The compute kernel never learns which phase it is in: the READER decides what fills `cb_local_operand` (input vs output read-back) and the WRITER decides where `cb_reduced` goes (fabric vs local output). The only compute-side distinction is the scaled final chunk (below).

### Tensix-to-Tensix / device-to-device contract

| Channel | From → To | Mechanism | Ordering guarantee |
|---|---|---|---|
| Partial-sum payload | device i dir-D writer → device i±1 dir-D interm region | `FabricStream::arm_unicast_write(page_size)` → `write_page(l1, interm_walker.next(), interm_addrgen)`, 1-hop unicast route | `noc_async_writes_flushed()` between `write_page` and `cb_pop_front` (CB slot reuse guard) |
| Arrival signal | same writer → neighbour's SAME logical core `out_ready_sem` | `arm_inc(1)` channel, `counter.inc(...)` per `SyncCadence.signal_due()` + `tail_due()` | same fabric connection as payload ⇒ inc lands after data |
| fwd→bwd handoff (mid devices) | FWD writer → BWD core `fwd_bwd_sem` (same device) | `noc_async_write_barrier()` then local `noc_semaphore_inc`, once per final chunk | full barrier (not flush) before inc — BWD reader reads those very output tiles back |
| Cross-call readiness | device j dir-D writer → upstream neighbour's OPPOSITE-direction core `peer_ready_sem` | one `counter.inc(...)` (same armed inc channel, different address) issued immediately after stream open, BEFORE its own peer_ready wait | closes the "call K+1 inc lands before call K's reader reset" race (see Program Cache section) |

Slice/tile addressing is agreed between the sender's writer and the receiver's reader by constructing character-for-character identical walkers: `SliceRowWalker(slice_Wt, input_tensor_Wt)` with base `slice_tile_offset(dim, slice, slice_C, slice_Ht, slice_Wt) + batch_offset + intermediate_full_offset`, `reset_offsets(start_pages_read_in_row, start_row_offset)` per channel, `bump_base(input_channel_num_pages)` per channel end.

## Work Distribution

| Field | Value |
|---|---|
| Work unit | One chunk of ≤ `tile_granularity` tiles of one (direction, slice-target-or-final, channel) walk step |
| Grid | 2 worker cores per device: FWD `(0,0)`, BWD `(0,1)` (logical; NoC coords via `mesh_device.worker_core_from_logical_core`, uniform across the mesh — the fabric inc targets the SAME logical core on the neighbour) |
| Per-core work | The direction's full slice range: 1 worker per direction ⇒ `start_tiles_read = 0`, `start_tiles_to_read = output_channel_num_pages`, `start_pages_read_in_row = 0`, `start_row_offset = 0` (host formulas mirror `reduce_scatter_program_utils.cpp:143-169`) |
| `tile_granularity` | **4** (uniform for bf16 and f32) — `fp32_dest_acc_en=True` clamps `DEST_AUTO_LIMIT` to 4 (`dest_helpers.hpp:103`); `BlockAccumulate::arm` asserts this |
| `chunks_per_sync` | host: `min(20, max(tiles_per_slice_per_worker // tile_granularity // 2, 1))` where `tiles_per_slice_per_worker = (start_tiles_to_read - start_tiles_read) · slice_C` (mirrors `reduce_scatter_program_utils.cpp:99-110` Linear default) |
| Remainder | Short final chunk per channel: `LineChannelWalk.tiles_this_chunk()` returns `n < tile_granularity`; math covers `n`, **CB protocol always runs at the full `tile_granularity`** (the C++ granule invariant — conflating them is a CB-wait deadlock) |
| Batches | `input_tensor_B = B` outer loop in ALL THREE kernels; `LineSliceCursor` constructed INSIDE the batch loop (per-batch restart — hoisting it is a silent wrong-slice bug) |
| Refinement (not Phase 0) | Multi-worker split per direction (`worker_id`-based tile ranges), multi-link, `arm_scatter_write` 2-page packet coalescing |

## Circular Buffers

All four CBs declared on BOTH worker cores (`CoreRange((0,0),(0,1))`), page size = `input_tensor.buffer_page_size()` (tile bytes: 2048 bf16 / 4096 f32, both 16-aligned), data format = input dtype. `num_pages = 3 · tile_granularity = 12` (triple-buffered granules, the C++ sizing at `reduce_scatter_minimal_async_program.cpp:1110-1117`).

| Semantic name | Index | Page size | Num pages | Producer | Consumer | Purpose / lifetime |
|---|---|---|---|---|---|---|
| `cb_local_operand` | 0 | tile bytes | 12 | reader | compute (SrcA) | Local operand of the reduction: input-slice tiles on forwarding + non-accumulate finals; OUTPUT read-back tiles on the accumulate-output final |
| `cb_partial_in` | 1 | tile bytes | 12 | reader | compute (SrcB) | Incoming partial sums, read back out of the local intermediate region after the `out_ready` wait |
| `cb_bypass` | 2 | tile bytes | 12 | reader | **writer** (directly) | First-device-in-direction only: raw input slices, no compute. The bypass is a CB retarget, not a separate code path — reader writes `cb_in0 = is_first_device_in_direction ? cb_bypass : cb_local_operand`; writer pops `is_first_device_in_direction ? cb_bypass : cb_reduced` in the forwarding loop |
| `cb_reduced` | 16 | tile bytes | 12 | compute | writer | Reduced partials (forwarding steps) and reduced+averaged tiles (scaled final) |

**CB sync ledger** (granules per batch; `CH = slice_C`, `K = ceil(per-worker channel tiles / G)` chunks/channel, `T = num_targets_in_direction`, `F = do_final_reduction`):

| CB | Producer pushes | Consumer waits/pops | Balanced |
|---|---|---|---|
| `cb_local_operand` | `(T+F)·CH·K` (non-first) / 0 (first) | compute: `steps·CH·K` where `steps = T+F` (non-first) / 0 (first) | ✓ |
| `cb_partial_in` | same as `cb_local_operand` | same | ✓ |
| `cb_bypass` | `T·CH·K` (first) / 0 (non-first) | writer forwarding loop, same expression | ✓ |
| `cb_reduced` | compute `(T+F)·CH·K` | writer: `T·CH·K` (forwarding) + `F·CH·K` (final) | ✓ |

Every `cb_reserve_back` / `cb_push_back` / `cb_wait_front` / `cb_pop_front` on every CB uses exactly `tile_granularity` pages.

## Semaphores

Three **GlobalSemaphores**, created ONCE per mesh (`_SEMAPHORE_CACHE` keyed by `id(mesh_device)`, `ttnn.create_global_semaphore(mesh_device, worker_cores, 0)` + one `ttnn.synchronize_device` inside the miss branch only), parked as `mesh_pd.semaphores = [out_ready_sem, peer_ready_sem, fwd_bwd_sem]`, addresses via `ttnn.get_global_semaphore_address(...)` baked into RT args. One shared L1 address per sem; FWD and BWD cores hold independent counters because they are different cores.

| Sem | Inc'd by | Waited by | Reset (re-arm for program-cache reuse) |
|---|---|---|---|
| `out_ready_sem` | upstream neighbour's same-direction writer, fabric `counter.inc`, per `SyncCadence` | local reader, `noc_semaphore_wait_min(out_ready, ++sem_target)` — `sem_target` monotonic across slices, finals, and batches | reader, unconditionally, at kernel end: `noc_semaphore_set(out_ready, 0)` (receiver-after-wait) |
| `peer_ready_sem` | downstream neighbour's OPPOSITE-direction writer (1 fabric inc at its start) | local writer (`num_targets > 0` only), `noc_semaphore_wait_min(peer_ready, 1)` BEFORE first payload send | writer, immediately after its wait: `noc_semaphore_set(peer_ready, 0)` |
| `fwd_bwd_sem` | FWD writer, local `noc_semaphore_inc` to the BWD core, once per final chunk (`hands_off` only) | BWD reader, `noc_semaphore_wait_min(fwd_bwd, ++fwd_sync_cnt)` per accumulate-final chunk, at the TOP of the chunk body | BWD reader, at kernel end, when `line_rs_accumulate_output(sync, is_forward)`: `noc_semaphore_set(fwd_bwd, 0)` |

**Readiness handshake ordering (deadlock rule):** every writer with `num_targets > 0` must `counter.inc(peer_ready @ opposite-core coords on its downstream neighbour)` **BEFORE** waiting on its own `peer_ready_sem`. Both neighbours signal unconditionally first, then wait — signal-then-wait is what makes the pairwise handshake deadlock-free. Rationale: a device's program K+1 kernels only start after program K fully completes (per-device CQ order), so receiving `peer_ready` proves the receiver's previous-call semaphore resets have executed; without it, device 0's FWD core (which never waits on anything) finishes call K almost instantly and its call-K+1 incs can land before the neighbour's call-K reader reset destroys them — the classic "green run 1, hang run 2". This replaces the C++ multicast startup barrier (`writer:180-190`) with only proven Python-side plumbing (p2p's reader `signal()` precedent, inverted onto the writer's existing stream — zero extra connections, zero multicast routes).

No batch-ready semaphore (the line variant of the C++ op has none). No multicast barrier.

## API Mapping

| Phase | Type | Function | File:Line | Key params | Input CB | Output CB | Owns CB ops? |
|---|---|---|---|---|---|---|---|
| Fabric egress (writer) | helper | `FabricStreamSender<>(size_t& conn_arg_idx, bool is_forward, uint32_t alignment)` | `ccl_helpers_dataflow.hpp:492` | DirectConn; conn block appended LAST in RT args, `has_forward` peeked at `conn_arg_idx`; **sender declared before the stream (lifetime)** | — | — | n/a |
| Fabric egress | helper | `FabricStream<> open(route)` / `unicast_route(num_hops=1)` | `:503` / `:302` | route bound ONCE at open | — | — | n/a |
| Payload | helper | `arm_unicast_write(page_size)` → `write_page(l1, page_idx, interm_addrgen)` | `:423` | one tile per packet (no scatter coalescing in Phase 0) | pops `cb_reduced`/`cb_bypass` (kernel-owned) | — | no — kernel owns wait/pop |
| Arrival + readiness incs | helper | `arm_inc(1)` → `counter.inc(noc_addr)` | `:435` | ONE armed channel serves both `out_ready` and `peer_ready` (value invariant, address per-issue) | — | — | n/a |
| Teardown | helper | `stream.close()` | `:461` | drains then closes; idempotent | — | — | n/a |
| Slice schedule (all 3 kernels) | helper | `LineChannelWalk(slice_C, tile_granularity, start_tiles_read, start_tiles_to_read)` | `ccl_helpers_schedule.hpp:690` | `reset()` per phase; `next_channel()`/`next_chunk()`/`tiles_this_chunk()`; no skip, chunks never empty | — | — | n/a |
| Slice schedule (reader+writer) | helper | `LineSliceCursor(is_forward, ring_size)` | `:627` | constructed INSIDE the batch loop; fwd decrements from N−1, bwd increments from 0 | — | — | n/a |
| Sync cadence (reader+writer) | helper | `SyncCadence(chunks_per_sync)` | `:588` | reader: `wait_due()` before `advance()`; writer: `advance()` then `signal_due()`, `tail_due()` after the channel loop; `reset()` once per slice target and once before the final block | — | — | n/a |
| Final-mode split | helper | `line_rs_accumulate_output(sync, is_forward)` / `line_rs_forward_hands_off(sync, is_forward)` | `:651` / `:656` | one shared definition of who accumulates vs who hands off | — | — | n/a |
| Tile addressing | helper | `slice_tile_offset(dim, slice, slice_C, slice_Ht, slice_Wt)`, `SliceRowWalker(slice_Wt, input_tensor_Wt)`, `SequentialTileWalker`, `rebase_row_offset` | `:466` / `:491` / `:543` / `:663` | accumulate-final reads output with `SliceRowWalker(slice_Wt, slice_Wt)`, base `b·output_batch_num_pages`, row0 = `rebase_row_offset(start_row_offset, input_tensor_Wt, slice_Wt)`, channel bump `output_channel_num_pages` | — | — | n/a |
| dim gate | helper | `is_supported_scatter_dim(dim)` | `:460` | `static_assert` in reader/writer | — | — | n/a |
| Reduction (compute) | helper | `BlockAccumulate::arm(cb_local_operand, cb_partial_in, cb_reduced, tile_granularity)` then `acc.run(n)` per chunk | `accumulate_helpers_compute.hpp:125` / `:132` | prerequisite `binary_op_init_common(cb_local_operand, cb_partial_in, cb_reduced)` at boot (kernel-owned); CB protocol at granularity, math at `n` | `cb_local_operand`, `cb_partial_in` | `cb_reduced` | **yes** — wait/pop/reserve/push all inside `run()`; do NOT wrap |
| State restore (compute) | helper | `acc.rearm()` after every scaled-final chunk | `:175` | restores `reconfig_data_format` + `add_tiles_init` clobbered by the epilogue | — | — | n/a |
| Averaging epilogue (compute) | **raw_api** | `binop_with_scalar_tile_init()` + `mul_unary_tile(idst, inv_n_bits)` inside a hand-rolled add+scale chunk | `tt_metal/hw/inc/api/compute/eltwise_unary/binop_with_scalar.h:151` / `:57` | `inv_n_bits = fp32 bit pattern of 1.0/N`, host-computed CT arg (`struct.pack('<f')`); SFPU on DEST while acquired | `cb_local_operand`, `cb_partial_in` | `cb_reduced` | kernel owns the whole chunk protocol (mirror of `run()`'s ordering, pops BEFORE reserve) |
| Interm/input/output access (dataflow) | raw_api (std) | `TensorAccessor(args, addr, page_size)` + `noc_async_read/write`, `noc_semaphore_wait_min/set/inc`, `safe_get_noc_addr` | `tech_reports/tensor_accessor/tensor_accessor.md`; usage per reference kernels | accessors chained: reader input→interm→output, writer interm→output | — | — | n/a |

### Helpers considered and rejected (mandatory justifications)

| Phase | Candidate helper | Why it cannot be used | Citation |
|---|---|---|---|
| Averaging epilogue | `BlockAccumulate::run/run_seeded/run_chunked` | No scaling hook anywhere in the class: `arm(cb_a, cb_b, cb_out, granularity)` and `run(num_tiles)` take no epilogue callable or scalar; the header explicitly assigns fused epilogues to the op: "any epilogue fused after the add (e.g. strided_reduce_scatter_async's addcmul — the op keeps that and re-arms afterwards)" | `accumulate_helpers_compute.hpp:125`, `:132`, `:65-68`; precedent `minimal_ring_reduction.cpp:129-231` (hand-rolled add + `mul_unary_tile` + `acc.rearm()` at `:226-231`) |
| Averaging epilogue | `sum_blocks(cb_in, cb_out, num_blocks, block_num_tiles, pop_input)` | Same absence of any scale parameter or post-hook; also wrong shape — it sums blocks resident in ONE CB, our operands stream in two CBs | `accumulate_helpers_compute.hpp:221-226` |
| Averaging epilogue | `eltwise_convenience.hpp` / `eltwise_chain.hpp` `MulUnary` | **Files do not exist in this tree** (only on unmerged branches under a different layout); verified via `git ls-tree -r HEAD` | `ttnn/cpp/ttnn/kernel_lib/` listing |
| Averaging epilogue | `reduce()` + `PostReduceOp` slot | `reduce_helpers_compute.hpp`'s reduce collapses the within-tile 32×32 dims — wrong math for an element-wise cross-device sum (same rejection all_reduce documented) | `reduce_helpers_compute.hpp:383-397` |
| Averaging epilogue (alternative) | `mul_tiles_bcast_scalar` + scaler CB | Usable but strictly worse here: adds a 5th CB, reader-side scaler prep, and a SEPARATE full pass over every final tile, where `mul_unary_tile` fuses into the already-open DEST window of the final add at zero extra CB/pass cost | `tt_metal/hw/inc/api/compute/bcast.h:441-451` |
| Reduction (all non-final chunks) | — | **`BlockAccumulate` IS used** (this row exists to state the mandate is met: no hand-rolled adds outside the single fused scaled-final chunk) | `accumulate_helpers_compute.hpp:111` |
| Packet framing | `ttnn._ttnn.fabric.ccl_packet_dims` | 1:1 tile-page↔packet framing, same deliberate rejection as all_gather/all_reduce/reduce_scatter (tile pages of 2048/4096 B ship whole; hardware-proven for both dtypes) | `reduce_scatter/op_design.md:227` |
| Startup fence | `arm_inc(multicast_route, 1)` barrier | Requires `line_multicast_route_info_t` CT plumbing with no Python-side binding precedent; replaced by the pairwise `peer_ready` handshake over the existing unicast stream (see Semaphores) | `ccl_helpers_dataflow.hpp:441-444`; `all_gather/verification_report.md:110-125` |

## Compute Phases

Per (device, direction) core; `steps = num_total_reduction_steps` (RT arg, host table above). Compute boot: `binary_op_init_common(cb_local_operand, cb_partial_in, cb_reduced)` then `BlockAccumulate::arm(...)` — both once, before any loop.

| # | Operation | Helper? | Consumes | Produces | CB state after |
|---|---|---|---|---|---|
| 0 | Forwarding reductions: for each batch, for step `i < steps−F` — chunk-wise `partial = local + incoming` | `acc.run(walk.tiles_this_chunk())` | `cb_local_operand` + `cb_partial_in`, G-granules | `cb_reduced` G-granules → writer fabric-forwards | empty per granule (streamed) |
| 1 | Unscaled final (hands-off side only, i.e. mid-device FWD): `out(slice i) = input(slice i) + interm_fwd` | `acc.run(n)` — identical to phase 0; the reader/writer change tensors, not the compute | same | `cb_reduced` → writer writes LOCAL output + per-chunk handoff inc | same |
| 2 | **Scaled final** (`scale_output` CT flag; last step of each batch): `out = (local + incoming) · (1/N)` | raw (justified above): mirror `run()`'s exact CB ordering — `cb_wait_front(a,G); cb_wait_front(b,G); tile_regs_acquire; add_tiles(a,b,j,j,j)×n; binop_with_scalar_tile_init(); mul_unary_tile(j, inv_n_bits)×n; tile_regs_commit; cb_pop_front(a,G); cb_pop_front(b,G); cb_reserve_back(out,G); tile_regs_wait; pack_tile(j,out,j)×n; tile_regs_release; cb_push_back(out,G);` then **`acc.rearm()`** | same | `cb_reduced` → writer writes LOCAL output (accumulate side: overwrites the fwd-written value with the averaged total) | same |

First-device-in-direction cores: `steps = 0` — the compute kernel's loop body never executes (boot init + arm still run, harmless; the C++ does the same). The bypass never touches compute.

Compute CT args carry `scale_output` and `inv_n_bits`; when `scale_output == 0` the epilogue path is `if constexpr`-eliminated and every step is `acc.run()`.

## Averaging Placement Verification

| Device | Writes to output, in order | Value after each write | Final value |
|---|---|---|---|
| 0 | BWD final (scaled) | `(input₀ + Σ₁..N−1) / N` | ✓ mean slice 0 |
| mid k | FWD final (unscaled) → `Σ₀..k`; then BWD final (accumulate + scaled) → `(Σ₀..k + Σ_{k+1..N−1}) / N` | fwd value is transient; per-chunk `fwd_bwd` handoff orders read-back after write | ✓ mean slice k |
| N−1 | FWD final (scaled) | `(Σ₀..N−2 + input_{N−1}) / N` | ✓ mean slice N−1 |

Forwarded partials are NEVER scaled (full-magnitude sums travel the chain); the 1/N lands exactly once, on the last write. `inv_n_bits` is exact for N = 2,4,8 (power-of-two reciprocal, no rounding); for non-power-of-two N the single fp32 multiply adds ≤ 1 ulp.

## Broadcast Verification

| Op | Broadcast | Valid region |
|---|---|---|
| `add_tiles` via `BlockAccumulate` / hand-rolled final | none (element-wise, tile i ↔ tile i) | All 32×32 |
| `mul_unary_tile` | scalar constant applied to the whole DEST tile | All 32×32 |

No reduce-direction section applies (no `reduce_tile` anywhere).

## Registry Contract & Host Assembly

| Item | Decision |
|---|---|
| Module layout | `ttnn/ttnn/operations/reduce_scatter_average/{__init__.py, reduce_scatter_average.py, reduce_scatter_average_program_descriptor.py, kernels/{reduce_scatter_average_reader.cpp, reduce_scatter_average_writer.cpp, reduce_scatter_average_compute.cpp}}` |
| Package re-exports | `__init__.py` re-exports exactly `reduce_scatter_average, SUPPORTED, EXCLUSIONS, INPUT_TAGGERS` (the golden harness imports from the package) |
| SUPPORTED (Phase 0) | `{"dtype": [ttnn.bfloat16, ttnn.float32], "layout": [ttnn.TILE_LAYOUT], "topology": [_Topology.Linear], "dim": [3]}` — `dim` MUST be a SUPPORTED key even at one value (harness derives xfails by iterating SUPPORTED). `TARGET − SUPPORTED` refinement candidates: `Topology.Ring`, `dim=2` |
| EXCLUSIONS / INPUT_TAGGERS | `[]` / `{}` |
| Refusal types | `try: from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue except ImportError:` local `NotImplementedError` subclasses (verbatim reference pattern) |
| validate() ordering | universal structural (`ValueError`: mesh N≥2, rank 4, TILE, interleaved, 16B page guard) → **axis gate** (`UnsupportedAxisValue` / `ExcludedCell`) → axis-dependent structural (`ValueError`: H/W tile-aligned, `shape[canonical_dim] % (N·32) == 0`, output_tensor spec match). Copy `reduce_scatter.py:137-211`'s order, NOT all_reduce's |
| Program build | One `ttnn.ProgramDescriptor(kernels=[fwd_reader, fwd_writer, fwd_compute, bwd_reader, bwd_writer, bwd_compute], semaphores=[], cbs=[cb_local_operand, cb_partial_in, cb_bypass, cb_reduced])` per `MeshCoordinate(0, i)`; 3 kernel sources instantiated twice with per-direction CT/RT args. `ReaderConfigDescriptor` / `WriterConfigDescriptor` / `ComputeConfigDescriptor(math_fidelity=HiFi4, fp32_dest_acc_en=True, math_approx_mode=False, dst_full_sync_en=False)` |
| Fabric wiring | `ttnn._ttnn.fabric.ccl_dm_route(mesh_device, coord_i, coord_neighbour, topology)` per direction (assert `num_hops == 1` — store-and-forward invariant); neighbour fabric ids via `mesh_device.get_fabric_node_id(...)` (NOT identity on ring-cabled boxes); conn block appended AFTER `ProgramDescriptor` construction via the reference `_append_fabric_rt_args` (`[has_forward][conn][has_backward][conn]`, placed LAST, `setup_fabric_connection` mutates the program) — on the WRITER kernels only. Writers with `num_targets == 0` still get their scalar RT args (they run the final reduction) but NO conn block; the kernel gates all fabric code on `num_targets_in_direction > 0` |
| Kernel include spellings (proven under generic_op JIT) | `"ttnn/cpp/ttnn/kernel_lib/ccl_helpers_dataflow.hpp"`, `"ttnn/operations/ccl/shared_with_host/ccl_helpers_schedule.hpp"` (note: NOT under kernel_lib), `"ttnn/cpp/ttnn/kernel_lib/accumulate_helpers_compute.hpp"`, `"api/compute/eltwise_binary.h"`, `"api/compute/eltwise_unary/binop_with_scalar.h"` |
| Dispatch | `ttnn.generic_op([input_tensor, intermediate_tensor, output_tensor], mesh_pd)` — ONE call, output LAST. NO per-call post-dispatch `synchronize_device` |

## Program Cache & Semaphore Lifecycle

| Mechanism | Detail |
|---|---|
| Hash-stable | kernel sources, CT args, core ranges, configs, RT-arg COUNTS, CB sizes/formats. Per-device programs differ by CT args (my_chip_id, sync flag, scale flag) — fine, cache is per device |
| Hash-excluded (re-pushed on hit) | RT-arg VALUES: tensor addresses (intermediate allocated fresh per call), semaphore addresses |
| GlobalSemaphores | created once per mesh + one `synchronize_device` in the miss branch; parked on `mesh_pd.semaphores` (generic_op copies them into the cached workload's `shared_variables`, keeping L1 alive across cache hits; excluded from the hash) |
| Kernel re-arm | `out_ready` reset by reader after last wait (unconditional, both cores — the line-end path matters); `peer_ready` reset by writer immediately after its wait; `fwd_bwd` reset by BWD reader at end when `accumulate_output`. Missing any reset = first run green, second hangs |
| Cross-call payload gating | `peer_ready` handshake (see Semaphores) — REQUIRED, not optional: device 0's FWD core has zero waits and finishes call K almost instantly, making the "call K+1 inc destroyed by call K's late reset" race live without it |
| The trap test | acceptance `test_reduce_scatter_average_program_cache` calls the op twice with identical specs and checks both results + `num_program_cache_entries` |

## Key Risks and Gotchas

| # | Risk | Rule |
|---|---|---|
| R1 | CB granule vs math count | ALL CB ops at `tile_granularity`; only tile loops use `tiles_this_chunk()`. Conflating = deadlock (`accumulate_helpers_compute.hpp:29-34`) |
| R2 | `LineSliceCursor` hoisting | Construct INSIDE the batch loop in reader AND writer (per-batch restart); hoisting silently walks wrong slices |
| R3 | Schedule drift | All three kernels build `LineChannelWalk` from the SAME (slice_C, G, start, end); never per-kernel chunk math |
| R4 | Epilogue state clobber | `acc.rearm()` after EVERY hand-rolled scaled chunk, before the next `acc.run()` (next batch's steps). `rearm` restores data formats too — `add_tiles_init` alone does not |
| R5 | Hand-rolled chunk ordering | Mirror `run()` exactly: pops BEFORE `cb_reserve_back(out)` (`accumulate_helpers_compute.inl:46-70`); DEST is zeroed by `tile_regs_release`, not acquire |
| R6 | Handoff visibility | FWD writer: full `noc_async_write_barrier()` (not flush) before each `fwd_bwd` inc — the BWD reader reads those output tiles back |
| R7 | Handoff wait placement | BWD reader waits `fwd_bwd` at the TOP of each accumulate-final chunk, before any read; `fwd_sync_cnt` monotonic across channels and batches |
| R8 | CB slot reuse under fabric | `noc_async_writes_flushed()` between `write_page` and `cb_pop_front` in the forwarding loop |
| R9 | Readiness deadlock | Writer must SIGNAL `peer_ready` before WAITING on its own (signal-then-wait); gate both on `num_targets > 0` |
| R10 | Idle-direction writers | 0-target writers still run the final reduction — they keep scalar RT args, skip only the fabric block (`if (num_targets_in_direction > 0)` guards sender construction, open, arming, handshake, close) |
| R11 | fp32 DEST | `fp32_dest_acc_en=True` → `DEST_AUTO_LIMIT = 4` (`dest_helpers.hpp:103`); `tile_granularity = 4` is at the limit — never raise G without dropping fp32 acc |
| R12 | Interm addressing | Reader and writer `interm_walker` expressions must be character-for-character identical (base = `slice_tile_offset + batch_offset + intermediate_full_offset`); BWD offset = `input_num_pages` (whole-shard page count) |
| R13 | Mesh/test contract | Tests MUST open exactly `(1, 8)` + `FABRIC_1D` (`wh_t3k_allmmio_reduce_scatter_average`); anything else hangs fabric init ("Fabric Router Sync: Timeout") or fails `system_mesh.cpp: requested_size <= system_size`. Drive via `scripts/run_multidevice_sim_pytest.py --op reduce_scatter_average` — NEVER `run_safe_pytest.sh` for this op |
| R14 | dim sign | Canonicalize `dim` to POSITIVE before the SUPPORTED membership test (`-1 ≡ 3`) |
| R15 | Topology import | `from ttnn._ttnn.operations.ccl import Topology as _Topology` — module-scope `ttnn.Topology` raises during auto-import |
| R16 | Accumulate-final walker | Output read-back uses `SliceRowWalker(slice_Wt, slice_Wt)` (dense), base `b·output_batch_num_pages`, row0 `rebase_row_offset(start_row_offset, input_tensor_Wt, slice_Wt)`, channel bump `output_channel_num_pages` — NOT the input-geometry walker |
| R17 | Reader chunk order | Main-tensor read first, THEN the `out_ready` wait, THEN the interm read (overlaps read latency with the wait), one `noc_async_read_barrier()` per chunk covering both |
| R18 | Sender lifetime | `FabricStreamSender` declared before (outliving) the stream; `stream.close()` before kernel end; the one invariant the typestate does not enforce |
| R19 | bf16 precision | Partials are stored bf16 in the interm buffer (one rounding per hop); DEST accumulates fp32. Acceptance PCC: bf16 → 0.99 (reduction budget, all_reduce precedent), f32 → 0.999. Do not tighten |

### Notes for the verifier / golden suite (pre-existing scaffold defects — not this op's files)

`eval/golden_tests/reduce_scatter_average/helpers.py` has two committed bugs that will fail every golden cell regardless of op correctness: (1) line ~95 calls `reduce_scatter(...)` while only `reduce_scatter_average` is imported → `NameError`; (2) lines ~75-81 build a SUM oracle with no `/ num_devices`, contradicting the op definition (mean). Both need a `/golden-tests` re-run or manual fix before golden results mean anything. `feature_spec.py` is authoritative and correct (pipeline mode; `INVALID = []` — no structural impossibilities to add for the TILE + float axis set).

## Acceptance

| Check | Criterion |
|---|---|
| Correctness | Per device i: output == slice i of the fp32-accumulated mean of all shards; PCC ≥ 0.99 (bf16) / 0.999 (f32) |
| Single dispatch | Exactly one `ttnn.generic_op` per invocation |
| output_tensor path | Writes into the supplied tensor; returns the same handle (`buffer_address()` equality) |
| Program cache | Second identical call correct; GlobalSemaphores survive the hit; cache entry count stable between calls 2 and beyond |
| Test file | `tests/ttnn/unit_tests/operations/reduce_scatter_average/test_reduce_scatter_average.py` — IMMUTABLE; the implementer does not modify it |
