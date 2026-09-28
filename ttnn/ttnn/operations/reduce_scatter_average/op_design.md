# Operation Design: reduce_scatter_average

## Overview

| Field | Value |
|-------|-------|
| Classification | CCL (compute-CCL: collective movement + tile reduction + scaling epilogue) |
| Goal | Element-wise MEAN of all N devices' same-shape shards on a MeshDevice line; device `i` keeps only slice `i` (of N equal slices along `dim`) of that mean. Scaling by 1/N is part of the op — the caller passes nothing but the tensor. |
| Math | `output_i[...] = ((Σ_{j=0..N-1} shard_j) / N)[..., i*(W/N) : (i+1)*(W/N)]` for `dim=3` (Phase-0) |
| Dispatch | **ONE `ttnn.generic_op` per invocation** — a single program per mesh coordinate in one `ttnn.MeshProgramDescriptor`. Compute overlaps fabric arrival via per-block semaphore signaling (see Dataflow Strategy). The reference `reduce_scatter`'s two-dispatch gather-then-reduce split is explicitly NOT copied. |
| Mode | Derivative (relay dataflow derived from the hardware-validated `reduce_scatter` Phase-A kernels; reduce pipeline is new) |
| References | `ttnn/ttnn/operations/reduce_scatter/` (relay kernels, host factory, registry contract, validate ordering), `ttnn/ttnn/operations/all_reduce/` (acceptance-test shape, PCC policy), `ttnn/cpp/ttnn/kernel_lib/ccl_helpers_dataflow.hpp`, `ttnn/cpp/ttnn/kernel_lib/accumulate_helpers_compute.hpp`, `ttnn/cpp/ttnn/operations/ccl/shared_with_host/ccl_helpers_schedule.hpp` |

**Do not wrap any existing CCL op.** All five kernels below are newly authored under `ttnn/ttnn/operations/reduce_scatter_average/kernels/`.

## Parameters

| Name | Type | Required | Valid Range | Default | Notes |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | TILE, interleaved, bf16/fp32, on a (1, N) MeshDevice line, N ≥ 2 | — | one SAME-shape shard per device |
| `dim` | `int` | no | Phase-0: 3 (canonical). Negative aliases accepted | `3` | **Canonicalize to POSITIVE (`dim % 4`, rank pinned to 4) BEFORE the SUPPORTED membership test** — the feature_spec TARGET uses the positive convention (`eval/golden_tests/reduce_scatter_average/feature_spec.py:38-47`), same as reference `reduce_scatter._canonicalize_dim` (`reduce_scatter.py:118-120`) |
| `topology` | `ttnn.Topology` | no | Phase-0: `Linear` | `Linear` | import as `from ttnn._ttnn.operations.ccl import Topology as _Topology` for the default (load-order idiom, `reduce_scatter.py:43`) |
| `output_tensor` | `ttnn.Tensor \| None` | no | slice shape, same dtype/layout/buffer_type as input | `None` | write into the supplied tensor and return the SAME handle |

## Tensors

### Input (per-device shard)

| Property | Requirement |
|----------|-------------|
| Shape | `(B, C, H, W)`, rank 4, `H % 32 == 0`, `W % 32 == 0` |
| Scatter constraint | `shape[dim] % N == 0` AND `(shape[dim] / N) % 32 == 0` — otherwise **ValueError** (loud, no silent padding) |
| Dtype | `bfloat16` (primary), `float32` |
| Layout | TILE |
| Memory | interleaved, DRAM or L1 (`ValueError` if `input_tensor.is_sharded()`) |
| Mesh | `(1, N)` line, `N = prod(mesh_device.shape) ≥ 2` |

### Output (per-device)

| Property | Value |
|----------|-------|
| Shape | shard shape with `shape[dim] //= N` (dim=3: `(B, C, H, W/N)`) |
| Dtype / Layout / Memory | same as input |
| Allocation when `output_tensor is None` | `ttnn.allocate_tensor_on_device(ttnn.Shape(out_shape), dtype, layout, mesh_device, input.memory_config())` (reference `reduce_scatter.py:242-251`); every output page is written, no seeding needed |

### Op-internal gather buffer (fabric landing target)

| Property | Value |
|----------|-------|
| Shape | `(B*N, C, H, W)` — block `c` (source device `c`) occupies pages `[c*P, (c+1)*P)` |
| Dtype / Layout / Memory | same as input (mesh-allocated interleaved ⇒ **uniform buffer address across devices**, which is what lets a fabric `write_page` target the neighbour's block through the LOCAL `TensorAccessor` routed one hop — reference `reduce_scatter.py:253-264`) |
| Lifetime | allocated fresh per call, passed in `io_tensors`; own block (`c == my_chip_id`) is never written (the reduce reader takes the own contribution directly from the input tensor — this deletes the reference's serialized self-copy wart, `reduce_scatter_reader.cpp:99-111` / `verification_report.md:91-93`) |

### Derived quantities (host computes once; symbols used throughout)

| Symbol | Formula | Meaning | Golden range |
|--------|---------|---------|--------------|
| `N` | `prod(mesh_device.shape)` | devices on the line | 8 (sim), 4 (hw) |
| `P` | `input.buffer_num_pages()` | tiles per shard | 64–128 |
| `Wt` | `W / 32` | shard tile-columns | 8–16 |
| `slice_Wt` | `Wt / N` | output tile-columns | 1–2 |
| `Rt` | `B * C * (H / 32)` | total tile-rows (batches are contiguous row-blocks in tiled page order, so the shard is walked as an `Rt × Wt` tile grid — no per-batch logic needed for dim=3) | 8–16 |
| `S` | `P / N = Rt * slice_Wt` | output tiles per device | 8–16 |
| `g` | largest of `{4, 2, 1}` dividing `S` | CB/DEST granule; `g ≤ DEST_AUTO_LIMIT = 4` under `fp32_dest_acc_en` + SyncHalf (`dest_helpers.hpp:90-103`); **`g` divides `S`** so no tail chunk ever exists |
| `page_size` | `input.buffer_page_size()` | tile bytes (bf16 2048 / fp32 4096) | |
| `aligned_page_size` | `ttnn.round_up(page_size, ttnn.get_l1_alignment())` | relay CB page | = page_size (already aligned) |
| `scaler_bits` | bf16: `(bits(bf16(1/N)) << 16) \| bits(bf16(1/N))`; fp32: `bits(float(1/N))` | packed scaler CT arg | 1/8, 1/4 — exact in bf16 for power-of-2 N |

## Dataflow Strategy

### Algorithm

Line store-and-forward **gather of whole shards** (the hardware-validated reference relay pattern) fused in the SAME program with an **arrival-ordered incremental reduce**. Every device ends up receiving all N−1 remote shards into its local `gather_buffer`; a dedicated reduce core consumes contributions one at a time — own shard first, then each arrival the moment its semaphore lands — so the accumulate of contribution *k* overlaps the fabric flight of contribution *k+1*. After the last accumulate, a 1/N broadcast-scalar multiply produces the output slice.

Why this decomposition (algorithm decision): the relay half is byte-for-byte the traffic pattern the reference proved on hardware, and the reduce half needs agreement only on a fixed per-contribution CB protocol (`g`-granule streaming) — the smallest cross-kernel drift surface that still satisfies the single-dispatch + overlap mandate. The bandwidth-optimal partial-sum line reduce-scatter (compute in the relay path, `LineSliceCursor`/`LineChannelWalk` step machine agreed across 3 kernels) is deferred as a refinement, exactly as the reference deferred it (`reduce_scatter/op_design.md:19-25`).

### Per-device data path

```
                    device i  (one program, one generic_op dispatch)
 core (0,0)  relay FWD:  input ──reader──▶ cb_relay_pages ──writer──▶ fabric 1 hop right
                          gather_buffer (fwd arrivals) ──reader──▶ cb_relay_pages ──writer──▶ (relay onward)
 core (0,1)  relay BWD:  mirror of (0,0), 1 hop left
 core (0,2)  reduce:     input slice i  ──reader──▶ cb_contributions ─┐
                          gather_buffer slice i of each arrived shard ─┤ (arrival order, g-granules)
                                                                       ▼
                          compute: seed-copy ▶ (N-1)× incremental add ▶ ×(1/N) ▶ cb_averaged
                                                                       ▼
                          writer: output tensor (dense tiles 0..S-1)
```

### Tensix-to-Tensix / device-to-device contract

| # | Contract | Detail |
|---|----------|--------|
| T1 | Fwd channel carries left→right traffic | device `i` fwd-sends `1 + i` blocks (own shard first, then relays of its `i` fwd arrivals) iff it has a right neighbour; fwd arrivals on device `i` = `i` blocks, in chain order **nearest-first**: shards of `i-1, i-2, …, 0`. Counts per reference table `reduce_scatter_program_descriptor.py:151-172` (Linear rows). |
| T2 | Bwd channel carries right→left traffic | mirror: bwd-sends `1 + (N-1-i)` iff left neighbour exists; bwd arrivals = `N-1-i`, order `i+1, i+2, …, N-1`. |
| T3 | Block indices on the wire | k=0 → own block `i`; relay k ≥ 1 → fwd `(i + N - k) % N`, bwd `(i + k) % N` (ring-modular form, reference `reduce_scatter_writer.cpp:80-84`; equals plain linear indices for Linear). Fabric `write_page` targets the NEXT device's `gather_buffer` pages `[c*P, (c+1)*P)` through the local accessor (uniform mesh address). |
| T4 | Arrival signaling — the overlap mechanism | after the last page of each block, the sending writer issues **two** fabric atomic incs on its armed inc channel, both 1 hop to the receiving device: `sem_dir` at the receiving relay core ((0,0) for fwd / (0,1) for bwd) AND `sem_dir` at the receiving reduce core (0,2). Incs are in-order behind the pages on the same connection, so `sem ≥ k` ⇒ blocks 1..k fully landed in `gather_buffer`. |
| T5 | Semaphores | TWO op-internal `GlobalSemaphore`s, `sem_fwd` and `sem_bwd` (one address each, a private counter per core). Consumers: relay reader (0,0) waits `sem_fwd`; relay reader (0,1) waits `sem_bwd`; reduce reader (0,2) polls BOTH. Each consumer re-arms its OWN core's counter to 0 after its last wait (cache-reuse footgun, `ccl_helpers_dataflow.hpp:116-121`). Waits/resets are op-owned — there is no receiver helper. |
| T6 | Relay forwarding | relay reader waits `sem_dir ≥ k+1`, reads arrival k's `P` pages from the LOCAL `gather_buffer` back into `cb_relay_pages`; writer forwards them one more hop. A line-end device (`num_sends == 0`) relays nothing but still waits all its arrivals and re-arms the sem. |
| T7 | Overlap timeline | reduce compute runs pass 0 (own) immediately; pass k runs as soon as the k-th arrival's double-inc lands, while arrival k+1 is still being relayed/flown. The scale + output write start after the last pass — the only serialized tail is `S` tiles of SFPU-free FPU work. |
| T8 | Deadlock freedom | relay chains are per-direction DAGs (no cycles); the reduce pipeline consumes only sems + DRAM reads + its own core's CBs; each device's egress (seed own shard) depends on nothing remote. No CB is shared across cores. |

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | whole op per device; within a device, fixed roles on 3 cores |
| Grid | logical cores `(0,0)` fwd relay, `(0,1)` bwd relay, `(0,2)` reduce — `ttnn.CoreRangeSet` of the three singleton ranges. NoC targets via `mesh_device.worker_core_from_logical_core(...)`, identical logical→physical mapping assumed across devices (reference precedent, `reduce_scatter_program_descriptor.py:352-356`) |
| Per-core work | (0,0)/(0,1): `num_sends_dir * P` pages relayed, `num_arrivals_dir` waits; (0,2): reader `N*S` tile reads, compute `N*S` tile-adds equivalent + `S` scaled tiles, writer `S` tile writes |
| Remainder | none — `g` divides `S` by construction; every CB interaction is a whole `g`-granule (or whole page for relay) |
| Multi-core reduce | deliberately NOT Phase-0: splitting `S` positions across reduce cores multiplies the per-block inc fan-out (each reduce core needs its own sem copies inc'd). Golden shapes have S ≤ 16 — one core is right-sized. Refinement 3. |

## Circular Buffers

All CBs are core-local (no CB spans cores). **Capacity rule**: every CB's capacity is a multiple of its interaction quantum, so a multi-page reserve/wait never straddles the ring wrap (linear `l1 += page_size` writes after a multi-page reserve require contiguity — reference reader precedent `reduce_scatter_reader.cpp:182-198`).

| Semantic Name | Index | Cores | Page Size | Num Pages | Format | Producer | Consumer | Lifetime / quantum |
|---------------|-------|-------|-----------|-----------|--------|----------|----------|--------------------|
| `cb_relay_pages` | 16 | (0,0), (0,1) | `aligned_page_size` | 2 | input dtype | relay reader | relay writer | streaming, quantum 1 page (double-buffered) |
| `cb_contributions` | 0 | (0,2) | `page_size` | `2*g` | input dtype | reduce reader | reduce compute | streaming, quantum `g` (double-buffered granules); carries all N contributions in arrival order, own first |
| `cb_scaler` | 8 | (0,2) | `page_size` | 1 | input dtype | reduce reader | reduce compute | persistent: pushed once, waited once (count 1), **never popped** |
| `cb_accumulator` | 24 | (0,2) | `page_size` | `S` | input dtype | reduce compute **only** | reduce compute (passes) + scale phase | resident running sum; quantum `g`; capacity exactly `S` (`g` divides `S` ⇒ wrap-safe). **Single producer invariant — see R4** |
| `cb_averaged` | 17 | (0,2) | `page_size` | `2*g` | input dtype | reduce compute | reduce writer | streaming, quantum `g` |

L1 budget (worst golden case, fp32 `(1,1,256,512)`: S=16, g=4, page 4096): 32 + 4 + 64 + 32 KB ≈ 132 KB on (0,2); 8 KB on each relay core. Growth cliff: `cb_accumulator = S` pages ⇒ Phase-0 must reject (ValueError) shards whose `S * page_size` exceeds a conservative L1 budget (suggest `S ≤ 256`); refinement 5 lifts it.

### Semaphores

| Name | Kind | Created | Inc'd by | Waited/reset by |
|------|------|---------|----------|-----------------|
| `sem_fwd` | GlobalSemaphore, initial 0, all worker cores | once per `mesh_device`, cached (`_SEMAPHORE_CACHE[id(mesh_device)] = (sem_fwd, sem_bwd)`), ONE `ttnn.synchronize_device` inside the miss branch only (reference `reduce_scatter.py:99-115`) | left neighbour's fwd writer: 2 fabric incs/block → cores (0,0) and (0,2) | (0,0) relay reader; (0,2) reduce reader — each resets its OWN core's counter to 0 after its final wait |
| `sem_bwd` | same | same call | right neighbour's bwd writer: 2 fabric incs/block → cores (0,1) and (0,2) | (0,1) relay reader; (0,2) reduce reader |

Both parked on `mesh_program_descriptor.semaphores = [sem_fwd, sem_bwd]` (kept alive across the program cache, excluded from the cache hash — `program_descriptors.cpp:1077-1087`). Addresses via `ttnn.get_global_semaphore_address(...)` baked into runtime args. **No per-call post-dispatch barrier.**

## Host Assembly (program factory)

One Python module pair, mirroring the reference layout: `reduce_scatter_average.py` (signature, registry contract, validate, semaphore cache, allocation, single `ttnn.generic_op` call) + `reduce_scatter_average_program_descriptor.py` (mesh PD factory).

| Duty | Mechanism |
|------|-----------|
| Mesh PD | `ttnn.MeshProgramDescriptor()`; one `ttnn.ProgramDescriptor(kernels=[...], semaphores=[], cbs=[...])` per `ttnn.MeshCoordinateRange(coord_i, coord_i)` — programs are per-device distinct (CT args: `my_chip_id`, send/arrival counts) |
| Dispatch | **exactly one** `ttnn.generic_op([input_tensor, gather_buffer, output_tensor], mesh_pd)` — output preallocated and LAST (`generic_op_nanobind.cpp:32-33`) |
| Routes | `ttnn._ttnn.fabric.ccl_dm_route(mesh_device, coord_i, coord_neighbour, topology)` per direction; `assert route.num_hops == 1` (store-and-forward invariant, reference `reduce_scatter_program_descriptor.py:263-296`). The route owns the fwd/bwd sign reversal — never hand-derive `is_forward` |
| Fabric conn args | `ttnn._ttnn.fabric.build_ccl_fabric_rt_args(src_fabric_node_id, neighbor_fabric_node_id, 0, program, worker_core, is_forward)` (`fabric.cpp:277-297`) — emits `[has_forward][fwd conn][has_backward][bwd conn]`, placed FIRST in each relay writer's rt args; it MUTATES the program (appends SemaphoreDescriptors), so append via the live `program.kernels[k].runtime_args[x][y]` view after `ProgramDescriptor` construction. Neighbour ids via `mesh_device.get_fabric_node_id(MeshCoordinate(0, j))` — never assumed identity |
| Packet framing | 1 page = 1 fabric packet (`arm_unicast_write(page_size)`); `ccl_packet_dims` NOT used — same documented rejection as all three reference collectives (`reduce_scatter/op_design.md:227`); tile pages (2048/4096 B) fit a single packet, hardware-validated by the references at both dtypes |
| Idle direction | empty rt-arg list `[]` + CT `num_sends = 0` / `num_arrivals = 0`; kernel no-ops under `if constexpr` (reference `:280,296`) |
| Compute config | `ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False, dst_full_sync_en=False)` (reference `:509-514`) — fixes `DEST_AUTO_LIMIT = 4` |
| TensorAccessors | `list(ttnn.TensorAccessorArgs(t).get_compile_time_args())` appended LAST after all scalar CT args, for: input + gather (relay reader), gather (relay writer), input + gather (reduce reader), output (reduce writer) |

Information each kernel needs (exact CT/RT index layout is the implementer's choice — derive from the CB table and helper signatures):

| Kernel (file under `kernels/`) | Core | Needs |
|---|---|---|
| `reduce_scatter_average_relay_reader.cpp` | (0,0)+(0,1), one source, CT-selected direction | direction, `my_chip_id`, `N`, `num_sends`, `num_arrivals`, `P`, page size; RT: input addr, gather addr, own-direction sem addr |
| `reduce_scatter_average_relay_writer.cpp` | (0,0)+(0,1), one source | direction, `my_chip_id`, `N`, `num_sends`, `P`, page size, L1 alignment; RT: fabric conn block FIRST, gather addr, `num_hops(=1)`, sem addr, NoC xy of neighbour's relay core AND reduce core |
| `reduce_scatter_average_reduce_reader.cpp` | (0,2) | `my_chip_id`, `N`, `fwd_arrivals`, `bwd_arrivals`, `S`, `g`, `Wt`, `slice_Wt`, `P`, `dim` (static_assert), `scaler_bits`, `scaler_is_fp32`; RT: input addr, gather addr, `sem_fwd` addr, `sem_bwd` addr |
| `reduce_scatter_average_reduce_compute.cpp` | (0,2) | `N`, `S`, `g` |
| `reduce_scatter_average_reduce_writer.cpp` | (0,2) | `S`, `g`; RT: output addr |

Four relay kernel *descriptors* (fwd reader, fwd writer, bwd reader, bwd writer — CT args differ per direction) + three reduce descriptors = 7 per program (reference structure).

## API Mapping

Every mechanism with verified file:line. Type `helper` or `raw_api`.

| Phase | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Requirements |
|-------|------|----------|-----------|------------------------|----------|-----------|--------------|
| Relay egress open | helper | `dataflow_kernel_lib::ccl::FabricStreamSender<>` ctor / `open(unicast_route(num_hops))` | `ttnn/cpp/ttnn/kernel_lib/ccl_helpers_dataflow.hpp:481,492,503`; `unicast_route` `:302` | `ConnT = DirectConn`; ctor `(conn_arg_idx, is_forward, alignment)`; peek `get_arg_val<uint32_t>(0)` for `is_forward` (reference `reduce_scatter_writer.cpp:77-78`) | — | — | sender declared before (outlives) the stream; route bound ONCE at open |
| Relay page send | helper | `FabricStream::arm_unicast_write(page_size).write_page(src_l1, page_idx, gather_accessor)` | `:423` (arm), `:327` (write_page) | invariant per-page payload | `cb_relay_pages` | remote `gather_buffer` | `noc_async_writes_flushed()` between `write_page` and `cb_pop_front` — CB-slot-reuse guard (reference `reduce_scatter_writer.cpp:100-104`) |
| Arrival signal ×2 | helper | `FabricStream::arm_inc(1)` then `channel.inc(noc_addr)` twice per block | `:435` (arm), `:368` (inc) | one armed channel, two issues: relay-core sem, reduce-core sem; `safe_get_noc_addr(x, y, sem_addr, 0)` | — | remote sems | in-order behind pages on the same connection |
| Egress close | helper | `FabricStream::close()` | `:461` | — | — | — | drains write + atomic barriers; idempotent |
| Arrival wait (relay) | raw_api (op-owned by design) | `noc_semaphore_wait_min(sem_ptr, k+1)`; re-arm `noc_semaphore_set(sem_ptr, 0)` | helper banner `ccl_helpers_dataflow.hpp:108-121` assigns the WAIT + reset to the op; reference `reduce_scatter_reader.cpp:131-151` | — | — | — | reset AFTER the final wait, on every role incl. pure receivers |
| Arrival poll (reduce) | raw_api (op-owned by design) | two-way poll: volatile reads of `sem_fwd`/`sem_bwd` L1 words with the same `invalidate_l1_cache()` spin `noc_semaphore_wait_min` uses; consume whichever direction has an unconsumed arrival | no helper exists — receive-side sync is explicitly outside `ccl_helpers_dataflow.hpp` (banner `:112-121`: "the *waiting* half is a plain op-owned `noc_semaphore_wait_min`"); a two-counter wait has no primitive | — | — | monotonic counters, no race; loop bound = `fwd_arrivals + bwd_arrivals`; reset BOTH after |
| Slice tile walk | helper | `ttnn::ccl::schedule::SliceRowWalker(slice_Wt, Wt)` + `set_base` / `reset_offsets(0,0)` / `next()` | `ttnn/cpp/ttnn/operations/ccl/shared_with_host/ccl_helpers_schedule.hpp:491,498,502,510,516`; base via `slice_tile_offset(dim, my_chip_id, C, Rt/…, slice_Wt)` `:466`; `static_assert(is_supported_scatter_dim(dim))` `:460` | own: base `my_chip_id*slice_Wt` over input; arrival from `src`: base `src*P + my_chip_id*slice_Wt` over gather_buffer; identical walk per contribution ⇒ positional alignment across passes | — | — | shard walked as `Rt × Wt` grid (covers batches); `next()` returns AND advances — call once per tile |
| DRAM reads/writes | helper | `TensorAccessor` + `noc_async_read/get_noc_addr` (reader), `noc_async_write` (writer) | `tech_reports/tensor_accessor/tensor_accessor.md`; reference `reduce_scatter_reader.cpp:182-198` | CT args from `ttnn.TensorAccessorArgs` | — | `cb_contributions` / output | per-granule (not per-tile) barriers |
| Scaler fill (bf16) | helper | `generate_bcast_unary_scalar(cb, scaler_bits)` | `ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp:43-49` | writes only `ptr[0] = scaler_bits >> 16`; owns reserve/push of 1 page | — | `cb_scaler` | scalar double-packed bf16 in u32 (host packs) |
| Scaler fill (fp32) | raw_api | reserve 1 page, store `scaler_bits` (raw IEEE-754 u32) to `ptr[0]`, push 1 | justification below | — | — | `cb_scaler` | mirrors the helper; only element (0,0) is read by SCALAR bcast |
| Compute boot | raw_api (mandated pre-condition) | `binary_op_init_common(cb_contributions, cb_accumulator, cb_averaged)` | pre-condition of both accumulate helpers: `accumulate_helpers_compute.hpp:116-117` ("@pre The kernel has already run its hardware startup") and `:211`; reference `reduce_scatter_compute.cpp:38` | once at kernel start | — | — | NOT interchangeable with per-op inits (banner `:82-92`) |
| Seed copy (contribution 0 = own) | helper | `compute_kernel_lib::sum_blocks(cb_contributions, cb_accumulator, /*num_blocks=*/1, /*block_num_tiles=*/g, /*pop_input=*/true)` × `S/g` | decl `accumulate_helpers_compute.hpp:221-222`; `num_blocks == 1` degenerates to a copy of block 0 (`:217`, `.inl:106-157`) | `pop_input=true` is load-bearing (deadlock otherwise — reference `reduce_scatter_compute.cpp:15-19`) | `cb_contributions` (g) | `cb_accumulator` (g) | leaves `add_tiles_init` in acc_to_dest mode (post `:212-213`) ⇒ `rearm()` before the runs |
| Incremental accumulate (arrivals 1..N−1) | helper | `compute_kernel_lib::BlockAccumulate::arm(cb_contributions, cb_accumulator, cb_accumulator, g)`; `rearm()`; `run(g)` × `(N-1) * S/g` | arm `accumulate_helpers_compute.hpp:125`, run `:132`, rearm `:175`; `.inl:46-70` | **in-place `cb_b == cb_out`**: sound because `run()` pops a and b BEFORE reserving out (`.inl:58-63`, "verified ordering") — with capacity exactly `S`, pop-then-reserve always finds `g` free pages | `cb_contributions` (g) + `cb_accumulator` front (g) | `cb_accumulator` back (g) | `g ≤ DEST_AUTO_LIMIT` asserted at arm (`hpp:122-123`); arm requires hw startup done |
| 1/N scale | raw_api | `mul_tiles_bcast_scalar_init_short(cb_accumulator, cb_scaler)` once, then per granule: `cb_wait_front(cb_accumulator, g)` → `tile_regs_acquire` → `mul_tiles_bcast_scalar(cb_accumulator, cb_scaler, t, 0, t)` ×g → `tile_regs_commit` → `cb_pop_front(cb_accumulator, g)` → `cb_reserve_back(cb_averaged, g)` → `tile_regs_wait` → `pack_tile` ×g → `tile_regs_release` → `cb_push_back(cb_averaged, g)` | `tt_metal/hw/inc/api/compute/bcast.h:441` (init_short), `:451` (op) | `icb1 = cb_scaler` is the scalar operand, read at tile 0 element (0,0); `cb_wait_front(cb_scaler, 1)` once, never pop | `cb_accumulator` (g) + `cb_scaler` (1, resident) | `cb_averaged` (g) | justification below; init_short AFTER the last `run()` (it reprograms the binary op state) |

### Raw-API justifications (helpers considered and rejected)

**1/N scale (`mul_tiles_bcast_scalar` pass) — helpers considered:**

| Candidate | Rejection (concrete, cited) |
|-----------|------------------------------|
| `eltwise_convenience.hpp` `mul<cb_a, cb_b, cb_out, BroadcastDim::Scalar>` — the designated helper for this exact phase | **ABSENT from this clone at HEAD.** `git ls-tree -r HEAD ttnn/cpp/ttnn/kernel_lib` lists exactly 20 files (`accumulate_helpers_compute`, `ccl_helpers_dataflow`, `dest_helpers`, `dfb_helpers_compute`, `dfb_helpers_dataflow`, `l1_helpers`, `reduce_helpers_common/compute/dataflow`, `tilize_helpers`, `untilize_helpers` — `.hpp` + `.inl` each); no `eltwise_convenience.hpp`, no `eltwise_chain.hpp`, no `eltwise_binary_sfpu.hpp`. A kernel `#include` of it cannot compile here. |
| `compute_kernel_lib::reduce()` (`reduce_helpers_compute.hpp`) | reduces WITHIN a tensor along a dim, collapsing the within-tile 32×32 dims (`accumulate_helpers_compute.hpp` banner contrast; same rationale as `all_reduce_compute.cpp:19-21`) — the scale must preserve every element |
| `BlockAccumulate` / `sum_blocks` (`accumulate_helpers_compute.hpp:125,221`) | addition only — `add_tiles`-based (`.inl:54,135`); no multiply, no scalar operand, no activation/epilogue hook |
| `prepare_reduce_scaler` (`reduce_helpers_dataflow.hpp:68`) for the scaler tile | header contract `:22-24`: scaler tiles "must ONLY be used" for the reduce LLK, not arbitrary constant tiles; its fill pattern is PoolType/ReduceDim-keyed, meaningless here |

**fp32 scaler fill — helper considered:** `generate_bcast_unary_scalar` (`generate_bcast_scalar.hpp:43`) assumes 16-bit tile elements (comment `:41-42`: "Tile is assumed to have 16-bit elements") and stores the high half of the packed u32 to a `uint32_t*` slot — wrong for a Float32 CB page. The raw fill is the same 4 lines with a full-width store. (`cb_scaler`'s format equals the input dtype precisely so the boot `binary_op_init_common` covers srcB with zero mid-kernel format reconfig.)

**Two-way semaphore poll — helper considered:** `ccl_helpers_dataflow.hpp` explicitly scopes receive-side sync OUT of the helper (banner `:108-121`); `noc_semaphore_wait_min` blocks on ONE counter and would serialize the two directions (losing overlap whenever the other direction is ready first).

## Compute Phases (reduce_compute kernel, core (0,2))

| # | Operation | Helper? | Input CB (tiles, state) | Output CB (tiles) | CB State After |
|---|-----------|---------|--------------------------|-------------------|----------------|
| C0 | `binary_op_init_common(cb_contributions, cb_accumulator, cb_averaged)`; `acc = BlockAccumulate::arm(cb_contributions, cb_accumulator, cb_accumulator, g)` | pre-condition + helper factory | — | — | hw configured; accumulator armed |
| C1 | Seed: copy contribution 0 (own slice) — `S/g` × `sum_blocks(cb_contributions, cb_accumulator, 1, g, true)` | helper | `cb_contributions` (g per call, popped) | `cb_accumulator` (g per call) | `cb_accumulator` holds S tiles = own contribution; `cb_contributions` empty |
| C2 | `acc.rearm()` — restore after `sum_blocks`'s acc_to_dest post-condition (`hpp:212-213`) | helper | — | — | — |
| C3 | Incremental accumulate: `for k in 1..N-1: for c in S/g: acc.run(g)` | helper | `cb_contributions` (g, popped) + `cb_accumulator` front (g, popped) | `cb_accumulator` back (g) | after pass k, `cb_accumulator` holds the S-tile running sum of contributions 0..k; FIFO order = walker order preserved every pass |
| C4 | 1/N scale: `cb_wait_front(cb_scaler, 1)`; `mul_tiles_bcast_scalar_init_short(cb_accumulator, cb_scaler)`; `S/g` granule passes (raw, see API Mapping) | raw | `cb_accumulator` (g, popped) + `cb_scaler` (1, resident) | `cb_averaged` (g) | `cb_accumulator` empty; `cb_scaler` still holds 1 page (never popped); output streamed to writer |

Compute is order-agnostic: it counts passes; the READER decides arrival order (own → whichever direction lands next). No step flags, no schedule agreement beyond "N contributions of S tiles in g-granules".

## Broadcast Verification

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast Dim |
|-------|----|--------------------|--------------------|---------------|
| C4 | `mul_tiles_bcast_scalar(cb_accumulator, cb_scaler, t, 0, t)` | `cb_accumulator`: full 2D `[32,32]` per tile — All | `cb_scaler` tile 0: only element (0,0) valid (fill writes only `ptr[0]`; SCALAR bcast reads only (0,0) — proven by shipped consumer `reader_bcast_scalar_interleaved_partitioned.cpp:40`) | Scalar (HW-bcast) |

## CB Sync Audit (push count == wait/pop count, per CB, per device)

| CB | Pushed | Waited/Popped | Balanced |
|----|--------|----------------|----------|
| `cb_relay_pages` (per relay core) | reader: `num_sends_dir * P` pages | writer: `num_sends_dir * P` (wait 1 / pop 1) | ✓ (0 = 0 on idle direction) |
| `cb_contributions` | reader: `N * S` (g-granules: own + fwd_arrivals + bwd_arrivals = N contributions) | compute: C1 pops `S` + C3 pops `(N-1)*S` | ✓ |
| `cb_accumulator` | compute only: C1 `S` + C3 `(N-1)*S` | compute: C3 waits/pops `(N-1)*S` + C4 pops `S` | ✓ |
| `cb_scaler` | reader: 1 | compute: `cb_wait_front(…, 1)` once; never popped (persistent) — all waits on this CB use count 1 | ✓ |
| `cb_averaged` | compute: `S` | writer: `S` (g-granules) | ✓ |

## Validation & Registry Contract (Phase-0)

| Item | Value |
|------|-------|
| Exports | `SUPPORTED`, `EXCLUSIONS`, `INPUT_TAGGERS`, `reduce_scatter_average` from both `reduce_scatter_average.py` and the package `__init__.py` (reference `reduce_scatter/__init__.py:10-12`) |
| Exceptions | `from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue` inside `try/except ImportError` with local `NotImplementedError`-subclass fallback (`reduce_scatter.py:45-53`) |
| Phase-0 `SUPPORTED` | `{"dtype": [ttnn.bfloat16, ttnn.float32], "layout": [ttnn.TILE_LAYOUT], "topology": [_Topology.Linear], "dim": [3]}` — **`"dim"` MUST be a SUPPORTED key even single-valued** (feature_spec `:42-47`: the harness xfails only declared axes) |
| `INPUT_TAGGERS` / `EXCLUSIONS` | `{}` / `[]` (golden INPUTS are all tile-aligned by construction — feature_spec `:56-59`) |
| `validate()` ordering | universal structural (ValueError: rank 4, TILE-constructible dims, not sharded, mesh line N ≥ 2) → axis gate (`UnsupportedAxisValue` per axis, then `ExcludedCell`) → axis-value-DEPENDENT structural (ValueError: `shape[dim] % N`, `(shape[dim]/N) % 32`, output_tensor spec match, L1 accumulator budget). This exact ordering is the verifier-blessed fix in `reduce_scatter.py:123-135` — do NOT copy `all_reduce`'s older all-structural-first ordering |
| `dim` canonicalization | negative → positive (`dim % 4`) BEFORE the SUPPORTED membership test (feature_spec positive convention) |
| `validate()` returns | `(num_devices, canonical_dim)` |
| TARGET − SUPPORTED refinement candidates | `topology=Ring`, `dim=2` — filed for `op_requirements.md` |

## Refinement candidates (not Phase-0)

| # | Refinement | Sketch |
|---|-----------|--------|
| 1 | `topology=Ring` | relay block indices are already ring-modular (T3); adopt the reference's Ring send/arrival depth table (`reduce_scatter_program_descriptor.py:151-160`: fwd `N/2`, bwd `(N-1)//2`) and `ccl_dm_route`'s short-way selection; reduce reader's per-direction source sequences stay `(i∓(1+a)) % N` |
| 2 | `dim=2` | slice = contiguous tile-row blocks per (batch,channel): per-(B,C) loop with `SliceRowWalker` degenerating to dense runs, base from `slice_tile_offset(dim=2, …)` (`ccl_helpers_schedule.hpp:466-478`) + per-batch `bump_base`; INPUTS already keep dim-2 slices tile-aligned |
| 3 | Multi-core reduce | split `S` across reduce cores; requires per-core arrival incs (fan-out) or a local mcast of the arrival signal |
| 4 | Bandwidth: slice-only relay or true partial-sum line RS | drops the N× gather traffic; the partial-sum variant is the `LineSliceCursor`/`LineChannelWalk` + `SyncCadence` machine — extend the host gtest schedule sweeps (`tests/ttnn/unit_tests/gtests/ccl/test_ccl_helpers_schedule.cpp`) BEFORE any new schedule variant |
| 5 | Large-S support | spill the accumulator (fp32 DRAM scratch) or chunk S; also fp32 accumulator CB (Float32 `cb_accumulator` under bf16 inputs) to cut the N−1 per-pass bf16 pack roundings |
| 6 | Packet coalescing | `ccl_packet_dims` multi-page packets + per-chunk (not per-block) incs |

## Key Risks and Gotchas

| # | Risk | Rule |
|---|------|------|
| R1 | **Cache-reuse semaphore re-arm** — first run green, second hangs (`ccl_helpers_dataflow.hpp:116-121`) | every consumer resets its OWN core's counter after its final wait: (0,0)→`sem_fwd`, (0,1)→`sem_bwd`, (0,2)→BOTH. Reset happens on every role, including devices with 0 arrivals in a direction (reference `reduce_scatter_reader.cpp:151`). The acceptance program-cache test exists to catch exactly this |
| R2 | **`cb_accumulator` single-producer invariant** | ONLY the compute kernel ever reserves/pushes `cb_accumulator`. Do NOT have the reader seed it directly: each RISC keeps a LOCAL CB write pointer, so a second producer starts writing at the CB base and corrupts the ring (it only coincidentally works when capacity exactly equals the pre-pushed count). The seed goes through `cb_contributions` + `sum_blocks` copy instead |
| R3 | **`sum_blocks` post-condition** — leaves `add_tiles_init` in acc_to_dest mode (`accumulate_helpers_compute.hpp:212-213`) | `acc.rearm()` between phase C1 and the first `run()` (also restores unpack/pack data formats — `hpp:169-173`) |
| R4 | **`pop_input=true` on `sum_blocks`** | default `false` deadlocks the reader on `cb_reserve_back` once `cb_contributions` fills (reference `reduce_scatter_compute.cpp:15-19`) |
| R5 | **Granularity contract** — CB protocol always moves whole `g`-granules; `run(n<g)` still moves `g` pages | host guarantees `g` divides `S` (`g ∈ {4,2,1}`), so `run(g)` is always full and no pad tiles exist. `g ≤ DEST_AUTO_LIMIT = 4` under `fp32_dest_acc_en` (asserted at `arm`, `hpp:122-123`) |
| R6 | **CB wrap contiguity** — multi-page reserve + linear `l1 += page_size` writes | every CB capacity is a multiple of its quantum (see CB table); never mix quanta on one CB |
| R7 | **`noc_async_writes_flushed()` before `cb_pop_front`** in the relay writer | the fabric write sources the CB slot; popping first lets the reader overwrite in-flight data (reference `reduce_scatter_writer.cpp:100-104`) |
| R8 | **Inc-after-pages ordering** | both `inc()` issues go on the SAME connection as the block's `write_page`s, after the last page — fabric delivery is in-order per connection, making `sem ≥ k` imply data-complete. Do not move the incs to another stream |
| R9 | **`static_assert(is_supported_scatter_dim(dim))`** must be guarded so discarded `if constexpr` branches don't trip it (reference `reduce_scatter_reader.cpp:59-61`) | |
| R10 | **`mul_tiles_bcast_scalar_init_short` placement** — it reprograms the binary-op state | issue it strictly AFTER the last `acc.run()`; no accumulate may follow without a `rearm()` |
| R11 | **Walker discipline** — `SliceRowWalker::next()` returns AND advances | call exactly once per tile; `reset_offsets(0,0)` + `set_base(...)` per contribution; identical walk order for every contribution is what keeps `add_tiles` positionally aligned across passes |
| R12 | **fp32 vs bf16** | `cb_scaler` format = input dtype (no mid-kernel srcB reconfig); fp32 fill is the raw one-word store; 1/N is exact in bf16 only for power-of-2 N — fine for the (1,8)/(1,4) meshes; PCC is scale-invariant regardless |
| R13 | **Mesh/topology contract** | acceptance + golden run on a `(1, 8)` mesh with `FabricConfig.FABRIC_1D`; any other shape hangs fabric init (`Fabric Router Sync: Timeout`) or fails `system_mesh.cpp: requested_size <= system_size` — a test/topology mismatch, not an op defect |
| R14 | **`gather_buffer` fresh per call** | reference-proven with program-cache hits (`test_all_reduce_program_cache` pattern); pass it in `io_tensors` so dispatch resolves/keeps it alive |
| R15 | **Overlap is arrival-major by necessity** | position-major (`sum_blocks` over N blocks per position) cannot overlap serialized chain arrivals — it waits for full presence. Do not "simplify" C1–C3 back into one big `sum_blocks(N, …)`; that reintroduces the reference's zero-overlap weakness inside a single dispatch |

## Structural impossibilities (pipeline-mode note)

`eval/golden_tests/reduce_scatter_average/feature_spec.py` already exists and is authoritative (not edited here). `INVALID = []` is correct for its TARGET (TILE-only layout, float dtypes — every cell constructible); no additional candidates.

**Golden-harness defects to fix via `/golden-tests` BEFORE golden verification** (they fail every cell regardless of op correctness):
1. `eval/golden_tests/reduce_scatter_average/helpers.py:95` calls `reduce_scatter(...)` — an undefined name in that module (only `reduce_scatter_average` is imported at `:26`) → `NameError` on every case.
2. `helpers.py:75-81` builds a **SUM** oracle (`.sum(dim=0)`, no `/ num_devices`) contradicting the op's MEAN semantics — PCC (scale-invariant) would pass but the `rms` half of `tolerance=(0.99, 0.05)` will not.

## Acceptance criteria mapping

| Requirement | Where satisfied |
|-------------|-----------------|
| Device i output = slice i of fp32-accumulated mean, PCC ~0.99 | `test_reduce_scatter_average` (bf16 0.99 / fp32 0.999, oracle fp32-accumulated over the quantized shards) |
| `output_tensor` path returns the supplied handle | `test_reduce_scatter_average_output_tensor` (`buffer_address()` equality) |
| Program-cache hit on 2nd identical call, semaphores survive | `test_reduce_scatter_average_program_cache` (2 calls; catches a missing R1 re-arm as a hang) |
| ONE `generic_op` dispatch, compute overlaps arrival | single MeshProgramDescriptor dispatch; T4/T7 overlap contract |
| Loud shape rejection | `test_reduce_scatter_average_rejects_unsplittable_width` (`pytest.raises(ValueError)`) |

## Hardware Constraints checklist

- [x] CB sync: push count = wait count for every CB (audit table above)
- [x] DEST: `g ≤ 4` under `fp32_dest_acc_en` (SyncHalf) — asserted at `BlockAccumulate::arm`
- [x] Reduce-scaler pool-type API: N/A — no `reduce_tile` in this op; the scaler here is a bcast-scalar operand, filled by `generate_bcast_unary_scalar` (bf16) / raw store (fp32), NOT by `prepare_reduce_scaler` (whose contract forbids non-reduce use)
- [x] Page sizes tile-aligned; relay CB pages rounded to L1 alignment
- [x] All `cb_wait_front` calls on a given CB use one page count (`g`, 1 page for relay, 1 for scaler)
- [x] Helpers not wrapped with extra CB ops (sum_blocks/BlockAccumulate/generate_bcast_unary_scalar own their protocols; the raw C4 pass owns its own)
- [x] `binary_op_init_common` (hw startup) before any helper usage
