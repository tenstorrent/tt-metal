# Operation Design: all_reduce

## Overview

| Field | Value |
|-------|-------|
| Classification | CCL (compute-CCL: collective movement + tile reduction) |
| Goal | Element-wise SUM of all N devices' same-shape shards on a MeshDevice line; EVERY device ends up holding the identical full sum. Output shape/dtype/layout == a single input shard. |
| Math | `output[...] = Σ_{j=0..N-1} shard_j[...]` — identical on every device |
| Dispatch | **ONE `ttnn.generic_op` per invocation** — a single program per mesh coordinate in one `ttnn.MeshProgramDescriptor`. Compute overlaps fabric arrival via per-block semaphore signaling (Dataflow Strategy, T4/T7). A sequential gather-then-reduce two-dispatch split (the reference `reduce_scatter` shape) is explicitly forbidden by the acceptance criteria. |
| Mode | Derivative (relay dataflow + arrival-major reduce pipeline derived from the hardware-validated `reduce_scatter_average` single-dispatch shape; all_reduce is that op with the slice walk, the 1/N scaler, and the scatter deleted — the reduce covers the FULL shard and the drain is a helper copy) |
| References | `ttnn/ttnn/operations/reduce_scatter_average/` (single-dispatch 5-kernel shape, two-semaphore overlap contract, host factory — verified on real 4-chip Blackhole), `ttnn/ttnn/operations/reduce_scatter/` (relay block-flow table, validate ordering, registry contract), `ttnn/ttnn/operations/all_gather/` (line store-and-forward, mesh fixture), `ttnn/cpp/ttnn/kernel_lib/ccl_helpers_dataflow.hpp`, `ttnn/cpp/ttnn/kernel_lib/accumulate_helpers_compute.hpp` |

**Do not wrap any existing CCL op.** All five kernels below are newly authored under `ttnn/ttnn/operations/all_reduce/kernels/`.

## Parameters

| Name | Type | Required | Valid Range | Default | Notes |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | TILE, interleaved, bf16/fp32, on a (1, N) MeshDevice line, N ≥ 2 | — | one SAME-shape shard per device |
| `topology` | `ttnn.Topology` | no | Phase-0: `Linear` | `Linear` | import as `from ttnn._ttnn.operations.ccl import Topology as _Topology` for the module-level default (the `ttnn.Topology` alias binds only after `ttnn.operations` auto-import — reference `reduce_scatter.py:43`) |
| `output_tensor` | `ttnn.Tensor \| None` | no | shard shape, same dtype/layout/buffer_type as input | `None` | write into the supplied tensor and return the SAME handle |

There is **no `dim` parameter** — all_reduce has no scatter/gather axis (the feature spec TARGET declares no index axis), so no canonicalization step exists.

## Tensors

### Input (per-device shard)

| Property | Requirement |
|----------|-------------|
| Shape | `(B, C, H, W)`, rank 4, `H % 32 == 0`, `W % 32 == 0` (Phase-0: tile-aligned; ValueError otherwise, no silent padding) |
| Dtype | `bfloat16` (primary), `float32` |
| Layout | TILE |
| Memory | interleaved, DRAM or L1 (`ValueError` if `input_tensor.is_sharded()`) |
| Mesh | `(1, N)` line, `N = prod(mesh_device.shape) ≥ 2` |

### Output (per-device — identical content on every device)

| Property | Value |
|----------|-------|
| Shape / Dtype / Layout / Memory | same as one input shard |
| Allocation when `output_tensor is None` | `ttnn.allocate_tensor_on_device(ttnn.Shape(shard_shape), dtype, layout, mesh_device, input.memory_config())`; every output page is written, no seeding needed |

### Op-internal gather buffer (fabric landing target)

| Property | Value |
|----------|-------|
| Shape | `(B*N, C, H, W)` — block `c` (source device `c`) occupies pages `[c*P, (c+1)*P)`; row-major page order makes this exact for any B/C (P = B·C·Ht·Wt) |
| Dtype / Layout / Memory | same as input (mesh-allocated interleaved ⇒ **uniform buffer address across devices**, which is what lets a fabric `write_page` target the neighbour's block through the LOCAL `TensorAccessor` routed one hop) |
| Lifetime | allocated fresh per call, passed in `io_tensors`; own block (`c == my_chip_id`) is never written — the reduce reader takes the own contribution directly from the input tensor (deletes the reference `reduce_scatter`'s serialized self-copy) |

### Derived quantities (host computes once; symbols used throughout)

| Symbol | Formula | Meaning | Acceptance range |
|--------|---------|---------|------------------|
| `N` | `prod(mesh_device.shape)` | devices on the line | 4 (bh_quietbox_1x4_hw) |
| `P` | `input.buffer_num_pages()` | tiles per shard = tiles per contribution = **output tiles** (full-shard reduce: the reduce_scatter_average `S` collapses to `P`) | 1–8 |
| `g` | largest of `{4, 2, 1}` dividing `P` | CB/DEST granule; `g ≤ DEST_AUTO_LIMIT = 4` under `fp32_dest_acc_en` + SyncHalf (`dest_helpers.hpp:103`); **`g` divides `P`** so no tail chunk ever exists |
| `page_size` | `input.buffer_page_size()` | tile bytes (bf16 2048 / fp32 4096) | |
| `aligned_page_size` | `round_up(page_size, ttnn.get_l1_alignment())` | relay CB page | = page_size (tile pages are already 16B-aligned) |

## Dataflow Strategy

### Algorithm

Line store-and-forward **gather of whole shards** (the hardware-validated relay pattern) fused in the SAME program with an **arrival-ordered incremental reduce over the full shard**. Every device receives all N−1 remote shards into its local `gather_buffer`; a dedicated reduce core consumes contributions one at a time — own shard first, then each arrival the moment its semaphore lands — so the accumulate of contribution *k* overlaps the fabric flight of contribution *k+1*. After the last accumulate, a helper copy drains the resident accumulator to the output writer.

Why this decomposition: it is byte-for-byte the `reduce_scatter_average` traffic + overlap contract that was verified green on this exact 4-chip Blackhole box (relay half = the reference `reduce_scatter` Phase-A pattern; reduce half = arrival-major `BlockAccumulate`), minus three things all_reduce doesn't need — the slice walker (the reduce covers dense pages `0..P-1`), the 1/N scaler CB + raw bcast-scalar pass (SUM needs no scale; the drain becomes a `sum_blocks` degenerate copy, making the compute kernel **all-helper**), and the scatter (output = full shard on every device). The bandwidth-optimal reduce-scatter+all-gather decomposition (N× less fabric traffic, `LineSliceCursor`/`LineChannelWalk`/`SyncCadence` step machine agreed across 3 kernels) is deferred — same deferral, same reasoning as both reference reduce collectives (`reduce_scatter/op_design.md` "Why gather-then-reduce", `reduce_scatter_average/op_design.md` Refinement 4).

### Per-device data path

```
                    device i  (one program, one generic_op dispatch)
 core (0,0)  relay FWD:  input ──reader──▶ cb_relay_pages ──writer──▶ fabric 1 hop right
                          gather_buffer (fwd arrivals) ──reader──▶ cb_relay_pages ──writer──▶ (relay onward)
 core (0,1)  relay BWD:  mirror of (0,0), 1 hop left
 core (0,2)  reduce:     input pages 0..P-1        ──reader──▶ cb_contributions ─┐
                          gather_buffer block c, pages c*P..c*P+P-1 (per arrival) ─┤ (arrival order, g-granules)
                                                                                   ▼
                          compute: seed-copy ▶ (N-1)× incremental add ▶ drain-copy ▶ cb_summed
                                                                                   ▼
                          writer: output tensor (dense tiles 0..P-1)
```

### Tensix-to-Tensix / device-to-device contract

| # | Contract | Detail |
|---|----------|--------|
| T1 | Fwd channel carries left→right traffic | device `i` fwd-sends `1 + i` blocks (own shard first, then relays of its `i` fwd arrivals) iff it has a right neighbour (`i < N-1`); fwd arrivals on device `i` = `i` blocks, nearest-first: shards of `i-1, i-2, …, 0`. |
| T2 | Bwd channel carries right→left traffic | mirror: bwd-sends `1 + (N-1-i)` iff `i > 0`; bwd arrivals = `N-1-i`, order `i+1, i+2, …, N-1`. Invariant: `fwd_arrivals + bwd_arrivals = N-1` on every device. |
| T3 | Block indices on the wire | send k=0 → own block `i`; relay k ≥ 1 → fwd `(i + N - k) % N`, bwd `(i + k) % N` (ring-modular form; never wraps on Linear). Fabric `write_page` targets the NEXT device's `gather_buffer` pages `[c*P, (c+1)*P)` through the local accessor (uniform mesh address). |
| T4 | Arrival signaling — the overlap mechanism | after each block's last page the sending writer issues **two** fabric atomic incs on its armed inc channel, both 1 hop: `sem_dir` at the receiving relay core ((0,0) fwd / (0,1) bwd) AND `sem_dir` at the receiving reduce core (0,2). Incs are in-order behind the pages on the same connection, so `sem ≥ k` ⇒ blocks 1..k fully landed in `gather_buffer`. |
| T5 | Semaphores | TWO op-internal `GlobalSemaphore`s, `sem_fwd` and `sem_bwd` (one address each, a private counter per core). Consumers: relay reader (0,0) waits `sem_fwd`; relay reader (0,1) waits `sem_bwd`; reduce reader (0,2) two-way-polls BOTH. Each consumer re-arms its OWN core's counter to 0 after its last wait (cache-reuse footgun, `ccl_helpers_dataflow.hpp:113-121`). Waits/resets are op-owned — there is no receiver helper. |
| T6 | Relay forwarding | relay reader waits `sem_dir ≥ k+1`, reads arrival k's `P` pages from the LOCAL `gather_buffer` back into `cb_relay_pages`; writer forwards them one more hop. Relayed blocks are a PREFIX of arrivals (`num_relays = num_sends-1 ≤ num_arrivals`). A line-end device (`num_sends == 0`) relays nothing but still waits ALL its arrivals and re-arms the sem. |
| T7 | Overlap timeline | reduce compute runs pass 0 (own shard) immediately; pass k runs as soon as the k-th arrival's double-inc lands, while arrival k+1 is still being relayed/flown. The only serialized tail is the drain copy of `P` tiles. |
| T8 | Deadlock freedom | relay chains are per-direction DAGs (no cycles); the reduce pipeline consumes only sems + DRAM reads + its own core's CBs; each device's egress (seed own shard) depends on nothing remote. No CB is shared across cores. |

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | whole op per device; within a device, fixed roles on 3 cores |
| Grid | logical cores `(0,0)` fwd relay, `(0,1)` bwd relay, `(0,2)` reduce — `ttnn.CoreRangeSet` of singleton ranges. NoC targets via `mesh_device.worker_core_from_logical_core(...)`, identical logical→physical mapping across devices (reference precedent, hardware-validated) |
| Per-core work | (0,0)/(0,1): `num_sends_dir * P` pages relayed, `num_arrivals_dir` waits; (0,2): reader `N*P` tile reads, compute `(N+1)*P` tile-ops (N·P adds/copies + P drain copies), writer `P` tile writes |
| Remainder | none — `g` divides `P` by construction; every CB interaction is a whole `g`-granule (or whole page for relay) |
| Multi-core reduce | deliberately NOT Phase-0: splitting `P` positions across reduce cores multiplies the per-block inc fan-out (each reduce core needs its own sem counter inc'd). Acceptance/golden shards have P ≤ 8 — one core is right-sized. Beyond-TARGET candidate. |
| `split_work_to_cores` | not used — fixed 3-core role assignment; there is no divisible tile-grid work unit to balance (the relay work is direction-asymmetric by the line position, the reduce is arrival-serialized) |

## Circular Buffers

All CBs are core-local (no CB spans cores; the relay CB is declared on the 2-core relay range, giving each core its own instance). **Capacity rule**: every CB's capacity is a multiple of its interaction quantum, so a multi-page reserve/wait never straddles the ring wrap (linear `l1 += page_size` writes after a multi-page reserve require contiguity).

| Semantic Name | Index | Cores | Page Size | Num Pages | Format | Producer | Consumer | Lifetime / quantum |
|---------------|-------|-------|-----------|-----------|--------|----------|----------|--------------------|
| `cb_contributions` | 0 | (0,2) | `page_size` | `2*g` | input dtype | reduce reader | reduce compute | streaming, quantum `g` (double-buffered granules); carries all N contributions in arrival order, own first, each in dense page order 0..P-1 |
| `cb_relay_pages` | 16 | (0,0), (0,1) | `aligned_page_size` | 2 | input dtype | relay reader | relay writer | streaming, quantum 1 page (double-buffered) |
| `cb_summed` | 17 | (0,2) | `page_size` | `2*g` | reduce compute | reduce writer | streaming, quantum `g` |
| `cb_accumulator` | 24 | (0,2) | `page_size` | `P` | reduce compute **only** | reduce compute | resident running sum; quantum `g`; capacity exactly `P` (`g` divides `P` ⇒ wrap-safe). **Single-producer invariant — see R2** |

No scaler CB — SUM needs no scaling operand (this deletes `reduce_scatter_average`'s `cb_scaler` and its raw bcast-scalar pass entirely).

L1 budget (worst acceptance case, bf16 `(1,1,64,128)`: P=8, g=4, page 2048): accumulator 16 KB + contributions 16 KB + summed 16 KB on (0,2); 4 KB per relay core. Growth cliff: `cb_accumulator = P` pages ⇒ `validate()` rejects (ValueError) shards with `P * page_size > 512 KiB`; large-P spill is a beyond-TARGET candidate.

### Semaphores

| Name | Kind | Created | Inc'd by | Waited/reset by |
|------|------|---------|----------|-----------------|
| `sem_fwd` | GlobalSemaphore, initial 0, reserved on the full worker grid | once per `mesh_device`, module cache `_SEMAPHORE_CACHE[id(mesh_device)] = (sem_fwd, sem_bwd)`, ONE `ttnn.synchronize_device` inside the miss branch only | left neighbour's fwd writer: 2 fabric incs/block → cores (0,0) and (0,2) | (0,0) relay reader; (0,2) reduce reader — each resets its OWN core's counter to 0 after its final wait |
| `sem_bwd` | same | same miss branch | right neighbour's bwd writer: 2 fabric incs/block → cores (0,1) and (0,2) | (0,1) relay reader; (0,2) reduce reader |

Both parked on `mesh_program_descriptor.semaphores = [sem_fwd, sem_bwd]` (the framework keeps their L1 alive across program-cache hits; excluded from the cache hash). Addresses via `ttnn.get_global_semaphore_address(...)` baked into runtime args. **No per-call post-dispatch barrier.** (`ttnn._ttnn.fabric.make_ccl_semaphore`, `fabric.cpp:268-270`, bundles allocation + barrier one-at-a-time; the two-semaphore module cache with a single shared barrier is kept instead — the proven reference shape.)

## Host Assembly (program factory)

Module layout mirrors the reference: `all_reduce.py` (signature, registry contract, validate, semaphore cache, allocation, single `ttnn.generic_op` call) + `all_reduce_program_descriptor.py` (mesh PD factory) + `__init__.py` re-exporting `all_reduce`, `SUPPORTED`, `EXCLUSIONS`, `INPUT_TAGGERS` at package level.

| Duty | Mechanism |
|------|-----------|
| Mesh PD | `ttnn.MeshProgramDescriptor()`; one `ttnn.ProgramDescriptor(kernels=[...], semaphores=[], cbs=[...])` per `ttnn.MeshCoordinateRange(coord_i, coord_i)` — programs are per-device DISTINCT (CT args: `my_chip_id`, send/arrival counts) |
| Dispatch | **exactly one** `ttnn.generic_op([input_tensor, gather_buffer, output_tensor], mesh_pd)` — output preallocated and LAST in `io_tensors` |
| Block flow | Linear table (T1/T2): `fwd_sends = 1 + i if i < N-1 else 0`, `fwd_arrivals = i`, `bwd_sends = 1 + (N-1-i) if i > 0 else 0`, `bwd_arrivals = N-1-i` |
| Routes | `ttnn._ttnn.fabric.ccl_dm_route(mesh_device, coord_i, coord_neighbour, topology)` per non-idle direction (`fabric.cpp:254-259`); `assert route.num_hops == 1` (store-and-forward invariant). The route owns the fwd/bwd sign reversal — never hand-derive `is_forward` |
| Fabric conn args | `ttnn._ttnn.fabric.build_ccl_fabric_rt_args(fabric_id_i, route.neighbor_id, 0, program, relay_core, route.is_forward)` (`fabric.cpp:278-297`) — emits `[has_forward][fwd conn][has_backward][bwd conn]`, placed FIRST in each relay writer's rt args; it MUTATES the program (appends SemaphoreDescriptors), so relay writers are constructed with EMPTY rt args and both the block and the op args are appended post-construction via the live `program.kernels[k].runtime_args[x][y]` view |
| Packet framing | 1 page = 1 fabric packet (`arm_unicast_write(page_size)`); `ccl_packet_dims` NOT used — same documented rejection as all three reference collectives; tile pages (2048/4096 B) fit a single packet, hardware-validated at both dtypes |
| Idle direction | empty rt-arg list `[]` + CT `num_sends = 0`; kernel no-ops under `if constexpr` and reads NO rt args (R10) |
| Compute config | `ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False, dst_full_sync_en=False)` — fixes `DEST_AUTO_LIMIT = 4`; fp32 DEST accumulation covers the bf16 sum-of-N rounding budget and the float32 secondary dtype |
| TensorAccessors | `list(ttnn.TensorAccessorArgs(t).get_compile_time_args())` appended LAST after all scalar CT args, for: input + gather (relay reader), gather (relay writer), input + gather (reduce reader), output (reduce writer) |

Information each kernel needs (exact CT/RT index layout is the implementer's choice — derive from the CB table and helper signatures):

| Kernel (file under `kernels/`) | Core(s) | Needs |
|---|---|---|
| `all_reduce_relay_reader.cpp` | (0,0)+(0,1), one source, CT-selected direction | CT: cb_relay_pages, direction, `my_chip_id`, `N`, `num_sends`, `num_arrivals` + input TA + gather TA. RT: input addr, gather addr, `P`, page_size, own-direction sem addr |
| `all_reduce_relay_writer.cpp` | (0,0)+(0,1), one source | CT: cb_relay_pages, direction, `my_chip_id`, `N`, `num_sends`, l1_alignment + gather TA. RT (num_sends > 0 only): fabric conn block FIRST, then gather addr, `P`, page_size, `num_hops(=1)`, sem addr, NoC xy of neighbour's same-direction relay core AND reduce core |
| `all_reduce_reduce_reader.cpp` | (0,2) | CT: cb_contributions, `my_chip_id`, `N`, `fwd_arrivals`, `bwd_arrivals`, `P`, `g` + input TA + gather TA. RT: input addr, gather addr, page_size, `sem_fwd` addr, `sem_bwd` addr |
| `all_reduce_reduce_compute.cpp` | (0,2) | CT: cb_contributions, cb_accumulator, cb_summed, `N`, `P`, `g`. No rt args |
| `all_reduce_reduce_writer.cpp` | (0,2) | CT: cb_summed, `P`, `g` + output TA. RT: output addr, page_size |

Five kernel sources, seven kernel *descriptors* per program (4 relay — fwd/bwd reader/writer with per-direction CT args — + 3 reduce).

## API Mapping

Every mechanism with verified file:line. Type `helper` or `raw_api`. All paths relative to repo root; dataflow helper = `ttnn/cpp/ttnn/kernel_lib/ccl_helpers_dataflow.hpp` ("dataflow.hpp"), accumulate helper = `ttnn/cpp/ttnn/kernel_lib/accumulate_helpers_compute.hpp` ("accum.hpp") / `.inl`.

| Phase | Type | Function | File:Line | Params / Notes | Input CB | Output CB | Owns CB ops? |
|-------|------|----------|-----------|----------------|----------|-----------|--------------|
| Relay egress open | helper | `dataflow_kernel_lib::ccl::FabricStreamSender<>` ctor / `open(unicast_route(num_hops))` | dataflow.hpp:492 (ctor), :503 (open), :302 (unicast_route) | `ConnT = DirectConn`; ctor `(conn_arg_idx, is_forward, alignment)`; `is_forward` peeked from the fabric block's leading `has_forward` rt arg; sender declared before (outlives) the stream; route bound ONCE at open | — | — | — |
| Relay page send | helper | `FabricStream::arm_unicast_write(page_size)` → `UnicastWriteChannel::write_page(src_l1, page_idx, gather_accessor)` | dataflow.hpp:423 (arm), :327 (write_page) | invariant per-page payload; page lands in the NEXT device's `gather_buffer` at `c*P + p` | `cb_relay_pages` (wait/pop owned by the KERNEL) | remote `gather_buffer` | no — kernel owns `cb_wait_front`/`noc_async_writes_flushed()`/`cb_pop_front` (R7) |
| Arrival signal ×2 | helper | `FabricStream::arm_inc(1)` → `AtomicIncChannel::inc(noc_addr)` twice per block | dataflow.hpp:435 (arm), :368 (inc) | one armed channel, two issues: relay-core sem, reduce-core sem via `safe_get_noc_addr(x, y, sem_addr, 0)`; in-order behind pages on the same connection (R8) | — | remote sems | — |
| Egress close | helper | `FabricStream::close()` | dataflow.hpp:461 | drains write + atomic barriers; RAII backstop at :418 | — | — | — |
| Arrival wait (relay) | raw_api (op-owned by design) | `noc_semaphore_wait_min(sem_ptr, k+1)`; re-arm `noc_semaphore_set(sem_ptr, 0)` | banner dataflow.hpp:113-121 assigns the WAIT half + reset to the op ("there is no FabricStreamReceiver") | reset AFTER the final wait, on every role incl. pure line-end receivers (R1) | — | — | — |
| Arrival poll (reduce) | raw_api (op-owned by design) | two-way poll: `invalidate_l1_cache()` + volatile reads of the `sem_fwd`/`sem_bwd` L1 words; consume whichever direction has an unconsumed arrival; reset BOTH after | no helper exists — receive-side sync is explicitly outside the dataflow helper (banner :113-121); a two-counter any-of wait has no primitive, and a single-counter `noc_semaphore_wait_min` would serialize the directions, losing overlap whenever the other direction lands first | loop bound = `fwd_arrivals + bwd_arrivals = N-1` | — | — | — |
| DRAM reads (reduce reader, dense) | raw_api over helper accessor | `TensorAccessor::get_noc_addr(idx)` + `noc_async_read` per tile, `noc_async_read_barrier` per g-granule | `tech_reports/tensor_accessor/tensor_accessor.md`; TA CT args from `ttnn.TensorAccessorArgs` | own contribution: pages `0..P-1` of input; arrival from `src`: pages `src*P + 0..P-1` of gather_buffer — plain `base + t`, identical dense order every contribution (positional alignment across passes, R11) | — | `cb_contributions` | kernel owns reserve/push per g-granule |
| Compute boot | raw_api (mandated pre-condition) | `binary_op_init_common(cb_contributions, cb_accumulator, cb_summed)` | accum.hpp:116-117 ("@pre The kernel has already run its hardware startup") and :211; ownership note :70-77 explains why `arm()` deliberately does not issue it | once at kernel start; NOT interchangeable with `compute_kernel_hw_startup` | — | — | — |
| Seed copy (contribution 0 = own) | helper | `compute_kernel_lib::sum_blocks(cb_contributions, cb_accumulator, /*num_blocks=*/1, /*block_num_tiles=*/g, /*pop_input=*/true)` × `P/g` | decl accum.hpp:221-222; `num_blocks == 1` degenerates to a copy of block 0 (accum.hpp:217, .inl:106-157) | `pop_input=true` is load-bearing (R4); post-condition: leaves `add_tiles_init` in acc_to_dest mode (accum.hpp:212-213) ⇒ `rearm()` before the runs (R3) | `cb_contributions` (g, popped) | `cb_accumulator` (g) | yes — owns wait/reserve/pop/push + tile_regs + DEST chunking |
| Incremental accumulate (arrivals 1..N−1) | helper | `compute_kernel_lib::BlockAccumulate::arm(cb_contributions, cb_accumulator, cb_accumulator, g)`; `rearm()`; `run(g)` × `(N-1) * P/g` | arm accum.hpp:125 / .inl:12-24 (asserts `g ≤ DEST_AUTO_LIMIT`, .inl:17), run .inl:46-70, rearm .inl:26-36 | **in-place `cb_b == cb_out`**: sound because `run()` pops a and b BEFORE reserving out (.inl:58-63, "verified ordering") — with capacity exactly `P` the reserve always finds `g` free pages | `cb_contributions` (g) + `cb_accumulator` front (g) | `cb_accumulator` back (g) | yes |
| Drain copy (final sum → writer) | helper | `compute_kernel_lib::sum_blocks(cb_accumulator, cb_summed, /*num_blocks=*/1, /*block_num_tiles=*/g, /*pop_input=*/true)` × `P/g` | accum.hpp:221-222, .inl:106-157 | replaces `reduce_scatter_average`'s raw 1/N bcast-scalar pass with a documented helper degenerate copy; runs strictly AFTER the last `run()`; nothing follows, so its acc_to_dest post-condition is moot | `cb_accumulator` (g, popped) | `cb_summed` (g) | yes |
| DRAM writes (reduce writer, dense) | raw_api over helper accessor | `TensorAccessor::get_noc_addr(chunk*g + t)` + `noc_async_write`, `noc_async_write_barrier` per g-granule | — | contribution stream order = dense page order = output page order ⇒ tile `t` drained maps to output page `t` | `cb_summed` (g) | output tensor | kernel owns wait/pop per g-granule |
| Route resolution (host) | helper | `ttnn._ttnn.fabric.ccl_dm_route(mesh_device, coord_i, coord_next, topology)` | `ttnn/cpp/ttnn-nanobind/fabric.cpp:254-259` | owns fwd/bwd sign reversal; `num_hops` → rt arg, `is_forward` → fabric block layout, `neighbor_id` → conn args | — | — | — |
| Fabric conn block (host) | helper | `ttnn._ttnn.fabric.build_ccl_fabric_rt_args(src, neighbor, 0, program, core, is_forward)` | fabric.cpp:278-297 | mutates the program — append post-construction; block FIRST in the relay writer's rt args | — | — | — |
| Semaphore host lifecycle | helper | `ttnn.create_global_semaphore(mesh_device, worker_cores, 0)` ×2 + one `ttnn.synchronize_device` + `ttnn.get_global_semaphore_address` | reference-proven module-cache pattern (`reduce_scatter_average.py` / `reduce_scatter.py`) | miss-branch-only barrier; parked on `mpd.semaphores` | — | — | — |

### Helpers considered and rejected (mandatory justifications)

| Candidate | Rejection (concrete, cited) |
|-----------|------------------------------|
| `FabricDuplexSender` / `FabricDuplexStream` (dataflow.hpp:49-58 banner; class ~:799) | duplex fans every issue out to ALL connected directions from ONE core. The two directions here send DIFFERENT block sequences (fwd: `i, i−1, …`; bwd: `i, i+1, …`) from DIFFERENT cores — a shared-issue duplex stream cannot express it. Same rejection as both reference reduce collectives; the banner's "all_reduce's shape" note refers to the in-tree C++ `all_reduce_async` multicast algorithm, not this store-and-forward gather. |
| `FabricStreamSender::signal()` (dataflow.hpp:86-90) | terminal one-shot (open + one inc + close); the relay writer issues `num_sends*P` pages + `2*num_sends` incs on a persistent connection — the staged `open → arm → issue → close` path is required. |
| Whole-op position-major `sum_blocks(cb_contributions, cb_summed, N, P, true)` | requires all N contributions resident (waits `N*P` tiles up front, .inl:107) — an `N*P`-tile CB and ZERO overlap of compute with serialized chain arrivals. Violates the single-dispatch-overlap acceptance criterion and reference R15 ("overlap must be arrival-major"). |
| `BlockAccumulate::run_seeded(cb_seed, n)` (.inl:72-104) for the seed phase | it is the 3-input terminal ring step `out = seed + a + b`: waits and pops THREE CBs per call (.inl:73-75). The seed phase has ONE operand stream (own contribution) and an EMPTY accumulator — there is no (a, b) pair to wait on; a `run_seeded` would deadlock on `cb_wait_front(cb_b_)`. `sum_blocks(num_blocks=1)` is the documented copy (accum.hpp:217). |
| `BlockAccumulate::run_chunked(n, cap)` (.inl:159-188) | reserves `cb_out` BEFORE popping a/b (.inl:165-167) — with the in-place `cb_b == cb_out` accumulator at capacity `P` (full during a pass), the reserve never finds free pages: guaranteed deadlock. Also unnecessary: the host clamps `g ≤ DEST_AUTO_LIMIT` by construction. |
| Second armed `BlockAccumulate` (e.g. `arm(cb_contributions, cb_accumulator, cb_summed, g)` for the final pass, saving the drain copy) | accum.hpp:188-191: "unpack/math config is SINGULAR hardware state, so two differently-armed accumulators cannot coexist". The drain-copy phase costs `P ≤ 8` extra tile copies and stays inside the helper contract. |
| `compute_kernel_lib::reduce()` (`reduce_helpers_compute.hpp:11-22`) | reduces WITHIN a tensor along a dimension, collapsing tile dims via `reduce_tile` — the banner itself contrasts it with cross-CB accumulation ("This header reduces WITHIN a tensor along a dimension", accum.hpp:12-16 mirror). all_reduce must preserve every element. |
| `prepare_reduce_scaler` / `calculate_and_prepare_reduce_scaler` (`reduce_helpers_dataflow.hpp`) | no `reduce_tile` in this op and no scaler operand at all — SUM needs no scale (the checklist's pool-type-aware-scaler rule is N/A). The header's contract restricts scaler tiles to the reduce LLK. |
| Schedule helpers `RingRsSchedule`/`ring_rs_step_flags`/`RingSliceCursor` (:411), `LineSliceCursor` (:627), `LineChannelWalk` (:690), `SyncCadence`, `DimZeroChunkWalk` (:794) — `ttnn/cpp/ttnn/operations/ccl/shared_with_host/ccl_helpers_schedule.hpp` | these encode multi-step transfer walks with per-step reduce/forward flags and chunk cadences agreed across 3 kernels. This algorithm has NO slice walk and NO step machine: the cross-kernel contract collapses to host-derived per-direction counts (CT args) + "N contributions × P dense tiles in g-granules, arrival-ordered" — the same deliberate drift-minimization both reference reduce ops chose. Adopting a schedule object would add agreement surface, not remove it. |
| `SliceRowWalker` (sched.hpp:491) / `SequentialTileWalker` (sched.hpp:543; `next() = base_ + offset_++` :556) | the full-shard walk is the identity: tile `t` of every contribution is page `base + t` (`base = 0` for input, `src*P` for gather blocks) inside an already-indexed loop. The walker adds a mutable cursor with zero agreement value — same documented rejection as `reduce_scatter`'s dense output path. |
| `ttnn._ttnn.fabric.ccl_packet_dims` (fabric.cpp:246-252) | deliberately unused — 1:1 page↔packet framing with tile-page payloads, the shape all three reference collectives shipped and validated. Multi-page coalescing is a perf refinement only. |
| `mcast_pipe.hpp` (`SenderPipe`/`ReceiverPipe`) for the arrival fan-out | ABSENT from this clone's `ttnn/cpp/ttnn/kernel_lib/` (verified: 20 files, no `mcast_pipe.hpp`); also unneeded — a single reduce core receives its arrival signal directly via the sender's second fabric inc (T4), no intra-device multicast exists. |
| `eltwise_convenience.hpp` `copy` / `eltwise_chain.hpp` `CopyTile` for the drain copy | ABSENT from this clone's `ttnn/cpp/ttnn/kernel_lib/` (verified by directory listing; a kernel `#include` cannot compile here). `sum_blocks(num_blocks=1)` is the in-clone documented copy primitive (accum.hpp:217). |
| `tilize_helpers` / `untilize_helpers` | TILE→TILE op; no layout conversion anywhere in the pipeline. |

## Compute Phases (reduce_compute kernel, core (0,2))

`chunks = P / g` throughout. Compute is order-agnostic by construction: it counts passes; the READER decides arrival order.

| # | Operation | Helper? | Input CB (state) | Output CB | CB State After |
|---|-----------|---------|-------------------|-----------|----------------|
| C0 | `binary_op_init_common(cb_contributions, cb_accumulator, cb_summed)`; `acc = BlockAccumulate::arm(cb_contributions, cb_accumulator, cb_accumulator, g)` | pre-condition + helper factory | — | — | hw configured; accumulator armed |
| C1 | Seed: contribution 0 (own shard) — `chunks × sum_blocks(cb_contributions, cb_accumulator, 1, g, true)` | helper | `cb_contributions` (g per call, popped) | `cb_accumulator` (g per call) | `cb_accumulator` holds P tiles = own contribution; `cb_contributions` empty |
| C2 | `acc.rearm()` — restore after `sum_blocks`'s acc_to_dest + format post-condition (accum.hpp:212-213, .inl:26-36) | helper | — | — | add init back in non-accumulate mode |
| C3 | Incremental accumulate: `for k in 1..N-1: chunks × acc.run(g)` | helper | `cb_contributions` (g, popped) + `cb_accumulator` front (g, popped) | `cb_accumulator` back (g) | after pass k, `cb_accumulator` holds the P-tile running sum of contributions 0..k; FIFO order = dense page order preserved every pass |
| C4 | Drain: `chunks × sum_blocks(cb_accumulator, cb_summed, 1, g, true)` — a degenerate copy of the final sum, streamed to the writer | helper | `cb_accumulator` (g, popped) | `cb_summed` (g) | `cb_accumulator` empty; writer drains `cb_summed` |

All four CBs share the input dtype, so the boot init's data formats cover every phase with zero mid-kernel reconfig (C2's `rearm()` re-establishes them anyway per .inl:32-33). No compute rt args; everything is CT.

## Broadcast Verification

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast Dim |
|-------|----|--------------------|--------------------|---------------|
| C1/C4 | `copy_tile` seed inside `sum_blocks` (odd count = 1) | full `[32,32]` tile | — | None |
| C3 | `add_tiles(cb_contributions, cb_accumulator, i, i, i)` inside `run()` | full `[32,32]` per tile | full `[32,32]` per tile | None |

No broadcast op anywhere (the reference's SCALAR-bcast 1/N multiply is deleted).

## CB Sync Audit (push count == wait/pop count, per CB, per device)

| CB | Pushed | Waited/Popped | Balanced |
|----|--------|----------------|----------|
| `cb_relay_pages` (per relay core) | relay reader: `num_sends_dir * P` pages (seed `P` + `(num_sends_dir−1) * P` relay read-backs) | relay writer: `num_sends_dir * P` (wait 1 / pop 1) | ✓ (0 = 0 on an idle direction — reader pushes nothing when `num_sends == 0`) |
| `cb_contributions` | reduce reader: `N * P` (g-granules: own + `fwd_arrivals` + `bwd_arrivals` = N contributions) | compute: C1 pops `P` + C3 pops `(N-1)*P` | ✓ |
| `cb_accumulator` | compute only: C1 `P` + C3 back-pushes `(N-1)*P` = `N*P` | compute: C3 front-pops `(N-1)*P` + C4 pops `P` = `N*P` | ✓ |
| `cb_summed` | compute: `P` (g-granules) | reduce writer: `P` (g-granules) | ✓ |

All waits on a given CB use a single count (`g` on the reduce CBs, 1 page on the relay CB).

## Validation & Registry Contract (Phase-0)

| Item | Value |
|------|-------|
| Exports | `SUPPORTED`, `EXCLUSIONS`, `INPUT_TAGGERS`, `all_reduce` from both `all_reduce.py` and the package `__init__.py` (the golden harness reads them at PACKAGE level) |
| Exceptions | `from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue` inside `try/except ImportError` with local `NotImplementedError`-subclass fallback |
| Phase-0 `SUPPORTED` | `{"dtype": [ttnn.bfloat16, ttnn.float32], "layout": [ttnn.TILE_LAYOUT], "topology": [_Topology.Linear]}` — covers the ENTIRE feature-spec TARGET; `TARGET − SUPPORTED = ∅`, so there are no refinement cells |
| `INPUT_TAGGERS` / `EXCLUSIONS` | `{}` / `[]` (golden INPUTS are all tile-aligned by construction) |
| `validate()` ordering | universal structural (ValueError: MeshDevice, `(1, N)` line with N ≥ 2, rank 4, `H % 32` / `W % 32`, not sharded) → axis gate (`UnsupportedAxisValue` per SUPPORTED axis, then `ExcludedCell` over EXCLUSIONS) → axis-value-DEPENDENT structural (ValueError: `page_size % 16` fabric-payload gate, `P * page_size ≤ 512 KiB` accumulator L1 budget, `output_tensor` spec match: shard shape / dtype / layout / `memory_config().buffer_type`). This ordering is the verifier-blessed reference fix — an out-of-SUPPORTED value must never surface as a shape-derived ValueError |
| `validate()` returns | `num_devices` |
| Output path | allocate shard-shape output iff `output_tensor is None`; always write every page; return the same handle |

## Feature spec (pipeline mode)

`eval/golden_tests/all_reduce/feature_spec.py` **already exists and is authoritative** (not edited here): `TARGET = {dtype: [bfloat16, float32], layout: [TILE_LAYOUT], topology: [Linear]}`, `INPUTS = [((1,1,32,32),), ((1,1,64,128),), ((1,1,128,64),)]`, `INVALID = []`.

**Structural impossibilities**: none to add — every TARGET cell (TILE × float dtypes × Linear) is constructible; `INVALID = []` is correct. No index axes exist, so no sign-canonicalization note is needed.

## Program Cache & Semaphore Lifecycle

1. First call: module cache miss → create `sem_fwd`, `sem_bwd` (initial 0, full worker grid), ONE `ttnn.synchronize_device(mesh_device)`, cache under `id(mesh_device)`.
2. Every call: fresh `gather_buffer` + (if needed) output allocation; build the mesh PD; `mpd.semaphores = [sem_fwd, sem_bwd]`; one `generic_op`.
3. Second identical call is a program-cache HIT: same rt-arg shapes, same CT args, semaphores alive (parked on the descriptor), counters at 0 because every consumer re-armed after its final wait (R1). The acceptance `test_all_reduce_program_cache` exists to catch exactly a missing re-arm (first run green, second hangs).
4. Distinct rt-arg counts on line-end vs interior devices produce per-device distinct programs — correct and cache-stable.

## Beyond-TARGET candidates (not refinements — TARGET is fully covered)

| # | Candidate | Sketch |
|---|-----------|--------|
| 1 | `topology=Ring` | kernels are already ring-modular (T3); adopt the short-way depth table (fwd `N//2`, bwd `(N-1)//2`) + `ccl_dm_route(.., Ring)` wrap links — probe the wrap route under FABRIC_1D first |
| 2 | Multi-core reduce | split `P` positions across reduce cores; needs per-core arrival inc fan-out |
| 3 | Large-P support | spill/chunk the accumulator (lifts the `P * page_size ≤ 512 KiB` gate); fp32 accumulator CB under bf16 inputs to cut per-pass pack rounding |
| 4 | Bandwidth-optimal RS+AG decomposition | drops the N× gather traffic; the `LineSliceCursor`/`LineChannelWalk` + `SyncCadence` machine — extend the host gtest schedule sweeps (`tests/ttnn/unit_tests/gtests/ccl/test_ccl_helpers_schedule.cpp`) BEFORE any new schedule variant |
| 5 | Packet coalescing | `ccl_packet_dims` multi-page packets + per-chunk incs |
| 6 | ROW_MAJOR / bfloat8_b / sharded / non-tile-aligned | each needs its own pipeline change (tilize wrap, exp-shared pages, shard-aware addressing, padding semantics) |

## Key Risks and Gotchas

| # | Risk | Rule |
|---|------|------|
| R1 | **Cache-reuse semaphore re-arm** — first run green, second hangs (dataflow.hpp:113-121) | every consumer resets its OWN core's counter after its final wait: (0,0)→`sem_fwd`, (0,1)→`sem_bwd`, (0,2)→BOTH. On every role, including line-end pure receivers (a device with `num_sends == 0` still waits all its arrivals, then resets). Safe against racing senders: all incs for the run have been OBSERVED before the reset. |
| R2 | **`cb_accumulator` single-producer invariant** | ONLY the compute kernel ever reserves/pushes `cb_accumulator`. Do NOT have the reader seed it directly: each RISC keeps a LOCAL CB write pointer, so a second producer corrupts the ring. The seed goes through `cb_contributions` + `sum_blocks` copy. |
| R3 | **`sum_blocks` post-condition** — leaves `add_tiles_init` in acc_to_dest mode and may leave formats reprogrammed (accum.hpp:212-213) | `acc.rearm()` between C1 and the first `run()`. The C4 drain needs no rearm after it — nothing follows. |
| R4 | **`pop_input=true` on `sum_blocks`** | the default `false` deadlocks the reduce reader on `cb_reserve_back` once `cb_contributions` fills. Both C1 and C4 pass `true`. |
| R5 | **Granularity contract** | the CB protocol always moves whole `g`-granules; host guarantees `g ∈ {4,2,1}` divides `P`, so every wait/reserve is full and no tail exists. `g ≤ DEST_AUTO_LIMIT = 4` under `fp32_dest_acc_en` (asserted at `arm`, .inl:17; `static_assert` the CT mirror in the compute kernel). |
| R6 | **CB wrap contiguity** — multi-page reserve + linear `l1 += page_size` writes | every CB capacity is a multiple of its quantum (CB table); never mix quanta on one CB. |
| R7 | **`noc_async_writes_flushed()` before `cb_pop_front`** in the relay writer | the fabric write sources the CB slot; popping first lets the reader overwrite in-flight data. |
| R8 | **Inc-after-pages ordering** | both `inc()` issues go on the SAME connection as the block's `write_page`s, after the last page — fabric delivery is in-order per connection, making `sem ≥ k` imply data-complete. Never move the incs to another stream/connection. |
| R9 | **Line-end writers read NO rt args** | the whole relay-writer body sits inside `if constexpr (num_sends > 0)`; its rt-arg list is literally `[]`. Any unconditional `get_arg_val` before the guard reads garbage. |
| R10 | **Fabric block FIRST, appended post-construction** | `build_ccl_fabric_rt_args` mutates the program; relay writers are constructed with empty rt args, then block + op args are appended via the live `runtime_args[x][y]` view. The kernel's arg cursor starts at 0 with the block. |
| R11 | **Positional alignment across passes** | every contribution is streamed in the SAME dense order (pages 0..P-1 of its block). This is what keeps `add_tiles(t, t, t)` element-aligned across passes AND makes the writer dim-agnostic (tile `t` drained = output page `t`). |
| R12 | **Two armed accumulators cannot coexist** (accum.hpp:188-191) | one `arm()` per kernel; the drain is `sum_blocks`, not a second differently-armed instance. |
| R13 | **Mesh/topology contract** | acceptance + golden run on a `(1, 4)` Blackhole mesh with `FabricConfig.FABRIC_1D` (`bh_quietbox_1x4_hw`, via `scripts/run_multidevice_sim_pytest.py --runtime hardware --op all_reduce`). Any other mesh shape hangs fabric init (`Fabric Router Sync: Timeout`) or fails `system_mesh.cpp: requested_size <= system_size` — a test/topology mismatch, not an op defect. The test defaults to `(1, 4)` with a `CCL_HW_MESH_SHAPE` env override. |
| R14 | **`gather_buffer` fresh per call, mesh-allocated** | pass it in `io_tensors` (middle position; output LAST) so dispatch resolves and keeps it alive; mesh allocation gives the uniform-address property T3 depends on. |
| R15 | **Overlap is arrival-major by necessity** | position-major (`sum_blocks` over N resident blocks) cannot overlap serialized chain arrivals. Do not "simplify" C1–C3 into one big `sum_blocks(N, …)` — that reintroduces zero overlap and an `N*P`-tile CB. |
| R16 | **Topology import** | `from ttnn._ttnn.operations.ccl import Topology as _Topology` at module scope; `ttnn.Topology` binds too late for a module-level default. |

## Acceptance criteria mapping

| Requirement | Where satisfied |
|-------------|-----------------|
| Every device's output = host fp32-accumulated element-wise SUM of all N shards, PCC ~0.99 | `test_all_reduce` (bf16 0.99 / fp32 0.999, 4 shard shapes × 2 dtypes, asserted per device) |
| `output_tensor` path writes into the supplied tensor and returns it | `test_all_reduce_output_tensor` (`buffer_address()` equality + per-device PCC) |
| Program-cache hit on 2nd identical call, GlobalSemaphores survive (created once) | `test_all_reduce_program_cache` (2 calls; catches a missing R1 re-arm as a hang) |
| ONE `generic_op` dispatch, compute overlaps fabric arrival | single MeshProgramDescriptor dispatch; T4/T7 overlap contract; two-way arrival poll |
| Registry contract (`SUPPORTED`/`EXCLUSIONS`/`INPUT_TAGGERS`, `UnsupportedAxisValue`) | `all_reduce.py` exports + `validate()` axis gate |

## Hardware Constraints checklist

- [x] CB sync: push count = wait/pop count for every CB (audit table above)
- [x] DEST: `g ≤ 4` under `fp32_dest_acc_en` + SyncHalf — asserted at `BlockAccumulate::arm` (.inl:17) and `static_assert`ed in the compute kernel
- [x] Reduce-scaler pool-type API: N/A — no `reduce_tile` and no scaler operand anywhere in this op (SUM needs neither)
- [x] Sequential-helper intermediates: `cb_accumulator` holds the FULL P-tile block (resident across all N passes); streaming CBs double-buffered at their quantum
- [x] Page sizes tile-aligned; relay CB pages rounded to L1 alignment; 16B page gate in validate (fabric on-wire round-up must not overrun the next page)
- [x] All `cb_wait_front` calls on a given CB use one page count (`g` on reduce CBs, 1 on the relay CB)
- [x] Helpers not wrapped with extra CB operations (`sum_blocks`/`BlockAccumulate` own their protocols; the relay writer's wait/flush/pop around `write_page` is the documented op-owned half)
- [x] `binary_op_init_common` (hw startup) before any accumulate-helper usage (accum.hpp:116-117)
