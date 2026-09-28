# Operation Design: reduce_scatter

## Overview

| Field | Value |
|-------|-------|
| Classification | CCL (compute-CCL: collective movement + tile reduction + per-device-distinct scatter) |
| Goal | Element-wise SUM of all N devices' same-shape shards on a MeshDevice line; device `i` keeps only slice `i` (of N equal slices along `dim`) of that sum. |
| Math | `output_i[...] = (Σ_{j=0..N-1} shard_j)[..., i*(W/N) : (i+1)*(W/N)]` for `dim=3` (Phase-0) |
| Dispatch | **ONE `ttnn.generic_op` per invocation** — a single program per mesh coordinate in one `ttnn.MeshProgramDescriptor`. Compute overlaps fabric arrival via per-block semaphore signaling (Dataflow Strategy T4/T7). A sequential gather-then-reduce two-dispatch split is explicitly forbidden by the acceptance criteria. |
| Mode | Derivative architecture: the hardware-adopted single-dispatch `reduce_scatter_average` design (commit `8a55e385b9`) minus its 1/N mean epilogue. **No wrapping** — all five kernels are newly authored under `ttnn/ttnn/operations/reduce_scatter/kernels/`; no import/call/dispatch to any existing CCL op. |
| References (read-only) | `ttnn/ttnn/operations/reduce_scatter_average/` (adopted sibling: kernels, host factory, validate ordering), `ttnn/ttnn/operations/all_reduce/` + `all_gather/` + `point_to_point/` (host-assembly idioms, acceptance-test shape), `experimental/ccl/reduce_scatter_minimal_async` (silicon-verified line algorithm, contrast case), `ttnn/cpp/ttnn/kernel_lib/ccl_helpers_dataflow.hpp`, `ttnn/cpp/ttnn/kernel_lib/accumulate_helpers_compute.hpp`, `ttnn/cpp/ttnn/operations/ccl/shared_with_host/ccl_helpers_schedule.hpp` |

### Files to produce (implementer)

| File | Role |
|------|------|
| `ttnn/ttnn/operations/reduce_scatter/__init__.py` | re-export `reduce_scatter`, `SUPPORTED`, `EXCLUSIONS`, `INPUT_TAGGERS` (package-level — the golden harness imports from the package) |
| `ttnn/ttnn/operations/reduce_scatter/reduce_scatter.py` | signature, registry contract, `validate()`, semaphore cache, allocation, the single `ttnn.generic_op` call |
| `ttnn/ttnn/operations/reduce_scatter/reduce_scatter_program_descriptor.py` | `MeshProgramDescriptor` factory (one `ProgramDescriptor` per mesh coordinate) |
| `kernels/reduce_scatter_relay_reader.cpp` | relay cores (0,0)+(0,1), NCRISC — one source, direction CT-selected |
| `kernels/reduce_scatter_relay_writer.cpp` | relay cores, BRISC — fabric egress |
| `kernels/reduce_scatter_reduce_reader.cpp` | reduce core (0,2), NCRISC — own slice + two-way arrival poll |
| `kernels/reduce_scatter_reduce_compute.cpp` | reduce core, TRISC — arrival-ordered incremental sum |
| `kernels/reduce_scatter_reduce_writer.cpp` | reduce core, BRISC — dense output write |

## Parameters

| Name | Type | Required | Valid Range | Default | Notes |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | TILE, interleaved, bf16/fp32, on a (1, N) MeshDevice line, N ≥ 2 | — | one SAME-shape shard per device |
| `dim` | `int` | no | Phase-0: 3 (canonical). Negative aliases accepted (`-4 ≤ dim ≤ 3`, else ValueError) | `3` | **Canonicalize to POSITIVE (`dim % 4`, rank pinned to 4) BEFORE the SUPPORTED membership test** — the feature spec's TARGET uses the positive convention (`eval/golden_tests/reduce_scatter/feature_spec.py:38-47`) and the golden driver pre-canonicalizes (`helpers.py:54`), so `-1` must alias to `3` |
| `topology` | `ttnn.Topology` | no | Phase-0: `Linear` | `Linear` | import as `from ttnn._ttnn.operations.ccl import Topology as _Topology` at module scope (the top-level `ttnn.Topology` alias binds only after `ttnn.operations` auto-imports — `all_reduce.py:37`) |
| `output_tensor` | `ttnn.Tensor \| None` | no | slice shape, same dtype/layout/buffer_type as input | `None` | write into the supplied tensor and return the SAME handle |

Signature must be positionally callable exactly as `reduce_scatter(input_tensor, dim=3, topology=_Topology.Linear, output_tensor=None)` — the golden driver calls `reduce_scatter(ttnn_input, dim=scatter_dim, topology=topology)` (`eval/golden_tests/reduce_scatter/helpers.py:90`). Do NOT make `dim` keyword-only.

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

### Output (per-device DISTINCT)

| Property | Value |
|----------|-------|
| Shape | shard shape with `shape[dim] //= N` (dim=3: `(B, C, H, W/N)`) |
| Dtype / Layout / Memory | same as input |
| Allocation when `output_tensor is None` | `ttnn.allocate_tensor_on_device(ttnn.Shape(out_shape), dtype, layout, mesh_device, input.memory_config())`; every output page is written, no seeding needed |

### Op-internal gather buffer (fabric landing target)

| Property | Value |
|----------|-------|
| Shape | `(B*N, C, H, W)` — block `c` (source device `c`) occupies tile pages `[c*P, (c+1)*P)` |
| Dtype / Layout / Memory | same as input; mesh-allocated interleaved ⇒ **uniform buffer address across devices**, which is what lets a fabric `write_page` target the neighbour's block through the LOCAL `TensorAccessor` routed one hop |
| Lifetime | allocated fresh per call via `ttnn.allocate_tensor_on_device`, passed in `io_tensors` so dispatch resolves and keeps it alive; own block (`c == my_chip_id`) is **never written** (the reduce reader takes the own contribution directly from the input tensor — no serialized self-copy) |

### Derived quantities (host computes once; symbols used throughout)

| Symbol | Formula | Meaning | Golden range |
|--------|---------|---------|--------------|
| `N` | `prod(mesh_device.shape)` | devices on the line | 4 (hw `bh_quietbox_1x4_hw`), 8 (sim) |
| `P` | `input.buffer_num_pages()` | tiles per shard | 64–128 |
| `Wt` | `W / 32` | shard tile-columns | 8–16 |
| `slice_Wt` | `Wt / N` | output tile-columns | 1–4 |
| `Rt` | `B * C * (H / 32)` | total tile-rows (batches are contiguous row-blocks in tiled page order, so the shard is walked as an `Rt × Wt` tile grid — no per-batch logic for dim=3) | 8–16 |
| `S` | `P / N = Rt * slice_Wt` | output tiles per device | 8–32 |
| `g` | largest of `{4, 2, 1}` dividing `S` | CB/DEST granule; `g ≤ DEST_AUTO_LIMIT = 4` under `fp32_dest_acc_en` + SyncHalf (`dest_helpers.hpp:90-103`); **`g` divides `S`** so no tail chunk ever exists |
| `page_size` | `input.buffer_page_size()` | tile bytes (bf16 2048 / fp32 4096) | |
| `aligned_page_size` | `ttnn.round_up(page_size, ttnn.get_l1_alignment())` | relay CB page (= page_size for tile pages, both already aligned) | |

## Dataflow Strategy

### Algorithm

Line store-and-forward **gather of whole shards** (the hardware-validated relay traffic pattern shared by the adopted `reduce_scatter_average` and `all_reduce` Phase A) fused in the SAME program with an **arrival-ordered incremental reduce** on a dedicated third core. Every device receives all N−1 remote shards into its local `gather_buffer`; the reduce core consumes only slice `i` of each contribution, one contribution at a time — own shard first, then each arrival the moment its semaphore lands — so the accumulate of contribution *k* overlaps the fabric flight of contribution *k+1*. The per-device-distinct scatter falls out of WHICH slice the reduce core walks (`slice_tile_offset(dim, my_chip_id, …)`), not out of any output-side selection.

Why this decomposition (algorithm decision): the relay half is byte-for-byte the traffic pattern already proven on this hardware by three adopted collectives; the reduce half needs cross-kernel agreement only on a fixed per-contribution CB protocol (`g`-granule streaming of S tiles, N times) — the smallest drift surface that still satisfies the single-dispatch + overlap mandate. The bandwidth-optimal partial-sum line reduce-scatter (compute in the relay path; the `LineSliceCursor`/`LineChannelWalk`/`SyncCadence`/`line_rs_*` step machine of `reduce_scatter_minimal_async`) is deferred as Refinement 4 — it is a different algorithm with a 3-kernel step-flag agreement surface, an FWD-leads/BWD-accumulates output race on middle devices, and an on-device handoff semaphore, none of which the acceptance criteria require.

### Per-device data path

```
                    device i  (one program, one generic_op dispatch)
 core (0,0)  relay FWD:  input ──reader──▶ cb_relay_pages ──writer──▶ fabric 1 hop right
                          gather_buffer (fwd arrivals) ──reader──▶ cb_relay_pages ──writer──▶ (relay onward)
 core (0,1)  relay BWD:  mirror of (0,0), 1 hop left
 core (0,2)  reduce:     input slice i  ──reader──▶ cb_contributions ─┐
                          gather_buffer slice i of each arrived shard ─┤ (arrival order, g-granules)
                                                                       ▼
                          compute: seed-copy ▶ (N-1)× incremental add ▶ move ▶ cb_output_tiles
                                                                       ▼
                          writer: output tensor (dense tiles 0..S-1)
```

### Tensix-to-Tensix / device-to-device contract

| # | Contract | Detail |
|---|----------|--------|
| T1 | Fwd channel carries left→right traffic | device `i` fwd-sends `num_fwd_sends = 1 + i` blocks iff `i < N-1` (else 0): own shard first (k=0, from the input tensor), then relays of its `i` fwd arrivals (k=1..i, from the local `gather_buffer`). Fwd arrivals on device `i` = `fwd_arrivals = i` blocks, in chain order **nearest-first**: shards of `i-1, i-2, …, 0`. |
| T2 | Bwd channel carries right→left traffic | mirror: `num_bwd_sends = 1 + (N-1-i)` iff `i > 0` (else 0); `bwd_arrivals = N-1-i`, order `i+1, i+2, …, N-1`. Total arrivals per device = `N-1`; reduce core consumes `own + N-1 = N` contributions. |
| T3 | Block indices on the wire | fwd send k: `c = (i + N - k) % N` (k=0 → own `i`; k ≥ 1 → `i-k`); bwd send k: `c = (i + k) % N`. Ring-modular form equals plain linear indices for Linear and is Ring-ready (Refinement 1). Fabric `write_page` targets the NEXT device's `gather_buffer` pages `[c*P, (c+1)*P)` through the local accessor (uniform mesh address). |
| T4 | Arrival signaling — the overlap mechanism | after the last page of each block, the sending writer issues **two** fabric atomic incs on its armed inc channel, both 1 hop to the receiving device: `sem_dir` at the receiving relay core ((0,0) for fwd / (0,1) for bwd) AND `sem_dir` at the receiving reduce core (0,2). Incs are in-order behind the pages on the same connection, so `sem ≥ k` ⇒ blocks 1..k fully landed in `gather_buffer`. |
| T5 | Semaphores | TWO op-internal `GlobalSemaphore`s, `sem_fwd` and `sem_bwd` (one L1 address each, a private counter word per core). Consumers: relay reader (0,0) waits `sem_fwd`; relay reader (0,1) waits `sem_bwd`; reduce reader (0,2) polls BOTH. Each consumer re-arms its OWN core's counter to 0 after its last wait (cache-reuse footgun, `ccl_helpers_dataflow.hpp:116-121`). Waits/resets are op-owned — there is no receiver helper. |
| T6 | Relay forwarding | fwd relay reader waits `sem_fwd ≥ k` for relay k (k=1..fwd_arrivals), reads arrival k−1's `P` pages from the LOCAL `gather_buffer` back into `cb_relay_pages`; the writer forwards them one more hop. A line-end device (`num_sends == 0` in that direction) relays nothing but still waits `sem_dir ≥ arrivals_dir` before re-arming (a reset racing an in-flight inc corrupts the counter). |
| T7 | Overlap timeline | reduce compute runs pass 0 (own contribution) immediately; pass k runs as soon as the k-th arrival's double-inc lands, while arrival k+1 is still being relayed/flown. The only serialized tail after the last arrival's pass is the S-tile move phase C4 (a block-0 copy) plus the writer drain. |
| T8 | Deadlock freedom | relay chains are per-direction DAGs (no cycles); the reduce pipeline consumes only sems + local NoC reads + its own core's CBs; each device's egress (seed own shard) depends on nothing remote. No CB spans cores. |

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | whole op per device; within a device, fixed roles on 3 cores |
| Grid | logical cores `(0,0)` fwd relay, `(0,1)` bwd relay, `(0,2)` reduce — `ttnn.CoreRangeSet` of the three singleton ranges. Uniform across devices so the peer NoC coordinates are mesh-wide identical; physical coords via `mesh_device.worker_core_from_logical_core(...)` |
| Per-core work | (0,0)/(0,1): `num_sends_dir * P` pages relayed, `arrivals_dir` waits; (0,2): reader `N*S` tile reads, compute `N*S` add/copy-equivalents + `S` moved tiles, writer `S` tile writes |
| Remainder | none — `g` divides `S` by construction; every CB interaction is a whole `g`-granule (or a whole page for relay) |
| Multi-core reduce | deliberately NOT Phase-0: splitting `S` across reduce cores multiplies the per-block inc fan-out. Golden shapes have S ≤ 32 — one core is right-sized. Refinement 3. |

## Circular Buffers

All CBs are core-local (no CB spans cores). **Capacity rule**: every CB's capacity is a multiple of its interaction quantum, so a multi-page reserve/wait never straddles the ring wrap (linear `l1 += page_size` writes after a multi-page reserve require contiguity).

| Semantic Name | Index | Cores | Page Size | Num Pages | Format | Producer | Consumer | Lifetime / quantum |
|---------------|-------|-------|-----------|-----------|--------|----------|----------|--------------------|
| `cb_relay_pages` | 16 | (0,0), (0,1) | `aligned_page_size` | 2 | input dtype | relay reader | relay writer | streaming, quantum 1 page (double-buffered) |
| `cb_contributions` | 0 | (0,2) | `page_size` | `2*g` | input dtype | reduce reader | reduce compute | streaming, quantum `g` (double-buffered granules); carries all N contributions in arrival order, own first |
| `cb_accumulator` | 24 | (0,2) | `page_size` | `S` | input dtype | reduce compute **only** | reduce compute (passes C3) + move phase C4 | resident running sum; quantum `g`; capacity exactly `S` (`g` divides `S` ⇒ wrap-safe). **Single-producer invariant — see R2** |
| `cb_output_tiles` | 17 | (0,2) | `page_size` | `2*g` | input dtype | reduce compute (C4) | reduce writer | streaming, quantum `g` |

There is **no scaler CB** — the sum is the output; nothing multiplies it. All four CBs carry the input dtype, so the boot `binary_op_init_common` covers every phase with zero mid-kernel data-format reconfig.

L1 budget (worst golden case, fp32 `(1,1,256,512)` on N=4: S=32, g=4, page 4096): `cb_contributions` 32 KB + `cb_accumulator` 128 KB + `cb_output_tiles` 32 KB ≈ 192 KB on (0,2); 8 KB on each relay core. Growth cliff: `cb_accumulator = S` pages ⇒ Phase-0 rejects (ValueError) shards with `S > 256`; Refinement 5 lifts it.

### Semaphores

| Name | Kind | Created | Inc'd by | Waited/reset by |
|------|------|---------|----------|-----------------|
| `sem_fwd` | GlobalSemaphore, initial 0, all worker cores | once per `mesh_device`, cached (`_SEMAPHORE_CACHE[id(mesh_device)] = (sem_fwd, sem_bwd)`), ONE `ttnn.synchronize_device` inside the miss branch only, covering both creations | left neighbour's fwd relay writer: 2 fabric incs/block → this device's cores (0,0) and (0,2) | (0,0) relay reader; (0,2) reduce reader — each resets its OWN core's counter to 0 after its final wait |
| `sem_bwd` | same | same miss branch | right neighbour's bwd relay writer: 2 fabric incs/block → cores (0,1) and (0,2) | (0,1) relay reader; (0,2) reduce reader |

Both parked on `mesh_program_descriptor.semaphores = [sem_fwd, sem_bwd]` (guard with `hasattr` for older bindings) — kept alive across the program cache, excluded from the cache hash (`program_descriptors.cpp:1077-1087`). Addresses via `ttnn.get_global_semaphore_address(...)` baked into runtime args. **No per-call post-dispatch barrier.**

## Host Assembly (program factory)

| Duty | Mechanism |
|------|-----------|
| Mesh PD | `ttnn.MeshProgramDescriptor()`; one `ttnn.ProgramDescriptor(...)` per `ttnn.MeshCoordinateRange(coord_i, coord_i)` — programs are per-device distinct (CT args: `my_chip_id`, send/arrival counts) |
| Dispatch | **exactly one** `ttnn.generic_op([input_tensor, gather_buffer, output_tensor], mesh_pd)` — output preallocated and LAST in `io_tensors` (`generic_op_nanobind.cpp:32-33`); `gather_buffer` fresh per call in `io_tensors` so dispatch resolves/keeps it alive |
| Routes | `ttnn._ttnn.fabric.ccl_dm_route(mesh_device, coord_i, coord_neighbour, topology)` per direction; `assert route.num_hops == 1` (store-and-forward invariant). The route owns the fwd/bwd sign reversal — never hand-derive `is_forward` |
| Fabric conn args | `ttnn._ttnn.fabric.build_ccl_fabric_rt_args(src_fabric_node_id, neighbor_fabric_node_id, 0, program, worker_core, is_forward)` (`ttnn/cpp/ttnn-nanobind/fabric.cpp:277-297`) — emits the `[has_forward][fwd conn][has_backward][bwd conn]` block, placed **FIRST** in each relay writer's rt args so the kernel consumes it at `conn_arg_idx = 0` and resumes op args at the advanced cursor. It MUTATES the program (appends SemaphoreDescriptors), so append via the live `program.kernels[k].runtime_args[x][y]` view after `ProgramDescriptor` construction. Neighbour ids via `mesh_device.get_fabric_node_id(MeshCoordinate(0, j))` |
| Packet framing | 1 tile page = 1 fabric packet (`arm_unicast_write(page_size)`); `ccl_packet_dims` NOT used — same documented rejection as all four adopted collectives; tile pages (2048/4096 B) fit a single packet, hardware-validated at both dtypes |
| Idle direction | empty rt-arg list `[]` + CT `num_sends = 0`; kernel no-ops the egress under `if constexpr` but still runs the T6 wait+reset when `arrivals_dir > 0` |
| Compute config | `ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False, dst_full_sync_en=False)` — fixes `DEST_AUTO_LIMIT = 4` |
| TensorAccessors | `list(ttnn.TensorAccessorArgs(t).get_compile_time_args())` appended LAST after all scalar CT args, for: input + gather (relay reader), gather (relay writer), input + gather (reduce reader), output (reduce writer) |
| Kernel descriptors | 4 relay (fwd reader, fwd writer, bwd reader, bwd writer — CT args differ per direction) + 3 reduce = **7 per program**, five distinct `.cpp` sources (no phase-selector CT superset — the separate-source layout is the cleaner adopted idiom) |

Information each kernel needs (exact CT/RT index layout is the implementer's choice — derive from the CB table and helper signatures):

| Kernel | Core(s) | Needs |
|---|---|---|
| `reduce_scatter_relay_reader.cpp` | (0,0)+(0,1) | CT: `cb_relay_pages`, direction, `my_chip_id`, `N`, `num_sends`, `arrivals_dir`, `P` + input/gather accessor args; RT: input addr, gather addr, page size, own-direction sem addr |
| `reduce_scatter_relay_writer.cpp` | (0,0)+(0,1) | CT: `cb_relay_pages`, direction, `my_chip_id`, `N`, `num_sends`, `P`, L1 alignment + gather accessor args; RT: fabric conn block FIRST, then gather addr, page size, sem addr, NoC xy of the neighbour's relay core AND reduce core |
| `reduce_scatter_reduce_reader.cpp` | (0,2) | CT: `cb_contributions`, `my_chip_id`, `N`, `fwd_arrivals`, `bwd_arrivals`, `S`, `g`, `Wt`, `slice_Wt`, `P`, `dim` + input/gather accessor args; RT: input addr, gather addr, page size, `sem_fwd` addr, `sem_bwd` addr |
| `reduce_scatter_reduce_compute.cpp` | (0,2) | CT: `cb_contributions`, `cb_accumulator`, `cb_output_tiles`, `N`, `S`, `g` |
| `reduce_scatter_reduce_writer.cpp` | (0,2) | CT: `cb_output_tiles`, `S`, `g` + output accessor args; RT: output addr, page size |

## API Mapping

Every mechanism with verified file:line. Type `helper` or `raw_api`.

| Phase | Type | Function | File:Line | Params / Notes | Input CB | Output CB | Requirements |
|-------|------|----------|-----------|----------------|----------|-----------|--------------|
| Relay egress open | helper | `dataflow_kernel_lib::ccl::FabricStreamSender<>` ctor / `open(unicast_route(num_hops))` | `ttnn/cpp/ttnn/kernel_lib/ccl_helpers_dataflow.hpp:481,492,503`; `unicast_route` `:302` | `ConnT = DirectConn`; ctor `(conn_arg_idx, is_forward, alignment)`; the fabric block sits at rt-arg index 0 | — | — | sender declared before (outlives) the stream (`:81-84`); route bound ONCE at open |
| Relay page send | helper | `FabricStream::arm_unicast_write(page_size)` → `channel.write_page(src_l1, page_idx, gather_accessor)` | `:423` (arm), `:327` (write_page) | invariant per-page payload, armed once, issued `num_sends*P` times | `cb_relay_pages` | remote `gather_buffer` | `noc_async_writes_flushed()` between `write_page` and `cb_pop_front` — CB-slot-reuse guard (R7) |
| Arrival signal ×2 | helper | `FabricStream::arm_inc(1)` → `counter.inc(noc_addr)` twice per block | `:435` (arm), `:368` (inc) | one armed channel, two issues per block: neighbour relay-core sem, neighbour reduce-core sem; addrs via `safe_get_noc_addr(x, y, sem_addr, 0)` | — | remote sems | in-order behind pages on the same connection (R8) |
| Egress close | helper | `FabricStream::close()` | `:461` | — | — | — | drains write + atomic barriers; idempotent |
| Arrival wait (relay) | raw_api (op-owned by design) | `noc_semaphore_wait_min(sem_ptr, k)`; re-arm `noc_semaphore_set(sem_ptr, 0)` | helper banner `ccl_helpers_dataflow.hpp:108-121` assigns the WAIT + reset to the op | — | — | — | reset AFTER the final wait, on every role incl. pure receivers (R1) |
| Arrival poll (reduce) | raw_api (op-owned by design) | two-way poll: `invalidate_l1_cache()` + volatile reads of the `sem_fwd`/`sem_bwd` L1 words; consume whichever direction has an unconsumed arrival (`*sem_dir > consumed_dir`) | no helper exists — receive-side sync is explicitly outside `ccl_helpers_dataflow.hpp` (banner `:108-121`); adopted-kernel precedent `reduce_scatter_average_reduce_reader.cpp:151-170` | — | — | — | monotone counters, loop bound `fwd_arrivals + bwd_arrivals`; reset BOTH after (R1) |
| Slice tile walk | helper | `ttnn::ccl::schedule::SliceRowWalker(slice_Wt, Wt)` + `set_base` / `reset_offsets(0,0)` / `next()`; base from `slice_tile_offset(dim, my_chip_id, C, slice_Ht, slice_Wt)`; `static_assert(sched::is_supported_scatter_dim(dim))` | `ccl_helpers_schedule.hpp:491-540` (walker), `:466-478` (offset), `:460-461` (dim gate) | own contribution: base `my_chip_id*slice_Wt` over the input accessor; arrival from `src`: base `src*P + my_chip_id*slice_Wt` over the gather accessor; IDENTICAL walk per contribution ⇒ positional alignment across passes (R11), and the walk order equals the output's row-major tile order ⇒ the dense writer needs no walker | — | — | `next()` returns AND advances — call once per tile |
| DRAM reads/writes | raw_api (no covering helper) | `TensorAccessor` + `noc_async_read`/`noc_async_write` + per-granule barriers | `tech_reports/tensor_accessor/tensor_accessor.md`; CT args from `ttnn.TensorAccessorArgs` | interleaved page addressing | — | `cb_contributions` / output tensor | per-granule (not per-tile) `noc_async_read_barrier` / `noc_async_write_barrier` |
| Compute boot | raw_api (mandated pre-condition) | `binary_op_init_common(cb_contributions, cb_accumulator, cb_output_tiles)` | pre-condition of the accumulate helpers: `accumulate_helpers_compute.hpp:116-117`, `:211`; ownership note `:70-77` (NOT interchangeable with per-op inits, deliberately not folded into `arm()`) | once at kernel start | — | — | before any helper usage |
| Seed copy (contribution 0 = own) | helper | `compute_kernel_lib::sum_blocks(cb_contributions, cb_accumulator, /*num_blocks=*/1, /*block_num_tiles=*/g, /*pop_input=*/true)` × `S/g` | decl `accumulate_helpers_compute.hpp:221-222`; `num_blocks == 1` degenerates to a copy of block 0 (`:217`) | `pop_input=true` is load-bearing (R4) | `cb_contributions` (g) | `cb_accumulator` (g) | leaves `add_tiles_init` in acc_to_dest mode (post `:212-213`) ⇒ `rearm()` before the runs (R3) |
| Incremental accumulate (arrivals 1..N−1) | helper | `compute_kernel_lib::BlockAccumulate::arm(cb_contributions, cb_accumulator, cb_accumulator, g)`; `rearm()`; `run(g)` × `(N-1) * S/g` | arm `accumulate_helpers_compute.hpp:125`, run `:132`, rearm `:175` | **in-place `cb_b == cb_out`**: sound because `run()` pops a and b BEFORE reserving out (`:147-151`, the verified ordering) — with capacity exactly `S`, pop-then-reserve always finds `g` free pages | `cb_contributions` (g) + `cb_accumulator` front (g) | `cb_accumulator` back (g) | `g ≤ DEST_AUTO_LIMIT` asserted at arm (`:122-123`); ONE armed instance only (R10) |
| Final move to writer | helper | `compute_kernel_lib::sum_blocks(cb_accumulator, cb_output_tiles, /*num_blocks=*/1, /*block_num_tiles=*/g, /*pop_input=*/true)` × `S/g` | `accumulate_helpers_compute.hpp:221-222`, degenerate copy `:217` | block-0 copy of the finished sum into the writer-facing CB; kernel ends after this phase, so its acc_to_dest post-condition is moot | `cb_accumulator` (g) | `cb_output_tiles` (g) | same-format CBs ⇒ no reconfig needed |

### Raw-API justifications (helpers considered and rejected)

**Final move (C4) — why not a second armed `BlockAccumulate` targeting `cb_output_tiles` for the LAST pass (which would delete C4 entirely):** `accumulate_helpers_compute.hpp:188-191` — "unpack/math config is SINGULAR hardware state, so **two differently-armed accumulators cannot coexist** — hence tracking the mode here rather than handing out two armed objects." The helper's own contract forbids the two-instance shape; the degenerate-copy `sum_blocks` C4 is the helper-native alternative (and mirrors the adopted sibling's C4 phase position exactly, with copy instead of scale). Cost: S extra tile copies (S ≤ 32 golden) after the last arrival — same tail class the adopted design shipped with.

**Eltwise convenience/chain helpers (`eltwise_convenience.hpp` `copy`, `eltwise_chain.hpp` `CopyTile`+`PackTile`) for C4:** **ABSENT from this clone.** `ttnn/cpp/ttnn/kernel_lib/` contains exactly: `accumulate_helpers_compute`, `ccl_helpers_dataflow`, `dest_helpers`, `dfb_helpers_compute`, `dfb_helpers_dataflow`, `l1_helpers`, `reduce_helpers_common/compute/dataflow`, `tilize_helpers`, `untilize_helpers` (`.hpp`+`.inl` each; verified by `ls`). A kernel `#include` of the eltwise headers cannot compile here.

**`reduce_helpers_compute.hpp` `reduce()` for the sum:** reduces WITHIN a tensor along a dim, collapsing the 32×32 within-tile dims via `reduce_tile` — the wrong shape for `out[i] = Σ_j shard_j[i]`, which must preserve every element. The accumulate header's banner draws exactly this contrast (`accumulate_helpers_compute.hpp:12-20`: "That header reduces WITHIN a tensor along a dimension… This one adds whole tile-blocks TOGETHER across separate CBs"). Consequently `prepare_reduce_scaler` is also N/A — no `reduce_tile` anywhere in this op.

**Position-major `sum_blocks(cb, cb_out, N, …)` for the whole reduction (instead of C1–C3):** `sum_blocks` waits the WHOLE input (`num_blocks * block_num_tiles`) up front (`accumulate_helpers_compute.hpp:199-201`) — it cannot start until all N contributions are present, reintroducing zero overlap inside the single dispatch (R13). Arrival-major C1–C3 is the design point.

**`FabricDuplexSender` for the relay writers:** the duplex form serves ONE core owning BOTH directions; this design gives each direction its own core and single-direction connection (the adopted pattern in all four in-tree collectives). Two `FabricStreamSender<>` instances on two cores are strictly simpler and proven.

**`LineSliceCursor` / `LineChannelWalk` / `SyncCadence` / `line_rs_*` predicates (`ccl_helpers_schedule.hpp:588-607, 627-645, 651-658, 690+`):** these are the step machine of the partial-sum line algorithm (per-step slice cursors, chunks-per-sync cadence, FWD/BWD output-accumulate mode split). This design has no per-step slice sequence (every relay block is a whole shard), and its signal quantum is a whole block (one double-inc per block), so a chunks-per-sync cadence has nothing to pace. Adopting them without the partial-sum algorithm would be dead machinery; with it, a different op (Refinement 4). The schedule helpers this algorithm DOES need — `slice_tile_offset`, `SliceRowWalker`, `is_supported_scatter_dim` — are used.

**Two-way semaphore poll — helper considered:** `noc_semaphore_wait_min` blocks on ONE counter and would serialize the two directions (losing overlap whenever the other direction lands first). Receive-side sync is explicitly scoped OUT of `ccl_helpers_dataflow.hpp` (banner `:108-121`); no two-counter primitive exists.

## Compute Phases (reduce_compute kernel, core (0,2))

| # | Operation | Helper? | Input CB (tiles, state) | Output CB (tiles) | CB State After |
|---|-----------|---------|--------------------------|-------------------|----------------|
| C0 | `binary_op_init_common(cb_contributions, cb_accumulator, cb_output_tiles)`; `acc = BlockAccumulate::arm(cb_contributions, cb_accumulator, cb_accumulator, g)` | pre-condition + helper factory | — | — | hw configured; ONE accumulator armed |
| C1 | Seed: copy contribution 0 (own slice) — `S/g × sum_blocks(cb_contributions, cb_accumulator, 1, g, /*pop_input=*/true)` | helper | `cb_contributions` (g per call, popped) | `cb_accumulator` (g per call) | `cb_accumulator` holds S tiles = own contribution; `cb_contributions` empty |
| C2 | `acc.rearm()` — restore after `sum_blocks`'s acc_to_dest post-condition | helper | — | — | plain add mode + data formats restored |
| C3 | Incremental accumulate: `for k in 1..N-1: for chunk in S/g: acc.run(g)` | helper | `cb_contributions` (g, popped) + `cb_accumulator` front (g, popped) | `cb_accumulator` back (g) | after pass k, `cb_accumulator` holds the S-tile running sum of contributions 0..k; FIFO order = walker order preserved every pass |
| C4 | Final move: `S/g × sum_blocks(cb_accumulator, cb_output_tiles, 1, g, /*pop_input=*/true)` — degenerate block-0 copy of the finished sum | helper | `cb_accumulator` (g, popped) | `cb_output_tiles` (g) | `cb_accumulator` empty; sum streamed to the writer; kernel ends (acc_to_dest post-condition moot) |

Compute is order-agnostic: it counts N contributions of S tiles in g-granules; the READER decides arrival order. No step flags, no schedule agreement beyond the fixed CB protocol. C4 exists because the writer cannot consume `cb_accumulator` directly — during C1–C3 the CB transiently holds partial sums indistinguishable from the final one, and the single-armed-instance rule (R10) forbids retargeting the last `run()` at `cb_output_tiles`.

Broadcast verification: N/A — every binary op in this design is a full-tile × full-tile `add_tiles` (BroadcastDim::None); there is no broadcast operand.

## CB Sync Audit (push count == wait/pop count, per CB, per device)

| CB | Pushed | Waited/Popped | Balanced |
|----|--------|----------------|----------|
| `cb_relay_pages` (per relay core) | reader: `num_sends_dir * P` pages | writer: `num_sends_dir * P` (wait 1 / pop 1) | ✓ (0 = 0 on idle direction) |
| `cb_contributions` | reader: `N * S` tiles in g-granules (own + fwd_arrivals + bwd_arrivals = N contributions) | compute: C1 pops `S` + C3 pops `(N-1)*S` | ✓ |
| `cb_accumulator` | compute only: C1 `S` + C3 `(N-1)*S` = `N*S` | compute: C3 pops `(N-1)*S` + C4 pops `S` = `N*S` | ✓ |
| `cb_output_tiles` | compute: `S` (C4) | writer: `S` (g-granules) | ✓ |

Semaphore audit (per device i): `sem_fwd` inc'd `fwd_arrivals` times at (0,0) and `fwd_arrivals` times at (0,2); (0,0) waits values `1..fwd_arrivals` (one per relayed block, or a single `≥ fwd_arrivals` wait at a line end) then resets; (0,2) consumes `fwd_arrivals` arrivals then resets. Mirror for `sem_bwd`. Waits observed == incs issued before every reset — a reset can never race an in-flight inc.

## Validation & Registry Contract (Phase-0)

| Item | Value |
|------|-------|
| Exports | `SUPPORTED`, `EXCLUSIONS`, `INPUT_TAGGERS`, `reduce_scatter` from both `reduce_scatter.py` and the package `__init__.py` (the golden harness imports from the package) |
| Exceptions | `from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue` inside `try/except ImportError` with local `NotImplementedError`-subclass fallbacks |
| Phase-0 `SUPPORTED` | `{"dtype": [ttnn.bfloat16, ttnn.float32], "layout": [ttnn.TILE_LAYOUT], "topology": [_Topology.Linear], "dim": [3]}` — **`"dim"` MUST be a SUPPORTED key even single-valued** (`feature_spec.py:42-47`: the harness derives xfail marks only from declared axes; an undeclared axis surfaces as a hard failure instead of the expected `UnsupportedAxisValue`) |
| `INPUT_TAGGERS` / `EXCLUSIONS` | `{}` / `[]` (golden INPUTS are all tile-aligned by construction) |
| `validate()` ordering | **universal structural (ValueError)** — MeshDevice, `(1, N)` line with N ≥ 2, rank 4, `H/W % 32`, not sharded, `-4 ≤ dim ≤ 3` → **axis gate** — canonicalize `dim = dim % 4`, then per-axis `UnsupportedAxisValue`, then `ExcludedCell` → **axis-value-DEPENDENT structural (ValueError)** — `shape[dim] % N == 0`, `(shape[dim]/N) % 32 == 0`, `S ≤ 256` L1 budget, `output_tensor` spec match (shape/dtype/layout/`memory_config().buffer_type`). This ordering (not all_reduce's older all-structural-first) is the verifier-blessed one: an out-of-SUPPORTED `dim` yields the typed refusal, never a shape ValueError computed under the wrong axis |
| `validate()` returns | `(num_devices, canonical_dim)`; `validate()` is called first in the public function, before any allocation |
| `output_tensor` path | spec-check in validate; write into it; return the SAME handle (acceptance asserts `buffer_address()` equality) |
| TARGET − SUPPORTED refinement candidates | `topology=Ring`, `dim=2` — to be filed in `op_requirements.md` |

## Refinement candidates (not Phase-0)

| # | Refinement | Sketch |
|---|-----------|--------|
| 1 | `topology=Ring` | block indices are already ring-modular (T3); adopt Ring send/arrival depths (fwd `N/2`, bwd `(N-1)//2`) and `ccl_dm_route`'s short-way selection; reduce reader's per-direction source sequences stay `(i ∓ (1+a)) % N` |
| 2 | `dim=2` | per-(batch,channel) dense runs: `walk_slice_Wt = Wt`, base from `slice_tile_offset(2, …)` (`ccl_helpers_schedule.hpp:466-478`), per-channel `bump_base(slice_Ht*N*Wt)` — the adopted sibling's reduce reader (`reduce_scatter_average_reduce_reader.cpp:77-86,115-145`) is the worked example |
| 3 | Multi-core reduce | split `S` across reduce cores; needs per-core arrival inc fan-out or a local mcast of the arrival signal |
| 4 | Bandwidth: true partial-sum line RS | the `LineSliceCursor`/`LineChannelWalk`/`SyncCadence`/`line_rs_*` machine (silicon-verified in `reduce_scatter_minimal_async`); drops the N× gather traffic; extend the host gtest schedule sweeps (`tests/ttnn/unit_tests/gtests/ccl/test_ccl_helpers_schedule.cpp`) before any schedule variant of it |
| 5 | Large-S support | chunk or spill the accumulator; also a Float32 `cb_accumulator` under bf16 inputs to cut the N−1 per-pass bf16 pack roundings |
| 6 | Packet coalescing | `ccl_packet_dims` multi-page packets + per-chunk incs |

## Key Risks and Gotchas

| # | Risk | Rule |
|---|------|------|
| R1 | **Cache-reuse semaphore re-arm** — first run green, second hangs (`ccl_helpers_dataflow.hpp:116-121`) | every consumer resets its OWN core's counter after its final wait: (0,0)→`sem_fwd`, (0,1)→`sem_bwd`, (0,2)→BOTH. Reset on every role, including line-end pure receivers that relay nothing (T6). The acceptance program-cache test exists to catch exactly this |
| R2 | **`cb_accumulator` single-producer invariant** | ONLY the compute kernel ever reserves/pushes `cb_accumulator`. Each RISC keeps a LOCAL CB write pointer — a reader seeding it directly corrupts the ring. The seed goes through `cb_contributions` + the C1 `sum_blocks` copy |
| R3 | **`sum_blocks` post-condition** — leaves `add_tiles_init` in acc_to_dest mode (`accumulate_helpers_compute.hpp:212-213`) | `acc.rearm()` between C1 and the first C3 `run()` (also restores unpack/pack data formats, `:169-173`) |
| R4 | **`pop_input=true` on both `sum_blocks` phases** | the default `false` deadlocks the upstream producer on `cb_reserve_back` once the input CB fills (C1: reader on `cb_contributions`; C4: compute itself on `cb_accumulator` in a later call) |
| R5 | **Granularity contract** | host guarantees `g ∈ {4,2,1}` divides `S`, so every CB interaction is a full granule and no tail chunk exists; `g ≤ DEST_AUTO_LIMIT = 4` under `fp32_dest_acc_en` (asserted at `arm`, `:122-123`); mirror with kernel `static_assert(S % g == 0)` |
| R6 | **CB wrap contiguity** — multi-page reserve + linear `l1 += page_size` writes | every CB capacity is a multiple of its quantum (CB table); never mix quanta on one CB |
| R7 | **`noc_async_writes_flushed()` before `cb_pop_front`** in the relay writer | the fabric write sources the CB slot; popping first lets the reader overwrite in-flight data |
| R8 | **Inc-after-pages ordering** | both `inc()` issues go on the SAME connection as the block's `write_page`s, after the last page — fabric delivery is in-order per connection, making `sem ≥ k` imply data-complete. Do not move the incs to another stream |
| R9 | **`static_assert(is_supported_scatter_dim(dim))`** in the reduce reader; `dim` reaches the kernel already canonical (host gate) | |
| R10 | **ONE armed `BlockAccumulate` per kernel** — unpack/math config is singular hardware state; two differently-armed instances cannot coexist (`accumulate_helpers_compute.hpp:188-191`) | do NOT "optimize" C4 away by arming a second accumulator with `cb_out = cb_output_tiles`; the final move is the degenerate-copy `sum_blocks` |
| R11 | **Walker discipline** — `SliceRowWalker::next()` returns AND advances | call exactly once per tile; `set_base(...)` + `reset_offsets(0,0)` per contribution; the IDENTICAL walk order for every contribution keeps `add_tiles` positionally aligned across passes, and that order equals the output's row-major tile order (dense writer contract) |
| R12 | **All reduce-core CBs share the input dtype** | this is what lets one boot `binary_op_init_common(cb_contributions, cb_accumulator, cb_output_tiles)` cover C1/C3/C4 with no mid-kernel reconfig; do not give any of them a different format |
| R13 | **Overlap is arrival-major by necessity** | a position-major `sum_blocks(cb, out, N, …)` waits full presence (`:199-201`) and reintroduces zero overlap — the failure the single-dispatch mandate exists to prevent. Keep C1–C3 arrival-major |
| R14 | **`gather_buffer` fresh per call, in `io_tensors`** | dispatch resolves and keeps it alive; uniform mesh address is what makes the neighbour-targeted `write_page` valid — allocate via `ttnn.allocate_tensor_on_device` on the MeshDevice, never per-device |
| R15 | **Mesh/topology contract** | acceptance runs on a `(1, 4)` Blackhole mesh with `FabricConfig.FABRIC_1D` (`bh_quietbox_1x4_hw` in `scripts/multidevice_sim_topologies.yaml:193-205`); any other mesh shape hangs fabric init (`Fabric Router Sync: Timeout`) or fails `system_mesh.cpp: requested_size <= system_size` — a test/topology mismatch, not an op defect. Drive verification via `scripts/run_multidevice_sim_pytest.py --runtime hardware --op reduce_scatter`; `run_safe_pytest.sh` is the wrong runner for CCL ops |
| R16 | **bf16 accumulation rounding** | `cb_accumulator` is bf16 under bf16 inputs → N−1 pack roundings; PCC threshold 0.99 (not 0.995), oracle accumulates in fp32 over the quantized shards then casts. Refinement 5 removes the intermediate roundings |

## Structural impossibilities (pipeline-mode note)

`eval/golden_tests/reduce_scatter/feature_spec.py` already exists and is authoritative (not edited here). `INVALID = []` is correct for its TARGET (TILE-only layout, float dtypes, orthogonal `topology`/`dim` axes — every cell constructible); no additional candidates. The golden `helpers.py` was audited: the oracle is the correct fp32-accumulated SUM-then-slice, the driver call matches this signature, and the mesh fixture honours `CCL_HW_MESH_SHAPE` — no harness defects to file.

## Acceptance criteria mapping

| Requirement | Where satisfied |
|-------------|-----------------|
| Device i output = slice i of the sum, PCC ~0.99 | `test_reduce_scatter` (bf16 0.99 / fp32 0.999, fp32-accumulated oracle over quantized shards) |
| Negative-dim alias | `test_reduce_scatter_negative_dim_alias` (`dim=-1` ≡ `dim=3`) |
| `output_tensor` path returns the supplied handle | `test_reduce_scatter_output_tensor` (`buffer_address()` equality + correctness) |
| Program-cache hit on 2nd call, semaphores survive | `test_reduce_scatter_program_cache` (2-call loop; catches a missing R1 re-arm as a hang) |
| ONE `generic_op` dispatch, compute overlaps arrival | single MeshProgramDescriptor dispatch; T4/T7 overlap contract; R13 |
| Loud shape rejection | `test_reduce_scatter_rejects_non_tile_aligned_slice` (`pytest.raises(ValueError)`) |
| Typed refusal for unsupported axis values | `test_reduce_scatter_rejects_unsupported_dim` (`pytest.raises(NotImplementedError)` — `UnsupportedAxisValue`) |

## Hardware Constraints checklist

- [x] CB sync: push count = wait/pop count for every CB (audit table)
- [x] DEST: `g ≤ 4` under `fp32_dest_acc_en` (SyncHalf) — asserted at `BlockAccumulate::arm`
- [x] Reduce-scaler pool-type API: N/A — no `reduce_tile` and no scaler anywhere in this op
- [x] Page sizes tile-aligned; relay CB pages rounded to L1 alignment
- [x] All `cb_wait_front` calls on a given CB use one page count (`g`; 1 page for relay)
- [x] Helpers not wrapped with extra CB operations (`sum_blocks`/`BlockAccumulate` own their protocols)
- [x] Hardware startup (`binary_op_init_common`) before any compute-helper usage
- [x] Every compute phase uses a helper; every raw-API fallback carries a file:line-cited rejection
