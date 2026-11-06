# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/dispatch`

- **`DispatchDeviceOperation`** (`device/dispatch_device_operation.hpp`)
  - `DispatchProgramFactory` (`device/dispatch_program_factory.cpp`): `create_workload_descriptor` builds **one `ProgramDescriptor` per mesh coordinate** (`dispatch_program_factory.cpp:1235-1245`). Row-major and tile layouts share this one factory; tile adds the untilize compute kernel.
- Kernels (all owned by the op, all referenced by the factory):
  - `device/kernels/dataflow/reader_worker_dispatch.cpp`: worker reader (RISCV_1)
  - `device/kernels/dataflow/writer_worker_dispatch.cpp`: worker writer
  - `device/kernels/dataflow/writer_sender_dispatch.cpp`: sender fabric writer
  - `device/kernels/compute/untilize_dispatch.cpp`: tile path only
  - `device/kernels/dataflow/dispatch_plan.hpp`: shared header (also included by `deepseek_prefill/combine/.../writer_combine.cpp`)
- First audit of this op. No earlier `METAL2_PREPORT_AUDIT.md` exists.

**Scope:** TTNN op, Gen1 (WH/BH) target, within scope of `audit/metal2_audit.md`.

**Recipe docs:** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`

**Code audited at:** `b0175cbbfe4` (branch `edwinlee/topk_large_indices_descriptor_port`); op dir identical at `main` `ad3ec49f6e2`. Readiness sheet fetched 2026-11-06.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/dispatch` |
| **Overall** | **RED**: no portable subset |
| **DOps / Factories** | `DispatchDeviceOperation` → `DispatchProgramFactory` |
| *Prereqs*: Device 2.0 (every kernel used) | **No**. RED, **broad Device 1.0** in all three dataflow kernels plus donor `ccl/common/kernels/moe_utils.hpp` (compute kernel is clean) |
| *Prereqs*: Cross-op escapes | skipped (whole-op RED, no portable subset; re-audit on unblock). The donor Device 2.0 blocker is named under Gate detail |
| *Feature Support*: overall | **RED**: `GlobalSemaphore` |
| *Feature Support*: Variadic-CTA | Ok (not a gate; see Appendix A, not listed) |
| *TTNN Readiness*: `Is able to port?` (the gate) | **No**: `Smuggled pointer (raw buffer addr in RTA/CRTA)` == `yes` (`Op Classification` == `Broken Op`); also `Concept` == `WorkloadDescriptor`, `Secretly SPMD Workload?` == `no` (genuine multi-program) |
| *TTNN Readiness*: Concept (current) | `WorkloadDescriptor` (`Execution Model` == `MPMD`) |
| *TTNN Readiness*: Secretly SPMD | **No**: one program per coordinate (`dispatch_program_factory.cpp:1235-1245`); per-coord fabric routing and mesh coordinate are baked into each program |
| *TTNN Readiness*: Custom hash | No (`attribute_values` backdoor only: `dispatch_types.hpp:61`) |
| *TTNN Readiness*: `get_dynamic_runtime_args` | No |
| *TTNN Readiness*: `override_runtime_arguments` | No |
| *TTNN Readiness*: Pybind `create_descriptor` | No |
| *TTNN Readiness*: Op-owned tensors | No (`WorkloadDescriptor::buffers` unused; only `semaphores` is populated) |
| *TTNN Readiness*: Target concept | Not recorded here (informational subject skipped). Recipe note: a per-coordinate Metal 2.0 concept now exists in code; see Recipe notes |
| *Port work*: Offset base pointer | none: GREEN |
| *Port work*: Tensor bindings | skipped (whole-op RED) |
| *TTNN Readiness*: TensorParameter relaxation | `none` (clears) |
| *Port work*: TensorAccessor 3rd arg | GREEN: 3 sites, all Class 2 (redundant → drop) |
| *Port work*: CB endpoints | deferred (re-evaluate after Device 2.0 migration) |

## Result

**RED → blocked on three gates, no portable subset** (one factory; every blocker is unconditional):

1. **Feature: `GlobalSemaphore`** (unsupported in Metal 2.0) → wait for Metal 2.0 semaphore support. The op creates three cross-device `GlobalSemaphore`s on every cache miss and hands their addresses to the sender kernel.
2. **TTNN factory concept** (`Is able to port?` == `no`) → **TTNN** team. The sheet flags a smuggled raw pointer, which the code confirms: the three `GlobalSemaphore` addresses ride runtime args as plain numbers. The op is also a genuine per-device (multi-program) workload.
3. **Device 2.0** → **Device 2.0 team**. All three dataflow kernels (and the borrowed `ccl/common/kernels/moe_utils.hpp` helpers) still broadly use old-style device calls. This is a full migration, not a few one-line fixes.

Gates 1 and 2 share one root cause: the cross-device `GlobalSemaphore`s. Gate 2 cannot clear in a way Metal 2.0 can use until gate 1's feature lands. Gate 3 is independent and can be worked now, in parallel.

This is **not** the "stale #57409 PD-batch sheet row" case. The sheet row matches the code, and the sheet is not the only RED, so no waiver applies.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **RED.** Row `experimental/deepseek_prefill/dispatch` / `DispatchDeviceOperation` / `DispatchProgramFactory`: `Is able to port?` = `no`.
  - **Blocking column:** `Smuggled pointer (raw buffer addr in RTA/CRTA)` = `yes`, with `Op Classification` = `Broken Op`. Route to **TTNN**, who fix the op first. Code evidence: `cross_device_semaphore.address()` and `init_semaphore.address()` go into the sender writer's base RTA block (`dispatch_program_factory.cpp:994-995`), and `exit_semaphore.address()` follows (`:1033`, annotated `// smuggled-rta-ok`). The kernel reads them back as raw L1 addresses (`writer_sender_dispatch.cpp:123-124, 133`). All six **tensor** addresses are already cache-safe: the `uint32_t` slots at `:988-993` are replaced with `Buffer*` entries by `promote_rt_args_with_buffer_bindings` (`:1006-1019`), and the worker kernels receive `Buffer*` directly (`:1142-1145, 1184-1185`, plus the optional `padding_config`/`scales` at `:1165, 1169, 1188`). So the smuggled pointers are the semaphore addresses only.
  - **Also present:** `Concept` = `WorkloadDescriptor`, `Secretly SPMD Workload?` = `no`, `Porting Target` = `(N/A)`. This is a genuine multi-program op, so neither single-program spec concept fits. See Recipe notes on the new per-coordinate concept.
  - **Primary-column cross-check, all clean:** `Concept` (`create_workload_descriptor`, `dispatch_program_factory.hpp:29`) ✓ · `Custom hash` = `no` (no `compute_program_hash`) ✓ · `get_dynamic_runtime_args` = `no` ✓ · `Override runtime args method?` = `no` ✓ · `Pybind descriptor` = `no` (no `create_descriptor` in `dispatch_nanobind.cpp`) ✓ · `Secretly SPMD` = `no` (one `programs` entry per coord) ✓ · `Op-owned tensors?` = `no` ✓ · factory set 1:1 ✓. No invariant broken.
  - **Path forward:** once Metal 2.0 has a cross-device semaphore object (see Feature compatibility), the semaphores move onto a typed binding and the raw-address RTAs go away. Until then the sheet will keep this row at `no`.

- **Device 2.0 (every kernel used):** **RED: broad Device 1.0**, routed to the Device 2.0 team. The kernels declare some Device 2.0 objects (`Noc noc`, `CircularBuffer`, `Semaphore<>`), but most of the work goes through legacy free functions. That makes this a full migration, not isolated holdovers. Rough size (lines with legacy calls): `reader_worker_dispatch.cpp` 47, `writer_sender_dispatch.cpp` 25, `writer_worker_dispatch.cpp` 25, `untilize_dispatch.cpp` 0 (compute kernel, clean). Representative sites:

  | File | Line | Call | Wrapper in scope |
  |---|---|---|---|
  | `kernels/dataflow/reader_worker_dispatch.cpp` | 187-191 | `cb_reserve_back(cb_offsets_id, …)`, `get_write_ptr(cb_offsets_id)`, `noc_async_read_page(i, offsets_addr_gen, …)` | `Noc noc` (`:67`); no `CircularBuffer` object for these CBs |
  | `kernels/dataflow/reader_worker_dispatch.cpp` | 271-293 | `cb_reserve_back`/`get_write_ptr`/`noc_async_read_page`/`noc_async_read_barrier`/`cb_push_back` on `cb_input_id`, `cb_scales_id` | `Noc noc` |
  | `kernels/dataflow/reader_worker_dispatch.cpp` | 209-213, 345 | raw `get_semaphore(turn_semaphore_id)` pointer + `noc_semaphore_set` / `noc_semaphore_wait_min` | `Semaphore<> next_turn_sem` (`:211`) |
  | `kernels/dataflow/writer_worker_dispatch.cpp` | 145, 165 | `reinterpret_cast<…>(get_semaphore(id))` raw pointers | `Semaphore<>` objects at `:139, 158, 163` |
  | `kernels/dataflow/writer_worker_dispatch.cpp` | 153-157 | `get_noc_addr(x, y, l1_addr)` raw remote addresses | `Noc noc` (`:88`) |
  | `kernels/dataflow/writer_worker_dispatch.cpp` | 253-294, 507-552, 601-608 | `noc_async_write_one_packet_with_trid`, `noc_async_write_barrier_with_trid`, `noc_semaphore_inc<true>(noc_addr, 1)`, `noc_async_full_barrier` | `Noc noc` |
  | `kernels/dataflow/writer_worker_dispatch.cpp` | 355-358 | `cb_reserve_back(cb_padding_config_id, 1)`, `get_write_ptr(cb_padding_config_id)` | none for this CB |
  | `kernels/dataflow/writer_sender_dispatch.cpp` | 173-184 | `get_semaphore`, `get_write_ptr(ring_*_cb[s])`, `get_noc_addr`, `noc_inline_dw_write`, `noc_async_write_barrier`, `noc_async_atomic_barrier` | `Noc noc`, `Semaphore<> addr_ready_sem` (`:171-172`) |
  | `kernels/dataflow/writer_sender_dispatch.cpp` | 192 | `get_read_ptr(cb_packet_header_id)` | none |
  | `kernels/dataflow/writer_sender_dispatch.cpp` | 268-284, 516-548 | `get_noc_addr(init/exit_semaphore_address)` from raw RTA, `noc_semaphore_wait`/`noc_semaphore_set` on raw pointers | none (GlobalSemaphore addresses) |
  | donor `ttnn/cpp/ttnn/operations/ccl/common/kernels/moe_utils.hpp` (ccl family) | 239-259 + | `fabric_send_noc_unicast` uses `noc_async_writes_flushed` and raw NoC writes; also called: `fabric_send_chip_sparse_multicast_noc_scatter_write_1d_in_direction`, `open_direction_connections_{async,barrier}` | n/a (donor) |

  `ttnn/cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp` (`ccl_routing_utils::fabric_set_line_unicast_route`) showed no legacy idioms in a grep. `tt_metal/fabric/hw/inc/**` is framework (class 1), so no concern.

- **Feature compatibility:**

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | no `global_circular_buffer` field, type, or remote-CB idiom |
  | CBDescriptor `address_offset` (non-zero) | N/A | no `address_offset` anywhere in the op |
  | GlobalSemaphore | **RED** | see below |

  #### GlobalSemaphore — RED
  - **Signal:** `#include <tt-metalium/global_semaphore.hpp>` and `<ttnn/global_semaphore.hpp>` (`dispatch_program_factory.hpp:13-14`, `dispatch_program_factory.cpp:22`); factory helper takes `const GlobalSemaphore&` ×3 (`dispatch_program_factory.cpp:132-134`); `ttnn::global_semaphore::create_global_semaphore(...)` ×3 (`:1212-1217`), parked in `WorkloadDescriptor::semaphores` (`:1223-1225`) with a cross-device `Synchronize` barrier (`:1220`).
  - **Use:** cross-device init / exit / final barriers between dispatch peers over fabric (`writer_sender_dispatch.cpp:268-284, 516-548`). This is core to the op, not one optional branch, so there is no clean subset.
  - **Expected resolution:** not yet supported in Metal 2.0. The port becomes possible once `GlobalSemaphore` bindings land (listed as "coming soon" in `kernel_spec.hpp`). Corroborated in-tree: `ttnn/api/ttnn/metal_v2_artifacts.hpp:42` says *"Semaphores for cross-device sync belong here once Metal 2.0 has its own semaphore object."*

- **CB endpoints:** deferred (re-evaluate after Device 2.0 migration). Device 2.0 is broadly RED, so the idioms the census relies on will change.

- **Offset base pointers:** **GREEN.** Every tensor address reaching a kernel is a bare base (`Buffer*` entries at `dispatch_program_factory.cpp:1006-1019, 1142-1145, 1165, 1169, 1184-1188`), with no host-side `+ offset`. The op is not in `2026-07-19_offset_base_pointers.md`. The raw L1 addresses the sender writes into a worker's `cross_addr` mailbox (`writer_sender_dispatch.cpp:175-181`) are the sender's own CB write pointers, exchanged on device. They are not host-folded tensor offsets.

- **TensorAccessor 3rd argument:** **GREEN.** Three sites, all **Class 2** (redundant). Each passes exactly the tensor's `buffer()->aligned_page_size()` (`dispatch_program_factory.cpp:32`), so it is inert on any accessor, sharded or interleaved:
  - `reader_worker_dispatch.cpp:173`: `input_addr_gen`, `aligned_input_page_size` = CTA 3 = `get_aligned_page_size(input_tensor)` (`dispatch_program_factory.cpp:808`)
  - `reader_worker_dispatch.cpp:175`: `scales_addr_gen`, `aligned_scales_page_size` = `get_aligned_page_size(scales_tensor)` (`:774, 852`)
  - `reader_worker_dispatch.cpp:177`: `indices_addr_gen`, `aligned_indices_page_size` = CTA 12 = `get_aligned_page_size(indices_tensor)` (`:817`)

  Not in the 2026-07-06 triage table; classified directly.

## Port-work summary  *(mirrors the brief)*

No brief is issued (RED). Known port work so far:

- **TensorAccessor 3rd arg:** drop the redundant page-size arg at `reader_worker_dispatch.cpp:173, 175, 177`.
- **TensorParameter relaxation:** `none`.
- Tensor bindings, CB endpoints, RTA varargs: skipped (whole-op RED, no portable subset; re-audit on unblock).

## Heads-ups  *(mirrors the brief)*

- **Shared header:** `device/kernels/dataflow/dispatch_plan.hpp` is also referenced by `deepseek_prefill/combine` (`combine/device/kernels/dataflow/writer_combine.cpp`, comments at `:17-20`). Keep the two ops in step when either is ported.
- **Sibling op:** `deepseek_prefill/dispatch_fabric2d` exists alongside. It was not audited here.
- Cross-op coupling inventory and RTA varargs: skipped (whole-op RED, no portable subset; re-audit on unblock). Note for the re-audit: `writer_sender_dispatch.cpp:146-148` reads a per-ring group of RTAs in a `num_workers`-bounded loop, and fabric connection args are appended by `append_routing_plane_connection_manager_rt_args` / `append_fabric_connection_rt_args`. Both are likely vararg candidates.

## Team-only

- **Out-of-directory coupling & donor shape:** skipped (whole-op RED, no portable subset; re-audit on unblock). The donor Device 2.0 blocker (`ccl/common/kernels/moe_utils.hpp`) is named under Gate detail.
- **Relaxation candidates:** none noted (no custom hash).
- **TTNN factory analysis:** `Concept` `WorkloadDescriptor`, genuine multi-program (per-coord fabric routing and linearized mesh coordinate in CTAs, `dispatch_program_factory.cpp:1230-1233`). No op-owned tensors. `WorkloadDescriptor::semaphores` carries 3 `GlobalSemaphore`s. No pybind `create_descriptor`, no custom `compute_program_hash`, no `get_dynamic_runtime_args`, no `override_runtime_arguments`.
- **Why the informational subjects were skipped** (Red-outcome scoping rule): of the three blockers, Device 2.0 is cleared by rewriting the op's own kernels, which changes exactly what those subjects describe. So this RED is judged **op-code side → skip**. The GlobalSemaphore gate alone would clear elsewhere (a framework feature), but the Device 2.0 rewrite still makes today's detail stale.

## Misc anomalies  *(team-only, non-gating)*

- `dispatch_program_factory.cpp:1033-1034`: the `smuggled-rta-ok` comment says the exit semaphore is a "persistent GlobalSemaphore (created once in TT_CCL, reused)". In fact all three semaphores are created on every cache miss inside `create_workload_descriptor` (`:1212-1217`) and kept alive by `WorkloadDescriptor::semaphores`. The conclusion (address stable across cache hits) still holds, but the stated reason is out of date.
- `DispatchParams::use_l1_small_for_semaphores` (`dispatch_types.hpp:30`) only affects where the GlobalSemaphores are allocated. It will need a home once semaphores move to a Metal 2.0 object.

## Questions for the user

1. **Smuggled-pointer attribution:** the sheet's `Smuggled pointer` = `yes` matches the code only via the three `GlobalSemaphore` **addresses** (tensor addresses are already `Buffer*`). If the TTNN team treats semaphore addresses kept alive across cache hits as acceptable (as the `smuggled-rta-ok` marker suggests), this conjunct might be cleared on the sheet. That would not change the verdict: the GlobalSemaphore feature gate and Device 2.0 still block.

## Recipe notes

- **TTNN porting shape / `ttnn_factory.md` "Multi-program" is out of date.** That section says *"As of 2026-08-31 there was none"* (no per-coordinate Metal 2.0 concept). The code now has `MeshWorkloadSpecFactoryConcept` (`ttnn/api/ttnn/operation_concepts.hpp:130`, `create_mesh_workload_artifacts` → `MeshWorkloadArtifacts` with per-range `ProgramSpec` + `ProgramRunArgs`), added in `9fb0ed54794 2026-09-04` (#54988). That is one day after this recipe's pinned commit. For this op it is moot (the sheet says `no`), but for any genuine multi-program op that clears, the target would be `MeshWorkloadSpecFactoryConcept`. The port procedure (`port/metal2_port.md`) still covers only the two single-program concepts, so the porter would stop at its coverage boundary.
- **Status-summary template:** there is no row for "target concept" when TTNN porting shape is skipped. I wrote "Not recorded here" with a pointer.
