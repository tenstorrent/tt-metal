# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/point_to_point`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ (from code; the readiness sheet was not read) · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `c41f7f4ef0a 2026-10-29 docs(metal_2.0): DFBAccessor -> DFBBindingToken; drop the dead .id form` *(carry this line into the port report's Provenance section)*

**Audited at:** `131ecf145ee` (after the three `[Prereq]` commits `2e5f720ff76`, `a2c4bc6610c`, `131ecf145ee`).

**Branch-specific overrides** (this branch is newer than the recipe):

1. GlobalSemaphore is supported. Host: `ProgramSpec::advanced_options.global_semaphore_parameters = {{.unique_id, .memory_type}}`; `KernelSpec::advanced_options.global_semaphore_bindings = {{.global_semaphore_parameter_name, .accessor_name}}` (data-movement kernels only); `ProgramRunArgs::advanced_options.global_semaphore_args = {{name, std::cref(semaphore)}}`. Kernel: `#include "api/dataflow/global_semaphore.h"`, `experimental::GlobalSemaphore gs(global_sem::<accessor>)` with `wait`, `wait_min`, `set`, `value`, `address`. See `tt_metal/api/tt-metalium/experimental/metal2_host_api/advanced_options.hpp` and `global_semaphore_parameter.hpp`.
2. Op-owned GlobalSemaphores go in `MeshWorkloadArtifacts::global_semaphores` (`ttnn/api/ttnn/metal_v2_artifacts.hpp:45`); the adapter keeps them alive for the cached workload.
3. Fabric connections replace `tt::tt_fabric::append_fabric_connection_rt_args`. Host: `ProgramSpec::advanced_options.fabric_connections = {FabricConnectionSpec{.unique_id, .src_fabric_node_id, .targets}}`, `targets` = `Table<NodeCoord, Group<FabricConnectionTarget{.dst_fabric_node_id, .link_idx}>>` (at most one target per node); `KernelSpec::advanced_options.fabric_connection_bindings = {{.fabric_connection_spec_name, .accessor_name}}` (single-thread DM kernels only). Kernel: `tt::tt_fabric::has_fabric_connection(fabric::<accessor>)`, `tt::tt_fabric::WorkerToFabricEdmSender::build_from_binding(fabric::<accessor>)` (`tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp`). See `experimental/metal2_host_api/fabric_connection_spec.hpp`.
4. **Sanctioned kernel-side exception** beyond the whitelist, and only this: replace (a) the raw GlobalSemaphore address RTA and its raw-pointer uses with a `global_sem::` binding, and (b) the `FabricConnectionManager::build_from_args` fabric-connection RTAs (incl. the forward/backward flags) with a `fabric::` binding. Any other non-whitelisted kernel change follows the recipe.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`). Carry them forward:

- **Current concept:** `WorkloadDescriptor` — `PointToPointOp::SendReceive::create_workload_descriptor` (`device/host/point_to_point_device_op.cpp:199`).
- **Op-owned tensors:** none. **Op-owned GlobalSemaphore:** one, created on each cache miss (`point_to_point_device_op.cpp:232`), then `distributed::Synchronize` (`:234`), kept in `WorkloadDescriptor::semaphores` (`:247`) → move it to `MeshWorkloadArtifacts::global_semaphores` (override 2). The local-copy path creates none.
- **Target concept:** `MeshWorkloadSpecFactoryConcept` (code-derived; the sheet's `Porting Target` was not read). Ported-from entry point: `SendReceive::create_workload_descriptor` (`point_to_point_device_op.cpp:199`). Per coordinate:
  - `send_coord != receive_coord`: a send program on `send_coord` (from `send_program_factory`) and a receive program on `receive_coord` (from `receive_program_factory`); no program on other coordinates.
  - `send_coord == receive_coord`: one local-copy program on that coordinate (from `local_copy_program_factory`); no GlobalSemaphore, no `Synchronize`, no fabric.
  - No `override_runtime_arguments` to translate.
- **Gate-cleared, confirmed absent** (from code): a `TensorParameter relaxation` that is neither `none` nor an analysis pointer · `get_dynamic_runtime_args`. Also absent, not gating: custom hash, `override_runtime_arguments`, pybound `create_descriptor`.

## Construct — to do

**Tensor bindings** (all **Case 1**: each base address goes to a `TensorAccessor` today as a `Buffer*` RTA):

- send program: input tensor → reader (`reader_unary_interleaved_start_id_gen.cpp:26`, RTA0); intermediate tensor (`tensor_return_value[0]`) → `writer_send.cpp:59` (RTA0) → express as `TensorParameter` / `TensorBinding`; kernel uses `TensorAccessor(tensor::name)`.
- receive program: intermediate tensor → `reader_receive.cpp:69` (RTA3); final output (`tensor_return_value[1]`) → writer (`writer_unary_interleaved_start_id_gen.cpp:63`, RTA0) → same.
- local-copy program: input tensor → reader; final output → writer → same.

**TensorParameter relaxation:** `none`.

**TensorAccessor 3rd arg:** drop the page-size argument at all four sites (Class 2; no `dynamic_tensor_shape`):
- `reader_unary_interleaved_start_id_gen.cpp:26`, `writer_unary_interleaved_start_id_gen.cpp:63` — `page_bytes` stays as a **named RTA**, because it is still the transfer size (lines 35, 71). Only its use as the accessor's 3rd argument drops.
- `writer_send.cpp:59`, `reader_receive.cpp:69` — the packet size stays as a named RTA too (it is the payload and read size, lines 87–88, 95–96, 99–100).

**CB endpoints:**
- 1P+1C (legal; producer and consumer are fixed by the FIFO ops): send `c_0` (reader → `writer_send`), receive `c_2` (`reader_receive` → writer), local copy `c_0` (reader → writer).
- **self-loop** (one toucher; bind the kernel as PRODUCER and CONSUMER): send `c_1` packet-header CB and `c_2` packet CB (`writer_send`); receive `c_0` packet-header CB and `c_1` packet CB (`reader_receive`).
- No multi-binding, no dead CB.

**GlobalSemaphore (override 4a):** one `GlobalSemaphoreParameter` per fabric program (send and receive), bound by the fabric kernel (`writer_send`, `reader_receive`), with the op-owned semaphore in the run args. Kernel side: `wait(1)` replaces `noc_semaphore_wait(ptr, 1)`, `set(0)` replaces `noc_semaphore_set(ptr, 0)`, and `get_noc_addr(gs.address())` replaces `get_noc_addr(sem_addr)` in the fabric atomic-inc header (`writer_send.cpp:71–75,116`; `reader_receive.cpp:52,74–75,109`). The semaphore address RTA (index 8 / index 7) drops.

**Fabric connection (override 4b):** one `FabricConnectionSpec` per fabric program. `src_fabric_node_id` = `mesh_device->get_fabric_node_id(<this coord>)`; one target on the single worker node: `{.dst_fabric_node_id = next_fabric_id, .link_idx = 0}` from `detail::fabric_1d_routing` (`point_to_point_device_op.cpp:67`; called at `send_program_factory.cpp:103`, `receive_program_factory.cpp:90`). Kernel side: `WorkerToFabricEdmSender::build_from_binding(fabric::<name>)` replaces `FabricConnectionManager::build_from_args<BUILD_AND_OPEN_CONNECTION_START_ONLY>` plus the direction pick through `common.hpp`. Keep today's order with the sender's own methods: `open_start()` where the manager was built, `open_finish()` where it was called, `close()` at the end (`edm_fabric_worker_adapters.hpp:468,537,646`). The forward/backward flag RTAs and the connection-arg tail drop.

## Watch for

- **CB endpoints (multi-binding):** none.
- **Cross-op / shared kernels:** all four kernels are op-owned and no other op factory binds them, but `tests/ttnn/unit_tests/operations/debug/test_generic_op.py:287,295,393,401` (`test_point_to_point`) drives all four through the V1 `ProgramDescriptor` path with positional RTAs. No `_metal2` fork exists yet — **this port creates `_metal2` forks of all four** beside the originals (rung 2) and leaves the originals unchanged apart from the pointer comment. Name bindings for the kernel's role, not for this op's locals. Sunset list (**not authorization to convert the kernels in place**): `test_generic_op.py::test_point_to_point`.
- **`device/kernels/common.hpp`** is also included by four `ccl/reduce_to_root` kernels. The forks no longer need it; do not edit or delete it.
- **RTA varargs:** none. The only variable-length tail (fabric-connection args, starting at index 9 and overlapping the direction flag) is replaced by the `fabric::` binding. Name every other RTA.
- **Intermediate tensor on the send program:** `writer_send` addresses the intermediate tensor on the **receiving** device with the base address of the send device's copy. This is correct because a mesh tensor has the same address on every device. Bind it as an ordinary `TensorParameter` on the send program.
- **Quasar debt:** the four self-loop CBs are DM self-loops, which the spec validator rejects on Gen2. Legal on WH/BH; record them in the port report as Quasar-uplift work.
- **`extract_tensor_buffers_t` specialization** (`device/host/point_to_point_device_op.hpp:149–156`) exists for the V1 `ProgramDescriptor` cache-hit patching. Check whether the Metal 2.0 adapter still uses it before you keep or remove it.
- **Kernels are already Device 2.0** (`2e5f720ff76`): the port is a binding-layer change (`CircularBuffer` → `DataflowBuffer`, tokens, named args) plus overrides 4a/4b, not an idiom rewrite. `CircularBuffer` objects are already the NoC source or destination in `noc.async_read/async_write`; a `DataflowBuffer` takes their place directly.
- **Do not read** `ttnn/cpp/ttnn/operations/experimental/quasar/**` for precedent or binding names.
- **Readiness sheet not read:** if River or the sheet owner reports `Is able to port?` = `no` for this op, stop.
