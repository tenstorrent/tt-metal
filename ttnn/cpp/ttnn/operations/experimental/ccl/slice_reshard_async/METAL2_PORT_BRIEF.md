# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/experimental/ccl/slice_reshard_async`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ (from code; the readiness sheet was not read) · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `c41f7f4ef0a 2026-10-29 docs(metal_2.0): DFBAccessor -> DFBBindingToken; drop the dead .id form` *(carry this line into the port report's Provenance section)*

**Audited at:** `c210a7093f6` (no `[Prereq]` commit was necessary).

**Branch-specific overrides** (this branch is newer than the recipe):

1. GlobalSemaphore is supported. Host: `ProgramSpec::advanced_options.global_semaphore_parameters = {{.unique_id, .memory_type}}`; `KernelSpec::advanced_options.global_semaphore_bindings = {{.global_semaphore_parameter_name, .accessor_name}}` (data-movement kernels only); `ProgramRunArgs::advanced_options.global_semaphore_args = {{name, std::cref(semaphore)}}`. Kernel: `#include "api/dataflow/global_semaphore.h"`, `experimental::GlobalSemaphore gs(global_sem::<accessor>)` with `wait`, `wait_min`, `set`, `value`, `address`. See `tt_metal/api/tt-metalium/experimental/metal2_host_api/advanced_options.hpp` and `global_semaphore_parameter.hpp`.
2. Op-owned GlobalSemaphores go in `MeshWorkloadArtifacts::global_semaphores` (`ttnn/api/ttnn/metal_v2_artifacts.hpp:45`); the adapter keeps them alive for the cached workload. **This op has none** — leave the field empty.
3. Fabric connections replace `tt::tt_fabric::append_fabric_connection_rt_args`. Host: `ProgramSpec::advanced_options.fabric_connections = {FabricConnectionSpec{.unique_id, .src_fabric_node_id, .targets}}`, `targets` = `Table<NodeCoord, Group<FabricConnectionTarget{.dst_fabric_node_id, .link_idx}>>` (at most one target per node); `KernelSpec::advanced_options.fabric_connection_bindings = {{.fabric_connection_spec_name, .accessor_name}}` (single-thread DM kernels only). Kernel: `tt::tt_fabric::has_fabric_connection(fabric::<accessor>)`, `tt::tt_fabric::WorkerToFabricEdmSender::build_from_binding(fabric::<accessor>)` (`tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp`). See `experimental/metal2_host_api/fabric_connection_spec.hpp`.
4. **Sanctioned kernel-side exception** beyond the whitelist, and only this: replace (a) the raw GlobalSemaphore address RTA and its raw-pointer uses with a `global_sem::` binding, and (b) the `FabricConnectionManager::build_from_args` fabric-connection RTAs (incl. the forward/backward flags) with a `fabric::` binding. Any other non-whitelisted kernel change follows the recipe.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`). Carry them forward:

- **Current concept:** `WorkloadDescriptor` — `SliceReshardAsyncProgramFactory::create_workload_descriptor` (`device/slice_reshard_async_program_factory.cpp:286`).
- **Op-owned tensors:** none. **Op-owned GlobalSemaphores:** none. Both semaphores (`final_semaphore`, `barrier_semaphore`) are caller-supplied in `SliceReshardAsyncParams` (`device/slice_reshard_async_device_operation_types.hpp:29-30`).
- **Target concept:** `MeshWorkloadSpecFactoryConcept` with the optional `override_runtime_arguments` (code-derived; the sheet's `Porting Target` was not read). Ported-from entry point: `create_workload_descriptor` (`slice_reshard_async_program_factory.cpp:286`), one program per coordinate of `tensor_coords`, `range = MeshCoordinateRange(coord)`. Per coordinate: ring index along `cluster_axis`, forward/backward neighbors, the outer-dim counts (RTAs), the `is_first_chip` / `is_last_chip` CTAs, and the fabric targets.
  - **Translate `override_runtime_arguments`** (`slice_reshard_async_program_factory.cpp:302`). It re-applies the final semaphore (reader RTA 9, writer RTA 14) and the barrier semaphore (writer RTA 18) on every hit. In Metal 2.0 these are `global_semaphore_args` for both semaphore parameters. The source is a `WorkloadDescriptor`, so the override also returns a `TensorArgument` for every io-tensor `TensorParameter` (input and output). No `kernel_run_args`.
- **Custom hash:** `SliceReshardAsyncDeviceOperation::compute_program_hash` (`device/slice_reshard_async_device_operation.cpp:50`). Leave it alone. It leaves out the semaphores on purpose.
- **Gate-cleared, confirmed absent** (from code): a `TensorParameter relaxation` that is neither `none` nor an analysis pointer · `get_dynamic_runtime_args`. Also absent, not gating: pybound `create_descriptor`.

## Construct — to do

**Tensor bindings** (both **Case 1**: each base address goes to a `TensorAccessor` today as a `Buffer*` RTA):

- input → reader (`minimal_default_reader.cpp:56`, RTA0) → `TensorParameter` / `TensorBinding`; kernel uses `TensorAccessor(tensor::<name>)`.
- output (`tensor_return_value`) → writer (`minimal_default_writer.cpp:66`, RTA1) → same.
- writer RTA0 `input_tensor_address` (`minimal_default_writer.cpp:40`) is never used. Drop it; the writer gets no input binding.

**TensorParameter relaxation:** `none`.

**TensorAccessor 3rd arg:** drop the page-size argument at `minimal_default_writer.cpp:66` (Class 2; no `dynamic_tensor_shape`). `stick_size` stays as a **named RTA**, because it is still the transfer size (lines 130, 134, 183-188, 207). The comment above the accessor (lines 64-65) explains the dropped argument; adjust it, do not leave it stale.

**CB endpoints:** 1P+1C (legal) on the one CB, `c_in0` (`slice_reshard_async_program_factory.cpp:122`): reader PRODUCER, writer CONSUMER, on every worker core. No self-loop, no multi-binding, no dead CB. The CB index also rides as reader/writer CTA 2 and as the unused named CTA `cb_forward_id`; both drop.

**GlobalSemaphore (override 4a):** two `GlobalSemaphoreParameter`s per program, `final_semaphore` and `barrier_semaphore`, both `BufferType::L1`. The reader binds the final semaphore; the writer binds both. Run args: `std::cref(args.final_semaphore)` / `std::cref(args.barrier_semaphore)`. Kernel side:
- reader: `out_ready_sem.wait_min(1)` replaces `noc_semaphore_wait_min(ptr, 1)` (`:109`), `out_ready_sem.set(0)` replaces `noc_semaphore_set(ptr, 0)` (`:112`).
- writer: same for the barrier semaphore (`:109`, `:111`); `safe_get_noc_addr(x, y, <gs>.address(), 0)` replaces the raw address in the two atomic-inc headers (`:85`, `:156`). The `*_noc0_x` / `*_noc0_y` RTAs stay as named RTAs (they name the remote core).
- The address RTAs (reader 9, writer 14 and 18) drop.

**Fabric connection (override 4b):** one `FabricConnectionSpec` per program. `src_fabric_node_id = mesh_device->get_fabric_node_id(coord)`. Targets: the direction-1 core of each link → `{.dst_fabric_node_id = forward neighbor, .link_idx = link}` when a forward neighbor exists; the direction-0 core → `{backward neighbor, link}` when a backward neighbor exists. Every writer `KernelSpec` binds the spec. Kernel side: a writer has at most one connection, so `has_fabric_connection(fabric::<name>)` replaces both `has_forward_connection()` and `has_backward_connection()`, and the sender from `build_from_binding` replaces both `get_forward_connection()` and `get_backward_connection()`. Build the sender only when the node has a connection (precondition of `build_from_binding`). Keep today's order: `open()` at `:76`, `close()` at `:217`, each guarded by the connection flag (the manager guarded them the same way). The flag RTAs and the connection-arg tail drop.

## Watch for

- **CB endpoints (multi-binding):** none.
- **Cross-op / shared kernels:** none. Both kernels are op-owned and nothing else binds them. Convert them in place.
- **RTA varargs:** none. Name every RTA.
- **Per-core multiplicity:** the legacy factory pushes one reader and one writer `KernelDescriptor` per worker core (`slice_reshard_async_program_factory.cpp:137-280`), each with per-direction CTAs (`is_first_chip` / `is_last_chip` swap with `direction`). Keep one `KernelSpec` per legacy descriptor and one work unit per core.
- **Reader `stick_size` is a file-scope `const`** initialized from a CTA and used as a template argument (`minimal_default_reader.cpp:26,85`). Keep it `const`, not `constexpr`.
- **`zeroPad` constructs its own `CircularBuffer` from a CB id** (`minimal_default_reader.cpp:29-35`). Rule 1/2 apply: the id becomes the `dfb::` token and the object a `DataflowBuffer`.
- **Kernels are already Device 2.0:** the port is a binding-layer change plus overrides 4a/4b, not an idiom rewrite.
- **Kernel on a core with no CB:** in the edge case `num_sticks_per_outer_dim < num_links`, legacy creates kernels on cores outside `worker_core_ranges`, which have no CB. In Metal 2.0 the DFB is placed on every node of a binding kernel. Record it; do not change the core set.
- **Do not read** `ttnn/cpp/ttnn/operations/experimental/quasar/**` for precedent or binding names.
- **Readiness sheet not read:** if River or the sheet owner reports `Is able to port?` = `no` for this op, stop.
