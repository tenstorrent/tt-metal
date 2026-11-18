# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/experimental/ccl/all_to_all_async`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ (from code; the readiness sheet was not read) · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `c41f7f4ef0a 2026-10-29 docs(metal_2.0): DFBAccessor -> DFBBindingToken; drop the dead .id form` *(carry this line into the port report's Provenance section)*

**Audited at:** `c116d1cba2f` (no `[Prereq]` commit was necessary).

**Branch-specific overrides** (this branch is newer than the recipe):

1. GlobalSemaphore is supported. Host: `ProgramSpec::advanced_options.global_semaphore_parameters = {{.unique_id, .memory_type}}`; `KernelSpec::advanced_options.global_semaphore_bindings = {{.global_semaphore_parameter_name, .accessor_name}}` (data-movement kernels only); `ProgramRunArgs::advanced_options.global_semaphore_args = {{name, std::cref(semaphore)}}`. Kernel: `#include "api/dataflow/global_semaphore.h"`, `experimental::GlobalSemaphore gs(global_sem::<accessor>)` with `wait`, `wait_min`, `set`, `value`, `address`.
2. Op-owned GlobalSemaphores go in `MeshWorkloadArtifacts::global_semaphores` (`ttnn/api/ttnn/metal_v2_artifacts.hpp`). **This op has none** — leave the field empty.
3. Fabric connections replace `tt::tt_fabric::append_fabric_connection_rt_args`. Host: `ProgramSpec::advanced_options.fabric_connections = {FabricConnectionSpec{.unique_id, .src_fabric_node_id, .targets}}`, `targets` = `Table<NodeCoord, Group<FabricConnectionTarget{.dst_fabric_node_id, .link_idx}>>` (at most one target per node per spec); `KernelSpec::advanced_options.fabric_connection_bindings = {{.fabric_connection_spec_name, .accessor_name}}` (single-thread DM kernels only). Kernel: `tt::tt_fabric::has_fabric_connection(fabric::<accessor>)`, `tt::tt_fabric::WorkerToFabricEdmSender::build_from_binding(fabric::<accessor>)`. Use fully qualified `tt::tt_metal::experimental::FabricConnectionSpec` in TTNN code.
4. **Sanctioned kernel-side exception** beyond the whitelist, and only this: replace (a) the raw GlobalSemaphore address RTA and its raw-pointer uses with a `global_sem::` binding, and (b) the `FabricConnectionManager::build_from_args` fabric-connection RTAs (incl. the per-direction flags) with `fabric::` bindings. Any other non-whitelisted kernel change follows the recipe.
5. **Caller-supplied GlobalSemaphores:** declare a `GlobalSemaphoreParameter` per semaphore, bind it in each kernel that uses it, and pass it as `global_semaphore_args` in both the cache-miss run args and the M2 `override_runtime_arguments`, built by one helper. The semaphore must span every node of every kernel that binds it.
6. **A binder node with no fabric target:** default-construct the sender, build it only under `has_fabric_connection(fabric::x)`, and guard `open*()` / `close*()` with the same flag.
7. **One node that talks to more than one destination:** one `FabricConnectionSpec` per destination slot, all bound by the kernel, each with a unique accessor name.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`). Carry them forward:

- **Current concept:** `descriptor (per-coord)` — `AllToAllAsyncProgram::create_descriptor(attrs, tensor_args, tensor_return_value, const std::optional<MeshCoordinate>&)` (`device/all_to_all_async_program_factory.cpp:149`).
- **Op-owned tensors:** none. **Op-owned GlobalSemaphores:** none. `semaphore` is caller-supplied in `AllToAllAsyncParams` (`device/all_to_all_async_device_operation_types.hpp:24`). The persistent intermediate and output buffers are caller-supplied io tensors (`AllToAllAsyncInputs`); `tensor_return_value` is the persistent output buffer (`create_output_tensors` returns it).
- **Target concept:** `MeshWorkloadSpecFactoryConcept` with the optional `override_runtime_arguments` (code-derived; the sheet's `Porting Target` was not read). Why: the ported-from factory builds a different program per coordinate (the ring index is CTA 0 of every kernel; RTAs and fabric targets differ), so one spec stamped over the mesh is not possible. Ported-from entry point: `create_descriptor(…, coord)` (`all_to_all_async_program_factory.cpp:149`); one `programs` entry per coordinate of `tensor_coords`, `range = MeshCoordinateRange(coord)`. Per coordinate: ring index (from `get_ring_coordinates()`), forward/backward fabric node ids, `num_targets_forward/backward`, the CTA `my_chip_id`, the sender RTAs (`out_row_start`, `out_col_start`, shard sizes, receiver core coordinates) and the fabric targets.
  - **Translate `override_runtime_arguments`** (`all_to_all_async_program_factory.cpp:607`). It rewrites on every hit, on every core of every program: input address (sender reader RTA0, receiver reader RTA1), intermediate address (sender writer RTA0, receiver reader RTA0), output address (sender writer RTA1, receiver writer RTA0) and semaphore address (sender writer RTA2, receiver reader RTA2). In Metal 2.0: `tensor_args` for `input`, `intermediate` and `output`, and `global_semaphore_args` for `semaphore`. No `kernel_run_args`. Build the miss run args and the override result from one helper (`make_dispatch_run_args`). The legacy override reads no coordinate; the M2 one ignores `range`.
- **Custom hash:** `AllToAllAsyncDeviceOperation::compute_program_hash` (`device/all_to_all_async_device_operation.cpp:246`). Leave it alone. It leaves out the semaphore on purpose.
- **Gate-cleared, confirmed absent** (from code): a `TensorParameter relaxation` that is neither `none` nor an analysis pointer · `get_dynamic_runtime_args`. Also absent, not gating: pybound `create_descriptor`.

## Construct — to do

**Tensor bindings** (all **Case 1**: each base address goes to a `TensorAccessor` today as a `Buffer*` RTA):

- input → sender reader (`interleaved_all_to_all_reader.cpp:242`, RTA0) and receiver reader (`interleaved_all_to_all_receiver_reader.cpp:52`, RTA1).
- intermediate → sender writer (`interleaved_all_to_all_writer.cpp:93`, RTA0) and receiver reader (`interleaved_all_to_all_receiver_reader.cpp:54`, RTA0).
- output → receiver writer (`interleaved_all_to_all_receiver_writer.cpp:177`, RTA0).
- sender writer RTA1 output (`interleaved_all_to_all_writer.cpp:51, 94-95`): the accessor is built and never used. Drop it; the sender writer gets no output binding.

**TensorParameter relaxation:** `none`.

**TensorAccessor 3rd arg:** none.

**CB endpoints:**
- sender data `c_in0` (`all_to_all_async_program_factory.cpp:75-83`, sender core): 1P+1C — sender reader PRODUCER, sender writer CONSUMER.
- packet header `c_in1` (`:86-94`, sender core): **self-loop** — sender writer PRODUCER and CONSUMER.
- receiver data `c_in0` (`:106-114`, receiver cores): 1P+1C — receiver reader PRODUCER, receiver writer CONSUMER.
- The two `c_in0` CBs are different CBs (different sizes, disjoint core sets): two `DataflowBufferSpec`s.
- The CB indices also ride as CTAs (sender reader CTA 2, sender writer CTAs 2 and 4, receiver CTA 7); they become `dfb::` tokens.

**GlobalSemaphore (override 4a, 5):** one `GlobalSemaphoreParameter`, `semaphore`, `BufferType::L1`. Bound by the sender writer and the receiver reader. Run args: `std::cref(args.semaphore)`. Kernel side:
- sender writer: `safe_get_noc_addr(receiver_core_x, receiver_core_y, <gs>.address(), 0)` replaces the raw address (`:102-103`). `receiver_core_x/y` stay named RTAs (they name the remote core).
- receiver reader: `<gs>.wait_min(wait_chunk_id)` replaces `noc_semaphore_wait_min(ptr, …)` (`:108`); `<gs>.set(0)` replaces `noc_semaphore_set(ptr, 0)` (`:130`); the raw pointer local (`:91-92`) drops.
- The address RTAs (sender writer 2, receiver reader 2) drop.

**Fabric connection (override 4b, 6, 7):** two `FabricConnectionSpec`s per program, `forward` and `backward`, `src_fabric_node_id = mesh_device->get_fabric_node_id(coord)`. On each link's sender core: `forward` target `{forward neighbor, link}` when a forward neighbor exists, `backward` target `{backward neighbor, link}` when a backward neighbor exists. The sender writer binds both. Kernel side: two `WorkerToFabricEdmSender`s, each built and `open_start()`ed only under its `has_fabric_connection` flag, in the legacy order (forward, then backward); `open_finish()` per flag where the manager called `open_finish()`; `close_start()` both, then `close_finish()` both, per flag, where the manager called `close()`. The forward/backward pick (`:125-126`) picks between the two senders. The flag RTAs and the connection-arg tail drop.

## Watch for

- **CB endpoints (multi-binding):** none.
- **Cross-op / shared kernels:** none. All four kernels are op-owned and nothing else binds them. Convert them in place.
- **RTA varargs:** none. Name every RTA. Keep the dead RTAs and CTAs (Misc anomalies) as named args: removing them is not port work.
- **Name collision in the receiver kernels:** the CTA and the RTA `num_pages_per_packet` share a name (the RTA local shadows the file-scope CTA). Give them distinct arg names; keep both variables.
- **Retained Device 1.0 read:** the receiver reader `noc_async_read(<accessor>.get_noc_addr(first_id, 0, 0), …)` (`:113-117`) stays as is; only the accessor construction changes.
- **Receiver kernel `CircularBuffer cb(cb_id)`** becomes `DataflowBuffer dfb(dfb::<name>)`.
- **Kernels are already Device 2.0:** the port is a binding-layer change plus overrides 4a/4b, not an idiom rewrite.
- **Do not read** `ttnn/cpp/ttnn/operations/experimental/quasar/**` for precedent or binding names.
- **Readiness sheet not read:** if River or the sheet owner reports `Is able to port?` = `no` for this op, stop.
