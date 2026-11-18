# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async` (llama-sharded subset)

> The audit is RED at op level; the subset `LlamaShardedMeshWorkloadFactory` is clear. This brief covers only that
> factory. `DefaultMeshWorkloadFactory` and `AllGatherViaBroadcastFactory` stay on the `WorkloadDescriptor` path. The
> full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared (llama-sharded):** Device 2.0 ✓ (after the `[Prereq]`) · Features ✓ · TTNN factory concept ✓ (from code;
the readiness sheet was not read) · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `c41f7f4ef0a 2026-10-29 docs(metal_2.0): DFBAccessor -> DFBBindingToken; drop the dead .id form` *(carry
this line into the port report's Provenance section)*

**Audited at:** `6420f30222a`. **Prerequisite:** `[Prereq] all_gather_async llama-sharded writer uses the Device 2.0 NoC
API` (the writer no longer calls the shared `write_and_advance_local_read_address_for_fabric_write`).

**Branch-specific overrides** (this branch is newer than the recipe):

1. GlobalSemaphore is supported. Host: `ProgramSpec::advanced_options.global_semaphore_parameters`;
   `KernelSpec::advanced_options.global_semaphore_bindings` (DM kernels only);
   `ProgramRunArgs::advanced_options.global_semaphore_args`. Kernel: `experimental::GlobalSemaphore
   gs(global_sem::<accessor>)` with `wait`, `wait_min`, `set`, `value`, `address`.
2. Op-owned GlobalSemaphores go in `MeshWorkloadArtifacts::global_semaphores`. **This factory has none.**
3. Fabric connections replace `append_fabric_connection_rt_args` (`FabricConnectionSpec`, `fabric_connection_bindings`,
   kernel `has_fabric_connection` / `WorkerToFabricEdmSender::build_from_binding`). Use fully qualified
   `tt::tt_metal::experimental::FabricConnectionSpec`.
4. Sanctioned kernel-side exception: (a) the raw GlobalSemaphore address CRTAs and their raw-pointer uses → `global_sem::`
   bindings; (b) the `FabricConnectionManager::build_from_args` RTAs (incl. the per-direction flags) → `fabric::` bindings.
5. Caller-supplied GlobalSemaphores: one `GlobalSemaphoreParameter` per semaphore, refreshed by one run-args helper on the
   miss and on the hit. The semaphore must span every node of every kernel that binds it.
6. A binder node with no fabric target: build each sender only under `has_fabric_connection`, and guard `open*` /
   `close*` with the same flag.
7. One node that talks to more than one destination: one spec per destination slot (`forward`, `backward`).

## TTNN factory analysis

- **Current concept:** `WorkloadDescriptor` — `LlamaShardedMeshWorkloadFactory::create_workload_descriptor`
  (`device/all_gather_async_llama_sharded_program_factory.cpp:346`), one `programs` entry per coordinate of
  `tensor_coords` (`range = MeshCoordinateRange(coord)`), each from `build_llama_sharded_program_descriptor` (`:31`).
- **Op-owned tensors:** none. **Op-owned GlobalSemaphores:** none. `semaphore` (`semaphore.at(0)` is used) and the
  optional `barrier_semaphore` are caller-supplied in `AllGatherAsyncParams`.
- **Target concept:** `MeshWorkloadSpecFactoryConcept` with the optional `override_runtime_arguments` (code-derived).
  Why: the ring index (CTA of both kernels), the target counts and route CTAs, the per-worker tile ranges and core lists,
  and the fabric targets depend on the coordinate.
  - **Translate `override_runtime_arguments`** (`:361-373`). It writes the writer's semaphore and barrier CRTAs. The
    framework patched the `Buffer*` CRTAs (input, output) itself. In Metal 2.0: `tensor_args` for `input` and `output`,
    `global_semaphore_args` for `semaphore` and (when bound) `barrier_semaphore`. No `kernel_run_args`. One helper
    (`make_dispatch_run_args`) for the miss and the hit.
- **Mixed variant:** the other two alternatives of `program_factory_t` stay `ProgramDescriptorFactoryConcept`; the
  framework dispatches each alternative on its own. Do not touch `select_program_factory` or the variant.
- **Custom hash:** `AllGatherAsyncDeviceOperation::compute_program_hash` (`all_gather_async_device_operation.cpp:243`).
  Leave it alone. It hashes `barrier_semaphore_present` and `using_persistent_buffers`, so the conditional barrier binding
  is stable per cache entry.
- **Gate-cleared, confirmed absent** (from code): `get_dynamic_runtime_args` · a non-`none` relaxation. Pybound
  `create_descriptor`: none.

## Construct — to do

**Tensor bindings** (both **Case 2**; both tensors are L1-sharded by the selection rule):

- `input` → reader (CRTA 0, `llama_shapes_sharded_reader.cpp:37`); base + `shard_tile_id * page_size`, read from listed
  cores (`:60-67`).
- `output` → writer (CRTA 0, `llama_shapes_sharded_writer.cpp:52`); base + offset for the local write and the fabric
  header (`:140-151`).
- Bridge: `LocalTensorAccessor<uint8_t>(tensor::x)` and `local_mem::detail::AddressAccess::address(x.local_mem())`. This
  branch has no `TensorAccessor::get_bank_base_address`.

**TensorParameter relaxation:** none. **TensorAccessor 3rd arg:** none.

**CB endpoints:**
- `c_in0` (`all_gather_async_llama_sharded_program_factory.cpp:123-131`): 1P+1C — reader PRODUCER, writer CONSUMER.
- `c_in1` packet header (`:136-144`): **self-loop** — writer PRODUCER and CONSUMER (DM).
- The CB indices also ride as CTAs (reader CTA 1, writer CTAs 1 and 3); they become `dfb::` tokens.

**GlobalSemaphores (override 4a, 5):** `semaphore` (L1), bound by the writer; `barrier_semaphore` (L1), declared and bound
only when `barrier_semaphore.has_value() && !using_persistent_buffers` (the legacy `use_barrier_sem` CTA, `:177-178`).
The writer's `if (use_barrier_sem)` becomes `#ifdef USE_BARRIER_SEM` (define emitted on the same condition). Kernel side:
`safe_get_noc_addr(x, y, gs.address(), 0)` for the two atomic-inc headers; `gs.wait_min` / `gs.set` for the raw-pointer
waits and resets; the drain-core increment keeps `noc_semaphore_inc(safe_get_noc_addr(x, y, gs.address()), 1)`.

**Fabric (override 4b, 6, 7):** `forward` and `backward` specs, `src_fabric_node_id = get_fabric_node_id(coord)`; worker
`link` gets `{forward neighbor, link}` / `{backward neighbor, link}` when that neighbor exists. Kernel side: two
`WorkerToFabricEdmSender`s, each built and `open_start()`ed under its flag (forward first); `open_finish()` per flag where
the manager called `open_finish()`; `close_start()` both, then `close_finish()` both, per flag, where it called `close()`.

## Watch for

- **RTA varargs:** the core lists in both kernels (`num_cores` x, then `num_cores` y). Name the scalars; the lists are
  varargs. The list length differs per worker, so pad every node's list with zeros to the longest one (the kernels read
  only `2 * num_cores`).
- **Positional route CTAs:** name the twelve route values (`forward_mcast_*`, `backward_mcast_*`) and build the two
  `line_multicast_route_info_t` in the kernel. Do not edit `kernel_common/worker_routing_utils.hpp`.
- **Cross-op / shared kernels:** none. Both kernels are op-owned; convert them in place.
- **Unused args:** keep `my_chip_id` (both kernels) as a named CTA.
- **2-device ring:** forward and backward neighbors are the same device; the validator rejects the repeated
  `(src, dst, link)`. Config edge case, not handled.
- **Do not read** `ttnn/cpp/ttnn/operations/experimental/quasar/**`.
- **Readiness sheet not read:** if River or the sheet owner reports `Is able to port?` = `no`, stop.
