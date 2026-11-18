# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/ccl/all_to_all_dispatch`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ (after `94bf191ce8f`) · Features ✓ · TTNN factory concept ✓ (from code; the readiness sheet was not read) · Offset base pointers ✓ · TensorAccessor 3rd arg ✓ (no sites)

**Recipe docs:** `c41f7f4ef0a 2026-10-29 docs(metal_2.0): DFBAccessor -> DFBBindingToken; drop the dead .id form` *(carry this line into the port report's Provenance section)*

**Audited at:** `94bf191ce8f` (after the `[Prereq]` Device 2.0 migration of both kernels).

**Branch-specific overrides** (this branch is newer than the recipe):

1. GlobalSemaphore is supported. Host: `ProgramSpec::advanced_options.global_semaphore_parameters = {{.unique_id, .memory_type}}`; `KernelSpec::advanced_options.global_semaphore_bindings = {{.global_semaphore_parameter_name, .accessor_name}}` (DM kernels only); `ProgramRunArgs::advanced_options.global_semaphore_args = {{name, std::cref(semaphore)}}`. Kernel: `#include "api/dataflow/global_semaphore.h"`, `experimental::GlobalSemaphore gs(global_sem::<accessor>)` with `wait`, `wait_min`, `set`, `value`, `address`.
2. Op-owned GlobalSemaphores go in `MeshWorkloadArtifacts::global_semaphores` (`ttnn/api/ttnn/metal_v2_artifacts.hpp:45`); the adapter keeps them alive for the cached workload. Precedent: `point_to_point`, `all_to_all_combine`. **This op has two.**
3. Fabric connections replace `append_fabric_connection_rt_args`. Host: `ProgramSpec::advanced_options.fabric_connections = {FabricConnectionSpec{.unique_id, .src_fabric_node_id, .targets}}`, `targets` = `Table<NodeCoord, Group<FabricConnectionTarget{.dst_fabric_node_id, .link_idx}>>` (at most one target per node per spec); `KernelSpec::advanced_options.fabric_connection_bindings = {{.fabric_connection_spec_name, .accessor_name}}`. Kernel: `tt::tt_fabric::has_fabric_connection(fabric::<accessor>)`, `tt::tt_fabric::WorkerToFabricEdmSender::build_from_binding(fabric::<accessor>)`. Use fully qualified `tt::tt_metal::experimental::FabricConnectionSpec` in TTNN code.
4. Sanctioned kernel-side exception, and only this: (a) the raw GlobalSemaphore address RTAs and their raw-pointer uses become `global_sem::` bindings; (b) the fabric-connection RTAs read by `build_from_args` (incl. the per-direction flags) become `fabric::` bindings.
5. Caller-supplied GlobalSemaphores: not applicable (both are op-owned; no legacy override).
6. A binder node with no fabric target: default-construct the sender, build it only under `has_fabric_connection(fabric::x)`, guard `open*()` / `close()` with the same flag.
7. One node that talks to more than one destination: one `FabricConnectionSpec` per destination slot, all bound by the kernel. The validator rejects a `(src, dst, link)` that appears twice in a program (2-device Ring along the axis: East == West) — a config edge case, not a blocker.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`). Carry them forward:

- **Current concept:** `WorkloadDescriptor` — `AllToAllDispatchSparse::create_workload_descriptor` (`device/all_to_all_dispatch_program_factory.cpp:501`).
- **Op-owned tensors:** none. **Op-owned GlobalSemaphores:** two, `init_barrier_semaphore` and `final_barrier_semaphore` (`:514-517`), created on `worker_core_range_set`, followed by one `distributed::Synchronize` per cache miss (`:518`). Keep the creation and the `Synchronize` as they are; park both in `MeshWorkloadArtifacts::global_semaphores` (reserve two slots first, then bind by reference to the vector elements).
- **Target concept:** `MeshWorkloadSpecFactoryConcept`, **no** `override_runtime_arguments` (code-derived). Ported-from entry point: `create_workload_descriptor` (`.cpp:501`), one program per coordinate of `tensor_coords`, `range = MeshCoordinateRange(coord)`. Per coordinate: `src_mesh_id`, `src_chip_id`, the linearized mesh index, the neighbors (→ fabric targets) and the `DIRECTIONS` define.
- **Custom hash:** none.
- **Gate-cleared, confirmed absent** (from code): a `TensorParameter relaxation` that is neither `none` nor an analysis pointer · `get_dynamic_runtime_args`. Also absent, not gating: pybound `create_descriptor`.

## Construct — to do

**Tensor bindings** (all **Case 1**: each base address goes to a `TensorAccessor` today as a `Buffer*` RTA):

- input → reader (`reader_all_to_all_dispatch.cpp:71`, RTA0), indices → reader (`:72`, RTA1), mapping → reader (`:73`, RTA2), metadata → reader (`:74`, RTA4) → `TensorParameter` / `TensorBinding`; kernel uses `TensorAccessor(tensor::<name>)`.
- output → writer (`writer_all_to_all_dispatch.cpp:207`, RTA3), metadata → writer (`:208`, RTA4) → same.
- The other address RTAs (reader RTA3, writer RTA0-2) are read but unused: they drop with no binding.

**TensorParameter relaxation:** `none`.

**TensorAccessor 3rd arg:** none.

**CB endpoints:**
- 1P+1C (reader PRODUCER, writer CONSUMER) on `c_0` input, `c_1` indices, `c_2` mapping.
- **Self-loop** on the writer: `c_3` packet header (no FIFO call at all) and `c_4` send preparation (only toucher).
- **Conditional DFB** `c_5` metadata buffer: declare it and bind it only when `impl == FullPacket`. Census there: two role-free touchers → 1P+1C, writer PRODUCER (it fills the buffer, locally and on the other devices), reader CONSUMER (it copies the buffer to the metadata tensor). Promote the CTA `write_page_by_page` (index 35) and its `if constexpr` blocks in both kernels to an `#ifdef` on a host define emitted with the binding (for example `FULL_PACKET`), per "Conditional / optional resource bindings"; feed the define to **both** kernels.
- No multi-binding, no dead CB. The CB indices (CTAs 0-4 and 34) drop (DFB bindings).

**GlobalSemaphore (override 4a):** two `GlobalSemaphoreParameter`s per program (L1).
- final (cross-device) semaphore: bound by the **reader** and the **writer**. Reader: `wait(...)` / `set(0)` replace `noc_semaphore_wait` / `noc_semaphore_set` (`reader…:123-132`). Writer: `get_noc_addr(<gs>.address())` replaces `get_noc_addr(global_semaphore_address)` (`writer…:320`); the local `noc_semaphore_inc` calls on that NoC address stay (no increment method).
- init semaphore: bound by the writer only. `get_noc_addr(<gs>.address())` (`writer…:223`), `wait(dispatch_devices - 1)` / `set(0)` (`:235-236`).
- The address RTAs (reader RTA5, writer RTA5-6) drop.

**Fabric connections (override 4b, 7):** four `FabricConnectionSpec`s per program — `east`, `west`, `north`, `south` — with `src_fabric_node_id = mesh_device->get_fabric_node_id(coord)`. For each sender core `c` (index `i`) and each direction `d` that has a neighbor: `targets[c] = {FabricConnectionTarget{.dst_fabric_node_id = neighbor_d, .link_idx = i}}`. Map neighbors to directions in `get_neighbors` order (`{E, W, N, S}`). The writer binds all four. Kernel side: replace `DIRECTIONS` with a direction array from `has_fabric_connection` and replace `open_direction_connections_async` (the only helper that reads RTAs) with a loop that builds each sender under `has_fabric_connection` and calls `open_start()`. `open_direction_connections_barrier` and `close_direction_connections` take that runtime array and the senders, read no RTA, and may stay as calls. Keep today's order. The `DIRECTIONS` define drops; `DEST_CHIP_ID`, `DEST_MESH_ID`, `AXIS` stay defines.

**Hardware configs:** both DM configs are custom: reader `RISCV_1` / `NOC_1`, writer `RISCV_0` / `NOC_0`, `DM_DEDICATED_NOC` (`.cpp:403-406`, `:415-418`). Copy them verbatim (`DataMovement1XXConfig`). `opt_level` unset on both (DM → O2).

## Watch for

- **CB endpoints (multi-binding):** none.
- **Cross-op / shared kernels:** none. Both kernels are op-owned → convert in place.
- **RTA varargs:** none. Name every RTA (`token_start_idx`, `token_end_idx`).
- **Shared positional CTA vector.** Both kernels receive the same 37 CTAs and read different subsets. Emit, per kernel, the named CTAs that kernel declares; do not copy the whole vector to both.
- **Kernels are already Device 2.0** after `94bf191ce8f`: the port is a binding-layer change plus overrides 4a/4b, not an idiom rewrite. Transfers stay as they are.
- **Cross-device CB address.** In `FullPacket` the writer computes the NoC destination for the other devices from its own `c_5` address (`writer…:405-406`). Keep the same DFB set on every coordinate so `c_5` lands at the same address everywhere, as in legacy.
- **Templated CTAs.** Many CTAs feed template arguments (`fabric_send_*<…>`, `send_init_semaphore_to_configured_targets<…>`, `is_configured_target<…>`); keep them `constexpr`.
- **Tensor references.** The legacy factory copies the input tensors (`auto input_tensor = tensor_args.input_tensor;`); a `TensorArgument` must reference the io tensor, not a copy — take them by `const&`.
- **2-device Ring along the dispatch axis:** East and West targets would both be `(src, neighbor, i)`; the validator rejects it. Record; do not work around it.
- **Do not read** `ttnn/cpp/ttnn/operations/experimental/quasar/**` for precedent or binding names.
- **Readiness sheet not read:** if River or the sheet owner reports `Is able to port?` = `no` for this op, stop.
