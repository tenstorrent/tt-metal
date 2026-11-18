# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/ccl/all_to_all_combine`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ (after `66b645994d5`) · Features ✓ · TTNN factory concept ✓ (from code; the readiness sheet was not read) · Offset base pointers ✓ · TensorAccessor 3rd arg ✓ (no sites)

**Recipe docs:** `c41f7f4ef0a 2026-10-29 docs(metal_2.0): DFBAccessor -> DFBBindingToken; drop the dead .id form` *(carry this line into the port report's Provenance section)*

**Audited at:** `66b645994d5` (after the `[Prereq]` Device 2.0 migration of the writer).

**Branch-specific overrides** (this branch is newer than the recipe):

1. GlobalSemaphore is supported. Host: `ProgramSpec::advanced_options.global_semaphore_parameters = {{.unique_id, .memory_type}}`; `KernelSpec::advanced_options.global_semaphore_bindings = {{.global_semaphore_parameter_name, .accessor_name}}` (DM kernels only); `ProgramRunArgs::advanced_options.global_semaphore_args = {{name, std::cref(semaphore)}}`. Kernel: `#include "api/dataflow/global_semaphore.h"`, `experimental::GlobalSemaphore gs(global_sem::<accessor>)` with `wait`, `wait_min`, `set`, `value`, `address`.
2. Op-owned GlobalSemaphores go in `MeshWorkloadArtifacts::global_semaphores` (`ttnn/api/ttnn/metal_v2_artifacts.hpp:45`); the adapter keeps them alive for the cached workload. Precedent: `point_to_point`. **This op has two.**
3. Fabric connections replace `append_fabric_connection_rt_args`. Host: `ProgramSpec::advanced_options.fabric_connections = {FabricConnectionSpec{.unique_id, .src_fabric_node_id, .targets}}`, `targets` = `Table<NodeCoord, Group<FabricConnectionTarget{.dst_fabric_node_id, .link_idx}>>` (at most one target per node per spec); `KernelSpec::advanced_options.fabric_connection_bindings = {{.fabric_connection_spec_name, .accessor_name}}`. Kernel: `tt::tt_fabric::has_fabric_connection(fabric::<accessor>)`, `tt::tt_fabric::WorkerToFabricEdmSender::build_from_binding(fabric::<accessor>)`. Use fully qualified `tt::tt_metal::experimental::FabricConnectionSpec` in TTNN code.
4. Sanctioned kernel-side exception, and only this: (a) the raw GlobalSemaphore address RTAs and their raw-pointer uses become `global_sem::` bindings; (b) the fabric-connection RTAs read by `build_from_args` (incl. the per-direction flags) become `fabric::` bindings.
6. A binder node with no fabric target: default-construct the sender, build it only under `has_fabric_connection(fabric::x)`, guard `open*()` / `close()` with the same flag.
7. One node that talks to more than one destination: one `FabricConnectionSpec` per destination slot, all bound by the kernel. The validator rejects a `(src, dst, link)` that appears twice in a program (2-device Ring along the axis: East == West) — a config edge case, not a blocker.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`). Carry them forward:

- **Current concept:** `WorkloadDescriptor` — `AllToAllCombineFromSparse::create_workload_descriptor` (`device/all_to_all_combine_program_factory.cpp:334`).
- **Op-owned tensors:** none. **Op-owned GlobalSemaphores:** two, `init_barrier_semaphore` and `final_barrier_semaphore` (`:347-350`), created on `worker_core_range_set`, followed by one `distributed::Synchronize` per cache miss (`:351`). Keep the creation and the `Synchronize` as they are; park both in `MeshWorkloadArtifacts::global_semaphores` (reserve two slots first, then bind by reference to the vector elements).
- **Target concept:** `MeshWorkloadSpecFactoryConcept`, **no** `override_runtime_arguments` (code-derived). Ported-from entry point: `create_workload_descriptor` (`.cpp:334`), one program per coordinate of `tensor_coords`, `range = MeshCoordinateRange(coord)`. Per coordinate: `src_chip_id`, the linearized mesh index, the neighbors (→ fabric targets) and the `DIRECTIONS` define.
- **Custom hash:** none.
- **Gate-cleared, confirmed absent** (from code): a `TensorParameter relaxation` that is neither `none` nor an analysis pointer · `get_dynamic_runtime_args`. Also absent, not gating: pybound `create_descriptor`.

## Construct — to do

**Tensor bindings** (all **Case 1**: each base address goes to a `TensorAccessor` today as a `Buffer*` RTA):

- mapping → reader (`reader_all_to_all_combine.cpp:82`, RTA0), metadata → reader (`:81`, RTA1), input → reader (`:83`, RTA2) → `TensorParameter` / `TensorBinding`; kernel uses `TensorAccessor(tensor::<name>)`.
- output (`tensor_return_value`) → writer (`writer_all_to_all_combine.cpp:168`, RTA0) → same.

**TensorParameter relaxation:** `none`.

**TensorAccessor 3rd arg:** none.

**CB endpoints:**
- 1P+1C (reader PRODUCER, writer CONSUMER) on `c_0` data, `c_2` local experts, `c_3` metadata.
- **Self-loop** `c_1` mapping on the reader (only toucher) and `c_4` packet header on the writer (only toucher; reserved and pushed, never popped).
- No multi-binding, no dead CB. The CB indices in the reader CTAs 0-3 and writer CTAs 0-3 drop (DFB bindings).

**GlobalSemaphore (override 4a):** two `GlobalSemaphoreParameter`s per program (L1), both bound by the writer only. Run args: the two vector elements of `global_semaphores`. Kernel side:
- init: `get_noc_addr(<gs>.address())` replaces `get_noc_addr(init_semaphore_addr)` (`:187`); `wait(replicate_group_devices - 1)` / `set(0)` replace `noc_semaphore_wait` / `noc_semaphore_set` (`:202-203`).
- final: `get_noc_addr(<gs>.address())` replaces `get_noc_addr(global_semaphore_addr)` (`:288`); `wait(expected_credits)` / `set(0)` replace `:315-316`. The local self-increment `noc_semaphore_inc(global_noc_semaphore_addr, 1)` (`:86`) already takes the NoC address built from the binding; it stays (the kernel `GlobalSemaphore` has no increment method).
- The address RTAs (writer RTA 1, 2) drop.

**Fabric connections (override 4b, 7):** four `FabricConnectionSpec`s per program — `east`, `west`, `north`, `south` — with `src_fabric_node_id = mesh_device->get_fabric_node_id(coord)`. For each sender core `c` (index `i`) and each direction `d` that has a neighbor: `targets[c] = {FabricConnectionTarget{.dst_fabric_node_id = neighbor_d, .link_idx = i}}`. The writer binds all four (`accessor_name` = the direction name). Map neighbors to directions in `get_neighbors` order (it pushes neighbors in `{E, W, N, S}` order and sets the matching flag). Kernel side: replace `DIRECTIONS` and `open_direction_connections_async` / `_barrier` / `close_direction_connections` (moe_utils helpers that call `build_from_args`; shared header, do not edit) with code in the writer that, for each of the four bindings in `{E, W, N, S}` order, builds the sender under `has_fabric_connection` and calls `open_start()`, later `open_finish()`, then `close()`, each under the same flag. Keep today's order (`open_start` before the packet-header setup, `open_finish` before the init send, `close` after the completion credits). The rest of the moe_utils senders take the `std::array<…, 4>` unchanged. The `DIRECTIONS` define drops (derivable from `has_fabric_connection`); `DEST_CHIP_ID`, `DEST_MESH_ID`, `REPLICATE_GROUP_AXIS` stay defines.

**Hardware configs:** both DM configs are custom: reader `RISCV_1` / `NOC_1`, writer `RISCV_0` / `NOC_0`, `DM_DEDICATED_NOC` (`.cpp:254-257`, `:266-269`). Copy them verbatim (`DataMovement1XXConfig`), not the reader/writer helpers. `opt_level` unset on both (DM → O2).

## Watch for

- **CB endpoints (multi-binding):** none.
- **Cross-op / shared kernels:** `reader_all_to_all_combine.cpp` is lent to `data_movement/moe_expert_token_remap` (`moe_expert_token_remap_program_factory.cpp:176`). No fork exists → create `device/kernels/dataflow/reader_all_to_all_combine_metal2.cpp` beside it (rung 2), add the pointer comment to the original, name its bindings for the kernel (`mapping`, `local_experts`, `metadata`, `data`). Sunset list — **not authorization to convert the kernel in place**: `moe_expert_token_remap`. The writer is op-owned: convert in place.
- **RTA varargs:** none. Name every RTA (`token_start_idx`, `token_end_idx`).
- **Kernels are already Device 2.0** after `66b645994d5`: the port is a binding-layer change plus overrides 4a/4b, not an idiom rewrite. Transfers stay as they are.
- **The packet-header base comes from `get_read_ptr()` after `reserve_back(3)`** (`writer_all_to_all_combine.cpp:180-181`; a comment explains why it is pinned). Keep the call as is on the `DataflowBuffer`.
- **Templated CTAs.** Many CTAs feed template arguments (`get_device_expert_indices<…>`, `send_init_semaphore_to_configured_targets<…>`, `fabric_send_*<…>`); keep them `constexpr` (`constexpr auto x = get_arg(args::x)`).
- **2-device Ring along the dispatch axis:** East and West targets would both be `(src, neighbor, i)`; the validator rejects it. Record; do not work around it.
- **Do not read** `ttnn/cpp/ttnn/operations/experimental/quasar/**` for precedent or binding names.
- **Readiness sheet not read:** if River or the sheet owner reports `Is able to port?` = `no` for this op, stop.
