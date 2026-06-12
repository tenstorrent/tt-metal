# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/transformer/sdpa`

This directory bundles five DeviceOperations sharing a kernel pool and common dataflow header. Audited as a bundle with per-DeviceOperation attribution where findings differ.

- **`SDPAOperation`**
  - `SDPAProgramFactory` (`device/sdpa_program_factory.cpp`)
- **`JointSDPADeviceOperation`**
  - `JointSDPAProgramFactory` (`device/joint_sdpa_program_factory.cpp`)
- **`RingDistributedSdpaDeviceOperation`**
  - `RingDistributedSdpaProgramFactory` (`device/ring_distributed_sdpa_program_factory.cpp`)
- **`RingJointSDPADeviceOperation`**
  - `RingJointSDPAMeshWorkloadFactory` (`device/ring_joint_sdpa_program_factory.cpp`)
- **`ExpRingJointSDPADeviceOperation`**
  - `ExpRingJointSDPAProgramFactory` (`device/exp_ring_joint_sdpa_program_factory.cpp`)

Shared kernels under `device/kernels/`: `reader_interleaved.cpp`, `writer_interleaved.cpp`, `joint_reader.cpp`, `joint_writer.cpp`, `ring_joint_reader.cpp`, `ring_joint_writer.cpp`, `exp_ring_joint_reader.cpp`, `exp_ring_joint_writer.cpp`, `compute/sdpa.cpp`, `compute/joint_sdpa.cpp` and the shared header `device/kernels/dataflow/dataflow_common.hpp`.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `port_op_to_metal2_audit.md`.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/transformer/sdpa` |
| **Overall** | RED |
| **DOps / Factories** | `SDPAOperation` → `SDPAProgramFactory`; `JointSDPADeviceOperation` → `JointSDPAProgramFactory`; `RingDistributedSdpaDeviceOperation` → `RingDistributedSdpaProgramFactory`; `RingJointSDPADeviceOperation` → `RingJointSDPAMeshWorkloadFactory`; `ExpRingJointSDPADeviceOperation` → `ExpRingJointSDPAProgramFactory` |
| *Prereqs* — ProgramDescriptor | Yes |
| *Prereqs* — Device 2.0 (every kernel used) | Yes-with-holdovers (YELLOW — fix on D2.0 track first) |
| *Prereqs* — Cross-op escapes | Ok |
| *Feature Support* — overall | RED |
| *Feature Support* — Variadic-CTA | Ok |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — MeshWorkload needed | Yes (genuine): RingJointSDPAMeshWorkloadFactory has `create_mesh_workload()` / `cached_mesh_workload_t`; ring protocol requires genuine cross-device coordination |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Other risky pybind | None |
| *TTNN Readiness* — Custom hash | Yes → delete (see Custom program hash) |
| *TTNN Readiness* — Custom override-RTA | Yes: `RingJointSDPAMeshWorkloadFactory::override_runtime_arguments` (`device/ring_joint_sdpa_program_factory.cpp:2148`) |
| *TTNN Readiness* — Fake CBs (address-only) | None |

**Fake CBs** = CBs used purely as an address source. **Litmus: does the CB have a producer *and* a consumer?** `cb_k.get_write_ptr()` at `reader_interleaved.cpp` captures the write pointer for NoC forwarding, but `cb_k` has `reserve_back` / `push_back` (reader producer) and `cb_wait_front(cb_k_in, ...)` in the compute kernel (consumer) — genuine FIFO. All other CB address-pointer captures examined are similarly bracketed by producer+consumer. No fake CBs found.

## Result

**RED — blocked on GlobalSemaphore**, routed to the wait-for-feature team. `RingJointSDPADeviceOperation` and `ExpRingJointSDPADeviceOperation` use `tt::tt_metal::GlobalSemaphore` as a direct input parameter (via `operation_attributes_t` / embedded struct). Metal 2.0 does not yet support `GlobalSemaphore`; these two DeviceOperations cannot be ported until that support lands on `KernelSpec`.

**Scoped subset that is clear:** `SDPAOperation`, `JointSDPADeviceOperation`, and `RingDistributedSdpaDeviceOperation` use no GlobalSemaphore. Their subset gate status is YELLOW only (one isolated Device 2.0 holdover in the shared `dataflow_common.hpp` kernel header). A scoped port of that subset is feasible once the Device 2.0 holdover is cleaned on the D2.0 track.

## Gate detail

- **ProgramDescriptor:** GREEN — all five DeviceOperations use `ProgramDescriptor` / `KernelDescriptor` / `CBDescriptor` / `SemaphoreDescriptor`. No `CreateProgram` / `CreateKernel` / `CreateCircularBuffer` imperative calls found. Includes `<tt-metalium/program_descriptors.hpp>`.

- **Device 2.0 (every kernel used):** YELLOW — one isolated holdover in the shared dataflow header.

  | File | Line | Call | Wrapper in scope |
  |---|---|---|---|
  | `device/kernels/dataflow/dataflow_common.hpp` | 514 | `get_write_ptr(cb_id)` (free-function form) | None — function `fill_diagonal_edge_tile_bf16()` takes `uint32_t cb_id`, no `CircularBuffer` wrapper in scope |

  All other kernel files (`reader_interleaved.cpp`, `writer_interleaved.cpp`, `joint_reader.cpp`, `joint_writer.cpp`, `ring_joint_reader.cpp`, `ring_joint_writer.cpp`, `exp_ring_joint_reader.cpp`, `exp_ring_joint_writer.cpp`, `compute/sdpa.cpp`, `compute/joint_sdpa.cpp`) are fully Device 2.0 compliant: use `Noc noc`, `CircularBuffer cb(id)` member-form API, `TensorAccessor`, `Semaphore<>`, etc.

  Fix: update `fill_diagonal_edge_tile_bf16()` to accept (or construct) a `CircularBuffer` wrapper and use `cb.get_write_ptr()`. This is Device 2.0 track work — **not** a port step.

- **Feature compatibility:**

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No reference to `experimental::GlobalCircularBuffer` found |
  | Dynamic CircularBuffer (borrowed memory) | N/A | No `CBDescriptor::buffer` set to non-null; no `set_globally_allocated_address` |
  | CBDescriptor `address_offset` (non-zero) | N/A | No non-zero `address_offset` set on any `CBDescriptor` |
  | Aliased Circular Buffers | N/A | All `format_descriptors` initializers are single-element |
  | GlobalSemaphore | RED | `RingJointSDPADeviceOperation` and `ExpRingJointSDPADeviceOperation` use `tt::tt_metal::GlobalSemaphore` — see detail block below |
  | Non-zero semaphore initial value | GREEN | `.initial_value = VALID` (=1) appears at three sites — heads-up; does not gate |
  | Dynamic TensorAccessor (`ArgConfig::Runtime*`) | N/A | No `ArgConfig::Runtime*` tokens in op host code |
  | `UpdateCircularBuffer*` | N/A | No `UpdateCircularBufferTotalSize` / `UpdateCircularBufferPageSize` / `UpdateDynamicCircularBufferAddressAndTotalSize` found |
  | Variable-count compile-time arguments (CTA varargs) | N/A | All `get_compile_time_arg_val(i)` calls use fixed-integer offsets, not loop variables; all tensor input lists have fixed count at factory construction time |

#### GlobalSemaphore — RED

Signal matches:
- `RingJointSDPADeviceOperation::create_program` takes `const std::vector<GlobalSemaphore>& multi_device_global_semaphore` (`device/ring_joint_sdpa_device_operation.hpp:48`)
- `RingJointSDPAParams` embeds `experimental::prim::RingAttentionAllGatherAsyncParams all_gather_operation_attributes` which carries `std::vector<GlobalSemaphore> semaphore` (`device/ring_attention_all_gather_async/device/ring_attention_all_gather_async_device_operation_types.hpp:33`)
- `ExpRingJointSDPADeviceOperation::create_program` takes `const std::vector<GlobalSemaphore>& multi_device_global_semaphore` (`device/exp_ring_joint_sdpa_device_operation.hpp:44`)
- `ExpRingJointSDPAParams` embeds `std::vector<GlobalSemaphore> semaphore` (`device/exp_ring_joint_sdpa_device_operation_types.hpp:34`)
- Both operation headers `#include <tt-metalium/global_semaphore.hpp>`
- Nanobind wrapper (`sdpa_nanobind.cpp`) passes `const std::vector<GlobalSemaphore>&` for both ring ops

Expected resolution: not yet supported in Metal 2.0; port will be possible once `GlobalSemaphore` support lands on `KernelSpec`. The two affected DeviceOperations (`RingJointSDPADeviceOperation`, `ExpRingJointSDPADeviceOperation`) must wait for that feature. The remaining three DeviceOperations are unblocked.

#### Non-zero semaphore initial value — GREEN (FYI-P heads-up)

Sites where `.initial_value = VALID` (VALID = 1, from `hostdevcommon/common_values.hpp:14`):
- `device/sdpa_program_factory.cpp:757` — `SemaphoreDescriptor{ ..., .initial_value = VALID }`
- `device/exp_ring_joint_sdpa_program_factory.cpp:508` — `SemaphoreDescriptor{ ..., .initial_value = VALID }`

Sites where `.initial_value = INVALID` (INVALID = 0 — fine, not a non-zero use):
- `device/sdpa_program_factory.cpp:745, 751`
- `device/exp_ring_joint_sdpa_program_factory.cpp:494, 501`

Note: the construct is supported on Gen1 today via `SemaphoreSpec::advanced_options.initial_value`, but that field is `[[deprecated]]` and unsupported on Gen2. Porter should acknowledge the deprecation warning when translating.

## Port-work summary

- **Tensor bindings** (per binding, all factories): All tensor buffer addresses are pushed into `KernelDescriptor::RTArgList` as `Buffer*` pointers (not via `.address()`) — **Case 1** across all five factories. The framework registers these as `BufferBinding` entries and patches addresses on cache hits. Port re-expresses each as a `TensorParameter` + `TensorBinding`; kernel-side code uses `TensorAccessor(ta::name)`.

  Affected bindings across factories (representative list):
  - `SDPAProgramFactory`: q, k, v, attn_mask (optional), page_table (optional), attention_sink (optional), chunk_start_idx_tensor (optional), output — all Case 1
  - `JointSDPAProgramFactory`: q, k, v, joint_q, joint_k, joint_v, output, joint_output — all Case 1
  - `RingDistributedSdpaProgramFactory`: q, k, v, page_table (optional), output — all Case 1
  - `RingJointSDPAMeshWorkloadFactory`: q, k, v, gathered_k, gathered_v, joint_q (optional), joint_k (optional), joint_v (optional), output, joint_output, stats_output — all Case 1
  - `ExpRingJointSDPAProgramFactory`: q, k, v and associated ring/joint tensors — all Case 1

- **Custom hash:** present on `SDPAOperation`, `RingJointSDPADeviceOperation`, `ExpRingJointSDPADeviceOperation`. In all cases the hash encodes tensor layout + shape attributes and applies conditional exclusion (e.g. `chunk_start_idx_for_hash` excluded when `flexible_chunked`). Delete custom `compute_program_hash` → default (sanctioned exception). Before deletion, mine the conditional exclusions for relaxation candidates.

## Heads-ups

- **Notable LANDED constructs:**
  - Non-zero semaphore initial value (`VALID=1`): `device/sdpa_program_factory.cpp:757`, `device/exp_ring_joint_sdpa_program_factory.cpp:508` → translate to `SemaphoreSpec::advanced_options.initial_value` (deprecated; Gen2-unsupported).

- **Fake CBs (address-only):** None. `cb_k.get_write_ptr()` in `reader_interleaved.cpp` (KV forwarding) is address-capture for NoC send on a CB that has both producer (`reserve_back` / `push_back`) and consumer (`cb_wait_front` in compute kernel) — not a fake CB.

- **Cross-op / shared kernels:**
  - `dataflow_common.hpp` is shared across all five DeviceOperations; the one Device 2.0 holdover at line 514 affects every factory that calls `fill_diagonal_edge_tile_bf16()` (SDPAOperation is the primary user).
  - `ExpRingJointSDPAProgramFactory` borrows `tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp` (a fabric-team-owned kernel, instantiated at line 1691 under `#ifdef USE_MUX`). The port of `ExpRingJointSDPADeviceOperation` must coordinate with the fabric team on that kernel's Metal 2.0 status; do not assume it is independently portable.

- **RTA varargs:** No kernel uses `get_vararg(i)`. All RTArgs are named by fixed index. No vararg concern.

- **TTNN factory analysis (porter-relevant):**
  - No pybind `create_descriptor` exposure found in `sdpa_nanobind.cpp` — nothing to delete.
  - No other migration-risky pybind identified.
  - Custom `override_runtime_arguments`: `RingJointSDPAMeshWorkloadFactory::override_runtime_arguments` at `device/ring_joint_sdpa_program_factory.cpp:2148` — calls `descriptor_adapter_t::apply_descriptor` plus `apply_ring_joint_scalar_runtime_args(program, ...)` using `GetRuntimeArgs`. Porter must translate the `GetRuntimeArgs` patching into the Metal 2.0 override-hook form.

## Team-only

- **TensorAccessor convertibility:** All bindings are standard interleaved or paged-interleaved (`TensorAccessorArgs(buf).append_to(...)`) — straightforwardly convertible via `TensorParameter` + `TensorBinding`. No exotic bank-walk identified.

- **Out-of-directory coupling and donor shape:**
  - `ExpRingJointSDPAProgramFactory` borrows `tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp` — fabric team boundary.
  - `RingJointSDPAParams` embeds `experimental::prim::RingAttentionAllGatherAsyncParams` from `ttnn/cpp/ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/` — ccl op coupling for ring protocol.

- **Relaxation candidates (mined from custom hash before deletion):**
  - `SDPAOperation::compute_program_hash` (`device/sdpa_device_operation.cpp:379`): conditionally excludes `chunk_start_idx_for_hash` when `flexible_chunked`, and `page_table_for_hash` when `flexible_chunked`. This suggests the default Metal 2.0 hash (over all tensor specs) would be too conservative for the flexible-chunked mode — a `match_padded_shape_only` or `dynamic_tensor_shape` relaxation on the chunk-start-idx tensor may be needed. **FALLIBLE — verify against actual dispatch patterns before adopting.**

- **TTNN factory analysis:**
  - Op-owned tensors: No. `create_output_tensors()` calls `create_device_tensor()` only for output tensors (not intermediate buffers the factory manages itself).
  - MeshWorkload: Yes (genuine). `RingJointSDPAMeshWorkloadFactory` has `create_mesh_workload()` and `cached_mesh_workload_t` at `device/ring_joint_sdpa_program_factory.hpp:43`. The ring protocol requires each device to run a distinct program coordinated with its ring neighbors — this is a real multi-program MeshWorkload need, not an op-owned-tensor artifact.
  - Pybind `create_descriptor`: None found in `sdpa_nanobind.cpp`.
  - Other risky pybind: None identified. `sdpa_nanobind.cpp` uses `ttnn::bind_function<>` for all op-function surface; `GlobalSemaphore` parameters are carried through but are already typed correctly.
  - Custom hash: Present on `SDPAOperation` (`device/sdpa_device_operation.cpp:379`), `RingJointSDPADeviceOperation` (present; `device/ring_joint_sdpa_device_operation.cpp`), `ExpRingJointSDPADeviceOperation` (present; `device/exp_ring_joint_sdpa_device_operation.cpp`). All are delete-and-default candidates per Metal 2.0 convention, with conditional-exclusion mining needed before deletion.
  - Custom `override_runtime_arguments`: `RingJointSDPAMeshWorkloadFactory::override_runtime_arguments` at `device/ring_joint_sdpa_program_factory.cpp:2148`.

## Per-DeviceOperation attribution

| DeviceOperation | ProgramDescriptor | Device 2.0 | GlobalSemaphore | Subset gate |
|---|---|---|---|---|
| `SDPAOperation` | Yes | YELLOW (shared `dataflow_common.hpp:514`) | N/A | YELLOW — clear once holdover fixed |
| `JointSDPADeviceOperation` | Yes | YELLOW (shared `dataflow_common.hpp:514`) | N/A | YELLOW — clear once holdover fixed |
| `RingDistributedSdpaDeviceOperation` | Yes | YELLOW (shared `dataflow_common.hpp:514`) | N/A | YELLOW — clear once holdover fixed |
| `RingJointSDPADeviceOperation` | Yes | YELLOW (shared `dataflow_common.hpp:514`) | **RED** | RED |
| `ExpRingJointSDPADeviceOperation` | Yes | YELLOW (shared `dataflow_common.hpp:514`) | **RED** | RED |

## Recipe notes

- The recipe's "per-DeviceOperation" bundling guidance (§ "Bundling multiple DeviceOperations") is well-suited to this directory's structure but does not prescribe whether to issue a scoped brief for the clean subset when the bundle-level result is RED. The recipe says "No brief on RED" at op level. The subset (SDPAOperation + JointSDPA + RingDistributed) clears all hard gates (YELLOW only). Audit reads this as RED at op level, no brief; the per-DeviceOperation attribution table above conveys which subset is clear.
- The MeshWorkload false-positive check (recipe §5, Q2) required verifying `ring_joint_sdpa_program_factory.hpp` to confirm `create_mesh_workload()` and `cached_mesh_workload_t` are the genuine MeshWorkload protocol (not just an op-owned-tensor workaround). The recognition guidance in the recipe is sufficient, but the presence of both `create_mesh_workload` and `override_runtime_arguments` on the same factory made the judgment non-trivial — a worked example of a genuine MeshWorkload would help.
