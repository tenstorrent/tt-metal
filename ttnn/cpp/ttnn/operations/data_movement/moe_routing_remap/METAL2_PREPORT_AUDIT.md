# Metal 2.0 Audit Findings — `data_movement/moe_routing_remap`

- **`MoeRoutingRemapDeviceOperation`**
  - `SingleCore` program factory (`device/moe_routing_remap_program_factory.cpp`), via `SingleCore::create_descriptor`

Single device-operation, single program factory. Kernels referenced by the factory:

- `device/kernels/dataflow/reader_moe_routing_remap.cpp` (own)
- `device/kernels/dataflow/writer_moe_routing_remap.cpp` (own)

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `597581e6151 2026-07-23 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/moe_routing_remap` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `MoeRoutingRemapDeviceOperation` → `SingleCore` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — both own kernels and both consumed donor functions are Device 2.0 native |
| *Prereqs* — Cross-op escapes | Ok — LLK + in-family `common.hpp` (Device-2.0-compliant); one unused cross-family include |
| *Feature Support* — overall | **GREEN** (all Appendix A entries `N/A`) |
| *Feature Support* — Variadic-CTA | Ok |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** |
| *TTNN Readiness* — Concept (current) | `descriptor` |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Is safe to port? | Yes |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — Runtime-args update | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `MetalV2FactoryConcept` |
| *Port work* — Offset base pointer | none (cleared) |
| *Port work* — Tensor bindings (per binding) | `routing_weights` Case 1 · `local_weights` (output) Case 1 |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | none present (no site) |
| *Port work* — CB endpoints | `c_0` 1:1 legal · `c_1` 1:1 legal · `c_2` self-loop |

**CB endpoints** are dispositions, not gates: every out-of-window CB has a port-time resolution. Here two CBs are ordinary 1:1 FIFOs and one is a single-toucher self-loop. Recorded per `(CB, config)` below; this op has a single config (single-core, fixed CB shapes).

## Result

**GREEN → brief issued.** All five gate-bearing subjects clear:

- **Device 2.0** — both kernels are written in Device 2.0 idioms (`Noc`, `DataflowBuffer`, object-method `.get_write_ptr()` / `.get_read_ptr()`, `noc.async_read` / `noc.async_write`, `CoreLocalMem`, `UnicastEndpoint`); the two donor functions they call are Device 2.0 compliant.
- **Feature compatibility** — no GlobalCircularBuffer, no non-zero `address_offset`, no GlobalSemaphore, no variable-count CTAs.
- **TTNN factory concept** — the readiness sheet's `Is able to port?` is `yes`; every cheaply-checkable column cross-checks clean against the code.
- **Offset base pointers** — no address RTA folds a host-side offset into its base; the one scalar offset the op passes (`device_weights_count_offset`) is an expert-count skip counter, never added to a device address.
- **TensorAccessor 3rd argument** — no accessor passes a page-size third argument.

No RED, no code-path subset needed — the whole op is portable.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN.** Readiness-sheet row (`data_movement/moe_routing_remap`, `MoeRoutingRemapDeviceOperation`, `SingleCore`): `Concept = descriptor`, `Custom hash = no`, `Runtime-args update (get_dynamic_runtime_args) = no`, `Runtime-args update (PD override_runtime_args) = no`, `Pybind descriptor = no`, `Smuggled pointer = no`, `Is safe to port? = yes`, `Is able to port? = yes`. Cross-check against code:
  - `Concept = descriptor` — [`create_descriptor`](device/moe_routing_remap_program_factory.cpp#L32-L36) returns a `ProgramDescriptor`. ✓
  - `Custom hash = no` — no `compute_program_hash` override anywhere in the op directory. ✓
  - `Runtime-args update = no` — no `get_dynamic_runtime_args` / `override_runtime_arguments` in the op. ✓
  - `Pybind descriptor = no` — [`moe_routing_remap_nanobind.cpp`](moe_routing_remap_nanobind.cpp#L53-L63) binds only the host op, not `create_descriptor`. ✓
  - `Smuggled pointer = no` — buffers reach the kernels as `Buffer*` binding entries via [`emplace_runtime_args`](device/moe_routing_remap_program_factory.cpp#L151-L152) (the framework-patched `BufferBinding` form), not as raw `->address()` values. ✓
  - Cross-column invariants hold (`Runtime-args update = no` on a `descriptor` row; `Op-owned tensors?` blank/absent on a `descriptor` row). ✓
- **Device 2.0 (every kernel used):** **GREEN.** No violations.

  Kernels are structurally Device 2.0:
  - [`reader_moe_routing_remap.cpp`](device/kernels/dataflow/reader_moe_routing_remap.cpp#L30-L64) — `Noc noc`, `DataflowBuffer`, `.reserve_back` / `.push_back` / object-method `.get_write_ptr()`, `noc.async_read(...)`, `noc.async_read_barrier()`, `TensorAccessor`.
  - [`writer_moe_routing_remap.cpp`](device/kernels/dataflow/writer_moe_routing_remap.cpp#L31-L63) — `Noc noc`, `DataflowBuffer`, `.reserve_back` / `.wait_front` / `.push_back` / object-method `.get_read_ptr()`, `CoreLocalMem`, `noc.async_write(...)`, `noc.async_write_barrier()`, `TensorAccessor`.

  The `get_write_ptr` / `get_read_ptr` sites are all the **object-method** form (`dfb.get_write_ptr()`) — Device 2.0 native — not the CB-index free-function holdover form. Donor functions actually called (see Out-of-directory coupling) are Device 2.0 compliant:
  - `tt::data_movement::common::tt_memmove<...>(noc, dst, src, bytes)` — the leading-`Noc` overload ([`common.hpp:89`](../common/kernels/common.hpp#L89)), Device 2.0 native. (The deprecated no-`Noc` overload at [`common.hpp:150`](../common/kernels/common.hpp#L150) is *not* the one called.)
  - `fill_with_val<T>(addr, n, val)` ([`common.hpp:157`](../common/kernels/common.hpp#L157)) — a plain L1 pointer fill, no NoC surface, Device-2.0-neutral.
  - `ByteSizeAddressType<Size>` ([`common.hpp:242`](../common/kernels/common.hpp#L242)) — a compile-time type trait, Device-2.0-neutral.

- **Feature compatibility:** all Appendix A entries `N/A` — no `GREEN` row exists (every entry is a gate-feature; absent ⇒ `N/A`).

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | Three plain `CBDescriptor`s ([factory L58-L102](device/moe_routing_remap_program_factory.cpp#L58-L102)); no `.global_circular_buffer` field, no `remote_index` / `remote_cb`, no 4-arg `CreateCircularBuffer(..., global_cb)`. |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `CBDescriptor` sets `address_offset` (default zero); no `set_address_offset`, no 4-arg `UpdateDynamicCircularBufferAddress`, no `cb_descriptor_from_sharded_tensor`. |
  | GlobalSemaphore | N/A | The op uses no semaphores at all (no `GlobalSemaphore`, no `CreateGlobalSemaphore`, no `global_semaphore.hpp`). |
  | Variable-count compile-time arguments (CTA varargs) | N/A | `tensor_args_t` is a fixed pair (`input_routing_weights` + `optional_output_routing_weights`), not a `std::vector<Tensor>`. Kernels read CTAs only at constexpr offsets (`get_compile_time_arg_val(0..5)`, `TensorAccessorArgs<5>` / `<6>`); no runtime-varying CTA index. |

- **CB endpoints (GATE-free):** three CBs, all with a clean port-time disposition; nothing blocks a Gen1 port. Single config (single-core `{0,0}`, fixed CB shapes), so one census.
  - **`c_0` — `routing_weights_dfb`** (input weights, [factory L56-L66](device/moe_routing_remap_program_factory.cpp#L56-L66)): reader FIFO-produces (`reserve_back` + `push_back`, [reader L34-L63](device/kernels/dataflow/reader_moe_routing_remap.cpp#L34-L63)), writer FIFO-consumes (`wait_front` + `get_read_ptr`, [writer L42-L45](device/kernels/dataflow/writer_moe_routing_remap.cpp#L42-L45)). Two touchers: one locked producer + one locked consumer → **plain 1:1, legal**. (The reader's raw read of its own filled buffer is a peek on its producer binding, not a second toucher.)
  - **`c_1` — `local_weights_idxs_dfb`** (per-device non-zero index scratch, [factory L69-L83](device/moe_routing_remap_program_factory.cpp#L69-L83)): reader produces (`reserve_back` + raw index write via `get_write_ptr` + `push_back`, [reader L43-L64](device/kernels/dataflow/reader_moe_routing_remap.cpp#L43-L64)), writer consumes (`wait_front` + `get_read_ptr`, [writer L43-L46](device/kernels/dataflow/writer_moe_routing_remap.cpp#L43-L46)). Two touchers, 1 producer + 1 consumer → **plain 1:1, legal**.
  - **`c_2` — `local_weights_dfb`** (output weights, [factory L86-L102](device/moe_routing_remap_program_factory.cpp#L86-L102)): touched by the **writer only** — `reserve_back` + `get_read_ptr` + `push_back`, `fill_with_val`, then `noc.async_write` from it ([writer L36-L62](device/kernels/dataflow/writer_moe_routing_remap.cpp#L36-L62)). Single toucher → **self-loop** (bind the writer as both PRODUCER and CONSUMER; legal on Gen1 for a DM kernel).
- **Offset base pointers:** **GREEN — cleared.** Both address RTAs resolve to clean bases:
  - Reader RTA `{routing_weights_buffer, device_weights_count_offset}` ([factory L151](device/moe_routing_remap_program_factory.cpp#L151)). `routing_weights_buffer` is a `Buffer*` (base, framework-patched), read in the kernel as `get_arg_val<uint32_t>(0)` and fed straight to `TensorAccessor(routing_weights_args, routing_weights_base_address)` ([reader L25-L28](device/kernels/dataflow/reader_moe_routing_remap.cpp#L25-L28)) — no host-folded offset. `device_weights_count_offset` ([factory L146-L147](device/moe_routing_remap_program_factory.cpp#L146-L147)) is **not** a device address: it is an expert-count skip value (`mesh_coordinate[axis] * non_zero_per_device`) used only as a loop comparison counter in the kernel (`weight_offset_count < device_weights_count_offset`, [reader L53](device/kernels/dataflow/reader_moe_routing_remap.cpp#L53)), never added to a NoC address. Not a Type 1/2 fold.
  - Writer RTA `{local_weights_buffer}` ([factory L152](device/moe_routing_remap_program_factory.cpp#L152)). `Buffer*` base → `get_arg_val<uint32_t>(0)` → `TensorAccessor(local_weights_args, local_weights_base_address)` ([writer L27-L29](device/kernels/dataflow/writer_moe_routing_remap.cpp#L27-L29)). Clean base.

  No op row appears in the offset-base-pointer triage prior (`2026-07-19_offset_base_pointers.md`); scan confirms clean, consistent with its absence. Type 3 / Type 4 do not apply.
- **TensorAccessor 3rd argument:** **GREEN — no site.** Both `TensorAccessor` constructions pass exactly two arguments (`args`, `base_addr`) — [reader L28](device/kernels/dataflow/reader_moe_routing_remap.cpp#L28), [writer L29](device/kernels/dataflow/writer_moe_routing_remap.cpp#L29). No explicit page-size override anywhere, so this subject does not fire. (Consistent with the op's absence from the `2026-07-06_tensor_accessor_3rd_arg_triage.md` prior.)

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding):
  - `routing_weights` (input, `c_0`) — **Case 1** (via `TensorAccessor`). Delivered today as a `Buffer*` binding; kernel reads the base at `get_arg_val<uint32_t>(0)` and builds `TensorAccessor(routing_weights_args, base)`. Express as a `TensorParameter` / `TensorBinding`; kernel builds `TensorAccessor(tensor::name)`; the address-via-RTA and the `TensorAccessorArgs<5>` plumbing both disappear.
  - `local_weights` (output, `c_2`) — **Case 1** (via `TensorAccessor`). Same shape; the writer's raw pointer arithmetic operates on **CB L1 addresses** (`get_read_ptr` results), not on the tensor base, so this is not Case 2. Express as a `TensorParameter` / `TensorBinding`; kernel builds `TensorAccessor(local_weights_args → tensor::name)`.
- **TensorParameter relaxation:** none (sheet `TensorParameter relaxation = none`; no custom hash).
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:** self-loop `c_2` (single config); `c_0` and `c_1` are ordinary 1:1 FIFOs (no action).

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none — no hidden second writer, no multi-reader. `c_0` / `c_1` are plain producer→consumer FIFOs; `c_2` is single-toucher.
- **Cross-op / shared kernels:** the op owns both its kernels (no file-path instantiation of a foreign kernel). It calls into the in-family shared header `ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp` (`tt_memmove`, `fill_with_val`, `ByteSizeAddressType`) — port the shared-header rewrite as one unit if it is touched. `ttnn/cpp/ttnn/operations/ccl/common/kernels/moe_utils.hpp` is `#include`d by both kernels but **no symbol from it is used** (see Misc anomalies).
- **RTA varargs:** none — reader reads two fixed distinct fields (`get_arg_val<uint32_t>(0)`, `(1)`); writer reads one (`(0)`). All nameable; no arg-indexed loop.
- **Mesh-coord-aware `create_descriptor`:** the factory's `create_descriptor` takes a `std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate` ([hpp L45-L49](device/moe_routing_remap_device_operation.hpp#L45-L49), [factory L32-L36](device/moe_routing_remap_program_factory.cpp#L32-L36)) and bakes a per-coordinate scalar (`device_weights_count_offset`) into each program. The readiness sheet classifies this as `descriptor` / able-to-port `yes`, and it is not an Appendix A feature — so it does not gate. Flagged so the porter wires the per-coord dispatch through the `MetalV2FactoryConcept` correctly rather than assuming a single coordinate-independent program. See Recipe notes.

## Team-only

- **Out-of-directory coupling & donor shape:**
  - **Op-level roll-up:** ✓ clean. Both kernels are op-owned; no foreign kernel is file-path-instantiated. Function-call escapes are LLK (`tt_metal/*`, no concern) and one in-family shared header whose consumed functions are Device-2.0-compliant.
  - **Summary table** (op kernel → donor include):

    | Op kernel | Donor include | Class | Status |
    |---|---|---|---|
    | reader, writer | `api/numeric/bfloat16.h`, `api/dataflow/noc.h`, `api/dataflow/dataflow_buffer.h`, `api/tensor/noc_traits.h`, `api/core_local_mem.h` (writer) | `tt_metal/*` LLK/HAL | ✓ no concern |
    | reader, writer | `ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp` | in-family shared | ✓ Device-2.0-compliant functions used |
    | reader, writer | `ttnn/cpp/ttnn/operations/ccl/common/kernels/moe_utils.hpp` | cross-family (ccl) | ✓ **unused** — no symbol referenced |

  - **Per-call detail** (`common.hpp`, the only donor with called functions): `tt_memmove<bool,bool,bool,uint32_t>(Noc, uint32_t, uint32_t, uint32_t)` — primitive + `Noc` signature, Device 2.0 native, ports cleanly; `fill_with_val<T>(uint32_t, uint32_t, T)` — primitive args, no resource handle; `ByteSizeAddressType<uint32_t>` — compile-time trait, no resource handle. No `Semaphore` / `CircularBuffer&` / addr-gen shapes in any consumed signature.
  - **Borrowed kernel files (file-path instantiation):** none. Both `KernelDescriptor::kernel_source` values point at the op's own `device/kernels/dataflow/` files ([factory L117-L118](device/moe_routing_remap_program_factory.cpp#L117-L118), [L134-L136](device/moe_routing_remap_program_factory.cpp#L134-L136)).
- **Relaxation candidates (mined from a custom hash):** none — the op has no custom hash.
- **TTNN factory analysis:** `descriptor` concept, single `SingleCore` factory. No op-owned tensors (no `WorkloadDescriptor`, no `buffers` vector). No pybind of `create_descriptor` or other migration-risky pybind. No custom hash, no custom `override_runtime_arguments`. Both tensor buffers are delivered via the framework-patched `Buffer*` (`BufferBinding`) form — correct-on-cache-hit today; the Metal 2.0 typed `TensorParameter` binding supersedes it. Per-coord dispatch via the mesh-coord-aware `create_descriptor` (see Heads-ups / Recipe notes). Target concept: `MetalV2FactoryConcept` (no op-owned tensors).

## Misc anomalies  *(team-only, non-gating)*

- **Operator-precedence bug in `validate_on_program_cache_miss`** — [`moe_routing_remap_device_operation.cpp:37-39`](device/moe_routing_remap_device_operation.cpp#L37-L39): the `TT_FATAL` condition `expert_parallel_size == (cluster_axis == 0) ? mesh_view.num_cols() : mesh_view.num_rows()` parses as `(expert_parallel_size == (cluster_axis == 0)) ? num_cols() : num_rows()` because `==` binds tighter than `?:`. The intended check is almost certainly `expert_parallel_size == ((cluster_axis == 0) ? num_cols() : num_rows())`. As written it compares `expert_parallel_size` against a bool (0/1) and then uses `num_cols()`/`num_rows()` (near-always non-zero) as the assertion condition, so the guard effectively never fires. Latent host-validation bug; routes to the ops team, not the port diff.
- **Unused cross-family include** — both kernels `#include "ttnn/operations/ccl/common/kernels/moe_utils.hpp"` ([reader L9](device/kernels/dataflow/reader_moe_routing_remap.cpp#L9), [writer L10](device/kernels/dataflow/writer_moe_routing_remap.cpp#L10)) but reference no symbol from it (`ttnn::operations::ccl::common`, `routing_state`, `Mux*`, `open_direction_*`, `polar_*` all absent). Dead include; harmless but removable. Removing it also drops the op's only cross-family coupling.

## Questions for the user  *(none)*

## Recipe notes

- **Mesh-coord-aware `create_descriptor` isn't covered by the concept-classification bullets.** The [TTNN factory concept prerequisite](../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/audit/metal2_audit.md) lists the `Concept` cues as `create_descriptor()` returning a `ProgramDescriptor` (descriptor), a mesh-workload return (`WorkloadDescriptor`), `create()` + `override_runtime_arguments()` (legacy), or `MetalV2`. This op's `create_descriptor` returns a `ProgramDescriptor` but takes an extra `std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate` and is dispatched once per mesh coordinate, baking a per-coord scalar into each program. That reads as a `descriptor`-with-per-coord-dispatch variant the classification text doesn't explicitly name. The readiness sheet resolves it cleanly (`descriptor`, able-to-port `yes`), so the gate is unambiguous here — but a one-line acknowledgement in the concept bullets that a per-coord-dispatched `create_descriptor` is still the `descriptor` concept would spare the next auditor the same double-take, and it would help the porter recipe be explicit about how per-coord dispatch maps onto `MetalV2FactoryConcept`.
