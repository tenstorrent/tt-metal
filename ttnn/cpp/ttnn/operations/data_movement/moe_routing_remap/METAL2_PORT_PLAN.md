# Port Plan — `data_movement/moe_routing_remap`

Port plan for `moe_routing_remap`, from the legacy `ProgramDescriptor` factory concept to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

> **Outcome: the port is BLOCKED at the factory-concept level and does not proceed.**
> During the planning step (the "Plan the spec" gate) I found a structural issue the audit
> did not catch: the op bakes a **per-mesh-coordinate** scalar (`device_weights_count_offset`)
> into each coordinate's program, and the `MetalV2FactoryConcept` the audit selected **cannot
> express per-mesh-coordinate runtime-arg variation**. This is a genuine capability gap, not a
> size or comprehension limit. The legacy inventory below is complete, and the planned spec
> shape is fully worked out (it is a head start for the future porter once the framework's
> multi-program concept lands), but **no code was constructed or changed**. The blocker is
> recorded in [TTNN ProgramFactory](#ttnn-programfactory) and [Deferred / Flagged](#deferred--flagged);
> the full handoff is in `METAL2_PORT_REPORT.md`. Recipe outcome: **CAPITULATED** (a success-tier
> grounded stop, per the recipe's operating posture).

## Legacy Inventory

*Observation step — complete.*

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` — `SingleCore::create_descriptor` returns a `tt::tt_metal::ProgramDescriptor` ([device/moe_routing_remap_program_factory.cpp#L32-L36](device/moe_routing_remap_program_factory.cpp#L32-L36)).
- Variants: single (`MoeRoutingRemapDeviceOperation::program_factory_t = std::variant<SingleCore>`, [device/moe_routing_remap_device_operation.hpp#L52](device/moe_routing_remap_device_operation.hpp#L52)).
- **Mesh-coord-aware**: `create_descriptor` takes a fourth argument `const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate` ([device/moe_routing_remap_device_operation.hpp#L45-L49](device/moe_routing_remap_device_operation.hpp#L45-L49)). The framework's descriptor adapter iterates every mesh coordinate and calls `create_descriptor` once per coordinate with that distinct coordinate (see [Flags](#flags) and the report).
- Custom `compute_program_hash`: none — the op uses the default reflection-based hash.

*(The Metal 2.0 factory concept the port targets was chosen during the audit — `MetalV2FactoryConcept`. Carried forward in [TTNN ProgramFactory](#ttnn-programfactory) below, where the concept-fit blocker is recorded.)*

### Kernels
| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | config |
|---|---|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/reader_moe_routing_remap.cpp` | `{{0,0},{0,0}}` | `[0]` routing_weights_cb_id (=`c_0`); `[1]` local_weights_idxs_cb_id (=`c_1`); `[2]` num_cluster_experts; `[3]` non_zero_per_device; `[4]` input_datum_size_bytes; then `TensorAccessorArgs(*routing_weights_buffer)` appended from index 5 ([factory L108-L114](device/moe_routing_remap_program_factory.cpp#L108-L114)) | none | `{routing_weights_buffer (Buffer*), device_weights_count_offset}` at core `{0,0}` ([factory L151](device/moe_routing_remap_program_factory.cpp#L151)) | none | none | `ReaderConfigDescriptor{}` |
| writer | `device/kernels/dataflow/writer_moe_routing_remap.cpp` | `{{0,0},{0,0}}` | `[0]` routing_weights_cb_id; `[1]` local_weights_idxs_cb_id; `[2]` local_weights_cb_id (=`c_2`); `[3]` num_cluster_experts; `[4]` non_zero_per_device; `[5]` input_datum_size_bytes; then `TensorAccessorArgs(*local_weights_buffer)` appended from index 6 ([factory L124-L131](device/moe_routing_remap_program_factory.cpp#L124-L131)) | none | `{local_weights_buffer (Buffer*)}` at core `{0,0}` ([factory L152](device/moe_routing_remap_program_factory.cpp#L152)) | none | none | `WriterConfigDescriptor{}` |

Both `config`s are the plain reader/writer defaults (no custom processor/NOC), so on Gen1 they map to the `create_reader_datamovement_config` / `create_writer_datamovement_config` helpers.

### CBs
| index | name | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|---|
| `c_0` | routing_weights_dfb | `aligned_routing_weight_page_size_bytes` = `align(routing_weights.tensor_spec().compute_page_size_bytes(), l1_alignment)` | `{{0,0},{0,0}}` | `datatype_to_dataformat_converter(routing_weights.dtype())` | same as total_size | not set |
| `c_1` | local_weights_idxs_dfb | `align(non_zero_per_device * sizeof(uint16_t), l1_alignment)` | `{{0,0},{0,0}}` | `datatype_to_dataformat_converter(convert_to_data_type<uint16_t>())` | same as total_size | not set |
| `c_2` | local_weights_dfb | `aligned_local_weights_page_size_bytes` = `align(tensor_return_value.tensor_spec().compute_page_size_bytes(), l1_alignment)` | `{{0,0},{0,0}}` | `datatype_to_dataformat_converter(tensor_return_value.dtype())` | same as total_size | not set |

All three are plain `CBDescriptor`s (single-element `format_descriptors`), each one page. No `GlobalCircularBuffer`, no aliasing, no `address_offset`, no borrowed memory. Factory allocation: [factory L56-L102](device/moe_routing_remap_program_factory.cpp#L56-L102). `TT_FATAL` enforces `c_2` format == `c_0` format ([factory L92](device/moe_routing_remap_program_factory.cpp#L92)).

### Semaphores
none.

### Tensor accessors
| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| reader kernel: `TensorAccessor(routing_weights_args, routing_weights_base_address)` ([reader L28](device/kernels/dataflow/reader_moe_routing_remap.cpp#L28)); host `TensorAccessorArgs(*routing_weights_buffer).append_to(reader_ct_args)` ([factory L114](device/moe_routing_remap_program_factory.cpp#L114)) | `routing_weights` (input, `tensor_args.input_routing_weights`) | reader RTA index 0 (`Buffer*`), read as `get_arg_val<uint32_t>(0)` ([reader L25](device/kernels/dataflow/reader_moe_routing_remap.cpp#L25)) |
| writer kernel: `TensorAccessor(local_weights_args, local_weights_base_address)` ([writer L29](device/kernels/dataflow/writer_moe_routing_remap.cpp#L29)); host `TensorAccessorArgs(*local_weights_buffer).append_to(writer_ct_args)` ([factory L131](device/moe_routing_remap_program_factory.cpp#L131)) | `local_weights` (output, `tensor_return_value`) | writer RTA index 0 (`Buffer*`), read as `get_arg_val<uint32_t>(0)` ([writer L27](device/kernels/dataflow/writer_moe_routing_remap.cpp#L27)) |

Both are **Case 1** (via `TensorAccessor`), 2-argument construction (no page-size 3rd argument). The writer's raw pointer arithmetic ([writer L50-L52](device/kernels/dataflow/writer_moe_routing_remap.cpp#L50-L52)) operates on **CB L1 addresses** (`get_read_ptr()` results), not on the tensor base — so it is not Case 2.

### Work split
n/a — single core `{0,0}`. No `split_work_to_cores`.

### Cross-op kernels
none file-path-instantiated — both `KernelDescriptor::kernel_source` values point at the op's own `device/kernels/dataflow/` files ([factory L117-L118](device/moe_routing_remap_program_factory.cpp#L117-L118), [L134-L136](device/moe_routing_remap_program_factory.cpp#L134-L136)).

Donor include (function-call escape, not file-path instantiation): the in-family shared header `ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp` — consumed functions `tt_memmove<...>(Noc, ...)`, `fill_with_val<T>(...)`, `ByteSizeAddressType<Size>`; all Device-2.0-compliant (audit-verified). If that header's Metal 2.0 rewrite is ever touched, port it as one unit across all consumers. These callees take raw `uint32_t` L1 addresses (not CB ids), so no `dfb::name` would cross the boundary here.

### Flags
- **Per-coord dispatch (THE BLOCKER — surfaced in planning).** The factory bakes `device_weights_count_offset = compute_weight_count_offset(*mesh_dispatch_coordinate, cluster_axis, non_zero_per_device)` = `mesh_coordinate[axis] * non_zero_per_device` into each program's reader RTA ([factory L15-L25](device/moe_routing_remap_program_factory.cpp#L15-L25), [L146-L151](device/moe_routing_remap_program_factory.cpp#L146-L151)). Because `create_descriptor` takes the coordinate, the framework's descriptor adapter dispatches **one program per mesh coordinate**, each with its own offset. This per-mesh-coordinate variation cannot be expressed on `MetalV2FactoryConcept` — see [TTNN ProgramFactory](#ttnn-programfactory) and [Deferred / Flagged](#deferred--flagged).
- **Unused cross-family include (incidental, out of scope).** Both kernels `#include "ttnn/operations/ccl/common/kernels/moe_utils.hpp"` ([reader L9](device/kernels/dataflow/reader_moe_routing_remap.cpp#L9), [writer L10](device/kernels/dataflow/writer_moe_routing_remap.cpp#L10)) but reference no symbol from it. Dead include — the audit's Misc anomalies also noted it.
- **Operator-precedence bug in `validate_on_program_cache_miss` (incidental, out of scope).** [device/moe_routing_remap_device_operation.cpp#L37-L39](device/moe_routing_remap_device_operation.cpp#L37-L39): `expert_parallel_size == (cluster_axis == 0) ? num_cols() : num_rows()` parses as `(expert_parallel_size == (cluster_axis == 0)) ? ...`, so the guard effectively never fires. Latent host-validation bug — routes to the ops team, not this diff (also noted by the audit).
- **`cluster_axis` docstring contradicts the code (incidental, out of scope).** The nanobind docstring says "0: columns, 1: rows" ([moe_routing_remap_nanobind.cpp#L42](moe_routing_remap_nanobind.cpp#L42)); the kernel/test/validate convention is the opposite (axis 0 = rows). Route to the ops team.

## TTNN ProgramFactory

*Filled during the planning step. The concept was chosen in the audit; this section carries it forward — and records why the op does not fit it.*

- **Concept (inherited from audit)**: `MetalV2FactoryConcept`.
- **Custom `compute_program_hash`**: none.
- **Implementation notes / concept-fit blocker**:

  **The op does not fit `MetalV2FactoryConcept`, and the port stops here.** The concept's factory entry point, `create_program_artifacts(attributes, tensor_args, tensor_return_value)`, receives **no mesh coordinate** and returns a **single** `ProgramArtifacts` (one `ProgramSpec` + one `ProgramRunArgs`). The framework's `MetalV2MeshWorkloadFactoryAdapter` calls it **once** and stamps that single artifact — the same spec and the *same* run-args — onto **every** mesh coordinate range ([`mesh_device_operation_adapter.hpp#L884`](../../../../../../ttnn/api/ttnn/mesh_device_operation_adapter.hpp#L884), [L904-L909](../../../../../../ttnn/api/ttnn/mesh_device_operation_adapter.hpp#L904-L909)). Runtime args are keyed by `NodeCoord` (a Tensix core within a single chip; `NodeCoord = CoreCoord`), which has **no mesh-coordinate axis**, and the cache-hit path refreshes only tensor bindings.

  This op requires the opposite: a **per-mesh-coordinate** reader RTA (`device_weights_count_offset`), load-bearing on every execution (see [Flags](#flags) and [Deferred / Flagged](#deferred--flagged)). There is no in-scope way to bridge that onto the single-program concept. This matches the `ttnn_factory.md` feasibility gate's **"multi-program / per-coord variation → BLOCKED"** case, which awaits a future `MeshWorkloadSpecFactoryConcept` (the framework itself names it as not-yet-existing). Per the recipe, I surfaced this disagreement with the audit to the invoker rather than overriding it or improvising a workaround.

The sections below record the spec design that *would* have been built, worked out fully before the blocker was confirmed. **None of it was constructed** — it is documentation for the future porter, not a description of shipped code.

## Planned Spec Shape

*Design only — NOT built. The port is blocked at the factory-concept level (above).*

- **KernelSpecs**: `reader` and `writer`, 1:1 with the legacy `KernelDescriptor`s. Both `hw_config` are the arch-agnostic defaults: `create_reader_datamovement_config(device->arch())` for the reader, `create_writer_datamovement_config(device->arch())` for the writer (resolved triples match the reader/writer defaults exactly).
- **DataflowBufferSpecs**: `routing_weights_dfb` (`c_0`), `local_weights_idxs_dfb` (`c_1`), `local_weights_dfb` (`c_2`). `entry_size` / `num_entries` computed from the same page-size math the legacy factory uses; `data_format_metadata` copied from each legacy CB's format; `tile_format_metadata` unset (all standard, no tile set). `c_2` is a **self-loop** (see [Applied Patterns](#applied-patterns)).
- **SemaphoreSpecs**: none.
- **TensorParameters**: `routing_weights` (input, from `tensor_args.input_routing_weights.tensor_spec()`), `local_weights` (output, from `tensor_return_value.tensor_spec()`). Both Case 1. Strict `TensorSpec` matching (no relaxation).
- **WorkUnitSpecs**: one — `{reader, writer}` on `target_nodes = NodeCoord{0,0}`.
- **Op-owned tensors**: none.

## Preserved Multiplicity

none — no work-split multiplicity in legacy (single core, one `KernelDescriptor` per role).

## Dropped Plumbing

*Design only — NOT built.*

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader RTA slot 0 ([factory L151](device/moe_routing_remap_program_factory.cpp#L151)) | `routing_weights_buffer` (`Buffer*`) → kernel `get_arg_val<uint32_t>(0)` | `TensorBinding(routing_weights)` + kernel `TensorAccessor(tensor::routing_weights)` |
| writer RTA slot 0 ([factory L152](device/moe_routing_remap_program_factory.cpp#L152)) | `local_weights_buffer` (`Buffer*`) → kernel `get_arg_val<uint32_t>(0)` | `TensorBinding(local_weights)` + kernel `TensorAccessor(tensor::local_weights)` |
| reader CTA slot 5+ ([factory L114](device/moe_routing_remap_program_factory.cpp#L114)); kernel `TensorAccessorArgs<5>()` ([reader L21](device/kernels/dataflow/reader_moe_routing_remap.cpp#L21)) | `TensorAccessorArgs(*routing_weights_buffer).append_to(...)` | binding mechanism end-to-end |
| writer CTA slot 6+ ([factory L131](device/moe_routing_remap_program_factory.cpp#L131)); kernel `TensorAccessorArgs<6>()` ([writer L24](device/kernels/dataflow/writer_moe_routing_remap.cpp#L24)) | `TensorAccessorArgs(*local_weights_buffer).append_to(...)` | binding mechanism end-to-end |
| reader CTA slots 0,1; writer CTA slots 0,1,2 | magic CB indices (`c_0`/`c_1`/`c_2`) | `DFBBinding`s (reader: `c_0` PRODUCER, `c_1` PRODUCER; writer: `c_0` CONSUMER, `c_1` CONSUMER, `c_2` self-loop) |
| reader CTA slots 2-4; writer CTA slots 3-5 | positional CTAs (`num_cluster_experts`, `non_zero_per_device`, `input_datum_size_bytes`) | named CTAs (same names) |

**Would-be-kept RTA — and the crux of the blocker**: `device_weights_count_offset` (reader RTA index 1). The brief directed porting it as a named RTA that "stays baked per coordinate." That is exactly the operation `MetalV2FactoryConcept` cannot perform: the single `create_program_artifacts` call produces one run-args value applied to every coordinate, so "baked per coordinate" is unreachable. This is the item that blocks the port.

## Applied Patterns

*Design only — NOT built.*

- [Sync-free / single-ended CB → self-loop DFB](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-sync-free-and-single-ended-cbs--self-loop-dfb): `c_2` (`local_weights_dfb`) is touched only by the writer (`reserve_back` + `get_read_ptr` + `push_back`, fill, then `noc.async_write` from it) — bind the writer as both PRODUCER and CONSUMER.
- `c_0` and `c_1` are ordinary 1:1 FIFOs (reader PRODUCER → writer CONSUMER) — no special pattern.

## Deferred / Flagged

- **New finding that blocks the port (audit gap).** The op requires **per-mesh-coordinate runtime-arg variation** (`device_weights_count_offset`), which `MetalV2FactoryConcept` cannot express. The audit's TTNN-factory feasibility gate should have fired **RED** ("multi-program / per-coord variation → BLOCKED"), but classified the op GREEN — treating the mesh-coord-aware `create_descriptor` as a portable `descriptor`-with-per-coord-dispatch variant and assuming the porter could "wire the per-coord dispatch through `MetalV2FactoryConcept` correctly." No such wiring exists today (verified against the framework: single-spec stamping, `NodeCoord`-only RTA keying, the per-coord `MakeMeshWorkloadFromSpecs` path not bridged to `create_program_artifacts`, and no multi-program concept defined on this branch). Full evidence and the sketch of the framework change needed are in `METAL2_PORT_REPORT.md` under Handoff points. Notably, the audit's own "Recipe notes" flagged uncertainty about "how per-coord dispatch maps onto `MetalV2FactoryConcept`" — the answer is that it does not map yet.
- **Incidental defects (out of scope — routed to the report, not fixed):** the operator-precedence bug in `validate_on_program_cache_miss`; the `cluster_axis` docstring/code contradiction; the unused `ccl/moe_utils.hpp` include. See [Flags](#flags).
