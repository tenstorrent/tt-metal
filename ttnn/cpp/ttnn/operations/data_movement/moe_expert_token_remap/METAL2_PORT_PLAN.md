# Port Plan — `data_movement/moe_expert_token_remap`

Port plan for `moe_expert_token_remap`, ported from the legacy `ProgramDescriptor` (`descriptor`) API to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

> **Outcome: the port stopped at the planning step — `CAPITULATED`.** Planning uncovered a
> structural blocker the audit did not catch: the factory produces a **per-mesh-coordinate
> specialized program** (a device-position value baked into a reader compile-time argument),
> and the target concept `MetalV2FactoryConcept` cannot express per-coordinate variation — its
> adapter builds one artifact and stamps it identically across the whole mesh. See
> [Deferred / Flagged](#deferred--flagged) for the full analysis. The legacy inventory below is
> complete and correct; the spec-shape planning is recorded for reference but was **not** built,
> because construction would produce silently-wrong numerics on every device except mesh
> coordinate `{0, 0}`.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` (`descriptor`). `Multicore::create_descriptor(...)` returns a `tt::tt_metal::ProgramDescriptor` ([moe_expert_token_remap_program_factory.cpp:18-22](device/moe_expert_token_remap_program_factory.cpp#L18-L22)).
- Variants: single (`Multicore`); `program_factory_t = std::variant<Multicore>` ([moe_expert_token_remap_device_operation.hpp:50](device/moe_expert_token_remap_device_operation.hpp#L50)).
- Custom `compute_program_hash`: none — already default reflection-based hash; `validate_on_program_cache_hit` is empty ([moe_expert_token_remap_device_operation.hpp:58-59](device/moe_expert_token_remap_device_operation.hpp#L58-L59)).
- **Per-coordinate specialization**: `create_descriptor` takes a 4th parameter, `const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate`, and folds the device's linearized mesh position into a reader compile-time argument (`flat_mesh_idx`, [factory:147-164](device/moe_expert_token_remap_program_factory.cpp#L147-L164)). Under the legacy descriptor adapter this makes the op **per-coordinate specialized** (a distinct compiled reader per device). This is the blocker — see [Deferred / Flagged](#deferred--flagged).

*(The Metal 2.0 factory concept the port targets was chosen during the audit — `MetalV2FactoryConcept`. This plan carries it forward in the [TTNN ProgramFactory](#ttnn-programfactory) section, where the disagreement with that choice is recorded.)*

### Kernels

Both kernels are data-movement kernels placed over the **same** `total_cores` grid (the full compute-with-storage grid, `CoreRange({0,0},{grid.x-1, grid.y-1})`). No compute kernel.

| unique_id | source | core_ranges | CTAs (positional) | RTAs | CRTAs | defines | config |
|---|---|---|---|---|---|---|---|
| reader | `ttnn/cpp/ttnn/operations/ccl/all_to_all_combine/device/kernels/dataflow/reader_all_to_all_combine.cpp` **(cross-op)** | `total_cores` | 14 scalar CTAs + 3 `TensorAccessorArgs` blocks (see below) | `{mapping_buffer (Buffer*), metadata_buffer (Buffer*), topk_buffer (Buffer*), page_idx_start, page_idx_end}` ([factory:230-231](device/moe_expert_token_remap_program_factory.cpp#L230-L231)) | none | none | `ReaderConfigDescriptor{}` |
| writer | `.../moe_expert_token_remap/device/kernels/dataflow/writer_moe_expert_token_remap.cpp` (op-owned) | `total_cores` | 11 scalar CTAs + 2 `TensorAccessorArgs` blocks | `{output_mapping_buffer (Buffer*), page_idx_start, page_idx_end, output_reduced_buffer (Buffer*), reduction_idx_start}` ([factory:236-237](device/moe_expert_token_remap_program_factory.cpp#L236-L237)) | none | none | `WriterConfigDescriptor{}` |

**Reader positional CTAs** (host emission order, [factory:155-172](device/moe_expert_token_remap_program_factory.cpp#L155-L172)):

| slot | value | kernel name | port disposition |
|---|---|---|---|
| 0 | `mapping_tensor_cb_id` (c_0) | `mapping_cb_id` | -> DFBBinding (self-loop) |
| 1 | `local_experts_cb_id` (c_1) | `local_experts_cb_id` | -> DFBBinding (PRODUCER) |
| 2 | `metadata_cb_id` (c_2) | `metadata_cb_id` | -> DFBBinding (PRODUCER) |
| 3 | `topk_cb_id` (c_3) | `data_cb_id` | -> DFBBinding (PRODUCER) |
| 4 | `experts_per_device` | `num_local_experts` | -> named CTA |
| 5 | `batch_size` | `batch_size` | -> named CTA |
| 6 | `seq_size` | `seq_size` | -> named CTA |
| 7 | `experts` | `num_mapping_pages` | -> named CTA |
| **8** | **`flat_mesh_idx`** | **`linearized_mesh_coord`** | **BLOCKER — per-coordinate value; no Metal 2.0 home** |
| 9 | `topk_page_size_bytes` | `data_size_bytes` | -> named CTA |
| 10 | `selected_experts_k` | `selected_experts_k` | -> named CTA |
| 11 | `mapping_page_size_bytes` | `mapping_page_size_bytes` | -> named CTA |
| 12 | `metadata_page_size_bytes` | `metadata_page_size_bytes` | -> named CTA |
| 13 | `local_reduce` (=`true`) | `locally_reduced` | -> named CTA |
| 14+ | `TensorAccessorArgs(topk)`, `(mapping)`, `(metadata)` | `data_args`, `mapping_args`, `metadata_args` | -> TensorBindings |

Slot 8 (`flat_mesh_idx`) is used kernel-side as a **non-type template parameter**: `detail::get_device_expert_indices<linearized_mesh_coord, ...>(...)`, which tests `mapping_ptr[DeviceIdx] == 1u` ([reader:100-101](../../ccl/all_to_all_combine/device/kernels/dataflow/reader_all_to_all_combine.cpp#L100-L101), [reader:47](../../ccl/all_to_all_combine/device/kernels/dataflow/reader_all_to_all_combine.cpp#L47)) — i.e. it selects *which experts are local to this device* from its position in the mesh. Its value differs for every device.

**Writer positional CTAs** (host emission order, [factory:184-198](device/moe_expert_token_remap_program_factory.cpp#L184-L198)):

| slot | value | kernel name | port disposition |
|---|---|---|---|
| 0 | `local_experts_cb_id` (c_1) | `local_experts_dfb_id` | -> DFBBinding (CONSUMER) |
| 1 | `metadata_cb_id` (c_2) | `metadata_dfb_id` | -> DFBBinding (CONSUMER) |
| 2 | `topk_cb_id` (c_3) | `data_dfb_id` | -> DFBBinding (CONSUMER) |
| 3 | `output_mapping_cb_id` (c_4) | `output_mapping_dfb_id` | -> DFBBinding (self-loop) |
| 4 | `output_reduced_cb_id` (c_5) | `output_reduced_dfb_id` | -> DFBBinding (self-loop) |
| 5 | `selected_experts_k` | `selected_experts_k` | -> named CTA |
| 6 | `experts_per_device` | `num_local_experts` | -> named CTA |
| 7 | `output_mapping_page_size_bytes` | `output_mapping_page_size_bytes` | -> named CTA |
| 8 | `output_datum_size_bytes` | `datum_size_bytes` | -> named CTA |
| 9 | `output_reduced_page_size_bytes` | `output_reduced_page_size_bytes` | -> named CTA |
| 10 | `reduction_size` | `reduction_size` | -> named CTA |
| 11+ | `TensorAccessorArgs(output_mapping)`, `(output_reduced)` | `output_mapping_args`, `output_reduced_args` | -> TensorBindings |

The writer carries **no** per-coordinate value: its CTAs are mesh-invariant, and its per-node work (`page_idx_start/end`, `reduction_idx_start`) is already an RTA.

### CBs

Single config (no sharding variants). Census re-derived independently from the kernel bodies; it agrees with the audit/brief.

| index | name | total_size | num_entries x entry_size | core_ranges | data_format | touchers on a node | disposition |
|---|---|---|---|---|---|---|---|
| c_0 | mapping | `aligned_mapping_page_size_bytes` | 1 x `aligned_mapping_page_size_bytes` | `total_cores` | mapping dtype (uint16) | reader only — producer + raw self-read ([reader:92-101](../../ccl/all_to_all_combine/device/kernels/dataflow/reader_all_to_all_combine.cpp#L92-L101)) | **self-loop** |
| c_1 | local_experts | `align(experts_per_device*2, l1)` | 1 x same | `total_cores` | uint16 | reader produces ([reader:92-102](../../ccl/all_to_all_combine/device/kernels/dataflow/reader_all_to_all_combine.cpp#L92-L102)), writer consumes ([writer:60-117](device/kernels/dataflow/writer_moe_expert_token_remap.cpp#L60-L117)) | **legal 1:1** |
| c_2 | metadata | `1 x aligned_metadata_page_size_bytes` | 1 x `aligned_metadata_page_size_bytes` | `total_cores` | metadata dtype (uint16) | reader produces ([reader:104-138](../../ccl/all_to_all_combine/device/kernels/dataflow/reader_all_to_all_combine.cpp#L104-L138)), writer consumes ([writer:65-115](device/kernels/dataflow/writer_moe_expert_token_remap.cpp#L65-L115)) | **legal 1:1** |
| c_3 | topk/data | `2 x aligned_topk_page_size_bytes` | 2 x `aligned_topk_page_size_bytes` | `total_cores` | topk dtype (bfloat16) | reader produces ([reader:121-131](../../ccl/all_to_all_combine/device/kernels/dataflow/reader_all_to_all_combine.cpp#L121-L131)), writer consumes ([writer:76-98](device/kernels/dataflow/writer_moe_expert_token_remap.cpp#L76-L98)) | **legal 1:1** |
| c_4 | output_mapping | `output_mapping_page_size_bytes` | 1 x same | `total_cores` | output_mapping dtype (bfloat16) | writer only — staging scratch ([writer:49-51](device/kernels/dataflow/writer_moe_expert_token_remap.cpp#L49-L51)) | **self-loop** |
| c_5 | output_reduced | `output_reduced_page_size_bytes` | 1 x same | `total_cores` | output_reduced dtype (uint16) | writer only — staging scratch ([writer:54-56](device/kernels/dataflow/writer_moe_expert_token_remap.cpp#L54-L56)) | **self-loop** |

No GlobalCircularBuffer, no aliased CBs, no dead CB, no multi-binding.

### Semaphores
none — the op uses no semaphores; reader/writer coordinate purely via CB FIFOs.

### Tensor accessors
All five are **Case 1** (base fed into a `TensorAccessor`; all access through the accessor). All delivered today via the `Buffer*`-binding form (a `Buffer*` in `emplace_runtime_args`), so there is no `buffer()->address()` fold anywhere.

| host site (file:line) | originating Tensor | RTA slot (host) | kernel use |
|---|---|---|---|
| [factory:230-231](device/moe_expert_token_remap_program_factory.cpp#L230-L231) | `mapping_tensor` (input) | reader slot 0 | `TensorAccessor(mapping_args, mapping_tensor_addr)` ([reader:82](../../ccl/all_to_all_combine/device/kernels/dataflow/reader_all_to_all_combine.cpp#L82)) |
| [factory:230-231](device/moe_expert_token_remap_program_factory.cpp#L230-L231) | `metadata_tensor` (input) | reader slot 1 | `TensorAccessor(metadata_args, metadata_tensor_addr)` ([reader:81](../../ccl/all_to_all_combine/device/kernels/dataflow/reader_all_to_all_combine.cpp#L81)) |
| [factory:230-231](device/moe_expert_token_remap_program_factory.cpp#L230-L231) | `topk_tensor` (input, "data") | reader slot 2 | `TensorAccessor(data_args, data_tensor_addr)` ([reader:83](../../ccl/all_to_all_combine/device/kernels/dataflow/reader_all_to_all_combine.cpp#L83)) |
| [factory:236-237](device/moe_expert_token_remap_program_factory.cpp#L236-L237) | `output_mapping_tensor` (output 0) | writer slot 0 | `TensorAccessor(output_mapping_args, output_mapping_base_addr)` ([writer:38](device/kernels/dataflow/writer_moe_expert_token_remap.cpp#L38)) |
| [factory:236-237](device/moe_expert_token_remap_program_factory.cpp#L236-L237) | `output_reduced_tensor` (output 1) | writer slot 3 | `TensorAccessor(output_reduced_args, output_reduced_base_addr)` ([writer:39](device/kernels/dataflow/writer_moe_expert_token_remap.cpp#L39)) |

### Work split
- Driver: `split_work_to_cores_even_multiples(grid, num_metadata_pages, reduction_size)` ([factory:210-213](device/moe_expert_token_remap_program_factory.cpp#L210-L213)), where `num_metadata_pages = metadata_tensor.buffer()->num_pages()` (= batch*seq).
- Returns `(core_page_increments, all_cores)`; the loop assigns each utilized core a `[page_idx_start, page_idx_end)` slice and a `reduction_idx_start = page_idx_start / reduction_size` ([factory:225-241](device/moe_expert_token_remap_program_factory.cpp#L225-L241)).
- No multi-`KernelDescriptor` work split — a single reader + single writer descriptor over `all_cores`. Per-group CTA multiplicity: none.

### Cross-op kernels
- `ttnn/cpp/ttnn/operations/ccl/all_to_all_combine/device/kernels/dataflow/reader_all_to_all_combine.cpp` — owned by op family `ccl/all_to_all_combine`; instantiated by file path here **and** by `all_to_all_combine` itself ([all_to_all_combine_program_factory.cpp:250](../../ccl/all_to_all_combine/device/all_to_all_combine_program_factory.cpp#L250)). A Metal 2.0 rewrite of this kernel cannot land in place without breaking `all_to_all_combine` (which is not being ported), so it would require the **fork-with-`_metal2`-suffix** path. This is *not* the reason the port stops; it is ordinary (if notable) port work that the capitulation makes moot.

### Flags
- **Per-coordinate program specialization via `mesh_dispatch_coordinate`** — the blocker. Recorded in [Deferred / Flagged](#deferred--flagged).
- The writer kernel is already structurally Device 2.0 (`DataflowBuffer` wrappers, `Noc`), and the reader kernel uses the `CircularBuffer` Device 2.0 wrapper. Neither uses a legacy CB-index free function or legacy addr-gen. (Device 2.0 is not the blocker.)
- No unreferenced kernel files in the op directory.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `MetalV2FactoryConcept`.
- **Custom `compute_program_hash`**: none (already default).
- **Disagreement with the audit's choice (surfaced, not overridden).** The recipe requires: *"If you find yourself disagreeing with the audit's choice, stop and surface the disagreement to the invoker — do not unilaterally override."* Planning found that `MetalV2FactoryConcept` is **not** a viable target for this op, because the op requires per-mesh-coordinate program specialization that the concept's single-program adapter cannot express (full analysis in [Deferred / Flagged](#deferred--flagged)). The correct target is the not-yet-implemented `MeshWorkloadSpecFactoryConcept` (or the op stays on its current per-coord `descriptor` adapter). This plan does **not** unilaterally substitute a different concept; it stops and records the disagreement for the invoker and the audit owners.

## Planned Spec Shape

*Recorded for reference only. This spec was **not** constructed — the port stopped at planning (see [Deferred / Flagged](#deferred--flagged)). Everything below except the `flat_mesh_idx` CTA maps cleanly; the entry documents that the blocker is isolated to one argument, not a pervasive mismatch.*

- **KernelSpecs**: 2 — `READER` (`reader_all_to_all_combine_metal2.cpp`, a fork), `WRITER` (`writer_moe_expert_token_remap.cpp`). Both DM; `hw_config` = `create_reader_datamovement_config(arch)` / `create_writer_datamovement_config(arch)` respectively.
- **DataflowBufferSpecs**: 6, one per CB (c_0..c_5), sizes as in the [CBs](#cbs) table; no aliasing, no borrowed memory.
- **SemaphoreSpecs**: none.
- **TensorParameters**: 5 — `mapping_tensor`, `metadata_tensor`, `topk_tensor` (inputs), `output_mapping_tensor`, `output_reduced_tensor` (outputs).
- **WorkUnitSpecs**: 1 — `{READER, WRITER}` over `all_cores`.
- **Op-owned tensors**: none — outputs are ordinary `create_output_tensors` device tensors.

Resolved DM configs (for the hardware-config diff, had construction proceeded): both kernels use `ReaderConfigDescriptor{}` / `WriterConfigDescriptor{}` in legacy, i.e. the plain reader/writer defaults `(RISCV_1, NOC_0, DEDICATED)` and `(RISCV_0, NOC_1, DEDICATED)` — so the arch-agnostic TTNN helpers reproduce them byte-for-byte on Gen1.

## Preserved Multiplicity

none — no work-split multiplicity in legacy (single reader + single writer `KernelDescriptor` over `all_cores`).

## Dropped Plumbing

*What the port would remove (recorded for reference). The one entry that has **no** replacement is the reason for the capitulation.*

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader RTA slots 0-2 ([factory:230-231](device/moe_expert_token_remap_program_factory.cpp#L230-L231)) | `Buffer*` (mapping / metadata / topk) | `TensorBinding` (Case 1) |
| writer RTA slots 0, 3 ([factory:236-237](device/moe_expert_token_remap_program_factory.cpp#L236-L237)) | `Buffer*` (output_mapping / output_reduced) | `TensorBinding` (Case 1) |
| reader CTAs 0-3, writer CTAs 0-4 | CB-index CTAs | `DFBBinding` |
| reader `TensorAccessorArgs` ([factory:170-172](device/moe_expert_token_remap_program_factory.cpp#L170-L172)) | host `append_to` + kernel `TensorAccessorArgs<N>()` | binding mechanism end-to-end |
| writer `TensorAccessorArgs` ([factory:197-198](device/moe_expert_token_remap_program_factory.cpp#L197-L198)) | host `append_to` + kernel `TensorAccessorArgs<N>()` | binding mechanism end-to-end |
| reader CTAs 4-7, 9-13; writer CTAs 5-10 | positional scalar CTAs | named CTAs |
| reader RTAs 3-4; writer RTAs 1, 2, 4 | positional scalar RTAs (page indices) | named RTAs |
| **reader CTA slot 8 `flat_mesh_idx`** ([factory:151](device/moe_expert_token_remap_program_factory.cpp#L151), [factory:164](device/moe_expert_token_remap_program_factory.cpp#L164)) | **per-coordinate value baked into a CTA via the `mesh_dispatch_coordinate` adapter** | **NONE — no Metal 2.0 primitive carries a per-mesh-coordinate value on `MetalV2FactoryConcept`** |

## Applied Patterns

Patterns that *would* have applied (recorded for reference; not exercised because the port stopped):
- Sync-free and single-ended CBs -> self-loop DFB: c_0 (mapping, reader self-loop), c_4 / c_5 (output staging, writer self-loop).
- Caution: Modifying a shared dataflow kernel: the reader would need the **fork** path (`reader_all_to_all_combine_metal2.cpp`) because `all_to_all_combine` co-borrows it.

## Deferred / Flagged

### BLOCKER — per-mesh-coordinate program specialization is not expressible on `MetalV2FactoryConcept`

**What the op does.** `Multicore::create_descriptor` takes `const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate` and computes
`flat_mesh_idx = coord[0] * mesh_view.num_cols() + coord[1]` ([factory:150-151](device/moe_expert_token_remap_program_factory.cpp#L150-L151)), then bakes it into the reader as compile-time arg slot 8 ([factory:164](device/moe_expert_token_remap_program_factory.cpp#L164)). Kernel-side it is a **non-type template parameter** selecting which experts belong to *this* device: `get_device_expert_indices<linearized_mesh_coord, ...>` -> `if (mapping_ptr[DeviceIdx] == 1u)` ([reader:100-101](../../ccl/all_to_all_combine/device/kernels/dataflow/reader_all_to_all_combine.cpp#L100-L101), [reader:47](../../ccl/all_to_all_combine/device/kernels/dataflow/reader_all_to_all_combine.cpp#L47)). Its value is **different on every device in the mesh** (0..N-1).

**Why the legacy op is correct.** The framework's `descriptor` adapter detects that `create_descriptor` accepts `mesh_dispatch_coordinate` (`create_descriptor_uses_mesh_dispatch_coordinate()` ([mesh_device_operation_adapter.hpp:425-444](../../../../../../ttnn/api/ttnn/mesh_device_operation_adapter.hpp#L425-L444))) and therefore iterates **per coordinate**, building a *distinct* program for each device with that device's `flat_mesh_idx`:

```cpp
if constexpr (create_descriptor_uses_mesh_dispatch_coordinate()) {
    for (const auto& coord : tensor_coords.coords()) {
        build_and_add_program(MeshCoordinateRange(coord), std::optional<MeshCoordinate>(coord));
    }
}
```

([mesh_device_operation_adapter.hpp:595-599](../../../../../../ttnn/api/ttnn/mesh_device_operation_adapter.hpp#L595-L599)).

**Why Metal 2.0 cannot reproduce it.** `MetalV2FactoryConcept::create_program_artifacts` has a **fixed 3-argument signature** (`attributes`, `tensor_args`, `tensor_return_value`) — **no mesh-coordinate parameter** ([operation_concepts.hpp:90-92](../../../../../../ttnn/api/ttnn/operation_concepts.hpp#L90-L92)). Its adapter calls the factory **once** and stamps the identical `spec` + `run_params` onto every coordinate range:

```cpp
auto artifacts = MetalV2Factory::create_program_artifacts(attrs, tensor_args, tensor_return_value);
...
for (const auto& range : tensor_coords.ranges()) {
    auto program = MakeProgramFromSpec(*mesh_device, artifacts.spec);   // same spec
    SetProgramRunArgs(program, artifacts.run_params);                   // same run_params
    mesh_workload.add_program(range, std::move(program));
}
```

([mesh_device_operation_adapter.hpp:884-910](../../../../../../ttnn/api/ttnn/mesh_device_operation_adapter.hpp#L884-L910)). This is the documented feasibility-gate BLOCKED case: *"Multi-program / per-coord variation. The op's programs genuinely differ across mesh coordinates (CCL-style). The single-program adapter stamps one spec everywhere."*

**No supported workaround exists.** Each avenue was checked and rejected against the recipe's stop-signals:
- **RTA / CRTA.** The same `run_params` is stamped to every coordinate range; RTAs are keyed by *node* (a core within a device), not by mesh coordinate, and the cache-hit path refreshes only tensor args. A value cannot be made to vary per mesh coordinate through any run-arg.
- **Op-owned / new tensor carrying the index.** `flat_mesh_idx` is a device *identity*, not tensor data the op computes; manufacturing a per-device index tensor is a functional redesign, and the kernel consumes the value as a `constexpr` template parameter — converting that to runtime data is kernel-logic surgery outside the whitelist. Both are exactly the "clever workaround" the recipe says to stop on.
- **A per-coord Metal 2.0 concept.** `MeshWorkloadSpecFactoryConcept` (the intended home for per-coord-varying ops) is explicitly *not yet implemented* ([metal_v2_artifacts.hpp:15-19](../../../../../../ttnn/api/ttnn/metal_v2_artifacts.hpp#L15-L19)).

**Consequence if ported anyway.** `create_program_artifacts` would compute one `flat_mesh_idx` (from the `{0,0}` fallback -> 0). Every device would select expert set 0. On the op's only real use (a multi-device mesh; the sole test is a 2x4 = 8-device T3000 job), 7 of 8 devices would produce silently-wrong numerics. The single-n150 bench available here **cannot run** that test, so the bug would not be caught locally.

**Decision.** Capitulate on the `Multicore` factory (the op's only factory). This is a framework capability gap, recorded as a Handoff-points entry in `METAL2_PORT_REPORT.md` with `Outcome: CAPITULATED`.
