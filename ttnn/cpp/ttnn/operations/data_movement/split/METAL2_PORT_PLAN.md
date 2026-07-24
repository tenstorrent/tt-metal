# Port Plan — `split` (native TILE device op)

Port plan for `ttnn/cpp/ttnn/operations/data_movement/split`, ported from the legacy
`ProgramDescriptor` API (`ProgramDescriptorFactoryConcept`) to Metal 2.0 (`MetalV2FactoryConcept`).
Written during the inventory and planning steps; committed alongside the port for review.

Scope: only the native TILE device operation (`SplitDeviceOperation` → `SplitProgramFactory`) is in
scope. The host-facing `ttnn::split` dispatcher (`split.cpp`) also has a slice-fallback backend
(`split_with_slice_impl` → `ttnn::slice`); that path runs a different op and is **not** part of this port.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` — `SplitProgramFactory::create_descriptor(...)` returns
  `tt::tt_metal::ProgramDescriptor` (`split_program_factory.hpp:14`, `split_program_factory.cpp:74`).
- Variants: single (`SplitProgramFactory`); `program_factory_t = std::variant<SplitProgramFactory>`
  (`split_device_operation.hpp:19`). No `select_program_factory` method — single-variant auto-dispatch.
- Custom `compute_program_hash`: none — the op uses the default reflection-based hash.

*(The Metal 2.0 factory concept the port targets — `MetalV2FactoryConcept` — was chosen during the
audit. Carried forward in the [TTNN ProgramFactory](#ttnn-programfactory) section below.)*

### Kernels
Two `KernelDescriptor`s, both sources owned by the op directory.

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | config |
|---|---|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/reader_tm_tile_layout_split_two_chunks.cpp` | `all_cores` = `CoreRange({0,0},{num_cores_r-1, num_cores_c-1})` | `{z/num_cores_z, per_core_tiles_x, per_core_tiles_y, z_stride_read, y_stride_read}` then `TensorAccessorArgs(*in0_buffer).append_to(...)` at slot 5 | none | slot0 `reader_core_id` (page index), slot1 `in0_buffer` (`Buffer*`), slot2 literal `0` (dead `split_last_dim`) | none | none | `ReaderConfigDescriptor{}` (reader default: RISCV_1 / NOC_0 / DM_DEDICATED_NOC) |
| writer | `device/kernels/dataflow/writer_split_n_chunks_tile.cpp` | `all_cores` (same range) | `{per_core_tiles_x, per_core_tiles_y, z/num_cores_z, z_stride_write, y_stride_write}` then `TensorAccessorArgs(*output_buffers[0]).append_to(...)` at slot 5 | none | slot0 `writer_core_id` (page index), slot1 `output_buffers[chunk_id]` (`Buffer*`, differs per chunk group) | none | none | `WriterConfigDescriptor{}` (writer default: RISCV_0 / NOC_1 / DM_DEDICATED_NOC) |

CTA value → kernel-side name mapping (positions are authoritative from the host emission order; the
`_y`/`_x` swap between factory locals and kernel names is a pre-existing legacy quirk, preserved verbatim):

- **reader**: slot0 `z/num_cores_z`→`z`; slot1 `per_core_tiles_x`→`out_num_tiles_per_tensor_y`;
  slot2 `per_core_tiles_y`→`out_num_tiles_per_tensor_x`; slot3 `z_stride_read`→`z_stride`;
  slot4 `y_stride_read`→`y_stride`. (`reader_tm_tile_layout_split_two_chunks.cpp:25-29`)
- **writer**: slot0 `per_core_tiles_x`→`out_num_tiles_per_tensor_y`; slot1 `per_core_tiles_y`→`out_num_tiles_per_tensor_x`;
  slot2 `z/num_cores_z`→`z`; slot3 `z_stride_write`→`z_stride`; slot4 `y_stride_write`→`y_stride`.
  (`writer_split_n_chunks_tile.cpp:26-30`)

All writer CTA values are **chunk-independent** (equal N-way split), so every per-chunk writer instance
carries identical CTAs.

### CBs
| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| 0 (`src0`) | `2 * single_tile_size` (double buffer, `num_input_tiles = 2`) | `all_cores` | `cb_data_format = datatype_to_dataformat_converter(input.dtype())` | `single_tile_size = tile_size(cb_data_format)` | not set (default 32×32) |

Single plain double-buffered CB (`split_program_factory.cpp:162-172`). Not a GlobalCircularBuffer;
no `address_offset`, no aliasing, no multi-element `format_descriptors`.

Census per node: the reader instance is the locked **producer** (`dfb_in0.reserve_back`/`push_back`,
`reader_...cpp:57,65`); the writer instance is the locked **consumer** (`dfb_out.wait_front`/`pop_front`,
`writer_...cpp:48,52`). Both kernels run on every node and bind DFB id `0` → clean **1 producer + 1
consumer per node**.

### Semaphores
none.

### Tensor accessors
| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| reader `TensorAccessor(in0_tensor_args, in0_tensor_addr)` (`reader_...cpp:37`) | input `in0` (`tensor_args.input`) | slot 1 (`in0_buffer`, `Buffer*`) |
| writer `TensorAccessor(out_tensor_args, out_tensor_addr)` (`writer_...cpp:37`) | output chunk (`tensor_return_value[chunk_id]`) | slot 1 (`output_buffers[chunk_id]`, `Buffer*`) |

Both are **Case 1** (kernel reads exclusively through the `TensorAccessor` page-access API — `page_id`
into `noc.async_read`/`async_write`; no raw base-pointer arithmetic). Both are 2-arg accessors (no 3rd
page-size argument to drop).

### Work split
Not `split_work_to_cores`; a custom two-axis grid split driven by
`get_max_cores_divisible_by_tiles_per_core_tiles` (`split_program_factory.cpp:108-137`):
- `num_cores_z = z` (parallelize dim-1 across core rows, the X grid axis).
- `(num_cores_x, per_core_tiles_x) = get_max_cores_divisible_by_tiles_per_core_tiles(num_tiles_dim_2, num_cores_x_limit / num_cores_z)` — dim-2 (height tiles) across X cores.
- `tiles_per_chunk = num_tiles_dim_3 / num_chunks`; `(num_cores_per_chunk, per_core_tiles_y) = get_max_cores_divisible_by_tiles_per_core_tiles(tiles_per_chunk, num_cores_y_limit / num_chunks)` — dim-3 (width tiles) across Y cores, grouped by chunk.
- Grid: `num_cores_r = num_cores_x * num_cores_z` (X axis, "rows"); `num_cores_c = num_chunks * num_cores_per_chunk` (Y axis, "cols").
- `all_cores = CoreRange({0,0}, {num_cores_r-1, num_cores_c-1})`.
- **Chunk → core-group mapping** (`setup_runtime`, `split_program_factory.cpp:55-66`): core `{x=id_r, y=id_c}`, with `id_c = chunk_id * num_cores_per_chunk + id_c_inner`. So chunk `k` owns the column band `y ∈ [k*num_cores_per_chunk, (k+1)*num_cores_per_chunk - 1]` across all rows `x ∈ [0, num_cores_r-1]`. Each core writes exactly one output chunk, selected today by the per-core `output_buffers[chunk_id]` `Buffer*` RTA.

### Cross-op kernels
none — both kernel sources live in the op's own directory; every `#include` is `api/*` (LLK/HAL) plus
`tensix_types.h` / `stdint.h`.

### Flags
Non-gating anomalies the inventory noticed (routed to the port report; the port does **not** act on them
beyond what the mechanical swap requires):
- **Dead RTA `split_last_dim`** — reader reads `get_arg_val<uint32_t>(2)` into `split_last_dim`
  (`reader_...cpp:22`) but never uses it; the factory always passes literal `0` (`...cpp:63`). Dropped
  on both sides by the port (no name assigned) — see [Dropped Plumbing](#dropped-plumbing).
- **Vestigial multi-tensor scaffolding in the reader** — `out_num_tensors = 1` (`reader_...cpp:32`)
  with the always-once `for (out_tensor_id …)` loop and `tensor_stride`/`tensor_stride_cum`
  (lines 43-44, 76) contributing a constant `0`. Inert; left as-is (kernel logic is not the port's to
  rewrite).
- **Stale kernel filename** — `reader_tm_tile_layout_split_two_chunks.cpp` is the generalized N-chunk
  reader (chunk count arrives via CTAs / grid layout), not 2-chunk-specific. Filename only; not renamed
  (out of scope).

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `MetalV2FactoryConcept`.
- **Custom `compute_program_hash`**: none — nothing to delete.
- **Pybind `create_descriptor`**: none — `split_nanobind.cpp` binds only the free function `ttnn::split`
  via `bind_function`; there is no pybind exposure of the factory entry point, so no pybind line is
  removed by this port.
- **Device-op-class edits forced by the port**: none. The device-op struct only names the factory type
  (`std::variant<SplitProgramFactory>`) and has no `select_program_factory`; the concept is detected from
  the factory's method. Renaming `create_descriptor` → `create_program_artifacts` (return type
  `ProgramDescriptor` → `ttnn::device_operation::ProgramArtifacts`) is entirely within the factory files.
- **Implementation notes**: see *Planned Spec Shape* — the one non-mechanical element is the N distinct
  output bindings.

## Planned Spec Shape

`N = num_splits` (fixed per compiled program; folded into the program hash via
`operation_attributes.num_splits`).

- **KernelSpecs**: `1 + N`.
  - `READER` — one KernelSpec (source `reader_tm_tile_layout_split_two_chunks.cpp`), placed on `all_cores`.
  - `WRITER_k` for `k = 0..N-1` — `N` KernelSpecs of the **same** source
    (`writer_split_n_chunks_tile.cpp`), each placed on chunk `k`'s disjoint column band, each bound to
    output chunk `k`. Identical CTAs across all `k`.
- **DataflowBufferSpecs**: 1 — `SRC0` (`entry_size = single_tile_size`, `num_entries = 2`,
  `data_format_metadata = cb_data_format`, `tile_format_metadata` unset). Bound PRODUCER by `READER`
  and CONSUMER by every `WRITER_k`.
- **SemaphoreSpecs**: none.
- **TensorParameters**: `1 + N`.
  - `IN0` — input, `spec = input_tensor.tensor_spec()`, bound on `READER` (accessor `in0`).
  - `OUT_k` for `k = 0..N-1` — output chunk `k`, `spec = tensor_return_value[k].tensor_spec()`, bound on
    `WRITER_k` (accessor `out`).
- **WorkUnitSpecs**: `N` — `WU_k = {kernels = {READER, WRITER_k}, target_nodes = chunk_k_band}`. The
  reader belongs to all `N` work units, so its effective node set is the union (`all_cores`); each
  `WRITER_k` belongs to exactly one.
- **Op-owned tensors**: none.

Accessor names (kernel-local, chosen to read naturally against the existing kernel variables):
- DFB `SRC0` → accessor `src0` on both kernels (`DataflowBuffer dfb_in0(dfb::src0)` in the reader,
  `DataflowBuffer dfb_out(dfb::src0)` in the writer).
- input tensor → accessor `in0` (`TensorAccessor(tensor::in0)`); output tensor → accessor `out`
  (`TensorAccessor(tensor::out)`).

## Preserved Multiplicity

The N-writer expansion is the disjoint-node work-split shape (each node sees exactly one writer instance,
so each is a legal single-role binding — no `allow_instance_multi_binding` flag). Note this is driven by
the **distinct per-chunk output binding**, not by per-group CTAs (writer CTAs are identical across
chunks), so there is **no** CTA→RTA demotion.

| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (endpoint role each binds) |
|---|---|---|---|
| writer (single descriptor over `all_cores`) | `WRITER_0 … WRITER_{N-1}` of `writer_split_n_chunks_tile.cpp` | `WU_0 … WU_{N-1}` (disjoint chunk column bands) | `SRC0` — each `WRITER_k` binds **CONSUMER** |
| reader (single descriptor over `all_cores`) | `READER` (single) | in all of `WU_0 … WU_{N-1}` (⇒ node set = `all_cores`) | `SRC0` — binds **PRODUCER** |

Per-node census: exactly 1 PRODUCER (`READER`) + 1 CONSUMER (that node's chunk's `WRITER_k`). The N
CONSUMER bindings on the one `SRC0` endpoint are legal per the `dataflow_buffer_spec.hpp` invariant
(lines 41-50): non-overlapping node coverage ✓, same kernel kind (all data-movement) ✓, identical
binding-site parameters (default `access_pattern = STRIDED`, `num_threads = 1`) ✓.

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader RTA slot 1 (`split_program_factory.cpp:63`) | `in0_buffer` (`Buffer*`) | `TensorBinding(IN0)` → kernel `TensorAccessor(tensor::in0)` |
| reader CTA slot 5 (`split_program_factory.cpp:147`) | `TensorAccessorArgs(*in0_buffer).append_to(...)`; kernel `TensorAccessorArgs<5>()` (`reader_...cpp:30`) | supplied by the binding token |
| reader RTA slot 2 (`split_program_factory.cpp:63`) | literal `0` fed to dead `split_last_dim` (`reader_...cpp:22`) | **dropped entirely** (dead on both sides; not named) |
| reader CTAs slots 0-4 (positional) | `get_compile_time_arg_val(0..4)` | named CTAs `z`, `out_num_tiles_per_tensor_y`, `out_num_tiles_per_tensor_x`, `z_stride`, `y_stride` |
| reader RTA slot 0 (`split_program_factory.cpp:63`) | `reader_core_id` positional | named RTA `in0_tensor_tile_id` (stays — it is a page index, not an address) |
| writer RTA slot 1 (`split_program_factory.cpp:65`) | `output_buffers[chunk_id]` (`Buffer*`) | `TensorBinding(OUT_k)` → kernel `TensorAccessor(tensor::out)` |
| writer CTA slot 5 (`split_program_factory.cpp:158`) | `TensorAccessorArgs(*output_buffers[0]).append_to(...)`; kernel `TensorAccessorArgs<5>()` (`writer_...cpp:32`) | supplied by the binding token |
| writer CTAs slots 0-4 (positional) | `get_compile_time_arg_val(0..4)` | named CTAs `out_num_tiles_per_tensor_y`, `out_num_tiles_per_tensor_x`, `z`, `z_stride`, `y_stride` |
| writer RTA slot 0 (`split_program_factory.cpp:65`) | `writer_core_id` positional | named RTA `out_tensor_tile_id` (stays — page index) |
| both kernels, CB id `0` (`reader_...cpp:33`, `writer_...cpp:34`) | `constexpr uint32_t dfb_id_in0 = 0;` / `dfb_id_out0 = 0;` → `DataflowBuffer(0)` | `DFBBinding(SRC0)` → `DataflowBuffer(dfb::src0)` |

No page-size 3rd-argument CTA/RTA to drop (both accessors already 2-arg). No semaphore-ID RTAs.

## Applied Patterns

- [Anti-pattern: Demoting per-group CTA to RTA](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md) — the *disjoint-node work-split* half of it: the single legacy writer descriptor over `all_cores` becomes N same-source `KernelSpec`s in N `WorkUnitSpec`s over disjoint node sets, all binding the same `SRC0` DFB. Here the multiplicity is per-chunk output binding, not per-group CTA, so no dimension is demoted to an RTA.
- [DFB endpoint invariant / multiple bindings on one endpoint](../../../../../../tt_metal/api/tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp) (lines 41-50) — the N CONSUMER bindings on `SRC0` are legal because their node coverage is non-overlapping, same kernel kind, identical binding-site parameters.
- No self-loop, no 1P+1C two-toucher assignment, no `allow_instance_multi_binding`, no aliasing, no conditional bindings, no varargs.

## Deferred / Flagged

- **Multi-output-per-core-subset binding shape** — the audit/brief flagged this as the one
  non-mechanical part and asked to confirm the shape before building. Resolved as N writer `KernelSpec`s
  over disjoint chunk node sets (above), grounded in the `dataflow_buffer_spec.hpp` endpoint invariant.
  The brief's phrasing "within the one writer KernelSpec" is imprecise for the current API: a
  `TensorBinding` is a property of a `KernelSpec` (and each writer uses a single `tensor::out` accessor),
  so binding N distinct outputs across disjoint core bands requires N `KernelSpec`s, not one. Same intent,
  correct expression. **Confirmed with the invoker before construction** — shape A (1 shared reader +
  N same-source writers over disjoint core bands) selected over the more conservative shape F (N readers
  + N writers, each pair in its own work unit); A is leaner and matches the catalog's disjoint-node
  work-split example. The two conceivable single-KernelSpec shapes are not expressible today (the
  `tensor::` binding tokens are per-KernelSpec compile-time types with no runtime index and no per-node
  scope).
- No structural issue uncovered that the audit missed; no feature gate that failed to fire.
