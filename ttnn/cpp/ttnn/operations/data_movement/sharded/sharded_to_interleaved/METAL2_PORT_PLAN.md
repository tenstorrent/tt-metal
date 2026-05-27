# Port Plan — `sharded_to_interleaved`

Port plan for `data_movement/sharded/sharded_to_interleaved`, ported from `ProgramDescriptor` to Metal 2.0.

> **Status**: planning. The port has not started. This document captures the structural decisions and stop signals discovered during planning, for user review before construction begins.

## Legacy Inventory

### Factory shape

- Concept: `ProgramDescriptorFactoryConcept` (single factory `ShardedToInterleavedProgramFactory::create_descriptor` returning `ProgramDescriptor`).
- Variants: single device-operation; two layout paths inside (`Layout::TILE` vs `Layout::ROW_MAJOR`) selecting different writer kernels and different RTA layouts.
- Custom `compute_program_hash`: **no** — uses default reflection-based hash.

### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | config |
|---|---|---|---|---|---|---|---|---|
| reader | `eltwise/unary/.../reader_unary_sharded.cpp` (cross-op) | `used_cores` | `[src0_cb_index]` | n/a | `[num_units_per_shard]` | none | none | `ReaderConfigDescriptor` |
| writer (TILE) | `data_movement/sharded/device/kernels/dataflow/writer_unary_sharded_blocks_interleaved_start_id.cpp` (in-family cross-op) | `used_cores` | `[out_cb_index, TensorAccessorArgs(*dst_buffer)…]` | n/a | `[dst_buffer*, num_units_per_shard_height, num_units_per_shard_width, shard_height, shard_width, num_units_offset, num_units_per_shard, curr_idx_h+w, starting_idx_h]` (9 args; arg 0 is `Buffer*` via `BufferBinding`) | none | none | `WriterConfigDescriptor` |
| writer (ROW_MAJOR) | `data_movement/sharded/device/kernels/dataflow/writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp` (in-family cross-op) | `used_cores` | `[out_cb_index, TensorAccessorArgs(*dst_buffer)…]` | n/a | `[dst_buffer*, num_units_per_row, shard_height, shard_width, padded_shard_width, curr_idx_w, curr_idx_h]` (7 args; arg 0 is `Buffer*` via `BufferBinding`) | none | none | `WriterConfigDescriptor` |
| compute (optional, on `convert_df`) | `ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp` (shared kernel pool) | `used_cores` | `[num_units_per_shard]` | n/a | none | none | none | `ComputeConfigDescriptor` |

### CBs

| index | total_size | core_ranges | data_format | page_size | tile (if set) | bound buffer |
|---|---|---|---|---|---|---|
| `CBIndex::c_0` (`src0_cb_index`) | `num_input_units * input_page_size` | `used_cores` | `input_cb_data_format` | `input_page_size` (= `align(input_unit_size, src_buffer->alignment())`) | not set (default) | `src_buffer` (borrowed-memory CB) |
| `CBIndex::c_16` (`out_cb_index`, conditional on `convert_df`) | `num_input_units * output_page_size` | `used_cores` | `output_cb_data_format` | `output_page_size` (= `align(output_unit_size, dst_buffer->alignment())`) | not set (default) | `nullptr` (regular allocated CB) |

### Semaphores

none

### Tensor accessors

| host site | originating Tensor | RTA slot (host) | CTA offset (kernel) |
|---|---|---|---|
| writer factory at `sharded_to_interleaved_program_factory.cpp:176` (`TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args)`) | `output_tensor` (interleaved DRAM) | arg 0 is `dst_buffer*` (BufferBinding) — kernel-side address injected by framework | starts at CTA index 1 (after `out_cb_index`) |

### Work split

- Driver: `corerange_to_cores(all_cores, std::nullopt, rm_orientation)` — derives the linear core list from the input tensor's shard grid; not a generic `split_work_to_cores`.
- `num_cores = all_cores.num_cores()`.
- `num_cores_unpadded` derived from sharding strategy and tensor extents; may be less than `num_cores` when the grid is over-allocated.
- `used_cores = select_from_corerangeset(all_cores, 0, num_cores_unpadded - 1, rm_orientation)` when `num_cores_unpadded < num_cores`; else `all_cores`.

### Cross-op kernels

All three dataflow kernel sources live outside the op's directory:

- `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp` — owner: `eltwise/unary`. Consumers (also reads it): `eltwise/unary` itself; `data_movement/sharded/{interleaved_to_sharded,sharded_to_interleaved}`; `data_movement/{tilize,untilize,untilize_with_unpadding}` (sharded paths). **Broadly-shared.**
- `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded_blocks_interleaved_start_id.cpp` — owner: `data_movement/sharded` (this op's family). Consumers: this op only (TILE path).
- `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp` — owner: `data_movement/sharded` (this op's family). Consumers: this op only (ROW_MAJOR path).

Plus the compute kernel: `ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp` — shared kernel pool.

### Flags

- No reference port: a `grep -rln "create_program_spec" ttnn/cpp` returns no hits. **This is the first TTNN op being ported to Metal 2.0 with `ProgramSpecFactoryConcept`.** No worked example to crib from.
- The TTNN device-op framework adapter supports `ProgramSpecFactoryConcept` (see `ttnn/api/ttnn/operation_concepts.hpp:90` and `ttnn/api/ttnn/metal2_artifacts.hpp`) — the integration surface exists.

## Planned Spec Shape

- **KernelSpecs**: one per legacy `KernelDescriptor`. Three total when `convert_df = true`, two when false (no compute kernel).
- **DataflowBufferSpecs**: one per legacy `CBDescriptor`. The `src0_cb` becomes a borrowed-memory DFB (`borrowed_from = INPUT`). The `out_cb` (only present when `convert_df`) is a regular DFB.
- **SemaphoreSpecs**: none.
- **TensorParameters**: two — `INPUT` (the sharded source tensor) and `OUTPUT` (the interleaved destination tensor).
- **WorkUnitSpecs**: one — `main`, targeting `used_cores`, containing all kernels.

### Names

```cpp
constexpr const char* READER  = "reader";
constexpr const char* WRITER  = "writer";
constexpr const char* COMPUTE = "compute";  // only when convert_df
constexpr const char* SRC_DFB = "src_dfb";
constexpr const char* OUT_DFB = "out_dfb";  // only when convert_df
constexpr const char* INPUT   = "input";
constexpr const char* OUTPUT  = "output";
```

## Preserved Multiplicity

none — no work-split multiplicity in legacy (no per-group CTAs).

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader CTA[0] = `src0_cb_index` | magic CB index in CTA | `DFBBinding{SRC_DFB, "shard", CONSUMER}` on reader `KernelSpec`; kernel reads `dfb::shard` |
| reader RTA[0] = `num_units_per_shard` | positional RTA | `named_runtime_args = {"num_units_per_shard"}`; kernel uses `get_arg(args::num_units_per_shard)` |
| writer CTA[0] = `out_cb_index` | magic CB index in CTA | `DFBBinding` to whichever DFB sources the writer's input (alias of `SRC_DFB` when `!convert_df`, `OUT_DFB` when `convert_df`); kernel reads `dfb::out` |
| writer CTA[1..N] = `TensorAccessorArgs(*dst_buffer)` plumbing | positional TensorAccessor CTAs | `TensorBinding{OUTPUT, "out"}` on writer `KernelSpec`; kernel uses `TensorAccessor(ta::out)` |
| writer RTA[0] = `dst_buffer*` (BufferBinding) | `Buffer*` framework slot | absorbed into the `TensorBinding` for OUTPUT (the tensor binding auto-injects the per-enqueue base address) |
| writer RTA[1..N] = the 7-9 named scalars | positional RTAs | `named_runtime_args = {"num_units_per_row" or "num_units_per_shard_height", ...}`; one per legacy RTA, names taken from the legacy variable names in the host code |

The two writer kernels (TILE vs ROW_MAJOR) have different argument sets — port preserves both as separate `KernelSpec`s of separate sources, just like the legacy.

## Applied Patterns

- **Conditional / optional DFB binding** — `OUT_DFB` (and the compute kernel) is present only when `convert_df`. The host conditionally constructs `OUT_DFB` and adds the compute `KernelSpec`. The reader and writer's DFB bindings switch source: when `convert_df`, the writer consumes `OUT_DFB`; when not, the writer consumes `SRC_DFB` directly. The catalog's [Pattern: Conditional / optional DFB bindings] applies.
- **Borrowed-memory DFB** — `SRC_DFB` is bound to the input buffer via `borrowed_from = INPUT`. The legacy `cb.buffer = src_buffer` becomes the new `DataflowBufferSpec::borrowed_from` field.

## Deferred / Flagged

### Cross-op kernel coordination (recipe → catalog: Modifying a shared dataflow kernel)

Three of the four kernels this factory instantiates live in other op directories:

1. **`reader_unary_sharded.cpp` (`eltwise/unary`)** — broadly-shared. Modifying in place requires co-migrating every consumer in the same PR. Per the catalog's "Caution: Modifying a shared dataflow kernel," the appropriate path during the bulk-port window is **fork with `_metal2` suffix**: create `reader_unary_sharded_metal2.cpp` next to the legacy file, rewrite it for Metal 2.0 named bindings, and reference the forked path from this op's factory. The legacy file stays in place for unmigrated consumers.
2. **`writer_unary_sharded_blocks_interleaved_start_id.cpp` (`data_movement/sharded`, in-family)** — consumed only by `sharded_to_interleaved`. Could be modified in place safely OR forked. Recommendation: modify in place (single-consumer, no fork sprawl).
3. **`writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp` (`data_movement/sharded`, in-family)** — same as above. Modify in place.
4. **`eltwise_copy.cpp` (shared kernel pool)** — compute kernel, broadly-shared. Compute kernels don't carry tensor bindings (TRISC builds can't compile the NoC includes), and the eltwise_copy kernel reads only the DFB CTA index — replaceable via `dfb::name`'s implicit cast to `uint32_t`. Fork might be needed to update the kernel-arg retrieval syntax. **Caution: this kernel is used by many ops.** Recommendation: fork with `_metal2` suffix.

### Question for the user before construction proceeds

This is the first TTNN op being ported. The cross-op kernel touches above coordinate with all four GREEN ops the user approved (`sharded_to_interleaved`, `interleaved_to_sharded`, `tilize`, `untilize`). Specifically:

- `reader_unary_sharded.cpp` is shared by all 4 GREEN ops + others. Forking it once with `_metal2` suffix gives a single Metal 2.0 reader that all 4 ports can use.
- `writer_unary_sharded.cpp` is shared by `interleaved_to_sharded`, `tilize`, `untilize` sharded paths. Same fork-once-use-many shape.
- `eltwise_copy.cpp` and `tilize.cpp` / `untilize.cpp` compute kernels are similarly cross-cutting.

**Proposed sequence**:

1. Port `sharded_to_interleaved` first (this plan). Fork the broadly-shared kernels and migrate the in-family writer kernels in place. Validate end-to-end (build + tests) — establishes the reference shape for all subsequent ports.
2. Port `interleaved_to_sharded` second. Reuses the `_metal2`-forked `reader_unary_sharded.cpp` and one new in-family kernel.
3. Port `tilize` (5 factories) and `untilize` (8 factories) third and fourth. Heavy use of the already-forked sharded reader/writer.

Each op's port stays a self-contained PR, but the kernel forks are placed once and reused.

**Stop signal**: I (the porter) want explicit user confirmation that this sequence and the fork strategy are acceptable before I write any port code. Specifically:

1. Confirm the fork path (`_metal2` suffix) over in-place modification for the broadly-shared kernels (`reader_unary_sharded.cpp`, `eltwise_copy.cpp`).
2. Confirm port order (start with `sharded_to_interleaved`, then iterate to others as proposed).
3. Confirm I should pause for a build/test cycle after `sharded_to_interleaved` before continuing to the next op (rather than producing all 4 ports back-to-back).

If user confirms: I write the construction (factory + forked kernels), surface any new friction in `METAL2_PORT_REPORT.md`, and stop for the build/test cycle. If user redirects (e.g., wants in-place modification): I update this plan and proceed accordingly.
