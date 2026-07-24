# Port Plan — `data_movement/sharded_partial/sharded_to_interleaved_partial`

Port plan for `sharded_to_interleaved_partial`, ported from the legacy `descriptor`
(`create_descriptor` returning `ProgramDescriptor`) API to Metal 2.0 (`MetalV2FactoryConcept`).
Written during the inventory and planning steps; committed alongside the port for review.

> **Outcome of this plan: the port CAPITULATES at the planning step on a scope limit.**
> Every kernel entry point this factory binds lives **outside** the op directory, and the
> orchestration constraint forbids editing or forking shared kernels outside the op directory.
> The recipe's atomic-unit rule (factory + every bound kernel entry point flip together — there is
> no half-Metal-2.0 factory) therefore cannot be satisfied within the permitted writeable surface.
> Full reasoning in `METAL2_PORT_REPORT.md`. The inventory below is complete; the spec plan is
> recorded up to the point the blocker makes construction impossible.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` (`descriptor`) — single
  `ShardedToInterleavedPartialProgramFactory::create_descriptor(...)` returning
  `tt::tt_metal::ProgramDescriptor` (`device/sharded_to_interleaved_partial_program_factory.hpp:15-18`).
- Variants: single (one `program_factory_t` variant;
  `device/sharded_to_interleaved_partial_device_operation.hpp:20`).
- Custom `compute_program_hash`: none — already default reflection-based hash (audit cross-checked;
  grep of op tree finds no `compute_program_hash`).
- Custom `override_runtime_arguments` / `get_dynamic_runtime_args`: none (audit cross-checked).
- Pybind `create_descriptor`: none — nanobind binds only the free function
  `&ttnn::sharded_to_interleaved_partial` (`sharded_to_interleaved_partial_nanobind.cpp`).

*(Target Metal 2.0 concept chosen during audit: `MetalV2FactoryConcept` — carried forward below.)*

### Kernels

The factory `CreateKernel`s (via `KernelDescriptor` / `FILE_PATH`) up to three kernels per program.
**The op owns none of them — all four selectable sources are file-path borrows OUTSIDE the op directory.**

| unique_id | source (all external) | core_ranges | CTAs (positional) | RTAs (positional) | defines | config |
|---|---|---|---|---|---|---|
| reader | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp` | `used_cores` | `[0]=src0_cb_index` (`c_0`) | `[0]=num_units_per_shard` | none | `ReaderConfigDescriptor{}` |
| writer (TILE, live path) | `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded_blocks_interleaved_start_id.cpp` | `used_cores` | `[0]=out_cb_index`; then `TensorAccessorArgs(*dst_buffer)` appended | `[0]=dst_buffer (Buffer*)`, `[1]=block_height_tiles`, `[2]=block_width_tiles`, `[3]=unpadded_block_height_tiles`, `[4]=unpadded_block_width_tiles`, `[5]=output_width_tiles`, `[6]=block_num_tiles`, `[7]=start_id_offset`, `[8]=start_id_base` | none | `WriterConfigDescriptor{}` |
| writer (RM, validate-blocked/dead path) | `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp` | `used_cores` | `[0]=out_cb_index`; then `TensorAccessorArgs(*dst_buffer)` appended | `[0]=dst_buffer (Buffer*)`, `[1]=num_units_per_row` (dead — never read), `[2]=shard_height`, `[3]=shard_width`, `[4]=padded_shard_width`, `[5]=curr_idx_w`, `[6]=curr_idx_h` | none | `WriterConfigDescriptor{}` |
| compute (only when `convert_df`) | `ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp` | `used_cores` | `[0]=num_units_per_shard` | none | none | `ComputeConfigDescriptor{}` |

Kernel-source selection axes:
- **Writer source** selected on `input.layout()`: TILE → `writer_unary_sharded_blocks_interleaved_start_id.cpp`;
  else → `writer_unary_stick_layout_...`. `validate_on_program_cache_miss` hard-asserts TILE
  (`device_operation.cpp:24`), so the RM writer path is **dead** at runtime (audit Misc anomaly) but
  is still a selectable source and would flip with the factory.
- **Compute kernel** conditionally present on `convert_df` (`input dtype != output dtype`).

Kernel-side legacy patterns requiring Metal 2.0 conversion (confirmed by reading the sources):
- reader: positional `get_compile_time_arg_val(0)` (CB id), positional `get_arg_val<uint32_t>(0)`.
- writer (TILE): positional CTA `get_compile_time_arg_val(0)`, `TensorAccessorArgs<1>()` plumbing,
  buffer-address RTA `dst_addr = get_arg_val<uint32_t>(0)`, positional RTAs `[1]..[8]`,
  `get_tile_size(cb_id_out)` free function.
- compute (`eltwise_copy.cpp`): positional CTA, `cb_*` FIFO-sync free functions (standard compute API).

All four are Device 2.0-compliant (object-based `DataflowBuffer` / `Noc` / `TensorAccessor`) per the
audit, but **none is on Metal 2.0** — each still reads positional CTAs/RTAs and (writers) uses
`TensorAccessorArgs` + buffer-address RTA.

### CBs (`CBDescriptor`)
Built by `push_s2i_partial_cb_pair` (`_program_factory.cpp:25-43`):
- `c_0` (`src0_cb_index`): `total_size = num_input_units * input_page_size`, `page_size = input_page_size`,
  `data_format = input_cb_data_format`, `core_ranges = used_cores`, **`buffer = src_buffer`**
  (borrowed-memory CB — bound to `input.buffer()` for cache-hit rebinding, `:140-147`).
  Not a GlobalCircularBuffer (`.global_circular_buffer` unset, no `remote_*` idiom — audit confirmed).
- `c_16` (`out_cb_index`, only when `convert_df`): `total_size = num_input_units * output_page_size`,
  `page_size = output_page_size`, `data_format = output_cb_data_format`, `core_ranges = used_cores`,
  `buffer = nullptr` (`:149-160`).

### Semaphores
None — the op uses no semaphores of any kind (audit confirmed).

### Tensor accessors
- input tensor (sharded): no `TensorAccessor` — data resides in the borrowed L1 CB `c_0`; reader only
  advances the FIFO (`dfb.push_back`). Would port as a **borrowed-memory DFB** (`borrowed_from` the
  input `TensorParameter`).
- output/cache tensor (interleaved): **Case 1** via `TensorAccessor`. Address surfaces as `Buffer*` in
  writer RTA `[0]` (`writer_rt.push_back(dst_buffer)`, `:243`/`:294`); kernel builds
  `TensorAccessor(dst_args, dst_addr)`. Would become a `TensorParameter` / `TensorBinding`.

### Work split
No `split_work_to_cores` — per-core work derives from the shard grid (`used_cores`) directly.
Reader RTA is identical on every core; writer RTAs are per-core (loop at `:211-309`). No multi-
`KernelDescriptor` work-split multiplicity.

### Cross-op kernels (out-of-directory)
**All four selectable kernel sources are outside the op directory** (see Kernels table). Co-borrower
counts (audit Team-only): reader ~18, TILE writer ~3, compute ~4; RM writer shared within the sharded
family. Per the audit, each shared kernel's Metal 2.0 rewrite must be adopted by every co-borrower in
one change (or the source forked).

### Flags
- Dead RM writer path (validate forces TILE); dead RTA `[1]` (`num_units_per_row`) in that path.
- `is_l1_aligned` hardcoded `true` (`:55`) — dead-branch simplification in the (dead) RM path.
- Both are pre-existing, out of scope (device-op / factory body outside the Metal 2.0 transformation).
  Routed to the report as findings; **not** touched.

## TTNN ProgramFactory
- **Concept (inherited from audit):** `MetalV2FactoryConcept`.
- **Custom `compute_program_hash`:** none to delete.
- **Implementation notes:** N/A — port does not reach construction (see blocker).

## Planned Spec Shape
Recorded for completeness; **not constructed** (blocked before construction). Had the kernels been
in-directory or touchable, the intended shape was:
- **KernelSpecs:** reader (1), writer (1, source selected by layout), compute (1, conditional on `convert_df`).
- **DataflowBufferSpecs:** `INPUT` (`c_0`, `borrowed_from` the input `TensorParameter`); `OUT`
  (`c_16`, only when `convert_df`).
- **SemaphoreSpecs:** none.
- **TensorParameters:** `INPUT` (sharded input, backs the borrowed DFB), `OUTPUT` (cache/output, Case 1).
- **WorkUnitSpecs:** one (`convert_df == false`: reader+writer) or one with compute added
  (`convert_df == true`).
- **Op-owned tensors:** none.

## Preserved Multiplicity
None — no multi-`KernelDescriptor` work-split in legacy.

## Dropped Plumbing (planned, not applied)
- Writer CTA `[0]` (`out_cb_index`, magic CB index) → `DFBBinding`.
- Writer `TensorAccessorArgs(*dst_buffer).append_to(...)` CTA (`:177`) → binding mechanism.
- Writer RTA `[0]` (`dst_buffer` `Buffer*` address) → `TensorBinding` (Case 1).
- Reader CTA `[0]` (`src0_cb_index`) → `DFBBinding` (borrowed DFB).
- All positional CTAs/RTAs → named args.

## Applied Patterns (planned, not applied)
- Borrowed-memory DFB: `INPUT` (`c_0`) `borrowed_from` the input `TensorParameter`.
- Conditional/optional binding + kernel gating: `OUT` (`c_16`) and the compute kernel, gated on `convert_df`.

## Deferred / Flagged — **BLOCKER (new finding at planning step)**

**Structural blocker not resolvable within scope: the factory owns zero in-directory kernels.**

The Metal 2.0 recipe's atomic-unit rule is explicit: a factory that speaks Metal 2.0 bindings can only
launch kernels whose entry points read those bindings, so the factory and **every** kernel-source entry
point it can select flip together — *there is no half-Metal-2.0 factory*. A Metal 2.0 factory emits only
named args (`dfb::`/`tensor::`/`args::`) via the framework-generated headers; a legacy kernel reading
positional `get_compile_time_arg_val(0)` / `get_arg_val<uint32_t>(N)` `static_assert`s at JIT.

All four selectable kernel sources are outside the op directory and none is on Metal 2.0. Converting the
factory therefore requires converting kernels outside the op directory. The orchestration constraint for
this port (which OVERRIDES the recipe) states: *"Only modify files UNDER this directory … do NOT fork or
edit shared kernels outside this op directory."* That removes **both** routes the recipe would otherwise
allow for cross-op kernels (in-place edit with caution, or fork).

Result: the port cannot proceed to construction without either (a) editing/forking out-of-directory
kernels (forbidden by the orchestration constraint) or (b) shipping a Metal 2.0 factory bound to legacy
kernels (a broken, non-building/JIT-failing half-port the recipe forbids). This is the recipe's
most-common stop signal — *"if you find yourself reaching past the op's own directory to make kernel
changes, that's the signal"* — and a scope-limit **CAPITULATION** (success-tier). See
`METAL2_PORT_REPORT.md` → Handoff points.
