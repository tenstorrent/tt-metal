# Port Plan — experimental/quasar/rotate

Port plan for `ttnn/cpp/ttnn/operations/experimental/quasar/rotate` (the quasar copy of `pool/rotate`), ported from the `ProgramDescriptor` API to Metal 2.0 **together with `experimental/quasar/grid_sample`** (shared kernels).
Written during the inventory and planning steps; committed alongside the port for review.

**Scope (user instruction):** quasar copies only; nothing under `pool/` changes.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` — `NearestProgramFactory` and `BilinearProgramFactory` are nested structs in the `program_factory_t` variant (`device/rotate_device_operation.hpp:32-46`), each with `create_descriptor`.
- Variants: Nearest {interleaved, height-sharded, ND-sharded}; Bilinear {interleaved, any input/output sharded}.
- Custom `compute_program_hash`: none (default reflection hash over `operation_attributes_t`) — left intact.

### Variant: Nearest (`device/rotate_nearest_program_factory.cpp`, legacy lines)

| unique_id | source | CTAs (positional) | RTAs | opt | config |
|---|---|---|---|---|---|
| reader | `kernels/dataflow/reader_rotate_nearest_interleaved.cpp` | 0 output_cb, 1 output_cb_page_size, 2 input_batch (read, unused), 3 H, 4 W, 5 channels, 6 num_cb_pages (read, unused), 7 fill_cb, 8 stick nbytes, 9 fill_is_zero, 10 burst; `TensorAccessorArgs(input)` | input `Buffer*`, num_sticks, start, cos, sin, cx, cy, fill bits | O2 | `ReaderConfigDescriptor{}` |
| writer | `kernels/dataflow/writer_rotate_nearest_interleaved.cpp` | 0 output_cb, 1 aligned output stick, 2 num_cb_pages (read, unused), 3 burst, 4 cb page size; `TensorAccessorArgs(output)` | output `Buffer*`, num_sticks, start | O2 | `WriterConfigDescriptor{}` |

| CB | size | format | buffer |
|---|---|---|---|
| fill | 1 × page | output | — |
| input (sharded only) | shard sticks × aligned input stick | input | input buffer — **no kernel references it** |
| output | (shard sticks, or num_cb_pages × 2) × page | output | output buffer when sharded |

### Variant: Bilinear (`device/rotate_bilinear_program_factory.cpp`)

| unique_id | source | CTAs | RTAs | opt | config |
|---|---|---|---|---|---|
| reader | `kernels/dataflow/reader_rotate_bilinear_interleaved.cpp` | 0 input_cb, 1 scalar_cb, 2 stick nbytes, 3 batch, 4 H, 5 W, 6 fill_cb, 7 channels, 8 fill_is_zero, 9 element_size; `TensorAccessorArgs(input)` | input `Buffer*`, num_sticks, start, cos, sin, cx, cy, fill bits | O2 | `ReaderConfigDescriptor{}` |
| compute ×1 (sharded) / ×2 (interleaved groups) | `pool/generic/.../compute_pool_2d.cpp` (**borrowed**) | as grid_sample's, with split = 0, force_4 = 0, CTA 3 per group | none | O3 | `ComputeConfigDescriptor{HiFi4, fp32 off, approx off}` |
| writer (interleaved) | `experimental/quasar/grid_sample/.../writer_grid_sample_interleaved.cpp` (**borrowed**) | 0 output_cb, 1 stick size, 2 out_ntiles_c; `TensorAccessorArgs(output)` | output `Buffer*`, num_sticks, start | O2 | `WriterConfigDescriptor{}` |

| CB | size | format | face geometry / tile | buffer |
|---|---|---|---|---|
| fill | stick nbytes | input | — | — |
| input | 2 × in_ntiles_c·TILE_HW·elem | input | {4, 2} | — |
| scalar | 2 × tile_size | input | {1, 2} | — |
| output | pages × FACE_WIDTH·elem | output | {1, 1 or 2}; tile {1,16} when single partial | output buffer when any sharded |

### Semaphores
none

### Shared kernels
- `experimental/quasar/grid_sample/.../writer_grid_sample_interleaved.cpp` and `grid_sample_reader_common.hpp` — borrowed from quasar grid_sample, converted in place by the joint port.
- `pool/generic/.../compute_pool_2d.cpp` — borrowed (main tree). Rebound to the grid_sample-local Metal 2.0 copy; nothing in `pool/` changes.

### Flags
- none.

## TTNN ProgramFactory
- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept` (both).
- **Custom hash**: none.

## Planned Spec Shape
- **Nearest:** TensorParameters `input`, `output`; DFBs `fill` (self-loop on reader), `out` (reader PRODUCER, writer CONSUMER; borrowed_from `output` when sharded); KernelSpecs `reader`, `writer`; one WorkUnit on all_cores. Legacy sharded input CB dropped (dead).
- **Bilinear:** TensorParameters `input`, `output`; DFBs `fill` (self-loop), `in`, `scalar` (reader → compute), `out` (compute → writer; or compute self-loop + borrowed_from `output` when sharded); face geometry → `tile_format_metadata` (as grid_sample). KernelSpecs `reader`, `compute` / `compute_group_1` / `compute_group_2`, `writer` (interleaved). WorkUnits: sharded one; interleaved one per core group.

## Preserved Multiplicity
| legacy | KernelSpecs | WorkUnits | shared DFBs |
|---|---|---|---|
| bilinear compute ×2 (CTA 3 differs) | `compute_group_1`, `compute_group_2` | `…_group_1`, `…_group_2` (disjoint) | `in`, `scalar` CONSUMER; `out` PRODUCER, each on its own nodes |

## Dropped Plumbing
| legacy | form | replacement |
|---|---|---|
| reader RTA 0 (both factories) | `input_tensor.buffer()` | `TensorBinding(input)` |
| writer RTA 0 (both) | `output_tensor.buffer()` | `TensorBinding(output)` |
| CB-index CTAs (nearest reader 0, 7; writer 0; bilinear reader 0, 1, 6; writer 0; compute 7–11, 13, 38) | CB indices | `DFBBinding`s |
| compute CTAs 17–37 | placeholders not read by the kernel | dropped |
| all `TensorAccessorArgs` appends + kernel chains | accessor plumbing | binding mechanism |
| nearest sharded input CB | dead CB (no kernel references it) | dropped |
| remaining positional args | positional | named (kernel local names; `cb` → `dfb` in `num_cb_pages`, `cb_page_size`) |

## Applied Patterns
- Self-loop (fill on both readers; bilinear `out` when sharded). Dead-CB drop (nearest sharded input). Preserved per-group compute KernelSpecs. Pass DFB tokens to helpers (`fill_four_val<dfb::scalar>`). Shared kernels with quasar grid_sample (joint port).

## Deferred / Flagged
- none beyond the audit's misc anomalies.
