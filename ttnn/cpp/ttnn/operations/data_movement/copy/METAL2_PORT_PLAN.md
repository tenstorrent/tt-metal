# Port Plan — `data_movement/copy` (`CopyDeviceOperation`)

Port plan for `ttnn/cpp/ttnn/operations/data_movement/copy`, ported from the
`ProgramDescriptor` (`descriptor`) concept to Metal 2.0 (`ProgramSpecFactoryConcept`).
Written during the inventory and planning steps; committed alongside the port for review.

One DeviceOperation (`CopyDeviceOperation`) with three factories in its
`program_factory_t` variant: `SameMemoryConfig`, `DefaultRowMajor`, `DefaultTilized`.
Per the recipe's atomic-unit rule, each factory is a complete sub-port; they are ported
in this order: **DefaultRowMajor → DefaultTilized → SameMemoryConfig**. A half-ported op
builds and runs — the framework dispatches per factory.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` (`descriptor`), all three factories define `create_descriptor()` returning `ProgramDescriptor` (`device/copy_device_operation.hpp:24-42`).
- Variants: three factories (not multi-variant within a factory).
- Custom `compute_program_hash`: **none** — default reflection-based hash (confirmed: no override anywhere in op).
- No `override_runtime_arguments`, no `get_dynamic_runtime_args`, no pybound `create_descriptor` (nanobind binds only `copy`/`assign` free functions).

*(Metal 2.0 target concept chosen by the audit: `ProgramSpecFactoryConcept` — carried forward below.)*

---

### Factory: `DefaultRowMajor`  (`copy_default_row_major_program_factory.cpp`)

Selected when input is ROW_MAJOR and cannot use the specialized (same-mem-config) factory.

#### Kernels
| unique_id | source | core_ranges | CTAs (positional) | RTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|
| reader | `copy/device/kernels/redistribute_pages_row_major_reader.cpp` (OWN) | all_cores | `[0]=c_0, [1]=c_1, [2]=num_output_pages_in_row(DEAD), [3]=num_input_pages_in_row, [4]=elements_per_output_page, [5]=bytes_per_element, [6]=elements_per_input_page, [7]=elements_per_tensor_row, [8]=input_subblock_size_bytes, [9]=output_subblock_size_bytes` + `TensorAccessorArgs(input.buffer())` @ `:136` | `[0]=input.buffer()` (Buffer*), `[1]=start_row_id`, `[2]=num_rows_to_process` | none | (unset → O2, DM) | ReaderConfigDescriptor{} (default reader triple) |
| writer | `copy/device/kernels/redistribute_pages_row_major_writer.cpp` (OWN) | all_cores | `[0]=c_1, [1]=num_output_pages_in_row, [2]=elements_per_output_page, [3]=bytes_per_element, [4]=elements_per_tensor_row, [5]=output_subblock_size_bytes` + `TensorAccessorArgs(output.buffer())` @ `:148` | `[0]=output.buffer()` (Buffer*), `[1]=start_row_id`, `[2]=num_rows_to_process` | none | (unset → O2, DM) | WriterConfigDescriptor{} (default writer triple) |

#### CBs
| index | total_size | core_ranges | data_format | page_size | tile |
|---|---|---|---|---|---|
| c_0 (`input_pages_cb_index`) | `input_page_size` | all_cores | `datatype_to_dataformat_converter(input.dtype())` | `input_page_size` | (unset) |
| c_1 (`output_page_cb_index`) | `2 * aligned_output_page_size` | all_cores | `datatype_to_dataformat_converter(output.dtype())` | `aligned_output_page_size` | (unset) |

- **c_0 endpoints:** reader only — `reserve_back`/`get_write_ptr`/`push_back`/`wait_front`/`pop_front` (`redistribute_pages_row_major_reader.cpp:42,61,183-185`). Used as an L1 scratchpad. **One toucher → SELF-LOOP** (reader bound both PRODUCER and CONSUMER).
- **c_1 endpoints:** reader PRODUCER (`reserve_back`/`get_write_ptr`/`push_back`), writer CONSUMER (`wait_front`/`noc.async_write(dfb_in1,…)`/`pop_front`). Legal 1:1.

#### Semaphores
none

#### Tensor accessors
| host site | originating Tensor | RTA slot (host) |
|---|---|---|
| reader `TensorAccessor(src_args, src_addr)` `redistribute_pages_row_major_reader.cpp:38` | input | reader RTA[0] `input.buffer()` — Case 1 |
| writer `TensorAccessor(dst_args, dst_addr)` `redistribute_pages_row_major_writer.cpp:31` | output | writer RTA[0] `output.buffer()` — Case 1 |

#### Work split
- Driver: `split_work_to_cores(compute_with_storage_grid_size, total_logical_rows)`
- num_cores/all_cores/core_group_1/core_group_2/num_rows_per_core_group_1/num_rows_per_core_group_2
- Per-core RTAs vary `start_row_id` (accumulated) and `num_rows_to_process` (group_1 vs group_2 count). No per-group CTA → single KernelSpec each, single WorkUnitSpec over all_cores.

#### Shared kernels
none — both `redistribute_pages_*` kernels are bound only by this factory (census run: only `copy_default_row_major_program_factory.cpp` binds them). Not lent, not borrowed. Convert in place.

#### Flags
- Dead CTA: reader CTA `[2]` `num_output_pages_in_row` declared (`reader.cpp:24`) but never used. Carry forward as-is (ops-team prune candidate — reported, not fixed).

---

### Factory: `DefaultTilized`  (`copy_default_tilized_program_factory.cpp`)

Selected for TILE layout when the specialized factory can't be used.

#### Kernels
| unique_id | source | core_ranges | CTAs (positional) | RTAs | opt_level | config |
|---|---|---|---|---|---|---|
| reader | `eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` (BORROWED — fork EXISTS `_metal2`) | all_cores | `TensorAccessorArgs(input.buffer())` only | `[0]=input.buffer()`, `[1]=num_tiles_to_process`, `[2]=start_tile_id` | O2 (DM) | ReaderConfigDescriptor{} |
| writer | `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` (BORROWED — fork EXISTS `_metal2`) | all_cores | `[0]=output_page_cb_index` + `TensorAccessorArgs(output.buffer())` | `[0]=output.buffer()`, `[1]=num_tiles_to_process`, `[2]=start_tile_id` | O2 (DM) | WriterConfigDescriptor{} |
| compute (only if `convert_df`) | `data_movement/sharded/device/kernels/compute/eltwise_copy.cpp` (IN-FAMILY — NO fork yet) | all_cores | `{}` (empty) | `[0]=num_tiles_to_process` | unset → **O3** (compute) | ComputeConfigDescriptor{} |

#### CBs
| index | total_size | data_format | page_size | tile |
|---|---|---|---|---|
| c_0 (`input_pages_cb_index`) | `2 * aligned_input_page_size` | input df | `aligned_input_page_size` | (unset) |
| c_16 (`output_page_cb_index`, only if convert_df) | `2 * aligned_output_page_size` | output df | `aligned_output_page_size` | (unset) |

- No-convert: c_0 reader→writer (1:1). Convert: c_0 reader→compute (1:1), c_16 compute→writer (1:1).
- `output_page_cb_index` = c_0 when no convert (writer reads c_0), c_16 when convert (writer reads c_16). Writer CTA slot 0 carries this index → becomes writer's DFB binding.

#### Work split
- `split_work_to_cores(grid, total_tiles)`. Per-core RTAs: num_tiles_to_process, start_tile_id (+ compute num_tiles). Compute is a single KernelSpec over all_cores (num_tiles is an RTA, **not** a per-group CTA) → no multiplicity.

#### Shared kernels
- reader/writer: BORROWED from `eltwise/unary` — `_metal2` forks **exist** (`reader_unary_interleaved_start_id_metal2.cpp`, `writer_unary_interleaved_start_id_metal2.cpp`): bind them, adopt their binding vocabulary. Do not re-fork.
- compute: `data_movement/sharded/device/kernels/compute/eltwise_copy.cpp` — IN-FAMILY, no fork yet. **Create fork** beside original + pointer comment. Remaining consumers (sunset list): `interleaved_to_sharded`, `interleaved_to_sharded_partial`.

---

### Factory: `SameMemoryConfig`  (`copy_same_memory_config_program_factory.cpp`)

Selected when input/output mem configs equal and CB fits L1. Runtime-selects reader/writer source across (tilized × sharded) axes; compute only on convert_dtype (TILE only).

#### Kernels (runtime source selection)
| unique_id | source (selected) | core_ranges | notes |
|---|---|---|---|
| reader | tilized→`copy/device/kernels/reader_unary_start_id.cpp` (OWN); RM+sharded→`copy/device/kernels/reader_unary_stick_start_id.cpp` (OWN); RM+interleaved→`ttnn/cpp/ttnn/kernel/dataflow/reader_unary_stick_layout_interleaved_start_id.cpp` (SHARED POOL — NO fork yet) | all_cores | see CTA/RTA below |
| writer | tilized→`copy/device/kernels/writer_unary_start_id.cpp` (OWN); RM+sharded→`copy/device/kernels/writer_unary_stick_start_id.cpp` (OWN); RM+interleaved→`ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp` (SHARED POOL — NO fork yet) | all_cores | |
| compute_g1 / compute_g2 (only if convert_dtype, TILE only) | `ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp` (SHARED POOL — NO fork yet) | core_group_1 / core_group_2 | **preserved multiplicity** — per-group CTA `num_units_per_core_group_{1,2}` |

- **tilized:** reader CTA = `TensorAccessorArgs(src)`; writer CTA = `[0]=output_cb_index` + `TensorAccessorArgs(dst)`. reader RTA `{src_buffer, num_units_per_core, start_id}`; writer RTA `{dst_buffer, num_units_per_core, start_id}`.
- **row-major:** reader CTA = `[0]=c_0, [1]=input_unit_size` + `TAA(src)`; writer CTA = `[0]=output_cb_index, [1]=output_unit_size` + `TAA(dst)`. reader RTA `{src_buffer, input_unit_size, num_units_per_core, start_id, full_input_row/input_unit_size}`; writer RTA `{dst_buffer, output_unit_size, num_units_per_core, start_id, full_output_row/output_unit_size}`.
- **compute:** CTA `{num_units_per_core_group_N}` per group; no RTA.

#### CBs
| index | total_size | data_format | page_size |
|---|---|---|---|
| c_0 (`src0_cb_index`) | `2 * aligned_input_unit_size` | input df | `aligned_input_unit_size` |
| c_16 (`output_cb_index`, only convert_dtype) | `2 * aligned_output_unit_size` | output df | `aligned_output_unit_size` |

- no-convert: c_0 reader→writer (writer's `output_cb_index`==c_0). convert (TILE): c_0 reader→compute; c_16 compute→writer. All legal 1:1.

#### Preserved Multiplicity (compute, convert_dtype path only)
Legacy compute is two `KernelDescriptor`s of `kernel/compute/eltwise_copy.cpp` over `core_group_1` and `core_group_2`, differing only on the per-group CTA `num_units_per_core_group_{1,2}` → two `KernelSpec`s (COMPUTE_G1, COMPUTE_G2) in two `WorkUnitSpec`s (disjoint node sets), each CONSUMER of c_0 and PRODUCER of c_16. (Reader/writer are single KernelSpec over all_cores with per-core RTAs.)

#### Shared kernels (SameMemoryConfig)
- `ttnn/cpp/ttnn/kernel/dataflow/reader_unary_stick_layout_interleaved_start_id.cpp` — SHARED POOL, no fork. Create fork. Sunset: `embedding`, `data_movement/concat`.
- `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp` — SHARED POOL, no fork. Create fork. Sunset: `embedding`, `data_movement/concat`.
- `ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp` — SHARED POOL, no fork. Create fork. Sunset: `sharded_to_interleaved`, `sharded_to_interleaved_partial`, `untilize_with_unpadding`.
- OWN kernels (reader/writer_unary_start_id, reader/writer_unary_stick_start_id) — census pending per-kernel; if lent, fork. (Checked during construction.)

#### Flags
- Function-call escape (DefaultRowMajor reader): `tt::data_movement::common::tt_memmove(Noc, …)` — Device 2.0-native, bridges cleanly, no donor change.

---

## TTNN ProgramFactory
- **Concept (inherited from audit):** `ProgramSpecFactoryConcept` (all three factories).
- **Custom `compute_program_hash`:** none — leave default.
- **Implementation notes:** device-op `.hpp` changes each factory struct's method from `create_descriptor` → `create_program_artifacts` (returns `ttnn::device_operation::ProgramArtifacts`) as that factory is ported. Mixed-concept variant is valid; framework dispatches per factory. Include `ttnn/metal_v2_artifacts.hpp`. No pybind `create_descriptor` to remove (nanobind binds only free functions).

## Planned Spec Shape (per factory)

### DefaultRowMajor
- **KernelSpecs:** READER (`redistribute_pages_row_major_reader.cpp`), WRITER (`redistribute_pages_row_major_writer.cpp`).
- **DataflowBufferSpecs:** INPUT_PAGES (c_0), OUTPUT_PAGE (c_1).
- **SemaphoreSpecs:** none.
- **TensorParameters:** INPUT (reader), OUTPUT (writer).
- **WorkUnitSpecs:** one — {READER, WRITER} over all_cores.
- **DFB bindings:** INPUT_PAGES → READER PRODUCER + READER CONSUMER (self-loop, one accessor name `in0`). OUTPUT_PAGE → READER PRODUCER (`in1`), WRITER CONSUMER (`in1`).
- **Tensor bindings:** INPUT → READER accessor `src`; OUTPUT → WRITER accessor `dst`.

### DefaultTilized
- **KernelSpecs:** READER (unary interleaved `_metal2` fork), WRITER (unary interleaved `_metal2` fork), COMPUTE (`sharded/.../eltwise_copy.cpp` new fork; only convert_df).
- **DataflowBufferSpecs:** c_0 (IN), c_16 (OUT; only convert_df).
- **TensorParameters:** input, output (binding vocab dictated by the existing `_metal2` reader/writer forks — read forks first).
- **WorkUnitSpec:** one over all_cores.

### SameMemoryConfig
- **KernelSpecs:** READER, WRITER (runtime source per tilized/sharded), COMPUTE_G1(+G2) (convert_dtype only).
- **DataflowBufferSpecs:** c_0 (IN), c_16 (OUT; convert_dtype).
- **TensorParameters:** input, output.
- **WorkUnitSpecs:** main {READER, WRITER} over all_cores; +convert: WU_G1 {COMPUTE_G1} over core_group_1, WU_G2 {COMPUTE_G2} over core_group_2.

## Preserved Multiplicity
- DefaultRowMajor: none.
- DefaultTilized: none (compute num_tiles is an RTA over all_cores).
- SameMemoryConfig: compute (convert_dtype) — two `KernelSpec`s of `kernel/compute/eltwise_copy.cpp` in two `WorkUnitSpec`s over disjoint core groups, per-group CTA `num_units_per_core_group_{1,2}`.

## Dropped Plumbing (DefaultRowMajor)
| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader RTA[0] | `input.buffer()` (Buffer*) | `TensorBinding(INPUT, "src")` |
| writer RTA[0] | `output.buffer()` (Buffer*) | `TensorBinding(OUTPUT, "dst")` |
| reader CTA[0] `c_0` | magic CB index | `DFBBinding(INPUT_PAGES, "in0", PRODUCER)+CONSUMER` (self-loop) |
| reader CTA[1] `c_1` | magic CB index | `DFBBinding(OUTPUT_PAGE, "in1", PRODUCER)` |
| writer CTA[0] `c_1` | magic CB index | `DFBBinding(OUTPUT_PAGE, "in1", CONSUMER)` |
| reader CTA `TensorAccessorArgs(input.buffer())` `:136` | TAA plumbing | binding |
| writer CTA `TensorAccessorArgs(output.buffer())` `:148` | TAA plumbing | binding |
| reader kernel `TensorAccessorArgs<10>()` `:37`, `get_arg_val<u32>(0)` src_addr | TAA + addr RTA | `TensorAccessor(tensor::src)` |
| writer kernel `TensorAccessorArgs<6>()` `:30`, `get_arg_val<u32>(0)` dst_addr | TAA + addr RTA | `TensorAccessor(tensor::dst)` |
| reader/writer positional CTAs | positional | named CTAs |
| reader/writer positional RTAs (start_row, num_rows) | positional | named RTAs |

## Applied Patterns
- [Self-loop DFB binding](../shared/port_patterns.md) — DefaultRowMajor c_0 (INPUT_PAGES) is a reader-only L1 scratchpad; bind reader as both PRODUCER and CONSUMER.
- [Two-toucher / disjoint-node work-split → multiple KernelSpecs](../shared/port_patterns.md) — SameMemoryConfig compute preserved multiplicity (disjoint node sets, single-role bindings, no flag).
- [Caution: Porting a shared kernel](../shared/port_patterns.md) — DefaultTilized (reuse existing `_metal2` forks + create sharded eltwise_copy fork) and SameMemoryConfig (create 3 forks).

## Deferred / Flagged
- **`SameMemoryConfig` capitulated during construction (audit gap).** Discovered at build time that the peer op `data_movement/move` (`move_program_factory.cpp:27`) reuses `SameMemoryConfig::create_descriptor` and depends on its `ProgramDescriptor` return + positional RTA layout. Porting `SameMemoryConfig` breaks `move`, which is outside this port's writeable surface; a factory cannot expose both `create_descriptor` and `create_program_artifacts` (dual-concept `static_assert`). Left `SameMemoryConfig` on the legacy concept and reverted its 4 own kernels + its 3 planned shared-pool forks. See `METAL2_PORT_REPORT.md` → Handoff points. DefaultRowMajor and DefaultTilized are unaffected (not consumed by `move`) and ported cleanly.
- No GlobalCircularBuffer, no address-offset, no semaphores, no custom hash, no override_runtime_arguments in any factory.
