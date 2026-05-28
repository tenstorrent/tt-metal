# Pre-port audit: `data_movement/untilize`

**Identifying section:**

- **`UntilizeDeviceOperation`**
  - `UntilizeMultiCoreInputAndOutputNDShardTypeAndShardSpecIdenticalProgramFactory` (`untilize_multi_core_input_and_output_nd_shard_type_and_shard_spec_identical_program_factory.cpp`)
  - `UntilizeMultiCoreInputAndOutputShardTypeAndShardSpecIdenticalProgramFactory` (`untilize_multi_core_input_and_output_shard_type_and_shard_spec_identical_program_factory.cpp`)
  - `UntilizeMultiCoreNDShardInputProgramFactory` (`untilize_multi_core_nd_shard_input_program_factory.cpp`)
  - `UntilizeMultiCoreParallelizeColumnProgramFactory` (`untilize_multi_core_parallelize_column_program_factory.cpp`)
  - `UntilizeMultiCoreProgramFactory` (`untilize_multi_core_program_factory.cpp`)
  - `UntilizeMultiCoreSubCoreGridsProgramFactory` (`untilize_multi_core_sub_core_grids_program_factory.cpp`)
  - `UntilizeSingleCoreProgramFactory` (`untilize_single_core_program_factory.cpp`)
  - `UntilizeMultiCoreBlockProgramFactory` (`untilize_multi_core_block_program_factory.cpp`)

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `port_op_to_metal2_audit.md`.

## Result

**GREEN** — all eight factories on `ProgramDescriptor`, Device-2.0-compliant kernels, no UNSUPPORTED features.

Handoff to the recipe doc is appropriate after explicit user go-ahead.

## Porting prerequisites

### ProgramDescriptor API: **GREEN**

All eight factories return `ProgramDescriptor`. No imperative-API calls.

### Device 2.0 DM: **GREEN**

All own dataflow kernels under `untilize/device/kernels/dataflow/` use the Device 2.0 API (`Noc`, `CircularBuffer`, `TensorAccessor`, `CoreLocalMem`, `UnicastEndpoint` as appropriate). Recent fix in the same family: `compute/untilize_variable_num_blocks.cpp` early-returns when `per_core_block_cnt == 0` to handle inactive cores in uneven nd-sharded launches.

### TensorAccessor usage: **GREEN**

The interleaved reader/writer kernels use `TensorAccessor` end-to-end. Sharded kernels (`reader_unary_sharded_blocks.cpp`, peer `reader_unary_sharded.cpp` / `writer_unary_sharded.cpp`) use the borrowed-memory CB pattern (causal-link gate applies).

## Feature compatibility check

| Feature | Status | Notes |
|---|---|---|
| GlobalCircularBuffer | N/A | not used |
| Dynamic CircularBuffer (CB on borrowed memory) | GREEN | sharded-spec-identical factories use `cb.buffer = <buffer>`; port uses `borrowed_from` |
| CBDescriptor `address_offset` (non-zero) | GREEN | not used |
| Aliased Circular Buffers | N/A | every `format_descriptors` initializer is single-element |
| GlobalSemaphore | N/A | no semaphores |
| Non-zero semaphore initial value | N/A | no semaphores |
| Dynamic TensorAccessor (`ArgConfig::Runtime*`) | N/A | not used |
| `UpdateCircularBuffer*` | GREEN | not used |

## Port complexity signals

### Variadic kernels

None.

### Custom `compute_program_hash`

None — `UntilizeDeviceOperation` relies on the default hash.

### `override_runtime_arguments` complexity

None — descriptor-form factories rely on framework binding-injection.

## Out-of-directory call surface

**Op-level roll-up:** ✓ clean — own kernels Device-2.0-native; instantiated peer kernels Device-2.0-compliant.

### Summary

| Op kernel | Donor file | Donor class | Functions consumed | Status roll-up |
|---|---|---|---|---|
| `reader_unary_*.cpp` / `writer_unary_*.cpp` | `tt_metal/api/.../noc.h`, `tensor_accessor.h` | LLK / tt_metal | `Noc::async_*`, `TensorAccessor` | ✓ |
| compute path | `compute/untilize.cpp`, `compute/untilize_variable_num_blocks.cpp` | own | LLK | ✓ |

### Per-call detail

Omitted — all summary rows are ✓.

### Borrowed kernel files

| Kernel file path | Owning op family | Also borrowed by |
|---|---|---|
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_*.cpp` | `eltwise/unary` | broadly-shared |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp` | `eltwise/unary` | broadly-shared |
| `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/reader_unary_nd_sharded_blocks.cpp` | `data_movement/sharded` (in-family) | also `untilize_with_unpadding` |
| `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp` | `data_movement/sharded` (in-family) | broadly-shared |

## TensorAccessor bypass

**Op-level roll-up:** ✓ clean — interleaved factories via `TensorAccessor`; sharded factories via borrowed-memory CB.

### Per-binding inventory

- **`input_tensor` (interleaved factories):** clean — `TensorAccessor`.
- **`output_tensor` (interleaved factories):** clean — `TensorAccessor`.
- **`input_tensor` / `output_tensor` (sharded-spec-identical factories):** clean — borrowed-memory CB.
- **`input_tensor` / `output_tensor` (nd-shard variants):** clean — `TensorAccessor` via `shard_pages` iterator.

## Path forward

GREEN — port is unblocked. Eight factories make this a sizable port; suggest tackling in subset order (single-core → multi-core interleaved → sharded variants → nd-sharded). Suggested handoff to `port_op_to_metal2_recipe.md` once the user approves.
