# Pre-port audit: `data_movement/tilize`

**Identifying section:**

- **`TilizeDeviceOperation`**
  - `TilizeMultiCoreBlockProgramFactory` (`tilize_multi_core_block_program_factory.cpp`)
  - `TilizeMultiCoreDefaultProgramFactory` (`tilize_multi_core_default_program_factory.cpp`)
  - `TilizeMultiCoreShardedProgramFactory` (`tilize_multi_core_sharded_program_factory.cpp`)
  - `TilizeMultiCoreWidthShardedProgramFactory` (`tilize_multi_core_width_sharded_program_factory.cpp`)
  - `TilizeSingleCoreProgramFactory` (`tilize_single_core_program_factory.cpp`)

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `port_op_to_metal2_audit.md`.

## Result

**GREEN** — all five factories on `ProgramDescriptor`, Device-2.0-compliant kernels, no UNSUPPORTED features.

Handoff to the recipe doc is appropriate after explicit user go-ahead.

## Porting prerequisites

### ProgramDescriptor API: **GREEN**

All five factories return `ProgramDescriptor` from `create_descriptor`. No imperative-API calls.

### Device 2.0 DM: **GREEN**

Own dataflow kernels (`reader_unary_stick_layout_split_rows_multicore.cpp`, `reader_unary_stick_layout_split_rows_singlecore.cpp`) use the Device 2.0 API. Sharded factories instantiate broadly-shared kernels (`reader_unary_sharded.cpp`, `writer_unary_sharded.cpp`) which are Device-2.0-compliant.

### TensorAccessor usage: **GREEN**

The own dataflow kernels use `TensorAccessor` with named `noc.async_read(...)` calls. Sharded factories rely on borrowed-memory CBs (no `TensorAccessor` needed; causal-link gate applies). Compute kernel uses the shared `ttnn/cpp/ttnn/kernel/compute/tilize.cpp` which only consumes from CBs — out of scope for Check 3.

## Feature compatibility check

| Feature | Status | Notes |
|---|---|---|
| GlobalCircularBuffer | N/A | not used |
| Dynamic CircularBuffer (CB on borrowed memory) | GREEN | sharded factories use `cb.buffer = <buffer>`; port uses `borrowed_from` |
| CBDescriptor `address_offset` (non-zero) | GREEN | not used |
| Aliased Circular Buffers | N/A | single-element `format_descriptors` throughout |
| GlobalSemaphore | N/A | no semaphores |
| Non-zero semaphore initial value | N/A | no semaphores |
| Dynamic TensorAccessor (`ArgConfig::Runtime*`) | N/A | not used |
| `UpdateCircularBuffer*` | GREEN | not used |

## Port complexity signals

### Variadic kernels

None.

### Custom `compute_program_hash`

None.

### `override_runtime_arguments` complexity

None — descriptor-form factories rely on framework binding-injection.

## Out-of-directory call surface

**Op-level roll-up:** ✓ clean — own kernels are Device-2.0-native; instantiated peer kernels are Device-2.0-compliant.

### Summary

| Op kernel | Donor file | Donor class | Functions consumed | Status roll-up |
|---|---|---|---|---|
| `reader_unary_stick_layout_split_rows_*core.cpp` | `tt_metal/api/.../noc.h`, `tensor_accessor.h` | LLK / tt_metal | `Noc::async_read`, `TensorAccessor` | ✓ |
| compute path | `ttnn/cpp/ttnn/kernel/compute/tilize.cpp` | shared kernel pool | LLK | ✓ |

### Per-call detail

Omitted — all summary rows are ✓.

### Borrowed kernel files

| Kernel file path | Owning op family | Also borrowed by |
|---|---|---|
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp` | `eltwise/unary` | broadly-shared |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | `eltwise/unary` | broadly-shared |
| `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp` | `data_movement/sharded` (in-family) | broadly-shared |
| `ttnn/cpp/ttnn/kernel/compute/tilize.cpp` | shared kernel pool | broadly-shared |

## TensorAccessor bypass

**Op-level roll-up:** ✓ clean — interleaved factories use `TensorAccessor`; sharded factories use borrowed-memory CBs.

### Per-binding inventory

- **`input_tensor` (interleaved factories):** clean — `TensorAccessor`.
- **`output_tensor` (interleaved factories):** clean — `TensorAccessor` (peer writer kernel takes the buffer through the standard sharded-writer arg).
- **`input_tensor` / `output_tensor` (sharded factories):** clean — borrowed-memory CB (causal-link).

## Path forward

GREEN — port is unblocked. Suggested handoff to `port_op_to_metal2_recipe.md` once the user approves.
