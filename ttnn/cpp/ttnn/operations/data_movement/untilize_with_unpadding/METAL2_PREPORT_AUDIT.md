# Pre-port audit: `data_movement/untilize_with_unpadding`

**Identifying section:**

- **`UntilizeWithUnpaddingDeviceOperation`**
  - `UntilizeWithUnpaddingMultiCoreBlockInterleavedProgramFactory` (`untilize_with_unpadding_multi_core_block_interleaved_program_factory.cpp`)
  - `UntilizeWithUnpaddingMultiCoreColInterleavedProgramFactory` (`untilize_with_unpadding_multi_core_col_interleaved_program_factory.cpp`)
  - `UntilizeWithUnpaddingMultiCoreInterleavedProgramFactory` (`untilize_with_unpadding_multi_core_interleaved_program_factory.cpp`)
  - `UntilizeWithUnpaddingMultiCoreNdShardedProgramFactory` (`untilize_with_unpadding_multi_core_nd_sharded_program_factory.cpp`)
  - `UntilizeWithUnpaddingMultiCoreShardedProgramFactory` (`untilize_with_unpadding_multi_core_sharded_program_factory.cpp`)
  - `UntilizeWithUnpaddingSingleCoreProgramFactory` (`untilize_with_unpadding_single_core_program_factory.cpp`)

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `port_op_to_metal2_audit.md`.

## Result

**YELLOW** — all six factories on `ProgramDescriptor`, Device-2.0-compliant kernels, no UNSUPPORTED features. Two Step 0.5 findings (buffer-address-in-RTA) flagged as port-time work.

### Yellow side-issues

- Step 0.5 bypass: `untilize_with_unpadding_multi_core_col_interleaved_program_factory.cpp:169,178` and `untilize_with_unpadding_multi_core_block_interleaved_program_factory.cpp:295,308` pass `src0_buffer->address()` / `dst_buffer->address()` as RTAs. Port-time work — re-express via `TensorParameter`.

## Porting prerequisites

### ProgramDescriptor API: **GREEN**

All six factories return `ProgramDescriptor`. No imperative-API calls (the `// SetRuntimeArgs` mention in `_multi_core_sharded_*.cpp:208` and the `// CreateCircularBuffer` mention in `_multi_core_block_interleaved_*.cpp:125` are explanatory comments, not active code).

### Device 2.0 DM: **GREEN**

All own dataflow kernels under `untilize_with_unpadding/device/kernels/dataflow/` use the Device 2.0 API. The op also instantiates the compute kernel from peer `untilize/.../compute/untilize.cpp` (Device-2.0-compliant per the `untilize` audit).

### TensorAccessor usage: **GREEN**

Interleaved factories use `TensorAccessor` end-to-end (`writer_unary_stick_layout_*.cpp`). Sharded and nd-sharded factories use the borrowed-memory CB pattern (causal-link gate applies — kernels read from CBs whose backing is the bound tensor buffer).

## Feature compatibility check

| Feature | Status | Notes |
|---|---|---|
| GlobalCircularBuffer | N/A | not used |
| Dynamic CircularBuffer (CB on borrowed memory) | GREEN | sharded variants use `cb.buffer`; port uses `borrowed_from` |
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

None.

### `override_runtime_arguments` complexity

None — descriptor-form factories rely on framework binding-injection.

## Out-of-directory call surface

**Op-level roll-up:** ✓ clean — own kernels Device-2.0-native; peer kernel instantiations Device-2.0-compliant.

### Summary

| Op kernel | Donor file | Donor class | Functions consumed | Status roll-up |
|---|---|---|---|---|
| `writer_unary_stick_layout_*.cpp` (own) | `tt_metal/api/.../noc.h`, `tensor_accessor.h` | LLK / tt_metal | `Noc::async_write`, `TensorAccessor` | ✓ |

### Per-call detail

Omitted — all summary rows are ✓.

### Borrowed kernel files

| Kernel file path | Owning op family | Also borrowed by |
|---|---|---|
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp` | `eltwise/unary` | broadly-shared |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_*.cpp` | `eltwise/unary` | broadly-shared |
| `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_blocks.cpp` | shared kernel pool | broadly-shared |
| `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/reader_unary_nd_sharded_blocks.cpp` | `data_movement/sharded` (in-family) | also `untilize` |
| `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize.cpp` | `data_movement/untilize` (sibling in family) | this op + `untilize` |

## TensorAccessor bypass

**Op-level roll-up:** ⚠ YELLOW — two interleaved factories pass buffer addresses through RTAs.

### Per-binding inventory

- **`input_tensor` (col_interleaved factory):** ⚠ YELLOW — `src0_buffer->address()` as RTA at `untilize_with_unpadding_multi_core_col_interleaved_program_factory.cpp:178`. Consumed kernel-side as the read base. Resolution: port re-expresses via `TensorParameter`.
- **`output_tensor` (col_interleaved factory):** ⚠ YELLOW — `dst_buffer->address()` as RTA at `untilize_with_unpadding_multi_core_col_interleaved_program_factory.cpp:169`. Resolution: same.
- **`input_tensor` (block_interleaved factory):** ⚠ YELLOW — `src0_buffer->address()` as RTA at `untilize_with_unpadding_multi_core_block_interleaved_program_factory.cpp:308`. Resolution: same.
- **`output_tensor` (block_interleaved factory):** ⚠ YELLOW — `dst_buffer->address()` as RTA at `untilize_with_unpadding_multi_core_block_interleaved_program_factory.cpp:295`. Resolution: same.
- **`input_tensor` / `output_tensor` (other 4 factories):** clean — `TensorAccessor` or borrowed-memory CB.

## Path forward

YELLOW — port is unblocked. Two factories have port-time bypass work (re-express buffer-address RTAs as `TensorParameter` bindings). No prereqs to wait for.

Suggested handoff to `port_op_to_metal2_recipe.md` once the user approves.

## Questions for the user

1. **Bypass scope:** confirm that re-expressing the `block_interleaved` and `col_interleaved` factory bypasses via `TensorParameter` (rather than `TensorAccessor::get_bank_base_address`) is the desired resolution. The access pattern in both is page-by-page tiled reads/writes, which `TensorAccessor` handles directly — no exotic walk.
