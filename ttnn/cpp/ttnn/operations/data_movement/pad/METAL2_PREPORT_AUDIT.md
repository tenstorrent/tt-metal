# Pre-port audit: `data_movement/pad`

**Identifying section:**

- **`PadDeviceOperation`**
  - **`ProgramDescriptor`-form factories (5):**
    - `PadRmReaderWriterMultiCoreDefaultProgramFactory` (`pad_rm_reader_writer_multi_core_default_program_factory.cpp`)
    - `PadRmShardedHeightOnlyProgramFactory` (`pad_rm_sharded_height_only_program_factory.cpp`)
    - `PadRmShardedWidthOnlyProgramFactory` (`pad_rm_sharded_width_only_program_factory.cpp`)
    - `PadTileMulticoreProgramFactory` (`pad_tile_multicore_program_factory.cpp`)
    - `PadTileProgramFactory` (`pad_tile_program_factory.cpp`)
  - **Imperative-API factories (2):**
    - `PadRmReaderWriterProgramFactory` (`pad_rm_reader_writer_program_factory.cpp`)
    - `PadRmReaderWriterMultiCoreProgramFactory` (`pad_rm_reader_writer_multi_core_program_factory.cpp`)

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `port_op_to_metal2_audit.md`.

## Result

**RED at op level; the 5 `ProgramDescriptor`-form factories form a clean subset that can be ported now.**

The 2 imperative-API factories trip Step 0.1 Check 1 RED: their `ProgramDescriptor` migration is a prerequisite to Metal 2.0, and is its own separate workstream. RED here is the **expected outcome** for any factory still on the legacy imperative API — not an alarm signal. The Metal 2.0 port for those two factories will be unblocked once their `ProgramDescriptor` migration lands.

### Clean subset

Can be ported now (subset, with the imperative-API factories left on legacy until their PD migration completes):

- `PadRmReaderWriterMultiCoreDefaultProgramFactory`
- `PadRmShardedHeightOnlyProgramFactory`
- `PadRmShardedWidthOnlyProgramFactory`
- `PadTileMulticoreProgramFactory`
- `PadTileProgramFactory`

Must wait for `ProgramDescriptor` migration first:

- `PadRmReaderWriterProgramFactory`
- `PadRmReaderWriterMultiCoreProgramFactory`

### Yellow side-issues

- Step 0.5 bypass: the 2 imperative-API factories each pass `src_buffer->address()`, `dst_buffer->address()`, and `pad_value_const_tensor.buffer()->address()` through RTAs (20 sites total). **Moot under Check 1 RED for those factories** — record for the eventual port-time work after their PD migration.

## Porting prerequisites

### ProgramDescriptor API: **RED (op level); GREEN for the 5-factory subset**

The 2 imperative-API factories use `CreateProgram` / `CreateKernel` / `CreateCircularBuffer` / `SetRuntimeArgs` (5+ sites each). RED here is the expected state for any factory still on the legacy imperative API; the `ProgramDescriptor` migration is a separate, ongoing effort and is the per-factory prereq for Metal 2.0. Step 0.3 Check C below tiers the `override_runtime_arguments` complexity, which is the best proxy for the eventual `ProgramDescriptor` lift difficulty.

The 5 `ProgramDescriptor`-form factories use the typed descriptor API throughout.

### Device 2.0 DM: **GREEN**

All own dataflow kernels under `pad/device/kernels/dataflow/` use the Device 2.0 API (`Noc`, `CircularBuffer`, `CoreLocalMem`, `UnicastEndpoint`, `TensorAccessor` as appropriate).

### TensorAccessor usage: **GREEN**

Interleaved kernels use `TensorAccessor`; sharded kernels use `UnicastEndpoint{}` (cross-core local-L1) and `CoreLocalMem<>` (self-L1). No legacy `InterleavedAddrGen`/`ShardedAddrGen` in the kernels.

## Feature compatibility check

| Feature | Status | Notes |
|---|---|---|
| GlobalCircularBuffer | N/A | not used |
| Dynamic CircularBuffer (CB on borrowed memory) | GREEN | sharded factories use `cb.buffer = <buffer>`; port uses `borrowed_from` |
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

The 2 imperative-API factories each define `override_runtime_arguments`:

- `pad_rm_reader_writer_program_factory.cpp:150-177` — **trivial** (~27 lines, RTA values only — `src_buffer->address()`, `dst_buffer->address()`, `pad_value_const_tensor.buffer()->address()`, no per-shape branching).
- `pad_rm_reader_writer_multi_core_program_factory.cpp:371-402` — **trivial** (~31 lines, same shape as above with a fixed-size 2D core grid loop).

Both translate cleanly into the framework's per-execution binding-injection once their `ProgramDescriptor` migrations land. The 5 `ProgramDescriptor`-form factories do not have `override_runtime_arguments` (binding-injection is handled by the framework via `cb.buffer = <buffer>` and the descriptor's runtime-args).

## Out-of-directory call surface

**Op-level roll-up:** ✓ clean — own kernels Device-2.0-native; peer kernel instantiations Device-2.0-compliant.

### Summary

| Op kernel | Donor file | Donor class | Functions consumed | Status roll-up |
|---|---|---|---|---|
| `reader_pad_*.cpp`, `writer_pad_*.cpp` (own) | `tt_metal/api/.../noc.h`, `tensor_accessor.h` | LLK / tt_metal | `Noc::async_*`, `TensorAccessor`, `UnicastEndpoint` | ✓ |

### Per-call detail

Omitted — all summary rows are ✓.

### Borrowed kernel files

| Kernel file path | Owning op family | Also borrowed by |
|---|---|---|
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` | `eltwise/unary` | broadly-shared |

## TensorAccessor bypass

**Op-level roll-up:** ⚠ YELLOW (confined to the 2 imperative-API factories, which are also Check 1 RED — the bypass becomes a port-time work item only after those factories' PD migration lands).

### Per-binding inventory

- **`input_tensor` (`pad_rm_reader_writer_program_factory.cpp`):** ⚠ YELLOW — `src_buffer->address()` as RTA at lines 110, 166, 173. Moot under Check 1 RED.
- **`output_tensor` (`pad_rm_reader_writer_program_factory.cpp`):** ⚠ YELLOW — `dst_buffer->address()` as RTA at lines 111, 167, 174. Moot under Check 1 RED.
- **`pad_value_const_tensor` (`pad_rm_reader_writer_program_factory.cpp`):** ⚠ YELLOW — `pad_value_const_tensor.buffer()->address()` at lines 43, 161. Moot under Check 1 RED.
- **`input_tensor` (`pad_rm_reader_writer_multi_core_program_factory.cpp`):** ⚠ YELLOW — `src0_buffer->address()` at lines 319, 389, 396. Moot under Check 1 RED.
- **`output_tensor` (`pad_rm_reader_writer_multi_core_program_factory.cpp`):** ⚠ YELLOW — `dst_buffer->address()` at lines 320, 390, 397. Moot under Check 1 RED.
- **`pad_value_const_tensor` (`pad_rm_reader_writer_multi_core_program_factory.cpp`):** ⚠ YELLOW — `pad_value_const_tensor.buffer()->address()` at lines 189, 381. Moot under Check 1 RED.
- **All bindings (the 5 `ProgramDescriptor`-form factories):** clean — `TensorAccessor` or borrowed-memory CB.

## Path forward

- **For the 5 `ProgramDescriptor`-form factories**: port can proceed now. Suggested handoff to `port_op_to_metal2_recipe.md` once the user approves the scoped subset.
- **For the 2 imperative-API factories**: blocked on the separate `ProgramDescriptor` migration workstream. Once that lands, those two factories' Step 0.5 YELLOW (buffer-address-as-RTA) becomes the only remaining port-time work item — straightforward to re-express via `TensorParameter`.

## Questions for the user

1. **Scoped subset:** confirm we should proceed with the 5-factory subset port, leaving the 2 imperative-API factories on legacy (the device-operation will continue to dispatch the imperative factories on the code paths that select them). Alternatively, hold the port until those factories' PD migration lands.
2. **Hybrid file handling:** the 2 imperative-API factory files and 5 PD-form factory files share the device-operation type and a single results-types header. A subset port leaves the imperative files in place — they continue to build against the legacy `host_api.hpp`. Confirm this is acceptable (vs requiring all-or-nothing).
