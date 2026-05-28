# Pre-port audit: `data_movement/sharded/interleaved_to_sharded`

**Identifying section:**

- **`InterleavedToShardedDeviceOperation`**
  - `InterleavedToShardedProgramFactory` (`interleaved_to_sharded_program_factory.cpp`)

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `port_op_to_metal2_audit.md`.

## Result

**GREEN** — single `ProgramDescriptor`-form factory, Device-2.0-compliant kernels, no UNSUPPORTED features in use.

### Yellow side-issues

- Custom `compute_program_hash` on the device-op (`interleaved_to_sharded_op.cpp:141`) omits `TensorSpec` from the hash key. Not a port blocker; flagged because the `UpdateTensorArgs` legality check on fast-path program-cache hits in Metal 2.0 keys off this. See [Questions for the user](#questions-for-the-user).

## Porting prerequisites

### ProgramDescriptor API: **GREEN**

`InterleavedToShardedProgramFactory::create_descriptor` returns a `ProgramDescriptor`. No imperative-API calls.

### Device 2.0 DM: **GREEN**

The op has no own dataflow kernels — it instantiates peer kernels:
- `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_*` (peer, Device-2.0-compliant)
- `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp` (in-family, Device-2.0-compliant)
- `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/compute/eltwise_copy.cpp` (in-family)

### TensorAccessor usage: **GREEN (N/A)**

No own dataflow kernels. The sharded destination CB binds the output buffer via `CBDescriptor::buffer = bound_buffer` (borrowed-memory pattern). Causal-link gate applies: lack of `TensorAccessor` is intended — the borrowed-memory CB **is** the tensor access. Port uses `DataflowBufferSpec::borrowed_from`.

## Feature compatibility check

| Feature | Status | Notes |
|---|---|---|
| GlobalCircularBuffer | N/A | not used |
| Dynamic CircularBuffer (CB on borrowed memory) | GREEN | used; port uses `borrowed_from` |
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

`interleaved_to_sharded_op.cpp:141` — `InterleavedToShardedDeviceOperation::compute_program_hash` hashes `output_mem_config`, `output_dtype`, `keep_l1_aligned`, plus `input_tensor.{dtype, memory_config, layout, padded_shape}`. **`TensorSpec` is not directly in the hash key**, though the relevant fields are spread across the inputs. The `UpdateTensorArgs` legality check on Metal 2.0 fast-path cache hits keys off `TensorSpec`; if a cache hit fires with a tensor whose spec differs in a way the custom hash doesn't separate, the framework may reject the cache hit. **Verification-time concern, not a port-time blocker** — fix when/if it bites is to drop the custom hash and use the default.

### `override_runtime_arguments` complexity

None — the factory exposes only `create_descriptor`; per-execution buffer-address patching is handled by the framework via `cb.buffer = <buffer>`.

## Out-of-directory call surface

**Op-level roll-up:** ✓ clean — no function-call escapes; only file-path kernel instantiations of broadly-shared Device-2.0-compliant donor kernels.

### Summary

| Op kernel | Donor file | Donor class | Functions consumed | Status roll-up |
|---|---|---|---|---|
| (none — op has no own kernel files) | n/a | n/a | n/a | n/a |

### Per-call detail

Omitted — all summary rows are ✓ (or n/a).

### Borrowed kernel files

| Kernel file path | Owning op family | Also borrowed by |
|---|---|---|
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` (and variants) | `eltwise/unary` | broadly-shared |
| `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp` | `data_movement/sharded` (in-family) | broadly-shared |
| `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/compute/eltwise_copy.cpp` | `data_movement/sharded` (in-family) | broadly-shared |

## TensorAccessor bypass

**Op-level roll-up:** ✓ clean — destination is via borrowed-memory CB; source is via the standard `TensorAccessor`-style reader.

### Per-binding inventory

- **`input_tensor` (interleaved src):** clean — `TensorAccessor` (peer reader kernel).
- **`output_tensor` (sharded dst):** clean — borrowed-memory CB (causal-link).

## Path forward

GREEN — port is unblocked. Suggested handoff to `port_op_to_metal2_recipe.md` once the user approves.

## Questions for the user

1. **Custom `compute_program_hash`:** the device-op defines a custom hash that omits direct `TensorSpec`. The audit doc notes this may trigger `UpdateTensorArgs` legality failures on fast-path cache hits under Metal 2.0. Should the port drop the custom hash (revert to default) preemptively, or wait for the failure to surface during verification? (`interleaved_to_sharded_op.cpp:141`)
