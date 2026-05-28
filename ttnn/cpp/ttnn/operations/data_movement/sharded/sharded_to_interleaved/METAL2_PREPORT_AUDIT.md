# Pre-port audit: `data_movement/sharded/sharded_to_interleaved`

**Identifying section:**

- **`ShardedToInterleavedDeviceOperation`**
  - `ShardedToInterleavedProgramFactory` (`sharded_to_interleaved_program_factory.cpp`)

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `port_op_to_metal2_audit.md`.

## Result

**GREEN** — single `ProgramDescriptor`-form factory, Device-2.0-compliant kernels, no UNSUPPORTED features in use.

Handoff to the recipe doc is appropriate after explicit user go-ahead.

## Porting prerequisites

### ProgramDescriptor API: **GREEN**

The sole factory `ShardedToInterleavedProgramFactory::create_descriptor` returns a `ProgramDescriptor` and populates `desc.kernels`, `desc.cbs` via the typed descriptor API. No imperative `CreateProgram` / `CreateKernel` / `CreateCircularBuffer` / `SetRuntimeArgs` calls.

### Device 2.0 DM: **GREEN**

The op references shared dataflow kernels (`eltwise/unary/.../reader_unary_sharded.cpp`, `sharded/.../writer_unary_sharded.cpp`) and the shared compute `ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp`. The borrowing kernels are already on the Device 2.0 API (no `noc_async_read`/`noc_async_write` legacy idioms). No CB-index-keyed free-function holdovers within the borrowed kernels at the op's call surface.

### TensorAccessor usage: **GREEN (N/A)**

The op has no own dataflow kernels. The peer-borrowed sharded reader/writer kernels are local-L1 sharded copies — they do not read DRAM-resident tensor data via host-managed `Buffer` addresses (the tensor body sits in the sharded CB, mapped via `CBDescriptor::buffer = bound_buffer`). Causal-link gate from Step 0.1 Check 3 applies: lack of `TensorAccessor` is intended (borrowed-memory CB **is** the tensor access). Port handles this via `DataflowBufferSpec::borrowed_from`.

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

None.

### `override_runtime_arguments` complexity

None — the factory exposes only `create_descriptor`; per-execution patching is handled by the framework via the `cb.buffer = <buffer>` binding.

## Out-of-directory call surface

**Op-level roll-up:** ✓ clean — all out-of-directory dependencies are file-path kernel instantiations of Device-2.0-compliant shared kernels.

- No `#include` escapes from this op's kernels (it has no own dataflow kernel files).
- File-path kernel instantiations are reported separately below.

### Summary

| Op kernel | Donor file | Donor class | Functions consumed | Status roll-up |
|---|---|---|---|---|
| (none — op has no own kernel files) | n/a | n/a | n/a | n/a |

### Per-call detail

Omitted — all summary rows are ✓ (or n/a).

### Borrowed kernel files

| Kernel file path | Owning op family | Also borrowed by |
|---|---|---|
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp` | `eltwise/unary` | broadly-shared |
| `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp` | `data_movement/sharded` (in-family) | broadly-shared |
| `ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp` | `kernel/compute` (shared kernel pool) | broadly-shared |

These do not gate the port — informational, in case the donor ops' Metal 2.0 ports modify the borrowed files.

## TensorAccessor bypass

**Op-level roll-up:** ✓ clean — the single tensor source is bound through the borrowed-memory CB (Dynamic CircularBuffer pattern), not through a `Buffer*->address()` RTA.

### Per-binding inventory

- **`input_tensor` (sharded src):** clean — borrowed-memory CB (causal-link).
- **`output_tensor` (interleaved dst):** clean — `TensorAccessor`-style host plumbing via the sharded writer kernel that takes the output buffer address as the standard sharded-writer arg (single-tile op, no bypass arithmetic).

## Path forward

GREEN — port is unblocked. Suggested handoff to `port_op_to_metal2_recipe.md` once the user approves.
