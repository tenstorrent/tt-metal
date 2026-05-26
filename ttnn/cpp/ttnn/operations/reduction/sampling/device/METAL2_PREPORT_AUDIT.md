# Pre-port audit: `ttnn/cpp/ttnn/operations/reduction/sampling/device`

- **`SamplingDeviceOperation`** (in `sampling_device_operation.{hpp,cpp}`)
  - `SamplingProgramFactory` (`sampling_program_factory.{hpp,cpp}`)

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `port_op_to_metal2_audit.md`.

## Result

**GREEN** — clean port candidate. Single factory satisfies `ProgramDescriptorFactoryConcept` (`create_descriptor`), all kernels use Device 2.0 wrappers (`Noc`, `CircularBuffer`) and standard `TensorAccessorArgs` (no `Runtime*` flavors). No borrowed-memory CBs, no semaphores, no `GlobalCircularBuffer` / `GlobalSemaphore`. Handoff to the recipe doc is appropriate.

## Porting prerequisites

### ProgramDescriptor API: **GREEN**

`SamplingProgramFactory::create_descriptor` returns a `tt::tt_metal::ProgramDescriptor`. Uses `KernelDescriptor`, `CBDescriptor`, `CBFormatDescriptor`, `ReaderConfigDescriptor`, `WriterConfigDescriptor`, `ComputeConfigDescriptor`. No `host_api.hpp`-style imperative builders.

### Device 2.0 DM: **GREEN**

- `reader_values_indices_tensor.cpp` uses `Noc noc;` + `CircularBuffer cb_xxx(cb_id_xxx);`.
- `writer_interleaved.cpp` uses `Noc noc;` + `CircularBuffer` for all CB declarations.
- No raw `noc_async_read` / `cb_reserve_back` / `cb_push_back` free-function calls remain in the op's kernels.

### TensorAccessor usage: **GREEN**

All tensor reads/writes go through `TensorAccessor` with `TensorAccessorArgs<N>()` plumbing:
- Reader: `TensorAccessorArgs<7>()` for `input_values`, then `next_compile_time_args_offset()` for `input_indices`.
- Writer: four `TensorAccessorArgs<N>()` chained — `output` / `temp` / `k` / `p`.

## Feature compatibility check

| Feature | Status | Notes |
|---|---|---|
| GlobalCircularBuffer | N/A | No `experimental::GlobalCircularBuffer` references. |
| Dynamic CircularBuffer (CB on borrowed memory) | N/A | No `CBDescriptor::buffer` set (verified with grep). |
| CBDescriptor `address_offset` (non-zero) | N/A | No `.address_offset` writes. |
| Aliased Circular Buffers | N/A | Every `format_descriptors` initializer is single-element. |
| GlobalSemaphore | N/A | No `GlobalSemaphore` references; the op uses no semaphores at all. |
| Non-zero semaphore initial value | N/A | No semaphores. |
| `ArgConfig::Runtime*` tensor-accessor flavors | GREEN | No `ArgConfig::Runtime` token anywhere under the op's directory. |
| `UpdateCircularBuffer*` | GREEN | No `UpdateCircularBufferTotalSize` / `UpdateCircularBufferPageSize` calls. |

## Path forward

GREEN — proceed with the port.

### Op shape notes (for the planning step)

- Six tensor parameters: `input_values`, `input_indices`, `k`, `p`, `temp`, `output`.
- 16 standard CBs (no borrowed memory). The `core_id` CTA varies per core (preserved-multiplicity case for the writer); reader and compute have identical CTAs across cores.
- One reader KernelSpec on the full core grid; per-core writer KernelSpecs (or `core_id` as a per-node RTA); single compute KernelSpec on the full core grid (compute CTAs are uniform).

## Questions for the user

None — proceeding with the port.
