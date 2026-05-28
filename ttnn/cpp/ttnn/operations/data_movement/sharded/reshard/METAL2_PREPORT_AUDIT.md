# Pre-port audit: `data_movement/sharded/reshard`

**Identifying section:**

- **`ReshardDeviceOperation`** (and ND-reshard variants)
  - `ReshardProgramFactoryGeneric` (`reshard_program_factory_generic.cpp`)
  - `ReshardProgramFactorySameHeight` (`reshard_program_factory_same_height.cpp`)
  - `ReshardProgramFactorySameWidth` (`reshard_program_factory_same_width.cpp`)
  - `NDReshardProgramFactoryCopyLocal` (`nd_reshard_program_factory_copy_local.cpp`)
  - `NDReshardProgramFactoryCopyPages` (`nd_reshard_program_factory_copy_pages.cpp`)

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `port_op_to_metal2_audit.md`.

## Result

**YELLOW** — all five factories on `ProgramDescriptor`, Device-2.0-compliant kernels, no UNSUPPORTED features. One Step 0.5 finding (buffer-address-in-RTA in the generic factory) flagged as port-time work.

### Yellow side-issues

- Step 0.5 bypass: `reshard_program_factory_generic.cpp` passes `input_buffer->address()` as RTA (4 sites at lines 746/754/764/774). Resolution at port time — re-express via `TensorParameter`. See [TensorAccessor bypass](#tensoraccessor-bypass).

## Porting prerequisites

### ProgramDescriptor API: **GREEN**

All five factories return `ProgramDescriptor` from `create_descriptor`. No imperative-API calls.

### Device 2.0 DM: **GREEN**

The own dataflow kernels (`nd_reshard_copy_local_shards.cpp`, `nd_reshard_copy_pages_reader.cpp`, `nd_reshard_copy_pages_writer.cpp`) use the Device 2.0 API (`Noc`, `CircularBuffer`, `CoreLocalMem`, `TensorAccessor`/`UnicastEndpoint` as appropriate). The generic and same-{height,width} factories use peer kernels under `data_movement/sharded/device/kernels/dataflow/reshard_*.cpp` which are also Device-2.0-compliant.

### TensorAccessor usage: **GREEN**

The `nd_reshard_copy_*` kernels use `TensorAccessor`; the `reshard_*_reader/writer` kernels use `UnicastEndpoint{}` for core-to-core L1 copies and `AllocatorBank<>` for bank-keyed DRAM/L1 access (Device-2.0-native). No legacy `InterleavedAddrGen` / `ShardedAddrGen` usage in the kernels.

## Feature compatibility check

| Feature | Status | Notes |
|---|---|---|
| GlobalCircularBuffer | N/A | not used |
| Dynamic CircularBuffer (CB on borrowed memory) | GREEN | used (sharded shards bind their buffers); port uses `borrowed_from` |
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

None — all factories rely on the framework's binding-injection.

## Out-of-directory call surface

**Op-level roll-up:** ✓ clean — own kernels are Device-2.0-native; cross-op donor instantiations use Device-2.0-compliant peer kernels in the same family (`sharded/`).

### Summary

| Op kernel | Donor file | Donor class | Functions consumed | Status roll-up |
|---|---|---|---|---|
| `nd_reshard_copy_local_shards.cpp` | `tt_metal/api/.../tensor_accessor.h` | LLK / tt_metal | `TensorAccessor::shard_pages`, `Page::noc_addr/page_id` | ✓ |
| `nd_reshard_copy_pages_reader.cpp` | (same) | LLK / tt_metal | `TensorAccessor::pages`, `Page::page_id` | ✓ |
| `nd_reshard_copy_pages_writer.cpp` | (same) | LLK / tt_metal | (same) | ✓ |

### Per-call detail

Omitted — all summary rows are ✓.

### Borrowed kernel files

| Kernel file path | Owning op family | Also borrowed by |
|---|---|---|
| `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/reshard_reader.cpp` | `data_movement/sharded` (in-family) | this op only |
| `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/reshard_reader_diff_width.cpp` | (same) | this op only |
| `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/reshard_same_{height,width}_{reader,writer}.cpp` | (same) | this op only |

These are in-family (`sharded/` umbrella). Not a coupling concern.

## TensorAccessor bypass

**Op-level roll-up:** ⚠ YELLOW — one binding bypasses in `reshard_program_factory_generic.cpp`.

### Per-binding inventory

- **`input_tensor` (in `reshard_program_factory_generic.cpp`):** ⚠ YELLOW — `input_buffer->address()` passed as RTA at `reshard_program_factory_generic.cpp:746` / `:754` / `:764` / `:774`. Consumed kernel-side as the base for core-to-core sharded transfers. **Resolution:** port re-expresses this binding via `TensorParameter`; the kernel reads the bound source from a sharded reader rather than an RTA-fed base.
- **`input_tensor` (in `reshard_program_factory_same_height.cpp` / `same_width.cpp`):** clean — uses peer reshard reader/writer kernels with shard-bound CBs.
- **`output_tensor` (all factories):** clean — sharded output bound via `CBDescriptor::buffer = <output_buffer>` (borrowed-memory CB).
- **`input_tensor` / `output_tensor` (ND-reshard factories):** clean — `TensorAccessor` end-to-end.

## Path forward

YELLOW — port is unblocked. The one Step 0.5 YELLOW is port-time work — re-express the bypassing binding in `reshard_program_factory_generic.cpp` via `TensorParameter`. No prereqs to wait for.

Suggested handoff to `port_op_to_metal2_recipe.md` once the user approves.

## Questions for the user

1. **Generic factory bypass scope:** the bypass in `reshard_program_factory_generic.cpp` is the standard cross-core sharded send/recv pattern. Confirm that re-expressing via `TensorParameter` (rather than `TensorAccessor::get_bank_base_address`) is the desired resolution.
