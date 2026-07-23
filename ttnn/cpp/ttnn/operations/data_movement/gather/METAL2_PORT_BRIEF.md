# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/data_movement/gather`

> **Scoped subset port.** The op is RED at op level, but the two **TILE-layout factories** clear every gate and form a clean, portable subset. This brief covers **only** that subset. The full record — including why the ROW_MAJOR factories are held back — is in `METAL2_PREPORT_AUDIT.md`.

**In scope for this port:**
- `GatherDeviceOperation::SingleRowSingleCore` (`device/gather_program_factory.cpp:50`) + kernels `gather_reader_single_row_single_core.cpp`, `gather_writer_single_row_single_core.cpp`
- `GatherDeviceOperation::SingleRowMultiCore` (`device/gather_program_factory.cpp:206`) + kernels `gather_reader_single_row_multi_core.cpp`, `gather_writer_single_row_multi_core.cpp`

**Out of scope (do not port yet):** `RmSingleRowSingleCore`, `RmSingleRowMultiCore` (the ROW_MAJOR factories). They have no readiness-sheet row, so their portability gate has not been decided; blocked pending the sheet owner adding and classifying them. Leave both factories, their four RM kernels, and the `select_program_factory` branch that routes to them untouched.

**Gates cleared (TILE subset):** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓ (no 3rd arg present)

**Recipe docs:** `1e584828bbf 2026-07-23 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper` *(carry this line into the port report's Provenance section)*

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); both TILE factories port to `MetalV2FactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` (both factories return a `ProgramDescriptor` from `create_descriptor`).
- **Op-owned tensors:** none.
- **Target concept:** `MetalV2FactoryConcept`.
- **Gate-cleared, confirmed absent** (each would have blocked the brief): custom hash · custom `override_runtime_arguments` / `get_dynamic_runtime_args` · pybind `create_descriptor` — all `no` on the sheet and code-confirmed.

## Construct — to do

**Tensor bindings** (per binding — all three are **Case 1**, via `TensorAccessor`; mechanical, low-risk):

- `input_index_tensor` — **Case 1**. Bound today as a `Buffer*` runtime arg (`reader_desc.emplace_runtime_args(core, {input_index_tensor_buffer, ...})`), consumed via `TensorAccessor(input_index_tensor_args, input_index_tensor_buffer_addr)` in the reader. → express as a `TensorParameter` / `TensorBinding`; the reader builds `TensorAccessor(tensor::name)`. The `Buffer*` RTA and the `input_index_tensor_args` CTAs both disappear.
- `input_tensor` — **Case 1**. `Buffer*` RTA in the writer; consumed via `TensorAccessor(input_tensor_args, input_tensor_buffer_addr)`. → `TensorParameter` / `TensorBinding`; writer uses `TensorAccessor(tensor::name)`.
- `output_tensor` — **Case 1**. `Buffer*` RTA in the writer; consumed via `TensorAccessor(output_tensor_args, output_tensor_buffer_addr)`. → `TensorParameter` / `TensorBinding`; writer uses `TensorAccessor(tensor::name)`.

The remaining runtime args are ordinary scalars (`work_per_core`, `tile_width`, `tile_height`, `core_id`) — port as named RTAs, inferring each name from the kernel's `get_arg_val` unpack site. The idle-core arm passes `0u` placeholders where the active arm passes a `Buffer*`; preserve that active/idle split when converting the bindings.

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none — every `TensorAccessor` in the four TILE kernels is 2-arg.

**CB endpoints:** three CBs, same disposition on every node, all configs:
- `c_1` input-index CB → **self-loop** (bind the reader as both PRODUCER and CONSUMER — the reader is its only toucher).
- `c_0` input-tensor CB → plain 1:1 (writer PRODUCER, reader CONSUMER) — no special action.
- `c_2` output CB → plain 1:1 (reader PRODUCER, writer CONSUMER) — no special action.

Metadata lookups the kernels currently make as free functions — `get_tile_size(cb)`, `get_dataformat(cb)`, `get_tile_hw(cb)` — move onto the bound `DataflowBuffer` object as the whitelisted syntax swap (they are sanctioned Device 2.0 today; the port relocates them).

## Watch for

- **CB endpoints (multi-binding):** none — no hidden second writer, no multi-reader, no dual-instance work-split. `c_1` is a straightforward single-toucher self-loop.
- **Cross-op / shared kernels:** none in scope — the four TILE kernels include only `gather_common.hpp` (in-directory) and `tt_metal/*` HAL. (The cross-op donor `data_movement/common/kernels/common.hpp` is used *only* by the out-of-scope RM kernels.)
- **RTA varargs:** none — all RTAs are read at fixed distinct indices; port them as named args, not varargs.
- **Idle-core binding arm:** both TILE factories emit a `0u`-placeholder RTA set for cores with `work_per_core == 0` (deliberately omitting the `BufferBinding` so cache-hit patching skips idle cores). Keep the active/idle distinction intact when converting the `Buffer*` args to typed bindings.
