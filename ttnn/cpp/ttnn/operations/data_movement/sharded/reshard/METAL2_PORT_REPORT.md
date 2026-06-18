# Metal 2.0 Port Report — Reshard legacy factories

Ported the three remaining legacy `ProgramDescriptor` reshard factories to the Metal 2.0 host API
(`create_program_artifacts` → `ProgramSpec` + `ProgramRunArgs`). They now satisfy
`MetalV2FactoryConcept`, the same path the `NdReshard*` factories already use. `select_program_factory`
in `reshard_device_operation.cpp` is unchanged — the device-operation adapter dispatches each factory
by concept, so a mixed legacy/Metal-2.0 variant is fine.

## Files changed

Factories (signature `create_descriptor` → `create_program_artifacts`):
- `device/reshard_program_factory_same_height.{hpp,cpp}`
- `device/reshard_program_factory_same_width.{hpp,cpp}`
- `device/reshard_program_factory_generic.{hpp,cpp}`

Forked kernels (new copies under the reshard op dir; legacy copies under
`sharded/device/kernels/dataflow/` left untouched):
- `device/kernels/dataflow/reshard_same_height_reader.cpp`, `reshard_same_height_writer.cpp`
- `device/kernels/dataflow/reshard_same_width_reader.cpp`, `reshard_same_width_writer.cpp`
- `device/kernels/dataflow/reshard_reader.cpp`, `reshard_reader_diff_width.cpp`

No CMake change needed (factory `.cpp` filenames preserved; kernels are JIT-compiled).

## Per-factory mapping

| Factory | Selected when | Notes |
|---|---|---|
| `ReshardSameWidthFactory<true/false>` | HEIGHT_SHARDED ↔ HEIGHT_SHARDED (`true`=out L1, `false`=out DRAM) | adds a scratch DFB + `UNALIGNED` define on the reader path when alignments differ |
| `ReshardSameHeightFactory<true/false>` | ROW_MAJOR WIDTH_SHARDED ↔ WIDTH_SHARDED, no padding | padded width falls through to generic |
| `ReshardGenericFactory` | everything else valid for legacy reshard (BLOCK↔BLOCK, TILE, padded width, cross-layout) | two kernel variants: `reshard_reader.cpp` (same page size) / `reshard_reader_diff_width.cpp` |

## Key transformations

- **Local sharded CB → borrowed DFB**: the local/output sharded CB (legacy `cb.buffer = buffer`) is a
  `DataflowBufferSpec` with `borrowed_from = <local TensorParameter>`. The kernel reads the base L1
  address from `DataflowBuffer(dfb::...).get_write_ptr()` / `get_read_ptr()`. Producer bound on the
  reader endpoint, consumer on the writer endpoint, satisfying the DFB ≥1-producer/≥1-consumer rule
  (the CB carries no FIFO traffic — it is an address source only).
- **Remote-buffer base-address RTA (`Buffer*`/`address()`) → `TensorAccessor`**: dropped from runtime
  args; the kernel now uses `TensorAccessor(ta::remote|ta::input).get_bank_base_address()` (Case 2).
- **Positional RTAs → named RTAs + positional varargs**: scalar per-core values are named RTAs; the
  variable-length per-segment / per-transfer / per-range tail is positional varargs read with
  `get_vararg(i)`. Since `num_runtime_varargs_per_node` is deprecated, the uniform `num_runtime_varargs`
  is set to the max-over-cores tail length and each core's vector is **zero-padded** to that length;
  kernels only read the real prefix (bounded by `num_segments`/`num_reads`/`num_ranges`).
- **Generic input-addr slot drop**: the generic kernels random-index into a leading
  `physical_core_coords` block and read the rest sequentially. The whole legacy positional vector is
  emitted as one vararg block with the `input_addr` element **erased**, keeping both the random indices
  (front block intact) and the sequential reads (shifted to skip the removed slot) aligned.

## Behavior preserved

No algorithmic changes. The generic factory still recomputes page ranges per core (legacy inefficiency
kept as-is). Uniform scalars are emitted as per-node named RTAs (CRTA promotion left as a future
optimization). The legacy writer-path scratch CB that was allocated-but-unused is simply not created in
Metal 2.0 (it would otherwise be an unbound DFB) — a behavior-preserving dead-allocation removal.

## Build note

Local `clang` lints report `'ttnn/device_operation.hpp' file not found` etc. — these are include-path
resolution failures of the standalone language server (the same headers the working `NdReshard*`
factories include), not code errors. A normal CMake build supplies the include dirs.
