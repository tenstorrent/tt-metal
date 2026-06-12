# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/data_movement/slice`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** ProgramDescriptor ✓ · Device 2.0 ▲ (holdovers — see Blocked-until) · Features ✓

> **BLOCKED until Device 2.0 cleanup.** This port **cannot begin** until these isolated Device 2.0 holdovers are fixed — *separately, on the Device 2.0 track; never in the port diff*:
>
> - `ttnn/operations/data_movement/common/kernels/common.hpp` — `enhanced_noc_async_read` (~line 29–35) and `enhanced_noc_async_write` (~line 43–47) call legacy `noc_async_read_one_packet` / `noc_async_read<N>` / `noc_async_write_one_packet` / `noc_async_write<N>` free functions. These are reached from `slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp` via `tt::data_movement::common::tt_memmove<false, false, false, 0>`. The call is a local L1-to-L1 copy (misalignment correction), not external tensor access. Fix: replace the `noc_async_*` calls inside the helper with `Noc` member-form equivalents, or convert `tt_memmove` to use the Device 2.0 `Noc` object passed in (or a locally constructed one). This is `data_movement/common` team scope.
>
> Once they're clean, proceed with this brief as-is — **no re-audit needed.**

---

## TTNN factory analysis

The factory concept is selected downstream from these facts (→ `port_op_to_metal2_ttnn_factory.md`); the port does not pick it here. Carry these forward:

- **Op-owned tensors:** none
- **MeshWorkload:** not needed
- **Pybind `create_descriptor`:** delete at `slice_nanobind.cpp:161` — `nb::class_<ttnn::prim::SliceTileProgramFactory>(mod, "SliceTileProgramFactory").def_static("create_descriptor", ...)` inside `bind_slice_descriptor`
- **Other risky pybind:** `slice_nanobind.cpp:149-158` exposes `SliceDeviceOperation::create_output_tensors` and `compute_output_specs`; `slice_nanobind.cpp:131-147` exposes `SliceParams` and `SliceInputs` structs — all inside `bind_slice_descriptor`; assess / remove during port
- **Custom `override_runtime_arguments`:** none

---

## Construct — to do

**Tensor bindings** (per factory, per binding):

**`SliceRmProgramFactory`:**
- `input` — **Case 1** → re-express via `TensorParameter` / `TensorBinding`; kernel builds `TensorAccessor(ta::input)`. Remove `input_tensor.buffer()->address()` from `common_reader_kernel_args[0]` (`slice_program_factory_rm.cpp:72`).
- `output` — **Case 1** → re-express via `TensorParameter` / `TensorBinding`; kernel builds `TensorAccessor(ta::output)`. Remove `output_buffer->address()` from writer RTAs (`slice_program_factory_rm.cpp:131`).

**`SliceRmShardedProgramFactory`:**
- `input` — **clean** (borrowed-memory DFB, `CBDescriptor::buffer = input.buffer()` at `slice_program_factory_rm_sharded.cpp:265`). Port: `DataflowBufferSpec::borrowed_from = <input TensorParameter name>`.
- `output` — **clean** (borrowed-memory DFB, `CBDescriptor::buffer = output.buffer()` at `slice_program_factory_rm_sharded.cpp:278`). Port: `DataflowBufferSpec::borrowed_from = <output TensorParameter name>`.

**`SliceRmStrideProgramFactory`:**
- `input` — **Case 1** → re-express via `TensorParameter` / `TensorBinding`. Remove `Buffer* input_buffer` pointer from `emplace_runtime_args` (4D path `slice_program_factory_rm_stride.cpp:126`; ND path `slice_program_factory_rm_stride.cpp:145`); kernel reads as `src_addr = get_arg_val<uint32_t>(0)` and passes to `TensorAccessor`.
- `output` — **Case 1** → re-express via `TensorParameter` / `TensorBinding`. Remove `Buffer* output_buffer` pointer (`slice_program_factory_rm_stride.cpp:134, 158`); kernel reads as `dst_addr = get_arg_val<uint32_t>(0)`.

**`SliceTileProgramFactory`:**
- `input` — **Case 1** → re-express via `TensorParameter` / `TensorBinding`. Remove `src0_buffer->address()` from `reader_common_args[0]` (`slice_program_factory_tile.cpp:77`); kernel reads as `src_addr = get_common_arg_val<uint32_t>(0)` and passes to `TensorAccessor`.
- `output` — **Case 1** → re-express via `TensorParameter` / `TensorBinding`. Remove `dst_buffer->address()` from per-core writer RTAs (`slice_program_factory_tile.cpp:162`); kernel reads as `dst_addr = get_arg_val<uint32_t>(0)` and passes to `TensorAccessor`.

**`SliceTileTensorArgsProgramFactory`:**
- `input` — **Case 1** → re-express via `TensorParameter` / `TensorBinding`. Remove `src_buffer->address()` from `reader_common_args[0]` (`slice_program_factory_tile_tensor_args.cpp:103`); kernel reads as `src_addr = get_common_arg_val<uint32_t>(0)`.
- `start_tensor` — **Case 1** → re-express via `TensorParameter` / `TensorBinding`. Remove `start_buffer->address()` from `reader_common_args[1]` (`slice_program_factory_tile_tensor_args.cpp:104`); kernel reads as `start_addr = get_common_arg_val<uint32_t>(1)`.
- `end_tensor` — **Case 1** → re-express via `TensorParameter` / `TensorBinding`. Remove `end_buffer->address()` from `reader_common_args[2]` (`slice_program_factory_tile_tensor_args.cpp:105`); kernel reads as `end_addr = get_common_arg_val<uint32_t>(2)`.
- `output` — **Case 1** → re-express via `TensorParameter` / `TensorBinding`. Remove `dst_buffer->address()` from per-core writer RTAs (`slice_program_factory_tile_tensor_args.cpp:159`); kernel reads as `dst_addr = get_arg_val<uint32_t>(0)`.

**Custom hash:** none — nothing to delete.

---

## Watch for

- **Notable constructs:** borrowed-memory DFB @ `slice_program_factory_rm_sharded.cpp:265,278` → `DataflowBufferSpec::borrowed_from` for both input and output CBs in the sharded factory
- **Cross-op / shared kernels:**
  - `ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` is instantiated by `SliceTileTensorArgsProgramFactory` (`slice_program_factory_tile_tensor_args.cpp:177`). This is a broadly-shared cross-family kernel. Its Metal 2.0 rewrite (CB→DFB, named-token bindings) must be coordinated with eltwise/unary — slice and eltwise/unary form a **port-together set** for this kernel. Do not rewrite it in isolation.
  - `ttnn/operations/data_movement/common/kernels/common.hpp` is included by `slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp`. After the Device 2.0 cleanup (required before port), the `tt_memmove` helper will use Device 2.0 forms and the slice kernel's usage will be transparent.
- **RTA varargs:** none
