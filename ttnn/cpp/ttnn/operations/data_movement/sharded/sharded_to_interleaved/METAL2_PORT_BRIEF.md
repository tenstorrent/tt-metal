# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/data_movement/sharded/sharded_to_interleaved`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** ProgramDescriptor ✓ · Device 2.0 ✓ · Features ✓

---

## TTNN factory analysis

The factory concept is selected downstream from these facts (→ `port_op_to_metal2_ttnn_factory.md`); the port does not pick it here. Carry these forward:

- **Op-owned tensors:** none
- **MeshWorkload:** not needed
- **Pybind `create_descriptor`:** none
- **Other risky pybind:** none
- **Custom `override_runtime_arguments`:** none

---

## Construct — to do

**Tensor bindings:**

- `input_tensor` (src_buffer, sharded input CB `c_0`) — **clean** via borrowed-memory DFB. The input CB is `CBDescriptor::buffer = src_buffer` (`push_s2i_cb_pair`, factory line 147). Port: declare `TensorParameter input_tensor` and set `DataflowBufferSpec::borrowed_from = input_tensor` on the input CB's `DataflowBufferSpec`. No RTA change needed — the reader kernel reads only `num_tiles_per_core` (a shape value, not a buffer address).

- `output_tensor` (dst_buffer) — **Case 1** → re-express via `TensorParameter` / `TensorBinding`. The factory currently pushes `dst_buffer` (a `Buffer*`) as RTA arg 0; the framework patches it as a `BufferBinding`. Both writer kernels consume it as `get_arg_val<uint32_t>(0)` and initialize `TensorAccessor(dst_args, dst_addr)` (tile-layout writer) or `TensorAccessor(dst_args, dst_addr + input_width_offset_bytes)` (stick-layout writer). Port: declare `TensorParameter output_tensor`, inject via `TensorBinding`, and have the kernels build `TensorAccessor(ta::output_tensor)`. The `TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args)` CTA dance disappears — the binding token handles it. **See open question in `METAL2_PREPORT_AUDIT.md` Q1** about the stick-layout offset-base pattern before finalizing.

**Custom hash:** none.

---

## Watch for

- **Borrowed-memory DFB (Dynamic CircularBuffer):** input CB at `device/sharded_to_interleaved_program_factory.cpp:41` (`cb.buffer = src_buffer`). Port via `DataflowBufferSpec::borrowed_from = input_tensor`. Reader kernel produces (`cb.push_back`) and writer consumes (`cb_out.wait_front`) — genuine DFB, not a fake CB.

- **Cross-op / shared kernels (port-together sets):**
  1. `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp` — cross-family donor (eltwise/unary). Shared broadly across data_movement ops (untilize_with_unpadding, untilize, tilize, transpose_wh_sharded, slice_write, and others). Its Metal 2.0 rewrite must cover all co-borrowers in one change. Rewrite is trivial (no tensor address RTAs — pure `cb.push_back`).
  2. `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded_blocks_interleaved_start_id.cpp` and `writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp` — in-family (data_movement/sharded). Also used by `sharded_to_interleaved_partial`. Rewrite must cover both ops in the same change.
  3. `ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp` — shared kernel library. Also used by `untilize_with_unpadding`, `sharded_to_interleaved_partial`, `copy` (copy_same_memory_config). Only active on the `convert_df=true` path. Port-together rewrite required.

- **RTA varargs:** none.
