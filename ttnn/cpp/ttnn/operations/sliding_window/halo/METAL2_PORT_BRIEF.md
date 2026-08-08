# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/sliding_window/halo`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** ProgramDescriptor ✓ · Device 2.0 ✓ · Features ✓

## TTNN factory analysis

The factory concept is selected downstream from these facts (→ `port_op_to_metal2_ttnn_factory.md`); the port does not pick it here. Carry these forward:

- **Op-owned tensors:** Yes — `UntilizeWithHaloProgramFactory::create_workload_descriptor` allocates four halo config device tensors (`pad_config_device_tensor0/1`, `gather_config_device_tensor0/1`) and parks them on `workload_descriptor.buffers` (`untilize_with_halo_program_factory.cpp:436–487`).
- **MeshWorkload:** Not needed — op-owned-tensor artifact only (the `WorkloadDescriptor` path was taken to keep config tensor lifetimes tied to the cached workload, not for genuine cross-device coordination).
- **Pybind `create_descriptor`:** None.
- **Other risky pybind:** None.
- **Custom `override_runtime_arguments`:** None.

## Construct — to do

**Tensor bindings** (per binding):

- `input_tensor` (declared input) — backed by `src_buffer`; set as `CBDescriptor::buffer = src_buffer` (`add_cb` call at factory line 152). Kernel: `src_cb` with `reserve_back/push_back` (producer, `halo_gather.cpp:317–318`) and `wait_front` / `pop_front` (consumer in `run_halo_gather`). Real borrowed-memory DFB with proper FIFO pair. **Case 1** → re-express via `TensorParameter` / `TensorBinding`; kernel builds `TensorAccessor(ta::name)` for this tensor.

- `output_tensor` (declared output) — backed by `dst_buffer`; set as `CBDescriptor::buffer = dst_buffer` (`add_cb` call at factory line 166). Kernel: `out_cb.get_write_ptr()` only (lines 108, 205 of `halo_gather.cpp`) — **fake CB** (no FIFO producer+consumer). Port: **Case 1** + fake-CB workaround (see port recipe). The binding flows through `TensorParameter`; the address-only access in the kernel uses the fake-CB workaround instead of a DFB.

- `padding_config_buffer0/1`, `gather_config_buffer0/1` (op-owned config tensors) — two code paths:
  - When `!config_tensors_in_dram`: set as `CBDescriptor::buffer = <config_buf>` (factory lines 238, 247, 257, 266). Kernel: `padding_config_cb.get_read_ptr()` (line 105) / `gather_config_cb.get_read_ptr()` (line 343) — **fake CBs**. Port resolves with fake-CB workaround; op-owned-tensor Metal 2.0 handling governs how the address reaches the kernel (consult `port_op_to_metal2_ttnn_factory.md`).
  - When `config_tensors_in_dram`: `buffer->address()` baked into CTAs (factory lines 307–317) + `TensorAccessorArgs(buffer).append_to(...)` also in CTAs (lines 319–323). Kernel: `TensorAccessorArgs<N>()` + `TensorAccessor(args, addr)` (lines 296–301 of `halo_gather.cpp`). **Case 1** — replace with `TensorParameter` / `TensorBinding`; CTA-baked address and `TensorAccessorArgs<N>()` dance both disappear; kernel builds `TensorAccessor(ta::name)`.

**Custom hash:** None.

## Watch for

- **Notable constructs:** Borrowed-memory DFB on `src_cb`, `out_cb`, and four config-tensor CBs (all via `CBDescriptor::buffer` non-null in the `add_cb` helper at `untilize_with_halo_program_factory.cpp:76`). Port: each uses `DataflowBufferSpec::borrowed_from` — except the three fake CBs (`out_cb`, `padding_config_cb`, `gather_config_cb`), which use the fake-CB workaround instead.

- **Fake CBs:** Three address-only CBs require the fake-CB workaround (see port recipe):
  - `out_cb` (`out_cb_id`, dst_buffer) — `halo_gather.cpp:108,205`
  - `padding_config_cb` (`padding_config0/1`) — `halo_gather.cpp:105`
  - `gather_config_cb` (`gather_config0/1`) — `halo_gather.cpp:343`

- **Cross-op / shared kernels:**
  - `halo_gather.cpp` includes `ttnn/operations/pool/device/kernels/experimental_device_api.hpp` (cross-family pool donor — Device 2.0 wrapper utilities only; no porting friction).
  - `pack_untilize.cpp` uses `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` (shared kernel lib; Device 2.0 compliant). If the kernel lib is updated as part of another op's port, confirm the update is compatible.

- **RTA varargs:** None.
