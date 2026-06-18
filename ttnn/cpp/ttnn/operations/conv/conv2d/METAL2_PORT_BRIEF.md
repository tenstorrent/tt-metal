# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/conv/conv2d`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** ProgramDescriptor ✓ · Device 2.0 ✓ · Features ✓

Op: `Conv2dDeviceOperation` (`ttnn::prim`) with two factories — `Conv2dShardedProgramFactory` (height/block/1D-depthwise) and `Conv2dWidthShardedProgramFactory` (width-sharded). They share the `conv_bmm_tilize.cpp` compute kernel and the `conv2d_op_program_factory_common.cpp` helpers; port them together.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `port_op_to_metal2_ttnn_factory.md`); the op ports to `MetalV2FactoryConcept`. Carry them forward:

- **Op-owned tensors:** Yes — `conv_reader_indices_tensor`, allocated per factory in `create_workload_descriptor` and parked on `workload_descriptor.buffers` (`conv2d_op_sharded_program_factory.cpp:1563-1572`, `conv2d_op_width_sharded_program_factory.cpp:727-736`). Wire it into `MetalV2FactoryConcept::op_owned_tensors`.
- **MeshWorkload:** not a genuine need — op-owned-tensor artifact only. Both factories sit on the `create_workload_descriptor` path solely to carry the config tensor; the per-coord program is structurally identical for every coord (the same `ProgramDescriptor` is copied into each range — `...sharded...:1574-1587`, `...width_sharded...:738-751`). Ports cleanly as morally single-program.
- **Pybind `create_descriptor`:** none.
- **Other risky pybind:** none (`conv2d_nanobind.cpp` binds only the `conv2d` function, the `prepare_conv_weights`/`prepare_conv_bias` helpers, and the `Conv2dConfig` config class).
- **Custom `override_runtime_arguments`:** none.

## Construct — to do

**Tensor bindings** (per binding; classification holds across both factories):

- **`b` (weights)** — **Case 1** (via `TensorAccessor`). Today the host pushes `Buffer* weights_buffer` as a runtime-arg `Buffer*` binding (`...sharded...:1302`, `...width_sharded...:667`); the kernel builds `TensorAccessor(s_weight_args, weight_addr_dram_base)` from the base (`reader_writer_tiled_out_1d_mcast_sender_...:135`, `writer_tiled_out_2d_mcast_sender_...:158`, `weights_reader_width_sharded.cpp:42`). Express as `TensorParameter`; kernel uses `TensorAccessor(ta::weights)`. The `Buffer*` arg + `TensorAccessorArgs` plumbing disappear.
- **`bias` (optional)** — **Case 1**, same shape (`...sharded...:1304`, `...width_sharded...:669` → `TensorAccessor(s_bias_args, bias_addr)`). Keep the `has_bias` kernel-side gate; when absent the host pushes literal `0`.
- **`conv_reader_indices` (op-owned config tensor)** — **Case 1** in the `config_tensors_in_dram == true` path (CT-baked address + `TensorAccessorArgs(...).append_to(...)` at `...sharded...:873-875`, `...width_sharded...:569-571`; kernel rebuilds a `TensorAccessor`, `conv_reader_common.hpp:361-364`). In the L1-small path it is the `READER_INDICES` borrowed-memory DFB (below). Bind as a `TensorParameter` over the op-owned tensor.
- **`a` (activation input shard)** — borrowed-memory DFB (`ACT_SHARDED` on the input buffer); but see the fake-CB note under Watch for.
- **Output (`OUT` / `MATMUL_PARTIALS`)** — borrowed-memory DFB on the output buffer (compute produces, writer consumes) → `borrowed_from = ta::out`.

**Custom hash:** delete custom `compute_program_hash` (`device/conv2d_device_operation.cpp:145-166`) → default (sanctioned exception). The custom hash also omits `groups` from its key (see audit Misc anomalies); the default fixes this incidentally — do not try to repair the custom hash.

## Watch for

- **Notable constructs:**
  - **Borrowed-memory DFBs** → `DataflowBufferSpec::borrowed_from`. Emitted by `emit_cb_descriptors` (`conv2d_op_program_factory_common.cpp:770-795`) for `ACT_SHARDED` (input), `OUT`/`MATMUL_PARTIALS` (output), `READER_INDICES` (config, L1-small path). Name the backing `TensorParameter` in each.
  - **`ACT_SHARDED` is a fake CB** (address-only) at the reader read-ptr endpoint — the resident input shard is read purely by base pointer (`reader_conv_activations_padded_with_halo_3x3_weights_v2.cpp:77`, `reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2.cpp:216`, `activation_reader_width_sharded.cpp:138,149`) with no kernel producer. Use the sanctioned fake-CB workaround (recipe); do not try to express it as a producer/consumer DFB. Contrast `READER_INDICES`, which *is* a real borrowed-DFB (reader writes config in, a `wait_front(1)` consumes — `reader_writer_tiled_out_1d_mcast_sender_...:103`).
  - **`conv_bmm_tilize.cpp` manual CB FIFO-pointer writes** via `get_local_cb_interface(cb_id).fifo_*_ptr` (e.g. `:71`, `:77`, `:369`, partials spill/reload around `:519-565`) are load-bearing (activation reuse, split reader, partials spill/reload) and sanctioned — leave them; there is no OO wrapper equivalent.
- **Cross-op / shared kernels:** every conv2d kernel includes `ttnn/cpp/ttnn/operations/pool/device/kernels/experimental_device_api.hpp` (pool's Device-2.0 convenience header, shared with pool). `conv_bmm_tilize.cpp` / `compute_depthwise_conv1d.cpp` also include `ttnn/cpp/ttnn/kernel_lib/{tilize,untilize}_helpers.hpp`. A Metal 2.0 rewrite of the shared pool header would need conv2d + pool migrated together — coordinate that change.
- **RTA varargs:** none.
