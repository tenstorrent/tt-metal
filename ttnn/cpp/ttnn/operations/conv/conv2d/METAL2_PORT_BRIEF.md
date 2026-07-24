# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/conv/conv2d`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** ProgramDescriptor ✓ · Device 2.0 ✓ (one open question on `fifo_wr_ptr` writes — see Watch-for) · Features ✓

> **Note:** There is one open question (Device 2.0) — `conv_reader_common.hpp:91,109` writes directly to `get_local_cb_interface(cb_id_act).fifo_wr_ptr` to reposition a hardware FIFO write pointer.  This is treated as **probably sanctioned** (the migration guide keeps `get_local_cb_interface` as a free function for architectural reasons; the activation-reuse code path appears to require direct pointer repositioning).  If the user confirms it is an **unsanctioned holdover**, the Device 2.0 gate becomes YELLOW and those two sites must be fixed on the Device 2.0 track **before** porting begins — *not* in the port diff. Until that confirmation, the brief issues as written.

## TTNN factory analysis

The factory concept is selected downstream from these facts (→ `port_op_to_metal2_ttnn_factory.md`); the port does not pick it here.  Carry these forward:

- **Op-owned tensors:** Yes — `conv_reader_indices_tensor` allocated in both factories.
  - `Conv2dShardedProgramFactory::create_workload_descriptor` — `conv2d_op_sharded_program_factory.cpp:1563–1572`
  - `Conv2dWidthShardedProgramFactory::create_workload_descriptor` — `conv2d_op_width_sharded_program_factory.cpp:727–736`
- **MeshWorkload:** op-owned-tensor artifact only (not a real need).  Both factories call `create_workload_descriptor` solely to park `conv_reader_indices_tensor` on `WorkloadDescriptor::buffers` — the op is single-device.
- **Pybind `create_descriptor`:** none.
- **Other risky pybind:** none.  `conv2d_nanobind.cpp` binds only the user-facing `conv2d` function, `Conv2dConfig`, `prepare_conv_weights`, and `prepare_conv_bias`.
- **Custom `override_runtime_arguments`:** none.

## Construct — to do

**Tensor bindings** (per binding, both factories):

- `weights` (tensor `b`) — **Case 1** → re-express via `TensorParameter` / `TensorBinding`.  Host side: replace `emplace_runtime_args(core, {weights_buffer, …})` (`Buffer*` binding form) with a named `TensorParameter`.  Kernel side already constructs `TensorAccessor(s_weight_args, weight_addr)` — swap that to `TensorAccessor(ta::weights)`.
  - Sharded: `conv2d_op_sharded_program_factory.cpp:1302`
  - Width-sharded: `conv2d_op_width_sharded_program_factory.cpp:667`
- `bias` (optional tensor) — **Case 1** → re-express via `TensorParameter` / `TensorBinding`.  Same `Buffer*` form when bias is present.  Kernel side: `TensorAccessor(s_bias_args, bias_addr)` → `TensorAccessor(ta::bias)`.
  - Sharded: `conv2d_op_sharded_program_factory.cpp:1304`
  - Width-sharded: `conv2d_op_width_sharded_program_factory.cpp:669`
- `conv_reader_indices` (op-owned config tensor) — **Case 1** → re-express via `TensorParameter` / `TensorBinding`.  Host side: `conv_reader_indices_buffer->address()` baked into CTAs (gated on `config_tensors_in_dram`); also `TensorAccessorArgs(conv_reader_indices_buffer).append_to(…)` appended to CTAs.  Kernel side: `TensorAccessorArgs<…>()` + `TensorAccessor(config_tensor_args, config_dram_addr)` in `conv_reader_common.hpp:361–364` — swap to `TensorAccessor(ta::reader_indices)`.
  - Sharded: `conv2d_op_sharded_program_factory.cpp:873–875` (CTA-baked address + TensorAccessorArgs)
  - Width-sharded: `conv2d_op_width_sharded_program_factory.cpp:569–571`
- `activations` (tensor `a`) — **clean** (borrowed-memory DFB via `ACT_SHARDED` CB, `CBDescriptor::buffer = input_buffer`).
- `output` — **clean** (borrowed-memory DFB via `OUT` / `MATMUL_PARTIALS` CBs, `CBDescriptor::buffer = output_buffer`).

**Custom hash:** delete custom `compute_program_hash` → default (sanctioned exception).
- Declaration: `conv2d_device_operation.hpp:44–45`
- Definition: `conv2d_device_operation.cpp:144–165`

## Watch for

- **Notable constructs — Dynamic CircularBuffer (borrowed-memory DFBs):** `ACT_SHARDED`, `OUT`, `MATMUL_PARTIALS`, and `READER_INDICES` CBs carry `CBDescriptor::buffer = <buffer>` — already expressed as Metal 2.0 `DataflowBufferSpec::borrowed_from`.  No action; the borrowed-from step is already done.  Sites: `conv2d_op_program_factory_common.cpp:756` (`emit_cb_descriptors`).

- **Fake CB — `READER_INDICES` in L1 mode:** when `config_tensors_in_dram = false`, the `READER_INDICES` borrowed-memory CB is accessed as a raw L1 address only — no push_back / pop_front producer-consumer pair in the L1 path.  The port resolves this with the sanctioned fake-CB workaround (see `port_op_to_metal2_recipe.md`); it does **not** gate.  Site: all reader kernels calling `cb_reader_idx.get_write_ptr()` on the L1 path; `conv_reader_common.hpp` `load_config_tensor_if_in_dram` template (the `#ifdef CONFIG_TENSOR_IN_DRAM` guard is what makes the non-DRAM path fake).

- **Cross-op donor — pool `experimental_device_api.hpp`:** `ttnn/operations/pool/device/kernels/experimental_device_api.hpp` is `#include`d by all dataflow kernels.  This is a Device 2.0 convenience header — `experimental::CB`, `Noc` aliases, `read_with_state` / `set_read_state` templates.  No action needed; it is Device 2.0 clean (Shape 1 / ✓ excellent) and will port alongside the conv2d kernels.

- **Shared kernel library:** `conv_bmm_tilize.cpp` and `compute_depthwise_conv1d.cpp` include `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` and `untilize_helpers.hpp`.  Both headers are Device 2.0 clean; coordinate the port-time rename/token update with the kernel_lib team if those helpers are shared with other ops that have not yet ported.

- **RTA varargs:** none.

- **Open question — `fifo_wr_ptr` write (Device 2.0):** before the port diff lands, confirm with the Device 2.0 team that `get_local_cb_interface(cb_id_act).fifo_wr_ptr = ...` writes in `conv_reader_common.hpp:91` and `:109` are sanctioned (architectural, activation-reuse CB pointer repositioning) or flag them as holdovers to clean first.  If a holdover: fix on the Device 2.0 track; **do not** include in the port diff.
