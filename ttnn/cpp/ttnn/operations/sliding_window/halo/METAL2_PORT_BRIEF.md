# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/sliding_window/halo`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** ProgramDescriptor ✓ · Device 2.0 ✓ · Features ✓

DOp / factory: `HaloDeviceOperation` → `UntilizeWithHaloProgramFactory` (`device/untilize_with_halo_program_factory.cpp`). Kernels: `device/kernels/dataflow/halo_gather.cpp` (split reader, RISCV_0 + RISCV_1), `device/kernels/compute/pack_untilize.cpp` (only when input is tiled).

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `port_op_to_metal2_ttnn_factory.md`); the op ports to `MetalV2FactoryConcept`. Carry them forward:

- **Op-owned tensors:** Yes — four intermediate halo-config device tensors (`pad_config0/1`, `gather_config0/1`) allocated in `create_workload_descriptor` (`untilize_with_halo_program_factory.cpp:436-459`) and parked on `workload_descriptor.buffers` (`:473-487`). Carry these via `MetalV2FactoryConcept::op_owned_tensors`.
- **MeshWorkload:** Not a genuine need — op-owned tensors only. The op is on the `create_workload_descriptor` / `WorkloadDescriptor` path purely to carry the four owned config tensors (the in-code comment at `:492-495` confirms single-device, structurally identical per-coord program). Ports cleanly as single-program.
- **Pybind `create_descriptor`:** none.
- **Other risky pybind:** none.
- **Custom `override_runtime_arguments`:** none.

## Construct — to do

**Tensor bindings** (per binding):

- **input tensor → `src_cb`** (`untilize_with_halo_program_factory.cpp:152`, `.buffer = src_buffer`) — **clean / borrowed-memory DFB.** Declare `DataflowBufferSpec::borrowed_from = <input TensorParameter>`. Producer = reader_0 fake-push (`halo_gather.cpp:317-318`); consumers = compute `untilize` (reads `src_cb_id`) + reader skip-untilize `src_cb.wait_front` (`:340`). **Confirm** the framework accepts one borrowed DFB bound to all three kernel instances (2 dataflow + 1 compute) before relying on it — this fan-out is broader than the recipe's two-endpoint sketch.
- **output tensor → `out_cb`** (`:166`, `.buffer = dst_buffer`) — **fake CB** (address-only over the borrowed output buffer; written by raw `noc.async_write` off `out_cb.get_write_ptr()`, no FIFO). Apply the sanctioned **fake-CB workaround** (see the porting recipe) over the borrowed output buffer.
- **config tensors → `padding_config0/1`, `gather_config0/1`** — split by code path:
  - **DRAM path** (`config_tensors_in_dram == true`): **Case 1** (via `TensorAccessor`). Today `buffer->address()` is baked into the reader's *compile-time* args (`:307-317`) and consumed by `TensorAccessor(args, addr)` (`halo_gather.cpp:300-301`). Express each config tensor as a `TensorParameter`; the CTA-baked address + kernel-side `TensorAccessorArgs<N>()` NTTP base both disappear, replaced by `TensorAccessor(ta::name)`.
  - **L1 path** (`config_tensors_in_dram == false`): **fake CB** (`.buffer` set @ `:238,248,258,268`; read-by-pointer via `*_config_cb.get_read_ptr()`, no FIFO). Apply the fake-CB workaround.
  - Record this per-path split via the Per-DeviceOperation/per-path attribution — same binding, two treatments.
- **`pad_cb0` / `pad_cb1`** (`:172-174`) — **fake CB** (scratch immediate-value buffers, no `.buffer`, `get_write_ptr`/`get_read_ptr` only). Fake-CB workaround.

**Custom hash:** none — leave the default reflection hash in place.

## Watch for

- **Notable constructs:**
  - Borrowed-mem DFB → `DataflowBufferSpec::borrowed_from`: `src_cb` @ `:152`, `out_cb` @ `:166`, config CBs (L1 path) @ `:238,248,258,268`.
  - Fake CBs (no producer/consumer FIFO) → fake-CB workaround: `out_cb` @ `:166`, `pad_cb0/1` @ `:172-174`, L1-path config CBs @ `:230-268`.
- **Cross-op / shared kernels:** no file-path kernel borrow (op owns both `.cpp`s). Two shared *header* includes only — `pool/.../experimental_device_api.hpp` and `kernel_lib/untilize_helpers.hpp` — both Device 2.0; do not modify them in the halo port (lib/shared-pool surface).
- **RTA varargs:** none — reader RTAs are single fixed scalars (`core_index` / `config_read_index`), compute RTA is a single `total_blocks`. Use named RTAs.
- **Note (not port scope):** `remote_read` is plumbed to CTA index 10 but `halo_gather.cpp:277` hard-`static_assert(!remote_read)`s — the kernel does not support it. Leave as-is.
