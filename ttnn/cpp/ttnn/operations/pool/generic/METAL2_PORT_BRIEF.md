# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/pool/generic`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** ProgramDescriptor ✓ · Device 2.0 ▲ *(see Blocked-until)* · Features ✓

> ⚠ **BLOCKED until Device 2.0 cleanup.** This port **cannot begin** until the following isolated Device 2.0 holdover is fixed — *separately, on the Device 2.0 track; never in the port diff*:
>
> - `device/kernels/dataflow/reader_pool_2d.cpp:88` — `get_write_ptr(in_cb_id)` (free-function form) → `in_cb.get_write_ptr()` (member form; `in_cb` is `experimental::CB in_cb(in_cb_id)` at line 53)
>
> Once this is clean, proceed with this brief as-is — **no re-audit needed.**

---

## TTNN factory analysis

The factory concept is selected downstream from these facts (→ `port_op_to_metal2_ttnn_factory.md`); the port does not pick it here. Carry these forward:

- **Op-owned tensors:** Yes — `reader_indices_tensor_owner` (always) and `scalar_config_tensor_owner` (avg-pool non-trivial-scalar path). Both allocated in `create_workload_descriptor` and parked in `WorkloadDescriptor::buffers`. See `METAL2_PREPORT_AUDIT.md` team-only section for full evidence.
- **MeshWorkload:** Not a genuine need — op-owned-tensor artifact only. `Pool2D` is a single-device op; the `WorkloadDescriptor` path is used solely to keep the op-owned buffers alive for the cached workload's lifetime.
- **Pybind `create_descriptor`:** None.
- **Other risky pybind:** None.
- **Custom `override_runtime_arguments`:** None.

---

## Construct — to do

**Tensor bindings** (per binding):

- `raw_in_cb_id` (input shard, `pool_multi_core_program_factory.cpp:456`) — **fake CB workaround**: backed by `input.buffer()`, accessed kernel-side via `in_shard_cb.get_read_ptr()` only (no FIFO). Cannot be expressed as a Metal 2.0 DFB (no producer+consumer pair). Apply the sanctioned fake-CB workaround per the porting recipe. Confirm the interpretation with the op owner before porting (see Questions in audit).
- `in_reader_indices_cb_id` L1-sharded path (`pool_multi_core_program_factory.cpp:486–491`) — **fake CB workaround**: backed by `reader_indices_buffer`, kernel reads base address only. Apply fake-CB workaround.
- `config_cb_id` L1-sharded path (`pool_multi_core_program_factory.cpp:754`) — **fake CB workaround**: backed by `scalar_config_buffer`, kernel reads base address only. Apply fake-CB workaround.
- `reader_indices_buffer` DRAM path (CTA-baked address at slot 35, `pool_multi_core_program_factory.cpp:793`) — **Case 1**: baked `reader_indices_buffer->address()` into CTA; used as base for `TensorAccessor` in `load_config_tensor_if_in_dram`. Re-express via op-owned `TensorParameter`/`TensorBinding`; CTA-baked address and `TensorAccessorArgs` wiring disappear.
- `scalar_config_buffer` DRAM path (CTA-baked address at slot 33, `pool_multi_core_program_factory.cpp:791`) — **Case 1**: same pattern as `reader_indices_buffer`. Re-express via op-owned `TensorParameter`/`TensorBinding`.
- `outputs[0]` (`out_cb_id`, `pool_multi_core_program_factory.cpp:691–698`) — **clean**: sharded output DFB; compute kernel produces via `reserve_back/push_back`. Port via `DataflowBufferSpec::borrowed_from`.
- `outputs[1]` (`out_idx_cb_id`, `pool_multi_core_program_factory.cpp:707–712`, `return_indices=true` path only) — **clean**: same pattern as `outputs[0]`. Port via `DataflowBufferSpec::borrowed_from`.

**Custom hash:** Delete custom `compute_program_hash` at `pool_op.cpp:168–185` → revert to default (sanctioned exception). Do not repair — delete entirely.

---

## Watch for

- **Aliased CBs:** `pre_tilize_cb_id` / `fast_tilize_cb_id` multi-format pair at `pool_multi_core_program_factory.cpp:657–684` — two `CBFormatDescriptor` elements in one `CBDescriptor`, allocated on the TILED output path only (`has_pre_tilize = true`). Port via `DataflowBufferSpec::advanced_options.alias_with` (ninja feature — see DFBAdvancedOptions header for legality constraints and migration guide for the porting shape).
- **Borrowed-memory DFBs:** Five `CBDescriptor::buffer = ...` sites (input shard, reader-indices L1 path, scalar-config L1 path, output, optional index output). Port via `DataflowBufferSpec::borrowed_from`.
- **Cross-op / shared kernels:** All kernel includes resolve to LLK/HAL or in-family pool shared headers — no cross-family coupling. No port-together obligation.
- **RTA varargs:** None.
