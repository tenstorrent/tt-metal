# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/data_movement/sharded/reshard`

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

### Tensor bindings (per binding, per factory)

**`NdReshardCopyPagesFactory`** and **`NdReshardCopyLocalShardFactory<T/F>`** — both clean:

- `input` — **clean** (proper `TensorAccessorArgs` + `TensorAccessor` end-to-end). No port work on these bindings.
- `output` — **clean** (same). No port work.

**`ReshardGenericFactory`:**

- `input` — **Case 1 (assumed — confirm with user first, see below)** → re-express via `TensorParameter` / `TensorBinding`; kernel replaces `input_shard_addr = get_arg_val<uint32_t>(...)` with a `TensorAccessor`-sourced base, or bridges via `get_bank_base_address` if user confirms Case 2. The `Buffer*` at RTA index `grid.x + grid.y` (`reshard_program_factory_generic.cpp` lines 785, 793) goes away; host-side stride-table RTAs remain as-is.
- `output` — **fake CB** (see Watch for — no `TensorParameter` needed; the fake-CB workaround in the porting recipe handles the L1 address sourcing).

> **Before constructing the `ReshardGenericFactory` input binding:** Ask the user whether the stride-table NoC walk in `reshard_reader.cpp` / `reshard_reader_diff_width.cpp` is Case 2. The access is a host-precomputed cross-core stride walk (`{.noc_x = core_id_x, .noc_y = core_id_y, .addr = input_shard_addr + addr_offset}`) — not standard page iteration. If Case 2 is confirmed: bind the tensor as a `TensorParameter`, pull the base via `TensorAccessor::get_bank_base_address`, and leave the stride-table NoC walk in place. Message for user if Case 2: "The use of `TensorAccessor` is an ergonomic choice on Gen1 architectures. It has meaningful performance implications on Gen2 architectures. Ideally, `TensorAccessor` should be updated to support the required iteration pattern; consider filing an issue requesting that support."

**`ReshardSameWidthFactory<T/F>`:**

- `remote` — **Case 1 (assumed — confirm with user first)** → re-express via `TensorParameter` / `TensorBinding`; the `Buffer*` at `kernel_args[0]` (`reshard_program_factory_same_width.cpp` line 156) goes away. Kernel replaces `src_addr` / `dst_addr = get_arg_val<uint32_t>(0)` with a `TensorAccessor`-sourced base, or bridges via `get_bank_base_address` if user confirms Case 2 (bank-id-keyed access via `{.bank_id = bank_id, .addr = remote_addr + offset}`).
- `local` — **fake CB** (see Watch for).

> **Before constructing the `ReshardSameWidthFactory` remote binding:** Same Case-2 question as above. The kernel uses pre-computed `bank_id` values from the host to address the remote shard. Ask the user.

**`ReshardSameHeightFactory<T/F>`:**

- `remote` — **Case 1 (assumed — confirm with user first)** → `Buffer*` at `runtime_args_0[3]` / `runtime_args_1[3]` (`reshard_program_factory_same_height.cpp` lines 127, 133) goes away. Kernel replaces `base_read_addr` / `base_write_addr = get_arg_val<uint32_t>(3)` with a `TensorAccessor`-sourced base, or Case 2 bridge.
- `local` — **fake CB** (see Watch for).

> **Before constructing the `ReshardSameHeightFactory` remote binding:** Same Case-2 question.

### Custom hash

None to delete.

---

## Watch for

- **Fake CBs (address-only):** Six CBs, one per legacy kernel endpoint, are used purely as L1 address sources — no FIFO producer-consumer pair. Each `cb.buffer = non_null_buffer` + `cb.get_write_ptr()` / `cb.get_read_ptr()` sequence is a fake CB. The port resolves each with the sanctioned fake-CB workaround (see the porting recipe).
  - `ReshardGenericFactory` output: `shard_cb` at `reshard_reader.cpp:30` and `reshard_reader_diff_width.cpp:30`
  - `ReshardSameWidthFactory` local: `shard_cb_id` at `reshard_same_width_reader.cpp:37` and `reshard_same_width_writer.cpp:37`
  - `ReshardSameHeightFactory` local: `shard_cb_id` at `reshard_same_height_reader.cpp:31` and `reshard_same_height_writer.cpp:31`

- **Dynamic CircularBuffer recognition signal fires but is overridden by fake-CB litmus:** The `cb.buffer = <Buffer*>` field is set in the three legacy factories. The Metal 2.0 LANDED path (`DataflowBufferSpec::borrowed_from`) does **not** apply here — apply the fake-CB workaround, not `borrowed_from`.

- **Cross-op / shared kernels:** None. All kernel files are exclusively used by this op. No port-together coupling.

- **RTA varargs:** The `reshard_reader.cpp` and `reshard_reader_diff_width.cpp` kernels loop over a runtime-count of range/block entries from RTAs. This is standard RTA loop retrieval, not CTA varargs. Prefer named RTAs or the Metal 2.0 runtime vararg mechanism at port time per the port recipe's kernel-side whitelist.
