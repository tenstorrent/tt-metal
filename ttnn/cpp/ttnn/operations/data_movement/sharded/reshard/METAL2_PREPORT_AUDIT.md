# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/data_movement/sharded/reshard`

**Op:** `ttnn::reshard` (primitive: `ttnn::prim::reshard`)

Device operations and factories in this directory:

- **`ReshardDeviceOperation`** (`device/reshard_device_operation.hpp`, `.cpp`)
  - `ReshardSameWidthFactory<true>` / `ReshardSameWidthFactory<false>` (`device/reshard_program_factory_same_width.cpp`)
  - `ReshardSameHeightFactory<true>` / `ReshardSameHeightFactory<false>` (`device/reshard_program_factory_same_height.cpp`)
  - `ReshardGenericFactory` (`device/reshard_program_factory_generic.cpp`)
  - `NdReshardCopyPagesFactory` (`device/nd_reshard_program_factory_copy_pages.cpp`)
  - `NdReshardCopyLocalShardFactory<true>` / `NdReshardCopyLocalShardFactory<false>` (`device/nd_reshard_program_factory_copy_local.cpp`)

All eight `program_factory_t` variants belong to the single `ReshardDeviceOperation` and share one DeviceOperation class. This is one bundled audit.

**Kernel files (all owned by this op's directory hierarchy):**

Legacy path (used by `ReshardGenericFactory`, `ReshardSameWidthFactory`, `ReshardSameHeightFactory`) — live one level up at `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/`:
- `reshard_reader.cpp`
- `reshard_reader_diff_width.cpp`
- `reshard_same_width_reader.cpp`
- `reshard_same_width_writer.cpp`
- `reshard_same_height_reader.cpp`
- `reshard_same_height_writer.cpp`

ND path (used by `NdReshardCopyPagesFactory` and `NdReshardCopyLocalShardFactory`) — live at `device/kernels/`:
- `nd_reshard_copy_pages_reader.cpp`
- `nd_reshard_copy_pages_writer.cpp`
- `nd_reshard_copy_local_shards.cpp`

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `port_op_to_metal2_audit.md`.

---

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/sharded/reshard` |
| **Overall** | GREEN |
| **DOps / Factories** | `ReshardDeviceOperation` → `ReshardSameWidthFactory<T/F>`, `ReshardSameHeightFactory<T/F>`, `ReshardGenericFactory`, `NdReshardCopyPagesFactory`, `NdReshardCopyLocalShardFactory<T/F>` |
| *Prereqs* — ProgramDescriptor | Yes |
| *Prereqs* — Device 2.0 (every kernel used) | Yes |
| *Prereqs* — Cross-op escapes | Ok |
| *Feature Support* — overall | GREEN |
| *Feature Support* — Variadic-CTA | N/A |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — MeshWorkload needed | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Other risky pybind | None |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — Custom override-RTA | No |
| *TTNN Readiness* — Fake CBs (address-only) | Present: `(shard_cb, reshard_reader.cpp)`, `(shard_cb, reshard_reader_diff_width.cpp)`, `(shard_cb_id, reshard_same_width_reader.cpp)`, `(shard_cb_id, reshard_same_width_writer.cpp)`, `(shard_cb_id, reshard_same_height_reader.cpp)`, `(shard_cb_id, reshard_same_height_writer.cpp)` (workaround applies) |

---

## Result

**GREEN → brief issued.**

All gates cleared. The ND-path factories (`NdReshardCopyPagesFactory`, `NdReshardCopyLocalShardFactory`) are fully clean (Device 2.0, `TensorAccessorArgs`-based). The legacy-path factories (`ReshardGenericFactory`, `ReshardSameWidthFactory`, `ReshardSameHeightFactory`) are on the `ProgramDescriptor` API, use Device 2.0 kernels throughout, pass buffer addresses via the `Buffer*`-binding form (Case 1 port work), and use fake CBs (FYI-P, workaround available). No UNSUPPORTED features in use. Port can proceed.

---

## Gate detail

### ProgramDescriptor

GREEN — all eight factory variants return a `tt::tt_metal::ProgramDescriptor` via `create_descriptor(...)` (all factory headers at `device/*.hpp` declare `static tt::tt_metal::ProgramDescriptor create_descriptor(...)`). All factories populate `ProgramDescriptor::kernels` via `KernelDescriptor` and CBs via `CBDescriptor`. No imperative-builder calls (`CreateProgram`, `CreateKernel`, `CreateCircularBuffer`, `SetRuntimeArgs`, etc.) anywhere in the factories.

### Device 2.0 (every kernel used)

GREEN — every kernel this op instantiates uses the Device 2.0 kernel-side API exclusively:

**Legacy-path kernels** (`sharded/device/kernels/dataflow/`):
- `reshard_reader.cpp`: uses `Noc`, `CircularBuffer`, `CoreLocalMem`, `UnicastEndpoint`. No `InterleavedAddrGen`, `ShardedAddrGen`, raw sem addresses, or manual CB index management.
- `reshard_reader_diff_width.cpp`: same Device 2.0 surface; adds `UnicastEndpoint{.noc_x, .noc_y, .addr}` and `CoreLocalMem`.
- `reshard_same_width_reader.cpp` / `reshard_same_width_writer.cpp`: use `Noc`, `CircularBuffer`, `AllocatorBank<bank_type>`, `CoreLocalMem`. Clean Device 2.0 throughout.
- `reshard_same_height_reader.cpp` / `reshard_same_height_writer.cpp`: same Device 2.0 surface.

**ND-path kernels** (`device/kernels/`):
- `nd_reshard_copy_pages_reader.cpp` / `nd_reshard_copy_pages_writer.cpp`: `TensorAccessorArgs`, `TensorAccessor`, `Noc`, `CircularBuffer`. Fully Device 2.0.
- `nd_reshard_copy_local_shards.cpp`: `TensorAccessorArgs`, `TensorAccessor`, `Noc`, `CoreLocalMem`. Fully Device 2.0.

No legacy holdovers of any kind. Device 2.0 gate: PASSED.

### Feature compatibility

| Feature | Status | Notes |
|---|---|---|
| GlobalCircularBuffer | N/A | No `GlobalCircularBuffer`, no `global_circular_buffer` CBDescriptor field set, no `CreateGlobalCircularBuffer` calls. |
| Dynamic CircularBuffer (borrowed memory) | GREEN | `cb.buffer = <non-null Buffer*>` in `ReshardGenericFactory` (line ~36, `reshard_program_factory_generic.cpp`), `ReshardSameWidthFactory` (line ~38, `reshard_program_factory_same_width.cpp`), and `ReshardSameHeightFactory` (line ~37, `reshard_program_factory_same_height.cpp`). Feature is LANDED; port uses `DataflowBufferSpec::borrowed_from`. However, all three fire the fake-CB check (see below). |
| CBDescriptor `address_offset` (non-zero) | N/A | No `address_offset` fields set anywhere in the factories. No calls to `cb_descriptor_from_sharded_tensor`. |
| Aliased Circular Buffers | N/A | Every CBDescriptor has exactly one `format_descriptors` entry (`push_back` called once). |
| GlobalSemaphore | N/A | No semaphores of any kind in this op. |
| Non-zero semaphore initial value | N/A | No semaphore creation anywhere. |
| Dynamic TensorAccessor (`ArgConfig::Runtime*`) | N/A | No `ArgConfig::Runtime` tokens in op host code. |
| `UpdateCircularBuffer*` | N/A | No `UpdateCircularBufferTotalSize`, `UpdateCircularBufferPageSize`, or `UpdateDynamicCircularBufferAddressAndTotalSize` calls. |
| Variable-count compile-time arguments (CTA varargs) | N/A | `tensor_args_t` (`ReshardInputs`) is a fixed struct with `Tensor input` and `std::optional<Tensor> preallocated_output`. No variable-count tensor containers. No kernel-level `get_compile_time_arg_val` loops over runtime-varying counts. |

Feature gate: GREEN (no UNSUPPORTED features in use).

---

## Port-work summary *(mirrors the brief)*

### Tensor bindings (per binding, per factory)

**`NdReshardCopyPagesFactory`** (`nd_reshard_program_factory_copy_pages.cpp`):
- `input` binding: **clean** — `TensorAccessorArgs(*input_buffer)` drives compile-time args (line 49); kernel `nd_reshard_copy_pages_reader.cpp` builds `TensorAccessor(args_src, bank_base_address_src)` using the CRTA-provided base. End-to-end `TensorAccessor`.
- `output` binding: **clean** — same pattern with `output_buffer` and `nd_reshard_copy_pages_writer.cpp`.

**`NdReshardCopyLocalShardFactory<T/F>`** (`nd_reshard_program_factory_copy_local.cpp`):
- `input` binding: **clean** — `TensorAccessorArgs(*input_buffer)` at line 24; `nd_reshard_copy_local_shards.cpp` uses `TensorAccessor(args_src, bank_base_address_src)`.
- `output` binding: **clean** — `TensorAccessorArgs(*output_buffer)` at line 25; same kernel uses `TensorAccessor(args_dst, bank_base_address_dst)`.

**`ReshardGenericFactory`** (`reshard_program_factory_generic.cpp`):
- `input` binding: **Case 1** — host side injects `input_buffer` (`Buffer*`) at RTA index `grid.x + grid.y` (line 785–786, line 793–794) via the `Buffer*`-binding form; kernel (`reshard_reader.cpp` line 23 / `reshard_reader_diff_width.cpp` line 23) reads `input_shard_addr = get_arg_val<uint32_t>(arg_index++)` and uses it as a raw base address in `{.addr = input_shard_addr + addr_offset}`. Port work: re-express via `TensorParameter` / `TensorBinding`; **question for user** on whether Case 2 applies (see Questions section).
- `output` binding: the output sharded tensor is accessed via `cb.buffer = output_buffer` (line ~698, `push_reshard_generic_cb_pair`). The kernel reads `cb.get_write_ptr()` as an L1 write pointer (line 30, `reshard_reader.cpp`). But the CB has **no FIFO producer + consumer** — both BRISC and NCRISC write directly to the CB's backing memory without `push_back`/`wait_front`. **Fake CB** (see Heads-ups).

**`ReshardSameWidthFactory<T/F>`** (`reshard_program_factory_same_width.cpp`):
- `remote` binding: **Case 1** — host side injects `remote_buffer` (`Buffer*`) at `kernel_args[0]` (line 156); kernel (`reshard_same_width_reader.cpp` line 24 / `reshard_same_width_writer.cpp` line 24) reads `src_addr` / `dst_addr = get_arg_val<uint32_t>(0)` and uses it as `{.bank_id = bank_id, .addr = addr}` (where `addr = src_addr + src_offset`). Port work: Case 1 re-express; **question for user** (see Questions).
- `local` binding: CB `cb.buffer = local_buffer` (line ~99, `push_reshard_same_width_cb_pair`). Kernel uses `shard_cb.get_write_ptr()` / `shard_cb.get_read_ptr()` as raw L1 address, no FIFO signaling. **Fake CB** (see Heads-ups).

**`ReshardSameHeightFactory<T/F>`** (`reshard_program_factory_same_height.cpp`):
- `remote` binding: **Case 1** — host side injects `remote_buffer` (`Buffer*`) at `runtime_args_0[3]` / `runtime_args_1[3]` (line 127, 133); kernel (`reshard_same_height_reader.cpp` line 21 / `reshard_same_height_writer.cpp` line 22) reads `base_read_addr` / `base_write_addr = get_arg_val<uint32_t>(3)` and uses it as `{.bank_id = bank_id, .addr = read_offset}` (where `read_offset = base_read_addr + offset`). Port work: Case 1 re-express; **question for user** (see Questions).
- `local` binding: CB `cb.buffer = local_buffer` (line ~79, `push_reshard_same_height_cb_pair`). No FIFO push/pop. **Fake CB** (see Heads-ups).

### Custom hash

None — `ReshardDeviceOperation` defines no `compute_program_hash`. No port work here.

---

## Heads-ups *(mirrors the brief)*

### Notable LANDED constructs

**Dynamic CircularBuffer (borrowed memory)** — `cb.buffer = <non-null Buffer*>` fires in three factories:
- `reshard_program_factory_generic.cpp` ~line 36 (`push_reshard_generic_cb_pair`, called at line ~698 with `output_buffer`)
- `reshard_program_factory_same_width.cpp` ~line 38 (`push_reshard_same_width_cb_pair`, called at line ~99 with `local_buffer`)
- `reshard_program_factory_same_height.cpp` ~line 37 (`push_reshard_same_height_cb_pair`, called at line ~79 with `local_buffer`)

All three are confirmed **fake CBs** (see below) — the "borrowed-memory DFB" Metal 2.0 translation does NOT apply. The port uses the sanctioned fake-CB workaround instead (see the porting recipe). The Dynamic-CB feature fires the LANDED recognition signal but the fake-CB litmus overrides the clean classification.

### Fake CBs (address-only)

**Six fake CBs, one per legacy kernel endpoint:**

Every CB that is bound via `cb.buffer = non_null_buffer` in the legacy factories is used purely as an L1 address source. The litmus: no `push_back`, `wait_front`, `pop_front`, or `reserve_back` calls appear in any of the six legacy kernel files. The CB is accessed exclusively via `cb.get_write_ptr()` / `cb.get_read_ptr()` to retrieve an L1 address, which is then used directly in NoC operations. No producer-consumer FIFO pair exists.

| CB identifier | Kernel endpoint | Factory / factory-path |
|---|---|---|
| `shard_cb` (CB index from CTA 0) | `reshard_reader.cpp:30 cb.get_write_ptr()` | `ReshardGenericFactory`, equal-page-size path |
| `shard_cb` (CB index from CTA 0) | `reshard_reader_diff_width.cpp:30 cb.get_write_ptr()` | `ReshardGenericFactory`, diff-page-size path |
| `shard_cb_id` (CTA 0) | `reshard_same_width_reader.cpp:37 shard_cb.get_write_ptr()` | `ReshardSameWidthFactory` |
| `shard_cb_id` (CTA 0) | `reshard_same_width_writer.cpp:37 shard_cb.get_read_ptr()` | `ReshardSameWidthFactory` |
| `shard_cb_id` (CTA 0) | `reshard_same_height_reader.cpp:31 shard_cb.get_write_ptr()` | `ReshardSameHeightFactory` |
| `shard_cb_id` (CTA 0) | `reshard_same_height_writer.cpp:31 shard_cb.get_read_ptr()` | `ReshardSameHeightFactory` |

The port resolves each with the sanctioned fake-CB workaround (see the porting recipe). These do **not** gate.

### Cross-op / shared kernels

The six legacy-path kernels (`reshard_reader.cpp`, `reshard_reader_diff_width.cpp`, `reshard_same_width_reader/writer.cpp`, `reshard_same_height_reader/writer.cpp`) live at `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/`. They are owned by and exclusively instantiated by this op's factories — no other op references them (verified by grep). They are in-family under the `sharded/` family tree. No port-together coupling required.

The three ND-path kernels live at `device/kernels/` inside this op's root — exclusively instantiated by this op.

### RTA varargs

Not present. The `reshard_reader.cpp` and `reshard_reader_diff_width.cpp` kernels consume a runtime-varying count of range / block entries from RTAs (the `num_ranges` / `num_blocks` count at `arg_index++` after the fixed header), but the loop counter is itself a runtime arg rather than a compile-time varargs index — this is standard RTA loop retrieval, not CTA varargs, and Metal 2.0 supports it via the runtime vararg mechanism or named RTAs. No special vararg flag needed beyond the standard port observation.

### TTNN factory analysis (porter-relevant)

- **Pybind `create_descriptor`:** None — `reshard_nanobind.cpp` only exposes `ttnn::reshard` via `bind_function<"reshard">`. No factory-innards binding.
- **Other risky pybind:** None.
- **Custom `override_runtime_arguments`:** None — no `override_runtime_arguments` definition found anywhere in the op.

---

## Team-only

### TensorAccessor convertibility (per Case-1 binding — no Case-2 confirmed)

All Case-1 bindings are marked `(user confirmation pending)` for whether they are genuinely Case 2. The team note:

- **`ReshardGenericFactory` input binding:** The kernel's access pattern is a custom stride-table walk: host pre-computes `PageStride`/`CompressedStrideBlock` structures encoding which source-core pages map to each output core, packs them into RTAs, and the kernel decodes these stride tables at runtime to issue individual NoC reads targeting `{noc_x, noc_y, input_shard_addr + offset}`. This is not page-by-page iteration in the `TensorAccessor` sense — the access is by physical-core + in-core-page-offset determined by the stride tables. Likely **genuinely exotic** (no standard `TensorAccessor` iterator covers arbitrary cross-core stride walks), but AI should not self-classify as Case 2 per recipe rules. User confirmation required.

- **`ReshardSameWidthFactory` remote binding:** The kernel reads from `{.bank_id = bank_id, .addr = src_addr + src_offset}` where `bank_id` and `src_offset` come from per-transfer RTAs. This is bank-id-keyed access (the host computes bank IDs from `device->allocator()->get_bank_ids_from_logical_core(...)`). Likely **genuinely exotic** — bank-id-keyed access without `TensorAccessor` is a pre-Device-2.0 pattern that `TensorAccessor` was designed to replace, but whether the exact layout can be expressed by a `TensorAccessor` iteration depends on the shard geometry. User confirmation required.

- **`ReshardSameHeightFactory` remote binding:** Same bank-id-keyed pattern — `{.bank_id = bank_id, .addr = base_write_addr + write_offset}`. Same analysis; user confirmation required.

### Out-of-directory coupling and donor shape

**Roll-up: ✓ clean** — no function-call escapes to cross-family donors, no cross-family borrowed kernel files.

**Summary table:**

| Op kernel | `#include` outside op dir | Donor class | Status |
|---|---|---|---|
| All six legacy-path kernels | `api/dataflow/dataflow_api.h`, `api/dataflow/noc.h`, `api/dataflow/circular_buffer.h`, `api/dataflow/endpoints.h`, `api/core_local_mem.h`, `api/tensor/noc_traits.h` | LLK / HAL (`tt_metal/*`) | ✓ No concern |
| All three ND-path kernels | `api/tensor/tensor_accessor.h`, `api/dataflow/dataflow_api.h`, `api/dataflow/noc.h`, `api/dataflow/circular_buffer.h`, `api/core_local_mem.h`, `api/tensor/noc_traits.h` | LLK / HAL (`tt_metal/*`) | ✓ No concern |

No out-of-op-directory function-call escapes. All includes are from the `api/` (Device 2.0 / LLK) tree. No `ttnn/operations/` cross-op includes in kernels.

**Borrowed kernel files:** All kernel files are instantiated exclusively by this op's factories. The six legacy kernels live at `sharded/device/kernels/dataflow/` (one level above the op root, still within the `sharded/` family). No other op instantiates them. The three ND kernels live inside `reshard/device/kernels/` — fully owned.

**Port-together sets:** None required. All kernel files are single-op or in-family.

### Relaxation candidates

No custom `compute_program_hash` to mine. No candidates available.

### TTNN factory analysis

**Q1 — Op-owned tensors:** No. `ReshardDeviceOperation::create_output_tensors` calls `create_device_tensor(compute_output_specs(...), ...)` (line 298, `reshard_device_operation.cpp`) — this is the declared output tensor, not an internal intermediate. No factory allocates or manages any intermediate tensors of its own. Answer: **No**.

**Q2 — MeshWorkload needed:** No. No `create_mesh_workload`, `create_workload_descriptor`, or `cached_mesh_workload_t` anywhere. Answer: **No** (not needed, and Q1 is No so no plumbing-artifact risk either).

**Q3 — Pybind `create_descriptor`:** No. `reshard_nanobind.cpp` only calls `bind_function<"reshard">` on the top-level op function. No `nb::class_<…Factory>(…).def_static("create_descriptor", …)` anywhere.

**Q4 — Other migration-risky pybind:** None. The nanobind file is minimal — one `bind_function` call. No DeviceOperation methods, factory parameters, or descriptor introspection exposed.

**Q5 — Custom hash:** No (confirmed — no `compute_program_hash` definition in the op).

**Q6 — Custom override-runtime-args:** No. No `override_runtime_arguments` static method definition in any factory.

---

## Misc anomalies *(team-only, non-gating)*

1. **`reshard_device_operation.cpp` unreachable code, lines 41–50.** The function `is_valid_for_legacy_reshard` has a `return` statement at line 39 (`return out_mem_config.buffer_type() == BufferType::L1`) that makes lines 41–50 (the `ROW_MAJOR` early-return logic) unreachable dead code. The final `return true` at line 50 is also unreachable. This appears to be a logic bug / leftover code. Route to op owner.

2. **`reshard_program_factory_generic.cpp` redundant `input_buffer->address()` in helper return vectors.** The detail helper functions `get_runtime_args_for_given_ranges` and `get_runtime_args_for_given_ranges_diff_width` return a `std::vector<uint32_t>` that includes the address at position `grid.x + grid.y` (lines 559, 605). The factory then iterates the vector and replaces position `grid.x + grid.y` with `input_buffer` (the `Buffer*` form, lines 784–798). This means the helper computes an address (`input_buffer->address()` indirectly via `input_addr` parameter) that is immediately discarded. Not incorrect (the `Buffer*` replacement is what actually registers the binding), but wasteful. Route to op owner.

---

## Questions for the user

1. **Case 1 vs Case 2 for `ReshardGenericFactory` input binding:** The kernel (`reshard_reader.cpp` / `reshard_reader_diff_width.cpp`) accesses the input tensor via host-precomputed stride tables encoding (source_core_x, source_core_y, page_offset) tuples. The kernel issues individual NoC reads to `{.noc_x = core_id_x, .noc_y = core_id_y, .addr = input_shard_addr + addr_offset}`. This is a cross-core stride walk that does not follow standard page-by-page iteration. **Is this genuinely exotic (Case 2 — bridge via `get_bank_base_address`, raw walk unchanged), or is it a case where `TensorAccessor` should be updated to support this iteration pattern (Case 1)?** Per the recipe, I cannot self-classify as Case 2. If Case 2 is confirmed, the recipe message applies: "The use of `TensorAccessor` is an ergonomic choice on Gen1 architectures. It has meaningful performance implications on Gen2 architectures. Ideally, `TensorAccessor` should be updated to support the required iteration pattern; consider filing an issue requesting that support."

2. **Case 1 vs Case 2 for `ReshardSameWidthFactory` / `ReshardSameHeightFactory` remote bindings:** These kernels access the remote tensor via bank-id-keyed calls (`{.bank_id = bank_id, .addr = remote_addr + offset}`), where `bank_id` values are pre-computed on the host from `device->allocator()->get_bank_ids_from_logical_core(...)`. This is a per-transfer bank-id walk. **Is this genuinely exotic (Case 2) or is it a candidate for `TensorAccessor` re-expression (Case 1)?** Same per-recipe message applies if Case 2 is confirmed.

---

## Recipe notes

1. **Fake-CB classification with `cb.buffer = non_null` in an op that has no FIFO semantics.** The Dynamic CircularBuffer LANDED entry fires its recognition signal (`.buffer` field set to non-null) in three factories. However, the causal-link gate immediately overrides all three to fake-CB status (no producer-consumer pair). The interplay between the LANDED recognition signal and the fake-CB litmus is correct per the recipe, but for an op like reshard — where the pattern is "use shard buffer as L1 backing, no FIFO" — it would be marginally clearer if the recipe's LANDED entry explicitly noted that borrowed-memory CBs with no FIFO signaling resolve to fake-CBs rather than `borrowed_from`. The current recipe flow (LANDED fires → causal-link gate → fake-CB) is logically correct; this is a clarity note, not a rule contradiction.
