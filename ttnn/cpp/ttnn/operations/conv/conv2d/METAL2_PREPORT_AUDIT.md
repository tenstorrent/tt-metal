# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/conv/conv2d`

**Identifying section:**

This directory contains a single `DeviceOperation` with two `ProgramFactory` implementations:

- **`Conv2dDeviceOperation`** (`conv2d_device_operation.cpp` / `.hpp`)
  - `Conv2dShardedProgramFactory` (`conv2d_op_sharded_program_factory.cpp`)
  - `Conv2dWidthShardedProgramFactory` (`conv2d_op_width_sharded_program_factory.cpp`)

Shared host-side factory helpers live in `conv2d_op_program_factory_common.cpp` / `.hpp`.  Kernels are all owned by this op under `device/kernels/`.  The pybind surface is in `conv2d_nanobind.cpp`.

> **Note:** Both factories implement `create_workload_descriptor` (not the older `create_program_factory` / `create_program`), meaning this op has **already crossed the Metal 2.0 host-API boundary** — `WorkloadDescriptor` + `ProgramDescriptor` are in active use.  The audit below nevertheless completes all seven subjects so the report serves as a full readiness record; findings that are moot because the port is effectively done are called out as such.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `port_op_to_metal2_audit.md`.

---

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/conv/conv2d` |
| **Overall** | GREEN (with one open question — see Questions) |
| **DOps / Factories** | `Conv2dDeviceOperation` → `Conv2dShardedProgramFactory`, `Conv2dWidthShardedProgramFactory` |
| *Prereqs* — ProgramDescriptor | Yes — both factories use `ProgramDescriptor` + `WorkloadDescriptor` |
| *Prereqs* — Device 2.0 (every kernel used) | Yes — all kernels use `experimental::CB` / `Noc` / `TensorAccessor` wrappers (one open question: `get_local_cb_interface().fifo_wr_ptr = …` writes in `conv_reader_common.hpp`) |
| *Prereqs* — Cross-op escapes | Ok — pool donor header and kernel_lib helpers are Device 2.0 clean |
| *Feature Support* — overall | GREEN |
| *Feature Support* — Variadic-CTA | N/A |
| *TTNN Readiness* — Op-owned tensors | Yes: `conv_reader_indices_tensor` allocated in both `create_workload_descriptor` implementations, parked on `WorkloadDescriptor::buffers` |
| *TTNN Readiness* — MeshWorkload needed | No (op-owned-tensor artifact only — see TTNN factory analysis) |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Other risky pybind | None |
| *TTNN Readiness* — Custom hash | Yes → delete (see Custom program hash) |
| *TTNN Readiness* — Custom override-RTA | No |
| *TTNN Readiness* — Fake CBs (address-only) | Present: `READER_INDICES` CB in L1 (non-DRAM) mode — no producer/consumer pair; see Heads-ups |

---

## Result

**GREEN → brief issued.**

Both factories already implement `create_workload_descriptor` with full `ProgramDescriptor` + `WorkloadDescriptor` construction; every Device 2.0 check clears; no UNSUPPORTED Appendix A features fire.  Open items are one question about a raw `fifo_wr_ptr` write (likely sanctioned / architectural but needs confirmation) and the standard port-work inventory below.  The op is essentially already ported at the host-API level; remaining work is largely the TTNN-layer touches (custom hash deletion, tensor-binding formalization).

---

## Gate detail

### ProgramDescriptor

**GREEN.**  Both `Conv2dShardedProgramFactory::create_workload_descriptor` and `Conv2dWidthShardedProgramFactory::create_workload_descriptor` construct a `tt::tt_metal::ProgramDescriptor desc;`, populate it with `CBDescriptor`, `SemaphoreDescriptor`, and `KernelDescriptor` objects, then wrap it in a `tt::tt_metal::WorkloadDescriptor` that is returned to the framework.  Neither factory uses the legacy `host_api.hpp` imperative builder (`CreateProgram`, `CreateKernel`, `CreateCircularBuffer`, `SetRuntimeArgs`).  Relevant files: `conv2d_op_sharded_program_factory.cpp:181–1457`, `conv2d_op_width_sharded_program_factory.cpp:56–688`.

### Device 2.0 (every kernel used)

**GREEN with one open question.**

All nine kernel files under `device/kernels/` use Device 2.0 wrapper objects throughout:

| Kernel file | CB wrapper | Noc wrapper | TensorAccessor |
|---|---|---|---|
| `reader_conv_activations_padded_with_halo_3x3_weights_v2.cpp` | `experimental::CB` | `Noc noc;` | via `conv_reader_common.hpp` |
| `reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2.cpp` | `experimental::CB` | `Noc noc;` | via `conv_reader_common.hpp` |
| `reader_depthwise_conv1d.cpp` | `experimental::CB` | `Noc noc;` | via `conv_reader_common.hpp` |
| `reader_writer_tiled_out_1d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp` | `experimental::CB` | `Noc noc;` | `TensorAccessorArgs<39>()` / `TensorAccessor(s_weight_args, …)` |
| `reader_writer_tiled_out_1d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp` | `experimental::CB` | `Noc noc;` | (reads from CB, no direct tensor accessor) |
| `writer_tiled_out_2d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp` | `experimental::CB` | `Noc noc;` | `TensorAccessorArgs<ct_arg_idx>()` / `TensorAccessor` |
| `writer_tiled_out_2d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp` | `experimental::CB` | `Noc noc;` | (reads from CB) |
| `activation_reader_width_sharded.cpp` | `experimental::CB` | `Noc noc;` | via `conv_reader_common.hpp` |
| `conv_bmm_tilize.cpp` | `experimental::CB` | (compute kernel) | — |
| `compute_depthwise_conv1d.cpp` | `experimental::CB` | (compute kernel) | — |

The in-directory header `conv_reader_common.hpp` uses `get_tile_size(cb_id)` (line 24) and `get_local_cb_interface(cb_id)` (line 25) — both are **sanctioned** Device 2.0 free functions per the migration guide.

**Open question on `fifo_wr_ptr` writes** (see Questions §1):  `conv_reader_common.hpp:91` and `:109` write directly to `get_local_cb_interface(cb_id_act).fifo_wr_ptr` to reposition the CB's hardware write pointer — a raw manipulation beyond the read-`fifo_num_pages` / `get_tile_size` pattern the recipe explicitly sanctions.  This does not appear in the Device 2.0 migration guide's migrated examples.  If this is sanctioned (architectural necessity for the activation-reuse path), Device 2.0 check stays GREEN.  If it is an unsanctioned holdover, it should be cleaned on the Device 2.0 track before the port proceeds — and this audit grades to YELLOW on that check.  **Pending user confirmation.**

### Feature compatibility

| Feature | Status | Notes |
|---|---|---|
| GlobalCircularBuffer | N/A | Header `global_circular_buffer.hpp` is `#include`d in `conv2d_device_operation.hpp:12` but no `GlobalCircularBuffer` type, `CreateGlobalCircularBuffer`, or `.global_circular_buffer` field assignment appears anywhere in the op's source. Dead include only. |
| Dynamic CircularBuffer (borrowed memory) | GREEN | `emit_cb_descriptors` (conv2d_op_program_factory_common.cpp:756) emits `CBDescriptor{…, .buffer = buffer}` for `ACT_SHARDED` (input buffer), `OUT` and `MATMUL_PARTIALS` (output buffer), and `READER_INDICES` (indices buffer). Port already uses `CBDescriptor::buffer` — the `borrowed_from` step is already done. |
| CBDescriptor `address_offset` (non-zero) | N/A | No `address_offset` field is set in any `CBDescriptor` emission; no `set_address_offset` call. |
| Aliased Circular Buffers | N/A | All `CBDescriptor::format_descriptors` initializers are single-element. The `overlapped_by_cb` field in `CBInfo` is an internal index-aliasing mechanism (zero-page CBs get their index remapped), distinct from the multi-element `format_descriptors` aliasing pattern. |
| GlobalSemaphore | N/A | No `GlobalSemaphore` type, `CreateGlobalSemaphore`, or `global_semaphore.hpp` include. |
| Non-zero semaphore initial value | N/A | All `SemaphoreDescriptor{…, .initial_value = 0}` (both factories, multiple sites). |
| Dynamic TensorAccessor (`ArgConfig::Runtime*`) | N/A | No `ArgConfig::Runtime*` token in any factory file. |
| `UpdateCircularBuffer*` | N/A | No `UpdateCircularBufferTotalSize`, `UpdateCircularBufferPageSize`, or `UpdateDynamicCircularBufferAddressAndTotalSize` calls. The comment at `conv2d_op_sharded_program_factory.cpp:684` ("framework tracks the globally-allocated CBs … patches their addresses on cache hits — no per-op UpdateDynamicCircularBufferAddress override needed") confirms these were intentionally avoided. |
| Variable-count compile-time arguments (CTA varargs) | N/A | `Conv2dInputs` has a fixed-count tuple (a, b, optional bias). No `std::vector<Tensor>` in `tensor_args_t`. No loop over `get_compile_time_arg_val(i)` with runtime-varying index. |

---

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings:**
  - `weights` (tensor `b`) — **Case 1** (re-express): host side uses `emplace_runtime_args(core, {weights_buffer, …})` (`Buffer*` binding form). Port re-expresses as a `TensorParameter` / `TensorBinding`; kernel side already constructs `TensorAccessor(s_weight_args, weight_addr)` from CTA-plumbed args.  Both factories. Sites: `conv2d_op_sharded_program_factory.cpp:1302`, `conv2d_op_width_sharded_program_factory.cpp:667`.
  - `bias` (optional tensor) — **Case 1** (re-express): same `Buffer*` binding form when bias is present. Sites: `conv2d_op_sharded_program_factory.cpp:1304`, `conv2d_op_width_sharded_program_factory.cpp:669`.
  - `conv_reader_indices` (op-owned config tensor) — **Case 1** (re-express): host side bakes `conv_reader_indices_buffer->address()` into CTAs (`conv2d_op_sharded_program_factory.cpp:873`, `conv2d_op_width_sharded_program_factory.cpp:569`; both gated on `config_tensors_in_dram`). CTA-baked address is not the silent-wrong hazard; port re-expresses via `TensorParameter`. Kernel side already uses `TensorAccessorArgs<…>()` + `TensorAccessor(config_tensor_args, config_dram_addr)` in `conv_reader_common.hpp:361–364`.
  - `activations` (tensor `a`, sharded in L1) — **clean**: accessed through the `ACT_SHARDED` borrowed-memory DFB (set via `CBDescriptor::buffer = input_buffer`). No raw address plumbing.
  - `output` — **clean**: accessed through the `OUT` borrowed-memory DFB.

- **Custom hash:** delete custom `compute_program_hash` → default (sanctioned exception). See Custom program hash below.

---

## Heads-ups  *(mirrors the brief)*

- **Notable LANDED constructs — Dynamic CircularBuffer (borrowed-memory DFBs):** `ACT_SHARDED`, `OUT`, `MATMUL_PARTIALS`, and `READER_INDICES` CBs are emitted with `CBDescriptor::buffer` set, already using the Metal 2.0 borrowed-memory DFB mechanism. No additional port work for these CBs (the `borrowed_from` field is already the destination).

- **Fake CBs (address-only) — `READER_INDICES` in L1 mode:**  When `config_tensors_in_dram = false` (the default), `READER_INDICES` is globally allocated to the L1-resident indices buffer (`CBInfo::is_globally_allocated = true`), but the kernel accesses it purely by calling `cb_reader_idx.get_write_ptr()` as a raw L1 address — there is no `push_back` / `pop_front` producer-consumer FIFO protocol in the L1 path (the `load_config_tensor_if_in_dram` helper that does call `push_back` runs only under `#ifdef CONFIG_TENSOR_IN_DRAM`). Sites: `conv_reader_common.hpp` (the `load_config_tensor_if_in_dram` template, conditional) and all reader kernels that call `cb_reader_idx.get_write_ptr()` on line ~36 without a preceding wait. The port resolves this with the sanctioned fake-CB workaround (see the porting recipe); it does **not** gate.

- **Cross-op / shared kernels:**  All kernel files `#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>` (cross-family donor, `pool` family).  This header is a Device 2.0 convenience wrapper (defines `experimental::CB = CircularBuffer`, `Noc` aliases, helper read functions) — it does not introduce any Device 1.0 idioms.  Classify as Shape 1 / clean donor.  Additionally: `conv_bmm_tilize.cpp` and `compute_depthwise_conv1d.cpp` include `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` and `untilize_helpers.hpp` — these are the shared kernel library (class 2), Device 2.0 compliant.

- **RTA varargs:** None. All runtime-arg plumbing uses named positional arguments or `emplace_runtime_args` with fixed-count vectors.

- **TTNN factory analysis (porter-relevant):**
  - Pybind `create_descriptor`: None.
  - Other risky pybind: None. `conv2d_nanobind.cpp` binds only the user-facing `conv2d` function, `Conv2dConfig` struct, `prepare_conv_weights`, and `prepare_conv_bias` — no `ProgramFactory` or `DeviceOperation` internals.
  - Custom `override_runtime_arguments`: None — neither factory defines this hook.

---

## Team-only

### TensorAccessor convertibility (per Case-2 binding)

No Case-2 bindings. All tensor access patterns are Case 1 (expressible via `TensorParameter` + `TensorBinding`):
- Weights / bias: already use `TensorAccessor` on the kernel side; host side needs the `Buffer*` form replaced with `TensorBinding`.
- Config tensor: already uses `TensorAccessorArgs` / `TensorAccessor` on the kernel side; CTA-baked address path needs `TensorParameter` wiring.

### Out-of-directory coupling & donor shape

**Op-level roll-up:** ✓ clean. No cross-family donors introduce any ⚠ or ✗ shape issues.

**Summary table:**

| Op kernel | Donor file | Classification |
|---|---|---|
| All dataflow kernels | `ttnn/operations/pool/device/kernels/experimental_device_api.hpp` | Class 6 (cross-family, pool); all functions use `Noc`, `CircularBuffer`, `UnicastEndpoint` — Shape 1 / ✓ excellent |
| `conv_bmm_tilize.cpp`, `compute_depthwise_conv1d.cpp` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` | Class 2 (shared kernel library) / ✓ |
| `conv_bmm_tilize.cpp` | `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` | Class 2 (shared kernel library) / ✓ |
| All kernel files | `device/kernels/conv_reader_common.hpp` | In-op (owned) |

**Per-call detail:** All donor functions in `experimental_device_api.hpp` take `Noc noc`, `CircularBuffer`, or `UnicastEndpoint` — these are Device 2.0 wrapper types (Shape 1: `✓ excellent`).  No `uint32_t sem_addr`, `TensorAccessorArgs<N>`, or old-style addr-gen shapes.

**Borrowed kernel files (file-path instantiation):** None.  All kernel `.cpp` files referenced in both factories are owned by this op's `device/kernels/` directory.  No borrowed kernel files from other families.

### Relaxation candidates (mined from custom hash before deletion — FALLIBLE)

`compute_program_hash` (`conv2d_device_operation.cpp:144–165`) hashes the `Conv2dHashableParams` struct.  Comparing `Conv2dParams` (all fields) vs. `Conv2dHashableParams` reveals these fields are **excluded from the hash**:
- `groups` (only used to compute `is_1d_depthwise_conv` at factory time)
- `full_inner_dim` (included in hash) — actually present
- `pre_op_l1_allocation_size_bytes` — explicitly excluded (transient L1-accounting field, correct to exclude)
- `full_inner_dim` — wait, re-checking: this field IS present in `Conv2dHashableParams` at line 234, so it IS hashed.

The hash covers `TensorSpec` indirectly via `tensor_args` (the second argument to `ttsl::hash::hash_objects_with_default_seed`).  Candidate: `groups` is not independently hashed (missing from `hashable_args`), which is correct since `groups` only affects `is_1d_depthwise_conv` which is derivable from the hashed fields (`sliding_window_config`, `output_channels`, `has_bias`, `input_tensor_shape`).  No relaxation candidates identified from this analysis; the custom hash looks intentionally minimal rather than incorrect.

**Note (FALLIBLE):** the above is a best-effort reading of the hash vs. param struct fields; the default hash on deletion supersedes this.

### TTNN factory analysis (six-question answers)

1. **Op-owned tensors?** **Yes.** Both `Conv2dShardedProgramFactory::create_workload_descriptor` (`conv2d_op_sharded_program_factory.cpp:1563–1572`) and `Conv2dWidthShardedProgramFactory::create_workload_descriptor` (`conv2d_op_width_sharded_program_factory.cpp:727–736`) allocate a `conv_reader_indices_tensor` on device, wrap it in `shared_ptr<Tensor>`, and push it to `workload_descriptor.buffers`.  This is the op-owned-tensor pattern described in the recipe.

2. **MeshWorkload concept needed?** **No (op-owned-tensor artifact only).** Both factories implement `create_workload_descriptor` rather than `create_program_factory`/`create_program`. Per the factory headers and comments, this is entirely because the op-owned `conv_reader_indices_tensor` must outlive the cached program and be parked on the `WorkloadDescriptor::buffers` vector.  The op is single-device (comments: "Single-device op: per-coord program is structurally identical for every coord in tensor_coords — conv2d doesn't depend on cluster position"). This is the false-positive trap the recipe describes: op-owned tensors → forced onto the WorkloadDescriptor path → **not** a genuine MeshWorkload need.

3. **Pybind `create_descriptor`?** **No.** `conv2d_nanobind.cpp` binds only the user-facing function and `Conv2dConfig` struct; no `nb::class_<Conv2dShardedProgramFactory>` or `nb::class_<Conv2dWidthShardedProgramFactory>` wrapping.

4. **Other migration-risky pybind?** **None.** No `DeviceOperation` methods or factory internals are exposed to Python.

5. **Custom hash?** **Yes** — `conv2d_device_operation.cpp:144`  (`ttsl::hash::hash_t Conv2dDeviceOperation::compute_program_hash(...)`).  See Custom program hash.

6. **Custom override-runtime-args?** **No.** Neither factory defines `override_runtime_arguments`.

---

## Custom program hash

**Present.** `Conv2dDeviceOperation::compute_program_hash` is defined at `conv2d_device_operation.cpp:144–165`.

The hash packages a `Conv2dHashableParams` struct (a strict subset of `Conv2dParams` that excludes the transient `pre_op_l1_allocation_size_bytes` field and the `groups` field) plus `tensor_args`.  `tensor_args` includes the input and weight tensors by value, which carries `TensorSpec` into the hash key.  This custom hash appears **functionally correct** (it avoids the common TensorSpec omission bug), but must still be deleted during the port: the default TTNN hash is correct-by-construction, and no Metal 2.0 factory concept reads the custom hash.

**Port work:** delete `compute_program_hash` declaration + definition and revert to default.  Files: `conv2d_device_operation.hpp:44–45`, `conv2d_device_operation.cpp:144–165`.

---

## Misc anomalies

- **Dead `#include` in device_operation header:** `conv2d_device_operation.hpp:12` includes `<tt-metalium/global_circular_buffer.hpp>` but no `GlobalCircularBuffer` type is used anywhere in the op's source.  This is a dead include that can be removed when convenient (not port work, but tidying the header is low-risk).

- **`post_conv2d_op_memory_checks` still present in common cpp:** The legacy `post_conv2d_op_memory_checks(tt::tt_metal::Program& program, ...)` function (`conv2d_op_program_factory_common.cpp:843`) is still compiled even though neither factory calls it in the descriptor path (both call `post_conv2d_op_memory_checks_descriptor` instead).  Appears to be kept for historical reference or potential fallback use; not breaking, but may become dead code.

---

## Questions for the user

1. **`get_local_cb_interface().fifo_wr_ptr` write access — sanctioned or Device 2.0 holdover?**  `conv_reader_common.hpp:91` (`get_local_cb_interface(cb_id_act).fifo_wr_ptr = l1_write_addr_act;`) and `:109` (`get_local_cb_interface(cb_id_act).fifo_wr_ptr = cb_start_addr;`) directly overwrite the hardware FIFO write pointer.  These sites are inside `pass_to_the_next_image_width` which is part of the `enable_activation_reuse` code path.  The recipe sanctions `get_local_cb_interface(cb_id)` for reading `fifo_num_pages` (and `get_tile_size(cb_id)`), but does not mention writing to `.fifo_wr_ptr`.  Is this a sanctioned architectural pattern for the activation-reuse CB pointer manipulation, or an unsanctioned Device 1.0 holdover that needs a Device 2.0 track fix before the port?

---

## Recipe notes

- The recipe's "ProgramDescriptor" prerequisite check (Check 1) has no defined outcome for an op that has **already passed ProgramDescriptor AND is already using `create_workload_descriptor`** (the Metal 2.0 factory API).  This op straddles both states: it uses `ProgramDescriptor` (clearly meets the prereq) but has also already adopted the Metal 2.0 factory interface.  The "Op is on the ProgramDescriptor API" description in the recipe implicitly assumes the op has not yet adopted `WorkloadDescriptor`.  For this op, Check 1 is unambiguously GREEN, but a note for the recipe maintainer: adding a "already on Metal 2.0 factory interface" recognition signal to Check 1 would be helpful for ops in this state.

- The recipe's MeshWorkload false-positive trap (recipe §TTNN factory analysis Q2) is exactly the case here: `create_workload_descriptor` is used not because the op is a genuine MeshWorkload consumer, but because the op-owned tensor requirement forces the `WorkloadDescriptor::buffers` path. The recipe language covers this correctly ("forces single-program ops that have op-owned tensors onto the MeshWorkload path as a plumbing artifact — that is NOT a genuine MeshWorkload need"), so this is a validation of the recipe, not friction with it.

- The `fifo_wr_ptr` direct write (Questions §1) was not anticipated by the recipe's Device 2.0 holdover section.  The recipe lists `get_read_ptr(cb_id)` and `get_write_ptr(cb_id)` as holdovers replaceable by `cb_obj.get_read_ptr()` / `cb_obj.get_write_ptr()`, but does not discuss *writes* to the hardware FIFO pointer fields via `get_local_cb_interface().fifo_wr_ptr`.  This appears to be a genuinely novel pattern for the activation-reuse feature that the recipe should address — either adding it to the sanctioned list with a note about the architectural reason, or calling it out as a Device 2.0 item to clean.
