# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/conv/conv2d`

- **`Conv2dDeviceOperation`** (`ttnn::prim`, `device/conv2d_device_operation.hpp:29`)
  - `Conv2dShardedProgramFactory` (`device/conv2d_op_sharded_program_factory.cpp`, ~1589 lines) — three sub-variants selected inside the factory body: height-sharded, block-sharded, 1D-depthwise.
  - `Conv2dWidthShardedProgramFactory` (`device/conv2d_op_width_sharded_program_factory.cpp`, ~755 lines).

Shared, non-factory host code lives at the op root: `conv2d_op_program_factory_common.cpp/.hpp` (CB-info build + `emit_cb_descriptors` + L1 checks). All 12 kernel files under `device/kernels/` are referenced by one or both factories; none are dead.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `port_op_to_metal2_audit.md`.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/conv/conv2d` |
| **Overall** | **RED** |
| **DOps / Factories** | `Conv2dDeviceOperation` → `Conv2dShardedProgramFactory`, `Conv2dWidthShardedProgramFactory` |
| *Prereqs* — ProgramDescriptor | Yes (both factories build `ProgramDescriptor` via `WorkloadDescriptor`) |
| *Prereqs* — Device 2.0 (every kernel used) | Yes (`experimental::CB` / `Noc` / `TensorAccessor`; only sanctioned `get_tile_size`/`get_local_cb_interface` free-fns) |
| *Prereqs* — Cross-op escapes | Ok (only donor: `pool/.../experimental_device_api.hpp`, itself Device 2.0; no cross-op kernel file-path instantiation) |
| *Feature Support* — overall | GREEN |
| *Feature Support* — Variadic-CTA | Ok (fixed `tensor_args_t = {a, b, optional bias}`) |
| *TTNN Readiness* — Op-owned tensors | **Yes (BLOCKER): both factories allocate+park `conv_reader_indices`** — sharded `device/conv2d_op_sharded_program_factory.cpp:1563-1572`; width `device/conv2d_op_width_sharded_program_factory.cpp:727-736` |
| *TTNN Readiness* — MeshWorkload needed | No (op-owned-tensor artifact only — single-program, structurally identical per coord; see Q2) |
| *TTNN Readiness* — Pybind `create_descriptor` | No (no factory/device-op class binding in `conv2d_nanobind.cpp`) |
| *TTNN Readiness* — Other risky pybind | None |
| *TTNN Readiness* — Custom hash | Yes → delete (`device/conv2d_device_operation.cpp:144`, `Conv2dHashableParams`) |
| *TTNN Readiness* — Custom override-RTA | No (no `override_runtime_arguments` in the op) |
| *TTNN Readiness* — Fake CBs (address-only) | None definitively identified (see Heads-ups for the READER_INDICES note) |

## Result

**RED — blocked on framework work, not porter-resolvable.**

The primary and only hard blocker is **op-owned device resources**: both program factories allocate an intermediate device tensor (`conv_reader_indices`) that is **not** in `tensor_args` or `tensor_return_value`, and park it (wrapped in `shared_ptr<Tensor>`) on the `WorkloadDescriptor::buffers` vector so its lifetime tracks the cached workload rather than the caller's tensors. Per `port_op_to_metal2_ttnn_factory.md`, the only Metal 2.0 factory concept on `main` is `ProgramSpecFactoryConcept`, which **requires single-program with no op-owned device resources** — every tensor a factory references must be reachable from `tensor_args` / `tensor_return_value`. The framework adapter has an explicit TODO for op-owned resources; today a `TensorArgument` that doesn't reference an input/output tensor `TT_FATAL`s. This is the "op allocates/parks its own intermediate `MeshTensor`" case the factory doc calls out as **blocked on framework, record RED and stop** — not a porter-resolvable item.

This is not a permanent block: the op is *morally* single-program (the `WorkloadDescriptor` here is a resource-lifetime workaround, not genuine multi-program / per-coord variation — see Q2), so it unblocks once op-owned-resource support lands in the Metal 2.0 factory framework. The finding routes to that framework workstream.

Every other audit subject is clean (ProgramDescriptor ✓, Device 2.0 ✓, Features GREEN, no risky pybind, no override-RTA). The only port-time work items behind the blocker are routine: re-express the tensor bindings as `TensorParameter`/`TensorBinding` and delete the custom `compute_program_hash`. No clean code-path subset exists — **both** factories carry the op-owned-tensor blocker, so a scoped-subset port is not available.

## Gate detail

- **ProgramDescriptor:** **GREEN.** Both factories populate `tt::tt_metal::ProgramDescriptor` (via `build_program_descriptor_sharded` / `build_program_descriptor`) and use `KernelDescriptor` / `CBDescriptor` / `SemaphoreDescriptor` (`device/conv2d_op_sharded_program_factory.cpp:186`, `:725`, `:786`; width `:61`, `:367`). No imperative `host_api.hpp` builder calls (`CreateProgram`/`CreateKernel`/`CreateCircularBuffer`/`SetRuntimeArgs`) in the op.

- **Device 2.0 (every kernel used):** **GREEN.** All 12 referenced kernels are Device 2.0 compliant. Data movement uses Device-2.0 wrappers — `experimental::CB` (alias for the kernel-side `CircularBuffer` wrapper), `Noc noc` objects with `.async_read`/`.async_write`, and the `experimental::set_read_state`/`read_with_state` helpers from the donor `experimental_device_api.hpp`. Tensor reads go through `TensorAccessor(args, addr)` (e.g. `device/kernels/conv_reader_common.hpp:361-362`, `weights_reader_width_sharded.cpp:42-45`). **No gating Device 1.0 idioms** anywhere: no `InterleavedAddrGen` / `ShardedAddrGen` / `InterleavedPow2AddrGen`, no `get_noc_addr_from_bank_id`, no raw `noc_async_read(`/`noc_async_write(`. The only CB-index free-functions present — `get_tile_size(cb_id)` and `get_local_cb_interface(cb_id)` (`conv_reader_common.hpp:24-25`, etc.) — are the **sanctioned** Device 2.0 free-functions the migration guide keeps in its migrated examples; they are **not** holdovers. (`get_compile_time_arg_val` / `get_arg_val` are positional CTA/RTA reads — these are Metal 2.0 port-time renames to named `args::`, not a Device 2.0 concern.)

  No holdover table — none to report.

- **Feature compatibility:** every Appendix A entry, in order:

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` / `CreateGlobalCircularBuffer` / `.global_circular_buffer` field. (`conv2d_device_operation.hpp` includes `<tt-metalium/global_circular_buffer.hpp>` but the type is never used in the op factories.) |
  | Dynamic CircularBuffer (borrowed memory) | GREEN | In use: `emit_cb_descriptors` sets `CBDescriptor::buffer` for `is_globally_allocated` CBs — ACT_SHARDED→input buffer, OUT/MATMUL_PARTIALS→output buffer, READER_INDICES→indices buffer (`conv2d_op_program_factory_common.cpp:771-794`). Port uses `DataflowBufferSpec::borrowed_from`. |
  | CBDescriptor `address_offset` (non-zero) | N/A | `address_offset` never set on any `CBDescriptor` (grep: zero hits). |
  | Aliased Circular Buffers | N/A | Every `format_descriptors` initializer is single-element (`conv2d_op_program_factory_common.cpp:789`); `overlapped_by_cb` (`:799-802`) reuses an index without a multi-element format vector. |
  | GlobalSemaphore | N/A | No `GlobalSemaphore` / `CreateGlobalSemaphore` / `global_semaphore.hpp`. |
  | Non-zero semaphore initial value | N/A | All `SemaphoreDescriptor.initial_value = 0` (`conv2d_op_sharded_program_factory.cpp:728`; width `:370`, `:376`). |
  | Dynamic TensorAccessor (`ArgConfig::Runtime*`) | N/A | No `ArgConfig::Runtime*` tokens; `TensorAccessorArgs(buffer)` is the standard single-arg form. |
  | `UpdateCircularBuffer*` | N/A | No `UpdateCircularBufferTotalSize` / `UpdateCircularBufferPageSize` / `UpdateDynamicCircularBufferAddressAndTotalSize`. (Two factory comments *mention* `UpdateDynamicCircularBufferAddress` only to note it is no longer needed — no call.) |
  | Variable-count compile-time arguments (CTA varargs) | N/A | `tensor_args_t = Conv2dInputs{a, b, optional bias}` (`conv2d_device_operation_types.hpp:240-244`) — fixed count, no `std::vector<Tensor>`. No kernel reads CTAs at a runtime-varying index. |

## Port-work summary  *(mirrors the brief — applicable only once the op-owned-resource framework gap is closed)*

- **Tensor bindings** (per binding):
  - `a` (activation, input) — **clean / borrowed-memory DFB.** Sharded; bound as the ACT_SHARDED globally-allocated CB (`emit_cb_descriptors` → input buffer). Port via `DataflowBufferSpec::borrowed_from`. No `TensorAccessor` for `a` is correct (causal-link gate).
  - `b` (weights, input) — **Case 1 (re-express).** Read via `TensorAccessorArgs(b.buffer()).append_to(...)` → kernel-side `TensorAccessor` (sharded `device/conv2d_op_sharded_program_factory.cpp:1035`; width `:578`, kernels `weights_reader_width_sharded.cpp:27-42`, `writer_tiled_out_2d_mcast_sender...:75-158`, `reader_writer_tiled_out_1d_mcast_sender...:56-135`). Re-express via `TensorParameter`/`TensorBinding`.
  - `bias` (optional, input) — **Case 1 (re-express)**, conditional binding. `TensorAccessorArgs(bias ? bias->buffer() : nullptr)` (`conv2d_op_sharded_program_factory.cpp:1036`; width `:579`). Bind only when `has_bias`; emit a matching define and `#ifdef`-gate kernel-side.
  - `output` — **clean / borrowed-memory DFB.** OUT / MATMUL_PARTIALS globally-allocated CB → output buffer. Port via `borrowed_from`.
  - `conv_reader_indices` (op-owned intermediate) — **BLOCKED (the gate).** Read via `TensorAccessorArgs(conv_reader_indices_buffer).append_to(...)` with the buffer's `->address()` and `->page_size()` baked into compile-time args (sharded `:873-875`; width `:569-571`), and consumed kernel-side through `TensorAccessor` at `conv_reader_common.hpp:361-362`. *Were it an input/output tensor* this would be Case 1; it is not — it is an op-owned device tensor, which is the framework gate. Resolving it cleanly needs op-owned-resource support (so the indices tensor can be a framework-managed resource) — it is **not** a porter Case-1 item today.

- **Custom hash:** delete custom `compute_program_hash` → default (sanctioned exception). Located `device/conv2d_device_operation.cpp:144-165`; builds a `Conv2dHashableParams` (defined `conv2d_device_operation_types.hpp:221-238`) and hashes it with `tensor_args`.

## Heads-ups  *(mirrors the brief)*

- **Notable LANDED constructs:**
  - **Borrowed-memory DFB** — `emit_cb_descriptors` (`conv2d_op_program_factory_common.cpp:771-794`) binds CBs onto `input`/`output`/`indices` buffers via `CBDescriptor::buffer`. Port uses `DataflowBufferSpec::borrowed_from` for ACT_SHARDED, OUT, MATMUL_PARTIALS, READER_INDICES.
- **Fake CBs (address-only):** none confirmed. The READER_INDICES CB is borrowed-memory on the op-owned indices buffer; the reader does build a real `TensorAccessor` against it (`conv_reader_common.hpp:361-362`) rather than reading it purely by base pointer, so it is not the recip-LUT fake-CB shape. (Moot regardless — the indices tensor itself is the gate.)
- **Cross-op / shared kernels:** all 12 kernel files are conv2d-owned (under `device/kernels/`); **no cross-op file-path kernel instantiation**. One donor `#include`: `ttnn/cpp/ttnn/operations/pool/device/kernels/experimental_device_api.hpp` (a Device-2.0 convenience header providing `experimental::CB` and NoC helpers), included by 6 kernels. It is header-only Device 2.0 glue, not an addr-gen donor — no gate.
- **RTA varargs:** none observed (no `num_runtime_varargs`, no counted `get_arg_val(i)` loop over a runtime-known `N`).
- **TTNN factory analysis (porter-relevant):** pybind `create_descriptor` — none. Other risky pybind — none. Custom `override_runtime_arguments` — none. (Only the custom hash is a device-op-class edit, carried above.)

## Team-only

### TTNN factory analysis (the six questions)

1. **Op-owned tensors? — YES (the blocker).** Both factories allocate `conv_reader_indices` via `construct_on_host_config_tensor` + `move_config_tensor_to_device`, wrap it in `std::make_shared<Tensor>`, and push `{owner, buffer}` onto `WorkloadDescriptor::buffers`:
   - Sharded: `device/conv2d_op_sharded_program_factory.cpp:1563-1572` (alloc + park), buffer consumed at `:1578` (`build_program_descriptor_sharded(..., conv_reader_indices_buffer)`), header comment `device/conv2d_op_sharded_program_factory.hpp:13-22`.
   - Width: `device/conv2d_op_width_sharded_program_factory.cpp:727-736` (alloc + park), consumed at `:742`, header comment `device/conv2d_op_width_sharded_program_factory.hpp:13-22`.
   - The tensor is neither in `tensor_args_t` (`Conv2dInputs{a, b, bias}`) nor `tensor_return_value_t` (`Tensor` = the conv output). The shared_ptr + `WorkloadDescriptor::buffers` parking exists specifically so `~Tensor` doesn't force-free the device buffer while the cached program is alive (see the rationale comments at the alloc sites).

2. **MeshWorkload concept needed? — NO (op-owned-tensor artifact only).** Both factories provide `create_workload_descriptor` and emit a `WorkloadDescriptor` with multiple `programs`, but the per-coord program is **structurally identical** — the factory builds one `ProgramDescriptor` and copies it into each coord range (`conv2d_op_sharded_program_factory.cpp:1574-1587`: *"Single-device op: per-coord program is structurally identical for every coord ... conv2d doesn't depend on cluster position. Build the ProgramDescriptor once and copy into each range entry."*; width `:738-751`). There is no genuine cross-program/cross-device coordination. The op is on the workload/multi-program path **only** because the legacy framework needed somewhere to park the op-owned `conv_reader_indices` tensor (Q1) — exactly the "legacy MeshWorkload is a resource workaround, not a real multi-program need" case in `port_op_to_metal2_ttnn_factory.md`. Records as morally single-program, blocked-on-framework (op-owned-resource support), a resource-workaround unwind.

3. **Pybind `create_descriptor`? — NO.** `conv2d_nanobind.cpp` binds only the normal op surface (the user-facing function, program-config/param structs, enums). No `nb::class_<...ProgramFactory>` and no `def_static("create_descriptor", ...)`.

4. **Other migration-risky pybind? — NO.** No `nb::class_<>` wrapping the `Conv2dDeviceOperation` or either factory; no device-op methods (`compute_program_hash`, `create_output_tensors`, `compute_output_specs`, `select_program_factory`) exposed to Python; no introspection entry point returning `ProgramDescriptor`.

5. **Custom hash? — YES.** `Conv2dDeviceOperation::compute_program_hash` at `device/conv2d_device_operation.cpp:144`. Treatment (delete → default) is carried in Port-work summary; see also relaxation-candidate note below.

6. **Custom override-runtime-args? — NO.** No `override_runtime_arguments` defined on either factory (grep: zero hits in `device/`).

### TensorAccessor convertibility (per Case-2 binding)
No Case-2 bindings. `b` and `bias` are awkward-but-routine Case-1 (standard page-iterable `TensorAccessor`). The op-owned `conv_reader_indices` is also read via a normal `TensorAccessor` (would be Case 1 if it were an input/output) — the obstacle is *ownership*, not the access pattern.

### Out-of-directory coupling & donor shape
- **Op-level roll-up: ✓ clean.** No function-call escape into another op family's signatures requiring conversion; the single donor header is Device-2.0 native.
- **Donor headers (cross-dir #includes in kernels):**
  - `ttnn/cpp/ttnn/operations/pool/device/kernels/experimental_device_api.hpp` — **cross-family donor**, but a Device-2.0 convenience header (aliases `experimental::CB`, NoC read-state helpers `set_read_state`/`read_with_state`/`local_addr`). Included by `activation_reader_width_sharded.cpp`, `compute_depthwise_conv1d.cpp`, `conv_reader_common.hpp`, `reader_depthwise_conv1d.cpp`, `conv_bmm_tilize.cpp`, `weights_reader_width_sharded.cpp`. The functions it provides take `Noc` / `uint32_t addr` / `experimental::CB` (Device-2.0 shapes) — ✓ excellent / OK, no conversion work, no gate.
  - `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp`, `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` — official shared kernel library (lib-team-owned), no concern.
  - All other includes are `tt_metal/*` / `api/*` LLK/HAL — no concern.
- **Borrowed kernel files (file-path instantiation):** none. All 12 kernel `.cpp` files instantiated by the factories live under `conv/conv2d/device/kernels/` and are conv2d-owned. No co-borrower port-together set.

### Relaxation candidates (mined from the custom hash before deletion) — FALLIBLE, candidates to verify, default strict
The custom hash (`Conv2dHashableParams`, `conv2d_device_operation_types.hpp:221-238`) keys on `sliding_window_config`, `parallelization_config`, `block_config`, `memory_config`, `dtype`, `input_tensor_shape`, `compute_kernel_config`, and several bool/enable flags — but **omits `groups`, `has_bias`'s companion `bias` tensor spec only implicitly, and `pre_op_l1_allocation_size_bytes`** relative to the full `Conv2dParams`. It does include `tensor_args` in the final hash call (`conv2d_device_operation.cpp:164`), so `TensorSpec` is keyed via the tensors. No clear-cut `dynamic_tensor_shape` / `match_padded_shape_only` candidate emerges — conv is highly shape-specialized (block/parallelization config derived from exact shapes), so strict matching is appropriate. No relaxation recommended; the default (strict) is the safe choice. Flag for the team only as "no relaxation candidate found."

## Misc anomalies  *(team-only, non-gating)*
- `conv2d_device_operation.hpp:12` includes `<tt-metalium/global_circular_buffer.hpp>` but `GlobalCircularBuffer` is never used in the device op or factories — a likely-stale include. Routes to the op owner; not porter work and not a gate.
- `Conv2dParams::pre_op_l1_allocation_size_bytes` (`conv2d_device_operation_types.hpp:217`) is carried in the attributes but excluded from `Conv2dHashableParams`. Intentional if it is purely a diagnostic/L1-budget input that doesn't change the program; noted for the op owner only.

## Per-DeviceOperation attribution
Single `DeviceOperation` (`Conv2dDeviceOperation`). Both factories carry the identical op-owned-`conv_reader_indices` blocker (Q1) and the same morally-single-program shape (Q2); the custom hash and pybind findings are device-op-class-level (shared). No per-factory divergence in the gate verdict — RED applies uniformly, no clean factory subset.

## Questions for the user
None — the blocker is unambiguous (op-owned device resource, framework gap not on `main`) and every other subject is clean. Surfaced as RED per the factory feasibility gate.

## Recipe notes
None. The audit doc and the factory feasibility-gate doc covered this op's situation directly (the "legacy MeshWorkload as a resource workaround for an op-owned tensor" case is explicitly described in `port_op_to_metal2_ttnn_factory.md`, and the conv factories' own header comments even cite the `workload_descriptor.hpp` WorkloadBuffer rationale — making the recognition unambiguous).

## TTNN ProgramFactory

### Concept
**BLOCKED** (op-owned device resources). Does **not** fit `ProgramSpecFactoryConcept` — the only Metal 2.0 factory concept on `main`.

### Fit
- Single vs multi-program: **single** — one `ProgramSpec` stamped across the mesh (the legacy multi-program `WorkloadDescriptor` is a resource-lifetime workaround, not genuine per-coord variation; see Q2 / the factory-doc heads-up).
- Op-owned device resources: **present — BLOCKED.** `conv_reader_indices` intermediate device tensor allocated + parked on `WorkloadDescriptor::buffers` by both factories (sharded `conv2d_op_sharded_program_factory.cpp:1563-1572`; width `conv2d_op_width_sharded_program_factory.cpp:727-736`). Not in `tensor_args` / `tensor_return_value`.
- Tensor-arg matching: strict (default; no deviation warranted — conv is shape-specialized).
- Legacy-to-Metal-2.0 shape: legacy `MeshWorkload`/`WorkloadDescriptor` was a **resource workaround** (op-owned tensor lifetime), not a real multi-program need — morally 1:1 single-program once op-owned-resource support lands.

### Custom compute_program_hash
Present at `device/conv2d_device_operation.cpp:144` (builds `Conv2dHashableParams`, `conv2d_device_operation_types.hpp:221`) → the port would delete it, reverting to the default reflection-based hash. (Applies only once the op is unblocked.)

### Stop signals
**BLOCKED — missing framework capability: op-owned device resources** (a factory-allocated `MeshTensor` whose lifetime tracks the cached entry). This support is not on `main`; the richer factory-concept design that would carry it is being reworked after fast-path performance findings. The overall audit result is **RED**. The op unblocks when op-owned-resource support lands in the Metal 2.0 factory framework; route this report to that framework workstream.
