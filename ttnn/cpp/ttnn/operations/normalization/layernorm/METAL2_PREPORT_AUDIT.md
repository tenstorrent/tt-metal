# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/normalization/layernorm`

Single `DeviceOperation` in this directory. Two program factories:

- **`LayerNormDeviceOperation`**
  - `LayerNormMultiCoreProgramFactory` (`layernorm_op_multi_core.cpp`)
  - `LayerNormShardedProgramFactory` (`layernorm_op_multi_core_sharded.cpp` + `sharded_layernorm_factory_helpers.cpp/.hpp`)

The sharded factory delegates CB and kernel descriptor building to helpers in
`sharded_layernorm_factory_helpers.cpp`. Those files are in scope and are
audited here as part of `LayerNormShardedProgramFactory`.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `port_op_to_metal2_audit.md`.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/normalization/layernorm` |
| **Overall** | GREEN |
| **DOps / Factories** | `LayerNormDeviceOperation` → `LayerNormMultiCoreProgramFactory`, `LayerNormShardedProgramFactory` |
| *Prereqs* — ProgramDescriptor | Yes |
| *Prereqs* — Device 2.0 (every kernel used) | Yes |
| *Prereqs* — Cross-op escapes | Ok |
| *Feature Support* — overall | GREEN |
| *Feature Support* — Variadic-CTA | Ok |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — MeshWorkload needed | No |
| *TTNN Readiness* — Pybind `create_descriptor` | Yes: `layernorm_nanobind.cpp:319-393` (both factory classes) |
| *TTNN Readiness* — Other risky pybind | Yes: `LayerNormDeviceOperation` methods + `LayerNormParams`/`LayerNormInputs` structs at `layernorm_nanobind.cpp:223-316` |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — Custom override-RTA | No |
| *TTNN Readiness* — Fake CBs (address-only) | present: `(CB25/cb_reciprocals, compute kernel)` in both factories (workaround) |

**Fake CBs** — `cb_reciprocals` (CB index `c_25`) is backed by `recip_tensor.buffer()` via
`CBDescriptor.buffer` and accessed in the Welford compute kernels via
`get_pointer_to_cb_data` → `get_tile_address` (raw address read, no FIFO
`wait_front` / `pop_front` anywhere). No kernel produces into it; it is a
pre-populated read-only LUT. Litmus fails: no producer + no consumer (FIFO
sense) → **fake CB**. The port resolves it with the sanctioned fake-CB
workaround (see porting recipe). Not a gate.

## Result

**GREEN → brief issued.** All gates clear. Port can proceed after explicit user go-ahead and loading of `port_op_to_metal2_recipe.md`.

## Gate detail

- **ProgramDescriptor:** GREEN. Both factories return `tt::tt_metal::ProgramDescriptor` and populate it with `KernelDescriptor`, `CBDescriptor`, `SemaphoreDescriptor`. Includes: `<tt-metalium/program_descriptors.hpp>` in both factory files. No imperative `CreateProgram` / `CreateKernel` / `SetRuntimeArgs` calls.

- **Device 2.0 (every kernel used):** GREEN. All own kernels use Device 2.0 wrappers consistently:
  - `Noc noc;` + `noc.async_read(...)`, `noc.async_write(...)`, `noc.async_write_multicast(...)`, `noc.async_read_barrier()`, `noc.async_write_barrier()`, `noc.async_atomic_barrier()`.
  - `CircularBuffer cb_obj(id);` + `cb_obj.wait_front(n)`, `cb_obj.pop_front(n)`, `cb_obj.reserve_back(n)`, `cb_obj.push_back(n)`, `cb_obj.get_read_ptr()`, `cb_obj.get_write_ptr()`.
  - `TensorAccessor(args, addr)` and `TensorAccessorArgs<N>()` for all interleaved tensor accesses.
  - No `InterleavedAddrGen`, `ShardedAddrGen`, `InterleavedAddrGenFast`, raw `noc_async_read` free-function, or raw semaphore addresses found.
  - All `get_read_ptr()` / `get_write_ptr()` calls are member-form on wrapper objects (not free-function calls).

  Cross-op / shared-pool kernel headers included (`ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp`, `ttnn/kernel/dataflow/generate_bcast_scalar.hpp`) are Device 2.0 compliant — no Device 1.0 idioms found.

- **Feature compatibility:** every Appendix A entry in order:

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type or `experimental::CreateGlobalCircularBuffer` |
  | Dynamic CircularBuffer (borrowed memory) | GREEN | Multiple `CBDescriptor.buffer = <buffer>` sites (sharded input, b, stats, output, recip LUT) — port uses `borrowed_from` |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `.address_offset` set to non-zero anywhere |
  | Aliased Circular Buffers | GREEN | Multiple `CBDescriptor.format_descriptors` with 2 elements (Welford-fp32 alias pattern) — port uses `advanced_options.alias_with` |
  | GlobalSemaphore | N/A | No `GlobalSemaphore` type or `experimental::CreateGlobalSemaphore` |
  | Non-zero semaphore initial value | N/A | All `SemaphoreDescriptor::initial_value = 0` |
  | Dynamic TensorAccessor (`ArgConfig::Runtime*`) | N/A | No `ArgConfig::Runtime*` usage found |
  | `UpdateCircularBuffer*` | N/A | No `UpdateCircularBufferTotalSize` / `UpdateCircularBufferPageSize` / `UpdateDynamicCircularBufferAddressAndTotalSize` calls |
  | Variable-count compile-time arguments (CTA varargs) | N/A | No `std::vector<Tensor>` in `LayerNormInputs`; all CTA reads use fixed positional indices |

## Port-work summary  *(mirrors the brief)*

**Tensor bindings** (per binding, per factory):

*LayerNormMultiCoreProgramFactory:*
- `input (a)` — **Case 1** → re-express via `TensorParameter` / `TensorBinding`. Host: `a_addr = a.buffer()->address()` in reader RTA arg[0] (`layernorm_op_multi_core.cpp:187`). Kernel: `TensorAccessor(src0_args, src_addr)` in `reader_unary_interleaved_ln.cpp:111`.
- `residual (b)` — **Case 1** → re-express. Host: `b_dram_addr = b ? b.value().buffer()->address() : 0` in reader RTA arg[8] (`layernorm_op_multi_core.cpp:188`). Kernel: `TensorAccessor(src1_args, b_addr)` in `reader_unary_interleaved_ln.cpp:123`.
- `gamma` — **Case 1** → re-express. Host: `gamma_dram_addr` in reader RTA arg[6] (`layernorm_op_multi_core.cpp:189`). Kernel: `TensorAccessor(gamma_args, gamma_addr)` in `reader_unary_interleaved_ln.cpp:115`.
- `beta` — **Case 1** → re-express. Host: `beta_dram_addr` in reader RTA arg[7] (`layernorm_op_multi_core.cpp:190`). Kernel: `TensorAccessor(beta_args, beta_addr)` in `reader_unary_interleaved_ln.cpp:119`.
- `output` — **Case 1** → re-express. Host: `dst_addr = output.buffer()->address()` in writer RTA arg[0] (`layernorm_op_multi_core.cpp:131`). Kernel: `TensorAccessor(dst_args, dst_addr)` in `writer_unary_interleaved_start_id_blocked.cpp:30`.
- `recip_tensor` — **clean** (borrowed-memory DFB via `CBDescriptor.buffer`; accessed as fake CB — see Fake CBs heads-up).

*LayerNormShardedProgramFactory:*
- `input (a)` — **clean**. Sharded; CBDescriptor.buffer → borrowed-memory DFB (`sharded_layernorm_factory_helpers.cpp:1006`).
- `residual (b)` — **clean**. Sharded; CBDescriptor.buffer → borrowed-memory DFB (`sharded_layernorm_factory_helpers.cpp:1017-1024`).
- `stats` — **clean**. Sharded; CBDescriptor.buffer → borrowed-memory DFB (`sharded_layernorm_factory_helpers.cpp:1193-1199`).
- `output` (CB16/CB17) — **clean**. Sharded; CBDescriptor.buffer → borrowed-memory DFB (`sharded_layernorm_factory_helpers.cpp:1219-1245`).
- `gamma` (DRAM) — **Case 1** → re-express. Host: `gamma_dram_addr` in writer RTA arg[3] (`sharded_layernorm_factory_helpers.cpp:1501`). Kernel: `TensorAccessor(gamma_args, gamma_addr)` in `writer_unary_sharded_ln.cpp:80`.
- `beta` (DRAM) — **Case 1** → re-express. Host: `beta_dram_addr` in writer RTA arg[4] (`sharded_layernorm_factory_helpers.cpp:1502`). Kernel: `TensorAccessor(beta_args, beta_addr)` in `writer_unary_sharded_ln.cpp:94`.
- `recip_tensor` — **clean** (borrowed-memory DFB via `CBDescriptor.buffer`; accessed as fake CB — see Fake CBs heads-up).

**Custom hash:** none.

## Heads-ups  *(mirrors the brief)*

- **Notable LANDED constructs:**
  - **Aliased CBs** (multiple sites). The Welford-fp32 alias pattern adds a second `CBFormatDescriptor` element to share SRAM between two buffer indices with different `UnpackToDestMode` settings:
    - *MultiCore factory:* CB0 (`c_0` + `c_29`) `layernorm_op_multi_core.cpp:686-694`; CB18 (`c_18` + `c_30`) `layernorm_op_multi_core.cpp:707-715`; CB19 (`c_19` + `c_31`) `layernorm_op_multi_core.cpp:731-740`; CB23 (`c_23` + `c_29`) `layernorm_op_multi_core.cpp:808-816`.
    - *Sharded factory:* CB0 (`c_0` + `c_29`) `sharded_layernorm_factory_helpers.cpp:1007-1012`; CB24 (`c_24` + `c_29`) `sharded_layernorm_factory_helpers.cpp:1066-1071`.
    - Port uses `DataflowBufferSpec::advanced_options.alias_with`. Note: aliasing is an advanced/ninja feature — see `DFBAdvancedOptions` header for legality constraints.
  - **Borrowed-memory DFB** (multiple sites). Many CBs are backed by sharded tensor buffers (CB0, CB1, CB7, CB14, CB16, CB17, CB25 depending on factory and path). Port uses `DataflowBufferSpec::borrowed_from` naming the relevant `TensorParameter`.

- **Fake CBs (address-only):** `cb_reciprocals` (CB `c_25`) in both factories. The Welford compute kernels read this CB via `get_pointer_to_cb_data` → `get_tile_address` (raw L1 address) — no FIFO `wait_front` / `pop_front`. There is no kernel producer for this CB; it is a pre-populated LUT. This is a fake CB. The port resolves it with the sanctioned fake-CB workaround (see porting recipe). Sites: `layernorm_op_multi_core.cpp:825-833` (MultiCore), `sharded_layernorm_factory_helpers.cpp:1177-1186` (Sharded). Kernel access: `layernorm_welford.cpp:104-106` and `layernorm_sharded_welford.cpp` (same pattern).

- **Cross-op / shared kernels:** All kernel source files are owned by the layernorm directory. Out-of-directory `#include`s are shared library/utility headers (see Team-only for full inventory). No file-path kernel instantiation from other op families.

- **RTA varargs:** None.

- **TTNN factory analysis (porter-relevant):**
  - **Pybind `create_descriptor`** — delete at `layernorm_nanobind.cpp:319-393`. Both `LayerNormMultiCoreProgramFactory` and `LayerNormShardedProgramFactory` have their `create_descriptor` static method pybind-exposed via `nb::class_<>`. The extra `core_range_set` optional parameter (beyond the standard factory signature) is the tell that this is factory-innards exposure. The port deletes these bindings (sanctioned device-op-class edit).
  - **Other risky pybind** — `layernorm_nanobind.cpp:223-316`:
    - `bind_normalization_layernorm_params_and_inputs` (line 223): pybinds `LayerNormParams` (operation_attributes) and `LayerNormInputs` (tensor_args) structs with all fields exposed as `def_rw`. These are the DeviceOperation's internal parameter/input types, not the user-facing config classes.
    - `bind_normalization_layernorm_device_operation` (line 250): pybinds `LayerNormDeviceOperation` itself, exposing `compute_program_hash`, `create_output_tensors`, `compute_output_specs`, and `select_program_factory` as static methods. Direct introspection into DeviceOperation internals.
  - **Custom `override_runtime_arguments`:** None.

## Team-only

### TensorAccessor convertibility (per Case-2 binding)
No Case-2 bindings identified. All non-clean bindings are Case 1.

### Out-of-directory coupling & donor shape

**Op-level roll-up:** ✓ clean. All out-of-directory `#include`s are either LLK/HAL (class 1), official shared kernel library (class 2), or in-family normalization utilities. No cross-family function-call escapes with ⚠/✗/⭐ shapes.

**Summary table:**

| Op kernel | Donor file | Donor class | Shape | Status |
|---|---|---|---|---|
| All dataflow kernels | `api/dataflow/dataflow_api.h` | LLK/HAL | — | ✓ |
| All kernels | `api/dataflow/noc.h`, `api/dataflow/circular_buffer.h`, `api/tensor/noc_traits.h` | LLK/HAL | — | ✓ |
| `reader_unary_interleaved_ln.cpp` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp` | Shared kernel lib (class 2) | utility | ✓ |
| `reader_unary_interleaved_ln.cpp` | `ttnn/kernel/dataflow/generate_bcast_scalar.hpp` | Shared kernel pool (class 3) | utility | ✓ |
| `reader_unary_interleaved_ln.cpp` + others | `ttnn/operations/normalization/kernel_util/generic/blocked_range.h` | In-family (normalization) | utility | ✓ |
| Compute kernels | `ttnn/operations/normalization/kernel_util/compute/numeric.h`, `compute/memory.h`, `generic/bit.h` | In-family (normalization) | utility | ✓ |
| Various dataflow | `layernorm_dataflow_utils.h` | In-family (same op dir) | utility | ✓ |
| `writer_unary_sharded_ln.cpp`, `writer_unary_sharded_ln_rm_gb.cpp` | `reshard_writer.hpp` | In-family (same op dir) | utility | ✓ |
| Various sharded dataflow | `hostdevcommon/common_values.hpp` | LLK/HAL | constants | ✓ |

**Per-call detail:** omitted — all rolls are ✓.

**Borrowed kernel files (file-path kernel instantiation):** all kernel `.cpp` files are owned by this op's own directory (`ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/`). No kernel is instantiated from another op's directory or from a shared pool via file-path. The normalization utility headers (`kernel_util/`) are header-only inlined utilities, not separately-compiled kernel files.

### Relaxation candidates
No custom `compute_program_hash` to mine.

### TTNN factory analysis

**Q1 — Op-owned tensors?** No. No `create_device_tensor` or `allocate_tensor_on_device` calls in any program factory or device-operation code outside the standard `create_output_tensors` implementation. The `create_output_tensors` method in `layernorm_device_operation.cpp:447-462` calls `create_device_tensor` for the standard output, which is the op's declared output tensor — not an intermediate or scratch tensor owned by the factory.

**Q2 — MeshWorkload concept needed?** No. No `create_mesh_workload`, `create_workload_descriptor`, or `cached_mesh_workload_t` in this op. Single-program structure throughout. Q1 is also No, so the MeshWorkload-path plumbing artifact trap does not apply.

**Q3 — Pybind `create_descriptor`?** Yes.
- `LayerNormMultiCoreProgramFactory::create_descriptor` pybind: `layernorm_nanobind.cpp:319-358`.
- `LayerNormShardedProgramFactory::create_descriptor` pybind: `layernorm_nanobind.cpp:361-393`.
Both expose `create_descriptor` as a `def_static` on the factory class. The extra `core_range_set` optional parameter (`std::optional<CoreRangeSet>`) is present in the actual C++ factory signatures and is the tell that this is factory-innards exposure (not the normal user-facing op function). Port deletes these bindings.

**Q4 — Other migration-risky pybind?**
- `bind_normalization_layernorm_device_operation` (`layernorm_nanobind.cpp:250-316`): `nb::class_<LayerNormDeviceOperation>` with four static methods exposed: `compute_program_hash` (delegates to `device_operation::detail::compute_program_hash<LayerNormDeviceOperation>`, which is the framework default — not a custom hash), `create_output_tensors`, `compute_output_specs`, `select_program_factory`. Exposing DeviceOperation internals creates a migration surface.
- `bind_normalization_layernorm_params_and_inputs` (`layernorm_nanobind.cpp:223-248`): pybinds `LayerNormParams` (the `operation_attributes_t`) and `LayerNormInputs` (the `tensor_args_t`) directly with all fields as `def_rw`. These are internal operation plumbing types, not user-facing config objects.

**Q5 — Custom hash?** No. The `LayerNormDeviceOperation` class has no `compute_program_hash` member/override. The pybind exposure at `layernorm_nanobind.cpp:253` calls `ttnn::device_operation::detail::compute_program_hash<LayerNormDeviceOperation>` (the framework default), not a custom override.

**Q6 — Custom `override_runtime_arguments`?** No. Neither factory defines a `static void override_runtime_arguments(...)`.

## Misc anomalies  *(team-only, non-gating)*

- `layernorm_op_multi_core.cpp:131`: `auto dst_addr = output.buffer()->address()` is assigned early in the function (the "Device Setup" section) before the `ProgramDescriptor` is built. It is then used at line 604 in the writer runtime-args. This is a slightly unusual ordering (extracting the address before the CB / kernel setup) but not incorrect given the `ProgramDescriptor` API. Incidental note only.

- The writer runtime-args for the sharded factory pass `gamma_dram_addr` and `beta_dram_addr` as raw uint32 values even when gamma/beta are absent (`= 0` default in `layernorm_op_multi_core_sharded.cpp:157-158`). Kernel-side, `TensorAccessor(gamma_args, gamma_addr)` is only constructed inside `#ifdef FUSE_GAMMA` / `#ifdef FUSE_BETA` guards, so no actual read occurs. Not a correctness issue, but the port should confirm the `TensorParameter` declarations handle the optional-tensor case cleanly.

## Questions for the user  *(none)*

## Recipe notes

- The recipe's "Fake CB" litmus test ("does the CB have a producer *and* a consumer?") was straightforward to apply for `cb_reciprocals`: the Welford kernels read it via `get_tile_address` (raw pointer, no FIFO). However, the recipe's phrasing "a sharded reader's fake-push satisfying a waiting compute consumer" as the "canonical legit case" could be read as implying that only dataflow kernels can be producers. In practice the borrowed-memory CB pattern here is unambiguously fake (compute reads by raw address with no preceding `wait_front`). The recipe is correct; just noting the signal that resolved it.
- The recipe's "Dynamic CircularBuffer" Appendix A entry lists `layernorm/device/layernorm_op_multi_core.cpp` as an example in the wild. That example is directly applicable — confirmed.
- The `core_range_set` parameter present on both factory `create_descriptor` signatures is beyond what the recipe's Q3 canonical example describes ("extra `core_range_set` parameter that exists **only** to drive the pybind hook"). Here the `core_range_set` parameter is a genuine functional parameter (sharded factory validates shard spec against it; interleaved factory uses it for core range selection). It is *also* exposed in the pybind — but the parameter is not pybind-only. Flagging both factory pybind exposures under Q3 regardless, since the pybind-of-factory-innards pattern is what Q3 targets; whether the extra parameter is purely pybind-driven or functional doesn't change the migration-risky classification.
