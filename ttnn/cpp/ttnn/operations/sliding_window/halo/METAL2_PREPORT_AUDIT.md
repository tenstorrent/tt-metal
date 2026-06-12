# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/sliding_window/halo`

Identifying section:

- **`HaloDeviceOperation`** (`ttnn::prim::HaloDeviceOperation`)
  - `UntilizeWithHaloProgramFactory` (`device/untilize_with_halo_program_factory.cpp`)

Factory entry point is `UntilizeWithHaloProgramFactory::create_workload_descriptor` (returns `tt::tt_metal::WorkloadDescriptor`), which internally calls `build_halo_program` (returns `ProgramDescriptor`). The factory allocates four op-owned halo config tensors and parks them on the `WorkloadDescriptor`; the `WorkloadDescriptor` path is a plumbing artifact of op-owned tensors, not a genuine MeshWorkload need.

Kernels exercised by this factory:
- `device/kernels/dataflow/halo_gather.cpp` (reader_0 and reader_1 — same file, two `KernelDescriptor` instances with different CTAs)
- `device/kernels/compute/pack_untilize.cpp` (only instantiated when input is tiled, i.e. `!skip_untilize`)

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `port_op_to_metal2_audit.md`.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/sliding_window/halo` |
| **Overall** | GREEN |
| **DOps / Factories** | `HaloDeviceOperation` → `UntilizeWithHaloProgramFactory` |
| *Prereqs* — ProgramDescriptor | Yes |
| *Prereqs* — Device 2.0 (every kernel used) | Yes |
| *Prereqs* — Cross-op escapes | Ok |
| *Feature Support* — overall | GREEN |
| *Feature Support* — Variadic-CTA | Ok |
| *TTNN Readiness* — Op-owned tensors | Yes: `UntilizeWithHaloProgramFactory::create_workload_descriptor` — `pad_config_device_tensor0/1`, `gather_config_device_tensor0/1` (lines 436–458, `untilize_with_halo_program_factory.cpp`) |
| *TTNN Readiness* — MeshWorkload needed | No (op-owned-tensor artifact only): `create_workload_descriptor` is on the `WorkloadDescriptor` path because the factory allocates the four halo config tensors internally and parks their `Tensor` owners on `workload_descriptor.buffers`. This is not a genuine multi-program / cross-device-coordination need. |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Other risky pybind | None |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — Custom override-RTA | No |
| *TTNN Readiness* — Fake CBs (address-only) | Present: `out_cb` (`out_cb_id`) at `halo_gather.cpp:108,205`; `padding_config_cb` (`padding_config0/1`) at `halo_gather.cpp:105`; `gather_config_cb` (`gather_config0/1`) at `halo_gather.cpp:343` (workaround — see Fake CBs section below) |

**Fake CBs** = CBs used purely as an address source, no producer+consumer pair.
- `out_cb` is backed by `dst_buffer` (output tensor). The kernel accesses it only via `out_cb.get_write_ptr()` (lines 108, 205 of `halo_gather.cpp`). No `reserve_back`/`push_back`/`wait_front`/`pop_front` on `out_cb`. → Fake CB.
- `padding_config_cb` is backed by `padding_config_buffer0/1` (when `!config_tensors_in_dram`). Accessed only via `padding_config_cb.get_read_ptr()` (line 105). → Fake CB.
- `gather_config_cb` is backed by `gather_config_buffer0/1` (when `!config_tensors_in_dram`). Accessed only via `gather_config_cb.get_read_ptr()` (line 343). → Fake CB.

Each fake CB will be resolved with the sanctioned fake-CB workaround at port time (see the port recipe). None of these gate the port.

## Result

**GREEN → brief issued.** All gates clear. Op is on the `ProgramDescriptor` API; all kernels are Device 2.0 compliant; no UNSUPPORTED Appendix A features are in use. Port can proceed after user go-ahead.

## Gate detail

- **ProgramDescriptor:** GREEN — `build_halo_program` in `device/untilize_with_halo_program_factory.cpp` populates a `ProgramDescriptor` using `CBDescriptor`, `KernelDescriptor`, and `KernelDescriptor::runtime_args` / `compile_time_args`. No legacy imperative `host_api.hpp` calls (`CreateProgram` / `CreateKernel` / `CreateCircularBuffer` / `SetRuntimeArgs`) anywhere in the factory.

- **Device 2.0 (every kernel used):** GREEN — both kernels are fully Device 2.0 compliant.

  **`halo_gather.cpp`:** Uses `Noc` wrapper object for all NoC operations (`noc.async_read`, `noc.async_write`, `noc.async_read_barrier`, `noc.async_write_barrier`), `experimental::CB` wrapper objects (`padding_config_cb`, `gather_config_cb`, `src_cb`, `in_cb`, `out_cb`, `pad_cb`) for all CB operations, member-function forms exclusively (`get_read_ptr()`, `get_write_ptr()`, `wait_front()`, `pop_front()`, `reserve_back()`, `push_back()`), and `use<experimental::CB::AddrSelector::READ_PTR>(in_cb)` for address-selector form. No free-function CB-index legacy calls; no `InterleavedAddrGen`/`ShardedAddrGen`; no raw semaphore addresses.

  **`pack_untilize.cpp`:** Uses `compute_kernel_lib::untilize_init`, `compute_kernel_lib::untilize`, `compute_kernel_lib::untilize_uninit` from `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp`. Internally (`untilize_helpers.inl`) these use `DataflowBuffer` wrapper objects (`wait_front`, `pop_front`, `push_back`, `reserve_back`) and `compute_kernel_hw_startup` — all Device 2.0 APIs. No Device 1.0 idioms.

  | File | Line | Call | Wrapper in scope | Note |
  |---|---|---|---|---|
  | *(no violations)* | | | | All calls use Device 2.0 wrapper methods |

- **Feature compatibility:**

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type or `CreateGlobalCircularBuffer` calls anywhere in op |
  | Dynamic CircularBuffer (borrowed memory) | GREEN | `src_cb_id`, `out_cb_id`, and four config-tensor CBs (when `!config_tensors_in_dram`) all use `CBDescriptor::buffer = <Buffer*>` (line 76 of factory, `add_cb` helper). Metal 2.0 port uses `DataflowBufferSpec::borrowed_from`. Note: `out_cb` and the config CBs are fake CBs (no FIFO pair) — see Fake CBs section. |
  | CBDescriptor `address_offset` (non-zero) | N/A | `add_cb` helper only sets `.total_size`, `.core_ranges`, `.format_descriptors`, and `.buffer`; `.address_offset` is never set (defaults to 0). |
  | Aliased Circular Buffers | N/A | Every `CBDescriptor` has a single-element `format_descriptors` initializer — no aliased CB pattern. |
  | GlobalSemaphore | N/A | No semaphores used anywhere in the op. |
  | Non-zero semaphore initial value | N/A | No semaphores. |
  | Dynamic TensorAccessor (`ArgConfig::Runtime*`) | N/A | No `ArgConfig::Runtime*` enumerators in host code. |
  | `UpdateCircularBuffer*` | N/A | No `UpdateCircularBuffer` calls. |
  | Variable-count compile-time arguments (CTA varargs) | N/A | `tensor_args_t = Tensor` (single tensor). No `std::vector<Tensor>`. CTA lists are fixed-count. |

## Port-work summary *(mirrors the brief)*

- **Tensor bindings** (per binding):
  - `input_tensor` (the op's declared input) — backed by `src_buffer`; passed to `add_cb` as `.buffer = src_buffer` (line 152 of factory). Kernel producer: `src_cb.reserve_back/push_back` (lines 317–318 of `halo_gather.cpp`); kernel consumer: `in_cb.wait_front/pop_front` in `run_halo_gather` (when `!skip_untilize`), or `src_cb.wait_front` (when `skip_untilize`). Real borrowed-memory DFB — **Case 1** (re-express via `TensorParameter` / `TensorBinding`; kernel builds `TensorAccessor(ta::name)`).
  - `output_tensor` (the op's declared output) — backed by `dst_buffer`; passed to `add_cb` as `.buffer = dst_buffer` (line 166 of factory). Fake CB (address-only, no FIFO pair). Port uses **fake-CB workaround** (see recipe); also **Case 1** for the binding itself (re-express via `TensorParameter`).
  - `padding_config_buffer0/1`, `gather_config_buffer0/1` (op-owned config tensors) — two code paths:
    - When `!config_tensors_in_dram`: passed to `add_cb` as `.buffer = <config_buf>` (lines 238, 247, 257, 266 of factory). Fake CBs (address-only). Port uses fake-CB workaround; **Case 1** for the binding.
    - When `config_tensors_in_dram`: `buffer->address()` is baked into CTAs (lines 307–317 of factory), and `TensorAccessorArgs(buffer).append_to(...)` is also appended to CTAs (lines 319–323). Kernel uses `TensorAccessorArgs<N>()` + `TensorAccessor(args, addr)` (lines 296–301 of `halo_gather.cpp`). This is the CTA-baked-address form. **Case 1** — re-express via `TensorParameter`; the CTA-baked address and `TensorAccessorArgs<N>()` CTA dance both disappear.
- **Custom hash:** None.

## Heads-ups *(mirrors the brief)*

- **Notable LANDED constructs:**
  - **Borrowed-memory DFB** — six `CBDescriptor::buffer` non-null assignments (via the `add_cb` helper at `untilize_with_halo_program_factory.cpp:76`). Port uses `DataflowBufferSpec::borrowed_from` for each. See the Dynamic CircularBuffer entry in the migration guide.
  - **CTA-baked buffer addresses** — `padding_config_buffer0/1->address()` and `gather_config_buffer0/1->address()` are baked into compile-time args at `untilize_with_halo_program_factory.cpp:307–317` (the `config_tensors_in_dram` branch). These are Case 1; the port replaces them with `TensorParameter` bindings and the kernel side uses `TensorAccessor(ta::name)`.

- **Fake CBs (address-only):** Three distinct fake CB usages — port resolves each with the sanctioned fake-CB workaround (see the porting recipe):
  - `out_cb` (`out_cb_id`, backed by `dst_buffer`): `halo_gather.cpp:108,205` — `out_cb.get_write_ptr()` is the only access; no FIFO producer+consumer.
  - `padding_config_cb` (`padding_config0/1`, backed by config tensor when `!config_tensors_in_dram`): `halo_gather.cpp:105` — `padding_config_cb.get_read_ptr()` only.
  - `gather_config_cb` (`gather_config0/1`, backed by config tensor when `!config_tensors_in_dram`): `halo_gather.cpp:343` — `gather_config_cb.get_read_ptr()` only.

- **Cross-op / shared kernels:**
  - `halo_gather.cpp` includes `ttnn/cpp/ttnn/operations/pool/device/kernels/experimental_device_api.hpp` (cross-family donor from the pool family). This header provides only Device 2.0 convenience wrappers (`Noc`, `CB` alias for `experimental::CB`, `set_read_state`, `read_with_state` etc.) — no Device 1.0 content. Shape: Device 2.0 native (see Out-of-directory coupling section for full inventory). No special port handling required — the include continues to work as-is.
  - `pack_untilize.cpp` borrows `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` from the shared kernel library. This library is Device 2.0 compliant (uses `DataflowBuffer` wrappers throughout).
  - **Port-together coupling:** `pack_untilize.cpp` is sourced from `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp`. If other ops also instantiate kernels using this lib, their Metal 2.0 ports may need to be coordinated. The `halo_gather.cpp` file is owned by this op (not shared). Verify before porting whether any other ops instantiate `halo_gather.cpp` directly.

- **TTNN factory analysis (porter-relevant):**
  - **Pybind `create_descriptor`:** None.
  - **Other risky pybind:** None. The `sliding_window_nanobind.cpp` binds only `ParallelConfig` and `Op2DSliceConfig` structs — no `DeviceOperation` or factory class internals.
  - **Custom `override_runtime_arguments`:** None.

## Team-only

### TensorAccessor convertibility (per Case-2 binding)

No Case-2 bindings found in this op. All bindings are Case 1 (re-express via `TensorParameter`).

### Out-of-directory coupling & donor shape

**Op-level roll-up:** ✓ clean — all donor includes are LLK/HAL, shared kernel library (Device 2.0), or a cross-family helper header that provides only Device 2.0 wrappers.

**Summary table:**

| Op kernel | Donor file | Donor class | Shape | Status |
|---|---|---|---|---|
| `halo_gather.cpp` | `api/dataflow/dataflow_api.h` | LLK / HAL | N/A (LLK) | ✓ no concern |
| `halo_gather.cpp` | `ttnn/operations/pool/device/kernels/experimental_device_api.hpp` | Cross-family (pool) | Device 2.0 wrapper utilities (`Noc`, `CB` alias, `set_read_state`, `read_with_state`) | ✓ excellent |
| `pack_untilize.cpp` | `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` | Shared kernel library | Device 2.0 `DataflowBuffer` wrappers | ✓ excellent |
| `pack_untilize.cpp` | `api/compute/pack_untilize.h` | LLK / HAL | N/A (LLK) | ✓ no concern |
| `pack_untilize.cpp` | `api/compute/cb_api.h` | LLK / HAL | N/A (LLK) | ✓ no concern |

**Per-call detail:** Omitted — all donor rolls are ✓ clean.

**Borrowed kernel files (file-path kernel instantiation):**

The factory instantiates kernels by file path:
- `ttnn/cpp/ttnn/operations/sliding_window/halo/device/kernels/dataflow/halo_gather.cpp` — owned by this op.
- `ttnn/cpp/ttnn/operations/sliding_window/halo/device/kernels/compute/pack_untilize.cpp` — owned by this op.

No kernel files are borrowed from other op families' directories. (The shared lib `untilize_helpers.hpp` is `#include`d, not instantiated as a kernel `.cpp` file by path.) No port-together set forced by file-path borrowing.

### Relaxation candidates

No custom `compute_program_hash` to mine.

### TTNN factory analysis

Full six-question answers:

1. **Op-owned tensors?** YES. `UntilizeWithHaloProgramFactory::create_workload_descriptor` allocates four device `Tensor` objects during factory invocation:
   - `pad_config_device_tensor0` — via `sliding_window::move_config_tensor_to_device(...)` (`untilize_with_halo_program_factory.cpp:436–440`)
   - `pad_config_device_tensor1` — ibid (`untilize_with_halo_program_factory.cpp:442–446`)
   - `gather_config_device_tensor0` — ibid (`untilize_with_halo_program_factory.cpp:448–452`)
   - `gather_config_device_tensor1` — ibid (`untilize_with_halo_program_factory.cpp:454–458`)
   These are halo sliding-window kernel config tensors (not in `tensor_args` / `tensor_return_value`). Their `Tensor` owners are parked on `workload_descriptor.buffers` (lines 473–487) so their device allocations outlive the cached workload. Their `Buffer*` pointers are passed to `add_cb` and to the factory's compile-time arg construction.

2. **MeshWorkload concept needed?** NO — op-owned-tensor artifact only. The factory uses `create_workload_descriptor` (returning `WorkloadDescriptor`) because the TTNN descriptor infra requires that path when op-owned tensors must outlive the cached programs. The op's algorithm is entirely single-device; there is no cross-device coordination or multi-program logic. The Q1 op-owned-tensor finding is the cause.

3. **Pybind `create_descriptor`?** NO. No `nb::class_<...ProgramFactory>(...).def_static("create_descriptor", ...)` or similar in any `*_nanobind.cpp` file for this op family.

4. **Other migration-risky pybind?** NO. `sliding_window_nanobind.cpp` binds only `ParallelConfig` and `Op2DSliceConfig` — plain config structs. No `DeviceOperation` methods, no factory parameter classes, no ProgramDescriptor introspection.

5. **Custom hash?** NO. No `compute_program_hash` definition or override in `halo_device_operation.hpp` or `halo_device_operation.cpp`.

6. **Custom override-runtime-args?** NO. No `override_runtime_arguments` definition anywhere in the op.

## Misc anomalies *(team-only, non-gating)*

- **Dead `remote_read` code path.** `HaloParams::remote_read` (types file line 17), `operation_attributes.remote_read` (factory, lines 97, 398), and the CTA at factory line 287 all thread `remote_read` through to `halo_gather.cpp`. However, the kernel has `static_assert(!remote_read, "Remote read is not supported in this kernel")` at line 277. The parameter exists in the API and in `HaloParams`, and the factory obediently serializes it as a CTA, but the kernel statically rejects it. This is harmless at runtime (callers pass `false`), but the op's public API (`halo()`) silently accepts `remote_read=true` arguments that would trigger the static_assert only at kernel compile time. The parameter and its threading could be removed. Routes to the op owner — not part of the port.

- **Double `#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"` in `pack_untilize.cpp`.** Lines 7 and 12 both include the same file. The `#pragma once` guard makes this harmless, but it is redundant. Routes to the op owner.

## Questions for the user

*(none)*

## Recipe notes

1. **Fake CB classification for `padding_config_cb` / `gather_config_cb` when `!config_tensors_in_dram`**: These CBs have `.buffer` set to the config tensors' buffers (borrowed-memory CB), but the kernel accesses them purely via `get_read_ptr()` — no FIFO wait/pop. Per the recipe's litmus test, they are fake CBs. However, unlike the typical fake-CB shape, these buffers are *op-owned tensors* (not declared TensorParameters). The fake-CB workaround in the recipe appears designed for bindings where the backing buffer *is* a declared TensorParameter. An op-owned-tensor–backed fake CB is a slightly different shape (the Metal 2.0 side must still provide the L1 address for the kernel to read; it just doesn't go through the TensorParameter channel). The recipe does not explicitly address this intersection. Recorded here for the recipe maintainer.

2. **`buffer->address()` baked into CTAs for op-owned tensors (the `config_tensors_in_dram` path):** The recipe's CTA-baked-address detection rule (recipe §TensorAccessor handling) specifically targets `buffer->address()` flowing into compile-time args and classifies these as Case 1. Here the buffers are op-owned config tensors, not declared input/output TensorParameters. The "re-express via TensorParameter" resolution is less direct for op-owned tensors, since op-owned tensors will need to be handled differently in the Metal 2.0 factory concept. Recorded as a nuance for the porter and for the downstream factory-concept selection (see `port_op_to_metal2_ttnn_factory.md`).
