# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/plusone`

- **`PlusOneDeviceOperation`**
  - `PlusOneProgramFactory` (`device/plusone_program_factory.cpp`)

The op has one device-operation and one program factory. A single kernel is owned by this directory:
`device/kernels/reader_plusone_interleaved.cpp`. No unreferenced kernel files present.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `port_op_to_metal2_audit.md`.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/plusone` |
| **Overall** | GREEN |
| **DOps / Factories** | `PlusOneDeviceOperation` → `PlusOneProgramFactory` |
| *Prereqs* — ProgramDescriptor | Yes |
| *Prereqs* — Device 2.0 (every kernel used) | Yes |
| *Prereqs* — Cross-op escapes | Ok |
| *Feature Support* — overall | GREEN |
| *Feature Support* — Variadic-CTA | Ok |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — MeshWorkload needed | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Other risky pybind | None |
| *TTNN Readiness* — Custom hash | Yes → delete (see Custom program hash) |
| *TTNN Readiness* — Custom override-RTA | No |
| *TTNN Readiness* — Fake CBs (address-only) | present: `(c_0, reader)` in both sharded and interleaved paths (workaround) |

**Fake CBs** = CBs used purely as an address source. `cb_id_in0` (`CBIndex::c_0`) has no producer–consumer FIFO pair in either the sharded or interleaved path — the kernel calls `cb_in0.get_write_ptr()` to extract a scratch/shard address, with no `reserve_back` / `push_back` / `wait_front` / `pop_front`. Not expressible as a Metal 2.0 DFB (spec validator requires ≥1 producer and ≥1 consumer); the port resolves this with the sanctioned fake-CB workaround. FYI-P heads-up, not a gate.

## Result

**GREEN → brief issued.** All gates cleared. One custom `compute_program_hash` to delete (PORT WORK). One `TensorParameter` binding — Case 1, re-express via `TensorParameter`/`TensorBinding`. Two FYI-P heads-ups: fake CB on `c_0` (both paths), Dynamic CircularBuffer LANDED feature (sharded path uses `CBDescriptor::buffer`). No UNSUPPORTED features found. No out-of-directory coupling beyond `tt_metal/` LLK/HAL headers (no concern).

## Gate detail

- **ProgramDescriptor:** GREEN — `PlusOneProgramFactory::create_descriptor` returns a `ProgramDescriptor` populated via `CBDescriptor`, `KernelDescriptor`, and `ProgramDescriptor::kernels` / `desc.cbs`. The op uses `program_descriptors.hpp`; no imperative `host_api.hpp` `CreateProgram` / `CreateKernel` / `CreateCircularBuffer` calls appear. `device/plusone_program_factory.cpp` line 22.

- **Device 2.0 (every kernel used):** GREEN — `device/kernels/reader_plusone_interleaved.cpp` exclusively uses Device 2.0 headers and wrappers:
  - `#include "api/dataflow/noc.h"` → `Noc noc;` (line 13)
  - `#include "api/dataflow/circular_buffer.h"` → `CircularBuffer cb_in0(cb_id_in0);` (line 28)
  - `#include "api/core_local_mem.h"` → `CoreLocalMem<uint32_t>(cb_addr)` (lines 36, 51)
  - `#include "api/tensor/noc_traits.h"` → `TensorAccessor`
  - `noc.async_read(...)` / `noc.async_write(...)` via `Noc` member form
  - No `InterleavedAddrGen`, `ShardedAddrGen`, `noc_async_read` free functions, raw CB-index free-function calls without a wrapper in scope, or other Device 1.0 idioms present.

- **Feature compatibility:** Every Appendix A entry scanned. No UNSUPPORTED signals fired.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type, `CreateGlobalCircularBuffer`, or `global_circular_buffer` CBDescriptor field |
  | Dynamic CircularBuffer (borrowed memory) | GREEN | `CBDescriptor::buffer = input.is_sharded() ? src_buffer : nullptr` at `plusone_program_factory.cpp:65`. Sharded path sets `.buffer = src_buffer` (non-null). Port uses `borrowed_from`. However see Fake CBs — no genuine FIFO; resolved via fake-CB workaround, not `borrowed_from`. |
  | CBDescriptor `address_offset` (non-zero) | N/A | `.address_offset` not set on any `CBDescriptor`. The `.buffer` field fires the Dynamic CircularBuffer signal; `.address_offset` is independent and absent. |
  | Aliased Circular Buffers | N/A | `CBDescriptor::format_descriptors` has exactly one `CBFormatDescriptor` element (single-element initializer at `plusone_program_factory.cpp:60–64`). |
  | GlobalSemaphore | N/A | No semaphores declared anywhere in the op. |
  | Non-zero semaphore initial value | N/A | No semaphores in this op. |
  | Dynamic TensorAccessor (`ArgConfig::Runtime*`) | N/A | `TensorAccessorArgs(*src_buffer).append_to(...)` at `plusone_program_factory.cpp:70` uses the single-argument form — no `ArgConfig::Runtime*` qualifier. |
  | `UpdateCircularBuffer*` | N/A | No calls to `UpdateCircularBufferTotalSize`, `UpdateCircularBufferPageSize`, or `UpdateDynamicCircularBufferAddressAndTotalSize`. |
  | Variable-count compile-time arguments (CTA varargs) | N/A | `tensor_args_t = Tensor` (single fixed tensor, not `std::vector<Tensor>`). Six fixed CTAs at indices 0–5, plus a fixed-size `TensorAccessorArgs<6>()` block. No runtime-varying CTA loop. |

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings:**
  - `input` / `src_buffer` — **Case 1** (re-express via `TensorParameter` / `TensorBinding`). Host: `Buffer*`-binding arg at `plusone_program_factory.cpp:85` (`emplace_runtime_args(core, {src_buffer})`); CTA-packed `TensorAccessorArgs` at `plusone_program_factory.cpp:70`. Kernel: `get_arg_val<uint32_t>(0)` → `src_addr` → `TensorAccessor(s0_args, src_addr)` at `reader_plusone_interleaved.cpp:16,26`. Note: in the sharded path `s0` is constructed but `noc.async_read`/`noc.async_write` are skipped (`if (src0_is_dram)` guard at lines 35/51). The port still re-expresses via `TensorParameter`; the kernel's `TensorAccessor` construction remains correct regardless of path.
- **Custom hash:** delete custom `compute_program_hash` → default (sanctioned exception). See `plusone_device_operation.cpp:26–39` and Custom program hash section below.

## Heads-ups  *(mirrors the brief)*

- **Fake CBs (address-only):** `cb_id_in0` (`CBIndex::c_0`, declared at `plusone_program_factory.cpp:45`), both paths:
  - **Interleaved path** (`CBDescriptor::buffer = nullptr`): L1 scratch. Kernel accesses it via `cb_in0.get_write_ptr()` only (`reader_plusone_interleaved.cpp:31`). No `reserve_back` / `push_back` / `wait_front` / `pop_front` anywhere. CB is used purely as a scratch address.
  - **Sharded path** (`CBDescriptor::buffer = src_buffer` when `input.is_sharded()`): borrowed-memory CB at the shard's L1 address. Kernel again accesses it via `get_write_ptr()` only. Same absence of FIFO operations. No genuine producer+consumer.
  - Both are the **fake-CB pattern**: the CB has no producer–consumer FIFO, so it cannot be expressed as a Metal 2.0 DFB. The port resolves this with the sanctioned fake-CB workaround (see `port_op_to_metal2_recipe.md`). Edge: `(c_0, reader)`.
- **Dynamic CircularBuffer LANDED:** sharded path sets `CBDescriptor::buffer = src_buffer` at `plusone_program_factory.cpp:65`. Although the Dynamic CircularBuffer rule fires on this (`.buffer` non-null), the causal-link gate fails — no producer+consumer FIFO — so this is classified as a fake CB, not a clean borrowed-memory DFB. The port does **not** use `DataflowBufferSpec::borrowed_from` for this CB; it uses the fake-CB workaround instead.
- **TTNN factory analysis (porter-relevant):** pybind `create_descriptor`: none. Other risky pybind: none. Custom `override_runtime_arguments`: none.

## Custom program hash

**File:line:** `device/plusone_device_operation.cpp:26–39`

The custom hash keys on `args` (includes `sub_core_grids`, `skip_negative_entries`), `input_tensor.dtype()`, `input_tensor.memory_config()`, and `input_shape` — but does **not** include `TensorSpec`. Per the audit recipe, the port **deletes** this and reverts to the default TTNN hash (a sanctioned device-op-class edit). The default hash is correct-by-construction and keys on what actually determines the program.

**Relaxation candidates (FYI-U, FALLIBLE):** The custom hash keys explicitly on `memory_config()` and `input_shape`. This hints that `memory_config` (sharding vs. interleaved, DRAM vs. L1) and shape are determining factors for the program. The `dtype` determines `input_cb_data_format` and `input_unit_size`. These are candidates for future `TensorParameter` relaxation analysis, but this hash is written against the old framework and may omit dependencies — treat as candidate-to-verify, not conclusion.

## Team-only

### TensorAccessor convertibility

- **`input` binding (Case 1):** The DRAM path uses `noc.async_read(s0, ..., {.page_id = h}, {})` and `noc.async_write(..., s0, ..., {.page_id = h})` — standard page-by-page iteration that `TensorAccessor` natively supports. Not exotic; straightforwardly Case 1.
- The sharded path constructs `s0` but does not use it for NoC access (all writes are in-place via the CB pointer). After re-expressing as `TensorParameter` the kernel still constructs `TensorAccessor(ta::input)` — the unused construction is a no-op and harmless. No Case 2 needed.

### Out-of-directory coupling & donor shape

**Op-level roll-up:** ✓ clean — all `#include`s in `reader_plusone_interleaved.cpp` resolve to `tt_metal/hw/inc/` or standard library headers (class 1: LLK/HAL/firmware). No function-call or file-path coupling to any other TTNN op family or shared-kernel pool.

**Summary table:**

| Op kernel | `#include` target | Donor class | Shape | Status |
|---|---|---|---|---|
| `reader_plusone_interleaved.cpp` | `api/dataflow/dataflow_api.h` | tt_metal/hw — LLK/HAL | n/a | ✓ no concern |
| `reader_plusone_interleaved.cpp` | `api/dataflow/noc.h` | tt_metal/hw — LLK/HAL | n/a | ✓ no concern |
| `reader_plusone_interleaved.cpp` | `api/dataflow/circular_buffer.h` | tt_metal/hw — LLK/HAL | n/a | ✓ no concern |
| `reader_plusone_interleaved.cpp` | `api/core_local_mem.h` | tt_metal/hw — LLK/HAL | n/a | ✓ no concern |
| `reader_plusone_interleaved.cpp` | `api/tensor/noc_traits.h` | tt_metal/hw — LLK/HAL | n/a | ✓ no concern |
| `reader_plusone_interleaved.cpp` | `<stdint.h>` / `<limits.h>` | standard library | n/a | ✓ no concern |

**Per-call detail:** omitted — all donors are class 1 (LLK/HAL/firmware), no ⚠/✗/⭐ entries.

**Borrowed kernel files (file-path kernel instantiation):** None — the only kernel referenced is `ttnn/cpp/ttnn/operations/experimental/plusone/device/kernels/reader_plusone_interleaved.cpp`, which is owned by this op directory. No file-path borrow from a shared pool or other op family.

### TTNN factory analysis

1. **Op-owned tensors?** No. `PlusOneProgramFactory::create_descriptor` takes `input` and `tensor_return_value` as arguments only. `create_output_tensors` at `plusone_device_operation.cpp:41–44` returns `input_tensor` directly (no separate allocation). No `create_device_tensor` / `allocate_tensor_on_device` calls anywhere. This is an in-place op — the output *is* the input.

2. **MeshWorkload concept needed?** No. No `create_mesh_workload`, `create_workload_descriptor`, or `cached_mesh_workload_t` present. No op-owned tensors (Q1 is No), so the MeshWorkload-artifact false-positive does not apply. Single-program, single-core (or sub-core-grid), no cross-device coordination.

3. **Pybind `create_descriptor`?** No. `plusone_nanobind.cpp` uses `ttnn::bind_function<"plus_one">` to bind the user-facing function. No `nb::class_<PlusOneProgramFactory>` or `.def_static("create_descriptor", ...)`. The binding is the normal op-function surface.

4. **Other migration-risky pybind?** None. `plusone_nanobind.cpp` wraps only the op function (`plus_one`). No `DeviceOperation` methods, no factory/param classes, no introspection entry points.

5. **Custom hash?** Yes — `PlusOneDeviceOperation::compute_program_hash` at `plusone_device_operation.cpp:26`. PORT WORK: delete it. See Custom program hash section.

6. **Custom override-runtime-args?** No. No `override_runtime_arguments` definition in `plusone_program_factory.cpp` or `.hpp`.

### Relaxation candidates (mined from custom hash — FALLIBLE)

The custom hash keys on `dtype`, `memory_config`, and `padded_shape`. This suggests:
- `dtype` is a potential per-`TensorParameter` constraint — the shape affects data format and `element_size`; likely must match.
- `memory_config` determines sharding vs. interleaved and DRAM vs. L1, which drive the compile-time `src0_is_dram` branch. Likely must match (different code paths).
- `padded_shape` affects `W`, `H`, `aligned_input_page_size`, and core assignment. Likely must match.

No obvious relaxation candidates — the hash keys appear structurally required. These are candidates to verify after deletion; the default hash is strict.

## Misc anomalies  *(omit if none; team-only, non-gating)*

- **Unused `TensorAccessor` in sharded path** — `reader_plusone_interleaved.cpp:25–26`: `s0_args` and `s0` are constructed unconditionally, but `s0` is only used inside `if (src0_is_dram)` blocks. In the sharded (`src0_is_dram = false`) path, `s0` is dead. Similarly, `src_addr` from `get_arg_val<uint32_t>(0)` is dead in the sharded path. Not a bug (the kernel operates correctly), but a minor latent overhead. Routes to op owner; the port does not act on it.

- **Comment mismatch** — `plusone_program_factory.cpp:55–56` says "For the interleaved path the CB is plain L1 scratch and the buffer address is passed through the reader runtime arg instead." The runtime arg at line 85 actually passes a `Buffer*` pointer (not `->address()`), which the framework auto-registers as a `BufferBinding`. The comment slightly mischaracterizes the mechanism. Non-critical; routes to op owner.

## Questions for the user  *(omit if none)*

None.

## Recipe notes  *(omit if none)*

- **Dynamic CircularBuffer + Fake CB overlap**: The op's sharded path sets `CBDescriptor::buffer = src_buffer` (non-null), which fires the Dynamic CircularBuffer recognition signal per Appendix A. However, the kernel has no producer+consumer FIFO (the "fake CB" test from the TensorAccessor-handling section). The recipe correctly handles this via the causal-link gate ("But 'clean' requires a real producer *and* consumer") and the fake-CB litmus. However, the resulting classification is slightly non-obvious: the Dynamic CircularBuffer row in the Feature table is marked GREEN (the feature *is* present in the sharded path), but the resolution is the fake-CB workaround (not `borrowed_from`). The recipe's intent is clear on re-reading, but the interaction between Appendix A's Dynamic CircularBuffer entry and the fake-CB classification in TensorAccessor handling required careful cross-referencing. A cross-reference note in one of the two sections pointing to the other would reduce re-reads for future auditors.
