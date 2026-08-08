# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/reduction/sampling`

Device operations and factories in this directory:

- **`SamplingDeviceOperation`**
  - `SamplingProgramFactory` (`device/sampling_program_factory.cpp`)

Kernels used:

- `device/kernels/dataflow/reader_values_indices_tensor.cpp` (reader, owned)
- `device/kernels/dataflow/writer_interleaved.cpp` (writer, owned)
- `device/kernels/compute/sampling.cpp` (compute, owned)

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `port_op_to_metal2_audit.md`.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/reduction/sampling` |
| **Overall** | RED |
| **DOps / Factories** | `SamplingDeviceOperation` → `SamplingProgramFactory` |
| *Prereqs* — ProgramDescriptor | Yes |
| *Prereqs* — Device 2.0 (every kernel used) | No |
| *Prereqs* — Cross-op escapes | issue (`generate_bcast_scalar.hpp` fully Device 1.0; `sdpa_decode/dataflow_common.hpp` isolated holdovers) |
| *Feature Support* — overall | GREEN (no UNSUPPORTED features) |
| *Feature Support* — Variadic-CTA | N/A |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — MeshWorkload needed | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Other risky pybind | None |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — Custom override-RTA | No |
| *TTNN Readiness* — Fake CBs (address-only) | present: CBIndex::c_14 (k), CBIndex::c_15 (p), CBIndex::c_13 (output) in `writer_interleaved.cpp` (workaround) |

**Fake CBs** = CBs used purely as an address source. The k, p, and output CBs have no `wait_front`/`pop_front` consumer FIFO path; they are used as raw L1 scratch via `CoreLocalMem` pointer reads/writes. The temp CB (c_16) is partially in this category but has its `reserve_back`/`push_back` commented out. The port resolves these with the sanctioned fake-CB workaround; they do **not** gate.

## Result

**RED → blocked on Device 2.0 gate**, routed to the **Device 2.0 team**.

Primary blocker: `ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp` — included by `writer_interleaved.cpp` — is entirely Device 1.0 (`cb_reserve_back`, `get_write_ptr`, `cb_push_back` free functions; no Device 2.0 wrapper objects at all). The `sdpa_decode/dataflow_common.hpp` donor adds isolated `get_write_ptr(cb_id)` holdovers in its `fill_tile` / `fill_tile_partial` helpers.

The ProgramDescriptor prerequisite is met and all feature-compatibility checks pass; this op can proceed directly to Metal 2.0 porting once the two donor Device 2.0 gaps are resolved.

## Gate detail

- **ProgramDescriptor:** GREEN — the factory populates a `ProgramDescriptor` and uses `CBDescriptor`, `KernelDescriptor`, `ComputeConfigDescriptor`, `ReaderConfigDescriptor`, `WriterConfigDescriptor`. No `host_api.hpp` imperative builder calls. (`sampling_program_factory.cpp:93–433`)

- **Device 2.0 (every kernel used):**

  **RED (GATE).** Two donor files pulled in by `writer_interleaved.cpp` are not Device 2.0 compliant:

  1. **`ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp`** — fully Device 1.0. All three functions (`generate_bcast_col_scalar`, `generate_bcast_row_scalar`, `generate_bcast_unary_scalar`) use only `cb_reserve_back(cb_id, n)`, `get_write_ptr(cb_id)`, and `cb_push_back(cb_id, n)` free functions. No `CircularBuffer` wrapper objects are in scope anywhere in this file. The writer kernel calls `generate_bcast_unary_scalar(cb_id_temp, temp_packed)` at line 124.

     | File | Line | Call | Wrapper in scope |
     |---|---|---|---|
     | `ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp` | 14 | `cb_reserve_back(cb_id, 1)` | No |
     | `ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp` | 15 | `get_write_ptr(cb_id)` | No |
     | `ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp` | 22 | `cb_push_back(cb_id, 1)` | No |
     | `ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp` | 29 | `cb_reserve_back(cb_id, 1)` | No |
     | `ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp` | 30 | `get_write_ptr(cb_id)` | No |
     | `ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp` | 37 | `cb_push_back(cb_id, 1)` | No |
     | `ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp` | 44 | `cb_reserve_back(cb_id, 1)` | No |
     | `ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp` | 46 | `get_write_ptr(cb_id)` | No |
     | `ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp` | 48 | `cb_push_back(cb_id, 1)` | No |

     Owner: shared-kernel pool (`ttnn/cpp/ttnn/kernel/` — class 3). The Device 2.0 team should migrate this file. Multiple ops likely share it.

  2. **`ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/dataflow_common.hpp`** — mostly Device 2.0, but `fill_tile` and `fill_tile_partial` use `get_write_ptr(cb_id)` as a free function without a `CircularBuffer` wrapper in scope in those function bodies (the non-zero `val` branch of `fill_tile`, lines 37, 58, 60; the same in `fill_tile_partial` and its derived helper). The writer kernel calls `generate_mask<cb_id_mask, one>(...)` which uses these helpers.

     | File | Line | Call | Wrapper in scope |
     |---|---|---|---|
     | `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/dataflow_common.hpp` | 37 | `get_write_ptr(cb_id)` (in `fill_tile`, val≠0 branch) | No (no CB wrapper in this branch) |
     | `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/dataflow_common.hpp` | 58 | `get_write_ptr(cb_id)` (in `fill_tile_partial`) | No |
     | `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/dataflow_common.hpp` | 60 | `get_write_ptr(cb_id)` (in `fill_tile_partial`) | No |
     | `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/dataflow_common.hpp` | 120 | `get_write_ptr(cb_id)` (in `fill_tile_partial_sliding_window`) | No |
     | `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/dataflow_common.hpp` | 122 | `get_write_ptr(cb_id)` (in `fill_tile_partial_sliding_window`) | No |

     Owner: `transformer/sdpa_decode` family (cross-family donor). The Device 2.0 team should coordinate with the sdpa_decode owners. These are isolated holdovers in a file that is otherwise mostly Device 2.0.

  Own kernels are clean:
  - `reader_values_indices_tensor.cpp`: Device 2.0 compliant. Uses `CircularBuffer`, `Noc`, `TensorAccessor`, `CoreLocalMem`. The call `get_tile_size(cb_id)` at line 78 is a sanctioned Device 2.0 free function.
  - `sampling.cpp` (compute): Device 2.0 compliant. Uses `CircularBuffer` wrappers throughout; does not include `generate_bcast_scalar.hpp`.
  - `writer_interleaved.cpp` (own code): Device 2.0 compliant for lines it owns; the violations come exclusively from the two donor includes.

- **Feature compatibility:**

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` usage; no `.global_circular_buffer` field on any `CBDescriptor` |
  | Dynamic CircularBuffer (borrowed memory) | N/A | No `.buffer` field set on any `CBDescriptor`; all CBs are static |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `.address_offset` field set on any `CBDescriptor` |
  | Aliased Circular Buffers | N/A | All `format_descriptors` initializers are single-element; no multi-index CBs |
  | GlobalSemaphore | N/A | No semaphores of any kind in this op |
  | Non-zero semaphore initial value | N/A | No semaphores |
  | Dynamic TensorAccessor (`ArgConfig::Runtime*`) | N/A | All `TensorAccessorArgs` calls use the single-argument form; no `ArgConfig::Runtime*` tokens |
  | `UpdateCircularBuffer*` | N/A | No calls to any `UpdateCircularBuffer*` variant |
  | Variable-count compile-time arguments (CTA varargs) | N/A | `tensor_args_t` is a fixed-field struct (not `std::vector<Tensor>`); no `get_compile_time_arg_val(i)` loops with runtime-varying `i` |

## Port-work summary *(mirrors the brief — no brief issued; see Result)*

*(For planning reference, once the Device 2.0 prereqs clear:)*

- **Tensor bindings** (per binding):
  - `input_values` — **Case 1** → re-express via `TensorParameter`/`TensorBinding` (reader kernel receives address as `get_arg_val<uint32_t>(0)` after auto-registration via `emplace_runtime_args({input_values_tensor, ...})`)
  - `input_indices` — **Case 1** → re-express via `TensorParameter`/`TensorBinding` (reader kernel `get_arg_val<uint32_t>(1)`)
  - `output` (output_mesh) — **Case 1** → re-express (writer kernel `get_arg_val<uint32_t>(0)`)
  - `temp` — **Case 1** → re-express (writer kernel `get_arg_val<uint32_t>(1)`)
  - `k` — **Case 1** → re-express (writer kernel `get_arg_val<uint32_t>(2)`)
  - `p` — **Case 1** → re-express (writer kernel `get_arg_val<uint32_t>(3)`)
- **Custom hash:** none — no action needed.

All six bindings use the `Buffer*`-binding form (`emplace_runtime_args` receives `Tensor` objects; the framework auto-registers `BufferBinding`s). They are correct-on-cache-hit today; the port re-expresses them as typed `TensorParameter`/`TensorBinding` channels. No exotic access patterns observed; no Case 2 bindings.

## Heads-ups *(mirrors the brief — no brief issued; for planning reference)*

- **Fake CBs (address-only):**
  - `cb_k` (CBIndex::c_14) at `writer_interleaved.cpp:81–99` — `reserve_back/push_back` present as a "producer" but the CB is consumed only via `CoreLocalMem<volatile uint32_t> k_ptr(cb_id_k_ptr)` raw pointer arithmetic. No `wait_front`/`pop_front` consumer. Fake CB.
  - `cb_p` (CBIndex::c_15) at `writer_interleaved.cpp:101–110` — same pattern.
  - `cb_temp` (CBIndex::c_16) at `writer_interleaved.cpp:112–125` — `reserve_back` and `push_back` are explicitly commented out; raw pointer read only. Fake CB.
  - `cb_out` (CBIndex::c_13) at `writer_interleaved.cpp:144–145` — writer writes `index_out[core_id]` via `CoreLocalMem<volatile uint32_t> index_out(out_addr)` raw pointer, with no `reserve_back`/`push_back` pair. Used as a scratch buffer; no FIFO semantics.
  
  The port resolves these with the sanctioned fake-CB workaround (see the porting recipe). They do not gate the port.

- **Cross-op / shared kernels:**
  - `writer_interleaved.cpp` includes `ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp` (Device 1.0; shared-kernel pool, broadly used — a Device 2.0 migration of this file will affect all consumers).
  - `writer_interleaved.cpp` includes `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/dataflow_common.hpp` (cross-family donor: `transformer/sdpa_decode`).
  - `writer_interleaved.cpp` includes `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp` (shared-kernel-lib pool; Device 2.0 compliant).
  - `sampling.cpp` includes `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp` (shared-kernel-lib pool; Device 2.0 compliant).
  
  The `generate_bcast_scalar.hpp` migration is required before this op's port can proceed. That migration is cross-op; coordinate with the shared-kernel team. The `sdpa_decode/dataflow_common.hpp` holdovers need to be resolved on the sdpa_decode Device 2.0 track.

## Team-only

### TensorAccessor convertibility

All six bindings are Case 1. The access patterns are standard page-by-page reads (`noc.async_read` with `.page_id = tile_id_...`) and single-page writes (`noc.async_write` with `.page_id = 0`). No exotic sub-page arithmetic or custom NoC walks observed. No Case 2 bindings; no convertibility annotation needed.

### Out-of-directory coupling & donor shape

**Op-level roll-up:** ⭐ blocked — `generate_bcast_scalar.hpp` is the Device 2.0 GATE (⭐ Shape 4 analog: file is pre-Device-2.0). The `sdpa_decode/dataflow_common.hpp` has isolated holdovers (⚠). The `reduce_helpers_*` shared-lib files are clean (✓).

**Summary table:**

| Op kernel | Donor file | Bucket |
|---|---|---|
| `writer_interleaved.cpp` | `ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp` | ⭐ Device 1.0 — GATE |
| `writer_interleaved.cpp` | `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/dataflow_common.hpp` | ⚠ isolated holdovers (`get_write_ptr(cb_id)` free function) |
| `writer_interleaved.cpp` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp` | ✓ clean (shared-lib class) |
| `sampling.cpp` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp` | ✓ clean (shared-lib class) |
| `reader_values_indices_tensor.cpp` | (none outside tt_metal/api) | ✓ clean |

**Per-call detail:**

**`generate_bcast_scalar.hpp`** (shared-kernel pool, `ttnn/cpp/ttnn/kernel/dataflow/`):
- Function `generate_bcast_unary_scalar(uint32_t cb_id, uint32_t scalar)` (called by `writer_interleaved.cpp:124`): takes `uint32_t cb_id` — entirely Device 1.0 body. No wrapper objects. Replacement requires re-writing the function body using a `CircularBuffer` wrapper.
- Functions `generate_bcast_col_scalar`, `generate_bcast_row_scalar`: same Device 1.0 structure (not called by sampling directly, but in same file).

**`sdpa_decode/dataflow_common.hpp`** (cross-family donor: `transformer/sdpa_decode`):
- Functions called by sampling (transitively via `generate_mask`): `fill_tile<tile_bytes>(cb_id, tile_id, val)` and `fill_tile_partial<tile_bytes>(cb_id, tile_id, ...)` — take `uint32_t cb_id` parameters and call `get_write_ptr(cb_id)` free function without a `CircularBuffer` wrapper in the call-site scope (Device 1.0 holdover). The `val==0` branch of `fill_tile` is already Device 2.0 (`CircularBuffer cb(cb_id); noc.async_write_zeros(...)`); only the `val!=0` branch is a holdover.
- Fix: refactor `fill_tile` / `fill_tile_partial` to accept a `CircularBuffer&` or construct a `CircularBuffer` wrapper and call `cb.get_write_ptr()`. Isolated; 5 call sites.

**Borrowed kernel files (file-path kernel instantiation):**

All three kernel `.cpp` files (`reader_values_indices_tensor.cpp`, `writer_interleaved.cpp`, `sampling.cpp`) are owned by this op and appear unique to it. No cross-op file-path borrowing observed. The op does not instantiate kernels owned by other families.

The coupling is via `#include` of donor headers, not file-path instantiation.

### Relaxation candidates

No custom `compute_program_hash` present; nothing to mine.

### TTNN factory analysis

1. **Op-owned tensors?** No. `SamplingDeviceOperation::create_output_tensors` calls `create_device_tensor(...)` to create the declared output tensor — that is the op's output `Tensor`, not an op-owned intermediate. The factory itself allocates no additional device tensors.

2. **MeshWorkload needed?** No. No `create_mesh_workload` / `cached_mesh_workload_t` in any factory or device-operation code. No genuine multi-program or cross-device coordination. Single-program op.

3. **Pybind `create_descriptor`?** No. `sampling_nanobind.cpp` uses `ttnn::bind_function<"sampling">` to bind the user-facing op function. No `nb::class_<...ProgramFactory>(...).def_static("create_descriptor", ...)` or `.def` binding of factory innards.

4. **Other migration-risky pybind?** None. The nanobind surface is limited to the normal op-function binding. No `DeviceOperation`, `SamplingProgramFactory`, or `SamplingParams` class is pybind-exposed in a way that would interfere with migration.

5. **Custom hash?** No. `SamplingDeviceOperation` does not define `compute_program_hash`. The default reflection-based hash is used.

6. **Custom `override_runtime_arguments`?** No. `SamplingProgramFactory` does not define `override_runtime_arguments`.

## Misc anomalies *(team-only, non-gating)*

- `writer_interleaved.cpp:59` — `args_base + 8` is described in a comment as `out_stick_size (passed from factory, unused in kernel)`. The factory passes `aligned_out0_unit_size` at that CTA position (`sampling_program_factory.cpp:376–377`), but the kernel skips it with no `get_compile_time_arg_val(args_base + 8)` call. Dead CTA slot. Low-risk but could confuse a future reader.

- `writer_interleaved.cpp:113,118` — `cb_temp.reserve_back(1)` and `cb_temp.push_back(1)` are commented out. The CB is used as a raw buffer (read via `get_write_ptr`, filled by `noc.async_read`). The commented-out calls suggest the fake-CB usage was deliberate but left as documentation debt. Port should resolve via the fake-CB workaround; no action needed before then.

## Questions for the user

1. **`generate_bcast_scalar.hpp` Device 2.0 migration ownership:** This shared-kernel file (`ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp`) is fully Device 1.0. Does the Device 2.0 migration team have it on their radar, or should this audit trigger a work item? Knowing the ETA affects whether the sampling port unblocks soon or not.

2. **`sdpa_decode/dataflow_common.hpp` holdovers:** The `fill_tile`/`fill_tile_partial` functions use `get_write_ptr(cb_id)` free-function form in several branches. Are these being tracked in the sdpa_decode Device 2.0 cleanup, or should they be filed separately? Since sampling borrows this file, the holdovers block sampling's port until resolved.

## Recipe notes

None — recipe guidance was clear and all cases encountered were well-anticipated.
