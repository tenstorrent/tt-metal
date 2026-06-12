# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/data_movement/slice`

**Identifying section.**

One `DeviceOperation` with five `ProgramFactory` variants sharing the directory:

- **`SliceDeviceOperation`** (`slice_device_operation.cpp` / `.hpp`, `slice_device_operation_types.hpp`)
  - `SliceRmProgramFactory` (`slice_program_factory_rm.cpp`)
  - `SliceRmShardedProgramFactory` (`slice_program_factory_rm_sharded.cpp`)
  - `SliceRmStrideProgramFactory` (`slice_program_factory_rm_stride.cpp`)
  - `SliceTileProgramFactory` (`slice_program_factory_tile.cpp`)
  - `SliceTileTensorArgsProgramFactory` (`slice_program_factory_tile_tensor_args.cpp`)

Unreferenced kernel files in the directory (out of scope): `strided_slice_reader_rm_interleaved_nd.cpp`, `strided_slice_writer_rm_interleaved.cpp`.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `port_op_to_metal2_audit.md`.

---

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/slice` |
| **Overall** | YELLOW |
| **DOps / Factories** | `SliceDeviceOperation` → `SliceRmProgramFactory`, `SliceRmShardedProgramFactory`, `SliceRmStrideProgramFactory`, `SliceTileProgramFactory`, `SliceTileTensorArgsProgramFactory` |
| *Prereqs* — ProgramDescriptor | Yes |
| *Prereqs* — Device 2.0 (every kernel used) | Yes-with-holdovers (YELLOW — see Gate detail) |
| *Prereqs* — Cross-op escapes | Ok (eltwise writer is Device 2.0 structurally; `common.hpp` holdover is also in-family) |
| *Feature Support* — overall | GREEN |
| *Feature Support* — Variadic-CTA | N/A |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — MeshWorkload needed | No |
| *TTNN Readiness* — Pybind `create_descriptor` | Yes: `slice_nanobind.cpp:161` — `SliceTileProgramFactory::create_descriptor` exposed via `bind_slice_descriptor` |
| *TTNN Readiness* — Other risky pybind | `SliceDeviceOperation::create_output_tensors` and `compute_output_specs` exposed at `slice_nanobind.cpp:149-158`; `SliceParams` and `SliceInputs` structs exposed |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — Custom override-RTA | No |
| *TTNN Readiness* — Fake CBs (address-only) | None |

---

## Result

**YELLOW** — all gates pass, but there are isolated Device 2.0 holdovers that must be cleaned before the port begins (on the Device 2.0 track; see Gate detail for the exact lines). The brief is issued. The port **cannot start** until those holdovers are fixed; once they are, proceed with the brief as-is — no re-audit needed.

Primary holdovers: `tt::data_movement::common::tt_memmove` calls raw `noc_async_read` / `noc_async_write` free functions inside the `common.hpp` helper, which is included by `slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp`.

---

## Gate detail

### ProgramDescriptor

**GREEN.** All five factories populate `tt::tt_metal::ProgramDescriptor` using `KernelDescriptor`, `CBDescriptor`, `CBFormatDescriptor`, `SemaphoreDescriptor` API objects. No `host_api.hpp` imperative calls (`CreateProgram` / `CreateKernel` / `CreateCircularBuffer` / `SetRuntimeArgs`) appear in any factory `.cpp`. Confirmed by inspection of all five factory files.

### Device 2.0 (every kernel used)

**YELLOW — isolated holdovers in one helper header; otherwise structurally Device 2.0.**

All in-scope kernel files directly include Device 2.0 headers (`api/dataflow/noc.h`, `api/dataflow/circular_buffer.h`, `api/tensor/noc_traits.h`) and use the Device 2.0 wrapper objects (`Noc noc;`, `CircularBuffer cb(...)`) as their primary access pattern.

The single holdover location is inside `ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp`, included by one kernel:

| File | Line | Call | Wrapper in scope |
|---|---|---|---|
| `ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp` | ~29–35 | `noc_async_read_one_packet(src_noc_addr, dst_l1_addr, bytes)` / `noc_async_read<N>(src_noc_addr, dst_l1_addr, bytes)` inside `enhanced_noc_async_read` | `Noc noc` in caller; but the call is inside the helper, not the Device 2.0 `noc.async_read()` path |
| `ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp` | ~43–47 | `noc_async_write_one_packet(...)` / `noc_async_write<N>(...)` inside `enhanced_noc_async_write` | same |

Called from `slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp` as `tt::data_movement::common::tt_memmove<false, false, false, 0>(src_buffer_l1_addr, src_buffer_l1_addr + misalignment, unpadded_stick_size)`. This is a local L1-to-L1 copy helper (loopback via `get_noc_addr(l1_addr)`), not an external tensor read. Functionally isolated and restricted to the misalignment-correction path in `SliceRmProgramFactory`'s reader kernel.

**Contextual note:** `common.hpp` is owned by the `data_movement/common` family, not the `slice` op directly. The Device 2.0 cleanup task belongs to the common utility team, not the slice porter. However, the slice port cannot start until it is resolved, per the Device 2.0 gate.

**Eltwise donor kernel (`writer_unary_interleaved_start_id.cpp`):** Uses `get_local_cb_interface(cb_id_out).fifo_page_size` at line ~19 — this is a **sanctioned** Device 2.0 free-function form per the migration guide (no holdover flag).

### Feature compatibility

| Feature | Status | Notes |
|---|---|---|
| GlobalCircularBuffer | N/A | No `GlobalCircularBuffer`, `experimental::CreateGlobalCircularBuffer`, or `.global_circular_buffer` CBDescriptor field in any factory or kernel |
| Dynamic CircularBuffer (borrowed memory) | GREEN | Used in `SliceRmShardedProgramFactory` — `CBDescriptor::buffer = input.buffer()` (src0, `slice_program_factory_rm_sharded.cpp:265`) and `CBDescriptor::buffer = output.buffer()` (c_16, line 278). Port uses `DataflowBufferSpec::borrowed_from`. |
| CBDescriptor `address_offset` (non-zero) | N/A | No `.address_offset` field set anywhere in any factory |
| Aliased Circular Buffers | N/A | All `CBDescriptor::format_descriptors` initializers use single `CBFormatDescriptor` elements throughout all five factories |
| GlobalSemaphore | N/A | No semaphores declared in any factory |
| Non-zero semaphore initial value | N/A | No semaphores declared |
| Dynamic TensorAccessor (`ArgConfig::Runtime*`) | N/A | No `ArgConfig::Runtime*` tokens anywhere in host code |
| `UpdateCircularBuffer*` | N/A | No `UpdateCircularBufferTotalSize`, `UpdateCircularBufferPageSize`, or `UpdateDynamicCircularBufferAddressAndTotalSize` calls. (Note: `slice_program_factory_rm.cpp` is listed as an "example in the wild" for this feature in Appendix A of the audit recipe — this appears stale; those calls no longer exist in the current code.) |
| Variable-count compile-time arguments (CTA varargs) | N/A | `SliceInputs::tensor_args_t` uses a fixed-shape struct, not `std::vector<Tensor>`. No kernel loops over `get_compile_time_arg_val(i)` with runtime-varying index. |

**Feature subject overall: GREEN — no gate fired.**

---

## Port-work summary *(mirrors the brief)*

**Tensor bindings** (per binding, per factory):

- `SliceRmProgramFactory`
  - `input` — **Case 1** (re-express): `input_tensor.buffer()->address()` baked into `common_reader_kernel_args[0]` at `slice_program_factory_rm.cpp:72`; kernel consumes as `src_addr` arg 0 and passes to `TensorAccessor(src_args, src_addr, padded_stick_size)`.
  - `output` — **Case 1** (re-express): `output_buffer->address()` in writer RTAs at `slice_program_factory_rm.cpp:131`; kernel consumes as `dst_addr` arg 0 and passes to `TensorAccessor(dst_args, dst_addr, stick_size)`.

- `SliceRmShardedProgramFactory`
  - `input` — **clean** (borrowed-memory DFB via `CBDescriptor::buffer = input.buffer()` at `slice_program_factory_rm_sharded.cpp:265`).
  - `output` — **clean** (borrowed-memory DFB via `CBDescriptor::buffer = output.buffer()` at `slice_program_factory_rm_sharded.cpp:278`).

- `SliceRmStrideProgramFactory`
  - `input` — **Case 1** (re-express): `Buffer* input_buffer` pushed directly via `emplace_runtime_args` (4D path: `slice_program_factory_rm_stride.cpp:126`; ND path: `slice_program_factory_rm_stride.cpp:145`). Framework registers these as `BufferBinding`s (correct-on-cache-hit today) but port re-expresses them as `TensorParameter`/`TensorBinding`. Kernels: `reader_multicore_slice_4d.cpp:52` and `reader_multicore_slice_nd.cpp:58` read as `src_addr = get_arg_val<uint32_t>(0)` and pass to `TensorAccessor(src_args, src_addr)`.
  - `output` — **Case 1** (re-express): `Buffer* output_buffer` pushed directly (`slice_program_factory_rm_stride.cpp:134, 158`). Kernels: `writer_multicore_slice_4d.cpp:52` and `writer_multicore_slice_nd.cpp:57` read as `dst_addr = get_arg_val<uint32_t>(0)`.

- `SliceTileProgramFactory`
  - `input` — **Case 1** (re-express): `src0_buffer->address()` in `reader_common_args[0]` at `slice_program_factory_tile.cpp:77`. Kernel (`reader_unary_unpad_dims_interleaved_start_id.cpp:15`) reads as `src_addr = get_common_arg_val<uint32_t>(0)` and passes to `TensorAccessor(src_args, src_addr)`.
  - `output` — **Case 1** (re-express): `dst_buffer->address()` in per-core writer RTAs at `slice_program_factory_tile.cpp:162`. Kernel (`writer_unary_interleaved_start_id.cpp` — slice-local copy) reads as `dst_addr = get_arg_val<uint32_t>(0)` and passes to `TensorAccessor(dst_args, dst_addr)`.

- `SliceTileTensorArgsProgramFactory`
  - `input` — **Case 1** (re-express): `src_buffer->address()` in `reader_common_args[0]` at `slice_program_factory_tile_tensor_args.cpp:103`. Kernel (`reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp:21`) reads as `src_addr = get_common_arg_val<uint32_t>(0)`.
  - `start_tensor` — **Case 1** (re-express): `start_buffer->address()` in `reader_common_args[1]` at `slice_program_factory_tile_tensor_args.cpp:104`. Kernel reads as `start_addr = get_common_arg_val<uint32_t>(1)`.
  - `end_tensor` — **Case 1** (re-express): `end_buffer->address()` in `reader_common_args[2]` at `slice_program_factory_tile_tensor_args.cpp:105`. Kernel reads as `end_addr = get_common_arg_val<uint32_t>(2)`.
  - `output` — **Case 1** (re-express): `dst_buffer->address()` in per-core writer RTAs at `slice_program_factory_tile_tensor_args.cpp:159`. Kernel (`eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp:11`) reads as `dst_addr = get_arg_val<uint32_t>(0)`.

**Custom hash:** None — no custom `compute_program_hash` defined anywhere in the op. Comments in factory headers mentioning it are editorial (describing which attributes drive the default hash). Nothing to delete.

---

## Heads-ups *(mirrors the brief)*

### Notable LANDED constructs

- **Borrowed-memory DFB (`SliceRmShardedProgramFactory`):** Two borrowed-memory CBs: input at `slice_program_factory_rm_sharded.cpp:257-266` (index 0, backed by `input.buffer()`) and output at `slice_program_factory_rm_sharded.cpp:268-278` (index `c_16`, backed by `output.buffer()`). Port uses `DataflowBufferSpec::borrowed_from` naming the respective `TensorParameter`.

### Fake CBs (address-only)

None detected. All CBs with `CircularBuffer` wrapper objects are used with proper `reserve_back`/`push_back` or `wait_front`/`pop_front` FIFO semantics, or are borrowed-memory DFBs (which are covered by the causal-link gate). No address-only (no-producer + no-consumer) fake CBs found.

### Cross-op / shared kernels

- **Borrowed kernel — cross-family:** `SliceTileTensorArgsProgramFactory` at `slice_program_factory_tile_tensor_args.cpp:177` instantiates `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` from the `eltwise/unary` family. That kernel's Metal 2.0 rewrite **must be coordinated** with the eltwise/unary port — slice and eltwise/unary form a **port-together set** for this kernel. Rewriting it for slice alone breaks eltwise/unary, and vice versa.
- **Shared header — in-family:** `slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp` includes `ttnn/operations/data_movement/common/kernels/common.hpp` (data_movement family). The `tt_memmove` function from this header calls legacy `noc_async_read/write` free functions — this is the Device 2.0 holdover flagged in the Gate detail. Cleanup is in the `data_movement/common` team's scope.

### RTA varargs

None. No kernel reads `num_runtime_varargs > 0`, and no factory produces a counted loop of `get_arg_val<uint32_t>(i)` with a runtime-varying index.

### TTNN factory analysis (porter-relevant)

- **Pybind `create_descriptor` — delete:** `bind_slice_descriptor` in `slice_nanobind.cpp` at line 161 exposes `SliceTileProgramFactory::create_descriptor` via `nb::class_<ttnn::prim::SliceTileProgramFactory>(mod, "SliceTileProgramFactory").def_static("create_descriptor", ...)`. The port deletes this (sanctioned device-op-class edit).
- **Other migration-risky pybind:** `bind_slice_descriptor` in `slice_nanobind.cpp` at lines 130-173 also exposes:
  - `nb::class_<ttnn::prim::SliceDeviceOperation>(mod, "SliceDeviceOperation")` with `.def_static("create_output_tensors", ...)` and `.def_static("compute_output_specs", ...)` at lines 149-158. These expose `DeviceOperation` methods to Python and will need assessment/removal during the port.
  - `nb::class_<ttnn::prim::SliceParams>(mod, "SliceParams")` at lines 131-137 and `nb::class_<ttnn::prim::SliceInputs>(mod, "SliceInputs")` at lines 139-147. These expose operation-attributes and tensor-args structs which may need redesign under Metal 2.0.
- **Custom `override_runtime_arguments`:** None.

---

## Team-only

### TensorAccessor convertibility

All bindings classified as Case 1 are straightforward page-by-page iteration — no exotic access patterns. No Case 2 bindings were identified. The `start_tensor` / `end_tensor` bindings in `SliceTileTensorArgsProgramFactory` read slice-boundary tensors (single tiles on device) via `TensorAccessor`; this is page-based and re-expressible.

### Out-of-directory coupling and donor shape analysis

**Op-level roll-up:** `⚠ workable` — the cross-family donor kernel (`eltwise/unary` writer) is structurally Device 2.0, and the shared `data_movement/common` header has isolated holdovers that are being cleaned on the Device 2.0 track.

**Summary table:**

| Op kernel | Donor file | Shape | Notes |
|---|---|---|---|
| `slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp` | `data_movement/common/kernels/common.hpp` | function-call escape | in-family; `tt_memmove` calls `noc_async_read/write` (Device 1.0 holdover, see Device 2.0 gate) |
| `slice_program_factory_tile_tensor_args.cpp` | `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | file-path instantiation | cross-family; Device 2.0 structurally; uses sanctioned `get_local_cb_interface(cb_id)` |

**Per-call detail — `data_movement/common/kernels/common.hpp` helpers called by slice:**

The slice kernel calls `tt::data_movement::common::tt_memmove<false, false, false, 0>(dst, src, bytes)`. Inside `tt_memmove`:

- When `use_read_datamover = false` and `guaranteed_16B_aligned = false` (the call-site instantiation): calls `enhanced_noc_async_write` → `noc_async_write_one_packet` or `noc_async_write<N>`. These are CB-index-free legacy NOC free-functions (Device 1.0 form).
- The operation is a local L1-to-L1 loopback copy (source and destination are both L1 addresses on the same core), not a DRAM/external access. Functionally isolated and restricted to the misalignment-correction path.

Donor shape: the functions take raw L1 `uint32_t` addresses, not `TensorAccessor` or `Semaphore` handles. No clean Metal 2.0 bridge today via named tokens; the Metal 2.0 port leaves this L1-copy utility in place (it operates entirely kernel-side on L1, not on tensor memory). Workaround: after the Device 2.0 cleanup of `common.hpp`, this function will use the Device 2.0 `Noc` member form internally, and the slice kernel's usage will be transparent.

**Borrowed kernel file — cross-family:**

`ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` is instantiated by `SliceTileTensorArgsProgramFactory`. This kernel is also used by many eltwise/unary ops (`unary_program_factory.cpp`, etc.) — it is a broadly-shared kernel. The Metal 2.0 rewrite of this kernel (CB→DFB, named-token bindings) must happen as a single coordinated change across slice and all eltwise/unary ops that share it. Port-together set: at minimum `slice` and the eltwise/unary ops; full set requires a grep across the codebase.

### Relaxation candidates

No custom `compute_program_hash` exists to mine. N/A.

### TTNN factory analysis (six questions)

1. **Op-owned tensors:** No. The `create_device_tensor` call at `slice_device_operation.cpp:231` is inside `create_output_tensors` — standard output tensor creation, not an intermediate/scratch tensor owned by a factory. No factory allocates device tensors beyond the declared input/output.

2. **MeshWorkload concept needed:** No. No factory provides `create_mesh_workload` / `create_workload_descriptor`, and the device-operation carries no `cached_mesh_workload_t`. Q1 is No (no op-owned tensors), so there is no op-owned-tensor artifact either. Single-program structure throughout.

3. **Pybind `create_descriptor`:** Yes — `slice_nanobind.cpp:161`: `nb::class_<ttnn::prim::SliceTileProgramFactory>(mod, "SliceTileProgramFactory").def_static("create_descriptor", ...)`. This binding exposes factory internals. The port deletes it (sanctioned device-op-class edit per `port_op_to_metal2_ttnn_factory.md`).

4. **Other migration-risky pybind:** Yes — `slice_nanobind.cpp:130-173` (`bind_slice_descriptor` function) also exposes `SliceDeviceOperation::create_output_tensors` and `compute_output_specs` (lines 149-158), and the `SliceParams` and `SliceInputs` structs (lines 131-147). The DeviceOperation method exposure is migration-risky; the struct exposure may need redesign. These all live inside `bind_slice_descriptor`, which is a separate binding hook from `bind_slice` (the normal op-function binding). The normal `bind_slice` at line 49 is expected surface and is not a finding.

5. **Custom hash:** No. No `compute_program_hash` method defined anywhere in the op. (Comments in factory header files that mention the hash name are editorial notes about which attributes feed the default hash, not custom method definitions.)

6. **Custom `override_runtime_arguments`:** No. No `static void <Factory>::override_runtime_arguments(...)` declarations or definitions found anywhere in the op.

---

## Misc anomalies *(team-only, non-gating)*

- **`slice_program_factory_rm.cpp:77` — `reader_common_args` size over-provisioned:** The common args vector is initialized with 4 trailing zero-padding slots at positions `[6..9]` (`{..., 0, 0, 0, 0}`) which are subsequently filled by per-core args. These are not actually common — they are per-core values (start_id, num_sticks_per_core, etc.) being clobbered by the per-core runtime_args patching path. This is a latent clarity issue (the common_runtime_args contract is being blurred with per-core RTAs), not a correctness issue today but worth noting for the porter.

- **Comments referencing `compute_program_hash()` behavior** in `slice_program_factory_rm.hpp:16`, `slice_program_factory_rm_sharded.hpp:18`, `slice_program_factory_rm.cpp:216`, `slice_program_factory_rm_sharded.cpp:252` describe how padded_shape is folded into the default hash. These are correct editorial notes but may confuse readers who expect a custom hash override method. Worth clarifying (e.g., "the default TTNN hash automatically keys on operation_attributes, which includes padded_shape").

- **`strided_slice_reader_rm_interleaved_nd.cpp`** at `slice/device/kernels/dataflow/` — unreferenced kernel (not instantiated by any factory). Uses `cb_in0.get_write_ptr()` without `reserve_back` at line 48 (potential fake CB pattern). Out of scope for the port audit but worth flagging to the op owner for eventual cleanup or documentation.

---

## Questions for the user

1. **`bind_slice_descriptor` consumers:** `slice_nanobind.cpp` exposes `SliceTileProgramFactory::create_descriptor`, `SliceDeviceOperation::create_output_tensors`, and `SliceDeviceOperation::compute_output_specs` to Python. Are there known callers of these Python bindings (test infrastructure, external tools)? The port plan should account for how those callers are migrated before this binding is deleted. `file:line`: `slice_nanobind.cpp:130-173`.

2. **Device 2.0 cleanup scheduling for `data_movement/common/kernels/common.hpp`:** The holdover `noc_async_read/write` calls inside `tt_memmove` are the only Device 2.0 blocker. Is this cleanup already tracked / in progress on the Device 2.0 migration team's backlog? The slice port is unblocked once it lands.

---

## Recipe notes

- **Appendix A — `UpdateCircularBufferTotalSize` examples list:** The recipe's "Examples in the wild" for the `UpdateCircularBuffer*` feature entry includes `ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_program_factory_rm.cpp`. In the current codebase there are no such calls in that file. The example is stale. This caused a brief false alarm during the scan; a note that examples may lag actual code would help future auditors.

- **`common.hpp` holdovers — GATE classification:** The recipe's Device 2.0 YELLOW holdover definition ("isolated CB-index-keyed free-function holdovers from the free-function family") is shaped around CB-index free functions. The `noc_async_read/write` calls in `common.hpp` are a different kind of holdover — raw NOC free functions inside a utility helper. The recipe doesn't explicitly address whether a helper-function indirection from a Device 2.0 kernel to a legacy NOC free function counts as a YELLOW holdover or a RED. I classified it YELLOW because: (a) the caller kernel is structurally Device 2.0, (b) the operation is a local L1 copy (not an external tensor access), and (c) the usage is functionally isolated to one code path. The recipe maintainer may want to extend the YELLOW definition or add a note about helper-function indirection.
