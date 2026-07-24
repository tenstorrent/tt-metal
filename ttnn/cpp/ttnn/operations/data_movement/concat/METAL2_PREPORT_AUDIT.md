# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/data_movement/concat`

**Op directory:** `ttnn/cpp/ttnn/operations/data_movement/concat`

This directory contains one `DeviceOperation` with five `ProgramFactory` variants:

- **`ConcatDeviceOperation`**
  - `ConcatProgramFactory` (`concat_program_factory.cpp`) — interleaved (tiled and RM layouts)
  - `ConcatS2STiledProgramFactory` (`concat_s2s_tiled_program_factory.cpp`) — sharded-to-sharded, tiled, two-tensor only
  - `ConcatS2SRMProgramFactory` (`concat_s2s_rm_program_factory.cpp`) — sharded-to-sharded, row-major, two-tensor only
  - `ConcatS2SMultiProgramFactory` (`concat_s2s_multi_program_factory.cpp`) — sharded-to-sharded, multi-tensor
  - `ConcatS2IProgramFactory` (`concat_s2i_program_factory.cpp`) — sharded-to-interleaved

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `port_op_to_metal2_audit.md`.

---

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/concat` |
| **Overall** | RED |
| **DOps / Factories** | `ConcatDeviceOperation` → `ConcatProgramFactory`, `ConcatS2STiledProgramFactory`, `ConcatS2SRMProgramFactory`, `ConcatS2SMultiProgramFactory`, `ConcatS2IProgramFactory` |
| *Prereqs* — ProgramDescriptor | Yes |
| *Prereqs* — Device 2.0 (every kernel used) | No |
| *Prereqs* — Cross-op escapes | issue (see Out-of-directory coupling) |
| *Feature Support* — overall | RED |
| *Feature Support* — Variadic-CTA | Unsupported (GATE — primary blocker) |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — MeshWorkload needed | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Other risky pybind | None |
| *TTNN Readiness* — Custom hash | Yes → delete (see Custom program hash) |
| *TTNN Readiness* — Custom override-RTA | No |
| *TTNN Readiness* — Fake CBs (address-only) | None |

---

## Result

**RED — blocked on two independent gates:**

1. **CTA varargs (UNSUPPORTED):** `ConcatDeviceOperation`'s `tensor_args_t` is `ConcatInputs` which holds a `std::vector<Tensor> input_tensors` — a runtime-variable count. The factories (`ConcatProgramFactory`) bake `num_input_tensors` as a compile-time arg and loop over `TensorAccessorArgs` per tensor; the kernels use `make_tensor_accessor_args_tuple<num_tensors, ...>()` where `num_tensors` is a CTA. Metal 2.0's `compile_time_args` schema requires a fixed-shape declaration; variable-count CTA varargs are not yet supported. This is the **primary blocker**; routed to the wait-for-feature team. This is the canonical example explicitly cited in Appendix A of the audit recipe.

2. **Device 2.0 non-compliance:** Three kernels used by this op are not Device 2.0 compliant — `reader_concat_interleaved_start_id.cpp`, `reader_concat_stick_layout_interleaved_start_id.cpp` (own kernels; use `noc_async_read` / `noc_async_read_barrier` raw free functions without a `Noc` object), and `writer_unary_stick_layout_interleaved_start_id.cpp` (borrowed from `ttnn/cpp/ttnn/kernel/dataflow/`; uses `cb_wait_front`, `get_read_ptr`, `cb_pop_front`, `noc_async_write` — Device 1.0 CB and NoC free functions). Routed to the Device 2.0 migration team.

No brief is issued. The port cannot proceed until the CTA-varargs feature lands (requiring a framework change) and the Device 2.0 kernel migrations complete. Both gates must clear before the port can begin.

---

## Gate detail

### ProgramDescriptor

**GREEN.** All five factories use `ProgramDescriptor` API: each `create_descriptor` populates a `tt::tt_metal::ProgramDescriptor`, pushes `CBDescriptor` and `KernelDescriptor` entries, and sets per-core `runtime_args`. The imperative `host_api.hpp` builder API (`CreateProgram`, `CreateKernel`, `SetRuntimeArgs`) is not used.

### Device 2.0 (every kernel used)

**RED (GATE).** Three of the kernels exercised by this op's program factories are not Device 2.0 compliant:

| File | Line | Call | Wrapper in scope |
|---|---|---|---|
| `device/kernels/dataflow/reader_concat_interleaved_start_id.cpp` | 50 | `noc_async_read(read_addr, l1_write_addr, tile_size_bytes)` | None (no `Noc noc` object declared) |
| `device/kernels/dataflow/reader_concat_interleaved_start_id.cpp` | 51 | `noc_async_read_barrier()` | None |
| `device/kernels/dataflow/reader_concat_stick_layout_interleaved_start_id.cpp` | 56 | `noc_async_read(read_addr, l1_write_addr, page_size)` | None |
| `device/kernels/dataflow/reader_concat_stick_layout_interleaved_start_id.cpp` | 65 | `noc_async_read(read_addr, l1_write_addr, page_size)` | None |
| `device/kernels/dataflow/reader_concat_stick_layout_interleaved_start_id.cpp` | 78 | `noc_async_read_barrier()` | None |
| `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp` | 26 | `cb_wait_front(cb_id_out0, 1)` | None (no `CircularBuffer` wrapper) |
| `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp` | 27 | `get_read_ptr(cb_id_out0)` | None |
| `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp` | 29 | `noc_async_write(l1_read_addr, dst_noc_addr, stick_size)` | None |
| `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp` | 31 | `cb_pop_front(cb_id_out0, 1)` | None |

**Notes:**
- `reader_concat_interleaved_start_id.cpp` and `reader_concat_stick_layout_interleaved_start_id.cpp` do use the `CircularBuffer` wrapper for CB operations (Device 2.0), but all NoC I/O is via raw legacy free functions (`noc_async_read`, `noc_async_read_barrier`). No `Noc` object is instantiated. These are not "structurally Device 2.0 with isolated holdovers" — the entire NoC data-movement layer is Device 1.0 free-function form.
- `writer_unary_stick_layout_interleaved_start_id.cpp` is a broadly-shared kernel in `ttnn/cpp/ttnn/kernel/dataflow/` (shared pool; used by many ops). It uses `TensorAccessor` for addressing but all CB and NoC operations are Device 1.0 free functions. Its Device 2.0 migration must land (on that kernel's own track) before this op can be ported.
- Device-2.0-compliant kernels (no issues): `reader_s2s_tensor_concat.cpp`, `writer_s2i_width.cpp`, `reader_height_sharded_width_concat_two_tensors.cpp`, `reader_height_sharded_width_concat_two_tensors_tiled.cpp`, `writer_height_sharded_width_concat_two_tensors_tiled.cpp`, `height_sharded_width_concat_two_tensors.cpp` (compute), and `writer_unary_interleaved_start_id.cpp` (eltwise/unary donor) all use `Noc noc`, `CircularBuffer` wrappers, or are compute-only. 

### Feature compatibility

Run regardless of gate outcome.

| Feature | Status | Notes |
|---|---|---|
| GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` references anywhere in op or kernel files |
| Dynamic CircularBuffer (borrowed memory) | GREEN | `CBDescriptor::buffer = input_tensors[input_id].buffer()` used in `ConcatS2SMultiProgramFactory` (lines 98, 117), `ConcatS2STiledProgramFactory` (lines 102, 116), `ConcatS2SRMProgramFactory` (lines 69, 86), and `ConcatS2IProgramFactory` (line 45); port uses `borrowed_from` |
| CBDescriptor `address_offset` (non-zero) | N/A | No `address_offset` field set anywhere |
| Aliased Circular Buffers | N/A | Every `CBDescriptor::format_descriptors` has a single `CBFormatDescriptor` element |
| GlobalSemaphore | N/A | No semaphores of any kind used |
| Non-zero semaphore initial value | N/A | No semaphores |
| Dynamic TensorAccessor (`ArgConfig::Runtime*`) | N/A | No `ArgConfig::Runtime` tokens in any factory |
| `UpdateCircularBuffer*` | N/A | No `UpdateCircularBufferTotalSize` / `UpdateCircularBufferPageSize` / `UpdateDynamicCircularBufferAddressAndTotalSize` calls |
| Variable-count compile-time arguments (CTA varargs) | **RED (GATE)** | See detail below |

#### CTA varargs detail (UNSUPPORTED — RED GATE)

**Signal:** `ConcatInputs::input_tensors` is `std::vector<Tensor>` (`concat_device_operation_types.hpp:20`) — a runtime-varying count input list. This is the op-level signal.

**Kernel-level signal:** `reader_concat_interleaved_start_id.cpp:21` and `reader_concat_stick_layout_interleaved_start_id.cpp:21` — `make_tensor_accessor_args_tuple<num_tensors, page_size_base_idx + num_tensors>()` where `num_tensors` is CTA index 1 (a runtime-varying count). `reader_concat_interleaved_start_id.cpp:27` and `reader_concat_stick_layout_interleaved_start_id.cpp:27` — `uint32_t num_tiles_per_block[num_tensors]` (VLA sized by CTA). This is the textbook kernel-level CTA-vararg pattern.

**Host-level confirmation:** `concat_program_factory.cpp:205-210` — the factory builds `reader_compile_time_args` as `{src0_cb_index, num_input_tensors, page_size_per_tensor[0], ..., page_size_per_tensor[N-1], TensorAccessorArgs[0], ..., TensorAccessorArgs[N-1]}` where `N = input_tensors.size()` at call time. The CTA schema length varies with the number of inputs.

**File:line sites:**
- `device/concat_device_operation_types.hpp:20` — `std::vector<Tensor> input_tensors` (op-level signal)
- `device/kernels/dataflow/reader_concat_interleaved_start_id.cpp:18,21,27` — `num_tensors` CTA + `make_tensor_accessor_args_tuple<num_tensors, ...>` + VLA
- `device/kernels/dataflow/reader_concat_stick_layout_interleaved_start_id.cpp:18,21,26-27` — same pattern
- `device/concat_program_factory.cpp:205-210` — factory-side CTA construction loop

**Note:** This is the example explicitly cited in the CTA varargs Appendix A entry: *"Examples in the wild: `ttnn/cpp/ttnn/operations/data_movement/concat/` — accepts a runtime-varying list of input tensors."* The expected resolution is: CTA-vararg support is on the Metal 2.0 host API roadmap; the port becomes possible once that feature lands.

**Note on `ConcatS2SMultiProgramFactory` also affected:** `concat_s2s_multi_program_factory.cpp:122` passes `num_input_tensors` as CTA index 3. The `reader_s2s_tensor_concat.cpp` kernel (line 17) reads `num_input_tensors` from CTA and loops over it at line 24. This is a runtime-loop over CTA-provided count. All five factories are blocked.

---

## Port-work summary *(mirrors the brief — moot until gates clear)*

**Tensor bindings** (observed-but-moot until gates clear):

The following bindings are enumerated for planning purposes. All are either accessed via `CBDescriptor::buffer` (borrowed-memory DFB, `ConcatS2SMultiProgramFactory`, `ConcatS2STiledProgramFactory`, `ConcatS2SRMProgramFactory`, `ConcatS2IProgramFactory`) or via `TensorAccessor` + raw-buffer-address RTA bypass (`ConcatProgramFactory`).

- `ConcatProgramFactory` — `input_tensors[i]` (each): **Case 1** (re-express). The factory extracts `src_addr[i] = buffer->address()` at `concat_program_factory.cpp:175,190` and injects them as RTAs (`common_reader_kernel_args`). The kernels read `src_addr_base_idx` via `get_arg_addr(src_addr_base_idx)` and pass to `TensorAccessor`. These are buffer-address RTAs — Case 1.
- `ConcatProgramFactory` — `output` tensor: **Case 1** (re-express). `dst_buffer->address()` is passed as RTA to the writer kernel (`concat_program_factory.cpp:283-287`).
- `ConcatS2SMultiProgramFactory`, `ConcatS2STiledProgramFactory`, `ConcatS2SRMProgramFactory`, `ConcatS2IProgramFactory` — inputs and output: **clean** (borrowed-memory DFB via `CBDescriptor::buffer`). These are genuine borrowed-memory DFBs — sharded input CBs have both producer (the sharded data in the buffer) and consumer (kernel reads via `cb.get_read_ptr()` or `cb.wait_front()`). The causal-link gate applies; these do not force into Case 1 or Case 2.
- `ConcatS2IProgramFactory` — `output` tensor in `writer_s2i_width.cpp`: `TensorAccessorArgs<2>()` used; `TensorAccessor(dst_args, dst_addr)` constructed with `dst_addr` from RTA. **Case 1** (re-express). The host side passes `output.buffer()->address()` as RTA (`concat_s2i_program_factory.cpp:82`).

**Custom hash:** delete custom `compute_program_hash` → default (sanctioned exception). See Custom program hash section.

---

## Heads-ups *(mirrors the brief — moot until gates clear)*

- **Notable LANDED constructs:**
  - **Borrowed-memory DFB** in `ConcatS2SMultiProgramFactory` (`concat_s2s_multi_program_factory.cpp:98,117`), `ConcatS2STiledProgramFactory` (`concat_s2s_tiled_program_factory.cpp:102,116`), `ConcatS2SRMProgramFactory` (`concat_s2s_rm_program_factory.cpp:69,86`), `ConcatS2IProgramFactory` (`concat_s2i_program_factory.cpp:45`). Port uses `DataflowBufferSpec::borrowed_from`.

- **Cross-op / shared kernels:**
  - `writer_unary_stick_layout_interleaved_start_id.cpp` borrowed from `ttnn/cpp/ttnn/kernel/dataflow/` — shared pool, used broadly. Device 2.0 migration of this kernel must land first (separately tracked).
  - `writer_unary_interleaved_start_id.cpp` borrowed from `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/` — eltwise/unary family. Already Device 2.0 compliant.

- **RTA varargs:** `ConcatS2SMultiProgramFactory` and `ConcatS2IProgramFactory` have loop-constructed runtime arg vectors whose length varies with `num_input_tensors` at call time. These are RTA varargs (runtime-varying count), **not** CTA varargs — RTA varargs ARE supported in Metal 2.0 via the vararg mechanism. Report for porter awareness; prefer named RTAs when the count is practically bounded. Recognition sites: `concat_s2s_multi_program_factory.cpp:124-136`, `concat_s2i_program_factory.cpp:79-91`.

- **TTNN factory analysis (porter-relevant):** No `create_descriptor` pybind, no other migration-risky pybind, no `override_runtime_arguments`.

---

## Team-only

### TensorAccessor convertibility (per Case-2 binding)

No Case-2 bindings found. All bindings are either clean (borrowed-memory DFB, causal-link gate) or Case 1 (buffer-address RTA → re-express via `TensorParameter`).

### Out-of-directory coupling & donor shape analysis

**Op-level roll-up:** ⭐ `blocked` — one borrowed kernel is Device 1.0 (gates D2.0 prerequisite); one is broadly shared with multiple co-borrowers.

**Summary table:**

| Op kernel | Donor file | Status | Shape |
|---|---|---|---|
| `ConcatProgramFactory` reader (tiled) | `device/kernels/dataflow/reader_concat_interleaved_start_id.cpp` | op-owned, D1.0 | — |
| `ConcatProgramFactory` reader (RM) | `device/kernels/dataflow/reader_concat_stick_layout_interleaved_start_id.cpp` | op-owned, D1.0 | — |
| `ConcatProgramFactory` writer (RM) | `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp` | shared-pool, D1.0 | TensorAccessor D2.0; CB/NoC D1.0 |
| `ConcatProgramFactory` writer (tiled) | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | cross-family borrow, D2.0 clean | — |
| `ConcatS2SMultiProgramFactory` reader+writer | `device/kernels/dataflow/reader_s2s_tensor_concat.cpp` | op-owned, D2.0 clean | — |
| `ConcatS2STiledProgramFactory` reader | `device/kernels/dataflow/reader_height_sharded_width_concat_two_tensors_tiled.cpp` | op-owned, D2.0 clean | — |
| `ConcatS2STiledProgramFactory` writer | `device/kernels/dataflow/writer_height_sharded_width_concat_two_tensors_tiled.cpp` | op-owned, D2.0 clean | — |
| `ConcatS2STiledProgramFactory` compute | `device/kernels/compute/height_sharded_width_concat_two_tensors.cpp` | op-owned, D2.0 clean (compute, CB-only) | — |
| `ConcatS2SRMProgramFactory` reader+writer | `device/kernels/dataflow/reader_height_sharded_width_concat_two_tensors.cpp` | op-owned, D2.0 clean | — |
| `ConcatS2IProgramFactory` reader | *(missing — see Misc anomalies)* | missing file | — |
| `ConcatS2IProgramFactory` writer | `device/kernels/dataflow/writer_s2i_width.cpp` | op-owned, D2.0 clean | — |

**Borrowed kernel files:**
- `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp` — shared pool; broadly used across many ops. Device 2.0 migration of this file is a pre-requisite for this op's port **and** for every other op that instantiates it. The Metal 2.0 rewrite of this file must be a coordinated single change across all co-borrowers.
- `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` — eltwise/unary family; already D2.0 clean. Port-together coupling exists: any Metal 2.0 rewrite of this kernel must coordinate with eltwise/unary ops.

**Cross-family function-call escapes:** No `#include` of headers outside the op directory or standard system headers was found in the kernel files. All cross-op coupling is via file-path kernel instantiation only (noted above).

### Relaxation candidates (mined from custom hash before deletion)

**FALLIBLE — candidates to verify, default strict.**

The custom `compute_program_hash` in `concat_device_operation.cpp:178–214` explicitly hashes:
- `tensor_args.input_tensors.size()` (line 186) — the op keys hard on the input count; no relaxation here.
- Per-tensor: `logical_shape`, `padded_shape`, `layout`, `dtype`, `memory_config` — all full-fidelity.
- Output spec: `logical_shape`, `padded_shape`, `layout`, `data_type`, `memory_config`.
- Also: `factory.index()` and `operation_attributes.dim/groups/output_mem_config/sub_core_grids`.

This hash is unusually comprehensive and does not suggest obvious relaxation candidates — it closely mirrors what the default hash would include. No relaxation candidates surfaced.

### TTNN factory analysis

1. **Op-owned tensors?** No. No factory calls `create_device_tensor` or `allocate_tensor_on_device` internally; all device tensors are passed in through `tensor_args` or are the output returned via `tensor_return_value`.

2. **MeshWorkload concept needed?** No. No `create_mesh_workload` / `create_workload_descriptor` / `cached_mesh_workload_t`. Standard single-program path.

3. **Pybind `create_descriptor`?** No. `concat_nanobind.cpp` only binds `&ttnn::concat` via `bind_function<"concat">` — the normal op-function surface, not factory innards. No `nb::class_<...ProgramFactory>` appears.

4. **Other migration-risky pybind?** None. The nanobind file is minimal: one `bind_function<"concat">` call.

5. **Custom hash?** Yes — `concat_device_operation.cpp:178–214`. See Custom program hash section for treatment (delete → default, sanctioned exception). `concat_device_operation.hpp:44` declares it.

6. **Custom override-runtime-args?** No. None of the five factories define a `static void override_runtime_arguments(...)`.

---

## Custom program hash

**Location:** `concat_device_operation.hpp:44` (declaration), `concat_device_operation.cpp:178–214` (definition).

**Treatment:** PORT WORK — the port deletes this custom `compute_program_hash` and reverts to the default TTNN hash. This is the sanctioned device-op-class exception (see port recipe). Relaxation candidates mined from the hash are recorded in Team-only above.

**Note on the hash's correctness:** The custom hash loops over `tensor_args.input_tensors` with a variable-count iteration (lines 189–200), hashing each tensor's shape/layout/dtype/memcfg. This was necessary because the input list is a `std::vector<Tensor>` — the vector size and per-element specs must all differentiate programs. The default hash will not naturally cover this variable-count case; this is one more manifestation of the CTA varargs structural issue. The hash will be revisited as part of the eventual Metal 2.0 port planning.

---

## Misc anomalies *(team-only, non-gating)*

- **Missing kernel file:** `concat_s2i_program_factory.cpp:54–55` references `"ttnn/cpp/ttnn/operations/data_movement/concat/device/kernels/dataflow/reader_s2i_width.cpp"` as `reader_desc.kernel_source`, but that file does not exist in the `device/kernels/dataflow/` directory. The `writer_s2i_width.cpp` exists; `reader_s2i_width.cpp` is absent. This would cause a build failure on the `ConcatS2IProgramFactory` path. Routes to the op owner; not porter-actionable (the port audit does not execute kernels).

- **`concat_program_factory.cpp:175,190`:** `src_addr[i] = buffer->address()` — buffer addresses extracted into a `std::vector<uint32_t> src_addr` that is then inserted into `common_reader_kernel_args` (line 199). These are the buffer-address RTAs enumerated as Case 1 bindings in the port-work summary. Recorded here as well for completeness; the correctness hazard (stale base on cache hit) is not the "silent-wrong" variety because the custom `compute_program_hash` includes per-tensor memory configs and would differentiate on storage change — but under Metal 2.0's fast-path-cache binding injection the RTA path would be bypassed. Treat as Case 1 at port time.

---

## Questions for the user

1. **ConcatS2IProgramFactory — missing reader kernel:** `reader_s2i_width.cpp` is referenced in the factory but does not exist (see Misc anomalies). Is this path currently dead code, a build-time failure being masked, or has the file been recently removed? This affects whether the S2I path needs to be treated as in-scope for the port or can be excluded as dead code.

2. **Device 2.0 for shared `writer_unary_stick_layout_interleaved_start_id.cpp`:** This kernel is in the shared pool (`ttnn/cpp/ttnn/kernel/dataflow/`) and used by many ops. Who owns its Device 2.0 migration, and is there a tracking issue? The concat port depends on that migration landing first.

---

## Recipe notes

- The `reader_concat_interleaved_start_id.cpp` and `reader_concat_stick_layout_interleaved_start_id.cpp` kernels present a nuanced D2.0 classification question: they use `CircularBuffer` wrappers (D2.0 CB API) but `noc_async_read` / `noc_async_read_barrier` free functions (D1.0 NoC API). The recipe's YELLOW holdover criterion says "uses `experimental::Noc`, `experimental::CircularBuffer`, etc. for the bulk of operations and has a small number of isolated legacy holdovers." These kernels are the inverse: CB ops are D2.0 but NoC ops are entirely D1.0. I classified them as RED (broadly D1.0 for NoC), not YELLOW holdover. The recipe could be clearer about the case where one API layer (CB vs. NoC) has been migrated while the other has not.

- The CTA varargs GATE fires clearly and the canonical concat example is cited in Appendix A. No ambiguity there.

- The `address_offset = 0` default is not explicitly set in any `CBDescriptor` in the op, which means the default applies (zero). I treated absence of the field as green for this check per the false-positive guard ("`.address_offset` not set (default zero) is fine").
