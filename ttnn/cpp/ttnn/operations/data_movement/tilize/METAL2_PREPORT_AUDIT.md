# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/data_movement/tilize`

Directory contains one `DeviceOperation` with five program factories:

- **`TilizeDeviceOperation`** (`tilize_device_operation.cpp` / `.hpp`, `tilize_device_operation_types.hpp`)
  - `TilizeSingleCoreProgramFactory` (`tilize_single_core_program_factory.cpp`)
  - `TilizeMultiCoreDefaultProgramFactory` (`tilize_multi_core_default_program_factory.cpp`)
  - `TilizeMultiCoreBlockProgramFactory` (`tilize_multi_core_block_program_factory.cpp`)
  - `TilizeMultiCoreShardedProgramFactory` (`tilize_multi_core_sharded_program_factory.cpp`)
  - `TilizeMultiCoreWidthShardedProgramFactory` (`tilize_multi_core_width_sharded_program_factory.cpp`)

Unreferenced kernel file: `tilize/device/kernels/compute/tilize.cpp` — present in the directory but no factory references it; all compute paths use the shared-pool kernel at `ttnn/cpp/ttnn/kernel/compute/tilize.cpp`. Noted here for clarity; not audited.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `port_op_to_metal2_audit.md`.

---

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/tilize` |
| **Overall** | YELLOW |
| **DOps / Factories** | `TilizeDeviceOperation` → `TilizeSingleCoreProgramFactory`, `TilizeMultiCoreDefaultProgramFactory`, `TilizeMultiCoreBlockProgramFactory`, `TilizeMultiCoreShardedProgramFactory`, `TilizeMultiCoreWidthShardedProgramFactory` |
| *Prereqs* — ProgramDescriptor | Yes |
| *Prereqs* — Device 2.0 (every kernel used) | **YELLOW (open question — see Questions for the user #1):** All own kernels and most donor kernels are Device 2.0 compliant. One cross-family donor, `reader_unary_pad_multicore_both_dims.cpp` (owned by `tilize_with_val_padding`), includes `common/kernels/common.hpp` and calls `tt_memmove` which internally uses Device 1.0 `noc_async_read` / `noc_async_write` free functions. Whether this constitutes an isolated holdover (YELLOW) or a broadly Device 1.0 dependency (RED) requires user confirmation. See Questions #1. |
| *Prereqs* — Cross-op escapes | ⚠ workable (file-path kernel borrowing; widely shared donor kernels; `reader_unary_pad_multicore_both_dims.cpp` is the ambiguous case) |
| *Feature Support* — overall | GREEN — no UNSUPPORTED features in use |
| *Feature Support* — Variadic-CTA | N/A |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — MeshWorkload needed | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Other risky pybind | None |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — Custom override-RTA | No |
| *TTNN Readiness* — Fake CBs (address-only) | None |

---

## Result

**YELLOW — open question on Device 2.0 compliance of one cross-family donor kernel.** All other gates clear. If the user confirms `reader_unary_pad_multicore_both_dims.cpp` is YELLOW (isolated holdover — Device 2.0 structurally, with `tt_memmove` as a legacy utility to clean up on the D2.0 track), the brief is issued and the port can proceed after that cleanup. If the user confirms RED (broadly Device 1.0 because of `common.hpp`), the Device 2.0 gate fails and the brief is withheld until `common.hpp` migrates.

All five factories share the ProgramDescriptor prerequisite (cleared), share the feature compatibility scan (all clear), and differ only in which donor kernels they use.

---

## Gate detail

### ProgramDescriptor: GREEN

All five factories return `ProgramDescriptor` from their `create_descriptor` static method. They use `CBDescriptor`, `KernelDescriptor`, `SemaphoreDescriptor`-free (no semaphores needed), and `emplace_runtime_args` / `Buffer*` slot injection. No `host_api.hpp` imperative builder calls (`CreateKernel`, `CreateCircularBuffer`, `SetRuntimeArgs`, etc.) appear anywhere in the op's device directory.

Evidence:
- `tilize_single_core_program_factory.cpp:18` — `ProgramDescriptor TilizeSingleCoreProgramFactory::create_descriptor(...)`
- `tilize_multi_core_default_program_factory.cpp:20` — `ProgramDescriptor TilizeMultiCoreDefaultProgramFactory::create_descriptor(...)`
- `tilize_multi_core_block_program_factory.cpp:76` — `ProgramDescriptor TilizeMultiCoreBlockProgramFactory::create_descriptor(...)`
- `tilize_multi_core_sharded_program_factory.cpp:18` — `ProgramDescriptor TilizeMultiCoreShardedProgramFactory::create_descriptor(...)`
- `tilize_multi_core_width_sharded_program_factory.cpp:18` — `ProgramDescriptor TilizeMultiCoreWidthShardedProgramFactory::create_descriptor(...)`

### Device 2.0 (every kernel used): YELLOW — pending user confirmation

**Own kernels (all Device 2.0 compliant):**

- `tilize/device/kernels/dataflow/reader_unary_stick_layout_split_rows_singlecore.cpp` — includes `api/dataflow/noc.h`, `api/dataflow/circular_buffer.h`, `api/core_local_mem.h`, `api/tensor/noc_traits.h`; uses `Noc`, `CircularBuffer`, `TensorAccessor`, `CoreLocalMem`. Clean Device 2.0.
- `tilize/device/kernels/dataflow/reader_unary_stick_layout_split_rows_multicore.cpp` — same headers and patterns. Clean Device 2.0.
- `tilize/device/kernels/compute/tilize_wh.cpp` — compute kernel only; uses `api/compute/tilize.h` and `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp`. Compute kernel, out of scope for Device 2.0 dataflow check.

**Shared-pool kernels (Device 2.0 compliant):**

- `ttnn/cpp/ttnn/kernel/compute/tilize.cpp` — compute kernel only; same pattern. Out of scope for dataflow check.

**Cross-family donor kernels (eltwise/unary):**

- `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` — uses `Noc`, `CircularBuffer`, `TensorAccessor`. Uses `get_local_cb_interface(cb_id_out)` at line 19 — sanctioned Device 2.0 free function per the migration guide. Clean Device 2.0.
- `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id_wh.cpp` — uses `Noc`, `CircularBuffer`, `TensorAccessor`. Uses `get_tile_size(cb_id_out)` at line 24 — sanctioned Device 2.0 free function. Clean Device 2.0.
- `eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp` — uses `CircularBuffer` only (`cb.push_back`). No NoC, no tensor access. Clean Device 2.0.

**Cross-family donor kernels (data_movement/sharded):**

- `data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp` — uses `CircularBuffer` only (`cb_out.wait_front`). Clean Device 2.0.

**Cross-family donor kernel (tilize_with_val_padding) — AMBIGUOUS:**

- `tilize_with_val_padding/device/kernels/dataflow/reader_unary_pad_multicore_both_dims.cpp` — The kernel's own code is Device 2.0: it includes `api/dataflow/noc.h`, `api/dataflow/circular_buffer.h`, `api/tensor/noc_traits.h`, `api/core_local_mem.h`, and uses `Noc`, `CircularBuffer`, `TensorAccessor`, `CoreLocalMem`, `UnicastEndpoint`. However, it also includes `cpp/ttnn/operations/data_movement/common/kernels/common.hpp` and calls `tt_memmove<false, false, true, 0>(...)` at line 159 (only on the misaligned code path). `tt_memmove` internally calls `noc_async_read(...)` and `noc_async_write(...)` Device 1.0 free functions via `enhanced_noc_async_read` / `enhanced_noc_async_write` in `common.hpp`. `common.hpp` broadly uses Device 1.0 patterns throughout.

  Holdover candidate at:
  | File | Lines | Call | D2.0 wrapper in scope |
  |---|---|---|---|
  | `common/kernels/common.hpp` | 30, 32 | `noc_async_read_one_packet`, `noc_async_read<…>` (inside `enhanced_noc_async_read`) | Yes — `Noc noc` is in scope in the calling kernel |
  | `common/kernels/common.hpp` | 43, 45 | `noc_async_write_one_packet`, `noc_async_write<…>` (inside `enhanced_noc_async_write`) | Yes |
  | `common/kernels/common.hpp` | 59, 65 | `get_noc_addr(src_l1_addr)` (inside `tt_memmove`) | Yes |

  **Open question:** Is this an isolated holdover (Device 2.0 structurally, holdovers in a utility function — YELLOW) or does `common.hpp`'s broad Device 1.0 use make the kernel broadly non-compliant (RED)? See Questions for the user #1.

### Feature compatibility: GREEN — no UNSUPPORTED features in use

| Feature | Status | Notes |
|---|---|---|
| GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type or `global_circular_buffer` field in any factory |
| Dynamic CircularBuffer (borrowed memory) | GREEN | `TilizeMultiCoreShardedProgramFactory` and `TilizeMultiCoreWidthShardedProgramFactory` set `CBDescriptor::buffer = src_buffer` and `= dst_buffer`. Port uses `DataflowBufferSpec::borrowed_from`. |
| CBDescriptor `address_offset` (non-zero) | N/A | `.address_offset` not set in any descriptor |
| Aliased Circular Buffers | N/A | No `CBDescriptor` with multi-element `format_descriptors` |
| GlobalSemaphore | N/A | No semaphores used by this op |
| Non-zero semaphore initial value | N/A | No semaphores |
| Dynamic TensorAccessor (`ArgConfig::Runtime*`) | N/A | No `ArgConfig::Runtime` references |
| `UpdateCircularBuffer*` | N/A | No `UpdateCircularBuffer` calls anywhere in the op |
| Variable-count compile-time arguments (CTA varargs) | N/A | `tensor_args_t` is `TilizeInputs` with a fixed `Tensor input_tensor` + `std::optional<Tensor> optional_input_tensor` — no variable-count container |

**Dynamic CircularBuffer detail (LANDED — FYI-P for porter):**

`TilizeMultiCoreShardedProgramFactory` (`tilize_multi_core_sharded_program_factory.cpp:53`, `:67`) and `TilizeMultiCoreWidthShardedProgramFactory` (`tilize_multi_core_width_sharded_program_factory.cpp:54`, `:68`) set `cb_src0.buffer = src_buffer` and `cb_output.buffer = dst_buffer`. This is the borrowed-memory CB (Dynamic CircularBuffer) pattern — LANDED in Metal 2.0 as `DataflowBufferSpec::borrowed_from`. No gate; port uses `borrowed_from`.

These borrowed-memory CBs are real DFBs (not fake CBs): the reader (`reader_unary_sharded.cpp`) produces into the source CB via `cb.push_back(num_tiles_per_shard)`, and the compute kernel consumes from it. The compute kernel produces into the output CB and the writer (`writer_unary_sharded.cpp`) consumes from it via `cb_out.wait_front(num_units)`. Both have genuine producer + consumer pairs.

---

## Port-work summary *(mirrors the brief)*

**Tensor bindings (per binding, per factory):**

All factories use the `Buffer*`-binding form for their runtime args (not raw `->address()`, but the pointer itself). This is the framework's interim mechanism — auto-registered as `BufferBinding`, patched on cache hits. Still classified Case 1 (re-express via `TensorParameter`) since the kernel consumes a raw `uint32_t` base address. Routine port work; not a correctness hazard.

- `TilizeSingleCoreProgramFactory`:
  - `input_tensor` → `src0_buffer` (Buffer* in reader RTA slot 0 — `tilize_single_core_program_factory.cpp:122`). **Case 1.**
  - `output_tensor` → `dst_buffer` (Buffer* in writer RTA slot 0 — `tilize_single_core_program_factory.cpp:142`). **Case 1.**

- `TilizeMultiCoreDefaultProgramFactory`:
  - `input_tensor` → `src0_buffer` (Buffer* in reader RTA slot 0 — `tilize_multi_core_default_program_factory.cpp:168`, `:195`). **Case 1.**
  - `output_tensor` → `dst_buffer` (Buffer* in writer RTA slot 0 — `tilize_multi_core_default_program_factory.cpp:181`, `:208`). **Case 1.**

- `TilizeMultiCoreBlockProgramFactory`:
  - `input_tensor` → `src0_buffer` (Buffer* in reader RTA slot 0 — `tilize_multi_core_block_program_factory.cpp:310`). **Case 1.**
  - `output_tensor` → `dst_buffer` (Buffer* in writer RTA slot 0 — `tilize_multi_core_block_program_factory.cpp:323`). **Case 1.**
  - Note: Both tensors also pass their buffer pointers as compile-time `TensorAccessorArgs` (reader CTA args, `tilize_multi_core_block_program_factory.cpp:205`; writer CTA args, `:218`). These are CTA-baked address pointers. Per the recipe, the CTA-baked-address form forces a recompile per address so it is not the silent-wrong hazard — but it is still a pointer argument to enumerate as Case 1 (re-express via `TensorParameter`; the CTA-baked address and NTTP base disappear).

- `TilizeMultiCoreShardedProgramFactory`:
  - `input_tensor` → borrowed-memory CB (`cb_src0.buffer = src_buffer` at `tilize_multi_core_sharded_program_factory.cpp:53`). **Clean** (causal-link gate: real DFB with producer + consumer).
  - `output_tensor` → borrowed-memory CB (`cb_output.buffer = dst_buffer` at `:67`). **Clean.**

- `TilizeMultiCoreWidthShardedProgramFactory`:
  - `input_tensor` → borrowed-memory CB (`cb_src0.buffer = src_buffer` at `tilize_multi_core_width_sharded_program_factory.cpp:54`). **Clean.**
  - `output_tensor` → borrowed-memory CB (`cb_output.buffer = dst_buffer` at `:68`). **Clean.**

**Custom hash:** None — `TilizeDeviceOperation` does not define `compute_program_hash`. The default hash applies. No action.

---

## Heads-ups *(mirrors the brief)*

- **Notable constructs — Dynamic CircularBuffer (borrowed-memory DFB):**
  - `TilizeMultiCoreShardedProgramFactory`: `tilize_multi_core_sharded_program_factory.cpp:53` (`cb_src0.buffer = src_buffer`), `:67` (`cb_output.buffer = dst_buffer`). Port uses `DataflowBufferSpec::borrowed_from`.
  - `TilizeMultiCoreWidthShardedProgramFactory`: `tilize_multi_core_width_sharded_program_factory.cpp:54`, `:68`. Same pattern.

- **Cross-op / shared kernels:**
  - `writer_unary_interleaved_start_id.cpp` (owner: `eltwise/unary`) — broadly shared across many data_movement and eltwise ops. Port-together set includes at minimum: `tilize`, `tilize_with_val_padding`, `slice`, `transpose`, `concat`, `reshape_on_device`, `copy`, and several `bcast` ops. Any Metal 2.0 rewrite of this kernel must coordinate all co-borrowers.
  - `writer_unary_interleaved_start_id_wh.cpp` (owner: `eltwise/unary`) — shared by `tilize` (block factory) and `tilize_with_val_padding` (block interleaved factory). Port-together set: at minimum `tilize` and `tilize_with_val_padding`.
  - `reader_unary_sharded.cpp` (owner: `eltwise/unary`) — shared by many data_movement ops including `tilize`, `tilize_with_val_padding`, `untilize`, `transpose`, `sharded_to_interleaved`, etc.
  - `writer_unary_sharded.cpp` (owner: `data_movement/sharded`) — shared by `tilize`, `tilize_with_val_padding`, `untilize`, `transpose`, `interleaved_to_sharded`, etc.
  - `reader_unary_pad_multicore_both_dims.cpp` (owner: `tilize_with_val_padding`) — borrowed by `tilize`'s block factory only. Port-together set: `tilize` + `tilize_with_val_padding`.
  - `ttnn/cpp/ttnn/kernel/compute/tilize.cpp` (shared kernel pool) — used by multiple tilize factories (single-core, multi-core default, sharded, width-sharded). Also likely used by other ops — check before porting.

- **RTA varargs:** None detected. All kernels use positional `get_arg_val<uint32_t>(i)` with fixed indices.

- **TTNN factory analysis (porter-relevant):**
  - Pybind `create_descriptor`: None. `tilize_nanobind.cpp` binds only `ttnn::tilize` (the user-facing function) — no ProgramFactory innards exposed.
  - Other risky pybind: None.
  - Custom `override_runtime_arguments`: None in any factory.

---

## Team-only

### TensorAccessor convertibility

All Case-1 bindings above are standard iteratable (page-by-page) patterns — not exotic NoC walks. The access pattern in `reader_unary_stick_layout_split_rows_singlecore.cpp` uses `noc.async_read(s, dst, width_size, {.page_id = stick_ids[k], .offset_bytes = stick_offset}, {.offset_bytes = 0})` — page-id + offset iteration, fully expressible via `TensorAccessor`. The multicore reader uses the same pattern. The block factory reader kernel (`reader_unary_pad_multicore_both_dims.cpp`) uses `s.get_noc_addr(size_2d + k)` — also standard accessor usage. All Case 1 bindings are **convertible** (not exotic).

### Out-of-directory coupling and donor shape analysis

**Op-level roll-up:** `⚠ workable` — no Shape 4 donors (pre-Device-2.0 addr-gens in the donor signatures themselves), but file-path borrowing is widespread and port-together coupling is significant. The one open question (`reader_unary_pad_multicore_both_dims.cpp` + `common.hpp`) may upgrade this to `⭐ blocked` if the Device 2.0 judgment for that kernel resolves as RED.

**Summary table:**

| Op kernel (factory) | Donor file | Donor class | Shape analysis |
|---|---|---|---|
| `reader_unary_stick_layout_split_rows_singlecore.cpp` | (no #include escapes outside own dir) | — | clean |
| `reader_unary_stick_layout_split_rows_multicore.cpp` | (no #include escapes outside own dir) | — | clean |
| `tilize_wh.cpp` (compute) | `ttnn/kernel_lib/tilize_helpers.hpp` | shared kernel lib | compute only; `uint32_t` CB indices → `dfb::name` constexpr cast handles |
| `ttnn/kernel/compute/tilize.cpp` | `ttnn/kernel_lib/tilize_helpers.hpp` | shared kernel pool | compute only; same |
| `writer_unary_interleaved_start_id.cpp` | `api/tensor/noc_traits.h` (TensorAccessor) | LLK/HAL | ✓ excellent — takes `TensorAccessorArgs` via CTA |
| `writer_unary_interleaved_start_id_wh.cpp` | `api/tensor/noc_traits.h` | LLK/HAL | ✓ excellent |
| `reader_unary_sharded.cpp` | (no significant escapes) | — | ✓ clean — CB-index parameter, `uint32_t cb_id_in0` |
| `writer_unary_sharded.cpp` | (no significant escapes) | — | ✓ clean — CB-index parameter |
| `reader_unary_pad_multicore_both_dims.cpp` (block factory) | `common/kernels/common.hpp` | in-family data_movement helper | ⚠ — `tt_memmove` uses Device 1.0 `noc_async_read`/`noc_async_write` internally; full analysis under Device 2.0 gate above |

**Per-call detail for `reader_unary_pad_multicore_both_dims.cpp`:**

The kernel's own code calls `noc.async_read(s, dst, width_size, ...)` and `noc.async_read(UnicastEndpoint{}, temp_dst, ..., {.noc_x=..., .noc_y=..., .addr=...}, ...)` — both Device 2.0. The `tt_memmove<false, false, true, 0>(l1_write_addr, temp_addr + ..., width_size)` call at line 159 reaches into `common.hpp`'s `enhanced_noc_async_read(get_noc_addr(src_l1_addr), dst_l1_addr, bytes)` which calls `noc_async_read<...>(...)` — Device 1.0 free function.

**Borrowed kernel files (file-path kernel instantiation):**

| Kernel file path | Owning family | Broadly shared? | Other known users |
|---|---|---|---|
| `ttnn/cpp/ttnn/kernel/compute/tilize.cpp` | `ttnn/kernel/` shared pool | Yes | used by `TilizeSingleCoreProgramFactory`, `TilizeMultiCoreDefaultProgramFactory`, `TilizeMultiCoreShardedProgramFactory`, `TilizeMultiCoreWidthShardedProgramFactory`; likely used by other ops |
| `eltwise/unary/.../writer_unary_interleaved_start_id.cpp` | `eltwise/unary` | Yes (widely shared) | `tilize_with_val_padding`, `slice`, `transpose`, `concat`, `reshape_on_device`, `copy`, `bcast`, etc. |
| `eltwise/unary/.../writer_unary_interleaved_start_id_wh.cpp` | `eltwise/unary` | Narrower | `tilize_with_val_padding` (block interleaved factory) |
| `eltwise/unary/.../reader_unary_sharded.cpp` | `eltwise/unary` | Yes (widely shared) | `tilize_with_val_padding`, `untilize`, `transpose`, `sharded_to_interleaved`, `untilize_with_unpadding`, etc. |
| `data_movement/sharded/.../writer_unary_sharded.cpp` | `data_movement/sharded` | Yes | `tilize_with_val_padding`, `untilize`, `transpose`, `interleaved_to_sharded`, etc. |
| `tilize_with_val_padding/.../reader_unary_pad_multicore_both_dims.cpp` | `tilize_with_val_padding` | Narrow (2 users) | `tilize_with_val_padding` (block interleaved factory) |

**Port-together sets:**
- The `reader_unary_sharded.cpp`, `writer_unary_sharded.cpp`, `writer_unary_interleaved_start_id.cpp` rewrites are high-impact multi-op coordinations — these should be planned as family-wide ports rather than op-by-op. Any Metal 2.0 rewrite of these kernels requires simultaneous migration of every op that instantiates them.
- `reader_unary_pad_multicore_both_dims.cpp` is a two-user kernel: `tilize` and `tilize_with_val_padding` must port together.

### Relaxation candidates

None — no custom `compute_program_hash` to mine.

### TTNN factory analysis

1. **Op-owned tensors?** No. `TilizeDeviceOperation::create_output_tensors` calls `create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input_tensor.device())` — this creates the declared output tensor, not an intermediate. No other device tensor allocations in any factory.

2. **MeshWorkload concept needed?** No. No `create_mesh_workload`, no `cached_mesh_workload_t`, no multi-program or cross-device coordination. Straightforward single-program op.

3. **Pybind `create_descriptor`?** No. `tilize_nanobind.cpp:39` binds only `ttnn::tilize` (the user-facing function via `bind_function<"tilize">`). No `nb::class_<…ProgramFactory>` or `create_descriptor` binding.

4. **Other migration-risky pybind?** None. The nanobind file is minimal — only the op function binding with its parameters.

5. **Custom hash?** No `compute_program_hash` defined in `TilizeDeviceOperation` or any factory.

6. **Custom override-runtime-args?** No `override_runtime_arguments` in any factory.

---

## Misc anomalies *(team-only, non-gating)*

- `tilize_device_operation_types.hpp:25` — `TilizeInputs` has an `optional_input_tensor` field, but no factory reads it, and `tilize_device_operation.cpp:255` passes `std::nullopt` for it unconditionally. Appears to be dead/unused infrastructure — present for potential future use or a vestige of an earlier design. Op owner's concern; not porter-actionable.

- Unreferenced kernel file: `tilize/device/kernels/compute/tilize.cpp` — this file is in the op's own directory but no factory references it. It has a different CTA interface than the shared-pool `ttnn/kernel/compute/tilize.cpp` (4 CTAs: cb_id_in0, cb_id_out0, per_core_block_cnt, per_core_block_tile_cnt vs. 2 CTAs in the shared version). Could cause confusion during the port if someone assumes the local file is in use. Consider removing or documenting.

- `tilize_single_core_program_factory.cpp:57` — `(a.device()->l1_size_per_core() / 2)` is a hardcoded ½ fraction for L1 budget. This is a pre-existing heuristic, not a port concern, but worth noting.

---

## Questions for the user

1. **Device 2.0 compliance of `reader_unary_pad_multicore_both_dims.cpp`:** This cross-family donor kernel (owned by `tilize_with_val_padding`, used by `TilizeMultiCoreBlockProgramFactory`) includes `common/kernels/common.hpp` and calls `tt_memmove<false, false, true, 0>(l1_write_addr, temp_addr + ..., width_size)` at line 159. `tt_memmove` internally calls Device 1.0 free functions (`noc_async_read`, `noc_async_write`, `get_noc_addr`) via `enhanced_noc_async_read`/`enhanced_noc_async_write` in `common.hpp`. The kernel's primary structure uses Device 2.0 wrappers (`Noc`, `CircularBuffer`, `TensorAccessor`, `CoreLocalMem`, `UnicastEndpoint`); the `tt_memmove` call is one call site only on the misaligned code path.

   **Question:** Does this constitute:
   - **(a) YELLOW — isolated holdover:** The kernel is structurally Device 2.0 and `tt_memmove` is an isolated utility using legacy internals. The fix (replace `tt_memmove` with a Device 2.0 equivalent) should happen on the Device 2.0 track before the port, but the port is still feasible and a brief should be issued.
   - **(b) RED — broadly non-compliant:** `common.hpp`'s Device 1.0 patterns are broad enough that `reader_unary_pad_multicore_both_dims.cpp` is not yet Device 2.0 compliant, and the Device 2.0 gate fails until `common.hpp` migrates.

   If (a): the brief will include a Blocked-until notice for this holdover. The `TilizeMultiCoreBlockProgramFactory` path is blocked until `reader_unary_pad_multicore_both_dims.cpp`'s `tt_memmove` call is replaced; the other four factories may be portable immediately.
   If (b): full RED on Device 2.0; no brief issued.

---

## Recipe notes

- The recipe's Device 2.0 yellow-vs-red boundary (§Prerequisites, Check 2) is clear for kernel-level holdovers but less clear for utility headers pulled in via `#include`. The `common.hpp` case — where the kernel's own code is fully Device 2.0 but an included helper contains raw `noc_async_read` — doesn't fit cleanly into either the YELLOW holdover description ("isolated CB-index free-function family") or the RED description ("broadly uses legacy Device 1.0 idioms"). Adding a note in the recipe about utility headers / `#include` escapes that pull in Device 1.0 code would help future auditors.

- The recipe says `get_tile_size(cb_id)` and `get_local_cb_interface(cb_id)` are sanctioned Device 2.0 free functions. Both appear in the donor kernels (`writer_unary_interleaved_start_id.cpp:19` and `writer_unary_interleaved_start_id_wh.cpp:24`) and were correctly not flagged.
