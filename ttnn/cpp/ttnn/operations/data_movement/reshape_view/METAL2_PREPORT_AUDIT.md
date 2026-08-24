# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/data_movement/reshape_view`

Single device operation, two program factories:

- **`ReshapeViewDeviceOperation`**
  - `ReshapeViewRMProgramFactory` (`device/reshape_rm_program_factory.cpp`) — `create_descriptor` → `ProgramDescriptor`; kernel `device/device/rm_reshape_interleaved.cpp` (instantiated once as reader on CBs 0/1, and again as writer on CBs 2/3 via a dual-instance work-split when `can_use_dual_kernel`).
  - `ReshapeViewTiledProgramFactory` (`device/reshape_tiled_program_factory.cpp`) — `create_workload_descriptor` → `WorkloadDescriptor`; op-owned host-computed mapping tensor parked on `workload_descriptor.buffers`; kernels `device/device/dataflow/reader_reshape_tiled.cpp` + `device/device/dataflow/writer_reshape_tiled.cpp`.

Both factories belong to one `DeviceOperation` and share the `compute_program_hash` / `validate` / `compute_output_specs` host code, so they are audited together and bundled into this one report. Where findings differ per factory they are attributed accordingly (see Per-DeviceOperation attribution).

Unreferenced files: none. `device/hostdevcommon/common.hpp` is a shared host/device header (`SegmentMapData`), referenced by the tiled kernels and factory — in scope.

**Out of bounds (not audited):** `ttnn/cpp/ttnn/operations/experimental/quasar/reshape_view/` holds a quasar copy of this op (readiness-sheet rows 249–250). Per the audit recipe it is not a source, precedent, or naming reference. The porter should **not** consult it.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `355760227dd 2026-08-24 docs(metal_2.0): a run in flight freezes the kernel sources`

> **Update (re-audit):** The RM factory's kernel `rm_reshape_interleaved.cpp` has since been migrated to Device 2.0 (CBs now go through `CircularBuffer` wrapper objects — `rm_reshape_interleaved.cpp:96-103`, with `api/dataflow/circular_buffer.h` included at line 33). That was the only op-level blocker in the prior pass; the Device 2.0 gate now clears for both factories and the op is **GREEN**. The RM factory's informational subjects (deferred while it was blocked) have been run and folded in below.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/reshape_view` |
| **Overall** | **GREEN** — brief issued (both factories) |
| **DOps / Factories** | `ReshapeViewDeviceOperation` → `ReshapeViewRMProgramFactory` (descriptor), `ReshapeViewTiledProgramFactory` (WorkloadDescriptor) |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all three kernels use `CircularBuffer` wrappers, `Noc`, `TensorAccessor`, `Noc`-first helpers. (RM kernel migrated since the prior pass.) |
| *Prereqs* — Cross-op escapes | Ok — only in-family donor `data_movement/common/kernels/common.hpp`, Device 2.0 native (`Noc`-first) |
| *Feature Support* — overall | GREEN (all Appendix A entries N/A) |
| *Feature Support* — GlobalCircularBuffer / address_offset / GlobalSemaphore | N/A / N/A / N/A |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** (both factories; sheet cross-check clean) |
| *TTNN Readiness* — Concept (current) | RM: `descriptor` · Tiled: `WorkloadDescriptor` |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | Tiled: **Yes** (single `desc` replicated across coord ranges; sheet "Why secretly SPMD?" = "Op-owned tensors") |
| *TTNN Readiness* — Custom hash | **Yes** (not a gate; port leaves it intact): `device/reshape_device_operation.cpp:48-63` |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No |
| *TTNN Readiness* — `override_runtime_arguments` | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Op-owned tensors | Tiled: **Yes** — mapping tensor @ `device/reshape_tiled_program_factory.cpp:459-461` |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept` (Tiled: + op-owned tensors) |
| *Port work* — Offset base pointer | none (both factories — no host-folded offset) |
| *Port work* — Tensor bindings (per binding) | RM: src Case 1 · dst Case 1 · Tiled: input Case 1 · mapping (op-owned) Case 1 · output Case 1 |
| *TTNN Readiness* — TensorParameter relaxation | `none` (both — clears) |
| *Port work* — TensorAccessor 3rd arg | none — no accessor passes a 3rd arg (both factories) |
| *Port work* — CB endpoints | RM: `c_0`/`c_1` self-loop; `c_2`/`c_3` self-loop (conditional on `can_use_dual_kernel`) · Tiled: `c_0` mapping 1:1 · `c_1` input 1:1 · `c_2` output/scratch self-loop |

## Result

**GREEN → brief issued.** Every gate clears for both factories: Device 2.0 ✓, features ✓ (all Appendix A N/A), TTNN factory concept ✓ (`Is able to port? = yes` for both), offset base pointers ✓, TensorAccessor 3rd arg N/A. `METAL2_PORT_BRIEF.md` covers the whole op (both factories → `ProgramSpecFactoryConcept`; tiled additionally carries op-owned tensors). The prior pass's RED (RM kernel Device 2.0) has been cleared by the kernel migration; the mixed-concept-variant question is therefore moot — both factories port together.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN — both factories `yes`.** Sheet cross-check clean and complete (readiness sheet fetched fresh this session; rows 71–72):
  - Row `ReshapeViewRMProgramFactory`: `Concept=descriptor` (verified — `create_descriptor` returns `ProgramDescriptor`, `reshape_row_major_program_factory.hpp:15`); `Custom hash=yes` (verified — `reshape_device_operation.cpp:48`); `get_dynamic_runtime_args=no` (verified — none in dir); `Override runtime args method?=no` (verified); `Pybind descriptor=no` (verified — `reshape_nanobind.cpp` has no `create_descriptor` / `nb::class_`); `Smuggled pointer=no` (verified — factory passes `Buffer*` binding-form args, not `->address()`); `TensorParameter relaxation=none`.
  - Row `ReshapeViewTiledProgramFactory`: `Concept=WorkloadDescriptor` (verified — `create_workload_descriptor`, `reshape_tiled_program_factory.hpp:21`); `Op-owned tensors?=yes` (verified — `reshape_tiled_program_factory.cpp:461` pushes the mapping buffer to `workload_descriptor.buffers`); `Secretly SPMD Workload?=yes` (verified — one `desc` built once and replicated across `tensor_coords` ranges, `reshape_tiled_program_factory.cpp:471-488`); `Custom hash=yes`; `get_dynamic_runtime_args=no`; `TensorParameter relaxation=none`.
  - Factory-set match: sheet rows 71–72 ↔ code's two factories, one-to-one. No phantom / missing rows. Cross-column invariant `Op-owned tensors? == yes` only on the `WorkloadDescriptor` row — satisfied.

- **Device 2.0 (every kernel used):** **GREEN.** All three kernels are structurally Device 2.0:
  - `rm_reshape_interleaved.cpp` — `Noc noc` (line 90), `TensorAccessor` (87-88), `enhanced_noc_async_read/write` + `tt_memmove` in `Noc`-first form, and CBs via `CircularBuffer cb_in0/cb_in1` wrapper objects with `.reserve_back()` / `.get_write_ptr()` / `.push_back()` (96-103). No free-function CB idioms, no legacy addr-gen, no raw `noc_async_*` remain (grep clean). *(Migrated since the prior audit pass, which had RED'd this kernel on manual CB index management.)*
  - `reader_reshape_tiled.cpp` — `CircularBuffer` wrappers (40-41), `Noc`, `TensorAccessor`, `enhanced_noc_async_read`.
  - `writer_reshape_tiled.cpp` — `CircularBuffer` wrappers (33-35), `Noc`, `TensorAccessor`, `enhanced_noc_async_write` / `tt_memmove`.

- **Feature compatibility:** every Appendix A entry, in order. No entry's recognition signals fire in either factory.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | Both factories use plain `CBDescriptor`s; no `.global_circular_buffer` field, no `remote_index`, no `experimental::…GlobalCircularBuffer` type. |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `CBDescriptor` sets `address_offset`; no `set_address_offset` / 4-arg `UpdateDynamicCircularBufferAddress` / `cb_descriptor_from_sharded_tensor`. |
  | GlobalSemaphore | N/A | Op uses no semaphores at all. |

- **Offset base pointers:** **GREEN (both factories).** No address RTA folds a host-side offset into a base. Both factories deliver buffer bases via the `Buffer*` binding form (`emplace_runtime_args(core, {src_buffer, dst_buffer, …})` @ `reshape_rm_program_factory.cpp:257-282`; `emplace_runtime_args(c, {input_buffer, mapping_buffer, …})` / `{output_buffer, …}` @ `reshape_tiled_program_factory.cpp:406-407`) — no `->address()` expression, no `base + offset` arithmetic anywhere. Type 3 (`address_offset`) and Type 4 (`narrow`) absent. (No entry in the dated offset-base-pointer triage; scan is the source of truth and it is clean.)

- **TensorAccessor 3rd argument:** **N/A — no accessor in the op passes a 3rd argument.** RM: `TensorAccessor(src_args, src_addr)`, `TensorAccessor(dst_args, dst_addr)` (`rm_reshape_interleaved.cpp:87-88`). Tiled reader: `TensorAccessor(input_args, input_addr)`, `TensorAccessor(map_args, map_addr)` (`reader_reshape_tiled.cpp:36-37`). Tiled writer: `TensorAccessor(output_args, output_base_addr)` (`writer_reshape_tiled.cpp:30`). All two-argument. The subject never fires. (No entry in the dated 3rd-arg triage; consistent.)

- **CB endpoints (GATE-free).** Census per CB, per node.
  - **RM factory** — the kernel is instantiated as two disjoint-CB instances: reader-config on CBs 0/1, writer-config on CBs 2/3 (the factory swaps `writer_compile_time_args[2]=src2_cb_index`, `[3]=src3_cb_index`, `reshape_rm_program_factory.cpp:207-209`). Each instance both fills and drains **its own** CB pair as scratch (`reserve_back`/`get_write_ptr`/`push_back`, never `wait_front`/`pop_front` — `rm_reshape_interleaved.cpp:98-103`), so every CB has exactly **one toucher** → **self-loop** (bind that kernel PRODUCER + CONSUMER).
    - `c_0` (src0), `c_1` (src1) — always allocated → self-loop.
    - `c_2` (src2), `c_3` (src3) — allocated **only when `can_use_dual_kernel`** (`reshape_rm_program_factory.cpp:187-205`) → self-loop, **conditional DFB** on `can_use_dual_kernel`.
    - Note: this is a dual-instance work-split (same `kernel_source`, Reader/Writer configs, one `total_cores`) but with **disjoint** CB sets per instance — so there are **no co-touched CBs** and no 1P+1C / multi-binding question here; every CB is a single-toucher self-loop.
  - **Tiled factory** — per node (single core group; identical program on every active core):
    - `c_0` mapping CB — reader FIFO-produces (`reader_reshape_tiled.cpp:44,49`), writer FIFO-consumes (`writer_reshape_tiled.cpp:41,75`) → 2 touchers, 1 locked P + 1 locked C → **plain 1:1, legal.**
    - `c_1` input CB — reader FIFO-produces (`reader_reshape_tiled.cpp:68,74`), writer FIFO-consumes (`writer_reshape_tiled.cpp:51,59,82`) → **plain 1:1, legal.**
    - `c_2` output/working CB — touched only by the writer, used as an L1 scratch page (`reserve_back`/`get_write_ptr`/`push_back`, never waited/popped — `writer_reshape_tiled.cpp:38-39,84`) → **1 toucher → self-loop.**

## Port-work summary  *(mirrors the brief — whole op)*

- **Tensor bindings** (per binding):
  - RM `src` — **Case 1** (`TensorAccessor`). `Buffer*` BufferBinding (`reshape_rm_program_factory.cpp:259`), consumed via `TensorAccessor(src_args, src_addr)` (`rm_reshape_interleaved.cpp:87`).
  - RM `dst` — **Case 1** (`TensorAccessor`). `Buffer*` BufferBinding (`reshape_rm_program_factory.cpp:260`), consumed via `TensorAccessor(dst_args, dst_addr)` (`rm_reshape_interleaved.cpp:88`).
  - Tiled `input` — **Case 1** (`TensorAccessor`). `Buffer*` BufferBinding (`reshape_tiled_program_factory.cpp:406`), consumed via `TensorAccessor(input_args, input_addr)` (`reader_reshape_tiled.cpp:36`).
  - Tiled `mapping` (**op-owned tensor**) — **Case 1** (`TensorAccessor`). Op-owned mapping buffer, `Buffer*` BufferBinding (`reshape_tiled_program_factory.cpp:406`), consumed via `TensorAccessor(map_args, map_addr)` (`reader_reshape_tiled.cpp:37`). Carried natively by `ProgramSpecFactoryConcept`'s op-owned-tensor support.
  - Tiled `output` — **Case 1** (`TensorAccessor`). `Buffer*` BufferBinding (`reshape_tiled_program_factory.cpp:407`), consumed via `TensorAccessor(output_args, output_base_addr)` (`writer_reshape_tiled.cpp:30`).
- **TensorParameter relaxation:** none.
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:** RM — self-loop `c_0`, `c_1` (always), `c_2`, `c_3` (conditional DFB on `can_use_dual_kernel`). Tiled — self-loop `c_2` (writer-only scratch); `c_0`/`c_1` already legal 1:1.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none in either factory — no hidden second writer, no multi-reader.
- **RM dual-instance work-split with disjoint CBs:** the RM factory pushes the same `kernel_source` into a reader-config and a writer-config KernelDescriptor over one `total_cores` (`reshape_rm_program_factory.cpp:175-217`). Recognition-wise this is the dual-instance shape, but the two instances use **disjoint** CB sets (0/1 vs 2/3) — so, unlike the canonical work-split, there is no shared CB to assign 1P+1C; every CB self-loops. Watch that the port keeps the CB→instance mapping (the CTA swap at 207-209) intact.
- **RM idle-core handling:** the factory creates the kernel on **all** cores in `total_cores`, marking spare cores idle with a trailing `nop=1` RTA and passing `0u` for the buffer slots (`reshape_rm_program_factory.cpp:225-227`); the kernel returns early on `nop==1` (`rm_reshape_interleaved.cpp:83-85`) **before** constructing any `TensorAccessor`. Under typed `TensorParameter` bindings the framework will still deliver the base to those cores, but the kernel never builds/uses the accessor, so it is harmless — confirm the binding model is content binding an idle core that early-returns.
- **Cross-op / shared kernels:** all three kernels `#include ttnn/operations/data_movement/common/kernels/common.hpp` (in-family shared helper pool) and call `enhanced_noc_async_read` / `enhanced_noc_async_write` / `tt_memmove` — all Device 2.0 native (`Noc`-first) → function-call escape is `✓ clean`, no donor-side work. **No borrowed kernel *files*** — each factory instantiates only its own kernels. No `_metal2` fork exists or is needed.
- **RTA varargs:** none. Every RTA is a fixed, distinct, nameable field (RM: `src_addr`,`dst_addr`,`source_read_size_bytes`,`read_start_page`,`read_end_page`,`write_start_page`,`write_start_offset`,`nop`; tiled reader: `input_addr`,`map_addr`,`start_output_page_idx`,`end_output_page_idx`; tiled writer: `output_base_addr`,`start_output_page`,`end_output_page`). No CTA varargs.

## Team-only

- **Out-of-directory coupling & donor shape:**
  - Op-level roll-up: **✓ clean.** Only escape is the in-family `data_movement/common/kernels/common.hpp` helper.

  | Op kernel | Donor file | Class | Status |
  |---|---|---|---|
  | `rm_reshape_interleaved.cpp` | `data_movement/common/kernels/common.hpp` | in-family shared | ✓ (functions `Noc`-first, Device 2.0) |
  | `reader_reshape_tiled.cpp` | `data_movement/common/kernels/common.hpp` | in-family shared | ✓ |
  | `writer_reshape_tiled.cpp` | `data_movement/common/kernels/common.hpp` | in-family shared | ✓ |

  Per-call detail: functions consumed — `enhanced_noc_async_read(Noc, uint64_t, uint32_t, uint32_t)`, `enhanced_noc_async_write(Noc, uint32_t, uint64_t, uint32_t)`, `tt_memmove(Noc, uint32_t, uint32_t, uint32_t)` — all take a leading `Noc` object (Device 2.0 native; the no-`Noc` overloads are `[[deprecated]]` and unused here). No `InterleavedAddrGen`/`ShardedAddrGen`/`Semaphore`/`CircularBuffer&` shapes cross the boundary. No gate.
  - Borrowed kernel files: none — all three kernels are owned by `reshape_view`.
- **TTNN factory analysis:** current concepts RM=`descriptor`, Tiled=`WorkloadDescriptor` (secretly SPMD, collapses to single-program); op-owned tensors on the tiled factory only (mapping tensor, `reshape_tiled_program_factory.cpp:459-461`); custom `compute_program_hash` present (`reshape_device_operation.cpp:48-63`, left intact by the port); no `get_dynamic_runtime_args`, no `override_runtime_arguments`, no pybound `create_descriptor`. Target concept for both: `ProgramSpecFactoryConcept` (tiled + op-owned tensors).
- **Relaxation candidates (FALLIBLE):** none noticed. The custom hash (`reshape_device_operation.cpp:55-62`) hashes `logical_output_shape`, `output_mem_config`, `sub_core_grid` presence+value, `tensor_args`, and `program_factory.index()` — no obvious property-narrowing candidate; `TensorParameter relaxation` is `none` on the sheet and the port leaves the hash alone.

## Misc anomalies  *(team-only, non-gating)*

- **`recreate_mapping_tensor` op attribute is accepted but unused.** Threaded through `ReshapeViewParams` (`reshape_device_operation_types.hpp:18`) and the `reshape_view` entry point, but the tiled factory explicitly ignores it (`reshape_tiled_program_factory.cpp:463-466`) and it is intentionally excluded from the program hash (`reshape_device_operation.cpp:54`). Effectively dead. Route to the ops team.
- **RM kernel `write_start_offset` (RTA 6) is a de-facto constant 0.** The factory always passes `0u` for it and the comment says it was "removed (always 0)" (`reshape_rm_program_factory.cpp:281`), yet the kernel still reads it and folds it into `writable`/`write_offset` (`rm_reshape_interleaved.cpp:66,95,106`). Dead-valued RTA. Team-only.

## Per-DeviceOperation attribution

Single DeviceOperation; both factories now clear:

| Factory | Concept → target | Device 2.0 | Gate result |
|---|---|---|---|
| `ReshapeViewRMProgramFactory` | descriptor → `ProgramSpecFactoryConcept` | ✓ (kernel migrated) | **Clear** — all gates pass |
| `ReshapeViewTiledProgramFactory` | WorkloadDescriptor (secretly SPMD, op-owned tensors) → `ProgramSpecFactoryConcept` + op-owned tensors | ✓ | **Clear** — all gates pass |

## Questions for the user  *(none)*

The prior pass's open question (mixed-concept variant on one DeviceOperation) is resolved: with the RM kernel's Device 2.0 migration landed, both factories clear and port together, so no mixed variant arises.

## Recipe notes

- **RM kernel was a Device-2.0 middle case the size buckets don't name cleanly** (recorded in the prior pass, retained for the maintainer). Before the fix, its addressing was fully Device 2.0 but its CB layer was entirely free-function with no wrapper in scope — neither an "isolated holdover" (wrapper-in-scope precondition failed) nor "broad Device 1.0" (no addr-gen to rewrite). It has since been migrated by introducing `CircularBuffer` wrappers (exactly the fix described). A third size label — "single-layer migration (CB or NoC only)" — would fit ops like this better and set the right re-audit-cost expectation.
