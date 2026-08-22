# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/data_movement/pad`

Single device operation, seven program factories:

- **`PadDeviceOperation`** (`device/pad_device_operation.{hpp,cpp}`)
  - `PadRmReaderWriterMultiCoreDefaultProgramFactory` (`pad_rm_reader_writer_multi_core_default_program_factory.cpp`) — `descriptor`
  - `PadRmReaderWriterMultiCoreProgramFactory` (`pad_rm_reader_writer_multi_core_program_factory.cpp`) — `WorkloadDescriptor` (secretly SPMD, op-owned tensor)
  - `PadRmReaderWriterProgramFactory` (`pad_rm_reader_writer_program_factory.cpp`) — `WorkloadDescriptor` (secretly SPMD, op-owned tensor)
  - `PadRmShardedHeightOnlyProgramFactory` (`pad_rm_sharded_height_only_program_factory.cpp`) — `descriptor`
  - `PadRmShardedWidthOnlyProgramFactory` (`pad_rm_sharded_width_only_program_factory.cpp`) — `descriptor`
  - `PadTileMulticoreProgramFactory` (`pad_tile_multicore_program_factory.cpp`) — `descriptor`
  - `PadTileCoreProgramFactory` (`pad_tile_program_factory.cpp`) — `descriptor`

Kernels (all under `device/kernels/dataflow/`, all owned by pad) + one cross-family donor kernel from `eltwise/unary/` and one in-family header from `data_movement/common/`. No unreferenced kernel files.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/pad` |
| **Overall** | **RED** |
| **DOps / Factories** | `PadDeviceOperation` → 7 factories (5 `descriptor`, 2 `WorkloadDescriptor`) |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** (GREEN — own kernels + eltwise donor + `data_movement/common` donor all Device 2.0) |
| *Prereqs* — Cross-op escapes | Ok (one cross-family donor kernel + one shared header; both Device 2.0; port-together coupling noted) |
| *Feature Support* — overall | **GREEN** (all Appendix A entries N/A) |
| *Feature Support* — Variadic-CTA | Ok (all `get_compile_time_arg_val` at constexpr indices) |
| *TTNN Readiness* — `Is able to port?` (the gate) | **No** — fails on 2 of 7 factories **+ readiness-sheet conflict** (see Gate detail) |
| *TTNN Readiness* — Concept (current) | 5 × `descriptor`, 2 × `WorkloadDescriptor` |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | Yes (both — identical `ProgramDescriptor` broadcast across mesh-coord ranges) |
| *TTNN Readiness* — Is safe to port? | Yes for 6 factories; **No** for `PadTileCoreProgramFactory` (→ readiness-sheet owner) |
| *TTNN Readiness* — Custom hash | No (all) |
| *TTNN Readiness* — Runtime-args update | **Yes** — device-op `get_dynamic_runtime_args` hook, active for `PadRmShardedHeightOnlyProgramFactory` (#48928) |
| *TTNN Readiness* — Pybind `create_descriptor` | No (nanobind binds only the op function) |
| *TTNN Readiness* — Op-owned tensors | **Yes** — pad-value const tensor on both `WorkloadDescriptor` factories |
| *TTNN Readiness* — Target concept | `MetalV2FactoryConcept` (+ op-owned tensor for the 2 WorkloadDescriptor factories) |
| *Port work* — Offset base pointer | none (GREEN — all bases clean) |
| *Port work* — Tensor bindings (per binding) | Case 1 (interleaved/tiled factories) · clean/borrowed-DFB (sharded factories) |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | drop (Class 2) — default-MC factory only |
| *Port work* — CB endpoints | 1P+1C (data CBs) · self-loop (pad-value/scratch CBs); no multi-binding, no dead CBs |

**CB endpoints** are dispositions, not gates. All resolvable at port time (details below).

## Result

**RED — blocked on the TTNN factory concept gate**, with three distinct threads that must be reconciled *together* because they are entangled:

1. **`PadRmShardedHeightOnlyProgramFactory` — Runtime-args update.** The device op declares a `get_dynamic_runtime_args` hook (`pad_device_operation.cpp:235`) added by #48928 to re-trip the descriptor fast-path for this factory's borrowed sharded CBs. Runtime-args-update is a **current TTNN-side gate** (`Is able to port? = no`), lifting when that Metal 2.0 infra ships. → **TTNN / ProgramDescriptor-migration team.**
2. **`PadTileCoreProgramFactory` — `Is safe to port? = no`.** The readiness sheet marks this factory unsafe (the sheet owner's correctness axis, not re-derived here). → **readiness-sheet owner.**
3. **Readiness-sheet conflict (spreadsheet-broken → gate).** The sheet's `Runtime-args update (get_dynamic_runtime_args)` column is **mis-attributed**: it marks `PadTileCoreProgramFactory = yes` and `PadRmShardedHeightOnlyProgramFactory = no`, but the code shows the exact reverse — the hook returns `{}` for every factory except the height-sharded one (`pad_device_operation.cpp:243-248`). → **readiness-sheet owner to reconcile** before any port proceeds.

**Candidate clean subset (do not port yet — see the open question):** the other five factories —
`PadRmReaderWriterMultiCoreDefaultProgramFactory`, `PadRmReaderWriterMultiCoreProgramFactory`, `PadRmReaderWriterProgramFactory`, `PadRmShardedWidthOnlyProgramFactory`, `PadTileMulticoreProgramFactory` —
carry `Is able to port? = yes` and independently clear every gate-bearing check (Device 2.0, Features, Offset base pointers, 3rd-arg). **`RED at op level; subset {default-MC, MC-workload, SC-workload, width-only-sharded, tile-multicore} is the candidate clean set, pending reconciliation.`**

No `METAL2_PORT_BRIEF.md` is issued. Two reasons: the recipe treats a readiness-sheet conflict as *"stop rather than proceed on data we can't trust"*, and the runtime-args-update hook lives on the **shared** `PadDeviceOperation` — so whether the 5-factory subset can port while the device op retains that hook is an open framework question (see *Questions for the user*), not something to hand a porter as a "go."

## Gate detail

- **TTNN factory concept (`Is able to port?`):** RED. Per-factory verdicts (sheet `Is able to port?`, cross-checked against code):

  | Factory | Concept | Custom hash | Runtime-args upd (code) | Pybind | Safe? | Sheet `Is able?` | Code verdict |
  |---|---|---|---|---|---|---|---|
  | `PadRmReaderWriterMultiCoreDefaultProgramFactory` | descriptor | no | no | no | yes | yes | **clear** |
  | `PadRmReaderWriterMultiCoreProgramFactory` | WorkloadDescriptor (SPMD) | no | no | no | yes | yes | **clear** |
  | `PadRmReaderWriterProgramFactory` | WorkloadDescriptor (SPMD) | no | no | no | yes | yes | **clear** |
  | `PadRmShardedWidthOnlyProgramFactory` | descriptor | no | no | no | yes | yes | **clear** |
  | `PadTileMulticoreProgramFactory` | descriptor | no | no | no | yes | yes | **clear** |
  | `PadRmShardedHeightOnlyProgramFactory` | descriptor | no | **yes** (#48928) | no | yes | yes ⚠ | **GATE (runtime-args upd)** |
  | `PadTileCoreProgramFactory` | descriptor | no | no | no | **no** | no | **GATE (safe=no)** |

  Cross-check confirmed on the cheaply-checkable columns: `Concept` matches (5 `create_descriptor`, 2 `create_workload_descriptor`); `Custom hash` — no `compute_program_hash` override anywhere; `Pybind descriptor` — `pad_nanobind.cpp` binds only `ttnn::pad`, no `create_descriptor`; `Op-owned tensors` — `wd.buffers.push_back(...)` on both workload factories (`pad_rm_reader_writer_multi_core_program_factory.cpp:419`, `pad_rm_reader_writer_program_factory.cpp:200`), which is why they are `WorkloadDescriptor` and satisfies the invariant (op-owned tensors only on `WorkloadDescriptor`). **The one conflict is `Runtime-args update`** — see Result item 3. `Is safe to port?` was **not** re-derived (owner's axis); noted the `PadTileCoreProgramFactory = no` for routing.

- **Device 2.0 (every kernel used):** **GREEN.** Every kernel the op instantiates is structurally Device 2.0 — `Noc` objects with `.async_read`/`.async_write`, `DataflowBuffer`/`dfb.*` wrapper methods (`get_write_ptr`/`get_read_ptr`/`reserve_back`/`push_back`/`wait_front`/`pop_front`), `TensorAccessor`. The only free functions seen are **sanctioned**: `get_tile_size(cb_id_out0)` (`writer_unary_pad_dims_interleaved.cpp:28`) and `get_local_cb_interface(cb_id_in0)` (donor `reader_unary_interleaved_start_id.cpp`). No raw `noc_async_read(`/`noc_async_write(` free calls, no `InterleavedAddrGen`/`ShardedAddrGen`/`get_noc_addr_from_bank_id`, no raw sem addresses (no semaphores at all). Donors: the two `*_v2` kernels call `tt::data_movement::common::noc_async_{read,write}_sharded` via the **Device-2.0 `Noc`-leading overload** passing a `TensorAccessor` (`reader_pad_dims_rm_interleaved_v2.cpp:45`, `writer_pad_dims_rm_interleaved_v2.cpp:48`); the tile-core reader donor is Device 2.0. Nothing to route.

- **Feature compatibility:** every Appendix A entry is **N/A** — a clean scan.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | no `GlobalCircularBuffer`, `remote_cb`, `.remote_index`, `.global_circular_buffer`, `CreateGlobalCircularBuffer` |
  | CBDescriptor `address_offset` (non-zero) | N/A | no `address_offset` / `set_address_offset` / 4-arg `UpdateDynamicCircularBufferAddress` / `cb_descriptor_from_sharded_tensor` anywhere |
  | GlobalSemaphore | N/A | no `GlobalSemaphore` / `CreateGlobalSemaphore` — the op uses **no** semaphores of any kind |
  | Variable-count compile-time arguments (CTA varargs) | N/A | all `get_compile_time_arg_val(N)` at constexpr literal indices; `tensor_args_t` is `PadInputs` (single input + optional preallocated output), not a variable-count container. The `std::vector<Tensor>` hits are only in `create_op_performance_model` signatures. |

- **CB endpoints (GATE-free):** every CB resolves at port time; **no multi-binding and no dead CBs anywhere.** Two structural facts make this simple: (a) the op has **no semaphores**, so the hidden-2nd-writer face (semaphore-gated raw co-fill) cannot exist; (b) every factory instantiates its reader and writer from **different** `kernel_source`s (no same-source dual-instance work-split), so that multi-binding face cannot exist either. Per-factory:
  - **Interleaved/tiled data CBs** (`c_0` main data CB in default-MC, both workload factories, tile-multicore, tile-core): reader FIFO-produces, writer FIFO-consumes → **plain 1:1** (1P+1C), legal, no action.
  - **Pad-value / pad-align scratch CBs** (`c_1`, and `c_2` in default-MC when front-pad/unaligned; `c_2` pad-val in tile-multicore; `c_1` in tile-core): touched (raw `get_write_ptr`/`get_read_ptr`, no cross-kernel FIFO handoff) by a **single** kernel → **self-loop** (bind that one kernel PRODUCER **and** CONSUMER).
  - **Sharded borrowed-memory CBs** (`c_0` input + `c_16` output, `cb.buffer = src/dst`, in height-only and width-only sharded): the borrowed-memory DFB *is* the tensor access (clean; see Tensor bindings). Endpoint-wise the output CB has one FIFO producer (reader) + one role-free raw toucher (writer) → **1P+1C**; the input CB is single-toucher → **self-loop**. (Height-only factory is gated for other reasons; width-only is in the candidate subset.)
  - `c_2` in default-MC is **conditional** (only allocated when `stick_size_padded_front != 0 || unaligned`, `pad_rm_reader_writer_multi_core_default_program_factory.cpp:128`) and is touched by the reader in that config — **not** a dead CB.

- **Offset base pointers:** **GREEN.** No address RTA folds a host-side offset into its base. Every `->address()` occurrence is in `log_debug(...)` or a page-size lookup; the actual base addresses reach kernels either as the **`Buffer*`-binding form** (`emplace_runtime_args({src0_buffer, ...})` / `push_back(src0_buffer)` — framework-patched on cache hits) or via **borrowed-memory CBs** (`cb.buffer = ...`). No `base + offset` device pointer is ever constructed. (Cross-checked against `analyses/2026-07-19_offset_base_pointers.md` conceptually — pad is not an offset-fold op; every base is clean.)

- **TensorAccessor 3rd argument:** **GREEN — Class 2 (drop).** Only two sites pass a 3rd arg, both in the default-MC factory's `*_v2` kernels: `TensorAccessor(src_args, src_addr, accessor_page_size)` (`reader_pad_dims_rm_interleaved_v2.cpp:95`) and `TensorAccessor(dst_args, dst_addr, accessor_page_size)` (`writer_pad_dims_rm_interleaved_v2.cpp:25`). Classification: (1) **interleaved** in the default (unsharded) path — realignment safety net applies; (2) the value is the **true logical stick size** — `input_accessor_page_size = stick_size = W*element_size` and `output_accessor_page_size = stick_size_padded = W_padded*element_size` (interleaved path), or `buffer->aligned_page_size()` (sharded fallback). Correct-magnitude on an interleaved accessor ⇒ inert ⇒ **Class 2, drop the arg** at port time. Not Class 1 (no relaxation / custom hash → each width is its own cache key, so the page size is fixed per compiled program). No Class 3/4/Special. (Cross-checked against `analyses/2026-07-06_tensor_accessor_3rd_arg_triage.md` conceptually.)

## Port-work summary  *(for the candidate subset; carried to a brief only after reconciliation)*

- **Tensor bindings** (per binding):
  - **default-MC** (`PadRmReaderWriterMultiCoreDefaultProgramFactory`): `input` **Case 1** (Buffer*→`TensorAccessor`), `output` **Case 1**.
  - **MC-workload / SC-workload**: `input` **Case 1**, `output` **Case 1**, **`pad_value_const`** **Case 1** (op-owned tensor; base delivered as Buffer*, fed to `TensorAccessor` in `reader/writer_pad_dims_rm_interleaved.cpp`).
  - **width-only-sharded**: `input` + `output` **clean** (borrowed-memory DFB via `cb.buffer`; causal-link gate); pad-value scratch is a plain local CB.
  - **tile-multicore**: `input` **Case 1**, `output` **Case 1**.
  - *(blocked factories, for completeness: **height-only-sharded** input/output **clean** borrowed-DFB; **tile-core** `input` Case 1 via eltwise donor, `output` Case 1.)*
- **TensorParameter relaxation:** none (sheet `none`, no custom hash).
- **TensorAccessor 3rd arg:** drop the redundant `accessor_page_size` arg @ `reader_pad_dims_rm_interleaved_v2.cpp:95` and `writer_pad_dims_rm_interleaved_v2.cpp:25` (default-MC only).
- **CB endpoints:** self-loop the pad-value/scratch CBs; 1P+1C the reader→writer data CBs and the sharded borrowed output CB; no multi-binding flag, no dead-CB drop.

## Heads-ups

- **CB endpoints (multi-binding shapes to watch):** none — no semaphores (no hidden-2nd-writer face) and no same-source dual-instance split. The porter does **not** need the multi-binding advanced option anywhere.
- **Cross-op / shared kernels:**
  - `PadTileCoreProgramFactory` file-path-instantiates the **cross-family donor** `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` (broadly shared across eltwise/DM ops) — a Metal 2.0 rewrite of that kernel is a **port-together** unit with its other co-borrowers. (This factory is gated on `safe=no` anyway.)
  - The two `*_v2` kernels `#include "ttnn/operations/data_movement/common/kernels/common.hpp"` (in-family shared pool) for `noc_async_{read,write}_sharded` — Device-2.0-native, Shape-1 (`TensorAccessor`) coupling.
- **RTA varargs** (prefer the vararg mechanism, don't name each):
  - `reader_pad_tiled.cpp` (tile-multicore): reads `input_page_shape`/`output_page_shape`/`input_id_per_dim`/`output_id_per_dim` as `num_dims`-length blocks via `get_arg_addr` (`:22-25`), consumed in `for (d < num_dims)` loops (`:46`) — rank-bounded ⇒ **RTA varargs**.
  - `reader_pad_dims_rm_interleaved_v2.cpp` (default-MC): `start_dim_offset` read as a rank-length block via `get_arg_addr(7)` (`:59`) ⇒ **RTA varargs** (the leading args 0-6 are fixed named scalars).
  - *(blocked, for completeness: `reader_pad_dims_rm_sharded.cpp` height-only reads a genuinely variable-count structure — `num_cores_read`, per-core NoC coords, per-core chunk counts, and (start,len) chunk pairs — in nested counted loops (`:38,44`) ⇒ **RTA varargs**.)*

## Team-only

- **Out-of-directory coupling & donor shape:**
  - **Op-level roll-up:** ✓ clean (workable). One cross-family file-path donor + one in-family shared header; both Device 2.0. No ⭐ blockers.
  - **Summary (op kernel → donor):**
    | Consuming factory | Donor | Class | Shape | Status |
    |---|---|---|---|---|
    | `PadTileCoreProgramFactory` (reader) | `eltwise/unary/.../reader_unary_interleaved_start_id.cpp` | cross-family (6) | file-path kernel instantiation; `TensorAccessor(src_args, src_addr)` internal | ✓ Device 2.0 |
    | `*_v2` kernels (default-MC) | `data_movement/common/kernels/common.hpp` | in-family shared (5) | function-call: `noc_async_{read,write}_sharded(Noc, …, TensorAccessor, …)` (Shape 1) | ✓ excellent |
  - **Borrowed kernel files (file-path instantiation):** only `reader_unary_interleaved_start_id.cpp` (owning family: `eltwise/unary`; broadly shared — many eltwise/DM readers instantiate it). All other kernels are pad-owned. Port-together set for that donor spans its co-borrowers.
- **TTNN factory analysis (sheet-derived, evidence-checked):**
  - Op-owned tensors: **yes** — the pad-value const tensor, built once on cache-miss and parked on `WorkloadDescriptor::buffers` (#44565) to defer `~Tensor` device-dealloc, on both `PadRmReaderWriterMultiCoreProgramFactory` (`:416-419`) and `PadRmReaderWriterProgramFactory` (`:197-200`). Forces those two to `WorkloadDescriptor`.
  - Secretly SPMD: **yes** — both workload factories broadcast an **identical** `ProgramDescriptor` across every `tensor_coords` range (`:424-429` / `:205-211`); morally one program. Note: the `programs` vector holds one entry per coord range (can be >1 on a multi-range mesh) but all entries are the same descriptor — SPMD in spirit (a mild deviation from the recipe's literal "single entry" test; recorded so the owner can confirm the SPMD classification holds).
  - Custom hash: no. Custom `override_runtime_arguments`: none (the descriptor factories have no such hook; the only runtime-args-update mechanism is the device-op `get_dynamic_runtime_args`). Pybind `create_descriptor`: no. Runtime-args update: the device-op `get_dynamic_runtime_args` (#48928), material only for the height-sharded factory.

## Misc anomalies  *(team-only, non-gating)*

- **`PadTileCoreProgramFactory` `Is safe to port? = no` is unexplained by the code visible here.** `Smuggled pointer = no` on the sheet, and the factory uses the framework-patched `Buffer*`-binding form for both `src0_buffer`/`dst_buffer` (`pad_tile_program_factory.cpp:122,124`) — no raw address smuggling. Combined with the mis-attributed `Runtime-args update = yes` on the *same* row (the hook is actually the height-sharded factory's), the whole `PadTileCoreProgramFactory` sheet row looks like it may have inherited the height-sharded factory's gate flags. Flagged for the readiness-sheet owner as part of the reconciliation — not re-derived here.
- **`writer_ct_args` in `PadTileCoreProgramFactory`** carries `{src0_cb_index, src1_cb_index}` (`:99`); confirm both indices are consumed by `writer_unary_pad_dims_interleaved.cpp` (it uses `cb_id_out0`=CTA0 and `dfb_id_out1`=CTA1) — they are, so no dead CTA. Noted only because CB-index CTAs are the usual place a dead-CTA hides.

## Per-DeviceOperation attribution

Single `PadDeviceOperation`; all findings above are attributed per **factory** in the Gate-detail table and Result. No bundling.

## Questions for the user

1. **Is the runtime-args-update gate per-factory or device-op-level here?** The `get_dynamic_runtime_args` hook lives on the shared `PadDeviceOperation` (`pad_device_operation.cpp:56,235`) and returns `{}` for every factory except `PadRmShardedHeightOnlyProgramFactory`. The readiness sheet treats runtime-args-update as a *per-factory* attribute (6 factories `Is able? = yes`), which implies the framework can port the other factories while the device op retains the hook. If that is correct, the 5-factory candidate subset is genuinely portable once the sheet is reconciled; if the hook blocks the *whole* device-op port until the runtime-args-update infra ships (or #48928's height-sharded fast-path is reworked off the hook), there is no independently-portable subset. This determines whether a subset port can be scheduled ahead of the gate lifting. *(Context: `pad_device_operation.cpp:235-249`.)*
2. **Readiness-sheet reconciliation.** The `Runtime-args update` column is inverted for pad (marks `PadTileCoreProgramFactory` yes / `PadRmShardedHeightOnlyProgramFactory` no; code is the reverse), and `PadTileCoreProgramFactory`'s `Is safe to port? = no` is unexplained by visible code. Both need the sheet owner (Diego) to reconcile before a port. Which of the two factories is *actually* meant to be unsafe / runtime-args-updating?

## Recipe notes

- **Device-op-level `get_dynamic_runtime_args` vs. per-factory gate.** The recipe's `Runtime-args update` cross-check says *"grep the factory for `get_dynamic_runtime_args`"*, but this op's hook is on the **DeviceOperation** and is factory-selective via `select_program_factory`. The recipe doesn't say how to attribute a device-op-level, factory-selective runtime-args hook to a single factory row, nor whether such a hook blocks a subset port of the *other* factories. This is the crux of Question 1 and would be worth a sentence in the *Runtime-args update* cross-check bullet. (§ TTNN factory concept prerequisite.)
- **Readiness-sheet conflict that is *localized* to already-blocked factories.** The recipe's sheet-conflict rule (*"stop rather than proceed on data we can't trust"*) and its Code-path-scope subset rule pull in opposite directions when the conflict touches only factories that are *independently* gated for other reasons, leaving a subset whose own rows are internally consistent. I resolved it by RED-ing the op, naming the candidate subset, and withholding the brief pending reconciliation — but the recipe could say explicitly whether a localized sheet conflict still forbids issuing a subset brief. (§ TTNN factory concept prerequisite / Code-path scope.)
- **SPMD "single entry" vs. identical-broadcast.** Both workload factories push one `PerCoordProgram` per mesh-coord range (potentially >1), all identical. The recipe's SPMD test is a literal "single entry in the `programs` vector," which this technically violates on a multi-range mesh while being SPMD in spirit. A note on the identical-broadcast shape would help. (§ TTNN factory concept prerequisite, `Secretly SPMD Workload?` bullet.)
