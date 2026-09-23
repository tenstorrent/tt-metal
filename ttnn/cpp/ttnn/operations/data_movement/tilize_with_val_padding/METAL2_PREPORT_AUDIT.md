# Metal 2.0 Audit Findings — `data_movement/tilize_with_val_padding`

**Audit scope (as requested):** the **`TilizeWithValPaddingMultiCoreBlockInterleavedFactory`** factory only — the one factory in this op that was blocked from Metal 2.0 by the per-region DFB-sizing wall of GitHub issue [#51305](https://github.com/tenstorrent/tt-metal/issues/51305). That issue is now **CLOSED (2026-09-04)** and the factory has been refactored to clear it; this audit verifies the block factory against that refactored code. The op's three sibling factories (`…SingleCore`, `…MultiCoreDefault`, `…MultiCoreSharded`) are **not** audited in depth here (they carry no such blocker); their readiness-sheet status is noted for context only.

- **`TilizeWithValPaddingDeviceOperation`** (`device/tilize_with_val_padding_device_operation.{hpp,cpp}`)
  - **`TilizeWithValPaddingMultiCoreBlockInterleavedFactory`** (`device/factories/tilize_with_val_padding_multi_core_block_interleaved_program_factory.{hpp,cpp}`) — **audited**
  - `TilizeWithValPaddingMultiCoreDefaultFactory` — context only (sheet: able to port = `yes`)
  - `TilizeWithValPaddingMultiCoreShardedFactory` — context only (sheet: able to port = `yes`)
  - `TilizeWithValPaddingSingleCoreFactory` — context only (sheet: able to port = `yes`)

**Kernels the block factory exercises (all audited):**

| Role | File | Owner | In scope why |
|---|---|---|---|
| Reader | `device/kernels/dataflow/reader_unary_pad_multicore_both_dims.cpp` | this op (owned) | `kernel_source` @ factory `.cpp:148-150` |
| Writer | `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id_wh.cpp` | eltwise/unary (cross-family donor) | `kernel_source` @ factory `.cpp:164-165` |
| Compute | `data_movement/tilize/device/kernels/compute/tilize_wh.cpp` | data_movement/tilize (in-family donor) | `kernel_source` @ factory `.cpp:188-189` |

**Shared host helper (the #51305 refactor lives here):** `data_movement/common/common.{hpp,cpp}` — `make_block_plan` / `BlockBufferSet` / `push_buffer_set` / `buffer_set_for_core`.

**Unreferenced by this factory (belong to sibling factories, not audited):** `reader_unary_pad_dims_split_rows.cpp`, `reader_unary_pad_dims_split_rows_multicore.cpp`, `reader_unary_pad_height_width_sharded.cpp`.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `d51708326b5 2026-09-23 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/` |
| **Overall** | **GREEN** (block factory) |
| **DOp / Factory** | `TilizeWithValPaddingDeviceOperation` → `TilizeWithValPaddingMultiCoreBlockInterleavedFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — reader + writer + compute all Device 2.0 compliant |
| *Prereqs* — Cross-op escapes | Ok — shared-lib / donor calls, all workable shapes |
| *Feature Support* — overall | **GREEN** (all Appendix A entries N/A) |
| *Feature Support* — GlobalCircularBuffer / address_offset / GlobalSemaphore | N/A / N/A / N/A |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** |
| *TTNN Readiness* — Concept (current) | `descriptor` |
| *TTNN Readiness* — Secretly SPMD | N/A (descriptor concept; `Secretly SPMD Workload?` cell empty) |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No |
| *TTNN Readiness* — `override_runtime_arguments` | No (block factory declares only `create_descriptor`) |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | **`ProgramSpecFactoryConcept`** (base — no override) |
| *Port work* — Offset base pointer | none (no host-folded offset; clean bases) |
| *Port work* — Tensor bindings (per binding) | input **Case 1** · output **Case 1** |
| *TTNN Readiness* — TensorParameter relaxation | `none` (clears) |
| *Port work* — TensorAccessor 3rd arg | none — no accessor passes a 3rd arg |
| *Port work* — CB endpoints | input c_0/c_2 **1P+1C** · output c_16/c_17 **1P+1C** · staging c_1/c_3 **self-loop** (DM) |

## Result

**GREEN → brief issued.** The `TilizeWithValPaddingMultiCoreBlockInterleavedFactory` clears every gate and is portable to `ProgramSpecFactoryConcept`.

The blocker that RED'd this factory in the prior tilize-family audits — **per-region (per-node) DFB sizing**, issue #51305 — has been resolved by the family-wide refactor (issue's **Option 1**: split into separately-sized buffer sets, each with its own CB indices). The shared helper `make_block_plan` now emits **two** `BlockBufferSet`s — `full` (indices c_0 / c_1 / c_16) and `cliffrow` (indices c_2 / c_3 / c_17) — and `push_buffer_set` sizes each `CBDescriptor` **once** over the set's whole `core_ranges` from a single scalar `block_tiles` (`common.cpp:843-862`). No CB index is ever allocated at two different sizes across nodes, which is exactly what `DataflowBufferSpec` (one `entry_size` + `num_entries` per named DFB, no per-core size field) requires. The readiness sheet reflects this: `Known op issues` is now **empty** for this factory (it previously read `Per-node CB size`).

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN** — the readiness sheet's `Is able to port?` == `yes` for this factory (row: `data_movement/tilize_with_val_padding` / `…MultiCoreBlockInterleavedFactory`). Lightweight cross-check against the code is **clean on every column**:
  - `Concept` == `descriptor` ✓ — `create_descriptor()` returns a `ProgramDescriptor` (factory `.hpp:14-17`); no mesh-workload return, no legacy `create()`+`override`.
  - `Custom hash` == `no` ✓ — no `compute_program_hash` override in `tilize_with_val_padding_device_operation.{hpp,cpp}`.
  - `Runtime-args update (get_dynamic_runtime_args)` == `no` ✓ — no such hook on the device op.
  - `Override runtime args method?` == `no` ✓ — block factory `.hpp` declares only `create_descriptor` (confirmed by the factory `.cpp:289-291` comment: "this factory declares no override_runtime_arguments, so `resolve_bindings` walks every kernel's bindings").
  - `Pybind descriptor` == `no` ✓ — `tilize_with_val_padding_nanobind.cpp` binds only the `tilize_with_val_padding` / `tilize_with_zero_padding` functions (lines 45-63); no `create_descriptor` / device-op `nb::class_`.
  - `Smuggled pointer` == `no` ✓; `TensorParameter relaxation` == `none` ✓; `Op-owned tensors?` empty ✓ (consistent with `descriptor`).
  - **Factory-set match** ✓ — the sheet has exactly 4 non-quasar rows for this op, one per factory in `program_factory_t` (`device_operation.hpp:27-31`). No phantom / missing rows.
  - **Cross-column invariants** ✓ — `get_dynamic_runtime_args=no` on a `descriptor` concept is valid; no op-owned tensors on a `descriptor` concept.
  - *(The `experimental/quasar/tilize_with_val_padding` rows in the sheet are out of bounds and ignored.)*

- **Device 2.0 (every kernel used):** **GREEN.** All three kernels are structurally Device 2.0 — object APIs (`Noc`, `DataflowBuffer`, `TensorAccessor`, `CoreLocalMem`), no legacy addr-gen, no manual CB-index management, no raw-address NoC walks.

  | File | Evidence | Note |
  |---|---|---|
  | `reader_unary_pad_multicore_both_dims.cpp` | `Noc noc;` (`:38`), `DataflowBuffer dfb_in0/dfb_in1` (`:39-40`), `TensorAccessor(src_args, src_addr)` (`:37`), `noc.async_read(...)` (`:65,84`), `CoreLocalMem`/`PrecomposedUnicastEndpoint` | Fully Device 2.0. Builds DFBs from numeric CTAs (`get_compile_time_arg_val(6/7)`, `:28-29`) — the Device-2.0 object-from-index idiom (allowed); the port swaps the CTA for a `dfb::` token. |
  | `writer_unary_interleaved_start_id_wh.cpp` | `Noc noc;` (`:28`), `DataflowBuffer dfb(cb_id_out)` (`:29`), `TensorAccessor(dst_args, dst_addr)` (`:26`), `noc.async_write(dfb, s, …)` (`:43`), `dfb.wait_front/pop_front` (`:42,45`) | Uses `get_tile_size(cb_id_out)` (`:24`) — a **sanctioned** CB-index free function (Green bullet); not a violation. |
  | `tilize_wh.cpp` | `compute_kernel_hw_startup(dfb_id_in, dfb_id_out)` (`:22`), `compute_kernel_lib::tilize<…>()` (`:28-35`), `compute_kernel_lib::is_fp32_input_format<dfb_id_in>()` (`:24`) | Device 2.0 compute-lib idioms; numeric DFB CTAs (`:19-20`). |

- **Feature compatibility:** every Appendix A entry is **N/A** (feature absent) — verified by targeted grep across the factory + `push_buffer_set` + all three kernels.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type, no `.global_circular_buffer` field on any `CBDescriptor`, no `remote_index`/`remote_cb`/`CreateGlobalCircularBuffer`. `push_buffer_set` builds plain `CBDescriptor`s (`common.cpp:833-862`). |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `.address_offset` set anywhere (defaults to 0); no borrowed-memory (`set_globally_allocated_address` / `.buffer=`). |
  | GlobalSemaphore | N/A | No semaphores of any kind in this factory (no `CreateSemaphore`, no `GlobalSemaphore`). |

- **CB endpoints (GATE-free):** every CB is legal 1:1 or a self-loop; no multi-binding flag, no dead CBs. Census is per-node; the input/output CBs each have exactly **one** producer and **one** consumer on any given node.

  | CB (config) | Producer | Consumer | Verdict / disposition |
  |---|---|---|---|
  | c_0 input (`full` set) | reader (`dfb_in0.reserve_back/push_back` + raw write @ reader `:56-118`) | compute (`tilize` helper `wait_front/pop_front`) | **1P+1C** |
  | c_16 output (`full` set) | compute | writer (`dfb.wait_front/pop_front` @ writer `:42,45`) | **1P+1C** |
  | c_1 staging (`full` set) | reader only (`dfb_in1.reserve_back(1)`/`get_write_ptr`/`push_back(1)` @ reader `:42-45`, then raw `temp_addr` scratch) | — | **self-loop** (single toucher; bind reader PRODUCER + CONSUMER) — a **DM self-loop**, legal on Gen1 |
  | c_2 / c_17 / c_3 (`cliffrow` set) | same shapes as c_0 / c_16 / c_1 | | same: **1P+1C** / **1P+1C** / **self-loop** |

  **Multiplicity note (not multi-binding):** the `full` set's I/O CBs (c_0/c_16) are bound by **one** reader instance + **one** writer instance (each over `full.core_ranges`) plus **two** compute instances over the *disjoint* `core_range` and `cliff_col_core_range` sub-ranges (`.cpp:217-232`). Per node this is exactly 1 reader + 1 compute + 1 writer → 1P+1C. This is Preserved Multiplicity (two same-source compute KernelSpecs over disjoint node sets), **not** a multi-binding — the porter lists the shared DFB per compute group with disjoint `target_nodes`. Same for the `cliffrow` set (compute over `cliff_row_core_range` and `cliff_col_row_core_range`).

- **Offset base pointers:** **GREEN** — no address RTA folds a host-side offset into its base. The two device-buffer addresses reach the kernels via the **`Buffer*`-binding form**: `src0_buffer` is pushed as reader RTA slot 0 (`.cpp:292-302`, the pointer object, not `->address()+offset`) and `dst_buffer` as writer RTA slot 0 (`.cpp:305-306`). Each kernel receives a clean base (`get_arg_val<uint32_t>(0)`) and feeds it straight to a `TensorAccessor`; all offsets (`start_column_id`, page ids, etc.) are computed on-device via separate scalar RTAs and the accessor. No Type 1/2 fold; Types 3/4 absent.

- **TensorAccessor 3rd argument:** **N/A** — no accessor in this factory passes a 3rd argument. Both sites are 2-arg: reader `TensorAccessor(src_args, src_addr)` (`:37`), writer `TensorAccessor(dst_args, dst_addr)` (`:26`). The subject never fires.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding):
  - **input** (`src0_buffer`) — **Case 1** (via `TensorAccessor`). `Buffer*`-binding RTA (`.cpp:294`) consumed by the reader as `TensorAccessor(src_args, src_addr)`. Port: express as `TensorParameter` (e.g. `tensor::src`); the reader builds `TensorAccessor(tensor::src)`, and the RTA slot + `TensorAccessorArgs(*src0_buffer)` CTA plumbing (`.cpp:145`) both disappear.
  - **output** (`dst_buffer`) — **Case 1** (via `TensorAccessor`). `Buffer*`-binding RTA (`.cpp:306`) consumed by the writer as `TensorAccessor(dst_args, dst_addr)`. Port: `TensorParameter` (e.g. `tensor::dst`); writer builds `TensorAccessor(tensor::dst)`; RTA slot + `TensorAccessorArgs(*dst_buffer)` CTA (`.cpp:161`) disappear.
- **TensorParameter relaxation:** `none`.
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:** self-loop `c_1`/`c_3` (staging, per set) · 1P+1C `c_0`/`c_2` (input) and `c_16`/`c_17` (output). All configs. No dead CBs, no multi-binding flag.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none — no hidden second writer, no multi-reader. (Staging self-loops and the 1P+1C I/O CBs are in Port-work.)
- **Cross-op / shared kernels (sunset lists — NOT an authorization to convert in place):**
  - `reader_unary_pad_multicore_both_dims.cpp` — **owned by this op**, but also file-path-instantiated by **tilize**'s block factory (`data_movement/tilize/device/tilize_multi_core_block_program_factory.cpp`). **No `_metal2` fork exists.**
  - `writer_unary_interleaved_start_id_wh.cpp` — owned by **eltwise/unary**; also used by tilize's block factory. **No `_metal2` fork** of the `_wh` variant (the `writer_unary_interleaved_start_id_metal2.cpp` sibling is a fork of the *non-`_wh`* kernel — do not reuse it here).
  - `tilize_wh.cpp` — owned by **data_movement/tilize**; also used by tilize's block factory. **No `_metal2` fork** (the `ttnn/cpp/ttnn/kernel/compute/tilize_metal2.cpp` fork is of the *other* tilize compute kernel, not `tilize_wh.cpp`).
  - Co-borrower set for all three = **{tilize_with_val_padding block factory, tilize block factory}**. This port creates the first `_metal2` fork of each beside the original; tilize's (still-legacy) block factory keeps binding the legacy copies until it is ported, at which point the legacy copies can sunset.
- **RTA varargs:** none — reader reads a fixed set (slots 0-8; slots 3-8 re-read at *constant* indices per `third_dim` iteration, not `arg_index++` or data-selected) and writer reads slots 0-3. All nameable.
- **Preserved-multiplicity wiring:** 4 compute KernelSpecs (`.cpp:217-232`) and 2 reader / 2 writer instances (`.cpp:173-176, 323-330`) over disjoint core ranges — list each shared DFB per group with disjoint `target_nodes` (see `metal2_port_gotchas`), and the framework derives placement.
- **Shared writer carries a `BACKWARDS` `#ifdef`** (writer `:31-41`) used by the untilize direction; this factory sets no such define (forward path). The `_metal2` fork must preserve that compile define for the co-borrowers.

## Team-only

- **Out-of-directory coupling & donor shape** — roll-up: **✓ clean** (no ⭐/✗ scheduling blockers).
  - *Summary table (function-call escapes):*

    | Op kernel | Donor include | Class | Shape / verdict |
    |---|---|---|---|
    | reader | `data_movement/common/kernels/common.hpp` (`tt_memmove`) | in-family shared | raw `uint32_t` addr + `Noc` args → ✓ |
    | reader | `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp` (`fill_l1_range`) | official kernel-lib | raw addr + scalars → ✓ |
    | compute | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` (`compute_kernel_lib::tilize`, `is_fp32_input_format`, `compute_kernel_hw_startup`) | official kernel-lib | `uint32_t cb_id` / NTTP CB index → ✓ (`dfb::name` constexpr cast handles runtime + template position) |
    | writer | `tt_metal` `api/*` only | LLK/HAL | ✓ no concern |
  - *File-path kernel instantiation:* the three borrowed kernel files + fork status are in the Heads-ups sunset lists above.
- **Relaxation candidates (from a custom hash):** none — the op has no custom hash.
- **TTNN factory analysis (sheet-derived, with code evidence):** current concept `descriptor`; target `ProgramSpecFactoryConcept`; no op-owned tensors; no genuine multi-program (the sheet's `Execution Model = SPMD` is descriptive — `Concept` is `descriptor`, `Secretly SPMD Workload?` empty); no pybound `create_descriptor`; no custom hash; no `override_runtime_arguments`; no `get_dynamic_runtime_args`. Sibling factories (Default / Sharded / SingleCore) all carry `Is able to port? = yes`, `none` relaxation, empty `Known op issues` — portable on their own audits, not covered here.

## Misc anomalies  *(team-only, non-gating)*

- **Stale comment** in the factory: `unpadded_row_size_bytes` / `padded_row_size_bytes` are annotated `// Assuming bfloat16 dataformat` (`.cpp:82-83`) yet are computed with `a.element_size()`, so they are correct for any dtype. The comment is misleading, not a bug.
- **Redundant per-iteration RTA reads** in the reader: slots 3-8 are re-read inside the `for (dim3)` loop (reader `:125-130`) although their values are constant across iterations. Harmless (constant indices); a micro-cleanup at most.

## Per-DeviceOperation attribution

N/A — a single `DeviceOperation`, single audited factory.

## Questions for the user

None.

## Recipe notes

- **A closed prereq issue + a since-refactored factory is the cleanest possible GREEN path, and the recipe handled it well.** The [Offset base pointers](#) and [TTNN factory concept] subjects' "your own scan is the source of truth; the dated triage / sheet is a prior" framing was exactly right here: the readiness sheet had already been updated (`Known op issues` cleared) to match the #51305 refactor, and the code cross-check confirmed it independently. No friction to report.
- The audit worked entirely from the *refactored* code; the value of #51305 being pointed out in the task prompt was in explaining *why* a previously-RED factory is now clean, which belongs in the Result narrative rather than any gate.
