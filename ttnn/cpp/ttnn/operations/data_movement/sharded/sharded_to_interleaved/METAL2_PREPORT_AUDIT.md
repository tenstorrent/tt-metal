# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/data_movement/sharded/sharded_to_interleaved`

> **Re-audit — now GREEN.** This is the third audit of this op. The first (`074a5166599`, 2026-07-23) was RED on **two** gates; the second (`9b2b2632b5e`, 2026-08-05) was RED on **one**. Both have since landed on `main`:
> - **Device 2.0** on the shared compute kernel — `0fb47949a27` / PR #51179.
> - **Offset base pointer** on the row-major writer — `6abdf94214d` / **PR #51747** *"[Cleanup] Fix offset pointers in I2S and S2I"*, merged 2026-08-05.
>
> **Every gate now clears. `METAL2_PORT_BRIEF.md` is issued alongside this report.**

Single device-operation directory:

- **`ShardedToInterleavedDeviceOperation`** (`device/sharded_to_interleaved_device_operation.{hpp,cpp}`)
  - `ShardedToInterleavedProgramFactory` (`device/sharded_to_interleaved_program_factory.cpp`) — one `descriptor` factory that selects kernels by input layout and dtype-conversion need.

**Kernels exercised** (all file-path-instantiated; the op owns none of its kernels):

| Role | Path | Selected when |
|---|---|---|
| Reader | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp` | always |
| Writer (tiled) | `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded_blocks_interleaved_start_id.cpp` | `input.layout() == TILE` |
| Writer (row-major) | `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp` | `input.layout() == ROW_MAJOR` |
| Compute (copy) | `ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp` | `convert_df` (input dtype ≠ output dtype; TILE only) |

No unreferenced kernel files sit in the op directory (it holds no kernels at all).

**Config matrix** (three reachable shapes, selected by runtime branch inside the single factory — *not* by separate `ProgramFactory`). All three are now clean; the port must build all three:

- **C1 — TILE, no conversion**: reader + tiled writer.
- **C2 — TILE, conversion** (`convert_df`): reader + tiled writer + compute.
- **C3 — ROW_MAJOR** (never converts; a dtype mismatch requires TILE per `validate_inputs:67-71`): reader + RM writer.
  - **C3a** — HEIGHT_SHARDED: per-core column offset is always `0`.
  - **C3b** — WIDTH_SHARDED / BLOCK_SHARDED: per-core column offset is non-zero on every core after the first. *(Formerly the blocked path; cleared by PR #51747.)*

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `metal2_audit.md`.

**Recipe docs:** `4386dc456a1 2026-07-29 docs(metal_2.0): require an explicit opt_level when porting compute kernels`
*(Pinned from the doc checkout at `/localdev/edwinlee/Port_Recipe`, re-fetched this run and unchanged since the previous audit. The `metal_2.0/` doc tree is not present in this op checkout — `/localdev/edwinlee/metal2_audit.md` symlinks into that separate checkout — so the hash pins the guidance, not this repo. The `analyses/` triage docs were available and are cross-checked below.)*

**Code state audited:** every source file below is **byte-identical to `origin/main` @ `2b7bf3396eb` (2026-08-05)**; the only diff in this working tree is this report and the brief. Relative to the previous audit's baseline (`f6a5267fa85`), the *only* change to any audited file is PR #51747's two-hunk edit to the RM writer.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/sharded/sharded_to_interleaved` |
| **Overall** | **GREEN** — all five gates cleared; brief issued |
| **DOps / Factories** | `ShardedToInterleavedDeviceOperation` → `ShardedToInterleavedProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** *(cleared by `0fb47949a27` / PR #51179)* |
| *Prereqs* — Cross-op escapes | Ok — no function-call escapes; all 4 kernels file-path-borrowed (coupling inventoried, FYI) |
| *Feature Support* — overall | **GREEN** (all Appendix A entries N/A) |
| *Feature Support* — Variadic-CTA | N/A (every CTA read at constexpr index 0) |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** (sheet re-fetched this run; cross-check clean, 1 row ↔ 1 factory) |
| *TTNN Readiness* — Concept (current) | `descriptor` |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A (not `WorkloadDescriptor`) |
| *TTNN Readiness* — Is safe to port? | Yes |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No |
| *TTNN Readiness* — `override_runtime_arguments` | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept` |
| *Port work* — Offset base pointer | **none — GREEN** *(cleared by `6abdf94214d` / PR #51747; the offset is now split out and consumed as a per-write destination `offset_bytes`)* |
| *Port work* — Tensor bindings (per binding) | `input_tensor` = clean (borrowed-memory DFB) · `output_tensor` = **Case 1** in all three configs |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | none (no 3rd-arg site anywhere) |
| *Port work* — CB endpoints | all legal 1P+1C, every CB in every config |

**CB endpoints** are dispositions, not gates. Here none is out of window — no self-loop, no 1P+1C assignment, no multi-binding flag, no dead-CB drop. See [Gate detail](#gate-detail).

## Result

**GREEN → brief issued** (`METAL2_PORT_BRIEF.md`, beside this file).

All five gates clear, and the two that previously blocked this op are resolved **on `main`**, not merely in flight:

- **Device 2.0 — cleared.** `ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp`, the first audit's blocker (four `cb_*` free-function FIFO calls with no wrapper in scope), was migrated on the Device 2.0 track by `0fb47949a27` (PR #51179). All four kernels the op exercises are now Device 2.0-native.
- **Offset base pointer — cleared.** The second audit's blocker was the RM writer feeding `dst_addr + input_width_offset_bytes` as a `TensorAccessor` **base** — a Type-2 accessor-fed offset with no Metal 2.0 seam. `6abdf94214d` (PR #51747, *"[Cleanup] Fix offset pointers in I2S and S2I"*) resolved it exactly as this audit's predecessor proposed and routed:

  ```diff
  -    const auto s0 = TensorAccessor(dst_args, dst_addr + input_width_offset_bytes);
  +    const auto s0 = TensorAccessor(dst_args, dst_addr);
  ...
  -            dfb_out, s0, block_width_bytes, {.offset_bytes = cb_read_offset}, {.page_id = stick_id, .offset_bytes = 0});
  +            {.page_id = stick_id, .offset_bytes = input_width_offset_bytes});
  ```

  The accessor now takes a **clean base**, and the per-core column shift rides each write as a relocatable destination `offset_bytes`. That is precisely the shape the [Offset base pointers] four-outcomes table calls already-split-out — and, critically, it **survives the port**: `TensorAccessor(tensor::out)` supplies the base, while `input_width_offset_bytes` continues as an ordinary named RTA feeding the unchanged `noc.async_write` destination args. No information is lost and no kernel logic changes. The `output_tensor` binding drops to a plain **Case 1** in every config, including C3.

  The fix was done on the **ops team's own track, ahead of the port** — the correct sequencing, and the reason the port diff can now be a pure syntax swap. It landed as a squashed commit (`6abdf94214d`), not the branch commit the previous audit cited (`0a40dce7acb`), which is why an ancestor check against the branch SHA reports "unmerged"; the content is identical plus the explanatory comment.

**Scope note.** All three configs live inside one `ProgramDescriptor` factory (kernel choice is a runtime branch on `input.layout()`, `program_factory.cpp:177-185`), so the porting unit is the whole factory and all four kernels. That was the reason no partial port was offered while C3 was blocked; now it simply means the port covers everything in one change. No code-path scoping applies.

**Nothing in the previous audits' team-only or informational findings has changed** — the factory, the reader, the tiled writer and the compute kernel are byte-identical to the previous baseline, so the CB census, tensor-binding inventory, RTA-vararg verdict, coupling inventory and misc anomalies below all carry forward re-verified rather than re-derived.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN.** Live readiness sheet (*"Operations analysis"*, `dgomez@`), re-fetched this run and **byte-identical to the previous run's copy**. Row `data_movement/sharded/sharded_to_interleaved`:

  | Column | Value |
  |---|---|
  | `Device operation` | `ShardedToInterleavedDeviceOperation` |
  | `Factory (variant)` | `ShardedToInterleavedProgramFactory` |
  | `Concept` | `descriptor` |
  | `Porting Target` | `ProgramSpecFactoryConcept` |
  | `Custom hash (compute_program_hash)` | `no` |
  | `Backdoor custom hash (attribute_values / to_hash)` | `no` |
  | `Runtime-args update (get_dynamic_runtime_args)` | `no` |
  | `Override runtime args method? (PD only)` | `no` |
  | `Pybind descriptor (nb::class_ of device op)` | `no` |
  | `Smuggled pointer (raw buffer addr in RTA/CRTA)` | `no` |
  | `Is safe to port?` | `yes` |
  | **`Is able to port?`** | **`yes`** |
  | `TensorParameter relaxation` | `none` |
  | `Op-owned tensors?` | *(blank)* |
  | `Secretly SPMD Workload?` | *(blank — N/A, not `WorkloadDescriptor`)* |
  | `Op Classification` | `PD Op (pointer-patching)` |
  | `Pointer patching perf issue?` | `OK` · `Formerly custom hashed?` `no` |

  Cross-check against code — all confirmed:
  - `Concept = descriptor` ✓ — `create_descriptor()` returns `tt::tt_metal::ProgramDescriptor` (`device/sharded_to_interleaved_program_factory.hpp:15`).
  - `Custom hash = no` ✓ — no `compute_program_hash` override anywhere in the op directory.
  - `Runtime-args update (get_dynamic_runtime_args) = no` ✓ — no such hook on the device-op (`sharded_to_interleaved_device_operation.hpp:22-31` declares only `validate_inputs`, `validate_on_program_cache_miss`, `compute_output_specs`, `create_output_tensors`, `create_op_performance_model`).
  - `Override runtime args method? = no` ✓ — no `override_runtime_arguments`; the concept is `descriptor`, so the legacy-signature reading does not apply either.
  - `Pybind descriptor = no` ✓ — `sharded_to_interleaved_nanobind.cpp:46-53` binds only the `sharded_to_interleaved` free function via `ttnn::bind_function`; no `nb::class_` of the device op, no `create_descriptor` binding.
  - `Op-owned tensors? = (blank/no)` ✓ — consistent with the `descriptor` concept (the cross-column invariant holds: a `descriptor` row cannot carry op-owned tensors).
  - **Factory-set match** ✓ — the sheet carries exactly **one** row for this op, and the code has exactly **one** factory (`program_factory_t = std::variant<ShardedToInterleavedProgramFactory>`, `device_operation.hpp:20`). No phantom row, no missing row.
  - Cross-column invariants hold. No spreadsheet conflict.

  **Target concept: `ProgramSpecFactoryConcept`** (no op-owned tensors) — the sheet's `Porting Target` column agrees. *(The first audit recorded `MetalV2FactoryConcept`; the current recipe's [TTNN porting shape] names `ProgramSpecFactoryConcept`.)*

- **Device 2.0 (every kernel used):** **GREEN — all four kernels compliant.** No violation table; nothing to route.

  | Kernel | Device 2.0 evidence |
  |---|---|
  | `reader_unary_sharded.cpp` | `DataflowBuffer dfb(cb_id_in0); dfb.push_back(...)` (`:15-16`). |
  | `writer_unary_sharded_blocks_interleaved_start_id.cpp` | `Noc noc; DataflowBuffer dfb_out(cb_id_out)` (`:30-31`), `noc.async_write(dfb_out, s, …)` (`:41`), `dfb_out.wait_front` / `pop_front` (`:36`, `:49`), `TensorAccessor` (`:28`). `get_tile_size(cb_id_out)` (`:26`) is a **sanctioned** free function — not a violation. |
  | `writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp` | `Noc noc; DataflowBuffer dfb_out(dfb_id_out0)` (`:27-28`), `noc.async_write(...)` (`:34-39`), `dfb_out.wait_front` / `pop_front` (`:31`, `:44`). PR #51747 touched only the accessor base and the destination args — the Device 2.0 idioms are unchanged. |
  | `eltwise_copy.cpp` | `CircularBuffer cb_in(tt::CBIndex::c_0)` / `cb_out(tt::CBIndex::c_16)` (`:19-20`) with method-form FIFO ops (`:26-27`, `:34-35`). `#include "api/dataflow/circular_buffer.h"` (`:10`) is the Device-2.0 header the migration guide's own migrated example uses. |

  Re-verified absent across all four kernels: `InterleavedAddrGen`, `ShardedAddrGen`, `InterleavedAddrGenFast`, `InterleavedPow2AddrGen*`, raw `noc_async_read(` / `noc_async_write(`, raw semaphore addresses, the `cb_wait_front(` / `cb_push_back(` / `cb_pop_front(` / `cb_reserve_back(` free-function form, and any `get_write_ptr` / `get_read_ptr` / `get_local_cb_interface` / `fifo_*_ptr` / `evil_set_*` access. The remaining CB-index free functions in `eltwise_copy.cpp` — `unary_op_init_common(c_0, c_16)`, `copy_tile_init(c_0)`, `copy_tile(c_0, …)`, `pack_tile(0, c_16)` — are **compute LLK**, outside the Device 2.0 *data-movement* API surface the migration guide covers, and are not holdovers.

- **Feature compatibility:** every Appendix A entry re-scanned against host and kernel code; all **N/A** (each entry is a gate-feature, so an absent one is N/A, not GREEN).

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type, no `.global_circular_buffer` field on the `CBDescriptor`, no `remote_cb*` / `.remote_index(` / `remote_circular_buffer.h`, no 4-arg `experimental::CreateCircularBuffer`. The input CB **is** Buffer-backed (`cb.buffer = bound_buffer`, `program_factory.cpp:41`, from `:147`) — the legacy **borrowed-memory** pattern, a mechanical `DataflowBufferSpec::borrowed_from` translation, explicitly *not* an Appendix A entry and *not* a GCB. |
  | CBDescriptor `address_offset` (non-zero) | N/A | `push_s2i_cb_pair` (`program_factory.cpp:25-43`) sets `total_size`, `core_ranges`, one `CBFormatDescriptor`, and `buffer` — it never touches `.address_offset` (defaults `0`). No `set_address_offset`, no 4-arg `UpdateDynamicCircularBufferAddress`, no `cb_descriptor_from_sharded_tensor` call anywhere in the op or its kernels. |
  | GlobalSemaphore | N/A | The op uses **no semaphores of any kind** — no `GlobalSemaphore`, no `CreateSemaphore`, no `global_semaphore.hpp`, no kernel-side semaphore wait/post. |
  | Variable-count compile-time arguments (CTA varargs) | N/A | Every kernel-side CTA read is at **constexpr index 0** and nothing else: reader `:13`, tiled writer `:22`, RM writer `:19`, compute `:13`. Host-side CTA lists are fixed-shape (`program_factory.cpp:168`, `:175-176`, `:195`); the writer's variable-length tail is `TensorAccessorArgs(*dst_buffer).append_to(...)` (`:176`), read kernel-side as the fixed `TensorAccessorArgs<1>()` NTTP, not a runtime-varying index. `tensor_args_t` is a fixed pair — `Tensor input_tensor` + `std::optional<Tensor> preallocated_output` (`device_operation_types.hpp:19-22`) — with no `std::vector<Tensor>`, so the op-level cue does not fire either. |

- **Offset base pointers:** **GREEN — no fold anywhere; the one former fold has been split out.**

  **Every address argument resolved.** The output buffer reaches both writers as a **`Buffer*` binding**, not a smuggled `->address()`: `writer_rt.push_back(dst_buffer)` (`program_factory.cpp:242` tiled, `:293` RM) into a `KernelDescriptor::RTArgList`, whose element type is `std::variant<uint32_t, Buffer*, std::reference_wrapper<const MeshTensor>>` (`tt_metal/api/tt-metalium/program_descriptors.hpp:186`). The framework auto-registers it as a `BufferBinding` and patches it on cache hits — consistent with the sheet's `Smuggled pointer = no`. The kernel receives a raw `uint32_t` base at arg 0. Both consumption sites are now clean:

  - **Tiled writer (C1/C2) — clean base.** `TensorAccessor(dst_args, dst_addr)` (`writer_unary_sharded_blocks_interleaved_start_id.cpp:28`), no arithmetic. All addressing is by **tile index**: `start_id = start_id_base + start_id_offset` (`:20`, from RTAs 7/8 = host `curr_idx_h + curr_idx_w` and `starting_idx_h`, `program_factory.cpp:249-250`) feeds `{.page_id = tile_id}` (`:41`). Page indices, not byte addresses — no fold, and never was one.
  - **RM writer (C3) — clean base, offset relocated.** `TensorAccessor(dst_args, dst_addr)` (`writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp:25`). `input_width_offset_bytes` (arg 5 = host `curr_idx_w`, `program_factory.cpp:298`) no longer touches the accessor base; it is consumed per write as `{.page_id = stick_id, .offset_bytes = input_width_offset_bytes}` (`:38`). The host side is unchanged by PR #51747 — it always passed a clean base plus a separate scalar offset — so this is now the textbook **"no fold, base reaches the accessor unmodified"** case: the offset is a relocatable trailing term the port carries through untouched as a named RTA.

  Offset magnitude, for the record (unchanged; explains why this mattered): `curr_idx_w` advances by `output_unit_size` per core and wraps at `num_units_per_row` (`:302-306`). **C3a HEIGHT_SHARDED** — `shard_spec.shape[1]` spans the full row, so it wraps to `0` every iteration → always `0`. **C3b WIDTH/BLOCK_SHARDED** — shard width < row width → non-zero on every core after the first, and reachable (`validate_inputs` admits row-major input whenever the shard page size is L1-aligned, `device_operation.cpp:61-66`). That is the case a mechanical Case-1 port would have silently mis-addressed before the fix.

  **Triage-doc cross-check** (`analyses/2026-07-19_offset_base_pointers.md`, dated 2026-07-19 — a dated prior, not an authority): `sharded_to_interleaved` appears in **none** of the four type tables; the Type-2 table lists only `slice`, `padded_slice` and `slice_write`. Outcome for this run: **"no fold, op not in the tables" → clean.** *(The previous audit reached "fold present, op not in the tables" via a kernel-side reading of the same site and gated it; that reading was borne out — PR #51747 fixed exactly it. See [Recipe notes](#recipe-notes): the recognition rule still does not describe this shape, and the doc's op tables could usefully record this op as a resolved former Type-2.)*

- **TensorAccessor 3rd argument:** **GREEN / none.** Both `TensorAccessor` constructions in the op's kernels are **2-argument** — `TensorAccessor(dst_args, dst_addr)` at tiled writer `:28` and RM writer `:25`. No explicit page-size third argument at any site, so the syntactic signal never fires and there is nothing to classify. Cross-checked against `analyses/2026-07-06_tensor_accessor_3rd_arg_triage.md` (a dated prior): the op is **not** in its op→class table — consistent with the code.

- **CB endpoints (GATE-free):** **all legal 1 producer + 1 consumer**, every CB on every node in every config. No self-loop, no 1P+1C assignment needed, no multi-binding flag, no dead CB. Device 2.0 idioms are intact across all four kernels, so the precondition for this scan holds and no deferral applies.

  Two CBs exist. `c_0` (`src0_cb_index`) is the borrowed-memory input CB (`cb.buffer = src_buffer`, `program_factory.cpp:41` from `:147`), allocated in every config. `c_16` (`out_cb_index`) is allocated **only** when `convert_df` (`:149-160`, `bound_buffer = nullptr`); when `!convert_df`, `out_cb_index == src0_cb_index == c_0` (`:129`), so the writer's DFB *is* `c_0`.

  | CB | Config | Producer (locked) | Consumer (locked) | Census | Verdict |
  |---|---|---|---|---|---|
  | `c_0` (borrowed-memory) | **C1** (TILE, no convert) | reader `dfb.push_back` (`reader_unary_sharded.cpp:16`) | tiled writer `dfb_out.wait_front` / `pop_front` (`writer_unary_sharded_blocks…:36,49`); its `noc.async_write(dfb_out, …)` (`:41`) is a peek on the same binding, not a second endpoint | 2 touchers: 1 locked P + 1 locked C | **plain 1:1 legal** |
  | `c_0` (borrowed-memory) | **C2** (TILE, convert) | reader `dfb.push_back` | compute `cb_in.wait_front` / `pop_front` (`eltwise_copy.cpp:26,34`) | 2 touchers: 1 P + 1 C | **plain 1:1 legal** |
  | `c_16` | **C2** only | compute `cb_out.reserve_back` / `push_back` (`eltwise_copy.cpp:27,35`) | tiled writer `dfb_out.wait_front` / `pop_front` | 2 touchers: 1 P + 1 C | **plain 1:1 legal** |
  | `c_0` (borrowed-memory) | **C3** (ROW_MAJOR) | reader `dfb.push_back` | RM writer `dfb_out.wait_front` / `pop_front` (`writer_unary_stick_layout…:31,44`) | 2 touchers: 1 P + 1 C | **plain 1:1 legal** |

  **Hidden-second-writer hunt: negative, positively.** All four kernels were scanned for a raw co-fill or co-read — `get_write_ptr` / `get_read_ptr` / `get_local_cb_interface(…).fifo_wr_ptr` / `fifo_rd_ptr` / `evil_set_write_ptr` / `evil_set_read_ptr` — with **zero** occurrences. The op allocates **no semaphores at all**, so face (a)'s semaphore-gated co-fill has no coordinating primitive available to it. No dual-instance work-split: each kernel source is pushed into exactly **one** `KernelDescriptor` (`:310-314`), never two configs over one core range. Face (b) multiple-readers does not fire either — each CB has exactly one reading kernel per config.

  **No dead CB.** `c_0`'s index reaches the reader as CTA 0 (`:168`) and the writer as CTA 0 (`:175`, when `!convert_df`); `c_16`'s reaches the writer as CTA 0 when `convert_df` (`:150`, `:175`). Both are consumed by real FIFO ops in every config in which they are allocated.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding):
  - **`input_tensor`** (`c_0`, borrowed-memory) — **clean**, via the causal-link gate. The CB is Buffer-backed (`cb.buffer = src_buffer`); the reader only `push_back`s the already-resident shard pages and constructs no `TensorAccessor`; downstream kernels read it through FIFO ops and as an `async_write` L1 source. The borrowed-memory DFB *is* the tensor access. Port via `DataflowBufferSpec::borrowed_from`. Neither Case 1 nor Case 2 — no work item beyond the `borrowed_from` wiring.
  - **`output_tensor`** (`dst_buffer`, delivered as a `Buffer*` binding → framework `BufferBinding`) — **Case 1 in all three configs.** Both writers feed the base into a `TensorAccessor` and address exclusively through it. Express as a `TensorParameter` / `TensorBinding`; each writer builds `TensorAccessor(tensor::out)`; the arg-0 base *and* the `TensorAccessorArgs` CTA plumbing (`program_factory.cpp:176`) both disappear. Mechanical, low-risk. **The former per-config split is gone** — C3 is no longer an exception (PR #51747).
- **TensorParameter relaxation:** **none.** Sheet says `none`; the op has no custom hash, so there is no hash logic to reconcile.
- **TensorAccessor 3rd arg:** **none** — no site passes one.
- **CB endpoints:** all legal 1P+1C. Nothing to self-loop, assign, flag, or drop.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** **none.** All three faces came back negative with positive evidence (no raw pointer access in any kernel, no semaphores at all, no dual-instance work-split, one reader per CB per config).
- **Cross-op / shared kernels:** **all four kernels are borrowed** — the op owns none, so the port creates four `_metal2` forks (rung 2). **No `_metal2` sibling exists beside any of the four originals** (checked locationally). One wrinkle: a real, non-quasar Metal 2.0 fork of the reader *does* exist at `ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/dataflow/reader_unary_sharded_metal2.cpp` (created by `cbde3d44ff3`, PR #51397, on `main`) — but in **typecast's own tree**, not beside the original, so rung 1's locational check reports "no fork." See [Questions](#questions-for-the-user); this is a convention call for the user, not the porter's to improvise. Full co-borrower / sunset inventory in [Team-only](#team-only).
- **RTA varargs:** **none.** Every kernel reads its runtime args at **distinct constant indices** — reader 0; tiled writer 0–8; RM writer 0, 2, 3, 4, 5, 6 — with no counted loop, no running `arg_index++`, no data-selected index. Every arg is nameable; this is the preferred non-signal case.

## Team-only

- **Out-of-directory coupling & donor shape:** roll-up **✓ clean** for *function-call* escapes. No kernel `#include`s another op's helper: the four kernels' includes are `api/dataflow/dataflow_api.h`, `api/dataflow/noc.h`, `api/dataflow/dataflow_buffer.h`, `api/dataflow/circular_buffer.h`, `api/tensor/noc_traits.h`, `api/compute/{common,tile_move_copy,eltwise_unary/eltwise_unary}.h`, `api/debug/dprint.h` — all `tt_metal/*` LLK/HAL (donor class 1, no concern). No per-call shape analysis is owed. Host-side, the factory `#include`s the in-family `sharded_common.hpp` for `calculate_starting_idx_h` (`program_factory.cpp:11`, `:206`) — host code, no kernel-token bridging, out of this subject's scope.

  **File-path kernel instantiation is the whole coupling story: the op owns none of its kernels.** Consumer sets below are a **sunset and coordination list — not authorization to convert any of these files in place.** Census by filename grep over `ttnn/cpp`, hits filtered to factory bindings, quasar copies excluded:

  | Borrowed kernel | Owning family / pool | Class | Sibling `_metal2` fork? | Co-binding ops (sunset list) |
  |---|---|---|---|---|
  | `reader_unary_sharded.cpp` | `eltwise/unary` | cross-family | **No** (but see the non-sibling typecast fork, above) | broadly shared — `sharded_to_interleaved_partial`, `tilize` (×2), `transpose_wh_sharded`, `untilize` (×3), `untilize_with_unpadding`, `slice_write` (×2) |
  | `writer_unary_sharded_blocks_interleaved_start_id.cpp` | `data_movement/sharded` | in-family | **No** — this port creates it | `sharded_to_interleaved_partial` |
  | `writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp` | `data_movement/sharded` | in-family | **No** — this port creates it | `sharded_to_interleaved_partial` |
  | `eltwise_copy.cpp` | `ttnn/cpp/ttnn/kernel/compute/` (shared pool) | shared-lib | **No** — this port creates it | `copy` (×2: default-tilized, same-memory-config), `interleaved_to_sharded`, `sharded_to_interleaved_partial`, `interleaved_to_sharded_partial`, `untilize_with_unpadding` |

  Sibling-fork check run **locationally** (`ls` of each original's directory), per the shared-kernel caution. Copies under `experimental/quasar/**` also bind same-named kernels; those are whole-op pre-port copies, do **not** count as forks to reuse, and are excluded above. **Porter warning worth carrying forward: `ttnn/cpp/ttnn/operations/experimental/quasar/sharded_to_interleaved/` is a hacky pre-port copy of this exact op.** It will look like a finished answer to every question this port raises; it is not one, and it carries idioms the port recipe forbids. Do not read it, template from it, or lift its binding names.

  Note `sharded_to_interleaved_partial` binds **all four** of this op's kernels — it is the single largest co-borrower and the natural next port after this one. The Device 2.0 migration of `eltwise_copy.cpp` (`0fb47949a27`) landed as one shared rewrite and equally unblocked `copy`, `interleaved_to_sharded`, `untilize_with_unpadding` and both `*_partial` ops.

- **Relaxation candidates (mined from a custom hash):** **none** — the op has no custom hash, so there is nothing to mine.

- **TTNN factory analysis (sheet-derived + `file:line`):** current concept `descriptor` (`program_factory.hpp:15`); target `ProgramSpecFactoryConcept`; **no** op-owned tensors (no `WorkloadDescriptor`, no `buffers` vector); **no** pybind `create_descriptor` and no other risky pybind (`sharded_to_interleaved_nanobind.cpp` exposes only the free function); **no** custom hash; **no** `get_dynamic_runtime_args`; **no** `override_runtime_arguments`; `Is safe to port? = yes`. Every gate conjunct absent → the TTNN gate clears cleanly. One factory, one sheet row, no MeshWorkload need.

## Misc anomalies  *(team-only, non-gating; the port does not act on these)*

All four carry forward from the previous audit and were re-verified against the current tree; PR #51747 did not touch any of them.

- **Dead RTA on the row-major path.** The factory pushes **7** writer RTAs for C3 — index 1 is `num_units_per_row` (`program_factory.cpp:294`) — but `writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp` reads indices 0, 2, 3, 4, 5, 6 and **never index 1** (`:12-17`). Dead plumbing on this path. *(Mildly relevant to the port: the named-RTA conversion will make the gap visible, since the schema names only the args the kernel reads. Dropping it is still not the porter's call — it is a behavior-neutral cleanup for the ops team.)*
- **`is_l1_aligned` is a hardcoded `true`** (`program_factory.cpp:55`), which makes the RM-path guard `if (is_blackhole or is_l1_aligned) { if (!dst_is_dram or is_l1_aligned) { … } }` (`:286-289`) unconditionally taken. Three consequences: `is_blackhole` (`:135`) and `dst_is_dram` (`:134`) are computed but effectively dead in this branch (`dst_is_dram` has no other use), and the first `padded_shard_width = tt::align(output_unit_size, dst_buffer->alignment())` (`:285`) is always overwritten at `:288`. A forced constant hiding a dead branch — worth a deliberate decision rather than leaving it as-is.
- **`num_slices` / `slice_index` are vestigial here.** The launch site hardcodes them to `1` / `0` (`sharded_to_interleaved_device_operation.cpp:147`, `ShardedToInterleavedParams{…, 1, 0}`), and `calculate_starting_idx_h` early-returns `0` when `num_slices <= 1` (`sharded_common.cpp:17-19`). So `starting_idx_h` — the tiled writer's arg 8 / `start_id_base` — is **always 0** for this op. The real user of the slicing parameters is the separate `sharded_to_interleaved_partial` op. Not a bug, but dead generality carried into hash-relevant attributes (both fields sit on `ShardedToInterleavedParams`).
- **The TILE/ROW_MAJOR decision is taken off two different tensors.** The unit-size and core-count blocks branch on `output.layout()` (`program_factory.cpp:81`, `:113`) while kernel selection and the per-core RTA loop branch on `input.layout()` (`:177`, `:213`, `:214`). They agree in practice — `compute_output_specs` builds the output with `PageConfig(input_tensor.layout())` (`device_operation.cpp:113`), and a preallocated output must match the input's layout (`:48-50`) — but the split reads as accidental and would diverge silently if either invariant were relaxed.
- **Stray debug include in the borrowed reader.** `reader_unary_sharded.cpp:9` includes `api/debug/dprint.h` with no `DPRINT` use in the file. Cosmetic, and it belongs to `eltwise/unary` — not this op's to fix.

## Per-DeviceOperation attribution

Not applicable — the directory holds exactly one `DeviceOperation` with exactly one program factory, and no finding now differs by config (the last per-config split, on the `output_tensor` binding, closed with PR #51747).

## Questions for the user

1. **Which fork does the reader bind?** `reader_unary_sharded_metal2.cpp` exists on `main` in **typecast's** directory rather than beside the original in `eltwise/unary/…/dataflow/`. Rung 1's *locational* check therefore reports "no fork," and rung 2 would have this port create a **second** fork of the same kernel. Options: **(a)** bind typecast's existing fork despite the non-sibling path (rung 1 by intent — check its `dfb::in` / `args::num_tiles_per_core` shape fits, which it appears to); **(b)** create the sibling fork per the letter of rung 2 and accept two forks in the tree; **(c)** relocate typecast's fork beside the original first, on the ops/porting track, so rung 1 works for every later consumer. The brief carries this as an open decision rather than a porter judgement call — worth settling **before** the port starts, since it determines one of the four kernel forks.
2. **Misc anomalies routing.** The `is_l1_aligned = true` forced constant and the dead RM arg 1 are pre-existing and non-gating, but the forced constant makes a real branch unreachable, and the named-RTA conversion will make the dead arg conspicuous. File them against the ops team now, or carry as-is?
3. **Next port in the family.** `sharded_to_interleaved_partial` binds **all four** of the same kernels and is `Is able to port? = yes` on the sheet with the same `descriptor` / `ProgramSpecFactoryConcept` shape. Sequencing it right after this one would let it reuse all four forks this port creates, at rung 1. Worth queueing together?

## Recipe notes  *(friction with the audit recipe itself)*

- **The offset-base recognition-rule gap is now demonstrated, not just argued — and is still open.** [Offset base pointers] resolves each address RTA "to its **host** computation," and its four-outcomes table classifies *clean base + separate scalar offset arg, summed in the kernel* as GREEN → hand to [TensorParameter analysis]. Two audits ago this op matched that description on paper while being a genuine Type-2 wall, because the kernel's sum landed on a **`TensorAccessor` base**. That reading has now been vindicated by the ops team: **PR #51747 fixed exactly the site the rule as written would have waved through**, and its own commit message states the diagnosis — *"These ops use offset base pointers … instead of keeping the actual base address and using the offset parameter. This blocks Metal 2.0 porting."* An auditor following the rule literally would have GREENed the op, and the port would have silently dropped the column offset on row-major width/block-sharded inputs. Two concrete edits close it: (1) add a kernel-side clause to Type-2 recognition — *"a base RTA plus a separately-delivered offset that are **summed and passed as a `TensorAccessor` base** is Type 2, wherever the sum is computed"*; (2) qualify the four-outcomes "No fold → clean" bullet with *"provided the base reaches the accessor unmodified."* The `roll` DRAM_RM precedent cited as the GREEN case should also state explicitly that its split-out offset is **raw-consumed**, which is what makes it green.
- **The offset triage doc could record resolved sites, not just open ones.** `analyses/2026-07-19_offset_base_pointers.md` never listed `sharded_to_interleaved`, and now the site is fixed — so a future auditor gets no signal in either direction from the doc. A short "resolved / formerly Type N" section (this op + the `interleaved_to_sharded` half of PR #51747) would let the doc's *"no fold, op in the tables"* staleness outcome actually fire, which is the one outcome that currently has no way to be reached for an unlisted op.
- **Readiness-sheet column names have drifted from the docs.** The live sheet's header is **`Override runtime args method?\n(PD only)`**, while both `ttnn_op_porting_readiness.md` and `metal2_audit.md` quote it as `Override runtime args method? (PD and legacy)`. The readiness doc's standing guarantee is *"existing column names never change, and no column is ever deleted"* — so a lookup keyed on the documented string finds nothing. It only worked here because I read the header row and matched by prefix. Worth reconciling in whichever direction is correct.
- **The sheet carries gate-adjacent columns the docs don't mention.** Beyond the documented set: `Op Classification`, `Execution Model`, **`Porting Target`**, **`Backdoor custom hash (attribute_values / to_hash)`**, `Known op issues`, `Pointer patching perf issue?`, `Formerly custom hashed?`. Two matter beyond the informational: **`Porting Target`** supplies directly the target concept that [TTNN porting shape] has the auditor derive by hand from `Concept` + `Op-owned tensors?` (they agreed here — `ProgramSpecFactoryConcept` — but the recipe should say which is authoritative); and **`Backdoor custom hash`** looks like a fifth custom-hash-shaped signal absent from the documented `Is able to port?` derivation, so an auditor cross-checking that derivation cannot tell whether it is a conjunct, a subsumed input, or informational. Both were benign here.
- **The rung-1 fork check is locational, but a real fork can be non-local — and the audit is where that gets caught.** [Caution: Porting a shared kernel] rung 1 checks for a **sibling** `_metal2` file and warns off tree-wide greps because they surface quasar copies. Correct as far as it goes, but it has a blind spot the quasar clause doesn't cover: a *legitimate, non-quasar* fork placed in the porting op's **own** tree instead of beside the original — `copy/typecast/device/kernels/dataflow/reader_unary_sharded_metal2.cpp` (PR #51397, on `main`) is exactly that. Rung 1 reports "no fork"; rung 2 then produces a second fork of one kernel, the duplication the convention exists to prevent. Suggest [Out-of-directory coupling]'s borrowed-kernel-file bullet ask the auditor to record **non-sibling non-quasar `_metal2` forks** as reuse candidates (a filename grep minus `experimental/quasar/**` suffices), since the auditor greps broadly anyway and the porter, working one file at a time, is least likely to find it.
- **Re-audit after a targeted unblock is cheap, and the recipe could say so.** This op has now been audited three times, each after a single narrow fix landed. Runs two and three were dominated by *re-verifying unchanged code* — the delta was two hunks in one kernel, confirmable with one `git diff` against the prior audit's recorded baseline. The **"Code state audited"** provenance line (this report records the exact `origin/main` SHA every audited file matched) is what made that cheap, and it isn't in the recipe's report template. Adding it would let a re-audit legitimately scope itself to the diff plus a full re-run of the affected subjects, instead of re-deriving a whole census to be sure nothing moved. *(Related: the previous run's disclosed deviation — running the informational subjects despite a no-portable-subset RED — paid off exactly as predicted here; that detail carried forward re-verified rather than re-derived.)*
