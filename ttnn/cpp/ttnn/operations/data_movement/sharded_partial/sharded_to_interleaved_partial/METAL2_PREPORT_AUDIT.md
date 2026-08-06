# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/data_movement/sharded_partial/sharded_to_interleaved_partial`

Single device-operation directory:

- **`ShardedToInterleavedPartialDeviceOperation`** (`device/sharded_to_interleaved_partial_device_operation.{hpp,cpp}`)
  - `ShardedToInterleavedPartialProgramFactory` (`device/sharded_to_interleaved_partial_program_factory.cpp`) — one `descriptor` factory.

**Relationship to `data_movement/sharded/sharded_to_interleaved`.** This op's program factory is a **verbatim clone** of that op's: `diff` after normalising the `Partial` / `_partial` identifiers shows only the copyright year and one line wrap. Same helper (`push_s2i_partial_cb_pair`), same CB layout, same four kernels, same `Buffer*` output binding, same RTA vectors. The two ops diverge **only** at the device-operation layer (validation, output handling, and live `num_slices` / `slice_index`). Findings below were derived independently but converge with the sibling audit at `../../sharded/sharded_to_interleaved/METAL2_PREPORT_AUDIT.md`; the differences that matter are called out explicitly.

**Kernels exercised** (all file-path-instantiated; the op owns none of its kernels — the same four as the sibling op):

| Role | Path | Selected when |
|---|---|---|
| Reader | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp` | always |
| Writer (tiled) | `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded_blocks_interleaved_start_id.cpp` | `input.layout() == TILE` — i.e. **always**, see below |
| Writer (row-major) | `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp` | `input.layout() == ROW_MAJOR` — **statically unreachable** |
| Compute (copy) | `ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp` | `convert_df` (cache-tensor dtype ≠ input dtype) |

No unreferenced kernel files sit in the op directory (it holds no kernels at all).

**Config matrix.** Unlike the sibling op, row-major input is **rejected outright**: `validate_on_program_cache_miss` opens with `TT_FATAL(input_tensor.layout() == Layout::TILE, "Currently, only tile layout is supported for partial S->I")` (`device_operation.cpp:24`). Layout is part of the default program hash, so a row-major input is always a cache miss and always hits that assert — the RM branch cannot execute.

- **C1 — TILE, no conversion**: reader + tiled writer. *(Reachable.)*
- **C2 — TILE, conversion** (`convert_df`): reader + tiled writer + compute. *(Reachable.)*
- **C3 — ROW_MAJOR**: reader + RM writer. **Unreachable** — the factory branch (`program_factory.cpp:182-186`, `:259-308`) and its kernel binding are live code the port must still carry, but no test can exercise them. See [Misc anomalies](#misc-anomalies-team-only-non-gating-the-port-does-not-act-on-these) and the brief's Watch-for.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `metal2_audit.md`.

**Recipe docs:** `4386dc456a1 2026-07-29 docs(metal_2.0): require an explicit opt_level when porting compute kernels`
*(Pinned from the doc checkout at `/localdev/edwinlee/Port_Recipe`, re-fetched this run. The `metal_2.0/` doc tree is not present in this op checkout — `/localdev/edwinlee/metal2_audit.md` symlinks into that separate checkout — so the hash pins the guidance, not this repo. The `analyses/` triage docs were available and are cross-checked below.)*

**Code state audited:** every source file below is **byte-identical to `origin/main` @ `2b7bf3396eb` (2026-08-05)**; the working tree carries no local modification to any of them. Note this baseline **includes** `6abdf94214d` (PR #51747), which cleared the offset-base fold in the shared row-major writer — see [Offset base pointers](#gate-detail).

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/sharded_partial/sharded_to_interleaved_partial` |
| **Overall** | **GREEN** — all five gates cleared; brief issued |
| **DOps / Factories** | `ShardedToInterleavedPartialDeviceOperation` → `ShardedToInterleavedPartialProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all four kernels Device 2.0-native |
| *Prereqs* — Cross-op escapes | Ok — no function-call escapes; all 4 kernels file-path-borrowed (coupling inventoried, FYI) |
| *Feature Support* — overall | **GREEN** (all Appendix A entries N/A) |
| *Feature Support* — Variadic-CTA | N/A (every CTA read at constexpr index 0) |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** (sheet fetched this run; cross-check clean, 1 row ↔ 1 factory) |
| *TTNN Readiness* — Concept (current) | `descriptor` |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A (not `WorkloadDescriptor`) |
| *TTNN Readiness* — Is safe to port? | Yes |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No |
| *TTNN Readiness* — `override_runtime_arguments` | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept` |
| *Port work* — Offset base pointer | **none — GREEN** (both writers take a clean base; RM writer cleared by `6abdf94214d` / PR #51747) |
| *Port work* — Tensor bindings (per binding) | `input_tensor` = clean (borrowed-memory DFB) · `cache_tensor` (the output) = **Case 1** |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | none (no 3rd-arg site anywhere) |
| *Port work* — CB endpoints | all legal 1P+1C, every CB in every reachable config |

**CB endpoints** are dispositions, not gates. None is out of window here — no self-loop, no 1P+1C assignment, no multi-binding flag, no dead-CB drop.

## Result

**GREEN → brief issued** (`METAL2_PORT_BRIEF.md`, beside this file).

**Yes — this op is clean, for the same reasons its sibling is, and it inherited the sibling's unblock for free.** The one construct that blocked `sharded_to_interleaved` through two audits — the row-major writer folding a per-core column offset into a `TensorAccessor` base — lived in a **shared kernel** that this op binds too. `6abdf94214d` (PR #51747) fixed that kernel, so this op cleared the same gate at the same moment, without ever being separately blocked by it. Likewise the Device 2.0 straggler `eltwise_copy.cpp`, cleared by `0fb47949a27` (PR #51179), is a kernel both ops share.

Two things make this op *easier* to port than its sibling, and one makes it slightly harder:

- **Easier — the reachable surface is smaller.** Row-major input is rejected at validation (`device_operation.cpp:24`), so only the tiled path can execute. The RM branch must still be ported (it is live code the factory binds), but it is untestable, so the port should treat it as carry-through, not as a path to verify.
- **Easier — every kernel fork it needs, the sibling port creates.** All four kernels are shared with `sharded_to_interleaved`. If that op ports first, this one is at **rung 1 on all four** — bind the existing `_metal2` forks, adopt their binding names, create nothing. That is the single strongest argument for sequencing these two together, in that order.
- **Slightly harder — the output is an input.** `create_output_tensors` returns `tensor_args.cache_tensor` itself (`device_operation.cpp:56-60`), so the tensor the writer binds as its output is the same object the op received as a tensor arg. That is a port-time wiring detail for the `TensorParameter` / `TensorBinding` declaration, not a gate — flagged in the brief.

The op also has live `num_slices` / `slice_index` (they are vestigial in the sibling), so `starting_idx_h` is genuinely non-zero. It is a **tile index**, not an address, and stays clean.

Nothing here is code-path-scoped: no gate fired, so no subset question arises.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN.** Live readiness sheet (*"Operations analysis"*, `dgomez@`), fetched this run. Row `data_movement/sharded_partial/sharded_to_interleaved_partial`:

  | Column | Value |
  |---|---|
  | `Device operation` | `ShardedToInterleavedPartialDeviceOperation` |
  | `Factory (variant)` | `ShardedToInterleavedPartialProgramFactory` |
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
  - `Concept = descriptor` ✓ — `create_descriptor()` returns `tt::tt_metal::ProgramDescriptor` (`device/sharded_to_interleaved_partial_program_factory.hpp:15-18`).
  - `Custom hash = no` ✓ — no `compute_program_hash` override anywhere in the op directory; no `attribute_values` / `to_hash` backdoor either.
  - `Runtime-args update (get_dynamic_runtime_args) = no` ✓ — no such hook on the device-op (`device_operation.hpp:23-31` declares only `validate_on_program_cache_miss`, `compute_output_specs`, `create_output_tensors`, `create_op_performance_model`).
  - `Override runtime args method? = no` ✓ — no `override_runtime_arguments`; the concept is `descriptor`, so the legacy-signature reading does not apply either.
  - `Pybind descriptor = no` ✓ — `sharded_to_interleaved_partial_nanobind.cpp:43-53` binds only the `sharded_to_interleaved_partial` free function via `ttnn::bind_function`; no `nb::class_` of the device op, no `create_descriptor` binding.
  - `Op-owned tensors? = (blank/no)` ✓ — consistent with the `descriptor` concept (cross-column invariant holds). **Note the near-miss:** the op *does* write into a caller-supplied `cache_tensor` returned verbatim by `create_output_tensors`, but that is a preallocated output, **not** an op-owned tensor (which would be a non-empty `buffers` vector on a `WorkloadDescriptor`). The `descriptor` concept is correct.
  - **Factory-set match** ✓ — the sheet carries exactly **one** row for this op, and the code has exactly **one** factory (`program_factory_t = std::variant<ShardedToInterleavedPartialProgramFactory>`, `device_operation.hpp:21`). No phantom row, no missing row.
  - Cross-column invariants hold. No spreadsheet conflict.

  **Target concept: `ProgramSpecFactoryConcept`** (no op-owned tensors) — the sheet's `Porting Target` column agrees.

- **Device 2.0 (every kernel used):** **GREEN — all four kernels compliant.** No violation table; nothing to route. These are the same four files the sibling op binds, re-verified against the current tree:

  | Kernel | Device 2.0 evidence |
  |---|---|
  | `reader_unary_sharded.cpp` | `DataflowBuffer dfb(cb_id_in0); dfb.push_back(...)` (`:15-16`). |
  | `writer_unary_sharded_blocks_interleaved_start_id.cpp` | `Noc noc; DataflowBuffer dfb_out(cb_id_out)` (`:30-31`), `noc.async_write(dfb_out, s, …)` (`:41`), `dfb_out.wait_front` / `pop_front` (`:36`, `:49`), `TensorAccessor` (`:28`). `get_tile_size(cb_id_out)` (`:26`) is a **sanctioned** free function — not a violation. |
  | `writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp` | `Noc noc; DataflowBuffer dfb_out(dfb_id_out0)` (`:27-28`), `noc.async_write(...)` (`:34-39`), `dfb_out.wait_front` / `pop_front` (`:31`, `:44`). *(Bound but unreachable for this op; audited anyway, per the "follow kernel references" scope rule.)* |
  | `eltwise_copy.cpp` | `CircularBuffer cb_in(tt::CBIndex::c_0)` / `cb_out(tt::CBIndex::c_16)` (`:19-20`) with method-form FIFO ops (`:26-27`, `:34-35`), migrated by `0fb47949a27` / PR #51179. `#include "api/dataflow/circular_buffer.h"` (`:10`) is the Device-2.0 header the migration guide's own migrated example uses. |

  Verified absent across all four kernels: `InterleavedAddrGen`, `ShardedAddrGen`, `InterleavedAddrGenFast`, `InterleavedPow2AddrGen*`, raw `noc_async_read(` / `noc_async_write(`, raw semaphore addresses, the `cb_wait_front(` / `cb_push_back(` / `cb_pop_front(` / `cb_reserve_back(` free-function form, and any `get_write_ptr` / `get_read_ptr` / `get_local_cb_interface` / `fifo_*_ptr` / `evil_set_*` access. The remaining CB-index free functions in `eltwise_copy.cpp` — `unary_op_init_common(c_0, c_16)`, `copy_tile_init(c_0)`, `copy_tile(c_0, …)`, `pack_tile(0, c_16)` — are **compute LLK**, outside the Device 2.0 *data-movement* API surface the migration guide covers, and are not holdovers.

- **Feature compatibility:** every Appendix A entry scanned against host and kernel code; all **N/A** (each entry is a gate-feature, so an absent one is N/A, not GREEN).

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type, no `.global_circular_buffer` field on the `CBDescriptor`, no `remote_cb*` / `.remote_index(` / `remote_circular_buffer.h`, no 4-arg `experimental::CreateCircularBuffer`. The input CB **is** Buffer-backed (`cb.buffer = bound_buffer`, `program_factory.cpp:41`, from `:140-147`) — the legacy **borrowed-memory** pattern, a mechanical `DataflowBufferSpec::borrowed_from` translation, explicitly *not* an Appendix A entry and *not* a GCB. |
  | CBDescriptor `address_offset` (non-zero) | N/A | `push_s2i_partial_cb_pair` (`program_factory.cpp:25-43`) sets `total_size`, `core_ranges`, one `CBFormatDescriptor`, and `buffer` — it never touches `.address_offset` (defaults `0`). No `set_address_offset`, no 4-arg `UpdateDynamicCircularBufferAddress`, no `cb_descriptor_from_sharded_tensor` call anywhere in the op or its kernels. |
  | GlobalSemaphore | N/A | The op uses **no semaphores of any kind** — no `GlobalSemaphore`, no `CreateSemaphore`, no `global_semaphore.hpp`, no kernel-side semaphore wait/post. |
  | Variable-count compile-time arguments (CTA varargs) | N/A | Every kernel-side CTA read is at **constexpr index 0** and nothing else: reader `:13`, tiled writer `:22`, RM writer `:19`, compute `:13`. Host-side CTA lists are fixed-shape (`program_factory.cpp:169`, `:176-177`, `:196`); the writer's variable-length tail is `TensorAccessorArgs(*dst_buffer).append_to(...)` (`:177`), read kernel-side as the fixed `TensorAccessorArgs<1>()` NTTP, not a runtime-varying index. `tensor_args_t` is a fixed pair — `Tensor input_tensor` + `Tensor cache_tensor` (`device_operation_types.hpp:19-22`) — with no `std::vector<Tensor>`, so the op-level cue does not fire either. |

- **Offset base pointers:** **GREEN — no fold at either writer.**

  **Every address argument resolved.** The output buffer reaches both writers as a **`Buffer*` binding**, not a smuggled `->address()`: `writer_rt.push_back(dst_buffer)` (`program_factory.cpp:243` tiled, `:294` RM) into a `KernelDescriptor::RTArgList`, whose element type is `std::variant<uint32_t, Buffer*, std::reference_wrapper<const MeshTensor>>` (`tt_metal/api/tt-metalium/program_descriptors.hpp:186`). The framework auto-registers it as a `BufferBinding` and patches it on cache hits — consistent with the sheet's `Smuggled pointer = no`. There is **no `->address()` call anywhere in the op directory** (grep-confirmed). Both consumption sites are clean:

  - **Tiled writer (C1/C2, the only reachable path) — clean base.** `TensorAccessor(dst_args, dst_addr)` (`writer_unary_sharded_blocks_interleaved_start_id.cpp:28`), no arithmetic. All addressing is by **tile index**: `start_id = start_id_base + start_id_offset` (`:20`) from RTAs 8 and 7 — host `starting_idx_h` and `curr_idx_h + curr_idx_w` (`program_factory.cpp:250-251`) — feeding `{.page_id = tile_id}` (`:41`).

    **Worth stating because it differs from the sibling op:** here `starting_idx_h` is **genuinely non-zero**. `num_slices` / `slice_index` are real user inputs (the sibling hardcodes `1` / `0`), so `calculate_starting_idx_h` (`sharded_common.cpp:16-28`) returns `num_tiles_per_slice * slice_index` — the whole point of the op. That value is a **page index**, not a byte address: it is added to another tile index and passed as `page_id`, never to an accessor base or a NoC address. Not an offset-base fold, and it ports as an ordinary named RTA.

  - **RM writer (C3, unreachable) — clean base.** `TensorAccessor(dst_args, dst_addr)` (`writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp:25`), with the per-core column shift riding each write as `{.page_id = stick_id, .offset_bytes = input_width_offset_bytes}` (`:38`). This kernel previously folded that offset into the accessor base and was the sibling op's blocking gate; `6abdf94214d` (PR #51747) split it out. Since this op binds the same shared file, **the fix cleared this op's copy of the gate too** — before it was ever separately reported. Even setting reachability aside, the site is clean on its own terms.

  **Triage-doc cross-check** (`analyses/2026-07-19_offset_base_pointers.md`, dated 2026-07-19 — a dated prior, not an authority): `sharded_to_interleaved_partial` appears in **none** of the four type tables. Outcome: **"no fold, op not in the tables" → clean.** *(As noted in the sibling audit's recipe notes, the doc has no way to record a site that was catalogued-in-effect and then fixed; both halves of PR #51747 would be worth adding as resolved entries.)*

- **TensorAccessor 3rd argument:** **GREEN / none.** Both `TensorAccessor` constructions in the op's kernels are **2-argument** — tiled writer `:28`, RM writer `:25`. No explicit page-size third argument at any site, so the syntactic signal never fires and there is nothing to classify. Cross-checked against `analyses/2026-07-06_tensor_accessor_3rd_arg_triage.md` (a dated prior): the op is **not** in its op→class table — consistent with the code.

- **CB endpoints (GATE-free):** **all legal 1 producer + 1 consumer**, every CB on every node in every reachable config. No self-loop, no 1P+1C assignment needed, no multi-binding flag, no dead CB. Device 2.0 idioms are intact across all four kernels, so the precondition for this scan holds and no deferral applies.

  Two CBs exist. `c_0` (`src0_cb_index`) is the borrowed-memory input CB (`cb.buffer = src_buffer`, `program_factory.cpp:41` from `:147`), allocated in every config. `c_16` (`out_cb_index`) is allocated **only** when `convert_df` (`:149-160`, `bound_buffer = nullptr`); when `!convert_df`, `out_cb_index == src0_cb_index == c_0` (`:129`), so the writer drains the same borrowed DFB the reader fills.

  | CB | Config | Producer (locked) | Consumer (locked) | Census | Verdict |
  |---|---|---|---|---|---|
  | `c_0` (borrowed-memory) | **C1** (TILE, no convert) | reader `dfb.push_back` (`reader_unary_sharded.cpp:16`) | tiled writer `dfb_out.wait_front` / `pop_front` (`writer_unary_sharded_blocks…:36,49`); its `noc.async_write(dfb_out, …)` (`:41`) is a peek on the same binding, not a second endpoint | 2 touchers: 1 locked P + 1 locked C | **plain 1:1 legal** |
  | `c_0` (borrowed-memory) | **C2** (TILE, convert) | reader `dfb.push_back` | compute `cb_in.wait_front` / `pop_front` (`eltwise_copy.cpp:26,34`) | 2 touchers: 1 P + 1 C | **plain 1:1 legal** |
  | `c_16` | **C2** only | compute `cb_out.reserve_back` / `push_back` (`eltwise_copy.cpp:27,35`) | tiled writer `dfb_out.wait_front` / `pop_front` | 2 touchers: 1 P + 1 C | **plain 1:1 legal** |
  | `c_0` (borrowed-memory) | **C3** (ROW_MAJOR) — *code present, unreachable* | reader `dfb.push_back` | RM writer `dfb_out.wait_front` / `pop_front` (`writer_unary_stick_layout…:31,44`) | 2 touchers: 1 P + 1 C | **plain 1:1 legal** (carry-through) |

  **Hidden-second-writer hunt: negative, positively.** All four kernels were scanned for a raw co-fill or co-read — `get_write_ptr` / `get_read_ptr` / `get_local_cb_interface(…).fifo_wr_ptr` / `fifo_rd_ptr` / `evil_set_write_ptr` / `evil_set_read_ptr` — with **zero** occurrences. The op allocates **no semaphores at all**, so face (a)'s semaphore-gated co-fill has no coordinating primitive available to it. No dual-instance work-split: each kernel source is pushed into exactly **one** `KernelDescriptor` (`:311-315`), never two configs over one core range. Face (b) multiple-readers does not fire — each CB has exactly one reading kernel per config.

  **No dead CB.** `c_0`'s index reaches the reader as CTA 0 (`:169`) and the writer as CTA 0 (`:176`, when `!convert_df`); `c_16`'s reaches the writer as CTA 0 when `convert_df` (`:150`, `:176`). Both are consumed by real FIFO ops in every config in which they are allocated.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding):
  - **`input_tensor`** (`c_0`, borrowed-memory) — **clean**, via the causal-link gate. The CB is Buffer-backed (`cb.buffer = src_buffer`); the reader only `push_back`s pages that are already resident and builds no `TensorAccessor`. The borrowed-memory DFB *is* the tensor access. Port via `DataflowBufferSpec::borrowed_from`. Neither Case 1 nor Case 2.
  - **`cache_tensor`** (the output; `dst_buffer`, delivered as a `Buffer*` binding → framework `BufferBinding`) — **Case 1** (via `TensorAccessor`). Express as a `TensorParameter` / `TensorBinding`; the writer builds `TensorAccessor(tensor::<out>)`. The **arg-0 base RTA** and the **`TensorAccessorArgs` CTA plumbing** disappear together — host-side `TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args)` (`program_factory.cpp:177`) and kernel-side `constexpr auto dst_args = TensorAccessorArgs<1>()` (tiled `:23`, RM `:20`). Mechanical, low-risk.
    **Wiring note:** this binding's tensor is `tensor_args.cache_tensor` — an *input* tensor arg returned verbatim as the output (`device_operation.cpp:56-60`), not a freshly allocated output. Same object on both sides; one binding.
- **TensorParameter relaxation:** **none.** Sheet says `none`; the op has no custom hash, so there is no hash logic to reconcile.
- **TensorAccessor 3rd arg:** **none** — no site passes one.
- **CB endpoints:** all legal 1P+1C. Nothing to self-loop, assign, flag, or drop.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** **none.** All three faces came back negative with positive evidence.
- **Cross-op / shared kernels:** all four kernels are borrowed; the op owns none. **No `_metal2` sibling exists beside any of the four originals today**, but every one of them is also bound by `data_movement/sharded/sharded_to_interleaved`, whose port is queued — so the rung depends on sequencing. Full inventory in [Team-only](#team-only).
- **RTA varargs:** **none.** Every kernel reads its runtime args at **distinct constant indices** — reader 0; tiled writer 0–8; RM writer 0, 2, 3, 4, 5, 6 — with no counted loop, no running `arg_index++`, no data-selected index.
- **Unreachable RM branch:** live code the port must carry but cannot test. Do not delete it; do not treat its untestability as licence to change it.

## Team-only

- **Out-of-directory coupling & donor shape:** roll-up **✓ clean** for *function-call* escapes. No kernel `#include`s another op's helper: the four kernels' includes are `api/dataflow/dataflow_api.h`, `api/dataflow/noc.h`, `api/dataflow/dataflow_buffer.h`, `api/dataflow/circular_buffer.h`, `api/tensor/noc_traits.h`, `api/compute/{common,tile_move_copy,eltwise_unary/eltwise_unary}.h`, `api/debug/dprint.h` — all `tt_metal/*` LLK/HAL (donor class 1, no concern). No per-call shape analysis is owed. Host-side, the factory `#include`s `ttnn/operations/data_movement/sharded/sharded_common.hpp` for `calculate_starting_idx_h` (`program_factory.cpp:11`, `:206-207`) — a **cross-directory host include** (this op lives under `sharded_partial/`, the helper under `sharded/`), but host code with no kernel-token bridging, so out of this subject's scope.

  **File-path kernel instantiation is the whole coupling story: the op owns none of its kernels.** Consumer sets below are a **sunset and coordination list — not authorization to convert any of these files in place.** Census by filename grep over `ttnn/cpp`, hits filtered to factory bindings, quasar copies excluded:

  | Borrowed kernel | Owning family / pool | Class | Sibling `_metal2` fork? | Other ops binding it (sunset list) |
  |---|---|---|---|---|
  | `reader_unary_sharded.cpp` | `eltwise/unary` | cross-family | **No** (see the non-sibling typecast fork below) | broadly shared — `sharded_to_interleaved`, `tilize` (×2), `transpose_wh_sharded`, `untilize` (×3), `untilize_with_unpadding`, `slice_write` (×2) |
  | `writer_unary_sharded_blocks_interleaved_start_id.cpp` | `data_movement/sharded` | cross-directory | **No** | `sharded_to_interleaved` |
  | `writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp` | `data_movement/sharded` | cross-directory | **No** | `sharded_to_interleaved` |
  | `eltwise_copy.cpp` | `ttnn/cpp/ttnn/kernel/compute/` (shared pool) | shared-lib | **No** | `copy` (×2), `interleaved_to_sharded`, `sharded_to_interleaved`, `interleaved_to_sharded_partial`, `untilize_with_unpadding` |

  **Sequencing is the actionable finding here.** This op and `data_movement/sharded/sharded_to_interleaved` bind **the same four kernels** and are both GREEN with the same `descriptor` → `ProgramSpecFactoryConcept` shape. Whichever ports first creates all four `_metal2` forks (rung 2) and fixes their binding names; the second is at **rung 1 on all four** — bind and adopt, create nothing. Porting them in either order works, but porting them *concurrently* would race on four shared files. Recommend sequencing them, sibling first (it is the more general op, so its binding names are the better interface to inherit).

  Also note: a real, non-quasar Metal 2.0 fork of `reader_unary_sharded.cpp` already exists at `ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/dataflow/reader_unary_sharded_metal2.cpp` (`cbde3d44ff3`, PR #51397, on `main`) — but in **typecast's** tree, not beside the original, so rung 1's locational check misses it. This is the same open convention question raised in the sibling audit; whichever port runs first should settle it once for both.

  **Porter warning worth carrying forward: `ttnn/cpp/ttnn/operations/experimental/quasar/sharded_to_interleaved/` is a pre-port copy of the sibling op** and binds the same kernels. It will look like a finished answer to every question this port raises; it is not one, and it carries idioms the port recipe forbids. Do not read it, template from it, lift its binding names, or count its `_metal2` files as forks to reuse.

- **Relaxation candidates (mined from a custom hash):** **none** — the op has no custom hash, so there is nothing to mine.

- **TTNN factory analysis (sheet-derived + `file:line`):** current concept `descriptor` (`program_factory.hpp:15`); target `ProgramSpecFactoryConcept`; **no** op-owned tensors (the caller-supplied `cache_tensor` is a preallocated output, not an op-owned buffer); **no** pybind `create_descriptor` and no other risky pybind (`sharded_to_interleaved_partial_nanobind.cpp` exposes only the free function); **no** custom hash; **no** `get_dynamic_runtime_args`; **no** `override_runtime_arguments`; `Is safe to port? = yes`. Every gate conjunct absent → the TTNN gate clears cleanly. One factory, one sheet row, no MeshWorkload need.

## Misc anomalies  *(team-only, non-gating; the port does not act on these)*

The first three are specific to this op and are **not** shared with the sibling — they live in the device-operation layer, which is where the two ops actually differ. The last two are inherited from the cloned factory and are already recorded in the sibling's audit.

- **⚠ Missing `cache_tensor` validation — the most substantive finding here.** The op validates the *input* thoroughly but the caller-supplied output barely at all. `validate_on_program_cache_miss` (`device_operation.cpp:12-48`) checks only that `cache_tensor`'s height is divisible by `num_slices` (`:25-27`). It never checks the cache tensor's **layout**, **dtype**, **storage type**, **buffer non-null**, or **device**. The sibling op checks every one of those for its preallocated output (`sharded_to_interleaved_device_operation.cpp:31-51`), including `output_tensor.layout() != input_tensor.layout()`.

  That gap is reachable and the factory is not robust to it: the factory takes the TILE/ROW_MAJOR decision off **two different tensors** — unit sizes and core counts branch on `output.layout()` (`program_factory.cpp:81`, `:113`), while kernel selection and the per-core RTA loop branch on `input.layout()` (`:178`, `:214`, `:215`). Input is forced TILE by `:24`, but a **row-major `cache_tensor`** would take the row-major sizing path at `:94-107` while still selecting the **tiled** writer and the tiled RTA layout — a silently inconsistent program. Recommend adding the sibling's layout/dtype/device checks. *(Independent of the port; the port neither causes nor fixes it.)*

- **`output_mem_config` is a completely dead, hash-keyed attribute.** It is threaded from the nanobind arg → `ttnn::sharded_to_interleaved_partial` (`sharded_to_interleaved_partial.cpp:20`, defaulting to `input_tensor.memory_config()`) → into `ShardedToInterleavedPartialParams` (`device_operation.cpp:85`), and then **read by nothing** — not the validator, not `compute_output_specs` (which returns `cache_tensor.tensor_spec()` and ignores the attributes entirely, `:50-54`), not the factory. Being an `operation_attributes_t` member it still feeds the default `compute_program_hash`, so distinct-but-irrelevant memory configs fragment the program cache. Textbook dead-but-hashed attribute.

- **`output_dtype`'s only use is a vacuous check.** `args.output_dtype` is read exactly once, at `device_operation.cpp:42-47`: `if (input_tensor.dtype() != args.output_dtype) TT_FATAL(input_tensor.layout() == Layout::TILE, …)`. That conclusion is already unconditionally enforced 18 lines earlier at `:24`, so the check can never fail. Note also that it does **not** drive the actual conversion decision: `convert_df` is computed from the *cache tensor's* dtype versus the input's (`program_factory.cpp:126`, via `:69`), so a caller passing `output_dtype` that disagrees with `cache_tensor.dtype()` gets conversion behaviour keyed to the tensor, not the argument — with no diagnostic. Like `output_mem_config`, it still keys the program hash.

- **Unreachable row-major branch.** `TT_FATAL(input_tensor.layout() == Layout::TILE, …)` (`device_operation.cpp:24`) makes the factory's entire `else` branch — RM unit sizing (`:94-107`), RM writer selection (`:182-186`), and the RM per-core RTA block (`:259-308`) — dead code, along with the RM writer kernel binding. The comment on `:24` (*"Currently, only tile layout is supported"*) suggests the restriction is intended to be temporary. Either the restriction should be lifted and the branch tested, or the branch retired; carrying untestable live code indefinitely is the worst of the three. *(For the port: carry it through unchanged — see the brief.)*

- **Inherited from the cloned factory** (same `file:line` as the sibling, already reported there): the RM path's **dead RTA at index 1** (`num_units_per_row`, pushed at `program_factory.cpp:295`, never read by the RM writer, which reads 0, 2, 3, 4, 5, 6) — doubly dead here, since the whole RM path is unreachable; and **`is_l1_aligned` hardcoded `true`** (`:55`), making the guard at `:287-290` unconditionally taken, `is_blackhole` (`:135`) and `dst_is_dram` (`:134`) effectively dead in that branch, and the first `padded_shard_width` assignment (`:286`) always overwritten.

- **Stray debug include in the borrowed reader.** `reader_unary_sharded.cpp:9` includes `api/debug/dprint.h` with no `DPRINT` use. Cosmetic, and it belongs to `eltwise/unary` — not this op's to fix.

## Per-DeviceOperation attribution

Not applicable — the directory holds exactly one `DeviceOperation` with exactly one program factory, and no finding differs by config.

## Questions for the user

1. **Sequencing with the sibling op.** `data_movement/sharded/sharded_to_interleaved` is GREEN as of the same date and binds **the same four kernels**. Porting them concurrently would race on four shared files; porting them in sequence lets the second reuse all four `_metal2` forks at rung 1. Recommend **sibling first**, this op second. Confirm the order (and that they are not assigned to two porters at once).
2. **Which fork does the reader bind?** Unchanged from the sibling audit: `reader_unary_sharded_metal2.cpp` exists on `main` in **typecast's** directory rather than beside the original, so rung 1's locational check reports "no fork" and rung 2 would create a second. Options: **(a)** bind typecast's existing fork despite the non-sibling path; **(b)** create the sibling fork per the letter of rung 2 and accept two forks; **(c)** relocate typecast's fork beside the original first, on the ops track. **This should be answered once and applied to both ports** — settling it per-op invites divergence.
3. **The `cache_tensor` validation gap.** The missing layout/dtype/device checks (first Misc anomaly) are a genuine latent bug, independent of the port, with a ready-made fix — copy the sibling's checks. File against the ops team now?
4. **The unreachable row-major branch.** Is the `Layout::TILE`-only restriction still intended to be temporary? If it is being lifted soon, that changes what the port has to get right in an untestable branch; if it is permanent, retiring the branch (on the ops track, before or after the port) would remove dead code the port otherwise has to carry.

## Recipe notes  *(friction with the audit recipe itself)*

- **The recipe has no notion of an audited-but-unreachable code path**, and this op has one. The RM branch is *referenced* by a `KernelDescriptor::kernel_source`, so the [Scope of the audit] rule ("follow kernel references") puts it squarely in scope — correct, and I audited it. But the [CB endpoints] subject's *"classify per instantiation"* rule and the config-matrix framing throughout both assume every config is reachable, so there is no sanctioned way to say "this `(CB, config)` row is carry-through, not a live instantiation." I recorded it inline. A one-line convention — mark an unreachable config and state that its dispositions are carry-through — would keep the distinction legible to the porter, who otherwise cannot tell which rows they can actually test against.
- **A "clone factory" is a real and repeatable shape the recipe could exploit.** This op's factory is byte-identical to its sibling's after identifier normalisation, and all four kernels are shared. Roughly two-thirds of this audit's evidence (Device 2.0, features, CB census, offset bases, 3rd arg, RTA varargs, coupling) was necessarily identical, and I re-derived it to be sure rather than assert it. That was the right call, but the recipe could make the pattern cheap and *safe*: a short instruction to (a) verify clone-ness mechanically (`diff` after normalising identifiers — the check that actually earns the shortcut), (b) inherit the shared-code findings by explicit reference, and (c) audit the *divergent* layer in full. Without that, an auditor either wastes effort or takes the shortcut informally and undocumented. **The failure mode to warn about is precisely what happened here:** the two ops' factories are identical, but every finding that mattered — the unreachable RM branch, the two dead-but-hashed attributes, the missing `cache_tensor` validation — lives in the device-operation layer where they diverge. Clone-ness is a licence to share the *kernel and factory* evidence, never the device-op evidence.
- **Carried forward from the sibling audit, still open** (all verified again this run, none op-specific): the offset-base recognition rule still describes only host-side folds; the offset triage doc has no way to record a *resolved* site (both halves of PR #51747 are now such sites); the readiness sheet's live header is `Override runtime args method?\n(PD only)` while the docs quote `(PD and legacy)`; the sheet carries undocumented gate-adjacent columns (`Porting Target`, `Backdoor custom hash`); and rung 1's locational fork check misses a legitimate non-sibling fork. See `../../sharded/sharded_to_interleaved/METAL2_PREPORT_AUDIT.md` → Recipe notes for the full write-ups; not repeated here.
