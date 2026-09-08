# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/data_movement/pad`

One device operation, seven program factories, all declared in the `program_factory_t` variant of
`device/pad_device_operation.hpp:32`:

- **`PadDeviceOperation`** (`device/pad_device_operation.{hpp,cpp}`)
  - `PadRmReaderWriterProgramFactory` (`pad_rm_reader_writer_program_factory.cpp`) — RM, single core, `create_workload_descriptor`
  - `PadRmReaderWriterMultiCoreProgramFactory` (`pad_rm_reader_writer_multi_core_program_factory.cpp`) — RM, multicore, `create_workload_descriptor` — **unreachable**: never returned by `select_program_factory` (see *Misc anomalies*)
  - `PadRmReaderWriterMultiCoreDefaultProgramFactory` (`pad_rm_reader_writer_multi_core_default_program_factory.cpp`) — RM, multicore, the default RM path
  - `PadRmShardedHeightOnlyProgramFactory` (`pad_rm_sharded_height_only_program_factory.cpp`) — RM height-sharded, optimized
  - `PadRmShardedWidthOnlyProgramFactory` (`pad_rm_sharded_width_only_program_factory.cpp`) — RM width-pad, sharded stickwise
  - `PadTileCoreProgramFactory` (`pad_tile_program_factory.cpp`) — TILE, single core
  - `PadTileMulticoreProgramFactory` (`pad_tile_multicore_program_factory.cpp`) — TILE, multicore

All 12 kernel files under `device/kernels/dataflow/` are referenced by at least one factory; none is
dead code. One kernel is instantiated from outside the op directory
(`eltwise/unary/.../reader_unary_interleaved_start_id.cpp`, by `PadTileCoreProgramFactory`).

> **`ttnn/cpp/ttnn/operations/experimental/quasar/pad/` exists and is out of bounds.** It is a
> whole-op shortcut copy, not a precedent. Nothing in it was read for this audit, and the porter
> should not read it either — in particular its `_metal2` kernels do **not** count as forks to reuse.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `metal2_audit.md`.

**Recipe docs:** `64668f470e4 2026-08-24 docs(metal_2.0): a run in flight freezes the kernel sources`

**Readiness sheet:** fetched live this session (7 rows for `data_movement/pad`, one per factory).

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/pad` |
| **Overall** | **GREEN** — all seven factories cleared. One readiness-sheet cell is **outdated** and is overridden here on code evidence; see *Result*. |
| **DOps / Factories** | `PadDeviceOperation` → 7 factories (listed above) |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all 12 op kernels + the one donor kernel are structurally Device 2.0 (`Noc`, `DataflowBuffer`, `TensorAccessor`, `CoreLocalMem`). No holdovers. |
| *Prereqs* — Cross-op escapes | Ok — one in-family header (`data_movement/common/kernels/common.hpp`, Shape 1) and one borrowed kernel file with an existing `_metal2` fork |
| *Feature Support* — overall | **GREEN** — every Appendix A entry `N/A` |
| *Feature Support* — Variadic-CTA | Ok (not an Appendix A entry; no varying-index CTA reads found either) |
| *TTNN Readiness* — `Is able to port?` (the gate) | **6 × `yes`** · 1 × `no` (`PadRmShardedHeightOnlyProgramFactory`) — **the `no` is outdated and is overridden here.** It is derived from two cells the code contradicts, both fixed by PR #52556 five days before this audit. Cleared on code evidence; the sheet row still needs correcting. |
| *TTNN Readiness* — Concept (current) | `descriptor` × 5 · `WorkloadDescriptor` × 2 (`PadRmReaderWriter*`) — cross-check ✓ |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | Yes on both `WorkloadDescriptor` rows (`Why secretly SPMD? = "Op-owned tensors"`) — cross-check ✓ with one nuance, see *Gate detail* |
| *TTNN Readiness* — Custom hash | No (sheet: `no` × 7; grep for `compute_program_hash` in the op: zero hits) ✓ |
| *TTNN Readiness* — `get_dynamic_runtime_args` | **No.** Sheet says `yes` on `PadRmShardedHeightOnlyProgramFactory`; the hook does not exist anywhere in the op. **Sheet outdated** — removed by PR #52556. |
| *TTNN Readiness* — `override_runtime_arguments` | **Yes, on one factory** — `device/pad_rm_sharded_height_only_program_factory.cpp:412` (declared `.hpp:22`). Sheet says `no` × 7. **Sheet outdated** — added by PR #52556. Not a gate: it selects `CustomProgramSpecFactoryConcept`. |
| *TTNN Readiness* — Pybind `create_descriptor` | No — `pad_nanobind.cpp` binds only `ttnn::pad` overloads; no `nb::class_` of the device op ✓ |
| *TTNN Readiness* — Op-owned tensors | **Yes** on the two `WorkloadDescriptor` factories: the pad-value const tensor parked on `WorkloadDescriptor::buffers` (`pad_rm_reader_writer_program_factory.cpp:200`, `pad_rm_reader_writer_multi_core_program_factory.cpp:419`) ✓ |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept` × 6 (two of them carrying op-owned tensors) · `CustomProgramSpecFactoryConcept` × 1 (the height-only factory, per the code — the sheet has not recorded its `override_runtime_arguments` yet) |
| *Port work* — Offset base pointer | **none** — no address RTA anywhere folds a host-side offset into its base |
| *Port work* — Tensor bindings (per binding) | Case 1 × 12 · clean (borrowed DFB) × 4 · **no Case 2** |
| *TTNN Readiness* — TensorParameter relaxation | `none` on all 7 rows → clears |
| *Port work* — TensorAccessor 3rd arg | 2 sites, both **Class 2** (redundant) → drop |
| *Port work* — CB endpoints | legal × 5 · self-loop × 7 · 1P+1C × 1 · **dead-CB drop × 1** · **conditional DFB × 1** |

**CB endpoints** are dispositions, not gates: every out-of-window CB below has a port-time
resolution. Recorded per `(CB, config)` in the *Gate detail* section.

## Result

**GREEN → brief issued for all seven factories.** Device 2.0, Appendix A feature compatibility,
offset base pointers, the TensorAccessor 3rd argument, and the `TensorParameter` relaxation column
all clear cleanly for every factory. The port can begin.

**One readiness-sheet cell is outdated and is overridden here.**
`PadRmShardedHeightOnlyProgramFactory`'s row reads `Is able to port? = no`, derived from
`Runtime-args update (get_dynamic_runtime_args) = yes`. **The code contradicts that cell, and so does
`Override runtime args method? (PD only) = no`.** PR **#52556** (`90ec10f4bf4`, merged to `main`
2026-08-19, five days before this audit) — *"[ttnn] pad: replace get_dynamic_runtime_args with a
factory override_runtime_arguments"* — removed the deprecated hook and replaced it with an
`override_runtime_arguments` that re-points only the two sharded CB base addresses. The sheet's row
still describes the pre-#52556 state; its `Diego validation = no`, `Op Classification = "Broken Op"`
and `Porting Target = "(N/A)"` cells are all downstream of the same lag.

Because the blocking column is factually wrong on a cheaply-verifiable point, and every gate this
audit checks independently clears for that factory, **the verdict is recorded as GREEN on the code
evidence** rather than on the stale cell. The factory's target concept is
`CustomProgramSpecFactoryConcept` (per the `override_runtime_arguments` the sheet has not yet
recorded), and it is included in the porter brief.

**Still to do, but not a blocker:** the readiness-sheet owner should correct the row —
`Runtime-args update (get_dynamic_runtime_args)` → `no`, `Override runtime args method? (PD only)` →
`yes`, citing #52556 / `90ec10f4bf4`. Leaving it stale will mislead the next reader of that row (the
port tracker and any other consumer), independently of this port.

**Reader's caveat on the override.** `Is able to port?` is a derived cell whose formula this audit
cannot see, so a `no` can in principle encode a consideration outside the audit's view. Here the one
visible input to it is demonstrably wrong, and the row also carries `Diego validation = no` (never
validated). The GREEN therefore rests on this audit's own gate-by-gate evidence for that factory —
Device 2.0 ✓, no Appendix A features, no offset folds, no `TensorAccessor` 3rd-arg sites, CB
endpoints all resolvable (self-loop ×2 + 1P+1C), relaxation `none` — not on a reconciled sheet.

**Subject coverage.** All twelve subjects were run in full for all seven factories; nothing was
skipped or deferred.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN for all seven** — six rows read `yes`; the
  seventh reads `no` on an **outdated** cell that is overridden here (see *Result*). Per-factory
  verdicts and the full cross-check:

  | Factory (sheet row) | `Concept` | `Is able to port?` | Cross-check |
  |---|---|---|---|
  | `PadRmReaderWriterMultiCoreDefaultProgramFactory` | `descriptor` | `yes` | ✓ `create_descriptor` at `..._default_program_factory.cpp:32` |
  | `PadRmReaderWriterMultiCoreProgramFactory` | `WorkloadDescriptor` | `yes` | ✓ `create_workload_descriptor` at `..._multi_core_program_factory.cpp:404`; `Op-owned tensors? = yes` ✓ (`wd.buffers` @ `:419`) |
  | `PadRmReaderWriterProgramFactory` | `WorkloadDescriptor` | `yes` | ✓ `create_workload_descriptor` at `..._program_factory.cpp:185`; `Op-owned tensors? = yes` ✓ (`wd.buffers` @ `:200`) |
  | **`PadRmShardedHeightOnlyProgramFactory`** | `descriptor` | `no` → **overridden, sheet outdated** | ✓ `create_descriptor` at `..._height_only_program_factory.cpp:195`; **two cells stale** — see below |
  | `PadRmShardedWidthOnlyProgramFactory` | `descriptor` | `yes` | ✓ `create_descriptor` at `..._width_only_program_factory.cpp:20` |
  | `PadTileCoreProgramFactory` | `descriptor` | `yes` | ✓ `create_descriptor` at `pad_tile_program_factory.cpp:18` |
  | `PadTileMulticoreProgramFactory` | `descriptor` | `yes` | ✓ `create_descriptor` at `pad_tile_multicore_program_factory.cpp:31` |

  **The two outdated cells** (both cheaply-checkable primary columns, both on the same row, both
  fixed by PR #52556):

  | Column | Sheet value (stale) | Code evidence (current) |
  |---|---|---|
  | `Runtime-args update (get_dynamic_runtime_args)` | `yes` | **Absent.** `grep -rn "get_dynamic_runtime_args" ttnn/cpp/ttnn/operations/data_movement/pad/` returns only a *comment* referencing its removal (`pad_rm_sharded_height_only_program_factory.hpp:21`: *"Replaces get_dynamic_runtime_args (#48928)"*). No hook on `PadDeviceOperation` (`device/pad_device_operation.hpp:41-52`). |
  | `Override runtime args method? (PD only)` | `no` | **Present.** `void PadRmShardedHeightOnlyProgramFactory::override_runtime_arguments(...)` at `device/pad_rm_sharded_height_only_program_factory.cpp:412`, declared at `.hpp:22`. It rebuilds a CB-address-only `ProgramDescriptor` and calls `apply_descriptor_runtime_args`. |

  The two are enforced mutually exclusive at the device-op level, and the code satisfies that
  (exactly one of them exists); it is the *sheet* that carries the older of the two states — it
  records the hook that was removed and misses the method that replaced it, which is exactly the
  signature of a row written before #52556 landed. The cross-column invariants hold on every row (`get_dynamic_runtime_args` only on a `descriptor`
  concept; `Op-owned tensors? = yes` only on `WorkloadDescriptor` rows). **Factory-set match: ✓** —
  all 7 code factories have a row and all 7 sheet rows map to a live factory; no phantom, no missing.

  *Nuance on `Secretly SPMD Workload? = yes`* (recorded, not a conflict). The recipe's code basis is
  "a single entry in the `programs` vector." Both `WorkloadDescriptor` factories actually push **one
  entry per range in `tensor_coords`** (`pad_rm_reader_writer_program_factory.cpp:205-211`,
  `..._multi_core_program_factory.cpp:423-429`) — but every entry carries the *identical*
  `ProgramDescriptor`, built once above the loop. That is SPMD in substance, and it collapses to the
  single-program concept; the sheet's `yes` is right. Flagged as recipe friction below.

- **Device 2.0 (every kernel used):** **GREEN.** Every kernel the op exercises is structurally
  Device 2.0 — `Noc`, `DataflowBuffer` objects, `TensorAccessor`, `CoreLocalMem`, `UnicastEndpoint`.
  No `InterleavedAddrGen` / `ShardedAddrGen` / raw `noc_async_*` / `cb_reserve_back` family anywhere.
  Every `get_write_ptr()` / `get_read_ptr()` in the tree is a **method on a `DataflowBuffer`
  instance**, not a CB-index free function — so there are no isolated holdovers either.

  The only CB-index free-function calls in scope are both on the **sanctioned** list and are
  therefore *not* violations:

  | File | Line | Call | Wrapper in scope | Verdict |
  |---|---|---|---|---|
  | `device/kernels/dataflow/writer_unary_pad_dims_interleaved.cpp` | 28 | `get_tile_size(cb_id_out0)` | `DataflowBuffer dfb_out0` (line 32) | sanctioned — not a holdover |
  | `eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` (donor) | 25 | `get_local_cb_interface(cb_id_in0).fifo_page_size` | `DataflowBuffer dfb` (line 33) | sanctioned — not a holdover |

  Both become member lookups at *port* time (kernel-side whitelist rule 7) — a port-stage change, not
  a Device 2.0 one. Note that for the donor the existing `_metal2` fork has already made that move
  (`dfb.get_entry_size()`), so the porter inherits it rather than performing it.

  The op owns no compute kernels — every `KernelDescriptor` in all seven factories carries a
  `ReaderConfigDescriptor` or `WriterConfigDescriptor`. (Consequence for the port: the compute
  `opt_level` question does not arise for this op.)

- **Feature compatibility:** every Appendix A entry scanned against host code, factory code,
  descriptors and kernels. All absent.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type, no `CreateGlobalCircularBuffer`, no `CBDescriptor::global_circular_buffer` field set, no `remote_cb_*` / `.remote_index(` / `remote_circular_buffer.h` idiom. Grep over the whole op directory: zero hits on every signal. |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `.address_offset`, no `set_address_offset`, no 4-arg `UpdateDynamicCircularBufferAddress`, no `cb_descriptor_from_sharded_tensor`. Zero hits. The two borrowed-memory CBs bind at base (`cb.buffer = src_buffer` / `dst_buffer`) with no offset field at all. |
  | GlobalSemaphore | N/A | The op uses **no semaphores of any kind** — `grep -rn "[Ss]emaphore"` over the op directory returns zero hits. |

- **CB endpoints (GATE-free):** censused per `(factory, CB, config)`, per node. All Device 2.0
  idioms intact, so the scan is reliable (no deferral).

  | Factory | CB | Touchers on a node | Verdict | Port-time resolution |
  |---|---|---|---|---|
  | `PadRmReaderWriterProgramFactory` | `c_0` | reader locked-producer (`reserve_back`/`push_back`, `reader_pad_dims_rm_interleaved.cpp:85,112`) + writer locked-consumer (`wait_front`/`pop_front`, `writer_pad_dims_rm_interleaved.cpp:41,50`) | plain 1:1 | legal — no action |
  | `PadRmReaderWriterMultiCoreProgramFactory` | `c_0` | same two kernels, same roles | plain 1:1 | legal — no action |
  | `PadRmReaderWriterMultiCoreDefaultProgramFactory` | `c_0` | reader locked-producer (`reader_..._v2.cpp:114,170`) + writer locked-consumer (`writer_..._v2.cpp:31,60`) | plain 1:1 | legal — no action |
  | ″ | `c_1` (`cb_pad`) | reader only — local-DFB `get_write_ptr` (`reader_..._v2.cpp:19`) + `get_read_ptr` (`:98`); the writer never touches it | 1 toucher | **self-loop** |
  | ″ | `c_2` (`dfb_pad_align`) | reader only (`:99`, `:134`, `:139`, `:142`) | 1 toucher **but config-dependent allocation** | **conditional DFB** — see the callout below |
  | `PadRmShardedHeightOnlyProgramFactory` | `c_0` (borrowed ← input) | reader raw peek only, `get_write_ptr` (`reader_pad_dims_rm_sharded.cpp:32`) | 1 toucher, role-free | **self-loop** (+ `borrowed_from` the input tensor) |
  | ″ | `c_16` (borrowed ← output) | reader locked-producer (`reserve_back`/`push_back`, `:31,69`) **+** writer raw peek `get_write_ptr` (`writer_pad_dims_rm_sharded.cpp:90`) | 2 touchers, 1 locked + 1 role-free | **1P+1C** — reader PRODUCER, writer CONSUMER. **Not** multi-binding. |
  | ″ | `c_1` (`cb_pad`) | writer only (`writer_pad_dims_rm_sharded.cpp:16,22,43,82`) | 1 toucher | **self-loop** |
  | `PadRmShardedWidthOnlyProgramFactory` | `c_0` (borrowed ← input) | reader raw peek only (`reader_..._stickwise.cpp:29`) | 1 toucher, role-free | **self-loop** (+ `borrowed_from`) |
  | ″ | `c_16` (borrowed ← output) | reader locked-consumer (`wait_front`/`pop_front`, `:37,55`) + writer locked-producer (`reserve_back`/`push_back`, `writer_..._stickwise.cpp:56,75`) | plain 1:1 | legal — no action (+ `borrowed_from`) |
  | ″ | `c_1` (`padding_value_cb`) | writer only (`writer_..._stickwise.cpp:24,60`) | 1 toucher | **self-loop** |
  | `PadTileCoreProgramFactory` | `c_0` | donor reader locked-producer (`reader_unary_interleaved_start_id.cpp:43,46`) + writer locked-consumer (`writer_unary_pad_dims_interleaved.cpp:62,66`) | plain 1:1 | legal — no action |
  | ″ | `c_1` | writer only — `reserve_back(1)` + `get_write_ptr()`, never pushed (`writer_unary_pad_dims_interleaved.cpp:35,37`) | 1 toucher | **self-loop** |
  | `PadTileMulticoreProgramFactory` | `c_0` | reader locked-producer (`reader_pad_tiled.cpp:54,57`) + writer locked-consumer (`writer_pad_tiled.cpp:76,81`) | plain 1:1 | legal — no action |
  | ″ | **`c_1`** | **0 touchers** | **DEAD CB** | **drop the allocation + the dead CTA** — see the callout below |
  | ″ | `c_2` (`pad_val`) | writer only (`writer_pad_tiled.cpp:48,49,57`) | 1 toucher | **self-loop** |

  No CB anywhere in this op needs the multi-binding advanced option: no node carries ≥3 distinct
  touchers, and no two kernels are locked to the same FIFO role. The hidden-second-writer face was
  hunted explicitly — every raw `get_write_ptr` / `get_read_ptr` in the tree was traced to its owning
  kernel, and the only cross-kernel raw touch is `writer_pad_dims_rm_sharded.cpp:90` on `c_16`, which
  is a role-free peek resolvable as 1P+1C. No semaphores exist in the op, so the semaphore-gated
  co-fill shape cannot be present.

  > **Conditional DFB — `c_2` in `PadRmReaderWriterMultiCoreDefaultProgramFactory`, with a kernel-side
  > catch.** The host allocates `c_2` **only** when `stick_size_padded_front != 0 || unaligned`
  > (`..._default_program_factory.cpp:127-139`). The kernel, however, constructs the buffer object
  > **unconditionally** (`reader_pad_dims_rm_interleaved_v2.cpp:90-93`) and calls
  > `dfb_pad_align_exp.get_read_ptr()` **unconditionally** at `:99` — only the *uses* at `:134/:139/:142`
  > sit behind `if constexpr (front_padding)` / `if constexpr (unaligned)`. Under the legacy CB model
  > the unguarded read of an unconfigured CB is harmless (the value is never used); under Metal 2.0 a
  > kernel may not reference a DFB it has not bound, so the port cannot simply make the spec
  > conditional and leave the kernel alone. Report it as *dead under
  > `stick_size_padded_front == 0 && !unaligned`, live otherwise* — **do not drop it.** Note that
  > `if constexpr` does **not** suppress `dfb::` name lookup, so gating the kernel line needs a real
  > preprocessor `#ifdef` paired with a host-side define, not a constexpr branch.

  > **Dead CB — `c_1` in `PadTileMulticoreProgramFactory`.** Allocated at
  > `pad_tile_multicore_program_factory.cpp:70-78` (`page_size * multi_buffering_size` bytes), and its
  > index is threaded to the writer as compile-time arg 1. **No kernel ever touches it.** Positively
  > confirmed: `output_cb_id` appears exactly once in the entire kernel tree — its own declaration at
  > `writer_pad_tiled.cpp:23` — and is never read again; the reader's CTA list
  > (`pad_tile_multicore_program_factory.cpp:116-120`) does not carry the index at all; neither tiled
  > kernel hardcodes `tt::CBIndex::c_1` or any aliased/computed index. This factory has a single
  > instantiation shape (the allocation is unconditional, behind no branch), so the CB is dead in
  > *every* config. A dead CB has no behavior, so removing it changes none — and a DFB with neither a
  > producer nor a consumer binding cannot be expressed at all. **Drop the allocation and the dead CTA,
  > and record both `file:line` in the port report.**

- **Offset base pointers:** **GREEN.** No address RTA in any factory folds a host-side offset into
  its base. Every buffer base reaches a kernel as a bare `Buffer*` pushed into
  `KernelDescriptor::RTArgList` (the framework's `BufferBinding` form), never as
  `buffer->address() + <expr>`. The only `->address()` call sites in the whole op are six
  `log_debug` arguments (`pad_rm_reader_writer_program_factory.cpp:115,116,128` — inside an `#if 0`
  block — and `pad_rm_reader_writer_multi_core_program_factory.cpp:285,286,298`); none reaches a
  runtime-arg context. Type 3 (`address_offset`) is absent (see the feature table). Type 4
  (`ttnn::narrow` / interior-base `MeshBuffer::create`) is absent. `data_movement/pad` does not appear
  in the `2026-07-19_offset_base_pointers.md` triage tables, and the scan agrees — *no fold, op not in
  the tables* → clean. All address RTAs hand off to *TensorParameter analysis* as clean bases.

- **TensorAccessor 3rd argument:** **GREEN — 2 sites, both Class 2 (redundant → drop).** Both live in
  `PadRmReaderWriterMultiCoreDefaultProgramFactory`'s kernel pair; every other `TensorAccessor` in the
  op is 2-arg.

  | Site | Value (host) | Specialization | Class |
  |---|---|---|---|
  | `reader_pad_dims_rm_interleaved_v2.cpp:95` — `TensorAccessor(src_args, src_addr, accessor_page_size)` | `input_accessor_page_size`, CTA 21 (`..._default_program_factory.cpp:57,63,163`) | interleaved **or** sharded, per input | **2** |
  | `writer_pad_dims_rm_interleaved_v2.cpp:25` — `TensorAccessor(dst_args, dst_addr, accessor_page_size)` | `output_accessor_page_size`, CTA 4 (`..._default_program_factory.cpp:68,73,171`) | interleaved **or** sharded, per output | **2** |

  Resolving the two classifying questions:

  1. **Sharded or interleaved?** Both. This factory serves interleaved RM *and* the sharded RM
     fall-through (`pad_device_operation.cpp:97`), so each accessor takes whichever specialization the
     tensor's memory config selects — which is exactly why the host branches on `is_sharded()`.
  2. **Correct or wrong magnitude?**
     - *Sharded branch* — the host passes `buffer->aligned_page_size()` **verbatim**
       (`:63`, `:73`). That is literally the value Metal 2.0 supplies implicitly, so the sharded
       specialization's verbatim use is a no-op. Class 2, sub-type (a).
     - *Interleaved branch* — the host passes the true logical page: `stick_size = W·element_size`
       for the input (`:47,57`) and `stick_size_padded = W_padded·element_size` for the output
       (`:48,68`). The output value is *exactly* the output buffer's page size, since
       `compute_output_specs` builds the output from the same `output_padded_shape`
       (`pad_device_operation.cpp:216-223`). The interleaved accessor then realigns whatever it is
       given — `InterleavedAddrGen::aligned_page_size = align_power_of_2(page_size, allocator_alignment)`
       (`tt_metal/hw/inc/internal/dataflow/dataflow_api_addrgen.h:289-290`) — so a correct-magnitude
       value is inert even before alignment. Class 2, sub-type (b).

  **Not Class 1**, despite being interleaved row-major: both values are **compile-time args**, not
  runtime args, so the page size cannot vary across shapes that reuse one compiled program (a change
  forces a recompile). Class 1's customers carry their extents as RTAs; pad carries `N/H/C/stick_size`
  as CTAs 0-9. The porter therefore drops the arg and must **not** set `dynamic_tensor_shape`.

  *Triage-doc reconciliation:* `data_movement/pad` is **absent** from
  `2026-07-06_tensor_accessor_3rd_arg_triage.md` (only the unrelated `fill_pad` and `padded_slice`
  appear). That is expected rather than a disagreement — the sites were introduced by PR #47507
  (`75ee03e9dc3`, 2026-07-08), two days *after* the dated analysis. Classified here from first
  principles, per the recipe.

  One residual, recorded as a question rather than a gate: the input value uses `a.logical_shape()[3]`
  while the same factory reads `a.padded_shape()` for the H/C/N bounds
  (`..._default_program_factory.cpp:39-40` vs `:197-198`). If a RM input ever carried last-dim padding
  (`padded[-1] > logical[-1]`), `stick_size` would understate the real page. That would be a
  *pre-existing* op bug rather than a port hazard — the same `stick_size` also drives the per-stick
  read length, so the op would already be mis-reading — and dropping the override moves the accessor
  onto the correct `aligned_page_size`. See *Questions for the user*.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding, per factory) — 12 × Case 1, 4 × clean, **no Case 2 anywhere**:

  | Factory | Binding | Delivery today | Kernel use | Case |
  |---|---|---|---|---|
  | `PadRmReaderWriterProgramFactory` | `input` | `Buffer*` reader/writer RTA slot 0 (`:143`) | `TensorAccessor(src_args, src_addr)` (`reader_pad_dims_rm_interleaved.cpp:75`) | **1** |
  | ″ | `output` | `Buffer*` slot 1 (`:144`) | `TensorAccessor(dst_args, dst_addr)` (`writer_pad_dims_rm_interleaved.cpp:32`) | **1** |
  | ″ | `pad_value_const` *(op-owned)* | `Buffer*` slot 13 (`:156`) | `TensorAccessor(pad_tensor_args, …)` (`reader_pad_dims_rm_interleaved.cpp:77`) | **1** |
  | `PadRmReaderWriterMultiCoreProgramFactory` | `input` / `output` / `pad_value_const` *(op-owned)* | `Buffer*` slots 0 / 1 / 13 (`:348,349,361`) | same three accessors | **1** ×3 |
  | `PadRmReaderWriterMultiCoreDefaultProgramFactory` | `input` | `Buffer*` reader slot 0 (`:221`) | `TensorAccessor(src_args, src_addr, page)` (`reader_..._v2.cpp:95`) | **1** |
  | ″ | `output` | `Buffer*` writer slot 0 (`:222`) | `TensorAccessor(dst_args, dst_addr, page)` (`writer_..._v2.cpp:25`) | **1** |
  | `PadRmShardedHeightOnlyProgramFactory` | `input` | borrowed CB `c_0` (`cb_src0.buffer` @ `:294`) | `dfb_in0_exp.get_write_ptr()` | **clean** |
  | ″ | `output` | borrowed CB `c_16` (`cb_output.buffer` @ `:309`) | FIFO + raw peek | **clean** |
  | `PadRmShardedWidthOnlyProgramFactory` | `input` | borrowed CB `c_0` (`cb_input.buffer` @ `:75`) | `dfb_input_shard.get_write_ptr()` | **clean** |
  | ″ | `output` | borrowed CB `c_16` (`cb_output.buffer` @ `:91`) | FIFO | **clean** |
  | `PadTileCoreProgramFactory` | `input` | `Buffer*` reader slot 0 (`:122`) | donor `TensorAccessor(src_args, src_addr)` (`reader_unary_interleaved_start_id.cpp:30`) | **1** |
  | ″ | `output` | `Buffer*` writer slot 0 (`:126`) | `TensorAccessor(dst_args, dst_addr)` (`writer_unary_pad_dims_interleaved.cpp:30`) | **1** |
  | `PadTileMulticoreProgramFactory` | `input` | `Buffer*` reader slot 0 (`:222`) | `TensorAccessor(dst_args, input_addr)` (`reader_pad_tiled.cpp:29`) | **1** |
  | ″ | `output` | `Buffer*` writer slot 0 (`:223`) | `TensorAccessor(dst_args, output_addr)` (`writer_pad_tiled.cpp:42`) | **1** |

  Every Case-1 binding today arrives via the **`Buffer*` BufferBinding form**, not a raw
  `->address()` RTA. That is the framework's interim patch-on-cache-hit mechanism, so **none of these
  is the silent-wrong hazard** — the sheet's `Smuggled pointer = no` on all 7 rows agrees with the
  code. They are still enumerated because the kernel receives a raw `uint32_t` base, and the port
  replaces the whole mechanism with a typed `TensorParameter` / `TensorBinding`.

- **TensorParameter relaxation:** `none` (all 7 rows) — nothing to apply.
- **TensorAccessor 3rd arg:** drop the redundant page-size arg at
  `reader_pad_dims_rm_interleaved_v2.cpp:95` and `writer_pad_dims_rm_interleaved_v2.cpp:25`, plus the
  two host CTAs that feed them (`..._default_program_factory.cpp:163` and `:171`, and the
  `input_accessor_page_size` / `output_accessor_page_size` computation at `:57-73`). Both Class 2 —
  **do not** set `dynamic_tensor_shape`.
- **CB endpoints:** self-loop `c_1`+`c_2` (default RM), `c_0`+`c_1` (sharded height-only), `c_0`+`c_1`
  (sharded width-only), `c_1` (tile single-core), `c_2` (tile multicore) · 1P+1C on `c_16` (sharded
  height-only) · dead-CB drop `c_1` @ `pad_tile_multicore_program_factory.cpp:70-78` (+ the dead CTA
  at `writer_pad_tiled.cpp:23`) · conditional DFB `c_2` (default RM) · all others legal.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none. No CB in this op reaches ≥3 touchers or
  doubles a FIFO role; the one two-toucher (`c_16` in the sharded height-only factory) resolves to a
  plain 1P+1C assignment.
- **Cross-op / shared kernels:** `PadTileCoreProgramFactory` file-path-instantiates
  `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`
  (`pad_tile_program_factory.cpp:104-105`). A `_metal2` fork **already exists beside it** — bind it,
  do not create a second. Details and the sunset list in *Team-only* below.
- **RTA varargs:** two genuine vararg sites, plus two look-alikes that must be **named**, not
  varargs — see *Team-only*.
- **Op-owned tensors:** the two `WorkloadDescriptor` factories each allocate a 32-element bfloat16
  pad-value const tensor and park it on `WorkloadDescriptor::buffers`
  (`pad_rm_reader_writer_program_factory.cpp:197-200`,
  `..._multi_core_program_factory.cpp:416-419`). `ProgramSpecFactoryConcept` carries these natively;
  the `WorkloadDescriptor` shape exists purely to unlock the feature, so the port drops the workload
  wrapper. **Keep holding the source `Tensor`, not just the `shared_ptr<MeshBuffer>`** — `~Tensor`
  force-deallocates through `DeviceStorage::deallocate` regardless of external `MeshBuffer` owners
  (issue #44565, cited in both factory headers).
- **Idle-core `0u` sentinel:** two factories push a literal `0u` instead of the `Buffer*` on cores
  with no work (`..._default_program_factory.cpp:224-225`,
  `pad_tile_multicore_program_factory.cpp:225-226`), to skip `BufferBinding` registration. Under
  Metal 2.0 the base rides a broadcast CRTA, so the branch has no equivalent and simply disappears;
  the kernels already short-circuit on `num_sticks_per_core == 0` / `num_pages_per_core == 0`. (In
  practice `split_work_to_cores` returns `all_cores == group_1 ∪ group_2`, so the branch is
  unreachable today anyway.)

## Team-only

### Out-of-directory coupling & donor shape

**Op-level roll-up: ✓ clean.** One in-family function-call escape, one cross-family file-path borrow.
No `CircularBuffer&` donor, no pre-Device-2.0 donor, no `uint32_t sem_*` donor (the op has no
semaphores at all).

**Summary table** — one row per (op kernel, donor file):

| Op kernel | Donor file | Donor class | Status |
|---|---|---|---|
| `reader_pad_dims_rm_interleaved_v2.cpp:13` | `ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp` | 5 — in-family shared | ✓ |
| `writer_pad_dims_rm_interleaved_v2.cpp:10` | same | 5 — in-family shared | ✓ |
| `reader_pad_tiled.cpp:8`, `writer_pad_tiled.cpp:8` | `device/kernels/dataflow/common.hpp` | in-directory — **not an escape** | ✓ |
| all kernels | `api/dataflow/*`, `api/tensor/*`, `api/core_local_mem.h`, `ckernel.h` | 1 — `tt_metal/*` LLK/HAL | ✓ no concern |

**Per-call detail** (only for the one real escape):

| Donor function | Signature shape | Verdict |
|---|---|---|
| `tt::data_movement::common::noc_async_read_sharded(Noc, uint32_t l1_addr, AddrGenType tensor, uint32_t src_id, uint32_t offset, uint32_t size)` (`common.hpp:375`) | `AddrGenType` is instantiated with `TensorAccessor<DSpec>` → **Shape 1** | ✓ excellent — porter constructs `TensorAccessor(tensor::name)` and passes it |
| `tt::data_movement::common::noc_async_write_sharded(...)` (`common.hpp:325`) | same — **Shape 1** | ✓ excellent |

Both callers already use the non-deprecated leading-`Noc` overload
(`reader_..._v2.cpp:45`, `writer_..._v2.cpp:48`); the deprecated no-`Noc` overloads exist in the donor
but are not reached from pad. No donor-side change, no fork.

**Borrowed kernel files (file-path instantiation).** Exactly one:

- **Path:** `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`
- **Owning family:** `eltwise/unary`
- **Broadly shared** — not a one-off borrow. In-tree consumers of the legacy file (the **sunset
  list**, i.e. who must migrate before the legacy copy can be deleted — *not* a bundled-port
  assignment):
  - `data_movement/pad` → `pad_tile_program_factory.cpp` *(this op)*
  - `data_movement/untilize_with_unpadding` → `untilize_with_unpadding_multi_core_interleaved_program_factory.cpp`, `untilize_with_unpadding_single_core_program_factory.cpp`
  - `examples/example` → `multi_core_program_factory.cpp`, `single_core_program_factory.cpp`
  - `examples/example_multiple_return` → `single_core_program_factory.cpp`
  - `experimental/transformer/nlp_create_qkv_heads_falcon7b` → `nlp_create_qkv_heads_falcon7b_program_factory.cpp`
  - `reduction/topk` → `topk_route_prep_program_factory.cpp`
  - plus `tests/ttnn/unit_tests/gtests/test_generic_op.cpp` and `tests/ttnn/unit_tests/operations/debug/test_generic_op.py`
- **`_metal2` fork already exists beside it:** `reader_unary_interleaved_start_id_metal2.cpp`, same
  directory. **Bind it; do not create another.** Its file header states the binding names are its
  interface and are not renamed once a consumer exists — the factory conforms to the kernel, never
  the reverse. Vocabulary: `dfb::in`, `tensor::src`, `args::num_pages`, `args::start_id`; the fork
  also already replaces `get_local_cb_interface(cb).fifo_page_size` with `dfb.get_entry_size()`.
- **Trap:** a *second*, differently-named fork of the same kernel exists at
  `ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/dataflow/reader_unary_interleaved_start_id_metal2.cpp`
  (vocabulary `tensor::input`, not `tensor::src`). The locational test — same stem, `_metal2` suffix,
  **same directory as the original** — selects the `eltwise/unary` one. Do not bind the typecast copy.

### RTA varargs

Two genuine vararg sites, and two look-alikes the porter should **name** instead:

- **Genuine vararg (a) — `reader_pad_dims_rm_sharded.cpp:17-22`** (`PadRmShardedHeightOnlyProgramFactory`).
  The arg block is `num_cores`, then `2·num_cores` NoC x/y values, then `num_cores` chunk counts, then
  `2·Σchunks` `(start_id, length)` pairs — read through `get_arg_addr()` at **runtime-computed**
  offsets (`get_arg_addr(1 + num_cores_read * 2)`, `… * 3`), with per-core counts driving the loops at
  `:38,44`. Host side at `pad_rm_sharded_height_only_program_factory.cpp:158-186`. There are no
  per-argument names to infer — use the RTA vararg mechanism.
- **Genuine vararg (a) — `reader_pad_tiled.cpp:22-25` and `writer_pad_tiled.cpp:35-38`**
  (`PadTileMulticoreProgramFactory`). Four consecutive `num_dims`-long blocks (`input_page_shape`,
  `output_page_shape`, `input_id_per_dim`, `output_id_per_dim`) reached by `get_arg_addr(rt_ind)` plus
  `+ num_dims` pointer strides, consumed in `for (d < num_dims)` loops (`:46`, `:66`, and
  `common.hpp:12`). `num_dims` is CTA 2 = `output_padded_shape.rank()`. A CTA-bounded loop still
  varies across instantiations, so this is a vararg by the recipe's rule. Host side at
  `pad_tile_multicore_program_factory.cpp:236-253`. *Caveat worth knowing:* `pad_impl` hard-asserts
  rank == 4 (`pad.cpp:193`) and `validate_on_program_cache_miss` caps rank at 4
  (`pad_device_operation.cpp:121`), so `num_dims` is 4 in every reachable configuration — naming the
  16 values would also be defensible. Flagged as vararg per the rule; the porter may take the named
  route if the kernel's rank-generic loops are simultaneously pinned.
- **Look-alikes — name these, don't varargs them.**
  `reader_pad_dims_rm_interleaved_v2.cpp:59` and `writer_pad_dims_rm_sharded.cpp:53` both take
  `start_dim_offset = get_arg_addr(k)` and then read only **fixed** indices `[1]`, `[2]`, `[3]`
  (`reader_..._v2.cpp:112`, `writer_..._sharded.cpp:93`). That is three distinct nameable fields
  (`start_h`, `start_c`, `start_n`) reached through legacy pointer arithmetic, not a data-directed
  pick — ordinary named RTAs.
- **CTA varargs: none.** No kernel reads `get_compile_time_arg_val()` at a varying index. The
  `kernel_compile_time_args[13]` / `[10..12]` reads (`reader_..._v2.cpp:85`,
  `writer_..._sharded.cpp:70-72`) use **constant** indices inside `if constexpr (not_pad_by_zero)`
  blocks, and the host emits those CTA slots unconditionally
  (`..._default_program_factory.cpp:141-163`, `..._height_only_program_factory.cpp:337-350`), so they
  are a fixed named set on both branches.

### Relaxation candidates

None. No custom `compute_program_hash` exists anywhere in the op, so there is no hash to mine for
tensor properties the op actually depends on. The sheet's `TensorParameter relaxation = none` on all
seven rows stands unchallenged.

### TTNN factory analysis

Sheet-derived facts with `file:line` evidence:

- **Op-owned tensors — yes, on two factories.** `wd.buffers.push_back({pad_value_owner, pad_value_const_buffer})`
  at `pad_rm_reader_writer_program_factory.cpp:200` and
  `pad_rm_reader_writer_multi_core_program_factory.cpp:419`. The owned object is a device-resident
  32-element bfloat16 L1 tensor built by `build_pad_value_const_tensor_sc` / `_mc`
  (`:25-36` / `:165-178`).
- **MeshWorkload need — an op-owned-tensor artifact, not genuine multi-program.** Both factories build
  **one** `ProgramDescriptor` and replicate the *same* object across `tensor_coords.ranges()`
  (`:202-211` / `:421-429`). Nothing is per-mesh-coordinate. The `create_workload_descriptor` entry
  point exists solely because the `descriptor` form cannot carry op-owned tensors.
- **Pybind `create_descriptor` — none.** `pad_nanobind.cpp:41-75` binds only the two public
  `ttnn::pad` overloads; no `nb::class_` of the device op, no descriptor internals exposed. Nothing
  for the port to delete, and no user-visible API change from this axis.
- **Custom hash — none** (and no backdoor `attribute_values` / `to_hash`). The default hash over
  `PadParams` + tensor specs applies.
- **`get_dynamic_runtime_args` — none** (removed by #52556). See the *Gate detail* conflict.
- **`override_runtime_arguments` — one**, at `pad_rm_sharded_height_only_program_factory.cpp:412`.
  It rebuilds a two-entry CB-address-only `ProgramDescriptor` mirroring `create_descriptor`'s CB push
  order positionally (input CB, output CB, then the unbound pad-value CB it deliberately omits) and
  calls `apply_descriptor_runtime_args`. This selects `CustomProgramSpecFactoryConcept` for that
  factory, and the porter translates the method into one returning a `ProgramRunArgs`. The
  positional-mirroring contract is fragile and worth carrying into the translation as an explicit
  comment. **Open question for the porter:** the method exists solely to re-point the two borrowed CB
  base addresses, which is precisely what `DataflowBufferSpec::borrowed_from` plus a `TensorBinding`
  does natively in Metal 2.0 — so the body may be wholly subsumed. If it is, the factory drops to the
  plain `ProgramSpecFactoryConcept`; that is a concept change, so raise it rather than deciding it
  silently. (`PadRmShardedWidthOnlyProgramFactory` has the same two borrowed CBs and no override,
  which is the evidence that the override may be unnecessary — see *Misc anomalies* #8.)
- **Target concepts.** `ProgramSpecFactoryConcept` for six factories — two of them
  (`PadRmReaderWriterProgramFactory`, `PadRmReaderWriterMultiCoreProgramFactory`) additionally
  carrying op-owned tensors. `CustomProgramSpecFactoryConcept` for
  `PadRmShardedHeightOnlyProgramFactory`, per the code (the sheet's `Override runtime args method?`
  cell is outdated), subject to the open question above.

## Misc anomalies  *(team-only, non-gating; the port does not act on these)*

1. **An entire factory is unreachable.** `PadRmReaderWriterMultiCoreProgramFactory` (433 lines) is
   declared in the `program_factory_t` variant (`pad_device_operation.hpp:33`) but is **never
   returned by `select_program_factory`** — the RM multicore path returns
   `PadRmReaderWriterMultiCoreDefaultProgramFactory` instead (`pad_device_operation.cpp:99-101`), and
   the only other reference in the tree is its own definition. It allocates an op-owned device tensor
   on every (hypothetical) cache miss and carries a hardcoded resnet-shape core split. Candidate for
   deletion; see *Questions*.
2. **Dead code after `TT_THROW` in that same factory.** `split_across_cores`'s `default:` branch
   throws at `..._multi_core_program_factory.cpp:97` and is followed by ~30 lines of unreachable
   "generic case -- TODO" logic (`:99-131`), including a second `TT_THROW` and commented-out
   assignments.
3. **Dead CB + dead CTA (tile multicore).** `c_1` allocated at
   `pad_tile_multicore_program_factory.cpp:70-78`; its index reaches `writer_pad_tiled.cpp:23` as
   `output_cb_id` and is never used. Wastes `2 × page_size` bytes of L1 per core. *(Also recorded as
   PORT WORK — the port drops it.)*
4. **Hardcoded constant overriding a live RTA.** `reader_pad_dims_rm_interleaved.cpp:52` hardcodes
   `pad_value_const_buffer_nbytes = 64` with the comment *"assumed to be 64 bytes, fails on BH when
   > 64. TODO: generalize? (Issue #21978)"*, while the host still computes and passes the real value
   at RTA slot 14 (`pad_rm_reader_writer_program_factory.cpp:157`). The passed value is silently
   ignored, so the arg is dead **and** the kernel is pinned to an assumption the host does not
   enforce.
5. **Several dead RTAs in the v1 RM kernels.** The reader and writer are handed *identical* 27-slot
   arg lists (`pad_rm_reader_writer_program_factory.cpp:172`, `..._multi_core_...:385`), so each
   receives roughly half it never reads. Beyond that, values that *are* unpacked and then unused:
   reader `num_total_W` (3), `num_total_Y` (7), `start_src_stick_wi` (18), `full_unpadded_X_nbytes`
   (23); writer `num_total_W` (3), `num_total_Y` (7), `num_total_X` (9), `num_local_unpadded_Y` (22),
   `full_padded_X_nbytes` (24), and `dst_stick_wi` (19, assigned at `:36` and never advanced).
6. **Dead compile-time args in the v1 writer.** `writer_ct_args = reader_ct_args`
   (`pad_rm_reader_writer_program_factory.cpp:85`) copies in the pad-value-const tensor's
   `TensorAccessorArgs`, which the writer never instantiates. Harmless but misleading — and the
   writer must still declare `src_args` (`writer_pad_dims_rm_interleaved.cpp:26`) purely to compute
   `dst_args`' offset.
7. **`start_dim_offset` seeded at the wrong length.** In both
   `..._default_program_factory.cpp:199-200,264` and
   `..._height_only_program_factory.cpp:65-66,116` the vector is initialized to `num_dims` zeros but
   reassigned to a literal `{0, curr_h, curr_c, curr_n}` (always 4) after the first core. Benign today
   because rank is pinned to 4 upstream (`pad.cpp:193`), but it silently breaks the rank-generic
   intent the initialization advertises.
8. **Sharded-factory asymmetry in cache-hit CB re-pointing.** `PadRmShardedHeightOnlyProgramFactory`
   defines an `override_runtime_arguments` specifically to re-point its two borrowed CB addresses
   (`..._height_only_program_factory.cpp:412-424`), while `PadRmShardedWidthOnlyProgramFactory` —
   which has the same two borrowed CBs — defines none and relies on the framework's `cb.buffer`
   patching (comment at `..._width_only_program_factory.cpp:170-173`). One of the two is doing
   unnecessary work, or the other has a latent stale-address hole on cache hits. Worth an ops-team
   look independent of the port.
9. **Unguarded read of a possibly-unallocated CB.** `reader_pad_dims_rm_interleaved_v2.cpp:99` calls
   `get_read_ptr()` on `c_2` unconditionally although the host allocates `c_2` only under a condition
   (`..._default_program_factory.cpp:127-139`). Harmless today; blocks a clean conditional-DFB port.
   *(Also recorded as PORT WORK.)*

## Per-DeviceOperation attribution

Single `DeviceOperation` (`PadDeviceOperation`); no bundling. Per-**factory** attribution — which is
where this op's findings actually differ — is carried inline throughout:

| Factory | Gate verdict | Target concept | In the port |
|---|---|---|---|
| `PadRmReaderWriterProgramFactory` | clear | `ProgramSpecFactoryConcept` + op-owned tensors | yes |
| `PadRmReaderWriterMultiCoreProgramFactory` | clear | `ProgramSpecFactoryConcept` + op-owned tensors | yes (but unreachable — see *Questions*) |
| `PadRmReaderWriterMultiCoreDefaultProgramFactory` | clear | `ProgramSpecFactoryConcept` | yes |
| `PadRmShardedHeightOnlyProgramFactory` | clear — sheet cell outdated, overridden on code evidence | `CustomProgramSpecFactoryConcept` | yes |
| `PadRmShardedWidthOnlyProgramFactory` | clear | `ProgramSpecFactoryConcept` | yes |
| `PadTileCoreProgramFactory` | clear | `ProgramSpecFactoryConcept` | yes |
| `PadTileMulticoreProgramFactory` | clear | `ProgramSpecFactoryConcept` | yes |

## Questions for the user

1. **Who corrects the readiness-sheet row, and when?** *(Does not block the port — recorded so it
   does not get lost.)* `PadRmShardedHeightOnlyProgramFactory`'s row still describes the pre-#52556
   code, so `Is able to port?` reads `no`. This audit overrides it, but the row itself will keep
   misleading the port tracker and the next auditor until
   `Runtime-args update (get_dynamic_runtime_args)` → `no` and
   `Override runtime args method? (PD only)` → `yes`.
2. **Should `PadRmReaderWriterMultiCoreProgramFactory` be deleted rather than ported?**
   (`pad_device_operation.hpp:33`, `pad_rm_reader_writer_multi_core_program_factory.cpp`.) It is
   unreachable from `select_program_factory`, and porting 433 lines of dead multicore code — including
   an op-owned-tensor allocation and a hardcoded resnet core split — spends porter effort on something
   no test can exercise. If it should go, deleting it *before* the port is cleaner than porting and
   then deleting. If it is being kept deliberately (a path someone intends to re-enable), say so and
   it stays in scope.
3. **Can a RM input reach `pad` with last-dim padding (`padded_shape[-1] > logical_shape[-1]`)?**
   `PadRmReaderWriterMultiCoreDefaultProgramFactory` derives `stick_size` from the **logical** last
   dim (`:39-40,47`) while taking its H/C/N bounds from the **padded** shape (`:197-198`). If such an
   input is reachable, the op already mis-reads (the same `stick_size` is both the accessor page and
   the per-stick read length) — a pre-existing bug for the ops team, not a port hazard. If it is not
   reachable, the mixed use is merely confusing. Either answer leaves the 3rd-arg site Class 2.

## Recipe notes

1. **`Secretly SPMD Workload?`'s code basis doesn't survive a multi-range mesh.** The recipe says
   *"a **single entry** in its `programs` vector ⇒ SPMD"*
   ([TTNN factory concept prerequisite](#) cross-check list). Pad's two `WorkloadDescriptor`
   factories push **one entry per range in `tensor_coords`** — `ranges.size()` entries, each carrying
   the *identical* `ProgramDescriptor` built once above the loop
   (`pad_rm_reader_writer_program_factory.cpp:205-211`). On a single-range mesh that is one entry and
   the test passes; on a multi-range one the test would say "not SPMD" about code that is plainly
   SPMD. Suggest restating the basis as *"every entry carries the same `ProgramDescriptor`"* rather
   than *"one entry."* I followed the substance (and the sheet) rather than the literal test.
2. **The recipe has no verdict for "the sheet is simply out of date," and this audit needed one.**
   *(Flagged prominently: this report deviates from the recipe as written. The deviation was the
   report owner's explicit decision, taken after the auditor recommended against it; recorded here so
   the maintainer sees the pressure the rule is under, not to claim the rule was followed.)*
   The recipe is unambiguous that `Is able to port?` is the gate — *"a `no` you cannot account for is
   still a `no`"* — and that a primary-column conflict means *"stop rather than proceed on data we
   can't trust,"* routed to the readiness-sheet owner. Applied literally, that RED's this op on a
   cell whose only visible input was fixed on `main` five days earlier by a PR the auditor can name
   (#52556 / `90ec10f4bf4`). Every gate the audit checks *itself* clears for that factory. The report
   owner judged that gating a port on a bookkeeping lag was not worth the cycle, and directed a GREEN
   with the staleness recorded; this document reflects that.
   The friction worth fixing: the recipe's two failure modes for a `no` are *"explained by a blocking
   column"* (→ that column's owner) and *"primary-column conflict"* (→ sheet owner, stop). Neither
   fits **"the blocking column is the conflicting column, and the code is already correct."** That
   case has no honest home today, so it lands as a stop-the-world RED for what is a two-cell edit.
   Suggest a third, explicitly-bounded outcome — *"stale-cell override"*: permitted only when the
   contradicted column is one of the cross-checked primaries, the auditor can cite the landing commit
   that made it stale, **and** every other gate-bearing subject clears independently; the report then
   reads GREEN, states the override, and still routes the correction to the sheet owner. Without
   something like that, auditors facing this shape will either over-block or quietly do what this
   report did without labelling it.
3. **The Red-scoping exception list doesn't mention spreadsheet-broken.** The *"Elsewhere → run
   them"* branch enumerates *"an unattributed or held readiness verdict, an Appendix A feature
   landing, any framework capability not yet released."* A stale/broken sheet row is the clearest
   possible "clears with the op untouched" case — the code is already correct — yet it isn't named,
   so a reader has to reason it in by analogy. Suggest adding it explicitly. (Moot for this report,
   which ran every subject in full, but it is the rule the auditor reached for first.)
4. **The conditional-DFB guidance stops at the host side.** [CB endpoints](#) covers *dead in some
   configs, live in others* by making the `DataflowBufferSpec` conditional, and warns that the legacy
   factory "allocates every CB unconditionally and gates only the *kernel-side* use behind an
   `#ifdef` or a branch." Pad's `c_2` is the mirror image: the **host** allocates conditionally and
   the **kernel** references it unconditionally
   (`reader_pad_dims_rm_interleaved_v2.cpp:99` vs `..._default_program_factory.cpp:127-139`). Making
   the spec conditional is then not sufficient — the kernel line needs gating too, and `if constexpr`
   will not do it because it does not suppress `dfb::` name lookup. Worth a sentence, since this is
   the shape that silently produces an unbound-DFB reference at build time.
5. **Minor: the audit template has no natural home for "the op has no compute kernels."** It matters
   (it retires the compute `opt_level` question wholesale), but it isn't a row in the status summary
   or a listed subject, so I filed it under the Device 2.0 bullet.
