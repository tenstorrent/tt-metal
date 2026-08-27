# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/data_movement/concat`

One DeviceOperation, six program factories (all in `device/`):

- **`ConcatDeviceOperation`** (`concat_device_operation.hpp` / `.cpp`)
  - `ConcatProgramFactory` (`concat_program_factory.cpp`) — interleaved inputs; also the ND-sharded fallback
  - `ConcatS2STiledProgramFactory` (`concat_s2s_tiled_program_factory.cpp`) — sharded→sharded, TILE, 2 inputs, dim 3
  - `ConcatS2SRMProgramFactory` (`concat_s2s_rm_program_factory.cpp`) — sharded→sharded, ROW_MAJOR, 2 inputs, dim 3
  - `ConcatS2SMultiProgramFactory` (`concat_s2s_multi_program_factory.cpp`) — sharded→sharded, dim 2 or 3, N inputs
  - `ConcatS2IProgramFactory` (`concat_s2i_program_factory.cpp`) — **dead code** (see Misc anomalies)
  - `ConcatBlockShardedProgramFactory` (`concat_block_sharded_program_factory.cpp`) — block-sharded, ≤16 inputs

All nine kernel files under `device/kernels/` are referenced by a factory; there are no unreferenced
kernel files. One factory references a kernel source that **does not exist** — see Misc anomalies #1.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `metal2_audit.md`.

**Recipe docs:** `0846547f407 2026-08-27 docs(metal_2.0): a run in flight freezes the kernel sources`

**Readiness sheet:** fetched fresh this session from the live *"Operations analysis"* sheet
(file ID `1KUMj8SyBGlNMZlLFgs1MbAZlO2g6EoUc4KaxSlcy8jw`); 6 rows for `data_movement/concat`.

> ### ⚠ One sheet cell is stale — read this before re-fetching the sheet
>
> The live readiness sheet carries `Known op issues` = `Awaiting variadic tensor in Metal 2.0` and
> `Is able to port?` = `no` on the **`ConcatProgramFactory`** row. **That variadic-tensor support has
> merged**, and the sheet owner has confirmed that clearing the `Known op issues` cell makes the row
> green. This audit therefore reads that row as `Known op issues` **empty**, `Is able to port?` =
> **`yes`**, and counts `ConcatProgramFactory` in the clear subset.
>
> **If you re-fetch the sheet and still see `no` on this row, that is the stale cell — not a new
> block.** Every other column on it is already clean and code-confirmed (see the cross-check table
> under Gate detail).
>
> **Action for the sheet owner:** refresh the cell, so the next auditor of this op does not re-gate a
> factory that is now portable.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/concat` |
| **Overall** | **RED at op level; subset `ConcatProgramFactory` + `ConcatS2SRMProgramFactory` + `ConcatS2STiledProgramFactory` is clear** |
| **DOps / Factories** | `ConcatDeviceOperation` → 6 factories (3 clear, 3 blocked) |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all 11 in-scope kernels (9 own + 2 donors) are structurally Device 2.0; zero Device 1.0 idioms; the only CB-index free functions found are both on the sanctioned list |
| *Prereqs* — Cross-op escapes | Ok — zero function-call escapes (every kernel include resolves under `tt_metal/hw/inc/api/`); two file-path borrows in `ConcatProgramFactory`, **both with a `_metal2` fork already checked in** (rung 1 — reuse, don't fork) |
| *Feature Support* — overall | **GREEN** — all three Appendix A entries `N/A` |
| *Feature Support* — Variadic-CTA | *(no such Appendix A entry exists — see Recipe notes #3)* |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Split: 3 × `yes`, 3 × `no`** — each `no` attributed to `Known op issues` (verbatim values in Gate detail). `ConcatProgramFactory` counted as `yes` per the stale-cell box above. |
| *TTNN Readiness* — Concept (current) | `descriptor` — all six factories (code-confirmed: each defines `static ProgramDescriptor create_descriptor(...)`) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A — concept is `descriptor`, not `WorkloadDescriptor` |
| *TTNN Readiness* — Custom hash | No — no `compute_program_hash`, no `attribute_values` / `to_hash` anywhere in the op |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No — hook absent from the device-op |
| *TTNN Readiness* — `override_runtime_arguments` | No — method absent from all six factories |
| *TTNN Readiness* — Pybind `create_descriptor` | No — `concat_nanobind.cpp` binds only the public `ttnn::concat`; no `nb::class_` of the device op |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | **`ProgramSpecFactoryConcept`** (matches the sheet's own `Porting Target` column) |
| *Port work* — Offset base pointer | **none** — **zero** `->address()` expressions exist anywhere in the op, so no host-folded offset is reachable |
| *Port work* — Tensor bindings (per binding) | `ConcatProgramFactory`: **N+1 × Case 1** · S2SRM + S2STiled: all **clean** (borrowed-memory DFB) · gated factories: **clean** |
| *TTNN Readiness* — TensorParameter relaxation | `none` on all six rows → clears |
| *Port work* — TensorAccessor 3rd arg | **N/A** — no accessor in the op passes a 3rd argument (the subject never fires) |
| *Port work* — CB endpoints | legal / self-loop / 1P+1C — no multi-binding, no dead CB (per-`(CB, config)` inventory below) |
| *Port work* — RTA varargs | `ConcatProgramFactory`: **one RTA vararg block + one CTA vararg** · S2SRM + S2STiled: **none** (zero runtime args) |

**CB endpoints** are dispositions, not gates: every out-of-window CB here resolves at port time via a
**self-loop** (one toucher) or a **1P+1C assignment** (two touchers). No CB in any factory reaches
≥3 distinct touchers or doubles a FIFO role, so the multi-binding advanced option is **not needed
anywhere in this op**. No dead CBs. Dispositions are recorded per `(CB, config)`.

## Result

**RED at op level; subset `ConcatProgramFactory` + `ConcatS2SRMProgramFactory` +
`ConcatS2STiledProgramFactory` is clear.**

Every gate this audit owns is **GREEN for the whole op** — Device 2.0, feature compatibility, offset
base pointers, TensorAccessor 3rd argument, and the relaxation column all clear. The op is RED solely
on the **TTNN factory concept prerequisite**, and only for **3 of its 6 factories**, each blocked by a
`Known op issues` entry on the readiness sheet:

| Factory | `Is able to port?` | `Known op issues` (verbatim) | Routed to |
|---|---|---|---|
| `ConcatProgramFactory` | `yes` *(promoted — see stale-cell box)* | ~~`Awaiting variadic tensor in Metal 2.0`~~ — **feature merged** | — **clear** |
| `ConcatS2SRMProgramFactory` | `yes` | *(blank)* | — **clear** |
| `ConcatS2STiledProgramFactory` | `yes` | *(blank)* | — **clear** |
| `ConcatS2SMultiProgramFactory` | `no` | `DFB misuse; will need semi-manual port` | **ops team / porting-recipe** — confirmed in code, `reader_s2s_tensor_concat.cpp:30` |
| `ConcatBlockShardedProgramFactory` | `no` | `DFB misuse; will need semi-manual port` | **ops team / porting-recipe** — confirmed in code, `reader_writer_block_sharded_concat.cpp:44` |
| `ConcatS2IProgramFactory` | `no` | `... this factory is dead code` | **ops team** — delete the factory (independently confirmed twice; Misc anomalies #1, #2) |

A brief **is** issued, scoped strictly to the three clear factories, per the config-scoped-GATE carve-out
in the finding-roles section of `metal2_audit.md` ("A *config-scoped* GATE … still issues a brief for the
clean subset"). This conflicts with two other lines in the same recipe — see Recipe notes #2.

**Path forward for the three still blocked.** Neither remaining blocker is permanent, and the two
`DFB misuse` factories are the same code shape twice — a `DataflowBuffer` constructed from a
runtime-selected index, which has no static `dfb::name` to bind to. `ConcatS2IProgramFactory`'s fix is
deletion, not a port.

**Note on scope after the promotion.** The clear subset now spans the op's **default** factory
(`ConcatProgramFactory` handles all interleaved inputs and the ND-sharded fallback) plus its two
two-tensor sharded specializations. This is a materially larger port than the pre-promotion subset —
`ConcatProgramFactory` is the only factory in the op with tensor bindings that need real work, the only
one with varargs, and the only one with borrowed donor kernels. The other two remain small.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **Split — RED on 3 of 6 factories.** The blocking column
  is `Known op issues` in every case; the per-factory cell values and routing are in the Result table
  above. The lightweight cross-check came back **clean on every primary column** — see below. No
  cross-column invariant is violated (`get_dynamic_runtime_args` is `no` on a `descriptor` concept;
  `Op-owned tensors?` is not `yes`). The **factory-set match** is exact: the sheet's 6 rows map
  one-to-one onto the 6 factory structs in `concat_device_operation.hpp:29-35`, names identical — no
  phantom row, no missing row. The sheet is **not** broken for this op; the one cell this audit
  overrides is overridden by instruction, not by conflicting evidence.

  | Sheet column | Sheet value | Code evidence | Agree? |
  |---|---|---|---|
  | `Concept` | `descriptor` (×6) | `static ProgramDescriptor create_descriptor(...)` in each of the 6 `device/*_program_factory.hpp:13-14` | ✓ |
  | `Custom hash (compute_program_hash)` | `no` (×6) | no match for `compute_program_hash` in the op tree | ✓ |
  | `Backdoor custom hash (attribute_values / to_hash)` | `no` (×6) | no match for `attribute_values` / `to_hash` | ✓ |
  | `Runtime-args update (get_dynamic_runtime_args)` | `no` (×6) | hook absent from `concat_device_operation.hpp` | ✓ |
  | `Override runtime args method?` | `no` (×6) | no `override_runtime_arguments` in the op tree | ✓ |
  | `Pybind descriptor (nb::class_ of device op)` | `no` (×6) | `concat_nanobind.cpp:39-49` binds only `ttnn::concat` via `bind_function` | ✓ |
  | `Smuggled pointer (raw buffer addr in RTA/CRTA)` | `no` (×6) | zero `->address()` sites; pointers travel as `Buffer*` (framework-registered) — consistent with `Op Classification` = `PD Op (pointer-patching)` | ✓ |
  | `TensorParameter relaxation` | `none` (×6) | n/a (read, not re-derived) | — |
  | Factory set | 6 rows | 6 factory structs, names match exactly | ✓ |

- **Device 2.0 (every kernel used):** **GREEN.** All 11 kernels the op exercises were read in full. Zero
  Device 1.0 idioms: no `noc_async_read`/`noc_async_write`/`*_barrier` free functions, no
  `InterleavedAddrGen` / `ShardedAddrGen` / `InterleavedAddrGenFast` / `InterleavedPow2AddrGen*`, no
  `get_noc_addr_from_bank_id`, no raw semaphore addresses (the op uses **no semaphores at all**). Every
  kernel is on `Noc` + `DataflowBuffer` (or the Device 2.0 `CircularBuffer` wrapper) + `CoreLocalMem` +
  the endpoint types.

  Kernels audited (own + donors):

  | Kernel | Bound by | Device 2.0 idioms present |
  |---|---|---|
  | `device/kernels/dataflow/reader_concat_interleaved_start_id.cpp` | `ConcatProgramFactory` (TILE) — **clear** | `Noc`, `DataflowBuffer`, `CoreLocalMem`, `TensorAccessor` |
  | `device/kernels/dataflow/reader_concat_stick_layout_interleaved_start_id.cpp` | `ConcatProgramFactory` (RM) — **clear** | same |
  | `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp` | `ConcatProgramFactory` (RM) — **donor, shared pool** | `Noc`, Device 2.0 `CircularBuffer` wrapper, `TensorAccessor` |
  | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | `ConcatProgramFactory` (TILE) — **donor, cross-family (`eltwise`)** | `Noc`, `DataflowBuffer`, `TensorAccessor` |
  | `device/kernels/dataflow/reader_height_sharded_width_concat_two_tensors.cpp` | `ConcatS2SRMProgramFactory` (×2 instances, ×2 core groups) — **clear** | `Noc`, `DataflowBuffer`, `CoreLocalMem`, `UnicastEndpoint`, `async_read_with_state` |
  | `device/kernels/dataflow/reader_height_sharded_width_concat_two_tensors_tiled.cpp` | `ConcatS2STiledProgramFactory` — **clear** | same |
  | `device/kernels/dataflow/writer_height_sharded_width_concat_two_tensors_tiled.cpp` | `ConcatS2STiledProgramFactory` — **clear** | same |
  | `device/kernels/compute/height_sharded_width_concat_two_tensors.cpp` | `ConcatS2STiledProgramFactory` — **clear** | `DataflowBuffer` + `api/compute/*` |
  | `device/kernels/dataflow/reader_s2s_tensor_concat.cpp` | `ConcatS2SMultiProgramFactory` (×2 instances) — gated | `Noc`, `DataflowBuffer`, `CoreLocalMem`, `UnicastEndpoint` |
  | `device/kernels/dataflow/reader_writer_block_sharded_concat.cpp` | `ConcatBlockShardedProgramFactory` (×2 instances) — gated | same |
  | `device/kernels/dataflow/writer_s2i_width.cpp` | `ConcatS2IProgramFactory` (dead) — gated | `Noc`, `DataflowBuffer`, `TensorAccessor` |

  **Two CB-index free functions found — both sanctioned, neither flagged.** Note that both now sit in
  the **clear** subset, so the porter will meet them:

  | File | Line | Call | Wrapper in scope | Verdict |
  |---|---|---|---|---|
  | `device/kernels/dataflow/reader_concat_interleaved_start_id.cpp` | 28 | `get_tile_size(cb_id_in)` | yes — `DataflowBuffer dfb_in` (line 45) | **sanctioned — not a violation** |
  | `.../eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | 27 | `get_local_cb_interface(cb_id_out).fifo_page_size` | yes — `DataflowBuffer dfb` (line 30) | **sanctioned — not a violation** |

  Both are on the recipe's sanctioned list, and both are the exact case the recipe warns misfires
  hardest: a `DataflowBuffer` is in scope and does expose its own `get_tile_size()`, but *sanctioned
  still means sanctioned* — the Device 2.0 surface, not the shape of the call site, is the test. The
  Device 2.0 boundary is unmoved. Moving these lookups onto the DFB object is a *port*-stage change
  under the kernel-side whitelist; the second one is already done for the porter — the `_metal2` fork of
  the eltwise donor replaces it with `dfb.get_entry_size()` (fork line 37).

- **Feature compatibility:** all three Appendix A entries scanned against all six factories, all nine
  own kernels, both donor kernels, and the host-facing `concat.cpp` / `concat_nanobind.cpp`.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | **N/A** | No signal of any kind: no `GlobalCircularBuffer` type, no `CreateGlobalCircularBuffer`, no `global_circular_buffer` field on any of the 13 `CBDescriptor` literals, no `remote_index()` / `remote_cb_*` / `num_global_cb_receivers`, no `<tt-metalium/global_circular_buffer.hpp>` include (either spelling). |
  | CBDescriptor `address_offset` (non-zero) | **N/A** | No `CBDescriptor` in any factory sets `.address_offset` (all 13 literals default it to 0). No `set_address_offset`, no `cb_descriptor_from_sharded_tensor`, no call to `UpdateDynamicCircularBufferAddress` in any form. **False-positive guard applied:** the single textual match in the op — `device/kernels/dataflow/reader_writer_block_sharded_concat.cpp:21` — is a *code comment* explaining why the kernel passes `(cb_id, offset)` pairs rather than absolute L1 addresses. It names `UpdateDynamicCircularBufferAddress` in prose; it is not a call, and no offset is delivered to a CB. |
  | GlobalSemaphore | **N/A** | No signal. The op declares **no semaphores at all** — no `SemaphoreSpec`-equivalent, no `CreateSemaphore`, no `GlobalSemaphore`, no `<tt-metalium/global_semaphore.hpp>`. |

  A clean scan is all-`N/A`; the subject verdict is **GREEN — no gate fired**.

- **Offset base pointers:** **GREEN — clean scan, not merely "unlisted."** The op contains **zero**
  occurrences of `->address()`, `.address()`, or `(*buffer).address()` — verified by grep over the whole
  op directory. With no address expression anywhere on the host side, there is no site into which a
  host offset could be folded, so Types 1 and 2 are structurally unreachable. Type 3 (`address_offset`)
  is `N/A` per the Appendix A row above. Type 4 (`ttnn::narrow` / interior-base `MeshBuffer::create`)
  does not appear.

  Every tensor base in this op reaches a kernel by one of two routes, neither of which carries an
  offset:
  1. **A borrowed-memory DFB** — `CBDescriptor::buffer = tensor.buffer()`, base only, no offset field
     set (all factories except `ConcatProgramFactory`).
  2. **A `Buffer*` pushed into an RTA list** — the pointer *object*, not its address; the framework
     resolves and patches it (`ConcatProgramFactory` `concat_program_factory.cpp:276, 285, 290`;
     `ConcatS2IProgramFactory` `concat_s2i_program_factory.cpp:83`).

  **Reconciliation against the dated triage** (`analyses/2026-07-19_offset_base_pointers.md`): concat is
  **not** in its op→type tables. Crossed with my scan, this is the *"no fold, op not in the tables"*
  outcome → clean; every address site hands off to TensorParameter analysis. The doc is a prior only;
  the verdict above rests on the grep and on reading all six factories, not on the doc's silence.

  *(Kernel-side offsets abound — `output_stick_offset`, `input_write_offset`, `src_l1_offset`,
  `dst_offset` — but every one is an offset **into an L1 CB**, added to a `get_write_ptr()` /
  `get_read_ptr()` result. None is added to a device-buffer base, and none crosses the host/kernel
  boundary as a pre-folded pointer. These are not this subject.)*

- **TensorAccessor 3rd argument:** **N/A — no accessor in the op passes a 3rd argument.** The subject
  never fires. Stated as *no sites*, not as *sites classified Class 2* — the two are different findings.

  Every accessor construction in the op and in both donor kernels was located and inspected:

  | Site | Form | Args |
  |---|---|---|
  | `device/kernels/dataflow/reader_concat_interleaved_start_id.cpp:36` | `make_tensor_accessor_tuple(args, 3)` | 2 (see below) |
  | `device/kernels/dataflow/reader_concat_stick_layout_interleaved_start_id.cpp:35` | `make_tensor_accessor_tuple(args, 3)` | 2 (see below) |
  | `.../kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp:25` | `TensorAccessor(dst0_args, dst_addr)` | 2 |
  | `.../eltwise/unary/.../writer_unary_interleaved_start_id.cpp:39` | `TensorAccessor(dst_args, dst_addr)` | 2 |
  | `device/kernels/dataflow/writer_s2i_width.cpp:24` | `TensorAccessor(dst_args, dst_addr)` | 2 |

  The two `make_tensor_accessor_tuple` sites are the only indirect ones, and they were resolved rather
  than assumed: `tt_metal/hw/inc/api/tensor/tensor_accessor.h:626` constructs each element as
  `TensorAccessor(std::get<Indexes>(args), get_arg_val<uint32_t>(address_rt_arg_index_start + Indexes))`
  — **two arguments, no page size**. The `ConcatS2SRMProgramFactory` and `ConcatS2STiledProgramFactory`
  kernels construct no `TensorAccessor` at all (their transfers are local-L1 NoC loopback).

  **Reconciliation against the dated triage** (`analyses/2026-07-06_tensor_accessor_3rd_arg_triage.md`):
  concat is **not** in its op→class table. That is consistent with the scan, and the scan — not the
  table's silence — is what establishes the finding.

- **CB endpoints (GATE-free):** censused per CB, per node, per config, for **all six factories**. No CB
  anywhere in the op reaches ≥3 distinct touchers or doubles a FIFO role, so **the multi-binding
  advanced option is not needed in this op**. No dead CBs and no conditional DFBs. Full inventory in
  the per-factory sections below. Not deferred — the Device 2.0 gate is GREEN, so the idioms the census
  keys on are intact.

  Two hunts worth recording as negatives, since the recipe names concat as a shape to census:
  - **Hidden second writer (face (a)):** actively hunted in every kernel that touches every CB —
    a `get_write_ptr()` / `fifo_wr_ptr` write by a kernel that is not the CB's FIFO producer, gated by
    a semaphore pair. **None found**, and none is possible: the op declares no semaphores at all, so
    there is no coordination primitive for a semaphore-gated raw co-fill to use.
  - **`ConcatS2STiledProgramFactory` ≥3-toucher intermediate:** the recipe flags concat `S2S`-tiled as
    a shape where "a co-touched intermediate may hit ≥3 touchers." Censused all seven of its CBs; the
    maximum is **2** touchers. The reader/compute/writer form a clean 3-stage chain in which each
    intermediate is touched by exactly two of the three kernels.

## Port-work summary  *(the subset sections mirror the brief)*

### `ConcatProgramFactory` — CLEAR (in the brief)

The op's **default** factory: every interleaved-input case, plus the ND-sharded fallback
(`concat_device_operation.cpp:29, 69`). Config axes: `rm_layout` (RM vs TILE — this swaps *both* the
reader and the writer kernel), `WIDTH_CONCAT` (set when `rm_layout && dim == rank-1`),
`sub_core_grids` present/absent, and CB depth 1 vs 2.

- **Tensor bindings: ⚠ port work — `N+1` bindings, all Case 1.** The only factory in the op whose
  tensors do *not* arrive via borrowed-memory DFBs. `N` is the input count, up to 47 for interleaved
  (`concat_device_operation.cpp:285`).
  - Delivery today is the **`Buffer*`-binding form**, *not* the silent-wrong `->address()` hazard: the
    factory pushes `Buffer*` objects into the RTA lists (`concat_program_factory.cpp:276` per input;
    `:285` and `:290` for the output). The framework auto-registers these as `BufferBinding`s and
    patches them on cache hits, so the op is correct today; the typed `TensorParameter` binding
    supersedes the mechanism.
  - Consumption is via `TensorAccessor` in every case → **Case 1** throughout. Reader:
    `make_tensor_accessor_tuple(tensor_accessor_args, src_addr_base_idx=3)` builds one accessor per
    input from RTA slots 3..3+N-1 (`reader_concat_interleaved_start_id.cpp:36`,
    `reader_concat_stick_layout_interleaved_start_id.cpp:35`). Writer: `TensorAccessor(dst_args,
    dst_addr)` with `dst_addr = get_arg_val<uint32_t>(0)` (both donor kernels).
  - **Port work:** declare `N` input `TensorParameter`s plus the output at ProgramSpec level. The
    reader's `N` accessors come from a `TensorBindingSequence` (the newly-landed variadic mechanism —
    see the brief for the exact translation); the output binding is `tensor::dst`, a name **fixed by the
    donor fork** the writer will bind. The per-input `TensorAccessorArgs` CTA plumbing
    (`concat_program_factory.cpp:203-205`) and the writer's (`:218`) both disappear.
- **TensorParameter relaxation:** `none`.
- **TensorAccessor 3rd arg:** none — both readers and both donor writers construct 2-argument accessors.
- **CB endpoints: ✓ legal in every config — no action.** Exactly one CB, and it is *not* borrowed:
  `src0_cb_index = 0`, a genuine 1- or 2-page FIFO staging buffer
  (`concat_program_factory.cpp:126-134`; depth falls from 2 to 1 when it would exceed the L1 budget).

  | CB | Backing | Producer | Consumer | Disposition |
  |---|---|---|---|---|
  | `0` (`src0_cb_index`) | L1 staging (no `.buffer`) | reader (`reserve_back`/`push_back`/`get_write_ptr`) | donor writer (`wait_front`/`pop_front`) | **plain 1:1 — no action** |

  Two touchers, one locked to each role. Holds across all four config axes: RM vs TILE swaps *which*
  donor writer is bound, but both are locked consumers; `WIDTH_CONCAT`, `sub_core_grids` and CB depth
  change nothing about the census. No dead CB.
- **RTA varargs: ⚠ one genuine RTA vararg block, plus a rare CTA vararg.**
  - **RTA, shape (a) — variable-count loop.** Reader RTA layout is
    `[0] num_pages_per_core, [1] curr_tensor, [2] curr_tensor_id, [3..3+N-1] Buffer* per input,
    [3+N..3+2N-1] num_pages_per_block, [3+2N..3+3N-1] page_id_per_tensor`
    (`concat_program_factory.cpp:270-281`). Slots 0-2 are read as distinct fixed fields and are
    **nameable**. Slots 3..3+N-1 become tensor bindings, not RTAs. The two trailing `N`-element blocks
    are read in a **counted loop** over `num_tensors` — `reader_concat_interleaved_start_id.cpp:39-43`
    and `reader_concat_stick_layout_interleaved_start_id.cpp:38-42`
    (`arg_ptr[<base> + i]` inside `for (i < num_tensors)`) — so they are a genuine vararg block. `N`
    arrives as a CTA rather than a runtime value, which per the recipe **still** makes it a vararg: it
    varies across instantiations, so there is no stable name set to infer.
  - **CTA vararg — the rare kind, RM path only.**
    `reader_concat_stick_layout_interleaved_start_id.cpp:57` and `:70` read
    `kernel_compile_time_args[page_size_base_idx + curr_tensor]` where `curr_tensor` is a **runtime**
    value. A compile-time arg read at a varying index → `KernelAdvancedOptions::compile_time_varargs`,
    read kernel-side with `get_compile_time_vararg(i)`. The TILE reader
    (`reader_concat_interleaved_start_id.cpp`) has **no** equivalent — it takes its page size from
    `get_tile_size(cb_id_in)` (line 28) and needs no per-tensor page-size CTA at all.
  - The donor writers' RTAs are all distinct fixed indices (0-3 RM, 0-2 TILE) → nameable, less slot 0
    which becomes the output tensor binding. Both forks already declare them as named args.
- **Out-of-directory coupling: ⚠ workable — two borrowed kernel files, both at rung 1.** Function-call
  escape is ✓ clean (all includes class 1). Full inventory in the Team-only section; the porter-facing
  summary is in the brief.

### `ConcatS2SRMProgramFactory` — CLEAR (in the brief)

- **Tensor bindings:** `input_0`, `input_1`, `output` — **all clean** (borrowed-memory DFB). The
  causal-link gate applies to all three: each is a `CBDescriptor` with `.buffer = <tensor>.buffer()`
  (`concat_s2s_rm_program_factory.cpp:69, 86`), and the kernel reads/writes tensor data through
  `input_dfb_0.get_read_ptr()` / `output_dfb.get_write_ptr()`. The DFB *is* the tensor access → port via
  `DataflowBufferSpec::borrowed_from`. No Case 1, no Case 2, no work items.
- **TensorParameter relaxation:** `none`.
- **TensorAccessor 3rd arg:** none — the kernel constructs no accessor.
- **CB endpoints** — the factory instantiates **one** kernel source into two `KernelDescriptor`s that
  differ only by `ReaderConfigDescriptor` / `WriterConfigDescriptor` and their per-instance work-split
  CTAs, both over the **same** `core_ranges` (`concat_s2s_rm_program_factory.cpp:166-188`). That is the
  dual-instance work-split; both instances hit every node, so every CB has exactly **two** touchers per
  node, and both touches are **sync-free raw peeks** (`get_read_ptr()` / `get_write_ptr()` + a NoC
  transfer, no FIFO ops anywhere in the kernel) → **role-free** → **1P+1C**, never the flag.

  | CB | Backing | Touchers per node | Roles | Disposition |
  |---|---|---|---|---|
  | `0` (`input_dfb_0`) | borrowed — `input_tensors[0]` | reader-config + writer-config instance | both role-free (`get_read_ptr()`, kernel:45) | **1P+1C** |
  | `1` (`input_dfb_1`) | borrowed — `input_tensors[1]` | reader-config + writer-config instance | both role-free (`get_read_ptr()`, kernel:70) | **1P+1C** |
  | `16` (`output_dfb`) | borrowed — `output` | reader-config + writer-config instance | both role-free (`get_write_ptr()`, kernel:42) | **1P+1C** |

  **Config dependence:** two configs, same disposition in both. When
  `num_output_rows_per_core_last == 0` there is one core group and two `KernelDescriptor`s. When
  `> 0` (`concat_s2s_rm_program_factory.cpp:190-197`) the cores split into `first_cores` / `last_cores`
  and there are **four** `KernelDescriptor`s of the one source — but the two groups cover **disjoint**
  node sets, so each node still sees exactly one reader-config and one writer-config instance. Census
  per node is 2 in both configs; the disposition does not flip.

- **RTA varargs:** **none** — the factory sets **zero** runtime args. Every kernel argument is a
  compile-time arg at a literal index (`get_compile_time_arg_val(0)`…`(13)`), so all fourteen are
  nameable with no vararg mechanism. No CTA varargs either.
- **Out-of-directory coupling:** **✓ clean** — binds only its own kernel; every include resolves under
  `tt_metal/hw/inc/api/` (donor class 1, LLK/HAL/firmware, no concern). No borrowed kernel files. Its
  kernel is bound by no other op (lent-shape census run: concat-only).

### `ConcatS2STiledProgramFactory` — CLEAR (in the brief)

- **Tensor bindings:** `input_0`, `input_1`, `output` — **all clean** (borrowed-memory DFB;
  `concat_s2s_tiled_program_factory.cpp:102, 116`). The compute kernel only consumes from / produces to
  CBs, so it is out of this subject's scope by the Scope rule.
- **TensorParameter relaxation:** `none`.
- **TensorAccessor 3rd arg:** none — none of the three kernels constructs an accessor.
- **CB endpoints** — seven CBs; three kernels (reader, writer, compute), one instance each, all over
  `all_cores`. Six CBs are plain 1:1 and need no action; one is a single-toucher self-loop.

  | CB | Name | Backing | Producer | Consumer | Disposition |
  |---|---|---|---|---|---|
  | `0` | input0 | borrowed — `input_tensors[0]` | reader (`push_back` @ reader:53) | compute (`wait_front`/`pop_front`) | **plain 1:1** |
  | `1` | input1 | borrowed — `input_tensors[1]` | reader (`push_back` @ reader:54) | compute (`wait_front`/`pop_front`) | **plain 1:1** |
  | `2` | output | borrowed — `output` | writer (`reserve_back`/`push_back` @ writer:43,61) | **— none** | **self-loop** (see below) |
  | `3` | input0_transpose | L1 scratch | compute (`reserve_back`/`push_back`) | reader (`wait_front`/`pop_front` @ reader:59,101) | **plain 1:1** |
  | `4` | input1_transpose | L1 scratch | compute (`reserve_back`/`push_back`) | reader (`wait_front`/`pop_front` @ reader:103,145) | **plain 1:1** |
  | `5` | concat | L1 scratch | reader (`reserve_back`/`push_back` @ reader:57,147) | compute (`wait_front`/`pop_front`) | **plain 1:1** |
  | `6` | output_transpose | L1 scratch | compute (`reserve_back`/`push_back`) | writer (`wait_front`/`pop_front` @ writer:44,60) | **plain 1:1** |

  **CB 2 (output) — one toucher → self-loop.** Only the writer accesses it. The reader declares
  `output_cb_id` as a `constexpr` (reader:20) but constructs no `DataflowBuffer` for it and never
  touches it. The compute kernel *does* construct
  `DataflowBuffer output_dfb(output_dfb_id)` (compute:57) — but that object is **never used anywhere in
  the kernel body**; it is a dead local. On the access test ("any kernel that touches the CB"),
  construction without access is not a touch, so the census is 1 and the resolution is a self-loop:
  bind the writer PRODUCER **and** CONSUMER. This is the one place in the subset where the census turns
  on a judgment the recipe does not explicitly settle — flagged in Recipe notes #4 and carried to the
  brief as a watch-for.

  **Config dependence:** the factory emits two `defines` — `BF8` (when inputs are `BFLOAT8_B`,
  changing the transpose CBs' data format and stride arithmetic) and `USE_SINGLE_PACKET_READ` (when
  both input strides fit in `NOC_MAX_BURST_SIZE`). Neither changes the CB set nor the toucher set —
  both only alter the reader's internal NoC read path. One census covers all four combinations; no
  disposition flips.

- **RTA varargs:** **none** — the factory sets **zero** runtime args. All three kernels share one
  14-element compile-time arg list, every element read at a literal index. All nameable. No CTA
  varargs.
- **Out-of-directory coupling:** **✓ clean** — binds only its own three kernels; every include resolves
  under `tt_metal/hw/inc/api/` (class 1). No borrowed kernel files, and none of its three kernels is
  bound by any other op.

## Heads-ups  *(the subset rows mirror the brief)*

- **CB endpoints (multi-binding shapes to watch):** **none in the entire op.** No CB in any factory
  reaches ≥3 distinct touchers or has two kernels locked to the same FIFO role. The
  hidden-second-writer hunt came back empty and is structurally impossible here (no semaphores). The
  two-toucher CBs are the dual-instance work-split shape and resolve to 1P+1C.
- **Cross-op / shared kernels:** **two, both in `ConcatProgramFactory`** — the RM and TILE donor
  writers. **Both already have a `_metal2` fork checked in beside the original**, so this port reuses
  the forks (rung 1) rather than creating them. Both forks were read and **both fit concat with no
  misfit in either direction** — details in the brief and in Team-only. The two clear sharded factories
  have no shared-kernel exposure at all.
- **RTA varargs:** `ConcatProgramFactory` — one genuine RTA vararg block (both readers) and one CTA
  vararg (RM reader only). The other two clear factories pass zero runtime args.
- **Also for the porter (subset):**
  - `ConcatS2SRMProgramFactory` has a **per-core-group CTA split** (`first_cores` / `last_cores`,
    `concat_s2s_rm_program_factory.cpp:190-197`) that must translate to **two `WorkUnitSpec`s over
    disjoint node sets**, each holding its own pair of same-source `KernelSpec`s with its own CTA
    values. Do **not** demote the per-group CTAs to runtime args — that is a named anti-pattern and
    costs compile-time loop unrolling. Note that `ConcatProgramFactory` does *not* have this shape: it
    already carries its per-core-group value (`num_pages_per_core`) in a **per-core RTA**
    (`concat_program_factory.cpp:247-248, 272`), with single reader and writer `KernelDescriptor`s over
    `all_cores`, so there is nothing to demote and nothing to promote.
  - `ConcatS2SRMProgramFactory` and `ConcatS2STiledProgramFactory` are **100% compile-time args, zero
    runtime args** — neither needs `runtime_arg_schema` or `ProgramRunArgs`.
  - `ConcatS2STiledProgramFactory`'s compute kernel carries the unused
    `DataflowBuffer output_dfb(output_dfb_id)` (compute:57) — resolve whether compute binds CB 2 at all
    before writing its `dfb_bindings`.

## Team-only

### Out-of-directory coupling & donor shape — full inventory

**Op-level roll-up: ⚠ workable.** No ⭐ entries. Two distinct escape types, and they land differently:

- **Function-call escape: ✓ clean, op-wide.** Every `#include` in all nine concat kernel files resolves
  under `tt_metal/hw/inc/api/` — `api/dataflow/{dataflow_api.h, noc.h, dataflow_buffer.h, endpoints.h,
  circular_buffer.h}`, `api/core_local_mem.h`, `api/tensor/noc_traits.h`, `api/compute/*`,
  `api/debug/dprint.h`. All are donor class 1 (LLK / HAL / firmware — no concern). **Zero** includes
  from `ttnn/cpp/ttnn/kernel_lib/`, `ttnn/cpp/ttnn/kernel/`,
  `ttnn/cpp/ttnn/operations/kernel_helper_functions/`, in-family shared code, or any cross-family op
  directory. There is no per-call shape analysis to do — the summary table would be empty of anything
  but class-1 rows, so it is omitted per the report format.

- **Borrowed kernel files (file-path instantiation): 2, both in `ConcatProgramFactory` — now in the
  port.** Both already have a `_metal2` fork checked in beside them, so this port **reuses the existing
  forks (rung 1)** rather than creating any. Both fork checks were run **locationally** (`ls` of the
  original's directory for a same-stem `_metal2` sibling), and neither fork is under
  `experimental/quasar/**`.

  | Donor kernel | Class | Bound for | `_metal2` fork beside it? | Other binding factories |
  |---|---|---|---|---|
  | `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp` | 3 — second shared-kernel pool (`ttnn/cpp/ttnn/kernel/`) | `ConcatProgramFactory`, RM path (`concat_program_factory.cpp:234`) | **yes** — `writer_unary_stick_layout_interleaved_start_id_metal2.cpp` | **1 other:** `data_movement/copy/device/copy_same_memory_config_program_factory.cpp:37` |
  | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | 6 — cross-family donor (`eltwise`) | `ConcatProgramFactory`, TILE path (`concat_program_factory.cpp:235`) | **yes** — `writer_unary_interleaved_start_id_metal2.cpp` | **~23 others** (list below); authoritative consumer list + sunset plan at **issue #52228**, named in the donor file's own header comment |

  These consumer sets are **sunset and coordination lists — not authorization to convert either file in
  place.** Each donor already carries the pointer comment and continues serving its legacy binders
  until the last one migrates. Because concat reuses forks rather than creating them, this port adds no
  new rung-2 carve-out and writes nothing into either peer directory.

  **Fork interfaces (read in full; both fit concat exactly).** The fork's binding names and named-arg
  set are the interface concat inherits — they are not concat's to rename:

  | Fork | DFB binding | Tensor binding | Named args | `#ifdef`s | Concat sets them? |
  |---|---|---|---|---|---|
  | `.../ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id_metal2.cpp` | `dfb::out0` | `tensor::dst` | `stick_size`, `num_sticks`, `start_id` | `BACKWARDS` | no |
  | `.../eltwise/unary/.../writer_unary_interleaved_start_id_metal2.cpp` | `dfb::out` | `tensor::dst` | `num_pages`, `start_id` | `OUT_SHARDED`, `BACKWARDS` | no |

  Concat's legacy writer args map onto both forks with nothing left over and nothing missing — see the
  brief for the per-arg mapping. **No handoff point, no fork edit, no fork-of-a-fork.**

  **A second, non-canonical fork of the TILE donor exists and must not be bound.**
  `ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/dataflow/writer_unary_interleaved_start_id_metal2.cpp`
  is a functionally identical fork sitting in a *consumer's* directory rather than beside the original,
  and it names its accessor **`tensor::output`** instead of `tensor::dst`. The canonical fork — the one
  the locational rung-1 test finds — is the sibling of the original under `eltwise/unary/`. The two are
  flagged for consolidation in the canonical fork's own header comment (lines 13-16) and under
  issue #52228. Binding the wrong one would silently inherit the wrong `tensor::` name.

  **Census disambiguation worth recording:** a bare-filename grep for the stick-layout donor also hits
  `data_movement/slice/device/slice_program_factory_rm.cpp:366`, but that binds
  `slice_writer_unary_stick_layout_interleaved_start_id.cpp` — slice's own **private copy** under a
  different filename, not this donor. Discarded per the census rule (check the bound *path*, not the
  filename). Real binder count is 2, not 4.

  Indicative co-borrower list for `writer_unary_interleaved_start_id.cpp` (filename grep, quasar and
  `_metal2` hits excluded; #52228 is authoritative): `data_movement/reshape_on_device`,
  `data_movement/slice` (×2 tile factories), `data_movement/tilize` (×5),
  `data_movement/transpose` (×2), `eltwise/unary_backward/gelu_bw`, `eltwise/unary_backward/tanh_bw`,
  `embedding`, `examples/example` (×2), `experimental/matmul/attn_matmul`,
  `experimental/transformer/nlp_concat_heads`, `experimental/transformer/nlp_concat_heads_boltz`,
  `matmul` (`matmul_multicore`), `reduction/generic` (×4).

### Per-factory detail for the three gated factories

The recipe's **Red** outcome scoping rule normally defers the seven informational subjects on a RED.
I judged the exception ("a RED that clears without touching the op's code: run them anyway") to apply
to two of the three, and ran them:

| Factory | Which side does the blocker clear on? | Judgment |
|---|---|---|
| `ConcatS2SMultiProgramFactory` | **Elsewhere** (judged) — the cell says the *port* will be semi-manual, not that the op needs rewriting; so the porting capability, not the op code, is what changes. | **Run** — ambiguity flagged in Questions #1 |
| `ConcatBlockShardedProgramFactory` | **Elsewhere** (judged) — same reasoning. | **Run** — ambiguity flagged in Questions #1 |
| `ConcatS2IProgramFactory` | **Op-code side** — the fix is deleting the factory and its kernel. Re-audit reads different (absent) code. | **Skipped** — `skipped — dead-code factory, blocker clears by deletion; re-audit on unblock` |

*(`ConcatProgramFactory` was in this table in the pre-promotion version of this audit, judged
"Elsewhere — a framework capability not yet released." That judgment is what made its promotion an edit
rather than a re-audit; its detail now lives in Port-work summary above.)*

#### `ConcatS2SMultiProgramFactory` — GATED (`DFB misuse; will need semi-manual port`)

- **The blocker, confirmed in code.** `device/kernels/dataflow/reader_s2s_tensor_concat.cpp:30`:
  `DataflowBuffer input_dfb(input_id);` where `input_id` is the **counter of a loop over
  `num_input_tensors`** (line 24). Metal 2.0 binds a DFB through a static `dfb::name` token, so a
  buffer selected by a loop variable has nothing to bind to. This is a **milder** form than
  `ConcatBlockShardedProgramFactory`'s (below): the loop bound is a compile-time arg, so the index set
  *is* statically enumerable and the loop could in principle be expanded over N static bindings.
- **Tensor bindings: all clean.** Every input and the output is a borrowed-memory DFB
  (`concat_s2s_multi_program_factory.cpp:98, 117`); no `->address()`, no `Buffer*` RTA. Causal-link
  gate applies throughout.
- **CB endpoints: 1P+1C on every CB.** Dual-instance work-split — one kernel source into two
  `KernelDescriptor`s differing only by Reader/Writer config and their RTA sets, both over the same
  `all_cores` (`:148-174`). Per node, each CB has two touchers and both are sync-free raw peeks:
  inputs via `input_dfb.get_read_ptr()` (kernel:32), output via `output_dfb.get_write_ptr()`
  (kernel:21), with the two instances writing disjoint stick ranges. Both role-free → **1P+1C**.
  Applies to CBs `0..N-1` (inputs, N ≤ 16) and CB `16` (output). No ≥3-toucher CB, no dead CB.
- **RTA varargs: ⚠ genuine vararg block, shape (a).** `reader_s2s_tensor_concat.cpp:24-28` — four RTAs
  pulled via `arg_idx++` **inside** a `for (input_id < num_input_tensors)` loop. No per-argument names
  to infer; the whole `4N` block becomes varargs.
- **Out-of-directory coupling: ✓ clean** — own kernel only, class-1 includes only.

#### `ConcatBlockShardedProgramFactory` — GATED (`DFB misuse; will need semi-manual port`)

- **The blocker, confirmed in code — the more severe form.**
  `device/kernels/dataflow/reader_writer_block_sharded_concat.cpp:44`:
  `DataflowBuffer src_dfb(src_dfb_id);` where `src_dfb_id = get_arg_val<uint32_t>(arg_idx++)` (line 36)
  — a **pure runtime arg**, supplied per transfer descriptor by the host
  (`concat_block_sharded_program_factory.cpp:349`). Unlike S2SMulti's CTA-bounded loop counter, this
  index is not statically enumerable from the kernel's own compile-time information; the port must bind
  all `N` input DFBs to both kernel instances and replace the runtime selection with a static dispatch.
  That is the "semi-manual" work the sheet refers to.
- **Tensor bindings: all clean.** Inputs and output are all borrowed-memory DFBs
  (`concat_block_sharded_program_factory.cpp:153, 166`); no `->address()`, no `Buffer*` RTA.
- **CB endpoints: 1P+1C on every CB.** Dual-instance work-split again (`:322-338`, one source, two
  configs, one `core_ranges`; the host splits each core's transfer list between the two RISCs at
  `:364-370`). All touches are sync-free raw peeks — `output_dfb.get_write_ptr()` (kernel:28) and
  `src_dfb.get_read_ptr()` (kernel:45) — so both touchers are role-free → **1P+1C** on CB `16`
  (output) and on each input CB `0..N-1`. No ≥3-toucher CB, no dead CB.

  *Note on the input CBs:* the kernel calls `get_read_ptr()` on its **local** instance of an input CB
  purely to learn the layout address, then issues the NoC read against that address on a **remote**
  core (`{.noc_x = src_noc_x, .noc_y = src_noc_y, .addr = src_l1_addr}`, kernel:55) — valid because
  every core's CB layout is identical. The local `get_read_ptr()` is still a local touch and is counted
  as one. Which input CBs a given node's reader-vs-writer instance touches varies with the per-core
  transfer split, so a uniform 1P+1C assignment across both instances is the disposition that holds
  for every core.
- **RTA varargs: ⚠ genuine vararg block, shape (a).** `reader_writer_block_sharded_concat.cpp:31-42` —
  `num_transfers` at slot 0 (nameable), then a `9 × num_transfers` block pulled via `arg_idx++` inside
  `for (t < num_transfers)`, with `num_transfers` a **runtime** value that differs between the reader
  and writer instances. The clearest vararg case in the op.
- **Out-of-directory coupling: ✓ clean** — own kernel only, class-1 includes only.

#### `ConcatS2IProgramFactory` — GATED (`... this factory is dead code`)

Informational subjects **skipped — dead-code factory, blocker clears by deletion; re-audit on unblock.**
Gate-bearing subjects were run over it and its kernel anyway: Device 2.0 ✓ (`writer_s2i_width.cpp` is
on `Noc` / `DataflowBuffer` / `TensorAccessor`), Appendix A all `N/A`, no `->address()`, no accessor
3rd argument. See Misc anomalies #1 and #2 for the two independent confirmations that it is dead.

### Relaxation candidates

**None to offer.** The op declares no custom `compute_program_hash` and no backdoor
`attribute_values` / `to_hash`, so there is no hash to mine for the tensor properties the op actually
depends on. `TensorParameter relaxation` is `none` on all six rows.

### TTNN factory analysis

Sheet-derived facts with `file:line` evidence:

- **Current concept:** `descriptor`, all six factories — `static tt::tt_metal::ProgramDescriptor
  create_descriptor(...)` at `device/concat_program_factory.hpp:14`,
  `concat_s2s_tiled_program_factory.hpp:14`, `concat_s2s_rm_program_factory.hpp:14`,
  `concat_s2s_multi_program_factory.hpp:14`, `concat_s2i_program_factory.hpp:14`,
  `concat_block_sharded_program_factory.hpp:13`.
- **Op-owned tensors:** none. No `WorkloadDescriptor`, hence no `buffers` vector.
- **MeshWorkload need:** none. The concept is `descriptor`, so the `WorkloadDescriptor` /
  secretly-SPMD question does not arise. The sheet's `Execution Model` column reads `SPMD` for all six.
- **Pybind `create_descriptor`:** absent. `concat_nanobind.cpp:39-49` binds only the public
  `ttnn::concat` free function through `ttnn::bind_function<"concat">`. No `nb::class_` of the device
  operation, no factory internals exposed. **Nothing for the port to delete**, and no user-visible API
  change on this axis.
- **Other risky pybind:** none observed.
- **Custom hash:** absent (grep-confirmed; not a gate in any case — the port would leave it intact).
- **`get_dynamic_runtime_args`:** absent (grep-confirmed) — the deprecated hook is not present, so this
  gate conjunct is confirmed clear.
- **`override_runtime_arguments`:** absent (grep-confirmed) → the target is the base
  **`ProgramSpecFactoryConcept`**, not `CustomProgramSpecFactoryConcept`.
- **Target concept:** `ProgramSpecFactoryConcept` — derived from `Concept == descriptor` plus
  `Override runtime args method? == no` plus no op-owned tensors. Independently matches the sheet's own
  `Porting Target` column (`ProgramSpecFactoryConcept` on all six rows).
- **Partial port and the factory variant — supported, no action needed.** Three of six factories
  convert. `ConcatDeviceOperation` keeps all six in its `program_factory_t` variant
  (`concat_device_operation.hpp:29-35`) and `select_program_factory` keeps dispatching to all six, so
  the device-op carries a **mixed** set of concepts — three `MetalV2`, three `descriptor` — for the
  duration. Confirmed with the TTNN framework owner that one device-op's `std::variant` supports a
  mixed set, so the subset port is viable as scoped and the porter needs to do nothing special here.

## Misc anomalies  *(team-only, non-gating; route to the ops team — the port does not act on these)*

1. **`ConcatS2IProgramFactory` binds a kernel source that does not exist.**
   `device/concat_s2i_program_factory.cpp:54-55` sets
   `reader_desc.kernel_source = ".../kernels/dataflow/reader_s2i_width.cpp"`, but **no file named
   `reader_s2i_width.cpp` exists anywhere in the repository** (`find`-confirmed; only the companion
   `writer_s2i_width.cpp` is present). Were this factory ever selected, program creation would fail on
   the missing source. This is a stronger statement than the sheet's "dead code": the factory is not
   merely unreachable, it is **unbuildable**.

2. **`select_program_factory`'s S2I branch is unreachable.** `device/concat_device_operation.cpp:32-34`
   returns `ConcatS2IProgramFactory{}` when the input is sharded and the output is not. `ttnn::prim::concat`
   has exactly four call sites, all inside `concat_device_operation.cpp` (`:364`, `:390`, `:455`, `:458`),
   and it is not pybound — so `concat_impl` is the only route in. Every sharded-input path converts the
   output config to a *sharded* one before calling (`:364` requires `output_mem_config.is_sharded()`;
   `:390` passes a constructed `temp_sharded_config`) or unshards the inputs and recurses (`:401-406`);
   the two remaining call sites (`:455`, `:458`) are past the sharded branch and always have interleaved
   inputs. The `input sharded && output interleaved` combination therefore never reaches
   `select_program_factory`. Independently corroborates #1. Suggested disposition: delete the factory,
   its header, its `program_factory_t` variant entry, the unreachable branch, and `writer_s2i_width.cpp`.

3. **Unused `DataflowBuffer` local in the tiled compute kernel.**
   `device/kernels/compute/height_sharded_width_concat_two_tensors.cpp:57` constructs
   `DataflowBuffer output_dfb(output_dfb_id);` and never uses it. Harmless today; under Metal 2.0 it
   becomes a live question (does compute bind CB 2?), so it is worth deleting on the ops-team track
   rather than leaving the porter to decide. Carried to the brief as a watch-for.

4. **A required-but-unread CTA slot in the legacy stick-layout donor — a donor-side quirk, not concat's.**
   `concat_program_factory.cpp:214` bakes `dst_buffer->page_size()` into writer CTA slot 1 on the RM
   path. The legacy donor kernel never reads that slot: it takes `cb_id_out0` from CTA 0, places its
   accessor args at CTA 2 (`TensorAccessorArgs<2>()`, donor line 23), and takes `stick_size` from
   **RTA** slot 1 (donor line 18) — so the same value is passed twice, once dead. This is the donor's
   contract rather than concat over-passing: the other consumer,
   `copy/device/copy_same_memory_config_program_factory.cpp:131`, likewise passes two CTAs
   (`{output_cb_index, output_unit_size}`) before its accessor args. Route to whoever owns
   `ttnn/cpp/ttnn/kernel/dataflow/`. **Moot on the port path** — the `_metal2` fork takes no CTAs at
   all, so the dead slot simply disappears.

5. **Unused debug includes.** `device/kernels/dataflow/writer_height_sharded_width_concat_two_tensors_tiled.cpp:6`
   (`#include <api/debug/dprint.h>`) and `device/kernels/dataflow/writer_s2i_width.cpp:6`
   (`#include "api/debug/dprint.h"`) — neither kernel body contains a `DPRINT`.

6. **`ConcatS2STiledProgramFactory` is loop-generalized over `N` inputs but hard-coded to 2.** It builds
   `N` input CBs in a loop (`concat_s2s_tiled_program_factory.cpp:91-104`) yet indexes
   `input_tensors[1]` and `num_tiles_for_each_input_shard[1]` directly (`:30-56`, `:127`, `:139`,
   `:178`, `:184`). Safe today — `select_program_factory` reaches it only when
   `input_tensors.size() == 2` (`concat_device_operation.cpp:46`) — but if that guard ever widened, the
   loop would allocate input CBs `2..N-1` that no kernel touches. Not a bug; a latent-fragility note.

7. **A comment names a legacy API that the code does not use.**
   `device/kernels/dataflow/reader_writer_block_sharded_concat.cpp:19-21` explains the `(cb_id, offset)`
   design by reference to `UpdateDynamicCircularBufferAddress`, which the descriptor-API factory never
   calls (it uses `CBDescriptor::buffer`). Accurate as history, but it is the one textual match for an
   Appendix A recognition signal in the whole op — a reader grepping for that signal will land here.
   Worth rewording to name the descriptor-API mechanism.

8. **Three duplicated `_metal2` forks of `eltwise/unary` kernels, two with divergent binding names.**
   Not concat's code, but concat is an op that has to choose between them — so it surfaced here, and
   is written up in full because the choice is silent and inherited. **Passed to the ops team
   2026-08-27.**

   `copy/typecast/device/kernels/dataflow/` holds three `_metal2` forks of `eltwise/unary` kernels, each
   of which *also* has a canonical fork beside its original:

   | Kernel | Canonical fork (beside original) | Orphan fork (`copy/typecast/`) | Diverges? |
   |---|---|---|---|
   | `reader_unary_interleaved_start_id_metal2.cpp` | `dfb::in`, **`tensor::src`** | `dfb::in`, **`tensor::input`** | **yes** |
   | `writer_unary_interleaved_start_id_metal2.cpp` | `dfb::out`, **`tensor::dst`** | `dfb::out`, **`tensor::output`** | **yes** |
   | `reader_unary_sharded_metal2.cpp` | `dfb::in` (no tensor binding) | `dfb::in` (no tensor binding) | no |

   Dataflow logic is identical in all three pairs; DFB names agree everywhere. Only the `tensor::`
   names, comment wording, and `const auto` vs `uint32_t` differ.

   **Not systemic — one PR.** A sweep of every `_metal2` file in `ttnn/cpp` outside `experimental/quasar`,
   checking whether the original sits beside it: **25 forks, 22 correct, 3 orphans** — these three, all
   added in `cbde3d44ff3` (2026-07-31, Edwin Lee, *[Cleanup] Port Typecast to Metal 2.0* #51397).

   **Mechanism.** The rung-1 fork check is deliberately **locational** (`ls` the original's directory for
   a same-stem `_metal2` sibling; a tree-wide filename grep is forbidden, to keep quasar copies out).
   That scope makes it structurally blind to a fork placed anywhere *other* than beside the original —
   and the rung-2 rule that prevents this ("the fork goes beside the file it forks, never relocated into
   your op's tree") is what #51397 did not follow. So later porters correctly ran rung 1, found nothing,
   and created second forks beside the originals, naming them independently. It caught the same author
   out: the typecast writer orphan landed 2026-07-31 (#51397) and the canonical writer fork five days
   later, 2026-08-05 (#51771, Edwin Lee), with a different accessor name; the `ALSO DUPLICATED BY`
   cross-reference only landed 2026-08-21 (#51743). The two *reader* forks still carry no cross-reference
   in either direction, and all three legacy originals say a fork "lives beside it" — **singular**.

   **Already spread beyond typecast.** `reader_unary_sharded_metal2.cpp` has two live forks with
   disjoint consumer sets in different families — canonical: `untilize` (×2 factories); orphan:
   `typecast_sharded` **and `data_movement/sharded/sharded_to_interleaved`**. The latter is deliberate:
   `sharded_to_interleaved_program_factory.cpp:132-134` states it binds "the shared Metal 2.0 reader fork
   that already lives in typecast's tree, so its accessor and argument names are that kernel's interface,
   not this op's choice" (`c7943ced965`, 2026-08-25, Edwin Lee, #52207) — **one day after** the canonical
   fork of that kernel was created (`2d32a5a54b2`, 2026-08-24, Gajanan Choudhary, #53586). Two ports a
   day apart reached opposite conclusions about where the fork lives, neither able to see the other, so
   "typecast's tree is the home of the shared fork" is being established by precedent. Other consumer
   sets — `reader_unary_interleaved`: canonical `untilize` (×2) + `data_movement/copy`, orphan `typecast`;
   `writer_unary_interleaved`: canonical `data_movement/copy`, orphan `typecast`.

   **Nothing is broken today.** No single factory mixes the two vocabularies (`typecast_program_factory`
   binds orphan+orphan; `copy_default_tilized` binds canonical+canonical). The exposure is to *future*
   ports that bind the wrong fork and inherit the wrong interface silently.

   **Fix looks cheap** (host-side only, no kernel edits): `reader_unary_sharded_metal2.cpp` needs a pure
   `.source` repoint for both its orphan consumers (names already agree, zero renames); the two
   interleaved kernels need `typecast_program_factory`'s two `.source` constants repointed plus four
   `accessor_name` strings changed — `"input"`→`"src"` (`:147`, `:311`), `"output"`→`"dst"` (`:157`,
   `:321`). Then delete the three orphans.

   **Two open questions for the ops team** (theirs to decide, not the auditor's): whether beside-the-
   original is in fact the intended location — 22/25 forks say yes, but `sharded_to_interleaved`
   documents the opposite in a comment, so the tree now holds both conventions; and whether `src`/`dst`
   or `input`/`output` wins, since every future consumer inherits the choice. (`src`/`dst` matches the
   legacy kernels' own locals, which is the tiebreak the porting docs suggest.)

   **Minor doc inaccuracy in the same files:** all three typecast fork headers state the legacy file is
   "instantiated by ~70 factories." A filename grep finds ~24 binders of
   `writer_unary_interleaved_start_id.cpp`. Likely a whole-family count in a per-kernel comment.

## Per-DeviceOperation attribution

Not applicable — the directory holds a single `DeviceOperation` (`ConcatDeviceOperation`). Per-factory
attribution, where findings differ, is carried throughout the sections above.

## Questions for the user

1. **Which side clears `DFB misuse; will need semi-manual port`?** The phrase reads as though the *port*
   absorbs the work (a porting-capability question), which is how I classified it — so I ran the seven
   informational subjects for `ConcatS2SMultiProgramFactory` and `ConcatBlockShardedProgramFactory`
   rather than deferring them. If instead the intent is that the **ops team rewrites the kernels** to
   make the DFB indices static, then that detail is against soon-to-change code and should be re-derived
   at re-audit. Either way nothing is lost — the work is done and disclosed — but the answer determines
   whether it should be trusted. (This is the same question the now-resolved variadic-tensor case
   answered favourably: running early cost one pass and saved a second.)

2. **Is `ConcatS2IProgramFactory` scheduled for deletion?** Given Misc anomalies #1 and #2 (missing
   kernel source *and* unreachable selection branch), it looks like an unambiguous delete rather than a
   port candidate. Confirming that would let it drop off the readiness sheet entirely instead of sitting
   as a permanent `no`.

*(Two further questions raised during this audit are now closed and folded into the body above: mixed
`descriptor` / `MetalV2` factories on one `program_factory_t` variant are **supported** — see TTNN
factory analysis — and the `ConcatProgramFactory` readiness cell is **stale, not blocking** — see the
box at the top.)*

## Recipe notes

*Consolidated, with the run's positive findings and process notes, in
`METAL2_AUDIT_RECIPE_FEEDBACK.md` beside this file — that is the copy to hand to the recipe maintainer.
Retained here in brief so this report stands alone.*

1. **The "run them anyway" exception (`:76`) paid off on this very op.** `ConcatProgramFactory`'s
   blocker was a pending framework feature, so I judged it to clear "elsewhere" and ran its seven
   informational subjects despite the RED. That feature has now merged — and because the detail already
   existed against unchanged code, promoting the factory into the clean subset was an **edit, not a
   re-audit**. Worth recording as evidence for the rule. See #5 for the one gap in applying it.

2. **Whether a config-scoped GATE issues a brief is stated both ways.** `:111` says a config-scoped GATE
   "still issues a brief for the clean subset" and `:74` says to run the informational subjects because
   "**its brief needs them**"; but `:507` says the brief is emitted "only on a fully GREEN audit … On
   any **RED** there is no brief," and the template header `:639` says "Never on RED." Concat lands on
   the seam. I followed `:74`/`:111` and emitted a subset-scoped brief, since `:74`'s rationale is void
   if no brief exists. Suggest amending `:507` and `:639` to carry the carve-out.

3. **The status-summary template has a `Variadic-CTA` row with no Appendix A entry behind it** (`:557`
   vs. Appendix A's three entries and the gate-detail table at `:590-594`). Awkward here specifically:
   concat *does* use a genuine CTA vararg
   (`reader_concat_stick_layout_interleaved_start_id.cpp:57,70`), which the RTA-varargs subject
   explicitly says does **not** gate — so neither `Ok` nor `Unsupported` is honest. Filled with a
   pointer to this note.

4. **The CB-endpoint census does not say whether constructing a DFB object without accessing it is a
   touch.** Concat's tiled compute kernel constructs `DataflowBuffer output_dfb(output_dfb_id)` and
   never accesses it (`height_sharded_width_concat_two_tensors.cpp:57`). Access test → census 1 →
   self-loop; a *needs-a-binding-to-compile* reading → census 2 → 1P+1C. I took the access test as
   written and flagged it in the brief.

5. **The `:76` op-code-side / elsewhere test has no anchor for free-text `Known op issues` cells.** All
   of concat's blocks arrived through that free-text column. `DFB misuse; will need semi-manual port`
   could be read either way (Questions #1). Suggest a line telling the auditor to decide from what the
   cell says will change, and to default to running when it doesn't say.

6. **The causal-link gate's borrowed-memory signal is given only in imperative form.** Both the gate
   (`:300`) and the false-positive guards (`:331-332`) name host-side
   `set_globally_allocated_address(buffer)`. Descriptor-API ops express borrowed memory as
   `.buffer = tensor.buffer()` on a `CBDescriptor` instead, and concat uses that form exclusively —
   13 `CBDescriptor` literals, **zero** occurrences of `set_globally_allocated_address`. An auditor
   grepping the documented signal finds nothing and could push borrowed-memory bindings into Case 1/2.
   Suggest naming both spellings. **This is the highest-risk item in the list**, because the failure is
   a grep returning nothing rather than a visible ambiguity.

7. **The recipe covers unreferenced kernel files but not referenced-but-absent ones** (`:85`). Concat has
   the mirror case (Misc anomalies #1), which decided a whole factory's disposition. Suggest a line in
   Scope.

8. **Minor: the "shapes to census" hints for concat were half-right.** `:423` predicts 1P+1C for
   `S2SRM` / `S2SMulti` / `BlockSharded` — held exactly on all three. It also predicts concat
   `S2S`-tiled "may hit ≥3 touchers"; its maximum census is **2**. Framed as a prompt, so not a defect.
