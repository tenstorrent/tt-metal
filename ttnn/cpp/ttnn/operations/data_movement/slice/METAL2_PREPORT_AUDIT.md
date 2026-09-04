# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/data_movement/slice`

> **Re-audit (2026-09-04) — the op is now GREEN.** Earlier audits of this op RED'd on the readiness
> sheet's TTNN gate plus the offset-base-pointer and `TensorAccessor`-3rd-argument subjects. All three
> have cleared: the sheet's `Is able to port?` now reads **`yes` on all five factory rows**, and the
> from-code scan finds **no host-folded offset base** anywhere in the op. Both dated triage analyses
> still list `slice`, and **both are stale for it** — each catalogues a construct the op no longer has.
> Per each subject's *"your own scan is the source of truth"* rule, the from-code scan decides.
> **A porter brief is issued:** `METAL2_PORT_BRIEF.md`, beside this file.

One DeviceOperation, five program factories — a single bundled report (they share the device-op, the
custom hash, the address-patching helper and, in part, their kernels). Findings are attributed per
factory throughout; see *Per-DeviceOperation / per-factory attribution*.

- **`ttnn::prim::SliceDeviceOperation`** (`device/slice_device_operation.hpp:31`)
  - `SliceRmProgramFactory` (`device/slice_program_factory_rm.cpp`)
  - `SliceRmShardedProgramFactory` (`device/slice_program_factory_rm_sharded.cpp`)
  - `SliceRmStrideProgramFactory` (`device/slice_program_factory_rm_stride.cpp`)
  - `SliceTileProgramFactory` (`device/slice_program_factory_tile.cpp`)
  - `SliceTileTensorArgsProgramFactory` (`device/slice_program_factory_tile_tensor_args.cpp`)

Factory selection (`device/slice_device_operation.cpp:309-341`) is what defines the *configs* this
report classifies per-`(CB, config)` against:

| Config | Selected factory |
|---|---|
| `use_tensor_args == true` (TILE only) | `SliceTileTensorArgs` |
| RM, HEIGHT-sharded in **and** out, no step, W-begin L1-aligned | `SliceRmSharded` |
| RM, any `step != 1` | `SliceRmStride` (rank ≤ 4 vs rank > 4 pick different kernels) |
| RM, otherwise (interleaved **or** BLOCK/WIDTH-sharded) | `SliceRm` |
| TILE, otherwise | `SliceTile` |

**Kernels in scope** (every kernel a factory `kernel_source`s — own and donor):

| Kernel | Factory | Owner |
|---|---|---|
| `slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp` | Rm | slice |
| `slice_writer_unary_stick_layout_interleaved_start_id.cpp` | Rm | slice |
| `slice_reader_unary_unpad_dims_rm_sharded.cpp` | RmSharded | slice |
| `reader_multicore_slice_4d.cpp` / `writer_multicore_slice_4d.cpp` | RmStride (rank ≤ 4) | slice |
| `reader_multicore_slice_nd.cpp` / `writer_multicore_slice_nd.cpp` | RmStride (rank > 4) | slice |
| `reader_unary_unpad_dims_interleaved_start_id.cpp` | Tile | slice |
| `writer_unary_interleaved_start_id.cpp` (slice's **own** copy) | Tile | slice |
| `reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp` | TileTensorArgs | slice |
| `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | TileTensorArgs | **eltwise/unary — cross-family donor** |

**Unreferenced kernel files in the op directory** (out of scope, contents not audited — listed only so a
reader does not mistake them for live code; verified unreferenced by a tree-wide filename grep):
`device/kernels/dataflow/strided_slice_reader_rm_interleaved_nd.cpp`,
`device/kernels/dataflow/strided_slice_writer_rm_interleaved.cpp`.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `metal2_audit.md`.

**Recipe docs:** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`

*Provenance note:* the recipe was consumed out-of-tree from `/localdev/edwinlee/metal2_audit.md`. That
file is **byte-identical** (`diff` clean) to
`docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/audit/metal2_audit.md` on branch
`akertesz/op-porting-recipe` in the doc-branch checkout at `/localdev/edwinlee/Port_Recipe`, so the hash
above is a valid pin. Note this is a **newer recipe** than the prior audit ran against
(`50e992a8ec2`), and the newest commit is specifically about this subject area.

**Reference data.** Readiness sheet *"TTNN Operations analysis"* **fetched fresh this session** from
Drive (`1KUMj8SyBGlNMZlLFgs1MbAZlO2g6EoUc4KaxSlcy8jw`, owner `dgomez@`) — 28 columns, 486 rows, five
`data_movement/slice` rows (one per factory). Both dated triage analyses named by the recipe were
available and are cross-referenced below: `analyses/2026-07-19_offset_base_pointers.md` and
`analyses/2026-07-06_tensor_accessor_3rd_arg_triage.md`. **Both list `slice`; both are now stale for it.**

---

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/slice` |
| **Overall** | **GREEN** — every gate cleared; brief issued |
| **DOps / Factories** | `SliceDeviceOperation` → `SliceRm`, `SliceRmSharded`, `SliceRmStride`, `SliceTile`, `SliceTileTensorArgs` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** ✓ — all 9 kernel files + the in-family helper header are structurally Device 2.0; zero holdovers |
| *Prereqs* — Cross-op escapes | **Ok** ✓ — one in-family helper header (all call shapes ✓), one cross-family donor kernel with a `_metal2` fork already checked in |
| *Feature Support* — overall | **GREEN** — no Appendix A entry fires |
| *Feature Support* — Variadic-CTA | Ok — every `get_compile_time_arg_val` index is a literal constant; no CTA vararg |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** ✓ — `yes` on all five factory rows; primary cross-check clean |
| *TTNN Readiness* — Concept (current) | `descriptor` (all five) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A — concept is `descriptor`, not `WorkloadDescriptor` |
| *TTNN Readiness* — Custom hash | **Yes** (not a gate; port leaves it intact): `device/slice_device_operation.cpp:343` |
| *TTNN Readiness* — `get_dynamic_runtime_args` | **No** ✓ — no such hook on the DeviceOperation; grep of the op returns zero hits |
| *TTNN Readiness* — `override_runtime_arguments` | **Yes** (not a gate; selects `CustomProgramSpecFactoryConcept`) — all five factories: `…_rm.cpp:396`, `…_rm_sharded.cpp:415`, `…_rm_stride.cpp:178`, `…_tile.cpp:189`, `…_tile_tensor_args.cpp:195` |
| *TTNN Readiness* — Pybind `create_descriptor` | **Yes** (not a gate; port deletes the binding): `slice_nanobind.cpp:168-179` (on `SliceTileProgramFactory` only). Sheet value is `PR` — handled in an in-flight PR |
| *TTNN Readiness* — Op-owned tensors | **No** — sheet cell blank; no `create_workload_descriptor`, no `buffers` vector |
| *TTNN Readiness* — Target concept | **`CustomProgramSpecFactoryConcept`** (no op-owned tensors) — matches the sheet's own `Porting Target` cell |
| *Port work* — Offset base pointer | **none** ✓ — every `->address()` in the op is a bare base; the W-begin byte offset rides a **separate** RTA |
| *Port work* — Tensor bindings (per binding) | 12 bindings: **10 × Case 1** (`TensorAccessor`), **2 × clean** (borrowed-memory DFB, RmSharded). No Case 2 |
| *TTNN Readiness* — TensorParameter relaxation | **`none`** (clears) — verbatim, all five rows. The `Provisional relaxation finding (Edwin)` cell (`needs fix, then none`) is **confirmed stale by the op owner**; relaxations are fine and the value is `none` |
| *Port work* — `TensorAccessor` 3rd arg | **2 sites, both Class 2** → mechanical drop. See the flagged sub-case in *Questions* |
| *Port work* — CB endpoints | 7 CBs: **4 × plain 1:1 legal**, **3 × self-loop**. No multi-binding, no dead CB, no conditional DFB |

**CB endpoints** are dispositions, not gates: every out-of-window CB here resolves to a **self-loop**
(one toucher). Nothing in this subject blocks a Gen1 port.

## Result

**GREEN → brief issued** (`METAL2_PORT_BRIEF.md`).

All five gate-bearing subjects clear:

1. **Device 2.0** ✓ — every kernel the op exercises, own and donor, is structurally Device 2.0.
2. **Feature compatibility** ✓ — all three Appendix A entries `N/A`.
3. **TTNN factory concept** ✓ — `Is able to port? == yes` on all five factory rows; cross-check clean.
4. **Offset base pointers** ✓ — the catalogued Type-2 fold has been **split out**; the scan finds none.
5. **`TensorAccessor` 3rd argument** ✓ — two sites, both Class 2 (mechanical drop).

Port work is real but mechanical: **10 Case-1 tensor bindings**, **2 page-size arg drops**, **3 CB
self-loops**, and **6 kernels carrying genuine RTA/CRTA vararg blocks**. Two items need the porter's
attention beyond mechanics, both recorded in the brief: the **existing `_metal2` fork** of the eltwise
donor (reuse it — rung 1), and **`ccl/mesh_partition`**, an out-of-directory host-side consumer that
calls slice's `create_descriptor` and `patch_slice_program_addresses` directly and will not compile
against a ported signature.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN.** All five `data_movement/slice` rows read
  `Is able to port? = yes`. The sheet also carries `Op Classification = PD Op (custom)`,
  `Execution Model = SPMD`, `Porting Target = CustomProgramSpecFactoryConcept`, and an empty
  `Known op issues` on every row. Nothing blocks.

  Lightweight cross-check against the code — every cheaply-checkable factual column agrees:

  | Column | Sheet value | Code evidence | Agrees |
  |---|---|---|---|
  | `Concept` | `descriptor` (×5) | all five factories declare `static tt::tt_metal::ProgramDescriptor create_descriptor(...)` (e.g. `slice_program_factory_rm.hpp:26`) | ✓ |
  | `Custom hash (compute_program_hash)` | `yes` (×5) | `slice_device_operation.hpp:51`, defined `slice_device_operation.cpp:343` | ✓ |
  | `Backdoor custom hash (attribute_values / to_hash)` | `no` (×5) | grep for `attribute_values` / `to_hash` over the op: **zero hits** | ✓ |
  | `Runtime-args update (get_dynamic_runtime_args)` | `no` (×5) | grep of the device-op and factories: **zero hits**; the two mechanisms are `static_assert`-exclusive and slice defines `override_runtime_arguments` | ✓ |
  | `Override runtime args method? (PD only)` | `yes` (×5) | all five factories define it (sites in the status summary) | ✓ |
  | `Pybind descriptor (nb::class_ of device op)` | `PR` | present in code at `slice_nanobind.cpp:168-179`; `PR` = handled in an in-flight PR, a status marker rather than a boolean claim → **not** a conflict | ✓ |
  | `Smuggled pointer (raw buffer addr in RTA/CRTA)` | `no` (×5) | every buffer address enters `create_descriptor` as a **`Buffer*` binding**, never as `->address()`; the five `->address()` calls in the op all sit inside `override_runtime_arguments`' patch path (`slice_program_factory_rm_sharded.cpp:381,389,395,398,399`), which is the sanctioned refresh route | ✓ |
  | `Op-owned tensors?` | *(blank)* | no `create_workload_descriptor`, no `buffers` vector | ✓ |
  | `Secretly SPMD Workload?` | *(blank)* | N/A on a `descriptor` concept | ✓ |
  | **Factory-set match** | 5 rows | 5 factories in `program_factory_t` (`slice_device_operation.hpp:36-41`); names and `Factory definition path` cells match one-to-one — no phantom row, no missing row | ✓ |

  Cross-column invariants hold: `get_dynamic_runtime_args == no` on a `descriptor` concept ✓;
  `Op-owned tensors?` not `yes` on a `descriptor` concept ✓.

  *Recorded, non-gating:* the sheet's `Provisional relaxation finding (Edwin)` column reads
  **`needs fix, then none`** on the `SliceRm` and `SliceTile` rows (blank on the other three). This is
  **not** the gating `TensorParameter relaxation` column — that one reads `none` on all five rows and
  clears. Flagged here because the two columns are easy to conflate; see *Questions*.

- **Device 2.0 (every kernel used):** **GREEN.** Every kernel the op exercises is structurally Device
  2.0 — `Noc` for all NOC traffic, `DataflowBuffer` for every CB, `CoreLocalMem` / `UnicastEndpoint`
  where a raw L1 address or an explicit NOC target is needed. The sweep looked for
  `InterleavedAddrGen` / `ShardedAddrGen` / `InterleavedAddrGenFast` / `InterleavedPow2AddrGen*`, bare
  `noc_async_read(` / `noc_async_write(` / `noc_async_*_barrier(`, `cb_reserve_back` / `cb_push_back` /
  `cb_wait_front` / `cb_pop_front`, raw semaphore addresses, `evil_set_*_ptr`, and a stale
  `api/dataflow/circular_buffer.h` include: **zero hits** across all 9 kernel files, the eltwise donor,
  and the in-family helper header.

  **No holdover table — there are no violations.** Two shapes were checked and deliberately *not*
  flagged:

  | File | Line | Call | Why not a violation |
  |---|---|---|---|
  | `eltwise/unary/…/writer_unary_interleaved_start_id.cpp` (donor) | `27` | `get_local_cb_interface(cb_id_out).fifo_page_size` | **Sanctioned** free function per the Device 2.0 Green bullet — the list is the whole test and does not turn on what object is in scope, so the adjacent `DataflowBuffer dfb(cb_id_out)` at `:30` does not unseat it. Moving this onto `dfb.get_entry_size()` is a **port-stage** change (kernel-side whitelist rule 7), not a Device 2.0 one — and here it is already done in the `_metal2` fork. |
  | all slice kernels | various | `dfb_x.get_write_ptr()` / `dfb_x.get_read_ptr()` | These are **methods on a `DataflowBuffer` object**, not the CB-index-keyed free functions the holdover rule targets. 13 such sites; all method-form. |

  The one *in-family* shared header the kernels include,
  `ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp`, is itself Device 2.0: its
  `noc_async_read_sharded` / `noc_async_write_sharded` / `tt_memmove` all take a leading `Noc`
  parameter (`:325`, `:375`, `:143`), and the slice kernels call those overloads — **not** the
  `[[deprecated]]` no-`Noc` forms at `:364`, `:413`, `:211`.

- **Feature compatibility:** every Appendix A entry, in order. All `N/A`.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | **N/A** | Grepped the whole op for `GlobalCircularBuffer`, `CreateGlobalCircularBuffer`, the `global_circular_buffer` `CBDescriptor` field, `remote_index(`, `remote_cb_*`, `remote_circular_buffer.h`, `UpdateDynamicCircularBufferAddress`, `num_global_cb_receivers` — **zero hits**. The two Buffer-backed CBs (`slice_program_factory_rm_sharded.cpp:290,302`) set only `.buffer` — the plain borrowed-memory pattern, a mechanical porting-recipe translation via `DataflowBufferSpec::borrowed_from`, not this entry. |
  | CBDescriptor `address_offset` (non-zero) | **N/A** | No `.address_offset`, no `set_address_offset`, no 4-arg `UpdateDynamicCircularBufferAddress`, no `cb_descriptor_from_sharded_tensor` anywhere in the op. All 7 `CBDescriptor`s leave `address_offset` defaulted to 0 — including the two that set `.buffer`. |
  | GlobalSemaphore | **N/A** | The op declares **no semaphores at all** — no `SemaphoreDescriptor`, no `CreateSemaphore`, no `GlobalSemaphore`, no `global_semaphore.hpp`. A grep for `Semaphore` / `semaphore` across the op returns nothing. |

- **CB endpoints (GATE-free):** every CB is either plain 1:1 or a one-toucher self-loop. Full
  per-`(CB, config)` census, counting **every** access (FIFO ops *and* raw-pointer peeks), per node:

  | Factory / config | CB | Touchers on a node | Verdict | Resolution |
  |---|---|---|---|---|
  | `SliceRm` (both the chunked and non-chunked paths) | `c_0` (`…_rm.cpp:322`) | reader `reserve_back`/`push_back` (+ own-binding `get_write_ptr` peek) = **locked producer**; writer `wait_front`/`pop_front` (+ `get_read_ptr` peek) = **locked consumer** | **plain 1:1** | bind 1 PRODUCER + 1 CONSUMER — no flag |
  | `SliceRmSharded` | `c_0`, **borrowed** from `input.buffer()` (`…_rm_sharded.cpp:282,290`) | reader only, and only `dfb_in.get_write_ptr()` (`slice_reader_…_rm_sharded.cpp:41`) — a raw peek, **no FIFO ops** → **role-free**, 1 toucher | **single-ended / sync-free** | **self-loop** (bind the reader PRODUCER *and* CONSUMER) |
  | `SliceRmSharded` | `c_16`, **borrowed** from `output.buffer()` (`…_rm_sharded.cpp:294,302`) | reader only: `reserve_back`(`:40`) / `get_write_ptr`(`:42`) / `push_back`(`:89`) → locked producer, 1 toucher, **nothing drains it** | **single-ended** | **self-loop** |
  | `SliceRmStride` rank ≤ 4 | `c_0` (`…_rm_stride.cpp:69`) | `reader_multicore_slice_4d.cpp` produces (`:152,153,179`); `writer_multicore_slice_4d.cpp` consumes (`:81,82,93`) | **plain 1:1** | 1 PRODUCER + 1 CONSUMER |
  | `SliceRmStride` rank > 4 | `c_0` (same descriptor) | `reader_multicore_slice_nd.cpp` produces (`:137,138,166`); `writer_multicore_slice_nd.cpp` consumes | **plain 1:1** | 1 PRODUCER + 1 CONSUMER |
  | `SliceTile` | `c_0` (`…_tile.cpp:53-60`) | reader produces (`reader_unary_unpad_dims_interleaved_start_id.cpp:39,42`); slice's own writer consumes (`writer_unary_interleaved_start_id.cpp:48,50`) | **plain 1:1** | 1 PRODUCER + 1 CONSUMER |
  | `SliceTileTensorArgs` | `c_0` (`…_tile_tensor_args.cpp:56`) | reader produces (`…_tensor_args.cpp:117,120`); **eltwise donor** writer consumes | **plain 1:1** | 1 PRODUCER + 1 CONSUMER |
  | `SliceTileTensorArgs` | `c_1` — the start/end staging scratchpad (`…_tile_tensor_args.cpp:65`) | reader **only**, but it runs a full handshake on both sides: `reserve_back`/`get_write_ptr`/`push_back`/`wait_front`/`pop_front` twice (`…_tensor_args.cpp:52-59,66,69-76,83`) → 1 toucher holding both roles | **single-ended** | **self-loop** — the legacy code already behaves as one; the kernel body is untouched |

  **Hidden-second-writer hunt (face a): none — and the mechanism is structurally absent.** The face
  requires a raw co-fill coordinated by a dedicated semaphore pair; **this op has no semaphores at
  all**, so there is nothing to coordinate one. Every `get_write_ptr()` / `get_read_ptr()` in the op was
  individually attributed to the kernel that already holds that CB's FIFO role (a peek on its own
  binding) or, in `SliceRmSharded`, to the *sole* toucher.

  **Multiple-readers hunt (face b): none.** No CB is read by 2+ co-resident kernels. `SliceRmSharded`'s
  borrowed input CB comes closest — it is a sync-free borrowed tensor view read by base pointer — but
  that factory instantiates **exactly one kernel**, so the census is 1.

  **Dual-instance work-split hunt (face c): none.** No factory pushes the same `kernel_source` into two
  `KernelDescriptor`s; every factory's reader and writer are distinct sources, and `SliceRmSharded` has
  a single kernel. So no CB has two co-resident same-source touchers.

  **No dead CB.** All 7 `buffer_index` values are referenced by a bound kernel in every config — traced
  through the indirection the recipe warns about: positional CTAs (`…_rm.cpp:332` → writer CTA 0),
  **named** CTAs (`…_tile.cpp:139,161` → `get_named_compile_time_arg_val("dfb_id_in"/"dfb_id_out")`),
  and `tt::CBIndex` constants baked kernel-side (`slice_reader_…_rm_sharded.cpp:32,33`). Nothing is
  allocated-and-untouched, so there is no drop and no conditional DFB.

- **Offset base pointers:** **GREEN — the triage doc is stale for `slice`.**

  `analyses/2026-07-19_offset_base_pointers.md` lists `slice` under **Type 2 — accessor-fed offset arg**
  (its table row: `slice_program_factory_rm.cpp`, *"reader RTA[0] — input base"*, offset expression
  `input->address() + begins_bytes − misalignment`, caveat *"the canonical case"*). Type 2 is the
  hardest gate in that subject — ops team **plus** framework, flag early. **It no longer applies.**

  The recognition scan was run on **every** address RTA in the op, independently of the doc. Result:
  **every `->address()` in the op is a bare base with no arithmetic** — the five sites are all inside
  `patch_slice_program_addresses` (`slice_program_factory_rm_sharded.cpp:381,389,395,398,399`), the
  `override_runtime_arguments` refresh path, not a host fold. And in `create_descriptor` the base does
  not travel as an address at all: it is pushed as a **`Buffer*` binding**
  (`…_rm.cpp:377,385`; `…_rm_stride.cpp:128,136,147,160`; `…_tile.cpp:143,180`;
  `…_tile_tensor_args.cpp:151,168,182,183,184`).

  The offset has been **split out into a separate scalar arg**, exactly the shape the recipe describes
  as the Type-1 remedy — and here applied to what the doc had catalogued as Type 2:

  - Host emits it as its own RTA: `begins_bytes - misalignment` at `slice_program_factory_rm.cpp:103`,
    the last element of `common_reader_kernel_args`.
  - Kernel reads it as its own argument: `src_offset_bytes = get_arg_val<uint32_t>(13)`
    (`slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp:30`).
  - The accessor is built on the **unshifted** base: `TensorAccessor(src_args, src_addr, padded_stick_size)`
    (`:43`) — `src_addr` is RTA 0, the bare `Buffer*`-bound base.
  - The shift is applied **per read**, as the accessor's own `offset` parameter, never to the base:
    `:67` (`offset = src_offset_bytes + cur * chunk_size`) and `:101`
    (`/*offset=*/src_offset_bytes`), both reaching `noc.async_read(…, {.page_id, .offset_bytes})`.

  The kernel says so in a comment at `:41-42` — *"The accessor base stays the unshifted buffer base:
  Metal 2.0 supplies it from the tensor binding and offers no seam for a pre-offset base. The W-begin
  shift rides each read as `src_offset_bytes`."*

  This is the recipe's **third reconciliation outcome** — *fold absent, op in the tables → the doc is
  stale → GREEN* — and the op drops to ordinary tensor-binding port work (Case 1 on the now-clean
  base). No Type 1, no Type 2, no Type 3 (`address_offset` — Appendix A, absent), no Type 4
  (`ttnn::narrow` / interior-base `MeshBuffer::create` — absent).

  Beyond the RM path, the other factories never had a fold: `SliceTile` and `SliceTileTensorArgs` pass
  the start offset as a clean **tile-index scalar** folded into `start_id`
  (`…_tile.cpp:97,119,125`; `…_tile_tensor_args.cpp:112` computes it *on-device* from the start
  tensor), and `SliceRmSharded` passes `begins_bytes` as a **CTA** (`…_rm_sharded.cpp:310`) that the
  kernel adds to a local L1 offset (`slice_reader_…_rm_sharded.cpp:69`), never to a tensor base.

  → **Routing:** nothing to route. Suggest the triage-doc owner retire `slice`'s Type-2 row.

- **`TensorAccessor` 3rd argument:** **GREEN — two sites, both Class 2 (mechanical drop).**

  Only two of the op's twelve accessor constructions pass a 3rd argument; the other ten omit it.

  | Site | 3rd arg | Host value |
  |---|---|---|
  | `slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp:43` | `padded_stick_size` (RTA 1) | `per_shard_page_size_bytes(input_tensor, padded_row_size_bytes)` — `slice_program_factory_rm.cpp:86` |
  | `slice_writer_unary_stick_layout_interleaved_start_id.cpp:32` | `page_size_override` (RTA 7) | `per_shard_page_size_bytes(output_tensor, unpadded_row_size_bytes)` — `slice_program_factory_rm.cpp:155` |

  Both are in `SliceRmProgramFactory`, whose tensors may be interleaved **or** sharded, so the
  load-bearing question — *which specialization?* — has three answers. `per_shard_page_size_bytes`
  (`data_movement/common/common.cpp:782`) resolves them:

  1. **Interleaved** → returns `row_bytes`, i.e. the tensor's true logical page
     (`input_shape[-1]·E` for the reader, `output_shape[-1]·E` for the writer). Correct magnitude, on an
     **interleaved** accessor, which realigns the value up to the allocator alignment → **inert. Class 2.**
  2. **HEIGHT-sharded** → returns `t.buffer()->aligned_page_size()` — literally the value Metal 2.0
     supplies implicitly (`TensorAccessorArgs::AlignedPageSize`, itself
     `buffer.aligned_page_size()`: `tt_metal/impl/buffers/tensor_accessor_args.cpp:179-185`).
     `== aligned_page_size` → **Class 2.**
  3. **BLOCK/WIDTH-sharded** → returns `shard_spec.shape[1] · element_size`, which for a sharded RM
     tensor *is* `buffer->page_size()` (page shape is `{1, physical_shard_width}`:
     `tt_metal/impl/tensor/spec/layout/page_config.cpp:111-117`). `buffer->page_size()` is on the
     recipe's **correct-magnitude** list → **Class 2.**

  Neither site is Class 1: the page size cannot vary across a cache-reused shape, because
  `compute_program_hash` folds in `input.padded_shape()`, `input.memory_config()`, and the output
  spec's `padded_shape()` / `memory_config()` (`slice_device_operation.cpp:365-408`), so any change in
  row width or shard width lands in a different cache entry. The 2026-07-06 triage doc classifies
  `slice (interleaved RM path)` as *"1 — Dynamic page size (+ **S** base-offset)"*; **both halves are
  now stale** — the `S` base-offset is the fold that has since been split out (above), and the
  hash work removed the dynamism that made it Class 1. Per the subject's *"your own read is the source
  of truth"* rule, the from-code read decides: **Class 2, drop the arg, no `dynamic_tensor_shape`.**

  → **Port action:** drop both 3rd arguments; the accessor then takes `aligned_page_size` implicitly.
  In every configuration where the op is correct today the passed value *equals* that implicit value,
  so the drop is behaviour-preserving. **One sub-case is worth the porter confirming rather than
  swapping blind, and is raised as a question below** — see *Questions* #1.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding, 12 total — 10 Case 1, 2 clean, 0 Case 2):

  | Factory | Binding | Delivery today | Kernel use | Case |
  |---|---|---|---|---|
  | `SliceRm` | `input` | `Buffer*` @ reader RTA 0 (`…_rm.cpp:377`) | `TensorAccessor(src_args, src_addr, …)` (`:43`) | **1** |
  | `SliceRm` | `output` | `Buffer*` @ writer RTA 0 (`…_rm.cpp:385`) | `TensorAccessor(dst_args, dst_addr, …)` (`:32`) | **1** |
  | `SliceRmSharded` | `input` | **no address arg** — CB `c_0` borrowed from `input.buffer()` | `dfb_in.get_write_ptr()` + remote NOC coords (`slice_reader_…_rm_sharded.cpp:41,65,76`) | **clean** (borrowed-memory DFB; causal-link gate) |
  | `SliceRmSharded` | `output` | **no address arg** — CB `c_16` borrowed from `output.buffer()` | `dfb_out.get_write_ptr()` (`:42`) | **clean** |
  | `SliceRmStride` | `input` | `Buffer*` @ reader RTA 0 (`:128`/`:147`) | `TensorAccessor(src_args, src_addr)` (4d `:89`, nd `:94`) | **1** |
  | `SliceRmStride` | `output` | `Buffer*` @ writer RTA 0 (`:136`/`:160`) | `TensorAccessor(dst_args, dst_addr)` (4d `:72`, nd) | **1** |
  | `SliceTile` | `input` | `Buffer*` @ reader **CRTA 0** (`…_tile.cpp:143`) | `TensorAccessor(src_args, src_addr)` (`:26`) | **1** |
  | `SliceTile` | `output` | `Buffer*` @ writer RTA 0 (`…_tile.cpp:180`) | `TensorAccessor(dst_args, dst_addr)` (`:36`) | **1** |
  | `SliceTileTensorArgs` | `input` | `Buffer*` @ reader CRTA 0 (`:182`) | `TensorAccessor(src_args, src_addr)` (`:33`) | **1** |
  | `SliceTileTensorArgs` | `start_tensor` | `Buffer*` @ reader CRTA 1 (`:183`) | `TensorAccessor(start_args, start_addr)` (`:44`) | **1** |
  | `SliceTileTensorArgs` | `end_tensor` | `Buffer*` @ reader CRTA 2 (`:184`) | `TensorAccessor(end_args, end_addr)` (`:45`) | **1** |
  | `SliceTileTensorArgs` | `output` | `Buffer*` @ writer RTA 0 (`:151`/`:168`) | `TensorAccessor(dst_args, dst_addr)` (donor `:35`) | **1** |

  **No Case 2 anywhere.** `SliceRmSharded`'s reader does raw address arithmetic, but on **L1 CB
  pointers** obtained from a borrowed DFB — not on a tensor base — so it is the clean borrowed-memory
  path, not a raw-pointer tensor consumption. And **none of the 12 is the silent-wrong hazard**: every
  one arrives as a `Buffer*` binding, which the framework already patches on cache hits. This is
  routine port work, not a correctness fix. The same `TensorParameter` is **clean in `SliceRmSharded`
  and Case 1 elsewhere** — the per-factory split the recipe anticipates.

- **TensorParameter relaxation:** **`none`** (verbatim, all five rows) — the port applies no
  relaxation, and no `analyses/relaxations/data_movement_slice.md` is needed or expected.
- **`TensorAccessor` 3rd arg:** drop the redundant page-size arg at
  `slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp:43` and
  `slice_writer_unary_stick_layout_interleaved_start_id.cpp:32`. Class 2 → no `dynamic_tensor_shape`.
- **CB endpoints:** self-loop `c_0` and `c_16` in `SliceRmSharded`, and `c_1` in
  `SliceTileTensorArgs`; the remaining four CBs are plain 1:1. No multi-binding flag, no dead-CB drop,
  no conditional DFB.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** **none.** No CB in this op reaches ≥3 touchers or
  doubles a FIFO role, and the hidden-second-writer face is structurally absent (no semaphores).
- **Cross-op / shared kernels:**
  - `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` — cross-family
    **borrowed** kernel, bound by `SliceTileTensorArgs` (`…_tile_tensor_args.cpp:133`). **A `_metal2`
    fork already exists beside it** at
    `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id_metal2.cpp` (a true
    locational sibling, not under `experimental/quasar/`). → **Rung 1: reuse it.** Its interface, which
    becomes slice's constraint: DFB `dfb::out`, tensor `tensor::dst`, named args `args::num_pages`,
    `args::start_id`. That maps exactly onto what the factory supplies today
    (`{dst_buffer, num_tiles_per_core, num_tiles_written}`). It gates `#ifdef OUT_SHARDED` and
    `#ifdef BACKWARDS`; slice sets **no** `defines`, so neither fires — the same as today.
  - Slice's **own** copy, `device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`, is bound by
    `SliceTileProgramFactory` **and by nothing else** (verified: no file outside the slice directory
    references the slice kernels path). It is neither borrowed nor lent → **no fork needed; convert in
    place.** Do **not** be tempted to redirect `SliceTile` onto the shared `_metal2` fork: slice's copy
    exists precisely because it takes its DFB index from a **named** CTA
    (`get_named_compile_time_arg_val("dfb_id_out")`, `…_tile.cpp:161`) so the fusion infrastructure can
    remap it — a capability the shared fork does not have.
  - **Sunset list** for the eltwise legacy copy — *coordination and sunset tracking only, **not**
    authorization to convert it in place*: at least 15 factories bind that path today, among them
    `data_movement/concat`, `data_movement/reshape_on_device`, five `data_movement/tilize` factories,
    `eltwise/unary_backward/tanh_bw`, `embedding`, `examples/example` (×2),
    `experimental/matmul/attn_matmul`, `experimental/transformer/nlp_concat_heads`(+`_boltz`), plus
    `SliceTileTensorArgs`. A filename-level census returns ~44 factories, but many of those bind their
    own same-named private copy (slice itself is one), so the bound *path* is what counts. Tracked
    upstream as **issue #52228**, which also records a **second, duplicate** `_metal2` fork at
    `copy/typecast/device/kernels/dataflow/writer_unary_interleaved_start_id_metal2.cpp` (names its
    accessor `tensor::output`); the eltwise-sited fork is the one to bind.
- **RTA varargs:** six kernels carry genuine variable-count blocks. Named args stay the default for
  everything else — the fixed `arg_index++` runs below are explicitly **not** varargs.

  | Kernel | Site | Shape | Verdict |
  |---|---|---|---|
  | `slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp` | `:32-34` (`get_arg_addr(14)`) | three blocks of `num_dims` each (`num_unpadded_sticks`, `num_padded_sticks`, `id_per_dim`), where `num_dims` is **RTA 4** — a runtime value | **RTA vararg** |
  | `slice_reader_unary_unpad_dims_rm_sharded.cpp` | `:26-30` (`get_arg_addr(1)`, `get_arg_addr(1 + num_cores_read*2)`, `get_arg_addr(1 + num_cores_read*3)`, `chunk_start_id + 1`) | interleaved noc-x/y pairs and per-core chunk descriptors, all indexed off the runtime `num_cores_read` (RTA 0) — count-driven **and** data-selected | **RTA vararg** — the most involved case in the op |
  | `reader_multicore_slice_nd.cpp` | `:73-87` | **five** consecutive `tensor_rank`-length blocks (`input_dims`, `output_dims`, `slice_starts`, `slice_ends`, `slice_steps`), `rt_args_idx += tensor_rank` between each | **RTA vararg** |
  | `writer_multicore_slice_nd.cpp` | `:73` | one `tensor_rank`-length block (`output_dims`) | **RTA vararg** |
  | `reader_unary_unpad_dims_interleaved_start_id.cpp` | `:17-18` (`get_common_arg_addr(1)`), `:23` (`get_arg_addr(2)`) | `2·num_dims` **CRTA** block + `num_dims` **RTA** block. `num_dims` is a CTA (`:13`), which per the recipe **still** makes it a vararg — it varies across instantiations | **CRTA vararg + RTA vararg** |
  | `reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp` | `:25-26` (`get_common_arg_addr(3)`), `:92` (`get_common_arg_addr(3 + 2*num_dims)`), `:31` (`get_arg_addr(2)`) | `2·num_dims` + `num_dims` CRTA blocks (the second at a **computed** offset) + `num_dims` RTA block | **CRTA vararg + RTA vararg** |

  **Not varargs — name these:** `reader_multicore_slice_4d.cpp:52-77` (a fixed run of **25** distinct
  fields via `rt_args_idx++`) and `writer_multicore_slice_4d.cpp:52-61` (**9** fields), plus every
  scalar ahead of a vararg block elsewhere (`src_addr`, `num_dims`, `start_id`, `num_tiles`,
  `tensor_rank`, `element_size`, `num_rows_for_this_core`, `start_row_for_this_core`, and the
  `SliceRm` writer's full 11-arg fixed set). A sequential counter over a fixed set is legacy positional
  plumbing, not a loop.

  **CTA varargs: none.** Every `get_compile_time_arg_val` in the op uses a literal constant index; no
  count-driven compile-time-arg loop exists, so `KernelAdvancedOptions::compile_time_varargs` is not
  needed.
- **`ccl/mesh_partition` drives slice's factories from outside the op directory — the port will break
  its build.** `ttnn/cpp/ttnn/operations/ccl/mesh_partition/device/mesh_partition_program_factory.cpp`
  calls `SliceOp::validate_on_program_cache_miss` and `SliceOp::select_program_factory` (`:126-127`),
  then `Factory::create_descriptor(...)` on whichever slice factory it selects (`:131`), wraps the
  result in a `Program`, and refreshes it through `ttnn::prim::patch_slice_program_addresses` (`:155`).
  `patch_slice_program_addresses` is *deliberately* shared for this — see its comment at
  `slice_program_factory_rm_sharded.cpp:350-353` and
  `mesh_partition_device_operation.hpp:47`. On the readiness sheet `ccl/mesh_partition` is
  `Concept = legacy (MeshWorkload)` with `Is able to port? = no` and
  `Porting Target = TBD (SPMD + coord args issue)`, so it **cannot co-migrate**. Details and options in
  *Team-only*.
- **The `id_per_dim` vararg block is written in place by the kernel**, not just read:
  `slice_reader_…_rm_interleaved_start_id.cpp:80`, `reader_unary_unpad_dims_interleaved_start_id.cpp:45`,
  `…_tensor_args.cpp:124`. Confirm the vararg mechanism gives a **writable** L1 view before assuming a
  read-only one; the host relies on it (it seeds the block per core and the kernel advances it).
- **`SliceRmSharded` reads a remote core's L1 at the address of its *own* borrowed CB.** The kernel
  takes `l1_read_addr = dfb_in.get_write_ptr()` (`:41`) — a **local** pointer into its own borrowed
  input CB — and then uses that same value as the `.addr` of reads targeting *other* cores
  (`:65`, `:76`). That is correct only because a sharded CB lands at the same L1 offset on every core
  in the range. Preserve the borrowed-from binding for `c_0` so the invariant survives; it is not
  obvious from the call site.
- **Two kernels are already part-modernized, so their port is a binding-layer change, not an idiom
  rewrite.** `reader_unary_unpad_dims_interleaved_start_id.cpp` already uses
  `get_named_compile_time_arg_val("dfb_id_in")` (`:12`), `dfb_in0.get_entry_size()` (`:33`) rather than
  `get_tile_size(cb)`, and passes the DFB object straight into `noc.async_read(s0, dfb_in0, …)`
  (`:40`); slice's own `writer_unary_interleaved_start_id.cpp` is the same shape. The host side already
  emits `named_compile_time_args` for both (`…_tile.cpp:139,161`).
- **A `constexpr`-vs-`const` detail that decides token-form vs member-getter:**
  `…_tensor_args.cpp:15-16` declares `tile_width` / `tile_height` as **`const`** (not `constexpr`)
  while reading them from `get_compile_time_arg_val(3)` / `(4)`; every other CTA in that file is
  `constexpr`. Check which form the named-CTA replacement needs before swapping.

## Team-only

- **Out-of-directory coupling & donor shape.**

  **Op-level roll-up: ✓ clean.** No donor call shape needs work, and no donor blocks scheduling.

  *Function-call escape.* Every `#include` across the nine slice kernels resolves to one of two places:

  | Slice kernel(s) | Donor | Class | Status |
  |---|---|---|---|
  | all nine | `api/dataflow/dataflow_api.h`, `api/dataflow/noc.h`, `api/dataflow/dataflow_buffer.h`, `api/dataflow/endpoints.h`, `api/core_local_mem.h`, `api/tensor/noc_traits.h` | 1 — `tt_metal/*` LLK / HAL | ✓ no concern |
  | `slice_reader_…_rm_interleaved_start_id.cpp`, `slice_writer_…_stick_layout_…`, `reader_multicore_slice_4d/nd`, `writer_multicore_slice_4d/nd` | `ttnn/operations/data_movement/common/kernels/common.hpp` | 5 — **in-family shared** (`data_movement`) | ✓ clean |

  Per-call shape analysis for the one non-framework donor (71 files across the tree include it, so it
  is broadly shared in-family):

  | Function | Signature shape | Status |
  |---|---|---|
  | `noc_async_read_sharded(Noc, uint32_t l1_addr, AddrGenType tensor, uint32_t src_id, uint32_t offset, uint32_t size)` (`:375`) | **Shape 1** — `TensorAccessor<DSpec>` by value (template) + `Noc` | ✓ excellent — porter constructs `TensorAccessor(tensor::name)` and passes it |
  | `noc_async_write_sharded(...)` (`:325`) | **Shape 1** — same | ✓ excellent |
  | `tt_memmove<…>(Noc, uint32_t dst_l1_addr, uint32_t src_l1_addr, uint32_t bytes)` (`:143`) | plain L1 addresses + `Noc`; carries **no** CB, semaphore, or accessor handle | ✓ no bridge needed |

  No `uint32_t sem_id`, no `uint32_t sem_addr` / `uint64_t` NOC-encoded semaphore, no
  `TensorAccessorArgs<N>`, no tensor-CTA-offset NTTP, no old-style addr-gen, and — the row that is easy
  to read backwards — **no `CircularBuffer` / `CircularBuffer&` parameter anywhere**. Per-call detail is
  therefore complete above; there is no ⚠ / ✗ / ⭐ entry to expand.

  *File-path kernel instantiation (borrowed kernel files).* One entry, covered in *Heads-ups*:
  `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` (owner: `eltwise/unary`;
  broadly shared — ≥15 binding factories; `_metal2` fork **already checked in** beside it → rung 1
  reuse). Slice owns its other eight kernels and lends none of them.

  *Host-side coupling — `ccl/mesh_partition` (the significant one).* This is not a kernel escape, so no
  subject owns it, but it is the largest cross-op consequence of this port. `MeshPartition` reuses
  slice's device-op statics and factories wholesale (`mesh_partition_program_factory.cpp:31,126-134,155`),
  and shares `patch_slice_program_addresses` so the RTA slot layout has exactly one home. Porting
  slice's five factories to `CustomProgramSpecFactoryConcept` replaces `create_descriptor` with a
  spec-returning method and replaces `override_runtime_arguments` with a `ProgramRunArgs`-returning
  one, so both of `MeshPartition`'s call sites stop compiling. `ccl/mesh_partition` is itself
  `legacy (MeshWorkload)` / `Is able to port? = no`, so a bundled port is not available. Someone needs
  to decide, **before the port lands**, between: (a) keeping the legacy `create_descriptor` +
  `patch_slice_program_addresses` pair alive alongside the new spec path purely for `MeshPartition`;
  (b) porting `MeshPartition`'s program construction in the same change (out of the port's scope, and
  blocked by that op's own gate); or (c) sequencing slice's port behind `MeshPartition`'s. This routes
  to the **ops team / TTNN**, not to the porter.

- **Relaxation candidates** (noticed in the custom hash while auditing): **FALLIBLE — candidates to
  verify; default strict; the ops team owns the real analysis.** `compute_program_hash`
  (`slice_device_operation.cpp:343-427`) keys on the **full** input spec (`logical_shape`, `rank`,
  `padded_shape`, `layout`, `dtype`, `memory_config`) plus the full output spec, plus the start/end
  tensors' and preallocated output's specs when present, plus `factory.index()`. The gating
  `TensorParameter relaxation` column reads `none`, and this hash is consistent with that: nothing here
  suggests slice depends on fewer tensor properties than the strict match provides. The one direction
  worth an eventual look is the *opposite* of a relaxation candidate — the hash is deliberately
  **over**-keyed relative to the default (see the comment at `:344-348`, and the `#53997` /
  `#47602` / `#45144` history), so any future relaxation work on this op should start from why the
  extra keys were needed, not from trimming them. The sheet's own
  `Provisional relaxation finding (Edwin)` cell (`needs fix, then none`, on the `SliceRm` and
  `SliceTile` rows) appears to predate that hash work; see *Questions* #2.

- **TTNN factory analysis** (sheet-derived facts, with `file:line` evidence):
  - **Concept (current):** `descriptor` on all five factories — `create_descriptor` returning a
    `ProgramDescriptor` (`slice_program_factory_rm.hpp:26` and the four peers).
  - **Op-owned tensors:** none. No `create_workload_descriptor`, no `buffers` vector. Consistent with
    the `descriptor` concept, which cannot carry them.
  - **MeshWorkload need:** none for slice itself. (`ccl/mesh_partition`, which wraps slice's factories,
    *is* a MeshWorkload and is `Secretly SPMD = yes / Why: Per-coord RTAs` — that is that op's finding,
    not slice's.)
  - **Pybind `create_descriptor`:** `slice_nanobind.cpp:168-179`, on `SliceTileProgramFactory` only.
    Not a gate — the port **deletes** this binding, which is a user-visible API change and gets its own
    entry in the port report. A replacement is expected later. The device-op is also pybound for
    `create_output_tensors` / `compute_output_specs` (`:156-166`); those are not descriptor internals
    and are not this signal. `SliceParams` / `SliceInputs` are pybound as plain structs (`:138-154`).
  - **Other risky pybind:** none beyond the above.
  - **Custom hash:** `slice_device_operation.cpp:343`. Not a gate; **the port leaves it exactly as it
    is** — no rewrite, no trimming, no re-derivation. It is independent of the concept choice.
  - **`get_dynamic_runtime_args`:** absent (zero grep hits). Gate conjunct confirmed clear.
  - **`override_runtime_arguments`:** present on all five factories (sites in the status summary), each
    a one-line delegation to `patch_slice_program_addresses`
    (`slice_program_factory_rm_sharded.cpp:354-413`). This is the **target-concept selector**, not a
    gate: slice ports to `CustomProgramSpecFactoryConcept`, and the porter **translates** that method
    into one returning a `ProgramRunArgs` rather than deleting it. Note the translation is unusual in
    shape: the five factories share one implementation that branches on the factory type
    (`std::holds_alternative` at `:362`, then `std::visit` at `:383`), reaching three different
    refresh mechanisms — `apply_descriptor_runtime_args` for the two borrowed CBs (RmSharded, `:366`),
    a positional `GetRuntimeArgs` slot-0 patch (`patch_slot0`, `:372-380`), and
    `apply_dynamic_runtime_args` over a `DynamicRuntimeArg` vector for the tile factories
    (`:394-409`, plus `slice_tile_dynamic_args` at `slice_program_factory_tile.cpp:198`). Note also
    that `tt::tt_metal::apply_dynamic_runtime_args` / `DynamicRuntimeArg` here is a **helper API**, not
    the deprecated device-op `get_dynamic_runtime_args` hook — do not conflate them.
  - **Target concept:** `CustomProgramSpecFactoryConcept`, no op-owned tensors. Independently derived
    from `Override runtime args method? == yes` and confirmed by the sheet's own `Porting Target` cell.

## Misc anomalies  *(team-only, non-gating, not porter-actionable)*

- **Dead CTA in four kernels.** `compile_time_element_size = get_compile_time_arg_val(1)` is declared
  and never used at `reader_multicore_slice_4d.cpp:81`, `writer_multicore_slice_4d.cpp:65`,
  `reader_multicore_slice_nd.cpp:67`, `writer_multicore_slice_nd.cpp:66`. All four kernels use the
  *runtime* `element_size` throughout. The host still emits it (`slice_program_factory_rm_stride.cpp:79,82`).
- **Dead RTAs in the 4D stride writer.** `output_h`, `output_d`, `output_n`
  (`writer_multicore_slice_4d.cpp:56-58`) are read and never used — only `output_w` is. The host emits
  all four (`slice_program_factory_rm_stride.cpp:139-142`). Same for `tensor_rank` (`:54`) in that file.
- **`SliceTileTensorArgs` reads the end tensor and discards it.** The reader does a full DFB-staged
  read of the end tensor and copies it into `end_indices`
  (`reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp:69-83`), but `end_indices` is declared
  `[[maybe_unused]]` (`:49`) and never read afterwards — only `start_indices` feeds the offset
  computation. The op nonetheless binds the tensor, allocates its accessor args
  (`slice_program_factory_tile_tensor_args.cpp:84`), hashes its spec
  (`slice_device_operation.cpp:388-398`), and `TT_FATAL`s if it is absent (`…_tensor_args.cpp:46`).
  Either the end tensor is genuinely unnecessary (a whole binding, a CB staging round-trip, and an
  accessor could go) or a use was intended and lost. **The port must keep binding it** — the read still
  happens — so this is an ops-team question, not port work.
- **`constexpr uint32_t start_offset = 0;`** at `slice_program_factory_tile_tensor_args.cpp:129`, added
  into `start_id` at `:158`. Always zero (the real offset is computed on-device from the start tensor),
  so the addition is a no-op kept for symmetry with `SliceTileProgramFactory`. Harmless; noted because
  it reads like a forgotten value.
- **Dead preprocessor branches in slice's own writer copy.** `writer_unary_interleaved_start_id.cpp`
  gates on `#ifdef OUT_SHARDED` (`:29`) and `#ifdef BACKWARDS` (`:38`), inherited from the eltwise
  original. No slice factory sets any `defines`, so neither can ever fire.
- **Belt-and-braces address refresh.** For `SliceRm` and `SliceRmStride`, arg slot 0 is *both* a
  `Buffer*` binding (which the framework patches on a cache hit) *and* manually rewritten by
  `patch_slot0` (`slice_program_factory_rm_sharded.cpp:381,389`). The manual patch is redundant for
  the op's own dispatch path; it is presumably load-bearing for `MeshPartition`, which builds a bare
  `Program` from the descriptor (`mesh_partition_program_factory.cpp:131-132`) and so may not get the
  framework's binding injection. Worth confirming, since if it *is* redundant it is also the thing
  making the tile factories' `patch_slot0`/`apply_dynamic_runtime_args` split necessary.
- **`get_arg_addr(rt_args_idx++)`** at `writer_multicore_slice_nd.cpp:73` — the post-increment has no
  effect (nothing reads `rt_args_idx` afterwards). Cosmetic.
- **Interleaved-pair pointer trick.** `slice_reader_unary_unpad_dims_rm_sharded.cpp:26-27` takes
  `read_noc_x = get_arg_addr(1)` and `read_noc_y = get_arg_addr(2)` — two pointers one word apart into
  the same interleaved x/y stream — then strides both by 2 (`:85`). Correct, but it is the kind of
  aliasing that a vararg-block port could easily get subtly wrong; called out here as well as in the
  porter heads-ups.

## Per-DeviceOperation / per-factory attribution

Single DeviceOperation (`SliceDeviceOperation`), so the per-DOp roll-up equals the op roll-up: **GREEN**.
Where findings differ, they differ by *factory*:

| Factory | `Is able to port?` | Tensor bindings | 3rd arg | CB dispositions | Kernels | Notable |
|---|---|---|---|---|---|---|
| `SliceRm` | yes | `input` C1, `output` C1 | **2 sites → drop** | `c_0` 1:1 | own ×2 | the ex-Type-2 offset fold, now split out |
| `SliceRmSharded` | yes | both **clean** (borrowed DFB) | none | `c_0` self-loop, `c_16` self-loop | own ×1 | only factory with borrowed CBs; no `TensorAccessor` at all |
| `SliceRmStride` | yes | `input` C1, `output` C1 | none | `c_0` 1:1 (both rank configs) | own ×4 | ND path is the heaviest RTA-vararg case |
| `SliceTile` | yes | `input` C1, `output` C1 | none | `c_0` 1:1 | own ×2 | already uses named CTAs; carries the pybound `create_descriptor` |
| `SliceTileTensorArgs` | yes | ×4, all C1 | none | `c_0` 1:1, `c_1` self-loop | own ×1 + **eltwise donor** | reuse the existing `_metal2` fork; end tensor read-and-discarded |

## Questions for the user  *(all three resolved)*

> **Update (2026-09-04), after reviewing PR #55433 (`akertesz/slice-test`, draft).** Questions 1 and 3
> below are **answered and closed**; the branch already carries the code. **Question 2 was closed by
> the op owner on 2026-09-04** — the spreadsheet cell is stale and the relaxation really is `none`.
> Nothing in this audit remains open. See *Post-audit reconciliation* at the end of this document for
> the branch delta and the recipe-version check.

1. ~~**`TensorAccessor` page size on a BLOCK/WIDTH-sharded RM tensor?**~~ — **CLOSED. Not a bug, and
   the answer was already in the op directory; I should have found it rather than asking.**
   `has_subaligned_shard_row` (`slice.cpp:47-57`) already exists at the audited HEAD and documents the
   exact mechanism this question derived, in the same terms: *"TensorAccessor strides pages by the
   buffer's **aligned** page size, but `noc_async_*_sharded` reads that same value back as the per-page
   **payload**. For a B/W-sharded RM buffer the page is the shard row, so the two only agree when the
   row is already a multiple of the buffer's alignment."* `needs_rm_composite_input` (`:60-77`) and
   `needs_rm_composite_output` (`:79-86`) then route any such tensor down the **composite** path, so
   `SliceRmProgramFactory` never receives one through `ttnn::slice`. So: an unaligned B/W shard width is
   a supported *slice* configuration but an unreachable *factory* configuration — the tensor is
   resharded first. **The 3rd-arg drop is safe, and it has already been done** (`87bd11a885e`), which
   also removed the two now-dead host RTAs and reindexed both kernels, exactly as the brief prescribed.
   PR #55433 additionally closes the one residual hole this audit identified — `MeshPartition` builds
   these programs off `select_program_factory` and never sees `slice.cpp`'s guard — with a
   `TT_FATAL` (`check_accessor_page_size`, `c65cafac4ee`) pinning the equivalence in
   `SliceRmProgramFactory::create_descriptor`. **My audit gap:** I skimmed `slice.cpp` with a keyword
   grep instead of reading it, and the pattern missed the guard. The mechanism analysis was right; the
   reachability conclusion was available in scope and I raised it as a question instead of answering it.

2. ~~**`Provisional relaxation finding (Edwin)` = `needs fix, then none` — still current?**~~ —
   **CLOSED by the op owner (2026-09-04): the spreadsheet cell is stale. The relaxations are fine and
   the value is `none`.** That matches the gating `TensorParameter relaxation` column (`none` on all
   five rows), this audit's own relaxation-candidate read (no candidate surfaced; the hash pins
   essentially the full `TensorSpec`), and the two factories already ported with no relaxation applied.
   **No relaxation work, no analysis doc, nothing to route.** The only residue is housekeeping: the
   `Provisional relaxation finding (Edwin)` cell should be cleared on the `SliceRm` and `SliceTile`
   rows so the next auditor doesn't weigh it against the gating column — and note the column is
   undocumented in `ttnn_op_porting_readiness.md`, which is what made it ambiguous to two independent
   auditors (see *Recipe notes*).

3. ~~**`ccl/mesh_partition` sequencing.**~~ — **CLOSED. Option (b) was chosen and is implemented**, in
   the same commit as the first factory port (`8c8b9eea947`). Rather than a name list, it adds a
   concept keyed on the *entry point* —
   `template <typename T> concept IsSliceSpecFactory = requires { &T::create_program_artifacts; };` —
   and branches both call sites on it: `create_at` builds a spec factory via `MakeProgramFromSpec` +
   `SetProgramRunArgs` and leaves the descriptor path untouched; `override_runtime_arguments` routes a
   spec factory through `UpdateProgramRunArgs(program, Factory::override_runtime_arguments(...))` and
   keeps `patch_slice_program_addresses` for the rest. Because it keys on the entry point, **each
   further slice factory that migrates needs no additional edit there**, and the branch retires when
   the last one converts. Two caveats to carry: the change was **explicitly authorized by the invoker**
   as an out-of-op-directory edit, and it is **not run-verified** — MeshPartition's tests are
   t3000/TG-only. It also invalidates my framing of this as a blocking sequencing decision: the port
   turns out to be **incrementable per factory** (see *Post-audit reconciliation*).

## Recipe notes

1. **The `TensorAccessor` 3rd-arg magnitude test does not resolve `page_size` vs `aligned_page_size`
   on a *sharded* accessor.** the `TensorAccessor 3rd argument` subject of `metal2_audit.md` question 2 says to compare the value to
   "the true logical page (`buffer->page_size()`)" and lists `buffer->page_size()` among the
   **correct-magnitude** values; Class 2's first clause is separately "`== aligned_page_size`". For an
   interleaved accessor these agree, because realignment makes the difference inert — and the section
   says so. For a **sharded** accessor they can disagree by up to `alignment - 1` per page, and the
   same section says sharded "uses the passed value **verbatim** … *any* wrong value mis-addresses."
   So a site passing exactly `buffer->page_size()` on a sharded accessor is simultaneously
   correct-magnitude (not Class 3/4) and not `== aligned_page_size` (not Class 2's first clause) — the
   taxonomy has no row for it. I resolved it toward Class 2 via Class 2's second clause plus the
   port-relevance test. **The gap is real and is now independently evidenced:** the ops team needed a
   bespoke predicate (`has_subaligned_shard_row`, `slice.cpp:47`) *and* a `TT_FATAL`
   (`check_accessor_page_size`, PR #55433) to pin exactly this equivalence — which is more machinery
   than a taxonomy with no row for the case would lead an auditor to expect. Suggest the section either
   add `buffer->page_size()` on a sharded accessor as an explicit Class-2 sub-case with the
   "coincides only when the shard row is alignment-aligned — check for an op-side guard" caveat, or
   name it a Special. (Slice is a live example, and not the only one:
   `per_shard_page_size_bytes` is shared with `data_movement/gather`, which passes the same value to
   four accessors, so whatever is decided here applies there too.)

   *Auditor's note, not a recipe defect:* the question this raised was answerable from the op directory
   (`slice.cpp`'s guard), and I asked it instead of answering it. The recipe told me to audit the op;
   I under-read its host-facing file. Recorded here only because the next auditor of a
   page-size-override site should **grep the op root for a guard predicate before escalating** —
   that specific habit would be a useful addition to the subject's two classifying questions.
2. **"Fixed `arg_index++` run" vs "vararg" is clear; "computed CRTA offset" is a third shape worth
   naming.** the `RTA varargs` subject of `metal2_audit.md` contrasts a counted loop (vararg) with a fixed `arg_index++` run
   (nameable). Slice's tile readers do neither: they take a **base pointer** at a *host-computed*
   offset — `get_common_arg_addr(3 + 2 * num_dims)`
   (`…_tensor_args.cpp:92`) — and index the block by loop variable. It is unambiguously a vararg by the
   spirit of shape (a), but no bullet quite describes "a `get_*_arg_addr` into a block whose *start*
   is a function of another arg." Naming it would save the next auditor a derivation, and it is common
   (six of nine slice kernels).
3. **Recipe assumed slice would still be the offset-base exemplar; it isn't, and the doc already knew.**
   the `Offset base pointers` subject of `metal2_audit.md` says "Reach that verdict from your scan, not from this paragraph. The
   slice family is the catalogued example, which is exactly why it is the one most likely to have been
   fixed since" — and the recipe's own newest commit is
   `state the offset-base wall as a category, not as slice's current state`. Both were exactly right
   and made the third-outcome call easy. Recording it as a positive: the two-source design (dated doc
   as prior, code as truth) worked as intended here, and the pre-warning is what stopped a
   doc-anchored lookup from RED'ing a GREEN op.
4. **No subject owns a host-side out-of-directory *consumer*.**
   the `Out-of-directory coupling` subject of `metal2_audit.md` covers two escape types, both kernel-facing: function-call escape and
   file-path kernel instantiation. Slice's most consequential cross-op coupling is neither — it is
   another op (`ccl/mesh_partition`) calling slice's **host** `create_descriptor` and a shared
   address-patching helper, which the port's signature change breaks. I put it in the brief's open
   "anything else" bullet and in Team-only, which works, but a one-line third escape type ("host-side:
   another op calls this op's factory statics or device-op statics — grep the factory type names and
   any exported helpers outside the op directory") would make it a found finding rather than an
   incidental one. It is cheap to check and, when present, is a scheduling blocker.

---

## Post-audit reconciliation — PR #55433 (`akertesz/slice-test`)

Reviewed after this audit was written, at the user's request. It does not change the **GREEN** verdict;
it closes two of the three open questions (above) and adds context this audit did not have.

**The port is already underway on that branch, and it is incremental per factory.** Two of the five
factories are ported to `CustomProgramSpecFactoryConcept` — `SliceTileProgramFactory`
(`8c8b9eea947`) and `SliceTileTensorArgsProgramFactory` (`aafc364bc0c`), both by Audrey Kertesz. The
remaining three stay on `ProgramDescriptorFactoryConcept`, and *"the framework dispatches per-factory,
so the op builds and runs with its factories on mixed concepts."* **This correction matters for
planning:** this audit's brief was written as though the five factories port together, and framed the
`ccl/mesh_partition` coupling as a blocking sequencing decision. Neither holds — the port can land one
factory at a time, and the `MeshPartition` bridge is already in place for all of them.

Reported verification for the first ported factory: Wormhole n150, Metal 2.0 legality checks forced on,
`test_slice.py` **448 passed / 38 skipped, identical to the pre-port baseline**.

**Ops-team work this audit called for is already done**, on the same branch and by the op's own author
(Edwin Lee) rather than as a separate track:

| Commit | What it does | This audit's corresponding item |
|---|---|---|
| `f91637843f4` | validate pre-allocated output on every input layout | — (new) |
| `87bd11a885e` | **drops both `TensorAccessor` 3rd args**, removes the two now-dead host RTAs, reindexes both RM kernels | *Port-work summary* → `TensorAccessor 3rd arg` |
| `c65cafac4ee` | adds `check_accessor_page_size` — a `TT_FATAL` pinning per-shard vs. aligned page size in `SliceRmProgramFactory::create_descriptor` | *Questions* #1 |
| `30f896705ba` | restores a row-major logical-shape check on the preallocated output | — (new) |

**Two artifacts on that branch differ from the ones in this directory.** The branch carries its own
`METAL2_PREPORT_AUDIT.md` (242 lines) and `METAL2_PORT_BRIEF.md` (131 lines), plus a
`METAL2_PORT_PLAN.md` and `METAL2_PORT_REPORT.md` from the port itself.

### Recipe-version delta — checked 2026-09-04

**This audit ran on the *newer* audit recipe, not the older one.** (An earlier revision of this section
said the opposite; that was wrong, and the commit dates are why.) The branch's docs cite
`1167faf7b42 2026-09-04 docs(metal_2.0): binary_ng relaxation analysis; invariant checks over commit
stamps`; this document is pinned to `4bd4bf42bfe 2026-09-03`. The later *date* belongs to a
**divergent branch** — `1167faf7b42` is reachable only from `origin/akertesz/slice-test`, and the merge
base with the recipe branch is `1c2aff5064f` on `main`. On the audit recipe's own content,
`4bd4bf42bfe` is the newer revision.

**The whole delta across the `metal_2.0` doc tree between those two commits is one file, one hunk:**
`ai/audit/metal2_audit.md`, *Offset base pointers → Code-path scope* (+1 / −3 lines).
`ai/port/metal2_port.md` and every shared/analyses doc are **byte-identical**, so nothing in the delta
touches the port recipe the porter will follow.

That one hunk is `4bd4bf42bfe` itself — *"state the offset-base wall as a category, not as slice's
current state"* — and it is precisely the paragraph that decides this op's offset-base verdict:

- **`1167faf7b42` (what the branch's audit read)** states the wall in the indicative and pre-writes the
  verdict: *"The wall is a row-major-layout phenomenon: the tiled variants of the same ops (`slice` /
  `padded_slice` / `slice_write`) … So a slice-family RED applies Code-path scope — RED the RM factory
  …"*
- **`4bd4bf42bfe` (what this audit read)** makes it conditional and adds:
  *"**Reach that verdict from your scan, not from this paragraph.** The slice family is the catalogued
  example, which is exactly why it is the one most likely to have been fixed since. A listed op with no
  fold is the third outcome above — the doc is stale, and the op is **GREEN**."*

The fix commit's own message names the failure it repairs: *"a cold auditor reported slice RED after
the blockers had been removed."* **The loop is closed in both directions:** the branch audit's
*Recipe notes* #1 diagnosed exactly this and proposed making the paragraph conditional; `4bd4bf42bfe`
implements that; and this audit, reading the fixed text, reached GREEN from its own scan. So the delta
is real, it is confined to this one subject, and it worked — no re-audit is warranted on account of it.

### Where the two audits differ, and why

They agree on all five gates, the GREEN verdict, the target concept, the `none` relaxation, the CB
census (4 legal + 3 self-loop), and *"no Case 2 anywhere"*. Three differences, all explainable:

| Item | This audit | Branch audit | Cause |
|---|---|---|---|
| `TensorAccessor` 3rd arg | **2 sites, Class 2 → drop** | **none — all 12 sites two-arg** | **Code state, not judgement.** The branch includes `87bd11a885e`, which already dropped both args. Each audit is correct for the tree it read. |
| Tensor-binding roll-up | 10 Case 1 + 2 clean | *"eight Case 1, two clean"* | **A miscount in the branch audit's summary line** — its own table lists 12 rows, 10 of them Case 1. Worth correcting there, since its brief and port report may inherit the figure. |
| Offset-base reasoning | reached from scan, recipe conditional | reached from scan **against** a recipe paragraph that pre-committed to RED | the recipe delta above |

**Whoever reconciles this directory should decide which pair survives** rather than merging blindly.
This document is longer and carries the per-`(CB, config)` census, the full vararg table, the
`MeshPartition` analysis and the resolved question trail; the branch's pair describes the
**post-fix** tree (no 3rd-arg work outstanding) and sits next to the port plan and report it produced.
The cleanest outcome is probably to keep the branch's pair as the live artifacts and fold this
document's extra detail into it.
