# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/padded_slice`

One device operation, two program factories, both declared in the `program_factory_t` variant of
`device/padded_slice_device_operation.hpp:25`:

- **`PaddedSliceDeviceOperation`** (`device/padded_slice_device_operation.{hpp,cpp}`)
  - `PaddedSliceRMProgramFactory` (`device/padded_slice_rm_program_factory.cpp`) — row-major input, sharded output; `create_descriptor` + `override_runtime_arguments`
  - `PaddedSliceTileProgramFactory` (`device/padded_slice_tile_program_factory.cpp`) — tiled input, untilized row-major sharded output; `create_descriptor` only

Kernels, all referenced (no dead kernel files in the directory):

| Kernel | Bound by | Owner |
|---|---|---|
| `device/kernels/dataflow/padded_slice_reader_rm_interleaved_start_id.cpp` | RM reader | this op |
| `device/kernels/dataflow/writer_unary_sharded_padded_rm.cpp` | RM writer when the output row is wider than the sliced row (`pad_output_row`) | this op |
| `data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp` | RM writer otherwise | **borrowed** (`data_movement/sharded`) |
| `device/kernels/dataflow/padded_slice_reader_tiled_interleaved_start_id.cpp` | Tile reader | this op |
| `sliding_window/halo/device/kernels/compute/pack_untilize.cpp` | Tile compute | **borrowed** (`sliding_window/halo`) |
| `device/kernels/dataflow/writer_unary_sharded_padded_tiled.cpp` | Tile writer | this op |

Host helpers `device/slice_cb_descriptor.hpp` and `device/padded_slice_utils.{hpp,cpp}` are **lent**: the
three `experimental/slice_write` factories include them (see *Team-only → Out-of-directory coupling*).

> **`ttnn/cpp/ttnn/operations/experimental/quasar/padded_slice/` exists and is out of bounds.** It is a
> whole-op shortcut copy, not a precedent. Nothing in it was read for this audit, and the porter must
> not read it either; its `_metal2` kernels do **not** count as forks to reuse.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `aab653ac9b9 2026-10-27 docs(metal_2.0): import the op-porting recipe docs for the padded_slice audit`
(verbatim import of `akertesz/op-porting-recipe` @ `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`; the recipe branch is an orphan history and cannot be merged into `main`)

**Code audited:** `origin/main` @ `497d9e2dc34` (2026-10-26) **plus the unmerged pre-port fix**
`f574e0b3430` — *"[ttnn] padded_slice: pass the RM reader a clean input base and port its TRID path to
the Noc API"* (PR #62499, since merged into `main` as `0226df45c4a` on 2026-10-26; this branch is rebased on a
`main` that carries it). **The GREEN below is conditional on that fix** — now met; against `main` without it the RM factory fails two
gates (see *Gate detail*). This is the ops-team pre-port fix the recipe requires to be a separate,
fully tested PR; it is deliberately not part of the port.

**Readiness sheet:** fetched live this session, twice (morning and after the sheet owner's update);
2 rows for `experimental/padded_slice` (sheet rows 199–200), one per factory. The morning export still
carried the pre-#57550 state (`legacy device-op`); the owner reconciled both rows during this session,
and the verdict below reads the reconciled rows.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/padded_slice` |
| **Overall** | **GREEN — both factories**, conditional on pre-port fix `f574e0b3430` (draft PR #62499) landing first |
| **DOps / Factories** | `PaddedSliceDeviceOperation` → `PaddedSliceRMProgramFactory`, `PaddedSliceTileProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** (with the fix) — all four op DM kernels are `Noc` / `CircularBuffer` (`experimental::CB` alias) / `TensorAccessor` / `UnicastEndpoint`; the borrowed DM writer is already on `DataflowBuffer`; the borrowed compute kernel is a TRISC kernel (DM migration N/A). **Without the fix:** RED on the RM reader's non-aligned path (raw `noc_async_read` + TRID polling) |
| *Prereqs* — Cross-op escapes | Ok — two donor headers (`data_movement/common/kernels/common.hpp`, `pool/device/kernels/experimental_device_api.hpp`), every called function takes raw `uint32_t` / `Noc`; two borrowed kernel files (one with an existing `_metal2` fork, one without) |
| *Feature Support* — overall | **GREEN** — every Appendix A entry `N/A` |
| *Feature Support* — Variadic-CTA | Ok — no varying-index compile-time-arg reads |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes × 2** (reconciled rows; the morning export read `yes (with PD step)` on a `legacy device-op` concept, which the code contradicted — resolved by the sheet owner this session) |
| *TTNN Readiness* — Concept (current) | `descriptor` × 2 — cross-check ✓ (`create_descriptor` at `padded_slice_rm_program_factory.cpp:260`, `padded_slice_tile_program_factory.cpp:369`; PR #57550) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Custom hash | No — sheet `no` × 2; no `compute_program_hash` / `attribute_values` / `to_hash` in the op ✓ |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No — sheet `no` × 2; absent from the device op ✓ |
| *TTNN Readiness* — `override_runtime_arguments` | **Yes on the RM factory** (`padded_slice_rm_program_factory.cpp:411`, declared `.hpp:20`) — sheet `yes` ✓; **No on the Tile factory** — sheet `no` ✓. Not a gate; selects the concept |
| *TTNN Readiness* — Pybind `create_descriptor` | No — `padded_slice_nanobind.cpp:69` binds only the `ttnn.experimental.padded_slice` wrapper; sheet `no` ✓ |
| *TTNN Readiness* — Op-owned tensors | No — sheet `no` × 2; both factories return a plain `ProgramDescriptor` ✓ |
| *TTNN Readiness* — Target concept | RM → **`CustomProgramSpecFactoryConcept`** · Tile → **`ProgramSpecFactoryConcept`** (sheet `Porting Target` agrees on both) |
| *Port work* — Offset base pointer | **none** (with the fix) — the RM reader base is a bare `->address()`, the W-begin / width offset ride as separate scalars (`padded_slice_rm_program_factory.cpp:226,234-236`); the Tile reader base is a `Buffer*`. **Without the fix:** Type 2 GATE on the RM factory (the fix is the split) |
| *Port work* — Tensor bindings (per binding) | `input`: Case 1 × 2 · `output`: clean (borrowed-memory CB) × 2 · no Case 2 |
| *TTNN Readiness* — TensorParameter relaxation | `none` × 2 → clears |
| *Port work* — TensorAccessor 3rd arg | 1 site, **Class 2** (redundant) → drop (RM reader `:45`); Tile reader passes none |
| *Port work* — CB endpoints | RM: 1P+1C × 1 (output CB, both writer configs) · self-loop × 2, each a **conditional DFB** (`temp_pad` on `pad_output_row`, `non_aligned` scratch on `is_non_aligned`) · Tile: legal 1:1 × 2 (input, untilized) · self-loop × 2 (output, padding) |

**CB endpoints** are dispositions, not gates: every CB above has a port-time resolution, recorded per
`(CB, config)` in *Gate detail*.

## Result

**GREEN → brief issued for both factories — conditional on the pre-port fix.** With
`f574e0b3430` applied, Device 2.0, Appendix A feature compatibility, the readiness sheet, offset base
pointers, the TensorAccessor 3rd argument and the `TensorParameter` relaxation column all clear for
both factories. The port can begin once that fix is on `main` (or on the port branch, which already
carries it).

**What the fix cleared, for the record.** Against `origin/main` @ `497d9e2dc34` the RM factory fails
two gates and the audit would read `RED at op level; subset PaddedSliceTileProgramFactory is clear`:

1. **Offset base pointer, Type 2** — `padded_slice_rm_reader_address()` folded
   `input->address() + begins_bytes + width_offset` into reader RTA 0 (pre-fix
   `padded_slice_rm_program_factory.cpp:70-72`), and the reader built its `TensorAccessor` on that
   shifted base (pre-fix reader `:45`). This is the catalogued `padded_slice` entry in
   `2026-07-19_offset_base_pointers.md` (Type 2, "two-stage host fold"), and PR #60876 (2026-10-16)
   had deepened it rather than removed it.
2. **Device 2.0** — the non-aligned reader path used raw `noc_async_read_set_trid` / `noc_async_read` /
   `ncrisc_noc_read_with_transaction_id_flushed` / `get_noc_addr` (pre-fix reader `:87-185`); the
   kernel comment claiming the `Noc` API lacked per-TRID polling was stale (`Noc::is_read_trid_flushed`,
   `async_read<NocOptions::TXN_ID>` exist on `main`).

The fix follows the shape `data_movement/slice`'s RM factory took in its own Metal 2.0 port (#56494):
bind the clean base, pass the aligned-down offset and the misalignment as scalars, apply the offset per
read through the accessor. Reviewers of the fix PR should hold it to the recipe's bar for pre-port
fixes: a separate PR, no functional change, fully tested before the port.

**Readiness-sheet note.** The morning export of rows 199–200 read `Concept = legacy device-op`,
`Override runtime args method? = n/a`, `Is able to port? = yes (with PD step)` — the pre-#57550 state,
contradicted by the code on two primary columns. Per the recipe that is a "spreadsheet broken" gate
routed to the sheet owner; it was routed, and the re-fetched rows now read `descriptor`,
`PD Op (custom)` / `PD Op (pointer-patching)`, override `yes` / `no`, `Is able to port? = yes`,
`Diego validation = yes`. The cross-check below is against the reconciled rows. (The sibling
`experimental/slice_write` rows 294–296, from the same PR #57550, are still in the stale state — not
this op's gate, flagged for the sheet owner.)

**Subject coverage.** All twelve subjects were run in full for both factories; nothing was skipped
or deferred.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN for both** — both rows read `yes`, every
  cheaply-checkable primary column matches the code, cross-column invariants hold, factory-set match ✓
  (2 rows ↔ 2 factories, no phantom, no missing):

  | Factory (sheet row) | `Concept` | `Op Classification` | `Override…?` | `Custom hash` | `get_dynamic…` | `Pybind descriptor` | `Op-owned tensors?` | `TensorParameter relaxation` | `Is able to port?` | Cross-check |
  |---|---|---|---|---|---|---|---|---|---|---|
  | `PaddedSliceRMProgramFactory` (199) | `descriptor` | `PD Op (custom)` | `yes` | `no` | `no` | `no` | `no` | `none` | `yes` | ✓ `create_descriptor` `.cpp:260`; `override_runtime_arguments` `.cpp:411` |
  | `PaddedSliceTileProgramFactory` (200) | `descriptor` | `PD Op (pointer-patching)` | `no` | `no` | `no` | `no` | `no` | `none` | `yes` | ✓ `create_descriptor` `.cpp:369`; no override; input address delivered as `Buffer*` (`.cpp:44`), patched by the framework |

  `Smuggled pointer = no` on both rows ✓: the RM reader's address RTA is refreshed by the override
  (`padded_slice_rm_program_factory.cpp:434`), the Tile reader's rides the `Buffer*` binding form.
  `Known op issues` is empty on both rows. `Uses llama kernels? = yes` on the RM row is informational
  (the borrowed `writer_unary_sharded.cpp`).

- **Device 2.0 (every kernel used):** **GREEN with the fix.** Per kernel:

  | Kernel | Idioms | Verdict |
  |---|---|---|
  | RM reader (`padded_slice_reader_rm_interleaved_start_id.cpp`) | `Noc`, `experimental::CB` (= `CircularBuffer`, `pool/device/kernels/experimental_device_api.hpp:24`), `TensorAccessor` (`:45`), `noc.async_read` (`:203`), `noc.async_read<NocOptions::TXN_ID>` (`:128`, `:159`), `noc.is_read_trid_flushed` (`:152`, `:183`), `UnicastEndpoint` + `experimental::local_addr`, `tt_memmove(noc, …)` (`:171`); `get_local_cb_interface(cb).fifo_page_size` (`:100`) is on the **sanctioned** list | ✓ |
  | RM pad writer (`writer_unary_sharded_padded_rm.cpp`) | `Noc`, `experimental::CB`, `cb.get_read_ptr()` methods (`:21-22`), `UnicastEndpoint` + `local_addr` (`:59-64`) | ✓ |
  | RM plain writer (borrowed `writer_unary_sharded.cpp`) | already `DataflowBuffer` (`:28-33`) | ✓ |
  | Tile reader (`padded_slice_reader_tiled_interleaved_start_id.cpp`) | `Noc`, `experimental::CB`, `TensorAccessor` (`:33`), `noc.async_read` (`:90`); `get_tile_size(cb_id)` (`:31`) sanctioned | ✓ |
  | Tile writer (`writer_unary_sharded_padded_tiled.cpp`) | `Noc`, `experimental::CB` methods (`:56`, `:141-142`, `:191`), `UnicastEndpoint` + `local_addr` (`:106`, `:148-163`, `:177-184`); `constexpr get_tile_size(cb_out_id)` (`:38`) sanctioned | ✓ |
  | Tile compute (borrowed `halo/…/pack_untilize.cpp`) | TRISC kernel: `compute_kernel_hw_startup`, `compute_kernel_lib::untilize*` over cb ids | N/A — not a data-movement kernel |

  No `InterleavedAddrGen` / `ShardedAddrGen` / raw `noc_async_*` / `cb_reserve_back`-family calls remain in
  any kernel the op binds. **Without the fix** the RM reader's non-aligned path is a Device 2.0 RED
  (isolated to that path, but the idioms are raw NoC calls, not one-line CB-index holdovers).

- **Feature compatibility:** every Appendix A entry `N/A`:

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | no `GlobalCircularBuffer` type, no `.global_circular_buffer` field, no `remote_cb` idiom; every CB comes from `make_slice_cb_descriptor` (`slice_cb_descriptor.hpp:14-31`), which sets only `.buffer` |
  | CBDescriptor `address_offset` (non-zero) | N/A | `make_slice_cb_descriptor` never sets `.address_offset`; no `UpdateDynamicCircularBufferAddress` 4-arg form (the override uses the 3-arg form, `padded_slice_rm_program_factory.cpp:89`) |
  | GlobalSemaphore | N/A | the op declares no semaphores at all |

- **CB endpoints (GATE-free):** classified per `(CB, config)`, per node. Every CB has one instance per
  node over the output shard grid.

  **RM factory** (`padded_slice_rm_program_factory.cpp`):

  | CB (index) | Built at | Touchers | Census | Disposition |
  |---|---|---|---|---|
  | `output` (0, borrowed from the output buffer) | `:339` | reader: `reserve_back` / `get_write_ptr` / `push_back` → locked PRODUCER; writer: `writer_unary_sharded` `wait_front`/`pop_front` (locked CONSUMER, `!pad_output_row`) **or** `writer_unary_sharded_padded_rm` `get_read_ptr` + NoC self-reads into it (role-free, `pad_output_row`) | 2 touchers, ≤1 locked each | **1P+1C** in both configs: reader PRODUCER, writer CONSUMER. With the pad writer the FIFO is fill-only (output is resident); do not add a pop |
  | `temp_pad` (1, only when `pad_output_row`) | `:344` | pad writer only: `get_read_ptr`, zero-fills, NoC self-reads | 1 toucher | **self-loop** on the pad writer; **conditional DFB** — exists only under `pad_output_row` (the host already allocates it conditionally, `:341-346`) |
  | `non_aligned` scratch (2, or 1 when there is no `temp_pad`; only when `is_non_aligned`) | `:353` | reader only: `reserve_back` / `get_write_ptr` / NoC reads into it / `push_back` | 1 toucher | **self-loop** on the reader; **conditional DFB** — exists only under `is_non_aligned` (`:349-355`). The index-reuse trick at `:346` ("so that CBs are continuous") evaporates with named DFBs |

  **Tile factory** (`padded_slice_tile_program_factory.cpp`):

  | CB (index) | Built at | Touchers | Census | Disposition |
  |---|---|---|---|---|
  | `input` (0) | `:450` | reader PRODUCER (`:67`, `:117`); compute CONSUMER via `compute_kernel_lib::untilize` (`pack_untilize.cpp:31-46`) | 1P + 1C | **legal 1:1** |
  | `untilized` (1) | `:457` | compute PRODUCER (`untilize` packs into `out_cb_id0` / `out_cb_id1`, **both = CB 1**, `:490-496`); writer CONSUMER (`wait_front` `:141`, `get_read_ptr` `:142`, `pop_front` `:191`) | 1P + 1C | **legal 1:1**; the compute kernel names the same CB twice → one DFB with two kernel-side names (catalog *Pattern: Same-FIFO aliasing*) |
  | `output` (2, borrowed from the output buffer) | `:470` | writer only: NoC self-reads into `cb_out` with `.offset_bytes` (`:148-163`, `:179-184`), no FIFO ops | 1 toucher | **self-loop** on the writer |
  | `padding` (3) | `:478` | writer only: `get_read_ptr` (`:56`), zero-fill, NoC self-reads from it | 1 toucher | **self-loop** on the writer |

  No hidden second writer: every `get_write_ptr` in the tree belongs to the CB's own FIFO producer, and no
  semaphore-gated co-fill exists (the op has no semaphores). No multi-binding flag anywhere.

- **Offset base pointers:** **GREEN with the fix.** Every address RTA resolved:

  | Factory | RTA | Host value | Kernel use | Verdict |
  |---|---|---|---|---|
  | RM | reader RTA 0 | `input_tensor.buffer()->address()` (`padded_slice_rm_program_factory.cpp:226`; refreshed `:434`) — bare base | `TensorAccessor(src_args, src_addr, …)` (`reader:45`); the W-begin + per-core width offset arrive as `src_offset_bytes` (`reader:22`, host `:235`) and `misalignment` (`reader:23`, host `:236`), applied as the accessor's per-read `offset_bytes` (`reader:128-132`, `:203-208`) | **no fold** — the catalogued Type-2 entry is now split out (the "no fold, op in the tables" outcome) |
  | Tile | reader RTA 0 | `Buffer*` (`padded_slice_tile_program_factory.cpp:44`) | `TensorAccessor(src_args, src_addr)` (`tiled reader:33`) + tile index `page_id` (`:90`) | **clean** |

  Type 3 (`address_offset`) is N/A per Appendix A; Type 4 (`narrow`) is not used. **Without the fix**
  the RM row is a Type-2 GATE (ops team + framework), as catalogued.

- **TensorAccessor 3rd argument:** **GREEN — one site, Class 2 (redundant) → drop.**
  `padded_slice_reader_rm_interleaved_start_id.cpp:45` passes `padded_stick_size` = reader RTA 1 =
  `input_page_size = input_shape[-1] * element_size` (`padded_slice_rm_program_factory.cpp:115`, slot 1
  of the common args). The two classifying questions:
  1. *Sharded or interleaved?* Interleaved — `padded_slice.cpp:57` rejects sharded inputs
     (`TT_FATAL(!input_tensor.memory_config().is_sharded(), …)`), so the interleaved specialization
     and its realignment apply.
  2. *Correct or wrong magnitude?* The true logical page: a row-major stick is exactly
     `width × element_size` bytes, so the value equals `buffer->page_size()` and realigns to the
     aligned page the binding would supply.

  It is **not Class 1**, although `2026-07-06_tensor_accessor_3rd_arg_triage.md` lists `padded_slice`
  there: Class 1 needs the page size to vary across shapes that reuse one compiled program, and this op
  has no custom hash — the default reflection hash keys the program cache on the input `TensorSpec`, so
  a cache hit implies the same width and the same page size. The triage row predates the op's
  ProgramDescriptor migration (#57550); trust the scan. The kernel comment at `reader:43-44` ("may be
  stale on program cache hits") describes a hazard the current hash does not admit — noted under *Misc
  anomalies*, not port work. Consistent with the sheet's `TensorParameter relaxation = none`.
  The Tile reader (`:33`) passes no 3rd argument.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding):
  - RM `input` — **Case 1** (`TensorAccessor`, `reader:45`) → `TensorParameter` / `TensorBinding`; the
    address RTA (slot 0) and the `TensorAccessorArgs` CTA tail (`padded_slice_rm_program_factory.cpp:361`,
    `reader:33`) disappear.
  - RM `output` — **clean** (borrowed-memory CB, `padded_slice_rm_program_factory.cpp:339`) →
    `DataflowBufferSpec::borrowed_from = OUTPUT`.
  - Tile `input` — **Case 1** (`Buffer*` delivery `padded_slice_tile_program_factory.cpp:44` →
    `TensorAccessor(src_args, src_addr)`, `tiled reader:33`) → `TensorParameter` / `TensorBinding`; the
    `Buffer*` RTA helper (`:38-50`) and the CTA tail (`:518`, `tiled reader:32`) disappear.
  - Tile `output` — **clean** (borrowed-memory CB, `padded_slice_tile_program_factory.cpp:470`) →
    `borrowed_from = OUTPUT`.
- **TensorParameter relaxation:** `none` (both rows).
- **TensorAccessor 3rd arg:** drop `padded_stick_size` from the accessor at `reader:45` (Class 2); the
  host-side slot 1 of the reader args exists only for it and drops with it.
- **CB endpoints:** RM `output` → 1P+1C (reader PRODUCER, writer CONSUMER, both writer configs);
  RM `temp_pad` → self-loop on the pad writer, conditional on `pad_output_row`; RM `non_aligned` → self-loop
  on the reader, conditional on `is_non_aligned`; Tile `input`, `untilized` → legal 1:1; Tile `output`,
  `padding` → self-loop on the writer.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none — no CB has three touchers or two kernels locked
  to one role. The Tile compute kernel binds `untilized` under two template names (`out_cb_id0`,
  `out_cb_id1`) for the one CB.
- **Cross-op / shared kernels:**
  - `data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp` (RM, `!pad_output_row`):
    **fork exists** — `writer_unary_sharded_metal2.cpp` beside it (vocabulary: `dfb::out` bound CONSUMER,
    `args::num_units`; issue #52228). Rung 1: bind it, adopt its names, do not touch the original.
    Sunset list (other legacy binders on `main`, quasar copies excluded):
    `data_movement/sharded_partial/interleaved_to_sharded_partial/device/interleaved_to_sharded_partial_program_factory.cpp`,
    `data_movement/untilize/device/factories/untilize_multi_core_input_and_output_nd_shard_type_and_shard_spec_identical_program_factory.cpp`.
    **Sunset list, not authorization to convert in place.**
  - `sliding_window/halo/device/kernels/compute/pack_untilize.cpp` (Tile compute): **no fork yet** —
    this port creates `pack_untilize_metal2.cpp` beside it (rung 2) plus the pointer comment in the
    original. Other legacy binder: `sliding_window/halo/device/untilize_with_halo_program_factory.cpp`
    (unported). Name the bindings for the kernel (`dfb::in`, `dfb::out`, `args::total_blocks`-style), not
    for padded_slice. Its donor calls (`compute_kernel_lib::untilize_init` / `untilize` /
    `untilize_uninit`, `ttnn/kernel_lib/untilize_helpers.hpp`) take CB ids as `uint32_t` template
    parameters — `dfb::name`'s constexpr conversion covers that position.
  - **Lent host helpers:** `device/slice_cb_descriptor.hpp` (a `CBDescriptor` builder) and
    `device/padded_slice_utils.hpp` are included by all three `experimental/slice_write` factories. The port
    stops using `slice_cb_descriptor.hpp` but **must leave the file in place** (slice_write still builds
    it); the recipe's "no `CBDescriptor` survives in the op directory" sweep will hit it — record it as
    the lent header it is, not as a leftover.
- **RTA varargs:** all three op DM kernels read `num_dims`-long blocks through `get_arg_addr` and index
  them in a `j < num_dims` loop → varargs:
  - RM reader `:24` (`get_arg_addr(11)`: `num_unpadded_sticks`, `num_padded_sticks`, `id_per_dim`);
  - Tile reader `:21` (`get_arg_addr(6)`: same three blocks);
  - Tile writer `:34-36` (`get_arg_addr(5)`, `5 + num_dims`, `5 + 2*num_dims`: `output_coord`,
    `output_start_in_input`, `output_end`; here `num_dims` is a CTA, still a vararg block).
  Every other RTA is a distinct named field. No CTA varargs.
- **Anything else the porter needs:**
  - **Mutable arg memory.** The RM reader (`:140`, `:212`), the Tile reader (`:106-114`) and the Tile
    writer (`:196-202`) **write back into** their vararg blocks (`id_per_dim[j]++`, `output_coord[…] +=`).
    Metal 2.0 varargs are read-only; the odometer needs writable storage. `data_movement/slice`'s ported
    RM reader seeds a `Scratchpad<uint32_t>` from the vararg block and advances that
    (`slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp:47-53` on `main`, `ScratchpadSpec` in the
    factory). Same shape here, in all three kernels.
  - **Target concept, RM:** `CustomProgramSpecFactoryConcept`. The ported-from override
    (`padded_slice_rm_program_factory.cpp:411-436`) refreshes exactly two things per dispatch: the output
    CB's backing buffer (`:88-89`, via `UpdateDynamicCircularBufferAddress`) and the reader's base-address
    RTA (`:434`). Both become `TensorArgument`s (`OUTPUT`, `INPUT`) in the returned `ProgramRunArgs`;
    nothing else is refreshed, so refresh nothing else. The guards at `:419-423` move with it.
  - **Target concept, Tile:** `ProgramSpecFactoryConcept` — the framework refreshes both bindings.
  - **`experimental::CB` is `CircularBuffer`.** The four op kernels take the wrapper through
    `pool/device/kernels/experimental_device_api.hpp`; rule 1 replaces it with `DataflowBuffer`. The
    header stays included for `experimental::local_addr` / `UnicastEndpoint` (both used by the writers
    and the RM reader) — do not fork or edit it.
  - **Compute hw_config / opt_level (Tile):** Style B — `ComputeConfigDescriptor{.fp32_dest_acc_en = false,
    .dst_full_sync_en = false}` (`padded_slice_tile_program_factory.cpp:506`), math fidelity default; no
    `opt_level` set anywhere (`grep -n opt_level` on both factories: zero hits), so the compute spec resolves
    to **O3** and needs it set explicitly; both DM kernels resolve to O2 (`ReaderConfigDescriptor{}` /
    `WriterConfigDescriptor{}` defaults).
  - **`constexpr` metadata:** Tile writer `:38` `constexpr uint32_t tile_size = get_tile_size(cb_out_id)`
    stays a free-function call on the binding token (`get_tile_size(dfb::out)`); the Tile reader's `:31`
    is a runtime `const` and moves to the DFB getter.
  - **Compute CTA duplication:** `compute_args` slots 1 and 2 are both `cb_untilized_index`
    (`padded_slice_tile_program_factory.cpp:490-496`); the kernel reads them as `out_cb_id0` / `out_cb_id1`
    (`pack_untilize.cpp:16-17`). Two names, one DFB.
  - **Runtime kernel-source selection, RM writer:** `pad_output_row` selects between the op's pad writer
    and the borrowed plain writer (`padded_slice_rm_program_factory.cpp:376-385`); both entry points convert
    with the factory (the plain one by reusing the existing fork).

## Team-only

- **Out-of-directory coupling & donor shape:** roll-up **✓ clean** (no `uint32_t sem_id` / sem-address /
  `TensorAccessorArgs<N>` / `CircularBuffer&` / addr-gen shapes in any called donor function).

  | Op kernel | Donor file | Class | Functions called | Shape |
  |---|---|---|---|---|
  | RM reader, Tile reader | `ttnn/operations/data_movement/common/kernels/common.hpp` | cross-family (`data_movement/common`) | RM reader: `tt::data_movement::common::tt_memmove<…>(Noc, uint32_t, uint32_t, uint32_t)` (`common.hpp:217`). Tile reader: includes it, calls nothing from it | ✓ raw addresses + `Noc` |
  | all four op kernels | `ttnn/operations/pool/device/kernels/experimental_device_api.hpp` | cross-family (`pool`) | `experimental::CB` (type alias of `CircularBuffer`, `:24`), `experimental::local_addr(uint32_t, uint8_t)` (`:37-39`, returns `UnicastEndpoint` src args) | ✓ no resource handles; header is a thin alias/helper layer, not a kernel |
  | Tile writer | `internal/dataflow/dataflow_api_addrgen.h`, `ckernel_defs.h`, `api/tile_constants.h` | `tt_metal/*` | `ckernel::TILE_HEIGHT`, `tt::constants::TILE_HEIGHT`; nothing from the internal addrgen header | ✓ (see anomalies) |
  | Tile compute (borrowed) | `ttnn/kernel_lib/untilize_helpers.hpp`, `api/compute/pack_untilize.h` | kernel-lib / LLK | `compute_kernel_lib::untilize_init<…>`, `untilize<…>`, `untilize_uninit<…>` (cb ids as `uint32_t` NTTPs, `untilize_helpers.hpp:64-72`, `:152-166`); `compute_kernel_hw_startup(cb, cb)` | ✓ `uint32_t cb_id` |

  Borrowed kernel files: `writer_unary_sharded.cpp` (fork present; consumers above) and
  `halo/…/pack_untilize.cpp` (no fork; consumer `untilize_with_halo_program_factory.cpp`). Lent host
  headers: `slice_cb_descriptor.hpp`, `padded_slice_utils.hpp` → `experimental/slice_write` (3 factories).
- **Relaxation candidates:** none — no custom hash to mine.
- **TTNN factory analysis:** single-program SPMD on both factories; no op-owned tensors; no custom hash;
  no `get_dynamic_runtime_args`; no pybound `create_descriptor`; RM `override_runtime_arguments` at
  `padded_slice_rm_program_factory.cpp:411` (refreshes the output CB buffer and the reader base only) →
  `CustomProgramSpecFactoryConcept`; Tile `ProgramSpecFactoryConcept`.
- **Pre-port fix provenance:** `f574e0b3430` on branch `iwrosz/padded-slice-rm-preport-fix` (draft PR
  #62499). Its tests: the existing `test_padded_slice.py` cases (cache-hit for both factories,
  height-sharded pad / unaligned-start / uint8) plus four new block-sharded unaligned-start cases, and the
  nightly `test_slice_for_conv.py` height / block / width-sharded sweeps.

## Misc anomalies  *(team-only, non-gating)*

- `padded_slice_reader_rm_interleaved_start_id.cpp:43-44` — comment says the runtime page size guards
  against a stale `AlignedPageSize` on cache hits; with no custom hash the cache key pins the input spec,
  so the override is inert (hence Class 2 above).
- `writer_unary_sharded_padded_tiled.cpp:8` includes `internal/dataflow/dataflow_api_addrgen.h` (an
  internal header) and uses nothing from it; `:14-19` defines a local `round_down` that duplicates the
  host/`tt` helper.
- `sliding_window/halo/device/kernels/compute/pack_untilize.cpp:7` and `:12` include
  `untilize_helpers.hpp` twice through two paths.
- `padded_slice_tile_program_factory.cpp:273-275` — two `get_tiled_start_offset` calls whose results are
  discarded; `:92`, `:183`, `:271`, `:422` — `[[maybe_unused]]` values computed and never used.
- `padded_slice_rm_program_factory.cpp` — `slice_row_size_bytes` means the per-core sliced row in
  `create_descriptor` (`:295-296`, divided by `num_cores_channels`) but the whole sliced row in the
  runtime-args builder (`:117`); the same name carries two meanings.
- `padded_slice_rm_program_factory.cpp:346` — CB index reuse "so that CBs are continuous": index 1 is
  `temp_pad` when the row is padded and the non-aligned scratch otherwise; harmless today, confusing to
  read, and gone with named DFBs.
- Device op `compute_output_specs` (`padded_slice_device_operation.cpp:83-84`) flattens N·C·H into dim 2
  and leaves dims 0–1 at 1; `validate_on_program_cache_miss` requires rank 4 (`:36`) but
  `padded_slice.cpp:34` already enforces it one level up.

## Questions for the user  *(none)*

## Recipe notes

1. **The workspace recipe cannot be followed literally.** `human/READ_ME_FIRST.md` says "create a branch
   off `akertesz/op-porting-recipe`, merge main into your branch". The recipe branch is an orphan history
   (`git merge-base` with `main` is empty; `git merge` refuses with "unrelated histories"), so the merge
   is impossible without `--allow-unrelated-histories` and thousands of conflicts. This audit branched
   off `main` and imported `docs/…/metal_2.0/` verbatim instead. Either the doc should describe that, or
   the recipe branch should be rebased onto a `main` ancestor.
2. **Audit against an unmerged pre-port fix.** The recipe assumes the ops-team fix has landed before the
   audit runs and says a RED op "is re-audited once the blockers clear". Here the fix and the audit were
   produced in one effort, so the audit ran on `main` + the fix and records the dependency. A sentence in
   the audit's *Output* section on how to record a conditional verdict (fix commit, PR, what fails without
   it) would make this shape reproducible.
3. **Mutable runtime-arg memory is not covered.** Kernels that `get_arg_addr` a block and then write into it
   (an odometer) are common in the slice family; the recipe's vararg rule only says varargs are read-only.
   The `slice` port's `Scratchpad` answer should be a catalog pattern so porters do not improvise.
4. **TensorAccessor 3rd-arg triage vs. cache key.** The triage doc classifies `padded_slice` as Class 1 on
   the assumption that shapes reuse one compiled program. For an op on the default reflection hash that
   cannot happen, which makes the site Class 2. The classification questions should ask about the op's
   cache key explicitly.
5. **`get_local_cb_interface(cb).fifo_page_size` is sanctioned but not a `DataflowBuffer` getter the
   whitelist names for the page size** — the porter will look for one. Worth a row in
   `cb_dfb_api_whitelist.md` §B if a getter exists.
