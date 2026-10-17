# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/data_movement/repeat/codegen`

- **`RepeatCodegenDeviceOperation`** (`repeat_codegen_device_operation.hpp/.cpp`) — the only DeviceOperation in this directory. `program_factory_t = std::variant<RepeatCodegenProgramFactory>`.
  - `RepeatCodegenProgramFactory` (`repeat_codegen_program_factory.cpp`) — one factory, one `create_descriptor`, **three internal code-path branches** selected by input layout / repeated dim (`repeat_codegen_program_factory.cpp:91-92`):
    - **TILE** (`!is_row_major`, lines 99-165) — binds two *borrowed* in-family shared kernels: `data_movement/common/kernels/codegen/reader_tile_interleaved_unified.cpp` (sequencer `seq_id = 1 == SEQ_REPEAT`) + `data_movement/common/kernels/codegen/writer_interleaved.cpp`.
    - **RM last-dim** (`is_last_dim_rm`, `rep_dim == 3`, lines 167-220) — own kernels `kernels/reader_repeat_last_dim_rm.cpp` + `kernels/writer_repeat_rm.cpp`.
    - **RM higher-dim** (fallthrough, lines 222-274) — own kernels `kernels/reader_repeat_higherdim_rm.cpp` + `kernels/writer_repeat_rm.cpp`.
  - Support / routing helpers: `repeat_codegen_supported.cpp` (`supported_by_codegen`, `is_demoted`) — host-only, no kernels.
- Layout note: this op has **no `device/` subdirectory**; the device-op, factory and kernels live at the op root (`codegen/`). The sibling `../device/` tree is the *native* `RepeatDeviceOperation` (already on `create_program_artifacts`, i.e. Metal 2.0) and is **out of scope** for this audit.
- Routing context: `ttnn::repeat` (`repeat/repeat.cpp:601-611`) tries this op first for every unsharded rank-2..4 interleaved same-buffer-type case (per-dim decomposition in `repeat_via_codegen`, `repeat.cpp:284-299`), falling back to native otherwise. `repeat_force_codegen` / `repeat_force_native` (`repeat_nanobind.cpp:79`) are verification-only entries.
- Unreferenced kernel files in the op directory: **none** (all three files under `kernels/` are bound by the factory).
- Kernel census (every `kernel_source`): 5 files — 3 own (`kernels/*.cpp`), 2 borrowed (`common/kernels/codegen/*.cpp`), plus the borrowed reader's `#include "sequencers.h"` (same shared dir).

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `8c5559389da 2026-09-22 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`

**Readiness sheet:** fetched 2026-09-23 via the Drive connector into `metal_2.0/analyses/ttnn_op_porting_readiness.csv` (ephemeral, not committed); row `data_movement/repeat/codegen` (CSV line 69).

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/repeat/codegen` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `RepeatCodegenDeviceOperation` → `RepeatCodegenProgramFactory` (3 branches: TILE / RM last-dim / RM higher-dim) |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all 5 kernels (3 own + 2 borrowed) are structurally Device 2.0 (`Noc`, `CircularBuffer`, `TensorAccessor`, `UnicastEndpoint`, `CoreLocalMem`); no CB-index holdovers, no legacy addr-gens |
| *Prereqs* — Cross-op escapes | Ok — one in-family function-call escape (`sequencers.h`, scalar-only signatures ✓); two borrowed kernel files (coordination cost, see Team-only) |
| *Feature Support* — overall | **GREEN** — no Appendix A entry fires |
| *Feature Support* — Variadic-CTA | Ok (none) |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** (`yes` verbatim; primary cross-check clean) |
| *TTNN Readiness* — Concept (current) | `descriptor` (`create_descriptor` → `ProgramDescriptor`, `repeat_codegen_program_factory.hpp:39`) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Custom hash | No (sheet `no`; no `compute_program_hash` / `attribute_values` / `to_hash` in the device-op) |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No (sheet `no`; grep clean) |
| *TTNN Readiness* — `override_runtime_arguments` | No (sheet `no`; grep clean) |
| *TTNN Readiness* — Pybind `create_descriptor` | No (sheet `no`; `repeat_nanobind.cpp` binds only `repeat`, `repeat_force_native`, `repeat_force_codegen`) |
| *TTNN Readiness* — Op-owned tensors | No (sheet cell blank; `descriptor` concept cannot carry them) |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept` |
| *Port work* — Offset base pointer | none — every address arg is a bare `Buffer*` (clean base) |
| *Port work* — Tensor bindings (per binding) | `src` (input) **Case 1** in all 3 branches · `dst` (output) **Case 1** in all 3 branches |
| *TTNN Readiness* — TensorParameter relaxation | `none` (clears) |
| *Port work* — TensorAccessor 3rd arg | **drop (Class 2)** — 2 live sites + 1 dead-branch site, all in the *borrowed* TILE-branch kernels; the op's own RM kernels pass no 3rd arg |
| *Port work* — CB endpoints | **legal** — one CB (`buffer_index 0`) per branch, 1 locked producer (reader) + 1 locked consumer (writer) on every node in every branch |

**CB endpoints** are dispositions, not gates (see `audit/metal2_audit.md` → CB endpoints): here every `(CB, config)` is a plain 1:1 FIFO — no self-loop, no 1P+1C assignment, no multi-binding flag, no dead CB.

## Result

**GREEN → brief issued.** Every gate clears: Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ (`Is able to port? == yes`, cross-check clean) · Offset base pointers ✓ · TensorAccessor 3rd arg ✓ (Class 2 only). The port targets `ProgramSpecFactoryConcept`. The one non-trivial piece of port work is that the **TILE branch binds two kernels this op does not own** (`common/kernels/codegen/reader_tile_interleaved_unified.cpp`, `writer_interleaved.cpp`), shared with `repeat_interleave/codegen`; no `_metal2` fork exists yet beside either, so this port creates the first fork of each (rung 2 of `port_patterns.md` → Caution: Porting a shared kernel).

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN** — the sheet's `Is able to port?` == `yes` for the op's single row (`data_movement/repeat/codegen`, `RepeatCodegenDeviceOperation`, `RepeatCodegenProgramFactory`). Cross-check of the primary columns against the code, all consistent:
  - `Concept` = `descriptor` ↔ `static ProgramDescriptor create_descriptor(...)` at `repeat_codegen_program_factory.hpp:39-42`, body returns `ProgramDescriptor` (`repeat_codegen_program_factory.cpp:80-274`). ✓
  - `Custom hash` = `no` / `Backdoor custom hash` = `no` ↔ no `compute_program_hash`, `attribute_values`, or `to_hash` anywhere in `repeat_codegen_device_operation.{hpp,cpp}`. ✓
  - `Runtime-args update (get_dynamic_runtime_args)` = `no` ↔ no such hook on the device-op. ✓
  - `Override runtime args method?` = `no` ↔ no `override_runtime_arguments` on the factory. ✓
  - `Pybind descriptor` = `no` ↔ `repeat_nanobind.cpp` has no `create_descriptor` / device-op class binding (only free functions `repeat`, `repeat_force_native`, `repeat_force_codegen` at lines ~60-85). ✓
  - `Smuggled pointer` = `no` ↔ the factory passes `Buffer*` objects (`src_buffer`, `dst_buffer`) into `emplace_runtime_args`, never `->address()` (lines 150-158, 212-213, 266-267) — the annotated `Buffer*`-binding form, not a smuggled raw address. ✓
  - `TensorParameter relaxation` = `none`; `Known op issues` = blank; `Op-owned tensors?` = blank; `Secretly SPMD Workload?` = blank (N/A for `descriptor`). ✓
  - Cross-column invariants hold (`get_dynamic_runtime_args == no`; no op-owned tensors on a `descriptor` row).
  - **Factory-set match:** exactly one sheet row ↔ exactly one factory in `program_factory_t` (`repeat_codegen_device_operation.hpp:20`). No phantom, no missing row. ✓
- **Device 2.0 (every kernel used):** **GREEN**. Per kernel:

  | Kernel | Owner | Device 2.0 idioms in use | Free functions present | Verdict |
  |---|---|---|---|---|
  | `codegen/kernels/reader_repeat_last_dim_rm.cpp` | this op | `Noc`, `CircularBuffer cb_in(cb_id)`, `TensorAccessor`, `noc.async_read(accessor, cb, …)`, `UnicastEndpoint self_ep` + `noc.async_read(self_ep, cb_in, …)` for the L1→L1 stick replication (lines 78-92), `CoreLocalMem<volatile T>` for the sub-16B RISC copies (lines 100-140) | `get_arg_val`, `get_compile_time_arg_val` (arg plumbing, not DM idioms); `cb_in.get_write_ptr()` **method** (line 60, not the free function); `my_x[noc.get_noc_id()]` / `my_y[…]` firmware globals for the self-endpoint coordinates (lines 79-80) — see Recipe notes | ✓ compliant |
  | `codegen/kernels/reader_repeat_higherdim_rm.cpp` | this op | `Noc`, `CircularBuffer`, `TensorAccessor`, `noc.async_read(s, cb_in, …)` | `get_arg_val`, `get_compile_time_arg_val` | ✓ compliant |
  | `codegen/kernels/writer_repeat_rm.cpp` | this op | `Noc`, `CircularBuffer`, `TensorAccessor`, `noc.async_write(cb, d, …)`, `noc.async_writes_flushed()`, `noc.async_write_barrier()` | `get_arg_val`, `get_compile_time_arg_val` | ✓ compliant |
  | `common/kernels/codegen/reader_tile_interleaved_unified.cpp` (borrowed) | `data_movement/common` shared codegen pool | `Noc`, `CircularBuffer`, `TensorAccessor`, `noc.async_read(accessor, cb, …)` (lines 123-144); PAD branch (dead under `SEQ_REPEAT`) also `UnicastEndpoint` + `CoreLocalMem` | `get_named_compile_time_arg_val`, `get_arg_addr(0)` + `reinterpret_cast<const ArgsRepeat*>` struct read of RTAs (lines 160, 179), `get_local_cb_interface(cb_id).fifo_page_size << cb_addr_shift` (line 164 — **sanctioned** per the Green bullet), `my_x`/`my_y` in the dead PAD branch (lines 240-241) | ✓ compliant |
  | `common/kernels/codegen/writer_interleaved.cpp` (borrowed) | `data_movement/common` shared codegen pool | `Noc`, `CircularBuffer`, `TensorAccessor`, `noc.async_write(cb, d, …)` | `get_arg_val`, `get_compile_time_arg_val`, `get_local_cb_interface(cb_out).fifo_page_size << cb_addr_shift` (line 39 — **sanctioned**) | ✓ compliant |
  | `common/kernels/codegen/sequencers.h` (included by the borrowed reader) | same pool | pure index arithmetic; includes only `api/dataflow/dataflow_api.h` | none | ✓ (no DM idioms at all) |

  No `noc_async_read`/`noc_async_write` free calls, no `get_noc_addr`, no `InterleavedAddrGen*`/`ShardedAddrGen`, no raw semaphore addresses, no `get_read_ptr(cb_id)`/`get_write_ptr(cb_id)` free-function holdovers in any of the six files. Nothing to route to the Device 2.0 team.

  | File | Line | Call | Wrapper in scope |
  |---|---|---|---|
  | — | — | *(no violations)* | — |

- **Feature compatibility:** scanned host (`repeat_codegen_*.cpp/.hpp`), all 5 kernels and `sequencers.h`. No entry fires.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | no `GlobalCircularBuffer` type, no `.global_circular_buffer` field on either `CBDescriptor` (lines 104-112, 173-181, 226-234), no `remote_index`/`remote_cb_*` |
  | CBDescriptor `address_offset` (non-zero) | N/A | no `.address_offset`, no `cb_descriptor_from_sharded_tensor`, no `UpdateDynamicCircularBufferAddress`; all three CBs are plain L1-allocated (no `buffer` field) |
  | GlobalSemaphore | N/A | no semaphores of any kind in the op |

- **CB endpoints (GATE-free):** **legal in every branch.** Each branch allocates exactly one CB, `buffer_index = 0`, over `split.all_cores`, depth `kRepeatCbDepth = 8` pages (`repeat_codegen_program_factory.hpp:20`). Per `(CB, config)` census on every node:
  - `(CB0, TILE)`: `reader_tile_interleaved_unified.cpp` FIFO-produces (`cb.reserve_back`/`cb.push_back`, lines 129/142 in `read_pages`) → locked producer; `writer_interleaved.cpp` FIFO-consumes (`cb.wait_front`/`cb.pop_front`, lines 61-98) → locked consumer. **2 touchers, 1P + 1C → legal.**
  - `(CB0, RM last-dim)`: `reader_repeat_last_dim_rm.cpp` FIFO-produces (`cb_in.reserve_back` line 57, `cb_in.push_back` line 145) and *also* peeks `cb_in.get_write_ptr()` (line 60) and uses `cb_in` as the destination of its own L1→L1 `noc.async_read(self_ep, cb_in, …)` (lines 84-89) — all within the **same kernel**, so still one toucher; `writer_repeat_rm.cpp` FIFO-consumes (lines 48-82). **2 touchers, 1P + 1C → legal.**
  - `(CB0, RM higher-dim)`: `reader_repeat_higherdim_rm.cpp` FIFO-produces (lines 46/70); `writer_repeat_rm.cpp` FIFO-consumes. **2 touchers, 1P + 1C → legal.**
  - Hidden-second-writer hunt: no kernel other than the producer touches the CB by raw pointer; no semaphores exist to coordinate a co-fill. Multiple-readers face: no borrowed-memory CB. Dual-instance work-split: reader and writer are different sources; each branch pushes exactly one reader + one writer `KernelDescriptor`. No dead CB (the index is reached by `cb_id` named CTA on TILE — `{"cb_id", 0}` line 125 — and by positional CTA on both RM branches — `reader_ct_args.push_back(0)` lines 185/238, `writer_ct_args = {0, …}` lines 197/252).
- **Offset base pointers:** **GREEN.** Every address argument is a bare `Buffer*` pushed into `emplace_runtime_args` — `src_buffer` at `repeat_codegen_program_factory.cpp:152, 212, 266`, `dst_buffer` at `:158, 213, 267` — with no host arithmetic folded in; `start` (per-core page offset) travels as a *separate* page-index scalar and is applied kernel-side as `page_id` through the `TensorAccessor` (`reader_repeat_last_dim_rm.cpp:65`, `reader_repeat_higherdim_rm.cpp:65`, `writer_repeat_rm.cpp:51`, `reader_tile_interleaved_unified.cpp:133-138`, `writer_interleaved.cpp:64-65`). No `->address() + …` anywhere in the op. Not in the `2026-07-19_offset_base_pointers.md` tables (grep `repeat` → no hit) → *no fold, op not in the tables → clean.* No Type 3 (`address_offset`) and no Type 4 (`narrow`) use.
- **TensorAccessor 3rd argument:** **GREEN — sites found and classified Class 2 (redundant); drop.** The op is not in the `2026-07-06_tensor_accessor_3rd_arg_triage.md` table (grep `repeat` → no hit), so each site was classified from the two questions. Neither of the op's own reader/writer kernels passes a 3rd arg (`reader_repeat_last_dim_rm.cpp:47`, `reader_repeat_higherdim_rm.cpp:33`, `writer_repeat_rm.cpp:36` — all 2-arg). The sites are all in the **borrowed TILE-branch kernels**:
  1. `common/kernels/codegen/reader_tile_interleaved_unified.cpp:169` — `TensorAccessor(src_args, base->src_addr, source_page_size)`. *Sharded or interleaved?* Interleaved: `supported_by_codegen` rejects any sharded input (`repeat_codegen_supported.cpp:120-122, 134-136`), and the factory's CB/accessor setup is interleaved-only. *Magnitude?* `source_page_size = src_page_pitch != 0 ? src_page_pitch : src_args.get_aligned_page_size()` (lines 161-163); this op passes `{"src_page_pitch", 0}` (`repeat_codegen_program_factory.cpp:131`), so the value **is** `aligned_page_size` → exactly the Class 2 definition. (The co-consumer `repeat_interleave/codegen` also passes `0` — `repeat_interleave_codegen_program_factory.cpp:119` — so the fork's drop is Class 2 for both current consumers.)
  2. `common/kernels/codegen/reader_tile_interleaved_unified.cpp:292` — `TensorAccessor(src_args, a->src_addr_1, source_page_size)` in the `SEQ_CONCAT` branch. **Dead code under `SEQ_REPEAT`** (`if constexpr`), same `source_page_size` expression → Class 2 regardless; the porter drops it identically when converting the fork.
  3. `common/kernels/codegen/writer_interleaved.cpp:28` — `TensorAccessor(dst_args, dst_addr, destination_page_size)` where `destination_page_size = dst_args.get_aligned_page_size()` (line 27). Interleaved output (the op's output is `output_mem_config`, interleaved by construction of the routing gate, `repeat.cpp:601-604`). Value == `aligned_page_size` → **Class 2**.

  No Class 1 (nothing varies across cache-reused shapes: the page pitch is the tile size, fixed by dtype), no Class 3/4 (no wrong-magnitude value), no Special. Nothing routes to the ops team.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding, identical in all three branches):
  - `src` (input tensor, `tensor_args.input`) — **Case 1** (via `TensorAccessor`). Legacy delivery is the `Buffer*`-binding form (`src_buffer` in `emplace_runtime_args`, lines 152/212/266 → framework-patched `BufferBinding`, correct-on-cache-hit today); the kernel feeds the base into `TensorAccessor(src_args, src_addr)` and reads only through `noc.async_read(accessor, cb, …, {.page_id})`. Port: `TensorParameter` + `TensorBinding`; kernel builds `TensorAccessor(tensor::src)`; the positional `TensorAccessorArgs<N>()` CTAs and the address RTA disappear.
  - `dst` (output tensor) — **Case 1**. Same shape: `dst_buffer` `Buffer*` at lines 158/213/267; kernel `TensorAccessor(dst_args, dst_addr)` → `noc.async_write(cb, d, …, {.page_id})`.
  - Op-level roll-up: **⚠ port work** (two Case-1 bindings; no Case 2; no borrowed-DFB binding).
- **TensorParameter relaxation:** `none`.
- **TensorAccessor 3rd arg:** drop the redundant page-size arg at `reader_tile_interleaved_unified.cpp:169` (and the dead-branch twin at `:292`) and `writer_interleaved.cpp:28` — in the **`_metal2` forks** of those files, not in the legacy originals. Class 2 → no `dynamic_tensor_shape`.
- **CB endpoints:** all legal — `(CB0, TILE)`, `(CB0, RM last-dim)`, `(CB0, RM higher-dim)` each bind reader PRODUCER + writer CONSUMER. No self-loop, no 1P+1C reassignment, no multi-binding flag, no dead-CB drop, no conditional DFB.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none.
- **Cross-op / shared kernels:** the TILE branch binds two kernels owned by the `data_movement/common/kernels/codegen/` shared pool — `reader_tile_interleaved_unified.cpp` and `writer_interleaved.cpp` — both also bound by `repeat_interleave/codegen/repeat_interleave_codegen_program_factory.cpp:38-40, 113, 124, 188`. **No `_metal2` fork exists beside either** (`ls` of the directory: `reader_tile_interleaved_unified.cpp`, `sequencers.h`, `writer_interleaved.cpp` only) → this port creates the first fork of each (rung 2). Consumer list = **sunset list, not authorization to convert in place.** `untilize/codegen/kernels/` holds a *same-named private copy* of the reader + `sequencers.h` at a different path — not a consumer of this file.
- **RTA varargs:** none for this op. The borrowed reader reads its RTAs as a **struct overlay** — `reinterpret_cast<const ArgsRepeat*>(get_arg_addr(0))` (`reader_tile_interleaved_unified.cpp:160, 179`) — six fixed fields under `SEQ_REPEAT` (`src_addr, num_pages, start_id, num_repeats, lower_pages, rep_dim_pages`, struct at lines 44-54). Fixed field set → **named args**, not varargs; but the conversion is not a 1:1 `get_arg_val` swap, so the fork must replace the struct overlay with per-field named reads. The same file's `SEQ_SLICE`/`SEQ_PERMUTE` branches read genuine variable-length tails (lines 197-199, 210-211) — those *would* be varargs, but no current consumer of this file binds them.
- **Anything else the porter needs:**
  - **Three code-path branches inside one factory** (`repeat_codegen_program_factory.cpp:99/167/222`): the ported factory keeps the branching and emits a different `KernelSpec` pair per branch; only the TILE branch touches the shared-pool forks.
  - **Named + positional CTA mix on the TILE reader** (`repeat_codegen_program_factory.cpp:114-132`): `TensorAccessorArgs` positional at index 0 plus named CTAs `seq_id`, `cb_id`, `batch`, `src_page_pitch`. Under Metal 2.0 the positional accessor args dissolve into the `tensor::src` binding; `cb_id` becomes a `dfb::` binding; `seq_id`/`batch` stay named CTAs. `src_page_pitch` is read unconditionally (`reader_tile_interleaved_unified.cpp:161`) and is what makes the 3rd-arg site Class 2 — once the 3rd arg is dropped in the fork, `src_page_pitch` has no remaining consumer under `SEQ_REPEAT`; leave it in place unless the whole fork drops the override (porter's call, documented in the port report).
  - **`get_local_cb_interface(cb_id).fifo_page_size << cb_addr_shift`** (`reader_tile_interleaved_unified.cpp:164`, `writer_interleaved.cpp:39`) — sanctioned for the Device 2.0 gate; the DFB equivalent is the entry-size getter (`cb_dfb_api_whitelist.md`, section B). Confirm the DFB getter's byte units before swapping rather than swapping blind.
  - **Firmware coordinate globals `my_x[noc.get_noc_id()]` / `my_y[…]`** (`reader_repeat_last_dim_rm.cpp:79-80`; dead PAD branch of the borrowed reader `:240-241`) feed a `UnicastEndpoint` self-address for the L1→L1 stick replication. Not a Device 2.0 violation (the `Noc` class itself reads the same globals, `tt_metal/hw/inc/api/dataflow/noc.h:160`, and exposes no accessor). The port leaves them alone.
  - **All RM `TensorAccessorArgs<N>()` offsets are positional** (`<3>`, `<2>`, `<3>`) with the trailing CTAs read at `next_compile_time_args_offset() + k` — they must become named CTAs (`cb_id`→`dfb::`, `NUM_REPEATS`, `LOWER_PAGES`, `REP_DIM_PAGES`, `BATCH`, `stick_size`, `in_read_size`/`xfer_size`, `out_l1_stride`/`l1_stride`).
  - **`constexpr` CTA reads drive `if constexpr` specialisations** (`reader_repeat_last_dim_rm.cpp:73-74, 100-132`; `reader_repeat_higherdim_rm.cpp:52-58`; both writers' `if constexpr (BATCH > 1)`) — the named CTA form must stay `constexpr`-evaluable.
  - **Tests for the baseline:** `tests/ttnn/unit_tests/operations/data_movement/test_repeat.py` and `tests/ttnn/nightly/unit_tests/operations/data_movement/test_repeat_codegen_routing.py` (uses `ttnn._ttnn.operations.data_movement.repeat_force_codegen` to pin the codegen leg). Run the baseline before the first kernel edit (kernels are JIT'd from the working tree).

## Team-only

- **Out-of-directory coupling & donor shape:**
  - **Op-level roll-up: ✓ clean** (function-call escapes). Issues: none blocking; the only coupling cost is the two borrowed kernel files (below).
  - **Summary table** (one row per (op kernel, donor file)):

    | Op kernel | Donor file | Donor class | Functions called | Shape | Status |
    |---|---|---|---|---|---|
    | `reader_repeat_last_dim_rm.cpp` | `api/dataflow/*.h`, `api/core_local_mem.h` | 1 — `tt_metal/*` | framework | — | ✓ |
    | `reader_repeat_higherdim_rm.cpp` | `api/dataflow/*.h` | 1 | framework | — | ✓ |
    | `writer_repeat_rm.cpp` | `api/dataflow/*.h` | 1 | framework | — | ✓ |
    | `common/kernels/codegen/reader_tile_interleaved_unified.cpp` (borrowed) | `common/kernels/codegen/sequencers.h` | 5 — in-family shared | `seq_repeat_init(uint32_t, uint32_t, uint32_t, uint32_t)` (`sequencers.h:133`), `seq_repeat_next(SeqRepeatState&)` (`:139`); constants `SEQ_*` | scalars / plain state struct — no resource handle in any signature | ✓ excellent |
    | `common/kernels/codegen/reader_tile_interleaved_unified.cpp` (borrowed) | `api/dataflow/*.h`, `api/core_local_mem.h`, `api/tensor/noc_traits.h` | 1 | framework | — | ✓ |
    | `common/kernels/codegen/writer_interleaved.cpp` (borrowed) | `api/dataflow/*.h`, `api/tensor/noc_traits.h` | 1 | framework | — | ✓ |

  - **Per-call detail:** omitted — all rolls are ✓.
  - **Borrowed kernel files (file-path instantiation):**

    | Kernel file | Owning pool | Other binders | `_metal2` fork beside it? |
    |---|---|---|---|
    | `ttnn/cpp/ttnn/operations/data_movement/common/kernels/codegen/reader_tile_interleaved_unified.cpp` | `data_movement/common` codegen shared pool | `repeat_interleave/codegen/repeat_interleave_codegen_program_factory.cpp` (`kTileReaderSrc`, line 38 → 113), with `seq_id = SEQ_REPEAT_INTERLEAVE` | **No** |
    | `ttnn/cpp/ttnn/operations/data_movement/common/kernels/codegen/writer_interleaved.cpp` | same | `repeat_interleave/codegen/repeat_interleave_codegen_program_factory.cpp` (`kWriterSrc`, line 40 → 124 and 188 — both its TILE and RM branches) | **No** |

    The three own kernels under `codegen/kernels/` are bound by this factory only (tree-wide grep → no other binder) — not lent.
    Note for planning: the shared reader is a *multi-sequencer* kernel (10 `SEQ_*` ids in `sequencers.h:28-37`) of which only `SEQ_REPEAT` (this op) and `SEQ_REPEAT_INTERLEAVE` (`repeat_interleave/codegen`) have binders in the tree today. The fork the first port creates fixes the binding vocabulary for both.
- **Relaxation candidates:** none — no custom hash to mine.
- **TTNN factory analysis:** `descriptor` concept (`repeat_codegen_program_factory.hpp:39`); no op-owned tensors; no MeshWorkload; no pybound `create_descriptor` (nanobind exposes `repeat_force_codegen` / `repeat_force_native` free functions only — verification hooks, not descriptor internals); no custom hash; no `get_dynamic_runtime_args`; no `override_runtime_arguments`; target `ProgramSpecFactoryConcept`. The device-op's `program_factory_t` is a single-alternative `std::variant` (`repeat_codegen_device_operation.hpp:20`) — the port keeps the variant form.
- **Sheet observation (sibling rows, not this op's gate):** the readiness CSV rows for the *native* `data_movement/repeat` (`RepeatProgramFactoryHigherDim`, `RepeatProgramFactoryLastDim`, CSV lines 67-68) read `Concept = descriptor`, but the code in `../device/repeat_program_factory_{higher_dim,last_dim}.hpp:14` is already `create_program_artifacts` (Metal 2.0). Those rows belong to a different `Op` key and do not enter this op's factory-set match, so this is **not** a spreadsheet-broken gate here — but the sheet owner may want to flip them to `MetalV2`.

## Misc anomalies  *(team-only, non-gating)*

- `reader_repeat_higherdim_rm.cpp:7` — header comment says it "mirrors the shared `reader_repeat_higherdim_rm.cpp`", naming itself; there is no shared file of that name (the shared reader is `reader_tile_interleaved_unified.cpp`). Comment-only.
- `repeat_codegen_program_factory.cpp:135` — TILE writer CTA `page_size` is `dst_buffer->aligned_page_size()`, which the borrowed writer then re-clamps against `dst_args.get_aligned_page_size()` and the CB page (`writer_interleaved.cpp:43-45`); harmless triple-authority, all equal on this path.
- `repeat_codegen_program_factory.hpp:29` — `RepeatCodegenParams::stick_size` is `0` on the TILE branch and still participates in the default attribute hash (documented in the struct comment; no behavioural effect).
- `repeat_codegen_supported.cpp:159-167` — `is_demoted` unconditionally returns `false`; kept as a routing extension point per its comment.

## Questions for the user

*(none)*

## Recipe notes

1. **Firmware coordinate globals in Device 2.0 kernels.** `reader_repeat_last_dim_rm.cpp:79-80` reads `my_x[noc.get_noc_id()]` / `my_y[…]` to build a `UnicastEndpoint` address for a self L1→L1 copy. The Device 2.0 gate's Green/Red bullets enumerate CB-index free functions and legacy addr-gens but say nothing about these firmware globals; the `Noc` class has no public coordinate accessor and uses the same globals internally (`noc.h:160`), so I treated them as non-violations. Suggest the gate name them explicitly (sanctioned or not) so the next auditor doesn't have to make the call.
2. **Struct-overlay RTA reads.** `reader_tile_interleaved_unified.cpp:160/179` reads RTAs via `reinterpret_cast<const ArgsRepeat*>(get_arg_addr(0))` rather than per-index `get_arg_val`. Neither the Device 2.0 gate nor the RTA-varargs subject anticipates this shape. I classified it as a fixed field set → named args (non-signal), and surfaced it as a heads-up because the named-arg conversion is a rewrite of the read pattern, not a 1:1 swap. The recipe might add "struct overlay over `get_arg_addr`" to the RTA-varargs *non-signal* list with that caveat.
3. **Op-directory resolution heuristic.** The recipe says to confirm a resolved directory "has a `device/` subdirectory containing a `*_device_operation.*`". This op (and the other `*/codegen` ops in the sheet, e.g. `untilize/codegen`) keeps device-op, factory and kernels at the op root with no `device/`; the user-specified path made the resolution unambiguous, but the heuristic would have flagged a false "not an op". Suggest allowing the flat layout.
4. **TensorAccessor 3rd-arg sites in borrowed kernels.** The subject's prose says "every accessor *in the op*"; all of this op's sites are in borrowed shared-pool kernels. I applied the audit-wide "follow kernel references, not directory boundaries" rule and classified them; a one-line note in the 3rd-arg subject confirming borrowed kernels are in scope would remove the ambiguity.
5. **Sibling-op sheet rows encountered incidentally.** While grepping the CSV for this op I saw stale rows for the native `data_movement/repeat` (see Team-only). The recipe defines "spreadsheet-broken" only for *this op's* rows; it does not say whether to report drift noticed on a neighbouring op's rows. I recorded it as a team-only observation rather than a gate.
