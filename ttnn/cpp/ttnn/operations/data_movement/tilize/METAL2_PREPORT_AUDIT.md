# Metal 2.0 Audit Findings — `data_movement/tilize` (`TilizeMultiCoreBlockProgramFactory`)

This is a **re-audit of a single factory**. The whole-op audit (2026-08-27) cleared five of the six
tilize factories — `{Default, SingleCore, Sharded, ShardedRetile, Retile}` were ported to
`CustomProgramSpecFactoryConcept` and merged (PR #54805). The lone RED was
`TilizeMultiCoreBlockProgramFactory`, blocked on the readiness sheet's `Known op issues =
"Per-node CB size"` — the block factories sized a CB per-node, which a named DFB cannot express.
[GitHub #51305](https://github.com/tenstorrent/tt-metal/issues/51305) tracked that blocker; its
**Option 1** (split the DM kernels into separate kernels, each with its own appropriately-sized DFBs)
landed for tilize in **PR #54140** ("Per-width buffer sets for the tilize/tilize_with_val_padding
block factories") plus bugfix #55810. This audit assesses whether the refactored Block factory is now
portable.

- **`TilizeDeviceOperation`** (`device/tilize_device_operation.{hpp,cpp}`) — one device operation, six program factories:
  - **`TilizeMultiCoreBlockProgramFactory`** (`device/tilize_multi_core_block_program_factory.cpp`) — **the factory in scope**; the only one still on the `descriptor` (`ProgramDescriptor`) concept.
  - `TilizeMultiCoreDefaultProgramFactory`, `TilizeSingleCoreProgramFactory`, `TilizeMultiCoreShardedProgramFactory`, `TilizeMultiCoreShardedRetileProgramFactory`, `TilizeMultiCoreRetileProgramFactory` — **already ported** (`create_program_artifacts` → `ProgramArtifacts`, `override_runtime_arguments` → `ProgramRunArgs`); out of scope here except as context.

**Kernels the Block factory exercises** (all referenced via `KernelDescriptor::kernel_source`):
- reader `data_movement/tilize_with_val_padding/device/kernels/dataflow/reader_unary_pad_multicore_both_dims.cpp` (borrowed)
- writer `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id_wh.cpp` (borrowed)
- compute `data_movement/tilize/device/kernels/compute/tilize_wh.cpp` (op-owned, also bound by `tilize_with_val_padding` block)

Unreferenced files in the compute dir: `kernels/compute/tilize.cpp` and `kernels/compute/retile.cpp` are bound by *other* tilize factories, not this one — out of scope for this factory's audit.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `bd9e9f36292 2026-09-23 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/tilize` |
| **Overall** | **GREEN** (factory-scoped: `TilizeMultiCoreBlockProgramFactory`) |
| **DOps / Factories** | `TilizeDeviceOperation` → `TilizeMultiCoreBlockProgramFactory` (5 sibling factories already ported) |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — reader/writer/compute all on the `Noc` + `DataflowBuffer` + `TensorAccessor` API; writer's only free-function CB call is the sanctioned `get_tile_size(cb_id_out)` |
| *Prereqs* — Cross-op escapes | Ok — 3 shared kernels, all already DFB; function-call escapes all ✓ / kernel-lib |
| *Feature Support* — overall | **GREEN** (all N/A) |
| *Feature Support* — GlobalCircularBuffer / `address_offset` / GlobalSemaphore | N/A / N/A / N/A |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** — `Known op issues` now **empty** (was `Per-node CB size`) |
| *TTNN Readiness* — Concept (current) | `descriptor` |
| *TTNN Readiness* — Secretly SPMD | N/A (`descriptor`, not `WorkloadDescriptor`) |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No |
| *TTNN Readiness* — `override_runtime_arguments` | Yes (not a gate; selects `CustomProgramSpecFactoryConcept`): `tilize_multi_core_block_program_factory.cpp:380` |
| *TTNN Readiness* — Pybind `create_descriptor` | No — `tilize_nanobind.cpp` binds `ttnn::tilize` via `bind_function`, no descriptor pybind |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `CustomProgramSpecFactoryConcept` |
| *Port work* — Offset base pointer | none — both slot-0 addresses are clean bases (bare `->address()`) |
| *Port work* — Tensor bindings (per binding) | input → Case 1 (`TensorAccessor`) · output → Case 1 (`TensorAccessor`) |
| *TTNN Readiness* — TensorParameter relaxation | `none` (clears) |
| *Port work* — TensorAccessor 3rd arg | none — no accessor passes a 3rd arg |
| *Port work* — CB endpoints | per set: input 1:1 · output 1:1 · staging **self-loop** (reader-only, DM self-loop). All conditional on the set being non-empty |

## Result

**GREEN → brief issued.** The single blocker from the prior audit — per-node CB sizing — is resolved.
PR #54140 rebuilt the factory on the shared `BlockBufferSet` model
(`data_movement/common/common.{hpp,cpp}`): the work split yields at most two block widths, and each
width gets its **own** buffer set with its **own** CB indices — full = `{c_0 in, c_1 staging, c_16 out}`,
cliffrow = `{c_2 in, c_3 staging, c_17 out}` (`common.cpp:919-936`). Each `buffer_index` is pushed
exactly once, at one `total_size`/`page_size`, over that set's disjoint cores (`push_buffer_set`,
`common.cpp:808-863`), so **no index is ever re-used at two different sizes** (the code says so verbatim
at `common.cpp:917-918`). That is precisely the uniform-size-per-named-DFB shape Metal 2.0 requires.

Every other gate is clear: Device 2.0 ✓, Feature compatibility ✓ (all N/A), TTNN factory concept ✓
(`Is able to port? = yes`), Offset base pointers ✓, TensorAccessor 3rd arg ✓ (N/A). Target concept is
`CustomProgramSpecFactoryConcept` — the same target the five sibling factories already ported to.

**Strong precedent:** the inverse op's block factory on the *same* `BlockBufferSet` model is already
ported and merged — `760544bad70 [Cleanup] Metal 2.0 port: untilize with UntilizeMultiCoreBlockProgramFactory (#56280)`.
The shared buffer-set helper is therefore proven portable.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN.** Sheet row (readiness CSV, block-factory
  row): `Is able to port? = yes`. Cross-check against the code is clean and consistent:
  - `Concept = descriptor` ✓ — `create_descriptor` returns `ProgramDescriptor` (`...block_program_factory.hpp:17`).
  - `Custom hash = no` ✓ — no `compute_program_hash` anywhere under `device/` (grep clean).
  - `Runtime-args update (get_dynamic_runtime_args) = no` ✓ — no hook on the device op.
  - `Override runtime args method? = yes` ✓ — `override_runtime_arguments` at `...block_program_factory.cpp:380` → target `CustomProgramSpecFactoryConcept`.
  - `Pybind descriptor = no` ✓ — nanobind binds `ttnn::tilize` only.
  - `Smuggled pointer = no` ✓ — slot-0 is a clean base (see Offset base pointers).
  - `TensorParameter relaxation = none` ✓ — clears.
  - Factory-set match: 6 sheet rows ↔ 6 code factories, one-to-one; no phantom / missing row. Cross-column invariants hold.

- **Device 2.0 (every kernel used):** **GREEN.** All three kernels are on the Device-2.0 (and beyond — DFB) API. No `InterleavedAddrGen` / `ShardedAddrGen`, no raw `noc_async_*` addr-gen, no `CircularBuffer&`, no CB-index-keyed pointer holdovers.
  - reader `reader_unary_pad_multicore_both_dims.cpp`: `Noc noc`, `DataflowBuffer dfb_in0/dfb_in1`, `TensorAccessor`, `CoreLocalMem`, `PrecomposedUnicastEndpoint`; CB indices are compile-time args (`dfb_id_in0 = get_compile_time_arg_val(6)`, `dfb_id_in1 = get_compile_time_arg_val(7)`).
  - writer `writer_unary_interleaved_start_id_wh.cpp`: `Noc noc`, `DataflowBuffer dfb(cb_id_out)`, `noc.async_write(dfb, s, ...)`, `dfb.wait_front`/`pop_front`. Its one free-function CB call — `get_tile_size(cb_id_out)` at line 24 — is **sanctioned** by the Device 2.0 green bullet (not a holdover).
  - compute `tilize_wh.cpp`: `compute_kernel_hw_startup(dfb_id_in, dfb_id_out)`, `compute_kernel_lib::tilize<...>`; CB indices are compile-time args (`dfb_id_in = get_compile_time_arg_val(3)`, `dfb_id_out = get_compile_time_arg_val(4)`).
  - Called helpers are all Device-2.0 or kernel-lib: `tt_memmove(noc, …)` (Noc-based, `data_movement/common/kernels/common.hpp`), `dataflow_kernel_lib::fill_l1_range` (`kernel_lib/l1_helpers.hpp`), `compute_kernel_lib::tilize` / `is_fp32_input_format` (`kernel_lib/tilize_helpers.hpp`).

- **Feature compatibility:** every Appendix A entry is **N/A** — none of the features is present.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer`, no `remote_cb`/`.remote_index`, no `.global_circular_buffer` field. The `push_buffer_set` CBs are plain L1 (`total_size` + `format_descriptors` only). |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `CBDescriptor` sets `address_offset`; no `.buffer`-backed / borrowed-memory CB. All CBs are plain interleaved L1 scratch (`common.cpp:833-862`). |
  | GlobalSemaphore | N/A | The factory creates no semaphores; the kernels use none. |

- **CB endpoints (GATE-free):** all legal — no dead CB, no multi-binding. Six named DFBs across two
  buffer sets, each set's CBs placed only on that set's disjoint cores; classify **per set** (each is
  conditional on the set being non-empty):
  - **full set** (cores `core_range ∪ cliff_col`): `c_0` input — reader FIFO-produces, compute consumes → **1:1**; `c_16` output — compute produces, writer consumes → **1:1**; `c_1` staging — touched **only** by the reader (`dfb_in1.reserve_back(1)` / `get_write_ptr()` / `push_back(1)`, then raw `temp_addr` scratch use, reader lines 42-45) → **self-loop** (bind reader PRODUCER + CONSUMER; DM self-loop → legal on Gen1, Quasar-uplift debt).
  - **cliffrow set** (cores `cliff_row ∪ cliff_col_row`): `c_2` input → **1:1**; `c_17` output → **1:1**; `c_3` staging → **self-loop** (same shape as `c_1`).
  - **Preserved-multiplicity note (not multi-binding):** the factory emits up to **4 compute** KernelSpecs (`core_range`→full, `cliff_col`→full, `cliff_row`→cliffrow, `cliff_col_row`→cliffrow, `...block_program_factory.cpp:225-240`) but one reader + one writer per set over the set's whole range. So `c_0`/`c_16` are consumed/produced by *two* compute KernelSpecs — over **disjoint** sub-ranges (`core_range` vs `cliff_col`). Per node the census is still 1P+1C; this is the demoting-per-group / preserved-multiplicity shape, **not** a multi-binding. Same for `c_2`/`c_17` across `cliff_row` and `cliff_col_row`.

- **Offset base pointers:** **GREEN — cleared.** Both address RTAs are clean bases with no host-folded offset:
  - reader slot 0 (`src_addr`) is emplaced as the `Buffer*` `src0_buffer` (`...block_program_factory.cpp:301-311`) and re-pointed on cache hit with the bare `tensor_args.input_tensor.buffer()->address()` (`...:410`).
  - writer slot 0 (`dst_addr`) is emplaced as `dst_buffer` (`...:314-315`) and re-pointed with the bare `tensor_return_value.buffer()->address()` (`...:411`).
  - The reader's per-row column offset (`start_column_id`, RTA slot 4) and DRAM-alignment fix-ups are kernel-side computations off the clean base (`s.get_noc_addr(page)` + `.offset_bytes`), not a host-folded `base + offset` RTA. No Type-1/2/3/4 fold present.

- **TensorAccessor 3rd argument:** **N/A** — no accessor in the factory's kernels passes a 3rd argument. Reader `TensorAccessor(src_args, src_addr)` (2 args); writer `TensorAccessor(dst_args, dst_addr)` (2 args). The subject never fires.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding):
  - `input_tensor` → **Case 1** (via `TensorAccessor`). Delivered today as a `Buffer*` in reader RTA slot 0 (`...block_program_factory.cpp:303`), fed to `TensorAccessor(src_args, src_addr)` in the reader; `TensorAccessorArgs(*src0_buffer)` baked into the reader CTAs (`...:152`). Port: bind as `TensorParameter`/`TensorBinding`; kernel builds `TensorAccessor(tensor::src)`; the RTA address, the slot-0 patch, and the CTA accessor-args all disappear.
  - `output_tensor` → **Case 1** (via `TensorAccessor`). Writer RTA slot 0 (`...:315`) → `TensorAccessor(dst_args, dst_addr)`; `TensorAccessorArgs(*dst_buffer)` in writer CTAs (`...:168`). Same treatment (`tensor::dst`).
- **TensorParameter relaxation:** `none`.
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:** self-loop `c_1` (full staging) and `c_3` (cliffrow staging); all other CBs are plain 1:1. Each set's DFB specs are **conditional** on the set being non-empty — mirroring the existing conditional `push_buffer_set` / `push_pair` structure (`...block_program_factory.cpp:102-120, 350-355`), so this is existing structure to carry, not new conditional plumbing.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none. No hidden second writer, no multi-reader, no ≥3-toucher CB. The only non-1:1 dispositions are the two staging self-loops.
- **Preserved multiplicity:** up to 4 compute + up to 2 reader + up to 2 writer KernelSpecs bind into 6 DFBs; ensure per-node 1P+1C bindings (two compute KernelSpecs may consume the same DFB over disjoint sub-ranges — this is preserved multiplicity, do **not** reach for the multi-binding flag).
- **Cross-op / shared kernels:** all 3 kernels are shared with `tilize_with_val_padding`'s block factory and have **no `_metal2` fork yet** — this port creates the first fork of each (rung 2). All 3 are already DFB-based, so each fork is a binding-layer conversion, not an idiom rewrite. Detail below.
- **RTA varargs:** none. All reader RTAs (9, slots 0-8) and writer RTAs (4, slots 0-3) are read at fixed constant indices as distinct named fields. The reader re-reads slots 3-8 each `third_dim` iteration but at the *same fixed* indices (a per-iteration reset of locals), not a variable-count loop.
- **Legacy patch plumbing dissolves on port:** the `override_runtime_arguments` hook, the `dm_kernel_metadata` common-args carrier (`{num_pairs, reader0, writer0, [reader1, writer1]}`, `...:366-371`), the belt-and-braces arg-width checks (`...:418-441`), and `patch_tilize_kernel_slot0` all exist only to re-point the slot-0 `Buffer*` on a cache hit. Under `CustomProgramSpecFactoryConcept` they collapse to an `override` that returns the two `TensorParameter` tensor_args (exactly as the 5 sibling factories did).

## Team-only

- **Out-of-directory coupling & donor shape.** Op-level roll-up: **✓ clean** (all function-call escapes are Device-2.0-shaped or kernel-lib; no ⭐/✗). No gate — the Device 2.0 gate (above) is GREEN for every donor.

  Borrowed / lent kernel files (file-path instantiation):

  | Kernel file | Owner | Also bound by | `_metal2` fork? | Rung |
  |---|---|---|---|---|
  | `reader_unary_pad_multicore_both_dims.cpp` | `data_movement/tilize_with_val_padding` (borrowed) | `tilize_with_val_padding` block | none | **2 — create fork** |
  | `writer_unary_interleaved_start_id_wh.cpp` | `eltwise/unary` (borrowed) | `tilize_with_val_padding` block | none¹ | **2 — create fork** |
  | `tilize_wh.cpp` (compute) | `data_movement/tilize` (lent — own dir) | `tilize_with_val_padding` block | none | **2 — create fork** |

  ¹ A `writer_unary_interleaved_start_id_metal2.cpp` sits in the same dir, but it is the `_metal2` fork of the **non-`_wh`** `writer_unary_interleaved_start_id.cpp` — a different kernel, **not** a reuse candidate for the `_wh` variant.

  **Sunset / coordination list (not authorization to convert in place):** all three kernels' consumer set is exactly `{data_movement/tilize block, data_movement/tilize_with_val_padding block}` — both the `_multi_core_block` factories that #51305 targets, both still on `descriptor`. TTNN-side gating means the two binder sets cannot co-migrate implicitly, so rung 2 (create `<stem>_metal2.cpp` beside each original, leave the legacy pointer comment) is the expected path; record the still-unmigrated `tilize_with_val_padding` block in the port report so the eventual last porter can sunset the legacy copies.

  Per-call function-call escapes (all workable / no donor rewrite): `tt_memmove` (Noc&, `data_movement/common/kernels/common.hpp`) ✓; `fill_l1_range` (`kernel_lib/l1_helpers.hpp`) — kernel-lib, out of porter scope; `compute_kernel_lib::tilize` / `is_fp32_input_format` / `compute_kernel_hw_startup` (`kernel_lib/tilize_helpers.hpp`, `api/compute/*`) — kernel-lib / framework.

- **TTNN factory analysis (sheet-derived, with code evidence):**
  - Current concept `descriptor`; `create_descriptor` at `...block_program_factory.cpp:43`.
  - Op-owned tensors: none (sheet empty; `descriptor` cannot carry them).
  - Custom hash: none; the port has nothing to preserve here.
  - `override_runtime_arguments`: present (`...:380`) → target `CustomProgramSpecFactoryConcept`; porter translates it into an `override` returning a `ProgramRunArgs` carrying the two tensor_args.
  - Pybind `create_descriptor`: none.

## Misc anomalies  *(team-only, non-gating)*

- **`patch_tilize_kernel_slot0` becomes dead when this factory ports.** It is declared/defined in the device op (`tilize_device_operation.hpp:45`, `tilize_device_operation.cpp:372`) but the Block factory (`...block_program_factory.cpp:435`) is now its **only** caller — the five ported siblings reference it in comments only ("replaces the legacy `patch_tilize_kernel_slot0`"). Once the Block factory ports, the helper (and its declaration) can be removed from the device op. A clean follow-up, not port-diff work.
- **Readiness-sheet staleness on the *sibling* factories (does not affect this gate).** The sheet lists all six tilize rows as `Concept = descriptor`, but the code shows the five non-Block factories on `create_program_artifacts` / `override_runtime_arguments` → `ProgramRunArgs` (i.e. `MetalV2`), merged via PR #54805. The sheet appears to lag that merge. This is a courtesy heads-up for the sheet owner; the **Block** factory's own row is accurate and consistent, so the gate stands.

## Questions for the user  *(low priority)*

1. **Sheet lag on ported siblings:** the five already-ported tilize factories still read `Concept = descriptor` in the readiness sheet (should be `MetalV2` after PR #54805). Worth flagging to the sheet owner? It does not affect the Block factory verdict.

## Recipe notes

- The audit doc's shared-kernel caution is filed under `ai/shared/port_patterns.md` and the readiness
  fetch doc references `analyses/ttnn_op_porting_readiness.md`; the `../../analyses/*` and
  `../shared/*` relative links in `audit/metal2_audit.md` resolve correctly (`analyses/` is a sibling
  of `ai/`, `shared/` is under `ai/`). No friction — noting only that a reader following the links
  literally from the op directory needs the repo open, as the doc already warns.
- No RED/GREEN boundary forced a judgment call here: the sole prior blocker was a named
  `Known op issues` cell that is now empty, and the code change that cleared it
  (per-width buffer sets, distinct indices) is exactly the DFB-uniform-size shape the audit's per-node
  reasoning requires.
