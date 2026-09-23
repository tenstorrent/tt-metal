# Metal 2.0 Port Brief — `data_movement/tilize` (`TilizeMultiCoreBlockProgramFactory`)

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.
> Scope is the **one** remaining factory — `TilizeMultiCoreBlockProgramFactory`. The other five tilize
> factories are already ported (`CustomProgramSpecFactoryConcept`, PR #54805); do not touch them.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `bd9e9f36292 2026-09-23 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state` *(carry this line into the port report's Provenance section)*

**Why this is now portable:** the prior blocker was per-node CB sizing. PR #54140 rebuilt the factory on
the shared `BlockBufferSet` model (`data_movement/common/common.{hpp,cpp}`): at most two block widths,
each with its **own** buffer set and its **own** CB indices — full `{c_0 in, c_1 staging, c_16 out}`,
cliffrow `{c_2 in, c_3 staging, c_17 out}` (`common.cpp:919-936`). Every `buffer_index` is pushed once,
at one size, over disjoint cores (`common.cpp:917-918`: "no index is ever re-used at two different
sizes") → each named DFB has a uniform size. The inverse op's block factory on this same model is
already ported (`760544bad70`, #56280) — a working precedent to mirror.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`):

- **Current concept:** `descriptor` — `create_descriptor` returns a `ProgramDescriptor` (`tilize_multi_core_block_program_factory.hpp:17`).
- **Op-owned tensors:** none.
- **Target concept:** `CustomProgramSpecFactoryConcept` — because `override_runtime_arguments` is present (`tilize_multi_core_block_program_factory.cpp:380`). This is the **same target the five sibling factories already ported to**; use them as your working template (e.g. `tilize_multi_core_default_program_factory.{hpp,cpp}`, `tilize_single_core_program_factory.{hpp,cpp}`), not the quasar copies.
- **Gate-cleared, confirmed absent** (each would have blocked the brief): a `TensorParameter relaxation` that is neither `none` nor an analysis pointer (it is `none`) · `get_dynamic_runtime_args` (absent). No custom hash and no pybound `create_descriptor` (neither gates; both simply absent here). An `override_runtime_arguments` is present — you translate it, not delete it.

## Construct — to do

**Tensor bindings** (per binding):

- `input_tensor` — **Case 1** (via `TensorAccessor`) → express as `TensorParameter` / `TensorBinding` (`tensor::src`). The reader builds `TensorAccessor(tensor::src)` instead of `TensorAccessor(src_args, src_addr)`. Deletes: reader RTA slot 0 (`src0_buffer`, `...:303`), the `TensorAccessorArgs(*src0_buffer)` CTAs (`...:152`), and the cache-hit slot-0 patch.
- `output_tensor` — **Case 1** (via `TensorAccessor`) → `TensorParameter` (`tensor::dst`); writer builds `TensorAccessor(tensor::dst)`. Deletes: writer RTA slot 0 (`dst_buffer`, `...:315`) and `TensorAccessorArgs(*dst_buffer)` CTAs (`...:168`).

**Translate `override_runtime_arguments` (custom concept):** the current hook (`...:380-442`) exists only
to re-point the slot-0 `Buffer*` on a cache hit — it reads a `dm_kernel_metadata` common-args carrier
(`{num_pairs, reader0, writer0, [reader1, writer1]}`, `...:366-371`), width-checks each kernel, and calls
`patch_tilize_kernel_slot0`. Under the target concept this **collapses** to an `override` returning a
`ProgramRunArgs` whose `tensor_args` are the two io `TensorParameter`s (bound to the input/output
`mesh_tensor()`), exactly as the sibling factories do. The metadata carrier, the width checks, and
`patch_tilize_kernel_slot0` all disappear (see Watch-for: the helper then becomes dead in the device op).

**TensorParameter relaxation:** `none`.

**TensorAccessor 3rd arg:** none — no accessor passes one; nothing to drop.

**CB endpoints:** six named DFBs across two conditional buffer sets. Bind per set (each set is emitted
only when non-empty — carry the existing conditionals at `...:102-120` and `...:350-355`):

- `c_0` input (full): reader **PRODUCER**, compute **CONSUMER** — plain 1:1.
- `c_16` output (full): compute **PRODUCER**, writer **CONSUMER** — plain 1:1.
- `c_1` staging (full): **self-loop** — bind the **reader** both PRODUCER and CONSUMER. It is reader-private scratch (`reserve_back(1)`/`get_write_ptr()`/`push_back(1)` then raw `temp_addr`, reader lines 42-45). DM self-loop → legal on Gen1.
- `c_2` / `c_17` / `c_3` (cliffrow set): same three dispositions as `c_0` / `c_16` / `c_1`.

Kernel-side, the CB-index compile-time args become the binding tokens: reader `dfb_id_in0`(CTA 6)→`dfb::in`, `dfb_id_in1`(CTA 7)→`dfb::staging`; writer `cb_id_out`(CTA 0)→`dfb::out`; compute `dfb_id_in`(CTA 3)/`dfb_id_out`(CTA 4)→`dfb::in`/`dfb::out`. The kernels are already DFB-based, so this is a binding-layer swap, not an idiom rewrite. While in the writer, move `get_tile_size(cb_id_out)` (line 24) onto the object it already holds — `dfb.get_tile_size()` — per kernel-side whitelist rule 7.

**Preserved multiplicity (bind carefully, don't over-flag):** the factory emits up to **4 compute**
KernelSpecs but one reader + one writer per set over the set's whole core range (`...:225-240`). So
`c_0`/`c_16` are touched by *two* compute KernelSpecs — over **disjoint** sub-ranges (`core_range` vs
`cliff_col`); per node the census is still 1P+1C. Bind each compute instance's consumer/producer role on
its own range; this is preserved multiplicity, **not** a reason to set the multi-binding advanced option.
Same for `c_2`/`c_17` (`cliff_row` vs `cliff_col_row`).

## Watch for

- **CB endpoints (multi-binding):** none — no hidden second writer, no multi-reader, no ≥3-toucher CB. The only non-1:1 dispositions are the two staging self-loops (`c_1`, `c_3`).
- **Cross-op / shared kernels — you create the first `_metal2` fork of all three (rung 2):**
  - reader `data_movement/tilize_with_val_padding/device/kernels/dataflow/reader_unary_pad_multicore_both_dims.cpp` — **borrowed**; create `reader_unary_pad_multicore_both_dims_metal2.cpp` beside it, leave the legacy pointer comment.
  - writer `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id_wh.cpp` — **borrowed**; create `writer_unary_interleaved_start_id_wh_metal2.cpp` beside it. Note: the adjacent `writer_unary_interleaved_start_id_metal2.cpp` is the fork of the *non-`_wh`* kernel — **not** a reuse target.
  - compute `data_movement/tilize/device/kernels/compute/tilize_wh.cpp` — **lent** (your own dir); create `tilize_wh_metal2.cpp` beside it.
  - All three are already DFB-based, so each fork is a named-binding conversion (`dfb::`/`tensor::` tokens, named RTAs), not a CB→DFB rewrite. Name the bindings for the **kernel's** role vocabulary, not this factory's locals — the fork's names become the interface `tilize_with_val_padding`'s block port inherits at rung 1.
  - **Sunset list, not authorization:** other binder = `data_movement/tilize_with_val_padding` block factory (still on `descriptor`). Do **not** convert any of these kernels in place; record the unmigrated `tilize_with_val_padding` block in `METAL2_PORT_REPORT.md` so the last porter can retire the legacy copies.
- **RTA varargs:** none — all reader (slots 0-8) and writer (slots 0-3) RTAs are fixed named fields; prefer named RTAs for every one. (The reader re-reads slots 3-8 each `third_dim` iteration at the *same fixed* indices — a local reset, not a vararg loop.)
- **Dead helper after your port:** `patch_tilize_kernel_slot0` (`tilize_device_operation.{hpp:45,cpp:372}`) has this factory as its **only** remaining caller (`...:435`). Once you port, remove the helper and its declaration from the device op — a clean follow-up in the same change.
- **`fp32_llk_acc` / `unpack_to_dest_mode`:** the factory sets `ComputeConfigDescriptor{.fp32_dest_acc_en, .unpack_to_dest_mode}` directly (`...:216-219`), including the UINT8 carve-out (`...:191-194`, no FP32 unpack-to-dest for UINT8). Build `ComputeGen1Config` directly and carry both faithfully — this matches what the sibling factories did (see `[[project_tilize_metal2_port]]` note: `UnpackToDest` is validator-safe here because `enable_32_bit_dest = fp32_llk_acc`). Set `opt_level = O3` on the compute `KernelSpec` (Metal 2.0 defaults compute to O2; legacy built O3).
