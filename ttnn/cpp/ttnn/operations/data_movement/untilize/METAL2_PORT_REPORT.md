# Metal 2.0 Port Report — untilize (7-factory native subset)

## Outcome
**PORTED** (6 of the 7 brief factories) **+ 1 CAPITULATED**. Six GREEN native factories of
`ttnn::prim::UntilizeDeviceOperation` converted to `ProgramSpecFactoryConcept`
(`create_program_artifacts`):
`UntilizeSingleCoreProgramFactory`, `UntilizeMultiCoreProgramFactory`,
`UntilizeMultiCoreNDShardInputProgramFactory`, `UntilizeMultiCoreParallelizeColumnProgramFactory`,
`UntilizeMultiCoreSubCoreGridsProgramFactory`,
`UntilizeMultiCoreInputAndOutputShardTypeAndShardSpecIdenticalProgramFactory`.

**CAPITULATED (reverted to legacy `ProgramDescriptorFactoryConcept`):**
`UntilizeMultiCoreInputAndOutputNDShardTypeAndShardSpecIdenticalProgramFactory` — a Metal 2.0
spec-validator over-strictness on borrowed-memory DFBs for **uneven ND sharding** rejects the faithful
port at spec-construction time. Full write-up in Handoff points #2. This surfaced only at test time
(the port compiled and 6 factories pass all tests); it is a **grounded capitulation on one factory**,
not a defect in the ported code. The op keeps building and running — the reverted factory dispatches
via the legacy path exactly as before the port.

Left on the legacy `ProgramDescriptorFactoryConcept` (out of scope, blocked before this port):
`UntilizeMultiCoreBlockProgramFactory` (#51305) and the whole `UntilizeCodegenDeviceOperation`
(no readiness-sheet row). The mixed variant builds and dispatches per-factory.

## Provenance
- Recipe docs (this port): `548e18500b3 2026-08-18 docs(metal_2.0): a direct-descriptor op converts to a real program factory`
- Audit docs (inherited): `548e18500b3 2026-08-18 docs(metal_2.0): a direct-descriptor op converts to a real program factory`

## TTNN ProgramFactory
### Concept realized
6 factories on `ProgramSpecFactoryConcept` (IdenticalNDShard capitulated → stays legacy; see Outcome +
Handoff #2). No `override_runtime_arguments` on any (base concept — framework refreshes tensor bindings
on cache hit). Every io-tensor `TensorParameter` has either a `TensorBinding` (interleaved / block-reader
/ ND-shard-input paths) or is named by a `DataflowBufferSpec::borrowed_from` (identical-shard, MultiCore
even-sharded input). `tensor_args` carries INPUT + OUTPUT in every ported factory.

### Device-op-class edits
- Pybind entry points removed: none (no `create_descriptor` was pybound — `untilize_nanobind.cpp` binds
  only the user-facing `untilize` + verification-only `untilize_force_*`).
- Custom `compute_program_hash`: none — nothing to touch. `select_program_factory` /
  `compute_output_specs` / `validate_*` in `untilize_device_operation.cpp` untouched.

### Open items
- **Relaxation candidates:** none applied. The sharded factories bind io tensors via `borrowed_from`
  with strict `TensorSpec`; no relaxation mirrored (the audit declared `TensorParameter relaxation = none`).
- The op would benefit from the block factory (#51305 per-node CB size) and codegen op landing once
  their prerequisites clear — both out of scope here.

## Handoff points
2. **Metal 2.0 spec-validator over-strictness on borrowed-memory DFBs for uneven ND sharding (Metal 2.0 framework team). — CAPITULATION.**
   `tt_metal/impl/metal2_host_api/program_spec.cpp:1567` enforces, at spec-construction time, a *coarse*
   check `dfb_bytes <= tensor_spec.compute_packed_buffer_size_bytes()` for any borrowed-memory DFB —
   comparing the (uniform, per-node) DFB size against the **whole logical packed tensor** size. The code
   comment there acknowledges this is imprecise for sharded tensors and that the *precise* per-bank check
   runs later in `AttachBorrowedDFBBuffers` (`program_run_args.cpp`) — but the coarse check is a hard
   `TT_FATAL`.
   For `UntilizeMultiCoreInputAndOutputNDShardTypeAndShardSpecIdenticalProgramFactory` on **uneven ND
   sharding**, the borrowed input/output DFBs are sized (faithfully, identical to the legacy backed-CB
   `total_size`) as `num_shards_per_core_in_group_1 × tile-padded-shard-tiles × tile_size`. For a heavily
   padded uneven shard this per-core size can exceed the whole logical packed tensor. Concrete failing
   case: tensor `[4,128,160]` bf16, shard `[3,96,96]`, 3-core grid → per-core DFB `3 shards × 27 tiles ×
   2048 = 165888 B` vs whole packed tensor `80 tiles × 2048 = 163840 B` — over by exactly one tile.
   *Why not worked around:* the port may not hand-tune sharded DFB sizing (silent-corruption risk,
   explicitly flagged for this tilize/untilize family in #51305), and no correct uniform DFB size can be
   both large enough to hold a core's shards and ≤ the whole logical packed tensor in this pathological
   case. The per-bank attach-time check would pass; only the coarse spec-time check rejects it.
   *Resolution:* capitulated on this one factory — reverted to legacy `create_descriptor`. **Suggested
   framework fix:** relax the spec-time coarse check for borrowed DFBs whose `borrowed_from` TensorSpec is
   L1-sharded (defer wholly to the per-bank attach-time check), or compare against the sharded per-core
   allocation rather than the logical packed size. The other 6 factories (including MultiCore even-sharded
   and 2D IdenticalShard borrowed DFBs, and all *even* ND cases) are unaffected.
3. **Latent RTA bug in `UntilizeMultiCoreParallelizeColumnProgramFactory` cliff writer (ops team).**
   The legacy factory passed the cliff-core writer **7** positional RTAs (an extra `stick_size` at
   index 2) while the writer kernel reads **6** — a positional misalignment that would corrupt the
   cliff core's writes. Latent / likely-unreached (this factory is selected only for single-tile-row
   wide-interleaved shapes and a cliff requires an uneven `nblocks` split). Audit Misc anomaly #1.
   *Port handling:* per the brief ("translate faithfully as-is; do not fix, do not replicate"), the
   named-argument model has no slot for the extra value, so it is dropped; the cliff core now receives
   the same well-formed 6-arg set as a full core (with the cliff tile count). This is not a deliberate
   fix — it is what a name-based translation produces — but it is a behavior change on the (unreached)
   cliff path relative to the legacy positional dispatch. Flagged for the ops team to fix the legacy
   factory / kernel contract properly. Code comment left at the cliff-writer arg site.

## Successes
- **Existing `_metal2` forks reused as-is** (Watch-for in the brief): `untilize_metal2.cpp`
  (dfb::src/dfb::out) for SingleCore/SubCoreGrids/ParallelizeColumn compute, and
  `reader_unary_interleaved_start_id_metal2.cpp` (dfb::in/tensor::src, args num_pages/start_id) for
  SubCoreGrids/ParallelizeColumn readers. No second fork created; binding vocab adopted from the forks.
- **Borrowed-memory DFBs** (migration guide + patterns): identical-shard and identical-nd-shard bind
  both c_0←INPUT and c_16←OUTPUT via `borrowed_from`; MultiCore even-sharded binds c_0←INPUT. The
  validator's "borrowed_from counts as used" exception let the sharded readers/writers carry no tensor
  binding, exactly as planned.
- **Disjoint-node work split (avoiding the CTA→RTA demotion anti-pattern):** MultiCore and
  ParallelizeColumn keep full+cliff compute as two `KernelSpec`s over disjoint `WorkUnitSpec`s, with
  reader+writer listed in both WUs so their union is the whole grid — no per-group CTA demoted to RTA.
- **`unpack_modes` FP32 rule:** `fp32_dest_acc_en` is true only for INT32/UINT32/FLOAT32 (all 32-bit
  formats), so `UnpackToDest` always fits the DFB format and the required-entry rule for Float32 is
  satisfied; mirrors legacy `UnpackToDestFp32` faithfully.

## Friction
- **Confusion — the ParallelizeColumn cliff-writer bug + "translate faithfully as-is, do not replicate."**
  These two instructions are in tension for a *positional-misalignment* bug: a name-based model cannot
  reproduce a positional misalignment without deliberately fabricating misaligned named values (which
  "do not replicate" forbids). Resolved as: name the 6 args, drop the 8th, document. A recipe note that
  "a positional-only legacy bug that the named model cannot express is dropped, not reproduced" would
  remove the ambiguity.
- **Gap (minor) — `KernelSpec::CompilerOptions` field order.** `defines` precedes `opt_level` in the
  struct, so `{.defines = ..., .opt_level = ...}` is the required designated-initializer order; the
  recipe/migration-guide examples only ever show `opt_level` alone, so the order isn't obvious. Noted
  here for the next porter that sets both (compute kernels needing DST_ACCUM_MODE + O3).

## Open items for downstream
### Shared kernel touches
Rung legend: **reuse** = bound an existing `_metal2` fork; **create** = added the `_metal2` fork beside
the original + pointer comment; **in-place** = converted an untilize-only kernel with no out-of-scope binder.

| kernel | rung | fork path | remaining unmigrated consumers |
|---|---|---|---|
| `untilize/device/kernels/compute/untilize.cpp` | reuse | `untilize_metal2.cpp` (owned by fold) | untilize_with_unpadding, pool/upsample; untilize's own block factory |
| `eltwise/unary/.../reader_unary_interleaved_start_id.cpp` | reuse | `reader_unary_interleaved_start_id_metal2.cpp` | copy, pad, untilize_with_unpadding, transformer, examples |
| `untilize/device/kernels/dataflow/reader_unary_start_id.cpp` | create | `reader_unary_start_id_metal2.cpp` | copy, tilize |
| `untilize/device/kernels/compute/untilize_variable_num_blocks.cpp` | create | `untilize_variable_num_blocks_metal2.cpp` | untilize_with_unpadding |
| `eltwise/unary/.../reader_unary_sharded.cpp` | create | `reader_unary_sharded_metal2.cpp` | sharded, sharded_partial, tilize, transpose, untilize_with_unpadding, slice_write (~7 families) |
| `data_movement/sharded/.../writer_unary_sharded.cpp` | create | `writer_unary_sharded_metal2.cpp` | sharded, sharded_partial, tilize, tilize_with_val_padding, transpose, padded_slice, transformer, reduction/generic (~9 families) |
| `data_movement/sharded/.../reader_unary_nd_sharded_blocks.cpp` | create | `reader_unary_nd_sharded_blocks_metal2.cpp` | untilize_with_unpadding |
| `untilize/.../writer_unary_stick_layout_split_rows_single_core.cpp` | in-place | — | none (untilize-only) |
| `untilize/.../writer_unary_stick_layout_split_rows_multi_core.cpp` | in-place | — | none (untilize-only) |
| `untilize/.../writer_unary_stick_layout_split_rows_interleaved_parallel_columns.cpp` | in-place | — | none (untilize-only; shared only by ported ParallelizeColumn + SubCoreGrids) |
| `untilize/.../writer_unary_stick_layout_split_rows_multi_core_nd_shard.cpp` | in-place | — | none (untilize-only) |
| `untilize/.../reader_unary_sharded_blocks.cpp` | in-place | — | none (untilize-only; MultiCore block-reader path) |

- **IdenticalNDShard reverted to legacy** (Handoff #2) now consumes the **legacy** originals of
  `reader_unary_sharded.cpp`, `writer_unary_sharded.cpp`, and `untilize_variable_num_blocks.cpp` — so it
  joins their remaining-unmigrated-consumer lists above. The three `_metal2` forks stay in use by the
  ported IdenticalShard / MultiCore(even-shard) / MultiCore / NDShardInput factories, so none is orphaned.
- The block factory (`UntilizeMultiCoreBlockProgramFactory`) still binds the **legacy** originals of
  `untilize.cpp` (compute), `reader_unary_interleaved_wh_multicore.cpp`, `untilize_wh.cpp`,
  `writer_unary_stick_layout_wh_multicore.cpp` — untouched. So the reused/created forks' sunset waits on
  it and the other listed consumer families.

### Legitimate guard loss (TT_FATAL census)
- None dropped in error. All factories retain the `output.buffer() != nullptr` guard (MultiCore's was
  restored during verification after the census flagged it).

### Dropped dead CTAs (zero-behavior-change)
- `writer_unary_stick_layout_split_rows_multi_core.cpp`: legacy CTA idx 1 (`output_stick_size`) — never
  read by the kernel; dropped from the host emission.
- `writer_unary_stick_layout_split_rows_multi_core_nd_shard.cpp`: legacy CTA idx 1 (`output_stick_size`)
  and idx 8 (`input_single_tile_size`) — never read; dropped.
- `writer_unary_stick_layout_split_rows_interleaved_parallel_columns.cpp`: CTA `stick_size` is read into
  a constexpr but unused by the kernel; **kept** as a named CTA to preserve the schema faithfully rather
  than change what the host emits.

### Test coverage note
- No dedicated C++ gtest exists for the `untilize` op (only `untilize_with_unpadding` has one). Coverage
  is the Python unit/nightly/base-functionality suites.
