# Metal 2.0 Port Report — upsample `UpsampleBilinearProgramFactory`

## Outcome

`PORTED` — `UpsampleBilinearProgramFactory` converted to `ProgramSpecFactoryConcept`. All 25 bilinear
cases in `tests/ttnn/unit_tests/operations/pool/test_upsample.py::test_bilinear_multi_core` pass on
Wormhole with the Metal 2.0 legality checks force-enabled (both `METAL2_CHECKS_FORCED` markers present)
and `TT_METAL_WATCHER=10`. The op's three sibling factories were already Metal 2.0 and are untouched.

## Provenance

- **Recipe docs (this port):** `23861284522 2026-09-08 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`
- **Audit docs (inherited):** `23861284522 2026-09-08 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`

## TTNN ProgramFactory

- Concept realized: `ProgramSpecFactoryConcept` (one `create_program_artifacts`, no override, no op-owned tensors).
- Custom hash: none (left intact — nothing to preserve).
- Device-op-class edits forced: **exception 1** — header method declaration changed from
  `create_descriptor(...) -> ProgramDescriptor` to `create_program_artifacts(...) -> ProgramArtifacts`.
  No pybind entry point removed (`create_descriptor` was never bound; nanobind binds only `upsample`).
- Open items: none with the concept fit.

## Handoff points

None. No capitulation, no out-of-directory edit, no pybind removal, no boundary-rule assumption violation.
The only cross-scope files touched were `tt_metal/impl/metal2_host_api/program_{spec,run_args}.cpp` for the
temporary legality-force scaffolding, reverted before commit (verified: no `tt_metal/` file in the diff).

## Successes

- **Compute config dropped-field guidance** (`metal2_port.md#compute-kernels`) fired correctly. The legacy
  factory resolves `dst_full_sync_en` via `get_compute_kernel_config_args` but never sets it on its
  `ComputeConfigDescriptor` — a dropped field. Following the recipe I forced `double_buffer_dest = true`
  after `to_compute_hardware_config` (`upsample_bilinear_program_factory_multicore.cpp:285`) rather than
  trusting the resolved config, preserving legacy behavior for any caller that sets the knob.
- **Self-loop vs two-toucher distinction** (`port_patterns.md`) resolved the two borrowed buffers cleanly:
  `out` (c_5) is a compute sole-toucher → self-loop (PRODUCER+CONSUMER, shared accessor); `halo` (c_0) is
  two role-free raw-readers (reader+writer over one grid) → 1P+1C, **not** the multi-binding flag. The
  audit's endpoint census matched the kernel-touch reality exactly.
- **Accessor-name vs spec-name split** (clone reference) let the shared reader/writer source bind different
  intermediate DFBs (c_1/c_3 vs c_2/c_4) behind the same `dfb::tilize_reduce` / `dfb::in_scalar` accessors —
  the natural expression of the dual-instance work-split without a multi-binding flag.

## Friction

### Gaps

None material. The recipe and the three already-ported sibling factories in this same directory covered
every construct.

### Confusion

- **`face_geometry` → which DFB field?** The legacy CBs set `CBFormatDescriptor::face_geometry`; the recipe's
  "copy `format_descriptors[i].tile`" line names the `tile` field, not `face_geometry`. The declaring header
  (`dataflow_buffer_spec.hpp`) resolved it: `face_geometry` maps to
  `DataflowBufferSpec::unpack_face_geometry_metadata` (distinct from `tile_format_metadata`). Going to the
  header first (per the recipe's "go to the headers" guidance) was faster than hunting a precedent — none of
  the ported siblings use partial-face tiles.

## Open items for downstream

- **Dead compute CTAs (ops-team cleanup, not port work).** `bilinear.cpp` reads `in_ntiles_hwc` and
  `window_size_hw` (now `args::in_ntiles_hwc` / `args::window_size_hw`) into `constexpr` locals that are never
  used; `num_output_tiles` (from `args::out_ntiles_c`) is likewise computed-but-unused. Carried across
  verbatim per the brief. Pruning them (and the matching factory CTA entries) is a separate cleanup.
- **`MAX_TILES_PER_REDUCTION = 8` duplicated** as a literal in the factory
  (`upsample_bilinear_program_factory_multicore.cpp`) and the compute kernel (`bilinear.cpp:79`). A single
  source of truth would be cleaner; not port work.
- **Optional Device-2.0 cleanup (not done).** `bilinear.cpp`'s `llk_push_pages_bilinear` uses the sanctioned
  `get_local_cb_interface(operand).fifo_*` free-function form and `reduce_h_fused` extracts `dfb.get_id()` to
  feed LLKs. Both are pre-existing Device-2.0 idioms; the brief marks moving the metadata lookup onto the DFB
  object as optional cleanup, deliberately left alone to keep this a binding-layer-only diff.
- **Shared kernels:** none. Both bilinear kernels are op-owned and bound by no other op; no `_metal2` fork
  exists or was created.
