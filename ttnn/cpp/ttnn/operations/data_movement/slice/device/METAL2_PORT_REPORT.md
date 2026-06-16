# Metal 2.0 port — slice (SliceTileProgramFactory → create_program_spec)

STATUS: PORTED

## Factory chosen
The simplest single-program factory: the non-strided TILE path (`SliceTileProgramFactory`,
`slice_program_factory_tile.{hpp,cpp}`). Two kernels (reader + writer), one CB, one work-split
into up to two core groups. No semaphores, no op-owned tensors, no per-mesh-coord variation.

## The two-concept conflict and how it was resolved
`ccl/mesh_partition` builds its own `MeshWorkload` by calling
`SliceTileProgramFactory::create_descriptor(...)` directly, per mesh coordinate, inside its
own `create_at` / `override_runtime_arguments` (mesh_partition_program_factory.cpp lines ~127
and ~157). It needs a `ProgramDescriptor` to stamp into a `Program` and to feed
`apply_descriptor_runtime_args`.

A single factory struct cannot satisfy BOTH `ProgramDescriptorFactoryConcept` (has
`create_descriptor`) and `ProgramSpecFactoryConcept` (has `create_program_spec`) — the
`AllFactoriesValid` check (operation_concepts.hpp) requires each `program_factory_t` variant
alternative to satisfy EXACTLY one concept.

Resolution (the recommended split):
- **Kept** `SliceTileProgramFactory::create_descriptor` unchanged → mesh_partition keeps building.
- **Added** a separate struct `SliceTileSpecProgramFactory::create_program_spec` (Metal 2.0)
  in the same .hpp/.cpp. slice's own `select_program_factory` now routes the TILE path to it,
  and the `program_factory_t` variant lists `SliceTileSpecProgramFactory` in place of the old
  descriptor factory.
- mesh_partition still `std::visit`s slice's variant; its two lambdas were given a one-line
  `if constexpr (std::is_same_v<Factory, SliceTileSpecProgramFactory>)` branch that calls
  `SliceTileProgramFactory::create_descriptor` (the descriptor variant of the identical TILE
  work) for that alternative, and `Factory::create_descriptor` for the rest. This is the only
  edit outside the slice op dir, and it is minimal + documented.

## Kernels
Both slice-local kernels are used ONLY by the TILE factory, but the legacy descriptor path
(kept for mesh_partition) still consumes the originals via `get_named_compile_time_arg_val` /
positional `get_arg_val` / `get_common_arg_addr`. The Metal 2.0 path needs the
`experimental/kernel_args.h` named mechanism. So both were **forked** (not edited in place):

- `kernels/dataflow/reader_unary_unpad_dims_interleaved_start_id_m2.cpp` (fork of the legacy reader)
- `kernels/dataflow/writer_unary_interleaved_start_id_m2.cpp` (fork of the legacy writer)

Port is access-mechanism-only (logic, #ifdefs, loop bounds, numeric paths UNCHANGED):
- src/dst addresses: raw `src_addr`/`dst_addr` runtime args → `TensorAccessor(ta::src)` /
  `TensorAccessor(ta::dst)` (address flows through TensorBinding; the appended
  `TensorAccessorArgs` CTAs are gone).
- CB ids: `get_named_compile_time_arg_val("cb_in"/"cb_out")` → `dfb::cb_in` / `dfb::cb_out`
  (one DataflowBufferSpec `cb_in_out`, reader=PRODUCER, writer=CONSUMER, both on `all_cores`,
  same WorkUnitSpec — local-DFB rule satisfied).
- `num_dims`: positional CTA → named CTA `args::num_dims`.
- reader `start_id`/`num_tiles`, writer `num_pages`/`start_id`: positional RTAs → named RTAs.
- reader per-dim `id_per_dim[]` array → runtime **varargs** (`get_vararg`); the common
  `[num_unpadded.., num_padded..]` arrays → common **varargs** (`get_common_vararg`). The
  m2 vararg API exposes value getters only (no writable pointer into the vararg region), so
  the per-dim running counters are copied into a local stack array (`uint32_t
  id_per_dim[num_dims]`, sized by the `num_dims` CTA) and mutated there — identical arithmetic.

## Files changed
- `device/slice_program_factory_tile.hpp` — added `SliceTileSpecProgramFactory` struct (kept
  `SliceTileProgramFactory`).
- `device/slice_program_factory_tile.cpp` — added `SliceTileSpecProgramFactory::create_program_spec`
  (legacy `create_descriptor` untouched).
- `device/slice_device_operation.hpp` — `program_factory_t` variant: `SliceTileProgramFactory`
  → `SliceTileSpecProgramFactory`.
- `device/slice_device_operation.cpp` — `select_program_factory` TILE path returns
  `SliceTileSpecProgramFactory{}`.
- `device/kernels/dataflow/reader_unary_unpad_dims_interleaved_start_id_m2.cpp` — NEW (forked).
- `device/kernels/dataflow/writer_unary_interleaved_start_id_m2.cpp` — NEW (forked).
- `ccl/mesh_partition/device/mesh_partition_program_factory.cpp` — two `if constexpr` branches
  mapping the spec factory alternative back to `SliceTileProgramFactory::create_descriptor`.
  (Minimal, documented; the only change outside the slice op dir.)

The `SliceTileProgramFactory::create_descriptor` pybind hook (slice_nanobind.cpp) was LEFT in
place — it is not dangling (the descriptor factory still exists). No new pybind hook was added
for the spec factory (matches rand; the spec path is dispatched by the framework adapter).

## Blockers
None. Both slice and ccl/mesh_partition retain everything they need to build.

## Not built / not measured / not committed
Per instructions. (This worktree has no build dir; cannot compile or measure here.)

## Note on this worktree's base
This worktree is checked out on a base commit that has the Metal 2.0 host API
(`tt-metalium/experimental/metal2_host_api/*`) and adapter (`ProgramSpecMeshWorkloadFactoryAdapter`,
`ttnn/metal2_artifacts.hpp`) merged, but does NOT contain the worked-example op ports (rand /
matmul on this base are still legacy `create_descriptor`). The port was written against the
in-tree framework API, using the `dgomez/rand-metal2` rand factory as the structural template
for `create_program_spec` and the forked-kernel named-arg conventions.
