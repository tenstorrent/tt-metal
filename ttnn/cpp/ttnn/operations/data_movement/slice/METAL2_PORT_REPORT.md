# Metal 2.0 Port Report — `data_movement/slice`

## Outcome

**`PORTED`** — all five factories of `SliceDeviceOperation` converted to
`CustomProgramSpecFactoryConcept`: `SliceRmProgramFactory`, `SliceRmShardedProgramFactory`,
`SliceRmStrideProgramFactory`, `SliceTileProgramFactory`, `SliceTileTensorArgsProgramFactory`.

Nine slice-owned kernels converted in place; the borrowed `eltwise/unary` writer bound to its existing
`_metal2` fork (rung 1, no new fork). Nothing is left on the descriptor API, so
`patch_slice_program_addresses` and `slice_tile_dynamic_args` are deleted outright and
`ccl/mesh_partition` drives one uniform spec path.

## Verification

Bench: a single Wormhole n150. `TT_METAL_WATCHER=10` on for every run.

| | result |
|---|---|
| Pre-port baseline | **769 passed, 42 skipped, 0 failed** |
| Post-port | **769 passed, 42 skipped, 0 failed** |

Same test set both times, confirmed with the invoker:
`tests/ttnn/unit_tests/operations/data_movement/test_slice.py`,
`tests/ttnn/nightly/unit_tests/operations/data_movement/test_slice_for_conv.py`,
`tests/ttnn/nightly/unit_tests/operations/data_movement/test_universal_input_tm_slice.py`.
The op has no C++ gtests.

**Every selected kernel source is exercised.** `select_program_factory` routes by layout, sharding and
step, and the suite covers all five branches: tile interleaved and sharded
(`SliceTileProgramFactory`), row-major unstrided (`SliceRmProgramFactory`), row-major height-sharded
in and out (`SliceRmShardedProgramFactory`), row-major strided at rank ≤ 4 and, through
`test_slice_5d`, at rank > 4 (`SliceRmStrideProgramFactory`, both kernel pairs), and the
`use_tensor_args` path (`SliceTileTensorArgsProgramFactory`).

**Legality checks forced on and proven live.** All nine `skip_validation` parameters in
`tt_metal/impl/metal2_host_api/` were pinned to `false`. The two markers were given **distinct**
strings this time (see Friction), and the post-port log carries 4630 of each, so both translation
units are demonstrably fresh rather than inferred. The scaffolding was reverted before committing and
the self-audit confirms no `tt_metal/` file appears in the diff.

## Provenance

- **Recipe docs (this port):** `f9451e2a21d 2026-09-09 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`
- **Audit docs (inherited):** `f9451e2a21d 2026-09-09 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`

## TTNN ProgramFactory

### Concept realized

`CustomProgramSpecFactoryConcept` on all five factories, as the audit chose. Each ported-from
`override_runtime_arguments` was translated, not deleted: it now returns a `ProgramRunArgs`.

Each override returns a `TensorArgument` for **every** io-tensor `TensorParameter` its factory
declares, matching the set the ported-from override re-pointed:

| Factory | tensor args refreshed | other run args refreshed |
|---|---|---|
| `SliceRmProgramFactory` | `input`, `output` | none — the ported-from override patched only the two addresses |
| `SliceRmShardedProgramFactory` | `input`, `output` | none — re-binding the tensors is what re-points both borrowed buffers |
| `SliceRmStrideProgramFactory` | `input`, `output` | none |
| `SliceTileProgramFactory` | `input`, `output` | the per-core reader and writer scalars, via `slice_tile_per_core_run_args` |
| `SliceTileTensorArgsProgramFactory` | `input`, `start`, `end`, `output` | same per-core set, with a zero `start_offset` |

The per-core re-emission on the two tile factories is not an addition: the ported-from override did it
through `slice_tile_dynamic_args`, because those scalars are excluded from the cache key and go stale
on a divergent-partition hit (issue #52651). There are no op-owned tensors, so none are excluded.

### Device-op-class edits

- **Pybind entry point removed:** the `create_descriptor` `def_static` on
  `SliceTileProgramFactory` at [slice_nanobind.cpp:168-179](slice_nanobind.cpp#L168-L179). See the
  Handoff-points entry — this is user-visible and has live Python callers.
- **Custom `compute_program_hash`:** left intact at
  [device/slice_device_operation.cpp:348-432](device/slice_device_operation.cpp#L348-L432), untouched.
- **`program_factory_t`:** unchanged; the op already had a conventional factory variant.

### Open items

- No relaxation candidates were noticed. The custom hash *widens* the cache key rather than narrowing
  it, so it reveals nothing about what the op could safely ignore, and the strict default is correct.

## Handoff points

### 1. `get_vararg()` is read-only, and three slice readers advanced their argument block in place

**Owner:** Metal 2.0 host/device API team. **Status:** worked around in-kernel; no port blocked.

Three readers used their runtime-argument region as mutable per-core scratch, taking a writable
pointer into it and incrementing through it:

```cpp
tt_l1_ptr uint32_t* id_per_dim = (tt_l1_ptr uint32_t*)(get_arg_addr(2));
...
id_per_dim[j]++;
if (id_per_dim[j] == num_unpadded_tiles[j]) { id_per_dim[j] = 0; ... }
```

Sites (pre-port): `reader_unary_unpad_dims_interleaved_start_id.cpp:23` and `:45-47`;
`slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp:31-33` and `:77-79`;
`reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp:31` and `:124-127`.

`id_per_dim` is `num_dims` long and indexed by a loop variable, so by the catalog's rule it is an
indexed-collection element and cannot be named. But the generated accessor is a pure value read —
[`genfiles.cpp:535-538`](../../../../../../tt_metal/jit_build/genfiles.cpp#L535-L538) emits
`get_vararg(idx)` as `get_arg_val<uint32_t>(...)`, with no address form and no writable variant, for
either runtime or common-runtime varargs.

**Resolution taken:** copy the block into a kernel-local array once at entry and advance the local.
This is behaviour-identical — nothing reads the block back from L1 after the kernel exits, and the
host reseeds it every dispatch either way. Two shapes, decided by where `num_dims` comes from: the two
tile readers take it from a CTA, so a plain `uint32_t id_per_dim[num_dims]` works; the RM reader takes
it from an RTA, so the local is bounded by `tensor_accessor::MAX_RANK` (8, already in scope in these
kernels) with a device `ASSERT`. `data_movement/pad` already established this pattern
([../pad/device/kernels/dataflow/reader_pad_tiled.cpp:21-33](../pad/device/kernels/dataflow/reader_pad_tiled.cpp#L21-L33)).

**What would remove the workaround:** a writable vararg accessor, e.g. `get_vararg_ref(idx)` returning
`uint32_t&`, or `get_vararg_addr(idx)` returning the vararg section's L1 address, emitted beside the
existing read accessor. Either restores the legacy capability with no local copy and no rank ceiling.
The rank ceiling is the part worth flagging: it is the one place the port bounds something the legacy
kernel did not, and it is only safe because the accessor imposes the same ceiling anyway.

### 2. The recipe has no category for a host-side cross-op consumer, and slice has one

**Owner:** Metal 2.0 recipe maintainers, plus the `ccl/mesh_partition` owners.

`ccl/mesh_partition` does not merely share a kernel with slice; it drives slice's **host** factory
entry points. Its `create_at` called `SliceOp::validate_on_program_cache_miss`,
`SliceOp::select_program_factory` and `Factory::create_descriptor` inside a `std::visit` over
`SliceDeviceOperation::program_factory_t`, and its `override_runtime_arguments` called
`ttnn::prim::patch_slice_program_addresses`.

This makes the port **all-or-nothing at the `create_descriptor` boundary**, in a way the recipe's scope
rules do not anticipate. `ProgramDescriptorFactoryConcept` is satisfied by the mere presence of
`create_descriptor` ([`operation_concepts.hpp:73-74`](../../../../../api/ttnn/operation_concepts.hpp#L73-L74)),
and both Metal 2.0 concepts require `!ProgramDescriptorFactoryConcept`, so a ported factory cannot keep
the method as a compatibility shim. Because the `std::visit` lambda is instantiated for **every**
alternative, removing `create_descriptor` from even one factory breaks mesh_partition's build. There is
no subset of slice that ports without touching a peer op's host code.

**The edit made, with the invoker's explicit authorization**, is confined to
`mesh_partition_program_factory.cpp`: `create_at` now calls `create_program_artifacts` →
`MakeProgramFromSpec` → `SetProgramRunArgs`, and `override_runtime_arguments` calls the factory's
translated override → `UpdateProgramRunArgs`. Because all five factories ported, no concept branching
is needed and the diff is smaller than a partial port would have required. Nothing else in that op
changed.

The readiness sheet lists `ccl/mesh_partition` as `legacy (MeshWorkload)` with `Is able to port? = no`.
**Recipe suggestion:** the audit's *Out-of-directory coupling* subject covers two kernel-level escapes;
a third category, "another op's host code calls this op's factory entry points", would have surfaced
this at audit time as a scope item rather than leaving the porter to discover it. The auditor made the
same suggestion independently (`METAL2_PREPORT_AUDIT.md`, Recipe notes 3).

### 3. Removed pybind surface, with live Python callers

**Owner:** the fusion / OpDescriptor owners. Tagged "API surface: removed entry point."

`ttnn.SliceTileProgramFactory.create_descriptor` is gone. The factory type is still exposed; only the
method was removed, because the type name is re-exported through
`ttnn/ttnn/operations/data_movement.py` and `ttnn/ttnn/__init__.py` and deleting the whole `nb::class_`
breaks `import ttnn` outright.

**Live callers.** `models/experimental/ops/descriptors/data_movement/slice.py:54` builds an
`OpDescriptor` from it, and four fusion tests import that helper
(`tests/ttnn/unit_tests/operations/fused/parallel_sequential/test_parallel_sequential.py` at four sites,
and `.../demo/test_fused_demo.py` at two). A fusion branch consumes a `ProgramDescriptor`, and slice now
produces a `ProgramSpec`, which no branch can consume yet.

**Accommodation taken:** slice was added to the existing skip list in
`tests/ttnn/unit_tests/operations/fused/parallel_sequential/conftest.py`, which already stood in for
exactly this situation for the two layernorm factories (issue #54365). The guard keys on the missing
method rather than on the list, so it becomes a no-op the day a spec-consuming branch exists. This is a
test-tree edit outside the op directory, taken because the mechanism and the precedent were already
there; the alternative was leaving six tests failing.

The `models/` helper itself is **left broken on purpose** — rewriting it needs the spec API exposed to
Python, which does not exist yet, and it is outside the porter's scope. It raises `AttributeError` if
called outside those tests.

## Successes

- **The catalog's shared-kernel rung 1, run *locationally*, found the fork.**
  `SliceTileTensorArgsProgramFactory` borrows the `eltwise/unary` writer, and
  `writer_unary_interleaved_start_id_metal2.cpp` already sits beside it. Listing the original's
  directory is what surfaced it; a tree-wide filename grep returns quasar-tree hits that are not
  siblings of anything. Binding the fork and adopting its vocabulary (`dfb::out`, `tensor::dst`,
  `args::num_pages`, `args::start_id`) cost nothing because slice's writer arguments were already
  exactly that pair of scalars.
- **[Unity-build hygiene](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-unity-build-hygiene-for-anonymous-namespace-symbols)
  was a real hazard, not a hypothetical.** `ttnn_op_data_movement` sets `TT_ENABLE_UNITY_BUILD` and
  slice has five factory translation units that all want a `READER` / `input` / `output` constant.
  The pattern's remedy is what `device/slice_metal2_names.hpp` implements, with per-factory buffer
  names kept local and distinctly identified.
- **The `hw_config` "match on values, not role names" instruction** confirmed every slice kernel
  resolves to the plain reader/writer defaults, so the arch-agnostic TTNN helpers reproduce them
  byte-for-byte. No custom triple anywhere.
- **The "preserve the bugs" rule fired on a real inconsistency.** The ported-from cache-miss path gives
  a no-op core writer args `{0, 0, 0}` while `slice_tile_dynamic_args` re-emitted that core's writer
  `start_id` as the running `num_tiles_written`. Both are inert (`num_pages == 0`), and the port
  reproduces the divergence rather than smoothing it, with a comment saying why.

## Friction

### Gaps

- **The recipe's vararg guidance never says varargs are read-only.** Handoff point 1. Neither the
  [caution](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#caution-avoid-varargs-unless-absolutely-necessary)
  nor the migration guide mentions it, and both the audit and the brief flagged the mutation here with
  the instruction that "the vararg has to be writable on the kernel side", which reads as though the
  capability exists. A sentence in the caution saying varargs are read-only, and that a kernel writing
  to its argument region copies the block into a local, would have saved the whole investigation.
- **The recipe has no category for a host-side cross-op consumer.** Handoff point 2.
- **"Remove pybound legacy factory entry points" does not say entry point, not class.** Exception 1 and
  the brief both name a line range that includes the `nb::class_` line. Deleting that range breaks
  `import ttnn`, because the class name is re-exported in two Python files — a failure that appears at
  test-collection time as an unrelated-looking `AttributeError` in `conftest.py`. The right granularity
  is the `def_static` alone. Worth one sentence in the pattern, plus a note that the removal may need
  a companion accommodation for existing Python callers.
- **`ProgramSpecFactoryConcept`'s exclusion of `ProgramDescriptorFactoryConcept` deserves a callout.**
  The recipe describes a half-ported op as routine, which it is. What it does not say is that a factory
  cannot retain `create_descriptor` as a temporary shim, because the method's mere presence flips the
  concept back. For an op whose entry points another op calls, that turns a per-factory decision into
  an op-wide one.

### Confusion

- **"Preserve the multiplicity" versus a runtime source selection.** `SliceRmStrideProgramFactory`
  selects *between* two kernel-source pairs by rank, which reads at a glance like the multi-variant
  case. It is not multiplicity: only one pair is ever built, so it is one `KernelSpec` pair whose
  `source` and `runtime_arg_schema` are chosen at construction. The distinguishing question is whether
  the legacy factory pushes *two* `KernelDescriptor`s of the same source into one program, or *chooses*
  between sources.
- **The two-marker legality proof does not work as written.** The recipe gives one
  `log_warning(tt::LogMetal, "METAL2_CHECKS_FORCED")` and says to add it "once in each file", then to
  expect "**two markers present**". Both copies emit the *same* string, so a log grep cannot tell two
  live translation units from one live and one stale — the exact failure the check exists to catch.
  This port used `METAL2_CHECKS_FORCED_program_spec` and `METAL2_CHECKS_FORCED_program_run_args`
  instead, which makes the check mean what it says. Suggest the recipe do the same.
- **The `opt_level` self-audit item reads as high-stakes but was vacuous here.** It is written around
  compute kernels, whose legacy default is `O3` against Metal 2.0's `O2`. Slice has no compute kernel,
  so every resolved level is `O2` on both sides. Establishing that still cost a pass over the whole op.

## Open items for downstream

- **Shared kernel touches.** One: `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`,
  borrowed by `SliceTileTensorArgsProgramFactory`. **Rung 1 — reused the existing fork**
  `writer_unary_interleaved_start_id_metal2.cpp` in that same directory. No new file was created and
  the legacy original was not touched, so it needs no pointer comment from this port. **Slice is now
  off the legacy copy.** Roughly 30 factories still bind it, tracked in issue
  [#52228](https://github.com/tenstorrent/tt-metal/issues/52228); that list is one shorter.

  Slice's own nine kernels are bound by no other non-quasar op and by no two slice factories, so all
  nine converted in place with no fork and nothing for a sibling porter to coordinate with.
- **`models/experimental/ops/descriptors/data_movement/slice.py` needs rewriting** once a `ProgramSpec`
  is exposed to Python. Until then its fusion tests skip through the conftest guard. Same issue as the
  layernorm factories, #54365.
- **Two unreferenced kernel files remain** in the op directory:
  `device/kernels/dataflow/strided_slice_reader_rm_interleaved_nd.cpp` and
  `device/kernels/dataflow/strided_slice_writer_rm_interleaved.cpp`. No factory names them and a
  repository-wide grep finds only themselves. They are the only files in the op still on legacy
  argument idioms, which now makes them conspicuous. Candidates for deletion, outside the port.
- **Test coverage note.** `tests/nightly/t3000/ccl/test_mesh_partition.py` is the only test exercising
  the `ccl/mesh_partition` path this port rewrote, and it needs a T3000. The bench used here is a
  single Wormhole n150, so **the mesh_partition change is compile-verified only and was not run.** It
  should be exercised on T3000 hardware before merge. This is the highest-risk untested surface in the
  change, because mesh_partition now takes an entirely new build-and-refresh path.
- **Sweeps were not run.** The confirmed baseline was the primary and nightly unit tests; the three
  `tests/sweep_framework/sweeps/data_movement/slice/` files are CI-run and were left to CI.
- The audit's misc anomalies are carried forward unchanged. Two are now easier to retire than before:
  `compile_time_element_size` survives as a named compile-time argument the stride kernels still never
  read, and it no longer has to occupy a slot to keep a `TensorAccessorArgs` offset correct, so it is a
  one-line deletion for whoever owns that cleanup. The dead `writer_kernel_args` half of the sharded
  helper's return pair is likewise now trivially removable.
