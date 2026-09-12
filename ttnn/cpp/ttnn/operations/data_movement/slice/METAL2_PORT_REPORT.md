# Metal 2.0 Port Report — `data_movement/slice`

## Outcome

**`PORTED`** — all five program factories (`SliceRmProgramFactory`,
`SliceRmShardedProgramFactory`, `SliceRmStrideProgramFactory`, `SliceTileProgramFactory`,
`SliceTileTensorArgsProgramFactory`) converted to `CustomProgramSpecFactoryConcept`, together with
the ten kernel entry points they bind. Nothing was left for a later pass.

**Tests match the pre-port baseline exactly**, over the set the invoker confirmed, with
`TT_METAL_WATCHER=10` on for every run and no Watcher report:

| | before the port | after |
|---|---|---|
| `tests/ttnn/unit_tests/operations/data_movement/test_slice.py` | 462 passed, 38 skipped | 462 passed, 38 skipped |
| the two nightly `data_movement` slice tests | 321 passed, 4 skipped | 321 passed, 4 skipped |
| **combined, on the exact committed tree** | **783 passed, 42 skipped** | **783 passed, 42 skipped** |

The Metal 2.0 legality checks were forced on for every run and proven live in the binary
(`METAL2_CHECKS_FORCED` in both translation units). Its count over the unit-test run rose from
**190 before the port to 9,163 after** — the jump is slice's own programs now being built and
validated through the Metal 2.0 spec path. The forcing scaffolding is working-tree only and is
**not** in this commit.

`ccl/mesh_partition` is the one part of this change that could not be run here; see
[Handoff points 1](#1-cclmesh_partition-re-wired-to-slices-metal-20-entry-points--out-of-directory-change).

## Provenance

- **Recipe docs (this port):** `c07a9a48c61 2026-09-12 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`
- **Audit docs (inherited):** `c07a9a48c61 2026-09-12 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`

## TTNN ProgramFactory

### Concept realized

`CustomProgramSpecFactoryConcept` on all five factories, as the audit chose. Each factory's
`override_runtime_arguments` was translated (not deleted) into one returning a `ProgramRunArgs`.

All five delegate to one shared helper, `ttnn::prim::slice_program_run_args`
(`device/slice_program_factory_rm_sharded.cpp`), which replaces the legacy
`patch_slice_program_addresses` in the same file and keeps the single-home shape the legacy code
had — MeshPartition calls it too.

**Every io-tensor `TensorParameter` gets a `TensorArgument` on every dispatch**, on all five
factories; none is deliberately skipped. That is a faithful translation of what the legacy
override did:

| Factory | What the legacy override refreshed | What the ported override returns |
|---|---|---|
| `SliceRmShardedProgramFactory` | the two borrowed-CB addresses, via a CB-address-only `ProgramDescriptor` | `tensor_args` for `input` + `output`; the borrowed DFBs draw their backing address from those |
| `SliceRmProgramFactory`, `SliceRmStrideProgramFactory` | reader slot 0 and writer slot 0 (the two base addresses) | `tensor_args` for `input` + `output` |
| `SliceTileProgramFactory`, `SliceTileTensorArgsProgramFactory` | the addresses **and** the per-core scalars, via `slice_tile_dynamic_args` | `tensor_args` for every bound tensor, plus `kernel_run_args` from `slice_tile_run_args` carrying the same per-core scalars |

The `patch_slot0` zero-slot skip disappears rather than being translated: it existed only because
`SliceTileProgramFactory` wrote a literal `0u` into writer slot 0 on no-op cores while
`SliceTileTensorArgsProgramFactory` bound the buffer on every core. A typed binding is uniform per
kernel, so the two factories no longer disagree and there is no zero to skip.

### Device-op-class edits

- **Pybind entry points removed:** `SliceTileProgramFactory.create_descriptor` and the class
  binding that carried it, `slice_nanobind.cpp:167-179`. See [Handoff points](#handoff-points) —
  this is a user-visible API removal with a live downstream caller.
- **Custom `compute_program_hash`:** left intact at `device/slice_device_operation.cpp:348`.
  Confirmed untouched — `device/slice_device_operation.cpp` is not in the port's diff at all, nor
  are `slice.cpp`, `slice.hpp` or `slice_device_operation_types.hpp`.

### Open items

- **Relaxation candidates.** None applied; the readiness sheet reads `none` on all five rows. Two
  conjuncts of the custom hash carry written justifications that are specific to the legacy
  binding model and are now stale as *reasons*, though the hash itself is still correct and must
  stay: the end tensor's memory config is hashed because "its memory config picks the bank table
  and cannot be refreshed on a hit" (`device/slice_device_operation.cpp:391-392`), and the
  preallocated output's spec because "every factory bakes a `TensorAccessorArgs` for that
  destination buffer into its writer's compile-time args" (`:415-418`). Under typed bindings both
  are refreshed on a hit. Whether the hash can be narrowed is an ops-team call, not a port one.
- No `TensorSpec` legality failure was observed on any cache hit, so the hash's containment of the
  (empty) relaxation set holds in practice across the whole test set.

## Handoff points

### 1. `ccl/mesh_partition` re-wired to slice's Metal 2.0 entry points — out-of-directory change

**Owner: the CCL / MeshPartition team.** Tagged "API surface: consumer of another op's factory."

`ttnn/cpp/ttnn/operations/ccl/mesh_partition/device/mesh_partition_program_factory.cpp` drives
slice's factories directly rather than going through `ttnn::prim::slice`. Every entry point it
used changed in this port, so it could not be left alone. The invoker confirmed before the port
started that it moves in the same change.

What changed, and nothing else:

- `create_at` (`:131-140`) — was `Factory::create_descriptor(...)` → `Program{descriptor}`. Now
  `Factory::create_program_artifacts(...)` → `MakeProgramFromSpec(...)` + `SetProgramRunArgs(...)`.
  These are the two steps TTNN's Metal 2.0 adapter performs for a single-program op; MeshPartition
  builds one `Program` per mesh coordinate itself, so it performs them itself.
- `override_runtime_arguments` (`:158-166`) — was
  `ttnn::prim::patch_slice_program_addresses(program, ...)`. Now
  `UpdateProgramRunArgs(program, ttnn::prim::slice_program_run_args(...))`.
- `mesh_partition_device_operation.hpp:46-47` — a comment naming the renamed helper.

**MeshPartition itself was NOT ported.** It is a `MeshWorkloadFactoryConcept` op (per-coordinate
`create_at` returning a `CachedProgram`), which the port recipe explicitly does not cover. It
still stores a `SliceDeviceOperation::program_factory_t` in its `shared_variables_t` and still
builds per-coordinate programs. This change only re-points it at slice's new entry points.

**It could not be verified on this host** — every MeshPartition test needs a multi-device mesh
and this bench is a single-chip N150 Wormhole. The change compiles, and its two call sites are
mechanical, but **it has not been run.** Dispatch these before merge:

| Pipeline | Job / entry | Command | SKU |
|---|---|---|---|
| **`(T3K) T3000 e2e tests`** (`.github/workflows/t3000-e2e-tests.yaml`; daily 06:00 UTC cron + dispatch, also reachable from `pipeline-select-t3k.yaml`) | `t3k_ccl_tests` (`tests/pipeline_reorg/t3k_e2e_tests.yaml:12-19`) | `pytest tests/nightly/t3000/ccl` | `wh_llmbox` |
| **`Nightly tt-metal L2 tests`** (`.github/workflows/tt-metal-l2-nightly.yaml`; daily 06:00 UTC cron + dispatch), category **`ccl`** | `Galaxy CCL tests` (`tests/pipeline_reorg/ops_unit_tests.yaml:437-445`) | `pytest tests/nightly/tg/ccl --ignore tests/nightly/tg/ccl/moe` | `wh_galaxy` |

`t3k_ccl_tests` is the one that matters most here: it collects the whole directory, so it picks up
`test_mesh_partition.py` without that file being named anywhere in CI config.

**The `ccl` category is mandatory for this PR, not optional.** `.github/workflows/test-command.md:582`
maps `ttnn/cpp/ttnn/operations/ccl/**` to category `ccl`, and the surrounding rule makes
`tt-metal-l2-nightly` mandatory whenever the diff reaches an op family with a category selector.
This diff touches both `ccl/**` and `data_movement/**`, so the dispatch is a single one carrying
`additional_test_categories: ccl,data_movement`.

**Other places MeshPartition is exercised** (found by grepping for the op, not the filename — a
filename search misses the first of these entirely):

- `tests/ttnn/docs_examples/test_ccl_examples.py:126` — `test_mesh_partition`, parametrized to a
  `(1, 2)` mesh. Runs in the same l2-nightly under category **`docs_examples`**
  (`ops_unit_tests.yaml:469-484`) on `wh_n150_civ2`, `wh_n300_civ2`, `bh_p100`,
  `bh_p150b_civ2_viommu` — so it **executes on n300** and skips on n150. That category is not
  auto-selected by this diff (it maps to no `operations/` path), so it has to be passed by hand
  to get the cheapest real coverage of this change.
- `models/demos/deepseek_v3/tests/unit/test_mesh_partition.py` — run by
  `tests/pipeline_reorg/models_unit_tests.yaml:807,2036` and
  `tests/scripts/multihost/run_quad_galaxy_tests.sh`.
- `models/demos/deepseek_v3/tests/fused_op_unit_tests/mla/test_mesh_partition.py`, which imports
  its generator from the t3000 test above.

### 2. Removed pybind surface — `SliceTileProgramFactory.create_descriptor`, with a live caller

**Owner: whoever maintains `models/experimental/ops/descriptors/`.** Tagged "API surface: removed
entry point."

`slice_nanobind.cpp:167-179` exposed `SliceTileProgramFactory::create_descriptor` to Python,
returning a `ProgramDescriptor` for the op-descriptor / fusion tooling to consume. The port
deletes it, because the function it binds no longer exists.

The class binding had no other member, so its whole Python re-export chain went with it:

- `ttnn/cpp/ttnn/operations/data_movement/slice/slice_nanobind.cpp` — the
  `nb::class_<SliceTileProgramFactory>` block.
- `ttnn/ttnn/operations/data_movement.py:550` — `SliceTileProgramFactory = ttnn._ttnn…`.
- `ttnn/ttnn/__init__.py:639` — the re-export into the `ttnn` namespace.

The last two are outside the op directory. They were forced: leaving either one turns every
`import ttnn` into an `AttributeError` at import time, which is how this surfaced — the first
post-port test run failed at collection, not in a test.

**There is one live caller, and the port does not fix it:**
`models/experimental/ops/descriptors/data_movement/slice.py:54` calls
`ttnn.SliceTileProgramFactory.create_descriptor(params, tensor_args, output_tensor)` and wraps the
result in an `OpDescriptor`. That module will now fail. It is not imported at `ttnn` import time
(nothing outside `models/experimental/ops/descriptors/` imports it), so it does not break the
test suite — but it needs a maintainer. There is no drop-in replacement: `create_program_artifacts`
returns a `ProgramSpec` + `ProgramRunArgs`, not a `ProgramDescriptor`, so the `OpDescriptor`
wrapper needs a different shape rather than a retargeted call.

The neighbouring bindings at `slice_nanobind.cpp:138-166` (`SliceParams`, `SliceInputs`, and
`SliceDeviceOperation::create_output_tensors` / `compute_output_specs`) are not factory entry
points and were left alone, along with their Python re-exports.

### 3. Metal 2.0 gap — no writable or addressable runtime-vararg accessor

**Owner: the Metal 2.0 API team.** Tagged "API: missing capability."

`get_vararg(i)` / `get_common_vararg(i)` return a value. There is no address form and no write
form: `tt_metal/jit_build/genfiles.cpp:535-538` emits exactly those two helpers, and no
`get_vararg_addr` exists anywhere in the tree.

Three of slice's readers used their runtime-argument buffer as **mutable per-dimension scratch** —
they took a raw pointer into it and incremented the values in place as an index odometer:

```cpp
tt_l1_ptr uint32_t* id_per_dim = (tt_l1_ptr uint32_t*)(get_arg_addr(2));
...
id_per_dim[j]++;
if (id_per_dim[j] == num_unpadded_tiles[j]) { id_per_dim[j] = 0; src_tile_id += num_padded_tiles[j]; }
```

The *read-only* vararg blocks in the same kernels translate mechanically to `get_vararg(i)`. This
one does not. Affected readers:
`device/kernels/dataflow/slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp` (rank arrives
as a runtime arg), `device/kernels/dataflow/reader_unary_unpad_dims_interleaved_start_id.cpp` and
`device/kernels/dataflow/reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp` (rank is a
compile-time arg).

**Resolution used, agreed with the invoker before any code was written:** a `ScratchpadSpec` per
affected kernel. That is Metal 2.0's own primitive for "a private, uninitialized region of
node-local SRAM for a kernel to use as working memory" (`scratchpad_spec.hpp:18-21`). The host
sizes it from the rank, binds it to the reader, and the kernel seeds it from the vararg values at
entry and mutates it there. Values, arithmetic and results are unchanged, and the scratch stays in
L1 where it was.

**This is a good outcome, and it is why this entry is a capability note rather than a blocker** —
but the recipe's kernel-side whitelist has no entry for it, the patterns catalog has no entry for
`ScratchpadSpec` at all, and the route was not obvious. See [Friction](#friction).

### 4. `num_runtime_varargs_per_node` is deprecated, and it is the only fit for one kernel

**Owner: the Metal 2.0 API team.** Tagged "API: deprecated field with no replacement for this shape."

`KernelAdvancedOptions::num_runtime_varargs_per_node` is marked
`[[deprecated]]` with the note "This feature is truly bizarre. It will be removed from the API
once existing uses are refactored to avoid it" (`advanced_options.hpp:85-92`).

`SliceRmShardedProgramFactory`'s reader needs it. Its argument-list length genuinely differs per
core: each output shard draws rows from a different number of input shards, and those rows
coalesce into a different number of chunks
(`device/slice_program_factory_rm_sharded.cpp`, `get_slice_runtime_args_rm_sharded`). The uniform
`num_runtime_varargs` cannot express that. The alternative — pad every core to the maximum —
would change the dispatch footprint, which the port is not entitled to do.

So this is a use that the deprecation note's "refactored to avoid it" does not obviously cover:
the per-core variation is intrinsic to the work split, not an artifact of legacy argument
packing. Worth checking before the field is removed. (It compiles clean today:
`-Wno-deprecated-declarations` is set tree-wide at `CMakeLists.txt:211`.)

## Successes

- **[Caution: Porting a shared kernel](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#caution-porting-a-shared-kernel),
  rung 1, fired exactly as written.** `SliceTileTensorArgsProgramFactory` binds
  `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`, which thirteen
  other factories also bind. The entry's instruction to `ls` the original's directory *before*
  doing anything found `writer_unary_interleaved_start_id_metal2.cpp` sitting beside it, and the
  fork's own header comment states that its names are its interface. The port bound the fork and
  took its vocabulary (`dfb::out`, `tensor::dst`, `num_pages`, `start_id`) rather than slice's own
  local names. No new file, no edit to the legacy original, no fork of the fork.

- **The same-basename warning in the brief paid for itself.** Slice owns its *own*
  `device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`, bound by
  `SliceTileProgramFactory`, which differs from the borrowed one. The two factories sit in
  adjacent files and read almost identically. Keying on the full path, as the brief said, is what
  kept `SliceTileProgramFactory` pointing at the slice-owned copy (converted in place, since no
  other op binds it) and `SliceTileTensorArgsProgramFactory` at the shared fork.

- **[Caution: Avoid varargs](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#caution-avoid-varargs-unless-absolutely-necessary)
  caught trap (1) on two kernels.** `reader_multicore_slice_4d.cpp` and
  `writer_multicore_slice_4d.cpp` walk their arguments with a running `rt_args_idx++` at the top
  of the kernel, which reads exactly like a vararg loop. The caution's explicit non-signal —
  "`arg_index++` is not a vararg signal… a distinct scalar read once is **named**" — is what made
  those 24 and 8 reads named arguments instead of varargs. The brief independently flagged the
  same two kernels, so the warning fired twice.

- **The endpoint re-derivation instruction changed nothing, which is itself the result.** The
  recipe says to re-run the kernel-touch census rather than transcribe the brief's dispositions.
  Doing so confirmed all three self-loops (one toucher each) and all four plain 1:1 DFBs. No
  disagreement with the brief, and `allow_instance_multi_binding` is set nowhere.

- **The `experimental/quasar/` warning was load-bearing.** A quasar copy of this op exists at
  `ttnn/cpp/ttnn/operations/experimental/quasar/slice`, and `padded_slice` sits beside it; both
  surfaced in build logs and directory listings during the port. Neither was opened.

## Friction

### Gaps

- **The recipe has no entry for a kernel that writes to its own argument buffer, and
  `ScratchpadSpec` is absent from the whole doc set.** This was the port's one real stuck point
  (see [Handoff points 3](#3-metal-20-gap--no-writable-or-addressable-runtime-vararg-accessor)).
  The kernel-side whitelist's rule 4 covers arguments as *inputs*; nothing covers an argument
  block the kernel mutates. `ScratchpadSpec` / `Scratchpad<T>` — the right answer — appears in
  neither the migration guide, the patterns catalog, nor the recipe; it was found by reading
  `kernel_spec.hpp` field by field, which is exactly what the recipe's "go to the headers first"
  instruction says to do, but only after the local-array workaround had already been proposed to
  the invoker. **Suggested fix:** a patterns-catalog entry, "Kernel-mutable scratch → ScratchpadSpec,"
  with the recognition signal being a legacy `get_arg_addr()` cast to a non-const pointer, and a
  mention of `ScratchpadSpec` in the migration guide's spec list (it is a `ProgramSpec` field,
  `program_spec.hpp:88`, and currently undocumented there).

- **The brief's vararg table did not distinguish read-only blocks from mutated ones.** It listed
  six kernels and the blocks each reads, which is accurate, but that is not the distinction that
  decides whether a block translates mechanically. Three of the six mutate their block, and that
  is the only thing about them that was hard. **Suggested fix:** have the audit's RTA-varargs
  subject ask whether the kernel *writes* to the block, and flag a write as port work rather than
  as a plain vararg row.

- **The recipe's "expect a long stretch with no green build" does not describe a descriptor-API
  port with no compute kernel.** The host side of all five factories compiled on the first build
  attempt, and every one of the ten kernel entry points compiled on the first test run that
  reached the JIT. That is worth knowing: the warning is calibrated for compute-heavy multi-source
  factories, and reading it set an expectation of a much rougher ride. What did bite instead was
  the pybind removal (below), which is not where the recipe points the porter's attention.

### Confusion

- **"Delete the pybind line(s)" understates the removal when the bound class has one member.**
  [Pattern: Removing pybound legacy factory entry points](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-removing-pybound-legacy-factory-entry-points)
  says to make the smallest change that restores compilation. Removing only the `def_static` does
  restore compilation — and leaves an empty class binding. Removing the class binding restores
  compilation too, and breaks `import ttnn`, because a Python module re-exports the class by name
  two files away. Neither outcome is what the entry describes, and the failure mode is a
  collection-time `AttributeError` in an unrelated conftest, which reads nothing like a port
  error. **Suggested fix:** tell the porter to grep the Python tree for the bound *class* name,
  not just the method, and to expect the re-export chain (`ttnn/ttnn/operations/<family>.py` and
  `ttnn/ttnn/__init__.py`) to be part of the mandatory removal.

- **The `Nodes` key type on `num_runtime_varargs_per_node` is a `std::variant`, and the field is
  a `Table` keyed by it**, so a per-node entry is written `num_runtime_varargs_per_node[Nodes{core}]`
  rather than by bare `NodeCoord`. Minor, but the header's comment ("Each entry pairs a node set
  with its vararg count") does not make the spelling obvious.

- **`AdvancedKernelRunArgs::runtime_varargs` says "length can vary per-node (as declared in
  schema)"** while `KernelAdvancedOptions::num_runtime_varargs` is a scalar. The two read as
  contradictory until you find the deprecated per-node override below them. A cross-reference on
  the `runtime_varargs` comment would resolve it.

## Open items for downstream

### Shared kernel touches

| kernel path | rung taken | remaining unmigrated consumers |
|---|---|---|
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | **rung 1 — reused the existing fork** `writer_unary_interleaved_start_id_metal2.cpp` beside it. No new file created; the legacy original was not touched, and its pointer comment was already in place. | Thirteen factories still bind the legacy copy: `data_movement/concat`, `data_movement/reshape_on_device`, four `data_movement/tilize` factories, `embedding`, `copy/typecast`, `eltwise/unary_backward/tanh_bw`, `experimental/matmul/attn_matmul`, `experimental/transformer/nlp_concat_heads_boltz`, and the two `examples/example` factories. Slice is now off that list. Sunset tracked by issue #52228. |

Slice's own `device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` is **not** shared — no
other op binds it — so it was converted in place with no fork.

The fork carries a note that a second Metal 2.0 fork of the same kernel exists at
`copy/typecast/device/kernels/dataflow/writer_unary_interleaved_start_id_metal2.cpp`, differing
only in its tensor accessor name (`tensor::output` vs `tensor::dst`). Consolidating the two is
still open; this port added a consumer to the eltwise/unary one, which is the copy the fork's own
comment nominates as the survivor.

### Findings — preserved, not fixed

Each of these is legacy behavior the port carried forward unchanged, per the porting invariant.

- **Dead compile-time argument in all four stride kernels.** `compile_time_element_size` is
  declared and never read: `reader_multicore_slice_4d.cpp:81`, `writer_multicore_slice_4d.cpp:65`,
  `reader_multicore_slice_nd.cpp:67`, `writer_multicore_slice_nd.cpp:66`. The value it carries
  also rides a runtime argument (`element_size`), which is the copy the kernels actually use. The
  port kept it as a named CTA the host still emits. Removing it is a two-line change on the ops
  track.
- **Dead runtime arguments in the 4D stride kernels.** `reader_multicore_slice_4d.cpp` reads
  `output_h`, `output_d`, `output_n` and never uses them; `writer_multicore_slice_4d.cpp` reads
  `tensor_rank`, `output_h`, `output_d`, `output_n` and never uses any of them. All are still
  emitted by the host and still declared in the kernels' runtime-arg schema.
- **Dead locals.** `output_bytes_per_row` is computed and never used in `reader_multicore_slice_4d.cpp`
  and `reader_multicore_slice_nd.cpp`; `old_src_tile_id` is assigned and never used in
  `reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp`.
- **Two unreferenced kernel files in the op directory, not ported and not compiled:**
  `device/kernels/dataflow/strided_slice_reader_rm_interleaved_nd.cpp` and
  `device/kernels/dataflow/strided_slice_writer_rm_interleaved.cpp`. No factory in the repository
  instantiates either, and both address DFB index 24, which no slice factory allocates. They are
  the only files left in the op directory still containing `TensorAccessorArgs`, and the only
  reason the op-wide sweep for legacy accessor plumbing is not zero. Deleting them is an ops-team
  call.
- **Stale comments in the device-operation class, deliberately left.**
  `device/slice_device_operation.cpp:171`, `:391-392` and `:415-418` explain parts of the custom
  hash in terms of `TensorAccessorArgs` being baked into writer compile-time args, and of
  addresses that "cannot be refreshed on a hit." Both statements described the legacy binding
  model and no longer describe how the op works. The file is off-limits to the port
  ([Host-side: stay in the lane](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#host-side-stay-in-the-lane)),
  so the comments were not touched. They should be refreshed by the op owner in the same change
  that revisits whether the hash can be narrowed.

### Test coverage notes

- **MeshPartition is untested on this bench** — every test of it needs a multi-device mesh and
  this host is a single-chip N150. The re-wire compiles but has not been run. **This is the one
  thing about this port that still needs a machine it has not had.** The pipelines that cover it,
  and the one that gives the cheapest real coverage (`docs_examples` on an n300), are listed under
  [Handoff points 1](#1-cclmesh_partition-re-wired-to-slices-metal-20-entry-points--out-of-directory-change).
- `models/experimental/ops/descriptors/data_movement/slice.py` has no test that imports it, which
  is why the removed pybind entry point surfaced only by grep and not as a test failure.

### Per-op carry-over

- Any op whose kernels use their runtime-arg buffer as mutable scratch faces the same wall this
  port hit, and `ScratchpadSpec` is the answer for all of them. The legacy signal to grep for is
  `get_arg_addr(` cast to a **non-const** `tt_l1_ptr uint32_t*` — the const / `volatile const`
  spelling is a read-only block and converts to `get_vararg` directly.
- `data_movement/padded_slice` and `experimental/slice_write` are close siblings of this op, share
  its odometer idiom, and will meet both the vararg-write wall and the same
  `writer_unary_interleaved_start_id` fork.
