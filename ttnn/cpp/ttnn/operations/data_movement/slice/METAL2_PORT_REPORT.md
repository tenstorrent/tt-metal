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
validated through the Metal 2.0 spec path. The forcing scaffolding is working-tree only and was
**not** committed.

**CI on the branch has since confirmed the one part that could not be run locally.** Dispatched on
`79a3e229aa3`; full triage below under [CI results](#ci-results). The short version: every
MeshPartition test that runs at all passed, on both t3000 and n300, matching main exactly. PR Gate,
Merge Gate and the models-extended pipeline were green.

**One regression escaped this report's first draft and was caught by CI** — the pybind removal
described under [TTNN ProgramFactory](#device-op-class-edits) was too broad and broke `import ttnn`.
Fixed in a follow-up commit; the corrected account is in
[TTNN ProgramFactory](#device-op-class-edits) and [Friction](#friction).

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

- **Pybind entry points removed:** `SliceTileProgramFactory.create_descriptor` only
  (`slice_nanobind.cpp:167-179`). The factory **class** stays bound, with no methods, and its
  Python re-exports stay — this is the shape layernorm settled when it hit the same point first.
  The first draft of this port removed the class and the re-exports too, which broke `import ttnn`;
  see [Friction](#friction).
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
mechanical. **It has since been run and passed** — see [CI results](#ci-results). The pipelines that
cover it, and which were dispatched on this branch:

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

### 2. Removed pybind surface — `SliceTileProgramFactory.create_descriptor`

**No owner needed; resolved in-branch by following an existing convention.** Retained here because
the first draft of this report got it wrong, and the way it was wrong is the useful part.

`slice_nanobind.cpp:167-179` exposed `SliceTileProgramFactory::create_descriptor` to Python,
returning a `ProgramDescriptor` that the op-descriptor / fusion tooling consumed. The port removes
that method, because the function it binds no longer exists. That much was right.

**What the first draft got wrong, in three ways:**

1. It removed the whole `nb::class_<SliceTileProgramFactory>` binding, not just the `def_static`,
   and with it the two Python re-exports (`ttnn/ttnn/operations/data_movement.py`,
   `ttnn/ttnn/__init__.py`).
2. It claimed the one caller,
   `models/experimental/ops/descriptors/data_movement/slice.py:54`, "is not imported at `ttnn`
   import time … so it does not break the test suite." Both halves are false:
   `models/experimental/ops/descriptors/__init__.py` exports it, and
   `tests/ttnn/unit_tests/operations/fused/parallel_sequential/test_parallel_sequential.py:1158`
   imports and *calls* it.
3. It concluded there was "no drop-in replacement" and routed the problem outward as a handoff.

The consequence was 8 failing sanity jobs across every SKU, each on the same one test, with
`AttributeError: module 'ttnn' has no attribute 'SliceTileProgramFactory'`.

**The convention already existed.** Layernorm was ported before slice and hit this exact wall. Its
resolution, in `tests/ttnn/unit_tests/operations/fused/parallel_sequential/conftest.py` under issue
**#54365**, is three parts:

- the factory class stays bound with **no** `create_descriptor`
  (`nb::class_<LayerNormShardedProgramFactory>(mod, "LayerNormShardedProgramFactory");`),
- its Python re-exports stay,
- an autouse conftest fixture monkeypatches a `pytest.skip` in place of the missing method, so only
  the branches that actually reach the call skip, and the fixture retires itself once a factory
  exposes the method again.

Slice now matches that exactly. The conftest's factory list was generalised from
`_LAYERNORM_FACTORIES` to `_FUSION_FACTORIES` and slice's factory added; it tests `hasattr` rather
than assuming, so it stays a no-op for any factory still on the descriptor API.

**What is genuinely still open, and is not slice's to fix:** the op-fusion infrastructure is built
on `ProgramDescriptor` and cannot consume a `ProgramSpec`. It wires up four ops — `rms_norm`,
`layer_norm`, `matmul`, `slice` — and each one that ports to Metal 2.0 drops out of fusion and gains
a skip. Two of the four are now out. That is issue #54365's subject, and the next port of `rms_norm`
or `matmul` will add the third and fourth. **A porter reaching this point should add their factory
to that conftest list, not open a new handoff.**

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

- **Caution: Porting a shared kernel,
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

- **Caution: Avoid varargs
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
  `ScratchpadSpec` is absent from the whole doc set.** This was the port's one stuck point during
  construction (see [Handoff points 3](#3-metal-20-gap--no-writable-or-addressable-runtime-vararg-accessor));
  the costlier one came later, at the pybind boundary, and is the entry below it.
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

- **The recipe sends the porter looking for a precedent in the wrong places, and this one cost a
  red CI run.** Pattern: *Removing pybound legacy factory entry points* says to "make the smallest
  change that restores compilation." Removing only the `def_static` does that and leaves an
  apparently pointless empty class; removing the class also compiles, and breaks `import ttnn` two
  files away. Nothing in the recipe distinguishes them, so I picked the tidier-looking one and was
  wrong. **The answer was already in the tree**: layernorm had been ported first and had settled
  all three parts of the shape (bare class, kept re-exports, conftest skip) under issue #54365.
  I did not find it, and then compounded that by asserting in this report that slice was "the first"
  of the fusion ops to port — a claim I had not checked and that was false.

  **Suggested fix, and it generalises past this one pattern:** the recipe tells the porter that
  already-ported ops are "skeptically-held reference at most" and warns them off `experimental/quasar/`.
  That is right for *spec shape*, and it reads as blanket discouragement from looking at other ports
  at all. For a **cross-cutting convention** — how a whole subsystem copes with ops leaving the
  descriptor API — an earlier port is not a weak reference, it is the authority, and diverging from
  it silently breaks things. Two concrete additions would have caught this:
  (a) in the pybind-removal pattern, tell the porter to grep the Python tree for the bound **class**
  name as well as the method, and to expect the `ttnn/ttnn/operations/<family>.py` +
  `ttnn/ttnn/__init__.py` re-export chain;
  (b) tell the porter that before writing a Handoff point for a *shared consumer* of the legacy API,
  they must check whether an earlier port already hit the same consumer and left a convention —
  because a handoff that duplicates a solved problem is worse than no handoff.

- **The `Nodes` key type on `num_runtime_varargs_per_node` is a `std::variant`, and the field is
  a `Table` keyed by it**, so a per-node entry is written `num_runtime_varargs_per_node[Nodes{core}]`
  rather than by bare `NodeCoord`. Minor, but the header's comment ("Each entry pairs a node set
  with its vararg count") does not make the spelling obvious.

- **`AdvancedKernelRunArgs::runtime_varargs` says "length can vary per-node (as declared in
  schema)"** while `KernelAdvancedOptions::num_runtime_varargs` is a scalar. The two read as
  contradictory until you find the deprecated per-node override below them. A cross-reference on
  the `runtime_varargs` comment would resolve it.

## CI results

Dispatched on `79a3e229aa3`: sanity, PR gate, merge gate, l2-nightly
(`additional_test_categories: data_movement,ccl,docs_examples,ops_docs_check`, `run_cpp_tests=true`),
pipeline-select `models-extended`, All Model Tests, and T3000 e2e.

| Pipeline | Result | Attributable to this port |
|---|---|---|
| PR Gate | pass | — |
| Merge Gate | pass | — |
| models-extended | pass | — |
| Sanity tests | fail: 9 jobs | **8 jobs — the pybind regression above.** The 9th is an SDPA perf-band assertion on `bh_quietbox_2`, unrelated. |
| L2 nightly | fail: 6 jobs | **none.** See breakdown below. |
| All Model Tests | fail: 31 jobs | **none** — zero hits of the regression signature across sampled jobs; the workflow is red on unrelated branches too. |
| T3000 e2e | fail: 1 job | **none** — byte-identical to main. |

**T3000 e2e** (`t3k_ccl_tests [wh_llmbox]`, the only failing job) is identical to main's run of the
same day: `4 failed, 77 passed, 167 skipped, 2 xfailed, 828 errors` on both, the same four failing
test ids, and the same fabric / Ethernet infrastructure errors behind the 828. Main has failed this
job daily for at least eight days.

**L2 nightly**, job by job: `data_movement [bh_p150b_civ2]` one tilize failure, also on main;
`data_movement [bh_p100]` job timeout, also on main; `docs examples [wh_n300_civ2]`
`1 failed, 83 passed` with the same failing test as main (`test_point_to_point`);
`Galaxy CCL [wh_galaxy]` died at collection in 1.57 s with
`Query mappings failed on device 17: No such device` — no tests ran; `Galaxy MoE [wh_galaxy]` job
timeout. The one delta from main was `data_movement [wh_n300_civ2]`, a single INT32 `permute`
numerical mismatch on a shard that is green on main; `permute` has no reference to slice, and the
job was re-run to settle flake-vs-regression rather than argued away.

**MeshPartition is no longer unverified.** The re-wire ran on real multi-device hardware and passed:

- t3000 (`tests/nightly/t3000/ccl/test_mesh_partition.py`) — all three test functions passed
  (`test_mesh_partition`, `test_mesh_partition_rm`, `test_mesh_partition_tile`), with the same
  3-passed / 46-setup-errored split as main. The 46 are the fabric cascade, erroring in fixture
  setup, not in the test body.
- n300 (`tests/ttnn/docs_examples/test_ccl_examples.py::test_mesh_partition`) — passed, on branch
  and on main.

Adding `docs_examples` to the l2-nightly categories by hand is what produced the n300 coverage; it
is not selected automatically by a diff touching only `operations/`.

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
  (Host-side: stay in the lane),
  so the comments were not touched. They should be refreshed by the op owner in the same change
  that revisits whether the hash can be narrowed.

### Test coverage notes

- **MeshPartition could not be tested on the porting bench** — every test of it needs a
  multi-device mesh and that host is a single-chip N150. This was closed by CI rather than left
  open: the re-wire passed on t3000 and on n300, matching main exactly
  ([CI results](#ci-results)). The cheapest real coverage is `docs_examples` on an n300, which is
  *not* auto-selected by a diff touching only `operations/` and has to be passed by hand to
  l2-nightly — worth knowing for the next port that touches a multi-device consumer.
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
