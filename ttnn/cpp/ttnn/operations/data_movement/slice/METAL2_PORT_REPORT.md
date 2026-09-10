# Metal 2.0 Port Report — `data_movement/slice`

## Outcome

**`PORTED`** — all five factories of `SliceDeviceOperation` converted to
`CustomProgramSpecFactoryConcept`: `SliceRmProgramFactory`, `SliceRmShardedProgramFactory`,
`SliceRmStrideProgramFactory`, `SliceTileProgramFactory`, `SliceTileTensorArgsProgramFactory`.

Nine slice-owned kernels converted in place; the borrowed `eltwise/unary` writer bound to its existing
`_metal2` fork (rung 1, no new fork). Nothing is left on the descriptor API, so
`patch_slice_program_addresses` and `slice_tile_dynamic_args` are deleted outright and
`ccl/mesh_partition` drives one uniform spec path.

**One regression.** `SliceRmShardedProgramFactory` now aborts on a shard grid that does not contain
core `(0, 0)`. The arithmetic responsible is unchanged from before the port; Metal 2.0's argument
validation is what turns it from a silent wrong answer into a stop. Handoff point 4 explains it in
full and sets out why there is no single obvious repair. **The port does not fix it**, on the
recipe's rule that a port preserves behaviour and reports what it finds.

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

**CI found one failure attributable to this change**: `test_llama_decoder_inference`
(Llama-3.3-70B Galaxy, T3000) stops with `Kernel 'reader' is setting runtime_varargs for node 0-0,
but the kernel does not run on that node.` Every other failure in the dispatched pipelines was
traced to an unrelated cause: a package-download outage across six or more shards, a `graph_report`
test already fixed on main by `1e19ef77ee8`, a tilize test that fails identically on main
(run 34317632754, `1 failed, 7506 passed`), and a T3000 ethernet bring-up wedging the box. The
attributable failure is handoff point 4.

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
  `SliceTileProgramFactory`. The bare class and the comment standing in for the removed method are at
  [slice_nanobind.cpp:168-171](slice_nanobind.cpp#L168-L171). See the Handoff-points entry — this is
  user-visible and has live Python callers.
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

### 4. `SliceRmShardedProgramFactory` addresses cores from the origin, and Metal 2.0 now rejects it

**Owner:** `data_movement/slice` owners. **Status:** unfixed, deliberately. Pre-existing arithmetic,
newly fatal. Prior report: issue [#53330](https://github.com/tenstorrent/tt-metal/issues/53330).

Written to stand alone for a reader who has not seen this op before, because the repair decision
belongs to the op's owners rather than to the porter.

#### The problem the code is solving

A Tenstorrent accelerator chip holds a grid of small processors. Each one is addressed by an
`(x, y)` pair, and this text calls one a **core**. A core has its own fast local memory, and it can
also read the local memory of another core on the same chip, given that core's address.

`slice` is the operation that copies a rectangular sub-region out of a tensor, the same thing
`t[2:6, 0:128]` does in PyTorch. The tensors here are stored one row after another, so a piece of a
tensor is a block of consecutive rows.

A tensor can be **sharded**: instead of living in one place, it is cut into pieces and one piece is
placed in the local memory of each of several cores. The description of that arrangement is the
tensor's **shard spec**. Its `grid` field, the set of cores holding a piece, is the one that matters
here.

When both the input and the output of a slice are sharded, each output core has to gather the rows
it needs from the local memory of whichever input cores hold them. So each output core needs its own
list of values telling it which cores to read and which rows to take. That list differs per core,
because each output core owns a different part of the result.

#### How a core gets its own runtime arguments

The program that runs on the chip is built by ordinary software running on the computer that drives
the accelerator, which this text calls the **host**. Part of what the host builds is the **runtime
arguments**: a list of 32-bit numbers handed to a core when the program starts. Two cores running
the same code can behave differently because they can receive different runtime arguments.

The host code that builds this description is called a **program factory**. The one in question is
`SliceRmShardedProgramFactory`. `Rm` is row-major, the row-after-row storage described above, and
`Sharded` means both the input and the output are sharded.

That factory produces one variable-length list of runtime arguments per output core. Each list names
the input cores to read from and, for each of them, which rows to take. The reader program on the
core walks the list ([slice_reader_unary_unpad_dims_rm_sharded.cpp:49-51](device/kernels/dataflow/slice_reader_unary_unpad_dims_rm_sharded.cpp#L49-L51)).

So the factory must answer one question correctly for every core: **which core does this particular
argument list belong to?**

One naming note before the code. This operation calls the input **padded** and the output
**unpadded**, because a slice discards the parts of the input outside the requested region. So
`shard_spec_padded` is the input's shard spec and `shard_spec_unpadded` is the output's
([slice_program_factory_rm_sharded.cpp:243](device/slice_program_factory_rm_sharded.cpp#L243),
[:263](device/slice_program_factory_rm_sharded.cpp#L263)).

#### Two summaries of a set of cores

A shard grid is a `CoreRangeSet`: a set of rectangles of cores. It can hold one rectangle or
several, and a rectangle can start anywhere on the chip, not only at the origin `(0, 0)`.

A `CoreRangeSet` offers two summaries, and the difference between them is the whole problem:

- `num_cores()`: how many cores are in the set.
- `bounding_box()`: the smallest single rectangle containing all of them, reported as a
  `start_coord` and an `end_coord`.

For a set that is one rectangle starting at `(0, 0)`, `num_cores()` equals the width times the
height of the bounding box, and counting `0, 1, 2, ...` across that box one grid row at a time
visits exactly the cores in the set. For any other set, both of those break.

#### The defect

The factory derives its grid dimensions like this
([slice_program_factory_rm_sharded.cpp:267-272](device/slice_program_factory_rm_sharded.cpp#L267-L272)):

```cpp
auto& all_cores_unpadded = shard_spec_unpadded.grid;
uint32_t num_cores_unpadded = shard_spec_unpadded.num_cores();
auto bbox_unpadded = all_cores_unpadded.bounding_box();
CoreCoord grid_size_unpadded = {bbox_unpadded.end_coord.x + 1, bbox_unpadded.end_coord.y + 1};
uint32_t num_cores_x_unpadded = grid_size_unpadded.x;
uint32_t num_cores_y_unpadded = grid_size_unpadded.y;
```

It reads `end_coord` and never reads `start_coord`. Adding 1 to `end_coord` gives a width and height
only if the rectangle starts at `(0, 0)`.

It then loops `i` from `0` to `num_cores_unpadded - 1`, turning each index into a core position by
dividing and taking a remainder
([slice_program_factory_rm_sharded.cpp:384-390](device/slice_program_factory_rm_sharded.cpp#L384-L390)):

```cpp
for (uint32_t i = 0; i < num_cores_unpadded; ++i) {
    CoreCoord core;
    if (row_major) {
        core = {i % num_cores_x_unpadded, i / num_cores_x_unpadded};
    } else {
        core = {i / num_cores_y_unpadded, i % num_cores_y_unpadded};
    }
    ...
}
```

The `row_major` flag here is not about tensor storage, despite the `Rm` in the factory name. It is
the shard spec's orientation
([:265](device/slice_program_factory_rm_sharded.cpp#L265)),
and it selects whether cores are visited across a grid row or down a grid column. Both branches
count from `(0, 0)`, so both carry the same defect with the axes swapped. Both examples below take
the first branch.

The walk therefore fills a rectangle of the derived width, one grid row of cores at a time, starting
at the origin. So the mapping is right only when the shard grid is one rectangle whose corner is
`(0, 0)`.

##### The same assumption appears in two live places

One fact about the code that every candidate repair has to reckon with: the origin assumption is in
two parts of the factory, not one.

1. **The output grid dimensions**
   ([:267-272](device/slice_program_factory_rm_sharded.cpp#L267-L272)),
   quoted above. These feed the walk at
   [:384-390](device/slice_program_factory_rm_sharded.cpp#L384-L390)
   that decides which core each list goes to. This is the site the error message is about.
2. **The input grid dimensions**
   ([:248-251](device/slice_program_factory_rm_sharded.cpp#L248-L251)),
   derived the same way from the input's bounding box. These decide the peer addresses written
   *into* the lists, in two spots: a shard number is turned into an input core position by the same
   divide-and-remainder from `(0, 0)`
   ([:154-160](device/slice_program_factory_rm_sharded.cpp#L154-L160)),
   and a guard then drops any peer landing outside the derived rectangle
   ([:161](device/slice_program_factory_rm_sharded.cpp#L161)).
   So on an input grid that is not origin-anchored, some source rows are never read, and the output
   rows that should have held them keep whatever was already in memory.

Correcting site 1 alone would pair right destinations with peer addresses still computed from the
old assumption, which would move the wrong answer rather than remove it: the error would stop and
the rows would still be gathered from the wrong places. In the failing case both grids are away
from the origin, as the next section shows, so both sites have to change together.

There is a third copy of the site 1 arithmetic, inside the helper that builds the lists
([:106-112](device/slice_program_factory_rm_sharded.cpp#L106-L112)).
It is dead: the `core` variable it computes is never read afterwards, in that helper or anywhere
else. It needs deleting rather than correcting, and it is worth mentioning only because anyone
grepping for the arithmetic will find it and wonder which of the copies matter.

##### A small illustration

Take a deliberately simple case first. Suppose the output is sharded across four cores forming the
rectangle from `(2, 0)` to `(5, 0)`: cores `(2,0)`, `(3,0)`, `(4,0)`, `(5,0)`.

- `num_cores()` is 4, so the loop runs `i = 0, 1, 2, 3`.
- `end_coord` is `(5, 0)`, so the derived width is 6 and the derived height is 1.
- The walk produces `(0,0)`, `(1,0)`, `(2,0)`, `(3,0)`.

Every list is shifted two positions to the left. Cores `(0,0)` and `(1,0)` are handed lists although
no program runs on them; cores `(2,0)` and `(3,0)` get the lists meant for the first and second
shard instead of the third and fourth; cores `(4,0)` and `(5,0)` run the reader with no list at all,
so they read whatever values happen to be sitting in their argument memory and treat them as core
addresses and row counts.

The correct answer is the four cores of the set, in order: `(2,0)`, `(3,0)`, `(4,0)`, `(5,0)`.

##### The case that actually fails

The real grid is worse than the illustration, and the difference decides which repair can work.

The failing test is `test_llama_decoder_inference` in
[test_llama_decoder.py](../../../../../../models/demos/llama3_70b_galaxy/tests/test_llama_decoder.py), which takes its
configuration from [model_config.py](../../../../../../models/demos/llama3_70b_galaxy/tt/model_config.py). That
configuration builds one set of cores and uses it in several places
([model_config.py:2987-2992](../../../../../../models/demos/llama3_70b_galaxy/tt/model_config.py#L2987-L2992)):

```python
sub_core_grids = ttnn.CoreRangeSet(
    [
        ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
        ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
    ]
)
```

That set is the shard grid of the memory configuration the failing slices write their output to
([model_config.py:3014-3024](../../../../../../models/demos/llama3_70b_galaxy/tt/model_config.py#L3014-L3024)). The
configuration is height-sharded, meaning the tensor is cut into blocks of whole rows, which is what
this factory requires.

That is **two** rectangles, holding columns 1 to 3 and columns 5 to 6. Column 0 and column 4 are
deliberately left out. A `CoreRange` includes both of its corners, so the corner `(3, 9)` means grid
rows 0 to 9, ten of them. So:

- `num_cores()` is 3 x 10 + 2 x 10 = 50.
- `bounding_box()` is `(1,0)` to `(6,9)`, so the derived width is 7 and the derived height is 10.
  That is a 70-core rectangle standing in for 50 cores.
- The walk emits `{i % 7, i / 7}` for `i` from 0 to 49. It starts at `(0,0)` and covers columns 0
  and 4 on grid rows 0 to 6, none of which is in the grid. Fifty indices at a width of 7 also run
  out part way through grid row 7, so the walk never reaches grid rows 8 and 9. Counting up, **15
  of the 50 grid cores are handed no list at all**: the five on each of grid rows 7, 8 and 9. The
  derived height of 10 plays no part in this branch.

The **input** to those slices is also away from the origin, so site 2 is wrong here as well. The
input is a rotation matrix built in
[llama_rope.py:78-80](../../../../../../models/demos/llama3_70b_galaxy/tt/llama_rope.py#L78-L80), which places its
shards inside the same two rectangles starting from core `(1, 0)`, and it is height-sharded at
[llama_rope.py:272-278](../../../../../../models/demos/llama3_70b_galaxy/tt/llama_rope.py#L272-L278). Its bounding box
therefore does not begin at `(0, 0)` either.

The failing call is one of four
([llama_attention.py:818-845](../../../../../../models/demos/llama3_70b_galaxy/tt/llama_attention.py#L818-L845)),
which slice the rotation matrices for the query and key halves of attention. Which of the four
reports the error first does not matter: all four pass the same output memory configuration, and
their inputs are the two rotation matrices built together by the same code, so all four see the
same pair of grids.

#### Why the defect stayed hidden until the migration

This arithmetic is not new. It was carried across unchanged when the operation was moved to a new
host interface for describing programs, `metal2_host_api`.

The old interface accepted the arguments and applied them. A list aimed at a core with no program on
it went nowhere, and nothing was reported.

The new interface validates the description before running it. It collects the set of cores each
program actually runs on, and checks every argument list against that set. It calls a core a
**node**, and it calls a variable-length argument list **runtime varargs**
([program_run_args.cpp:241-246](../../../../../../tt_metal/impl/metal2_host_api/program_run_args.cpp#L241-L246)):

```cpp
TT_FATAL(
    kernel_nodes.contains(node_coord),
    "Kernel '{}' is setting runtime_varargs for node {}, but the kernel does not run on that node.",
    kernel_name,
    node_coord.str());
```

So the operation now stops with an error where it used to continue. `node 0-0` in the message is
core `(0, 0)`:

```
Kernel 'reader' is setting runtime_varargs for node 0-0, but the kernel does not run on that node.
```

The migration did not create the wrong mapping; it made it impossible to ignore. On any layout that
triggers it, the old code already sent rows to the wrong cores and reported nothing. Stopping with
an error is better than producing wrong data silently, but it is still a change in what the
operation does, and it is why `test_llama_decoder_inference` now fails.

Those calls can also carry a discarded request for a specific group of cores, the
`sub_core_grids` argument, which the operation drops with a warning
([:253-255](device/slice_program_factory_rm_sharded.cpp#L253-L255))
before using the tensor's own shard grid instead. Candidate 3 below returns to it.

#### Why the repair is not one obvious edit

The error names a single kernel and a single core, which makes the repair look like a small edit. It
is not. Four repairs are possible, and they differ in what they treat as the real defect: an
arithmetic slip, a missing use of an existing helper, a discarded parameter, or an undocumented
restriction. Two of the four turn out not to apply to this grid.

**Candidate 1: honour `start_coord`.** Compute the width from `end_coord.x - start_coord.x + 1` and
add `start_coord` to each generated position, at both of the sites described earlier. This is the
smallest edit. It is correct for a shard grid that is a single rectangle anywhere on the chip, and
still wrong for a grid made of several rectangles, because a divide-and-remainder walk can only
produce a rectangle.

**It does not fix the failing case.** That grid is two rectangles with column 4 left out between
them. Honouring `start_coord = (1, 0)` gives a width of 6 and starts the walk at `(1,0)`, so the
walk then covers columns 1 to 6. Column 4 is still in that span and still not in the grid, so the
same check fails on `(4,0)` instead of on `(0,0)`, and as before the walk runs out before the last
grid row. This candidate only ever produces a rectangle, and the grid is not one.

**Candidate 2: enumerate the set directly.** The codebase already has `corerange_to_cores`, which
returns the cores of a `CoreRangeSet` in a defined order: each rectangle walked one grid row at a
time, rectangles in the order the set holds them. Replacing the arithmetic with that call is correct
for any grid layout. But it changes which core gets which argument list for any grid where the
bounding-box walk and the actual set of cores disagree, and those are exactly the cases that are
wrong today. That is the intent, and it is also the risk: nobody has characterised what the
operation produced on those layouts before the migration, so there is no known-good result to
compare against. It needs numerical checks on real sharded cases, not only a successful build.

Like candidate 1, it has to be applied at both sites described earlier.

**Candidate 3: stop discarding the caller's requested core group.** The operation throws away the
`sub_core_grids` placement with the warning quoted above. The four calls pass that argument when the
model runs with its prefetcher resident, and pass nothing otherwise
([llama_attention.py:817](../../../../../../models/demos/llama3_70b_galaxy/tt/llama_attention.py#L817)). I did not
check which of the two the failing run took. When the argument is passed, the group is the same
two-rectangle set as the output shard grid
([model_config.py:616-621](../../../../../../models/demos/llama3_70b_galaxy/tt/model_config.py#L616-L621)). A comment
just above those calls
([llama_attention.py:812-816](../../../../../../models/demos/llama3_70b_galaxy/tt/llama_attention.py#L812-L816))
says the argument is there to keep slice off cores the model reserves for other work, and describes
slice as otherwise packing its cores from the origin. So the caller has met origin-anchored packing
before and written a workaround the operation ignores.

The comment does not describe the failure examined here, and it should not be read as if it did. It
speaks of a 32-core column-major packing. The grid above holds 50 cores and is row-major, so it
takes the other branch of the walk. My guess is that the comment describes a different code path
inside slice, one that picks its own cores when the output is not already sharded. The comment is
evidence that the argument is meant to be honoured, not evidence about the failing case.

Honouring the argument would not fix the arithmetic on its own. The requested group is the same
two-rectangle set as the shard grid, so a divide-and-remainder walk over it fails the same way; this candidate only
helps combined with candidate 2. It also changes which cores the operation runs on, and so its speed
and its memory use. It belongs on the list because a caller evidently expects the argument to mean
something, which is a question about the operation's contract rather than about this defect.

**Candidate 4: reject the layout.** Add a check that the shard grid is a single origin-anchored
rectangle and fail clearly if it is not. This makes the limit explicit, and it turns a
currently-silent wrong answer into a refusal. It also turns away a layout a caller is already
passing in, so it needs someone who knows the callers to agree.

For this grid the field narrows. Candidate 1 cannot express a two-rectangle set, so it is out.
Candidate 4 would turn away a layout a shipping model already passes, so it needs the model owners
to agree first. That leaves **candidate 2 as the only one that repairs the failing case**, applied
at both sites, with candidate 3 as a separate question about whether the operation should honour a
placement argument it currently discards.

What is still open is not which mechanism to use but what the operation promises: whether a
non-rectangular shard grid is supported, and whether `sub_core_grids` means anything here. Those are
questions for whoever owns the operation, and the answers decide whether candidate 3 ships with
candidate 2 or separately.

What should *not* happen is either silencing the check, or widening the kernel's placement to the
whole bounding box so that the existing walk stops complaining. Widening the placement is
especially tempting, because it makes the error go away with one line. It would also place the
kernel on column 0 and column 4, which the model excluded on purpose, and it would bring back the
old behaviour of writing rows to the wrong cores without reporting anything. That is worse than
stopping with an error.

#### This was already reported, from the other side

Issue [#53330](https://github.com/tenstorrent/tt-metal/issues/53330) (open, filed 2026-08-17) is the
same defect seen before Metal 2.0 gave it a name. On Blackhole Galaxy the watcher stops the device
with `NCRISC accessed unique runtime arg index out of bounds` in
[slice_reader_unary_unpad_dims_rm_sharded.cpp](device/kernels/dataflow/slice_reader_unary_unpad_dims_rm_sharded.cpp),
which is what a starved core does: it runs the reader with no list and indexes past the arguments it
was given.

The core the watcher names matches. That leg is Qwen3-32B, where the same two rectangles are capped
to grid rows 0 to 7
([qwen_model_config.py:242-243](../../../../../../models/demos/llama3_70b_galaxy/tt/qwen_model_config.py#L242-L243)),
giving 40 cores instead of 50:

| grid | cores | derived width | starved | first starved |
|---|---|---|---|---|
| grid rows 0-9 | 50 | 7 | 15 | `(1,7)` |
| grid rows 0-7 | 40 | 7 | 12 | **`(5,5)`** |

`(5,5)` is the lowest-indexed grid core the walk never reaches, and `(5,5)` is the core the watcher
reports. The issue does not identify the cause, but its last action item, "Verify all cores in the
shard grid receive the full unique-arg vector (no short-populated core)", is this defect stated as a
symptom.

Two other open issues are the same class of defect in neighbouring code, and neither covers this one:
[#51442](https://github.com/tenstorrent/tt-metal/issues/51442), `ttnn.concat` accepting
`sub_core_grids` and silently dropping it, which is candidate 3's question asked about another op;
and [#51214](https://github.com/tenstorrent/tt-metal/issues/51214) item 5, a reversed Horner loop in
[reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp](device/kernels/dataflow/reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp),
a slice kernel this port converted. That one is unrelated to the grid walk and is still unfixed here:
the port preserved it, as the "preserve the bugs" rule requires.

#### Sequencing

The fix can land **before** the port, and probably should. The arithmetic at all three sites is
identical at the merge-base, so the repair applies unchanged to the legacy factory, and doing it
first keeps this port what it is meant to be, a syntax conversion with no behaviour change. The catch
is that the defect is silent before the port, so the repair needs a numerical test on a sharded grid
that is not anchored at `(0, 0)` written first, or nothing demonstrates it.

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
- **The `SliceRmShardedProgramFactory` grid walk is unfixed.** Handoff point 4. This is the one
  behaviour change in the port and the only CI failure attributable to it. It needs a decision from
  the op's owners on what the operation promises for a shard grid that is not a single
  origin-anchored rectangle, and it needs a numerical test on such a grid, which does not exist
  today. Issue [#53330](https://github.com/tenstorrent/tt-metal/issues/53330) already tracks the
  symptom and should be linked to whatever fixes it.
- **Sweeps were not run.** The confirmed baseline was the primary and nightly unit tests; the three
  `tests/sweep_framework/sweeps/data_movement/slice/` files are CI-run and were left to CI.
- The audit's misc anomalies are carried forward unchanged. Two are now easier to retire than before:
  `compile_time_element_size` survives as a named compile-time argument the stride kernels still never
  read, and it no longer has to occupy a slot to keep a `TensorAccessorArgs` offset correct, so it is a
  one-line deletion for whoever owns that cleanup. The dead `writer_kernel_args` half of the sharded
  helper's return pair is likewise now trivially removable.
