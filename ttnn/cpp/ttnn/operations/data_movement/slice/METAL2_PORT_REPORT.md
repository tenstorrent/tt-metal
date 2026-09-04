# Metal 2.0 Port Report — `data_movement/slice`

*(Opened at the start of the port; friction captured as it happened, polished at the end.)*

## Outcome

**`PORTED`** — all five factories of `SliceDeviceOperation` converted to
`CustomProgramSpecFactoryConcept`: `SliceRm`, `SliceRmSharded`, `SliceRmStride`, `SliceTile`,
`SliceTileTensorArgs`. Ten slice-owned kernels converted in place; the cross-family donor reused via
its existing `_metal2` fork (rung 1). Nothing left on the descriptor API, so
`patch_slice_program_addresses` and `slice_tile_dynamic_args` are deleted outright.

### Verification

Metal 2.0 legality checks forced on at all nine `skip_validation` sites and **proven live** — 9254
`METAL2_CHECKS_FORCED` markers in the final run — and Watcher on (`TT_METAL_WATCHER=10`) for every
run. The final numbers were measured on the committed tree, with the forcing re-applied for the
measurement and reverted afterwards; the diff contains none of that scaffolding.

| Suite | Pre-port baseline | Post-port |
|---|---|---|
| `unit_tests/operations/data_movement/test_slice.py` | 445 passed, 38 skipped *(on the original base)* | **448 passed, 38 skipped** |
| `nightly/…/test_slice_for_conv.py` + `test_universal_input_tm_slice.py` | not separately baselined | **321 passed, 4 skipped, 0 failed** |
| C++ gtests | — | none exist for this op (see *Open items*) |

The `test_slice.py` count moved 445 → 448 across the rebase described below, and the three are the
newer base's, not this port's: the branch this landed on carries ops-team preallocated-output fixes
that the original base lacked. 448 / 38 is also exactly what the audit recorded for PR #55433 on a
branch carrying those same fixes, which is a useful independent check that this tree is the state that
branch describes.

Anti-pattern self-audit, over the op directory (**29 `.cpp`/`.hpp`/`.h` files scanned**, and 25 files
in the change set for the diff-scoped checks) — every check zero over a non-zero denominator:

| Check | Result |
|---|---|
| buffer address in run args (`->address()`, `emplace_runtime_args`, `Buffer*`) | 0 |
| magic CB indices / `CBIndex::` in CTAs | 0 |
| `TensorAccessorArgs<N>()` in ported kernels | 0 (4 hits total: 2 in the *unreferenced* kernels, 2 in comments in the off-limits device-op class) |
| `cb`-shaped names (`[Cc][Bb]_`, `\bCB[A-Z]`, …) | 0 |
| `.id` extraction at LLK call sites | 0 |
| `allow_instance_multi_binding` | 0 |
| positional `compile_time_args = {…}` | 0 |
| `opt_level` | 0 — correct here: the op has **no compute kernel at all**, so every kernel's legacy resolved level is `O2`, which is also Metal 2.0's default. No `opt_level` line is owed. |
| `hw_config` fidelity | all ten DM kernels resolve to the reader/writer defaults and use `create_reader_datamovement_config` / `create_writer_datamovement_config`; no custom `(processor, noc, noc_mode)` existed to reproduce |
| forced-legality scaffolding in the diff | 0 files under `tt_metal/`, 0 marker hits |
| ephemeral `.md` cited from code | 0 |
| `TT_FATAL` / `TT_ASSERT` / `TT_THROW` census | one intended **+1** (`check_accessor_page_size`, mandated by the brief). One guard was lost in a first pass and **restored** (`TT_ASSERT(output.buffer() != nullptr)` in the tile factory) — the census is what caught it. |

The device-operation class (`slice_device_operation.cpp`) and the op's top-level `slice.cpp` are
**byte-identical** to pre-port.

**One capability lost outside the op, with no test left red:** the brief-mandated removal of the
pybound `SliceTileProgramFactory.create_descriptor` takes slice out of the `OpDescriptor` path
(`models/experimental/`), which the port cannot repair — that path needs a `ProgramDescriptor` object
and a ported factory has none. Six tests are converted from failures to skips by the same guard
layernorm's port uses, under issue #54365 — `0 failed, 50 passed, 75 skipped`. See
*Handoff points* #1.

**Bench caveat:** this machine's multi-chip ethernet topology defeats UMD's cluster discovery, so all
runs used `TT_VISIBLE_DEVICES` to restrict to a single Wormhole card. That matches the
single-device scope of the op's tests, but `ccl/mesh_partition` — which this port had to change —
could not be exercised.

## Provenance

`git log -1 --format='%h %cs %s' -- docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/`
prints **nothing** in this checkout: the recipe docs are not in this tree at all. They were consumed
out-of-tree from `/localdev/edwinlee/metal2_port.md`, so the recipe version cannot be pinned by hash.

- **Recipe docs (this port):** *unpinnable* — `/localdev/edwinlee/metal2_port.md`, 1200 lines,
  consumed 2026-09-04. Its five `../shared/*.md` references (`port_patterns.md`,
  `migration_guide.md`, `ttnn_factory.md`, `cb_dfb_api_whitelist.md`, `workspace_setup.md`) do not
  exist in this tree or beside the recipe; the port worked from the declaring headers under
  `tt_metal/api/tt-metalium/experimental/metal2_host_api/` instead.
- **Audit docs (inherited):** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a
  category, not as slice's current state`
- **Port base commit:** `6ebddf3088a` — the tip of `origin/edwinlee/Port_Slice`. Merge-base with
  `origin/main` is `2a8253ad20a`.

  *The port was originally written against a local `edwinlee/Port_Slice` at `d1f66c276f2`, which had
  diverged from the remote branch, and was rebased onto the remote tip before landing. The rebase
  matters to two claims below and is described under* Rebase onto the live branch *at the end of this
  report.*

## TTNN ProgramFactory

- **Concept realized:** `CustomProgramSpecFactoryConcept` on **all five** factories, as the audit
  chose. Each now defines `create_program_artifacts` returning `ProgramArtifacts` and
  `override_runtime_arguments` returning `ProgramRunArgs`; `create_descriptor` is gone from all five.
- **Op-owned tensors:** none — `ProgramArtifacts::op_owned_tensors` is omitted throughout.
- **Custom `compute_program_hash`: left intact**, `device/slice_device_operation.cpp:343`. Not
  rewritten, not trimmed, not re-derived. No `TensorSpec` legality failure appeared on any
  second-or-later invocation, so the ops team's pre-port verdict on the hash holds.
- **Pybind entry points removed:** one — `SliceTileProgramFactory.create_descriptor`, plus its two
  Python re-exports. See *Handoff points* #1; it has live callers the port does not fix.
- **Device-op-class edits the port forced:** none of the three documented exceptions applied. The
  device-operation class itself (`slice_device_operation.cpp`) is **byte-identical**; the only edit to
  `slice_device_operation.hpp` is the deletion of the `patch_slice_program_addresses` declaration,
  whose definition the port removes.
- **What collapsed.** `patch_slice_program_addresses`
  (formerly `slice_program_factory_rm_sharded.cpp:354-413`) and `slice_tile_dynamic_args`
  (formerly `slice_program_factory_tile.cpp:198-281`) are **both deleted**. Between them they held
  three separate refresh mechanisms — `apply_descriptor_runtime_args` over a CB-address-only
  descriptor, a positional `GetRuntimeArgs` slot-0 rewrite, and an `apply_dynamic_runtime_args`
  vector — and all three were doing the same thing the typed bindings now do. What genuinely remained
  (the tile factories' per-node scalar re-emission for #52651) is a dozen lines inside two
  `override_runtime_arguments`, sharing one work-split helper with the cache-miss path so the two
  cannot drift.
- **Open items with the concept:** the `CustomProgramSpecMeshWorkloadFactoryAdapter` cache-hit path
  (`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:963-983`) applies **only** what the override
  returns — it does not also run the base adapter's `UpdateTensorArgs`. Three of slice's five
  factories need no per-dispatch scalars at all and exist on this concept purely to re-supply their
  tensor bindings. That is correct but easy to get wrong (omit a binding and its address silently
  freezes at the cache-miss value); a base-concept variant that still refreshed tensors would suit
  those three better.

## Handoff points

### 1. Removed pybind surface — `SliceTileProgramFactory.create_descriptor`
*Tagged: API surface: removed entry point. Owner: the descriptor-fusion team, under issue #54365.*

The port replaces `create_descriptor` with `create_program_artifacts` on all five factories, so the
`def_static` that bound it (`slice_nanobind.cpp:167-179`) had to go —
`ProgramDescriptorFactoryConcept` and `ProgramSpecFactoryConcept` are mutually exclusive by
construction (`ttnn/api/ttnn/operation_concepts.hpp:120-136`), so the method cannot stay on the
factory at all.

**The type itself stays bound.** `slice_nanobind.cpp` keeps an empty
`nb::class_<SliceTileProgramFactory>` and both Python re-exports
(`ttnn/ttnn/operations/data_movement.py:550`, `ttnn/ttnn/__init__.py:639`). That is deliberate and
mirrors `layernorm_nanobind.cpp:339`, which is exactly the same shape: a factory that ported to Metal
2.0, still bound, with no methods. It costs nothing and it means a caller gets a clear
missing-*method* error rather than a missing-*symbol* one.

**What genuinely cannot be preserved.** The consumer does not merely *call* `create_descriptor`; it
needs the returned `ProgramDescriptor` as an object. `OpDescriptor.launch()` hands it straight to
`ttnn.generic_op` (`op_descriptor.py:334-338`); a *fused* tree goes further and consumes it
structurally — `fusion.py:805` iterates `op.descriptor.kernels` to group them by RISC type,
`op_descriptor.py:227` hashes it, and codegen emits a merged `ProgramDescriptor`. A ported factory
produces a `ProgramSpec`, which neither path can consume. No binding shape recovers this.

**Scope — wider than "fusion", but narrower than it first looks.** The entry point is
`models/experimental/ops/descriptors/data_movement/slice.py:54`, and *both* its uses break: standalone
`OpDescriptor.launch()`, and membership in a `Sequential` / `Parallel` tree (a "branch", in this
infrastructure's terms, is one `OpDescriptor` inside such a tree — `fusion.py:441-458`). Against
that: `Sequential` / `Parallel` are **already disabled by default** — `fusion.py:85-89` raises unless
`TT_METAL_ENABLE_PARALLEL_SEQUENTIAL=1`, for the same *"until ProgramSpec is exposed to Python"*
reason that #54365 tracks — and everything affected sits under `models/experimental/`. `ttnn.slice()`
itself is untouched; it runs the spec path, which is what the 445 passing tests exercise.

**Resolution — the layernorm precedent, applied.** The fusion suite already handles exactly this: an
autouse fixture in `tests/ttnn/unit_tests/operations/fused/parallel_sequential/conftest.py` stands a
`pytest.skip` in for the missing `create_descriptor`, skipping only the tests that actually reach the
call and self-retiring once a factory exposes one again. Slice is the second op through that door, so
it is added to the fixture's tuple (renamed `_LAYERNORM_FACTORIES` → `_BRANCH_FACTORIES`, and the
skip message generalized, since it is no longer layernorm-specific).

| Suite | Before the guard | After |
|---|---|---|
| `…/parallel_sequential/test_parallel_sequential.py` + `…/demo/test_fused_demo.py` | 6 failed, 50 passed, 69 skipped | **0 failed, 50 passed, 75 skipped** |

Both figures above were measured with Watcher **off**, so they compare like for like. With
`TT_METAL_WATCHER=10` on, a further 20 tests in `test_fused_demo.py` skip on a pre-existing condition
at `test_fused_demo.py:2392` (*"pytest-timeout plugin interacts with watcher on device reopen"*) —
unrelated to this port, but worth knowing before comparing a watcher-on run against these numbers.
Re-run post-rebase, watcher on: `test_parallel_sequential.py` 21 passed / 68 skipped / 0 failed, and
`test_fused_demo.py` 9 passed / 27 skipped / 0 failed.

The six that convert to skips are `TestBranchingTopology::test_three_way_split_with_slice`,
`TestBranchingTopology::test_nested_split_with_slice`,
`TestParallelExecution::test_two_disjoint_trees_parallel`,
`TestDocExample::test_matmul_slice_ln_rms_tree`,
`TestAsymmetricBarrier::test_narrow_wide_with_slice`, and
`TestPerfDemos::test_sharded_tree_ln_slice_matmul_slice_ln_fused[perf_mode=none]`.

**Two notes for the owners.** The conftest edit is *outside the op directory* — sanctioned here only
because it is the established mechanism for this exact event and issue #54365 already owns it; it
should be reviewed by the fusion team rather than waved through with the port. And the coupling runs
both ways: slice keeps its **own** copy of the unary writer specifically so the fusion infrastructure
can remap the DFB index, so re-enabling these branches is worth real consideration rather than
indefinite skipping.

### 2. `ccl/mesh_partition` — out-of-op-directory edit, and it is now unconditional
*Tagged: cross-op host coupling. Owner: ops team / TTNN.*

`ttnn/cpp/ttnn/operations/ccl/mesh_partition/device/mesh_partition_program_factory.cpp` drives slice's
factories directly and had to move with them, at both call sites (`create_at` and
`override_runtime_arguments`). Three things to carry forward:

- The change is **outside the porter's writeable surface**. It is taken here on the audit's record
  that option (b) was chosen and *"explicitly authorized by the invoker"* (*Questions* #3) — the
  branch that implemented it, PR #55433 `8c8b9eea947`, is **not in this tree**, so it was
  re-implemented rather than inherited.
- Because all five factories converted in this change, the `IsSliceSpecFactory` concept that made the
  bridge incremental is **already retired**: both sites now go unconditionally through
  `MakeProgramFromSpec` + `SetProgramRunArgs` and `UpdateProgramRunArgs`. That is the intended end
  state, one step earlier than expected.
- It remains **not run-verified**: MeshPartition's tests are t3000/TG-only and this bench is a single
  Wormhole card. Unchanged from the audit's caveat.

### 3. `get_vararg()` has no address form, and three readers write back into their vararg block
*Tagged: API: missing accessor. Owner: the Metal 2.0 API team.*

`tt_metal/jit_build/genfiles.cpp:441` emits only `get_vararg(idx)` / `get_common_vararg(idx)` — value
getters. Three slice readers advance a host-seeded `id_per_dim` block in place, so the block must be
copied into a kernel local and mutated there:

- `reader_unary_unpad_dims_interleaved_start_id.cpp:26` and
  `reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp:31` — `num_dims` is a CTA, so the
  local is exactly sized.
- `slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp:44` — `num_dims` is a **runtime** arg, so
  no exactly-sized local is possible. Sized by `tensor_accessor::MAX_RANK`
  (`tt_metal/hw/inc/internal/tensor/const.h:11`) with a device `ASSERT`. Promoting `num_dims` to a CTA
  would have fixed it but changes compile/cache behaviour, which is not port work.

`data_movement/pad` hit the same wall and resolved it the same way
(`pad/device/kernels/dataflow/reader_pad_tiled.cpp:22-32`), so this is now two independent ports
reporting it. A `get_vararg_addr(i)` — or the planned typed `std::array` arguments — removes the
workaround and the `MAX_RANK` bound with it. This is the **only** place in this port where a value
changes *where it lives* rather than how it is spelled; it is behaviour-identical because nothing
reads the block back from L1 after the kernel exits.

## Successes

- **The recipe's insistence on forcing the legality checks paid for itself immediately.** The op's own
  runtime config reports `validate_program_args=false` (visible in every `ttnn.CONFIG` dump in the
  test logs), so on this bench the cache-hit validator would have been off for the whole port. Forcing
  all nine `skip_validation` sites and proving it with the two `METAL2_CHECKS_FORCED` markers — both
  present in every test run — is the difference between a verified port and a false green. *Recipe:
  "Ensure the Metal 2.0 host-side legality checks are enabled."*

- **"Re-derive the endpoint dispositions from the census, don't transcribe them" held up.** The brief
  listed three self-loops; re-deriving each from the kernel-touch census reached the same three, and
  more usefully it forced reading *why* `in_shard` is sync-free
  (`slice_reader_unary_unpad_dims_rm_sharded.cpp:41` is a bare `get_write_ptr()` with no FIFO ops).
  That is the same fact that makes the borrowed binding load-bearing for correctness, not just for
  addressing — which is now written down at the DFB spec
  (`slice_program_factory_rm_sharded.cpp:296-305`) instead of being invisible at the call site.

- **The `Table`-is-not-a-vector warning fired correctly.** `runtime_arg_values` is keyed name-first,
  then node, and every legacy loop here is node-first. `AddRuntimeArgsForNode` let all five factories
  keep their existing per-core loops verbatim; inverting them by hand across ~40 arguments would have
  been the single most error-prone edit in the port. *Recipe: `KernelSpec` ↔ `KernelRunArgs`.*

- **The scope-discipline rule caught a real temptation.** The tile factories diverge between their
  cache-miss and cache-hit paths for a no-op node's writer `start_id` (0 vs. the running tile count) —
  inert, since `num_pages` is 0 there. It reads like a bug and is a one-line "fix". Preserved instead,
  and written up below. Same for the dead `compile_time_element_size` CTA in four stride kernels and
  the dead `old_src_tile_id` local, both of which the port carries forward unchanged.

## Friction

*(running notes)*

- **Gap — the recipe's reference docs are not in this checkout.** `metal2_port.md` links
  `../shared/port_patterns.md`, `../shared/migration_guide.md`, `../shared/ttnn_factory.md`,
  `../shared/cb_dfb_api_whitelist.md` and `../shared/workspace_setup.md`; none exists in this tree or
  beside the recipe at `/localdev/edwinlee/`. Worked from the declaring headers instead (which the
  recipe itself recommends as the stronger reflex).

- **Gap — the environment `workspace_setup.md` assumes was not present.** No `build_Release`, no
  `python_env`, submodules uninitialized, and neither `clang-20` nor `g++-12` installed (no sudo).
  `./build_metal.sh` fails at cmake configure. Resolved by building and testing inside the project's
  own CI image, `ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-22.04-dev-amd64`.

- **Gap — UMD cannot build a cluster descriptor on this bench, and nothing in the recipe covers it.**
  Every device open (host *and* container, so not a Docker artifact) threw a bare
  `std::out_of_range("unordered_map::at")` from
  `tt::umd::TopologyDiscovery::fill_cluster_descriptor_info()` — no TT_FATAL, no message, no hint.
  The eight Wormhole cards are individually healthy; it is the multi-chip ethernet map that fails.
  `TT_VISIBLE_DEVICES=<n>` restricts discovery to one card and everything works, which is what the
  whole verification run used. Worth a line in `workspace_setup.md`: *a device-open failure that
  surfaces as `unordered_map::at` is a cluster-topology problem — retry with `TT_VISIBLE_DEVICES`
  before suspecting the port.* Cost here: ~40 minutes.

- **Gap — the `TensorAccessor` 3rd-arg Class-2 rule is stated as "the value equals what Metal 2.0
  supplies implicitly", and on an interleaved accessor that is false while the drop is still safe.**
  The brief calls both slice sites *"a pure no-op"* because the passed value equals the implicit one.
  For the interleaved case it does not: `per_shard_page_size_bytes` returns the true logical row
  (`common.cpp:782-792`), while the implicit value is that row rounded up to the allocator alignment.
  They address identically only because an *interleaved* accessor realigns the value internally —
  equivalence of effect, not of value. Writing the brief-mandated `check_accessor_page_size` guard as
  the equality the brief describes made **24 of 483 tests fail** on the first full run, every one of
  them an interleaved RM slice whose row bytes are not 32-aligned (e.g. `2 B` vs `32 B`, `126 B` vs
  `128 B`). The guard is only meaningful where the accessor consumes the value verbatim, so it is now
  scoped to sharded tensors (`slice_program_factory_rm.cpp:293-312`).

  This is the concrete form of the gap the audit already predicted in its own *Recipe notes* #1 — that
  the taxonomy has no row for `buffer->page_size()` on a sharded accessor, and that the two Class-2
  clauses ("correct magnitude" vs "`== aligned_page_size`") come apart there. It is worth the
  subject saying explicitly that the *equality* clause is a sharded-accessor test and the
  *magnitude* clause is the interleaved one; stated as one rule it invites exactly this guard.

- **Confusion — the brief describes a tree this port did not start from.** Its two "already done on
  `akertesz/slice-test`, verify and skip" call-outs (the `TensorAccessor` 3rd-arg drop, the
  `ccl/mesh_partition` bridge) are both *absent* here, and one of them is a hard build blocker for the
  very first factory. The brief does say the text is kept "for a port that starts from an unpatched
  tree", which is what saved it — but the tree state is worth an explicit *check this first* line
  rather than a note attached to one item. See `METAL2_PORT_PLAN.md` → *Tree state this port starts
  from*.

## Open items for downstream

### Shared kernel touches

| Kernel | Relationship | Rung taken | Remaining unmigrated consumers |
|---|---|---|---|
| `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | **borrowed** by `SliceTileTensorArgs` | **Rung 1 — reused the existing `_metal2` fork beside it** (`…/writer_unary_interleaved_start_id_metal2.cpp`). No new file; the fork was not edited; no pointer comment added to the legacy original (it already has one). | see sunset list below |
| `slice/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` (slice's own copy) | neither borrowed nor lent — `SliceTileProgramFactory` is its only binder | **converted in place, no fork** | none |
| slice's other seven kernels | slice-owned, single-binder | converted in place | none |

**Sunset list for the legacy eltwise copy** — coordination only, *not* authorization to convert it in
place. Slice is now off that path, so it drops out of the consumer set. At least fifteen factories
still bind it, among them `data_movement/concat`, `data_movement/reshape_on_device`, five
`data_movement/tilize` factories, `eltwise/unary_backward/tanh_bw`, `embedding`,
`examples/example` (×2), `experimental/matmul/attn_matmul`,
`experimental/transformer/nlp_concat_heads` (+ `_boltz`). Tracked as **issue #52228**, which also
records the duplicate fork at
`copy/typecast/device/kernels/dataflow/writer_unary_interleaved_start_id_metal2.cpp` — the
eltwise-sited one is the one to bind.

### Pre-existing findings, preserved not fixed

Everything here is behaviour the port carried forward deliberately. Each is an ops-team question.

1. **Cache-miss / cache-hit divergence in the tile factories.** For a node the work split left idle,
   the build path emitted writer `start_id = 0` while the refresh path re-emitted the running tile
   count. Inert — `num_pages` is 0 there, so the writer's loop never runs — but the two paths
   genuinely disagreed. Preserved on both sides
   (`slice_program_factory_tile.cpp:232-236` and `:266-271`, with the same split in
   `…_tile_tensor_args.cpp`). A follow-up should just make the idle node's value 0 in both.
   *Related:* the legacy `patch_slot0` deliberately skipped an arg slot holding 0, so an idle node's
   address slot kept its zero across hits; a `TensorBinding` patches every node uniformly, so idle
   nodes now receive the real address. Also inert, same reason.
2. **Dead `compile_time_element_size` CTA** in all four stride kernels
   (`reader_multicore_slice_4d.cpp:79`, `writer_multicore_slice_4d.cpp:63`,
   `reader_multicore_slice_nd.cpp:65`, `writer_multicore_slice_nd.cpp:64`). Declared, never used; the
   kernels use the runtime `element_size`. Still emitted by the host
   (`slice_program_factory_rm_stride.cpp:98`). Kept, as the audit directs.
3. **Dead RTAs in the 4D stride writer** — `output_h`, `output_d`, `output_n` are read and never used;
   only `output_w` is. Still named and still emitted. Same for `tensor_rank` in that file.
4. **`SliceTileTensorArgs` reads the end tensor and discards it.** The reader does a full DFB-staged
   read into `end_indices`, which is `[[maybe_unused]]` and never read afterwards. The port keeps the
   whole apparatus: the `end` `TensorParameter`, its binding, its accessor, and the staging round
   trip — because the read still happens. If the tensor really is unnecessary, a binding, a DFB
   round trip and an accessor can all go; that is an ops-team call, not a port one.
5. **Dead `old_src_tile_id` local** at `reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp:117`
   — assigned each iteration, never read. Preserved verbatim.
6. **Dead preprocessor branches** — slice's own writer copy still gates on `#ifdef OUT_SHARDED` and
   `#ifdef BACKWARDS`, and no slice factory sets any `defines`, so neither can fire. Preserved (and
   the reused eltwise fork has the identical pair).
7. **Stale comments in the off-limits device-op class.** `slice_device_operation.cpp:386` and `:412`
   explain the hash's contents in terms of `TensorAccessorArgs` baked into the writer's compile-time
   args. The mechanism is now the tensor binding, so the wording is stale — the *reasoning* still
   holds (the accessor layout is still baked per cache entry, from the `TensorParameter` spec). Not
   touched: that file is outside the port's writeable surface.

### Test coverage notes

- **The op has no C++ gtests.** `unit_tests_ttnn` and every sibling binary return zero tests for
  `*Slice*`; the 67 hits in `unit_tests_ttnn_ccl` are CCL slice-helper tests, unrelated to this op.
  Python is the whole safety net, which is worth knowing before the next structural change here.
- **`ccl/mesh_partition` remains unexercised on this bench** (t3000/TG only), so the out-of-directory
  change in *Handoff points* #2 is compile-verified only.
- **Two unreferenced kernel files** remain in the op directory and were deliberately not touched or
  audited: `device/kernels/dataflow/strided_slice_reader_rm_interleaved_nd.cpp` and
  `device/kernels/dataflow/strided_slice_writer_rm_interleaved.cpp`. They still contain legacy
  `TensorAccessorArgs<N>()` / positional-CTA idioms, which is the only reason a tree-wide grep for
  those patterns is non-zero after this port. If they are genuinely dead they should be deleted.

### Per-op carry-over

- `data_movement/pad` is the closest structural sibling (same sharded-gather reader shape, same
  `id_per_dim` write-back, same borrowed-shard DFBs) and is already ported. Anyone porting a third op
  in this family should read `pad`'s factories first — the two ports independently reached the same
  answers on every shared construct, which is a good sign those answers are the intended ones.

---

## Rebase onto the live branch

The port was written against a local `edwinlee/Port_Slice` at `d1f66c276f2` and was later rebased onto
the **remote** tip of the same branch, `6ebddf3088a`. The two had diverged, and the remote was ahead —
so this is not bookkeeping; it changed the port in three ways worth recording.

**1. Two items the port did itself were already done upstream.** The plan's *Tree state this port
starts from* table reports both as absent, which was true of the base it was written against and is
no longer true of the base it landed on:

| Item | Original base `d1f66c276f2` | Landed base `6ebddf3088a` |
|---|---|---|
| `TensorAccessor` 3rd-arg drop + host arg reindex | absent — the port did it | **already present** |
| `check_accessor_page_size` | absent — the port wrote one | **already present** |

The kernels and the host arg lists agree either way (12 named reader RTAs, 9 named writer RTAs), so
the conflict was textual rather than semantic: upstream had the legacy-shaped post-drop version and
this port has the Metal 2.0 named-argument version of the same state.

**2. Upstream's `check_accessor_page_size` is the better one, and is the one that survived.** This
port's version skipped interleaved tensors entirely, on the reasoning that an interleaved accessor
realigns the page size internally so any correct-magnitude value is inert. Upstream's instead
*checks* the interleaved case by rounding —
`effective = is_sharded ? per_shard : round_up(per_shard, alignment)`, compared against
`aligned_page_size` — which catches a genuinely wrong row size there rather than waving it through.
Upstream's is now in `slice_program_factory_rm.cpp`; this port's was discarded.

**3. Ops-team fixes the port never saw, carried through untouched.** The landed base also contains
preallocated-output validation in `slice_device_operation.cpp` (padded-vs-logical shape, a row-major
logical-shape check, and a `tensor_layout()` comparison) and a `can_land_in_preallocated` guard in
`slice.cpp`. Both files are outside the port's writeable surface and neither is touched, so they
survive intact — `git diff origin/edwinlee/Port_Slice HEAD` over those two paths is empty. The
"byte-identical" claim in *Verification* is against this landed base.

The rebase is also where `test_slice.py` moved from 445 to 448 passing: those three are the
preallocated-output fixes above, not this port.
