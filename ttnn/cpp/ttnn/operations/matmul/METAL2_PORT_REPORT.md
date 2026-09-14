# Metal 2.0 Port Report — `ttnn/cpp/ttnn/operations/matmul`

**Factory ported: `MatmulMultiCoreReuseOptimizedProgramFactory`** (one of eight in the op).

## Outcome

**`PORTED`** — `MatmulMultiCoreReuseOptimizedProgramFactory` converted to
`ProgramSpecFactoryConcept`, host build green on the first attempt, and the confirmed test set
identical before and after (see Verification). The op's other seven factories are untouched and
remain on their current concepts; the `program_factory_t` variant dispatches per factory, so the op
builds and runs with one factory on Metal 2.0 and seven on the descriptor API.

Nothing about this port reached the capitulation off-ramp: no construct failed to translate, no
change was needed outside the op directory beyond the three invoker-authorized pybind/re-export
deletions, and the shared compute kernel's existing fork fit without modification.

## Provenance

- **Recipe docs (this port):** read out of the recipe branch, not this checkout (this branch is
  based on plain `main`, so `docs/…/metal_2.0/` is absent and the provenance command prints
  nothing). Branch tip used: `akertesz/op-porting-recipe` @ `4bd4bf42bfe`
  `docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`.
- **Audit docs (inherited):** `058100de698 2026-08-31 docs(metal_2.0): let the sheet gate
  multi-program ops, and bound what the port covers`.
- **Doc delta since the audit's pin:** 8 files, and nothing that reaches this port — six are anchor
  renames (`conditional--optional-dfb-bindings` → `…-resource-bindings`), two are relaxation
  analyses for other ops. The one substantive edit (`ttnn_factory.md`: relaxation source authority,
  and the custom-hash containment rule generalised from "relaxations are always none" to "the hash
  must not be looser than the declaration") does not apply — this op's relaxation is `none` and it
  has no framework-visible custom hash.

## TTNN ProgramFactory

### Concept realized

**`ProgramSpecFactoryConcept`** (base), as the audit chose. `create_program_artifacts` is the only
method; **no `override_runtime_arguments` was added**, so the framework refreshes tensor bindings on
cache hit. Nothing about the port prompted revisiting the audit's choice.

### Device-op-class edits

- **Pybind entry points removed:** `ttnn/cpp/ttnn/operations/matmul/matmul_nanobind.cpp` — the whole
  `nb::class_<MatmulMultiCoreReuseOptimizedProgramFactory>` block (was `:1239-1258`), covering
  `create_descriptor` and `default_core_range`.
- **Factory members removed:** `create_descriptor`'s pybind-only `core_range_set` parameter (and the
  work-split branch only it could reach), and `default_core_range` in full — invoker-authorized, no
  production C++ caller.
- **Custom `compute_program_hash`:** none to leave intact. The device-op's
  `compute_descriptor_program_hash` (`device/matmul_device_operation.hpp:50`) is deliberately not
  named `compute_program_hash` and is reached only through a pybind alias — untouched.
- **One guard message updated, not the guard:** the `TT_FATAL` at the top of the factory said
  "program_config must be provided for **create_descriptor**". The condition is unchanged; only the
  function name in the message moved to `create_program_artifacts`, since the old name no longer
  exists anywhere.

## Handoff points

### API surface: removed entry point — and it reaches outside the op directory

`create_descriptor` and `default_core_range` are gone from
`MatmulMultiCoreReuseOptimizedProgramFactory`, so three files lost lines:

| File | What was removed | Why it was mandatory |
|---|---|---|
| `ttnn/cpp/ttnn/operations/matmul/matmul_nanobind.cpp` | the factory's `nb::class_` block | references vanished symbols; would not link |
| `ttnn/ttnn/operations/matmul.py` (was `:25`) | `MatmulMultiCoreReuseOptimizedProgramFactory = ttnn._ttnn.operations.matmul.…` | **module-scope** attribute read — without this deletion `import ttnn` raises `AttributeError` for every Python consumer in the repo |
| `ttnn/ttnn/__init__.py` (was `:512`) | the same name in the `from ttnn.operations.matmul import (…)` list | same |

This is the **first** matmul port to touch a file outside the op directory: this was the only matmul
*factory* re-exported into the public `ttnn` namespace, which is why #55224, #55961 and #56114 each
stayed entirely inside `ttnn/cpp/ttnn/operations/matmul/`. A reviewer applying "a port touches only
the op directory" as a heuristic will see three out-of-directory files here; all three are forced,
and all three were authorized by the invoker before conversion began.

### The Python descriptor framework loses this factory — ~15 tests, not one

`models/experimental/ops/descriptors/matmul.py` calls `factory.create_descriptor(...)` (`:120`) on
whatever `ttnn.matmul_select_program_factory` returns (`:97`). It guards only
`MatmulMeshWorkloadMultiCoreReuseMcast1DProgramFactory` (`:29`, `:98`), so after this port a
`MatmulMultiCoreReuseProgramConfig` through that path raises a **nanobind `TypeError` at `:97`** — at
the select call, not at the `create_descriptor` call. That makes **3 unbound alternatives / 2
unguarded** (#55224 and #55961 each widened this gap before this port; the general fix — one guard
covering every alternative the select can return unbound — is main-targeted and deliberately not in
this diff).

The invoker's instruction was to leave the affected tests failing and record them. The count is
larger than the one test named in the launch prompt, because **both** fused test files reach matmul
*exclusively* through the descriptor framework (`test_parallel_sequential.py`: 26
`descriptors.matmul` imports, **zero** direct `ttnn.matmul(` calls; `test_fused_demo.py` likewise):

- `tests/ttnn/unit_tests/operations/fused/parallel_sequential/test_parallel_sequential.py` — 10 test
  functions build the config directly (`test_matmul_plus_fused_chain:1088`,
  `test_matmul_slice_ln_rms_tree:1561`, `test_persistent_matmul_deferred:2169`,
  `test_reuse_optimized_factory:2705`, `test_reuse_optimized_rms_chain:2840`,
  `test_parallel_different_factories:2992`, `test_branching_rms_to_different_mm_factories:3051`,
  `test_deep_chain_alternating_factories_and_norms:3120`,
  `test_parallel_mm_chains_different_factories:3186`, `test_nested_tree_mixed_factories:3285`) and 5
  more reach it through the `_mm_config()` helper at `:581` (`test_matmul_standalone`,
  `test_multicore_matmul_chain`, `test_matmul_followed_by_n_rms`, `test_fp32_mismatch_error`, and
  `mk_mm`). Several carry `@stress_test_program_cache`, so each is 5 runs.
- `tests/ttnn/unit_tests/operations/fused/parallel_sequential/demo/test_fused_demo.py` — the
  `*_fused` / `*_unfused` chain and sharded-tree tests, via `_linear_chain_setup:315`, `_shard_mem`
  (three copies) and `_sharded_tree_setup:1041`.

Two mitigations worth knowing: only the ops selecting **this** factory fail — the Mcast2D/1D configs
in the same mixed-factory tests keep working, since those factories keep `create_descriptor` — and
`parallel_sequential` appears nowhere in `.github/workflows/` or `tests/scripts/` outside its own
`conftest.py`, so none of it is CI-gating today. The in-flight merge gate for matmul/fused/reduction
ttnn ops (#56284) would change that, which is the deadline this handoff is really against.

**Owner:** the fusion-framework / descriptor-framework maintainers. Port 5 (Mcast2D) will hit the
identical problem for `test_mcast_2d_factory` and its siblings.

### Shared kernel touches

| kernel | rung taken | detail |
|---|---|---|
| `device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp` | **rung 1 — reused the existing `_metal2` fork** | bound `…_metal2.cpp` (created by #55961, already reused by #56114). No new file, no edit to the fork, and no pointer comment added to the legacy original — rung 1 forbids touching it. |
| `device/kernels/dataflow/reader_bmm_tile_layout_in0.cpp` | in place | private to this factory (1 binder) |
| `device/kernels/dataflow/reader_writer_bmm_tile_layout_in1.cpp` | in place | private to this factory (1 binder) |

**Remaining legacy consumers of the un-forked compute original: 3** — `mcast_2d`, `mcast_1d` (one
file, two factories), and the sparse device-op's factory. The fork can be sunset and take over the
original's name when the last of those migrates.

## Successes

- **The fork's own inline comment did the job it was written for.** At
  `…_metal2.cpp:205-209` it says the transpose selection arrives as a define rather than an argument
  and that *"Future factories adopting this fork must keep the `#ifdef`, not restore a ternary."*
  That is precisely the decision a rung-1 consumer is most likely to get wrong — legacy passes
  `in0_transpose_tile` as compute CTA slot 17, and the reflex is to carry it across as a named arg.
  Reading the fork's contract first turned a would-be build error (`args::in0_transpose_tile` does
  not exist) into a one-line define.
- **"Go to the headers first; they are ground truth" earned its place in the recipe** — see the
  Friction entry below. The patterns catalog's aliased-DFB legality list would have sent this port
  either into an invented DFB endpoint or into a capitulation; the declaring header said the rule is
  something weaker, and the port is correct because the header was consulted.
- **The rung-1 fit check is cheap and worth doing before writing any code.** Enumerating the fork's
  gated vs ungated argument reads up front established that this factory feeds none of the gates it
  cannot supply (`MATMUL_DRAM_SHARDED`, `SFPU_ACTIVATION`, `MM_PARTIALS_RELOAD_ALIAS`), so the reuse
  could not fail late. `last_subblock_w_valid` in particular has an `#else` supplying
  `out_subblock_w`, so passing it — as a neighbouring ported factory does — would have been an
  unread named arg here.

## Friction

### Gap — the patterns catalog states the aliased-DFB legality rule more strictly than the header

`port_patterns.md` → *Pattern: Aliased DFBs* lists the constraints as "Same `num_entries *
entry_size`", "**Bound to the same set of kernels**", "Borrowed-memory consistency". The declaring
header says something materially weaker:

```
//   - Every DFB in the alias group must list every other member as an alias
//   - Aliased DFBs must have the same total size (num_entries * entry_size).
//   - All members must target the same node set
//     (derived from their bound kernels' WorkUnitSpecs).
```
`tt_metal/api/tt-metalium/experimental/metal2_host_api/advanced_options.hpp:168-171`

The difference decides this port. `OUT` is bound by the compute specs **and** the reader_writer;
`INTERMED0` is compute-only. Under the catalog's wording the alias is illegal and the options are to
invent a reader_writer binding for the intermediate or to capitulate. Under the header's wording it
is legal on inspection: both DFBs' binders resolve to `core_group_1 ∪ core_group_2 = all_cores`, the
same node set. **Suggested fix:** restate the catalog's third bullet as "target the same node set
(derived from their bound kernels' WorkUnitSpecs)", and note that the kernel *sets* may differ.

The same header paragraph (`:182-186`) also states the disjoint-node work-split case explicitly —
"a *DFBSpec* (spanning multiple nodes) can have more than one KernelSpec producer or more consumer
bindings, as long as every node's DFB instance has one producer and one consumer" — which is the
cleanest confirmation available that two compute specs binding one DFB in one role need no advanced
option. Worth citing from the catalog's *Demoting per-group CTA to RTA* entry, which currently
argues the same point from the invariant without quoting it.

### Gap — an audit brief carries no freshness signal, and this one's rung instruction had expired

The brief (13 days old) instructs **rung 2 — create the fork**, and supports it with a locational
check: *"`find` over `matmul/device/kernels/` returns **zero** `*_metal2*` files."* Both statements
were true when written and both were false by the time the port ran: #55961 created the compute fork
and #56114 already reused it. A porter following the brief literally creates a second fork of a
kernel that already has one — the precise failure the locational rung-1 check exists to prevent,
arriving through the document that prescribes it.

The invoker caught this and said so in the launch prompt, which is what saved it. But that is a
human remembering, not a procedure. **Suggested fix:** have the audit emit the rung as a *check to
re-run* rather than a verdict — one line in the brief's shared-kernel table, e.g. "rung as of
`<date>`: 2 (re-run the locational check before binding; a fork may have landed since)". Cheap to
write, and it converts a silently-stale instruction into a two-second `ls`.

### Gap — the `cb`-name self-audit sweep has no usable denominator on a single-factory port

The self-audit's sweep is specified over the **op directory**:

```bash
grep -rnE '[Cc][Bb]_|_[Cc][Bb]\b|\b[Cc][Bb]\b|\bCB[A-Z]' <op-dir>
```

with "Expect **zero** hits: post-port the op has no CBs". That holds for an op ported in full. This
op has **eight** factories and one is ported, so seven legitimately still build `CBDescriptor`s and
carry `cb_*` names — the op-wide sweep returns hundreds of hits that are all correct, and the check
degenerates into something a porter either ignores or, worse, "fixes" by touching factories outside
the port's scope.

Scoping it to the files the port actually touched restores the signal (2 host files here, plus the
two converted kernels), and it caught a real one: a comment I had written in the new factory still
named the legacy `CBDescriptor` type while explaining why two buffers are aliased. Rewritten to
state the same reason without the legacy type. **Suggested fix:** in the checklist item, scope the
sweep to the ported factory's own files plus its converted kernels — the diff-derived file list the
ephemeral-doc check already builds is exactly the right input — and keep the op-wide form only for
ops ported in one change.

### Confusion — the `opt_level` pairing check misreads a shared spec-building helper

The checklist says to enumerate the compute `KernelSpec`s from the construction code and "pair each
one with a line of that output. **A compute `KernelSpec` with no line is the defect.**" This factory
builds both compute specs from one `make_compute` lambda, so `grep -nE opt_level` returns **one**
line for **two** specs and the stated pairing fails even though both specs carry `O3`. The item does
anticipate the shape ("Specs built in a loop or through a shared helper are no exception — the level
is per `KernelSpec`"), but the mechanical check it prescribes contradicts that sentence. **Suggested
fix:** phrase the check as "every construction path that produces a compute `KernelSpec` sets the
level", so one line covering a shared builder reads as a pass.

### Confusion — `disable_dfb_implicit_sync_for_all` on the arch-agnostic DM helpers

`ttnn::create_reader_datamovement_config(arch, disable_dfb_implicit_sync_for_all = false)` takes a
flag that is, per its own comment, "a Gen2 (Quasar) concept only — ignored on the Gen1 (WH/BH)
placement". A neighbouring ported matmul factory passes `true`
(`matmul_multicore_program_factory.cpp:191`, `:217`). The recipe says to build only the Gen1 config
and author no Gen2 judgment, so this port omits the argument — Gen1-identical either way. It took a
read of the helper to establish that the sibling's `true` was not something this port had to match.
**Suggested fix:** one sentence in the recipe's *Data movement kernels* section saying the flag is
Gen2-only and a Gen1 port should leave it defaulted, so the next porter does not have to derive that
from a sibling's diff.

## Open items for downstream

- **Stale comment in off-limits code.** `device/matmul_device_operation.cpp:919` reads "Mirror the
  shard_spec priority in `MatmulMultiCoreReuseOptimizedProgramFactory::create_descriptor`" — a
  method this port deletes. The line is in the device-operation class, which is off-limits, so it is
  reported rather than edited. The priority it mirrors is unchanged; only the name is now dangling.
- **`packer_l1_acc && (num_blocks > 2)`** (factory `:112` pre-port). Every sibling matmul factory
  uses `> 1`. Carried verbatim, as instructed. It is not inert: it selects `interm0_data_format`,
  which selects whether `OUT`/`INTERMED0` are aliased at all, so a "correction" would change the L1
  topology, not just a flag. Flagging for the op owner as a question, not a defect.
- **Explicit `unpack_modes` where legacy had none.** Legacy set no `unpack_to_dest_mode`, i.e.
  `Default` for every buffer. The port states `UnpackMode::UnpackToSrc` explicitly for each buffer
  the compute kernel consumes (`in0`, `in1`, `intermed0`, plus `bias` / `in0_transposed` when
  bound). That is the faithful translation of `Default`, and it satisfies unconditionally the
  Metal 2.0 rule that a Float32 buffer consumed under `enable_32_bit_dest` must carry an explicit
  entry — which `intermed0` hits whenever `fp32_dest_acc_en` is set (`interm0_data_format` becomes
  `Float32` at factory `:114-116`). Stated for every consumed buffer rather than conditionally, so
  there is no branch that can silently miss the required case.
- **Gen1-only token-form metadata sites.** In `reader_bmm_tile_layout_in0.cpp` the legacy
  `constexpr` metadata reads keep the free-function form with the binding token —
  `get_tile_size(dfb::in0)` and `get_dataformat(dfb::in0)` — because a `DataflowBuffer` member
  getter cannot yield a constant expression and these values feed `constexpr` context. The token's
  `uint32_t` conversion is documented Gen1-only, so these are Quasar-uplift debt. The
  `reader_writer_bmm_tile_layout_in1.cpp` equivalents were already non-`constexpr` member getters
  and needed no change.
- **`disable_dfb_implicit_sync_for_all` consistency across the op.** Port 1's factory passes `true`
  on both DM helpers; this port leaves it defaulted. Identical on Gen1, divergent on Quasar — worth
  settling once during the uplift rather than per factory.
- **Relaxation candidates:** none noticed. All four `TensorParameter`s stay strict, matching the
  audit's `none`.

## Verification

*Filled in after the build and test runs; see the summary at the end of the port.*

### Confirmed test set

Discovered by sweeping every test tree for the op name and then filtering to the files that build
`MatmulMultiCoreReuseProgramConfig` — the only selector that reaches this factory
(`select_program_factory`, `device/matmul_device_operation.cpp:2196-2197`); no auto-selected path
does, so a generic matmul test never exercises it. Confirmed with the invoker before use:

| Tree | Tests |
|---|---|
| C++ gtest | `tests/ttnn/unit_tests/gtests/test_matmul.cpp` → **`MatmulSmoke.ReuseBmm` (`:324`) only** |
| pytest | `tests/ttnn/unit_tests/operations/matmul/{test_matmul,test_linear,test_custom_grids}.py` |

⚠ **`MatmulSmoke.ReuseBmmBatchBroadcast` (`:375`) does *not* exercise this factory, despite the
name.** It calls `ttnn::matmul(ta, tb)` with pure auto-dispatch, which
`create_simple_matmul_program_config` routes to `MatmulMultiCoreReuseMultiCastProgramConfig` — the
2D mcast factory. Its own comment says so: *"An explicit MatmulMultiCoreReuseProgramConfig would
also accept this… but the auto path is what models actually hit."* The test is named for the
batch-broadcast escape class, not for the Reuse factory. It was on the confirmed set as this
factory's coverage (mine and the invoker's list both had it) and it is not; **C++ coverage of this
factory is a single test**, which is worth knowing for whoever sizes the risk here — the pytest
files, and `test_custom_grids.py` in particular, carry the rest.

How it surfaced is worth recording, because it is a use for the forced-check markers beyond proving
they are on: post-port, the two tests together produced only **one** `BuildProgramFromSpec` marker.
A cache-behaviour change was the obvious suspect (the same test had also gone 1469 ms → 14 ms). The
benign explanation turned out to be the right one — one test is on the spec path and the other is
still on the descriptor path, which emits no marker — and the marker count is what distinguished
"this factory ran once" from "this factory ran twice and hit the cache". Counting spec-path
constructions is a cheap way to confirm *which* tests actually reach a ported factory.

`test_custom_grids.py`'s "Factory B" class is the load-bearing one: it drives this factory at the
default grid, at custom grids, and on a sub-device via `allowed_worker_cores`, which is exactly the
work-split branch that survives the deletion of `core_range_set`.

Two files with Reuse-config coverage are **not** in the set and remain open questions for the
invoker: `tests/ttnn/nightly/unit_tests/operations/matmul/test_matmul2.py` (4 sites) and
`tests/device_perf_tests/matmul_stagger/test_run_8x7_matmul.py` — the latter being the only coverage
of the stagger / throttle defines this factory emits. `test_matmul_program_cache.py` was considered
and excluded: it calls bare `ttnn.matmul(a, b)` with no program config, so it cannot select this
factory.

### Pre-port baseline

Both runs with the forced legality checks live and `TT_METAL_WATCHER=10`:

- `./build/test/ttnn/unit_tests_ttnn --gtest_filter='MatmulSmoke.*'` → **22/22 passed**, including
  `ReuseBmm` and `ReuseBmmBatchBroadcast`. `METAL2_CHECKS_FORCED` appeared from **both** forced
  translation units (`program_spec.cpp:2950` and `program_run_args.cpp:565`, alternating in pairs),
  so the checks were provably running in the binary under test — the markers carry their source
  location, which distinguishes the two files more reliably than counting occurrences.
- The three pytest files → **1097 passed, 317 skipped, 2 xfailed, 1 xpassed, 0 failed** of 1417
  collected, in 30m42s.

### Post-port

Host build: **green on the first attempt**, no errors.

`import ttnn` verified working after the pybind and re-export removals:
`MatmulMultiCoreReuseOptimizedProgramFactory` is gone from the namespace (intended) while
`matmul_select_program_factory` and `MatmulMultiCoreReuseProgramConfig` remain bound — so normal
`ttnn.matmul` with an explicit Reuse config still dispatches, and only the descriptor path loses its
entry point.

| Run | Baseline | Post-port |
|---|---|---|
| `MatmulSmoke.*` gtest | 22/22 passed | **22/22 passed** |
| the three pytest files | 1097 passed, 317 skipped, 2 xfailed, 1 xpassed, **0 failed** | **1097 passed, 317 skipped, 2 xfailed, 1 xpassed, 0 failed** |

Identical, test for test. During the post-port pytest run the forced checks logged **477**
`BuildProgramFromSpec` constructions, so the ported factory was genuinely exercised through the
validated spec path rather than skipped over.

**Two numbers that look like results and are not.** The gtest suite went 45.4 s → 3.0 s and the
pytest set 1842 s → 1054 s. Both are JIT kernel-cache warmth: the baseline runs paid for a full
kernel recompile because enabling `TT_METAL_WATCHER` invalidates that cache, and the post-port runs
did not. Nothing here is evidence about op performance in either direction, and no timing
measurement in this report was taken with Watcher off.

Additional evidence beyond the confirmed set: `tests/ttnn/nightly/unit_tests/operations/matmul/test_matmul2.py`
(290 tests, 4 Reuse-config sites) was run post-port because C++ coverage of this factory turned out
to be a single test — result recorded below. `tests/device_perf_tests/matmul_stagger/test_run_8x7_matmul.py`
was **not** run: it is a timing test, and every run in this port had Watcher on, whose checks are
deliberately expensive. It remains the only coverage of the stagger / throttle defines this factory
emits, so it is worth one run with Watcher unset before merge — flagged for the invoker rather than
done here.

Nightly result: **258 passed, 32 skipped, 0 failed** (390 s), with 178 `BuildProgramFromSpec`
constructions. Read this as "green post-port", **not** as a comparison — it was added after the
baseline was taken, so there is no pre-port run of this file to diff against. The three confirmed
pytest files and the gtest are the only true before/after pairs.
