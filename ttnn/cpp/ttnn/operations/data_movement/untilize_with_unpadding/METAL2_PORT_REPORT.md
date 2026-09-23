# Metal 2.0 Port Report — `data_movement/untilize_with_unpadding`

> **Revision, 2026-09-23 — five of five.** This replaces the three-of-five report (still in branch
> history). That report recorded a narrowing forced by untilize codegen's `build_native_equivalent`
> calling `create_descriptor` on two of these factories. PR #56280 deleted that caller. All five
> factories are now on Metal 2.0, and this revision re-verifies them on the tree after merging
> `main`.

## Outcome

**`PORTED` — all five factories on `ProgramSpecFactoryConcept`.** Tests pass on the merged tree, with
Watcher on and the Metal 2.0 legality checks forced and proven live (see [Verification](#verification)).

| factory | concept | tensor bindings | kernel sources bound |
|---|---|---|---|
| `SingleCore` | `ProgramSpecFactoryConcept` | 2, Case 1 | 2 forks reused, 1 converted in place |
| `MultiCoreInterleaved` | `ProgramSpecFactoryConcept` | 2, Case 1 | 2 forks reused, 1 converted in place |
| `MultiCoreSharded` | `ProgramSpecFactoryConcept` | 1 Case 1 + 2 borrowed-memory DFBs | 3 forks reused, 1 fork created, 4 converted in place |
| `MultiCoreBlockInterleaved` | `ProgramSpecFactoryConcept` | 2, Case 1 | 3 forks reused (all created by #56280) |
| `MultiCoreNDSharded` | `ProgramSpecFactoryConcept` | 3, Case 1 | 2 forks reused, 1 converted in place |

**Two changes in this diff deviate from the recipe, both by explicit invoker decision.** They are
recorded as [Handoff points](#handoff-points) 1 and 2 so a reviewer finds them before reading the code:

1. **The shared-kernel sunset.** Five orphaned legacy kernels are deleted, with their build-file
   entries and fork header notes. The recipe says the sunset "is not the porter's to perform".
2. **A `BACKWARDS` correction** in the `reader_unary_interleaved_wh_multicore_metal2.cpp` fork,
   which `data_movement/untilize` also binds. The recipe says a port preserves bugs.

Neither changes observable behaviour in any current configuration. The deleted files have no
binder, and no consumer defines `BACKWARDS`.

### What this revision re-verified, and why the restore needed it

The two restored factories (`MultiCoreInterleaved`, `MultiCoreBlockInterleaved`) and their op-owned
writer are **byte-identical** to the five-of-five state that passed earlier with the checks forced.
Several things around them had moved, though, so that earlier green was not inherited on trust:

- **The BlockInterleaved factory now binds #56280's forks, not this branch's own.** Each was diffed
  against the fork this branch created earlier and against its legacy original. The kernel bodies
  are behaviourally identical, apart from the `BACKWARDS` walk (Handoff 1), and the named-arg /
  binding vocabulary matches what the factory emits, name for name.
- **`main` moved under the legacy side.** The legacy factories are unchanged since the port was
  written; `git diff` of the five factory files is empty. What did change was `untilize_output_dtype`
  and `get_pending_l1_output_reservation` (#56280 and a follow-up). They change which factory is
  *selected* for BFLOAT8_B inputs, not what any factory builds. So the BFLOAT8_B coverage was re-run
  rather than cited (`test_untilize_bfloat8_b.py` joined the baseline for that reason).

## Provenance

- **Recipe docs (this port):** `git log -1 --format='%h %cs %s' -- docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/`
  prints **nothing**, because the doc tree is not on this branch. Pinned by content instead. The
  copy used (extracted outside the repo) is blob-identical to `akertesz/op-porting-recipe` at
  `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`:
  `port/metal2_port.md` `33e573fa1b6`, `shared/port_patterns.md` `8d79949fffe`,
  `shared/ttnn_factory.md` `7f3e389a0fc`, `shared/migration_guide.md` `9356d8cf4e4`,
  `shared/cb_dfb_api_whitelist.md` `798733a9c6c`.
- **Audit docs (inherited):** audit recipe blob `d0576d6d739`; newest `metal_2.0/` commit on
  `akertesz/op-porting-recipe` is `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`.
- **Code under test.** The record run (run 3, below) is on `HEAD` `30dbe7153e0`: four commits on
  `main` at `e4a9afc5927`, plus the uncommitted `eltwise_copy_metal2.cpp` install entry (Handoff 3).
  Runs 1–2 were on the pre-rewrite tree (`c3038aee531` plus working tree). Its code is identical to
  `HEAD`'s. Its tests differ only by the coverage commit's new parametrizations: `git diff
  c3038aee531 HEAD -- tests/` is those 13 lines of `test_untilize_with_unpadding.py`.

## TTNN ProgramFactory

### Concept realized

`ProgramSpecFactoryConcept` (plain) on **all five** factories, as the audit chose. Each
`create_descriptor` became `create_program_artifacts`, returning
`ProgramArtifacts{spec, run_params}`. `op_owned_tensors` stays defaulted, because the op allocates no
device tensors of its own. The mixed-concept variant of the three-of-five state is gone, and with it
the only reason anything outside the op directory named one of these factories.

### Device-op-class edits

- **Pybind entry points removed:** none. `untilize_with_unpadding_nanobind.cpp` binds only the
  user-facing op, so **there is no user-visible API change**.
- **Custom `compute_program_hash`:** none. The op uses the default reflection-based hash, with no
  backdoor. Nothing touched.
- All eight non-factory files in the op directory are **byte-identical** to `main`: the device
  operation `.cpp`/`.hpp`, its types header, the op entry point `.cpp`/`.hpp`, the two nanobind
  files, and the dead `…_shared_variables.hpp`.

### Open items

- **Relaxation candidates:** none. Strict `TensorSpec` matching throughout; no `dynamic_tensor_shape`.
- **API dependency:** the MultiCoreInterleaved writer's per-core vararg count uses a field whose
  header says it will be removed. See Handoff 4.

## Handoff points

1. **The `BACKWARDS` correction to a shared fork is in this diff, by invoker decision. The
   `data_movement/untilize` owners should review it.** *(Owners: `eltwise/unary` kernel owners;
   `data_movement/untilize` as the fork's other consumer.)*
   - **File:** `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_wh_multicore_metal2.cpp:39-47`.
     #56280 created this fork; `untilize_multi_core_block_program_factory.cpp:58-60` (bound at
     `:179`) also binds it.
   - **The bug.** `dim`, `c` and `r` are `uint32_t`, so `dim > -third_dim` compares `0` against a
     huge unsigned value. The `BACKWARDS` loop never runs, and `-start_id` wraps.
   - **The change.** One loop nest for both directions, with the page id computed as
     `start_id ± offset`. The non-`BACKWARDS` path is arithmetically identical to before (same
     `uint32_t` sum, split into a named `offset`).
   - **Why it is inert today.** Nothing defines `BACKWARDS` for this kernel. The only
     `kernel_defines["BACKWARDS"]` in the tree
     (`data_movement/copy/device/copy_same_memory_config_program_factory.cpp:137`) binds a different
     file.
   - **Recipe status: deviation.** The port preserves bugs (§Scope discipline), and a fork with a
     consumer is read-only to a porter (*Caution: Porting a shared kernel*). The invoker chose to keep
     it. The sunset (Handoff 2) deletes the legacy original that carried the same bug, so no
     fork/original divergence is left behind.

2. **The shared-kernel sunset is in this diff, by invoker decision, and the docs disagree about
   whose job it is.** *(Owner: recipe maintainers.)*
   - **What is deleted:** five legacy kernels that no factory, test or build file anywhere binds any
     more:
     - `eltwise/unary/…/reader_unary_interleaved_wh_multicore.cpp`
     - `untilize_with_unpadding/…/writer_unary_stick_layout_wh_multicore.cpp`
     - `untilize/…/compute/untilize_wh.cpp`
     - `sharded/…/reader_unary_nd_sharded_blocks.cpp`
     - `ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_blocks.cpp`
   - **Build files:** the redundant explicit entry for the deleted writer is removed from
     `data_movement/CMakeLists.txt`. The legacy entry at `ttnn/sources.cmake:173` is **replaced** by
     the fork's; see Friction, gap 2, for why that directory needs an entry at all.
   - **Fork headers:** the surviving forks' "the original serves the legacy consumers" notes are
     rewritten. `untilize/…/compute/untilize.cpp` is **kept**, because
     `tests/…/parallel_sequential/test_parallel_sequential.py:1435` reads its source text. Its fork's
     note now says so, citing the full path.
   - **Safety evidence:** a repo-wide `git grep -F` for each deleted filename, across every file
     type, finds only the forks' own history notes. `experimental/quasar/` was checked without
     reading it: none of its 382 kernel path literals points into any of the five shared directories,
     and each deleted file has its own private copy there. Both sweeps were validated with a positive
     control (see [Successes](#successes)).
   - **The contradiction.** `port_patterns.md` → *Caution: Porting a shared kernel* → **Sunset**
     says: *"That cleanup is not the porter's to perform, but the porter feeds it."* This op's audit
     brief routed it as **PORT WORK**, citing #56280's description (*"… is the sunset point for the
     three legacy copies"*). The recipe's scope boundary also sanctions only fork creation plus a
     pointer comment outside the op directory, and treats a fork with a consumer as read-only. Five
     of the header rewrites touch such forks.
   - **The recipe's sunset is also more than was done.** *"The legacy copy is deleted and the fork
     takes over its name."* The forks here keep their `_metal2` names, and their notes say the suffix
     is historical. Renaming would also repoint two of `data_movement/untilize`'s factories (its
     block and ND-shard-input factories), which the port may not touch.
   - **Ask:** the recipe should say who performs the sunset, and whether the rename is part of it.
     The audit should not route PORT WORK that the port recipe forbids.

3. **`ttnn/cpp/ttnn/kernel/compute/eltwise_copy_metal2.cpp` was not installed — fixed in this
   diff.** *(FYI: TTNN build; the S2I port, #52207, which created the fork.)*
   - **The gap.** `ttnn/cpp/ttnn/kernel/` is installed from the explicit
     `TTNN_CORE_JIT_API_HEADERS` list in `ttnn/sources.cmake`, not from a glob
     (`ttnn/CMakeLists.txt:140-143`). On `main` that list names `eltwise_copy.cpp` but not its fork.
   - **Why it is this port's to fix.** Legacy, this op's Sharded factory bound the installed
     `eltwise_copy.cpp` on its W=16 path. The port rebinds that path to the fork
     (`…_multi_core_sharded_program_factory.cpp:363`). So without the entry, **the port itself**
     would break that path in installed builds: a dev checkout resolves the JIT path from the source
     tree and hides it. An earlier revision of this report called it someone else's pre-existing gap
     and left it; review (Copilot) correctly pointed out that this op newly depends on it.
   - **The fix.** `cpp/ttnn/kernel/compute/eltwise_copy_metal2.cpp` is added beside the legacy entry.
     It also mends S2I's installed build, which already bound the fork.
   - **Checked across the whole op.** Every one of the 17 kernel sources the five factories bind is
     now covered by an install rule:
     - the two `ttnn/kernel/` forks, by the explicit list;
     - the rest, by the `data_movement` and `eltwise/unary` `GLOB_RECURSE kernels` rules.

     The regenerated `build_Release/ttnn/cmake_install.cmake` names the new entry.

4. **`KernelAdvancedOptions::num_runtime_varargs_per_node` — this port depends on a field slated
   for removal.** *(Owner: Metal 2.0 API team; carried from the earlier report.)*
   - **The use.** The MultiCoreInterleaved writer's `BlockRep` payload is a genuinely per-core,
     variable-length vararg block (5 values per run, run count varying per core and per shape). The
     per-node count override is the only construct that reproduces the legacy per-core RTA layout
     exactly (`…_multi_core_interleaved_program_factory.cpp:268`).
   - **The risk.** Its header describes it as due for removal once existing uses are refactored.
     If it goes before a typed-array replacement exists, this factory needs a plan.

## Successes

- **Rung 1's "check fit before committing to reuse" did real work.** *(`port_patterns.md` → Caution:
  Porting a shared kernel.)* Three forks this port used to create were found locationally, beside
  their originals, already on `main`. The recipe's advice was to verify fit rather than assume it,
  so each was diffed against this branch's earlier fork and the legacy original. The result: the
  vocabulary matches name for name, and the bodies are equivalent. The one declaration difference
  resolved in #56280's favour. `untilize_wh_metal2.cpp` keeps the legacy `const uint32_t` CTAs,
  where this branch's own earlier copy had promoted them to `constexpr auto`. Under whitelist §A,
  "the legacy declaration is the entire test."
- **"Print the denominator" caught a false zero in the sunset's safety sweep.** *(§Anti-pattern
  self-audit, preamble.)* The first repo-wide `git grep` used `:!` exclude pathspecs and returned
  **nothing**, which was the passing answer for a deletion. A positive control (the forks' own header
  lines, which had to match) exposed the silent pathspec failure. The corrected sweep found the real
  reference set. The quasar path-literal check was given the same control (`data_movement/untilize` →
  14 hits) before its zero was trusted. For a *deletion*, an unvalidated zero is exactly how a live
  consumer gets broken; `untilize.cpp` was already nearly lost that way once.
- **"A bare filename fails by default" caught a new citation.** *(§Generated docs in the op
  directory.)* The rewritten `untilize_metal2.cpp` header named `test_parallel_sequential.py` bare.
  It now carries the full repo path, which `git cat-file -e origin/main:` resolves. The self-audit's
  own grep is `.md`-only and would not have flagged it (Friction, gap 3).
- **Forcing the checks, with markers that name their translation unit.** *(§Ensure the Metal 2.0
  host-side legality checks are enabled.)* All nine `bool skip_validation` sites were forced. The two
  markers carry distinct text (`METAL2_CHECKS_FORCED[program_spec]` / `…[program_run_args]`), so one
  live translation unit cannot pass for two. Both fired on every run (counts in Verification).

## Friction

### Gaps

1. **Sunset ownership contradicts the audit.** Handoff 2 has the detail. For a porter, the practical
   effect is that the brief hands over a task the recipe says is not theirs, with no tie-breaker. The
   audit's own Recipe notes (item 3) asks for a sunset check in the audit, which would make the
   contradiction systematic unless the port recipe changes too.
2. **Rung 2's "No build-system change is needed for the new file" is false for
   `ttnn/cpp/ttnn/kernel/`, and rung 1 does not ask the question at all.** That pool is installed
   from an explicit list, not a `file(GLOB_RECURSE …)`.
   - **Two forks had landed there without an install entry,** and both are now listed by this diff:
     this branch's own `writer_unary_stick_layout_interleaved_blocks_metal2.cpp`, created at rung 2,
     and S2I's `eltwise_copy_metal2.cpp`, reused at rung 1 (Handoff 3).
   - **The second one slipped past me.** My install check covered only the fork this branch
     *created*. It treated a reused fork as already shipped, because another op already bound it.
     A reused fork is newly shipped *for this op*, though. An automated review caught it.
   - **Dev checkouts pass either way, so no test catches either.**
   - **Suggested fix:** rungs 1 **and** 2 should say *"confirm the fork's directory has an install rule
     covering it. If the directory is installed from an explicit list, add the fork beside the
     original's entry."* That is a narrow build edit the carve-out should sanction, because without it
     the port does not ship. A mechanical form is a check that every `KernelSpec::source` the factory
     binds is matched by an install rule; this port's sweep of its 17 sources was that check.
3. **The ephemeral-doc check mechanizes only half its rule.** The rule is "code may only cite a path
   that resolves on `main`; a bare filename fails by default". The command greps only for `.md`.
   - **Suggested fix:** also grep changed comments for `\b[\w./-]+\.(py|cpp|hpp|h)\b` that are not
     `#include` targets, and run `git cat-file -e origin/main:` on each.
4. **No mode for re-checking a restored port.** This pass verified a byte-identical restore of
   factories that had already passed, on a tree whose dependencies had moved. The recipe is written
   for a first port.
   - **What I did:** treated construction as re-reviewed rather than rebuilt. I re-derived every
     moved dependency (the forks, `main`'s legacy side, the selection logic), then re-ran the full
     verification.
   - **Suggested fix:** a short paragraph sanctioning that shape would stop the next porter
     re-deriving it. The audit raised the same point for audits (its Recipe notes, item 1).
5. **The census snippets have no positive control.** `grep -rl <kernel-filename> …` and friends are
   presented as self-validating. Success 2 shows they are not, when combined with the exclusions a
   porter needs, such as quasar.
   - **Suggested fix:** state *"run one query you know must hit"* next to the snippet.

### Confusion

1. **The `cb`-name sweep's "expect zero hits" collides with the off-limits rule, now with more
   hits.** There are 12 hits, all in the off-limits `untilize_with_unpadding.cpp` and
   `device/untilize_with_unpadding_device_operation.cpp`:
   - **7 predate this branch:** device op comments `:144`, `:146`, `:221`, `:273`, and in the
     top-level file the `input_cb_data_format` local (`:103`, `:105`) and the "CB budget" comment
     (`:111`).
   - **5 arrived with #56280:** the output-CB-estimate comment (`:96-98`) and the
     `output_cb_data_format` local (`:104`, `:106`).

   All are reported, not edited.
   - **Suggested fix:** scope the sweep to factory bodies and kernels, as the earlier report
     suggested.
2. **Dead non-CB compile-time args still have no rule, and this branch now does both.** Three are
   dropped:
   - the NDSharded writer's `output_stick_size` and `input_single_tile_size`, per the brief;
   - the Sharded config-d writer's `output_row_size`, per review.

   Two are kept: the SingleCore writer's `unpadded_stick_size`, and config b′'s `aligned_page_size`.
   The latter shares its CTA set with the config b writer, which does read it. A drop is zero-functional-change: it alters only the
   generated `args::` set and the kernel's compile hash. The inconsistency is still worth a line in
   the recipe's Dropped Plumbing section.

## Open items for downstream

### Shared kernel touches

| kernel | relation | rung | remaining legacy consumers |
|---|---|---|---|
| `eltwise/unary/…/reader_unary_interleaved_start_id.cpp` | borrowed | **1 — reused** `…_metal2.cpp` | 5: `reduction/topk`, `experimental/transformer/nlp_create_qkv_heads_falcon7b`, `examples/example` (×2), `examples/example_multiple_return` |
| `eltwise/unary/…/reader_unary_sharded.cpp` | borrowed | **1 — reused** | 4: `experimental/slice_write` (×2), `untilize` ND-identical factory, `sharded_partial/sharded_to_interleaved_partial` |
| `data_movement/untilize/…/compute/untilize.cpp` | borrowed | **1 — reused** | 0 factories. `TestCrossOpCompilation` reads it, so it is kept |
| `data_movement/untilize/…/compute/untilize_variable_num_blocks.cpp` | borrowed | **1 — reused** | 1: `untilize` ND-identical factory |
| `ttnn/kernel/compute/eltwise_copy.cpp` | borrowed | **1 — reused** | 3: `data_movement/copy`, `sharded_partial/sharded_to_interleaved_partial`, and `interleaved_to_sharded_partial` (which binds a different `eltwise_copy.cpp`). The fork's install entry is added by this diff (Handoff 3) |
| `data_movement/sharded/…/reader_unary_nd_sharded_blocks.cpp` | borrowed | **1 — reused** | 0 → **deleted** (sunset) |
| `eltwise/unary/…/reader_unary_interleaved_wh_multicore.cpp` | borrowed | **1 — reused** (#56280's fork) | 0 → **deleted** (sunset). Fork carries the `BACKWARDS` fix |
| `data_movement/untilize/…/compute/untilize_wh.cpp` | borrowed | **1 — reused** (#56280's fork) | 0 → **deleted** (sunset) |
| `…/untilize_with_unpadding/…/writer_unary_stick_layout_wh_multicore.cpp` | **lent** | **1 — reused** (#56280's fork, in this op's directory) | 0 → **deleted** (sunset) |
| `ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_blocks.cpp` | borrowed | **2 — created** `…_metal2.cpp` (this branch) | 0 → **deleted** (sunset); the fork is now listed in `ttnn/sources.cmake` |

Every reused fork also has consumers in other ops (1–7 other factories each). So its named-arg and
binding vocabulary is frozen by more than this op (see the plan's vocabulary table). Three of them,
the BlockInterleaved trio, are shared with `untilize`'s block factory specifically. The fork created
here is bound only by this op. The seven op-owned writers have this op as their sole consumer and
were converted in place.

### Findings — bugs and oddities carried forward unchanged

Every item below is preserved in behaviour. None is fixed.

1. **Dead args still emitted.**
   - SingleCore writer CTA `unpadded_stick_size` (`…_single_core_program_factory.cpp:172`). The
     kernel reads the same quantity as RTA `num_unpadded_X` × element size instead.
   - Sharded config b′ writer CTA `aligned_page_size`: the vector is shared with config b, which
     reads it.
   - SingleCore writer RTA `num_blocks_w_input`, read into an unused local
     (`writer_unary_unpad_dims_split_rows.cpp`). It occupies a dispatch slot.
2. **`device/factories/untilize_with_unpadding_multi_core_shared_variables.hpp` is unreferenced.** It
   is a leftover of the pre-`ProgramDescriptor` era, listed only in `data_movement/CMakeLists.txt`.
   Safe to delete together with that entry.
3. **Stale "CB" vocabulary in off-limits files.**
   - `device/untilize_with_unpadding_device_operation.cpp:144,221,273` refer to the sharded
     factory's `sharded_output_cb_index` CB, which is now `SH_SHARDED_OUT`.
   - `untilize_with_unpadding.cpp:96-111` speaks of output CB estimates and CB budgets (`:96-106`
     from #56280, `:111` older). That reasoning still holds; only the words are stale.
4. **`round_down_32` in `writer_unary_unpad_dims_split_rows.cpp` is never called.**
5. **A self-flagged doubt in the Sharded factory:** *"I am not sure it is correct to ever use the
   shard_spec here"*, above the `out_shard_spec` fallback that substitutes the *input's* shard spec
   when the output has none. The fallback is reachable, from the interleaved-output branches. Worth
   a deliberate answer from the owner.
6. **`untilize_output_dtype` (`data_movement/common/common.hpp`) maps BFLOAT8_B but not BFLOAT4_B.**
   A BFLOAT4_B tiled input would get a ROW_MAJOR BFLOAT4_B output spec. Either that is unreachable
   (and a `TT_FATAL` should say so), or the mapping is incomplete.
7. **The audit's open question stands:** can a 1-byte output element size reach the interleaved →
   BLOCK/WIDTH-sharded path? If it can, the dropped 3rd argument was mis-addressing in legacy, and
   the port's aligned page size is the fix.

### Doc-evolution and carry-over

- **A spec-side `push_buffer_set`.** Two ported block factories now reproduce its sizing inline: this
  op's BlockInterleaved and #56280's `untilize` block factory. The helper exists so those rules
  cannot drift, and each further Metal 2.0 block factory (tilize, tilize_with_val_padding) will
  duplicate them again. A
  `Group<DataflowBufferSpec> make_buffer_pair(const BlockBufferSet&, …)` in `data_movement/common`
  would restore the single source. That is shared-code work.
- **`make_block_plan` is valid only on a cache miss.** `untilize`'s block factory says so in a
  comment; this op's BlockInterleaved factory relies on the same property silently. A matching
  comment would help, but it is out of this diff's scope.

### Test coverage notes

- **MultiCoreInterleaved cache hit with a new input address — closed.**
  - **The gap.** `test_untilize_with_unpadding_spec_factories_program_cache_addr_change` was written
    while only three factories were ported, so it parametrized `single_core`, `multi_core_sharded` and
    `multi_core_nd_sharded` only. MultiCoreInterleaved's refresh was covered only for the output, by
    the #46533 regression test (same input, fresh output). BlockInterleaved's was already covered, by
    `test_untilize_with_unpadding_block_per_node_cb_size` (two iterations, `keep_alive`, cache reuse
    asserted).
  - **Interim cover.** The [routing probe](#routing-probe--the-two-restored-factories-attributed)
    exercised it before the test changed: new input and output addresses, a cache hit and exact
    output, on bf16, fp32 and BFLOAT8_B.
  - **The fix.** `multi_core_interleaved` was added to the regression test's parametrization; the
    interleaved-L1 input it already builds for `single_core` routes there under `use_multicore`.
    Confirmed from the inspector's `kernels.yaml`: the case builds
    `writer_unary_stick_layout_split_rows_multicore.cpp` + `untilize_metal2.cpp`, MultiCoreInterleaved's
    kernels, not the block triple.
- **BFLOAT8_B on BlockInterleaved — closed.** The attribution run showed `test_untilize_bfloat8_b.py`
  building only SingleCore and MultiCoreInterleaved programs, while the BlockInterleaved cache test
  was bf16-only, leaving the brief's item 4 combination on the uncommitted probe alone.
  `test_untilize_with_unpadding_block_per_node_cb_size` is now parametrized over BFLOAT8_B as well,
  so all six wide-row shapes run on both dtypes. Its reference had to change with it: BFLOAT8_B
  quantizes on the way to the device, so the golden is now the device's view of the input rather than
  the original torch tensor, which leaves the comparison exact for both dtypes. Routing was confirmed
  rather than assumed — the inspector's `kernels.yaml` shows the BFLOAT8_B cases binding
  `untilize_wh_metal2.cpp`, `reader_unary_interleaved_wh_multicore_metal2.cpp` and
  `writer_unary_stick_layout_wh_multicore_metal2.cpp`, the BlockInterleaved triple. This matters
  because `enough_space_height` is *more* permissive for BFLOAT8_B (1088 B input tile against the same
  2048 B output tile), so these shapes reach the block factory through the dtype-independent wide-row
  heuristic, not through the space check.
- **The untilize-codegen non-tile-aligned fallback is gone with `build_native_equivalent`.** The
  earlier report's "untested path" caveat is therefore moot.
- **`test_to_layout.py` was run unfiltered.** `-k untilize_with_unpadding` selects only 8 of its 724
  tests, while the op is reached from many more through `to_layout`.

## Verification

### Build

`./build_metal.sh --build-tests`: **exit 0** on every tree tested, including the final one. The
libraries Python loads (`ttnn/ttnn/_ttnn.so` → `build/lib/_ttnncpp.so`, `build/lib/libtt_metal.so`)
were checked to be newer than the last host source edit.

### Legality checks — forced, and proven live

- **Forced:** `skip_validation = false` as the first statement of all nine functions
  `grep -n 'bool skip_validation' tt_metal/impl/metal2_host_api/*.cpp` names. That is
  `SetProgramRunArgs`, `UpdateTensorArgs`, `MergeKernelRunArgsInto`, `UpdateProgramRunArgs`,
  `MergeProgramRunArgs`, `BuildProgramFromSpec`, `MakeProgramFromSpec` and both
  `MakeMeshWorkloadFromSpec[s]`.
- **Proven:** one marker in `BuildProgramFromSpec` and one in `SetProgramRunArgs`.
- **Removed after the runs**, with both files restored to `HEAD`. The tree was then rebuilt, so the
  binaries match the source again. `git diff --name-only $(git merge-base origin/main HEAD) | grep ^tt_metal/`
  is empty, and no code file in the diff matches `METAL2_CHECKS_FORCED|DO NOT COMMIT`. The only match
  is this report's own prose. For the two attribution runs the `[program_spec]` marker temporarily
  also printed `spec.name`; that went with it.

Both markers fired in every session of every run, in equal numbers: one `BuildProgramFromSpec`
and one `SetProgramRunArgs` per Metal 2.0 program the session built. So both translation units were
fresh, and both validators ran on every program. Run 3, the record:

| session | `[program_spec]` | `[program_run_args]` |
|---|---|---|
| gtest `*UntilizeWithUnpadding*` | 1 | 1 |
| `test_untilize_with_unpadding.py` | 373 | 373 |
| `test_untilize.py` | 767 | 767 |
| `test_to_layout.py` | 515 | 515 |
| `test_untilize_bfloat8_b.py` | 152 | 152 |
| `test_tilize_untilize_2D.py` | 231 | 231 |
| nightly `test_untilize.py` | 8 | 8 |
| `test_sharded.py -k untilize_with_unpadding` | 26 | 26 |
| **total** | **2073** | **2073** |

### Tests (Watcher on: `TT_METAL_WATCHER=10`)

Confirmed baseline, agreed with the invoker before relying on it. Each entry ran as its own pytest
session. The table is **run 3, on the final tree**. Its wall times are with a warm kernel cache; the
cold first run took about 22 minutes.

| session | collected | passed | failed | skipped | xfailed | wall |
|---|---|---|---|---|---|---|
| `./build/test/ttnn/unit_tests_ttnn --gtest_filter='*UntilizeWithUnpadding*'` | 1 | 1 | 0 | 0 | 0 | 1 s |
| `tests/ttnn/unit_tests/operations/data_movement/test_untilize_with_unpadding.py` | 385 | 373 | 0 | 6 | 6 | 1 m 18 s |
| `tests/ttnn/unit_tests/operations/data_movement/test_untilize.py` | 844 | 838 | 0 | 4 | 2 | 2 m 55 s |
| `tests/ttnn/unit_tests/base_functionality/test_to_layout.py` (unfiltered) | 724 | 671 | 0 | 45 | 8 | 3 m 33 s |
| `tests/ttnn/unit_tests/base_functionality/test_untilize_bfloat8_b.py` | 176 | 176 | 0 | 0 | 0 | 13 s |
| `tests/ttnn/unit_tests/base_functionality/test_tilize_untilize_2D.py` | 240 | 240 | 0 | 0 | 0 | 11 s |
| `tests/ttnn/nightly/unit_tests/operations/data_movement/test_untilize.py` | 6 | 6 | 0 | 0 | 0 | 8 s |
| `tests/tt_eager/python_api_testing/unit_testing/misc/test_sharded.py -k untilize_with_unpadding` | 12 | 12 | 0 | 0 | 0 | 10 s |
| **total** | **2388** | **2317** | **0** | **55** | **16** | **≈ 8.5 min** |

**Every non-passing outcome is pre-existing, and none is a failure:**

- **`test_untilize_with_unpadding.py`:**
  - 6 skips: "blocked until reshape supports ND-sharded tensors without using `ttnn::experimental::view`".
  - 6 xfails: they fail in `from_torch` **setup** (the bank manager runs out of memory building the
    sharded input), before the op runs.
- **`test_untilize.py`:**
  - 4 skips: a width-sharded shard narrower than a tile, on the single-core path.
  - 2 xfails: the same setup failure.
- **`test_to_layout.py`:**
  - 45 skips: 32 "Modifying logical shape with borrowed buffer is not supported!", 12 Blackhole-only,
    and 1 that needs 8 devices.
  - 8 **strict** xfails (`raises=RuntimeError`): interleaved → ND-sharded output, which the device
    operation's validation rejects by design. A pass, or any other exception, would have been
    reported as a failure.

Watcher reported nothing in any session of any run: no `0xdeadc0de`, and no Watcher-attributed
error.

**Against the earlier five-of-five run.** That run had 1195 passed across its two files. The same two
files now give 1211 passed:
- +3 from the program-cache regression test added on this branch since;
- +7 from the coverage commit: 6 BFLOAT8_B block-cache cases and the `multi_core_interleaved`
  regression case;
- +6 from `test_untilize.py` cases #56280 added on `main`.

No existing outcome changed category.

**Three runs, and what the last one attributes.**

- **Runs 1 and 2** were on the tree before the coverage commit. They were identical test by test:
  every one of 2380 pytest outcomes matched.
- **Run 3, the record,** is on the final tree, with the `[program_spec]` marker temporarily printing
  `spec.name`. Every test the two trees share has the same outcome as in run 2. The only difference
  is the coverage commit's 7 new tests, all passing:
  - six `…_block_per_node_cb_size[dtype=BFLOAT8_B-…]` cases;
  - `…_program_cache_addr_change[factory=multi_core_interleaved]`.

Run 3 shows which of this op's factories the baseline exercises (programs built, i.e. cache misses):

| session | SingleCore | MultiCoreInterleaved | MultiCoreSharded | BlockInterleaved | NDSharded | other ops' Metal 2.0 programs |
|---|---|---|---|---|---|---|
| `test_untilize_with_unpadding.py` | 1 | 50 | 33 | 12 | 277 | 0 |
| `test_untilize.py` | 0 | 2 | 0 | 2 | 0 | 763 |
| `test_to_layout.py` | 0 | 82 | 53 | 6 | 2 | 372 |
| `test_untilize_bfloat8_b.py` | 64 | 64 | 0 | 0 | 0 | 24 |
| `test_tilize_untilize_2D.py` | 20 | 20 | 0 | 0 | 0 | 191 |
| nightly `test_untilize.py` | 0 | 2 | 0 | 6 | 0 | 0 |
| `test_sharded.py -k untilize_with_unpadding` | 0 | 2 | 10 | 0 | 0 | 14 |
| **total** | **85** | **222** | **96** | **26** | **279** | 1364 |

Every factory is exercised. BlockInterleaved is still the thinnest, at **26** programs:
- 12 from `…_block_per_node_cb_size` (six shapes × bf16 and BFLOAT8_B);
- 6 from the nightly wide-row tests (bf16);
- 2 via `ttnn::untilize`;
- 6 via `to_layout`.

Against run 2, the BlockInterleaved count rose by exactly the six BFLOAT8_B cases, and
MultiCoreInterleaved's by the one new regression case.

**BFLOAT8_B on BlockInterleaved is now committed coverage.** Those six shapes reach the block factory
through the dtype-independent wide-row heuristic. The one route no committed test pins is the
*space-check* route: a BFLOAT8_B shape that #56280's output-dtype sizing pushed out of
`enough_space_height`. Only the probe's `bf8b_shift` exercises that. Selection happens before the
factory runs, and the factory builds the same kinds of program either way, so the residual is a
selection-logic gap in the device operation, not a factory-coverage gap.

### Routing probe — the two restored factories, attributed

This probe ran **before** the coverage commit, and is how the two gaps that commit closes were
found. At the time, the baseline showed the op green but not *which* factory each case reached, and
the brief asked specifically for BFLOAT8_B on the block path (brief item 4). The probe is scratch and
not committed. For attribution, the `[program_spec]` marker was temporarily extended to print
`spec.name`, and removed with the rest of the scaffolding. The probe runs each case in its own
process and calls the op twice, keeping every tensor alive, so the second call is a program-cache hit
with **both** tensors at new addresses. It then compares the output with the host's view of the
device input:

| case | input dtype | shape → output | factory built (from `spec.name`) | numerics | 2nd call |
|---|---|---|---|---|---|
| `bf8b_shift` | BFLOAT8_B | (1,1,2048,16384) → (1,1,2000,16300) | `…_multi_core_block_interleaved` | exact | cache hit, new in/out addresses |
| `bf8b_wide` | BFLOAT8_B | (1,1,32,7328) → (1,1,30,7300) | `…_multi_core_block_interleaved` | exact | cache hit |
| `bf8b_wide_cliff` | BFLOAT8_B | (1,1,160,6304) → (1,1,150,6301) | `…_multi_core_block_interleaved` | exact | cache hit |
| `bf8b_mci` | BFLOAT8_B | (1,1,2048,256) → (1,1,2000,250) | `…_multi_core_interleaved` | exact | cache hit |
| `bf16_mci` | BFLOAT16 | (1,1,256,128) → (1,1,201,101) | `…_multi_core_interleaved` | exact | cache hit |
| `fp32_mci` | FLOAT32 | (1,1,256,128) → (1,1,201,101) | `…_multi_core_interleaved` | exact | cache hit |
| `fp32_wide` | FLOAT32 | (1,1,32,7328) → (1,1,30,7300) | `…_multi_core_block_interleaved` | exact | cache hit |

- **`bf8b_shift` is the brief's case.** Its row is 512 tiles wide, and it is tall enough that the
  wide-row heuristic prefers MultiCoreInterleaved. So it reaches BlockInterleaved only through
  `enough_space_height`:
  - With #56280's output-dtype sizing, the estimate is 512 × (1088 + 2048) B, which no longer fits
    in L1.
  - With the old input-dtype sizing, it was 512 × (1088 + 1088) B, which did fit.

  That makes it a shape #56280 moved from MultiCoreInterleaved to BlockInterleaved, and it is
  bit-exact on the block path.
- **The `fp32` rows** run with `fp32_dest_acc_en = true`, so the `UnpackToDest` entry and
  `enable_32_bit_dest` are live on both restored factories.
- **The `mci` rows and the BFLOAT8_B block rows** were the interim cover for the two gaps under Test
  coverage notes. The coverage commit has since made both permanent; see run 3 above.
  `bf8b_shift`'s space-check route is the one thing still covered only here.

### Anti-pattern self-audit

Scope: the op directory (26 `.cpp`/`.hpp`, of which 5 are factories and 8 are kernels) plus the 9
forks it binds outside the directory. Each result is reported as hits / files scanned.

| check | result |
|---|---|
| No buffer address in run-args (`address()`, `emplace_runtime_args`, `Buffer*`) | **0 / 5** factories |
| No CB indices / legacy CB API in factories (`CBIndex`, `c_N`, `buffer_index`, `CBDescriptor`, `CBFormatDescriptor`, `.cbs`, `CircularBuffer`) | **0 / 5** |
| No `TensorAccessorArgs`, positional `get_compile_time_arg_val` / `get_arg_val` / `get_common_arg_val`, or `CircularBuffer` in any bound kernel | **0 / 17** kernels |
| No `cb` in DFB names, spec names or host variables | **0** in factories and kernels. 12 hits in the two off-limits files, reported under Confusion 1 |
| Conditional bindings follow the pattern | `SH_SHARDED_OUT` and the BI cliffrow pair are bound only by `KernelSpec`s built under the same condition. No `#ifdef` is needed (see the plan) |
| No `.id` extraction on `dfb::` handles | **0 / 17** |
| No CTA→RTA demotion | none. Per-group CTAs are preserved on all 2 + 4 same-source compute splits |
| No unnecessary multi-binding flag, never stacked with a self-loop | **0** occurrences of `allow_instance_multi_binding` in code |
| All CTAs named | yes: every `compile_time_args` is `{{name, value}, …}` |
| No nameable argument smuggled into varargs | two vararg blocks, both genuine indexed collections. Their leading scalars are named |
| No forced-legality scaffolding in the diff | **clean after removal.** `git diff --name-only $BASE \| grep ^tt_metal/` prints nothing. `git diff $BASE \| grep -E 'METAL2_CHECKS_FORCED\|DO NOT COMMIT'` hits only this report's own description of the markers, and zero code files |
| No ephemeral doc cited from code | **0 `.md` citations / 27** changed code files (23 present, 4 deleted). The fifth deleted kernel shows as a rename into its fork. The one repo-path comment citation resolves on `origin/main` |
| Every legacy `TT_FATAL` / `TT_ASSERT` / `TT_THROW` accounted for | **census clean.** Every code file's count equals `main`'s. The only deltas are mentions inside the `METAL2_*.md` docs |
| Every `hw_config` reproduces the legacy resolved values | DM: plain Reader/Writer defaults → TTNN helpers. Compute: Style B, with only `enable_32_bit_dest` and `unpack_modes` set, both gated on `fp32_dest_acc_en` (per-instance input DFB in BI) |
| Every `KernelSpec`'s `opt_level` matches | legacy sets none (0 hits). `grep -n opt_level` → **5** lines, one per factory, each inside the single construction site (two of them lambdas) of that factory's compute specs, so **every compute `KernelSpec` carries `O3`**. No DM spec sets one |
