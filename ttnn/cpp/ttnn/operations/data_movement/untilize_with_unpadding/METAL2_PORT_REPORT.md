# Metal 2.0 Port Report — `data_movement/untilize_with_unpadding`

## Outcome

**`PORTED` (partial, by owner request) — 3 of 5 factories on Metal 2.0; 2 reverted to
`create_descriptor`.**

### Current concept split

| factory | concept | reachable from untilize codegen's live-L1 fallback? |
|---|---|---|
| `MultiCoreInterleaved` | `ProgramDescriptorFactoryConcept` — **reverted** | **yes** |
| `MultiCoreBlockInterleaved` | `ProgramDescriptorFactoryConcept` — **reverted** | **yes** |
| `SingleCore` | `ProgramSpecFactoryConcept` | no |
| `MultiCoreSharded` | `ProgramSpecFactoryConcept` | no |
| `MultiCoreNDSharded` | `ProgramSpecFactoryConcept` | no |

The op ran fully ported (all five factories, green — results below). Owners then asked to keep the
Metal 2.0 conversion only where untilize codegen's native fallback does **not** consume the factory,
so the two consumed ones went back to the descriptor API. **The port is not descoped** — the three
remaining factories keep every Metal 2.0 construct the port introduced (typed tensor bindings,
DFB bindings, borrowed-memory DFBs, the `c_17` self-loop, named args, the two vararg blocks).

### Which factories codegen actually consumes — the derivation

`build_native_equivalent`'s non-tile-aligned branch
(`untilize/codegen/untilize_codegen_program_factory.cpp:439-446`) calls
`UntilizeWithUnpaddingDeviceOperation::select_program_factory` with `use_multicore = true` hardcoded,
on a tensor that already passed `supported_by_codegen()` — which
`untilize_codegen_device_operation.cpp:25-31` asserts as a hard `TT_FATAL` precondition. That
predicate rejects **sharded input** (`untilize_codegen_supported.cpp:49-51`) and **sharded output**
(`:52-54`). Walking `select_program_factory` under those constraints:

| branch | condition | reachable? |
|---|---|---|
| `MultiCoreSharded` / `MultiCoreNDSharded` | `input.memory_config().is_sharded()` | **no** — rejected by `supported_by_codegen` |
| `MultiCoreInterleaved` (via sharded output) | `output_mem_config.is_sharded()` | no — rejected |
| `SingleCore` | `!use_multicore` | **no** — codegen hardcodes `true` |
| `MultiCoreBlockInterleaved` | `!enough_space_height`, or the wide-row heuristic | **yes** |
| `MultiCoreInterleaved` (fallthrough) | neither of the above | **yes** |

So exactly **two** factories are live on that path, and those are the two reverted. This is a
derivation from the gating code, not an inference from test coverage — the path itself is untested
(see [Handoff points](#handoff-points) item 1).

### The guard stays — dropping it does not compile

The request included dropping the `if constexpr (requires { … })` + `TT_THROW` guard added by the
earlier consumer fix. **That is not possible while any factory remains ported, and the build proves
it.** `std::visit` instantiates its callable for **every** alternative of the five-alternative
`program_factory_t` variant — including the three that codegen can never select at runtime. With the
guard removed:

```
untilize_codegen_program_factory.cpp:448:75: error: no member named 'create_descriptor' in
    'ttnn::prim::UntilizeWithUnpaddingSingleCoreProgramFactory'
untilize_codegen_program_factory.cpp:447:16: error: no matching function for call to 'visit'
```

The guard was therefore restored. **The good news is that the partial revert makes it behaviorally
inert:** its `if constexpr` true-branch now covers both runtime-reachable factories, so the
non-tile-aligned fallback builds a native program again exactly as it did pre-port, and the `TT_THROW`
else-branch is instantiated only for the three unreachable alternatives. **The regression this report
previously recorded is resolved by the revert itself, not by removing the guard.** A comment at the
call site now records this so the guard is not deleted again; the only way to drop it is to revert all
five factories, which would descope the port.

### Test results

`./build_metal.sh --build-tests` completes with **0 errors** in both states. The sentinel run is
**identical in both**, outcome for outcome:

| test file | collected | passed | failed | skipped | xfailed |
|---|---|---|---|---|---|
| `tests/ttnn/unit_tests/operations/data_movement/test_untilize_with_unpadding.py` | 375 | **363** | **0** | 6 | 6 |
| `tests/ttnn/unit_tests/operations/data_movement/test_untilize.py` | 838 | **832** | **0** | 4 | 2 |
| **total** | **1213** | **1195** | **0** | **10** | **8** |

- fully ported (5/5): `1195 passed, 10 skipped, 8 xfailed in 806.58s`, `TT_METAL_WATCHER=10` on, no
  watcher trip (no `0xdeadc0de`, no device-side assertion).
- partial revert (3/5, current): `1195 passed, 10 skipped, 8 xfailed in 690.78s`, Watcher **off** (the
  current recipe revision does not ask for it, and the sentinel command specified did not set it).

> **Correction to an earlier revision of this report.** It listed 361 / 834 passed and "8 xfails, all
> in `test_untilize_with_unpadding.py`". That was wrong: the per-file passes were derived by assuming
> every xfail landed in the unpadding file. Re-reading both logs, the split is **6 xfails in
> `test_untilize_with_unpadding.py` and 2 in `test_untilize.py`** in *both* runs, so the per-file
> passes are 363 / 832. The totals (1195 / 0 / 10 / 8) were correct and unchanged.

`test_untilize.py` is included because it exercises the untilize-codegen paths that consume these
factories — the paths this revert is about.

All 18 non-passing outcomes are pre-existing and unrelated to the port or the revert:
- **6 skips** — `test_untilize_with_unpadding.py:372`, "blocked until reshape supports ND-sharded
  tensors without using `ttnn::experimental::view`".
- **4 skips** — `test_untilize.py:183`, "Width sharded case results in shard with width < tile width,
  which is not supported in single core implementation."
- **8 xfails** — 6 × `test_untilize_with_unpadding_multi_core_nd_sharded_to_interleaved` and
  2 × `test_untilize_multi_core_nd_sharded_to_interleaved`, all on the same
  `shard_core_grid={[0-0 - 0-2]}` / `[4, 4, 256, 512]` shape and all failing in *test setup*
  (`from_torch failed while building sharded tensor: TT_FATAL @
  tt_metal/impl/allocator/bank_manager.cpp:462`), i.e. before the op under test is reached.

**The legality checks were provably live for the fully-ported run.** The log carries source locations,
so the two markers are distinguishable rather than merely counted:
`METAL2_CHECKS_FORCED (program_spec.cpp:2950)` **1118** times and
`METAL2_CHECKS_FORCED (program_run_args.cpp:565)` **1118** times, interleaved as adjacent pairs —
both translation units fresh, `ValidateProgramSpec` and the run-args validation both running on every
one of the 1118 programs those tests construct. (Counting the bare marker string would not have shown
this: both sites log identical text, so a single live TU would look the same at half the count.)

**That forcing scaffolding is no longer in the tree.** It was removed between sessions, and the
recipe revision now in `docs/…/metal_2.0/` has no forced-legality-check step at all (zero mentions of
`skip_validation`), so it was not re-applied. Consequence for the post-revert run: the checks are in
whatever state TTNN's own `skip_validation` handling leaves them, and this report can no longer
*prove* they ran. Re-apply the forcing if that proof is wanted again — see
[Handoff points](#handoff-points) item 2.

## Provenance

- **Recipe docs (the port itself):** `9c1a0466220 2026-09-07 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`
- **Audit docs (inherited):** `9c1a0466220 2026-09-07 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`
- **Recipe docs (this partial-revert pass):** the `docs/…/metal_2.0/` tree was **replaced and is now
  untracked** (`git log` over that path returns nothing; `git status` shows `?? docs/…/metal_2.0/`),
  so no commit can be pinned. The revision present is a restructured, smaller one
  (`ai/port_op_to_metal2_recipe.md`, 841 lines). **It disagrees with the code and with the revision
  the port was executed against, in two ways worth flagging** — see
  [Friction → Gaps](#gaps) for the detail:
  1. it names the target concept `MetalV2FactoryConcept`, which does not exist in
     `ttnn/api/ttnn/operation_concepts.hpp` (the code has `ProgramSpecFactoryConcept`);
  2. it instructs the port to **delete** a custom `compute_program_hash`, where the revision this
     port followed forbade touching it. Moot here (this op has no custom hash) but a live
     contradiction for other ports.
  Doc links inside these artifacts were retargeted to the new filenames where an equivalent exists.

## TTNN ProgramFactory

### Concept realized

`ProgramSpecFactoryConcept` (plain) on **three of five** factories after the owner-requested partial
revert — `SingleCore`, `MultiCoreSharded`, `MultiCoreNDSharded`. `MultiCoreInterleaved` and
`MultiCoreBlockInterleaved` are back on `ProgramDescriptorFactoryConcept`; see
[Outcome](#outcome) for which factories codegen consumes and why those two. The audit's concept
choice was not re-decided — the scope was narrowed by owners after the port landed.

For the three ported factories, `create_descriptor` became `create_program_artifacts` returning
`ttnn::device_operation::ProgramArtifacts{spec, run_params}`; `op_owned_tensors` is left defaulted
(the op allocates no device tensors of its own). **The mixed variant is legal and dispatches
per-factory at runtime** — that is the recipe's own "atomic unit is one ProgramFactory" property,
exercised here in the reverse direction.

The op already had a `program_factory_t` variant, so [exception 3](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port_op_to_metal2_ttnn_factory.md#3-give-a-direct-descriptor-op-a-conventional-program-factory)
(direct-descriptor conversion) did not apply — the port is a method swap inside the existing structs.

### Device-op-class edits

- **Pybind entry points removed:** none. `untilize_with_unpadding_nanobind.cpp` binds only the
  user-facing op; no `create_descriptor` was ever exposed. **This port carries no user-visible API
  change**, as the brief predicted.
- **Custom `compute_program_hash`:** none — the op uses the default reflection-based hash, and there
  is no backdoor `attribute_values` / `to_hash` either. Nothing to preserve, nothing touched.
- **`override_runtime_arguments`:** none to translate.
- `device/untilize_with_unpadding_device_operation.cpp` and `untilize_with_unpadding.cpp` are
  **byte-identical** to their pre-port state.

### Open items

- **Relaxation candidates:** none identified. The audit declared `none` on all five rows, and nothing
  during construction suggested a kernel that would tolerate a relaxed `TensorSpec` match. Strict
  matching retained everywhere.
- **Capability the op would benefit from:** a spec-side twin of the shared `push_buffer_set` helper —
  see [Open items for downstream](#open-items-for-downstream).

## Handoff points

1. **RESOLVED — the codegen non-tile-aligned regression is gone, and the path is still untested.**
   The earlier fully-ported state made untilize codegen's non-tile-aligned L1 fallback throw, because
   the factory it selects had no `create_descriptor`. The partial revert puts both selectable
   factories (`MultiCoreInterleaved`, `MultiCoreBlockInterleaved`) back on the descriptor API, so that
   fallback builds a native program again, exactly as pre-port. **No behavioral regression remains.**

   Two things still worth an owner's attention:
   - **The guard cannot be deleted** while the other three factories stay ported — `std::visit`
     instantiates over all five variant alternatives. Build evidence and the reasoning are in
     [Outcome](#outcome); a comment at the call site now records it.
   - **The path has no test coverage in either state.** Reaching `build_native_equivalent` needs live
     L1 occupancy high enough that *no* codegen CB plan fits (`get_max_l1_space` reads
     `lowest_occupied_compute_l1_address`, sampled per dispatch — `kUsableL1Note`, `:509-517`) *and* a
     non-tile-aligned logical shape. No unit test constructs that occupancy, so the branch's only
     coverage is the compiler. That is why the fully-ported breakage surfaced as a build error rather
     than a test failure — and why the fix's correctness rests on the derivation above, not on a
     green run. **Owner: untilize-codegen.** A test that pins L1 occupancy would retire this caveat.

   **A `create_descriptor` shim on the ported factories is still NOT an option** — worth keeping on
   record because it is the obvious-looking alternative to reverting:
   `ProgramSpecFactoryConcept` requires `!ProgramDescriptorFactoryConcept`
   (`ttnn/api/ttnn/operation_concepts.hpp:137-140`), and `ProgramDescriptorFactoryConcept` is
   satisfied by the mere *presence* of `create_descriptor`. A factory declaring both is classified as
   descriptor-concept and silently keeps running the legacy path — the port would appear to land while
   changing nothing.

2. **The forced-legality scaffolding is gone from the tree, and this recipe revision no longer asks
   for it.** The fully-ported run proved both validator TUs live (1118 markers each). That scaffolding
   — `skip_validation = false` at all 9 `grep -n 'bool skip_validation'` sites in
   `tt_metal/impl/metal2_host_api/program_{run_args,spec}.cpp`, plus one `METAL2_CHECKS_FORCED` marker
   per file — was removed between sessions and **not** re-applied, because the recipe revision now in
   `docs/…/metal_2.0/ai/port_op_to_metal2_recipe.md` contains no forced-legality step (zero mentions
   of `skip_validation`). So the post-revert run cannot prove the checks ran. If that proof is wanted,
   re-apply the forcing and expect **two distinguishable** markers (they share text; the source
   location in the log is what tells the two TUs apart). **Owner: whoever owns the recipe** — the
   forcing step existed in the revision this port was executed against and is absent from the current
   one; if it was dropped deliberately, this report's earlier proof method is no longer expected of
   porters, and if not, the step needs restoring.

3. **The BACKWARDS page-id fix is no longer in the working tree — it lives only in history.** Deleting
   the three orphaned forks (below) removes the file that carried it, which means the content of
   commit **`b334c1ae941` "Fix BACKWARDS page id loop"** (mcw-anasuya, 2026-09-09) is no longer in the
   tree. Recorded here verbatim so it is recoverable without archaeology.

   The fix was in `eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_wh_multicore_metal2.cpp`.
   The **legacy original still carries the broken form**, and it is a genuine bug: `dim` is `uint32_t`,
   so `-third_dim` is a huge unsigned value and `0 > huge` is false — the BACKWARDS loop never
   executes at all, and `-start_id` wraps.

   ```diff
   -#ifdef BACKWARDS
   -    for (uint32_t dim = 0; dim > -third_dim; dim--) {
   -        for (uint32_t c = 0; c > -single_block_size_col_arg; c--) {
   -            for (uint32_t r = 0; r > -single_block_size_row_arg; r--) {
   -                uint32_t tile = -start_id + dim * num_tiles_per_2d + c * total_tiles_per_row + r;
   -#else
        for (uint32_t dim = 0; dim < third_dim; dim++) {
            for (uint32_t c = 0; c < single_block_size_col_arg; c++) {
                for (uint32_t r = 0; r < single_block_size_row_arg; r++) {
   -                uint32_t tile = start_id + dim * num_tiles_per_2d + c * total_tiles_per_row + r;
   +                const uint32_t offset = dim * num_tiles_per_2d + c * total_tiles_per_row + r;
   +#ifdef BACKWARDS
   +                const uint32_t tile = start_id - offset;
   +#else
   +                const uint32_t tile = start_id + offset;
    #endif
   ```

   **It is latent, not live.** Nothing defines `BACKWARDS` for this kernel — the only definer in the
   tree is `data_movement/copy/device/copy_same_memory_config_program_factory.cpp:137`, and it binds
   `copy/device/kernels/reader_unary_start_id.cpp`, a different file. So the branch is dead code in
   both copies today and deleting the fork causes no behavior change. But the bug is real and will
   bite the first consumer that defines `BACKWARDS`.

   **Owner: `eltwise/unary` kernel owners.** The clean route is applying the same correction to the
   legacy `reader_unary_interleaved_wh_multicore.cpp` as a standalone one-line change — deliberately
   *not* done here, because that kernel is shared (`data_movement/untilize`'s block factory binds it)
   and lives outside this op's writeable surface, so it needs its owner's review rather than a
   revert-shaped port commit.

3. **Audit gap — the brief's shared-kernel table missed a *lent* kernel.** The brief lists
   `device/kernels/dataflow/writer_unary_stick_layout_wh_multicore.cpp` among "8 op-owned writers …
   none is dead code" and does not flag it as shared. It is: `data_movement/untilize`'s block factory
   binds it by full path
   (`ttnn/cpp/ttnn/operations/data_movement/untilize/device/factories/untilize_multi_core_block_program_factory.cpp:150-152`).
   Converting it in place would have broken that op. Handled as a rung-2 shared kernel (fork beside
   the original, pointer comment added). **Owner: the audit tooling / next auditor** — the census in
   the audit brief appears to have been run only for kernels *outside* the op directory, so the
   *lent* direction (a kernel inside the op's own directory that other ops bind) was not swept. That
   is the exact failure mode the [shared-kernel Caution](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/metal2_port_patterns.md#caution-porting-a-shared-kernel)
   warns about ("Nothing about the path warns you").

4. **`num_runtime_varargs_per_node` — a `[[deprecated]]` API this port newly depends on.** The
   MultiCoreInterleaved writer's `BlockRep` payload is a genuinely per-core-variable-length vararg
   block (5 values per run, run count varying per core and per shape), and the *only* construct that
   reproduces the legacy per-core RTA layout exactly is the per-node vararg-count override on
   `KernelAdvancedOptions`. Its header comment says *"This feature is truly bizarre. It will be
   removed from the API once existing uses are refactored to avoid it."* — this port adds an existing
   use. **Owner: the Metal 2.0 API team.** The alternatives considered and rejected: declaring a
   scalar `num_runtime_varargs` at the per-core maximum and zero-padding every shorter node (changes
   the dispatch payload size on most cores, which the port is not entitled to do), or restructuring
   the kernel's group walk (kernel-logic surgery, out of scope). If the field is removed before a
   typed-array replacement lands, this factory needs a plan.

## Successes

- **[Caution: Porting a shared kernel](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/metal2_port_patterns.md#caution-porting-a-shared-kernel)
  fired correctly, twice.** First on the six rung-1 forks: reading each fork *before* writing the
  factory made the binding vocabulary a constraint rather than a choice, and it is not the vocabulary
  this op's locals would have suggested — the untilize compute forks use `dfb::src`/`dfb::out` while
  the readers use `dfb::in`, so the sharded factory binds the *same* `SH_IN` buffer under the name
  `in` on the reader and `src` on the compute kernel
  (`…_multi_core_sharded_program_factory.cpp:352-357`). Deriving those names from this op instead
  would have silently broken every other consumer of the forks.
  Second, and more valuable: the entry's insistence that the *lent* direction is invisible from the
  path ("Nothing about the path warns you: the file sits inside your writeable surface, so converting
  it in place feels safe, and it breaks every borrower the moment you do") is the only reason I ran
  the census on the op's **own** writers at all — the brief said they were exclusively this op's. One
  of them wasn't. See Handoff point 3.

- **The `constexpr` carve-out in CB→DFB whitelist §A *(the CB→DFB whitelist was a standalone doc at port time; the current docs tree has folded it away)*
  is keyed on exactly the right thing.** Two `get_tile_size` sites in this op, and the rule splits
  them correctly on the legacy declaration alone:
  `writer_unary_unpad_width_16_sharded.cpp:23` was `constexpr` and feeds a `static_assert` and two
  `NOC_MAX_BURST_SIZE` template arguments → kept the free-function form with the token,
  `get_tile_size(dfb::out)`; `reader_unary_interleaved_wh_multicore.cpp:27` was plain `const` → moved
  onto the object, `dfb.get_tile_size()`. Reaching for the member getter at the first site would not
  have compiled; the rule got there without needing the build I could not run.

- **[Two-toucher / self-loop endpoint-assignment procedure](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/metal2_port_patterns.md#pattern-two-toucher-dfb--assign-1p1c-dual-instance-work-split),
  re-derived rather than transcribed.** The census agreed with the brief everywhere: `c_17` is a
  genuine one-toucher (the writer `reserve_back`s, fills by write pointer, `push_back`s; nothing
  drains it) → self-loop; every other buffer is an ordinary 1P+1C. The brief's "Watch for" note about
  the BlockInterleaved factory's same-source multiplicity was also correct and load-bearing — those
  are **disjoint-node** work splits, not same-grid two-touchers, so no 1P+1C assignment and no
  multi-binding flag. **`allow_instance_multi_binding` appears nowhere in this port.**

- **The `opt_level` "absent line" warning earned its emphasis.** The recipe's insistence that this is
  the field which survives an otherwise careful port because there is *nothing to read and object to*
  is accurate: `grep -n opt_level` over the five legacy factories returns **zero** hits, so nothing in
  the legacy source hints that compute kernels resolve to `O3`. All 10 compute `KernelSpec`s now carry
  an explicit `O3`; the DM specs correctly carry nothing.

## Friction

### Gaps

- **The recipe tree was swapped mid-work, and the revision now present contradicts both the code and
  the revision the port followed.** `docs/…/metal_2.0/` is now **untracked** and restructured
  (`ai/port_op_to_metal2_recipe.md`, 841 lines, replacing `ai/port/metal2_port.md`). Two
  contradictions, found by checking the code rather than trusting the doc:
  1. **Concept name.** The current recipe calls the target `MetalV2FactoryConcept`.
     `ttnn/api/ttnn/operation_concepts.hpp:138` defines `ProgramSpecFactoryConcept`; no
     `MetalV2FactoryConcept` exists anywhere in the tree. A porter following this revision literally
     would look for a concept that isn't there.
  2. **Custom `compute_program_hash`.** The current revision says the port **deletes** it
     (`:155`, `:297`, `:617`). The revision this port followed said the opposite in the strongest
     terms — leave it alone, touching it is a scope violation. These cannot both be right, and the
     difference is a behavior change (the op's cache-equivalence class). Moot for this op (no custom
     hash) but decidable only by the doc owner.

  Also absent from the current revision: the **forced-legality-check** step (zero mentions of
  `skip_validation`), which the port's verification leaned on to prove the validator was live. See
  [Handoff points](#handoff-points) item 2. **Suggested fix:** track the docs tree in git so a port
  can pin a provenance line, and reconcile the concept name and the custom-hash instruction against
  the code before the next porter reads either.

- **The recipe has no guidance for reverting a landed port.** This pass was a partial revert —
  narrowing a five-factory port to three because a consumer needs the descriptor API on two of them.
  Nothing in the recipe covers un-porting, and the one non-obvious hazard is not written down
  anywhere: a consumer that `std::visit`s the op's `program_factory_t` needs **every** alternative to
  satisfy the API it calls, so a mixed-concept variant only compiles behind an
  `if constexpr (requires { … })` guard. That makes "which factories can this consumer actually
  select?" a *compile-time* question about the whole variant, not just a runtime-reachability one —
  the distinction that made "drop the guard" impossible here. **Suggested fix:** a short
  "partial ports and consumers" note in the recipe, stating that a legacy consumer visiting the
  variant pins every alternative to the descriptor API unless guarded.

- **The recipe has no answer for a per-node-variable vararg count.** [Caution: Avoid varargs](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/metal2_port_patterns.md#caution-avoid-varargs-unless-absolutely-necessary)
  and the [migration guide](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_migration_guide.md#programrunargs)
  both describe varargs as a single `num_runtime_varargs` count, and the guide's worked example is a
  CTA-bounded shape where the count is uniform. Neither mentions `num_runtime_varargs_per_node`, which
  is the only construct that fits a payload whose length varies per core — the shape this op's
  MultiCoreInterleaved writer has. I found it by reading `advanced_options.hpp`, which is exactly what
  ["go to the headers first"](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port_op_to_metal2_recipe.md#read-this-first)
  promises — but a porter who trusted the docs alone would have reached for a max-count-plus-padding
  workaround, which silently changes the dispatch payload on most cores. **Suggested fix:** one
  sentence in the varargs caution naming the per-node override and its deprecation status, so the
  choice is made deliberately rather than discovered.

- **No guidance on a *dead* compile-time arg.** Five CTAs in this op are emitted by the host and never
  read by the kernel (listed under [Open items](#open-items-for-downstream)). The recipe covers dead
  *CBs* explicitly ("build no spec, drop the allocation and any dead CTA carrying its index") but says
  nothing about a CTA that is dead on its own. In the positional world it was invisible plumbing; in
  the named world the porter must decide whether to declare a name nothing reads. I kept all five
  (uniform, faithful, no per-site judgment — an unread named CTA lowers to an unused
  `constexpr experimental::CtaVal<uint32_t>` in the generated header and costs nothing), but the
  opposite choice is just as defensible, and two porters will split. **Suggested fix:** a line in
  [Dropped Plumbing](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port_op_to_metal2_recipe.md#dropped-plumbing)
  stating which way to go. The decision matters more than usual when the CTA lands in a **shared
  fork's** interface, where it becomes every future consumer's obligation — as
  `args::output_row_size` now is in `writer_unary_stick_layout_interleaved_blocks_metal2.cpp`.

- **A shared host-side helper that emits legacy descriptors has no migration story.** The
  BlockInterleaved factory's circular buffers were built by
  `ttnn::operations::data_movement::push_buffer_set` (`data_movement/common/common.cpp:795-850`),
  which takes a `ProgramDescriptor&`. It is out of this port's writeable surface, so the two
  `DataflowBufferSpec`s are now built inline in the factory, duplicating its sizing rules. The helper
  exists *precisely* to stop those rules drifting between the four block factories (its own comment:
  "a private copy can drift and reintroduce the corruption the split prevents"), so the port has
  reintroduced the drift risk it was written to prevent — correctly, per the scope boundary, but the
  recipe offers no route for this shape. See [Open items](#open-items-for-downstream).

### Confusion

- **The `cb`-name sweep's "expect zero hits" collides with the off-limits rule.** The sweep flags
  `input_cb_data_format` (a legitimate rename — done) but also four stale `CB` comments in
  `device/untilize_with_unpadding_device_operation.cpp` and `untilize_with_unpadding.cpp`, which
  [Host-side: stay in the lane](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port_op_to_metal2_recipe.md#host-side-stay-in-the-lane)
  forbids touching. The self-audit item says "Expect **zero** hits … every hit is a real leftover",
  which reads as a hard gate; the resolution (the off-limits rule is the more specific one, so those
  hits are *reported*, not fixed) took a re-read of both sections to be confident about. **Suggested
  fix:** scope the sweep to the factory bodies and kernels, and say the device-op class is excluded
  and reported instead.

- **Near-miss on run-arg ordering, caught by reading the legacy loop twice.** The MultiCoreInterleaved
  writer's `start_stick_id` is pushed **before** the inner loop that advances `row_start_id` past that
  core's own blocks. Restructuring the legacy `RTArgList` build into `AddRuntimeArgsForNode` moves the
  arg emission to the *end* of the loop body, which silently shifts every core's start row by its own
  block count. Caught and fixed with an explicit `core_row_start_id` capture
  (`…_multi_core_interleaved_program_factory.cpp:236-238`). The recipe's advice to "keep the legacy
  per-node loop as-is and let the helper transpose" is right, but the helper's natural call position
  is not always where the legacy pushed the value. **Suggested fix:** a warning beside the
  `AddRuntimeArgsForNode` example that a legacy `push_back` interleaved with mutation of the value it
  pushed must keep its original position, not migrate to the end of the loop.

## Open items for downstream

### Shared kernel touches

Ten kernels touched during the port. None was modified in place. **After the partial revert, three
of the four created forks were deleted** — the two reverted factories bind the legacy originals
again, leaving those forks with zero consumers. Their pointer comments in the legacy originals were
reverted with them, so no comment now advertises a fork that does not exist. The forks are recoverable
from commit `a947a3cdf5d` (and the reader's BACKWARDS fix from `b334c1ae941` — Handoff 3).

| kernel | relation | rung taken | remaining unmigrated consumers |
|---|---|---|---|
| `eltwise/unary/…/reader_unary_interleaved_start_id.cpp` | borrowed | **1 — reused** `…_metal2.cpp` (no new file) | `examples/example`, `examples/example_multiple_return`, `experimental/transformer/nlp_create_qkv_heads_falcon7b`, `reduction/topk` |
| `eltwise/unary/…/reader_unary_sharded.cpp` | borrowed | **1 — reused** | `data_movement/tilize`, `data_movement/untilize`, `data_movement/sharded_partial/sharded_to_interleaved_partial`, `experimental/slice_write` |
| `data_movement/sharded/…/reader_unary_nd_sharded_blocks.cpp` | borrowed | **1 — reused** | none (this op was the only consumer) |
| `data_movement/untilize/…/compute/untilize.cpp` | borrowed | **1 — reused** | `data_movement/fold` |
| `data_movement/untilize/…/compute/untilize_variable_num_blocks.cpp` | borrowed | **1 — reused** | `data_movement/untilize` |
| `ttnn/kernel/compute/eltwise_copy.cpp` | borrowed | **1 — reused** | `data_movement/copy`, `data_movement/sharded/interleaved_to_sharded`, `data_movement/sharded_partial/interleaved_to_sharded_partial`, `data_movement/sharded_partial/sharded_to_interleaved_partial` |
| `eltwise/unary/…/reader_unary_interleaved_wh_multicore.cpp` | borrowed | **fork created, then DELETED** on revert — original untouched again (pointer comment reverted). Carried the BACKWARDS fix; see Handoff 3 | `data_movement/untilize` (now the only consumer, as pre-port) |
| `data_movement/untilize/…/compute/untilize_wh.cpp` | borrowed | **fork created, then DELETED** on revert — purely mechanical conversion, nothing unique lost | `data_movement/untilize` (as pre-port) |
| `ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_blocks.cpp` | borrowed | **2 — created** `…_metal2.cpp`; pointer comment added ✓ — **survives**, bound by the still-ported `MultiCoreSharded` | none (this op is the only consumer) |
| `…/untilize_with_unpadding/…/writer_unary_stick_layout_wh_multicore.cpp` | **lent** (the audit missed it) | **fork created, then DELETED** on revert — purely mechanical conversion, nothing unique lost | `data_movement/untilize` (as pre-port) |

Binding vocabulary the four new forks establish, for the next consumer to inherit:

| fork | `dfb::` | `tensor::` | named args |
|---|---|---|---|
| `reader_unary_interleaved_wh_multicore_metal2.cpp` | `in` | `src` | CTA `num_tiles_per_2d`, `third_dim`, `total_tiles_per_row`; RTA `start_id`, `single_block_size_row_arg`, `single_block_size_col_arg` |
| `untilize_wh_metal2.cpp` | `src`, `out` | — | CTA `block_size_col`, `block_size_row`, `third_dim` |
| `writer_unary_stick_layout_interleaved_blocks_metal2.cpp` | `out` | `dst` | CTA `float32_dtype`, `output_row_size` (unread — see below); RTA `num_rows_block`, `block_row_size`, `batch`, `num_blocks_h`, `num_blocks_w`, `last_block_row_size_unpadded`, `num_output_rows_unpadded`, `block_start_row_id`, `block_start_row_offset` |
| `writer_unary_stick_layout_wh_multicore_metal2.cpp` | `out` | `dst` | CTA `total_num_rows`, `third_dim`, `tile_height`, `unpadded_X_size`; RTA `width_size`, `start_row_id`, `start_column_id`, `single_block_size_row_arg`, `single_block_size_col_arg`, `sub_block_width_size`, `single_sub_block_size_row_arg` |

`untilize_wh_metal2.cpp` deliberately reuses the `dfb::src` / `dfb::out` vocabulary of its two sibling
untilize compute forks in the same directory, so a factory can bind any of the three with one
vocabulary. **Sunset:** `data_movement/untilize` is the last unmigrated consumer of three of the four
new forks; when it ports, those three legacy originals can be retired.

### Findings — bugs and oddities carried forward unchanged

Every item below is preserved byte-for-byte in behavior. None is fixed.

1. **Five dead compile-time args** — emitted by the host, never read by the kernel. Kept as named CTAs
   (see the Friction gap above):
   - `writer_unary_unpad_dims_split_rows.cpp`: `unpadded_stick_size` (legacy CTA 1). The kernel gets
     the same quantity as RTA `num_unpadded_X` × element size instead.
   - `writer_unary_stick_layout_interleaved_blocks_metal2.cpp`: `output_row_size` (legacy CTA 1).
   - `writer_unary_unpad_width_16_sharded.cpp`: `aligned_page_size` (legacy CTA 2) — the same
     `writer_ct_args` vector feeds both same-shard-type writers and only the general one reads it.
   - `writer_unary_stick_layout_split_rows_multicore_nd_sharded.cpp`: `output_stick_size` (legacy CTA
     1) and `input_single_tile_size` (legacy CTA 8). Both were already dead under positional args (the
     kernel's `TensorAccessorArgs<17>` offset accounted for them without reading them).

2. **One dead runtime arg.** `writer_unary_unpad_dims_split_rows.cpp:29` reads
   `num_blocks_w_input` into a local the kernel body never uses. It still occupies a dispatch slot, so
   it is preserved as a named RTA. Removing it would shrink the writer's per-core RTA payload by one
   word — a real but out-of-scope saving.

3. **`device/factories/untilize_with_unpadding_multi_core_shared_variables.hpp` is unreferenced dead
   code.** It defines `UntilizeWithUnpaddingMultiCoreSharedVariables` (reader/writer `KernelHandle`s, a
   core vector, an `ncores`) — a leftover from the pre-`ProgramDescriptor` `ProgramFactoryConcept` era.
   Nothing in the tree names the struct; the header is only listed in
   `ttnn/cpp/ttnn/operations/data_movement/CMakeLists.txt:333`. Left in place (out of the port's
   scope). Now that the op is on Metal 2.0 it is unambiguously dead and can be deleted with its
   CMakeLists entry.

4. **Stale `CB` comments in off-limits files.** `device/untilize_with_unpadding_device_operation.cpp`
   lines 144, 221 and 273 each say "binds the output buffer directly as an L1 circular buffer (see the
   sharded factory's `sharded_output_cb_index` CB)"; `untilize_with_unpadding.cpp:102` refers to "the
   CB budget". The identifier `sharded_output_cb_index` no longer exists — it is now the
   `SH_SHARDED_OUT` dataflow buffer. The *reasoning* in all four comments is still correct (that is
   why the DRAM rejection they justify is still right); only the vocabulary is stale. Not edited: both
   files are outside the port's writeable surface.

5. **`num_blocks_w_input` and the `round_down_32` helper.** `writer_unary_unpad_dims_split_rows.cpp:12`
   defines `inline uint64_t round_down_32(...)`, which nothing calls. Pre-existing; left alone.

### Doc-evolution and carry-over

- **Candidate: a spec-side `push_buffer_set`.** Four factories share the block-buffer model
  (tilize, tilize_with_val_padding, untilize, untilize_with_unpadding) and the helper exists to keep
  their sizing rules identical. The first of them to port must inline the rules (as this one did), and
  each subsequent port will inline them again — which is precisely the drift the helper prevents.
  Worth deciding, before the second block factory ports, whether
  `data_movement/common` should grow a `Group<DataflowBufferSpec> make_buffer_pair(const BlockBufferSet&, …)`
  alongside the descriptor one. That is a shared-code change, outside any single port's scope.
- **Sibling ops that would benefit from the same pattern:** `data_movement/untilize` is the direct
  sibling — it shares four kernels with this op and is the last unmigrated consumer of three of the
  four new forks. Porting it next would let three legacy kernel copies be retired at once, and its
  block factory can bind `writer_unary_stick_layout_wh_multicore_metal2.cpp` and
  `untilize_wh_metal2.cpp` at rung 1.

### Test coverage notes

The confirmed no-regression baseline (agreed with the invoker before relying on it) is:

0. `tests/ttnn/unit_tests/operations/data_movement/test_untilize.py` — added as a sentinel *after* the
   baseline was agreed, because it exercises the untilize-**codegen** paths that consume these
   factories. It is the only file in the set that covers the consumer this port broke, so it belongs
   in the permanent baseline for any further work on these factories, not just this session.
1. `tests/ttnn/unit_tests/operations/data_movement/test_untilize_with_unpadding.py` — primary, 992
   lines, 41 references.
2. `./build/test/ttnn/unit_tests_ttnn --gtest_filter='*UntilizeWithUnpadding*'` — the C++ gtest
   `TestGraphCaptureArgumentsUntilizeWithUnpadding`
   (`tests/ttnn/unit_tests/gtests/test_graph_capture_arguments_untilize_with_unpadding.cpp`, built into
   `unit_tests_ttnn` via `tests/ttnn/unit_tests/gtests/sources.cmake:28`).
3. `tests/ttnn/unit_tests/base_functionality/test_to_layout.py -k untilize_with_unpadding`.

Note for whoever runs it: in (3) the `-k untilize_with_unpadding` filter matches by **test name**, and
only `test_untilize_with_unpadding_W_16` matches; the file's other ~15 `untilize_with_unpadding` call
sites live in differently-named tests (e.g. the ND-sharded and sharded-output cases around lines
588-630, 967-1040 and 1608-1620) and are **not** selected by that filter. Running the file unfiltered
would cover the W=16 fast path *and* the ND-sharded path this port touches. Flagging rather than
overriding the invoker's chosen command.

## Verification status

Two verified states, in order.

**1 — fully ported (5/5 factories).** Run and green, with the forcing scaffolding in place:

```bash
./build_metal.sh --build-tests                      # 0 errors
export TT_METAL_WATCHER=10
pytest tests/ttnn/unit_tests/operations/data_movement/test_untilize_with_unpadding.py \
       tests/ttnn/unit_tests/operations/data_movement/test_untilize.py
# -> 1195 passed, 10 skipped, 8 xfailed in 806.58s
# -> METAL2_CHECKS_FORCED: 1118x (program_spec.cpp:2950) + 1118x (program_run_args.cpp:565)
```

**2 — partial revert (3/5 factories), the current tree.** Rebuilt (0 errors) and re-tested:
`1195 passed, 10 skipped, 8 xfailed in 690.78s` — **outcome-for-outcome identical to state 1**
(table in [Outcome](#outcome)). Differences from state 1 that bear on how much the green is worth:

- The forcing scaffolding is **gone** and was not re-applied (Handoff 2), so this run does **not**
  prove the validator ran. State 1's proof stands for the three factories that are unchanged between
  the two states, since neither their specs nor their kernels were touched by the revert.
- `test_untilize.py` matters more in this state than in state 1: it exercises the untilize-codegen
  paths, and codegen's native fallback now calls back into the two reverted factories.
- Also note `d900e3339dd [Nightly L2] Use NO_DISPATCH capture in test_untilize.py` landed from
  someone else between the two runs, so `test_untilize.py` is not byte-identical across them.

A third state was built deliberately and is **not** a candidate: guard dropped as originally
requested, which fails to compile (error text in [Outcome](#outcome)).

Notes on how the builds got there, worth knowing for the next port:

- **The first build attempt failed for a reason unrelated to the port.** The literal text `git status`
  had been prepended to line 1 of `tt_metal/impl/metal2_host_api/program_run_args.cpp` in the working
  tree — a stray shell command written into a source file while the forcing scaffolding was being
  applied — so that translation unit could not parse (`error: unknown type name 'git'`). Diffing the
  two scaffolding files against `HEAD` separated the corruption (one line) from the intended
  scaffolding (11 lines, all `skip_validation`/marker) immediately. Worth a diff-before-build habit
  whenever the scaffolding is applied by hand: a corrupted forcing file fails in `tt_metal/impl`,
  which reads at a glance like a framework problem rather than a typo.
- **Nothing in the port needed fixing to compile.** The one code change in that pass was the required
  consumer fix in untilize codegen, committed separately.
- **The revert clobbered nothing.** Three commits landed from another author between passes
  (`b334c1ae941` BACKWARDS fix, `2eb85a3d2f7` Remove dead CTA, `d900e3339dd` NO_DISPATCH capture).
  `git log a947a3cdf5d..HEAD -- <the six reverted paths>` was checked before reverting and returns
  only this port's own guard commit, so restoring those six files from the pre-port tree discarded no
  one else's work. `2eb85a3d2f7` touched `MultiCoreSharded` and the surviving
  `writer_unary_stick_layout_interleaved_blocks_metal2.cpp` — both **kept ported**, so its change is
  intact.

**Still not covered by any test:** the codegen non-tile-aligned L1 fallback
([Handoff points](#handoff-points) item 1) — in *either* state. Its only coverage is the compiler,
which is why the guard question had to be settled by building rather than by testing.

**Remaining from the originally-confirmed baseline** — not run this session, and cheap to add:
`./build/test/ttnn/unit_tests_ttnn --gtest_filter='*UntilizeWithUnpadding*'` (the graph-capture gtest,
now built) and `tests/ttnn/unit_tests/base_functionality/test_to_layout.py`. On the latter, note that
`-k untilize_with_unpadding` selects only `test_untilize_with_unpadding_W_16`; the file's other ~15
`untilize_with_unpadding` call sites sit in differently-named tests (around lines 588-630, 967-1040,
1608-1620), so running it **unfiltered** is what actually covers the W=16 fast path *and* the
ND-sharded path this port touches.

### Anti-pattern self-audit results

**Recorded against the fully-ported (5/5) tree.** The results below are the audit as run then; they
are the audit of record for the three factories that remain ported, since the revert did not touch
their factory `.cpp`s, their kernels, or their bindings.

They are **no longer a whole-directory clean sweep**, because the two reverted factories and
`writer_unary_stick_layout_split_rows_multicore.cpp` are legacy code again: the `cb`-name,
`CBDescriptor`/`CircularBuffer`, positional-CTA and `TensorAccessorArgs<N>` checks all have
legitimate hits in those three files now, by design. Re-running the sweep over the op directory as a
pass/fail gate would be misreading it — scope it to the three ported factories and their kernels.

Run over the op directory (**27 `.cpp`/`.hpp` files scanned** — non-zero denominator) plus the four
new forks. Static checks only; the build-dependent items are unverified.

| check | result |
|---|---|
| No buffer address in run-args (`->address()`, `emplace_runtime_args`, bare `Buffer*`) | **0 hits / 27 files** |
| No magic CB indices, `CBDescriptor`, `CBFormatDescriptor`, `CircularBuffer` | **0 hits** |
| No `TensorAccessorArgs<N>()` in any ported kernel | **0 hits** (the one remaining occurrence is in the *legacy original* `writer_unary_stick_layout_wh_multicore.cpp`, deliberately untouched for its other consumer) |
| No `cb` in any DFB name, spec-name string, or host variable | **0 hits** in the five factories and the four forks, after renaming `*_cb_data_format` → `*_dfb_data_format` and dropping the dead `cb_utils.hpp` includes. 4 hits remain in comments in the two **off-limits** files — reported, not edited (finding 4) |
| Conditional bindings follow the pattern | `SH_SHARDED_OUT` is bound only in the two configurations that allocate it. **No `#ifdef` needed**: the host selects a different kernel *source* per configuration, so no kernel ever name-looks-up a token its own build does not bind |
| No `.id` extraction on `dfb::` handles | **0 hits** |
| No CTA→RTA demotion | none — per-group CTAs preserved on all 2 + 4 same-source compute splits |
| No unnecessary multi-binding flag; never stacked with a self-loop | **0 occurrences of `allow_instance_multi_binding`** anywhere |
| All CTAs named | **yes** — every `compile_time_args` is `{{name, value}, …}` |
| No nameable argument smuggled into varargs | 2 vararg blocks, both genuine indexed collections (runtime-count `BlockRep` runs; CTA-bounded shape stream). Their 3 leading scalars are named |
| No forced-legality scaffolding in the diff | the 2 `tt_metal/` files are **excluded from the commit**; no other `tt_metal/` path is touched |
| No ephemeral doc cited from code | **0 hits / 27 files** |
| Every legacy `TT_FATAL`/`TT_ASSERT`/`TT_THROW` accounted for | **census clean — no output.** Three `dst_buffer != nullptr` guards were initially lost when the `Buffer*` locals went away; restored as `output.buffer() != nullptr` (same condition, same message, no stray `Buffer*`) |
| Every `hw_config` reproduces the legacy resolved values | DM: every kernel resolved to the plain reader/writer defaults → arch-agnostic TTNN helpers. Compute: **Style B**, `ComputeGen1Config` built directly; only `enable_32_bit_dest` and `unpack_modes` set, all four other fields left at defaults that coincide with the legacy `ComputeConfigDescriptor` defaults |
| Every `KernelSpec`'s `opt_level` matches | `grep -n opt_level` → **5 lines**, one per factory, each inside the single construction site (or lambda) that builds that factory's compute specs → **all 10 compute `KernelSpec`s carry explicit `O3`**. No DM spec sets one (legacy `O2` = Metal 2.0 default) |
