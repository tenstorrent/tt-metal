# Metal 2.0 Port Report — `data_movement/slice`

*Opened at the start of the port; entries captured as they happened, polished at the end.*

## Outcome

**`PORTED`** — `SliceTileProgramFactory` converted to `CustomProgramSpecFactoryConcept`, together with
the two kernels it binds. The op's other four factories remain on `ProgramDescriptorFactoryConcept` and
are enumerated for a later pass in `METAL2_PORT_PLAN.md` (*Deferred / Flagged*). The op builds and runs
with its factories on mixed concepts, as designed.

**Verification.** Confirmed baseline `tests/ttnn/unit_tests/operations/data_movement/test_slice.py`:

| | build | tests | legality checks live |
|---|---|---|---|
| pre-port | SUCCESS | **448 passed, 38 skipped** | ✓ 81 × `program_spec.cpp`, 81 × `program_run_args.cpp` |
| post-port | SUCCESS | **448 passed, 38 skipped** | ✓ 188 × each |

Test-name sets are identical between the two runs (diffed, empty). The single `TT_FATAL` in both logs is
an expected negative-path assertion, present pre- and post-port. Run wall-clock differed (218 s → 99 s)
purely from JIT kernel-cache warmth — **not** a performance claim; this port makes no perf assertion
either way.

**The realized concept was proven, not assumed.** A factory whose `override_runtime_arguments` has the
wrong return type falls back to `ProgramSpecFactoryConcept` *silently*, and the override never runs. A
temporary `static_assert` triple (custom-concept true; base-concept and descriptor-concept both false)
was compiled against the factory and then removed before commit. Recommend the recipe suggest this — it
is a one-build check against an otherwise-invisible failure mode.

**Carries one out-of-scope edit, authorized by the invoker**: `ccl/mesh_partition` (1 file). See
Handoff points — it was unavoidable, not a convenience, and it is **not run-verified** on this bench.

## Provenance

```
git log -1 --format='%h %cs %s' -- docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/
```

- **Recipe docs (this port):** `1167faf7b42 2026-09-04 docs(metal_2.0): binary_ng relaxation analysis; invariant checks over commit stamps`
- **Audit docs (inherited):** `1167faf7b42 2026-09-04 docs(metal_2.0): binary_ng relaxation analysis; invariant checks over commit stamps`

## TTNN ProgramFactory

### Concept realized

**`CustomProgramSpecFactoryConcept`**, as the audit chose. Not re-decided.

The override returns a `TensorArgument` for **every** `TensorParameter` bound to an io tensor — `INPUT`
and `OUTPUT`, both on every dispatch. None deliberately skipped. This matches what the ported-from
override did: `patch_slice_program_addresses` refreshed the input base (as a common-RTA slot) and the
output base (as writer RTA slot 0) on every cache hit.

**One subtlety consciously dropped**, per the recipe's instruction to preserve-or-consciously-drop it:
the legacy `patch_slot0` helper (`slice_program_factory_rm_sharded.cpp:372-378`) deliberately **skipped
slots holding 0** — the zero-filled no-op cores. In Metal 2.0 the output address is not an RTA slot at
all but a `TensorBinding`, so there is no per-node slot to skip; the framework applies the binding
uniformly. This is a no-op behaviourally: a no-op core receives `num_pages = 0`, never enters the write
loop, and never dereferences the accessor. Dropped deliberately, recorded here.

### Device-op-class edits

- **Pybind entry points removed:** `slice_nanobind.cpp` — `SliceTileProgramFactory.create_descriptor`
  (the `.def_static` at `:168-179`). See Handoff points for the downstream consumer.
- **Custom `compute_program_hash`:** left intact at `slice_device_operation.cpp:348`. Confirmed
  untouched — `git diff` shows no change to that file.

### Open items

- **Relaxation candidates:** none applied, none newly identified. The audit's `none` verdict held; the
  op's hash pins essentially the full `TensorSpec` (audit *Relaxation candidates*), which is consistent
  with strict matching and gave no trouble on the cache-hit path.
- **Capability the op would benefit from:** see Handoff point 1 — a supported way for one op to consume
  another op's ported factory would remove this port's only out-of-scope edit.

## Handoff points

### 1. `ccl/mesh_partition` consumes slice's factories through the legacy host entry point (CCL team + Metal 2.0 team)

**The blocker.** `mesh_partition_program_factory.cpp:126-134` calls
`Factory::create_descriptor(slice_attrs, slice_tensor_args, tensor_return_value)` inside a `std::visit`
**generic lambda**. A generic lambda under `std::visit` is instantiated for *every* alternative of
`SliceDeviceOperation::program_factory_t` — so deleting `create_descriptor` from **any one** of slice's
five factories breaks that translation unit's compile. There is no slice factory that can be ported in
isolation.

**Why no in-directory shim exists.** The obvious mitigation — keep `create_descriptor` alongside the new
`create_program_artifacts` — is worse than it looks, and it fails **silently**. From
`ttnn/api/ttnn/operation_concepts.hpp:120-135`, both spec concepts are defined with
`&& !ProgramDescriptorFactoryConcept<T>`, and `ProgramDescriptorFactoryConcept` (`:73`) is satisfied by
the mere presence of `&T::create_descriptor`. A factory declaring both members therefore classifies as a
**descriptor** factory. `AllFactoriesValid` still passes (exactly one concept matches, count == 1), the
build is green, the tests pass — and `create_program_artifacts` is **never called**. The port would look
complete and be inert.

**What was done.** The invoker was consulted and authorized carrying MeshPartition along. Exactly **one**
file outside the op directory changed:
- `ccl/mesh_partition/device/mesh_partition_program_factory.cpp` — `create_at` now `if constexpr`-branches
  on the ported factory and builds its `Program` via `MakeProgramFromSpec` + `SetProgramRunArgs`;
  `override_runtime_arguments` routes it through `UpdateProgramRunArgs`. The other four factories keep the
  `Program{descriptor}` / `patch_slice_program_addresses` path unchanged.
- The branch predicate is a local concept, `detail::IsSliceSpecFactory`, keyed on the presence of
  `create_program_artifacts` rather than on a list of factory names — so the next slice factory to
  migrate needs **no** further edit here, which was the point of the recommendation below.
- `mesh_partition_device_operation.hpp` needed no change: the stored `program_factory_t` still
  discriminates correctly.

**Verification gap — please read.** MeshPartition's own tests are `tests/nightly/t3000/ccl/test_mesh_partition.py`
and `tests/nightly/tg/ccl/test_mesh_partition_6U.py`, both **multi-device**. This port was verified on a
single-chip Wormhole n150, so **the MeshPartition path is compile-verified only, not run-verified.** It
needs a t3000/TG run before merge.

**Recommendation.** The concept-keyed predicate means the *next* four slice ports should need no further
edit to this file — but each will still silently re-enter an untested code path on this bench, and the
branch only disappears when the last factory converts. Two options worth a decision by the owning teams:
(a) run the t3000/TG MeshPartition suites once per slice port as a gate; or (b) give TTNN a supported way
for one op to consume another op's ported factory, so this cross-op coupling stops being a per-port cost
at all. Option (b) is the one that would have made this port scope-tight.

### 2. Removed pybind surface — `SliceTileProgramFactory.create_descriptor` (owners of `models/experimental/ops/descriptors`)

- **File:** `ttnn/cpp/ttnn/operations/data_movement/slice/slice_nanobind.cpp`, the `.def_static` at `:168-179`.
- **What it was for:** exposing the tiled slice factory's `ProgramDescriptor` to Python so a descriptor
  can be built and inspected without dispatching the op.
- **Known downstream caller:** `models/experimental/ops/descriptors/data_movement/slice.py:54` —
  `ttnn.SliceTileProgramFactory.create_descriptor(params, tensor_args, output_tensor)`. **This call will
  now raise `AttributeError` at runtime.** Left unmodified (out of scope); it needs a follow-up owned by
  whoever maintains that descriptor-export path. There is no drop-in replacement:
  `create_program_artifacts` returns a `ProgramSpec` + `ProgramRunArgs` pair, not a `ProgramDescriptor`,
  so the Python consumer needs a real port rather than a retarget.
- **Deliberately *not* removed:** the enclosing `nb::class_<SliceTileProgramFactory>(mod, "SliceTileProgramFactory")`.
  `ttnn/ttnn/__init__.py:635-640` and `ttnn/ttnn/operations/data_movement.py:548` import that symbol at
  **module scope**, so removing the class binding would break `import ttnn` for the entire package. Only
  the one `def_static` went. (The brief flagged exactly this — see Successes.)

### 3. Boundary-rule assumption violations

**None.** No call site outside the op directory required a `sem::` or `tensor::` handle. The only
out-of-op callees the ported kernels reach are `noc.*` framework primitives, which take the
`DataflowBuffer` / `TensorAccessor` objects the porter constructs locally — the recipe's documented
second exception, not a handoff.

### 4. Kernel-lib gaps

**None.** Both kernels were already fully Device 2.0 (`Noc`, `DataflowBuffer`, `TensorAccessor`), and
neither reaches a kernel-lib helper. Slice's own writer already used `dfb_out.get_entry_size()` rather
than the `get_local_cb_interface(...)` free function, so whitelist rule 7 had nothing to do on this
factory. (The rule-7 breadcrumb the brief carries applies to the *borrowed* eltwise writer, which belongs
to `SliceTileTensorArgsProgramFactory` — a later pass.)

### 5. Framework gaps

**None bit during this port.** All audit-flagged features were N/A and stayed N/A.

## Findings

*Bugs and oddities read out of the legacy code and **preserved, not fixed**, per
[§Scope discipline — Bugs are part of the behavior you preserve]. (The recipe repeatedly says to file
these "under findings" but its report structure defines no such section — see Friction → Gap 6. Filed
here.)*

1. **Legacy's two code paths disagree on one no-op-node value, and the port had to pick one.**
   For a node in neither core group, the writer's third runtime slot (`start_id`) is written as **`0`**
   by `create_descriptor` (`slice_program_factory_tile.cpp:176` pre-port, `{0u, 0u, 0u}`) but as the
   running **`num_tiles_written`** by the cache-hit override (`slice_tile_dynamic_args`, `:274` pre-port
   — the two writer pushes sit *outside* that function's `if (active)` guard, unlike every reader push
   above them). So legacy writes one value on a cache miss and a different one on every subsequent hit.

   **It is unobservable.** The same node's `num_pages` is `0` on both paths, and `num_pages` gates the
   only two uses of `start_id` in the kernel: `for (i = start_id; i < start_id + 0; ++i)` and, under the
   never-defined `BACKWARDS`, `end_id = start_id - 0`. Neither loop body executes, and the accessor is
   never dereferenced.

   **What the port does:** the ported factory computes per-node values in one function shared by both
   paths, so it cannot reproduce the disagreement. It follows the **override's** value
   (`num_tiles_written`, unguarded), because that is the function the shared helper is modelled on and
   the one that runs on every dispatch after the first. Net effect: the cache-*miss* dispatch now writes
   `num_tiles_written` where legacy wrote `0`, into a slot that is provably not read on that node.
   Flagged rather than silently normalised. If the ops team would rather have the zero-fill, it is a
   one-line `active ? num_tiles_written : 0`.

2. **Latent out-of-bounds write for a rank-1 input, in both the pre- and post-port code.**
   `accumulated_total_per_dim`, `num_unpadded_tiles_per_dim` and `num_padded_tiles_per_dim` are sized
   `num_dims` and then have index `[1]` assigned unconditionally
   (`slice_program_factory_tile.cpp:76-77, 84-87` pre-port; `:88-93` post-port). With `num_dims == 1`
   that is a heap overflow. It cannot fire today — this factory is selected only for TILE layout, and a
   tiled tensor has rank ≥ 2 — so it is a latent trap rather than a live bug. **Preserved verbatim**
   (writing the guard would be a functional change the port is not entitled to make). Worth an
   assertion by the op owners if rank-1 tiled tensors ever become representable. Note the same shape
   appears in `slice_tile_dynamic_args`, which this port leaves untouched.

3. **`patch_slice_program_addresses`' `SliceTileProgramFactory` arm is now dead code.** The shared
   function (`slice_program_factory_rm_sharded.cpp:383-411`) still carries an
   `if constexpr (… SliceTileProgramFactory || … SliceTileTensorArgsProgramFactory)` branch. Nothing
   routes the ported factory there any more — slice's own dispatch goes through the new override, and
   MeshPartition now branches before the call. **Left exactly as it is**: the branch is shared with the
   still-unported `SliceTileTensorArgsProgramFactory`, and narrowing the condition would be a cleanup
   bundled into a port. Whoever ports the tensor-args factory retires the whole arm.

## Successes

- **[Brief — pybind `create_descriptor` to delete]** told me to remove *"the one `def_static`"* and
  explicitly warned that the neighbouring `nb::class_` bindings *"are not `create_descriptor` bindings —
  read the surrounding code before removing anything beyond the one `def_static`."* That precision was
  load-bearing and saved a serious break. The recipe's own wording (*"just remove the line(s)"*,
  [Pattern: Removing pybound legacy factory entry points]) reads naturally as "remove the binding," and
  removing the whole `nb::class_` block would have broken `import ttnn` for the entire package —
  `ttnn/ttnn/__init__.py:639` imports `SliceTileProgramFactory` at module scope. The brief's extra
  sentence is the reason that didn't happen. **Keep it verbatim, and consider promoting the distinction
  into the pattern entry itself** (see Friction → Gaps).

- **[Brief — Same-basename trap]** fired exactly as intended. Slice owns
  `device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` *and* borrows a different file of the
  same basename from `eltwise/unary`. I ran the shared-kernel census on the basename per
  [Caution: Porting a shared kernel] and got hits for both; the brief's warning is what made me check the
  bound **path** at `slice_program_factory_tile.cpp:157` rather than the filename. Converting the wrong
  one in place would have silently broken 15 other ops. The catalog's own instruction — *"check the bound
  path, not the filename"* — is correct but easy to skim past; a concrete same-op example is what made it
  land.

- **[Recipe — Ensure the Metal 2.0 host-side legality checks are enabled]** paid off immediately and
  cheaply. The `grep -n 'bool skip_validation'` found **9** sites (not a fixed list), including
  `UpdateTensorArgs`, which the recipe warns may or may not be present depending on tree freshness — it
  was. Forcing all 9 and proving with the markers took a few minutes and removed the question entirely.
  The recipe's insistence on *proving* rather than *editing* was right: see Friction → Confusion for the
  one part of the proof step that doesn't work as written.

- **[Recipe — the atomic unit is one ProgramFactory]** with its "don't raise a which-factory question"
  instruction was the right call for a five-factory op. Picking `SliceTileProgramFactory` autonomously and
  enumerating the rest kept this pass tractable; the instruction to *not* ask saved a round trip.

- **The repo already enforces two of the recipe's anti-patterns as pre-commit hooks, and the recipe
  never mentions them.** The commit ran `Detect smuggled buffer-address runtime args in descriptor
  factories` and `Detect ProgramDescriptor rebuilds inside override_runtime_arguments` — both passed.
  These are automated versions of self-audit checks the porter is told to run by hand. Worth naming in
  [§Anti-pattern self-audit]: it tells the porter that a class of mistake is caught mechanically at
  commit time, which is both reassuring and a reason to trust the manual sweep less as the sole guard.
  (It also means a porter who skips the manual sweep is *not* fully unprotected — useful calibration for
  how much emphasis those items need.)

- **[Recipe — Check that your first commit actually landed]** fired exactly as described, on the first
  commit. `clang-format` reformatted two files and **aborted** the commit while printing a wall of
  `Passed` lines; `HEAD` had not moved. Without the warning I would plausibly have read that output as
  success. The recipe's follow-up instruction was also right: `git diff` showed pure reflow (three
  non-braced constructs re-wrapped), so no re-verification was needed. Worth keeping verbatim.

  Related and also correct: the **trailing-comma-on-every-braced-initializer** instruction. Every
  `DFBBinding` / `TensorBinding` / spec initializer written that way survived the hook untouched; the
  only things it moved were the three constructs that are not braced lists. Had the commas been missing,
  the reformat would have been the large unexpected diff the recipe predicts.

- **[Patterns catalog — Two-toucher DFB → assign 1P+1C]**, step 3 (*"Re-derive, don't transcribe"*),
  worked as designed. The brief called this CB "legal 1:1"; re-running the census independently
  (reader = locked PRODUCER at `reader_...:39,42`; writer = locked CONSUMER at `writer_...:45,48`)
  reached the same verdict, so the transcription would have been fine — but the check cost about two
  minutes and the guidance correctly distinguishes it from the factory-concept choice, which the porter
  is told to inherit rather than verify. Good calibration.

## Friction

### Gaps

1. **The recipe has no bucket for host-side cross-op coupling, and this GREEN op arrived with a
   guaranteed out-of-scope edit.** The [scope boundary] permits exactly one kind of write outside the op
   directory (a shared-kernel `_metal2` fork). `ccl/mesh_partition` is neither a kernel nor a fork — it
   is another op reusing slice's *host* entry points. So the recipe's only available response was
   capitulation ([§When the discipline doesn't fit]), while the **brief** told me to *"decide early
   whether MeshPartition moves with slice or gets a compatibility shim."* Those two instructions are in
   direct conflict, and neither document resolves it. I stopped and asked the invoker.

   Note the auditor saw this coming and said so — `METAL2_PREPORT_AUDIT.md` *Recipe notes* 4 asks for a
   third escape type (*host-side factory reuse by another op*) and records that *"the recipe gave me
   nowhere to put it."* Two independent agents hit the same missing bucket on the same op. **Suggested
   fix, in two parts:**
   - *Audit side:* add the third escape type as Recipe note 4 proposes, **and make it gate.** An op whose
     factories are driven by another op cannot be ported scope-tightly; that is a feasibility fact known
     at audit time, and it should reach the porter as a gate decision (or an explicit pre-authorization),
     not as a "decide early" in the brief.
   - *Port side:* give [§When the discipline doesn't fit] an explicit third outcome besides PORTED and
     CAPITULATED — something like *AUTHORIZED SCOPE EXTENSION*: the porter stops, states the required
     out-of-directory change and its blast radius, and asks. Right now a porter who asks is off-script,
     and a porter who capitulates delivers nothing for a port that is otherwise entirely tractable.

2. **The "keep the legacy entry point as a compatibility shim" idea fails silently, and nothing warns
   about it.** This is the first thing any porter will try when an out-of-op consumer blocks the port,
   and it is the single most dangerous move available: a factory declaring both `create_descriptor` and
   `create_program_artifacts` satisfies `ProgramDescriptorFactoryConcept`, which both spec concepts
   negate — so `AllFactoriesValid` passes with count == 1, the build is green, every test passes, and
   `create_program_artifacts` is dead code. A "successful" port that ported nothing.

   The recipe's build-failure list mentions `AllFactoriesValid` firing when *"a factory satisfies two
   concepts (likely a stale `cached_program_t` declaration)"* — which is the **loud** version of this
   family, and its presence makes the quiet version *more* surprising, not less: a porter who has read
   that line reasonably expects the framework to catch concept confusion. **Suggested fix:** a short
   note under [§Construct] or the [TTNN integration doc]'s concept section — *"deleting the legacy entry
   point is mandatory, not cosmetic: leaving it re-classifies the factory as a descriptor factory and
   your `create_program_artifacts` is never called. It fails green."* Cheap to add, and it forecloses the
   worst available mistake.

3. **The vararg API has no writable accessor, and the recipe doesn't mention it.** The reader kernel uses
   its RTA vararg block as a mutable odometer — legacy takes `tt_l1_ptr uint32_t* id_per_dim =
   (tt_l1_ptr uint32_t*)(get_arg_addr(2))` (`reader_unary_unpad_dims_interleaved_start_id.cpp:23`) and
   does `id_per_dim[j]++` / `= 0`. Metal 2.0's generated `get_vararg(idx)` returns a **value**
   (`tt_metal/jit_build/genfiles.cpp:457`); there is no `get_vararg_addr`, and the named-section offset
   it bakes in is not exposed, so the address cannot be reconstructed without hardcoding an offset —
   exactly the "clever workaround" the recipe forbids.

   Resolution taken: copy the block into a fixed-size local at kernel entry (`num_dims` is a CTA, so no
   VLA) and mutate the local. Behaviour-preserving, because the host re-supplies the whole block on every
   dispatch. But I had to reason that out from the codegen, and a porter who instead reached for the
   offset arithmetic would land somewhere bad. **Suggested fix:** one line in
   [whitelist rule 4] / [Caution: Avoid varargs] — *"varargs are read-only values; a legacy kernel that
   wrote back into its arg buffer copies the block into a local first. The count is usually a CTA, so the
   local is a fixed-size array."* This is likely to recur: using the RTA buffer as scratch is a common
   legacy idiom in `unpad_dims`-style kernels, and the brief lists **six** slice kernels with vararg
   blocks.

4. **`Group<T>` initialization from a runtime-computed list is undocumented.** The migration guide and
   the recipe show `Group<T>` only as a brace-initialized literal (`.kernels = {READER, WRITER}`), and the
   recipe documents `Table` carefully (*"maps, not vectors — no `push_back`"*) but says nothing about
   `Group`. I had to read `utility/group.hpp` to find out whether `push_back` exists and whether a
   conditionally-built binding list is expressible. Given the recipe went to the trouble of warning about
   `Table`'s shape, the omission of its sibling reads as an oversight. **Suggested fix:** one sentence
   beside the `Table` note stating what `Group` is and how to build one incrementally.

5. **The `TT_FATAL` census and the `tt_metal/` scope check both use the wrong base revision on exactly
   the branch the workflow puts you on.** Both self-audit items compute
   `BASE=$(git merge-base origin/main HEAD)` and call it *"the pre-port revision."* That holds only if
   the branch is otherwise unmodified relative to `main` with respect to the op. Here it was not, and the
   census reported two files as having changed guard counts —
   `slice_device_operation.cpp` 22→21 and `slice_program_factory_rm.cpp` 3→4 — **neither of which this
   port touches at all.** The deltas are branch-vs-`main` drift (`git diff $BASE HEAD` on those two files
   shows 68 insertions). Re-running the census against `HEAD`, the true pre-port revision here, is clean.

   This is not an exotic setup: the recipe itself says the op owner's pre-port functional fixes land *"on
   a separate branch/PR before the port"*, so a porter working on or after such a branch is the expected
   case, and it is exactly the case where `merge-base` misfires. The same flaw hits the scope check
   `git diff --name-only "$BASE" | grep -E '^tt_metal/'`, which would flag pre-existing `tt_metal/`
   changes as a scope violation. **Suggested fix:** define BASE as the revision the port started from —
   `HEAD` when the porter has not committed, or an explicitly recorded start commit — and say why
   `merge-base` is not it. A one-line "sanity-check that BASE names your pre-port tree" would also do.

6. **The report structure has no `Findings` section, but the recipe repeatedly files things there.**
   [§Scope discipline] says *"Any bug you find goes in the port report under findings"*, *"write it up in
   the report under findings"*, and *"leave the `TT_FATAL` exactly as it is, and write a finding in the
   port report."* [§Capture the port report] then enumerates the sections — Outcome, Provenance, TTNN
   ProgramFactory, Handoff points, Successes, Friction, Open items for downstream — and *"do not omit
   sections."* None of them is Findings, and none is an obvious home: a preserved legacy bug is not a
   handoff (nobody outside the op owns it), not friction (the docs did fine), and not really a downstream
   open item. This port had three such findings and added a `Findings` section for them.
   **Suggested fix:** add `Findings` to the enumerated structure, or change the Scope-discipline wording
   to name the section that is actually meant.

### Confusion

7. **The "prove the legality checks are live" step cannot be run as written, before the port.** The recipe
   says: *"Rebuild, run one test, and grep the log for `METAL2_CHECKS_FORCED` — **two markers present**
   means both translation units are fresh."* But `BuildProgramFromSpec` and `SetProgramRunArgs` only
   execute when something builds a Metal 2.0 program. Pre-port, the op you are about to port is by
   definition *not* on Metal 2.0, so its own tests need not exercise either function. My baseline run
   happened to produce markers (81 from each file) because other already-ported ops run in the same
   pytest session — but that is luck, not the procedure working. On an op whose test session touches no
   ported op, the porter follows the recipe exactly, sees zero markers, and is told by the recipe to
   *"stop and fix the forcing before you read a single test result"* — chasing a non-problem.

   **Also: "two markers present" is ambiguous as stated.** Both files log the *same string*, so
   `grep -c` cannot distinguish "both files fresh" from "one file fired twice." It happens to work
   because `tt-logger` appends `(file.cpp:line)`, so
   `grep -o 'METAL2_CHECKS_FORCED.*' | sed 's/.*(\(.*\))/\1/' | sort | uniq -c` gives the real answer
   (here: 81 × `program_run_args.cpp:565`, 81 × `program_spec.cpp:2934`). **Suggested fix:** (a) say the
   proof runs *after* the first ported build, or name a known-ported op's test as the pre-port probe;
   and (b) give the disambiguating command, or tell the porter to use two distinct marker strings.

8. **`Buffer*`-in-an-`RTArgList` reads as an anti-pattern check failure when it is the thing being
   removed.** The self-audit says to search the factory for *"`emplace_runtime_args` / a bare `Buffer*`"*
   and expect zero hits. In this op **every** address travels that way pre-port — the audit's own Recipe
   note 3 makes the same observation about the roll-up vocabulary. Post-port the check is clean for the
   ported factory, but the grep is naturally run over the op *directory*, where the four unported
   factories still legitimately use the form. The check has no notion of "this factory is ported, those
   four are not," which is awkward for exactly the multi-factory ops the recipe elsewhere tells you to
   port one factory at a time. **Suggested fix:** note in the self-audit that on a partial-op port the
   sweeps are scoped to the ported factory's files, and say which files those are.

9. **Doc-size note, offered as calibration rather than a complaint.** The mandatory reading before the
   first edit is ~62k tokens of recipe plus the brief, the audit, `ttnn_factory.md`, `port_patterns.md`
   and parts of `migration_guide.md` — comfortably inside a 1M primary session (as the recipe intends),
   but it front-loads a large fixed cost onto every port, including a two-kernel one like this. The
   recipe's own "load on demand" framing for the reference docs worked well; the 1201-line recipe itself
   is the part that doesn't chunk. No suggested fix — flagging only because "is this scaling to N ports?"
   seems like a question the doc owners are asking.

## Open items for downstream

- **Shared kernel touches:** **none for this factory.** Both sources are bound only by
  `SliceTileProgramFactory` and were converted in place. Census run and disambiguated —
  see `METAL2_PORT_PLAN.md` → *Shared kernels*. The op's *other* factories do have a shared-kernel case
  (`SliceTileTensorArgsProgramFactory` borrows `eltwise/unary`'s
  `writer_unary_interleaved_start_id.cpp`, which already has a `_metal2` fork to bind at rung 1); that is
  the next porter's item, not this pass's.

- **Two near-identical writer kernels are now both on Metal 2.0 and could be consolidated.** Slice's own
  `device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` (converted by this port) and
  `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id_metal2.cpp` (pre-existing fork)
  are now functionally identical: same `dfb::out`, same `tensor::dst`, same `num_pages` / `start_id`
  named args. Slice's copy exists only because the legacy original took its DFB index from a *positional*
  CTA while slice needed a *named* one — a distinction the binding model erases, since neither reads a CB
  index any more. The fork's own header comment already tracks a **third** copy under
  `copy/typecast/` and issue **#52228** for the sunset. Slice's copy is now a fourth candidate for that
  consolidation. Deliberately not done here (out of scope, and it would reach into a peer op's directory
  for a non-forced reason).

- **Dead code the audit catalogued, still dead, still not fixed** — repeated here because the audit's
  *Misc anomalies* is team-only and this factory's items belong with the port:
  - `reader_unary_unpad_dims_interleaved_start_id.cpp` has no dead args (the four strided kernels do —
    audit anomalies 1-3, a later pass).
  - `#ifdef OUT_SHARDED` in slice's writer (`writer_unary_interleaved_start_id.cpp:29`) is unreachable on
    every slice path: no slice factory sets kernel `defines`. Both branches preserved verbatim per scope
    discipline. Whoever retires it should check the eltwise original and the `_metal2` fork together —
    all three carry the same dead branch.

- **Test coverage note.** `tests/ttnn/unit_tests/operations/data_movement/test_slice.py` (the confirmed
  baseline: 448 passed / 38 skipped) exercises this factory well. Not run, and worth a look before merge:
  the two nightly suites (`test_slice_for_conv.py`, `test_universal_input_tm_slice.py` — the latter names
  `SliceTileProgramFactory` explicitly in its docstring as the path under test) and the three
  `tests/sweep_framework/sweeps/data_movement/slice/` sweeps. And, per Handoff point 1, MeshPartition's
  t3000/TG suites, which **cannot** run on the single-chip n150 this port was verified on.

- **Doc-evolution suggestion (broader than a Gap entry).** Friction 1 and 2 are the same underlying
  shape: the recipe reasons about coupling almost entirely in terms of *kernel* sharing (the whole
  [Caution: Porting a shared kernel] apparatus), and has comparatively little to say about *host-side*
  coupling — one op calling another's factory, or a pybind/Python re-export chain that breaks at module
  scope. Both of this port's genuinely risky moments were host-side coupling, and both were caught by the
  **brief** rather than the recipe. Worth considering a short "host-side coupling" section alongside the
  kernel-sharing one: who else calls this factory, who else imports this symbol, what breaks at compile
  time vs. import time vs. call time.
