# Metal 2.0 Port Report — `ttnn/cpp/ttnn/operations/experimental/minimal_matmul`

## Outcome

**`PORTED`** — `MinimalMatmulDeviceOperation::ProgramFactory`, the op's only factory, converted to
Metal 2.0 on `ProgramSpecFactoryConcept`. **231 functional tests pass, 0 fail**, matching the
pre-port baseline exactly (see *Post-port verification*).

The port also created the first `_metal2` forks of this op's four kernel sources; the legacy
originals are untouched and continue to serve the fused-CCL emitter.

## Precondition — how this port was authorized

**This port ran without the recipe's stated precondition being met, on the explicit instruction of
the invoker, Igor Wroński (`iwrosz@tenstorrent.com`), who took responsibility for the decision.**
Recording it here in full because the recipe makes the precondition non-negotiable
(`ai/port/metal2_port.md`, *Read this first* → *Precondition*), so a reviewer needs to see that it
was overridden deliberately rather than missed.

The two conditions and their actual state at port start:

| Condition | State |
|---|---|
| The audit produced an **overall GREEN** result | **Not met.** `METAL2_PREPORT_AUDIT.md` (2026-09-10) is **RED** |
| The user explicitly asked to proceed | **Met.** Explicit go-ahead, with responsibility accepted |

**What the RED actually is.** It is not a code finding. Every code-side gate cleared — Device 2.0,
feature compatibility, offset base pointers, TensorAccessor 3rd argument. The single blocker is
that the TTNN readiness sheet's row for this op still describes it *before* its `ProgramDescriptor`
migration (#55738): `Concept` reads `legacy device-op`, `Override runtime args method?` reads
`n/a`, and the `Factory (variant)` row names `MinimalMatmulProgramFactory`, which is no longer this
device op's factory. Three primary cross-checked columns conflict with the code, which is the
audit's *spreadsheet-broken* trigger.

**The sheet had not landed when this port started.** The invoker reported the refresh as in flight.
The sheet was re-fetched at port start (2026-09-10) and the row was **byte-identical** to the
morning's fetch — still stale. The port proceeded anyway on the invoker's instruction.

> **Resolved during the port.** The row was refreshed while the port was in verification, and it
> **clears the gate**: `Concept` → `descriptor`, `Is able to port?` → `yes`. The audit's verdict is
> now GREEN (see `METAL2_PREPORT_AUDIT.md` → *Addendum*), so this port ran on a premise that turned
> out to be correct rather than merely authorized. The refreshed `Porting Target` and
> `Override runtime args method? = no` also **corroborate the base-concept decision** recorded
> below — the sheet and the port agree on `ProgramSpecFactoryConcept`. Two fields
> (`Factory (variant)`, `Factory definition path`) still name the legacy CCL helper; cosmetic, and
> routed to the sheet owner in the audit addendum.

**Consequence for this report's reader.** The port is built on the audit's own findings rather than
on a sheet-confirmed clearance. Two things follow, and both are load-bearing:

1. **`METAL2_PORT_BRIEF.md` was written by the auditor under the same override**, not issued by a
   passing audit. It is marked provisional at its head. Its content is derived entirely from the
   RED audit's PORT WORK and FYI-P findings, which the audit ran **in full** — the Red-outcome
   scoping rule's exception applied, because a sheet-only RED clears without touching the op's code.
   So the brief's substance is what a GREEN audit would have carried; only its authorization
   differs.
2. **If the refreshed row does not say what the audit predicts**, this port's premise is wrong and
   the work needs re-validating against the corrected row. The audit predicts the row resolves to
   `Concept = descriptor`, `Override runtime args method? = yes`, target concept
   `CustomProgramSpecFactoryConcept`. The one that would actually change the port is the last:
   the whole port targets the **custom** concept because the descriptor factory carries an
   `override_runtime_arguments`. A refreshed row that instead reads `ProgramSpecFactoryConcept`
   (the value the stale row's `Porting Target` cell carries today) contradicts the code and would
   need reconciling before this port is trusted.

## Concept deviation — base instead of custom, on the invoker's decision

**The port targets `ProgramSpecFactoryConcept`, not the `CustomProgramSpecFactoryConcept` the
recipe's rule selects.** The legacy `override_runtime_arguments` is **deleted rather than
translated**. Decided by the invoker (`iwrosz@tenstorrent.com`) after the trade-off was surfaced;
recorded here because the recipe forbids it in two places
(`ttnn_factory.md` → *"Do not delete the ported-from override in place of translating it"*, and
`metal2_port.md` → *Translating `override_runtime_arguments`* → *"The governing rule is fidelity,
not analysis"*).

**The evidence the decision rests on.** The rule's stated purpose is to avoid silently discarding
the op's *non-tensor* refreshes. This override has none — every write in
`minimal_matmul_program_descriptor.cpp:1019-1027` is a buffer address (`in0`, `in1`, `in2`, `in3`,
`ternary_a`, `ternary_b`, and all N outputs). It refreshes no scalar runtime arg and never touches
the compute kernel's args. The base concept refreshes precisely that set on every cache hit
(`operation_concepts.hpp:134` — *"Base spec factory: cache hit refreshes only tensor bindings"*), so
the two reach the same end state.

**What is actually given up:** mechanism fidelity, not correctness. On the `ProgramDescriptor` path
the adapter does no address inference for a factory that declares an override — the descriptor's own
comment says so (*"Bindings stay declared for the cache miss, then ignored"*, `:950`) — so today the
override is load-bearing. Replacing it with the framework's refresh is an equivalent-outcome change
of mechanism, which the zero-functional-change invariant would normally rule out. A reviewer
comparing cache-hit behaviour will find the same addresses applied by a different actor.

**A correction to an argument raised during the decision.** I initially cited the override's own
`~7%` dispatch-cost comment (`:949-950`) as a reason to keep it. That figure compares a hand-hoisted
override against `apply_resolved_bindings` on the **legacy descriptor** path; it says nothing about
`UpdateTensorArgs` vs `UpdateProgramRunArgs` on the Metal 2.0 spec path. It should not be read as
evidence either way, and **the port has not measured dispatch cost on either concept** — worth doing
before this pattern is generalized to other ops.

**Safety check performed.** No code outside the framework adapter calls the descriptor factory's
override. `minimal_matmul_strided_reduce_scatter_async_program.cpp:165` calls the *legacy*
`MinimalMatmulProgramFactory::override_runtime_arguments` — a different symbol in a different
struct, which this port does not touch.

## Provenance

- **Recipe docs (this port):** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`
- **Audit docs (inherited):** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`
- **Code base:** `main` @ `bc119c5338b`
- **Readiness sheet:** fetched twice on 2026-09-10; unchanged between fetches (see above)

## TTNN ProgramFactory

- **Concept realized:** **`ProgramSpecFactoryConcept`** — `create_program_artifacts` returning
  `ProgramArtifacts{.spec, .run_params}`, no `op_owned_tensors`, no override. Selected implicitly:
  the concept predicate `detail::HasSpecRuntimeArgsOverride` is keyed on an
  `override_runtime_arguments` returning `ProgramRunArgs`, and this factory declares none. Confirmed
  by the refreshed readiness row (`Porting Target = ProgramSpecFactoryConcept`,
  `Override runtime args method? = no`). See *Concept deviation* for why the legacy override was
  deleted rather than translated.
- **Spec shape:** 5 `KernelSpec`s (in0 sender/receiver, in1 sender/receiver, compute) · 4-7
  `DataflowBufferSpec`s (3 conditional) · 6 `SemaphoreSpec`s · 2-6 + N `TensorParameter`s ·
  4 `WorkUnitSpec`s. No `alias_with`, no `allow_instance_multi_binding`, no borrowed memory.
- **Notable mechanism:** the N output tensors are bound through a
  `KernelAdvancedOptions::TensorBindingSequence` named `outputs`, which the kernels consume with
  `make_tensor_accessors(tensor::outputs)`. **This op appears to be that mechanism's first
  production user** — before this port, `tensor_binding_sequences` appeared only in the framework
  itself. It worked as documented, first try, and is a strong candidate for a patterns-catalog entry
  (see *Open items*).
- **Custom `compute_program_hash`:** none (no `compute_program_hash`, no `attribute_values` /
  `to_hash` backdoor) — nothing to leave intact
- **Pybind entry points removed:** expected **none** — neither `minimal_matmul_nanobind.cpp` nor
  `minimal_matmul_split_nanobind.cpp` binds `create_descriptor`; they bind only the op and the
  `MinimalMatmulConfig` struct
- **Device-op-class edits forced:** expected **none** — the op already has a nested `ProgramFactory`
  with `program_factory_t = std::variant<ProgramFactory>`
  (`device/minimal_matmul_device_operation.hpp:28-42`), so the *direct-descriptor* exception
  (`ttnn_factory.md` §3) does **not** apply; the port is a method swap inside the existing struct.
  The two method *signatures* in that header change (`create_descriptor` → `create_program_artifacts`,
  and the override's return type `void` → `ProgramRunArgs`), which is the factory struct itself, not
  the device-op class around it.

## Pre-port baseline (the no-regression reference)

Run before any host-side change, on the legacy factory, with `TT_METAL_WATCHER=10`:

```
tests/ttnn/nightly/unit_tests/operations/experimental/test_minimal_matmul.py
  => 168 passed, 1 failed, 276 skipped  (+1 setup error)
```

**All 168 functional tests pass. The two failures are both the same Watcher/profiler collision and
are NOT port-related** — the host factory was untouched at this point and the `_metal2` forks are
bound by nothing:

| test | outcome | cause |
|---|---|---|
| `test_performance` | FAILED | profiler firmware + Watcher overflows BRISC L1: `brisc.elf: segment[0] [0x3ae0,+0x2324) overflows region:0 limit of 0x2200 bytes` |
| `test_run_performance` | ERROR at setup | same `TT_THROW @ tt_metal/llrt/tt_elffile.cpp:405` |

Both tests enable the device profiler (`KERNEL_PROFILER`); with Watcher also on, the combined
firmware exceeds the BRISC code region. They need to be run with Watcher **off**, separately.

So the post-port comparison is **168 passed** on the functional set, with the two perf tests
deselected (`-k 'not performance'`) or run without Watcher. A post-port run that reproduces exactly
these two failures and nothing else is a pass.

*(276 skipped are multi-device / arch-gated cases this single-p150b bench does not cover.)*

## Post-port verification

Rebuilt with the legality checks forced, then run with `TT_METAL_WATCHER=10`. `tt_metal/` is
byte-identical to `HEAD` — the forcing scaffolding was removed before the final build.

| suite | result |
|---|---|
| `test_minimal_matmul.py` (`-k 'not performance'`) | **167 passed, 0 failed**, 276 skipped |
| `test_minimal_matmul_split.py` + `test_dit_minimal_matmul_addcmul_fused.py` (`-k 'not performance'`) | **64 passed, 0 failed** |
| **total** | **231 passed, 0 failed** |

Matches the pre-port baseline exactly: baseline was 168 passed + 1 Watcher-incompatible perf
failure over the same 169 non-skipped cases; post-port is 167 passed with those 2 perf tests
deselected. `METAL2_CHECKS_FORCED` appeared **334** times in the main run and **120** in the
split/ternary run, so the spec validator was live for every result above.

Coverage of the port's novel parts: `chunks = 1, 2, 3, 6` exercises the **TensorBindingSequence**
(the variadic output bindings); the addcmul suite exercises the **conditional ternary bindings**;
the SwiGLU and fused-concat cases exercise the `IS_OUTPUT_WRITER` define split.

### Anti-pattern self-audit

Denominator: **28** `.cpp`/`.hpp` files in the op directory.

| check | hits | verdict |
|---|---|---|
| buffer address / `Buffer*` / `emplace_runtime_args` in the ported factory | 0 | pass |
| `TensorAccessorArgs<N>()` in ported kernels | 0 | pass |
| `cb`-shaped names in ported kernels + factory | 0 | pass (4 stale comments found and fixed) |
| `.id` extraction on a `dfb::` handle | 0 | pass |
| `allow_instance_multi_binding` in code | 0 | pass (3 hits are this port's own docs saying not to set it) |
| positional `compile_time_args` | 0 | pass |
| varargs in ported kernels | 0 | pass |
| forced-legality scaffolding in `tt_metal/` | 0 | pass — see note below |
| `.md` cited from code | 0 | pass |
| `opt_level` per compute `KernelSpec` | 1 spec, 1 explicit `O3` | pass |

> **The scaffolding check needs reading, not just running.** `git diff "$BASE" | grep -nE
> 'METAL2_CHECKS_FORCED|DO NOT COMMIT'` returns 6 hits here — **all of them inside the recipe docs
> themselves** (`ai/port/metal2_port.md`, which documents the scaffolding and this very check), which
> are in the diff because this branch carries the doc-branch merge. The check's real subject is
> `git diff --name-only "$BASE" | grep -E '^tt_metal/'`, which is **empty**. Worth a note in the
> recipe: on a doc-branch checkout the second grep false-positives on the recipe's own text.

### TT_FATAL census

`minimal_matmul_program_descriptor.cpp` drops from **4 to 3** guards. The lost one is the
cache-miss RTA-layout drift check (pre-port `:921-941`), which asserted that the emitted arg lists
still matched the `kIn0FixedArgCount` / `kIn1FixedArgCount` constants that
`override_runtime_arguments` indexed with. Both the override and those constants are gone, so this
is a **subject-deleted** loss — the one legitimate kind — and it was predicted in
`METAL2_PORT_PLAN.md` before the edit. Nothing replaces it because nothing indexes runtime args
positionally any more: the port's args are named, and a name mismatch is a build error rather than
a silent stale address. The three surviving guards (the SwiGLU evenness check and the two ternary
format/tile-size checks) are unchanged. No count changed outside the factory.

## Handoff points

*Captured as they arise.*

1. **Multi-device test coverage cannot be exercised on this bench.** The op's kernels are also
   bound by the legacy CCL emitter, whose consumers
   (`minimal_matmul_strided_reduce_scatter_async`, `strided_all_gather_minimal_matmul_async`,
   `all_gather_minimal_matmul_async`) are covered only by t3000 / TG / galaxy tests. This host has a
   **single Blackhole p150b**, so those tests cannot run here. The port's fork strategy is designed
   so the legacy copies are untouched and those consumers are unaffected by construction — but that
   is an argument, not a measurement. Someone with multi-device hardware should run the CCL nightly
   set before this merges.

## Successes

*Captured as they arise.*

1. **The scope-discipline rule against tidying caught a real bug in my own conversion.** Renaming
   the compute helpers' CB-id parameters to binding tokens, I applied the rename with a
   file-wide regex, which rewrote helper *bodies* to reference `dfb::out` instead of their own
   `out_dfb` parameter. `matmul_blocks` is called with the **intermediate** token in its `out`
   position (`compute.cpp:494-503` pre-port), so the result would have packed matmul partials into
   the output DFB instead of the accumulator — wrong numerics, and it would have compiled and
   validated cleanly. Caught by re-reading the diff against the pre-port call sites rather than the
   new code, which is exactly what the recipe's *"a validation guard can vanish … only visible
   against what used to be there"* framing trains you to do. The fork was reverted from `HEAD` and
   redone with per-function scoped renames.

2. **The spec validator caught a real `unpack_modes` omission, loudly and precisely.** The first
   post-port run failed 21 tests with one message:

   ```
   TT_FATAL: Compute kernel 'compute' consumes FP32 DFB 'in0' with enable_32_bit_dest=true,
   but provides no unpack_modes entry for this DFB.
   ```

   Exactly the "newly-required explicit entry" the recipe documents, named down to the offending
   DFB. This is the class of mistake the recipe calls out as otherwise silent, and the validator
   turned it into a one-line diagnosis. It is also the concrete payoff of forcing the legality
   checks: with `skip_validation` left alone this would have shipped.

3. **`Table` is a map, and the recipe said so before the compiler did.** `to_defines` was first
   written with an iterator-pair constructor (`Defines(m.begin(), m.end())`), which does not exist.
   The recipe's *"`Table`s are maps, not vectors"* note names the single-argument range constructor
   `Table(existing_map)` as the fix for exactly this case — converting a legacy `std::map` of
   defines. Cost one build cycle instead of a hunt.

4. **The audit's conditional-binding findings fired correctly, and early.** The audit flagged four
   places where a kernel instance names a CB it never touches
   (`METAL2_PREPORT_AUDIT.md` → Gate detail → CB endpoints). Each is invisible to an endpoint
   census — the census reads *legal 1:1* — and each becomes a compile error or a needless extra
   binding under Metal 2.0. Having them enumerated before construction is the difference between
   planning for them and meeting them as four separate build failures.

## Friction

*Captured as they arise.*

1. **Confusion — the 3rd-argument rule reads as "the CTA drops too", and here it must not.**
   Kernel-side whitelist rule 3's page-size note says the third argument "simply falls away … (its
   host-side CTA/RTA emission drops with it — see Dropped Plumbing)", and *Dropped Plumbing* lists
   "**Page-size 3rd-argument CTAs/RTAs**: a `page_size` value emitted **solely** to feed a
   `TensorAccessor`'s third constructor argument." The word *solely* is the whole condition, and it
   is easy to read past — the rule is stated twice and only one statement carries the qualifier.
   In this op the CTA is shared: `out_tile_size` feeds the accessor's third argument **and** is the
   `tile_size_bytes` L1 stride at eight further sites in each DM kernel
   (`dm_in0_sender.cpp:302, 312, 330, 340, 502, 516, 530, 540`). Dropping it would have silently
   broken every output write's addressing — a wrong-numerics bug with no build or validator signal.
   The audit brief this port inherited had it wrong for the same reason, and it was caught only by
   grepping the CTA's uses before editing. Suggest putting the qualifier in rule 3 as well, and
   adding the check explicitly: *before dropping a page-size CTA, grep its other uses in the kernel.*

2. **Gap — `METAL2_CHECKS_FORCED` cannot be proven on a pre-port op, which is when the recipe asks
   for the proof.** *Ensure the Metal 2.0 host-side legality checks are enabled* says to force the
   nine `skip_validation` sites, then "Rebuild, run one test, and grep the log for
   `METAL2_CHECKS_FORCED` — two markers present means both translation units are fresh and the
   checks are running." But the markers live in `BuildProgramFromSpec` and `SetProgramRunArgs`,
   which are **Metal 2.0 entry points**. A not-yet-ported op runs the `ProgramDescriptor` path and
   never reaches them, so the baseline run here logged **zero** markers — correctly, and with the
   scaffolding in place. Taken literally the instruction reads as a failed setup at exactly the
   moment it is impossible to satisfy. The fix is to prove it against **an op that is already on
   Metal 2.0** (any ported op's test), or to say plainly that the check moves to the first
   post-port run. Recording it because the stated failure mode — "if the markers are missing, stop
   and fix the forcing before you read a single test result" — would otherwise send a porter
   hunting a non-existent problem.

   **Confirmed after the port:** the first post-port run logged **135** `METAL2_CHECKS_FORCED`
   markers, so the scaffolding was correct all along and the checks were live for every result
   below. (And they earned their keep immediately — see Successes.)

3. **Gap — "run every test with Watcher on" is unsatisfiable for profiler-based perf tests.**
   *Run tests* says to `export TT_METAL_WATCHER=10` once and run everything that way. This op's test
   file contains two tests that enable the **device profiler**, and profiler firmware plus Watcher
   firmware together overflow the BRISC L1 code region — a hard `TT_THROW`
   (`brisc.elf: segment[0] … overflows region:0 limit of 0x2200 bytes`), not a threshold miss. The
   tests cannot run at all in the configuration the recipe mandates. It cost a full 11-minute
   baseline run plus a diagnosis to establish that the two red results were the harness, not the
   code — exactly the "previously-passing test now fails → stop and report" trap, arrived at from
   the other direction. Suggest a sentence in *Run tests*: perf/profiler tests are excluded from the
   Watcher rule and run separately with Watcher unset.

4. **Confusion — the `unpack_modes` guidance emphasises one half of the trap, and I fell into the
   other half.** The recipe's italicised warning is *"**The trigger is the DFB's format, not the
   op's tensor dtypes** — an op deriving an intermediate format as
   `fp32_dest_acc_en ? Float32 : data_format` … has no Float32 *tensor* anywhere, yet every such
   buffer its compute kernel consumes needs an entry."* That is this op's intermediate DFB exactly,
   so the warning landed — and I added the entry for `intermediate` and stopped there. The converse
   case is the one that bit: when the *input tensors* are FP32, `in0` and `in1` are Float32 DFBs
   too, and each needs its own entry. Read quickly, "not the op's tensor dtypes" reads as *tensor
   dtypes are irrelevant here*, when the actual rule is the plain one in the sentence before it:
   **every** Float32 DFB the compute kernel consumes. Suggest stating the rule first and the
   intermediate case as an illustration of it, rather than leading with the contrast. 21 of the
   op's tests exercise fp32 inputs, so the failure was loud and cheap — but only because the
   validator exists.

5. **Gap — `AddRuntimeArgsForNode` takes an `initializer_list`, so a conditional arg set needs a
   second call.** The recipe's example shows one call per node
   (`AddRuntimeArgsForNode(kra.runtime_arg_values, core, {{"start_page", …}, …})`). This op has a
   fixed run of 11 args plus one that exists only under `FUSE_TERNARY`, and an `initializer_list`
   cannot be built conditionally — a `std::vector<std::pair<std::string, uint32_t>>` does not bind.
   Two calls work fine and read clearly, but the constraint is worth a half-sentence in the recipe
   so the next porter does not first write the vector form and have it rejected.

6. **Gap — the recipe's build instructions assume a checkout, not a git worktree.**
   *Ensure the Metal 2.0 host-side legality checks are enabled* and *Build* both go straight to
   `./build_metal.sh`, which fails immediately in a fresh **worktree** with
   `CMake Error at CMakeLists.txt:5 (message): Missing submodules.` Git worktrees do not inherit the
   parent clone's submodules, and `workspace_setup.md` covers the clone case only. One line in
   *Workspace bootstrap* — run `git submodule update --init --recursive` first if you are in a
   worktree — would save the cycle. Worth fixing because the bulk-port effort makes worktrees the
   natural way to run several ports off one clone.

## Open items for downstream

*Captured as they arise; the shared-kernel entry is filled once the fork rung is executed.*

1. **Shared kernel touches — rung 2, four forks created.** All four sources are *lent*: they live in
   this op's directory and a second emitter in the same directory
   (`minimal_matmul_factory_helper_common`) binds them for the CCL composite. No `_metal2` sibling
   existed, so this port created them.

   | fork created | legacy original |
   |---|---|
   | `device/kernels/dm_in0_sender_metal2.cpp` | `dm_in0_sender.cpp` |
   | `device/kernels/dm_in1_sender_out_metal2.cpp` | `dm_in1_sender_out.cpp` |
   | `device/kernels/compute_metal2.cpp` | `compute.cpp` |
   | `device/kernels/matmul_dataflow_common_metal2.hpp` | `matmul_dataflow_common.hpp` |

   The pointer comment landed in all four legacy originals; nothing else in them changed.
   **Remaining unmigrated consumer (the sunset list):**
   `ttnn/cpp/ttnn/operations/experimental/ccl/minimal_matmul_strided_reduce_scatter_async`, via
   `minimal_matmul_factory_helper_common`. When it migrates, the legacy copies can be deleted and
   the forks take their names.

   **The forks deliberately do not convert the CCL-only regions.** `FUSE_AG`,
   `SRS_FUSE_OP_SIGNALER`, `MM_WINDOW_BLOCKS` and `READ_FROM_LOCAL_INPUT` are defined only by the
   legacy emitter, so the forks never compile those blocks. They are carried verbatim (a dead
   `#ifdef` branch need not be valid C++) with a header comment saying so. They still read
   positional runtime args the forked files no longer have, so **the CCL port must convert them
   along with giving the spec the matching bindings** — that work belongs to the CCL op, not here.
   This also means the forks' Device 2.0 residue (the raw-L1 semaphore waits the audit discusses)
   is unreachable in every configuration this factory emits.

2. **`TensorBindingSequence` deserves a patterns-catalog entry.** The mechanism is documented only
   in `advanced_options.hpp` and `tensor_accessor.h`; the catalog has nothing on variadic tensor
   bindings, and the audit recipe's *RTA varargs* subject does not mention it (a porter meeting a
   CTA-sized output set is likely to reach for varargs, which would be the smuggling anti-pattern).
   The shape here is small enough to lift verbatim: declare `out0..out{N-1}` `TensorParameter`s in a
   loop, add one sequence naming them, and consume with `make_tensor_accessors(tensor::<name>)`.

3. **Two dead legacy args were carried across unchanged** (they are the ops team's, not the port's):
   `in3_tile_size` — emitted as a CTA and never read by the kernel — and `max_defer_write_k_block`,
   read but used only under `SRS_FUSE_OP_SIGNALER`. Both are named args now, so the dead CTA is
   visible as a named value nothing consumes rather than a silently skipped positional slot.

4. **Dispatch cost was not measured on either concept.** See *Concept deviation*; worth measuring
   before the delete-the-override shape is generalized to other ops.
