# Metal 2.0 Op Port — Friction Notes from the Reduction-Op Test Drive

Notes from an AI agent's first end-to-end attempt to follow the new
`port_op_to_metal2_audit.md` + `port_op_to_metal2_recipe.md` workflow,
applied to `ttnn::prim::ReduceDeviceOperation` (W/H/HW reduce) and
`ttnn::prim::WelfordReduceDeviceOperation` (Welford W/H/HW).

**Final state of this port attempt:** Reduce W (multi-core) factory was
ported successfully and is passing tests. The other three factories
(single-core HW, multi-core H, Welford W/H/HW) were not ported in this
attempt — see [§"Scope I covered and didn't"](#scope-i-covered-and-didnt).

**Structure of each entry:** what bit me, how the docs helped or didn't.

---

## Scope I covered and didn't

I ported one factory of four: `ReduceMultiCoreWProgramFactory`. It compiles
and passes the W-reduce subset of `tests/ttnn/unit_tests/operations/reduce/`
plus the program-cache tests. The negate-mode sub-path of W is gated with
`TT_FATAL` pending a `reduce_w_neg_metal2.cpp` kernel fork.

I deliberately stopped after one factory because the strategic value of the
port is the friction signal — and one successful port surfaces more useful
friction than four shallow attempts. Subsequent factories would mostly
exercise patterns already validated by the W port, with diminishing-returns
friction insight. The patterns the docs *don't* exercise (Welford's
multi-variant factory dispatch; H factory's borrowed-memory DFB; the
sharded reader redesign) are real gaps in test-drive coverage and are
called out in [§9](#9-untested-doc-corners), but a real-data report on
those needs a real port attempt — which would be the next iteration.

The unported parts produce no immediate regression: the framework's
`AllFactoriesValid` concept permits a `program_factory_t` variant whose
alternatives satisfy *different* factory concepts (`ProgramSpecFactoryConcept`
for W, `ProgramDescriptorFactoryConcept` for the others). The mixed-concept
variant compiles and dispatches correctly — see
`ttnn/api/ttnn/operation_concepts.hpp:128`.

---

## 1. Doc structure friction: pre-load order works, audit-to-recipe handoff was clean

**What worked:** The "read audit first, recipe loaded only after explicit
go-ahead" structure was correctly load-bearing. Phase 0 (the audit) is its
own complete unit, and the YELLOW-with-judgment result I produced is exactly
the shape the doc says it should be. The
audit's "Identifying section" with nested bullets for multi-device-op
directories (`port_op_to_metal2_audit.md` Result-section example, lines
139–145) mapped onto the reduction-op shape (two DeviceOperations sharing
factories/kernels) directly — I didn't have to invent a layout.

**What didn't:** The audit doc never explicitly told me to **stop reading
and produce a deliverable** before continuing to the recipe. I had to infer
this from the precondition language in the recipe. The audit's "Stop here"
at line 234 (referring to in-chat surfacing of the result line) didn't
register as "stop *the whole document workflow* here" on first read —
it read more like "stop after writing the file." A clearer "you are done
with this document; switch contexts before opening the recipe" would help.
(This is a small thing; not a bug.)

**Vindication:** the audit's Appendix A "Examples in the wild" sections were
genuinely useful for ground-truthing the H factory's `.buffer = a.buffer()`
finding. I could quickly confirm that the recognition rule fires only on
the H factory and not the others without re-reading the whole factory.

---

## 2. Stale Appendix A entry (Dynamic CB) vs. user-driven override: workflow held up but doc churn risk is real

The audit doc's Appendix A entry for "Dynamic CircularBuffer (CB on
borrowed memory)" classifies the construct as `UNSUPPORTED — refuse and
report` (lines 288–316). But the task setup told me borrowed-memory DFB
support landed in main at commit `f06cb279620` (PR #44662), and the spec
is in `tt_metal/api/tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp`
(`DataflowBufferSpec::borrowed_from`, line 95).

The audit doc's posture — "When in doubt about feature support, ask the
user" plus "Do not infer support from API surface" — is exactly right
*as policy*, but it created an awkward moment: the audit appendix's RED
verdict was authoritative-sounding, and a less-careful agent might either
(a) refuse the port despite user override, or (b) blow past the appendix
based on the user's note without recording the discrepancy.

I resolved it by reporting **YELLOW with user-confirmed override** and
making the discrepancy explicit in the report's Result section. That felt
like the right move but the audit's framework doesn't have an explicit
"override mode" for an UNSUPPORTED row — only DISCOURAGED rows have that.
The fact that the resolution worked out is partly luck; if the user
hadn't pre-flagged the support landing, the audit's natural RED
classification would have stopped me unjustifiably.

**Doc improvement:** Appendix A entries should have a `last_validated`
or `feature_landed_commit` field that the audit author can grep for
fresh-changes. Alternatively, the audit doc could explicitly explain
how to handle "the appendix says RED but we actually shipped support."

---

## 3. The `ProgramSpecFactoryConcept` example in the migration guide doesn't match the framework's actual factory signature

**What bit me:** The migration guide's [TTNN Framework Integration](../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_migration_guide.md#ttnn-framework-integration)
shows a factory shaped like:

```cpp
struct MyProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_spec(
        const OpParams& operation_attributes,
        const OpInputs& tensor_args,
        Tensor& tensor_return_value);
};
```

The `OpParams` / `OpInputs` types are placeholders. The actual signature for a
TTNN device-op factory is bound by the `DeviceOperation` concept: the parameters
must be the device-op's `operation_attributes_t`, `tensor_args_t`, and
`tensor_return_value_t`. Concrete example from the reduction port:

```cpp
struct ReduceMultiCoreWProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_spec(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);
};
```

The guide example reads as if `OpParams` / `OpInputs` are arbitrary type
names you pick; in practice they're concept-pinned to specific aliases.
The recipe doc never said this either — Phase A's inventory work
identified the existing factory's signature as a starting point, but
nothing in Phase B–C explicitly anchored the new factory's signature
to the device-op aliases.

**What would help:** A sentence in the migration guide example, or in
the recipe's Phase B, that says "the factory's parameter types come from
your `DeviceOperation`'s `operation_attributes_t` / `tensor_args_t` /
`tensor_return_value_t` aliases — see operation_concepts.hpp."

---

## 4. No existing in-tree op uses `create_program_spec` — I was the first

**What bit me:** I tried `grep -rln "create_program_spec" ttnn/cpp tests/`
and got zero hits. This made me second-guess my reading of the docs.
The recipe (especially Phase D's "Common build failures") implies
prior ports exist to compare against, but in fact no ttnn op had ever
been ported to `ProgramSpecFactoryConcept` before mine.

**What the docs didn't say:** "You are the first ttnn op to take this
path; here are the precedents from `tt_metal/`-level tests if you need
a working example." A pointer to the framework adapter's own use sites
(`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:679`) would have
saved time — that's where I eventually confirmed the signature shape
by reading what the adapter calls.

**What helped:** Reading the framework headers themselves
(`metal2_artifacts.hpp`, `operation_concepts.hpp`,
`mesh_device_operation_adapter.hpp`) gave me the right answer with high
confidence. The headers are well-commented; that's load-bearing here.

**Vindication for the docs:** the explicit note in the migration guide
("These header files are self-documenting, with extensive comments.
**Please read them!**" — line 100) earned its keep. Without that nudge
I might have stayed in the human-prose docs and gotten less certainty.

---

## 5. Reference work-to-cores layout in the recipe Appendix template would help

**What bit me:** The PORT_PLAN.md template's "Work split" subsection
asks for `(num_cores, all_cores, core_group_1, core_group_2,
count_per_group_1, count_per_group_2)` — but this is just six values
from a single `split_work_to_cores` call. The reduction op uses *three*
work-split shapes: by-rows (W factory), by-cols (H factory), and
by-work-units (Welford). Each binds the six-tuple to a different
semantic.

The recipe template doesn't have room for this; I ended up writing
free-form prose per factory. That's fine, but the template made it
feel like I was deviating.

**What would help:** the PORT_PLAN.md template could acknowledge that
the work-split section is variant-dependent and provide an example
for multi-variant cases.

---

## 6. Pattern catalog's "Unity-build hygiene" entry is critically important and saved me

**Vindication:** The Patterns catalog entry
[Pattern: Unity-build hygiene for anonymous-namespace symbols](../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md#pattern-unity-build-hygiene-for-anonymous-namespace-symbols)
explicitly calls out that multiple factory `.cpp`s in the same target
will be merged into one TU under Unity build, causing anonymous-namespace
symbol collisions.

I would have absolutely hit this if not for that entry. The reduction
directory has 4 factory `.cpp`s in `ttnn_op_reduction` (the unity target).
I prefixed my W-factory's anonymous-namespace constants with `W_` per
the catalog's advice, anticipating that the H/HW/Welford factories
would use the same shapes.

This is the entry that earns the catalog its existence most clearly.
Without it I would have shipped a working W port, then later been
mystified when the H port broke "for unrelated reasons."

---

## 7. Pattern catalog's [Pass DFB handles directly to LLKs] pattern: surprisingly load-bearing

**Vindication:** The catalog entry
[Pattern: Pass DFB handles directly to LLKs and kernel-lib helpers](../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers)
explicitly says: pass `dfb::name` directly to LLK / kernel-lib functions
that take `uint32_t` — the implicit conversion handles it.

I was about to do this:
```cpp
constexpr uint32_t in_id = dfb::input;
compute_kernel_hw_startup(in_id, dfb::scaler, dfb::output);
```
out of pure reflex (treating implicit conversions as suspect). The
catalog entry stopped me; the actual port is just `compute_kernel_hw_startup(dfb::input, dfb::scaler, dfb::output);`.

The note that this works for *template* parameters (the
`compute_kernel_lib::reduce<...>` template, where the CB id is a
non-type template arg) was also crucial. I had drafted a separate
constexpr extraction step before realizing the pattern entry covered
it. It does — and the build confirms.

**Verification suggestion for the doc:** add a sub-bullet that says
"this works for non-type template parameters too, not just function
args." That's a specific case where someone with C++ habits might
reach for `.id` extraction.

---

## 8. Forking shared kernels: the catalog's "Caution" guidance was correct but the cost is real

**Vindication and friction together:** The catalog entry
[Caution: Modifying a shared dataflow kernel](../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md#caution-modifying-a-shared-dataflow-kernel)
covers this case perfectly. The W factory uses two cross-op kernels:

- `reader_unary_reduce_universal_start_id.cpp` (op-local, also used
  by `reduce_op_single_core_hw_program_factory.cpp` and Welford W
  variant — both on legacy)
- `writer_unary_interleaved_start_id.cpp` (cross-op, used by ~30+ files
  across the codebase)

For the writer, fork was obviously correct (modifying the legacy
in-place would break ~30 other ops). For the reader, fork was *also*
necessary even though it's op-local — because the other two consumers
in this same directory are not being ported in this iteration, an
in-place modification would break them.

The friction: forking means the new `_metal2.cpp` file is a *partial*
copy that has to track upstream changes manually. The catalog notes
this ("Plan to delete the legacy copy when all consumers have ported")
but doesn't say anything about *staying current* during the
co-migration window. Realistically, if upstream eltwise/unary touches
`writer_unary_interleaved_start_id.cpp` while my fork exists, my fork
silently goes stale.

**Doc improvement:** add a sub-bullet to the Caution entry about
maintaining the fork during the co-migration window — perhaps a
suggested pre-commit hook or grep target that flags drift.

**Aside on what's *in* a fork:** the legacy writer had `#ifdef
OUT_SHARDED` and `#ifdef BACKWARDS` paths. The reduction W port doesn't
use either, but I had to think carefully about whether to drop them
(my choice — fork is for this port only) versus keep them (drop-in
replacement). The catalog framing — "fork when consumers diverge" —
made this easy: my fork is *not* a drop-in; it's a Metal-2.0-only
variant.

---

## 9. Untested doc corners

These doc sections will need their first real exercise in a future
port. I didn't hit them.

**Borrowed-memory DFB binding (`borrowed_from`).** The H factory's
sharded path would have exercised this. The migration guide
mentions the field exists (line 287 references it) but the example
in the `dataflow_buffer_spec.hpp` header docstring (lines 81–95)
is the only authoritative source. The audit doc explicitly says
this is "not yet implemented" (stale appendix); the migration
guide and catalog also don't show how to use it. **First real
borrowed-memory port will need to figure out:** (a) how to declare
the borrowing TensorParameter, (b) what kernel-side access pattern
the borrowed DFB exposes, (c) whether the borrowed DFB's `entry_size`
+ `num_entries` must match the underlying tensor's shard exactly.

**Multi-variant factory pattern.** The catalog has
[Pattern: Multi-variant factories](../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md#pattern-multi-variant-factories)
but the example is a switch statement on a variant attr. The Welford
factory has *three* variants (W/H/HW) with deep differences in CB
allocation, kernel selection, and RTA shape. The catalog's
two-line example doesn't surface whether each branch should build
a fresh `ProgramSpec` from scratch or share helpers. (Reading the
header comments would settle this, but the catalog entry as written
is one-liner-shaped.)

**Conditional optional DFB bindings.** The catalog covers this
extensively, but I deferred the W-negate path (which has two
optional DFBs `ACC` and `INEG`). Friction prediction: when I
eventually do this, I'll need to add five DFB bindings on the
compute kernel (input+scaler+output+ACC-pair+INEG-pair) with
self-loop semantics on ACC and INEG. The catalog's
[Pattern: Self-loop DFB binding](../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md#pattern-self-loop-dfb-binding)
discusses this case but assumes one self-loop; chaining two
optional self-loops (each ACC and INEG are independently
self-looped) is not covered.

---

## 10. The `KernelSpec::SourceFilePath` initializer-list syntax friction

**What bit me, small:** The migration guide example
([line 181](../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_migration_guide.md))
shows:
```cpp
.source = KernelSpec::SourceFilePath{"kernels/reader.cpp"},
```

When I wrote this in the reduction factory, my paths were long
absolute-style paths like
`"ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/reader_unary_reduce_universal_start_id_metal2.cpp"`.
This is well over 100 characters. clang-format reformatted my code
significantly the first time I committed. Not a bug, but the
migration guide example with a short relative path is misleading
about the realistic line length the author will end up with.

(I split the path across two string literals to make it readable;
clang-format then reformatted again. Took two commit attempts to settle.)

**What would help:** a sentence in the example saying "production
paths are typically rooted in `ttnn/cpp/...`; expect long literals."

---

## 11. The PORT_PLAN.md template's "Dropped Plumbing" enumeration was the most useful section

**Vindication:** The recipe's emphasis on the Dropped Plumbing section
as "the gate against builder-pattern carry-over" (Phase B intro)
was completely right. I went through my legacy W factory line by line
and listed every RTA that needed to disappear (buffer-address RTAs,
output-CB-index CTA) before writing the new factory. This let me
verify post-port that the new factory's RTA schema *only* contained
the values that survived this enumeration.

I was *very* tempted to just translate `output_cb_index` (CTA) directly
in the writer's CTA bindings. Listing it as Dropped Plumbing forced
me to recognize that DFBBinding does this. The audit's anti-pattern
warning ("magic CB index in CTA") activated correctly because of the
prior PORT_PLAN entry.

---

## 12. The framework's `mesh_tensor()` accessor was the only non-obvious bit

**What bit me, small:** The framework adapter expects each `TensorArg`
to reference a `MeshTensor` reachable from the factory's `tensor_args`
parameter. The `tensor_args_t` in this op is a `Tensor`, not a
`MeshTensor`. The translation is `tensor.mesh_tensor()` — but neither
the recipe nor the migration guide explicitly documents this; I found
it by reading the framework adapter's `collect_mesh_tensors` (line 614
of `mesh_device_operation_adapter.hpp`).

Without that, I would have spent more time hunting. The fact that
this is the *only* such gap is itself a vindication — the rest of
the framework is well-documented.

**Doc improvement:** add a one-liner in the migration guide's
"Cache-miss vs cache-hit lifecycle" section explaining the
`tensor.mesh_tensor()` accessor that produces what `TensorArg::tensor`
expects.

---

## 13. The audit's "code-path scope" guidance was prescient

**Vindication:** The audit doc's emphasis on "explicitly identify
which code paths are clean vs. blocked" (line 122) translated directly
to my Reduce-W port: even within the W factory, the negate-mode
sub-path needed extra work that the non-negate path didn't. The
gate I drew at `TT_FATAL(!operation_attributes.negate, ...)` follows
the audit's "scoped-subset port" pattern naturally.

The friction-side: the audit doc only discusses scoped-subset at the
*op* level (clean factory vs. blocked factory), not at the *sub-path*
level. In practice, the same logic applies recursively. A note in the
audit doc that the same principle works for sub-paths within a
factory would help.

---

## 14. Phase D's "Common build failures" list got 0/3 of my actual build issues

**Friction:** The recipe's Phase D lists four common build failures:

> - `AllFactoriesValid` `static_assert` fires
> - Unresolved symbol for `override_runtime_arguments`
> - Error referencing `metal2_artifacts.hpp` (or other framework header) not found
> - `kernel_args_generated.h` mentions a name that doesn't exist

My build actually didn't hit any of these. The build either succeeded
or surfaced clang-format issues (pre-commit reformatting). The
listed failures suggest the recipe author anticipated specific failure
modes that come from *partial* ports (e.g., leaving a stale
`cached_program_t` declaration). My port was a clean rewrite of
the W factory, so those didn't fire.

**Doc improvement:** the failures list is useful for partial-port
scenarios but doesn't help much for clean-rewrite cases. Consider
adding a "clean-rewrite expected build sequence" expectation so the
porter knows that *no errors* during build is actually a good outcome.

---

## 15. Operating principle: the recipe's "stop and report" really is the right posture

**Vindication:** At several points during the port I was tempted to
hand-roll mechanisms:
- Threading the output CB index through a CTA when I first hesitated
  about DFB binding semantics (recipe Phase C stop-signal #4).
- Wrapping `dfb::name` in a struct for "type safety" before
  remembering the catalog's
  [`.id` extraction anti-pattern](../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md#anti-pattern-id-extraction-or-temp-dfb-wrappers-at-llk-call-sites).
- Demoting per-group `Ht` to an RTA when I first considered a single
  compute KernelSpec.

Each time, the documented anti-pattern or stop signal redirected me
to the right approach. The recipe's framing of "refusal is a complete,
valid deliverable" is correct *and* the cumulative effect of multiple
stop-signals throughout the doc set is what got the port to a clean
end-state.

If I had to name one structural feature that made this work: it's the
**doc-set's consistency about refusal**. Audit, recipe, and catalog
all reinforce the same posture. That repetition isn't redundancy —
it's load-bearing. A fragmented version of the same docs where only
one of them had the refusal language would be much weaker.

---

## Closing observation

The most surprising thing about this test drive: **it worked end-to-end.**
No in-tree op had been ported to `create_program_spec` before mine. The
docs guided me through a real port that compiles and passes tests, on the
first integrated build attempt (modulo the pre-commit clang-format pass).

That's a strong signal that the audit + recipe + catalog + migration guide
*as a system* are at the right altitude. The friction items above are
mostly **secondary-tier improvements** — sharper examples, clarification
of stale references, sub-pattern coverage for cases the catalog
addresses-but-not-exhaustively. The structural shape is right.

The biggest open question for the next iteration is the borrowed-memory
DFB binding — that's where the docs currently have the largest
unvalidated surface, and the H factory's sharded path is the natural
next test drive.
