# Metal 2.0 Port Report — `moreh_nll_loss_step2`

## Outcome

**`PORTED`** — the op's single factory (`MorehNllLossStep2DeviceOperation::Factory`) is converted to
`ProgramSpecFactoryConcept`, together with all seven kernel entry points it can bind. All twelve
reachable configurations (3 rank paths × weight present/absent × divisor present/absent) build, and
the confirmed test set passes at the pre-port baseline with the Metal 2.0 legality checks forced on
and proven live. No factories remain unported — the op has exactly one.

## Provenance

- **Recipe docs (this port):** `28c1b0b4224 2026-08-25 docs(metal_2.0): a run in flight freezes the kernel sources`
- **Audit docs (inherited):** `28c1b0b4224 2026-08-25 docs(metal_2.0): a run in flight freezes the kernel sources`

## TTNN ProgramFactory

### Concept realized

**`ProgramSpecFactoryConcept`** — as the audit chose, unchanged. The ported-from factory had no
`override_runtime_arguments`, so nothing was translated and the framework refreshes tensor bindings on
a cache hit. `Factory::create_descriptor` became `Factory::create_program_artifacts` **inside the
existing `Factory` struct**: the struct was already the single alternative of
`program_factory_t = std::variant<Factory>`, so the direct-descriptor exception (`ttnn_factory.md`
exception 3) did not apply and no nested struct had to be introduced.

The three rank paths stayed three file-local builders — internal code paths, not factories, exactly as
the brief said. Each now returns `ttnn::device_operation::ProgramArtifacts`;
`create_program_artifacts` remains the thin rank dispatcher it was.

### Device-op-class edits

- **Pybind entry points removed:** **none.** `moreh_nll_loss_nanobind.cpp` binds only the user-facing
  `ttnn::moreh_nll_loss`, never `create_descriptor`, so the port removes no user-visible Python
  surface and exception 1 never fired.
- **Custom `compute_program_hash`:** **none** — the op uses the default reflection-based hash, with no
  backdoor `attribute_values` / `to_hash`. Nothing to leave intact.
- **The one header edit, forced by the return-type change:**
  `device/moreh_nll_loss_step2_device_operation.hpp` swaps
  `#include <tt-metalium/program_descriptors.hpp>` for `#include "ttnn/metal_v2_artifacts.hpp"` and
  changes the `Factory` method's declaration. Nothing else in the device-operation class was touched —
  `device/moreh_nll_loss_step2_device_operation.cpp` is **byte-identical** to its pre-port revision
  (confirmed: `git diff --stat` against the merge-base is empty), so `validate_inputs`,
  `compute_output_specs` and `create_output_tensors` are untouched.

### Open items

- **Relaxation candidates:** none applied, and none confidently identifiable. The audit recorded
  `TensorParameter relaxation: none`, and its usual mining source — a custom `compute_program_hash`
  revealing which tensor properties the op actually depends on — does not exist here. Strict
  `TensorSpec` matching throughout.
- **A capability this op would benefit from:** nothing missing. It needs no op-owned tensors, no
  op-owned `GlobalSemaphore`s (it has no semaphores at all), and no per-coord program variation, so the
  base concept fits without strain.
- **Concept-fit friction:** none. This is close to the ideal shape for the base concept.

## Handoff points

1. **Framework team — audit Question 1 remains open, and the port did not need it answered.**
   Whether `compute_kernel_hw_startup(dfb::tmp_weight, dfb::tmp_input, dfb::output)`
   (`device/kernels/moreh_nll_loss_step2_kernel.cpp:25`) constitutes an endpoint *binding*, as opposed
   to a hardware-configuration reference that needs the binding token but not an endpoint, is still
   unsettled. Per the invoker's instruction the port took the brief's guidance and moved on. Recording
   what the port actually did, so the answer can be checked against it later: `tmp_weight`'s
   `DataflowBufferSpec` is built **unconditionally** in all twelve configs, and in the six no-weight
   configs the compute kernel is bound to it as both PRODUCER and CONSUMER (a self-loop). That choice
   is correct under **both** readings of the question — one role-free toucher → self-loop, or zero
   touchers → a conditional DFB that would nonetheless have to exist, because the token must resolve
   for the `compute_kernel_hw_startup` call *and* for the unconditional accessor construction at
   `moreh_nll_loss_step2_kernel.cpp:14` to compile. If the answer turns out to be "not a binding," the
   only consequence for this op is that the self-loop label was cosmetic rather than earned; nothing
   about the spec changes. **Owner:** Metal 2.0 framework team. **Reporter:** this port.

2. **No boundary-rule assumption violations, no kernel-lib gaps, no framework gaps.** No call site
   outside the op directory required a `sem::` or `tensor::` handle — both donor headers
   (`ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp`,
   `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp`) take `DataflowBuffer` **by value** at every symbol
   this op consumes, so every call site is unchanged and no donor-side edit or fork was needed. No
   `_metal2` fork was reused or created anywhere; the port wrote no file outside the op's own
   directory. No audit-time UNSUPPORTED finding bit during the port.

## Successes

- **[Anti-pattern: Demoting per-group CTA to RTA](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#anti-pattern-demoting-per-group-cta-to-rta)
  fired exactly as designed, on the op it was written about.** The legacy factory populates a per-core
  compute runtime-arg vector carrying precisely `units_per_core` — and the compute kernel reads no
  runtime args at all, taking that count from a compile-time arg instead. Collapsing the two compute
  descriptors into one `KernelSpec` fed by that existing RTA is the obvious-looking simplification, and
  it is wrong: `per_core_tile_cnt` bounds the kernel's main loop
  (`device/kernels/moreh_nll_loss_step2_kernel.cpp:12`, loop at `:49`), so the demotion would cost
  compile-time unrolling — a perf regression the port has no licence to make. The port kept two
  `KernelSpec`s in two `WorkUnitSpec`s and **deleted** the dead RTA rather than adopting it
  (`device/moreh_nll_loss_step2_program_factory.cpp:360-372` and the two sibling sites). The brief's
  "this op baits the trap unusually well" warning was accurate, and the catalog entry's worked example
  transferred one-to-one.

- **[Two-toucher DFB → assign 1P+1C](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-two-toucher-dfb--assign-1p1c-dual-instance-work-split)'s
  "re-derive, don't transcribe" instruction, plus the *Constraint* paragraph distinguishing the
  disjoint-node work split from the same-grid one, is what kept the multi-binding flag out of this
  port.** Four DFBs (`tmp_input`, `output`, `tmp_weight`, `divisor`) are referenced by **three**
  `KernelSpec`s — a reader-or-writer plus *both* compute specs — which counts to three bindings and
  reads like a multi-binding case. Re-running the census per *node* rather than per spec gives two, and
  `dataflow_buffer_spec.hpp:42-50` states the rule at the field: multiple specs may share one endpoint
  role given non-overlapping node coverage, the same kernel kind, and identical binding-site
  parameters. Going to the header — as [§Read this first](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#read-this-first)
  recommends over hunting for a precedent — settled it definitively in less time than finding a
  comparable port would have taken.

- **[Compiler options](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#compiler-options)
  caught a defect that has nothing in the code to look at.** `grep -n opt_level` over the legacy
  factory returns *nothing*, which reads as "no setting, nothing to carry" — and is precisely the
  dominant shape of the miss: an absent `KernelDescriptor::opt_level` still resolves to **`O3`** on a
  `ComputeConfigDescriptor`, while `KernelSpec::compiler_options` defaults to `O2`. Without that
  section this port would have silently dropped an optimization level on all six compute specs. The
  section's insistence that this be run as its own mechanical item, rather than eyeballed alongside
  `hw_config`, is what made the difference. Landed at
  `device/moreh_nll_loss_step2_program_factory.cpp:177`, and deliberately inside the shared
  `make_compute_spec` helper so it cannot be set on one instance and missed on the other.

- **[Hardware configuration](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#hardware-configuration)'s
  `unpack_modes` warning — all three of its "three things change at once" — was needed here.** The
  legacy vector is `vector<UnpackToDestMode>(NUM_CIRCULAR_BUFFERS, Default)`: uniform, and `Default`
  maps to `UnpackToSrc`, which one normally expresses by *omitting* every entry. That reasoning gets you
  a spec the validator rejects, because the op's intermediates become `Float32` exactly when
  `enable_32_bit_dest` is on (`fp32_dest_acc_en ? Float32 : data_format` — the "very common idiom" the
  section names), and the compute kernel consumes all five. The section's note that **the trigger is
  the DFB's format, not the op's tensor dtypes** is the load-bearing sentence: this op has no `Float32`
  *tensor* anywhere — `validate_inputs` pins every input to `BFLOAT16` — so a dtype-led reading would
  have concluded no entries were needed. Five explicit `UnpackToSrc` entries at
  `device/moreh_nll_loss_step2_program_factory.cpp:102-110`, each derived from the legacy vector rather
  than guessed.

- **The dead-declaration trap was worth the brief spending a table on it.** Deleting rather than
  converting the four dead CB constants (`cb_output` in all three readers, `cb_weight` in the compute
  kernel) is counter-intuitive under the port's own "convert every CB index to a `dfb::` handle" rule —
  and converting them would have manufactured endpoints that don't exist, pushing `output`'s per-node
  census to three and turning `weight`'s clean self-loop into a spurious 1P+1C. Both would have been
  *silent*: the spec builds, and the wrong disposition looks deliberate.

## Friction

### Gaps

1. **The recipe's `unpack_modes` guidance does not say what to do when the *conditional-gating* rule and
   the *required-entry* rule could interact — and this op came within one design choice of hitting it.**
   The section says a conditionally-bound DFB's `unpack_modes` entry "must be gated on the same
   condition as its binding," and separately that a `Float32` DFB consumed under `enable_32_bit_dest`
   *requires* an entry. Here the two rules happen not to collide: all five `Float32` intermediates are
   bound in every one of the twelve configs, so one ungated table is correct. But that is luck, not
   structure — had `tmp_weight`'s spec been conditional (which is exactly what audit Question 1's other
   answer would have implied), the entry would have had to be gated on the same condition, and the two
   rules would have had to be composed by the porter with no worked example. **Suggested addition:** one
   sentence under item 3 of the `unpack_modes` subsection noting that when a required `Float32` entry
   names a *conditionally-bound* DFB, the entry inherits the binding's condition — and that a
   conditionally-bound `Float32` DFB consumed under `enable_32_bit_dest` therefore needs *both* rules
   applied at once, not either one.

2. **The [anti-pattern self-audit](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#anti-pattern-self-audit)
   has no item for a local left dangling by dropped plumbing, and that was this port's only build
   failure.** Dropping the dead `N` runtime arg from the 4d reader left
   `const auto origin_N = input_shape_without_padding[0];` with no remaining reader, which `-Werror`
   `-Wunused-variable` turned into a hard build failure in the unity build. Cheap to diagnose here —
   one error, named file:line — but the general shape is worth a line, because dropping plumbing is a
   *mandated* part of every port and every dropped RTA is a candidate to orphan the host local that
   computed it. Note the failure is loud only when the value came from a plain local: a value that
   arrives via a structured binding (this factory's `math_fidelity`, `math_approx_mode`,
   `dst_full_sync_en`, and `all_cores`, all genuinely unused post-port) produces **no** warning at all,
   so the compiler is an incomplete backstop. **Suggested addition:** a checklist item — *"every host
   local that fed a dropped RTA/CTA is either still read or deleted; note that a value destructured from
   a structured binding will not warn."*

3. **The `METAL2_*.md` artifacts are the one thing whose own relative links the recipe never discusses,
   and they are easy to get wrong in the direction that looks right.** [Generated docs in the op
   directory](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#generated-docs-in-the-op-directory)
   is emphatic that *code* must not cite these files or the recipe, and the self-audit greps `.cpp` /
   `.hpp` / `.h` for `.md` — correctly. But the plan and report are themselves asked to "cite the doc
   section by name and link," and a link from an op-directory `.md` into `docs/…/metal_2.0/` is a deep
   relative path (`../../../../../../../docs/…`) that no check validates and that is trivially
   miscounted — while *looking* exactly like the sanctioned citation the Successes section asks for.
   Since these artifacts are deleted before merge, a broken link costs only a reviewer's click, so this
   is minor. **Suggested addition:** one line saying whether the four artifacts should link into the
   recipe docs by relative path at all, or cite section names as plain text — and if by path, that the
   depth is worth checking with `ls`.

### Confusion

4. **"Preserve the multiplicity" and "don't re-architect the legacy loop" pull in opposite directions
   for a factory whose three internal paths are near-copies, and the recipe does not say which wins.**
   This factory's three rank builders are ~85% identical, and the port had to decide how much to share:
   the three legacy `impl_*` functions duplicate their CB creation, kernel wiring and compute config
   wholesale. [Scope discipline](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#host-side-stay-in-the-lane)
   says don't refactor adjacent code; [Unity-build hygiene](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-unity-build-hygiene-for-anonymous-namespace-symbols)
   says hoist truly-identical declarations into shared scope; and *Compiler options* effectively
   *requires* sharing the compute-spec construction, since "specs built through a shared helper are no
   exception — the level is per `KernelSpec`" is much easier to guarantee with one helper than with six
   inline sites. The port settled on: hoist the name constants and the three mechanical helpers
   (`make_dfb`, `bind_self_loop`, `make_compute_spec`, plus `make_compute_hw_config`) to file scope, and
   keep the three builders otherwise parallel to the legacy three, duplication included. That reads as
   the right balance, but it was a judgment call made without guidance, and a different porter could
   defensibly have collapsed the three builders into one parameterized function — a materially different
   diff. **Suggested addition:** a sentence in *Scope discipline* stating that hoisting mechanically
   identical spec-construction helpers is in scope (it is forced by the per-`KernelSpec` fields), while
   merging the legacy code paths themselves is not.

5. **The [self-audit](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#anti-pattern-self-audit)'s
   forced-scaffolding check is guaranteed to fail while the port is being verified, which briefly reads
   as a real finding.** `git diff --name-only "$BASE" | grep -E '^tt_metal/'` necessarily lists
   `program_spec.cpp` and `program_run_args.cpp` for as long as the legality-check forcing is in place —
   and the forcing has to stay in place *through* the test run that the same section says to trust.
   So the check can only pass after a revert that must happen after verification. That ordering is
   implicit in "belong to your working tree only," but the checklist presents the two as peers.
   **Suggested addition:** a clause on that item — *"run this last, after the verification run and after
   reverting the forcing; it is expected to fail until then."*

6. **A small proving-the-markers gap.** [Ensure the legality checks are enabled](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#ensure-the-metal-20-host-side-legality-checks-are-enabled)
   says to add the *same* marker string in each file and then confirm "**two markers present**." With
   one identical string, a grep cannot tell two fresh translation units from one fresh and one stale —
   which is the exact failure the step exists to rule out. Suffixing each marker with its filename
   (`METAL2_CHECKS_FORCED program_spec.cpp` / `… program_run_args.cpp`) makes the check actually
   conclusive while still matching a grep for the bare string. **Suggested change:** state the marker as
   including the file name.

## Open items for downstream

- **Shared kernel touches: none.** The op owns all seven kernel `.cpp` files and no other op binds any
  of them (verified with `grep -rl <filename> ttnn/cpp/ttnn/operations/`, disambiguated to this
  factory's own `source` assignments). Nothing borrowed, nothing lent, and the three rank paths bind
  disjoint reader/writer sources so there is no intra-op sharing either. The one source the paths share,
  `moreh_nll_loss_step2_kernel.cpp`, is shared *within the single factory being ported* and so converted
  atomically with it. **No `_metal2` fork reused, none created, no sunset list, nothing to coordinate.**

- **Three dead runtime args the audit and brief both missed, dropped as an extension of the invoker's
  decision.** The brief listed four dead RTAs and the invoker chose to drop rather than name them. The
  inventory found three more of the identical kind, each read into a local and never referenced
  (confirmed by word-boundary grep — one occurrence each, the declaration itself):
  - the 3d reader's `N` — host `:431` pre-port, read at `reader_..._3d.cpp:21`
  - the 4d reader's `N` — host `:661` pre-port, read at `reader_..._4d.cpp:20`
  - the 4d reader's `Wt` — host `:663` pre-port, read at `reader_..._4d.cpp:22`

  The port dropped these three too, taking the total to seven, on the reasoning that the invoker's
  decision was about a *class* and the brief simply had an incomplete list. Flagged here rather than
  folded in silently, because it is a deviation from the literal instruction ("the four the brief
  lists") and is cheaply reversible — naming each instead is a one-line change on the host plus one in
  the kernel. Two of the seven kept a near-twin alive, and the distinction is easy to get backwards:
  the **2d** reader's `N` **is** used (`:48`) and the **3d** reader's `element_size` **is** used (`:64`),
  so those two were kept.

- **Ops-team findings the port deliberately did not fix** (each preserved as-is; the audit's *Misc
  anomalies* section carries the full list, these are the ones the port touched code adjacent to):
  - **A redundant double `reserve_back` in the 4d reader.** `reader_..._4d.cpp:42` reserves
    `weight_num_tile` entries, then `:45` calls `read_line`, whose `do_reserve` parameter defaults to
    `true` and reserves the same count again (`ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp:745`,
    `:749-751`). Harmless — the second wait is satisfied immediately, nothing having been pushed in
    between — but it is dead work on every core, and it reads as a misunderstanding of `read_line`'s
    default. `step1`'s small reader calls `read_line` with no preceding `reserve_back`, which is the
    correct usage, so the two sibling ops disagree. Carried across verbatim.
  - **A hardcoded `target_element_size` in the 3d reader.** `reader_..._3d.cpp:21` sets
    `uint32_t target_element_size = 4;  // sizeof(int32)` rather than deriving it or receiving it.
    Correct today, since `validate_inputs` requires the target tensor be `INT32`, but it silently
    couples the kernel to that assertion. Carried across verbatim.
  - **An unused include in the 4d writer.** `writer_..._4d.cpp:6` includes
    `ttnn/kernel/dataflow/moreh_common.hpp` and uses nothing from it. Kept — removing it is unrelated
    to the Metal 2.0 transformation, and the 3d writer's identical include *is* used
    (`get_noc_offset`).
  - **`c_27`'s stale factory comment.** The legacy factory labelled the buffer `// tmp2` while the
    compute kernel calls it `cb_divisor_recip` and stores `1/divisor`. The port did not propagate the
    stale label into the new name — the DFB is `divisor_recip`, after the kernel's actual use — but it
    also did not go and correct anything else about the naming.
  - **`reduction` participates in the program hash without affecting the program.** The factory takes
    `reduction` and every builder marks it unused; the program never varies on it, yet it feeds the
    default reflection hash, so two invocations differing only in `reduction` miss the cache and
    compile a byte-identical program. Untouched — the cache key is not the port's to change.
  - **The `/1024` element-size derivation in the shared donor.**
    `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp:709` computes an element size as
    `tile_size / 1024`, which is wrong for block-float formats. Safe in this op only because
    `validate_inputs` pins input and weight to `BFLOAT16`. It lives in a shared header, so it is one
    fix serving both `step1` and `step2` — out of scope for either port.

- **Three `TT_FATAL`s legitimately lost — subject deleted, and the condition is still guarded.** The
  pre-port factory carried six guards; the ported factory carries three. The three that went are
  `TT_FATAL(false, "Core not in specified core ranges.")` at pre-port `:242`, `:455`, `:685` — each the
  `else` arm of the `if/else` that populated the dead per-core compute RTA. That RTA is deleted, so the
  branch it guarded no longer exists; and the identical condition is still checked, a few lines earlier
  in the same loop iteration, by the surviving `TT_THROW("Core not in specified core ranges")` (now
  `:402`, `:659`, `:938`). No guard was weakened and none was moved. Verified in both directions by the
  per-file census: `moreh_nll_loss_step2_device_operation.cpp` holds at 16 and is byte-identical, so
  nothing leaked into the off-limits device-operation class.

- **Test coverage notes.** The confirmed set covers all twelve configs, but unevenly, and the shape is
  worth knowing before someone trims it:
  - `test_moreh_nll_loss_compute_kernel_options` is the only test that reaches **all three** rank paths
    with **both** `none_weight` values **and** both reductions **and** an `fp32_dest_acc_en` sweep. It
    is therefore the single test carrying the `unpack_modes` requirement described above — without it,
    the five required `Float32` entries would be exercised by nothing. Worth knowing if that test is
    ever trimmed for runtime.
  - The rank-4 path is reached in `test_moreh_nll_loss` only via a **rank-6** shape
    (`[5, 50, 2, 7, 50, 70]`), since `impl_4d` serves `rank >= 4`. There is no rank-4 shape in that
    test's own parametrisation; rank 4 proper comes from the other two tests.
  - 44 of the 97 collected cases skip themselves (`bfloat8_b`, "Support for bfloat8_b is currently
    unavailable"), identically before and after the port. The op's `validate_inputs` hard-asserts
    `BFLOAT16`, so those skips are consistent with the op rather than with a gap in the port — but it
    does mean roughly half the parametrised surface is inert.
  - No C++ gtest coverage exists for this op (`grep -rl moreh_nll_loss tests/ --include=*.cpp` is
    empty), so the pytest layer is the whole safety net.
