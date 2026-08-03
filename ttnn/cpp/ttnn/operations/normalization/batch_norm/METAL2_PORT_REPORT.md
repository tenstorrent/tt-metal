# Metal 2.0 Port Report — `normalization/batch_norm`

*Opened at the start of the port; entries captured as they happened and polished at the end.*

## Outcome

**`PORTED`** — both device-operations in the directory, complete. `BatchNormOperation::BatchNormFactory`
and `RunningStatistics::RunningStatisticsProgramFactory` both converted from
`ProgramDescriptorFactoryConcept` (`create_descriptor`) to `ProgramSpecFactoryConcept`
(`create_program_artifacts`), together with all **8** kernel entry points each factory can bind
(reader, writer, and **both** runtime-selected compute sources). Nothing is left for a later pass.

Build green; the confirmed no-regression baseline passes with an identical outcome count before and
after:

| | pre-port (`a38e7b405db`, unmodified) | post-port |
|---|---|---|
| `test_batch_norm.py` + `test_batch_norm_program_cache.py` | **1588 passed, 786 xfailed**, 0 failed | **1588 passed, 786 xfailed**, 0 failed |

`test_batch_norm_program_cache.py` — the highest-value test for this port, since it pins program-cache
keying *and* the running-statistics in-place side effect across cache hits, which is exactly the surface
Metal 2.0's `UpdateTensorArgs` cache-hit path touches — was run first on its own: **5 passed**.

**The baseline does not reach the typecast configurations, so they were verified separately.** Both
baseline files parametrize a *single* dtype across every tensor, which makes `interm_data_format` equal
to the output / running-stat format — so `needs_output_typecast` and `stat_format_needs_typecast` are
**always false** throughout the 2374 outcomes above. That leaves the port's most novel work uncovered:
the three conditional DFBs (`writer_out`, `writer_updated_mean`, `writer_updated_var`), all three
`#ifdef` defines, the `#ifdef`-gated handle aliases, and the typecast-config self-loop dispositions on
`out` / `updated_mean` / `updated_var`. Those paths *are* reachable — `batch_norm.cpp:60-90` requires the
**parameter** tensors to share one dtype but leaves the **input** dtype independent — so I drove them
directly with an ad-hoc scratchpad probe (not added to the diff; the port adds no tests):

| configuration | reaches |
|---|---|
| input `bfloat16` + `float32` params, eval mode, weight+bias / neither / weight-only | BatchNorm `needs_output_typecast` |
| input `float32` + `bfloat16` running stats, training, both stats | RS `needs_mean_typecast` **and** `needs_var_typecast` |
| same, `running_var` absent | mean typecasts, var does not — the independently-keyed case |
| same, `running_mean` absent | var typecasts, mean does not |

All seven checks agree with a `torch.nn.functional.batch_norm` reference. More usefully, the probe was
then run against the **pre-port** sources (`git checkout HEAD~1 -- device/`, rebuild) and the outputs are
**identical to six significant figures** — `max_abs_err` 0.00195163 / 0.00194758 / 0.000970334 (BatchNorm)
and 0.0130129 / 0.00834846 (RS), the same values before and after. So the typecast paths are a genuine
no-regression result, not merely "correct against torch". See
[Open items](#open-items-for-downstream) for the recommendation that this gap get a real test.

## Provenance

- **Recipe docs (this port):** `a38e7b405db 2026-08-03 docs(metal_2.0): require an explicit opt_level when porting compute kernels`
- **Audit docs (inherited):** `a38e7b405db 2026-08-03 docs(metal_2.0): require an explicit opt_level when porting compute kernels`

## TTNN ProgramFactory

### Concept realized

**`ProgramSpecFactoryConcept`** for both factories, as the audit chose. No re-decision, nothing surfaced
to the invoker on this axis.

Each factory returns a single `ProgramArtifacts` with `op_owned_tensors` left defaulted (neither
allocates a device tensor beyond its io), one `ProgramSpec` and one `ProgramRunArgs`. The runtime
compute-source selection is a `KernelSpec::source` choice inside one spec, **not** a multi-variant
factory branch — so the [Multi-variant factories](#) pattern does not apply here.

### Device-op-class edits

- **Custom `compute_program_hash` deleted:** none — neither device-op defined one (`grep`: zero hits in
  the directory). Already the default reflection-based hash.
- **Pybind entry points removed:** none. `batch_norm_nanobind.cpp` binds only the user-facing
  `ttnn::batch_norm` — no `nb::class_` of a device op and no `create_descriptor` binding — so sanctioned
  exceptions 2 and 3 never fired. **No user-visible API surface change.**
- **The only device-op-class edit** was each factory's declared signature plus its header include:
  - `batch_norm_device_operation.hpp:38-42` and `running_statistics_device_operation.hpp:35-39`:
    `static tt::tt_metal::ProgramDescriptor create_descriptor(...)` →
    `static ttnn::device_operation::ProgramArtifacts create_program_artifacts(...)`
  - both headers: `#include <tt-metalium/program_descriptors.hpp>` → `#include "ttnn/metal_v2_artifacts.hpp"`
- **`BatchNormOperation::operation_attributes_t::to_hash()` was left untouched**, per the brief's
  explicit warning. It is a *backdoor* custom hash, a different mechanism from `compute_program_hash`,
  and device-op-class code — not one of the three sanctioned exceptions. Verified still present and
  unmodified at `batch_norm_device_operation.cpp:121-123`.

### Open items

- **Relaxation candidates: none.** Confirmed independently rather than inherited: zero
  `ArgConfig::Runtime*` uses in the directory and no `TensorAccessor` third argument. All shape
  information travels as ordinary scalar RTAs (`HtWt`, `n_stride`, `c_stride`, `N`, `C`) and the tensors
  are tile-layout interleaved, so the accessor configuration does not vary with shape and a shape change
  *should* invalidate the cache entry. **Strict matching kept** on all 11 `TensorParameter`s.
- **No capability gap.** Neither factory wanted anything the concept lacks: no op-owned
  `GlobalSemaphore`s, no genuine per-coordinate program variation, no `override_runtime_arguments`.
- Entry-point wiring was frictionless: `ttnn::device_operation::ProgramArtifacts` +
  `MeshTensor`-by-reference `TensorArgument`s worked first try, and the framework adapter needed nothing
  from the op.

## Handoff points

No capitulation, no boundary-rule violation, no kernel-lib gap that blocked anything, and no removed
pybind surface. One informational escalation:

- **Kernel-lib, informational — `fill_cb_with_value` now has a Metal 2.0 caller.**
  `ttnn/cpp/ttnn/kernel/dataflow/cb_fill_helpers.hpp` takes a raw `uint32_t cb_id` and internally
  constructs a **legacy `CircularBuffer`** from it (the header `#include`s
  `api/dataflow/circular_buffer.h`). After this port, `reader_running_statistics.cpp:56` reaches it with
  a **DFB** handle (`fill_cb_with_value(dfb::one, one_u)`, via the constexpr
  `DFBBindingToken → uint32_t` conversion). On Gen1 a DFB lowers to a plain circular buffer, so this is
  byte-for-byte correct and needed no change — the port did **not** touch the helper (out of scope,
  kernel-side whitelist rule 9). Recorded so the kernel-lib owners know the helper is now on a Metal 2.0
  call path: the eventual Quasar uplift of any caller will need it converted.
  *Tagged: kernel-lib / Quasar-uplift dependency. Not a Gen1 blocker.*

## Successes

- **[Compiler options — the `opt_level` rule](#) fired exactly as designed, and would otherwise have
  been a silent 4-kernel perf loss.** `grep -n opt_level` over the op directory returns **zero hits**,
  which reads as "nothing to carry over." The recipe's clause that a genuinely absent
  `KernelDescriptor::opt_level` still resolves to **`O3`** on a `ComputeConfigDescriptor` — while Metal
  2.0's `CompilerOptions` defaults to `O2` for every kernel kind — is the only thing that flagged it.
  Applied explicitly on both compute specs (`batch_norm_program_factory.cpp`,
  `running_statistics_program_factory.cpp`, `.compiler_options = {… .opt_level = KernelBuildOptLevel::O3}`).
  Nothing in the build or the 2374-outcome test run would have caught the omission.

- **[Two-toucher DFB → assign 1P+1C](#) — the "re-derive, don't transcribe" instruction paid off on the
  role-free cases.** Re-running the census per `(DFB, config)` produced agreement with the brief on all
  24 DFBs, but it was the census that made the *reasoning* safe for the four DFBs whose touchers do
  nothing at all in some configs: BatchNorm `weight`/`bias` when their tensor is absent, and RS
  `updated_mean`/`updated_var` when the corresponding stat is absent (both kernels' bodies sit inside a
  skipped `if constexpr`). Those look like dead CBs on a first read — which would have meant *building
  no spec* per the dead-CB rule, dropping an allocation legacy makes unconditionally and changing the L1
  footprint. The pattern's "a role-free toucher takes whichever side is open (cosmetic)" line is what
  distinguishes them; they are 1P+1C, not dead. `allow_instance_multi_binding` is set nowhere.

- **The brief's conditional-DFB table was right to list only the *compute* alias sites — and the
  near-miss was mine.** The two writers read the same host-selected buffer index through their own CTAs
  (`writer_batch_norm.cpp` CTA `[3]`; `writer_running_statistics.cpp` CTAs `[6]`/`[7]`), so I first
  concluded they needed the `#ifdef` alias too, and wrote it into the plan before catching it. They do
  not: each writer touches its path-dependent buffer through exactly **one** handle and binds none of
  the candidates otherwise, so a single `DFBBinding` with a fixed `accessor_name` and a host-chosen
  `dfb_spec_name` emits the same `dfb::dst` (resp. `dfb::new_mean` / `dfb::new_var`) token on both paths
  — no define, no `#ifdef`, no second binding. The compute kernels genuinely need the alias precisely
  because they *already* bind `out` / `updated_mean` / `updated_var` under another accessor name, which
  makes a second binding for the same DFB illegal. Corrected in `METAL2_PORT_PLAN.md`
  (Deferred / Flagged item 1). See the matching Gap below — the rule that resolves this is not stated
  anywhere.

- **[Pass DFB handles directly to LLKs and kernel-lib helpers](#) covered every escape with no bridge
  work.** All three donor headers the audit flagged crossed on the implicit conversion with the call
  sites otherwise unchanged: `fill_cb_with_value` (`uint32_t cb_id`), the `dest_format_helpers.hpp`
  trio (`pack_tile_with_dt`, `copy_tile_to_dst_init_short_with_dt`, `copy_tile_init_with_dt`), and
  `fill_tile_utils.hpp` (raw `l1_write_ptr` — nothing to bridge, kept `dfb.get_write_ptr()`). No `.id`
  extraction and no temporary `DataflowBuffer` wrapper anywhere in the diff.

- **The brief's runtime-DFB-selection heads-up prevented a wrong "improvement."** Both BatchNorm compute
  sources pick `dfb_affine_or_out` / `dfb_scaled_output` from `weight_has_value` / `bias_has_value` at
  **runtime** (`batch_norm_sfpu_kernel.cpp:42-43`), and the `has_value` guards in those kernels are
  runtime `if`s, not `if constexpr`. The pull to make them `constexpr` (and to `#ifdef` the weight/bias
  bindings for L1) is strong and would have been wrong on both counts. Left exactly as-is; weight and
  bias are bound unconditionally on compute, which is what the kernels require.

## Friction

### Gaps

- **The kernel-side docs name a type that does not exist: `DFBAccessor`.** The recipe
  ([kernel-side whitelist rule 2](#), the boundary note in §Read this first), the patterns catalog
  ([Pass DFB handles directly](#), [`.id` extraction anti-pattern](#)) and the migration guide all
  describe the bridge as `DFBAccessor::operator uint32_t()`. The generated type is **`DFBBindingToken`**
  (`tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:46-59`, emitted at
  `tt_metal/jit_build/genfiles.cpp:196`); `grep -rn DFBAccessor tt_metal/` returns **nothing**. Since the
  whole port leans on that conversion, a symbol that greps to zero hits is an unsettling first check —
  it reads as "the affordance was renamed away" rather than "the doc is stale." Resolved by reading the
  declaring header, which the recipe's "go to the headers first" advice points at, so the cost was
  small. Right answer: `constexpr operator uint32_t()` on `DFBBindingToken`, exactly as described.
  **Suggested fix:** s/`DFBAccessor`/`DFBBindingToken`/ across the three docs (`dfb::name` as a
  *spelling* is correct everywhere).

- **The conditional-binding pattern's host-side example is not valid code, and its shape misleads.**
  [Pattern: Conditional / optional DFB bindings](#) shows
  `.dfb_bindings = fuse_pre_add ? Group<DFBBinding>{INPUT, OUTPUT, FUSION} : Group<DFBBinding>{INPUT, OUTPUT}`.
  `INPUT` / `OUTPUT` / `FUSION` are `DFBSpecName`s, and a `DFBBinding` additionally needs
  `accessor_name` and `endpoint_type` — so the braced list does not compile, and more importantly it
  suggests bindings can be enumerated by DFB name with the roles inferred. They cannot. What this port
  used, on all six `KernelSpec`s with conditional content: declare `Group<DFBBinding>` as a named local
  holding the unconditional full-form bindings, then `push_back` the conditional ones alongside the
  matching `defines` entry, and `std::move` it into the spec. The same shape applies to
  `Group<TensorBinding>` for optional tensors. **Suggested fix:** replace the ternary example with the
  declare-then-`push_back` form; it is what the recipe's own "prefer designated initializers" rule
  forces anyway, since the full `DFBBinding{…}` form cannot be written inline in a ternary without
  repeating every field.

- **No rule for a *dead RTA slot*, although there is one for a dead CB.** The recipe's Construct step is
  explicit about a dead CB (build no spec, drop the allocation and any dead CTA carrying its index,
  record each with `file:line`), and about dead per-source CTAs falling out of the per-source
  `KernelSpec`. It says nothing about an RTA value the factory emits that **no kernel reads**. This op
  has **8** of them: `cHt` and `cWt` in the last two slots of all four dataflow kernels
  (`batch_norm_program_factory.cpp:98-99`, `:123-124`; `running_statistics_program_factory.cpp:96-97`,
  `:120-121`), plus `freq` and `counter` on the RS compute kernel (`:126` emits three, both RS compute
  sources read only slot 0). Dropping them — declaring no named arg — is obviously right and obviously
  zero-functional-change, and the brief's arity table implies it arithmetically ("reader 9 → 8 named"),
  but I had to derive the licence rather than read it. **Suggested fix:** one line under Dropped
  Plumbing, parallel to the dead-CB clause: *an RTA slot no kernel reads is dropped, not named; record
  each with `file:line`.* All 8 are recorded in `METAL2_PORT_PLAN.md` → Flags item 1.

- **The `accessor_name`-absorbs-the-path-choice rule is not stated anywhere**, and it is the thing that
  decides whether a path-dependent DFB needs the `#ifdef` scaffolding at all. The two rules interact:
  *(a)* a kernel binds a given DFB under exactly one accessor name, so a second name for the same DFB is
  rejected; therefore *(b)* if a kernel reaches a path-dependent buffer through **one** handle and does
  **not** otherwise bind any candidate, choose the `dfb_spec_name` host-side under a **fixed**
  `accessor_name` and the kernel needs no gate — but if it *does* already bind a candidate under another
  name, the `#ifdef`-gated handle alias is the only option. This port needed (b)-no-gate on both writers
  and (b)-with-gate on both SFPU compute sources, for the *same* host value. Getting it backwards costs
  either a redundant define plus dead `#ifdef` branches (harmless but misleading) or a validator
  rejection. **Suggested fix:** add the two-way test to
  [Pattern: Same-FIFO aliasing → path-dependent variant](#), which is where a porter looks.

- **The `unpack_modes` validator tolerates producer-side entries, and no doc says so.** The migration
  guide's troubleshooting table and the recipe both cover the *consumer* side (an entry is **required**
  for a consumed Float32 DFB under `enable_32_bit_dest`) and the rejection cases (unbound DFB; 32-bit
  format into a 16-bit Dest; ≤16-bit format on Gen1). Neither mentions what happens to an entry for a
  DFB the compute kernel only **produces**. It is explicitly accepted as inert
  (`tt_metal/impl/metal2_host_api/program_spec.cpp:1005-1007`, *"Compute kernel is bound as the DFB
  Producer: inert, tolerated"*). That mattered here: the legacy RS list includes `output` (`c_2`), which
  the compute kernel only ever packs into, and the faithful re-key keeps it. Without the header read I
  would have had to choose between dropping a key (a silent deviation from legacy) and risking a
  validator failure. **Suggested fix:** one clause in the `unpack_modes` subsection — *an entry for a
  producer-only DFB is tolerated as inert, so a legacy list may be re-keyed wholesale without pruning
  producer-side entries.*

- **Verification has no step for "does the confirmed test set actually reach each conditional binding?"
   — and for a config-gated `#ifdef` port it very plausibly does not.** The recipe's standard is *all
  tests passing pre-conversion should continue to pass post-conversion*, which is necessary but, for this
  shape of port, quietly insufficient: a conditionally-bound DFB only compiles on the configurations that
  bind it, so a fully green baseline can leave the entire `#ifdef` mechanism — the define emission, the
  gated alias, the conditional `DFBBinding`, and the flipped endpoint disposition — **never built**. That
  is exactly what happened here: all 2374 outcomes are green with `needs_output_typecast` and
  `stat_format_needs_typecast` pinned to `false`, because both baseline files parametrize one dtype
  across every tensor. A mistake in that machinery would not have produced a test failure; it would have
  produced a JIT `static_assert` or a name-lookup error the first time a user passed a mixed-dtype
  configuration. I only found this by asking, after the suite was green, *which* configuration selects
  each conditional DFB — and then had to derive the reachable mixes from the op-level dtype validation
  (`batch_norm.cpp:60-90`: parameters share a dtype, input is independent) and drive them with a
  throwaway script. **Suggested fix:** add a bullet to [Run tests](#) — *for each conditional binding
  and each `defines` entry the port introduced, identify the configuration that selects it and confirm
  the confirmed test set contains a case that hits it; where it does not, drive that configuration by
  hand (ad-hoc script, not a checked-in test) and record the gap under Open items.* Cheap to do, and it
  is the difference between "the tests pass" and "the code I wrote was compiled."

- **`workspace_setup.md` should say: build before you trust an inherited `build/`.** The checkout I was
  handed had a complete-looking `build/lib/_ttnn.so` dated the same day, but importing `ttnn` failed with
  `ImportError: cannot import name 'FabricType' from 'ttnn._ttnn.fabric'` — a stale/partial incremental
  build that presents as a broken environment, not a stale one. The recipe's environment precondition
  covers the venv (`create_venv.sh` / `source python_env/bin/activate`) but assumes the build is
  trustworthy. One `./build_metal.sh --build-tests` fixed it. **Suggested fix:** add to the
  environment-precondition callout — *establish the pre-port baseline by building first; do not infer
  from the presence of `build/` that the Python bindings match the checkout.*

- **`tt-smi` can report the card as dead while `ttnn` drives it fine — don't gate on it.**
  `tt-smi -ls` printed *"Error in detecting devices! Chip initialization failed: … DRAM Status: Timeout
  … CPU Status: Timeout … ARC Status: Timeout"* on this host, both before and after the port, while
  2374 test outcomes ran green through `ttnn` in the same session. Read literally that output says the
  port cannot be verified on hardware, which is a plausible reason to stop and report — and would have
  been wrong. **Suggested fix:** a line in `workspace_setup.md` — *`tt-smi` detection failure is not a
  verification blocker on its own; confirm with an actual test run before treating the device as
  unavailable.*

### Confusion

- **"Both factories are Style A" needed one extra step to act on.** The brief prescribes
  `ttnn::to_compute_hardware_config(device->arch(), operation_attributes.compute_kernel_config)`, which
  is right — but the factory *also* needs the raw `fp32_dest_acc_en` bool, twice: it selects the compute
  source and it gates the whole `unpack_modes` list. Legacy got it from the five-way
  `get_compute_kernel_config_args` destructure, which after the port leaves four of the five names
  unused. The port uses `ttnn::get_fp32_dest_acc_en(operation_attributes.compute_kernel_config)`
  instead, having checked it reads the identical field
  (`compute_kernel_config.cpp:70-75` vs `:99-107`). Minor, but the Style A instruction reads as "replace
  the destructure with the helper," and for any op whose factory *branches* on a config knob that is
  half the story. **Suggested fix:** note that a knob the factory reads for its own branching still
  needs a direct read; the helper only fills `hw_config`.

- **The unity-build catalog entry prescribes a remedy this codebase has already solved structurally.**
  [Pattern: Unity-build hygiene](#) says to prefix per-factory constants (`W_READER_KERNEL`,
  `H_READER_KERNEL`, …). `ttnn_op_normalization` *is* a unity-build target and both factories declare
  same-named constants — but `cmake/unity.cmake:9` sets
  `UNITY_BUILD_UNIQUE_ID "CMAKE_UNIQUE_NAMESPACE"`, and both files already wrapped their file-local code
  in `namespace { namespace CMAKE_UNIQUE_NAMESPACE { … } }`. Declaring the spec names inside that
  existing wrapper (plus `using namespace CMAKE_UNIQUE_NAMESPACE;` in the factory body) means `READER`,
  `INPUT_DFB`, `make_dfb` and friends can be spelled identically in both files with no collision and no
  prefix noise. I drafted the prefixed form first before noticing the wrapper was already there.
  **Suggested fix:** lead the entry with "check for an existing `CMAKE_UNIQUE_NAMESPACE` wrapper — most
  TTNN op factories in unity-build targets already have one"; keep prefixing as the fallback for
  constants that must be visible outside it.

## Open items for downstream

- **Shared kernel touches: none.** All eight kernel sources live in this op's directory and are bound
  only by this directory's two factories (verified per-file with
  `grep -rl <filename> ttnn/cpp/ttnn/operations/` — every hit inside `normalization/batch_norm/`). No
  `_metal2` fork was reused or created, no peer-op directory was written to, and no legacy kernel was
  converted in place for another consumer. Nothing for a future sibling-op porter to coordinate with,
  and no sunset checklist.

- **Idle nodes divide by zero in both readers — pre-existing, preserved verbatim.** Both factories place
  kernels on **every** device core and hand the nodes outside both work groups an all-zero argument set.
  The readers then compute `tiles_per_batch = HtWt * C` (= 0) and `start_tile_id / tiles_per_batch`
  unconditionally, before any `num_tiles == 0` check
  (`reader_batch_norm.cpp:36-39`, `reader_running_statistics.cpp:41-45`; the writers do the same at
  `writer_batch_norm.cpp:65-68`, `writer_running_statistics.cpp:57-61`). The result is unused — the
  subsequent loops all test `num_tiles_read < num_tiles` and do not execute — so it is harmless in
  practice, and it behaved identically before the port. Flagged because it is invisible in the legacy
  code (the zero-fill and the division are in different files) and because both plausible cleanups are
  **behavior changes** the port must not make: narrowing the work unit to the working cores changes
  kernel placement, and adding an early `num_tiles == 0` return to the dataflow kernels changes their
  control flow. For the op owners to decide.

- **RTA → CRTA is *not* available here, despite appearances.** `HtWt`, `n_stride`, `c_stride`, `N`, `C`
  hold the same value on every *working* node, so they look like textbook common-runtime-arg candidates
  (the recipe flags this as a later-pass tidy-up). They are not, as emitted: the idle nodes get `0` for
  them, so the values are **not** node-invariant across the kernels' actual node set. Promoting them to
  CRTAs would require dropping the zero-padding first, which is the placement change above. The two are
  coupled — worth knowing before someone picks up the CRTA cleanup in isolation.

- **Pre-existing anomalies confirmed still present, deliberately not touched** (the audit flagged all
  four; the port reproduces each exactly):
  - `packer_l1_acc` is resolved and dropped on the floor (`batch_norm_program_factory.cpp`,
    `running_statistics_program_factory.cpp` — now via `to_compute_hardware_config`, which likewise has
    nowhere to put it). `ComputeGen1Config` has no equivalent field, so Metal 2.0 cannot restore it
    either. Neither port work nor a port regression.
  - `b_num_tiles_per_cb` is still a redundant alias of `num_tiles_per_cb`, kept so the DFB `num_entries`
    expressions read the same as the legacy `total_size` ones.
  - `eps` / `momentum` / `one` still hold one tile with `num_entries = 2`. An L1 saving is available;
    changing it is a footprint change.
  - The `to_hash` inertness on `BatchNormOperation` still depends on `input_dtype` never diverging from
    `tensor_args.input.dtype()` — see `METAL2_PREPORT_AUDIT.md` → *The `to_hash` backdoor*. Untouched.

- **Test coverage notes.** The confirmed baseline is the two `tests/ttnn/unit_tests/operations/fused/`
  pytests (invoker-confirmed; the coverage is under the `fused/` slug, not `normalization/`). Two gaps
  the verification step surfaced but did not act on:
  - `tests/sweep_framework/sweeps/normalization/batch_norm/batch_norm.py` could **not** be run — sweep
    framework modules are currently unimportable (the `tests/` packaging breaks `tests.ttnn.*` imports).
    So the sweep contributed nothing to this port's no-regression evidence.
  - There are **no C++ gtests** for either device-op (`grep -rli batch_norm tests/**/*.cpp`: zero hits)
    and no nightly variants. All verification is Python-level.
  - **The biggest coverage gap, and the one worth acting on: no test reaches a typecast configuration.**
    Both baseline files parametrize one dtype across all tensors, and a uniform dtype forces
    `needs_output_typecast` / `stat_format_needs_typecast` to **false** — so the three conditional DFBs,
    all three `#ifdef` defines, and the typecast-config self-loop dispositions have **zero automated
    coverage**, before this port and after it. This port verified them with an ad-hoc probe (details in
    [Outcome](#outcome); identical output pre- and post-port), but that probe is not checked in.
    **Recommendation:** add mixed-dtype rows to `test_batch_norm.py` — `bfloat16` input with `float32`
    parameters (hits BatchNorm's typecast) and `float32` input with `bfloat16` running stats (hits both
    RunningStatistics typecasts, and with one stat omitted, the independently-keyed case). Both are
    legal per `batch_norm.cpp:60-90`, which constrains only the *parameter* tensors to a shared dtype.
    This is a **pre-existing** coverage gap, not one the port introduced — but the port converts that
    machinery to `#ifdef`-gated conditional bindings, where a mistake fails at JIT rather than
    numerically, so the gap now hides a sharper edge than it used to.
  - Not a coverage gap but worth recording for whoever reads the numbers: **786 of the 2374 outcomes are
    `xfail`**. They come from one functional precondition — `test_batch_norm.py:77`,
    *"running_mean and running_var must be defined in evaluation mode"* — not from a dtype limitation, so
    the `float32` rows do execute. They xfail identically before and after, so they constrain the port no
    further than the 1588 passes do.

- **Per-op carry-over.** The `weight`/`bias`-present-but-DFB-bound-anyway shape here (an optional tensor
  whose DFB is allocated unconditionally by legacy, with the kernel referencing the handle outside its
  `if constexpr` guard) is likely common across the `normalization` and `eltwise` families. The
  resolution this port used — **tensor** binding conditional + `#ifdef`, **DFB** binding unconditional,
  the two decoupled — is worth reaching for directly rather than re-deriving: the two conditionalities
  are independent, and conflating them either wastes the `#ifdef` or breaks the unconditional handle
  reference.
