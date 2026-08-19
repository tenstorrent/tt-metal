# Metal 2.0 Port Report — `moreh/moreh_linear_backward` (`MorehBiasAddBackwardOperation`)

## Outcome

**`PORTED`** — **both** factories converted in this change (`SingleCoreProgramFactory` and
`MultiCoreProgramFactory`), together with all five kernels they bind. Nothing left for a later pass:
the device operation has no other factory.

The confirmed test set passes **identically before and after** the port:

| run | result |
|---|---|
| pre-port baseline (binaries built from the unmodified op) | `173 passed, 168 skipped, 66 deselected` |
| post-port | `173 passed, 168 skipped, 66 deselected` |

(The 168 skips are the `bfloat8_b` parameterizations, which the test file skips explicitly for
backward; the 66 deselected are the forward-only tests excluded by `-k backward`.)

Metal 2.0 legality checks were **forced on and proven live** for both runs — `METAL2_CHECKS_FORCED`
appears from both translation units (`program_spec.cpp:2847` in `BuildProgramFromSpec` and
`program_run_args.cpp:502` in `SetProgramRunArgs`), 958 times in the post-port run. All nine
`skip_validation` sites were forced; none of that scaffolding is in the commit.

## Provenance

```
git log -1 --format='%h %cs %s' -- docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/
```

- **Recipe docs (this port):** `9440205cf62 2026-08-19 docs(metal_2.0): have the porter prove the legality checks are running`
- **Audit docs (inherited):** `9440205cf62 2026-08-19 docs(metal_2.0): have the porter prove the legality checks are running`

The working-tree docs were verified byte-identical to `origin/akertesz/op-porting-recipe`
(`17fbf9bebe5`) before starting, so the recipe followed is the current upstream one.

## TTNN ProgramFactory

### Concept realized

`ProgramSpecFactoryConcept` (base concept) on **both** factories — exactly the audit's choice, with
no disagreement to surface. Each factory implements one `create_program_artifacts` returning
`ProgramArtifacts{spec, run_params}`; `op_owned_tensors` is left defaulted (the op allocates no
device tensors of its own). The op has no `override_runtime_arguments`, so the framework refreshes
the tensor bindings on a cache hit and the port writes no override —
`test_moreh_linear_backward_enable_cache` exercises that path and passes, with the cache-entry count
stable across two dispatches as before.

### Device-op-class edits

- **Pybind entry points removed:** **none.** `moreh_linear_backward_nanobind.cpp` binds only the
  user-facing `ttnn::moreh_linear_backward`; there was no pybound `create_descriptor`, so
  exception 1 never fired. The nanobind files are untouched.
- **Custom `compute_program_hash`:** none — the op uses the default reflection-based hash and has no
  backdoor `attribute_values` / `to_hash`. Nothing to leave alone, nothing touched.
- The only device-operation-**class** change is in the header
  (`device/moreh_linear_backward_device_operation.hpp`): the two factory method signatures
  (`ProgramDescriptor create_descriptor` → `ttnn::device_operation::ProgramArtifacts
  create_program_artifacts`) and swapping `#include <tt-metalium/program_descriptors.hpp>` for
  `#include "ttnn/metal_v2_artifacts.hpp"`. The device-operation `.cpp`
  (`validate_inputs`, `select_program_factory`, `compute_output_specs`, `create_output_tensors`, the
  `ttnn::prim` entry point) is **byte-identical** — confirmed by the TT_FATAL census below.

### Open items

- **Tensor-arg matching kept strict** on both `TensorParameter`s, as the audit recorded
  (`relaxations` left default). No `ArgConfig::Runtime*` use anywhere in the op's kernels, so no
  known-required relaxation applied. No relaxation candidate spotted worth flagging: both kernels
  index by `page_id` through a `TensorAccessor` whose layout the binding pins, and neither reads a
  shape.
- No capability gap. The op needed no op-owned tensors, no op-owned `GlobalSemaphore`s, and no
  per-coord program variation, so the base concept fit with room to spare.

## Handoff points

**None.** No capitulation, no boundary-rule assumption violation, no kernel-lib gap, no framework
gap, no removed pybind surface. Recorded explicitly because each was checked:

- **`sem::` / `tensor::` never cross a donor boundary.** The op declares no semaphores at all
  (`grep -i semaphore` over the op directory: zero hits), and both `TensorAccessor`s are constructed
  and consumed inside the op's own kernels. The recipe's boundary assumption holds.
- **Every donor call site crossed on `dfb::name`'s `uint32_t` conversion, with no callee change.**
  `fill_cb_with_value` and `generate_mask_h_w`
  (`ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp:98,262`) already take a `DataflowBuffer` by
  value; `dataflow_kernel_lib::calculate_and_prepare_reduce_scaler`
  (`ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp:83`) and
  `compute_kernel_lib::reduce` (`reduce_helpers_compute.hpp:392`) take DFB ids as `uint32_t`
  non-type template parameters, which `dfb::name` satisfies because the conversion is `constexpr`;
  `compute_kernel_lib::Accumulate::at` (`:193`) and the LLKs take a runtime `uint32_t`. Nothing
  outside the op directory was edited.
- **No Case 2 binding.** Both tensor bindings are Case 1 — the base flows only into a
  `TensorAccessor`, accessed by `{.page_id = …}` — so `get_bank_base_address` is never reached and
  the compute-kernel Case-2 block never arose.
- **No shared-kernel fork needed.** `device/kernels/writer_moreh_bias_backward.cpp` is bound by both
  factories (*intra-op* sharing), and both convert in this change, so the file was converted in
  place at **rung 0**: no `_metal2` fork, no pointer comment, no consumer left behind. See Open
  items for the census that establishes there is no external consumer.

## Successes

- **[CB→DFB whitelist §A — `constexpr` metadata values](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/cb_dfb_api_whitelist.md#tile--format-metadata-jit-descriptors)
  fired correctly, and the brief pre-checked it.** All three `get_tile_size(cb_id)` sites
  (`device/kernels/reader_moreh_bias_backward_hw.cpp:41`, `reader_moreh_bias_backward_h.cpp:34`,
  `writer_moreh_bias_backward.cpp:21`) were declared `const auto`, not `constexpr`, so the member
  getter `dfb.get_tile_size()` is the correct form and the `get_tile_size(dfb::name)` token form is
  used nowhere. Because the whitelist makes the *legacy declaration* the entire test, this needed no
  judgement — and it means the port carries **no** Gen1-only token-form debt to record. The
  whitelist's warning against demoting a `constexpr` to `const` to make a getter fit never had to be
  resisted, precisely because the rule is stated as a lookup rather than a preference.
- **[Pattern: Conditional / optional DFB bindings](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-conditional--optional-dfb-bindings)
  described this op's single-core mask buffer exactly, including the trap.** The pattern's
  "*`if constexpr` still performs name lookup on the discarded branch*" rationale is what makes the
  `#ifdef` non-optional here, and its instruction to gate *every* reference — not just the
  construction — caught the two nested `copy_tile` sub-blocks at
  `device/kernels/moreh_bias_backward_single_core_hw.cpp:63-83`, which sit inside an
  already-runtime-gated `if (do_mask)` and so look unreachable rather than un-nameable. The
  pattern's "don't bind unconditionally as an alternative" paragraph pre-empted the obvious
  shortcut, which would also have changed the op's L1 footprint.
- **[Anti-pattern: Demoting per-group CTA to RTA](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#anti-pattern-demoting-per-group-cta-to-rta)
  plus the brief's warning stopped a very tempting simplification.** The multi-core factory's two
  compute descriptors differ *only* in a compile-time arg the kernel never reads
  (`device/moreh_linear_backward_multi_core_program_factory.cpp:297`), so collapsing them into one
  `KernelSpec` looks free — no CTA would even have to be demoted, since the value is already an RTA
  as well. Both documents naming the multiplicity as non-negotiable, and the brief naming
  `units_per_core` as the precedent name, made this a two-minute decision instead of a judgement
  call.
- **The recipe's `Table` warning saved a compile-error detour.** `compile_time_args`, `defines` and
  `unpack_modes` are all `Table`, and all three were built conditionally in this port
  (`compute_defines.emplace(...)`, `unpack_modes.emplace(...)`); reaching for `push_back` would have
  been the reflex.
- **`AddRuntimeArgsForNode` / `MakeRuntimeArgsForSingleNode` made the run-args translation
  mechanical.** The multi-core factory's node-first core loop is preserved verbatim with the
  transposing helper, and the single-core factory's three one-core calls read as data. The recipe's
  explicit "do **not** re-architect the legacy loop into name-first form" removed the temptation to
  tidy it.

## Friction

### Gaps

- **The brief's `unpack_modes` reasoning has an incomplete premise, and the gap is the silent
  kind.** The brief states: "*`c_24` needs no entry in either factory — its format is
  `cb_data_format`, never Float32*". `cb_data_format` is
  `datatype_to_dataformat_converter(output_grad.dtype())`, so it **is** `Float32` when `output_grad`
  is a Float32 tensor — and nothing gates that dtype (see Open items). With Float32 `output_grad`
  **and** `fp32_dest_acc_en`, the validator's required-entry rule
  (`tt_metal/impl/metal2_host_api/program_spec.cpp:1049-1077`, which keys on the *DFB's*
  `data_format_metadata`, exactly as the recipe says) fires for `in0`, `scaler`, `mask_h_w` and
  `intermed0` too, not only `intermed1`. Following the brief literally would have produced a spec
  that `TT_FATAL`s in a configuration legacy ran — a behavior change, and one no test would catch
  because the confirmed test set is bfloat16-only.

  Resolved by transcribing the **whole** legacy `Default` row rather than the one element the brief
  named: `UnpackToSrc` for every compute-consumed DFB, plus `UnpackToDest` on `intermed1` in the
  multi-core factory under fp32. This is not an added value —
  `BuildUnpackToDestModeVector` (`program_spec.cpp:2711-2714`) maps `UnpackToSrc` back to
  `UnpackToDestMode::Default`, and `UnpackToSrc` is unconditionally legal
  (`program_spec.cpp:1004-1005`) — so the internal vector the JIT sees is byte-identical to
  legacy's in every configuration, and the required-entry rule is satisfied in all of them.
  Sites: `device/moreh_linear_backward_single_core_program_factory.cpp:268-279` and
  `device/moreh_linear_backward_multi_core_program_factory.cpp:221-231`.

  **Suggested doc change:** the recipe's `unpack_modes` item already says "*The trigger is the DFB's
  format, not the op's tensor dtypes*" and gives the `fp32_dest_acc_en ? Float32 : data_format`
  idiom as its example — that is exactly right and is what caught this. The gap is one level up, in
  the **audit**: an auditor reasoning "this DFB's format is `<some data_format variable>`, so never
  Float32" is making a claim about the op's reachable dtypes, which needs the dtype-gate check to
  back it. A one-line prompt in the audit's `unpack_modes` guidance — *before asserting a DFB is
  never Float32, confirm the op gates the dtype that feeds its format* — would close it. The
  general, dtype-independent move (transcribe the whole legacy row, since `UnpackToSrc` is always
  legal and always means `Default`) is also worth stating in the recipe as the safe default: it
  removes the reachability analysis from the porter's plate entirely.

- **The "prove the legality checks are running" step works, but only because of a detail the recipe
  doesn't mention.** The recipe says to put a marker in each file the grep named and expects "**two
  markers present**". Both markers log the *identical* string, so `grep -c METAL2_CHECKS_FORCED`
  cannot distinguish one fresh translation unit from two — it just counts hits. What makes the check
  work is that `tt-logger` appends the emitting source location, so the log reads
  `METAL2_CHECKS_FORCED (program_spec.cpp:2847)` and `METAL2_CHECKS_FORCED
  (program_run_args.cpp:502)`. Worth one clause in the recipe (*grep for the two distinct
  `(file:line)` suffixes, not for the marker count*), because the naive count passes with one file
  stale.

- **Three of the nine `skip_validation` grep hits are continuation lines, not signature lines.**
  `grep -n 'bool skip_validation' tt_metal/impl/metal2_host_api/*.cpp` returns
  `program_run_args.cpp:799`, `:874` and `program_spec.cpp:3251` as bare
  `… bool skip_validation) {` fragments of multi-line signatures; the function *name* is on an
  earlier line. Inserting "as the first statement" is still unambiguous (the matched line is the one
  ending the signature), but identifying *which* function you are forcing — needed to reason about
  the two named choke points the recipe calls out — requires widening the context first. A
  parenthetical (*several hits are continuation lines of multi-line signatures; widen the grep
  context to identify the function*) would save a step.

### Confusion

- **The CB-name sweep's "expect zero hits" collides with comments that deliberately document the
  legacy mapping.** The port's highest-risk lines are the `unpack_modes` tables, and the most useful
  thing a reviewer can be handed there is what the legacy code did — which naturally reads
  "*legacy set `unpack_to_dest_mode[CBIndex::c_25] = UnpackToDestFp32`*". That comment is
  information a reviewer needs (the port report does not reach `main`), but it trips both the
  `\bCB[A-Z]` sweep and the "no `CircularBuffer` / `CBDescriptor` survives" statement, whose
  wording ("*only legacy-comparison artifacts in the port report, if any*") reads as ruling such
  comments out of the code entirely.

  Resolved by rewording each to prose that keeps the mapping and drops the legacy identifiers
  ("*the intermed1 buffer (index 25)*", "*the host-side tensor-accessor argument plumbing*",
  "*the legacy two-kernel-descriptor work split*") — which is arguably better anyway, since a
  ported reader has no `CBIndex` vocabulary. But the recipe could say so directly: *explanatory
  comments about what legacy did are welcome and belong in the code; write them without legacy
  API identifiers so the sweeps stay clean.* Right now the porter has to choose between a clean
  sweep and a traceable comment, and the obvious resolution is to delete the comment.

- **One residual `cb`-name hit is a callee's name and cannot be fixed from inside the port.** After
  renaming the host local `cb_data_format` → `dfb_data_format`, the sweep
  `grep -rnE '[Cc][Bb]_|_[Cc][Bb]\b|\b[Cc][Bb]\b|\bCB[A-Z]'` over the op directory returns exactly
  one hit in code: the call `fill_cb_with_value(dfb_scaler, scaler.u)` at
  `device/kernels/reader_moreh_bias_backward_hw.cpp:25`. The helper lives in
  `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp:98` — outside the porter's writable surface, and
  already migrated to take a `DataflowBuffer`, so only its *name* is stale. The recipe's sweep
  expects zero and the surrounding text frames every hit as "a real leftover"; it would help to note
  that a **donor helper's name** at a call site is the one hit a port cannot clear, and that the
  right disposition is a report entry rather than an edit. (Entry filed under Open items.)

## Open items for downstream

### Shared kernel touches

- `ttnn/cpp/ttnn/operations/moreh/moreh_linear_backward/device/kernels/writer_moreh_bias_backward.cpp`
  — **intra-op** shared kernel, bound by both of this op's factories.
  - **Rung taken: 0 — converted in place, no fork created.** Both binding factories converted in
    this same change, so nothing is left on the legacy API. No `writer_moreh_bias_backward_metal2.cpp`
    exists or was created, and no pointer comment was added (there is nothing to point at).
  - **Remaining unmigrated consumers: none.** Census: `grep -rl <filename>
    ttnn/cpp/ttnn/operations/` for each of the op's five kernels returns only this op's own two
    factories (and this port's `METAL2_*.md` files); the wider sweep over `ttnn/ tests/ models/`
    adds only `ttnn.egg-info/SOURCES.txt`, a packaging manifest rather than a consumer. So this port
    creates **no** sunset list and no cross-op coordination cost.
  - Binding vocabulary the writer now defines, for anyone who later binds it: `dfb::out`,
    `tensor::dst`, `args::num_tiles`, `args::start_id` — taken from the kernel's own names
    (`cb_id_out`, `dst_addr`), not from either factory's locals.

### Owner-facing findings (observed, deliberately not fixed)

1. **`output_grad`'s dtype is ungated, and a Float32 `output_grad` would compute wrong masks.**
   Neither `MorehBiasAddBackwardOperation::validate_inputs`
   (`device/moreh_linear_backward_device_operation.cpp:17-26`) nor `ttnn::moreh_linear_backward`
   (`moreh_linear_backward.cpp:106-170`) restricts the dtype, and `ttnn::prim::moreh_bias_add_backward`
   is reachable directly. But `generate_mask_h_w`
   (`ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp:262`) writes the mask through a `uint16_t*` and
   defaults `single_tile_size = 2048` — a bfloat16 tile — so a Float32 `output_grad` with a
   non-tile-aligned shape gets a malformed mask. Pre-existing and untouched by the port (the tests
   are bfloat16-only, so nothing exercises it). Two candidate owner fixes: gate the dtype in
   `validate_inputs`, or make the mask generator format-aware. Flagging it because it is also the
   reachability question behind the `unpack_modes` Gap above.
2. **Dead compile-time arg on both multi-core compute specs** — `units_per_core`
   (`device/moreh_linear_backward_multi_core_program_factory.cpp:297`, from legacy
   `compile_time_args = {num_cols_per_core_group_N}` at old `:171`/`:188`). The kernel reads **no**
   compile-time argument at all (`grep -n get_compile_time_arg_val` over all five kernels: zero
   hits); the value it uses arrives as the `Wt_per_core` RTA from the same `num_cols_per_core`. The
   port preserves it verbatim as a named CTA, because it is what distinguishes the two per-group
   specs and dropping it would collapse the preserved multiplicity. Removing it — and deciding
   whether the two specs should then merge — is an owner call.
3. **`batch_num` is now a *named* argument whose name understates its value.** The multi-core factory
   passes `batch_num * Ht` (`device/moreh_linear_backward_multi_core_program_factory.cpp:44`, then
   `:346`) into locals named `batch_num` in both `reader_moreh_bias_backward_h.cpp:13` and
   `moreh_bias_backward_multi_core_h.cpp:12` — where the latter then computes
   `num_tiles = batch_num * Ht`, i.e. `batch_num * Ht * Ht` by the name's own reading. The value is
   correct; the name is not. Named after the kernel-side local as the recipe prescribes, with a
   comment at the schema site stating what the value actually is. A rename of the kernel locals (and
   the matching argument name) is an ops-team cleanup; it is strictly nicer to do now that the name
   is part of an interface rather than a positional slot.
4. **The multi-core factory reserves the mask buffer unconditionally.**
   `device/moreh_linear_backward_multi_core_program_factory.cpp:114-119` (legacy `:64,92-100`) always
   declares a 2-tile `mask_h_w` DFB on every core, even when no mask applies — unlike the
   single-core factory, which guards the same allocation. Wasted L1 in the common aligned-shape
   case. Preserved faithfully; the single-core factory's conditional-DFB shape is exactly the pattern
   to copy if an owner wants to tighten it.
5. **A latent wart the port happened to retire.** `moreh_bias_backward_single_core_hw.cpp` used to
   construct `DataflowBuffer dfb_mask_h_w_obj(cb_mask_h_w)` unconditionally (old `:21-22`) on a CB
   index that the single-core factory allocated only under masking — benign on Gen1, but a real
   mismatch. The Metal 2.0 conditional binding makes the construction impossible to get wrong: with
   no binding there is no token, so the compiler enforces the gate. No action needed; noted because
   it is a small, concrete example of the typed-binding model catching something the legacy
   `uint32_t` channel could not.
6. **Donor helper name still says `cb`.** `fill_cb_with_value`
   (`ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp:98`) already takes a `DataflowBuffer`; only its
   name is pre-Metal-2.0, and it is the single residual hit of this port's CB-name sweep (call site:
   `device/kernels/reader_moreh_bias_backward_hw.cpp:25`). Out of the porter's writable surface, so
   not renamed. A `fill_dfb_with_value` rename in the moreh kernel-helper pool would clear it for
   every moreh op at once; `generate_mask_h_w` and friends in the same header are already
   CB-name-free.
7. **Pre-existing `clang-format` violation in a file this port does not touch.**
   `device/moreh_linear_backward_device_operation.cpp:79` exceeds the column limit and
   `clang-format --style=file` wants to wrap it. The file is **not** in this port's diff and was not
   staged, so the pre-commit hook leaves it alone — but it will surface for whoever next edits that
   file.

### Test coverage notes

- **The op's only coverage is nightly**, in
  `tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_linear.py`; there is no non-nightly
  file and no C++ gtest. Confirmed with the invoker as the no-regression baseline before it was
  relied on. Worth knowing that a CI configuration which skips nightly has **zero** coverage of this
  op, ported or not.
- **The coverage does reach both factories and both fp32 settings**, which is why the *missing*
  `unpack_modes` entry would have failed loudly: `test_moreh_linear_backward` parameterizes
  `fp32_dest_acc_en ∈ {False, True}` over shapes with both scalar bias (`[1,1]` → SingleCore) and
  1-D bias (→ MultiCore), and `test_moreh_bias_backward_fp32` covers MultiCore under fp32.
- **It also reaches *both* sides of the single-core conditional DFB**, which is the part of this port
  that could not be verified by inspection alone: `([31, 31], [30, 31], [1, 1], [31, 30])` gives an
  `output_grad` of `[31, 30]`, so both `do_mask_h` and `do_mask_w` hold and the `DO_MASK_H_W` build
  runs; `([32, 64], [1024, 64], [1, 1], [32, 1024])` is tile-aligned in both dimensions, so the
  no-mask build — the one where `dfb::mask_h_w` does not exist and every gated reference must have
  been caught — runs too. Both compile and pass, so the `#ifdef` set is complete rather than merely
  plausible. What the coverage does **not** reach: a *wrong-valued* `unpack_modes` entry (no test distinguishes
  `UnpackToSrc` from `UnpackToDest`), a Float32 `output_grad` (finding 1), and `bfloat8_b` (skipped
  by the test itself).
- The same test file drives `moreh_matmul` and `moreh_sum` through the composite entry point, so a
  failure in it is not necessarily attributable to this op. Both are already ported, which is why
  the *pre-port* baseline run already emitted 776 `METAL2_CHECKS_FORCED` markers — a useful
  side-effect: it proved the forcing scaffolding was live before the port's own code existed.

### Per-op carry-over

- **`moreh/moreh_dot` looks to be missing its explicit compute `opt_level`, and it is already
  merged.** Noticed incidentally while reading it for spec *shape*; verified before reporting, and
  **not** touched (another op's code):
  - Pre-port, its compute kernel used `compute_desc.config = ComputeConfigDescriptor{…}` with no
    `opt_level` field anywhere in the factory
    (`git show 420b36650af^:ttnn/cpp/ttnn/operations/moreh/moreh_dot/device/moreh_dot_program_factory.cpp`,
    line 147) — which resolves to **`O3`**, the legacy `ComputeConfigDescriptor` default.
  - Post-port, `ttnn/cpp/ttnn/operations/moreh/moreh_dot/device/moreh_dot_program_factory.cpp` sets
    `.compiler_options = {.defines = {…}}` on its compute `KernelSpec` with **no** `opt_level`
    (`grep -n opt_level` over the file: zero hits), so it takes Metal 2.0's `CompilerOptions`
    default of **`O2`**.
  - That is a one-level drop on the compute kernel's compile *and* link — precisely the silent
    perf-only regression the recipe's *Compiler options* section exists to prevent, landed in
    commit `420b36650af` (PR #51085, "verified passing", which it would be: no test distinguishes
    optimization levels). Worth a one-line follow-up PR, and worth a sweep of the other merged ports
    for the same omission, since the rule is easy to miss when reading a reference port for shape.
- The three sibling reduce-shaped moreh ops that share this op's donor helpers
  (`ttnn/cpp/ttnn/kernel_lib/reduce_helpers_{dataflow,compute}.hpp`,
  `ttnn/cpp/ttnn/kernel/{dataflow,compute}/moreh_common.hpp`) all cross the boundary the same way,
  on `dfb::name`'s `constexpr uint32_t` conversion in non-type-template-parameter position. Nothing
  in this port needed a donor change, so any moreh op in the same family should expect a clean
  crossing too — the family's helpers are already DFB-aware.
- This op's `SingleCoreProgramFactory` is a compact worked example of a **conditionally declared**
  DFB (spec, both endpoint bindings, the `defines` entry, the `unpack_modes` entry, and the
  kernel-side `#ifdef`s all sharing one host-time condition). The merged `moreh_mean` H factory
  binds its mask DFB *unconditionally* and self-loops it on the off-path instead, which is a
  different resolution to a different shape (there the DFB is always allocated). A porter comparing
  the two should read the difference as "is the legacy CB allocation itself conditional?", not as
  two competing styles.
