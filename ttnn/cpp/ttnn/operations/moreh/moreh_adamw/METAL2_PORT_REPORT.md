# Metal 2.0 Port Report — `moreh_adamw`

*(Opened at the start of the port; friction captured as it happened, polished at the end.)*

## Outcome

**`PORTED`** — `MorehAdamWDeviceOperation::MultiCoreProgramFactory`, the op's only factory, converted to
`CustomProgramSpecFactoryConcept`. No factories left for a later pass. Tests match the pre-port baseline
exactly: **23 passed, 8 skipped**, both before and after
(`tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_adamw.py`; the 8 skips are the tests' own
`bfloat8_b`-without-`fp32_dest_acc` guards, unchanged by the port).

## Provenance

- **Recipe docs (this port):** `086a669ff5e 2026-08-15 docs(metal_2.0): two porter-facing gaps a blind cold read turned up`
- **Audit docs (inherited):** `086a669ff5e 2026-08-15 docs(metal_2.0): two porter-facing gaps a blind cold read turned up`

## TTNN ProgramFactory

### Concept realized

`CustomProgramSpecFactoryConcept`, as the audit chose. Not re-decided.

The override returns a `TensorArgument` for **every** io-tensor `TensorParameter` — all seven, or all nine when
`amsgrad` is on (`device/multi_core_program_factory.cpp:652-664`). None deliberately skipped. This mirrors the
ported-from override, whose address writes at the pre-port `:374-384` / `:398-400` / `:409-411` covered all
nine. The op has no op-owned tensors, so none are excluded by construction.

The ported-from override was **translated, not deleted**: it changed shape (lost its `Program&` parameter,
now returns `ProgramRunArgs`) as well as body. Its refresh set is reproduced exactly — reader `lr`,
`beta1_exponent`, `beta2_exponent`, `step`, and compute `step`, plus the nine addresses as bindings. Nothing
added: the reader's `beta1` / `beta2` / `eps` / `weight_decay` / `amsgrad` / `num_tiles_per_core` / `start_id`
and the writer's two RTAs were **not** refreshed by the ported-from override and are not refreshed here, so
the writer gets no `kernel_run_args` entry at all.

### Device-op-class edits

- **`program_factory_t` + nested factory struct introduced** (`device/moreh_adamw_device_operation.hpp:61-82`)
  — a forced edit outside the recipe's two documented exceptions. See Handoff points entry 1.
- Declarations removed from the device-op struct: `create_descriptor`, and the `void`-returning
  `override_runtime_arguments`. Both now live on `MultiCoreProgramFactory`.
- `#include <tt-metalium/program_descriptors.hpp>` removed from the device-op header — the port removed its
  only use (`ProgramDescriptor` in the `create_descriptor` return type).
- **Pybind entry points removed: none.** `moreh_adamw_nanobind.cpp` binds only
  `ttnn::bind_function<"moreh_adamw">` @ `:43`; `create_descriptor` was never pybound, so nothing to delete
  and no user-visible surface change.
- **Custom `compute_program_hash`: none; backdoor hash left intact** at
  `device/moreh_adamw_device_operation.hpp:35-40`, comment at `:31-34` untouched. Confirmed byte-identical.

### Open items

- **Relaxation candidates: none.** Every `TensorParameter` is strict; no kernel reads `ArgConfig::Runtime*`
  (grep clean), and the backdoor hash narrows only scalar attributes, never a `TensorSpec` property — so it
  reveals no tensor-property independence to relax. Consistent with the audit's `TensorParameter relaxation ==
  none`.
- **No `UpdateTensorArgs` legality failure.** The backdoor hash pins the whole `TensorSpec` via `tensor_args`,
  so the failure mode the docs warn about (second-and-later dispatch, cache hot) did not fire — and
  `test_moreh_adamw_callback`, which runs the op twice on the same key, is the test that would have caught it.

## Handoff points

1. **Framework gap — a device operation that declares its factory methods directly has no Metal 2.0 path,
   forcing a `program_factory_t` edit.** *Owner: TTNN framework team.*
   `MeshDeviceOperationAdapter` has a `DirectDescriptorFactory` shim
   (`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:170-187`) for ops that put `create_descriptor` on the
   device-op struct itself, and `HasDirectDescriptor` (`ttnn/api/ttnn/operation_concepts.hpp:139`) keys
   `DeviceOperationConcept` on that. There is **no** equivalent for `create_program_artifacts`, so removing
   `create_descriptor` without adding a `program_factory_t` fails the concept — an op in this shape cannot be
   ported without editing the device-op header beyond the two exceptions the recipe documents.
   What the port did: added `struct MultiCoreProgramFactory` holding the two static methods plus
   `using program_factory_t = std::variant<MultiCoreProgramFactory>;`
   (`device/moreh_adamw_device_operation.hpp:61-82`), keeping both definitions in the existing factory `.cpp`.
   Either a `DirectSpecFactory` shim mirroring `DirectDescriptorFactory`, or an explicit third documented
   exception in the recipe, would close this. Confirmed with the invoker before proceeding.
2. **Kernel-lib naming — donor functions carry `_cb` in their names, which the recipe's post-port `cb` sweep
   cannot distinguish from a real leftover.** *Owner: kernel-lib / `ttnn/cpp/ttnn/kernel/` maintainers.*
   `fill_cb_with_value` (`ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp:98`) and `mul_tiles_to_cb` /
   `sub_tiles_to_cb` / `add_tiles_to_cb` / `copy_tile_to_cb`
   (`ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp`) all match the checklist's pattern. They are outside the
   porter's writeable surface, so 26 hits survive in this op's kernels that no porter can fix. A `_dfb`
   rename (or a `_to_dfb` alias) when those donors are next touched would let the sweep mean what it says.
   No functional issue — the donors take `DataflowBuffer` by value and needed no change at all.
3. **Boundary-rule assumption violations: none.** No out-of-op call site required a `sem::` or `tensor::`
   handle. The op declares no semaphores, and both `TensorAccessor`s stay inside the op's own DM kernels.
4. **Kernel-lib / LLK capability gaps: none.** Every donor call compiled untouched: the object-taking helpers
   kept the kernels' existing local `DataflowBuffer` objects (now constructed from tokens), and the
   raw-CB-id LLK calls (`binary_op_init_common`, `sub_tiles`, `mul_tiles`, `add_tiles`, `copy_tile`) took
   `dfb::<name>` directly through the token's `uint32_t` conversion.

## Successes

- **The audit's config-scoped dead-CB finding was load-bearing, and the recipe's framing of it was correct.**
  The brief said in bold *"Do not read this as 'drop them'"* for `c_4` / `c_19` / `c_27`. Dropping them — the
  reflex the plain "dead CB → drop the allocation" rule invites — would have silently broken the entire
  `amsgrad == true` path, where all three are live. Instead they became `amsgrad`-conditional DFB specs
  (`device/multi_core_program_factory.cpp:185-190`). This is the finding that most needed to arrive as a
  warning rather than as a test failure, and it did.
- **[Conditional / optional DFB bindings](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-conditional--optional-dfb-bindings)
  fit this op exactly, with the promotion step already done.** The pattern's harder half — promoting an
  `if constexpr` CTA gate to a preprocessor `#ifdef` — was unnecessary: the legacy op already gated every
  `max_exp_avg_sq` reference on `#ifdef AMSGRAD` fed by `defines`, in all three kernels. So the port's work
  was purely making the *host* bindings conditional on the same flag
  (`device/multi_core_program_factory.cpp:276-286, 308-318, 437-448`). Having read the pattern first, I knew
  not to bind unconditionally to dodge the name-lookup problem.
- **[Compiler options](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#compiler-options)
  caught a silent perf loss I would otherwise have shipped.** `grep -n opt_level` over the legacy factory
  returns *nothing*, which reads as "this op doesn't care." The recipe's insistence that an absent
  `KernelDescriptor::opt_level` still resolves to **O3** on a compute descriptor is what put the explicit
  `KernelBuildOptLevel::O3` on both compute specs
  (`device/multi_core_program_factory.cpp:487`) — verified against
  `tt_metal/impl/program/program.cpp:456` (`opt_level.value_or(KernelBuildOptLevel::O3)`). Nothing in the
  build or the tests would have flagged the O2 drop.
- **The custom concept's "the override owns the tensor bindings too" warning is stated three times in
  `ttnn_factory.md`, and that is the right number.** It is the one mistake on this port that compiles, runs,
  and passes the first dispatch of every test — `test_moreh_adamw_callback` allocates fresh tensors on its
  second run, so an omitted `tensor_args` would have surfaced as wrong numerics only there.
- **The identity footgun generalized past where the recipe places it.** The recipe documents pointer-identity
  binding under *op-owned tensors*, which this op has none of. But the legacy factory took a **copy** —
  `const std::optional<Tensor> max_exp_avg_sq_out = amsgrad ? tensor_return_value.at(3) : std::nullopt;` — and
  binding through that copy is the same hazard in a different place. Having read the footgun note, I bound a
  pointer to the vector element instead (`device/multi_core_program_factory.cpp:139`).

## Friction

### Gaps

1. **Nothing documents the direct-`create_descriptor` → Metal 2.0 transition, and the audit recorded the shape
   as a neutral fact.** `ttnn_factory.md` and the migration guide both present the factory as a separate
   struct (`struct MyProgramFactory { ... }`) without saying where it has to be reachable from; the audit's
   opening line notes *"there is no separate `ProgramFactory` class"* as description, not as port work. So the
   need for a `program_factory_t` is invisible until the concept check fails, and the fix is an edit the
   recipe's scope discipline otherwise forbids. **Suggest:** name this shape in `ttnn_factory.md`'s
   *Device-operation-class edits the port forces* section as a third exception, with the
   `HasDirectDescriptor` reasoning; and have the audit flag `Concept == descriptor` **without** a
   `program_factory_t` as a port-work item rather than a description. Full detail in Handoff points 1.

2. **The compute-`hw_config` Style A / Style B test misclassifies a hybrid, and getting it wrong is silent.**
   Style A's recognition signal is "the op resolves a TTNN `ComputeKernelConfig` via
   `get_compute_kernel_config_args`" — this op does, at the pre-port `:98-99`. Style B's is "the op sets a
   Metal `ComputeConfigDescriptor` directly, **with no TTNN `ComputeKernelConfig` feeding them**" — this op
   also does, at the pre-port `:241-245`, but the values *are* TTNN-fed, so the parenthetical excludes it.
   Both signals half-fire. Following the Style A label and calling
   `to_compute_hardware_config(arch, config)` would set `double_buffer_dest = !config.dst_full_sync_en`,
   whereas the legacy op destructures `dst_full_sync_en` and then **drops** it, so it always resolves to the
   `ComputeConfigDescriptor` default `false` ⇒ `double_buffer_dest = true`, whatever the caller passed. For
   any caller setting `dst_full_sync_en = true` that is a silent throughput/capacity change with no test net —
   exactly the class of error the *Hardware configuration* section exists to prevent, arrived at *by
   following the section*. Resolved by building `ComputeGen1Config` by hand from the three fields the legacy
   descriptor actually set (`device/multi_core_program_factory.cpp:456-462`).
   **Suggest:** make the test the config the op *hands to the framework*, not the provenance of its values —
   e.g. *"if the op hand-assembles a Metal compute config, copy field-by-field from that config and leave
   unset fields at their Metal defaults, regardless of what fed the values. A field the op destructures and
   then drops is a field the op leaves at the descriptor default; carrying it across from the TTNN config is
   a behavior change."* That phrasing also covers the `packer_l1_acc` case in the same op.

3. **The `cb`-leftover sweep's "expect zero hits" is unreachable for any op consuming `_cb`-named donors.**
   26 hits survive in this op, every one a donor *function name* the porter may not rename (Handoff points 2).
   The checklist's wording — *"post-port the op has no CBs, so every hit is a real leftover rather than
   noise"* — is what makes this friction rather than a shrug: it asserts the count is zero, so a porter
   either edits out of scope or is left unable to complete a checklist item as written. **Suggest:** add a
   third exclusion class beside `cbegin` / `cbrt` — *"a hit resolving to a kernel-lib or donor function name
   (`*_to_cb`, `fill_cb_with_value`) is out of the porter's writeable surface: record the count in the report
   and move on."*

4. **`TT_KERNEL` is the most prominent thing in the one header the recipe tells you to add, and the recipe
   never mentions it.** The whitelist says the port adds `experimental/kernel_args.h`; that header's macro
   block (`tt_metal/hw/inc/experimental/kernel_args.h:44`) documents `TT_KERNEL` as *"marks the named-arg
   entry point; the JIT generates kernel_main() from its signature"*, which reads as the required new entry
   form for a named-arg kernel. I kept plain `kernel_main()` only after reading
   `tt_metal/jit_build/kernel_signature_parser.hpp:34` and confirming a source with no marker is treated as
   *"a legacy / hand-written kernel_main()"*. **Suggest** one clause in whitelist rule 4: *"the kernel keeps
   its existing `void kernel_main()` entry point; the `TT_KERNEL` marker in that header is a separate,
   optional mechanism and not part of this port."*

### Confusion

5. **"Float32 DFB" in the `unpack_modes` rule needs a DFB-format reading, and the tensor-dtype reading of it
   is both natural and wrong.** The rule fires on a consumed **Float32 DFB** with `enable_32_bit_dest`. In
   this op *no tensor is ever Float32* — `validate_inputs` admits only `BFLOAT16` / `BFLOAT8_B` — yet **ten
   DFBs are**, because the intermediates take `fp32_dest_acc_en ? Float32 : data_format` (pre-port `:105`).
   A porter who checks "does this op deal in fp32?" against the op's dtypes concludes the rule is
   inapplicable, and ships a spec that validates fine on the default path and fails only under
   `test_moreh_adamw_compute_kernel_options`. I nearly took that route; what redirected me was reading the
   validator itself (`tt_metal/impl/metal2_host_api/program_spec.cpp:1049-1078`), which keys on
   `dfb_spec->data_format_metadata`, not on any tensor. **Suggest** the *Hardware configuration* item name the
   idiom: *"the trigger is the DFB's `data_format_metadata`, not the op's tensor dtypes — the common shape is
   an op whose intermediate buffers are `fp32_dest_acc_en ? Float32 : data_format`, so the requirement appears
   only in the fp32 configuration of an op with no fp32 tensors."* Ten entries were needed here
   (`device/multi_core_program_factory.cpp:468-476`), all `UnpackToSrc` per the legacy all-`Default` vector.

6. **Minor — the `TT_FATAL` census command's `BASE` is wrong on a branch that already carries commits.** The
   recipe sets `BASE=$(git merge-base origin/main HEAD)` and calls it "the pre-port revision." This branch
   already had a prior port and several doc commits, so the merge-base is well behind the port's true
   starting point; it happened not to matter here (this op's directory was untouched by those commits, and
   both bases gave *no output*), but on a branch where an earlier commit touched the same op the census would
   silently compare against the wrong tree. **Suggest:** *"`HEAD` is the base while the port is uncommitted;
   use the merge-base only if you have already committed part of the port."*

## Open items for downstream

- **Shared kernel touches: none.** All three kernel sources are owned exclusively by this op; the census
  (`grep -rl <filename> ttnn/cpp/ttnn/operations/`) finds no other binder, no `_metal2` fork exists or was
  created, and there is no sunset list. The one hit to discard is `moreh/sources.cmake`, which lists the
  **host** wrapper `moreh_adamw/moreh_adamw.cpp` — same filename as the compute kernel, different file.
- **RTA → CRTA candidates (a later cleanup pass, deliberately not done here).** Of the reader's 11 named
  RTAs, **nine** are set to the same value on every node — `lr`, `beta1`, `beta2`, `eps`, `weight_decay`,
  `beta1_exponent`, `beta2_exponent`, `step`, `amsgrad` — as is the compute kernel's `step`. Only
  `num_tiles_per_core` and `start_id` genuinely vary per node (on both DM kernels). Promoting the nine to
  `common_runtime_arg_values` would cut per-node dispatch work substantially, but it changes dispatch
  semantics, so the port left them as per-node RTAs.
- **Name-first RTA restructure (same later pass).** The port kept the legacy node-first core loop and bridged
  it with `AddRuntimeArgsForNode` (`device/multi_core_program_factory.cpp:531-560`), per the recipe's
  guidance not to invert the loop as port work.
- **The audit's brittle-override-guard concern is resolved as a side effect, not carried forward.** The
  ported-from override keyed its liveness check on a **dead** argument —
  `if (a.size() <= kReaderStepIdx) continue;` with `kReaderStepIdx = 12` (pre-port `:372`, `:395`) — which the
  audit flagged as anchored to an index that cleanup could remove without anyone connecting the two. Metal
  2.0 addresses runtime args by name and the translated override walks the work split's own node list, so the
  positional guard is gone entirely. Removing reader `step` is now a safe cleanup for the ops team, where
  before the port it would have broken the override.
- **Three dead runtime args carried forward untouched**, as the audit directed: reader `step`
  (`device/kernels/reader_moreh_adamw.cpp:23`), reader `amsgrad` (`:24`), compute `step`
  (`device/kernels/moreh_adamw.cpp:18`). Each is read into a named local and never referenced. Removing them
  is an ops-team functional change. Note the compute kernel's `step` is its **only** runtime arg — dropping it
  would let the compute `KernelRunArgs` entry disappear from both the miss and hit paths.
- **Two unused includes left in place** (audit *Misc anomalies*), and the port makes the first of them more
  clearly dead: `<tt-metalium/experimental/program_descriptor_patching.hpp>` @
  `device/moreh_adamw_device_operation.hpp:16` (nothing from it referenced, and the descriptor path it belongs
  to is now gone), and `"ttnn/operations/moreh/moreh_helper_functions.hpp"` @
  `device/multi_core_program_factory.cpp:12`. Both were already unused pre-port, so removing them is not
  attributable to the port.
- **`packer_l1_acc` and `dst_full_sync_en` are still destructured from the resolved compute config and
  dropped** (`device/multi_core_program_factory.cpp:157-158`, audit *Misc anomalies*). The port preserves the
  drop exactly — that fidelity requirement is what forced the hand-built `ComputeGen1Config` (Friction 2). Two
  user-settable fields therefore still participate in the cache key while having no effect on the program.
  `dst_full_sync_en` now has a direct Metal 2.0 home (`double_buffer_dest`), so if the ops team decides the
  drop was an oversight, wiring it up is a one-line change — in a separate PR.
- **Test-coverage gap, now higher-value than before the port.** No test varies `lr` or `step` across a
  program-cache *hit* — which is the exact scenario the backdoor hash exists to enable and the translated
  override exists to service. `test_moreh_adamw_callback` runs the op twice with **identical** hyperparameters,
  so it covers the tensor-binding refresh (fresh tensors each call) but not the scalar refresh: if the
  override dropped `lr` / `step` / the β-exponents, every current test would still pass. A callback variant
  that steps `step` (and ideally `lr`) between iterations and checks numerics against torch would close it.
  Pre-existing, but the port changed the code that depends on it.
- **L1 footprint improves under `amsgrad == false`** — three fewer per-core buffers (2 × `data_tile_size` +
  1 × `intermed_tile_size`), because the three config-dead DFBs are no longer declared. This is the port's
  **only** observable behavior difference, and it is forced by the validator rather than chosen: a bindingless
  DFB cannot be expressed. Flagging it explicitly since the porting invariant is otherwise zero-change.
- **Latent legacy inconsistency, not fixed:** `amsgrad == true` with `max_exp_avg_sq_in` absent. Legacy appends
  the 5th `TensorAccessorArgs` block only `if (max_exp_avg_sq_in.has_value())` (pre-port `:197-199`) while the
  kernel's `#ifdef AMSGRAD` block unconditionally reads a 5th accessor, so that combination already mismatches
  CTA supply against kernel expectation. Nothing enforces the coupling — `validate_inputs` only dtype-checks
  when present — and the only caller (the pytest) always passes it iff `amsgrad`. Post-port the combination
  raises from `std::optional::value()` instead of reading a garbage accessor. If the coupling is real, a
  `TT_FATAL` in `validate_inputs` is the right home; that is a device-op-class edit and an ops-team call.
- **Per-op carry-over.** The other `moreh` optimizer ops (`moreh_sgd`, `moreh_adam`, …) share this op's shape
  — `fill_cb_with_value`-fed scalar buffers, compute-scratch self-loops, an `intermed_cb_format` keyed on
  `fp32_dest_acc_en`, and per-group compute CTAs. The `unpack_modes` finding (Friction 5) and the
  hand-built-`ComputeGen1Config` finding (Friction 2) will very likely recur verbatim across that family, as
  will the `_cb`-named donor sweep noise. A porter picking up the next one should read those two entries first.
