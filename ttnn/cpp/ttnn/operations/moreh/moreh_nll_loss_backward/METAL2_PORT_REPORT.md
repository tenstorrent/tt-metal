# Metal 2.0 Port Report — `moreh_nll_loss_backward`

## Outcome

**`PORTED`** — `MorehNllLossBackwardDeviceOperation::Factory`, the op's only factory, converted to the
base `ProgramSpecFactoryConcept` across all three of its rank-dispatched code paths (2d / 3d / 4d)
together with all five kernels it binds. Nothing is left for a later pass: the op has no other
factory and no other kernel.

## Provenance

- **Recipe docs (this port):** `38da2cdbd29 2026-08-13 docs(metal_2.0): stop gating on a pybound create_descriptor, and drop a stale reference port`
- **Audit docs (inherited):** `38da2cdbd29 2026-08-13 docs(metal_2.0): stop gating on a pybound create_descriptor, and drop a stale reference port`

No reference port was supplied by the invoker, and none was sought. The spec shape was derived from
the recipe, the patterns catalog, the migration guide, and the Metal 2.0 headers themselves.

## Verification

**Build:** `./build_metal.sh --build-tests` — SUCCESS, **zero warnings**.

**Tests** (confirmed with the invoker as the complete no-regression baseline:
`tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_nll_loss.py`, the only file exercising
this op; no C++ gtests, no sweeps. `test_moreh_nll_loss_unreduced.py` is a different device op and
was excluded):

| Selection | Pre-port | Post-port |
|---|---|---|
| `-k backward` (60 tests — the 3 backward test functions) | **32 passed, 28 skipped, 0 failed** | **32 passed, 28 skipped, 0 failed** |
| whole file (130 tests, incl. the untouched forward op) | *(not measured)* | 70 passed, 60 skipped, 0 failed |

Identical tallies before and after. All 28 backward skips are `Support for bfloat8_b is currently
unavailable` — a dtype guard evaluated before the op runs, so unaffected by the port.

Coverage of the paths this port had to get right: all three rank paths (2d `[400, 300]`, 3d
`[20, 300, 320]`, 4d `[5, 2, 5, 40, 70]` / `[10, 20, 30, 40]`); both optional-tensor axes
(`none_weight` True/False, `reduction_mean` True/False driving `divisor`); both `fp32_dest_acc_en`
settings — which is what exercises the new `unpack_modes` entries; and the program-cache hit path
(`test_moreh_nll_loss_backward_test_callback`, 4 passed), which is what exercises the framework's
`TensorArgument` re-patching that replaced the legacy `Buffer*` RTAs.

**Anti-pattern self-audit:** all checklist items pass. Zero hits in code for the `cb` sweep
(`grep -rnE '[Cc][Bb]_|_[Cc][Bb]\b|\b[Cc][Bb]\b|\bCB[A-Z]'`), `CircularBuffer` / `CBDescriptor` /
`ProgramDescriptor` / `.cbs`, `TensorAccessorArgs`, `get_arg_val` / `get_compile_time_arg_val` /
`get_vararg`, `.id` extraction, `allow_instance_multi_binding`, and `.md` citations from code. The
only `.buffer()` hits left in the directory are the pre-existing null-checks in the device-op class's
`validate_inputs`, which the port did not touch. `TT_FATAL` census: see *Open items* below — 6 → 3 in
the factory, all three losses subject-deleted, 0 change in every other file.

## TTNN ProgramFactory

### Concept realized

`ProgramSpecFactoryConcept` — the base concept the audit chose, unchanged. `Factory::create_descriptor`
became `Factory::create_program_artifacts` returning `ttnn::device_operation::ProgramArtifacts`.
There was no `override_runtime_arguments` to translate, so nothing routed to
`CustomProgramSpecFactoryConcept`, and `op_owned_tensors` is left defaulted (the factory allocates no
device tensors beyond the op's io).

### Device-op-class edits

- **Pybind entry points removed:** none. `moreh_nll_loss_backward_nanobind.cpp` binds only the
  user-facing op (`ttnn::bind_function<"moreh_nll_loss_backward">`); nothing referenced
  `create_descriptor`. **This port carries no user-visible API change.**
- **Custom `compute_program_hash`:** none present (default reflection-based hash). Nothing to
  preserve, nothing touched.
- **The one edit the port forced:** `device/moreh_nll_loss_backward_device_operation.hpp` — the
  `Factory::create_descriptor` declaration became `create_program_artifacts`, and the now-unused
  `<tt-metalium/program_descriptors.hpp>` include was replaced by `"ttnn/metal_v2_artifacts.hpp"`.
  Two lines; the rest of the device-operation class is byte-identical (verified with
  `git diff` against the merge-base — `moreh_nll_loss_backward_device_operation.cpp`,
  `moreh_nll_loss_backward.{cpp,hpp}` and the nanobind file show no diff at all).

### Open items

- **Relaxation candidates:** none identified. The audit's `TensorParameter relaxation` cell reads
  `none` and all five `TensorParameter`s are left strict. Nothing in the kernels suggested a
  tolerable relaxation — the readers index tiles by exact padded shape.
- **Capabilities not yet on this concept:** none needed.
- **Concept fit:** clean. The op is single-program, has no semaphores, no op-owned tensors and no
  custom hash — the simplest shape this concept serves.

## Handoff points

**None.** No capitulation, no boundary-rule violation, no kernel-lib gap, no framework gap, no
removed pybind surface.

Specifically, the things the recipe warns can force a stop did not occur: no `GlobalCircularBuffer`,
no `get_cb_tiles_acked_ptr` / `get_cb_tiles_received_ptr`, no Case 2 (raw base pointer) binding, no
host-computed `base + offset` fold, and no call site outside the op directory demanding a `sem::` or
`tensor::` handle. Every donor call site bridged with no work at all — the shared-pool helpers take
`DataflowBuffer` by value (implicit from `dfb::name`) and the accessor as a template parameter, and
the raw-`uint32_t` LLKs are covered by the token's `constexpr operator uint32_t()`.

No shared kernels: all five sources live in this op's own `device/kernels/` and are bound only by
this factory, so no `_metal2` fork was created or reused and no peer directory was written to.

## Successes

- **[Patterns catalog — Anti-pattern: Demoting per-group CTA to RTA] gave the exact `WorkUnitSpec`
  shape.** The legacy factory emits the compute kernel twice over disjoint core groups
  (`program_factory.cpp:169`/`:185` pre-port). The entry's "Correct port" block — two same-source
  `KernelSpec`s in two `WorkUnitSpec`s over the disjoint groups, both binding the same DFBs — was
  directly transcribable, and its "Constraint" paragraph pre-empted the misread I would otherwise
  have made: this is the *disjoint-node* work-split, so each node sees one compute instance and every
  shared DFB stays an ordinary single-role binding. No `allow_instance_multi_binding`, no 1P+1C
  reassignment. Landed at `moreh_nll_loss_backward_program_factory.cpp:362-374` (2d) and the 3d/4d
  equivalents.

- **[Port recipe — Hardware configuration, `unpack_modes`] fired exactly as advertised, and it is
  genuinely the dangerous one.** With `fp32_dest_acc_en` on, this op's `tmp_weight` / `tmp1` / `tmp2`
  buffers become `Float32` while `enable_32_bit_dest` becomes `true` — precisely the combination the
  validator requires an explicit entry for, and legacy supplied none (it passed
  `UnpackToDestMode::Default` for every CB). The recipe's instruction to *derive* the value from the
  legacy vector rather than guess (`Default` → `UnpackToSrc`) is what makes this safe; guessing
  `UnpackToDest` would have compiled, passed nothing loudly, and silently changed the precision
  tradeoff. Implemented as `require_unpack_mode` (`program_factory.cpp:61-67`) and applied per rank
  path. Exercised by `test_moreh_nll_loss_backward_compute_kernel_options[...fp32_dest_acc_en=True...]`.

- **[Port recipe — Compiler options] caught a silent level drop.** `grep -n opt_level` over the
  legacy factory returns zero hits, which reads as "nothing to carry." The recipe's rule 2 is the
  only reason the two compute `KernelSpec`s got an explicit `KernelBuildOptLevel::O3` — legacy
  `ComputeConfigDescriptor` defaults to `O3`, Metal 2.0's `CompilerOptions` to `O2`. Nothing in the
  build or the tests would have flagged the difference.

- **[Port brief — CB endpoints] was correct on all nine buffers.** I re-derived the census
  independently per the recipe's instruction to verify rather than transcribe; every disposition
  agreed (4 plain 1P+1C, 5 self-loops across configs, 1 dead-CB drop, 2 config-conditional). The
  brief's pre-emptive "you do not need to re-hunt multi-binding" was also right — the reader's raw
  `get_write_ptr()` write into `tmp_weight` is bracketed by its own `reserve_back`/`push_back`, which
  is the producer's own peek and exactly the shape the brief said would invite a misread.

## Friction

### Gaps

- **The brief under-counted the unguarded kernel-side DFB constructions — three, not two.** The
  brief's *"And move two kernel-side constructions inside the guard"* names `dfb_tmp1_obj` and
  `dfb_tmp2_obj` (`moreh_nll_loss_backward_kernel.cpp:23,25` pre-port). There is a **third**:
  `DataflowBuffer dfb_divisor_obj(cb_divisor)` at pre-port `:17`, constructed unconditionally while
  *every* use sits inside `#if defined(DIVISOR)` (`:35`, `:113`). `c_3` is not allocated in the
  no-divisor config either (`push_cb` early-returns on `num_tiles == 0`), so `dfb::divisor` does not
  exist there and the no-divisor build would have failed to compile on an undeclared binding. Fixed
  in the port (`moreh_nll_loss_backward_kernel.cpp:15-17`).

  This is the audit's own recipe-note 3 biting the brief that recipe note produced, which makes it
  worth more than a one-off correction: a hand-written list of unguarded mentions is not a reliable
  instrument. **Suggested doc change:** state the *procedure* rather than the list — for every
  conditionally-declared DFB, grep the kernel for each mention of its handle (construction, metadata
  lookup, alias) and confirm each one sits inside the guard. That mechanically finds all three here;
  reading the brief's list finds two.

- **The migration guide's "identical `WorkUnitSpec` membership" line contradicts the catalog's
  work-split shape, and the catalog is right.** [migration guide — Troubleshooting] states: *"**Local
  DFB invariant.** A local DFB's producer and consumer kernels must share *identical* `WorkUnitSpec`
  membership."* Under the shape the catalog prescribes (and that this op needs), the reader belongs
  to **both** work units while `compute_group_1` belongs to only one — so read literally, the
  guide forbids the catalog's own worked example. I could not resolve it from the docs and went to
  the validator, which settles it: the check is a **per-node census** of producer/consumer
  *instances* (`tt_metal/impl/metal2_host_api/program_spec.cpp:1324-1400`), not a set-equality test
  on work-unit membership, and the catalog's shape satisfies it. **Suggested doc change:** restate
  that bullet in per-node terms — *"every node a local DFB is instantiated on must run exactly one
  producer instance and exactly one consumer instance"* — which is both what the validator enforces
  and what a porter can check by hand.

- **The dead-CB resolution list still has no "config-conditional" entry.** Already logged by the
  audit as recipe note 2; confirming it from the porting side because it cost real time. `c_25` /
  `c_26` are live under `DIVISOR` and dead without it, while the legacy *allocation* is
  unconditional. "Drop it" breaks the divisor path; "keep it" fails validation on the other. The
  actual fix — gate the `DataflowBufferSpec` on the same predicate the factory already uses for
  `c_3` — is not among the four dispositions the recipe lists (self-loop / 1P+1C / multi-binding /
  drop). **Suggested doc change:** add the fifth disposition. An op with optional tensors hits this
  routinely and both wrong answers look plausible.

### Confusion

- **`get_compute_kernel_config_args` vs `to_compute_hardware_config` — the recipe says "mirror the
  op's style" without saying the two read the same thing.** The legacy factory takes its four compute
  knobs from `get_compute_kernel_config_args(arch, config)` (`:73-74` pre-port), while the recipe's
  Style A routes the port through `to_compute_hardware_config(arch, config)`. Since this is the one
  place a port can silently shift a precision/perf setting, I could not take the substitution on
  faith and read `ttnn/cpp/ttnn/operations/core/compute_kernel/compute_kernel_config.cpp:99-136` to
  confirm that `get_compute_kernel_config_args` is a pure field passthrough that ignores `arch`, so
  the two see identical values. **Suggested doc change:** one clause in the Style A bullet — *"both
  helpers read the same `ComputeKernelConfig` fields, so the substitution is exact"* — removes the
  check for every porter.

- **Minor: the `unpack_modes` free-function accessor collides with the obvious local variable name.**
  `tt::tt_metal::experimental::unpack_modes(config)` is how you reach the field through the variant,
  so a local named `unpack_modes` shadows the accessor and the assignment fails to compile in a
  confusing way. Named the local `compute_unpack_modes` instead (`program_factory.cpp:307`). Not a
  doc bug — just a sharp edge worth one line in the Hardware configuration section, since the
  header's own usage example (`auto& dfb_unpack_modes = unpack_modes(compute_hw);`) already dodges
  it without saying why.

## Open items for downstream

- **Shared kernel touches:** none. No fork created, none reused, no kernel modified outside this op's
  directory. Nothing for a future sibling-op porter to coordinate with.

- **Dead plumbing the port removed, for the ops team's awareness.** The brief instructed that these
  not be carried across as named args; because the host stops emitting them, the matching kernel-side
  reads had to go with them or they would read positional args that no longer exist:
  - reader `element_size` — host computation (`:205`/`:419`/`:636` pre-port) and the unused kernel
    local in all three readers.
  - compute RTA 0 (never read) and RTA 1 `tile_offset` (read into an unused local at
    `moreh_nll_loss_backward_kernel.cpp:14` pre-port). Both dead, so the compute kernels now have no
    runtime args at all.
  - compute CTA 1 `divisor_has_value` (`:169` etc.) — never read; the kernel branches on the
    `DIVISOR` define, which the factory also supplies.
  - the nine dead `get_dataformat(cb_id)` locals across the three readers — deleted, per the ops
    team's confirmation relayed by the invoker (audit Question 2).

- **Guard census: three assertions lost, all subject-deleted.** Because the compute kernels now carry
  no runtime args, the per-group compute RTA assignment block disappeared, and with it the only
  assertion inside it: `TT_FATAL(false, "Core not in specified core ranges.")` in the 2d impl
  (`:243` pre-port) and `TT_ASSERT(false, ...)` in 3d (`:458`) and 4d (`:675`). **The condition is
  still checked** — the `TT_THROW("Core not in specified core ranges")` earlier in the same loop
  (`:215`/`:429`/`:646` pre-port) survives untouched in all three impls, and it is the one that
  actually fires first. Net census: 6 → 3 in the factory, 0 change everywhere else.

  Incidentally this retires the audit's *"Inconsistent assertion macro"* anomaly — the `TT_FATAL` vs
  `TT_ASSERT` divergence lived entirely on the deleted lines. It was not fixed; its subject went away.

- **Findings left in place, for the op owners.** Each of these is real and each was deliberately
  *not* touched, per scope discipline:
  - **`reduction_mean` is accepted, hashed, and ignored by all three impls** (`:52`/`:265`/`:480`
    pre-port, now `:76`/`:435`/`:799`). Left completely alone on the invoker's explicit instruction:
    it is public API with an external consumer (`tt-train/sources/ttml/ops/losses.cpp`), so removing
    it is a separate decision on a separate track. Flagged here only because it still participates in
    the default program hash, so two calls differing *only* in `reduction_mean` occupy two cache
    entries that compile to identical programs.
  - **The 2d reader tiles a width dimension with `TILE_HEIGHT`.**
    `uint32_t Ct = (C + TILE_HEIGHT - 1) / TILE_HEIGHT;` (`reader_..._2d.cpp:46` post-port) computes
    the same quantity the host computes as `tt::div_up(channel_size, tt::constants::TILE_WIDTH)`, and
    the derived index uses `c = ct * TILE_WIDTH + w`. For `input_grad: (N, C)` the channel is a width
    dimension, so `TILE_WIDTH` is the semantically correct divisor. Numerically identical today (both
    32) and therefore invisible; it would break silently if the tile dimensions ever diverged.
  - **Dead local `n` in the 2d reader** (`reader_..._2d.cpp:64` post-port) — computed in the
    innermost loop and never used; the row within the tile is addressed via `get_tilized_idx(0, h)`.
    Carried across unchanged because it is not port-forced (unlike the dead *args* above, nothing
    about Metal 2.0 makes this line stop compiling). The 3d reader's `n` *is* used.
  - **Unreachable output-allocation path.** `compute_output_specs` unconditionally `TT_FATAL`s when
    `input_grad_tensor` is absent (`...device_operation.cpp:87`), so the `create_device_tensor` call
    in `create_output_tensors` (`:96-97`) is dead — the op always requires a preallocated
    `input_grad_tensor`, while the pybind default advertises `input_grad_tensor = None`. Device-op
    class code; off-limits to the port.

- **Test coverage note.** 28 of the 60 backward tests skip on this bench with *"Support for bfloat8_b
  is currently unavailable"*, so the port's `bfloat8_b` behaviour is unverified here — identically
  before and after, so it is not a regression risk introduced by this change, but it does mean
  `data_format = Bfp8_b` never reached the DFB specs in this run. The `fp32_dest_acc_en=True` path
  (the one that drives the new `unpack_modes` entries) *is* covered, on `bfloat16`.
