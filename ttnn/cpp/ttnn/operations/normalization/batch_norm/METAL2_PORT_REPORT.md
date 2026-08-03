# Metal 2.0 Port Report — `normalization/batch_norm`

Port of both device-operations under `ttnn/cpp/ttnn/operations/normalization/batch_norm/` from
`ProgramDescriptor` to Metal 2.0. Structural decisions and the legacy inventory are in
`METAL2_PORT_PLAN.md`; this report records what happened during the port.

## Outcome

**`PORTED`** — both factories, in one PR, as the invoker scoped it:

- `RunningStatistics::RunningStatisticsProgramFactory` + its 2 dataflow and 2 compute kernels
- `BatchNormOperation::BatchNormFactory` + its 2 dataflow and 2 compute kernels

Ten files converted (2 factories + 8 kernels) plus one entry-point retarget per device-op header.
Nothing was left for a later pass; no factory capitulated.

**Verification** (single-card Wormhole; `./build_metal.sh --build-tests` succeeded with **zero compiler
errors on the first post-conversion build**, and on every rebuild after it):

| Run | Result |
|---|---|
| **Pre-port baseline** — `test_batch_norm.py` + `test_batch_norm_program_cache.py` | **1560 passed, 786 xfailed, 0 failed** (250s) |
| **Post-port** — same set | **1560 passed, 786 xfailed, 0 failed** (171s) |
| **Post-port, after a readability edit** — same set | **1560 passed, 786 xfailed, 0 failed** (83s) |
| **Post-port, after `clang-format` (pre-commit hook)** — same set | **1560 passed, 786 xfailed, 0 failed** (151s) |

Identical counts every time, no `TT_FATAL`, no `0xdeadc0de`, no segfault, no JIT `static_assert`. The
wall-clock spread across the four runs is **JIT-cache warmth, not a perf signal** — the baseline
compiled every kernel cold, each post-port run recompiled whichever of the eight kernels had changed
text, and the fastest run hit a fully warm cache. Do not read any of these numbers as a speed-up; see
the perf note under Open items for what would actually measure that.

The `clang-format` reformat was verified to be line-wrapping only (`diff -w` against the pre-format
sources is content-identical), and the full test set was re-run against it regardless.

Two further checks were run because the confirmed pytest set leaves gaps:

- **RunningStatistics' typecast path is not reachable from any committed test, so it was verified
  directly.** `needs_{mean,var}_typecast` requires `interm_data_format == Float32` *and* the running
  statistics *not* being `Float32` — i.e. a **float32 input with bfloat16 running stats**, since in
  training mode `batch_mean`/`batch_var` inherit the input's dtype (`batch_norm.cpp:104-116`). The
  suite parametrizes a single `testing_dtype` for everything, and
  `test_batch_norm_compute_config`'s `(input_dtype, param_dtype)` pairs are `(bf16,bf16)`,
  `(bf16,fp32)`, `(fp32,fp32)` — never `(fp32,bf16)`. A scratch script driving that config plus its
  three siblings passed all four (`pcc ≥ 0.99999` on output and both updated statistics), so the
  `MEAN_NEEDS_TYPECAST` / `VAR_NEEDS_TYPECAST` `#ifdef` branches and the `writer_updated_*` DFB
  bindings are exercised. (BatchNorm's `needs_output_typecast` **is** covered, by the `(bf16,fp32)`
  pair.) This is a **pre-existing** coverage gap, not one the port introduced — but the port
  restructures exactly that path, so shipping it unexercised was not acceptable.
- **The sweep is unrunnable on this branch, so the axes it would have added were sampled instead.**
  `tests/sweep_framework/sweeps/normalization/batch_norm/batch_norm.py` fails at **import** —
  `ModuleNotFoundError: No module named 'tests.ttnn.unit_tests.operations.eltwise.backward'` (line 20)
  — both under `sweeps_parameter_generator.py` and under plain `pytest --collect-only`. Cause:
  `tests/__init__.py` makes `tests` a regular package while `tests/ttnn/**` has no `__init__.py`, so
  the dotted path cannot resolve. Entirely independent of this port (the port touches no file under
  `tests/`, and the failure happens before any op runs); fixing it means editing test infrastructure,
  which is out of scope. What the sweep adds over the pytest set is `input_memory_config`
  (**L1** as well as DRAM) and a larger shape range, so a scratch script covered
  3 shapes — `(1,1,32,32)`, `(3,5,192,192)`, `(5,9,256,256)` — × {bf16, fp32} × {L1, DRAM} ×
  6 mode combinations (training on/off, asymmetric running-stat presence, weight/bias on/off) =
  **72 cases, all passed**. **Explicitly not covered:** the sweep's full 3072-vector product (4 shapes
  × 2 dtypes × 2 memory configs × 2 training × 2 check_mean × 2 check_var × 2 weight × 2 bias × 4 eps
  × 3 momentum); at its 30 s timeout that is over a day of device time. The `eps` and `momentum` axes
  in particular were sampled at one value each here, though the committed pytest set does parametrize
  both.

## Provenance

- **Recipe docs (this port):** `4386dc456a1 2026-07-29 docs(metal_2.0): require an explicit opt_level when porting compute kernels`
- **Audit docs (inherited):** `4386dc456a1 2026-07-29 docs(metal_2.0): require an explicit opt_level when porting compute kernels`

## TTNN ProgramFactory

### Concept realized

`ProgramSpecFactoryConcept` for **both** factories, exactly as the audit chose. Each
`create_program_artifacts` returns a `ttnn::device_operation::ProgramArtifacts` with `spec` +
`run_params` and `op_owned_tensors` left defaulted — neither factory allocates device tensors beyond
its declared io. No disagreement with the audit's decision arose at any point.

Each factory's realized shape: 3 `KernelSpec`s, one `WorkUnitSpec` over `all_device_cores`, no
`SemaphoreSpec`s, and no `advanced_options` on any spec (no `alias_with`, no
`allow_instance_multi_binding`, no varargs, no `TensorParameter` relaxation, no `borrowed_from`).

### Device-op-class edits

- **Custom `compute_program_hash` deleted:** none — neither device-op had one, so the sanctioned
  deletion did not trigger.
  - `BatchNormOperation::operation_attributes_t::to_hash()` (declared
    `device/batch_norm_device_operation.hpp:22`, defined `device/batch_norm_device_operation.cpp:121`)
    is the ttsl attribute-hash protocol, **not** a `compute_program_hash` override. The brief told the
    porter to leave it and the invoker confirmed the recipe default; it is untouched.
- **Pybind entry points removed:** none — `batch_norm_nanobind.cpp` exposes only the user-facing
  `ttnn::batch_norm`, so no factory entry point vanished from the Python surface.
- **Entry-point retarget (the only edit in each header), 2 files:**
  - `device/running_statistics_device_operation.hpp:11,36`
  - `device/batch_norm_device_operation.hpp:11,39`

  In each: `#include <tt-metalium/program_descriptors.hpp>` → `#include "ttnn/metal_v2_artifacts.hpp"`,
  and `static tt::tt_metal::ProgramDescriptor create_descriptor(...)` →
  `static ttnn::device_operation::ProgramArtifacts create_program_artifacts(...)`. Nothing else in
  either device-op class changed.

### Open items

- **Relaxation candidates: none applied, and none identified.** `grep` for `ArgConfig::Runtime*`
  across the op and its three donor headers is empty, matching the readiness sheet. Strict
  `TensorSpec` matching is kept on all 11 `TensorParameter`s. Worth noting for the doc's
  *family heads-up* list: these kernels **look** shape-agnostic in the eltwise sense (they iterate
  tile-by-tile over `N`/`C`/`HtWt` runtime args), yet the legacy factory declared no relaxation — so
  the eltwise-family hint does **not** generalize to `normalization/batch_norm`.
- **Capabilities the op would benefit from:** none missing. `ProgramSpecFactoryConcept` fits both
  factories with no strain.
- **Concept-fit friction:** none. See the Friction section for API-ergonomics notes, which are about
  the run-args helpers rather than the concept.

## Handoff points

**None.** No capitulation, no boundary-rule assumption violation, no kernel-lib gap, no framework gap
that bit, and no pybind surface removed. Specifically:

- **`sem::` / `tensor::` boundary assumption held.** All three donor headers this op calls into take
  `uint32_t cb_id` or a raw L1 address, so `dfb::name`'s `constexpr operator uint32_t()` bridged every
  out-of-op call site and no call site demanded a `sem::` or `tensor::` handle:
  - `ttnn/cpp/ttnn/kernel/dataflow/cb_fill_helpers.hpp` — `fill_cb_with_value(dfb::one, one_u)`
    (`device/kernels/dataflow/reader_running_statistics.cpp:56`).
  - `ttnn/cpp/ttnn/kernel/compute/dest_format_helpers.hpp` — `pack_tile_with_dt`,
    `copy_tile_init_with_dt`, `{add,sub,mul}_tiles_init_with_dt`, `{mul,add,sub}_tiles_to_cb`, all four
    compute kernels.
  - `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp` — fed a
    raw L1 address from a member `dfb.get_write_ptr()` peek; unchanged, per the whitelist's
    transfers-stay-as-is rule.
- **No shared kernels**, in either direction, so no `_metal2` fork was created or reused and nothing
  was written outside this op's directory.
- **No out-of-directory edits of any kind.** The port's whole writeable surface was this op's own
  directory.

## Successes

- **The brief's "Optional tensors — the DFB and the tensor answers are OPPOSITE" section fired
  exactly as written, and it is the single highest-value warning in the whole brief.** My instinct on
  reaching `weight`/`bias` (and `running_mean`/`running_var`) was to treat the absent optional as a
  conditional DFB and reach for
  [Pattern: Conditional / optional DFB bindings](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md)
  for *both* the DFB and the tensor. That is wrong in two independent ways, and the brief named both:
  the legacy host allocates the CB in every config (so dropping it is a functional L1 change), and the
  writers construct `DataflowBuffer dfb_weight(...)` / `dfb_old_mean(...)` **outside** the
  `if constexpr` (`device/kernels/dataflow/writer_batch_norm.cpp:33-37`,
  `device/kernels/dataflow/writer_running_statistics.cpp:29-34`), so a conditional binding would not
  even compile. Splitting the answer — DFBs unconditional, `TensorParameter`s conditional with
  `#ifdef` — is what the port does.
- **[Pattern: Same-FIFO aliasing](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md)
  stopped a real correctness bug.** On the typecast-off path the legacy factory sets
  `writer_updated_m_cb = updated_m_cb` (`device/running_statistics_program_factory.cpp:283` pre-port)
  and `writer_output_cb = output_tensor_cb` (`device/batch_norm_program_factory.cpp:226` pre-port) —
  one CB reached through two kernel-side names. Having just read the *Aliased DFBs* entry for the
  config-flip DFBs, `advanced_options.alias_with` was the reflex reach. The entry's side-by-side
  comparison table ("Buffers: two **distinct** DFBs vs **one** DFB"; "FIFO pointers: independent vs
  **shared**") is what made the distinction land, and it explicitly labels the reach a bug: two
  independent FIFOs at one address, losing the produce-via-one-name / consume-via-the-other coherence
  the kernels depend on. The port instead keeps one DFB, resolves the writer's name **host-side**, and
  uses an `#ifdef`-gated `constexpr` handle alias on compute
  (`device/kernels/compute/running_statistics_sfpu_kernel.cpp:66-77`,
  `device/kernels/compute/batch_norm_sfpu_kernel.cpp:220-229`).
- **The endpoint-assignment procedure's "re-derive, don't transcribe" instruction paid for itself.**
  Re-running the kernel-touch census over all 24 DFBs confirmed the brief's dispositions exactly —
  including the counter-intuitive ones (the *writer* is a PRODUCER on four DFBs in each factory, and
  `temp_1` is a compute self-loop even in the config where it is never touched at runtime). It also
  surfaced a CTA-range under-count in the brief (see Confusion below), which transcription would have
  propagated.
- **The `unpack_modes` hazard section, plus its instruction to go to the headers/source rather than
  guess, resolved what looked like a blocker.** Translating the legacy vector faithfully means setting
  `UnpackMode::UnpackToDest` on ≤16-bit-format DFBs whenever `fp32_dest_acc_en && !any_float32` — and
  the recipe warns the Gen1 validator *rejects* a ≤16-bit format with `UnpackToDest`. Reading
  [`tt_metal/impl/metal2_host_api/program_spec.cpp:1011-1039`](../../../../../../tt_metal/impl/metal2_host_api/program_spec.cpp)
  showed `enable_32_bit_dest` short-circuits that check, so the faithful translation is accepted — and
  since legacy only builds the vector under `fp32_dest_acc_en`, `enable_32_bit_dest` is always true
  where an entry exists. A guess in either direction would have been wrong.
- **[Compiler options](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md)
  caught the `opt_level` trap.** `grep -n opt_level` over both factories returns nothing, which reads
  as "no setting, nothing to carry." The section's flat statement that an absent
  `KernelDescriptor::opt_level` still resolves to `O3` on a `ComputeConfigDescriptor` is the only
  reason both compute `KernelSpec`s carry an explicit
  `opt_level = KernelBuildOptLevel::O3` (`device/running_statistics_program_factory.cpp:577`,
  `device/batch_norm_program_factory.cpp:515`). Nothing else in the toolchain would have flagged it.

## Friction

### Gaps

- **Promoting an `if constexpr` gate to an `#ifdef` orphans the CTA that fed it — the recipe doesn't
  say what to do with it.** [Kernel-side whitelist rule 6](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md)
  and the conditional-binding pattern both say to move the condition from a CTA to a `#define`, but
  neither says whether the CTA should still be declared. In both writers here the `*_has_value` flag
  had **exactly one** reader — the `if constexpr` being promoted — so after promotion nothing reads it
  and the port drops it from the writer's `compile_time_args`
  (`device/kernels/dataflow/writer_batch_norm.cpp`, `writer_running_statistics.cpp`). The compute
  kernels keep theirs, because they consume the same flag through a *runtime* `if`
  (`batch_norm_kernel.cpp:65,93`) or to derive `needs_{mean,var}_typecast`
  (`running_statistics_sfpu_kernel.cpp:78-79`). One sentence — *"drop the CTA if the promoted gate was
  its only reader; keep it where the kernel also reads the value"* — would make this reproducible.
- **The host can resolve a config-dependent DFB name without any kernel-side `#ifdef`, and no doc says
  so.** The *Same-FIFO aliasing* entry presents only the kernel-side handle alias (with the
  path-dependent `#ifdef` variant). But when the kernel binds **only one** of the two aliased names,
  the cleaner fix is entirely host-side: give the kernel one fixed `accessor_name` and point its
  `DFBBinding` at whichever `DFBSpecName` the config selects. Both writers here take that shape —
  `dfb::new_mean` / `dfb::new_var` / `dfb::dst` are bound to the writer-facing DFB or the staging DFB
  as the config dictates (`device/running_statistics_program_factory.cpp:245-247`,
  `device/batch_norm_program_factory.cpp:268-269`) — so the writers need **no** define and **no**
  preprocessor gate at all. Only compute, which binds both names, needs the alias. That host-side
  option is strictly better where it applies (no define to keep in sync, no `#ifdef` in the kernel)
  and deserves to be the *first* branch of the entry's Decision, with the kernel-side alias as the
  fallback for a kernel that genuinely needs both names.
- **`AddRuntimeArgsForNode` forces the RTA name list to be written three times, and only one of the
  three is compiler-checked.** The schema names live on `KernelSpec::runtime_arg_schema`; the live
  values and the idle-core zero-fill values each repeat the same names as string literals in
  `populate_runtime_arguments`. A name typo'd in *only* the zero-fill branch compiles fine and fails at
  `SetProgramRunArgs` — and it fails on a config the tests may not reach, since it is the branch for
  cores outside both work groups. Something like
  `MakeZeroRuntimeArgsForNode(kernel_spec.runtime_arg_schema, node)` (or an
  `AddRuntimeArgsForNode` overload taking a value-less name list) would remove the hazard outright for
  the very common "legacy zero-fills idle cores" shape that the brief's *Work split and placement*
  section already tells porters to preserve.
- **[Locate and confirm the op's tests](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md)
  lists `tests/sweep_framework/sweeps/<family>/<op>/` as a tree to check, but says nothing about what
  running one costs or requires.** Two things bit here. (a) The harness needs a vector-generation step
  and a vector source / result destination, and this op's sweep module does not even import on this
  branch — so "confirm the set with the invoker" produced an agreed baseline that turned out to be
  partly unrunnable, discovered only at verification time. (b) Sweep grids are full Cartesian products:
  this one is 3072 vectors at a 30 s timeout, which no port pass can absorb. A sentence noting that
  sweeps are *usually* out of a port's practical reach, and that the porter should say so at the
  confirmation checkpoint rather than at the end, would have saved a round trip — and would prompt the
  invoker to nominate a subset up front.
- **A legacy-CB include can be load-bearing transitively.** [Rule 1](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md)'s
  sweep says unused CB-API `#include`s should all be gone. Both factories included
  `ttnn/operations/cb_utils.hpp` without using its only symbol (`create_cb`) — but it pulls
  `tt-metalium/host_api.hpp`, which is what was supplying `bfloat16.hpp` for
  `pack_two_bfloat16_into_uint32`. Dropping the legacy-CB include therefore *also* dropped an
  unrelated symbol, so the port adds `#include <tt-metalium/bfloat16.hpp>` explicitly. Worth half a
  sentence in the sweep instruction: check what the dropped include was transitively providing.

### Confusion

- **The brief under-counts Unit 1's CB-index CTA range.** It says "CTAs carrying CB indices (reader
  0–2, writer 2–7, compute 2–13) all become `dfb::` bindings" — but compute CTAs **14 and 15** are
  `writer_updated_m_cb` / `writer_updated_v_cb`, which are CB indices too
  (`device/running_statistics_program_factory.cpp:431-432` pre-port,
  `device/kernels/compute/running_statistics_sfpu_kernel.cpp:61-62` pre-port). Fourteen slots, not
  twelve. Followed the code. Unit 2's equivalent slot (compute CTA 11 = `writer_output_cb`) *is*
  inside the brief's stated range, which is probably why the Unit 1 range was not re-derived.
- **The brief's per-unit "Named CTAs" list reads as "keep these everywhere."** Each unit's list is a
  single flat set covering all three kernels (`weight_has_value`, `bias_has_value`, `fill_eps_fp32`,
  `batch_stat_is_fp32`, …), so the reader has to cross-reference the *Optional tensors* section to
  learn that two of them become defines on the writer and stay CTAs only on compute. Splitting the
  list per kernel would have removed a re-derivation step.
- **The brief describes the config-flip DFBs' endpoint disposition but not their aliasing.** Its
  treatment of `updated_m`/`updated_v` and `output_0` is thorough on 1P+1C ↔ self-loop, which is the
  part the validator would have caught anyway. It does not mention that on the typecast-off path
  `writer_updated_*_cb == updated_*_cb` — one CB under two names — which is the part that is silently
  wrong if mis-modelled. See the Successes entry: the patterns catalog covered it, the brief did not.

## Open items for downstream

- **Shared kernel touches: none.** No kernel in this op is borrowed, lent, or shared between the two
  device-operations; no `_metal2` fork exists, was reused, or was created. Nothing to coordinate and
  nothing to sunset.
- **RTA → CRTA candidates (real dispatch-efficiency win, deliberately not taken).** In both factories
  `momentum`/`eps`, `HtWt`, `n_stride`, `c_stride`, `N` and `C` carry the **same value on every
  working core** — they vary per node only in that idle cores get `0`. Since the DM kernels' loops are
  bounded by `num_tiles` (0 on idle cores → zero iterations) and the compute kernels early-return on
  `num_tiles == 0`, broadcasting the real values to idle cores as CRTAs would be behaviourally inert
  and would shrink the dispatch payload from `num_cores × 6` words to 6. Only `start_tile_id` and
  `num_tiles` (and BatchNorm's `tile_start`) are genuinely per-node. **Not done here:** RTA→CRTA
  changes dispatch semantics, which is outside a no-behaviour-change port.
- **The idle-core zero-fill could go away entirely.** It exists because legacy `SetRuntimeArgs`
  demanded a value on every core the kernel was created on. Metal 2.0 derives placement from
  `WorkUnitSpec::target_nodes`, so a second `WorkUnitSpec` covering only `core_group_1 ∪ core_group_2`
  would let the kernels not run on idle cores at all — no zero-fill, and fewer kernel instances to
  dispatch. That changes which cores the program touches, so it is a follow-up, not port work.
- **Dead RTAs dropped (audit anomaly A1) — now explicit in the named schema.** Both readers' trailing
  `cHt`/`cWt`, both writers' trailing `cHt`/`cWt`, and RunningStatistics' compute `freq`/`counter`
  were emitted but never read. They are simply absent from the ported `runtime_arg_schema`, and the
  idle-core zero-fill was narrowed to match. Legacy sites (pre-port line numbers):
  `running_statistics_program_factory.cpp:85-97,107-121,126`,
  `batch_norm_program_factory.cpp:87-99,109-124`.
- **Dead CTAs in the non-SFPU compute sources (A2) — still declared as the superset.** Both compute
  `KernelSpec`s declare the union of what their two selectable sources read, so
  `running_statistics_kernel.cpp` gets five named CTAs it ignores and `batch_norm_kernel.cpp` gets
  four. That is the correct call today (one `KernelSpec`, two sources). If the factory is ever split
  into a per-source `KernelSpec` — or once NTTP-ified CTAs make `if constexpr` gating sufficient — the
  plain source could declare only its own set.
- **`push_back` with no matching `reserve_back` (A3) carried forward unchanged.**
  `device/kernels/compute/running_statistics_kernel.cpp:57-59` packs into `dfb_out0` and
  `push_back`s without ever reserving, while the SFPU sibling *does* reserve
  (`running_statistics_sfpu_kernel.cpp:95`). A legacy FIFO-protocol asymmetry between the two sources;
  the recipe forbids "balancing" a FIFO during a port, so it is untouched. **Ops team.**
- **Duplicated `extract_shape_dims` / `populate_runtime_arguments` (A4) left in place.** The two
  factories still carry near-identical private copies inside their own
  `namespace { namespace CMAKE_UNIQUE_NAMESPACE { … } }`
  (`device/running_statistics_program_factory.cpp:19-125`,
  `device/batch_norm_program_factory.cpp:19-135`). The single-PR scoping puts both in one diff, which
  makes the duplication conspicuous; de-duplicating it is a separate change. Note that the post-port
  copies have diverged slightly (each now builds its own kernel's named-RTA set), so a shared helper
  would need the name lists parameterized.
- **`RunningStatistics` mutates its inputs in place (A6).**
  `device/kernels/dataflow/writer_running_statistics.cpp` writes the updated statistics back into the
  `running_mean` / `running_var` **input** tensors while the declared output receives a duplicate of
  one stat. Ported as-is: one `TensorParameter` with one `TensorBinding` serves both directions, and
  the accessor is used as both an `async_read` source and an `async_write` destination. Relevant to
  anyone reasoning about aliasing or program-cache behaviour — the op's true output set is wider than
  its declared one.
- **Perf comparison not gathered (invoker's call, recipe default).** Worth flagging for whoever picks
  it up: the readiness sheet's `Pointer patching perf issue? = suspect perf regression (+ fixed latent
  bug)` attaches to the `Buffer*`-patching mechanism, and **this port removes that mechanism
  entirely** — all 11 tensor addresses now ride typed `TensorBinding`s that the framework refreshes on
  its own channel, and no `Buffer*` reaches a runtime arg. If the suspected regression is attributable
  to patching, this commit is plausibly its fix. A before/after measurement on this op would settle it
  cheaply. Nothing perf-anomalous was observed during testing, but the test runs are not a perf
  measurement.
- **Readiness-sheet data quality (carried from the audit).** The `Backdoor custom hash` cell reads
  `yes` for the `RunningStatistics` row, but that device-op has no `to_hash()` and no
  `attribute_values` — confirmed again during the port. Non-gating; **readiness-sheet owner**.
- **Two test-coverage gaps worth closing, neither introduced by the port.**
  1. **No committed test reaches RunningStatistics' typecast path.** It needs a float32 input with
     bfloat16 running statistics; every parametrization uses one dtype throughout, and
     `test_batch_norm_compute_config`'s three `(input_dtype, param_dtype)` pairs omit `(fp32, bf16)`.
     Adding that fourth pair to the existing `@pytest.mark.parametrize` at
     `tests/ttnn/unit_tests/operations/fused/test_batch_norm.py:498` would cover it in one line, and
     would exercise `running_statistics_sfpu_kernel.cpp`'s `maybe_typecast_stat` and the
     `writer_updated_m` / `writer_updated_v` DFBs. Verified manually during this port (see Outcome).
  2. **The sweep module is unimportable**, so it is presumably not running in CI either:
     `tests/sweep_framework/sweeps/normalization/batch_norm/batch_norm.py:20` imports
     `tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs`, which cannot resolve because
     `tests/__init__.py` exists while `tests/ttnn/**` has no `__init__.py`. **Sweep-framework owner.**
     Separately, that sweep's grid includes `check_mean=True, check_var=False`, and its
     `run_batch_norm` hands both running stats straight to `torch.nn.functional.batch_norm`, which
     rejects one-without-the-other — so those vectors would raise `ValueError` even once the import is
     fixed. The committed pytest substitutes zeros/ones for the missing stat
     (`test_batch_norm.py:101-108`); the sweep should do the same.
- **Test-coverage note.** The confirmed pytest baseline is
  `tests/ttnn/unit_tests/operations/fused/test_batch_norm.py` (dtype × shape × optional-tensor ×
  training × eps × momentum sweeps, exercising both prims) and `test_batch_norm_program_cache.py`.
  There are **no C++ gtests** for this op (`grep` for `batch_norm` / `BatchNorm` across
  `tests/**/*.cpp` is empty). All runs were made with `Fast Runtime Mode` **ON** (the conftest default
  here), which skips tests tagged `@pytest.mark.requires_fast_runtime_mode_off`; no test in this set
  carries that marker, so nothing was skipped (the summary reports no `skipped`) — but a validation
  pass with fast-runtime-mode off would be a strictly wider net.
