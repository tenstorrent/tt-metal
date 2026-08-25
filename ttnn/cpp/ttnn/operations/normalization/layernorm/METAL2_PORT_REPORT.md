# Metal 2.0 Port Report — `layernorm`

The op's two program factories were ported in two passes, one factory each, and each pass has its own
report below:

- [Part 1 — `LayerNormMultiCoreProgramFactory`](#part-1--layernormmulticoreprogramfactory)
- [Part 2 — `LayerNormShardedProgramFactory`](#part-2--layernormshardedprogramfactory)

---

# Part 1 — `LayerNormMultiCoreProgramFactory`

## Outcome

**`PORTED`** — `LayerNormMultiCoreProgramFactory` (the interleaved / non-sharded path) and the ten
kernel entry points it can select are on `ProgramSpecFactoryConcept`.

`LayerNormShardedProgramFactory` was **not** ported in this pass and was left on
`ProgramDescriptorFactoryConcept` at the end of it. The two factories share no kernel source, so the
`program_factory_t` variant is valid with one factory on each concept and the op built and ran
throughout. What Part 1 handed to the sharded pass is recorded under
[Open items for downstream](#open-items-for-downstream); Part 2 below is that pass, and both
factories are on `ProgramSpecFactoryConcept` in the final tree.

**No-regression result.** The confirmed test set gives **2236 passed, 22 skipped** both before and
after the port, with an identical pass/skip split per file. One behavior change falls outside that
test set: the sanctioned pybind removal. A second one was found by measurement and then fixed inside
this port: a Welford configuration with a 32-bit float input and `fp32_dest_acc_en=False` failed
while the spec was built, and now matches the pre-port output in every digit. Both are characterized
precisely below.

## Provenance

- **Recipe docs (this port):** `93fb1b95d03 2026-08-24 docs(metal_2.0): a run in flight freezes the kernel sources`
- **Audit docs (inherited):** `93fb1b95d03 2026-08-24 docs(metal_2.0): a run in flight freezes the kernel sources`

## TTNN ProgramFactory

### Concept realized

`ProgramSpecFactoryConcept`, as the audit chose. The ported-from factory has no
`override_runtime_arguments`, so the framework owns the cache-hit binding refresh and the factory
implements a single method, `create_program_artifacts`. Nothing about the concept changed during the
port.

### Device-op-class edits

- **Pybind entry points removed:** `create_descriptor` on `LayerNormMultiCoreProgramFactory`
  ([layernorm_nanobind.cpp:320-346](layernorm_nanobind.cpp#L320-L346) pre-port). See the Handoff
  points entry below — this one has live downstream consumers.
- **`default_core_range` kept.** The recipe's pybind exception is narrow: it removes bindings whose
  symbol vanishes. `default_core_range` still exists (the factory body calls it to pick the
  work-split grid), so its nanobind compiles and stays.
- **`LayerNormShardedProgramFactory::create_descriptor` and its nanobind were untouched by this
  part**, since Part 1 covers only the multi-core factory. Part 2 ported that factory and removed
  both, so neither survives in the final tree.
- **Custom `compute_program_hash`:** none. The device operation defines no override and no backdoor
  `attribute_values` / `to_hash`; the `compute_program_hash` nanobind at
  [layernorm_nanobind.cpp:252-263](layernorm_nanobind.cpp#L252-L263) forwards to the framework
  default. Nothing to preserve, nothing touched.

### Open items

- **Relaxation candidates:** none identified. Every `TensorParameter` matches strictly, which is
  what the audit recorded (`TensorParameter relaxation = none`). No kernel in this factory reads
  `ArgConfig::Runtime*`.
- **One `TT_FATAL` added inside the factory body**, guarding that
  `to_compute_hardware_config` returned the Gen1 alternative before the port reads Gen1-only fields
  off it. It cannot fire on Wormhole or Blackhole; it replaces what would otherwise be a
  `std::bad_variant_access` on a Gen2 device. This is the one count delta in the TT_FATAL check
  (11 → 12 in the factory), and it is an addition, not a loss.
- **`Wt` and `eps` are per-node runtime args whose value is identical on every node**, so they are
  really common runtime args. The port keeps them as RTAs; converting changes dispatch semantics and
  is a separate cleanup.

## Handoff points

### Removed pybind surface: `LayerNormMultiCoreProgramFactory.create_descriptor`

*Tagged: API surface — removed entry point. This one has live consumers, so it needs an owner.*

- **File / function:** [layernorm_nanobind.cpp:320-346](layernorm_nanobind.cpp#L320-L346)
  (pre-port line numbers), `LayerNormMultiCoreProgramFactory.create_descriptor`.
- **What it was for:** it let Python drive the interleaved factory directly and receive a
  `ProgramDescriptor`, including a fourth `core_range_set` argument that the C++ op never sets.
- **Why it had to go:** the `create_descriptor` symbol no longer exists, so the block naming it stops
  compiling. It could not be pointed at `create_program_artifacts` either: that returns a
  `ProgramArtifacts`, and neither `ProgramArtifacts` nor `ProgramSpec` is bound to Python anywhere in
  the repo (`ProgramDescriptor` is bound in four files, these in none), so exposing the new method
  would first mean binding the whole Metal 2.0 type family. ttnn_factory's exception 2, which names
  layernorm's `core_range_set` explicitly, prescribes deleting the binding *and* dropping the
  parameter. **Only the first half is followed here** — see the next bullet and the Friction entry on
  that exception below.
- **`core_range_set` is kept**, as
  `create_program_artifacts(attributes, tensor_args, tensor_return_value, core_range_set = std::nullopt)`,
  matching the sharded factory. Nothing technically prevented keeping it: `ProgramSpecFactoryConcept`
  tests only for the presence of `create_program_artifacts`, and the adapter calls it with exactly
  three arguments, so a defaulted fourth parameter compiles and is simply never supplied by the
  framework path. It is retained on the invoker's decision, so the behavior the parameter carries
  stays in the tree even with no caller reaching it today. Unlike the sharded factory, where the
  parameter only *validates* a containment property, here it genuinely *selects* the work-split grid
  ([device/layernorm_op_multi_core.cpp:277-286](device/layernorm_op_multi_core.cpp#L277-L286)):
  supplying it restricts which cores the tile rows are distributed over, and omitting it takes
  `default_core_range(device)`, the device's whole compute grid. Keeping it therefore preserves the
  pre-port behavior exactly rather than merely preserving a check.
- **What the binding removal costs.** The C++ capability is intact per the bullet above, and every
  C++ call already omitted the argument, so nothing changed for them. What is gone is the *Python
  route* to it: the one Python consumer passed a genuinely restricted range on the interleaved path
  ([_utils.py:104](../../../../../../models/experimental/ops/descriptors/normalization/_utils.py#L104),
  `cr_arg = None if input_tensor.is_sharded() else core_range_set`), and with no binding it can no
  longer reach the factory at all. The removal is also **not** clean from the caller's side:
  `layer_norm.py` and `rms_norm.py` still take a `core_range_set` argument and forward it, and
  `op_descriptor.py` still documents it as live behavior, so a caller who passes one is not told it
  is unsupported — it fails with `AttributeError` further down. Because the parameter survives in
  C++, restoring the capability is now a binding question rather than a factory one.
- **Known downstream consumers** (all outside the porter's writeable surface; neither consumer was
  itself edited, though the test directory's `conftest.py` was — see *Measured impact* below):
  - [models/experimental/ops/descriptors/normalization/_utils.py:109](../../../../../../models/experimental/ops/descriptors/normalization/_utils.py#L109)
    — `factory.create_descriptor(operation_params, tensor_args, out, cr_arg)`, reached through
    `select_program_factory`, so it breaks for interleaved inputs and keeps working for sharded
    ones. The same file also calls `default_core_range` at
    [:44](../../../../../../models/experimental/ops/descriptors/normalization/_utils.py#L44), which
    still exists.
  - [tests/ttnn/unit_tests/operations/fused/parallel_sequential/test_parallel_sequential.py:198](../../../../../../tests/ttnn/unit_tests/operations/fused/parallel_sequential/test_parallel_sequential.py#L198)
    — does not call `create_descriptor` itself; it drives the helper above with interleaved tensors
    and an explicit `core_range_set`, and fails through it.
- **Measured impact.** Immediately after this pass, that file gave **60 failed, 29 passed** — every
  failure the same line, `AttributeError: 'LayerNormMultiCoreProgramFactory' object has no attribute
  'create_descriptor'`, with no numerics failure and no second failure mode. The 29 passes were the
  sharded-input branches, still reaching the sharded factory's `create_descriptor`. Once the sharded
  factory was ported too those 29 went the same way, taking the whole
  `parallel_sequential/` directory to **90 failed, 178 passed** (61 from the interleaved factory, 29
  from the sharded one).
- **These tests are now skipped rather than failing**, on the invoker's instruction, by an autouse
  fixture in
  [tests/ttnn/unit_tests/operations/fused/parallel_sequential/conftest.py](../../../../../../tests/ttnn/unit_tests/operations/fused/parallel_sequential/conftest.py).
  It stands a `pytest.skip` in for the absent `create_descriptor` on both factory classes, so a test
  is skipped only if it actually reaches the call, and the fixture goes inert the moment a factory
  exposes the method again. The directory now reports **178 passed, 90 skipped**. **This is the one
  place where green CI stops meaning "works"**: the fused-descriptor framework still has no working
  layernorm or rms_norm path, and it needs a Metal 2.0 route (or a branch that stops going through
  `create_descriptor`) before any of it works again. The skip hides that from CI, so the owner
  should be told directly rather than discovering it from a passing run.

### Welford with a 32-bit float input and `fp32_dest_acc_en=False`

*Tagged: op behavior. Fixed in this port for the case the port broke; a pre-existing bug in a
neighbouring case is reported here and deliberately left alone. Measured on hardware, not inferred.*

**What the port broke.** With Welford on, a 32-bit float input in tile layout, and a
`compute_kernel_config` carrying `fp32_dest_acc_en=False`, the first ported revision failed while
the spec was built. Both branches of Metal 2.0's unpack-mode rule were reachable, one per
configuration:

| case | input | `fp32_dest_acc_en` | residual | pre-port | first ported revision |
|---|---|---|---|---|---|
| A | float32 | False | no | ran | `TT_FATAL` [program_spec.cpp:1036](../../../../../../tt_metal/impl/metal2_host_api/program_spec.cpp#L1036) `!is_32bit_element_format(fmt)` |
| B | float32 | False | yes | ran | `TT_FATAL` [program_spec.cpp:1044](../../../../../../tt_metal/impl/metal2_host_api/program_spec.cpp#L1044) `is_gen2` |
| C | float32 | True | no | ran | ran |
| D | bfloat16 | False | no | ran | ran |

Measured with a four-case probe against the pre-port revision and against the ported code, on a
build with **no** forced-validation scaffolding, so this is the shipped configuration. The residual
is *not* required for the failure: case A has none.

**Why it happened.** `welford_fp32_alias` gated the alias on the *input* being 32-bit float but
never on `fp32_dest_acc_en`, the switch that sets the Dest register's width. With the switch off,
the alias is marked to unpack straight into a 16-bit Dest: in case A it carries the input's 32-bit
format, which cannot fit; in case B it carries the 16-bit intermediate format, where the bypass buys
nothing. The port copied the marking across unchanged, so the rule is what is new, not the setting.

**The fix applied here.** `&& fp32_dest_acc_en` was added to the alias condition
([device/layernorm_op_multi_core.cpp:451-456](device/layernorm_op_multi_core.cpp#L451-L456)). With
the switch off there is no alias, and the compute kernel reads the buffer it would have aliased,
which is the path it already takes whenever the alias is absent
([device/kernels/compute/layernorm_welford.cpp:104-109](device/kernels/compute/layernorm_welford.cpp#L104-L109)).
Measured against the PyTorch reference:

| case | pre-port | first ported revision | with the guard |
|---|---|---|---|
| A (no residual) | PCC 0.0172, max err `inf` | `TT_FATAL` | PCC 0.0056, max err `inf` |
| B (residual) | PCC 0.999970288388242 | `TT_FATAL` | PCC 0.999970288388242 |

Case B is restored to the pre-port result in every digit, which is the whole point: it produced
correct output before, and it does again.

**Case A is a pre-existing bug and is left exactly as it was.** It returned garbage before the port
and returns garbage now, because a 32-bit float Welford input cannot reach the compute unit intact
when Dest is 16 bits wide, by either route. The first ported revision happened to turn that silent
wrong answer into a loud failure, but a port is not the place to change behavior, so the guard
restores the pre-port behavior here as well rather than preserving the accidental improvement.

**The pre-port numerics were measured too.** Running the same probe against the pre-port revision
gives:

| case | pre-port PCC | pre-port max abs error |
|---|---|---|
| A | **0.0172** | **`inf`** |
| B | 0.999970288388242 | 4.354e-02 |
| C | 0.9999996686705196 | 4.253e-03 |
| D | 0.9999893967933305 | 2.344e-02 |

Cases B, C and D all match the ported factory to every digit, which is the no-change result the port
promises. Case A matches pre-port in being wrong.

**What the owner should follow up on.** Case A's combination should be rejected outright in the op's
validation, since no binding arrangement makes it produce correct numbers, and today it returns
`inf` without complaint. That belongs in its own change.
[test_layer_norm_ulp.py:20](../../../../../../tests/ttnn/nightly/unit_tests/operations/fused/test_layer_norm_ulp.py#L20)
already asserts in a comment that "device enforces this for FP32 inputs" — no such enforcement
exists, so that comment describes intended behavior that was never implemented. Adding it would make
the comment true and would close this out.

**The sharded factory already carries the same guard**
([device/layernorm_op_multi_core_sharded.cpp:323](device/layernorm_op_multi_core_sharded.cpp#L323)),
so the two factories now agree.

### `get_pointer_to_cb_data` keeps a CB-vocabulary name after the port

*Tagged: kernel-lib naming, non-blocking.*

The two Welford compute kernels reach the reciprocal LUT through
`norm::kernel_util::compute::memory::get_pointer_to_cb_data<T>(uint32_t, ...)`
([layernorm_welford.cpp:133](device/kernels/compute/layernorm_welford.cpp#L133),
[layernorm_large_tensor_welford.cpp:430](device/kernels/compute/layernorm_large_tensor_welford.cpp#L430)).
`dfb::reciprocals` flows straight into its `uint32_t` parameter, so nothing is blocked and the donor
needed no change — this is audit Question 1's shape, answered as GREEN. But the *name* keeps the CB
vocabulary in two otherwise CB-free kernels, and the helper lives in
`ttnn/cpp/ttnn/operations/normalization/kernel_util/`, outside this op's directory. It is the only
residue left by the port's `cb` → `dfb` name sweep. The in-family owner may want to rename it (the
whitelist's own mapping for this call is `DataflowBuffer::get_tile_address`).

## Successes

- **Forcing the legality checks caught a real defect before any test could.** The `unpack_modes`
  entry for the large-tensor accumulator was gated on `float32_reduction` alone, mirroring legacy
  ([device/layernorm_op_multi_core.cpp:517-519](device/layernorm_op_multi_core.cpp#L517-L519)
  pre-port), which wrote into a vector slot for a buffer index that may never have been allocated.
  Metal 2.0 rejected it immediately: `Kernel 'compute' unpack_modes entry references DFB
  'accumulate', which the kernel does not bind`. The recipe's parenthetical — *"a conditionally-bound
  DFB's `unpack_modes` entry must be gated on the same condition as its binding"* — is exactly the
  fix, and the error message named the buffer, so the diagnosis took one read. Without the forced
  checks this would have been a silently-wrong precision setting on the Welford path.
- **The self-loop versus 1P+1C distinction held up under re-derivation.** The brief listed how the
  interleaved factory's buffers should be bound, and re-counting the touching kernels per selected
  kernel source (rather
  than per buffer) confirmed all of them and additionally surfaced the three that *move* with the
  configuration: `IN`'s producer shifts from the reader to compute on the row-major path, `OUT`
  becomes a compute self-loop there, and `X_WELFORD` is a 1P+1C when the pre-add is not fused and a
  self-loop when it is. The recipe's insistence on mapping roles *per selected source path* is what
  made those visible; a per-buffer reading would have mis-bound three of the twenty-one buffers.
  **No multi-binding flag is set anywhere in this factory**, and no buffer is both self-looped and
  multi-bound.
- **The `#ifdef`-gated conditional-binding pattern covered every case the kernels presented**,
  including the two the catalog calls out as traps: file-scope names resolved from a ternary
  (`dfb_im_or_out`, `dfb_outg`) and CTA gates that had to be promoted to defines. Three promotions
  were needed and all three were invisible from the brief: `do_gamma` / `do_beta` (they reach
  `GAMMA`, `BETA` and `FUSION` from `if constexpr` branches in every compute kernel), `use_welford`
  in the reader (it gates `SCALER`), and `fuse_pre_add` in the two Welford compute kernels, which
  today receive no `FUSE_PRE_ADD` define at all. The catalog's *"watch the emission target"* note is
  precisely this case.

## Friction

### Gaps

- **ttnn_factory's exception 2 gives a reason that does not hold, using this op as its example.**
  [ttnn_factory.md](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/ttnn_factory.md)
  says *"The fixed `create_program_artifacts` signature (`attributes`, `tensor_args`,
  `tensor_return_value`) cannot carry it"*, naming layernorm's `core_range_set` as the case. It can
  carry it: `ProgramSpecFactoryConcept` tests only for the presence of `create_program_artifacts`,
  and the adapter calls it with exactly three arguments, so a defaulted fourth parameter compiles and
  goes unnoticed. **Both layernorm factories now keep the parameter on that basis**, so the
  exception's prescription was not followed here either: only its pybind half was. Two things are
  worth correcting in the doc. First the reason, because a porter who applies it to a parameter that
  *does* still have a C++ caller would drop something a caller depends on. Second the prescription
  itself, which conflates two separable decisions — deleting a binding whose symbol has vanished is
  forced, while dropping the parameter that binding fed is a judgment call about whether the behavior
  is worth keeping without a caller. Layernorm answered that second question the other way for both
  factories: the parameter carries behavior (grid selection in the interleaved factory, containment
  validation in the sharded one) that is cheaper to keep than to reconstruct later.
- **A fully-skipped test session and a stale build look identical when proving
  `METAL2_CHECKS_FORCED`.** The recipe says to rebuild, run one test, and grep the log. The op I
  first picked (`tests/ttnn/unit_tests/operations/fused/test_distributed_layernorm.py`) skipped all
  72 cases on a single-device bench, so the grep returned zero markers with exit code 0 — which
  reads exactly like a stale object file. Checking the pass/skip line first turns a ten-minute
  detour into one command. A sentence telling the porter to confirm the probe test actually *ran*
  cases would prevent it.
- **The two markers carry the same string, so "two markers present" cannot actually be checked.**
  The recipe asks for a marker in each file the grep named and says two markers prove both
  translation units are fresh, but with identical text a grep can only report a count, and that
  count scales with the number of programs the run builds (I measured 278 across 142 tests). Making
  the strings distinguishable — `METAL2_CHECKS_FORCED program_spec` and `... program_run_args` —
  turns it back into the yes/no it is meant to be, and both then appear in equal counts. Worth
  putting in the recipe's snippet.
- **`unpack_modes()` does not accept `KernelSpec::hw_config`.** The Hardware-configuration section
  shows `std::get<ComputeGen1Config>(compute_hw).unpack_modes = …` against the *helper's return
  value*, and the common-field accessor takes a `ComputeHardwareConfig&`. But `KernelSpec::hw_config`
  is the outer `std::variant<DataMovementHardwareConfig, ComputeHardwareConfig>`, so reaching the
  table on a built `KernelSpec` needs
  `unpack_modes(std::get<ComputeHardwareConfig>(kernel.hw_config))`. The compiler says only "no
  matching function for call to 'unpack_modes'", which does not point at the extra variant layer.
- **No guidance for a runtime-arg slot whose meaning differs across the sources one factory can
  select.** The recipe says to pick argument names that match the variables they were going to be
  assigned to, which assumes one kernel per name. Here reader slot 3 is `start_tile_row` in two of
  the four readers and `tile_offset` in the other two, and the writer's slot 3 likewise. Naming
  after the host's own variable (`reader_start` / `writer_start`) was the only choice accurate for
  every selected source; a line sanctioning that tie-breaker would save the deliberation.

### Confusion

- **"Declare the conditional-side binding unconditionally" is easy to miss when the asymmetry is
  between two *kernels* rather than between two branches of one kernel.** The migration guide frames
  it as a reader that always produces versus a compute kernel that only conditionally consumes. The
  shape here is different: `IN_RM` and `OUT_RM` are allocated whenever the input is row-major, but
  only two of the four compute kernels carry the tilize / untilize blocks that touch them, so in the
  other configurations no kernel touches the buffer at all, rather than exactly one. The same sentence resolves
  it, but it took a while to recognise that it applied.
- **The `Group<T>` / `Table<K,V>` heads-up says these are Metal 2.0 types but the examples use them
  unqualified.** They live in `tt::tt_metal::experimental`, so a factory that aliases the namespace
  (`namespace m2 = …`, as the sibling port does) has to write `m2::Group<…>`. Mentioning the
  namespace once in the type-system heads-up would help.

## Open items for downstream

### The sharded factory was handed over, and Part 2 picked it up

*Closed. Kept as the record of what Part 1 handed across; Part 2 below is the outcome.*

At the end of Part 1, `LayerNormShardedProgramFactory` was still on
`ProgramDescriptorFactoryConcept`. It was left for a later pass because this op is broad rather than
deep and one factory is the recipe's atomic unit, not because anything blocked it. Part 2 ported it,
so both factories are on `ProgramSpecFactoryConcept` today and the op has no `create_descriptor`
anywhere.

**An earlier revision of this report said audit Question 2 blocked it. That is withdrawn.** The
audit has since retracted Question 2: a buffer-backed CB does not take a per-core SRAM allocation at
all, so `c_17`'s address is the output tensor's own address and the storage half of its core range
reserves nothing. The port can bind `c_17` on the worker nodes alone and needs nothing on the
storage nodes. What the finding became instead is a **binding reclassification**: the sharded
`output`, reached through `c_17` under POST without `skip_write_back`, is **Case 2** rather than
clean — the writer takes only that binding's base address and does its own remote NOC writes, so it
needed binding as a `TensorParameter` with the write-back arithmetic left alone. Part 2 did that
([device/layernorm_op_multi_core_sharded.cpp:375](device/layernorm_op_multi_core_sharded.cpp#L375)).

Everything else the sharded factory needed was already scoped by the audit and brief: five
`allow_instance_multi_binding` buffers, five runtime-vararg sites with the `get_arg_addr` pointer
wrinkle, three semaphores, and the borrowed-memory list. See Part 2 for how each landed.

### Shared kernel touches

- **Reused an existing `_metal2` fork (rung 1):**
  [ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar_metal2.hpp](../../../kernel/dataflow/generate_bcast_scalar_metal2.hpp),
  whose `generate_bcast_col_scalar(DataflowBuffer&, uint32_t)` replaces the legacy
  `generate_bcast_scalar.hpp` version taking a `CircularBuffer`. All four ported readers now bind
  the fork. **No new file was created and no pointer comment was added** to the legacy original —
  the fork already existed, so this is rung 1, where the original is not the porter's to annotate.
  Its parameter is a non-const reference, so every call site passes a named local
  (`DataflowBuffer dfb_eps(dfb::eps); generate_bcast_col_scalar(dfb_eps, eps);`).
  **Remaining unmigrated consumers of the legacy `generate_bcast_scalar.hpp`:** this op's *sharded*
  writers, [writer_unary_sharded_ln.cpp:71](device/kernels/dataflow/writer_unary_sharded_ln.cpp#L71)
  and [writer_unary_sharded_ln_rm_gb.cpp:74](device/kernels/dataflow/writer_unary_sharded_ln_rm_gb.cpp#L74),
  plus whatever binds it outside this op. The legacy copy cannot be sunset until those migrate.
- **No fork was created and no kernel outside the op directory was modified.** All ten sources this
  factory binds live in the op's own directory and have no consumer outside it, so there is no
  borrowed or lent kernel and no sunset list to open.
- **Two in-op kernel headers are shared with the sharded factory** and were touched only in ways
  that cannot reach it. `layernorm_dataflow_utils.h` had four helpers renamed
  (`read_block_to_cb` → `read_block_to_dfb`, and the three row-major siblings) as part of the
  `cb` → `dfb` sweep; the sharded kernels use only `compute_single_stage_noc_addrs` /
  `compute_two_stage_noc_addrs` from that header and call none of the renamed four.
  `layernorm_compute_utils.h` is interleaved-only. Neither header needed a functional change: both
  were already parameterized on `DataflowBuffer&` and `TensorAccessor` templates with no buffer
  indices and no argument reads.

### Findings the port carried forward unchanged

These are defects and dead code the port preserved rather than fixed, per the porting invariant.
Each is an ops-team call.

1. **Two configurations hang or produce garbage today, and nothing rejects them.** Both involve a
   row-major input, and the op's own tests skip both
   ([tests/ttnn/nightly/unit_tests/operations/fused/test_layernorm.py:384-386](../../../../../../tests/ttnn/nightly/unit_tests/operations/fused/test_layernorm.py#L384-L386)):
   - `use_welford && input_is_row_major` — the factory emits `TILIZE_IN` / `UNTILIZE_OUT` and
     allocates `IN_RM` / `OUT_RM`, but neither Welford compute kernel contains a tilize or untilize
     block, so nothing fills `IN` and nothing fills `OUT_RM`. This generalizes the audit's misc
     anomaly 1, which covers only the large-tensor half; the non-large-tensor half behaves the same.
   - `input_is_row_major && use_row_major_kernel && !large_tensor` —
     [reader_unary_interleaved_ln_rm_gb.cpp](device/kernels/dataflow/reader_unary_interleaved_ln_rm_gb.cpp)
     is selected, and it has no `TILIZE_IN` branch, so it reads a row-major tensor as tiles and
     never fills `IN_RM` while compute waits on it.

   The port keeps both behaving exactly as they do now: `IN_RM` and `OUT_RM` declare both endpoints
   under the row-major gate regardless of which kernels were selected, which keeps the spec valid
   and the SRAM footprint byte-identical. A `TT_FATAL` on either combination would be the real fix,
   and it belongs to the op owner.
2. **A `#define` typo makes a reconfigure run on a path it was never meant to.**
   [layernorm.cpp:283](device/kernels/compute/layernorm.cpp#L283) reads
   `#if defined RMSNORM and not defined FUSED_PRE_ADD` — `FUSED_PRE_ADD`, not `FUSE_PRE_ADD`. The
   host never defines that spelling, so the guarded `reconfig_data_format` is compiled into *every*
   RMSNORM build, including the fused-pre-add one it was written to exclude. Carried forward
   verbatim, typo included.
3. **Dead plumbing dropped** (each read by no kernel this factory binds; dropping is zero functional
   change, and the values are listed here so the owner can confirm):
   - Reader RTA slot 4, `packed_one_value`, with its `bfloat16(1)` /
     `pack_two_bfloat16_into_uint32` computation
     ([device/layernorm_op_multi_core.cpp:552-553, 592](device/layernorm_op_multi_core.cpp#L552-L553)
     pre-port). The reader kernels' own header comments already labelled it *"legacy; unused, scaler
     is generated in-kernel"*.
   - The reader's trailing size CTA in three of its four branches
     ([:365-373](device/layernorm_op_multi_core.cpp#L365-L373) pre-port): only the
     `input_is_row_major` branch (`a.element_size()`) is ever read. The `gamma_stick_size`,
     `beta_stick_size` and `tile_size` branches fed nothing.
   - Compute CTA slot 7 on the Welford path, `rms_norm`
     ([:504](device/layernorm_op_multi_core.cpp#L504) pre-port): neither Welford compute kernel
     reads index 7, and the value is a compile-time `false` on that path anyway.
4. **`FUSION` is allocated but untouched under `large_tensor && use_welford && (gamma || beta)`.**
   [layernorm_large_tensor_welford.cpp](device/kernels/compute/layernorm_large_tensor_welford.cpp)
   declares the handle and never uses it, routing gamma/beta staging through `XMM` instead. The port
   keeps the allocation and self-loops the buffer on the compute kernel so the SRAM footprint is
   unchanged; dropping the allocation in that configuration is an owner decision, not a port one.
5. **A reconfigure named a buffer index that may not have been allocated.** Pre-port,
   [layernorm.cpp:287](device/kernels/compute/layernorm.cpp#L287) called
   `reconfig_data_format_srca(cb_fusion, cb_xmm)` under `RMSNORM && !FUSE_PRE_ADD` regardless of
   gamma/beta, so with neither present it read the format descriptor of an unallocated index. The
   ported line is gated on `FUSE_GAMMA || FUSE_BETA`, which is behavior-preserving: the call is a
   restore paired with the `reconfig_data_format_srca(cb_xmm, cb_fusion)` at the end of the same
   loop body, and that one runs only when gamma or beta is present, so with neither there is
   nothing to restore and the preceding `reconfig_data_format(cb_xmm, cb_ex2pe)` has already left
   SrcA where the call would have put it.

### Doc-evolution candidates

- **The `cb` → `dfb` name sweep needs a rule for a partially-ported op.** The self-audit's sweep is
  specified over the whole op directory and expects zero hits, but when only one of two factories is
  ported the other factory's kernels legitimately still speak CB, so a directory-wide grep is
  guaranteed to fire. Scoping the sweep to the ported file set (and saying so) would make the check
  runnable on the multi-factory ports the recipe otherwise encourages. Reported here as *hits /
  files scanned*: **0 / 13** over the ported files after the sweep, with the two
  `get_pointer_to_cb_data` call sites excluded as an out-of-directory helper name (Handoff points
  above).
- **A named-CTA carrying a boolean that gates a conditional buffer is always a define, never a named
  arg.** Whitelist rule 2's note covers a named CTA carrying a buffer *index*; the sibling case is a
  named CTA carrying the *flag* that decides whether that index exists (`welford_fp32_alias`,
  `welford_state_fp32_alias`, `use_welford` here). It resolves the same way — promote to
  `compiler_options.defines` — but the rule as written does not quite reach it.

### Test coverage notes

- Nothing in the confirmed test set exercises `use_welford && input_is_row_major` or
  `input_is_row_major && use_row_major_kernel`; both are explicitly skipped. Those are the two
  broken configurations in finding 1, so the skips are currently the only thing keeping them out of
  CI. If either is ever fixed, the skip should go with the fix.
- **All four alias groups are exercised.** `ttnn.layer_norm` resolves its compute config with
  `fp32_acc = true` by default
  ([layernorm.cpp:16-18](layernorm.cpp#L16-L18)), so the intermediate format is `Float32` in the
  ordinary case and `test_large_layer_norm_with_weight_bias_and_residual_input[use_welford=True]`
  reaches `welford_state_fp32_alias` (`EX` ↔ `EX_WELFORD`, `EX2` ↔ `EX2_WELFORD`) while the
  float32-input Welford cases reach `welford_fp32_alias` on both its fused and non-fused shapes.
- The plan's `unpack_modes` watch item **fires, and is wider than the plan predicted.** It is written
  up as a Handoff point above (*Welford with a 32-bit float input and `fp32_dest_acc_en=False`*); an
  earlier revision of this section claimed it "did not fire" and could not be reached, which was
  wrong on both counts. It was measured, not reasoned about.

---

# Part 2 — `LayerNormShardedProgramFactory`

## Outcome

**`PORTED`** — `LayerNormShardedProgramFactory` and the thirteen kernel entry points it can select
across its three `distributed_norm_stage` values are on `ProgramSpecFactoryConcept`. With Part 1
already done, **both** of the op's factories are now on the spec concept and the op has no
`create_descriptor` left anywhere.

**No-regression result.** The confirmed test set gives **2631 passed, 664 skipped, 10 xfailed** after
the port, against **2667 passed, 628 skipped, 10 xfailed** before it. The whole difference is the 36
parameterizations of the two tests that drive the removed pybind, which are now skipped with the reason
stated at the test; every other test's outcome is unchanged. C++ side: the 17 normalization gtests and
the eager `test_layernorm_op` pass, identical to the baseline.

## Provenance

- **Recipe docs (this port):** `93fb1b95d03 2026-08-24 docs(metal_2.0): a run in flight freezes the kernel sources`
- **Audit docs (inherited):** `93fb1b95d03 2026-08-24 docs(metal_2.0): a run in flight freezes the kernel sources`

## TTNN ProgramFactory

### Concept realized

`ProgramSpecFactoryConcept`, as the audit chose. The ported-from factory has no
`override_runtime_arguments`, so the framework owns the cache-hit binding refresh and the factory
implements a single method, `create_program_artifacts`. Nothing about the concept changed during the
port.

### Device-op-class edits

- **`core_range_set` kept**, as the invoker decided when asked audit Question 2. The signature is
  `create_program_artifacts(attributes, tensor_args, tensor_return_value, core_range_set = std::nullopt)`.
  Part 1 keeps its equivalent parameter too, so both factories now carry it; what differs is what the
  parameter does. Here it only *validates* that the shard grid's multicast bounding box lies inside
  the given range, and keeping the parameter keeps that validation in the tree, whereas in Part 1 it
  genuinely chooses the work-split grid. The cost is stated plainly: with
  the pybind gone the parameter has no caller at all, so the validation is dead code until something
  calls the factory with a range again.
- **Pybind entry point removed:** `create_descriptor` on `LayerNormShardedProgramFactory`
  (the `create_descriptor` static method that stood on the class registration now at
  [layernorm_nanobind.cpp:339](layernorm_nanobind.cpp#L339)). See the Handoff points entry below.
- **The `nb::class_<LayerNormShardedProgramFactory>` registration stays**, now with no methods.
  `select_program_factory` returns that type to Python, and nanobind needs the type registered to
  convert the variant; the factory's own output (`ProgramArtifacts` / `ProgramSpec`) is not bound to
  Python anywhere in the repo, so there is nothing to expose in its place.
- **Custom `compute_program_hash`:** none, same as Part 1. Nothing to preserve, nothing touched.
- **`<tt-metalium/program_descriptors.hpp>` dropped** from
  [device/layernorm_device_operation.hpp](device/layernorm_device_operation.hpp): with both factories
  converted, the op no longer names a descriptor type.

### Open items

- **Relaxation candidates:** none identified, matching the audit (`TensorParameter relaxation = none`).
  No kernel in this factory reads `ArgConfig::Runtime*`.
- **Two `TT_FATAL`s added inside the new spec builders**, both guarding an invariant of the new code
  rather than tightening an op-level check: `add_dfb` rejects a zero entry size (it divides the buffer's
  byte size by it), and `add_tensor_parameter_specs`' callers rely on the optional tensors being present
  under the same conditions the buffers are declared. These are the only two count deltas in the
  TT_FATAL census (5 → 7 in `sharded_layernorm_factory_helpers.cpp`), and both are additions. Every
  legacy guard is still in place, including the two the `core_range_set` validation carries.
- **`eps_u` and the packed `cinv` / `winv` scalars are per-node runtime args whose value is identical
  on every node** (except `cinv` on a two-stage reduce, which genuinely varies), so the other two are
  really common runtime args. The port keeps them as RTAs; converting changes dispatch semantics and is
  a separate cleanup.

## Handoff points

### Removed pybind surface: `LayerNormShardedProgramFactory.create_descriptor`

*Tagged: API surface — removed entry point.*

- **File / function:** `LayerNormShardedProgramFactory.create_descriptor`, whose class registration survives at
  [layernorm_nanobind.cpp:339](layernorm_nanobind.cpp#L339).
- **What it was for:** it let Python drive the sharded factory directly and receive a
  `ProgramDescriptor`, including a fourth `core_range_set` argument used only to validate the shard
  grid's multicast footprint.
- **Why it had to go:** the `create_descriptor` symbol no longer exists. It could not be pointed at
  `create_program_artifacts` either, for the same reason Part 1 recorded: neither `ProgramArtifacts`
  nor `ProgramSpec` is bound to Python anywhere in the repo.
- **It had live callers in this op's own tests, which is a revision to the audit.** Audit Question 2
  asked whether `core_range_set` survives the port, on the understanding that the parameter was
  reachable only through the pybind and had no C++ caller. It also had two *test* callers:
  `test_layer_norm_sharded_non_rectangular_grid_rejects_excluded_hole_cores`
  ([tests/ttnn/unit_tests/operations/fused/test_layer_norm_sharded.py:659](../../../../../../tests/ttnn/unit_tests/operations/fused/test_layer_norm_sharded.py#L659))
  and its RMSNorm sibling
  ([tests/ttnn/unit_tests/operations/fused/test_rms_norm_sharded.py:441](../../../../../../tests/ttnn/unit_tests/operations/fused/test_rms_norm_sharded.py#L441)),
  36 parameterizations between them. Each drives `create_descriptor` twice: once with the bare shard
  grid, expecting the hole-rejection `RuntimeError`, and once with the bounding box, then walking
  `descriptor.kernels` and `descriptor.semaphores` to assert that placement covers exactly the
  multicast bounding box and nothing outside it.
  **Neither half survives.** The rejection needs a Python entry point that reaches the validation, and
  the placement assertion needs a Python-visible program spec; there is no `ProgramSpec` or
  `ProgramArtifacts` binding anywhere in the repo. On the invoker's decision both tests are
  `pytest.mark.skip`ped with that reason stated in full at the test, so the bodies stay intact and can
  be revived the moment either surface exists. **This is the port's one coverage loss**, and it is a
  loss of *validation* coverage rather than numerics: the underlying check is still in the factory, and
  the placement it asserts is separately exercised by the sibling
  `test_layer_norm_sharded_width_non_rectangular_grid` tests, which run the op end to end on the same
  non-rectangular grids and pass.
- **Known downstream consumers.** The same helper Part 1 identified,
  [models/experimental/ops/descriptors/normalization/_utils.py:109](../../../../../../models/experimental/ops/descriptors/normalization/_utils.py#L109),
  reaches this factory through `select_program_factory` for **sharded** inputs. Part 1 measured that
  file's driver test as 60 failed / 29 passed, where the 29 passes were exactly the sharded-input
  branches still reaching this `create_descriptor`. With this pass those 29 broke the same way, so
  the fused-descriptor framework now has **no** working layernorm path and needs a Metal 2.0 route
  (or a branch that stops going through `create_descriptor`) before any of it works again. All 90
  are since skipped rather than failing, by the conftest fixture described in Part 1's entry; the
  capability gap is unchanged, only its visibility in CI. That is the same owner and the same ticket
  as Part 1's entry, with the scope now complete rather than partial.

### The write-back's arguments and its compile gate did not agree, and Metal 2.0 cannot reproduce the mismatch

*Tagged: op behavior — one configuration changes. This is the single place the port does not preserve
legacy behavior, so it is called out at the top of the list.*

- **What legacy did.** `writer_unary_sharded_ln.cpp` and `writer_unary_sharded_ln_rm_gb.cpp` compile
  their write-back block under `#ifndef SKIP_WRITE_BACK`, i.e. whenever `skip_write_back` is false.
  But `build_write_back_args` emits the runtime arguments that block reads — the segment count, the
  storage-core start offset, and the segment array — only when the stage is `POST_ALL_GATHER`
  ([device/sharded_layernorm_factory_helpers.cpp:1641-1646](device/sharded_layernorm_factory_helpers.cpp#L1641-L1646)
  pre-port).
- **The configuration that falls between them.** A **non-distributed** sharded layernorm whose output
  shard spec differs from its input's. Nothing validates the two equal
  ([device/layernorm_device_operation.cpp:163-235](device/layernorm_device_operation.cpp#L163-L235)),
  and `compute_output_specs` takes the output shard spec straight from `output_mem_config`, so a caller
  can produce it. In that build the writer reads `get_arg_val<uint32_t>(6)`, `(7)` and
  `get_arg_addr(8)` past the end of its own six-argument list, then writes through a `c_17` its core
  has no configuration for.
- **Why the port cannot reproduce it.** Metal 2.0 has no way to read a runtime argument the host never
  declared: the name would not exist in the kernel's generated header. The port therefore makes the two
  conditions agree, compiling the write-back when `POST_ALL_GATHER && !skip_write_back` — exactly when
  its arguments exist. Every post-all-gather program behaves identically to legacy.
- **What changes.** The non-distributed-with-differing-output-shard-spec program stops issuing that
  undefined write-back and simply leaves the output where the compute kernel put it. Legacy's behavior
  there was reading uninitialized argument memory and writing to an unconfigured buffer index, so this
  is not a case of trading one defined behavior for another; but it *is* a change, and if that
  configuration is meant to be supported it needs a real write-back path (and its arguments emitted),
  which is an op-owner change rather than a port one.
- **Related, and left alone:** under `POST_ALL_GATHER` **with** `skip_write_back`, legacy emitted the
  write-back arguments and the kernel did not read them. Those are dead runtime args; the port does not
  emit them.

### `get_pointer_to_cb_data` keeps a CB-vocabulary name after the port (third call site)

*Tagged: kernel-lib naming, non-blocking. Extends Part 1's entry with the sharded call site.*

The sharded Welford compute kernel reaches the reciprocal LUT through the same in-family helper Part 1
reported, `norm::kernel_util::compute::memory::get_pointer_to_cb_data<T>(uint32_t, ...)`
([layernorm_sharded_welford.cpp:331](device/kernels/compute/layernorm_sharded_welford.cpp#L331)).
`dfb::reciprocals` flows straight into its `uint32_t` parameter, so nothing is blocked and the donor
needed no change. With this pass all three call sites in the op are converted and the helper's name is
the only CB-vocabulary residue left anywhere in the directory.

## Successes

- **The self-audit's "never stack a self-loop with the multi-binding flag" check was exactly right,
  and the forced legality checks proved it in one run.** The brief named five buffers for
  `allow_instance_multi_binding` and argued the case at length ("The flag is correct here, and it is
  worth knowing why… Do not spend time hunting a 1P+1C assignment for them"). Built that way, the very
  first sharded test failed with a precise message: *"DFB 'ex2' is self-looped … but the set of
  producer KernelSpecs differs from the set of consumer KernelSpecs. When a DFB is self-looped, every
  same-side binding must come from a self-loop participant."* All five are plain **1P+1C**: the kernel
  that pushes takes the producer side, the second toucher the consumer side, and the pushing kernel's
  read-back of its own result needs no endpoint of its own. The recipe's instruction to re-derive the
  census rather than transcribe the brief is what made this a ten-minute correction instead of a
  shipped defect.
- **The dead-CB rule, applied to a per-stage census, found thirteen buffers the brief said did not
  exist.** The brief states "No dead CBs anywhere in this op"; re-deriving the census per
  `(buffer, stage)` shows seven under `PRE_ALL_GATHER` (`c_3`, `c_8`, `c_9`, `c_10`, `c_15`, `c_18`,
  `c_20`) and six under `POST_ALL_GATHER` (`c_8`, `c_9`, `c_10`, `c_11`, `c_13`, `c_20`), plus `c_1`
  when a residual reaches the post stage. Three of them even have declared-but-unused kernel-side
  aliases, which is what put the search on the right track. The validator would have rejected each as
  a bindingless buffer, so the recipe's "build no spec, drop the allocation" is both the rule and the
  only thing that compiles.
- **The alias-group legality rules caught a real inconsistency at spec time.** `c_0`'s legacy
  descriptor is buffer-backed and carries the Welford alias as a second format descriptor, so both
  indices share borrowed memory. Declared as two DFBs, only the primary borrowed at first; the
  `alias_with` rule that every member must agree on `borrowed_from` is what surfaced it, and it is
  documented at the field rather than only in the recipe.
- **Reading the declaring headers beat hunting for a precedent, repeatedly.**
  `advanced_options.hpp` settled the vararg schema, the per-node override's deprecation, and the three
  alias rules; `dataflow_buffer_spec.hpp` settled what `borrowed_from` does and does not validate;
  `program_run_args.hpp` settled the vararg run-args shape. None of that is in the recipe at that level
  of detail, and the headers answered each question in one read.

## Friction

### Gaps

- **The endpoint rule the port actually needed is not written down anywhere.** The recipe and catalog
  frame the choice as self-loop (one toucher) / 1P+1C (two touchers) / flag (census cannot fit 1P+1C),
  and the brief read "compute pushes, then waits on its own result, and a reader also waits" as the
  third case. The framework's rule is narrower and decides it outright: *a self-looped DFB may not have
  any same-side binding from a kernel that is not itself a self-loop participant.* Stated that way, a
  kernel that reads back what it pushed is simply the producer, and the flag never enters. That
  sentence belongs in the endpoint-assignment pattern, next to the "≥2 kernels locked to the same FIFO
  role" line it contradicts in practice.
- **Endpoint counting is per *node*, and the recipe reads as per DFB.** Three separate binding
  decisions here turned on it: a buffer bound as producer only on the gathering compute spec has no
  producer at all on the other nodes where the reader still names it; a consumer bound only on the
  sender reader leaves the other all-to-all nodes without one; and a buffer nothing reads needs its
  producing kernel to hold both ends. The validator's own message says "instance", and
  `advanced_options.hpp` says it plainly ("a DFB on a particular node"), but the recipe's prose does
  not, so the first pass got all three wrong in the same way.
- **No guidance for a kernel whose whole body is compiled out on some nodes.** This factory places an
  idle reader / writer / compute triple on the holes of a non-rectangular shard grid, purely so those
  nodes carry the buffers the reduction multicasts across (legacy did it with a post-pass that widened
  every non-buffer-backed CB's core range). The resolution is worth a line in the catalog because it is
  not obvious: gate the whole kernel body on the preprocessor so it references nothing, and declare the
  bindings **host-side only** — placement comes from the binding, not from the kernel naming the token.
  That also avoids inventing runtime-argument values for a kernel that reads none.
- **A per-node vararg block has no non-deprecated expression.** The write-back segment block's length
  varies per core, `num_runtime_varargs` is a per-kernel scalar, and
  `num_runtime_varargs_per_node` carries a `[[deprecated]]` attribute, so using it risks the build's
  `-Werror`. The port declares the longest block on every node and zero-pads the shorter ones; the
  kernel reads exactly `num_segments_to_write_back` segments, so the padding is never looked at. Worth
  saying in the varargs Caution: if the count varies per node, pad rather than reach for the deprecated
  table.
- **Copying a vararg block into a local array needs a compile-time bound the legacy kernel did not.**
  Legacy took `get_arg_addr(8)` and indexed the argument region in place. `get_vararg` returns a value,
  so the kernel must copy into an array, which needs a size — a new named CTA
  (`max_write_back_segments`) that carries the same number the host used for the vararg count. The
  brief's "fill a small local array from `get_vararg`" is right, but the extra compile-time argument
  that comes with it is not mentioned, and it is not optional.
- **"Reconcile each placement mismatch" needs the mechanism, not just the instruction.** Four buffers
  are declared on `sender_cores` while the compute kernel naming them runs on the wider
  `all_to_all_cores`. The way to express that in Metal 2.0 is to promote the `is_allgather_worker` CTA
  to a preprocessor define and `#ifdef`-gate the aliases, so only the narrower kernel spec binds them —
  the same CTA-to-define promotion the conditional-binding pattern describes, applied to placement
  rather than to existence. Naming that in the brief's placement bullet would have saved a pass.

### Confusion

- **"Declare the conditional-side endpoint unconditionally" and the self-loop rule pull against each
  other, and the recipe gives no order.** Declaring an omitted endpoint is right when it completes a
  producer/consumer pair on a node that would otherwise have only one side. It is *illegal* when it
  turns a 1P+1C into a self-loop plus a third endpoint. Both cases occur in this factory, one node set
  apart. The distinguishing question is whether the kernel being given the extra endpoint is the only
  toucher on that node; a sentence to that effect in the conditional-binding pattern would resolve it.
- **A named compile-time argument carrying a per-spec *value* versus one that must become a define.**
  The reader's `is_all_to_all_worker` stays a named CTA (its two specs differ only in the value), while
  the compute kernel's must become a define (its two specs differ in their runtime-argument *schema*,
  and a named argument has to exist at compile time). Same flag, same op, opposite answers. The rule is
  "does anything downstream of the flag change what the kernel's generated headers contain", which took
  a while to formulate.

## Test results

The set the invoker confirmed: the sharded-path pytests, the interleaved-path pytests as a regression
check on the two kernel headers both factories share, the normalization gtests, and the eager
`test_layernorm_op`.

| | before | after |
|---|---|---|
| pytest | 2667 passed, 628 skipped, 10 xfailed | 2631 passed, **664** skipped, 10 xfailed |
| `unit_tests_ttnn --gtest_filter='*LayerNorm*:*RmsNorm*:*Distributed*Norm*'` | 17 passed | 17 passed |
| `tt_eager/ops/test_layernorm_op` | pass | pass |

2631 + 36 = 2667, and the skipped count rises by exactly the same 36: the only tests whose outcome
changed are the two that drive the removed pybind. Nothing else in the set moved, in either direction.

**The legality checks were live for the whole run.** `METAL2_CHECKS_FORCED program_spec` and
`METAL2_CHECKS_FORCED program_run_args` appear 2924 times each in the pytest log — equal counts, which
is what proves both translation units were rebuilt rather than serving a stale object. Distinguishing
the two marker strings by file (rather than emitting the same text twice, as the recipe's snippet does)
is what makes that a yes/no check instead of a count; Part 1 asked for this and it paid off here,
because the run that caught the endpoint defect needed to be trusted immediately.

## Open items for downstream

### Shared kernel touches

- **Reused the existing `_metal2` fork (rung 1):**
  [ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar_metal2.hpp](../../../kernel/dataflow/generate_bcast_scalar_metal2.hpp),
  whose `generate_bcast_col_scalar(DataflowBuffer&, uint32_t)` replaces the legacy
  `generate_bcast_scalar.hpp` version taking a `CircularBuffer`. Both sharded writers now bind the
  fork. No new file was created and no pointer comment was added to the legacy original: the fork
  already existed, so this is rung 1, where the original is not the porter's to annotate.
  **Sunset update:** Part 1 recorded this op's sharded writers as the remaining unmigrated consumers of
  the legacy header. They are migrated now, so **this op no longer binds
  `generate_bcast_scalar.hpp` at all.** Whatever binds it outside this op is the only thing left
  holding the legacy copy open.
- **`reshard_writer.hpp` changed signature, in-directory, both consumers converted together.**
  [device/kernels/dataflow/reshard_writer.hpp](device/kernels/dataflow/reshard_writer.hpp)'s
  `write_resharded_data` took a `DataflowBuffer&` for the resharded-output buffer and used it only to
  read an address; it now takes that address as a `uint32_t`, because the output is reached through a
  `TensorBinding` rather than a borrowed buffer. Its only two consumers are this factory's two
  write-back writers, both converted in this pass. The identically-named header under
  `ttnn/cpp/ttnn/operations/experimental/ccl/rms_allgather/device/kernels/dataflow/` is **that op's own
  separate file** (it defines `write_minimal_resharded_data` and resolves from its own directory), so
  nothing outside this op is affected — worth stating because a `grep -rl reshard_writer.hpp` makes it
  look shared.
- **Two in-op kernel headers are shared with the Part 1 factory and were left alone.**
  [device/kernels/dataflow/layernorm_dataflow_utils.h](device/kernels/dataflow/layernorm_dataflow_utils.h)
  needed no change: the sharded readers use only `compute_single_stage_noc_addrs` /
  `compute_two_stage_noc_addrs`, whose `L1Ptr` parameters keep working once each kernel copies its
  vararg coordinate block into a local array.
  [device/kernels/layernorm_scaler_tiles.h](device/kernels/layernorm_scaler_tiles.h) is used by both
  factories and needed only a comment sweep from CB to buffer vocabulary.

### Findings the port carried forward unchanged

These are defects and dead code the port preserved rather than fixed, per the porting invariant. Each
is an ops-team call.

1. **Dead plumbing dropped** (each read by no kernel this factory binds; dropping is zero functional
   change, and the values are listed so the owner can confirm):
   - The writer's `gamma_stick_size` / `beta_stick_size` compile-time arg
     ([device/sharded_layernorm_factory_helpers.cpp:706-712](device/sharded_layernorm_factory_helpers.cpp#L706-L712)
     pre-port). Both writer sources skip that slot: the plain writer reads
     `beta_args.next_compile_time_args_offset() + 2 … + 4` and the row-major one `+ 1 … + 5`, and the
     slot exists only in the row-major case, where the row-major writer's `+ 1` already starts past it.
   - The trailing duplicate `use_welford` appended to the writer-receiver list only
     ([:728](device/sharded_layernorm_factory_helpers.cpp#L728) pre-port) — the audit's misc anomaly 4.
   - The pre-all-gather writer's runtime-arg slots 2, 3 and 4 (`eps_u`, `gamma_dram_addr`,
     `beta_dram_addr`) — the audit's misc anomaly 3. The two addresses became the gamma / beta
     `TensorBinding` in the writers that do read them; under `PRE_ALL_GATHER` the port simply does not
     declare the names.
   - The `is_top_row` compute compile-time arg (a literal `0` in every configuration) and
     `FLOAT32_REDUCTION` (compute compile-time arg 11): both are declared in the compute kernels and
     read by none of them.
   - The `block_h_size_bytes` reader compile-time arg: declared in all three reader-sender sources and
     read by none.
   - The write-back runtime args under `POST_ALL_GATHER` **with** `skip_write_back`, which the kernel's
     `SKIP_WRITE_BACK` build never reads.
   - The `dfb_ex2_global` declaration at
     [reader_mcast_sender_unary_sharded_ln_pre_allgather.cpp:49](device/kernels/dataflow/reader_mcast_sender_unary_sharded_ln_pre_allgather.cpp#L49)
     pre-port — the audit's misc anomaly 5 — along with the equally unused `dfb_ex`, `dfb_in1` and
     `dfb_reciprocal` declarations in the pre- and post-all-gather compute kernels.
2. **Thirteen buffers are allocated in configurations where nothing touches them, and the port drops
   those allocations.** Listed under Successes above. A dropped dead buffer has no behavior, but it does
   free SRAM and shift the addresses of the live buffers on the affected cores, so it is worth the
   owner knowing: the pre-all-gather stage gives back seven buffers' worth of SRAM and the
   post-all-gather stage six. Nothing reads them in either stage; the allocations were unconditional in
   code that also serves the non-distributed stage, where most of them are live.
3. **A buffer the writer fills and nobody drains, under `POST_ALL_GATHER`.** The writer generates the
   per-core reduce scaler (`c_2`) in every non-Welford build, but the post-all-gather compute kernel
   reduces the gathered statistics with the *global* scaler alone and never reads it. The port keeps the
   allocation and self-loops it on the writer, which is the sanctioned shape for a single-ended buffer;
   the generation is a few cycles per program and preserved exactly.
4. **Placement widens in two spots, in both cases for buffers that take no SRAM.** The idle triple binds
   the same buffers its active counterparts do, which widens the *borrowed* ones (`in0`, `in1`,
   `in_pre_add`, `out`, `stats`, `reciprocals`) from the active cores to the whole multicast bounding
   box; legacy widened only the non-borrowed ones. A borrowed buffer takes the tensor's address rather
   than a per-core allocation, so the SRAM layout is unchanged and only a configuration entry appears on
   cores that never read it — the same mechanism, and the same harmlessness, as legacy's own `c_17`
   entry on the storage cores. Separately, the four `sender_cores`-only buffers land on
   `all_to_all_cores`, which **equals** `sender_cores` in every non-two-stage pre- or post-all-gather
   program; under a two-stage reduce it is wider, and `c_21` / `c_19`-as-`cb_var` then gain a real
   allocation on the non-sender gathering cores. That is a configuration where the legacy kernels were
   already touching buffer indices their own core had no configuration for, so it is worth an owner
   look independently of this port.
5. **The `#define` typo Part 1 reported has a sibling here.** Part 1 recorded
   `#if defined RMSNORM and not defined FUSED_PRE_ADD` (for `FUSE_PRE_ADD`) in the interleaved compute
   kernel. The same misspelling sits in the sharded one at
   [layernorm_sharded.cpp:401](device/kernels/compute/layernorm_sharded.cpp#L401): the host never
   defines `FUSED_PRE_ADD`, so the guarded `reconfig_data_format` is compiled into every RMSNORM build,
   including the fused-pre-add one it was written to exclude. Carried forward verbatim, typo included.

### Doc-evolution candidates

- **The `cb` → `dfb` sweep over a whole op directory needs the partial-port rule Part 1 asked for, and
  one more thing: the host-side *size and format* variable names.** The sweep's headline case is the
  buffer's own name, and those were clean from the start here. What it actually caught was 130 hits in
  the sharded factory's host code — `in0_CB_size`, `ex_partial_CB_size`, `cb_data_format`,
  `stats_cb_size`, `CBSizeParams` — legacy names for the *sizes and formats of* the buffers, which
  survive a careful rename of the buffers themselves. Reported as *hits / files scanned*: **0 / 42**
  over the op directory after the sweep, counting the three `get_pointer_to_cb_data` call sites
  separately as an out-of-directory helper name (Handoff points above) rather than as leftovers this
  port could fix.
- **A worked example of the per-node endpoint count would carry more than the rule does.** The three
  mistakes this port made are all the same shape and all cheap to describe: producer-only on a subset of
  nodes, consumer-only on a subset, and no consumer at all. A short table of "what the validator says"
  against "what to change" would be the most useful addition to the endpoint pattern.

### Test coverage notes

- **The idle triple is well covered**, which was the most welcome thing the test run said.
  `test_layer_norm_sharded_width_non_rectangular_grid` and its RMSNorm sibling drive shard grids that
  do not fill their bounding box, in both orientations, at an offset origin, with and without Welford,
  and they pass. That is the one part of this factory whose placement the port had to reconstruct
  rather than copy (legacy widened the core ranges in a post-pass; the port gets there by binding the
  buffers on three idle `KernelSpec`s), so having it under test rather than reasoned about is worth
  a lot.
- **Nothing in the confirmed set exercises a two-stage reduce**, which is the only configuration where
  `all_to_all_cores` is wider than `sender_cores` and therefore the only one where the placement
  widening in finding 4 has any effect.
- **Nothing exercises the non-distributed-with-differing-output-shard-spec configuration** described in
  the second Handoff-points entry, which is why the behavior change there is invisible to the test set.
  If that configuration is meant to work, it needs a test before it needs a fix.
