# Metal 2.0 Port Report — `layernorm` (`LayerNormMultiCoreProgramFactory`)

## Outcome

**`PORTED`** — `LayerNormMultiCoreProgramFactory` (the interleaved / non-sharded path) and the ten
kernel entry points it can select are on `ProgramSpecFactoryConcept`.

`LayerNormShardedProgramFactory` is **not** ported in this pass and stays on
`ProgramDescriptorFactoryConcept`. The two factories share no kernel source, so the `program_factory_t`
variant is valid with one factory on each concept and the op builds and runs throughout. The sharded
factory's blocker is recorded under [Open items for downstream](#open-items-for-downstream).

**No-regression result.** The confirmed test set gives **2236 passed, 22 skipped** both before and
after the port, with an identical pass/skip split per file. The only behavior change anywhere is the
sanctioned pybind removal, characterized precisely below.

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
- **`LayerNormShardedProgramFactory::create_descriptor` and its nanobind are untouched**, because
  that factory is not ported.
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
  `std::bad_variant_access` on a Gen2 device. This is the one count delta in the TT_FATAL census
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
- **Why it had to go:** `create_program_artifacts` has a fixed three-parameter signature that cannot
  carry `core_range_set`, and the `create_descriptor` symbol no longer exists. This is
  ttnn_factory's exception 2 (which names layernorm's `core_range_set` explicitly): the parameter is
  dropped and its production default, `default_core_range(device)`, is inlined at the work-split
  site.
- **Known downstream consumers** (all outside the porter's writeable surface; none were edited):
  - [models/experimental/ops/descriptors/normalization/_utils.py:113](../../../../../../models/experimental/ops/descriptors/normalization/_utils.py#L113)
    — `factory.create_descriptor(operation_params, tensor_args, out, cr_arg)`, reached through
    `select_program_factory`, so it breaks for interleaved inputs and keeps working for sharded
    ones. The same file also calls `default_core_range` at
    [:44](../../../../../../models/experimental/ops/descriptors/normalization/_utils.py#L44), which
    still exists.
  - [tests/ttnn/unit_tests/operations/fused/parallel_sequential/test_parallel_sequential.py](../../../../../../tests/ttnn/unit_tests/operations/fused/parallel_sequential/test_parallel_sequential.py)
    — drives that helper with interleaved tensors and an explicit `core_range_set`.
- **Measured impact.** Running that file post-port gives **60 failed, 29 passed**, and every one of
  the 60 failures is the same line: `AttributeError: 'LayerNormMultiCoreProgramFactory' object has
  no attribute 'create_descriptor'`. There is no numerics failure and no second failure mode; the 29
  that pass are the sharded-input branches, which still reach the sharded factory's surviving
  `create_descriptor`. The fused-descriptor framework needs a Metal 2.0 path (or an interleaved
  layernorm branch that stops going through `create_descriptor`) before the interleaved half of it
  works again.

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
- **The two-toucher / self-loop distinction held up under re-derivation.** The brief listed the
  interleaved factory's dispositions, and re-running the census per selected kernel source (rather
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
  other configurations the buffer has *zero* touchers rather than one. The same sentence resolves
  it, but it took a while to recognise that it applied.
- **The `Group<T>` / `Table<K,V>` heads-up says these are Metal 2.0 types but the examples use them
  unqualified.** They live in `tt::tt_metal::experimental`, so a factory that aliases the namespace
  (`namespace m2 = …`, as the sibling port does) has to write `m2::Group<…>`. Mentioning the
  namespace once in the type-system heads-up would help.

## Open items for downstream

### The sharded factory is not ported, and audit Question 2 blocks it

`LayerNormShardedProgramFactory` remains on `ProgramDescriptorFactoryConcept`. The blocker is the
audit's **Question 2**, which is a construction decision no porter can route around: `c_17`
(`cb_out_resharded`, POST without `skip_write_back`) is declared over `all_worker_and_storage_cores`
— *wider* than any kernel's range — and `write_resharded_data` reads its **local** `get_write_ptr()`
and uses that value as the destination address on a **remote** storage node
([device/kernels/dataflow/reshard_writer.hpp:39](device/kernels/dataflow/reshard_writer.hpp#L39)).
That only works because the buffer sits at the same SRAM address on every node in the union. Metal
2.0 derives a buffer's node set from its bound kernels, so a buffer with no bound kernel on a node
cannot be placed there, and dropping it on the storage nodes would corrupt whatever the allocator
put at that address. Either the framework needs a way to place a borrowed-memory buffer on nodes
with no binding, or the sharded port has to bind something on the storage nodes. **Get that answer
before starting the sharded factory.**

Everything else the sharded factory needs is already scoped by the audit and brief: five
`allow_instance_multi_binding` buffers, five runtime-vararg sites with the `get_arg_addr` pointer
wrinkle, three semaphores, and the borrowed-memory list.

### Shared kernel touches

- **Reused an existing `_metal2` fork (rung 1):**
  [ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar_metal2.hpp](../../../../kernel/dataflow/generate_bcast_scalar_metal2.hpp),
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
- The watch item the plan flagged — `UnpackToDest` on a `Float16_b` buffer when
  `welford_fp32_alias && fuse_pre_add && !fp32_dest_acc_en`, which the Gen1 validator can reject —
  **did not fire** across the full run with the checks forced on, and it cannot: reaching it needs a
  residual *and* an explicit `compute_kernel_config` with `fp32_dest_acc_en=False` *and* a float32
  input *and* a non-large tensor. `test_layer_norm_ulp.py` is the only file that varies
  `fp32_dest_acc_en`, and it passes no residual, so nothing in the confirmed set covers the
  combination. It is worth a targeted test rather than leaving it to chance.
