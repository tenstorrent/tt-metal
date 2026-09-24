# Metal 2.0 Port Report — `matmul` / `SparseMatmulMultiCoreReuseMcast1DProgramFactory`

## Outcome

**`PORTED`** — `SparseMatmulMultiCoreReuseMcast1DProgramFactory` converted from
`ProgramDescriptorFactoryConcept` to `ProgramSpecFactoryConcept`. The gtest and the four pytest
suites of the confirmed test set pass with identical counts pre- and post-port, under Watcher, with
the Metal 2.0 host-side legality checks forced on and both markers proven live. That factory is the
only alternative in `SparseMatmulDeviceOperation::program_factory_t`, so the whole device-operation
is now on Metal 2.0.

**One caveat on the evidence:** the sweeps were part of the requested no-regression set and could
not be measured — the sweep framework cannot open a device on this bench, on any op. See
*Verification → Sweeps* for the attribution evidence and *Handoff points* 3. The port is green on
everything that could be measured; sweep coverage is absent rather than passing.

**The port's diff is host-side only — zero kernel edits.** All four kernels were already forked to
`_metal2` by earlier dense-matmul ports, and every name the sparse path needs was already in those
forks (see *Successes* 1).

`MatmulDeviceOperation`'s factories were never in scope and are untouched.

## Provenance

- **Recipe docs (this port):** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`
- **Audit docs (inherited):** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`

Both lines are the same because the audit was re-run in this checkout: the audit on the branch
(`377932c93f3`) predated the op's `ProgramDescriptor` migration and REDed the factory-concept gate,
so it was stale rather than wrong. #57369 cleared that gate on 2026-09-23 and the re-audit came back
GREEN.

## TTNN ProgramFactory

### Concept realized

**`ProgramSpecFactoryConcept`** — as the audit chose. The ported-from factory had no
`override_runtime_arguments`, so the framework owns the cache-hit tensor refresh and the factory
implements one method, `create_program_artifacts`. The recipe's
*Translating `override_runtime_arguments`* step did not apply and was skipped.

### Device-op-class edits

- **Pybind entry points removed:** none. Nothing under `device/sparse/` was ever bound —
  `matmul_nanobind.cpp` binds only the user-facing `sparse_matmul` function — so there was no
  `create_descriptor` binding to delete. **No user-visible API surface changed by this port.**
- **Custom `compute_program_hash`:** none. The declaration and definition are both commented out
  (`device/sparse/sparse_matmul_device_operation.hpp:33`, `.cpp:504`) and were left byte-identical.
- No other device-op-class file was touched. `git diff --stat` covers exactly the factory `.cpp` /
  `.hpp` plus the four `METAL2_*.md` artifacts.

### Open items

- **Relaxation candidate:** none identified. The op has no live custom hash, so there is no narrowed
  cache key to read a tolerated deviation out of, and nothing in the kernels suggested a
  padding-only tolerance worth proposing. Tensor matching stayed strict, as the audit's
  `relaxation = none` requires.
- **Capability the op would benefit from:** `SemaphoreSpec` has no counterpart to the legacy
  `SemaphoreDescriptor::initial_value`. See *Friction → Gaps* — it cost a paragraph of reasoning to
  convince myself the omission was safe, and the reasoning is not local to the factory (it lives in
  the kernels).

## Handoff points

1. **Readiness-sheet refresh — route to the sheet owner (Diego).** The sheet's row for
   (`matmul`, `SparseMatmulDeviceOperation`, `SparseMatmulMultiCoreReuseMcast1DProgramFactory`)
   still reads `Concept = legacy device-op` and `Op Classification = Legacy Op`, while the code has
   been on `descriptor` since #57369 (2026-09-23). `Override runtime args method? (PD only)` reads
   `n/a`, whose post-PD value is `no`. `Is able to port?` reads `yes (with PD step)`; the PD step has
   landed, so it can drop the parenthetical. After this port the row's `Concept` becomes `MetalV2`.

   The audit treated the `Concept` mismatch as staleness rather than as the recipe's
   *spreadsheet-is-broken* GATE, on the reasoning in `METAL2_PREPORT_AUDIT.md` → *Gate detail*. That
   was a judgement call and it is flagged as a question there — **if the reviewer disagrees, this
   port is premature and should wait on the sheet update**, not be merged over it.

2. **`global_cb` is accepted on the public `sparse_matmul` API and never used — route to the ops
   team.** `sparse_matmul(...)` takes
   `const std::optional<const GlobalCircularBuffer>& global_cb`
   (`device/sparse/sparse_matmul_device_operation.hpp:61`, `:79`), threads it through
   `SparseMatmulParams`, and the factory uses it only to let `get_program_config` derive a program
   config — no buffer is ever created from it. A caller passing one gets it discarded with no
   diagnostic, **and** because `SparseMatmulParams` feeds the default reflection hash, two otherwise
   identical calls differing only in `global_cb` miss the program cache while producing identical
   programs.

   This is the one finding with a Metal 2.0 consequence: if the parameter is ever wired up, Metal
   2.0 has no `GlobalDataflowBuffer` and this factory's port would have to be reverted. The audit
   adjudicated Appendix A as N/A on the strength of it being unconsumed today; that verdict is only
   as durable as the parameter staying vestigial. **Worth an explicit owner decision: wire it up (and
   re-open the Metal 2.0 gate) or remove it from the signature.**

3. **The sweep framework cannot open a device on this bench — route to whoever owns the sweep
   runner / the bench.** Every sweep vector dies in a forked child with
   `Bus error (7) / Non-existent physical address (2)`, stack
   `tt::umd::ArcTelemetryReader::read_entry` → `tt::Cluster::get_device_aiclk`, during device
   bring-up and before any program is built. Reproduced on `matmul.sparse.sparse_matmul` (14/14
   attempted) **and** on the unrelated `eltwise.unary.abs.abs` (9/9), so it is not op-specific.
   In-process pytest and gtest runs open the same board fine, so it is specific to the runner's
   per-vector subprocess model. Detail in *Verification → Sweeps*. **Consequence for this port:** the
   requested sweep coverage is missing rather than green, and should be re-run elsewhere before merge.

4. **No boundary-rule assumption violations, no kernel-lib gaps, no Metal 2.0 framework gaps.** No
   call site required passing a `sem::` or `tensor::` handle outside the op directory, and no shared
   kernel needed a change (see *Successes* 1).

## Successes

1. **[Caution: Porting a shared kernel] rung 1 was the whole port, and the entry's ordering is what
   made that visible.** All four kernels this factory binds are *lent* — matmul-owned but also bound
   by the dense factories — and all four already had `_metal2` forks with live consumers. Running the
   entry's rung 1 check *first* (`ls` the original's directory, locationally, not a tree-wide grep)
   turned what looked like a four-kernel conversion into **zero kernel edits**. The port is
   host-side only.

   The entry's "check fit before committing to reuse" step then paid for itself twice:
   - Each fork's header note already names `SPARSITY` as a define-gated optional resource family,
     and both readers already read `batchB`, `bcast_A`, `get_batch_from_reader`, `num_active`,
     `sparsity_pagesize`, `num_batch_compute` and `compact_output`. Every name the sparse path needs
     was present. Whoever wrote those forks left the sparse porter a complete interface, and said so
     in the header — that is the convention working exactly as designed.
   - It also told me what I was *not* allowed to do. "A fork that already has a consumer is
     read-only to you" removed the temptation to tidy two fork-side things I noticed (below).

2. **[Pattern: Conditional / optional resource bindings] explained a define I would otherwise have
   read as redundant.** `SPARSITY` is *constant true* for this factory — the sparsity operand is
   mandatory and `batchB >= 1` always — so emitting a define for a condition that never varies looks
   like dead weight. The pattern's *Why this is hard* paragraph is what says otherwise: the define
   does not gate a *branch*, it gates whether `dfb::sparsity` / `tensor::sparsity` reach C++ name
   lookup at all, and the shared forks default to sparsity-**off**. Omitting it would have produced
   a kernel that fails to compile on a name that the host did bind. The factory comment at the
   define now states that reasoning inline, since the pattern doc does not reach `main`.

3. **The hw_config section's "match on the values, not the role name" warning fired correctly, in
   the direction I did not expect.** The sparse factory puts the in0 sender on `RISCV_0` and the in1
   sender/writer on `RISCV_1` — the **opposite** of the dense metal2 factory sitting in the same
   directory, whose in0-mcast path is otherwise the closest available template. Copying the dense
   `in0_sender_hw_config` / `in1_sender_hw_config` locals would have swapped both kernels' RISC
   cores, and the recipe is right that nothing would have caught it: the spec validator only checks
   that the two DM kernels on a node use *distinct* cores and *distinct* NOCs, which the swapped
   assignment still satisfies. Diffing resolved triples against the legacy descriptors, as the
   section instructs, is what surfaced it.

4. **The `opt_level` check found the defect it is written to find.** `grep -n opt_level` over the
   ported-from factory returns **nothing**, which reads as "nothing to carry" and is wrong: an
   absent `KernelDescriptor::opt_level` on a `ComputeConfigDescriptor` resolves to `O3`, while Metal
   2.0's `CompilerOptions` defaults to `O2`. The compute `KernelSpec` therefore needed an explicit
   `KernelBuildOptLevel::O3` that no error, test or validator would have asked for. The section's
   insistence that this is an *absent line* rather than a wrong value — "do not eyeball this one" —
   is what got it checked at all.

5. **The compute-config "dropped field" check came back negative, and the check is what made that
   knowable.** The dense metal2 factory has to force `double_buffer_dest(compute_hw) = true` because
   its ported-from factory resolves `dst_full_sync_en` and then never passes it on. The sparse
   factory *does* pass it (`ComputeConfigDescriptor{… .dst_full_sync_en = dst_full_sync_en …}`), so
   `to_compute_hardware_config` is faithful with nothing to pin — and copying the dense pin would
   have silently overridden a knob this op honours. Comparing "fields resolved" against "fields set",
   per the section, is the only way to tell those two ops apart.

## Friction

### Gaps

1. **The recipe has no instruction for a legacy field with no Metal 2.0 destination, and
   `SemaphoreDescriptor::initial_value` is one.** The ported-from factory sets
   `SemaphoreDescriptor{.id = …, .core_ranges = all_cores, .initial_value = INVALID}` on both
   semaphores. `SemaphoreSpec` carries `unique_id`, `target_nodes` and `advanced_options` — there is
   no initial-value field, and the recipe's *Construct* bullet for `SemaphoreSpec` ("Build with
   `target_nodes`") does not mention that anything is being dropped.

   Dropping a field silently is exactly what the porting invariant forbids, so resolving it meant
   leaving the factory and reading the kernels: the in0 sender fork sets the receiver semaphore to
   `VALID` at entry and the receivers reset to `INVALID` per batch, so the observable initial state
   is re-established in-kernel and the legacy descriptor value was inert. That is a sound answer, but
   it took a kernel read to reach, and a porter who *didn't* go looking would have dropped the field
   without noticing there was a question. **Suggested fix:** a sentence under `SemaphoreSpec` saying
   that the legacy `initial_value` has no spec counterpart because Metal 2.0 initialises semaphores
   itself, and that a kernel relying on a specific initial value establishes it in-kernel — so the
   field is dropped, not translated.

2. **The `TT_FATAL` census counts comment text, and the port is expected to *add* comments.** The
   census compares per-file `grep -cE 'TT_FATAL|TT_ASSERT|TT_THROW'` counts across the port and
   expects no output. Mine reported `…mcast_1d_optimized.cpp:4` → `:5` — an *increase*, from a
   comment I wrote explaining that omitting an `unpack_modes` entry is "a TT_FATAL at program build".
   All four real guards were present and unchanged.

   The check's failure mode is benign (I re-worded the comment), but two things are worth noting:
   the instructions only describe what to do when a count **drops**, so an increase has no documented
   interpretation; and a rise is the *likelier* direction for this port, because the recipe elsewhere
   (correctly) pushes porters to explain validator requirements in comments, and the clearest way to
   say "the validator rejects this" names the macro. **Suggested fix:** grep for the macro *call*
   (`TT_FATAL\(`) rather than the bare token, and say explicitly that a count increase is
   comment-only until proven otherwise.

3. **The `cb`-leftover sweep has no adjudication step for a hit the port may not touch.** The sweep
   (`grep -rnE '[Cc][Bb]_|_[Cc][Bb]\b|\b[Cc][Bb]\b|\bCB[A-Z]'`) is specified as "expect **zero**
   hits: post-port the op has no CBs, so every hit is a real leftover". Over this op's directory it
   returns eleven, and **none** is a leftover: ten are `global_cb` / `GlobalCircularBuffer` in the
   device-operation class and its types header — off-limits host code the port may not edit — and the
   eleventh is the factory's own `operation_attributes.global_cb`, a field name on the attributes
   struct that the factory must read to build `MatmulParams`.

   So the honest result is "eleven hits, eleven adjudicated, zero actionable," which the check has no
   vocabulary for. **Suggested fix:** note that a `global_cb` / GlobalCircularBuffer hit outside the
   factory body is device-op-class code and out of scope, and that the sweep's zero-hit expectation
   applies to the *factory* and its kernels rather than the whole op directory.

### Confusion

4. **"Reuse an existing `_metal2` fork" and "the port adds exactly two headers" pull in opposite
   directions, and the resolution is only implicit.** The kernel-side whitelist opens with a precise
   account of what the port does to kernel code — two headers added, `CircularBuffer` →
   `DataflowBuffer`, named args throughout — while rung 1 of the shared-kernel Caution means the
   port does **none** of it: the forks are already converted and read-only. I spent time confirming
   that a port whose kernel diff is empty is a complete port rather than a half-finished one, and the
   thing that settled it was the *Open items* instruction to record the rung taken, which presupposes
   that "reused, no edit" is a legitimate terminal state.

   It reads as a gap in framing rather than in content. **Suggested fix:** one line at the top of the
   kernel-side whitelist noting that on rung 1 the whitelist describes work an *earlier* port already
   did, and this port's kernel diff is legitimately empty — the whitelist then applies to *reading*
   the fork to check fit, not to editing it.

5. **The audit/port split assumes the audit's factory-concept verdict is current, and a re-audit is
   not a documented workflow state.** `READ_ME_FIRST.md` describes audit → review → port, each a
   fresh session, and the port recipe's precondition is "the audit … produced an overall GREEN
   result". This op arrived with a **RED** audit that was correct when written and obsolete by the
   time the port was requested, because the blocker it named — the `ProgramDescriptor` migration —
   had since landed. The previous audit even asked, in its *Questions*, whether re-audit should be
   automatic on that event.

   Nothing in either document says who re-runs the audit, or that a port session may. Re-running it
   was clearly right, and the audit recipe's own "the op is re-audited once the cleanup lands"
   language supports it — but the two documents' *workflow* sections do not, and a porter reading
   only the precondition would either stop or port against a RED audit. **Suggested fix:** a bullet
   in the port recipe's *Inputs the invoker should have supplied* for the case where the brief is
   absent because the audit REDed on a blocker that has since cleared: re-run the audit in this
   session, note it in the report's Provenance, and proceed on the fresh verdict.

## Open items for downstream

### Shared kernel touches

Four kernels, all **rung 1 (reused an existing `_metal2` fork)**. **No file was created, forked or
modified**, and per rung 1 no pointer comment was added to any legacy original.

| kernel (fork bound by this port) | rung | remaining unmigrated consumers of the legacy original |
|---|---|---|
| `…/dataflow/reader_bmm_tile_layout_in0_sender_padding_metal2.cpp` | reused | `matmul_multicore_reuse_mcast_1d_program_factory.cpp` (descriptor + MeshWorkload paths), `matmul_multicore_reuse_mcast_2d_program_factory.cpp` |
| `…/dataflow/reader_bmm_tile_layout_in0_receiver_metal2.cpp` | reused | `…mcast_1d…`, `…mcast_2d…` |
| `…/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding_metal2.cpp` | reused | `…mcast_1d…`, `…mcast_2d…` |
| `…/compute/bmm_large_block_zm_fused_bias_activation_metal2.cpp` | reused | `…mcast_1d…`, `…mcast_2d…` |

This is the **sunset checklist**, not authorization to convert anything in place. The legacy copies
still serve the dense factories' descriptor paths; whichever port migrates the last of those can
delete them.

**This port is the first to set `SPARSITY` on any of these forks.** The region was checked in with
the forks and, until now, no factory enabled it — so this port is also its first execution through a
Metal 2.0 spec. A later porter debugging that region should know the code is older than its first
run.

### Fork-side observations (deliberately not acted on — the forks are read-only)

1. **`reader_bmm_tile_layout_in0_sender_padding_metal2.cpp` marks `num_batch_compute`
   `[[maybe_unused]]`** even though, under `SPARSITY`, the kernel's on-device `count_nonzero`
   validation does use it. The attribute was presumably added when no factory set the define. Not a
   bug — `[[maybe_unused]]` is inert when the variable *is* used — but the attribute now says
   something false about the only configuration that compiles the line.
2. **The in1 fork's `#ifdef SPARSITY` block does `reserve_back(1)` with no matching `push_back`**
   (unlike the in0 sender, which does both). Faithful to the legacy kernel, which behaves the same
   way, and harmless because the buffer's only toucher is that kernel and per-execution buffer state
   is reinitialised. Recorded only so a future reader does not take it for a port error introduced
   here — and it is precisely the shape the recipe warns against "fixing" by adding a `push`.

### Dead legacy plumbing removed by the translation (all inert, all recorded in the audit)

The port dropped these because Metal 2.0 has no channel for them, not as cleanup. None had
behaviour:

- **Five trailing in1 runtime-arg slots** (indices 21–25). The legacy kernel's last read in this
  configuration is `last_num_blocks_w_dim` at index **20**; the five zeros after it were placeholders
  for the `IN1_DRAM_*_SHARDED` reads, which no Metal 2.0 factory can enable. Named args live in their
  own section, so there is nothing to pad.
- **Two bias placeholder slots** (indices 18–19), which the legacy kernel explicitly steps over
  (`rt_args_idx += 2;` in its `#else` branch). *This is the one place a careless port would break
  the op:* the placeholders are why `last_num_blocks_w_dim` correctly lands at index 20 and not 18,
  and a porter who counted the kernel's visible reads and stopped at 18 would have mapped the wrong
  value to that name.
- **Three named CB arguments pointing at indices the factory never allocated** —
  `cb_in0_sharded`→`c_2`, `cb_bias`→`c_3`, `cb_in0_transposed`→`c_10`. Each was read by its kernel
  only under a feature this factory never enables. Under Metal 2.0 they simply have no binding.
- **Five dead `->address()` reads** that filled a staging `uint32_t` vector whose slots the `Buffer*`
  assignments then overwrote at the same index.
- **Six dead locals** — four mcast-list / no-work core sets left over from the block-sharded path
  this factory does not have, and the `*_cb_index` variables that existed only to feed `log_debug`.
- **The four `log_debug(LogOp, "CB {} :: PS = …")` calls.** Their subject was the CB index, which no
  longer exists; the DFB spec states entry size and count as data. Dropped with the concept rather
  than rewritten. Flagging it because it is the one *observable* (log-only) difference in the diff.

### Carried-forward oddities the port preserved deliberately

- **`FUSE_ACTIVATION = "0"` is still emitted to the compute kernel.** No matmul compute kernel —
  legacy or fork — references the name; it has been dead since some earlier refactor. Carried because
  the port preserves behaviour including inert behaviour, and because a define whose value is `"0"`
  would read as *true* under `#ifdef` if anyone revived the name. Routed to the ops team, not fixed.
- **`batchA` is passed twice to the in0 sender**, as both `in0_B` and `in1_B`. Correct for this op —
  the sparse 1D path broadcasts in1 across the A-batch — and the named form now makes the intent
  legible where the positional list read like a copy-paste bug.
- **The in1 sender/writer binds two semaphores it never touches.** `SKIP_MCAST` is unconditional
  here, so the kernel constructs both `Semaphore` objects and uses neither. The legacy factory passed
  literal ids `0`/`0` into those slots — i.e. it pointed *both* in1 handles at semaphore 0 — whereas
  this port binds the sender and receiver specs respectively, matching what the dense metal2 factory
  does from the same legacy `0`/`0`. Behaviour-neutral (the usage is compiled out), and following the
  already-tested precedent rather than reproducing a value the legacy code only got away with because
  nothing read it.

### Test coverage notes

- **`test_sparse_matmul.py` skips 2 of 31 cases as "Quasar-only API"** in both the pre- and post-port
  runs. Out of scope for a Gen1 port, but it means the op's Quasar surface has no coverage on this
  bench.
- **The sweeps were requested as part of the no-regression baseline and could NOT be measured — the
  sweep framework cannot run on this host at all.** Full detail and the attribution evidence are in
  *Verification → Sweeps*. In short: every sweep vector dies with a host-side `Bus error` in
  `Cluster::get_device_aiclk`, and an unrelated op (`eltwise.unary.abs.abs`) fails identically, so it
  is a bench/UMD problem rather than anything to do with this port. **The port's green result does not
  include sweep coverage.** Someone should re-run
  `tests/sweep_framework/sweeps/matmul/sparse/{sparse_matmul,batched_sparse_matmul}.py` on a host
  where the framework works before this merges.
- The op's pytest coverage lives at `tests/ttnn/unit_tests/operations/matmul/` (not a
  `sparse_matmul/` subdirectory), and the C++ coverage is a single fixture, `SparseMatmulFp32Test`,
  in the umbrella `unit_tests_ttnn` binary.

## Verification

**Legality checks forced and proven live.** `skip_validation` was forced `false` as the first
statement of all nine functions `grep -n 'bool skip_validation' tt_metal/impl/metal2_host_api/*.cpp`
names, with a distinguishing marker per file (`METAL2_CHECKS_FORCED program_spec` /
`… program_run_args`) so "both translation units are fresh" is checkable rather than inferred from a
single ambiguous count. **None of that scaffolding is in the diff** — see the self-audit below.

Both markers appeared **48 times each, exactly paired**, across the post-port run — so both
translation units were fresh and both the spec-side and run-args-side checks were live for every
program this op built.

| test | pre-port | post-port |
|---|---|---|
| `unit_tests_ttnn --gtest_filter='SparseMatmulFp32Test.*'` | 2 passed | **2 passed** |
| `tests/ttnn/unit_tests/operations/matmul/test_sparse_matmul.py` | 29 passed, 2 skipped | **29 passed, 2 skipped** |
| `tests/ttnn/unit_tests/operations/matmul/test_sparse_matmul_indexed.py` | 18 passed | **18 passed** |
| `tests/ttnn/docs_examples/test_matrix_multiplication_examples.py` | 7 passed | **7 passed** |
| `tests/ttnn/unit_tests/base_functionality/test_comparison_mode.py` | 90 passed, 1 skipped | **90 passed, 1 skipped** |

Identical counts on both sides, which is the result a port should produce. All runs with
`TT_METAL_WATCHER=10`, on a single Blackhole p100a; **no** Watcher trip, no `0xdeadc0de`, and no
unexpected `TT_FATAL` (the `TT_FATAL`s in the log are all inside the tests' own
`[EXPECTED_ERROR BEGIN/END]` brackets — negative-path assertions whose owning tests pass).

The two skips in `test_sparse_matmul.py` are "Quasar-only API" and the one in `test_comparison_mode.py`
is `requires_fast_runtime_mode_off`; both skip pre- and post-port alike.

### Sweeps — requested, attempted, and not measurable on this host

The invoker asked for the sparse sweeps in the no-regression set. They were generated and run, and
**they produce no signal here: the sweep framework itself cannot open a device on this bench.**

- `sweeps_parameter_generator.py` worked (32 vectors for `matmul.sparse.sparse_matmul`, 22 for
  `matmul.sparse.batched_sparse_matmul`).
- `sweeps_runner.py --vector-source file` then failed **every** vector identically: the forked
  per-vector child dies with `Signal: Bus error (7) / Signal code: Non-existent physical address (2)`,
  with `tt::umd::ArcTelemetryReader::read_entry` → `tt::Cluster::get_device_aiclk` on the stack —
  i.e. a host-side ARC-telemetry read over PCIe during device bring-up, before any program is built.
  The framework marks each `FAIL_CRASH_HANG`, runs `tt-smi -r`, and the next vector does the same.
  14 consecutive failures before the run was stopped.
- **Attribution (the reason this is reported rather than investigated as a port bug):** the same
  harness fails the same way on **`eltwise.unary.abs.abs`** — 9/9 vectors, same signal, same
  `get_device_aiclk` frame. An op with no relationship to matmul. So the failure is in the sweep
  runner's subprocess/device-open path on this host, not in anything this port touched.
- Corroborating: the five suites above exercise this exact factory — including the fp32
  partial-reload path and the full indexed/gather path — under Watcher, in-process, with zero
  failures. A defect that crashed the sweeps at device open would not be invisible there.

**What this means for the port's evidence:** the gtest + four pytest suites are the measured
no-regression baseline, pre- and post-port, and they match exactly. Sweep coverage is **absent, not
green** — it needs re-running on a host where the framework works. Flagged here rather than quietly
dropped from the requested set.

### Anti-pattern self-audit

Denominator: **5** files under `device/sparse/` (`find … -name '*.cpp' -o -name '*.hpp' | wc -l`).

| check | result |
|---|---|
| No buffer address in run-args (`->address()`, `emplace_runtime_args`, bare `Buffer*`) | **0 / 5** |
| No magic CB indices in CTAs (`CBIndex`, `c_<n>`) | **0 / 5** |
| No `TensorAccessorArgs<N>()` in a ported kernel | **0 / 5** (and no kernel was edited) |
| No `cb` in a DFB name or variable | **11 / 5 hits, all adjudicated** — see *Friction → Gaps 3*: ten are `global_cb` / `GlobalCircularBuffer` in off-limits device-op-class code, one is the factory reading `operation_attributes.global_cb` to build `MatmulParams`. Zero are DFB names or port-introduced |
| Conditional DFB bindings follow the pattern | ✓ — `SPARSITY` define emitted to both readers alongside the bindings; the forks carry the `#ifdef` gating |
| No `.id` extraction at LLK call sites | **0 / 5** |
| No CTA→RTA demotion | ✓ — no work-split multiplicity existed to demote |
| No unnecessary multi-binding flag, never stacked with a self-loop | **0 / 5** `allow_instance_multi_binding`. Three self-loops (`intermed0`, `sparsity`, `in1_sparsity`), each a genuine one-toucher; none also multi-bound |
| All CTAs named | ✓ — four `compile_time_args`, all `{{name, value}, …}` |
| No nameable argument in varargs | **0 / 5** vararg uses; every runtime arg is a distinct field read once |
| No forced-legality scaffolding in the diff | ✓ — `git diff --name-only $BASE \| grep -E '^tt_metal/'` and `git diff $BASE \| grep -nE 'METAL2_CHECKS_FORCED\|DO NOT COMMIT'` both empty |
| No ephemeral doc cited from code | **0 hits** over the diffed `.cpp`/`.hpp` set |
| Every legacy `TT_FATAL` accounted for | ✓ — all four guards present and unchanged; the one count delta was comment text (*Friction → Gaps 2*) |
| Every `hw_config` reproduces the legacy resolved values | ✓ — in0 sender/receiver `(RISCV_0, in0_noc, DM_DEDICATED_NOC)`, in1 sender/writer `(RISCV_1, in1_noc, DM_DEDICATED_NOC)`, diffed against the legacy descriptors. Compute via `to_compute_hardware_config` with no pinned field (nothing dropped); `bfp_pack_precision_mode` left default (legacy did not set it); `unpack_modes` reindexed from CB id to DFB name with `UnpackToDest` on the partials buffer and explicit `UnpackToSrc` on every other Float32 buffer the kernel consumes |
| Every `KernelSpec`'s `opt_level` matches its legacy kernel's | ✓ — one compute `KernelSpec`, one `opt_level` line (`O3`, explicit, because the legacy `ComputeConfigDescriptor` resolved to `O3` and Metal 2.0 defaults to `O2`); the three DM specs correctly carry none (legacy `O2` → Metal 2.0 `O2`) |
