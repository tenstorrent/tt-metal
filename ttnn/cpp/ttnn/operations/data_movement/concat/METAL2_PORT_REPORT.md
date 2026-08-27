# Metal 2.0 Port Report — `data_movement/concat`

## Outcome

**`PORTED`** — all **three** factories in the audit brief's clean subset converted to
`ProgramSpecFactoryConcept`, and the confirmed test set passes **identically to the pre-port baseline**:

| | result |
|---|---|
| pre-port baseline | `672 passed, 20 skipped, 3 xfailed` |
| post-port (all 3 factories) | `672 passed, 20 skipped, 3 xfailed` |

Ported: `ConcatProgramFactory`, `ConcatS2SRMProgramFactory`, `ConcatS2STiledProgramFactory`.
Untouched (audit-gated, still on the `descriptor` concept): `ConcatS2SMultiProgramFactory`,
`ConcatBlockShardedProgramFactory`, `ConcatS2IProgramFactory`. `ConcatDeviceOperation`'s
`program_factory_t` variant therefore holds a 3/3 mix of `MetalV2` and `descriptor` factories, which the
brief records as confirmed-supported; it builds and runs, and the gated factories' tests still pass.

Verified with the host-side legality checks **forced on and proven live** — 2680 `METAL2_CHECKS_FORCED`
lines across the run, from both translation units (`program_spec.cpp`, `program_run_args.cpp`).

## Provenance

- **Recipe docs (this port):** `0846547f407 2026-08-27 docs(metal_2.0): a run in flight freezes the kernel sources`
- **Audit docs (inherited):** `0846547f407 2026-08-27 docs(metal_2.0): a run in flight freezes the kernel sources`

## TTNN ProgramFactory

### Concept realized

`ProgramSpecFactoryConcept` on all three factories — the concept the audit chose, unchanged. No
`override_runtime_arguments` exists anywhere in the op, so the custom concept never came into play and
[Translating `override_runtime_arguments`] was skipped entirely.

### Device-op-class edits

- **Pybind entry points removed: none.** `concat_nanobind.cpp` binds only the public `ttnn::concat` free
  function; `create_descriptor` was never pybound, so there is no user-visible API surface change.
- **Custom `compute_program_hash`: none** — `grep -rn 'compute_program_hash|attribute_values|to_hash|override_runtime_arguments|get_dynamic_runtime_args'`
  over the whole op directory returns zero hits. Default reflection-based hash, untouched.
- `concat_device_operation.{hpp,cpp}` is **byte-identical** to pre-port. `select_program_factory` still
  dispatches to all six factories; the variant still lists all six. Confirmed by the TT_FATAL census
  below, which shows no count change outside the ported factories.

### Open items

- **Relaxation candidates: none identified.** All nine `TensorParameter`s keep strict `TensorSpec`
  matching, matching the brief's `relaxation: none`. Nothing in the kernels suggested a relaxation
  would be tolerated, and the port is not the place to decide one anyway.
- **`ConcatProgramFactory` carries up to 47 `TensorBinding`s on one `KernelSpec`** (`N` inputs, capped
  at 47 by `concat_device_operation.cpp:285`). Every binding consumes a CRTA slot for its
  auto-injected base address. The validator accepted it and the tests pass at the widths the suite
  exercises, but the op's ceiling is unusually high for this mechanism and is worth a deliberate
  look from the framework side rather than being discovered by a model.

## Handoff points

**None.** No capitulation, no boundary-rule assumption violation, no kernel-lib gap, no framework gap,
no removed pybind surface. Specifically:

- Both borrowed donor writers were at **rung 1** — a `_metal2` fork already existed beside each
  original, both fit concat exactly in both directions, so the port **binds and adopts**. Nothing was
  written into either peer directory, no fork was edited, and no fork was created.
- No `sem::` or `tensor::` handle was needed at an out-of-op call site (the op declares no semaphores,
  and the two donor forks already construct their own accessors). The boundary assumption held.
- No kernel outside the op's own directory was modified.

## Successes

- **[Compiler options — `opt_level`] fired exactly as designed, on the one spec that needed it.**
  `grep -n opt_level` over the legacy op printed **nothing at all**, which reads as "no setting,
  nothing to carry." The section's insistence that an absent `KernelDescriptor::opt_level` is *not*
  "no setting" — `std::nullopt` resolves to `O3` for a `ComputeConfigDescriptor` while Metal 2.0's
  type-agnostic `CompilerOptions` defaults to `O2` — is what put an explicit
  `KernelBuildOptLevel::O3` on `concat_s2s_tiled_program_factory.cpp:323`. Nothing else in the port
  would have caught it: it is an absent line, no validator or test distinguishes the levels, and
  `to_compute_hardware_config`-style helpers do not carry it. The paired check (enumerate the compute
  `KernelSpec`s from the construction code, pair each with a line of `grep -nE opt_level`) resolved to
  **one spec, one line**, which is the whole verification.

- **The two-toucher / self-loop split held up on both of the shapes it distinguishes, in one op.**
  `ConcatS2SRMProgramFactory` is the dual-instance work-split verbatim — one source pushed into a
  Reader-config and a Writer-config descriptor over the *same* grid, all touches raw cursor peeks,
  and the kernel contains no FIFO call at all. Two role-free touchers → **1P+1C** on all three DFBs
  (`concat_s2s_rm_program_factory.cpp:196-221`). `ConcatS2STiledProgramFactory`'s `output` buffer is
  the *other* shape: one toucher → **self-loop** (`concat_s2s_tiled_program_factory.cpp:265-283`).
  Having the census procedure state the hard gate ("self-loop applies **only** when exactly one kernel
  touches it") is what kept those two apart; without it, "sync-free raw touches" reads like one case.

- **"Re-derive the census, don't transcribe it" earned its place on the tiled factory's `output`.**
  The brief's own warning pointed at it, but the load-bearing move was reading the compute kernel's
  *body*: it constructs `DataflowBuffer output_dfb(...)` and never uses it. Binding `output` on compute
  to satisfy that construction would have manufactured a two-toucher where the code has one, turning a
  correct self-loop into a wrong 1P+1C. My census agreed with the brief on all seven of that factory's
  buffers and all three of the RM factory's.

- **The shared-kernel Caution's "run the census, *then disambiguate it*" was the difference between
  1 and 5 consumers.** `grep -rl writer_unary_stick_layout_interleaved_start_id ttnn/cpp/ttnn/operations/`
  returned five factories beyond concat. Four are not consumers:
  `embedding/device/embeddings_rm_program_factory.cpp:329` and
  `embedding/device/embeddings_tilized_indices_program_factory.cpp:231` bind the **`_metal2` fork**
  (they are the fork's existing consumers — which is precisely what makes it read-only to this port);
  `data_movement/slice/device/slice_program_factory_rm.cpp:366` binds a *different* file,
  `slice_writer_unary_stick_layout_interleaved_start_id.cpp`, a substring match; and the
  `experimental/quasar/slice` hit is out of bounds. Only
  `data_movement/copy/device/copy_same_memory_config_program_factory.cpp:37` is a real remaining
  legacy consumer — which is exactly what the brief said. Reading the hit list as a consumer list
  would have inflated the sunset list fourfold and, worse, suggested the fork was mine to change.

- **The locational rung-1 test kept `experimental/quasar/` out with no effort.** Broad greps for both
  donor filenames surfaced quasar hits (`quasar/slice`, `quasar/matmul`, `quasar/tilize`,
  `quasar/typecast`, …) — the tree holds a dozen copies of `writer_unary_interleaved_start_id*.cpp`.
  Running the fork check as `ls` the original's directory for a `_metal2` sibling, rather than a
  tree-wide filename grep, meant none of them ever became a candidate. That distinction is easy to
  read past and is the entire defence.

- **The ⚠ two-forks warning was live, not hypothetical.**
  `copy/typecast/device/kernels/dataflow/writer_unary_interleaved_start_id_metal2.cpp` exists, is
  functionally identical, and names its accessor `tensor::output` instead of `tensor::dst`. Binding it
  would have compiled and then failed on a name that reads perfectly plausible. The canonical fork
  under `eltwise/unary/` is the one bound.

- **`Table` is a map, not a vector.** Flagged in the construct step and it landed immediately —
  `compile_time_args`, `defines` and `unpack_modes` are all built with brace-init, `emplace` or
  `operator[]` in this port, and reaching for `push_back` on any of them would have been the reflex.

- **The `[[deprecated]]` `compile_time_varargs` field turned out to be a non-event**, because the
  repo sets `-Wno-deprecated-declarations` globally (`CMakeLists.txt:209`). Worth recording only
  because "the field I need is deprecated" looks like a stop signal for a moment.

## Friction

### Gaps

1. **The GREEN precondition and a subset brief are mutually exclusive as written — HIGH.**
   The recipe's precondition is unambiguous: the audit must have produced an **overall GREEN** result,
   and "If either condition is unmet, stop. Return to the audit document. Do not improvise."
   `METAL2_PORT_BRIEF.md` opens with a box stating the op is **RED at op level**, with three of six
   factories clear. Both statements are load-bearing and they contradict, so a cold porter's first
   action on this op is either to stop with nothing delivered, or to decide unilaterally that the
   precondition means something other than what it says. I raised it with the invoker and was told to
   proceed on the subset — but an unsupervised run has no such channel, and the two readings produce
   *no port* versus *a complete three-factory port*.

   **Suggested fix:** amend the precondition to name the case the audit already produces — e.g.
   *"an overall GREEN audit, **or** a config-scoped RED whose brief is scoped to a clean factory
   subset; in the latter case the brief's scope box is the port's scope."* The audit doc's
   finding-roles section already carves this out for the auditor; the port doc's precondition has not
   caught up. This is the same seam the audit's own feedback file flags from the other side.

2. **Two of the anti-pattern self-audit sweeps assume a whole-op port and mis-fire on a subset — MEDIUM.**
   - The `cb`-name sweep says to run `grep -rnE '[Cc][Bb]_|...' <op-dir>` and *"Expect **zero** hits:
     post-port the op has no CBs, so every hit is a real leftover."* On a subset port that is false by
     construction: the three untouched factories still legitimately hold `CBDescriptor`,
     `cb_data_format`, `src0_cb_index` and so on. Run over the op directory here it returns dozens of
     hits, none of them leftovers. I scoped it to the 12 ported files (printing the denominator) and it
     came back **0 hits over 12 files**.
   - The `TT_FATAL` census's closing rule — *"A count delta **outside** the factory … is a scope
     violation on its own"* — is right, but "the factory" is singular. With three factories converted
     the census output needs reading per-file against a three-file allowlist, which the phrasing does
     not anticipate.

   **Suggested fix:** in both checks, say the scanned set is *the ported files*, not the op directory,
   and keep the denominator requirement (which is what makes a scoped sweep trustworthy). The
   denominator rule is the good half of this section and it did its job — printing *hits / files
   scanned* is what let me state a scoped pass honestly instead of quietly narrowing the check.

3. **The forced-legality scaffolding check has a false positive on any branch carrying the recipe docs — LOW, but it fires every time.**
   The check is `git diff "$BASE" | grep -nE 'METAL2_CHECKS_FORCED|DO NOT COMMIT'`, expecting no
   output. On this branch the recipe documents themselves are part of the diff, and
   `metal2_port.md` *contains* both strings (it prescribes them). So the check reports five hits with
   the working tree completely clean. Attributing them took a per-file loop. The companion check
   (`git diff --name-only "$BASE" | grep -E '^tt_metal/'`) is unambiguous and correctly reported
   nothing.

   **Suggested fix:** scope the second grep to code — `git diff "$BASE" -- '*.cpp' '*.hpp' '*.h' | grep -nE ...`
   — which is the rule's actual intent and returns a clean zero here. Given that this bulk-port effort
   iterates the docs on the same branches as the ports, this will misfire for most porters.

4. **`compile_time_varargs` has no worked example anywhere in the corpus — LOW.**
   `grep -rn compile_time_varargs ttnn/cpp/ttnn/operations/` returns only prose (this op's audit
   artifacts and pad's brief telling its porter *not* to use it). Concat is the first actual user. The
   mechanics are fully documented at the field in `advanced_options.hpp:95-110` and the generated
   accessors are exactly as described, so nothing was blocked — but two things were only learnable by
   reading `jit_build/genfiles.cpp:380-408`: the accessor is **0-based over the vararg prefix**, not
   over `kernel_compile_time_args` as a whole (so the legacy `page_size_base_idx + curr_tensor` index
   becomes plain `curr_tensor`), and there are three forms (`get_compile_time_vararg(i)`,
   `get_compile_time_vararg<i>()`, `get_num_compile_time_varargs()`).

   **Suggested fix:** the recipe mentions `get_compile_time_vararg(i)` in passing under rule 4; one
   sentence that the index is relative to the vararg block, not the CTA array, would close it. The RTA
   side already says this ("`get_vararg(0)` is the first vararg, regardless of named-arg count") — the
   CTA side just needs the same sentence.

### Confusion

5. **`KernelSpec::DFBEndpointType::PRODUCER` in the migration guide does not name a real type — LOW.**
   `migration_guide.md:288` and `:295` write `.endpoint_type = KernelSpec::DFBEndpointType::PRODUCER`.
   The enum is `KernelSpec::DFBBinding::EndpointType`, with a namespace-scope alias
   `DFBEndpointType` at `kernel_spec.hpp:230`. The port recipe's spelling
   (`DFBEndpointType::CONSUMER`) is the correct one; the guide's is a compile error. Since the recipe
   outranks the guide this cost nothing, but the guide is where a porter reads the DFB section first.

6. **"Extract `MeshTensor` at the top" runs out of API for two queries every factory here needs — LOW.**
   The guide is emphatic that the factory body should hold Metalium objects and that reaching back
   through `.mesh_tensor()` per site is not the recommended style. But `MeshTensor` has no `buffer()`,
   and concat needs `Buffer::page_size()` and `Buffer::alignment()`. The routes are
   `mesh_buffer().page_size()` and `mesh_buffer().get_reference_buffer()->alignment()`. I verified the
   translation is *exact* rather than approximate — `ttnn::Tensor::buffer()` is literally
   `mesh_buffer().get_reference_buffer()` (`ttnn/core/tensor/tensor.cpp:469` →
   `ttnn/core/tensor/storage.cpp:156`) — but that verification is the kind of thing a porter either
   does or silently assumes.

   **Suggested fix:** a two-line table in the guide's Factory-skeleton note mapping the common
   `Tensor::buffer()->…` queries to their `MeshTensor` routes, with the identity above stated so
   nobody has to re-derive it.

### Process note (mine, not the recipe's)

7. I lost one build-and-test cycle to my own shortcut. The recipe says to background the build and use
   its **exit** as the completion signal, which is *turn-durable* and therefore a real signal. I
   instead armed a waiter on a log-marker heuristic; the marker I picked appears during the install
   phase, so the wait released early and the test ran against a half-relinked `_ttnn.so`. The symptom
   was 90 failures whose compile line showed `-DKERNEL_COMPILE_TIME_ARGS=16,64,…,0,1` — the *legacy*
   positional list, which the ported factory cannot emit. That mismatch is what identified it as a
   stale binary rather than a port bug, and re-running on the finished build gave `244 passed`.
   Recorded because the diagnostic generalizes: **if a ported kernel fails with `'args' has not been
   declared` / `'dfb' has not been declared`, read the `KERNEL_COMPILE_TIME_ARGS` on the failing
   compile line.** Legacy-shaped values there mean the host you are running is not the host you
   wrote, and no amount of staring at the kernel will show it. The recipe's own instruction was
   correct and I should have followed it; a sentence naming this symptom would still help.

## Open items for downstream

### Shared kernel touches

Both are **rung 1 — reused an existing `_metal2` fork. No new file created, no fork edited, no write
into any peer directory.**

| fork bound | rung | vocabulary adopted | remaining **legacy-original** consumers |
|---|---|---|---|
| `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id_metal2.cpp` | 1 (reuse) | `dfb::out0`, `tensor::dst`; args `stick_size`, `num_sticks`, `start_id`; gates on `BACKWARDS` (concat sets no defines) | **1** — `data_movement/copy/device/copy_same_memory_config_program_factory.cpp:37` |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id_metal2.cpp` | 1 (reuse) | `dfb::out`, `tensor::dst`; args `num_pages`, `start_id`; gates on `OUT_SHARDED`, `BACKWARDS` | **23** — authoritative list and sunset plan at issue #52228 |

The RM fork already had **two** other consumers before this port
(`embedding/device/embeddings_rm_program_factory.cpp:329`,
`embedding/device/embeddings_tilized_indices_program_factory.cpp:231`), which is what makes it
read-only. Concat is now a third. **The RM donor's legacy original is one consumer away from sunset** —
when `copy_same_memory_config_program_factory.cpp` migrates, that legacy copy can be deleted and the
fork can take its name. Whoever ports `data_movement/copy` should know they are the last one.

The three concat-owned kernels converted in place are **not** shared: filename census across
`ttnn/cpp/ttnn/operations/` gives exactly one binder each, and no other op binds any concat kernel.

### Findings — preserved, not fixed

1. **A dead compute-kernel local had to be deleted, and the deletion was forced rather than chosen.**
   `device/kernels/compute/height_sharded_width_concat_two_tensors.cpp` constructed
   `DataflowBuffer output_dfb(output_dfb_id)` and never used it (pre-port line 57). The brief's
   recommendation was to bind the buffer to the writer only and *leave the dead local for the ops
   team* — that is not available: with no binding on compute there is no `dfb::output` token in that
   kernel's generated header, so the line cannot compile. It is deleted. Zero functional change (the
   local was dead), but it is a kernel-body deletion a reviewer should see named rather than discover.
   **The same forcing removed six more dead buffer-index locals**: reader slots 5-6
   (`output_transpose_cb_id`, `output_cb_id`) and writer slots 0-4 in the tiled factory. In every case
   the legacy kernel read a buffer index it never used.

2. **Dead *scalar* arguments were preserved, deliberately.** Nothing forces these, so they stay:
   `num_output_pages` in the RM kernel (CTA slot 5, read into an unused local); `tile_size` and
   `groups` in the tiled compute kernel (slots 11-12, declared and unused); `input1_num_tiles_height`
   in the tiled writer (slot 9, unused); and `input0_stride` / `input1_stride` in the tiled writer
   (computed at lines 24-25, unused — `width_len_bytes` is what the body reads). Each is a word of
   argument space and a line of kernel code. Cleaning them up is a real, separate improvement and is
   not port work.

3. **A dead CTA slot disappeared on its own, and it is not a behaviour change.**
   `concat_program_factory.cpp:214` (pre-port) baked `dst_buffer->page_size()` into RM writer CTA
   slot 1, which the legacy donor never read — it takes `stick_size` from RTA slot 1, and the slot
   existed only to fill the offset its `TensorAccessorArgs<2>()` chain expected. The fork takes no
   CTAs at all, so the slot is gone with the plumbing.

4. **RTAs that are really CRTAs — noted for a later pass, not converted.**
   In `ConcatProgramFactory`'s RM path, `stick_size` is set to
   `output.mesh_buffer().page_size()` on **every** node — the same value throughout the loop
   (`concat_program_factory.cpp` per-core loop). That is a common runtime arg wearing an RTA's clothes.
   It was **not** converted: RTA→CRTA changes dispatch semantics, and the port preserves behaviour.
   Same shape, same reason, for the `num_pages_per_block` half of the reader's vararg block, which is
   also node-invariant (`page_id_per_tensor`, the other half, genuinely varies per node). A later
   cleanup pass could move both.

5. **`ConcatS2IProgramFactory` is dead code and the evidence is mechanical.** It binds
   `.../kernels/dataflow/reader_s2i_width.cpp`, and **no file of that name exists anywhere in the
   repository** (`find` over the whole tree). Its sibling `writer_s2i_width.cpp` does exist and is
   otherwise unreferenced. The factory is reachable from `select_program_factory`
   (`concat_device_operation.cpp:32-34`, the sharded-input / interleaved-output path), so any input
   that selects it fails at kernel build. Out of scope here — the audit gated it — but it is a
   deletion candidate with proof, not a suspicion.

6. **The two gated factories remain the port's unfinished business.**
   `ConcatS2SMultiProgramFactory` and `ConcatBlockShardedProgramFactory` are blocked as
   *"DFB misuse; will need semi-manual port"*. They are untouched and still on the `descriptor`
   concept. Both bind kernels no other factory binds
   (`reader_s2s_tensor_concat.cpp`, `reader_writer_block_sharded_concat.cpp`), so whoever picks them up
   inherits no fork coordination from this port.

### Doc-evolution suggestions

7. **`compile_time_varargs` and `TensorBindingSequence` are both catalog-entry candidates**, and concat
   is now the worked example for each. The sequence in particular has a non-obvious pair of
   consequences a catalog entry could carry: the tensor-count argument becomes redundant because the
   sequence carries its own length (`std::tuple_size_v<decltype(tensor::inputs)>`), and the per-input
   `TensorAccessorArgs` CTA blocks vanish because the framework builds accessor args from the
   bindings. Both are real deletions a porter might otherwise translate by reflex. The kernel-side
   collapse is genuinely two lines:
   `make_tensor_accessors(tensor::inputs)` + `make_abstract_tensor_accessor_wrappers(...)`.

8. **A worked example of a *mixed* factory verdict would help**, as the audit's own feedback also asks.
   Concat's three-ported / three-gated split drove the scope box, the subset self-audit scoping (Gap 2),
   the per-factory plan structure, and the Outcome line's shape all at once. Each is covered
   individually; the composition is where the judgment went.

### Test coverage notes

9. **No dedicated C++ gtest exists for concat.** The recipe's recommended
   `./build/test/ttnn/unit_tests_ttnn --gtest_filter='*Concat*'` has nothing to match — the only C++
   hits are incidental uses of `ttnn::concat` inside `tensor/test_partition.cpp` and
   `tensor/test_distributed_tensor.cpp`. So the fast-fail layer the recipe recommends running *before*
   pytests does not exist for this op, and the whole no-regression signal is Python. A small gtest over
   the interleaved and the two sharded paths would give the next porter of a concat factory a
   30-second signal instead of a 7-minute one.

10. **The verification here did not cover Blackhole.** `use_single_packet_read` in the tiled factory
    branches on `arch == BLACKHOLE` for `noc_max_burst_size` (16384 vs 8192), and this run was on
    Wormhole (n150). The branch is carried over verbatim and is host-side arithmetic feeding a
    `define`, so the risk is low — but the BH value of that define was not exercised.
