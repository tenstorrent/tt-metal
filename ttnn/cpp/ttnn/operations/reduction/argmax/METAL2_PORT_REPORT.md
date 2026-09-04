# Metal 2.0 Port Report — `reduction/argmax` (`ArgMaxDeviceOperation`)

## Outcome

**`PORTED`** — both in-scope factories converted to `ProgramSpecFactoryConcept` and their tests
pass: `ArgMaxSingleCoreProgramFactory` (all three runtime-selected kernel sources) and
`ArgMaxMultiCoreProgramFactory`.

`ArgMaxNCDeviceOperation` / `ArgMaxNCProgramFactory` remains on the legacy imperative API, out of
scope by the audit gate, and is **untouched**. `ttnn::argmax` now dispatches to a Metal 2.0
device-op and a legacy one side by side, which is expected.

Verification (all with the host-side legality checks forced on and *proven* live — see Successes):

| Layer | Result |
|---|---|
| `./build_metal.sh --build-tests` | SUCCESS, 0 errors; `_ttnncpp.so` relinked; `nm` confirms both factories export `create_program_artifacts` and no longer export `create_descriptor` |
| `unit_tests_ttnn --gtest_filter='*Argmax*:*ArgMax*:*argmax*'` | **6/6 passed** (incl. `TestGenericOpArgmaxSingleCore`, the out-of-directory consumer of the retained legacy kernel) |
| `tests/ttnn/unit_tests/operations/reduce/test_argmax.py` | **66 passed**, 0 failed |
| `tests/ttnn/nightly/.../reduction/test_reduction_op_corners.py` + `test_generic_ops.py` | **937 passed, 8 skipped**, 0 failed |

The 8 skips are argmax cases the tests skip by their own capability guards (`dim=None` with TILE
layout; non-last-dim reduction) — conditions evaluated before the op is called, so they are
unrelated to the port.

## Provenance

- **Recipe docs (this port):** `b73b958088a 2026-08-24 docs(metal_2.0): a run in flight freezes the kernel sources`
- **Audit docs (inherited):** `b73b958088a 2026-08-24 docs(metal_2.0): a run in flight freezes the kernel sources`

## TTNN ProgramFactory

### Concept realized

`ProgramSpecFactoryConcept` (base), as the audit chose. Neither factory had an
`override_runtime_arguments`, so the framework refreshes the tensor bindings on a cache hit and
each factory writes exactly one method. Nothing about the concept choice was revisited.

Both factories already lived in a `program_factory_t` variant
(`argmax_device_operation.hpp:31`), so the direct-descriptor exception did not apply — the port
was a method swap inside the existing structs.

### Device-op-class edits

- **Pybind entry points removed:** none. `argmax_nanobind.cpp` binds only the user-facing
  `ttnn::argmax`; there was no pybound `create_descriptor`, so this port carries **no
  user-visible API change**.
- **Custom `compute_program_hash`:** none — the op uses the default reflection-based hash, and
  no backdoor `attribute_values` / `to_hash`. Nothing to leave alone, nothing touched.
- The only device-op-class edit is the two factory-method declarations in
  `argmax_device_operation.hpp` (`create_descriptor` → `create_program_artifacts`) and the
  matching include swap. That is the port itself, not one of the three sanctioned exceptions.

### Open items

- **Relaxation candidates:** none identified. Both `TensorParameter`s are left strict, matching
  the audit's `relaxation: none`. The two TILE readers do reason explicitly about padded-vs-
  logical shape, but they take both as compile-time arguments and bake them into the spec, so a
  `dynamic_tensor_shape` relaxation would be actively wrong here, not merely unexercised.
- **Concept fit:** no friction. The single-core factory ends up with **zero** runtime arguments
  once the two address RTAs become `TensorBinding`s, so its `ProgramRunArgs` carries only
  `tensor_args` and it declares no `KernelRunArgs` entry at all. That is a nice demonstration of
  the binding model paying for itself, and it worked exactly as the header documents.

## Handoff points

**No capitulation, no boundary-rule violation, no kernel-lib gap, no framework gap.** No call
site required a `sem::` or `tensor::` handle outside the op directory; no compute kernel is
involved (both factories build DM kernels only), so the Case-2 / compute-`TensorBinding` block
never came up.

One item worth routing, though it is environmental rather than API:

- **[Environment / invoker] The checkout could not build as handed over.**
  `tt_metal/third_party/umd` was checked out ~85 commits behind the commit this branch records
  (`4ed96fb3` vs. the recorded `0b263b2c400`), which is the `M tt_metal/third_party/umd` present
  in the session's opening `git status`. `tt_metal/llrt/tt_cluster.cpp:363,365,372` call
  `umd::restore_default_tdp_limit`, `umd::set_tdp_limit`, and
  `FirmwareInfoProvider::get_tdp_limit`, none of which exist at that older commit, so every build
  failed in `llrt` long before reaching any TTNN op. The submodule had no local edits and the
  recorded commit was already in its object store, so `git submodule update --init
  tt_metal/third_party/umd` restored it cleanly and reversibly (it now matches HEAD and does not
  appear in the diff at all). Flagging it because it is invisible from the port diff and because
  anyone else picking up this branch will hit the same wall.

## Successes

- **[Compiler options](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md) — the "answer it mechanically" instruction did its job.**
  The section insists on `grep -n opt_level` rather than reading `config`. Doing that returned
  nothing across both factories, which — combined with the section's table — resolves to `O2` on
  a reader/DM descriptor and therefore needs no action, since Metal 2.0 also defaults to `O2`.
  The compute-kernel `O3` trap the section is really aimed at cannot fire here: both factories
  build DM kernels only. Worth recording that the mechanical check produced a confident *"nothing
  to do"* rather than a guess.

- **[Hardware configuration](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md) — "match on the *values*, not the constructor spelling" caught a real trap.**
  The multi-core factory's legacy config reads
  `DataMovementConfigDescriptor{.processor = RISCV_1, .noc = NOC::RISCV_1_default}`. The name
  `RISCV_1_default` reads like "the default for the RISCV_1 kernel", i.e. the reader default —
  and `create_reader_datamovement_config()` was the obvious reach. But `NOC::RISCV_1_default == NOC_1`
  (`kernel_types.hpp:33-38`), and the *reader* default is `NOC_0`. Routing this through the reader
  helper would have flipped the NOC on every multi-core dispatch: no build error, no test
  failure, a silent bandwidth regression. Replicated verbatim instead as a raw
  `DataMovementGen1Config` at `argmax_multi_core_program_factory.cpp:428`. This is precisely the
  failure mode the section says it exists to prevent, and the section is what stopped it.

- **[CB→DFB whitelist §A](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/cb_dfb_api_whitelist.md) — the `constexpr` carve-out outranked the brief, correctly.**
  See the Friction entry below; the whitelist's rule is stated precisely enough that the
  disagreement was resolvable by reading one constructor declaration.

- **[Two-toucher / endpoint-assignment procedure](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md) — "re-derive, don't transcribe" confirmed the brief rather than correcting it.**
  Running the census independently over all four kernels found what the brief said: every DFB has
  exactly one toucher per node, and every touch is a bare `get_write_ptr()` peek with no FIFO op
  and no `evil_set_*` cursor surgery anywhere. Ten self-loops, no `allow_instance_multi_binding`.
  The procedure's explicit warning that *cursor surgery is not a peek* is what made the sweep
  conclusive rather than approximate — it named the one thing that would have changed the answer.

- **The `_metal2` fork convention held up under a case it was not written for.**
  The shared-kernel Caution is written around *factories* that will not co-migrate. Here the
  stranded consumer is a **gtest** (`test_generic_op.cpp:126`) that file-path-instantiates
  `reader_argmax_interleaved.cpp` from a hand-built `ProgramDescriptor` — a shape that cannot be
  re-pointed, because `generic_op` cannot supply named bindings at all. Rung 2 applied unchanged
  and worked: the gtest still passes, untouched, on the legacy copy.

## Friction

### Gaps

- **The brief and the CB→DFB whitelist give opposite instructions for `get_dataformat`, and the
  brief is wrong.** `METAL2_PORT_BRIEF.md` says to move all five sites onto the DFB member
  getter, reasoning that "`DataflowBuffer::get_dataformat()` is `constexpr`
  (`dataflow_buffer.h:279`), so this works" for the NTTP uses. It does not.
  [Whitelist §A](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/cb_dfb_api_whitelist.md)
  and [recipe kernel-side rule 7](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md)
  state the actual rule: a member getter cannot produce a constant expression, because the
  *getter* being `constexpr` is irrelevant when no `DataflowBuffer` object can be — its only
  usable constructor, `DataflowBuffer(uint16_t)` at `dataflow_buffer.h:113`, is a non-`constexpr`
  out-of-line declaration. So a legacy-`constexpr` metadata read keeps the free-function form
  with the binding token. Followed the whitelist; all five sites are
  `constexpr DataFormat … = get_dataformat(dfb::…)` at
  `reader_argmax_interleaved_metal2.cpp:49`, `reader_argmax_interleaved_multicore.cpp:299,316`,
  `reader_argmax_tile_layout.cpp:51`, `reader_argmax_tile_layout_h.cpp:44`. That is also the form
  landed ports on `main` already use (e.g. `data_movement/scatter`,
  `experimental/reduction/integral_image`).

  *Suggested fix:* the audit's own *Recipe notes* #1 asked for `get_dataformat` to be added to a
  sanctioned list, and the resulting brief text over-corrected into a member-getter instruction.
  The audit template should route a `constexpr`-site metadata read to the whitelist's carve-out
  by name rather than restating the reasoning, since restating it is where the error entered.
  These are **Gen1-only token-form sites** (the token's `uint32_t` conversion is documented as
  such), so they are Quasar-uplift debt, recorded here as §A asks.

- **The anti-pattern self-audit's `cb`-name sweep has no defined answer when the port creates a
  `_metal2` fork.** The checklist says to run the sweep over `<op-dir>` and expect **zero** hits,
  on the reasoning that "post-port the op has no CBs". That reasoning does not survive rung 2:
  the whole point of a fork is that the legacy copy *stays in the directory*, still full of
  `CircularBuffer` and `cb_*`, and (here) so does an entirely out-of-scope second DeviceOperation
  the audit gated. Reported both ways instead — **0 hits over the 10 ported files**, with the
  residue enumerated as exactly `reader_argmax_interleaved.cpp` (the retained fork) plus the four
  NC-half files. *Suggested fix:* have the checklist define the sweep's denominator as the ported
  file set, and require the residue to be enumerated and attributed, rather than asking for a
  zero that a correct rung-2 port cannot produce.

- **`ProgramRunArgs` has no worked example of a kernel with no runtime arguments.** The
  single-core factory ends with zero RTAs and zero CRTAs. The `program_run_args.hpp` comment does
  cover it ("except for kernels that have no runtime or common runtime arguments"), but every
  example in the migration guide and the recipe shows a populated `kernel_run_args`, so it took a
  header read to be confident that omitting the entry entirely — rather than pushing an empty
  `KernelRunArgs{.kernel = READER}` — was the intended shape. One sentence in the migration
  guide's `ProgramRunArgs` section would settle it.

### Confusion

- **"Preserve the value mapping, not the label" is the right rule for named CTAs, but no doc
  states it.** The multi-core factory deliberately swaps start/end coordinates — the host comment
  reads `// end comes before start for NOC1` — so the kernel's `start_core_x0` has always
  received `end_core0.x`. Under positional CTAs that is invisible; under *named* CTAs it produces
  `{"start_core_x0", static_cast<uint32_t>(end_core0.x)}`
  (`argmax_multi_core_program_factory.cpp:353`), which reads like a transcription bug and is the
  faithful port. The alternative — renaming the kernel's variables so the names line up — would
  be kernel-logic surgery outside the whitelist and would obscure the NOC1 convention the
  original comment documents. Recorded the reasoning at the emission site so the next reader does
  not "fix" it. A line in the recipe's *Dropped Plumbing* / naming guidance saying that naming a
  CTA never licenses re-pairing name to value would have made this a non-decision.

- **The recipe's "build in the background, read the log with a subagent" pattern silently
  tolerates a broken exit code.** Piping `build_metal.sh` into `tail` to bound the log makes `$?`
  report `tail`, so the harness reported the first build as exit 0 when ninja had in fact stopped.
  What actually caught it was the memory-driven habit of checking `nm` on the relinked library —
  the `.so` was still dated a month earlier and still exported `create_descriptor`. The recipe
  already warns that `build_metal.sh` can return 0 over a failed unity build; the pipeline
  variant is a second, independent way to manufacture the same false green. *Suggested fix:*
  show the redirect form (`> log 2>&1`) explicitly rather than a pipeline, and add "confirm the
  library was relinked" to the build step.

## Open items for downstream

### Shared kernel touches

- **`ttnn/cpp/ttnn/operations/reduction/argmax/device/kernels/reader_argmax_interleaved.cpp`** —
  *lent* (an out-of-directory consumer binds it).
  - **(a) Kernel path:** as above.
  - **(b) Rung taken: 2 — created the fork.** New file:
    `ttnn/cpp/ttnn/operations/reduction/argmax/device/kernels/reader_argmax_interleaved_metal2.cpp`.
    The pointer comment landed at the top of the legacy original and is that file's **only**
    change. Resolution confirmed with the invoker before any kernel edit (audit *Questions* #1,
    option (a)).
  - **(c) Remaining unmigrated consumer:** `tests/ttnn/unit_tests/gtests/test_generic_op.cpp:126`
    (`TTNNFixtureWithDevice.TestGenericOpArgmaxSingleCore`), which builds a `ProgramDescriptor`
    around the kernel with the CTA layout (indices 0-7 plus two `TensorAccessorArgs` blocks) and
    the RTA pair (`src_buffer->address()`, `dst_buffer->address()`) hardcoded at `:105-121`.
    **Sunset condition:** the legacy copy can be deleted once that gtest is retired or moved off
    `ttnn::generic_op`. It cannot simply be re-pointed at the fork — `generic_op` /
    `ProgramDescriptor` has no way to supply Metal 2.0 named bindings.

  Note this is a **test**, not an op, so it is invisible to the borrowed-kernel inventory (which
  only looks outward, at what *this* op binds). The audit's *Recipe notes* #4 already proposes
  adding a "who else instantiates my kernels" grep to that step; this port is a concrete instance
  of why. Nothing else in the repo binds any of the other three in-scope kernels, so those three
  were converted in place.

### Findings routed to the ops team (observed, deliberately **not** fixed)

The audit already listed items 1-4 under *Misc anomalies*; they are repeated here with their
**post-port** locations, because the port changed how two of them present.

1. **Dead CTA — `src_page_size` in the multi-core reader.** The factory emits it
   (`argmax_multi_core_program_factory.cpp:341`) and the kernel never reads it; reads are sized
   from the `src_read_size` runtime argument instead. **This is now visible in a way it was not
   before:** under positional CTAs it was a silent hole at index 4, and it is now a *named* CTA
   with no kernel-side reader, which a reader of the factory can spot directly. Kept and named
   rather than deleted, per the brief.
2. **Dead CTA — `dst_page_size` in both TILE readers.** Same shape: emitted at
   `argmax_single_core_program_factory.cpp:76` (RM branch, where it *is* read) and `:98` (TILE
   branch, where neither TILE reader reads it — both derive the write size from
   `output_page_elements` in `write_to_output`). One shared argument builder serving two reader
   families is what leaves the hole. Kept and named.
3. **A core index narrowed through `bool`.**
   `reader_argmax_interleaved_multicore.cpp:260` reads
   `constexpr uint32_t reduce_core_id = (bool)get_arg(args::reduce_core_id);`. CTA
   `reduce_core_id` is the reducer's **core index**, not a flag. It is 0 today so the cast is
   inert, but the factory explicitly anticipates changing it — *"We can do perf optimization by
   tuning this in the future"* (`argmax_multi_core_program_factory.cpp:283`). The first non-zero
   value collapses to 1 and both the `is_reduce_core` test and the worker-skip test silently pick
   the wrong core. Carried across verbatim, cast included.
4. **Dummy `(0,0)` core-range arguments for the single-group case.**
   `argmax_multi_core_program_factory.cpp:292` substitutes
   `CoreRange(CoreCoord(0,0), CoreCoord(0,0))` for the absent second group, so
   `start_core_*1` / `end_core_*1` describe core (0,0) rather than "no group". Inert, because
   every use is guarded by `num_cores1 > 0`, but it hands the kernel plausible-looking
   coordinates for a group that does not exist. Preserved.
5. **The NOC1 start/end argument swap is undocumented at the kernel.** The host knows why
   (`// end comes before start for NOC1`); the kernel just declares `start_core_x0` and feeds it
   to `set_multicast` as a start coordinate. The swap is correct but is stated only on the host
   side, and only in the factory the arguments happen to come from. The port added an explanatory
   comment at the emission site; a matching note at the kernel's declaration would be a real
   improvement, but writing one is an edit to kernel documentation that the port did not need,
   so it is left here.

### Other notes

- **Test-coverage note.** `test_argmax.py` plus the two nightly reduction files cover all four
  kernel configurations (RM last-dim, RM reduce-all, TILE dim=W, TILE dim=H) and both multi-core
  paths. The brief asked specifically for a multi-core check with an odd `red_dim` and
  `sub_core_grids` unset, to exercise the two core groups at *different* `SRC` sizes and confirm
  the shared `RED_IDXS` / `RED_VALS` still land at one uniform L1 address; the suites include such
  shapes and pass. The underlying property was also confirmed by reading the allocator rather
  than inferred from the green run — `ProgramImpl::allocate_dataflow_buffers`
  (`tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:2505-2571`) assigns one address per DFB,
  taken as the max region-end across every core range it spans, and writes that same address to
  every core, mirroring the legacy `allocate_circular_buffers` behaviour. The port also preserves
  the legacy DFB declaration order, since the allocator walks `ProgramSpec::dataflow_buffers` in
  user order.
- **Pre-port baseline — a deviation worth disclosing.** The recipe asks for the numerical
  baseline *before* the first kernel edit. Build and test commands were unavailable at the start
  of this session (blocked by the environment's permission layer), so the kernels were converted
  first and the baseline was left recoverable via `git stash` rather than captured up front. It
  was never needed: the post-port run is green across every suite, which rules out a regression
  without a comparison point. Had anything failed, the baseline would have been taken by stashing
  the port. Noting it because the recipe's rule is sound and the ordering here was forced, not
  chosen.
- **Per-op carry-over.** `ArgMaxNCDeviceOperation` is the obvious next port for this directory and
  is blocked only on its `ProgramDescriptor` migration (audit gate; every other gate is already
  clear for it). When that lands, its two CBs are a textbook plain 1:1 and the re-audit should be
  short. Whoever takes it should re-derive the CB census rather than trusting the audit's
  provisional one, since the migration rewrites the very factory it describes.
