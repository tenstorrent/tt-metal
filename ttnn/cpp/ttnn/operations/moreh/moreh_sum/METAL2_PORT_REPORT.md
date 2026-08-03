# Metal 2.0 Port Report — `ttnn/cpp/ttnn/operations/moreh/moreh_sum`

## Outcome

**`PORTED`** — all **six** factories (`MorehSumHFactory`, `MorehSumWFactory`, `MorehSumNCFactory`,
`MorehSumHIntFactory`, `MorehSumWIntFactory`, `MorehSumNCIntFactory`) and all **16** kernels they
bind, converted together to `ProgramSpecFactoryConcept`. Nothing left for a later pass.

Verified against the pre-port baseline captured on the same checkout (identical numbers before and
after):

| test | pre-port | post-port |
|---|---|---|
| `tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_sum.py` | 229 passed, 155 skipped | 229 passed, 155 skipped |
| `tests/ttnn/nightly/unit_tests/operations/eltwise/backward/test_backward_repeat.py` (`repeat_bw` → NC factory) | 8 passed | 8 passed |
| `./build/test/ttnn/unit_tests_ttnn --gtest_filter='*TestAsync*'` | 3 passed | 3 passed |

The 155 skips are pre-existing (`bfloat8_b not supported`, plus batch-dim configs the int test skips)
and identical in both runs. Coverage note: the parametrized shapes include both
`TILE_HEIGHT * 10 - 1, TILE_WIDTH * 10 - 1` (→ `do_mask_h` / `do_mask_w` **true**) and exact
`TILE_HEIGHT, TILE_WIDTH` (→ **false**), so *both* branches of every conditional DFB binding and
every config-scoped self-loop in this port are exercised, on the float and the int32 paths alike.
`test_moreh_sum_enable_cache` exercises the program-cache hit path (`UpdateTensorArgs`).

## Provenance

```
git log -1 --format='%h %cs %s' -- docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/
```

- **Recipe docs (this port):** `20c1692eb08 2026-08-03 docs(metal_2.0): require an explicit opt_level when porting compute kernels`
- **Audit docs (inherited):** `20c1692eb08 2026-08-03 docs(metal_2.0): require an explicit opt_level when porting compute kernels`

## TTNN ProgramFactory

### Concept realized

`ProgramSpecFactoryConcept` (plain), exactly as the audit chose — for all six factories, with one
wiring pattern. No op-owned tensors, no semaphores, no deviation to surface.

### Device-op-class edits

- **Custom `compute_program_hash` deleted:** none — the op never had one (already on the default
  reflection-based hash).
- **Pybind entry points removed:** none — `moreh_sum_nanobind.cpp:19-29` binds only
  `&ttnn::moreh_sum`; no `create_descriptor` was ever exposed, so no user-visible surface changed.
- The only edit outside a factory body is `device/moreh_sum_device_operation.hpp`: the six
  `static tt::tt_metal::ProgramDescriptor create_descriptor(...)` declarations became
  `static ttnn::device_operation::ProgramArtifacts create_program_artifacts(...)`, and
  `<tt-metalium/program_descriptors.hpp>` was replaced by `"ttnn/metal_v2_artifacts.hpp"`. Nothing in
  `validate_on_program_cache_miss`, `compute_output_specs`, `create_output_tensors` or
  `select_program_factory` was touched.

### Open items

- **Relaxation candidates: none applied, one latent.** Every `TensorParameter` keeps strict
  `TensorSpec` matching. The op has no custom hash to mine, so no relaxation could be pending. The
  one shape that *would* be a relaxation candidate if the op ever gains a custom hash is the
  forced-`fp32_dest_acc_en` anomaly below.
- **`fp32_dest_acc_en` is forced but still hashed** (`moreh_int_sum_h_program_factory.cpp:54-57`,
  `moreh_int_sum_w_program_factory.cpp:56-59`, `moreh_int_sum_nc_program_factory.cpp:52-55`). The
  three int factories override the caller's value to `true`, but the *un-forced* value still rides
  `operation_attributes.compute_kernel_config` into the default program hash — so two INT32 calls
  differing only in `fp32_dest_acc_en` occupy two cache entries for a byte-identical program. The
  port preserves this exactly (it carries the *forced* value into `enable_32_bit_dest`, matching
  legacy); flagged for the owner as a cache-efficiency item, not a correctness one.

## Handoff points

**None.** Nothing in this port required a change outside the op's own directory, and no construct
resisted conversion:

- **Zero donor-side edits**, as the audit predicted. Every out-of-directory callee took either
  `DataflowBuffer` by value (`generate_mask_h` / `generate_mask_w` / `generate_mm_scaler`,
  `copy_tile_to_dst` / `pack_tile_from_dst`) or a `uint32_t` CB id as an NTTP
  (`dataflow_kernel_lib::prepare_zero_tile<>`, `calculate_and_prepare_reduce_scaler<>`,
  `compute_kernel_lib::reduce<>`), and a `dfb::name` token satisfies both natively. The broadly-shared
  headers under `ttnn/cpp/ttnn/kernel/` and `ttnn/cpp/ttnn/kernel_lib/` were read but not modified.
- **No boundary-rule assumption violation** — no out-of-op call site needed a `sem::` or `tensor::`
  handle (the op has no semaphores, and both accessors are consumed inside its own kernels).
- **No `_metal2` fork created and none needed.** The two intra-op shared kernels
  (`moreh_sum_nc_impl_kernels/reader_moreh_sum_nc.cpp`, `writer_moreh_sum_nc.cpp`, each bound by
  `MorehSumNCFactory` *and* `MorehSumNCIntFactory`) had **both** binders converted in this change, so
  the shared-kernel Caution's rungs never fired: nothing was left behind on the legacy API, so no
  fork, no pointer comment, and no sunset list. See *Open items* for the durable record.
- **No capitulation.** No blocked construct, no GlobalCircularBuffer, no Case-2 binding, no
  `get_cb_tiles_acked_ptr`-class API.

## Successes

- **[Hardware configuration → Compute kernels](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md)
  — the "derive `unpack_modes` from *this* op's legacy vector, don't copy a sibling" warning fired
  correctly, and it was the highest-stakes line in the port.** The merged `moreh_mean` port is this
  op's structural twin and its W factory sets `UnpackMode::UnpackToSrc` — because *moreh_mean's*
  legacy left `unpack_to_dest_mode` all-`Default`. `moreh_sum`'s W factory legacy does **not**: it
  sets `unpack_to_dest_mode[CBIndex::c_24] = UnpackToDestFp32` under `fp32_dest_acc_en`
  (`moreh_sum_w_program_factory.cpp:184-187`, pre-port), the same as its H sibling. Copying the twin
  would have silently flipped a precision/perf tradeoff with no compile error and no test signal.
  Ported value: `UnpackToDest` in both H and W (`moreh_sum_h_program_factory.cpp:207-213`,
  `moreh_sum_w_program_factory.cpp:208-214`), no entry in the four factories whose legacy vector was
  all-`Default`.
- **[Compiler options](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md)
  — "a genuinely absent `KernelDescriptor::opt_level` is *not* 'no setting'".** `grep -n opt_level`
  over the whole op returned zero hits, which reads as "nothing to carry over"; the recipe's rule 2
  is the only reason all six compute `KernelSpec`s got an explicit
  `KernelBuildOptLevel::O3` instead of silently dropping to Metal 2.0's `O2` default. Applied at the
  `make_compute` lambda in each of the six factories.
- **[Two-toucher DFB → assign 1P+1C](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md)
  — its *Constraint* paragraph (distinguish the disjoint-node work split).** Every factory here
  instantiates compute twice, which pattern-matches the dual-instance shape at a glance. The
  Constraint note is what made the distinction explicit: the two instances cover **disjoint** core
  groups, so each node sees one compute instance and every compute-bound DFB stays an ordinary
  single-role binding — no 1P+1C question, and emphatically no multi-binding flag (zero in the diff).
- **The endpoint-census "re-derive, don't transcribe" instruction.** My census agreed with the brief
  on every row, but re-deriving it is what surfaced *why* `mask_h` is a live toucher of the H float
  compute kernel in both configurations: lines 36 and 95 of `moreh_sum_h.cpp` guard with a **plain**
  `if (do_mask_h)` (compiled either way) while line 54 uses `if constexpr`. Transcribing the
  brief's disposition without reading the guards would have produced the same bindings by luck, not
  by reasoning — and the neighbouring `masked_input` case shows how thin that margin is.

## Friction

### Gaps

- **No pattern entry for a runtime-selected DFB handle.** Three kernels here pick *which* DFB to act
  on at runtime: `moreh_sum_w.cpp` reassigns a mutable `cb_input` from the input DFB to the
  masked-input DFB mid-loop (lines 19, 44, 92 post-port) and drives temporaries
  `DataflowBuffer(cb_input)` off it; `moreh_int_sum_nc.cpp:41` picks its pack target with a ternary
  over two tokens. The recipe's *Watch for* text says only that `dfb::name` tokens are static and to
  "flag it if the shape resists a clean rewrite" — but it never states the resolution, and the audit
  filed the same absence against the *audit* template (its Recipe note 5). The resolution is
  mechanical once you read the declaring header: `DFBBindingToken`
  (`tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:46-58`) is a **single non-template struct** with a
  `constexpr operator uint32_t`, so (a) a `uint32_t` local initialised from one token and reassigned
  to another is legal, and (b) a **ternary between two tokens has a common type** and converts
  cleanly. Both DFBs simply have to be bound to that kernel in every configuration so both tokens
  exist. Suggest a short catalog entry — *"Runtime-selected DFB handle → keep the legacy
  `uint32_t`-valued local; bind every candidate DFB"* — since a porter who assumes the tokens are
  distinct types will reach for something worse (a `#ifdef` ladder, or two `DataflowBuffer` objects).
  This was the single sharpest thing in the op and the docs left the last step to the porter.
- **Whitelist rule 1's "the port adds exactly two headers" over-counts for an op already on Device
  2.0.** All 16 kernels here already used `DataflowBuffer`, so the port added exactly **one** header
  (`experimental/kernel_args.h`); `api/dataflow/dataflow_buffer.h` was already present in 15 of them,
  and `moreh_int_sum_nc.cpp` uses `DataflowBuffer` *without* including it at all (it arrives
  transitively via `ttnn/kernel/compute/moreh_common.hpp`) — which I left alone as the
  minimal-diff choice rather than "completing" the rule. One clause in rule 1 ("if the kernel is
  already Device 2.0, `dataflow_buffer.h` is already there — add only `kernel_args.h`") would settle
  whether adding it anyway is expected.

### Confusion

- **The build snippet is working-directory-sensitive in a way that fails silently.** My first build
  invocation did nothing: an earlier tool call had `cd`'d into the op directory, the shell's working
  directory persisted, and `./build_metal.sh --build-tests > log 2>&1` resolved to nothing — while
  the `echo "BUILD EXIT: $?"` I had appended reported the *echo's* status, so it looked like a clean
  exit-0 build with an empty log. Cheap to lose 10 minutes to. Suggest the §Running builds and tests
  snippet use `cd "$TT_METAL_HOME" && ./build_metal.sh --build-tests > … 2>&1` (or an absolute path),
  and note that `$?` after a redirect-plus-`echo` chain is not the build's status.
- **§Running builds and tests prescribes a Sonnet subagent for log reading; this session's operating
  rules forbade spawning agents.** I kept the logs out of context the other way — `grep -cE "error:"`
  for the verdict, then a narrow slice around any hit — which achieved the same context hygiene. Worth
  one sentence that the subagent is *a* way to keep logs out of context, not the only sanctioned one,
  so a porter under a no-subagent policy doesn't read the recipe as blocking.
- **Near-miss on the reader's dead `HtWt` CTA.** `reader_moreh_sum_h.cpp` reads `HtWt` and never uses
  it (the column loop strides by `Wt`). Naming CTAs one-by-one makes a dead argument conspicuous in a
  way the positional list did not, and the pull to just drop it is real. §Scope discipline is
  unambiguous that it stays — but the *specific* case of "a CTA the kernel provably ignores" isn't
  named in the dropped-plumbing list, whose entries are all things that legitimately disappear.
  Preserved verbatim, with a comment at the emission site saying why.

## Open items for downstream

- **Shared kernel touches — nothing outstanding.** `moreh_sum_nc_impl_kernels/reader_moreh_sum_nc.cpp`
  and `writer_moreh_sum_nc.cpp` are *intra-op* shared (`MorehSumNCFactory` + `MorehSumNCIntFactory`).
  Rung taken: **neither** — both binders converted in this change, so the sources were
  Metal-2.0-ified in place with no consumer left behind, no `_metal2` fork, and no pointer comment.
  Remaining unmigrated consumer op directories: **none**. Recorded here because this is the only
  durable home for the fact.
  - The one constraint that outlives the port: those two sources are still shared, and the two
    factories' DFB sets **differ** — `MorehSumNCIntFactory` allocates no zero buffer and emits no
    `USE_FPU`. Anyone editing the shared reader must keep the zero-tile block preprocessor-gated, or
    the int32 factory will reference a `dfb::zero` token it does not bind.
- **Pre-existing dead / vestigial items, preserved verbatim** (each a candidate for a separate
  cleanup PR, none touched here):
  - `HtWt` named CTA in `reader_moreh_sum_h.cpp:21` — read, never used.
  - `dst1` in `moreh_sum_nc.cpp:20` — declared, never used.
  - `[[maybe_unused]] num_tiles` in `moreh_int_sum_h_program_factory.cpp:36` and
    `moreh_int_sum_w_program_factory.cpp:37`.
  - `moreh_sum_h.cpp:11-13` / `moreh_sum_w.cpp:12-14` read `Ht` / `Wt` / `NC` into **non-`constexpr`**
    locals while their int siblings use `constexpr`, demoting compile-time-known values to runtime
    branches (`is_h_single_tile`, `is_w_single_tile`). A missed unroll, not a bug.
- **L1: unconditional mask / intermediate buffers.** `mask_h` / `mask_w` are allocated in every
  configuration even though `do_mask_*` is known at spec-construction time, and per the ops team's
  decision H's `masked_input` is likewise unconditional (mirroring the merged `moreh_mean` port).
  Under `!do_mask_*` that is one to two tiles of L1 per core with no producer. Pre-existing waste,
  now *visible* in the spec: making them conditional is a mechanical application of the
  conditional-DFB `#ifdef` pattern whenever an L1 pass wants the tiles back. Note the coupling that
  makes it non-trivial for `masked_input`: `moreh_sum_h.cpp:20` constructs the buffer object outside
  the `do_mask_h` guard, so dropping the binding requires `#ifdef`-gating that construction too.
- **Hardcoded tile geometry in the int32 writers.** `writer_moreh_int_sum_h.cpp:31-35` and
  `writer_moreh_int_sum_w.cpp:30-34` fold sub-tile faces with literal `16`, `4`/`8`, `256`/`512`
  strides, assuming a 32×32 int32 tile and a specific face layout, with no assert guarding it.
  Untouched by the port (the fold operates on DFB memory via `get_read_ptr()`, never on tensor
  memory — no Case-2 binding). Also flagged by the audit's Misc anomalies.
- **Test coverage.** The confirmed set covers both mask configurations on all six factories and the
  program-cache hit path. Two gaps a future pass could close: the int32 factories are exercised only
  by `test_moreh_sum_integer` (5 shapes × 4 dims, no `compute_kernel_options` sweep), and nothing in
  the confirmed set asserts *performance*, so the `opt_level` / `unpack_modes` settings this port had
  to carry by hand have no automated regression net — exactly the silent-failure class the recipe
  warns about.
- **Sibling carry-over.** `moreh_sum_backward` (`ttnn/cpp/ttnn/operations/moreh/moreh_sum_backward/`)
  is still on the descriptor concept and is the obvious next port in this family: it is a single
  factory, its kernels are already Device 2.0, and its shape closely follows the NC factory here.
  It is untouched by this change and its tests (in the same `test_moreh_sum.py` file) pass unchanged.
