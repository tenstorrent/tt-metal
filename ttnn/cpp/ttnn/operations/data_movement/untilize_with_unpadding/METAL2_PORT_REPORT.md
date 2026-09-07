# Metal 2.0 Port Report — `data_movement/untilize_with_unpadding`

## Outcome

**`PORTED` — conversion complete for all five factories and every configuration; NOT YET VERIFIED.**

All five factories now satisfy `ProgramSpecFactoryConcept`, all 8 op-owned writer kernels are
converted, 6 existing `_metal2` forks are bound and 4 new ones created. The static
[anti-pattern self-audit](#anti-pattern-self-audit-results) passes in full, including the `TT_FATAL`
census.

**The build and test steps could not be run in this environment** — `./build_metal.sh --build-tests`
was refused by the session's command sandbox on every invocation (both foreground and background
form). So this port has **not been compiled and no test has been run**, and the recipe's own standard
("the factory converted *and its tests pass*") is not met. Treat the `PORTED` label as *conversion
complete, verification outstanding*: the next step is exactly the recipe's
[Verification](#verification-status) section, unchanged, in a shell that can run the build. The
forced-legality scaffolding is in the working tree ready for that run and is **excluded from the
commit** (see [Handoff points](#handoff-points)).

This is not a capitulation: nothing about Metal 2.0 failed to express this op, and no construct
required a workaround. The gap is a harness permission, not a fit gap.

## Provenance

- **Recipe docs (this port):** `9c1a0466220 2026-09-07 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`
- **Audit docs (inherited):** `9c1a0466220 2026-09-07 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`

## TTNN ProgramFactory

### Concept realized

`ProgramSpecFactoryConcept` (plain) on **all five** factories, exactly as the audit chose. No
re-decision, nothing surfaced back to the invoker. Each factory's `create_descriptor` became
`create_program_artifacts` returning `ttnn::device_operation::ProgramArtifacts{spec, run_params}`;
`op_owned_tensors` is left defaulted (the op allocates no device tensors of its own).

The op already had a `program_factory_t` variant, so [exception 3](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/ttnn_factory.md#3-give-a-direct-descriptor-op-a-conventional-program-factory)
(direct-descriptor conversion) did not apply — the port is a method swap inside the existing structs.

### Device-op-class edits

- **Pybind entry points removed:** none. `untilize_with_unpadding_nanobind.cpp` binds only the
  user-facing op; no `create_descriptor` was ever exposed. **This port carries no user-visible API
  change**, as the brief predicted.
- **Custom `compute_program_hash`:** none — the op uses the default reflection-based hash, and there
  is no backdoor `attribute_values` / `to_hash` either. Nothing to preserve, nothing touched.
- **`override_runtime_arguments`:** none to translate.
- `device/untilize_with_unpadding_device_operation.cpp` and `untilize_with_unpadding.cpp` are
  **byte-identical** to their pre-port state.

### Open items

- **Relaxation candidates:** none identified. The audit declared `none` on all five rows, and nothing
  during construction suggested a kernel that would tolerate a relaxed `TensorSpec` match. Strict
  matching retained everywhere.
- **Capability the op would benefit from:** a spec-side twin of the shared `push_buffer_set` helper —
  see [Open items for downstream](#open-items-for-downstream).

## Handoff points

1. **Build and test could not be run — verification is outstanding.** `./build_metal.sh --build-tests`
   was refused by this session's command sandbox. Nothing in the port depends on that refusal; the
   next session should run [Verification](#verification-status) as written. **Highest-priority item in
   this report** — the diff is unproven.

2. **The forced-legality scaffolding is in the working tree and must not be committed.**
   `tt_metal/impl/metal2_host_api/program_run_args.cpp` and `program_spec.cpp` carry
   `skip_validation = false;  // TEMP: … DO NOT COMMIT.` at all **9** sites `grep -n 'bool skip_validation'`
   named, plus one `METAL2_CHECKS_FORCED` marker per file (in `SetProgramRunArgs` and
   `BuildProgramFromSpec`). They are deliberately excluded from the commit. Re-apply them before the
   verification run, and confirm **two** markers appear in the test log before trusting any green.

3. **Audit gap — the brief's shared-kernel table missed a *lent* kernel.** The brief lists
   `device/kernels/dataflow/writer_unary_stick_layout_wh_multicore.cpp` among "8 op-owned writers …
   none is dead code" and does not flag it as shared. It is: `data_movement/untilize`'s block factory
   binds it by full path
   (`ttnn/cpp/ttnn/operations/data_movement/untilize/device/factories/untilize_multi_core_block_program_factory.cpp:150-152`).
   Converting it in place would have broken that op. Handled as a rung-2 shared kernel (fork beside
   the original, pointer comment added). **Owner: the audit tooling / next auditor** — the census in
   the audit brief appears to have been run only for kernels *outside* the op directory, so the
   *lent* direction (a kernel inside the op's own directory that other ops bind) was not swept. That
   is the exact failure mode the [shared-kernel Caution](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#caution-porting-a-shared-kernel)
   warns about ("Nothing about the path warns you").

4. **`num_runtime_varargs_per_node` — a `[[deprecated]]` API this port newly depends on.** The
   MultiCoreInterleaved writer's `BlockRep` payload is a genuinely per-core-variable-length vararg
   block (5 values per run, run count varying per core and per shape), and the *only* construct that
   reproduces the legacy per-core RTA layout exactly is the per-node vararg-count override on
   `KernelAdvancedOptions`. Its header comment says *"This feature is truly bizarre. It will be
   removed from the API once existing uses are refactored to avoid it."* — this port adds an existing
   use. **Owner: the Metal 2.0 API team.** The alternatives considered and rejected: declaring a
   scalar `num_runtime_varargs` at the per-core maximum and zero-padding every shorter node (changes
   the dispatch payload size on most cores, which the port is not entitled to do), or restructuring
   the kernel's group walk (kernel-logic surgery, out of scope). If the field is removed before a
   typed-array replacement lands, this factory needs a plan.

## Successes

- **[Caution: Porting a shared kernel](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#caution-porting-a-shared-kernel)
  fired correctly, twice.** First on the six rung-1 forks: reading each fork *before* writing the
  factory made the binding vocabulary a constraint rather than a choice, and it is not the vocabulary
  this op's locals would have suggested — the untilize compute forks use `dfb::src`/`dfb::out` while
  the readers use `dfb::in`, so the sharded factory binds the *same* `SH_IN` buffer under the name
  `in` on the reader and `src` on the compute kernel
  (`…_multi_core_sharded_program_factory.cpp:352-357`). Deriving those names from this op instead
  would have silently broken every other consumer of the forks.
  Second, and more valuable: the entry's insistence that the *lent* direction is invisible from the
  path ("Nothing about the path warns you: the file sits inside your writeable surface, so converting
  it in place feels safe, and it breaks every borrower the moment you do") is the only reason I ran
  the census on the op's **own** writers at all — the brief said they were exclusively this op's. One
  of them wasn't. See Handoff point 3.

- **The `constexpr` carve-out in [CB→DFB whitelist §A](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/cb_dfb_api_whitelist.md#tile--format-metadata-jit-descriptors)
  is keyed on exactly the right thing.** Two `get_tile_size` sites in this op, and the rule splits
  them correctly on the legacy declaration alone:
  `writer_unary_unpad_width_16_sharded.cpp:23` was `constexpr` and feeds a `static_assert` and two
  `NOC_MAX_BURST_SIZE` template arguments → kept the free-function form with the token,
  `get_tile_size(dfb::out)`; `reader_unary_interleaved_wh_multicore.cpp:27` was plain `const` → moved
  onto the object, `dfb.get_tile_size()`. Reaching for the member getter at the first site would not
  have compiled; the rule got there without needing the build I could not run.

- **[Two-toucher / self-loop endpoint-assignment procedure](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-two-toucher-dfb--assign-1p1c-dual-instance-work-split),
  re-derived rather than transcribed.** The census agreed with the brief everywhere: `c_17` is a
  genuine one-toucher (the writer `reserve_back`s, fills by write pointer, `push_back`s; nothing
  drains it) → self-loop; every other buffer is an ordinary 1P+1C. The brief's "Watch for" note about
  the BlockInterleaved factory's same-source multiplicity was also correct and load-bearing — those
  are **disjoint-node** work splits, not same-grid two-touchers, so no 1P+1C assignment and no
  multi-binding flag. **`allow_instance_multi_binding` appears nowhere in this port.**

- **The `opt_level` "absent line" warning earned its emphasis.** The recipe's insistence that this is
  the field which survives an otherwise careful port because there is *nothing to read and object to*
  is accurate: `grep -n opt_level` over the five legacy factories returns **zero** hits, so nothing in
  the legacy source hints that compute kernels resolve to `O3`. All 10 compute `KernelSpec`s now carry
  an explicit `O3`; the DM specs correctly carry nothing.

## Friction

### Gaps

- **The recipe has no answer for a per-node-variable vararg count.** [Caution: Avoid varargs](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#caution-avoid-varargs-unless-absolutely-necessary)
  and the [migration guide](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/migration_guide.md#programrunargs)
  both describe varargs as a single `num_runtime_varargs` count, and the guide's worked example is a
  CTA-bounded shape where the count is uniform. Neither mentions `num_runtime_varargs_per_node`, which
  is the only construct that fits a payload whose length varies per core — the shape this op's
  MultiCoreInterleaved writer has. I found it by reading `advanced_options.hpp`, which is exactly what
  ["go to the headers first"](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#read-this-first)
  promises — but a porter who trusted the docs alone would have reached for a max-count-plus-padding
  workaround, which silently changes the dispatch payload on most cores. **Suggested fix:** one
  sentence in the varargs caution naming the per-node override and its deprecation status, so the
  choice is made deliberately rather than discovered.

- **No guidance on a *dead* compile-time arg.** Five CTAs in this op are emitted by the host and never
  read by the kernel (listed under [Open items](#open-items-for-downstream)). The recipe covers dead
  *CBs* explicitly ("build no spec, drop the allocation and any dead CTA carrying its index") but says
  nothing about a CTA that is dead on its own. In the positional world it was invisible plumbing; in
  the named world the porter must decide whether to declare a name nothing reads. I kept all five
  (uniform, faithful, no per-site judgment — an unread named CTA lowers to an unused
  `constexpr experimental::CtaVal<uint32_t>` in the generated header and costs nothing), but the
  opposite choice is just as defensible, and two porters will split. **Suggested fix:** a line in
  [Dropped Plumbing](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#dropped-plumbing)
  stating which way to go. The decision matters more than usual when the CTA lands in a **shared
  fork's** interface, where it becomes every future consumer's obligation — as
  `args::output_row_size` now is in `writer_unary_stick_layout_interleaved_blocks_metal2.cpp`.

- **A shared host-side helper that emits legacy descriptors has no migration story.** The
  BlockInterleaved factory's circular buffers were built by
  `ttnn::operations::data_movement::push_buffer_set` (`data_movement/common/common.cpp:795-850`),
  which takes a `ProgramDescriptor&`. It is out of this port's writeable surface, so the two
  `DataflowBufferSpec`s are now built inline in the factory, duplicating its sizing rules. The helper
  exists *precisely* to stop those rules drifting between the four block factories (its own comment:
  "a private copy can drift and reintroduce the corruption the split prevents"), so the port has
  reintroduced the drift risk it was written to prevent — correctly, per the scope boundary, but the
  recipe offers no route for this shape. See [Open items](#open-items-for-downstream).

### Confusion

- **The `cb`-name sweep's "expect zero hits" collides with the off-limits rule.** The sweep flags
  `input_cb_data_format` (a legitimate rename — done) but also four stale `CB` comments in
  `device/untilize_with_unpadding_device_operation.cpp` and `untilize_with_unpadding.cpp`, which
  [Host-side: stay in the lane](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#host-side-stay-in-the-lane)
  forbids touching. The self-audit item says "Expect **zero** hits … every hit is a real leftover",
  which reads as a hard gate; the resolution (the off-limits rule is the more specific one, so those
  hits are *reported*, not fixed) took a re-read of both sections to be confident about. **Suggested
  fix:** scope the sweep to the factory bodies and kernels, and say the device-op class is excluded
  and reported instead.

- **Near-miss on run-arg ordering, caught by reading the legacy loop twice.** The MultiCoreInterleaved
  writer's `start_stick_id` is pushed **before** the inner loop that advances `row_start_id` past that
  core's own blocks. Restructuring the legacy `RTArgList` build into `AddRuntimeArgsForNode` moves the
  arg emission to the *end* of the loop body, which silently shifts every core's start row by its own
  block count. Caught and fixed with an explicit `core_row_start_id` capture
  (`…_multi_core_interleaved_program_factory.cpp:236-238`). The recipe's advice to "keep the legacy
  per-node loop as-is and let the helper transpose" is right, but the helper's natural call position
  is not always where the legacy pushed the value. **Suggested fix:** a warning beside the
  `AddRuntimeArgsForNode` example that a legacy `push_back` interleaved with mutation of the value it
  pushed must keep its original position, not migrate to the end of the loop.

## Open items for downstream

### Shared kernel touches

Ten kernels, six reused and four forked. None was modified in place.

| kernel | relation | rung taken | remaining unmigrated consumers |
|---|---|---|---|
| `eltwise/unary/…/reader_unary_interleaved_start_id.cpp` | borrowed | **1 — reused** `…_metal2.cpp` (no new file) | `examples/example`, `examples/example_multiple_return`, `experimental/transformer/nlp_create_qkv_heads_falcon7b`, `reduction/topk` |
| `eltwise/unary/…/reader_unary_sharded.cpp` | borrowed | **1 — reused** | `data_movement/tilize`, `data_movement/untilize`, `data_movement/sharded_partial/sharded_to_interleaved_partial`, `experimental/slice_write` |
| `data_movement/sharded/…/reader_unary_nd_sharded_blocks.cpp` | borrowed | **1 — reused** | none (this op was the only consumer) |
| `data_movement/untilize/…/compute/untilize.cpp` | borrowed | **1 — reused** | `data_movement/fold` |
| `data_movement/untilize/…/compute/untilize_variable_num_blocks.cpp` | borrowed | **1 — reused** | `data_movement/untilize` |
| `ttnn/kernel/compute/eltwise_copy.cpp` | borrowed | **1 — reused** | `data_movement/copy`, `data_movement/sharded/interleaved_to_sharded`, `data_movement/sharded_partial/interleaved_to_sharded_partial`, `data_movement/sharded_partial/sharded_to_interleaved_partial` |
| `eltwise/unary/…/reader_unary_interleaved_wh_multicore.cpp` | borrowed | **2 — created** `reader_unary_interleaved_wh_multicore_metal2.cpp`; pointer comment added to the original ✓ | `data_movement/untilize` |
| `data_movement/untilize/…/compute/untilize_wh.cpp` | borrowed | **2 — created** `untilize_wh_metal2.cpp`; pointer comment added ✓ | `data_movement/untilize` |
| `ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_blocks.cpp` | borrowed | **2 — created** `…_metal2.cpp`; pointer comment added ✓ | none (this op was the only consumer) |
| `…/untilize_with_unpadding/…/writer_unary_stick_layout_wh_multicore.cpp` | **lent** (audit missed it — Handoff 3) | **2 — created** `…_metal2.cpp` beside the original, inside this op's own directory; pointer comment added ✓ | `data_movement/untilize` |

Binding vocabulary the four new forks establish, for the next consumer to inherit:

| fork | `dfb::` | `tensor::` | named args |
|---|---|---|---|
| `reader_unary_interleaved_wh_multicore_metal2.cpp` | `in` | `src` | CTA `num_tiles_per_2d`, `third_dim`, `total_tiles_per_row`; RTA `start_id`, `single_block_size_row_arg`, `single_block_size_col_arg` |
| `untilize_wh_metal2.cpp` | `src`, `out` | — | CTA `block_size_col`, `block_size_row`, `third_dim` |
| `writer_unary_stick_layout_interleaved_blocks_metal2.cpp` | `out` | `dst` | CTA `float32_dtype`, `output_row_size` (unread — see below); RTA `num_rows_block`, `block_row_size`, `batch`, `num_blocks_h`, `num_blocks_w`, `last_block_row_size_unpadded`, `num_output_rows_unpadded`, `block_start_row_id`, `block_start_row_offset` |
| `writer_unary_stick_layout_wh_multicore_metal2.cpp` | `out` | `dst` | CTA `total_num_rows`, `third_dim`, `tile_height`, `unpadded_X_size`; RTA `width_size`, `start_row_id`, `start_column_id`, `single_block_size_row_arg`, `single_block_size_col_arg`, `sub_block_width_size`, `single_sub_block_size_row_arg` |

`untilize_wh_metal2.cpp` deliberately reuses the `dfb::src` / `dfb::out` vocabulary of its two sibling
untilize compute forks in the same directory, so a factory can bind any of the three with one
vocabulary. **Sunset:** `data_movement/untilize` is the last unmigrated consumer of three of the four
new forks; when it ports, those three legacy originals can be retired.

### Findings — bugs and oddities carried forward unchanged

Every item below is preserved byte-for-byte in behavior. None is fixed.

1. **Five dead compile-time args** — emitted by the host, never read by the kernel. Kept as named CTAs
   (see the Friction gap above):
   - `writer_unary_unpad_dims_split_rows.cpp`: `unpadded_stick_size` (legacy CTA 1). The kernel gets
     the same quantity as RTA `num_unpadded_X` × element size instead.
   - `writer_unary_stick_layout_interleaved_blocks_metal2.cpp`: `output_row_size` (legacy CTA 1).
   - `writer_unary_unpad_width_16_sharded.cpp`: `aligned_page_size` (legacy CTA 2) — the same
     `writer_ct_args` vector feeds both same-shard-type writers and only the general one reads it.
   - `writer_unary_stick_layout_split_rows_multicore_nd_sharded.cpp`: `output_stick_size` (legacy CTA
     1) and `input_single_tile_size` (legacy CTA 8). Both were already dead under positional args (the
     kernel's `TensorAccessorArgs<17>` offset accounted for them without reading them).

2. **One dead runtime arg.** `writer_unary_unpad_dims_split_rows.cpp:29` reads
   `num_blocks_w_input` into a local the kernel body never uses. It still occupies a dispatch slot, so
   it is preserved as a named RTA. Removing it would shrink the writer's per-core RTA payload by one
   word — a real but out-of-scope saving.

3. **`device/factories/untilize_with_unpadding_multi_core_shared_variables.hpp` is unreferenced dead
   code.** It defines `UntilizeWithUnpaddingMultiCoreSharedVariables` (reader/writer `KernelHandle`s, a
   core vector, an `ncores`) — a leftover from the pre-`ProgramDescriptor` `ProgramFactoryConcept` era.
   Nothing in the tree names the struct; the header is only listed in
   `ttnn/cpp/ttnn/operations/data_movement/CMakeLists.txt:333`. Left in place (out of the port's
   scope). Now that the op is on Metal 2.0 it is unambiguously dead and can be deleted with its
   CMakeLists entry.

4. **Stale `CB` comments in off-limits files.** `device/untilize_with_unpadding_device_operation.cpp`
   lines 144, 221 and 273 each say "binds the output buffer directly as an L1 circular buffer (see the
   sharded factory's `sharded_output_cb_index` CB)"; `untilize_with_unpadding.cpp:102` refers to "the
   CB budget". The identifier `sharded_output_cb_index` no longer exists — it is now the
   `SH_SHARDED_OUT` dataflow buffer. The *reasoning* in all four comments is still correct (that is
   why the DRAM rejection they justify is still right); only the vocabulary is stale. Not edited: both
   files are outside the port's writeable surface.

5. **`num_blocks_w_input` and the `round_down_32` helper.** `writer_unary_unpad_dims_split_rows.cpp:12`
   defines `inline uint64_t round_down_32(...)`, which nothing calls. Pre-existing; left alone.

### Doc-evolution and carry-over

- **Candidate: a spec-side `push_buffer_set`.** Four factories share the block-buffer model
  (tilize, tilize_with_val_padding, untilize, untilize_with_unpadding) and the helper exists to keep
  their sizing rules identical. The first of them to port must inline the rules (as this one did), and
  each subsequent port will inline them again — which is precisely the drift the helper prevents.
  Worth deciding, before the second block factory ports, whether
  `data_movement/common` should grow a `Group<DataflowBufferSpec> make_buffer_pair(const BlockBufferSet&, …)`
  alongside the descriptor one. That is a shared-code change, outside any single port's scope.
- **Sibling ops that would benefit from the same pattern:** `data_movement/untilize` is the direct
  sibling — it shares four kernels with this op and is the last unmigrated consumer of three of the
  four new forks. Porting it next would let three legacy kernel copies be retired at once, and its
  block factory can bind `writer_unary_stick_layout_wh_multicore_metal2.cpp` and
  `untilize_wh_metal2.cpp` at rung 1.

### Test coverage notes

The confirmed no-regression baseline (agreed with the invoker before relying on it) is:

1. `tests/ttnn/unit_tests/operations/data_movement/test_untilize_with_unpadding.py` — primary, 992
   lines, 41 references.
2. `./build/test/ttnn/unit_tests_ttnn --gtest_filter='*UntilizeWithUnpadding*'` — the C++ gtest
   `TestGraphCaptureArgumentsUntilizeWithUnpadding`
   (`tests/ttnn/unit_tests/gtests/test_graph_capture_arguments_untilize_with_unpadding.cpp`, built into
   `unit_tests_ttnn` via `tests/ttnn/unit_tests/gtests/sources.cmake:28`).
3. `tests/ttnn/unit_tests/base_functionality/test_to_layout.py -k untilize_with_unpadding`.

Note for whoever runs it: in (3) the `-k untilize_with_unpadding` filter matches by **test name**, and
only `test_untilize_with_unpadding_W_16` matches; the file's other ~15 `untilize_with_unpadding` call
sites live in differently-named tests (e.g. the ND-sharded and sharded-output cases around lines
588-630, 967-1040 and 1608-1620) and are **not** selected by that filter. Running the file unfiltered
would cover the W=16 fast path *and* the ND-sharded path this port touches. Flagging rather than
overriding the invoker's chosen command.

## Verification status

Not run. The full sequence, unchanged, for the next session:

```bash
# 0. re-apply the forced-legality scaffolding (Handoff 2), then:
./build_metal.sh --build-tests                      # background, log to a file
export TT_METAL_WATCHER=10                          # required for every test run
./build/test/ttnn/unit_tests_ttnn --gtest_filter='*UntilizeWithUnpadding*'
pytest tests/ttnn/unit_tests/operations/data_movement/test_untilize_with_unpadding.py -x -v
pytest tests/ttnn/unit_tests/base_functionality/test_to_layout.py -k untilize_with_unpadding -x -v
# then: grep the logs for METAL2_CHECKS_FORCED — expect BOTH markers before trusting any green.
```

Expect the first build to surface ordinary mechanical errors (a misspelled named arg between a
`runtime_arg_schema` and its kernel, a designated-initializer order slip). The
[cryptic-error table](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/migration_guide.md#cryptic-error--likely-cause)
covers the spec-validator failures most likely here; the two spots I would check first are the
per-node vararg counts on the MultiCoreInterleaved writer (Handoff 4) and the `SH_SHARDED_OUT`
self-loop's borrowed-memory binding.

### Anti-pattern self-audit results

Run over the op directory (**27 `.cpp`/`.hpp` files scanned** — non-zero denominator) plus the four
new forks. Static checks only; the build-dependent items are unverified.

| check | result |
|---|---|
| No buffer address in run-args (`->address()`, `emplace_runtime_args`, bare `Buffer*`) | **0 hits / 27 files** |
| No magic CB indices, `CBDescriptor`, `CBFormatDescriptor`, `CircularBuffer` | **0 hits** |
| No `TensorAccessorArgs<N>()` in any ported kernel | **0 hits** (the one remaining occurrence is in the *legacy original* `writer_unary_stick_layout_wh_multicore.cpp`, deliberately untouched for its other consumer) |
| No `cb` in any DFB name, spec-name string, or host variable | **0 hits** in the five factories and the four forks, after renaming `*_cb_data_format` → `*_dfb_data_format` and dropping the dead `cb_utils.hpp` includes. 4 hits remain in comments in the two **off-limits** files — reported, not edited (finding 4) |
| Conditional bindings follow the pattern | `SH_SHARDED_OUT` is bound only in the two configurations that allocate it. **No `#ifdef` needed**: the host selects a different kernel *source* per configuration, so no kernel ever name-looks-up a token its own build does not bind |
| No `.id` extraction on `dfb::` handles | **0 hits** |
| No CTA→RTA demotion | none — per-group CTAs preserved on all 2 + 4 same-source compute splits |
| No unnecessary multi-binding flag; never stacked with a self-loop | **0 occurrences of `allow_instance_multi_binding`** anywhere |
| All CTAs named | **yes** — every `compile_time_args` is `{{name, value}, …}` |
| No nameable argument smuggled into varargs | 2 vararg blocks, both genuine indexed collections (runtime-count `BlockRep` runs; CTA-bounded shape stream). Their 3 leading scalars are named |
| No forced-legality scaffolding in the diff | the 2 `tt_metal/` files are **excluded from the commit**; no other `tt_metal/` path is touched |
| No ephemeral doc cited from code | **0 hits / 27 files** |
| Every legacy `TT_FATAL`/`TT_ASSERT`/`TT_THROW` accounted for | **census clean — no output.** Three `dst_buffer != nullptr` guards were initially lost when the `Buffer*` locals went away; restored as `output.buffer() != nullptr` (same condition, same message, no stray `Buffer*`) |
| Every `hw_config` reproduces the legacy resolved values | DM: every kernel resolved to the plain reader/writer defaults → arch-agnostic TTNN helpers. Compute: **Style B**, `ComputeGen1Config` built directly; only `enable_32_bit_dest` and `unpack_modes` set, all four other fields left at defaults that coincide with the legacy `ComputeConfigDescriptor` defaults |
| Every `KernelSpec`'s `opt_level` matches | `grep -n opt_level` → **5 lines**, one per factory, each inside the single construction site (or lambda) that builds that factory's compute specs → **all 10 compute `KernelSpec`s carry explicit `O3`**. No DM spec sets one (legacy `O2` = Metal 2.0 default) |
