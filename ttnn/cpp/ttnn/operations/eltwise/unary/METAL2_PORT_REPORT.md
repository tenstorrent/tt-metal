# Metal 2.0 Port Report — `ttnn/cpp/ttnn/operations/eltwise/unary`

Written for: the Metal 2.0 porting / runtime team, the doc maintainers, and the TTNN eltwise owners.

## Outcome

**`PORTED`.** The single `ProgramFactory` and all eleven kernel entry points it can bind are
converted to `CustomProgramSpecFactoryConcept`. It builds, and the confirmed sentinel set passes
at **985 passed / 6 skipped / 3 xfailed — an exact match to the pre-port baseline on this branch**.
The row-5 regression test named in the port brief was restored by the invoker and passes. Every
static check in the recipe's anti-pattern self-audit passes.

No Metal 2.0 capability gap was hit, and nothing the port needed fell outside the op's directory
beyond the one sanctioned shared-kernel fork.

**One verification step did not happen**, and it is the one the recipe treats as load-bearing:
the cache-hit re-validation path was almost certainly off during the passing run, so the
relaxation declaration has not been checked by the validator. It is a single re-run to close, with
no rebuild — see *Verification* below. Nothing else is outstanding.

## Provenance

```
git log -1 --format='%h %cs %s' -- docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/
```

printed nothing, so the recipe version cannot be pinned.

- **Recipe docs (this port):** *not pinnable* — the `metal_2.0/` doc tree is staged-but-uncommitted
  in this checkout. Working tree: branch `anasuya/metal2_port_unary`, HEAD
  `6e4a9a9b588 2026-09-17 Fix ND-sharded tensors reading an absent 2D shard spec`.
- **Audit docs (inherited):** *not pinnable* — the `metal_2.0/` doc tree is staged-but-uncommitted in
  this checkout (`git log -1 -- .../metal_2.0/` prints nothing). Working tree: branch
  `anasuya/metal2_port_unary`, HEAD `6e4a9a9b588 2026-09-17`.

## TTNN ProgramFactory

### Concept realized

**`CustomProgramSpecFactoryConcept`**, as the audit chose. No re-decision, no disagreement.
`ProgramFactory::create_descriptor` became `create_program_artifacts`; the `void`-returning
`override_runtime_arguments` was **translated**, not deleted, into one returning a
`ProgramRunArgs` (only the return type is concept-enforced, so a deletion would have dropped the
op silently onto the base concept).

The override returns a `TensorArgument` for **every** `TensorParameter` bound to an io tensor —
both of them, `src` and `dst`, on every dispatch. Nothing is deliberately skipped. That matches
what the ported-from override did: it wrote `input.buffer()->address()` and
`output.buffer()->address()` into reader/writer slot 0 on every hit
(`unary_program_factory.cpp:585-586,613,616` pre-port) *and* re-pointed both tensor-backed CBs via
`apply_descriptor_runtime_args` (`:657-665` pre-port). Both of those statements became
`tensor_args` entries; neither became a runtime-arg value.

The override also re-applies every per-core runtime arg the miss path writes, through the *same*
`build_kernel_run_args` → `enumerate_core_rt_args` chain `create_program_artifacts` uses. The
ported-from code shared `enumerate_core_rt_args` between its two methods for exactly this reason,
and the port preserves that rather than re-deriving the split in the override.

Nothing was added to the refresh set. The two legacy blocks that *did* disappear did so because
the binding model owns them now, not because the port judged them unnecessary: the accessor
common-arg rebuild (`:638-653` pre-port) and the CB-address patch (`:657-665` pre-port).

### Device-op-class edits

- **Pybind entry points removed:** **none.** `unary_nanobind.cpp` binds no `create_descriptor` and
  no descriptor internals (grep clean), so the port makes no user-visible API change.
- **Custom `compute_program_hash`:** **left intact**, untouched, at
  `device/unary_device_operation.cpp:179`, along with the backdoor
  `operation_attributes_t::to_hash()` at `:16`. `git status` shows
  `device/unary_device_operation.cpp` modified only by the pre-port commits already on this branch,
  not by the port.
- **Other device-op-class edits:** the only edit outside the factory body is in
  `device/unary_device_operation.hpp`, and it is confined to the `ProgramFactory` struct's two
  method declarations plus the include swap those force (`program_descriptors.hpp` +
  `experimental/program_descriptor_patching.hpp` out; `metal_v2_artifacts.hpp` +
  `experimental/metal2_host_api/program_run_args.hpp` in). Nothing in `validate_on_program_cache_miss`,
  `compute_output_specs`, `create_output_tensors`, `compute_program_hash` or `skip_launch` changed.

### Open items

- **Relaxation declared, and it is the tree's first shipped one.** Both `TensorParameter`s carry
  `.relaxations = {.dynamic_tensor_shape = true, .relax_logical_rank = true}`, transcribed from
  `analyses/relaxations/eltwise_unary.md` §2 (verdict CONFIRMED; all five validity checks re-run by
  the audit and passing). `match_page_size` and `match_padded_shape_only` are deliberately unset.
  Unary is the first non-experimental shipped factory to declare any relaxation, and
  `relax_logical_rank` has no shipped precedent at all — so a validation throw here should be
  treated as a plausible framework-side gap before it is treated as a mis-declaration.
- **Capability the op would benefit from:** a relaxation that admits a *sharded* slot whose
  distribution geometry re-squeezes with the shape values. That is precisely the gap behind analysis
  §3 row 5, and it is why that row rates Low confidence rather than being dischargeable by the
  declaration. The header is explicit that such an argument is rejected rather than mis-addressed
  (`tensor_spec_relaxations.hpp:63-74`), so the failure mode is a spurious throw, not corruption.

---

## Handoff points

1. **Shared-kernel fork created — `eltwise_sfpu_metal2.cpp` (rung 2).** Owner: the Metal 2.0
   porting effort / the `examples` op owners.
   - **Fork:** `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu_metal2.cpp`
     (new file, beside the original, same stem + `_metal2`). Pointer comment landed in the legacy
     original at `eltwise_sfpu.cpp:5-9`. No build-system change was needed — the op's
     `CMakeLists.txt:16` already does `file(GLOB_RECURSE kernels device/kernels/*.cpp)`.
   - **Binding vocabulary this fork establishes** (inherited by every later consumer, so not to be
     renamed once one exists): `dfb::input`, `dfb::output`, `args::num_tiles`.
   - **Remaining unmigrated consumers — the sunset list.** All five still bind the legacy original:
     - `ttnn/cpp/ttnn/operations/examples/example/device/single_core_program_factory.cpp:91`
     - `ttnn/cpp/ttnn/operations/examples/example/device/multi_core_program_factory.cpp:89`
     - `ttnn/cpp/ttnn/operations/examples/example_multiple_return/device/single_core_program_factory.cpp:80`
     - `tests/ttnn/unit_tests/gtests/test_generic_op.cpp:246`
     - `tests/ttnn/unit_tests/operations/fused/parallel_sequential/test_parallel_sequential.py:1436`

     When the last of these migrates, delete `eltwise_sfpu.cpp` and rename the fork onto its name.

2. **~~The regression test the invoker named as the relaxation's safety net does not exist.~~
   RESOLVED — restored by the invoker and passing.** Kept in the report because the coverage it
   provides is what licenses the row-5 declaration, and a future reader deciding whether to trust
   that declaration needs to know which test carries it.

   `test_unary_sharded_input_on_interleaved_path_cache_reuse` now lives at
   `tests/ttnn/unit_tests/operations/eltwise/test_unary_program_cache.py:558`, and it targets
   exactly the right axis: a **DRAM height-sharded** input (so `is_native_L1_sharding` is false,
   `get_shard_specs` returns `nullopt`, and the op takes the interleaved path with the accessor live
   over a genuinely sharded buffer), driving **two shapes through one asserted cache entry**. Its
   four parameterisations vary the two things the TILE branch of the key omits — logical **rank**
   (`rank4_then_rank2`, `rank2_then_rank4`, which is what `relax_logical_rank` exists for) and
   **even-vs-uneven padded shape** (`even_then_uneven`, `uneven_then_even`) — in both orders. It
   asserts the golden on each dispatch *and* `cache_entries_counter.total == 1`, so a rebuild cannot
   mask a stale arg.

   That closes the gap the earlier version of this entry described. For the record, it is a
   genuinely different test from the two sharded tests that already existed:
   `test_unary_nd_sharded_fallback` (`test_unary_sharding.py:205`) and
   `test_unary_uneven_sharding_fallback` (`:163`) each run a **single** dispatch, and a wrong
   relaxation is accepted on the first dispatch — it only bites on the second at a different shape.
   Every other cache-hit-across-shapes test (`test_unary_cache_reuse_same_volume_different_shapes`,
   `test_unary_cache_reuse_different_volumes`, `test_unary_inplace_cache_reuse_different_shapes`)
   uses `ttnn.DRAM_MEMORY_CONFIG`, i.e. plain interleaved — analysis row 1, the High-confidence
   regime. This test is the only coverage of row 5's cache-hit axis in the tree.

   **Caveat, carried from *Verification* above:** the test proves the *numerics* of the reused
   program at both shapes, which holds regardless of the validation knob. It does **not** prove the
   framework accepts the declaration, because the check that reads
   `TensorParameter::relaxations` runs only on the cache-hit path that
   `validate_program_args` gates, and that is off by default. Re-run it with the knob on to get
   that half.

3. **Non-32×32 tiles: the family-wide defect now fails *loud* under sharding.** Owner: the eltwise
   team (this is the same issue the audit already routed under *Misc anomalies* #8 — this is a new
   consequence of it, not a new defect). The port preserves the legacy sizing verbatim: both DFBs
   take `entry_size = tile_size(DataFormat)`, which assumes 32×32, while `num_entries` comes from
   the tensor's real tile. Metal 2.0 adds a check legacy did not have — a borrowed DFB may not
   exceed its backing tensor's per-bank allocation (`tt_metal/impl/metal2_host_api/program_spec.cpp:1628-1641`).
   A **sharded** tensor with, say, a 16×32 bf16 tile computes roughly twice the true shard size, so
   post-port it raises a `TT_FATAL` at spec build where pre-port it silently produced wrong data.

   Not reachable from the sentinel suites: the only non-default-tile tests
   (`test_unary_program_cache.py:170`, `:225`) are both interleaved, and interleaved DFBs allocate
   their own L1 and so bypass the check. Flagged because it changes the failure *mode* of an
   already-broken configuration, and because fixing the sizing would resolve it — which is
   out of this port's scope by the brief.

4. **~~Build and test execution was refused by the environment.~~ RESOLVED — the invoker built and
   ran them** (results in *Verification*). Recorded as a workflow note rather than an escalation:
   every attempt from the porting session to run `./build_metal.sh --build-tests` (foreground and
   background, with and without the venv activation prefix) and every test invocation was denied by
   the sandbox with *"Permission for this action was denied … Reason: [Modify Shared Resources]"*.
   A port cannot self-verify under that permission set, so the recipe's whole Verification step has
   to be handed back to the invoker. Worth knowing when scheduling the next port in the same
   environment: budget for the round trip, and expect the forced-checks proof
   (`METAL2_CHECKS_FORCED` in a test log) to be the step that falls through the gap, since it is the
   one that needs the build and the run and the log all in the same pair of hands.

5. **Boundary-rule assumption violations:** none. No out-of-op call site required a `sem::` or
   `tensor::` handle. The op has no semaphores at all, and both `TensorAccessor`s are constructed
   inside the op's own kernels.

6. **Kernel-lib gaps:** none. Every escape lands in `tt_metal/*` or `ttnn/cpp/ttnn/kernel_lib/`, and
   every consumed signature that carries a resource handle takes a plain `uint32_t` CB id, which the
   `DFBBindingToken`'s `constexpr operator uint32_t()` bridges. No kernel-lib or LLK file was
   modified.

---

## Verification

Run by the invoker; the porting session's own sandbox refused every build and test invocation
(*"[Modify Shared Resources]"*), so the numbers below are reported, not measured here.

| Step | Result |
|---|---|
| `./build_metal.sh --build-tests` | **pass** |
| Sentinel pytests — `test_unary.py`, `test_unary_sharding.py`, `test_unary_program_cache.py` | **985 passed / 6 skipped / 3 xfailed — exact match to the pre-port baseline** |
| `test_unary_sharded_input_on_interleaved_path_cache_reuse` (restored by the invoker) | **pass** |
| Cache-hit re-validation live during the run (`validate_program_args` / the forced-checks proof) | **NOT confirmed — see below** |

Three risks flagged before the build are now **closed by it**:

1. **The compute kernels' NTTP nesting compiles.** Seven of the nine bind DFBs through
   `compute_kernel_lib`, where the id sits inside an `InputSpec` / `OutputSpec` used as a non-type
   template parameter: `ckl::CopyTile<ckl::input(dfb_input_id, …), ckl::Dst::D0>{}`. The port keeps
   each kernel's legacy declaration shape and swaps only the initializer —
   `constexpr auto dfb_input_id = dfb::input;` — so `DFBBindingToken`'s
   `constexpr operator uint32_t()` had to fold inside that nesting. It does. The audit predicted
   this and it held; the fallback spelling (`constexpr uint32_t dfb_input_id = dfb::input;`) was not
   needed. **This is the first confirmation that a `dfb::` token survives NTTP-inside-a-struct**, so
   the six other `compute_kernel_lib` ops behind this one need not re-litigate it.
2. **The two replacement includes are sufficient.** Dropping the now-unused
   `<tt-metalium/program_descriptors.hpp>` also dropped a *transitive* `fmt` provider (that header
   specialises `fmt::formatter`), and the factory still calls `fmt::format` for the compute kernel
   path; `<fmt/format.h>` and `<tt_stl/assert.hpp>` are now included directly.
3. **Concept selection is right.** The factory landed on `CustomProgramSpecFactoryConcept`, not the
   base one — a silent fall-through would have shown as an `AllFactoriesValid` `static_assert` or an
   unresolved `override_runtime_arguments`.

### The one step still open: cache-hit re-validation

**The forced-legality scaffolding was reverted before the port's state was frozen**
(`git diff HEAD -- tt_metal/` is empty, and `grep 'skip_validation = false'` over
`tt_metal/impl/metal2_host_api/` returns nothing), and the knob that gates the same checks defaults
to **off**:

```cpp
// ttnn/api/ttnn/config.hpp:29-31
// Re-validate Metal 2.0 program args on the cache-hit fast path (Update{Tensor,ProgramRun}Args).
// The cache-miss build path always validates. Off by default; CI turns it on.
bool validate_program_args = false;
```

**What that does and does not leave unverified** — the split is clean, and most of it is fine:

- **Cache-miss validation ran, unconditionally.** The adapter calls
  `MakeMeshWorkloadFromSpecs(*mesh_device, program_specs)` and
  `SetProgramRunArgs(program, run_params.at(range))` with no `skip_validation` argument
  (`mesh_device_operation_adapter.hpp:982-986`), and both default to `false`
  (`metal2_host_api/program.hpp:45-51,74`). Its own comment says so: *"Cache miss is the cold path:
  always validate here, so every cached program was built from a checked spec."* So the **whole
  `ProgramSpec` validator** was exercised by every test that built a program — DFB endpoint
  census and the `tmp0` self-loop pair, the `unpack_modes` legality table and its required-entry
  rule, the borrowed-DFB L1-residency and per-bank size checks, tensor-parameter usage, the Gen1
  DM node invariants. That is the larger half of the spec's legality surface and it is verified.
- **The cache-*hit* re-check did not run.** `UpdateProgramRunArgs` is called with
  `skip_validation = !ttnn::CONFIG.get<"validate_program_args">()`
  (`mesh_device_operation_adapter.hpp:1020`), which is `true` at the default. And that is precisely
  where `tensorspecs_match_with_relaxation` lives — the only check that reads
  `TensorParameter::relaxations`.

**So the relaxation declaration itself is the thing still unchecked**, and it is the port's one
novel, correctness-sensitive declaration (unary is the first shipped factory to declare any
relaxation; `relax_logical_rank` has no shipped precedent at all).

Note what this does *not* undermine: `test_unary_sharded_input_on_interleaved_path_cache_reuse`
asserts the **numerics** of both dispatches plus a single cache entry, and that holds regardless of
the knob — the reused program demonstrably produces correct data at both shapes, which is the
substantive question. What the knob gates is whether the framework would have *accepted* the
declaration, and with validation off a wrong declaration (say, a missing `relax_logical_rank`
against the test's rank4→rank2 pair) is silently tolerated rather than thrown. Passing the test
therefore does not discriminate a correct declaration from an over-permissive one.

**Closing it is one re-run, no rebuild** — `validate_program_args` is a runtime config knob read at
`_ttnn` load, not a compile-time switch:

```bash
TTNN_CONFIG_OVERRIDES='{"validate_program_args": true}' \
  pytest tests/ttnn/unit_tests/operations/eltwise/test_unary.py \
         tests/ttnn/unit_tests/operations/eltwise/test_unary_sharding.py \
         tests/ttnn/unit_tests/operations/eltwise/test_unary_program_cache.py
```

Expect the same 985 / 6 / 3. A `TensorSpec` legality failure on a *second or later* dispatch is the
signature to watch for; per the recipe that is a stop-and-report against the op's cache key, not
something to fix by loosening the declaration.

(`TT_METAL_WATCHER=10` was likewise not confirmed for the run. Worth setting on the re-run above,
since it costs only a kernel recompile and this port's kernels are all newly JIT-compiled.)

---

## Successes

- **The `_metal2` name-adjacency warning fired exactly as intended.** `device/kernels/dataflow/`
  contains `reader_unary_interleaved_start_id_metal2.cpp`, `reader_unary_sharded_metal2.cpp` and
  `writer_unary_interleaved_start_id_metal2.cpp`, and the natural reading of "is there a `_metal2`
  fork beside it?" as a directory question would have concluded that `reader_unary.cpp` and
  `writer_unary.cpp` were already forked and needed no conversion. The brief's *"The fork test is
  per-stem, not per-directory"* stopped that, and the per-stem `ls` confirmed no fork of either.
  Both were then converted in place — correctly, since the census shows no external consumer of
  either (`device/kernels/dataflow/reader_unary.cpp` at
  `device/kernels/dataflow/reader_unary.cpp:1`, writer likewise).
- **The same warning's positive half paid off too.** Those three forks were read as idiom
  precedent, and `dfb.get_entry_size()` — the replacement for
  `get_local_cb_interface(cb_id).fifo_page_size` at `device/kernels/dataflow/reader_unary.cpp:57`
  and `writer_unary.cpp:60` pre-port — came from them rather than from guessing at the whitelist's
  §B table. Both forks carry the same comment about it working for TILE and ROW_MAJOR alike.
- **"Read the legacy declaration, not the value" saved the `opt_level` line.** `grep -rn opt_level`
  over the op directory returns *nothing*, which reads as "the op has no opinion." The recipe's
  insistence that an absent `KernelDescriptor::opt_level` still resolves to **O3** on a
  `ComputeConfigDescriptor` is what put the explicit `KernelBuildOptLevel::O3` on the compute
  `KernelSpec` (`device/unary_program_factory.cpp:686`). Confirmed at the source:
  `tt_metal/impl/program/program.cpp:485` uses `value_or(KernelBuildOptLevel::O3)` for the compute
  descriptor and `value_or(O2)` for the DM ones. Without that the port would have silently dropped
  a level on every unary compute kernel, with no test able to see it.
- **"Re-derive the endpoint census, don't transcribe it" confirmed the brief rather than correcting
  it.** `c_1`/`tmp0`'s census was re-run from the kernel bodies: `logit_kernel.cpp` is its only
  toucher, packing into it (`:41-45`) and copying out of it (`:49-56`). One toucher locked to both
  roles → self-loop, which is what the brief said. Worth recording that the verify step agreed.
- **The "two Metal-only compute fields" callout caught the `unpack_modes` reindex.** Going through
  the required-entry rule field by field turned up a configuration the legacy vector does not
  cover: `BITCAST` to `FLOAT32` from a non-`FLOAT32` input makes the input DFB Float32 and sets
  `fp32_dest_acc_en` while leaving `preserve_fp32_precision` false — so the validator requires an
  explicit entry exactly where legacy silently defaulted. Emitting an entry for every consumed DFB
  (`device/unary_program_factory.cpp:628-634`) covers it without changing behaviour.

## Friction

### Gaps

- **The self-audit's `cb`-leftover sweep cannot reach zero for this op, and the recipe's phrasing
  assumes it can.** The check says to run
  `grep -rnE '[Cc][Bb]_|_[Cc][Bb]\b|\b[Cc][Bb]\b|\bCB[A-Z]' <op-dir>` and *"Expect **zero** hits:
  post-port the op has no CBs, so every hit is a real leftover rather than noise."* Over
  `ttnn/cpp/ttnn/operations/eltwise/unary` it returns **34** hits across 37 files, and **none is a
  leftover**:
  - **9 hits are `CBRT`/`cbrt`** — the cube-root unary op (`common/unary_op_types.hpp:132`,
    `unary.cpp:177`, `common/unary_op_utils.cpp:26,901`). The recipe says the pattern "excludes
    `cbegin` / `cbrt`", but only lowercase `cbrt` escapes; **uppercase `CBRT` trips `\bCB[A-Z]`**.
  - **~16 hits are the nine unreferenced dataflow kernels** in `device/kernels/dataflow/` that this
    op never instantiates and that the audit explicitly marks *"out of scope, and not this op's to
    touch"* — they are lent to `untilize` / `tilize` / `transpose` / `copy` and are still on the
    legacy API by design.
  - **8 hits are `eltwise_sfpu.cpp`**, the legacy original of the kernel this port forked, which
    rung 2 requires be left untouched apart from the pointer comment.
  - **1 hit is a comment** in `common/unary_utils.cpp:71` ("for the sharded CB-aliasing path to
    work"), outside the factory body and so off-limits.

  Scoped to the 13 files the port actually converted, the sweep is **0 hits / 13 files**.
  **Suggest:** scope the check to the ported surface (the factory + the kernel entry points it
  binds + the fork) rather than the op directory, and note that the pattern's `\bCB[A-Z]` arm also
  matches SCREAMING_CASE op names. As written, an op whose directory lends kernels to other
  families can only "pass" this item by adjudicating a long list by hand, which is exactly the
  reading-not-running the item exists to replace.

- **The forced-checks procedure patches `tt_metal/impl/` when TTNN already exposes a runtime knob
  for half of it — and the knob is the half that matters most.** *Ensure the Metal 2.0 host-side
  legality checks are enabled* has the porter edit nine `skip_validation` sites across two
  `tt_metal/impl/metal2_host_api/` files, add `METAL2_CHECKS_FORCED` markers, rebuild, grep a log
  for two markers, and then remember to revert all of it — with a self-audit item that exists
  purely to catch the forgetting. But `ttnn::CONFIG`'s `validate_program_args`
  (`ttnn/api/ttnn/config.hpp:31`) gates exactly the cache-hit re-checks, is settable per-run via
  `TTNN_CONFIG_OVERRIDES='{"validate_program_args": true}'` with **no source edit and no rebuild**,
  and its own comment says *"Off by default; CI turns it on."* The cache-miss half needs no forcing
  at all: the adapter calls `MakeMeshWorkloadFromSpecs` / `SetProgramRunArgs` without the argument
  and both default to validating (`metal2_host_api/program.hpp:45-51,74`).

  So for a `ProgramSpecFactoryConcept` or `CustomProgramSpecFactoryConcept` port the source-patching
  route buys nothing the env var doesn't, and costs a rebuild plus a revert plus a self-audit item.
  It also failed in the obvious way here: this port did the patch, could not build, the patch was
  reverted, and the run that produced the green went out with the hit-path checks **off** — which is
  precisely the false-green the section is written to prevent, arrived at *through* following the
  section. **Suggest:** lead that section with the env var and the "cache-miss always validates"
  fact, and keep the source patch as the fallback for the non-TTNN entry points that the knob
  genuinely cannot reach.

- **No guidance on what to do when the *sentinel* infrastructure is wrong.** *(Lower stakes than it
  looked — it resolved by restoration; recorded because the recipe still has no shape for it.)* The
  recipe covers a missed test and tells the porter to confirm the set with the invoker. It does not
  cover this: the invoker named a specific test as the safety net for a correctness-sensitive
  declaration, pre-answered the stop-and-ask on that basis, and the test was absent (it had been
  removed by mistake and was restored once reported). Proceeding was right — the invoker's decision
  stands and the port is not the place to relitigate it — but there was no prescribed shape for
  recording "I proceeded on an authorisation whose stated evidence I could not find." **Suggest:**
  one line in *Locate and confirm the op's tests* — if a test the invoker cites by name is absent,
  continue, and report the absence as a Handoff point rather than treating it as a stop.

- **The `override_runtime_arguments` translation step says nothing about *sharing* the derivation
  with the miss path.** Its Step 1/2/3 are framed as "inventory what the ported-from override
  touches, then mirror it," which reads as writing a second, parallel implementation. For this op
  the ported-from code had already solved that by sharing one enumerator between the two methods,
  with a comment saying why ("so a cache-hit patch cannot drift from the layout the miss path
  built"). Mirroring the *inventory* faithfully while duplicating the *code* would have thrown that
  property away. **Suggest:** add a sentence — where the ported-from op shares a derivation between
  `create_descriptor` and its override, keep it shared; the fidelity rule is about the value set,
  and a duplicated derivation is a new drift risk the ported-from op did not have.

### Confusion

- **Pruning a legacy include can remove a transitive dependency the factory still needs.** The
  "CB transition is total — sweep unused `#include`s" rule is clear about removing
  `program_descriptors.hpp`, and it is genuinely unused post-port. But it was also what supplied
  `fmt` (it specialises `fmt::formatter` for `TileDescriptor`), and the factory still formats the
  compute kernel path. Nothing in the recipe suggests checking what a removed header was
  transitively providing, and under a unity build the failure would surface as an error in whatever
  else lands in the same blob. Resolved by adding `<fmt/format.h>` and `<tt_stl/assert.hpp>`
  explicitly. **Suggest:** one clause in the sweep instruction — after removing a legacy header,
  re-check the symbols the factory still uses that were not locally included.

- **`hw_config` is a variant of variants, and the natural spelling leans on two conversions.**
  `KernelSpec::hw_config` is `std::variant<DataMovementHardwareConfig, ComputeHardwareConfig>`, and
  `ComputeHardwareConfig` is itself `std::variant<ComputeGen1Config, ComputeGen2Config>`. Writing
  `.hw_config = my_gen1_config` asks the outer variant's converting constructor to reach the inner
  one through a user-defined conversion. The recipe's examples all show
  `.hw_config = create_reader_datamovement_config(...)` (which returns the *inner* variant, so one
  step) and `std::get<ComputeGen1Config>(compute_hw)` for the compute side, so neither shows the
  direct-construction case. Written as `ComputeHardwareConfig{compute_gen1_config}`
  (`device/unary_program_factory.cpp:690`) to keep it to a single conversion. **Suggest:** show the
  explicit wrap in the Style B example, since Style B is the branch that builds a `ComputeGen1Config`
  by hand.

- **Near-miss on the reader/writer RTA schema.** The brief says *"whatever schema you choose, a
  core that flips must not retain stale args"*, which invites dropping the five chunk args in the
  TILE-interleaved case (they are read only under `RM_INTERLEAVED`, and with *named* args the
  uniform-slot-layout reason for zero-filling them evaporates). The audit's *Misc anomalies* #1-#4
  closing line — *"none is yours to remove, since dropping an arg is a functional change"* — is what
  stopped that. The two documents are consistent, but they pull in opposite directions on a first
  read, and the more specific one wins. **Suggest:** have the brief's schema sentence name the
  anomaly list, so "whatever schema you choose" is bounded by "and you still emit every legacy arg."

## Open items for downstream

- **Shared kernel touches.** One, recorded in full under [Handoff points] #1: `eltwise_sfpu.cpp`,
  **lent**, rung **2 — created the fork** at
  `device/kernels/compute/eltwise_sfpu_metal2.cpp`, pointer comment confirmed landed in the legacy
  original, five remaining unmigrated consumers listed there as the sunset checklist. Bindings named
  for the kernel (`dfb::input` / `dfb::output` / `args::num_tiles`), not for unary's locals, so the
  next consumer can adopt them unchanged. Nothing was borrowed and nothing was converted in place
  that another op binds.

- **Test coverage the verification step surfaced but did not act on.** The port's confirmed sentinel
  set is the three files the invoker named, and it passes at an exact baseline match. Coverage that
  exists but was **not** in that set, and that a reviewer may want run before merge — this is the
  larger residual risk now that the row-5 gap is closed:
  `tests/ttnn/unit_tests/operations/eltwise/test_unary_activation.py`, `test_unary_fp32.py`,
  `test_unary_int32.py`, `test_unary_uint32.py`, `test_unary_uint16.py`, `test_unary_i1.py`,
  `test_unary_lgamma.py`, `test_unary_pow.py`, `test_unary_category{1,2,5}_bfloat16.py`,
  `test_unary_ops_ttnn.py`; the nightlies
  `tests/ttnn/nightly/unit_tests/operations/eltwise/test_unary_shard.py` and
  `test_unary_subcoregrids.py`; the gtest `tests/tt_eager/ops/test_eltwise_unary_op.cpp`; and
  `tests/sweep_framework/sweeps/eltwise/unary/**`. The dtype-specific files matter more than usual
  here because the compute source is selected on `(op_type, input_dtype)` and all nine sources
  flipped together — `test_unary_lgamma.py` is the only place the
  `lgamma_kernel` / `lgamma_fast_kernel` split is exercised on both sides.

- **The five chunk args are a CRTA candidate for the later name-first pass.** `RmChunkConstants` is
  computed once per call and is core-invariant by construction (its own comment says so). The only
  reason `chunks_per_row` / `chunk_size` / `last_chunk_size` / `rows_per_tile` / `total_rows` vary
  per node in the emitted args is the no-op zero-fill, which exists to serve the legacy uniform slot
  layout. Drop the zero-fill and all five become `common_runtime_arg_values`, saving five per-node
  words on every interleaved dispatch. **Not converted here** — RTA→CRTA changes dispatch semantics,
  and the zero-fill is behaviour the sentinels hold fixed.

- **Stale terminology in `common/unary_utils.cpp:71`.** The comment reads *"for the sharded
  CB-aliasing path to work (it requires whole-tile pages)"*. Post-port that path is a
  borrowed-memory DFB, and there is no CB and no aliasing involved (the port uses
  `DataflowBufferSpec::borrowed_from`, not `advanced_options.alias_with`). The file is outside the
  factory body so the port left it alone. One-word fix for whoever next touches `common/`.

- **Per-op carry-over for the eltwise family.** Two things here generalise to the sibling ops
  (`binary_ng`, `copy/typecast`, `eltwise/ternary`):
  1. The **emit-an-entry-for-every-consumed-DFB** approach to `unpack_modes`
     (`device/unary_program_factory.cpp:628-634`). It is behaviour-identical to omitting the
     `UnpackToSrc` ones, satisfies today's Float32 required-entry rule, and is already correct for
     the Int32/UInt32 extension deferred under issue #49936 — so those ops will not need revisiting
     when that lands. `copy/typecast` currently does the per-dtype case analysis instead
     (`typecast_program_factory.cpp:36-43`), which is correct today and will need a second pass.
  2. The **borrowed-DFB size check vs. `tile_size(DataFormat)`** interaction in [Handoff points] #3
     applies to any eltwise op that both sizes DFBs from `tile_size(DataFormat)` and uses
     `borrowed_from` for sharding — which is the whole family.

- **~~Working-tree scaffolding that must not be committed.~~ RESOLVED — reverted.** The
  forced-legality scaffolding that had been applied to
  `tt_metal/impl/metal2_host_api/program_run_args.cpp` and `program_spec.cpp` (nine
  `skip_validation = false;` forces — every site `grep -n 'bool skip_validation'` named across both
  files — plus two `METAL2_CHECKS_FORCED` markers, one per translation unit, all tagged
  `DO NOT COMMIT`) is gone. Confirmed two ways: `git diff HEAD -- tt_metal/` is empty, and
  `grep -n 'skip_validation = false\|METAL2_CHECKS_FORCED' tt_metal/impl/metal2_host_api/*.cpp`
  returns nothing. The self-audit's *no-forced-legality-scaffolding-in-the-diff* item now **passes**,
  and the port's diff is confined to the op's own directory. The cost of the revert is the open
  verification step in *Verification* above.
