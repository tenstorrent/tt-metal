# Metal 2.0 Port Report: `ttnn/cpp/ttnn/operations/reduction/topk`

## Outcome

**`PORTED`** for `TopKSingleCoreProgramFactory`: it is on `ProgramSpecFactoryConcept` and its tests
pass with no change in results. **`CAPITULATED`** for `TopKMultiCoreProgramFactory`: it cannot be expressed in
Metal 2.0 today; it stays on `ProgramDescriptorFactoryConcept` and keeps running. Both halves are
success-tier outcomes; the capitulation is [Handoff point 1](#1-topkmulticoreprogramfactory-cannot-be-expressed-in-metal-20-capitulation)
and is the port's most valuable finding.

Test results, same set run before and after the port
(`tests/ttnn/unit_tests/operations/reduce/test_topk.py` plus the two nightly experimental topk
files, 309 items):

| | passed | skipped | xfailed | failed |
|---|---|---|---|---|
| pre-port | 131 | 98 | 80 | 0 |
| post-port | 131 | 98 | 80 | 0 |
| post-rebase (`test_topk.py` only) | 191 | 8 | 80 | 0 |

The first two rows are identical, which is the no-regression result the port is judged on. The third
row is the same `test_topk.py` re-run after the rebase brought in fp32 input support, which adds cases
and un-skips others (see *Rebase*). Most of these shapes route to the single-core factory (`multi_core_min_width` is 8192, and
the multi-core path additionally needs a power-of-two reduction dim and `k <= 64`), so the ported
factory carries the bulk of this coverage. The set also covers shapes that route to the *untouched*
multi-core factory (e.g. `W=8192, k=50` and `W=16384, k=32`), which confirms the mixed-concept
`program_factory_t` variant dispatches correctly: the ported factory and the legacy-descriptor one
both run in the same build.

The program-cache-hit path was verified separately, since that is where a mis-wired spec factory
usually surfaces: four repeated identical calls on one device reuse a single cache entry
(`num_program_cache_entries() == 1`) and return correct values every time, so `UpdateTensorArgs`
refreshes the tensor bindings correctly.

The confirmed set's broader half (`test_graph_capture.py`, `test_reduction.py`,
`nightly/.../test_reduction_ops.py`, 5435 items) was run post-port only and finished clean:
**3621 passed, 1814 skipped, 0 failed, 0 errors** in 35 minutes, including every `test_5d_topk` and
`test_graph_capture_topk` case. It was not baselined pre-port because it takes roughly
an order of magnitude longer than the topk-specific set and almost none of it touches topk. The
reason that split is safe: this diff changes only `TopKSingleCoreProgramFactory`, its three kernel
sources, and the one factory declaration in the device-op header. Nothing outside the op reaches any
of them (the shared-kernel census below confirms no cross-op binding), and the factory is reachable
only through `TopKDeviceOperation::select_program_factory`, so a non-topk reduction op has no path to
this change.

## Provenance

- **Recipe docs (this port):** `ccf3df7c4ab 2026-08-03 docs(metal_2.0): require an explicit opt_level when porting compute kernels`
- **Audit docs (inherited):** `ccf3df7c4ab 2026-08-03 docs(metal_2.0): require an explicit opt_level when porting compute kernels`

## TTNN ProgramFactory

### Concept realized

`ProgramSpecFactoryConcept`, exactly as the audit chose, for `TopKSingleCoreProgramFactory`. No
disagreement with the audit's decision arose. `TopKMultiCoreProgramFactory` keeps
`create_descriptor`, so `program_factory_t` now holds one factory on each concept; the build is
green and `select_program_factory` dispatches per factory unchanged.

### Device-op-class edits

- Custom `compute_program_hash` deleted: **none**; the op never had one, so the default reflection
  hash was already in use.
- Pybind entry points removed: **none**. `topk_nanobind.cpp` binds only the user-facing `ttnn::topk`,
  never a factory entry point.

The only edit to `device/topk_device_operation.hpp` is the one the port forces: the single-core
factory's declaration changes from `create_descriptor` returning `ProgramDescriptor` to
`create_program_artifacts` returning `ttnn::device_operation::ProgramArtifacts`, plus the
`ttnn/metal_v2_artifacts.hpp` include that return type needs. Nothing else in the device-operation
class was touched.

### Open items

- **Relaxation candidates: none applied, and one worth a look.** All four `TensorParameter`s stay
  strict. The reader and writer kernels are written tile-index-wise and never bake in a dimension, so
  they would likely tolerate `match_padded_shape_only`; that is a correctness-sensitive opt-in and not
  a port-time call, so it is only noted here.
- **A capability this op would benefit from:** a way for a kernel to obtain a *remote* node's DFB
  address (or cross-node DFB support). That is precisely what blocks the multi-core factory, see
  Handoff point 1.
- No friction with the concept itself: the entry-point wiring was mechanical, and a variant holding
  factories on two different concepts caused no build or dispatch trouble.

## Handoff points

### 1. `TopKMultiCoreProgramFactory` cannot be expressed in Metal 2.0 (capitulation)

**Owner:** Metal 2.0 host-API / DFB framework team.

**Op / factory:** `ttnn/cpp/ttnn/operations/reduction/topk`, `TopKMultiCoreProgramFactory`
(`device/topk_multi_core_program_factory.cpp`).

**The construct that does not convert.** The factory moves data from each local core to the single
final core by having the local core read the write pointer of its **own** `c_4` / `c_5`
(`gathered_values_cb` / `gathered_indices_cb`) instance and use that value as the *destination
address on the final core*: `writer_local_topk.cpp:45-50`, used at `:69` and `:89`. That is correct
under the legacy allocator only because a CB declared over a core range set is placed at one common
address on every core in that range; a property the factory documents and orders its allocations
around (`topk_multi_core_program_factory.cpp:158-168`).

**Why mechanical conversion fails.** For the trick to keep working, `c_4` / `c_5` must each be **one**
`DataflowBufferSpec` whose derived node set spans the local cores *and* the final core. Metal 2.0's
spec validator rejects every possible endpoint assignment for such a spec:

- On a local node the only kernel that touches the buffer is `writer_local`, and it only *peeks*
  (`get_write_ptr`); no FIFO ops. The per-node census requires **exactly one producer and exactly one
  consumer instance on every node in the DFB's footprint**
  (`tt_metal/impl/metal2_host_api/program_spec.cpp:1355-1390`), so `writer_local` would have to hold
  both roles, i.e. self-loop.
- Self-looping it then trips the self-loop rule: when any kernel appears on both sides, the set of
  producer `KernelSpec`s must **equal** the set of consumer `KernelSpec`s
  (`program_spec.cpp:1438-1444`). It cannot here; the final node needs `reader_final` as producer (it
  does the `reserve_back` / `push_back`, `reader_final_topk.cpp:34-57`) and `compute_final` as
  consumer (it does the `wait_front` / `pop_front`, `topk_final.cpp:85-86`, `:107`, `:126`).
- Giving `writer_local` a single role instead leaves its node with zero of the other role, which the
  same per-node census rejects.
- `advanced_options.allow_instance_multi_binding` does **not** rescue it. It skips the role-uniformity
  checks and relaxes the census upper bound (`program_spec.cpp:1250-1306`), but the self-loop
  set-equality check at `:1438` runs **unconditionally**. It would also be wrong on the merits: per
  node this DFB is a plain 1P+1C, not a genuine multi-binding.

**Why the obvious alternative is worse.** Splitting `c_4` / `c_5` into a local-side spec and a
final-side spec satisfies the validator but breaks the transfer. Two specs over disjoint node ranges
get independent addresses, and the local instance is allocated after `c_0` / `c_1` while the final
instance is not (see the answer to audit Question 1 below), so the local core's write pointer no
longer names the final core's buffer. **The failure mode is silent mis-addressing, not an error** ,
which is why this is a stop rather than a "try it and see".

**What the off-rules change would have been**, for evaluation: either (a) cross-node DFB support, so
the transfer is declared rather than hand-addressed, `CrossNodeDataflowBufferSpec` exists in the API
surface but is rejected at validation (`program_spec.cpp:1454-1459`); or (b) a sanctioned way for a
kernel to read a bound DFB's address *on a named remote node*; or (c) relaxing the self-loop
set-equality check so a role-free peeker can coexist with a real producer/consumer pair on other
nodes; the narrowest change, and the one that would make this factory port as-is. Options I
deliberately did **not** take: threading the address through an RTA (off-whitelist raw pointer), and
inserting a padding DFB on the final core to realign the allocator watermarks (fabricating a resource
to hand-tune addresses).

**Both audit questions are answered here, for the audit's benefit:**

- **Question 1, may a `DataflowBufferSpec` keep a declared core range covering nodes where no kernel
  binds it?** There is no choice to make: `DataflowBufferSpec` has **no placement field at all**
  (`tt_metal/api/tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp:79-136`), and
  placement is derived as the union of the bound kernels' work-unit nodes
  (`program_spec.cpp:591-600`). `c_0` / `c_1` therefore narrow to the local cores necessarily.
- **Question 2, does Metal 2.0 give a DFB one common address across its whole node set?** Yes, for a
  DFB whose derived node set spans both node classes: DFB allocation runs through the same
  max-watermark allocator as legacy CBs (`tt_metal/impl/program/program.cpp:1560-1583`; the address
  is the maximum region end across the buffer's ranges, then marked on all of them). The Question 1
  narrowing alone would *not* have broken the transfer; the final core would simply carry an unused
  gap. It is the endpoint-census rejection above that makes the single spanning spec impossible.

### 2. Boundary-rule assumption violations

None. No call site outside the op directory needed a `sem::name` or a `tensor::name`. The two
in-directory kernel headers the ported kernels reach (`topk_dataflow_common.hpp` directly,
`topk_common_funcs.hpp` not at all on the single-core path) take buffer indices as plain `uint32_t`,
which the `dfb::name → uint32_t` constexpr conversion satisfies without touching either header.

### 3. Kernel-lib gaps

None. Every LLK and compute-API call site in the ported kernels
(`compute_kernel_hw_startup`, `transpose_init`, `transpose_tile`, `pack_tile`, `copy_tile`,
`copy_tile_to_dst_init_short_with_dt`, `reconfig_data_format_srca`, `pack_reconfig_data_format`,
`ckernel::topk_local_sort`) takes `uint32_t`, so named DFB handles flow in unchanged.

### 4. Removed pybind surface

None.

## Successes

- **[Go to the headers first](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#read-this-first) was the decisive advice.** Both
  of the audit's open questions, which the brief told me to get answered rather than guess, were
  answerable definitively from `dataflow_buffer_spec.hpp` and
  `tt_metal/impl/metal2_host_api/program_spec.cpp`. Hunting for a precedent would have found none
  (nothing in the tree does a cross-node CB-address transfer) and would have left me guessing on the
  exact question that decides whether the multi-core factory ports.
- **The [naming caution](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md), "take binding names from the factory, never
  from the kernel-side variable", fired correctly.** The multi-core kernels name CTAs 9/10
  `final_values_dfb_index` / `final_indices_dfb_index` while the factory passes the *gathered* buffers
  there; inferring DFB names from kernel locals would have produced crossed bindings. Even on the
  single-core path the same hazard exists in miniature: the reader calls `c_0` / `c_1`
  `dfb_id_in0` / `dfb_intermed_index`, and the compute kernel calls them `input_val` / `input_ind`,
  while the factory calls them `input_cb` / `index_cb`. Taking the factory's vocabulary
  (`dfb::input`, `dfb::index`) gave one consistent name per buffer across three kernels.
- **The [Compiler options](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#compiler-options) rule caught a silent perf loss.**
  No `KernelDescriptor` in this op sets `opt_level`, so the field reads as irrelevant. It is not: a
  `ComputeConfigDescriptor` resolves to `O3` while Metal 2.0's `CompilerOptions` defaults to `O2`, so
  the compute `KernelSpec` needed an explicit `KernelBuildOptLevel::O3`
  (`topk_single_core_program_factory.cpp:267`). Nothing would have flagged the omission; the op
  compiles and every test passes either way.
- **[Scope discipline](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#scope-discipline) held on `GENERATE_INDICES`.** The
  single-core factory hardcodes it to `"1"` with the intended expression sitting commented out beside
  it, which makes the whole precomputed-indices path (and its tensor binding) dead. The pull to "fix
  it while I'm converting the binding anyway" was real; the brief's explicit instruction not to made
  the call easy, and the port carries the binding exactly as written.
- **The shared-kernel census's "disambiguate the hits" instruction paid off on the first grep.** `grep -rl` for the three converted kernel filenames returns hits outside this op:
  `reduction/moe/device/moe_program_factory.cpp` names `reader_create_index_tensor.cpp`, and two ops
  name `topk.cpp`. Every one is a false positive of exactly the kinds the census calls out, `moe`
  binds its **own private copy** at
  `reduction/moe/device/kernels/dataflow/reader_create_index_tensor.cpp` (same filename, different
  path), the `topk.cpp` hits are two `sources.cmake` build files plus substring matches on
  `moe_grouped_topk.cpp` / `reader_moe_grouped_topk.cpp`. Taking the hit list at face value would have
  produced a needless `_metal2` fork and a pointer comment in a peer op's directory; checking the bound
  *path* rather than the filename settled all of them in a minute.
- **The [endpoint-assignment procedure](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-two-toucher-dfb--assign-1p1c-dual-instance-work-split)
  is genuinely re-derivable.** Re-running the census on the single-core factory from the kernel bodies
  took a few minutes and agreed with the brief on all eight buffers, which then made the *disagreement*
  on the multi-core side (where no endpoint assignment works) trustworthy rather than a suspected mistake of
  my own.

## Friction

### Gaps

- **The audit framed Question 1 as a porter choice, and it is not one.** The brief says "Two readings
  are open: keep the range and accept a DFB spanning nodes where nothing binds it, or narrow the two
  specs to `local_cores_range_set`", and the audit's Recipe note 3 asks for a table row covering
  "live on one class of node, unreferenced on another". Both frame it as a range decision. But
  `DataflowBufferSpec` has no range field, so there is nothing to keep or narrow; the answer is
  structural, not a judgment call. The [CB endpoints](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/audit/metal2_audit.md#cb-endpoints) subject
  would be clearer if it stated up front that a DFB's node set is *always* exactly the union of its
  binding kernels' nodes, and that the only real question is which kernels bind it.
- **Nothing in the recipe, catalog, or audit covers a kernel that learns a *remote* node's buffer
  address from its own instance of that buffer.** This is the multi-core blocker, and it is not a
  variant of any catalogued pattern: it is not a hidden co-filler, not a two-toucher work-split, not a
  borrowed-memory DFB. The audit did spot the *dependency* (its Question 2, correctly called the
  highest-value thing to verify) but had no gate to fail on, so a factory that provably cannot port
  cleared GREEN. A catalogued anti-pattern, "a kernel that uses one node's DFB address as an address
  on another node cannot port; check for `get_write_ptr` / `get_read_ptr` feeding a NoC address whose
  `noc_x` / `noc_y` is a *different* node", would turn this into an audit gate and save the next
  porter the derivation. The recognition grep is cheap and specific.
- **The self-loop set-equality rule is invisible from the docs.** The recipe and catalog explain that
  self-loop and multi-binding must never stack, and that the flag is the last resort for ≥3 touchers.
  Neither says that the self-loop check *runs unconditionally*, i.e. that the flag cannot rescue a
  self-looping kernel that shares a DFB with an unrelated kernel. That single fact is what closed off
  the last option for `c_4` / `c_5`, and it is only visible at `program_spec.cpp:1425-1444`. Worth a
  sentence in the self-loop pattern entry.
- **No guidance on a host-declared named CTA the kernel never reads.** `Ht` is declared and unused in
  *all three* single-core kernels (the audit recorded only the reader and writer; the compute kernel
  has it too, now at `topk.cpp:134`). The recipe's build-failure list covers the opposite direction (a
  kernel referencing a name the host did not declare) but says nothing about the dead-CTA direction. I
  ported them across as the brief instructed, and nothing complained; a sentence confirming that a
  host-declared, kernel-unread named CTA is harmless would remove the doubt.
- **The recipe's `-k` advice does not work for ops under `tests/ttnn/`.** It suggests excluding
  not-yet-converted paths with `-k`, but the `pytest_make_parametrize_id` hook renders IDs as
  `argname=value`, and pytest's `-k` grammar rejects `=` ("expected end of input; got ="), including in
  quoted form on this pytest version. Selecting a subset of such tests needs full node IDs (or
  `--deselect`). Minor, but it cost a few minutes mid-verification.
- **Nothing warns that editing a kernel source while a test run is live reproduces the
  unconverted-kernel crash.** The recipe warns that a *selected-but-unconverted* kernel path can take
  down a whole pytest session with a JIT `static_assert` on `get_compile_time_arg_val`, and tells you
  to exclude those paths with `-k`. What it does not say is that the same symptom appears if you
  simply *touch* a kernel file mid-run: host code is compiled into `_ttnn.so`, but kernel sources are
  read from the working tree at JIT time. I hit this exactly once, and self-inflicted; I reverted
  `topk.cpp` to re-apply its edits without whole-file formatting while a 5400-test suite was running,
  and the JIT immediately compiled the legacy positional-CTA kernel against the new named-arg host
  schema (`get_ct_arg<8>` on an empty CTA list, plus a `topk_local_sort` overload mismatch). Re-running
  after the edits settled was clean. Worth one sentence in [Running builds and
  tests](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#running-builds-and-tests-without-flooding-your-context):
  host edits are safe during a run, kernel edits are not. (The silver lining: it is a decisive
  confirmation that the ported host schema really is driving the JIT; a stale kernel cannot compile
  against it.)
- **`clang-format` on a whole file is the wrong tool for this repo, and the recipe does not say so.**
  This repo's pre-commit hook is `git-clang-format`, which formats **changed lines only**. Several
  op files are not whole-file clang-format-clean, so running `clang-format -i` on one silently adds
  unrelated comment-realignment churn to the port diff, 22 extra lines in `topk.cpp` in my case,
  exactly the attribution-muddying the [scope discipline](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#scope-discipline)
  section exists to prevent. `git-clang-format --diff` is the check to run instead.
- **The log-reading step assumes a subagent is available.** The recipe prescribes handing build/test
  logs to a Sonnet subagent; in this session that route was closed, and redirecting to a file plus
  grepping targeted slices worked just as well for both a cold build and a 309-test run. Phrasing the
  step as "keep the raw output out of context, delegate it or grep targeted slices" would make it
  robust to the harness.

### Confusion

- **Whether accessor names must be unique per kernel or per resource kind.** The writer ends up with
  both a `dfb::values` and a `tensor::values` (its output-values buffer and its output-values tensor),
  which reads oddly. Reading the validator settled it, accessor names are checked for uniqueness
  within each binding kind, and the generated tokens live in separate namespaces, so this is legal. I
  kept it because it matches the kernel's own locals (`values_dfb`, `values_tensor_accessor`), but the
  docs never state the scoping rule and the examples never exercise the collision.
- **Style B and the `double_buffer_dest` inversion.** The recipe's table correctly says
  `double_buffer_dest = !dst_full_sync_en`, but not that the overwhelmingly common legacy value
  (`dst_full_sync_en = false`) inverts to the Metal default (`true`), so the field usually needs no
  action at all. I spent a while confirming that setting it explicitly was not accidentally changing
  something. I did set it explicitly, with a comment, so the inversion is visible to a reviewer. A
  parenthetical in the table ("the common legacy `false` maps to the Gen1 default, so most ports need
  no assignment here") would resolve this instantly.
- **The `unpack_modes` forced-entry rule needs a "check the reachable dtypes first" lead, and a
  warning that the answer can change under you.** Deciding whether the rule fires meant tracing the
  op's dtype validation through to every buffer's format. At the time it did not fire: the op accepted
  only `BFLOAT16` / `BFLOAT8_B`, so no consumed DFB could carry a 32-bit float format and the table
  stayed empty. A rebase then landed fp32 input support, which flipped the answer, and the port now
  carries three explicit `UnpackToDest` entries (see *Rebase* below). Two doc suggestions fall out:
  lead the section with "first check which dtypes the op's validator actually admits, for many ops no
  consumed DFB can be Float32", and note that a port whose table is empty *because of a dtype
  restriction* should say so in a comment, since the restriction is the only thing holding the entry
  back.

## CI

All six standard workflows were dispatched on the port branch. **No failure is attributable to the
port**; each red traces to one of four causes already present on `main` or to CI infrastructure.

| Workflow | Result | Cause of any red |
|---|---|---|
| PR Gate | pass | |
| Merge Gate | pass | |
| Sanity tests | 3 jobs red | pre-existing on `main` (below) |
| Blackhole sanity | 1 job red | pre-existing on `main` (below) |
| Nightly L2 | 20 pass / 2 red | suite timeout + runner setup timeout |
| Performance / models | 91 pass / 10 red | 6 infrastructure, 4 pre-existing |

Notably `ttnn nightly reduction tests [bh_p100]` **passed**, so the category covering topk is green on
Blackhole hardware.

The four root causes:

1. **A tie-break regression from `3cff0510b9e` (#50687)**, which accounts for every sanity and
   blackhole-sanity failure. All of them are
   `test_tiebreak_input_adjust.py::test_tiebreak_boosts_lowest_global_index_for_greedy_users`. That
   commit *added* the failing test and is an ancestor of this branch's merge-base. `main` fails the
   identical jobs on its three most recent runs of both workflows. The failing path is
   `ttnn.max` / `eq` / `lt` / `min` / `abs` / `multiply` inside
   `TTSampling._adjust_values_for_tiebreak`, which touches no topk code. A local A/B settled it: the
   file fails **15 of 18 identically with the port applied and with it reverted**.
2. **`ttnn nightly reduction tests` no longer fits its 45-minute budget.** It timed out mid-progress
   with zero assertion failures, having passed 3564 tests and still passing roughly one per second in
   `test_generic_ops_w_scalar` (a var/std test). `main`'s same job times out the same way.
3. **CI infrastructure.** Six models jobs and one L2 job failed with GitHub `codeload` archive
   download timeouts after 3 attempts, or a runner setup timeout that left the test step *skipped*.
   None ran a test.
4. **Pre-existing model failures.** `bge_m3` and `stable_diffusion` are red on `main`'s
   frequent-models runs; `efficientdetd0` fails a PCC 0.92 threshold. None of those models references
   `ttnn.topk` or `ttnn.sampling`.

Two items worth routing onward, neither belonging to this branch:

- **#50687's breakage is under-reported by CI.** Blackhole reports one failing parametrization; on
  Wormhole hardware all 15 cases in that file fail. The boost is never applied at all (the tied maxima
  stay exactly equal), which points at the eltwise / reduce chain rather than anything arch-specific.
- **The nightly reduction suite needs a longer timeout or a split.**

## Rebase

The branch was rebased onto a newer `main` after the port was written. One conflict, in
`device/topk_single_core_program_factory.cpp`, from upstream adding **fp32 input support** to the same
factory. Three upstream changes were folded into the Metal 2.0 version:

1. **`is_fp32_input`** (`input_cb_data_format == tt::DataFormat::Float32`) plus its rationale comment:
   fp32 is deliberately *not* downcast to bf16 the way bfp8 / bfp4 are, because with a 32-bit dest
   register and unpack-to-dest the value buffers stay fp32 through the sort. Carried over verbatim.
2. **The reader's index-width flag now derives from the index dtype**, not from the dimension size:
   `output_ind_cb_data_format == tt::DataFormat::UInt16` rather than `uint16_output`. The named
   argument keeps its `uint16_output` name (the kernel side is unchanged upstream, and the name still
   describes the flag), only the host-side value changes. This matters because the factory's local
   `uint16_output` is a pure dimension-size test and does not know about fp32, whereas the index dtype
   does.
3. **The compute hardware config** gains `enable_32_bit_dest = !uint16_output || is_fp32_input` and,
   under fp32, the three `unpack_modes` entries. The legacy form is a `vector<UnpackToDestMode>`
   indexed by buffer index with `UnpackToDestFp32` on `c_0` / `c_2` / `c_4`; the Metal 2.0 form is the
   name-keyed table shown in `METAL2_PORT_PLAN.md`. The index reordering and the
   `UnpackToDestFp32` → `UnpackMode::UnpackToDest` value translation are the two transforms the
   recipe warns flip silently, so both were done against the table rather than by eye.

Nothing about the port's structure changed: same DFB set, same bindings, same work units, same
argument schema. Upstream did not touch any of the three kernel sources, so those merged cleanly.

**The fp32 path is verified, not just merged.** `unpack_modes` was empty when the port was written and
is now populated, and it is a setting the recipe calls out as silently wrong in either direction, so it
needed a real run rather than a careful reading. Upstream added `ttnn.float32` to `test_topk.py`'s dtype
parametrization in the same change, so the coverage exists: post-rebase that file is
**191 passed, 8 skipped, 80 xfailed, 0 failed**, with 164 distinct `FLOAT32` cases among them. For
comparison the pre-rebase run of the same file was 131 passed / 98 skipped / 80 xfailed, so the fp32
dtype both added cases and un-skipped existing ones. A wrong `UnpackToDest` entry, or a missing one
under a 32-bit dest register, would corrupt the sort's comparisons and fail these.

## Open items for downstream

- **Shared kernel touches; none taken, one to keep coordinating.**
  `device/kernels/compute/topk_common_funcs.hpp` is **lent** by this op to two others
  (`experimental/reduction/deepseek_grouped_gate/device/kernels/compute/deepseek_grouped_gate.cpp:13`
  and
  `experimental/deepseek_prefill/moe_grouped_topk/device/kernels/compute/moe_gate_common_compute.hpp:24`).
  Rung taken: **none**; no `_metal2` fork was created and the header was not modified. It is reached
  only from the multi-core compute kernels, which this pass does not port, and every function in it
  takes its buffer index as a `uint32_t`, so a future multi-core port can pass `dfb::` tokens straight
  into the existing signatures. Remaining unmigrated consumers: the two op directories above (both
  gated on the readiness sheet today). `device/kernels/dataflow/topk_dataflow_common.hpp` is private to
  this op, is included by the ported reader, and needed no change for the same `uint32_t` reason.
- **`TopKMultiCoreProgramFactory` is the remaining work for this op**, and it is blocked on framework
  capability rather than porter effort, see Handoff point 1. Its six kernels are all still on the
  legacy positional-CTA form, so they convert together with it and not before.
- **Per-op carry-over.** Any op that hand-rolls a cross-core transfer by reading a local CB's pointer
  as a remote address hits the same wall. `writer_local_topk.cpp` is the pattern to grep for. Worth
  screening the reduction and CCL-adjacent families before they are audited, so they are not cleared
  GREEN on a factory that cannot port.
- **Pre-existing dead code, left in place** (all noticed during the port, none touched):
  - `topk_single_core_program_factory.cpp:78` declares `cores` (`corerange_to_cores(...)`) and never
    uses it; the per-core loop walks `group.ranges()` instead.
  - `Ht` is a declared-and-unused CTA in all three single-core kernels
    (`reader_create_index_tensor.cpp:21`, `writer_binary_interleaved.cpp:17`, `topk.cpp:134`). The
    audit recorded the first two; the compute kernel is a third instance.
  - The audit's Misc anomalies 1-7 remain open for the ops team; the port neither fixed nor worsened
    any of them. Anomaly 2 (`GENERATE_INDICES` hardcoded, GH 36329) is the one that most affects
    review of this diff: it is why the reader's `tensor::indices` binding is ported but unreachable.
- **The fp32 input path is covered on Wormhole only.** It arrived with the rebase and is verified
  there (see *Rebase*), but the run was local, on Wormhole hardware. Blackhole exercises the same
  `unpack_modes` and 32-bit-dest configuration and has its own dest-register behaviour, so the
  `reduction` category should be confirmed green on Blackhole for the rebased branch before merge.
- **Test coverage notes.**
  - There are **no C++ gtests** for topk anywhere; coverage is Python-only. The recipe's
    gtests-first-then-pytests order does not apply to this op.
  - The single-core `indices` input path is dead under the hardcoded `GENERATE_INDICES`, so the
    conditional `INPUT_INDICES_TENSOR` binding this port introduces is **compiled but never
    executed**. When GH 36329 is fixed, that binding needs a test that actually supplies an indices
    tensor to the single-core path (today `test_topk_sub_core_grids` and `test_topk_large_2d_shapes`
    pass one, but the define ignores it).
  - The confirmed test set's broader half (`test_reduction.py`,
    `nightly/.../test_reduction_ops.py`, `test_graph_capture.py`) is ~5400 items and dominates
    runtime. I ran the 309 topk-specific items as the pre/post no-regression baseline and the broader
    suites once post-port; a future porter of the multi-core factory can use the same split.
- **RTA→CRTA candidates: none.** `id` differs per node and `work_per_core` differs between the two
  core groups, so neither is broadcast-uniform.
