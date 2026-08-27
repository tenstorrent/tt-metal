# Metal 2.0 Port Report — `ttnn/cpp/ttnn/operations/experimental/paged_cache`

## Outcome

**`PORTED` (partial) + `CAPITULATED` (partial)** — the op's eight factories split cleanly in two, and
this pass delivers both halves of that split as a deliberate result rather than a stopping point:

- **`PORTED`** — `PagedUpdateCacheProgramFactory` and `PagedFillCacheProgramFactory` are on
  `CustomProgramSpecFactoryConcept`, with five `_metal2` kernel forks. These are the factories
  `select_program_factory` picks whenever `mesh_coords` is `nullopt`, which is the default on every
  public entry point — so this is the common path, not a corner.
- **`CAPITULATED`** — the four `*MeshWorkloadFactory` factories are **blocked on framework work**:
  they need a per-mesh-coordinate `ProgramSpec` / `ProgramRunArgs`, which no Metal 2.0 TTNN factory
  concept provides. This is `ttnn_factory.md`'s own *Feasibility gate* → **"Multi-program / per-coord
  variation"** RED case; the audit cleared it in error. See [Handoff points](#handoff-points) #1.
- **Deferred, not capitulated** — the two *fused* single-device factories
  (`PagedTiledFusedUpdateCacheProgramFactory`, `PagedRowMajorFusedUpdateCacheProgramFactory`) are
  left for a later pass: the audit's own open design **Question #1** is still unanswered, and the
  brief instructs *"Get an answer before you write the fused specs."* See
  [Open items](#open-items-for-downstream) #1.

**⚠ Not built, not tested — see [Friction](#friction) #1.** The required toolchain (`clang-20`) is
not installed on the machine this port ran on, so `./build_metal.sh` cannot configure. Every check in
[Verification performed](#verification-performed) below is static. The invoker asked to run the build
and tests themselves; the commands, including the mandatory legality-check forcing, are in
[Handing the build and test run back](#handing-the-build-and-test-run-back).

## Provenance

- **Recipe docs (this port):** `50e992a8ec2 2026-08-21 docs(metal_2.0): a run in flight freezes the kernel sources`
- **Audit docs (inherited):** `50e992a8ec2 2026-08-21 docs(metal_2.0): a run in flight freezes the kernel sources`

The working checkout carries no `metal_2.0` doc tree, so
`git log -1 --format='%h %cs %s' -- docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/`
prints nothing there. The hash above is pinned from the sibling doc-branch checkout
`/localdev/edwinlee/Port_Recipe`, whose `ai/port/metal2_port.md` was verified byte-identical
(`diff -q`) to the recipe this port was handed — the same pin, and the same verification, the audit
recorded.

## TTNN ProgramFactory

### Concept realized

`CustomProgramSpecFactoryConcept`, as the audit chose, on the two factories this pass ported. Each
implements `create_program_artifacts` plus a `ProgramRunArgs`-returning `override_runtime_arguments`.

**Cache-hit tensor-binding completeness** (the silent failure `ttnn_factory.md` warns about — on this
concept the framework refreshes *nothing* for you): each override returns a `TensorArgument` for
**every** `TensorParameter` the spec declares, in every configuration. Nothing is skipped.

- `PagedUpdateCacheProgramFactory::override_runtime_arguments` → `cache`, `input`, and (when present)
  `index`, `page_table`. The `input` entry is load-bearing beyond its own accessor: it is what
  refreshes the **borrowed-memory** `input` DFB's backing L1 address, the job the ported-from body
  did with `UpdateDynamicCircularBufferAddress`.
- `PagedFillCacheProgramFactory::override_runtime_arguments` → `input`, `cache`, `page_table`, and
  (when present) `batch_idx`, `valid_seq_len`.

**Non-tensor refreshes mirror the ported-from set exactly, no more and no less:**

| ported-from override wrote | Metal 2.0 |
|---|---|
| `update_cache` reader `[1]`, writer `[1]`/`[2]` — only when `offsets` is non-empty | `cache_start_id` / `cache_tile_offset_B` named RTAs, under the same `offsets.empty()` guard |
| `update_cache` reader `[0]`/`[2]`/`[4]`, writer `[0]`, and the input CB re-point | `tensor_args` (4 entries) |
| `fill_cache` reader `[3]`, writer `[5]` (`noop`) | `noop` named RTA on both kernels |
| `fill_cache` writer `[4]` (`batch_idx_fallback`, scalar path only) | `batch_idx_fallback` named RTA, declared and refreshed only on the `!use_batch_idx_tensor` path |
| `fill_cache` reader `[0]`, writer `[0]`/`[1]`/`[4]`(tensor path)/`[6]` | `tensor_args` (3–5 entries) |
| *(neither: `start_tile_id` / `start_row_num` / `num_rows` / `my_batch_idx` / `wait_to_start` / `send_*`)* | **not** refreshed — identical, deliberately |

`UpdateProgramRunArgs` is a partial update, so everything omitted keeps its cache-miss value —
which is exactly the ported-from behaviour.

### Device-op-class edits

- **Pybind entry points removed: none.** `paged_cache_nanobind.cpp` binds only the three public entry
  points via `ttnn::bind_function` (`:48`, `:89`, `:134`); no `create_descriptor` was ever pybound, so
  the port makes **no user-visible API change**.
- **Custom `compute_program_hash`: left intact, untouched**, on all three DeviceOperations —
  `paged_update_cache_device_operation.cpp:313`, `paged_fill_cache_device_operation.cpp:207`,
  `paged_fused_update_cache_device_operation.cpp:371`.
- **No device-operation-class file was edited at all.** The `TT_FATAL` census below confirms the three
  device-op `.cpp` files are byte-identical in guard count (45 / 25 / 40, unchanged).
- One structural change *inside the port's own writeable surface*, forced by the split: in each ported
  factory `.cpp` the ported-from `create_descriptor` body and the ported-from `Program&`-mutating
  patch moved verbatim into anonymous-namespace helpers
  (`build_paged_update_cache_descriptor` / `patch_paged_update_cache_runtime_args`, and the
  `fill_cache` equivalents), because the blocked `*MeshWorkloadFactory` sibling still needs them.
  `fill_cache` already had this shape for the descriptor; `update_cache` acquired it.

### Open items

See [Open items for downstream](#open-items-for-downstream).

---

## Handoff points

### 1. **Framework gap — a Metal 2.0 spec factory cannot vary its program across mesh coordinates.** *(the port's headline finding; owner: Metal 2.0 / TTNN framework)*

**What is needed.** `paged_cache`'s four `*MeshWorkloadFactory` factories build a *different* program
per mesh coordinate. Both Metal 2.0 factory concepts are single-program:
`create_program_artifacts(attrs, tensor_args, tensor_return_value)` takes **no**
`mesh_dispatch_coordinate`, and `ProgramSpecMeshWorkloadFactoryAdapter::create_mesh_workload`
(`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:912-921`) emplaces *the same* `artifacts.spec` for
every range in `tensor_coords` and applies *the same* `artifacts.run_params` to every resulting
program through `SetProgramRunArgs`. There is no per-coordinate hook on the cache-miss path.

**The two legacy idioms that need one** (the brief is explicit that neither is the port's to
normalise — *"Preserve both behaviours as they are"*):

- **Empty-descriptor idiom** — `paged_update_cache_program_factory.cpp:1132-1139` (post-port; `:448-453`
  in the pre-port tree the brief cites), tiled fused `:544-549`, RM fused `:547-552` (both unchanged). A coordinate outside `operation_attributes.mesh_coords` gets an
  **empty `ProgramDescriptor`**, and the descriptor adapter then *skips adding a program for that
  coordinate entirely* (`mesh_device_operation_adapter.hpp:588-592`). A `ProgramSpec` has no way to
  say "no program here" — a valid spec needs at least one `WorkUnitSpec` with every kernel referenced.
- **`noop`-RTA idiom** — `paged_fill_cache_program_factory.cpp:62-70` and `:879-890` (post-port;
  `:33-40` / `:348-359` pre-port). The spec *is*
  identical across coordinates; only the initial value of the `noop` runtime arg differs. The
  cache-**hit** path is fine (`override_runtime_arguments` receives the coordinate), so this gap is
  narrower — **per-coordinate run args on the cache miss** — but the miss dispatch executes, so a
  coordinate the caller excluded would do a real cache fill on the first call.

**Why it matters that this reached the port.** `ttnn_factory.md` → *Feasibility gate* already names
this case: *"Multi-program / per-coord variation. The op's programs genuinely differ across mesh
coordinates (CCL-style). The single-program adapter stamps one spec everywhere."* → *"the port is
blocked on framework work, not porter-resolvable. Record RED and stop."* The audit cleared these four
factories GREEN anyway — it filed the two mesh idioms under *Watch for* (behaviour to preserve)
rather than running them through the factory-concept gate. **The gate exists and it should have
fired**; see [Friction](#friction) #2 for the doc/process suggestion.

**The intended vehicle already has a name.** `ttnn/api/ttnn/metal_v2_artifacts.hpp:20-22`: *"A future
`MeshWorkloadSpecFactoryConcept` will return a different (multi-program) artifact type for ops whose
programs vary across the mesh."* This op is a concrete customer for it, with two distinct shapes
(a program that is *absent* on some coordinates, and one that is *present but parameterised* by
coordinate). A narrower fix that would unblock `fill_cache` alone: a per-coordinate hook on the
cache-miss `ProgramRunArgs` (the cache-hit side already has one).

**Cost paid in this port because of it:** five kernel forks (next entry).

### 2. **Five intra-op `_metal2` kernel forks created.** *(coordination signal; owner: this op's next porter)*

Because each `*MeshWorkloadFactory` binds the *same* kernel sources as the single-device sibling that
did convert, converting those sources in place would have broken the four blocked factories. Per
*Caution: Porting a shared kernel* — **rung 2 (create the fork), intra-op shape** — each fork was
created beside its original, the original was left untouched apart from the mandated pointer comment,
and the originals keep serving the mesh factories.

| fork created (all under `device/kernels/`) | forked from | remaining consumers of the original |
|---|---|---|
| `dataflow/reader_update_cache_interleaved_start_id_metal2.cpp` | `reader_update_cache_interleaved_start_id.cpp` | `PagedUpdateCacheMeshWorkloadFactory` |
| `dataflow/writer_update_cache_interleaved_start_id_metal2.cpp` | `writer_update_cache_interleaved_start_id.cpp` | `PagedUpdateCacheMeshWorkloadFactory` |
| `compute/update_cache_metal2.cpp` | `compute/update_cache.cpp` | `PagedUpdateCacheMeshWorkloadFactory` |
| `dataflow/reader_fill_cache_interleaved_metal2.cpp` | `reader_fill_cache_interleaved.cpp` | `PagedFillCacheMeshWorkloadFactory` |
| `dataflow/writer_fill_cache_interleaved_metal2.cpp` | `writer_fill_cache_interleaved.cpp` | `PagedFillCacheMeshWorkloadFactory` |

No `_metal2` fork existed beside any of them beforehand (checked locationally, by `ls` of each
original's directory — not by a tree-wide grep). No build-system change was needed: the op's kernels
are installed by a `file(GLOB_RECURSE …)` that already covers these directories.

**Sunset:** when handoff point #1 lands and the four mesh factories convert, all five originals are
deleted and the forks take their names. Until then, **a fix to either copy should be evaluated for
the other** — these five pairs are now a drift-discipline liability inside a single op directory, which
is an unusually tight coupling for the fork convention and a reason to prioritise #1.

### 3. **`ttnn.experimental.paged_fill_cache`'s `noop` attribute is dead API surface.** *(owner: ops team; carried forward from the audit, confirmed during the port)*

`PagedFillCacheParams::noop` (`paged_fill_cache_device_operation_types.hpp:16`) is never set `true` by
any caller — the sole public entry point hardcodes `.noop = false`
(`paged_fill_cache_device_operation.cpp:237`). The functional noop path is driven entirely by the
mesh-coordinate test inside `paged_fill_cache_noop`. **The port preserves it**, and it is called out
here because it interacts with handoff point #1: once a per-coordinate mechanism exists, the attribute
and the coordinate test are two sources for one flag and should be reconciled together.

---

## Successes

**1. `dataflow_buffer_spec.hpp`'s PLACEMENT note stopped a wrong answer to the audit's open question
before any code was written.** Approaching the fused factories, the natural first move was the
audit's own first option — *"bind both `src1` and `src2` and rely on per-node existence."* The
header's one-line contract (*"PLACEMENT: Derived — the DFB's effective node set is the union of its
bound kernels' `WorkUnitSpec` `target_nodes`"*) makes it immediately clear that binding both on a
`KernelSpec` spanning `all_cores_bb` **places both DFBs across the whole bounding box**, where legacy
allocated each over only its own half — and that both are `borrowed_from` an L1-sharded tensor with no
shard on the other half. That is a different question from the one the audit posed, and reading the
declaring header answered it faster than any precedent would have. It is why the fused factories are
deferred rather than guessed at. (Recipe: *"Go to the headers first; they are ground truth."*)

**2. *Pattern: Conditional / optional DFB bindings* fired exactly where the brief predicted, on a
site the brief flagged and the recipe explained.** The brief's *"Kernels declare CB wrapper objects
unconditionally for conditionally-allocated CBs"* heads-up names
`reader_update_cache:56-57`, `writer_update_cache:61-62` and `writer_fill_cache:92-93` — harmless
under the legacy `uint32_t` CB id, a compile error under a named binding. The catalog's
*"Promote a CTA gate to a define"* paragraph is precisely this port's shape: `use_index_tensor`,
`is_paged_cache`, `use_batch_idx_tensor` and `use_valid_seq_len` were all CTAs gating
`if constexpr` blocks that name conditionally-bound resources, and all four became
`compiler_options.defines` + kernel-side `#ifdef`. The catalog's warning that the define must reach
**every** kernel that references the resource caught the `update_cache` case, where both the reader
*and* the writer touch `index` / `page_table` — hence one shared `conditional_defines` table fed to
both (`paged_update_cache_program_factory.cpp:762-768`, fed to both kernel specs at `:773` and `:839`).

**3. The `hw_config` section's insistence on porting *values, not spellings* caught a real
divergence.** `update_cache` resolves a full TTNN `ComputeKernelConfig` but hand-copies only
`fp32_dest_acc_en` onto its `ComputeConfigDescriptor` — the recipe's *"Check for a dropped field
before using the helper"* case, verbatim. Routing that through `to_compute_hardware_config` would have
substituted the helper's **high-performance** defaults for `math_fidelity`, `math_approx_mode` and
`dst_full_sync_en`, none of which this op ever applied. Building `ComputeGen1Config` directly
reproduces the legacy `ComputeConfigDescriptor` defaults exactly (verified field by field:
HiFi4 = HiFi4; `math_approx_mode=false` = `Precision::Precise`; `bfp8_pack_precise=false` =
`Precision::Approximate`; `dst_full_sync_en=false` = `double_buffer_dest=true`).

**4. The `opt_level` "absent line" check earned its place in the checklist.** `grep -n opt_level` over
both ported-from factories returns **nothing** — which reads as "nothing to carry across" and is
exactly the trap the recipe describes. `update_cache`'s compute kernel resolves to `O3` on a
`ComputeConfigDescriptor` and would have silently dropped to `O2` on `KernelSpec::compiler_options`.
Set explicitly at `paged_update_cache_program_factory.cpp:938`.

---

## Friction

### Gaps

**1. The port could not be built or tested: the toolchain is absent from the machine.**
`./build_metal.sh` fails at CMake configure with
`The CMAKE_C_COMPILER: clang-20 is not a full path and was not found in the PATH`; only clang-14 is
installed (`/usr/bin/clang-14`, `/usr/lib/llvm-14`). A sibling checkout's `CMakeCache.txt` shows
`/usr/bin/llvm-ar-20`, so the box *did* carry clang-20 at some earlier point — it is gone now.
Installing a system toolchain was out of scope, and the invoker had asked to run the build and tests
themselves.

Two secondary things this surfaced, both worth a line in `workspace_setup.md`:

- **A `git worktree` does not inherit submodules.** The first configure failed with
  `Missing submodules. Run: git submodule update --init --recursive` — obvious in hindsight, but the
  recipe's *Workspace bootstrap* assumes a fresh clone and says nothing about worktrees, which is a
  natural way to isolate a port. Fixed by running the command inside the worktree.
- **The legality-check forcing step is wasted work if the build then fails.** The recipe orders
  *force → build → prove*, so the `skip_validation` scaffolding was applied to all nine grep-named
  sites before the build was attempted, then reverted when the build proved impossible. Cheap here,
  but a one-line "confirm you can build before you force" would save the round trip.

**Everything in [Verification performed](#verification-performed) is therefore static.** The port's
highest-risk unverified surfaces, in the order I would check them:
1. Does the whole thing compile (host and the five JIT kernel forks)?
2. Does `ValidateProgramSpec` accept the `update_cache` alias group and the three `fill_cache`
   self-loops?
3. Do the ported factories still cache-hit and produce identical numerics on the second dispatch?

**2. `ttnn_factory.md`'s feasibility gate is not represented in the audit's own gate set, so a
listed-RED condition passed GREEN.** The per-coord-variation blocker (handoff point #1) is documented
clearly in `ttnn_factory.md` → *Feasibility gate*, which is the doc the audit's *final step* invokes.
But the audit's status summary is organised around Appendix A features + the named gates (Device 2.0,
Features, TTNN factory concept, Offset base pointers, TensorAccessor 3rd arg), and "TTNN factory
concept" was answered as *which* concept rather than *whether the op fits one*. The brief's *Watch
for* section then describes both mesh idioms accurately — it just files them as behaviour to
preserve.

*Suggestion:* make the **fit** half of the factory gate explicit and mechanical, alongside the
concept choice — a checkbox the auditor must answer with evidence, e.g. *"Does any factory's
`create_descriptor` read `mesh_dispatch_coordinate` for anything other than pass-through? If yes →
per-coord variation → RED."* That single question would have caught all four factories here: three
return an empty descriptor on it, and the fourth feeds it into `paged_fill_cache_noop`. As written,
an auditor can complete the *TTNN factory analysis* section fully and correctly without ever
evaluating the gate the section's own doc defines.

### Confusion

**3. "Both members of a factory pair convert together" and "the atomic unit is one ProgramFactory"
point opposite ways when the pair straddles a framework gap, and the recipe does not adjudicate.**
The brief says *"Do not port one member of a factory pair without the other — they share the body, so
they convert together."* The recipe says a `program_factory_t` variant is *valid* with alternatives on
different concepts, and that a finished factory with the rest reported is a shippable deliverable.
Here the pair genuinely cannot convert together (#1), so one of the two had to give.

The resolution turned out to be already written down, just not where it was being looked for: the
*atomic-unit* note routes the **shared top-level entry point** case to *Caution: Porting a shared
kernel*, whose **intra-op** rung says the fork lands in your own directory and your op's other
factories are the "remaining consumers." It also says co-porting the sibling is *"only the simpler
move when that factory doesn't drag in shared kernels of its own"* — which quietly assumes the
sibling *can* be co-ported at all. It cost maybe twenty minutes of re-reading to be confident the
fork was sanctioned rather than an improvisation, mostly because a brief-level "do not split this
pair" reads as more categorical than a catalog rung.

*Suggestion:* one sentence in the shared-kernel Caution's intra-op bullet — *"if the sibling factory
is blocked on framework work, the fork is the answer; the brief's 'convert together' guidance assumes
both are portable."*

**4. The `_metal2` fork naming rule and the intra-op case pull in different directions.** *Name the
bindings for the kernel, not for your op* exists so a later consumer can reuse the fork. In the
intra-op case there is no later consumer — the fork's only future is to be deleted when the sibling
converts and to give up its name to the original. The rule was followed anyway (accessor names come
from the kernel's own vocabulary: `tensor::src` from `src_addr`, `dfb::in` from `cb_in`), and it cost
nothing, but it is worth noting that the guidance's stated *reason* does not apply here. One place it
bit for real: the `update_cache` writer's own vocabulary calls the **output** DFB `cache`
(`cache_cb_id` kernel-side is `output_cb_index` host-side — the brief flags this exact trap). Keeping
the kernel's word was the minimal-diff choice and is defensible — the buffer really does hold cache
data — but the host-side binding needed an explicit comment
(`paged_update_cache_program_factory.cpp:869-872`) to keep the mapping legible.

---

## Open items for downstream

### 1. The two **fused** single-device factories, and the design question that gates them

`PagedTiledFusedUpdateCacheProgramFactory` and `PagedRowMajorFusedUpdateCacheProgramFactory` are not
ported. They are not blocked by handoff point #1 (that blocks only their mesh siblings); they are
blocked by the audit's own **Question #1**, which the brief says must be answered upstream *before*
the fused specs are written. Reading `dataflow_buffer_spec.hpp` sharpens it into a concrete fork:

- **Option A — bind both `src1` and `src2` on every `KernelSpec`.** Placement is *derived* as the
  union of the bound kernels' `WorkUnitSpec::target_nodes`, so both DFBs land on **all of**
  `all_cores_bb`, whereas legacy allocated `c_1` only over `input1_cores` and `c_2` only over
  `input2_cores` (validated disjoint at `paged_fused_update_cache_device_operation.cpp:350-351`).
  Both are `borrowed_from` an L1-sharded input tensor which has **no shard** on the other core set,
  so `AttachBorrowedDFBBuffers`' per-bank sizing check has nothing correct to resolve there. Needs a
  ruling on what a borrowed DFB means on a node where its backing tensor has no shard.
- **Option B — split into per-core-set `KernelSpec`s** (`reader@input1_cores` binding `src1`,
  `reader@input2_cores` binding `src2`, plus a third over `unused_cores`). Placement then matches
  legacy exactly, and the DFBs get one PRODUCER binding each over non-overlapping node sets — legal
  by the `DataflowBufferSpec` INVARIANT note. **But** the kernel picks its input DFB from the
  *runtime* arg `is_input1`, so a split forces that host-computed per-core value onto a compile-time
  `#define`, changing the arg schema. That is a structural change the port is not entitled to make
  unilaterally.

The same fork appears a second time on the tensor channel: reader RTA[2] / writer RTA[1] carry
`cache_tensor1` on `cores1` and `cache_tensor2` on `cores2` (tiled `:438` vs `:483`; RM `:435` vs
`:481`), and a `tensor::name` binding is per-`KernelSpec` — so one reader spec would need two
`TensorParameter`s reaching one argument position, by node. **Resolve both together**; Option B
answers both at once, Option A answers neither.

Also waiting on that pass: the fused factories' *variable per-core runtime-arg count*
(`unused_cores` nodes get one arg and early-return, working cores get 8–9) has to be reconciled with
`runtime_arg_schema` being one schema per `KernelSpec` — Option B dissolves this too, which is worth
weighing in the decision.

### 2. Dead compile-time args, carried through unchanged

The audit catalogued these as team-only anomalies; the port carried each across as a named arg rather
than dropping it, because removing one changes the arg schema. Now that they are *named*, they are
also easy to find and drop in a follow-up:

- `log_base_2_of_page_size` — `reader_update_cache_interleaved_start_id_metal2.cpp:23`; host value is
  a local initialised to `0` and never assigned (`paged_update_cache_program_factory.cpp:582`).
- `log2_page_table_stick_size` — read and unused in
  `reader_update_cache_interleaved_start_id_metal2.cpp:31` and
  `writer_fill_cache_interleaved_metal2.cpp:46`.
- `max_blocks_per_seq` — read and unused in both `update_cache` `_metal2` dataflow kernels. It *is*
  load-bearing in `validate_on_program_cache_miss` as a bound, but no kernel range-checks
  `virtual_block_id` against it before indexing `page_table_ptr[virtual_block_id]` — a missing
  on-device bound check worth the ops team's attention independently.

### 3. Naming and structure the port deliberately left alone

- **`update_cache`'s writer calls the output DFB `cache`.** Kernel-side `cache_cb_id` was host-side
  `output_cb_index` (`c_16`); the fork keeps the kernel's word and binds the `OUTPUT` spec under
  accessor `cache`. Semantically defensible (it holds re-tilized cache data), but it is the kind of
  thing a reader trips over — worth renaming when the mesh sibling converts and the fork takes over
  the original's name.
- **Two unbalanced FIFO `wait_front`s in the row-major fused writer** (`:84`, `:95`, no matching
  `pop_front`, unlike both sibling writers) — untouched, and unreached by this pass since the RM
  fused factory is deferred. Flagged again here because it will land in whoever ports it.

### 4. Test-coverage note

Nothing in the confirmed test set exercises the **cache-miss dispatch on a mesh coordinate excluded
by `mesh_coords`** on the `fill_cache` path — which is precisely the behaviour handoff point #1's
`noop`-RTA idiom would silently change if someone ported that factory anyway.
`test_paged_fill_cache_mesh_coords` and `test_paged_fill_cache_batched_mesh_coords`
(`tests/ttnn/nightly/.../test_paged_update_cache.py:946`, `:1300`) assert the *result*, so they would
catch it — but only on a multi-device mesh, never on an N150. Worth stating explicitly on the ticket
for #1 so the fix is not validated on a single-device bench.

---

## Verification performed

**All static — the build could not run ([Friction](#friction) #1).** Denominators printed per the
recipe's note, so a check that scanned nothing is distinguishable from a check that found nothing.

| check | result |
|---|---|
| Diff scope — `git diff --name-only $BASE \| grep '^tt_metal/'` | **no output** (11 files changed, all under the op directory) |
| Forced-legality scaffolding — `git diff $BASE \| grep -E 'METAL2_CHECKS_FORCED\|DO NOT COMMIT'` | **no output** (applied to all 9 grep-named sites, then reverted with `git checkout --`) |
| `cb`-name sweep over the 5 `_metal2` kernels | **0 hits / 5 files** |
| `cb`-name + `CBDescriptor` + `CircularBuffer` + `TensorAccessorArgs` + `buffer()->address` + `emplace_runtime_args` sweep over both Metal 2.0 factory regions (`paged_update_cache_program_factory.cpp:556-1128`, `paged_fill_cache_program_factory.cpp:435-878`) | **0 hits / 2 regions** after rewording two comments that named the legacy API |
| Ephemeral-doc citation — `.md` references in changed + untracked `.cpp/.hpp` | **0 hits / 14 files** |
| `.id` extraction on a `dfb::` handle | **none** (the one `.id` hit is `SemaphoreDescriptor::id` in the retained legacy body) |
| `allow_instance_multi_binding` | **not set anywhere** — the census is 1P+1C or self-loop throughout |
| Varargs (`get_vararg` / `compile_time_varargs` / `num_runtime_varargs`) | **none** — every argument is named |
| `TT_FATAL` / `TT_ASSERT` / `TT_THROW` census vs. `$BASE` | **no file's count dropped.** `paged_fill_cache_program_factory.cpp` 3 → **6** (the same three guards now also in the Metal 2.0 body); the three device-op `.cpp` files unchanged at 45 / 25 / 40 |
| `opt_level` audit | `grep -n opt_level` over both ported factories returns **one** line — `KernelBuildOptLevel::O3` at `paged_update_cache_program_factory.cpp:938`, paired with the **one** compute `KernelSpec` the factory builds. `fill_cache` builds **zero** compute specs; both its DM specs correctly take Metal 2.0's `O2` default, matching the legacy DM default |
| `hw_config` value diff | reader `RISCV_1 / NOC_0 / DM_DEDICATED_NOC`, writer `RISCV_0 / NOC_1 / DM_DEDICATED_NOC` — traced from `ReaderConfigDescriptor{}` → `ReaderDataMovementConfig` (`kernel_types.cpp:19-22`, with `preferred_noc_for_dram_read` = `NOC_0` on every Gen1 arch) and confirmed identical to `CreateReaderGen1DataMovementConfig()` / `CreateWriterGen1DataMovementConfig()`, which the TTNN helpers wrap. Compute: only `enable_32_bit_dest` carried across; all four remaining `ComputeGen1Config` defaults verified equal to the legacy `ComputeConfigDescriptor` defaults |
| DFB endpoint census re-derived from the kernels (not transcribed from the brief) | agrees with the audit: `update_cache` 8 × 1P+1C; `fill_cache` 1 × 1P+1C + 3 self-loop. Self-loop shape (one accessor name, PRODUCER + CONSUMER on one kernel) confirmed legal against `program_spec.cpp:296-370` |
| Alias-group legality (`update_cache` `c_24`/`c_25`) | mutual `alias_with`, equal `entry_size * num_entries`, same derived node set, neither borrows — all four rules at `program_spec.cpp:1619-1699` satisfied |
| Borrowed-DFB legality (`update_cache` `input`) | named `TensorParameter` exists, its `TensorSpec` is L1-resident (the op requires a sharded input), and `entry_size * num_entries` ≤ the tensor's packed size — the three checks at `program_spec.cpp:1570-1615` |

### Handing the build and test run back

**Confirmed test set.** Located with a broad sweep
(`find tests -iname '*paged*' -o -iname '*update_cache*' -o -iname '*fill_cache*'`) and filtered.
There is **no C++ gtest coverage** for this op — pytest only.

| file | fixture | covers |
|---|---|---|
| `tests/ttnn/unit_tests/operations/transformers/test_paged_cache_flexible_geometry.py` | `device` | both **ported** factories (block-size / num-kv-heads overrides, negatives) |
| `tests/ttnn/nightly/unit_tests/operations/transformers/test_paged_update_cache.py` | `device` (18 tests) | both **ported** factories incl. the index-tensor path, the batched fill path, and the program-cache/cache-hit tests — the highest-value file for this port |
| `tests/ttnn/unit_tests/operations/transformers/test_paged_fused_update_cache.py` | `device` | the **deferred** fused factories — no-regression only |
| `tests/ttnn/unit_tests/operations/transformers/test_paged_cache_mask.py` | `mesh_device` | the **blocked** mesh path — no-regression only; meaningful coverage needs a multi-device mesh |
| `tests/sweep_framework/sweeps/model_traced/paged_{update,fill}_cache_model_traced.py` | sweep harness | optional breadth |

**Excluded as a false positive:** `tests/tt_eager/python_api_testing/unit_testing/misc/test_update_cache.py`
drives `ttnn.update_cache` / `ttnn.fill_cache` — the separate `kv_cache` op, not this one.

**Before trusting any green, force the Metal 2.0 legality checks** (recipe: *Ensure the Metal 2.0
host-side legality checks are enabled*). TTNN sets `skip_validation` behind the factory concepts and
has got it wrong before; a pass with the checks bypassed is the most expensive false green available
here, because it means every spec mistake in this port is still sitting in it:

```bash
grep -n 'bool skip_validation' tt_metal/impl/metal2_host_api/*.cpp
# make `skip_validation = false;` the first statement of EVERY function that grep names,
# and add one `log_warning(tt::LogMetal, "METAL2_CHECKS_FORCED");` per file (not in
# UpdateProgramRunArgs -- it fires on every cache hit and buries the log).
./build_metal.sh --build-tests
```

Run one test, then grep its log for `METAL2_CHECKS_FORCED`; **two** markers means both translation
units are fresh and the checks are live. **Revert the forcing and the markers before the PR** —
`tt_metal/impl/` is far outside a port's scope.

**Then, in order:**

```bash
export PYTHONPATH=$(pwd)
source python_env/bin/activate          # ./create_venv.sh first if there is no python_env

# 1. The two ported factories -- primary no-regression baseline.
pytest tests/ttnn/unit_tests/operations/transformers/test_paged_cache_flexible_geometry.py -x -v
pytest tests/ttnn/nightly/unit_tests/operations/transformers/test_paged_update_cache.py -x -v

# 2. The cache-hit path in isolation -- this is where the translated
#    override_runtime_arguments lives, and the failures it can produce are
#    second-dispatch-only, so they hide behind a first-call pass.
pytest tests/ttnn/nightly/unit_tests/operations/transformers/test_paged_update_cache.py -v \
  -k "program_cache or program_caching or attr_idxs"

# 3. Untouched-by-this-pass factories -- confirm the split broke nothing.
pytest tests/ttnn/unit_tests/operations/transformers/test_paged_fused_update_cache.py -x -v
pytest tests/ttnn/unit_tests/operations/transformers/test_paged_cache_mask.py -x -v
```

**What to expect if something is wrong**, in the order the failures would surface:

1. **Compile.** Host: a `Table` built like a vector, or a designated initializer out of field order.
   Kernel (JIT, at first dispatch): a `dfb::` / `tensor::` / `args::` name the host did not emit, or
   an `#ifdef` gate that does not match its binding condition.
2. **`TT_FATAL` from `program_spec.cpp` at the first dispatch** — a spec-validation reject. The two
   constructs most worth suspecting are `update_cache`'s alias group (`untilized_cache` /
   `untilized_cache2`) and its borrowed `input` DFB, and `fill_cache`'s three writer self-loops.
3. **Wrong numerics on the second and later calls only** — a missing `TensorArgument` in an override.
   The completeness table under [Concept realized](#concept-realized) is the checklist.
4. **A `TensorSpec` legality failure on the second call, never the first** — that is the custom-hash
   failure mode, and it is a stop-and-report, not a fix: do not touch the hash. See
   `ttnn_factory.md` → *The cache key: leave the custom hash alone*.
