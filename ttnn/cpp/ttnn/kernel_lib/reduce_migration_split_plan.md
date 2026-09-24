# Splitting the host-planned reduce migration (PR #56063) into mergeable PRs

Status: discussion draft. The source branch (`malimpic/migrate-to-host-reduce-helpers`) is
work in progress, so everything below describes the *shape* of each PR, not the exact code.
File and line counts are taken from the branch at `bc00a35f3be` (auxiliary tiles produced by
writer kernels) and will drift.

## 1. Baseline

**What the PR is.** #56063 moves every consumer of the kernel_lib reduce helpers to a
host-planned interface: the program factory plans the reduction (algorithm, chunking,
partial-tile handling, accumulation, auxiliary tiles) and serializes it into compile-time
args; the compute kernel issues `reduce<Call>()`; one dataflow kernel materializes the
planning unit's auxiliary-tile recipe. The old dataflow scaler helpers are removed from all
callers.

**Real size.** GitHub shows 625 files / +60k because the PR's base branch
(`llk_helper_library`) was rewritten after the branch was cut. The PR's own work is the last
66 commits. Against its true parent the diff is:

| | files | added | removed |
|---|---|---|---|
| Whole PR (own commits only) | 334 | ~14,350 | ~8,230 |
| of which library (`ttnn/cpp/ttnn/kernel_lib`) | 10 | ~3,100 | ~900 |
| of which library tests (kernel_lib pytest + gtest) | 8 | ~3,300 | ~300 |
| of which op migrations (`ttnn/cpp/ttnn/operations`, models, op tests) | ~286 | ~7,100 | ~5,700 |
| of which `llk_helper_library`-only content (examples, toy ops, Python ProgramSpec binding) | 25 | ~1,100 | ~1,300 |
| of which migration tooling (`scripts/*reduce_migration*`) | 5 | ~830 | 0 |

**"Compared to main" is larger than "compared to llk_helper_library".** `main` and
`llk_helper_library` already differ by ~1,200 lines in the reduce helpers. `llk_helper_library`
carries the `algorithm` parameter (ReduceTile / AccumulateViaAdd), within-tile collapse
control, accumulator reload modes, partial-scaler descriptors, and a `reduce_mean` wrapper;
`main` has none of that. The plan below targets `main`, so phase 1 must carry those
additions too, in the final form the PR settled on (not the intermediate
`llk_helper_library` form).

**What lives where today (high level).**

| Surface | main | llk_helper_library | PR head |
|---|---|---|---|
| Compute `reduce<...>` explicit template interface | pool, dim, 3 CBs, input policy, reconfig, fp32 mode | + algorithm (Auto default), within-tile, partial-scaler runtime arg, `reduce_mean` | + algorithm (ReduceTile default), within-tile, `reduce_factor`, partial *mode*, input chunk, auxiliary offset; `reduce_mean` and Auto removed |
| Compute planned interface `reduce<Call>()` | – | – | yes |
| Dataflow scaler helpers (`prepare_reduce_scaler`, `calculate_and_prepare_reduce_scaler`) | yes, called from ~41 readers and ~20 writers | yes + partial/mask variants | **removed**, replaced by `prepare_reduce_auxiliary_tiles<Auxiliary>()`, called from **writers** (63 sites) |
| Host planner (`kernel_lib/host/reduce_host.*`), wire format (`reduce_plan_args*.hpp`), Python `ttnn.reduce_planner` | – | – | yes |
| Consumers on the planned interface | 0 | 0 | 51 compute kernels, ~60 factories; 1 kernel (indexer_score) still explicit |

Consumers on `main` today: 56 compute kernels include the compute helper, 66 dataflow
kernels include the dataflow helper, spread over moreh, normalization, reduction,
experimental (quasar, ccl, kda, ssm, indexer_score, deepseek), transformer/sdpa, one model
custom op (bge_m3) and the udm gtests.

**Who produces the auxiliary tiles: the writer.** The latest branch state moves auxiliary
tile production (scalers, masks, zero tiles) from the reader to the writer kernel wherever
an op has both. Rationale: the reader is on the input-streaming critical path, the writer
is idle at kernel start, and about a third of the `main` scaler producers were writers
already (sdpa, groupnorm, sharded layernorm, moe, sampling). Two implementation patterns
appear:

- a small block guarded by a define (`REDUCE_AUXILIARY_CB`) in the op's existing writer, so a
  writer shared by several factories of the same op (for example moreh_mean H and W) only
  produces auxiliaries when the factory asks for it;
- a dedicated tiny "auxiliary writer" kernel where the op's writer is multi-purpose
  (softmax attention, layernorm_distributed).

Consequences for the split: every op migration now touches four kernels-side files instead
of three (compute, reader loses scaler production, writer gains auxiliary production,
factory moves the auxiliary args and the auxiliary CB producer binding to the writer), the
per-writer change is 4–18 lines, and two of the writers are **generic kernels shared with
unrelated ops** (see §7.3).

## 2. The four phases at a glance

```
P1  explicit compute interface, defaults only ──► P2  planner + planned interfaces (additive) + 1 op
                                                     │
                     ┌───────────────────────────────┴──────────────────────────────┐
                     ▼            ▼            ▼            ▼             ▼          ▼
P3  moreh (3 PRs)   normalization (3 PRs)   reduction   quasar   sdpa team   small experimental / models / udm
                     └───────────────────────────────┬──────────────────────────────┘
                                                     ▼
P4  delete old dataflow helpers, make explicit compute path internal-ish, doc cleanup
```

P1 and P2 are additive to `main` and gate everything else. P3 groups are independent of
each other and can be reviewed and merged in parallel (each depends only on P2). P4 waits
for the last P3 group.

## 3. Phase 1 — explicit compute interface on main, behavior-neutral

**Goal.** Land the *library* half of the compute-side interface change so that later
kernel migrations never need a helper change again. No kernel changes its behavior.

**Contents.**

- The explicit `reduce<...>` gets its final parameter set: `algorithm` (default ReduceTile),
  `within_tile` (default Collapse), `reduce_factor` (default 1), and the runtime arguments
  `partial_mode` (default None), `input_chunk` (default automatic), `auxiliary_tile_offset`
  (default 0). New parameters are defaulted and sit before the deduced functor parameters,
  so existing call sites compile unchanged and mean exactly what they meant before.
- The compute-side enums move to a small shared header (`reduce_types.hpp`) so P2 can
  share them with the host planner without touching the compute header again.
  The host-side `ReduceOpMath/ReduceOpDim` enums stay where they are until P2.
- The implementation for the new modes (AccumulateViaAdd datapath, within-tile skip,
  compile-time AVG factor, partial mask/scaler handling, reload modes) comes in with it.
  Behavior of the default path (ReduceTile, no partial, no additive accumulate) must be
  identical to `main`. See §7.1 for the internal changes that ride along and need a
  keep/defer decision.
- Dataflow: the two existing scaler helpers stay exactly as they are. To be able to test
  the new compute modes on device, P1 also adds the low-level "fill one auxiliary tile of a
  given pattern" primitive (first-row / first-column / per-face-row / zero) that P2's
  recipe executor is built on. This is additive and does not change any existing helper.
- Kernels: **no mandatory kernel edits**. All 107 `reduce<...>` call sites on `main` were
  checked against the new signature: none names the functor template parameters
  explicitly, none passes more than four positional arguments, and the eight that pass
  `fp32_mode` still bind to the same slot. If reviewers prefer call sites to spell the new
  defaults explicitly, that is ~56 compute kernels × 1–2 lines and can be a follow-up; it is
  not needed for P1 to be correct and it is not what this phase is for.
- Two extraction details: the host-side enums must not ride into `reduce_types.hpp` in P1
  (they would collide with the two `common.hpp` copies until P2 moves them), and the
  "no auxiliary CB" sentinel the explicit path uses for scaler-less reductions
  (AccumulateViaAdd, SFPU) needs a home outside the wire-format header, e.g. in
  `reduce_types.hpp`.

**Tests.** Keep `main`'s existing kernel_lib reduce pytest untouched and green (that is the
behavior-neutrality proof). Add a second test file that drives the new explicit
parameters directly (defines → template args), covering: AccumulateViaAdd vs ReduceTile
for SUM/AVG on row/col, partial edges (scaler and mask modes), within-tile skip on
pre-reduced inputs, cross-call accumulation with each reload mode, explicit input chunk.
The current branch already enumerates this matrix, but through the planner; P1 re-expresses
it without the planner. The auxiliary tiles for these tests are produced by a writer-config
test kernel, matching the convention the ops use.

**Estimate.**

| Area | files | added | removed |
|---|---|---|---|
| `reduce_helpers_compute.hpp` / `.inl` (minus the `reduce<Call>` adapter) | 2 | ~1,100 | ~130 |
| `reduce_types.hpp` (new, compute enums), `reduce_helpers_common.hpp`, `common_types.hpp` (NoOp/NoAccumulation, mirrors llk_helper_library) | 3 | ~110 | ~10 |
| `reduce_helpers_dataflow.hpp` / `.inl` (additive tile-fill primitive only) | 2 | ~140 | 0 |
| kernel_lib tests: new explicit-mode pytest + 2 test kernels | 3 | ~800 | 0 |
| **Total** | **~10** | **~2,150** | **~140** |

**Reviewers.** kernel_lib owners (`metalium-developers-kernel-lib`, `metalium-developers-mmfusedreduce`).

**Verification.** kernel_lib pytest (old + new); L2 nightly categories `kernel_lib`,
`moreh`, `fused`, `reduction`, `sdpa`, `experimental` (every consumer recompiles against
the new header, so all consumer categories should run once).

## 4. Phase 2 — planner, planned interfaces (additive), one migrated op

**Goal.** Make the new way of writing a reduction available on `main` without forcing
anyone onto it, and prove it end to end on one production op.

**Contents.**

- Host planner: `kernel_lib/host/reduce_host.hpp/.cpp` (plan one block; plan an
  accumulated call sequence; CB requirements; auxiliary recipe; serializers), added to
  `ttnn/sources.cmake`.
- Wire format shared by host and device: `reduce_plan_args_common.hpp` (packed words) and
  `reduce_plan_args.hpp` (constexpr device views: `ReduceCallArgs`, `ReduceAuxiliaryArgs`,
  the Metal 2.0 `Bound*` rebinding views, tail/runtime-shape support).
- `reduce_types.hpp` grows the host/device shared enums (accumulation mode, auxiliary tile
  type, path) and takes over `ReduceOpMath/ReduceOpDim` from the two `reduction/generic`
  `common.hpp` copies (reduction and quasar) so the planner can use them.
- Compute: **add** the `reduce<Call>(post_op)` overload that lowers a planned call into the
  P1 explicit call. The explicit overload stays public.
- Dataflow: **add** `prepare_reduce_auxiliary_tiles<Auxiliary>()`. The two old scaler
  helpers stay; the header documents the new one as preferred and states the convention
  that the writer produces the auxiliary CB.
- Shared generic writers: add the define-guarded auxiliary block to the two shared writer
  kernels the generic reduction op uses (`eltwise/unary/.../writer_unary_interleaved_start_id_metal2.cpp`,
  `data_movement/sharded/.../writer_unary_sharded_metal2.cpp`). Compiles to nothing for every
  other user. Landing this in P2 gets the one-time eltwise and data-movement review out of
  the way so G7 stays a single-team PR (see §7.3 for the alternative).
- Python: `ttnn.reduce_planner` nanobind module (needed by the planner pytest and by the
  bge_m3 custom op later) plus the two one-line Python re-exports.
- Tests: host-only planner gtests (in `tests/ttnn/unit_tests/gtests/test_reduction.cpp`),
  the planner-driven kernel_lib pytest (sequence kernel, writer-side auxiliary kernel,
  tail-stream and bound-tail kernels). Keep it in a separate file from P1's explicit test.
- One op migrated: **moreh_mean (H reduction)**. Recommended because it is the smallest
  real production path that exercises what the planner is for: a partial last tile, AVG
  normalization, an optional auxiliary CB, Metal 2.0 `dfb::` rebinding via
  `compile_time_varargs`, and a writer shared with the W factory (so it shows the
  define-guarded writer pattern). The compute kernel shrinks from ~100 lines of hand-written
  mask + accumulate to a loop around `reduce<Call>()`; the reader loses its scaler
  production; the writer gains the guarded auxiliary block; the factory drops three CBs
  (mask, accumulator, masked input) and its define plumbing. Net ~-155 lines across 4 op
  files, plus a small nightly test addition. Runner-up: moreh_sum (H), same shape, net ~-180,
  but without the optional-auxiliary path. Both are owned by the moreh reviewers.

**Delivery.** One logical phase, best delivered as two PRs so reviewers do not block each
other: **2a** library + tests + shared writers (kernel_lib owners, ttnn-core for
`sources.cmake` and nanobind, eltwise and data-movement for the two guarded writer blocks),
**2b** moreh_mean_h (moreh owners), stacked on 2a.

**Estimate.**

| Area | files | added | removed |
|---|---|---|---|
| host planner + `ttnn/sources.cmake` | 3 | ~1,600 | 0 |
| wire format headers | 2 | ~520 | 0 |
| `reduce_types.hpp` growth + enum move out of 2 `common.hpp` | 3 | ~50 | ~20 |
| compute `reduce<Call>` adapter (hpp + inl) | 2 | ~250 | 0 |
| dataflow recipe executor (additive) | 2 | ~80 | 0 |
| two shared generic writers (guarded block) | 2 | ~15 | 0 |
| Python planner binding + re-exports + cmake | 6 | ~450 | 0 |
| planner gtests | 1 | ~1,600 | 0 |
| planner kernel_lib pytest + 4 test kernels | 5 | ~2,100 | 0 |
| **2a total** | **~26** | **~6,650** | **~20** |
| 2b moreh_mean_h: compute, reader, writer, factory, nightly test | 5 | ~80 | ~240 |

The gtest and pytest files are half of 2a. If 2a is too large to review, the planner
gtests can be a separate PR that lands right after (host-only, no device).

## 5. Phase 3 — migrate the remaining consumers, in owner-aligned groups

Every group does the same four things per op: the factory plans the reduction and appends
the compute call list to the compute kernel's args and the auxiliary recipe to the
**writer's** args, moving the auxiliary CB producer binding to the writer; the compute kernel
replaces the explicit `reduce<...>` (and any hand-written mask / accumulate scaffolding) with
`reduce<Call>()`; the reader drops its scaler production; the writer gains the guarded
`prepare_reduce_auxiliary_tiles` block (or the op gets a dedicated auxiliary writer). Groups
are cut along CODEOWNERS boundaries on `main` so each PR has one reviewing team, and no
group depends on another.

| # | Group | Owner(s) on main | files | added | removed | Notes |
|---|---|---|---|---|---|---|
| G1 | moreh A: sum, dot, linear_backward, clip_grad_norm, norm | moreh owners (`razorback3`, `dongjin-na`, `ayerofieiev-tt`, `nmauriceTT`, `aczajkowskiTT`, mmfusedreduce) | 30 | ~670 | ~1,510 | 6 writers gain the guarded block; deletes two dead `moreh_norm` kernels (unreferenced on main) and their CMake globs; adds one nightly boundary test |
| G2 | moreh B: softmax fwd + bwd | moreh owners | 31 | ~970 | ~910 | 8 compute + 8 reader + 6 writer kernels + 8 factories; adds nightly boundary test |
| G3 | moreh C: layer_norm fwd/bwd, group_norm fwd/bwd | moreh owners | 35 | ~920 | ~890 | Introduces shared `moreh_reduce.hpp` host helper and two per-op kernel headers; 6 writers; `sources.cmake`/CMake glob edits |
| G4 | normalization: softmax (general + attention) | mmfusedreduce | 20 | ~620 | ~330 | Adds `softmax_reduce_plans.hpp` (host), `softmax_reduce.hpp` (kernel) and a dedicated `writer_reduce_auxiliary.cpp`; CMake hpp glob |
| G5 | normalization: layernorm, layernorm_distributed, rmsnorm_distributed | mmfusedreduce | 33 | ~520 | ~740 | Dedicated `writer_reduce_auxiliary.cpp` for layernorm_distributed, guarded blocks in the blocked writers; deletes two unreferenced rmsnorm_distributed kernels; updates fused tests incl. uneven-shard gamma handling (see §7.4) |
| G6 | normalization: groupnorm | mmfusedreduce | 11 | ~350 | ~225 | Writers already produced the scaler on main; adds `groupnorm_reduce_plans.hpp`; new DRAM test shapes |
| G7 | reduction: generic, moe, sampling + experimental deepseek_grouped_gate | mmfusedreduce | 27 | ~540 | ~310 | Adds `generic/device/common.cpp` planning helper; relies on the two shared generic writers from P2a; nightly + unit tests for H reduction |
| G8 | experimental/quasar: reduction + sdpa | mmfusedreduce | 25 | ~320 | ~270 | Mirrors G7 with quasar's own writer copies (no cross-team files); deletes one unreferenced sharded reader |
| G9 | sdpa team: transformer/sdpa, sdpa_decode, experimental indexer_score, kda | `metalium-developers-sdpa` | 27 | ~170 | ~70 | sdpa writers already owned the scaler; indexer_score/kda move it reader → writer; indexer_score compute stays on the explicit path |
| G10 | experimental/ccl: dit_fused_distributed_rmsnorm, rms_allgather | `metalium-developers-ops-data-movement`, `jonathansuTT` | 8 | ~160 | ~130 | Writers already owned the scaler |
| G11 | experimental without a specific owner: ssm hc_sum_reduce, transformer/fused_distributed_rmsnorm | falls to `metalium-developers-ops-leads` | 12 | ~130 | ~70 | |
| G12 | experimental/deepseek_prefill attn_res_gather_softmax | `metalium-developers-ds-prefill` | 4 | ~30 | ~15 | |
| G13 | experimental/transformer dit_layernorm_pre_all_gather | `metallium-maintainers-llama-models` | 1 | ~1 | ~7 | Removes a now-unused scaler fill |
| G14 | models: bge_m3 encoder_sdpa custom op + its test | `gtobarTT`, `cse-developer-ttnn` | 3 | ~70 | ~7 | First model-side consumer of `ttnn.reduce_planner` |
| G15 | udm reduction gtests | `yugaoTT`, `SeanNijjar` | 5 | ~20 | ~15 | Test kernels keep the auxiliary in the reader (no writer in that harness) |

**How many PRs.** 15 owner-aligned groups is the finest cut that keeps single-team review.
Recommended consolidation to **12 PRs**: G1–G10 as listed, G11+G12+G13 as one "small
experimental ops" PR (17 files; three reviewing teams but each reviews ~4 files), and
G14+G15 as one "models + udm tests" PR (8 files). If a team prefers one review, G4+G5+G6
(normalization, 64 files) or G7+G8 (reduction + quasar, 52 files) can also be merged
into single PRs, at the cost of review size.

**Order.** None required. Suggested: G7 first (the generic reduction op is the reference
implementation and the most-tested), then moreh and normalization in parallel, then the
small groups. Rebase each group right before merge: these kernels keep changing on `main`.

## 6. Phase 4 — remove the old interfaces

- Delete the two old dataflow scaler helpers and their constants from the dataflow
  header/impl (dataflow header collapses to the recipe executor only). At this point no
  reader includes the dataflow reduce helper any more.
- Drop the compatibility bits P1/P2 kept only for the transition (public tile-fill
  primitive if only tests use it; any include shims).
- The explicit compute `reduce<...>` **stays**: it is the implementation `reduce<Call>` lowers
  into, and at least one kernel legitimately uses it. Decide whether to keep it documented
  as the low-level API or move it under `detail::`.
- Update the kernel_lib docs (the reduce plan/API notes live on `llk_helper_library`).

Estimate: 3–5 files, ~-250 lines, ~+30 doc lines. Reviewers: kernel_lib owners.

## 7. Obstacles and decisions

### 7.1 Helper-internal changes bundled with the interface change

P1 lands a new implementation of the compute helper, not only new parameters. The default
path (ReduceTile, no partial, no additive accumulate, factor 1) was compared against `main`
step by step: init placement, scaler wait, register protocol, per-tile output publishing,
input policies, accumulator reload, reconfig, MAX/MIN handling and all existing asserts are
unchanged. The 3-argument vs 2-argument `compute_kernel_hw_startup` wording is
documentation only; the helper configures both operands itself in either case. Three items
differ and need a **keep / drop / defer** decision:

| | What | Who on main is affected | Recommendation |
|---|---|---|---|
| a | SFPU row-reduction packer edge-offset override (zeroes the right-hand faces of the output tile). This is a **temporary workaround** for a bug in the LLK's reduce pack-mask setup; the proper fix is being made in the LLK. | int32 and accurate-fp32 row reductions: generic `ttnn.sum/mean/max/min`, distributed layernorm / rmsnorm pre-allgather | **Drop before P1 merges.** P1's SFPU path then stays as on `main`. If the LLK fix has not landed by then, P1 tests that check padding of SFPU row outputs must not assert on it |
| b | Quasar only: packer re-initialised on every `reduce()` call regardless of reconfig mode (needed when a kernel chains reductions into different output CBs). Not device-validated on Quasar. | quasar reduction kernels | Not a pure fix (extra per-call cost). Either gate it behind the OUTPUT reconfig mode or defer to the quasar group (G8) |
| c | Quasar only: `reduce_uninit` receives the real input CB instead of CB 0. | quasar kernels whose input is not CB 0 | Keep; neutral on Wormhole/Blackhole, fix on Quasar |

Everything else in the implementation growth (runtime chunk size, partial-scaler index
plumbing, batch stride, new static asserts, the extra fields on `Accumulate` and the memory
layout struct) folds to identical code at the defaults. The SFPU post-scaling change on the
branch is planner-side and belongs to P2.

### 7.2 Target branch and the `llk_helper_library` fork

The PR is opened against `llk_helper_library`; this plan targets `main`. Once P1 lands on
`main`, the next `llk_helper_library` sync will break the branch-only consumers of the
intermediate interface (the `reduce_block` example, `toy_reduce_partial`, `toy_variance`,
their tests, the Python ProgramSpec binding tweak): 25 files. The PR already contains the
migrated versions of those files, so the fix is to port that slice to `llk_helper_library`
at sync time. Someone has to own that sync; it is outside the `main` PR series.

### 7.3 Auxiliary production in shared generic writers

Moving auxiliary production to the writer means the generic reduction op now needs two
writer kernels it does not own to produce auxiliaries: the eltwise unary interleaved writer
(also used by copy, typecast, bcast, permute, tilize_with_val_padding, transpose,
nlp_concat_heads, gelu_backward, kv_cache fill and prod) and the data-movement sharded writer
(also used by transpose, untilize, tilize_with_val_padding and quasar untilize). The branch
adds a define-guarded block plus an unconditional include of the reduce dataflow header to
each. Points to settle:

- **Where it lands.** Recommended: in P2a, as part of "open the interface", with eltwise and
  data-movement reviewing 7 lines each once. Alternative: give the generic reduction op its
  own writer copies (it already owns `writer_reduce_rm_scalar.cpp` and `writer_welford_hw.cpp`),
  which keeps every P3 group single-team but duplicates two small generic writers.
- **Legacy twins.** Both shared writers have a non-Metal-2.0 twin with a "keep in sync"
  note. Decide whether the twins get the same guarded block (they have no reduce users) or
  the note is amended.
- **Compile cost.** The unconditional include pulls the reduce dataflow header into every
  JIT build of those writers. It is header-only and cheap, but if it matters the include
  can sit inside the guard.
- **Contract visibility.** The define name is now part of the helper's contract but lives in
  kernels far from kernel_lib. The dataflow header should document it.

Quasar keeps its own copies of these writers, so G8 is unaffected.

### 7.4 Behavior and numerics visible to op tests

- G5 changes how a sharded distributed layernorm treats gamma for an uneven last shard
  (the test stops padding gamma to the combined shard width). Reviewers must confirm the
  op contract change is intended.
- New boundary tests (moreh norm/softmax/normalization, reduction H with short tails)
  encode the new tail handling. They are additive but reviewers should know they pin
  behavior the old kernels did not have.
- Tolerances were not loosened anywhere in the branch's test changes.
- Moving the scaler producer from the reader to the writer changes which RISC-V fills the
  auxiliary CB before compute waits on it. Compute waits on the CB either way, so this is a
  scheduling change, not a correctness one, but it is worth a sentence in each group's PR
  description because it shows up in profiler traces.

### 7.5 Things to keep out of the `main` series

- `scripts/*reduce_migration*` (~830 lines of one-off migration tooling).
- The falcon40b softmax model-test rewrite (unrelated config drift fix; separate PR).
- The Python ProgramSpec `compile_time_varargs` binding (Metal 2.0 Python binding exists
  only on `llk_helper_library`).

### 7.6 CI coverage on main

kernel_lib pytests run only in the L2 nightly `kernel_lib` category (scheduled, or via the
`/test` command which already maps `kernel_lib/**` and each op directory to a category).
Post-commit does not build or run them. P1 and P2 should therefore be validated with an
explicit nightly run over `kernel_lib,moreh,fused,reduction,sdpa,experimental` before
merge, and each P3 group with its own category. P2a's shared-writer edits additionally
warrant the `data_movement` and `eltwise` categories once. The planner gtests live in
`unit_tests_ttnn`, which runs in the T3000 unit-test job; confirm it also runs in the
post-commit C++ job before relying on it.

### 7.7 Cross-cutting edits in P2

Moving `ReduceOpMath/ReduceOpDim` into kernel_lib touches host code owned by the reduction
team (two `common.hpp` files), `ttnn/sources.cmake` / nanobind owned by ttnn-core, and (if
§7.3 lands there) two writers owned by eltwise and data-movement. Small, but P2a collects
up to five reviewing teams; the 2a/2b split keeps moreh out of it.

### 7.8 Deleted kernels

Five kernel files are deleted across G1, G5, G8. All were checked against `main`: none is
referenced by a factory. One (`rmsnorm_distributed/.../rmsnorm_post_allgather.cpp`) is named
in a cross-op compilation test's path table, which the PR updates in G5.

### 7.9 Merge-conflict exposure

56 compute, 66 dataflow-with-scaler and now ~40 additional writer kernels on `main` keep
evolving. P1/P2 are additive and low risk. P3 groups touch the kernels themselves; the
longer a group waits, the more it drifts. Sizing groups so each merges within a couple of
weeks of opening matters more than the exact cut.

## 8. Estimate summary

| PR | files | added | removed | reviewers |
|---|---|---|---|---|
| P1 explicit compute interface | ~10 | ~2,150 | ~140 | kernel_lib |
| P2a planner + planned interfaces + shared writers + tests | ~26 | ~6,650 | ~20 | kernel_lib, ttnn-core, reduction (enum move), eltwise + data-movement (guarded writer blocks) |
| P2b moreh_mean_h | 5 | ~80 | ~240 | moreh |
| G1 moreh A | 30 | ~670 | ~1,510 | moreh |
| G2 moreh B | 31 | ~970 | ~910 | moreh |
| G3 moreh C | 35 | ~920 | ~890 | moreh |
| G4 norm softmax | 20 | ~620 | ~330 | mmfusedreduce |
| G5 norm layernorm/rmsnorm | 33 | ~520 | ~740 | mmfusedreduce |
| G6 norm groupnorm | 11 | ~350 | ~225 | mmfusedreduce |
| G7 reduction (+ deepseek_grouped_gate) | 27 | ~540 | ~310 | mmfusedreduce |
| G8 quasar | 25 | ~320 | ~270 | mmfusedreduce |
| G9 sdpa team | 27 | ~170 | ~70 | sdpa |
| G10 experimental ccl | 8 | ~160 | ~130 | ops-data-movement |
| G11–G13 small experimental | 17 | ~160 | ~90 | ops-leads, ds-prefill, llama-models |
| G14–G15 models + udm | 8 | ~90 | ~20 | bge_m3 owners, udm owners |
| P4 cleanup | ~4 | ~30 | ~250 | kernel_lib |
| **Total** | **~317** | **~14,400** | **~6,150** | |

The total exceeds the PR's own diff because P1/P2 add tests the PR does not have (explicit
mode coverage) and keep both dataflow interfaces alive until P4.

## Appendix A — files per PR (snapshot of the current branch)

Counts are `added/removed` per file on the current branch. Kernels and factories will be
re-derived when each PR is cut. "Writer +N" means the guarded auxiliary block was added to
that writer.

### P1 (library only)
- `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp`, `.inl` (explicit interface + implementation; the branch's version minus the `reduce<Call>` adapter)
- `ttnn/cpp/ttnn/kernel_lib/reduce_types.hpp` (new; compute enums), `reduce_helpers_common.hpp` (enum moved out), `common_types.hpp` (new; NoOp / NoAccumulation)
- `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp`, `.inl` (additive: single auxiliary-tile fill primitive)
- `tests/ttnn/unit_tests/kernel_lib/reduce/`: existing `test_reduce_helpers.py` + kernels unchanged; new explicit-mode test + compute/writer test kernels

### P2a (library + tests)
- `ttnn/cpp/ttnn/kernel_lib/host/reduce_host.hpp` 350/0, `reduce_host.cpp` 1231/0; `ttnn/sources.cmake` 1/0
- `ttnn/cpp/ttnn/kernel_lib/reduce_plan_args_common.hpp` 147/0, `reduce_plan_args.hpp` 377/0
- `ttnn/cpp/ttnn/kernel_lib/reduce_types.hpp` (+ host enums), `ttnn/cpp/ttnn/operations/reduction/generic/device/common.hpp`, `.../experimental/quasar/reduction/generic/device/common.hpp` (enum relocation only)
- `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp`, `.inl` (`reduce<Call>` adapter), `reduce_helpers_dataflow.hpp`, `.inl` (`prepare_reduce_auxiliary_tiles`)
- `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id_metal2.cpp` 7/0, `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded_metal2.cpp` 7/0 (guarded blocks; see §7.3)
- `ttnn/cpp/ttnn/operations/reduction/reduce_planner_nanobind.cpp` 433/0, `.hpp` 15/0, `reduction_nanobind.cpp` 2/0, `reduction/sources.cmake` 1/0, `ttnn/ttnn/__init__.py` 1/0, `ttnn/ttnn/operations/reduction.py` 1/0
- `tests/ttnn/unit_tests/gtests/test_reduction.cpp` 1614/0
- `tests/ttnn/unit_tests/kernel_lib/reduce/`: planner pytest (~1,900), `kernels/reduce_plan_sequence.cpp` 73, `reduce_plan_sequence_aux.cpp` 15 (writer config), `reduce_bound_tail.cpp` 23, `reduce_tail_stream_reader.cpp` 38

### P2b (moreh_mean_h)
- `ttnn/cpp/ttnn/operations/moreh/moreh_mean/device/kernels/moreh_mean_h.cpp` 9/87, `reader_moreh_mean_h.cpp` 0/16, `writer_moreh_mean_unary_interleaved_start_id.cpp` 8/0 (shared with the W factory; guarded), `moreh_mean_h_program_factory.cpp` 54/135
- `tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_mean.py` 11/0

### G1 moreh A (30 files)
- moreh_sum: compute 11/85, reader 0/12, writer +5, factory 30/133
- moreh_dot: compute 18/27, reader 0/63, writer +5, factory 46/8
- moreh_linear_backward: multi-core H compute 18/98, single-core HW compute 22/17, 2 readers 0/24, writer +5, 2 factories 76/116
- moreh_clip_grad_norm step1: compute 69/71, reader 0/13, writer +5/1, factory 72/40
- moreh_norm (ord_other H and W): 2 compute 104/265, 2 readers 0/34, 2 writers +10, 2 factories 111/128; delete `moreh_norm_h/kernels/moreh_norm_h_kernel.cpp` and `moreh_norm_w/kernels/moreh_norm_w_kernel.cpp` (172 each, unreferenced); `moreh/CMakeLists.txt` glob removal
- `tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_norm.py` 23/0

### G2 moreh B (31 files)
- moreh_softmax: 4 compute kernels 168/308, 4 readers ~10/70, 4 writers +40, 4 factories ~275/95
- moreh_softmax_backward: 4 compute kernels 114/184, 4 readers ~0/45, 2 writers +8, 4 factories ~265/180
- `tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_softmax_reduce_boundaries.py` 68/0

### G3 moreh C (35 files)
- `ttnn/cpp/ttnn/operations/moreh/moreh_reduce.hpp` 57/0 (shared host helper), `moreh/sources.cmake` 3/0, `moreh/CMakeLists.txt` hpp globs
- moreh_layer_norm: 2 compute kernels 14/406, new `moreh_layer_norm_reduce.hpp` 138/0, 2 readers ~0/16, writer +5/1, factory ~75/50
- moreh_layer_norm_backward: 3 compute kernels ~190/310, new `moreh_norm_backward_reduce.hpp` 27/0, 3 readers ~0/30, 2 writers +10, 2 factories ~65/30
- moreh_group_norm: 2 readers ~0/16, writer +5/1, factory ~55/20
- moreh_group_norm_backward: 3 readers ~0/28, 2 writers +10, 2 factories ~90/45
- tests: `test_moreh_group_norm.py` 9/0, `test_moreh_layer_norm.py` 7/0, `test_moreh_normalization_reduce_boundaries.py` 147/0

### G4 normalization softmax (20 files)
- attention compute: `softmax.cpp` 8/23, `softmax_large_tensor.cpp` 54/50, `softmax_sharded.cpp` 8/22, new `softmax_reduce.hpp` 26/0
- attention dataflow: 5 readers ~15/110, `writer_unary_interleaved_start_id_blocked_sm.cpp` +10, new `writer_reduce_auxiliary.cpp` 18/0
- factories: attention optimized + sharded ~60/10, general h/w small/large ~260/75, new `softmax_reduce_plans.hpp` 94/0
- `normalization/CMakeLists.txt` 1/0 (hpp glob), `normalization/sources.cmake` (softmax line)

### G5 normalization layernorm / rmsnorm (33 files)
- layernorm: 2 compute 46/117, 6 dataflow ~40/100, 2 blocked writers +14, `layernorm_op_multi_core.cpp` ~25/5, `layernorm_op_multi_core_sharded.cpp` 59/4, `sharded_layernorm_factory_helpers.cpp/.hpp` 56/17
- layernorm_distributed: 2 compute 14/35, 3 readers ~5/25, blocked writer +7, new `writer_reduce_auxiliary.cpp` 12/0, 4 factories ~130/80
- rmsnorm_distributed: 2 compute 10/17; delete `rmsnorm_post_allgather.cpp` (200) and `rmsnorm_pre_allgather_2d.cpp` (132)
- tests: nightly `fused/test_distributed_layernorm_post_allgather.py` 62/9, unit `fused/test_distributed_layernorm_sharded.py` 5/6, `fused/sharded_test_utils.py` 11/2, `fused/parallel_sequential/test_parallel_sequential.py` 2/2

### G6 normalization groupnorm (11 files)
- compute `groupnorm.cpp` 48/45, `groupnorm_sharded_v2.cpp` 21/71; 2 writers 12/74 (already scaler producers on main)
- 3 factories 113/32, `groupnorm_program_utils.hpp` 2/2, new `groupnorm_reduce_plans.hpp` 152/0, `normalization/sources.cmake` (groupnorm line)
- `tests/ttnn/unit_tests/operations/fused/test_group_norm_DRAM.py` 2/1

### G7 reduction (27 files)
- generic: `common.hpp` 19/10 (planning helper part), new `common.cpp` 66/0, 3 compute 38/82, 5 readers ~10/70, `writer_reduce_rm_scalar.cpp` +10/1, `writer_welford_hw.cpp` +5, 4 factories ~140/40 (they select the shared eltwise / data-movement writers from P2a and set the define)
- moe: compute 13/21, writer 3/3, factory 34/4
- sampling: compute 12/21, writer 8/4, factory 34/8
- experimental/reduction/deepseek_grouped_gate: compute 3/8, writer 3/4, factory 18/2
- tests: nightly `reduction/test_reduce.py` 28/0, unit `reduce/test_reduce_migration_height.py` 84/0

### G8 quasar (25 files)
- reduction/generic: `common.hpp` 16/10, new `common.cpp` 66/0, 3 compute 48/97, 6 readers ~20/70 (+ delete one unreferenced sharded reader, 76), 4 writers +17 (quasar-owned copies), 4 factories ~140/35
- transformer: sdpa factories 15/2, sdpa_decode factory 6/1, 3 writers 12/15

### G9 sdpa team (27 files)
- transformer/sdpa: 7 factories 28/0, 6 writers 15/28; sdpa_decode: factory 4/0, writer 2/5
- experimental/indexer_score: compute 4/12 (stays explicit), reader ~0/9, writer +7, 2 factories ~45/5
- experimental/kda: prepare_chunk_recurrence compute 5/6, writer +6, factory ~28/0; sigmoid_gated_rms_norm compute 4/3, reader ~0/9, writer +5, factory ~25/1

### G10 experimental ccl (8 files)
- dit_fused_distributed_rmsnorm: factory 54/30, compute 14/32, 2 writers 8/6, `dit_rmsnorm_scalar_setup.hpp` 4/10
- rms_allgather: compute 19/31, writer 13/14, factory 50/5

### G11–G13 small experimental (17 files)
- ssm/hc_sum_reduce: factory ~22/1, reader ~0/7, writer +4, compute 3/9
- transformer/fused_distributed_rmsnorm: 2 factories ~75/10, 2 compute 28/36, 2 readers ~0/13, 2 writers +8
- deepseek_prefill/attn_res_gather_softmax: factory ~20/5, compute 3/7, reader ~0/5, writer +7
- transformer/dit_layernorm_pre_all_gather: reader 1/7

### G14–G15 models + udm (8 files)
- `models/demos/wormhole/bge_m3/tt/custom_ops/encoder_sdpa/kernels/writer.cpp` 7/7, `op.py` 5/0, `tests/ttnn/unit_tests/operations/sdpa/test_bge_encoder_sdpa_reduce_migration.py` 58/0
- `tests/ttnn/unit_tests/gtests/udm/reduction/`: 3 kernels 9/15 (auxiliary stays reader-side here), 2 tests 9/0

### Not carried to main
- `scripts/generate_reduce_migration_sanity_reports.py`, `reduce_migration_gtest_adapter.py`, `reduce_migration_pytest_plugin.py`, `run_reduce_migration_sanity.py`, `run_reduce_migration_tests.py`
- `ttnn/ttnn/operations/examples/*` (4 examples, 9 files), `toy_reduce_partial/*`, `toy_variance/*` (incl. their writers), their tests, `ttnn/cpp/ttnn-nanobind/program_specs.cpp` — port to `llk_helper_library` at its next sync instead
- `models/demos/t3000/falcon40b/tests/unit_tests/test_falcon_softmax.py` — unrelated fix, separate PR
