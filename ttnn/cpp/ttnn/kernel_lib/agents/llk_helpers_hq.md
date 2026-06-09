# Kernel Helper Library HQ

Entry point for creating and maintaining `compute_kernel_lib` helpers — unified APIs that hide LLK init/compute/pack complexity in tt-metal compute kernels.

## Quick Start

| What you need | Document |
|---|---|
| **Writing helper code** (file structure, op structs, policies, CRTP bases, perf testing) | [llk_helpers_conventions.md](llk_helpers_conventions.md) |
| **Running the agent pipeline** (catalog, investigate, propose, validate, implement) | [llk_helpers_pipeline.md](llk_helpers_pipeline.md) |

## When to Use What

| Situation | Action |
|---|---|
| Adding ops to an existing helper | Read [conventions](llk_helpers_conventions.md) section 5 (op structs). Add CRTP struct + init/call. |
| Creating a new helper (known ops, known LLK calls) | Read [conventions](llk_helpers_conventions.md), write .hpp/.inl directly, add perf tests. |
| Creating a new helper (unknown territory) | Run the [pipeline](llk_helpers_pipeline.md) from Phase 0. |
| Updating/improving an existing helper | Run [pipeline](llk_helpers_pipeline.md) from Phase 4 (validation only). |

## Agent Files

| Agent | Pipeline Phase | Purpose |
|-------|---------------|---------|
| `llk_catalog_agent.md` | 0 | Enumerate ops, group, locate source files |
| `llk_investigation_agent.md` | 1 | Device + host + usage analysis (per group) |
| `llk_verification_agent.md` | 2 | Confirm/deny investigation claims |
| `llk_helper_proposal_agent.md` | 3 | Design helper API + op structs |
| `llk_validation_agent.md` | 4 | Raw LLK -> params -> integration -> perf |
| `llk_review_fix_agent.md` | 4 | Document review and fix loop (within validation) |
| `llk_device_validation_agent.md` | 4 | Device-side test generation (within validation) |

## Helpers Location

```
ttnn/cpp/ttnn/kernel_lib/
  {name}_helpers.hpp      <- declarations, enums, structs, examples
  {name}_helpers.inl      <- implementation
  agents/                 <- this directory (pipeline docs + agent prompts)
```

## Approval Gates

Two BLOCKING gates between phases of any pipeline run. Both gates require an
**explicit go-ahead from the human**. These gates exist because a previous
helper-creation run conflated a scope answer ("yes, full surface migration")
with API approval and shipped a 24-file design + a 12-test pytest with no
review. Don't repeat that.

### Gate 1 — API proposal requires explicit user approval

> BLOCKING REQUIREMENT — do NOT write any `.inl`, `.cpp`, kernel source, or
> test file until the user has explicitly approved the API proposal.

After Phase 3 (Proposal), the agent's turn ENDS. The agent must:

1. Write the proposal artifact at the standard path (`{category}_helper_proposal.md`).
2. Output one line: `Proposal at <path>. Awaiting sign-off.`
3. Stop. Do not start implementation, validation kernels, or tests.

Implementation phases (4 onwards) DO NOT START until the user replies with
explicit approval on the proposal.

### Gate 2 — Test plan requires explicit user approval

> BLOCKING REQUIREMENT — do NOT add, remove, skip, retitle, change tolerance,
> change parameterization, change dtype matrix, or mark XFAIL↔PASS on any
> test until the user has explicitly approved the test plan.

The validation phase posts a test plan (kernel-by-kernel: shape, num_tiles,
dtype matrix, PCC threshold, skip rationale) as a SEPARATE artifact and
ends its turn. Implementation of test kernels and pytest only begins after
explicit user approval on that plan.

What counts as a test change for this gate: adding a test, removing a test,
skipping a test, changing the dtype matrix, changing the tolerance, changing
the parameterization (num_tiles, shapes, etc.), marking XFAIL→PASS or
PASS→XFAIL.

### What counts as "explicit approval"

- An explicit message containing "approved", "sign off", "ship it", "looks
  good", "go ahead", or an explicit list of accepted/rejected sections.
- An empty user message, a `<system-reminder>`, or a tool-result echo does
  NOT count.
- **Answers to clarifying questions about scope, naming, or coverage do NOT
  count as design or test approval.** Scope sign-off and design sign-off are
  different things. If the human says "yes, full surface migration", that
  approves *what* gets migrated — not *how* the API looks or *which* tests
  cover it. The agent must still post a written proposal and a written test
  plan, and wait for explicit approval on each, before any implementation.

When in doubt, ASK. The cost of one extra round-trip is small. The cost of
landing an unapproved API or unapproved test list is much larger — the
artifact is harder to redline once it exists than to design on paper.

Compression modes (caveman, ultra, terse) do NOT override these gates.

### Commit the artifact on acceptance

The moment a proposal artifact (Gate 1: `{category}_helper_proposal.md`) or a test plan artifact (Gate 2: `{category}_test_plan.md`) clears its gate, **commit the approved file before starting the next phase**. The commit message names the gate and the user's approval message.

Reason: the artifact must be git-permanent at the version that was approved. If implementation reveals a needed delta, the new version is a *new* commit that re-enters the gate — not an in-place edit of the approved file. In-place edits dissolve the audit trail of "what was actually signed off."

Concretely:

1. After Gate 1 sign-off, commit `{category}_helper_proposal.md` verbatim, then begin Phase 3.5 (Test Plan).
2. After Gate 2 sign-off, commit `{category}_test_plan.md` verbatim, then begin Phase 4 (Validation).
3. If the user lists deltas instead of a clean approval, revise the artifact, re-post for second sign-off, and commit only after that explicit approval — never before.

Compression modes do NOT skip the commit step.

## Helper Design Principles (general)

Rules below apply across every helper family, not just eltwise. Per-helper specifics (policy enum values, op-struct catalog, batching pitfalls) live in the helper's own header doc-comment.

### Helper owns the CB lifecycle it can see

If the helper takes a CB id as a parameter, the helper itself does the `cb_wait_front` / `cb_pop_front` on that CB. Don't push wait/pop back to the caller "because the policy could be different" — the caller already told the helper what CB it is by passing it. The helper's policy enum picks *which* shape of wait/pop, not *whether* the helper does them. Caller-side wait/pop only survives where the helper genuinely cannot see the CB (sharded tensors, persistent prologues, fan-out across helper boundaries) — and that path is named explicitly (`NoWait*` / `*NoPop`), not the default.

A helper that asks the caller to do `cb_wait_front(cb_in0, 1)` *before* invoking it is leaking lifecycle state across the helper boundary, and any chain / composition trait checks the helper performs cannot reason about waits the helper never sees. Swallow the wait/pop or document the gap explicitly as a known-unsupported lifecycle in the helper's header.

### DEST capacity is compile-time, never literal

Half-sync fp16 has 8 DEST slots; full-sync has 16; fp32-DEST mode halves both. Use `DEST_AUTO_LIMIT` from `ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp` (constexpr, derived from `get_dest_limit()` / `DST_ACCUM_MODE`) anywhere a helper bounds DEST slot indices, batch sizes, or chain widths. A literal `8` in an enum or `static_assert` ships a helper that silently miscompiles the moment the kernel runs in a different DEST mode. `DEST_AUTO_LIMIT` is already used across `binary_op_helpers.inl`, `sfpu_helpers.hpp`, `untilize_helpers.inl` — match that convention.

### Helper API takes `uint32_t` cb id; `CircularBuffer` supported internally

Public helper signatures take **only** the raw `uint32_t` cb id — never an `CircularBuffer` reference, never a templated CB-type wrapper. The single `uint32_t` parameter is the helper's contract with the caller.

Internally, the helper is free to use `CircularBuffer` machinery — e.g. constructing an `CircularBuffer` from the id to query metadata, drive iterator-based access, or invoke features only the experimental type exposes. That is an implementation detail, not part of the API.

Reason: kernels mid-transition to `CircularBuffer` and legacy kernels on the bare uint id share one API surface. The id is what every kernel already has; constructing the experimental wrapper inside the helper is cheap. Lifting the type into the signature forces every caller to construct one (or maintain two helper variants), and the surface drifts as some helpers take the wrapper and others don't.

Where the helper queries CB metadata (page size, num pages, dtype), prefer compile-time `static_assert` paths when the metadata is reachable through the `CircularBuffer` constructed from the id, falling back to runtime reads when it is not. `if constexpr` selects between paths.

### Reconfig is a first-class helper capability

Every helper that bridges init / dtype / format state must expose reconfiguring as an in-helper option (policy enum, not caller-side fixup). Use `with_dt_tree`-style decomposition to map each reconfig variant to its underlying LLK calls — the mapping is the canonical reference for migration and pattern matching.

## Validation Tests

Every helper change (new helper OR update) MUST land alongside at least one
pytest that runs a custom compute kernel on device and validates against a
torch golden. Host-side `./build_metal.sh` is NOT sufficient — kernels JIT
at runtime, so compilation success is not correctness.

Existing suites — run these AND add to them for any change that touches the
covered surface:

| Suite | What it covers | Location |
|---|---|---|
| `test_helpers_chain_and_binary.py` | `sfpu_chain` Load lifecycle (fan-out, `LoadPolicy`, `NoWaitPop`); `binary_op` same-CB dedup; `DestReuseOp` as chain element; Load inside a PostOp chain (FPU-clash reinit) | `tests/ttnn/unit_tests/kernel_lib/` + kernels in `ttnn/cpp/ttnn/kernel_lib/tests/chain_and_binary/` |

When a new surface is added (new enum value, new policy, new op struct, new
reconfig path) the update MUST:

1. Add or extend a test kernel that exercises the exact new path.
2. Parameterize over at least `num_tiles ∈ {1, 8, 64}` — single tile, fits in
   DEST, and spans multiple DEST windows. These three cover most off-by-one
   and batching regressions.
3. Include a torch golden using `comp_pcc(...)` (>= 0.9999 for bf16-only
   paths; >= 0.999 when fp32 mixed dtypes are involved).
4. Skip Blackhole unless the feature is explicitly Blackhole-tested.

Run via:
```
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/kernel_lib/*.py
```

Do NOT prefix with the shell `timeout` command — `run_safe_pytest.sh` and `tt-probe.sh` have built-in dispatch-timeout detection that runs `tt-triage` and emits the JSON triage report. An outer `timeout` kills the wrapper before triage runs and the hang artifact is lost.

Phase 2 (Validate) of the pipeline is responsible for adding these tests —
see [llk_validation_agent.md](llk_validation_agent.md) for the concrete
sub-stage 2c / 2d steps.

## Kernel Migration Steps

Migration is the FINAL step — it consumes helpers that already exist and are validated. If a missing op struct, missing enum value, or missing API surface is discovered mid-migration, stop and close that gap first (Helper Update / Helper Creation), then resume.

The steps below are helper-agnostic. Helper-specific guidance (policy enums, batching rules, op-struct catalog, fusion patterns) lives in the per-helper `.hpp` doc-comments and the helper's section of `llk_helpers_conventions.md` — read those for the helper you are migrating to before Step 1.

### Step 0 — Slim context before migration

Before auditing the first kernel of a migration cycle, drop stale context that would mislead pattern matching:

- Archive prior `agent_logs/` entries from helper-creation runs that preceded this migration cycle.
- Strip closed gap-map entries; keep only open `GAP-N` rows that the migration may hit.
- Drop superseded proposal artifacts and old investigation outputs that no longer reflect the helper's current API.

Reason: bloated migration context surfaces dead op names, rejected enums, and pre-redesign claims as phantom requirements during audit. Pipeline-wide cycles (catalog → implementation) do NOT need this reset between phases — only the migration phase does.

### Step 1 — Find tests that exercise the migration path

Before reading a single LLK call in the kernel under migration:

1. Look up the kernel's pytest in the [Pytest Manifest](#pytest-manifest).
2. If the manifest row is missing, run the discovery procedure ([Pytest Manifest → Discovery procedure](#pytest-manifest)) and append the row before continuing.
3. Verify the test JIT-compiles THIS exact kernel file (per [Verifying the Test Exercises the Changed Kernel](#verifying-the-test-exercises-the-changed-kernel)). Same-named files in other directories silently shadow the one being migrated; finding out post-migration that the test never touched the kernel is the worst failure mode.
4. Run the test once UNCHANGED to establish a green baseline. A test that was already failing pre-migration is not a regression detector.

Only after a green-baseline test exists does the migration proceed to audit (Step 2). Migration without a known-exercising test is unsupported — the result cannot be verified.

### Step 2 — Audit the target kernel

For the kernel being migrated, enumerate:

- **Raw LLK calls it makes** (`*_tile_init`, `*_tile`, `reconfig_data_format*`, `cb_*`, `tile_regs_*`, `pack_tile*`, `matmul_*`, `reduce_*`, etc.). For each, decide: covered by an existing helper API, requires a helper update, or out of scope.
- **CB lifecycle per operand** — classify each input/output CB as one of:
  - *Per-tile*: `cb_wait_front(N, 1) / ... / cb_pop_front(N, 1)` inside the loop body.
  - *Upfront*: `cb_wait_front(N, count)` once, processed in a loop, `cb_pop_front(N, count)` once at the end.
  - *Persistent*: waited once before the block, never popped inside (caller-managed).
  - *Streaming, pre-pushed*: tiles already present when the block starts (no wait in compute), popped per tile.
  - *Cumulative*: `cb_wait_front(N, base + i)` growing each iteration — usually unsupported by helpers.
  Map each lifecycle to the policy enum the target helper exposes (consult the helper header).
- **Dtype assumptions**: any `*_with_dt` calls, explicit `reconfig_data_format_srca/_srcb`, or per-iteration format swaps indicate FP32_DEST_ACC or mixed-dtype paths. If the helper only does one entry-time reconfig and the kernel switches dtypes mid-loop, flag this as a correctness risk; the migration MUST be tested with `fp32_dest_acc_en=True` and any mixed-dtype combos the original kernel supported.
- **Control-flow shape of the replaceable block**:
  - `if constexpr (compile_time_flag)` → trivially migratable; one helper invocation per arm.
  - `if (runtime_bool)` → emit one helper call per branch; do NOT try to lift the runtime condition inside a compile-time helper composition.
  - Interleaved op classes in one DEST window (e.g. FPU+SFPU, matmul+eltwise) → only migratable if the target helper exposes a fusion / post-op extension point. Otherwise split into separate helper calls with an intermediate CB.

### Step 3 — Gate-check against the helper API

If the audit turns up ANY of these, return to a prior step before writing the migration:

- A needed op struct, API, or fusion point does not exist → Helper Update (conventions §5) or Helper Creation, then resume.
- A required policy / enum value is missing (e.g. a CB lifecycle the helper does not yet model, a dtype-reconfig mode it does not emit) → Helper Update via pipeline, then resume.
- The control-flow or LLK pattern is fundamentally unsupported (cumulative waits, deeply interleaved op classes with no fusion point, exotic pack patterns) → leave that block on raw LLK and move on.

### Step 4 — Write the migration

Keep the scope surgical:

- **Only replace the LLK-call block.** Do NOT refactor surrounding code, rename CBs, reorder operations, or touch unrelated paths in the same kernel. Other helper calls already in the file stay as-is.
- **Always fully qualify helper symbols** (`compute_kernel_lib::<Symbol>`). Do NOT introduce `using namespace compute_kernel_lib;` or namespace aliases (`namespace ckl = compute_kernel_lib;`). Namespace pollution — even block-scoped — leaks helper names into later additions to that scope and hides the dependency on the helper library at the call site. The verbosity of full qualification is the point: it is searchable, auditable, and prevents unrelated identifiers from silently binding to helper symbols.
- **Delete dead locals.** DEST-slot indices, scratch counters, and lifecycle bookkeeping that the helper now owns become unreachable. Remove them; do not keep them "for safety".
- **Preserve surrounding CB lifecycle the helper does not own.** If the helper's policy declares it does not wait/pop a given CB (caller-managed / persistent), the kernel's existing `cb_wait_front` / `cb_pop_front` at the surrounding scope MUST stay.
- **Trim includes only when fully unused.** Swap `api/compute/*.h` includes for `ttnn/cpp/ttnn/kernel_lib/*_helpers.hpp` only after every raw LLK call from that header is gone. Headers still needed by unmigrated paths in the same file stay.

### Step 5 — Verify on device

Build is necessary but **not** sufficient — kernels JIT at runtime, so a clean `./build_metal.sh` says nothing about correctness. Every migration MUST pass on real device.

- Build the host library: `./build_metal.sh`.
- Run the kernel_lib validation suite (covers the helper itself):
  `scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/kernel_lib/*.py`.
- Run the operation's pytest(s). Resolve the path via the [Pytest Manifest](#pytest-manifest) instead of re-searching the tree each session.
- Confirm the test exercises THIS kernel file — see [Verifying the Test Exercises the Changed Kernel](#verifying-the-test-exercises-the-changed-kernel). Same-named files in other directories silently shadow the one you changed.
- Cover the dtype matrix the kernel supported pre-migration. Read the **program-factory dispatch table** for the original kernel and mirror that exact list as pytest params — do NOT just run the dtype the migration author happened to test on. Common silent omissions: `float32` input, `int32` input, `bfloat8_b` packed format, mixed `bfloat16 × float32`. At minimum run `fp32_dest_acc_en` ∈ {False, True} when the op tests FP32 accumulation, and any mixed-dtype combos the original kernel handled. If a dtype is intentionally dropped, document why in the test file. "It worked on bf16" is not coverage.
- During a multi-kernel migration cycle, run ONE representative pytest per kernel (the row from the Pytest Manifest). Defer the full kernel_lib suite + cross-helper regression set to the end of the cycle. Per-kernel full-suite runs multiply wall time without adding coverage.
- If any test fails: diagnose whether it's a helper gap (→ Step 3) or a migration mistake (→ Step 4). Do NOT relax the test or skip dtype combos to make it pass.

#### Migrations untestable on the local agent

Some migrations cannot be validated on the local agent's hardware — multi-chip ops, arch-gated paths (e.g. Blackhole-only LLK), kernels behind unavailable models. These migrations still land when structurally correct, with both:

1. A `untestable_locally: <reason>` annotation in the commit message.
2. A named CI job (or specific test id) that *will* exercise the kernel after merge.

Do not skip the migration "because we can't verify" and do not silently mark it complete with no CI handoff. The annotation + CI pointer is the contract; the reviewer knows the local run did not exercise this kernel.

### Pytest Manifest

The migration pipeline maintains a per-operation pytest manifest — a plain
text file listing, for each migrated kernel, the pytest(s) that exercise it.
The manifest is the single source of truth for test discovery; agents read
from it and append to it, never re-grep the repo ad-hoc.

**Where it lives**: `ttnn/cpp/ttnn/kernel_lib/pytest_map.md` (one file,
repo-wide). Seed rows are created the first time a kernel is touched by a
pipeline run; later runs append rows as new kernels are encountered.

**Row format** (kept trivially greppable — one kernel per line):
```
<repo-relative kernel path> :: <repo-relative pytest path>[; <additional pytest path> ...]
```

**When each step writes to it**:

| Pipeline step | Write responsibility |
|---|---|
| Step 1 (find tests) | If the target kernel is missing from the manifest, run the discovery procedure (grep program factories → test imports), verify the test JIT-compiles the kernel, then append a row before continuing. |
| Step 5 (verify) | Run every pytest listed for the touched kernel(s). If the manifest row was stale (path moved, test renamed), fix the row before reporting the migration done. |
| Step 6 (record) | If the migration adds coverage on a *new* kernel that the test file didn't previously exercise, note that in the `*_analysis.md`; the manifest row needs no change (same pytest still covers it). |

**Infrastructure regression set**: a separate section at the top of the
manifest lists pytests that must run after any change to the shared helpers
(`binary_op_helpers`, `sfpu_chain`, `reduce_helpers_compute`). Agents
*read* this set, they don't re-derive it — new rows get appended when a
pipeline run proves a particular pytest exercises a newly-added surface.

**Discovery procedure** (used once per kernel, then the result lives in the
manifest):

1. `grep -rn "<kernel relative path>" ttnn/cpp --include="*.cpp"` → finds the
   program factory that compiles it.
2. Trace the factory back to the `ttnn.*` op it registers.
3. `grep -rn "ttnn\.<op_name>" tests/ --include="*.py"` → candidate tests.
4. Run the shortest candidate with `scripts/run_safe_pytest.sh` and
   confirm it actually JIT-compiles the target kernel (optional paranoid
   check: temporarily plant `static_assert(false, "sentinel")` in the kernel
   and confirm the test fails to compile).
5. Append the row; commit the manifest change alongside the migration.

### Step 6 — Record the migration

- Update `migration_blockers.md` or the relevant `*_analysis.md` to mark the kernel as migrated.
- Cross-reference the helper feature(s) it now consumes so later removals notice the dependency.

### CB Lifecycle Taxonomy

Every helper that wraps a CB-consuming or CB-producing block exposes some form of policy enum that selects the wait/pop or reserve/push pattern it emits. The names differ per helper; the underlying lifecycles are the same. Use this taxonomy to read the raw kernel, then look up the matching enum in the target helper's header.

**Input lifecycles**

| Raw pattern | Lifecycle |
|---|---|
| `cb_wait_front(A, 1); ... cb_pop_front(A, 1)` per tile | per-tile (helper waits + pops) |
| `cb_wait_front(A, N); ... cb_pop_front(A, N)` at end of block | upfront (helper waits + pops at end) |
| `cb_wait_front(A, N)` once before loop, popped once after loop, but **outside** the helper-replaced block | pre-waited (helper pops at end, caller waits) |
| `cb_wait_front(A, N)` once before loop, never popped in loop | persistent / caller-managed (helper neither waits nor pops) |
| Tiles already present (pushed by reader, no wait in compute) | streaming, pre-pushed (helper neither waits nor pops; caller pops downstream) |
| Cumulative wait (`cb_wait_front(A, base + i)` growing each iteration) | **unsupported** — leave on raw LLK |

**Output lifecycles**

| Raw pattern | Lifecycle |
|---|---|
| `cb_reserve_back(1); pack; cb_push_back(1)` per tile | per-tile |
| `cb_reserve_back(N)` upfront, pack sequential, `cb_push_back(N)` at end | bulk |
| `cb_reserve_back(chunk); pack chunk; cb_push_back(chunk)` repeated | per-chunk |
| CB pre-reserved by caller, helper packs sequentially, helper or caller pushes at end | bulk (a duplicate `cb_reserve_back` on an already-reserved CB is a no-op — safe) |

### Partial Migration

A kernel that has SOME migratable stages and SOME blocked stages is **PARTIAL** — not NOT-MIGRATED. Replace only the migratable stages; leave the rest on raw LLK. Document which stages were migrated and the specific blocker per blocked stage.

Good indicators that partial migration is worth doing:
- The migratable stage involves multiple CB / DEST / init lifecycle calls — the helper collapses all of them into one call.
- The blocker is in a different stage with no data dependency on the migratable stage's output CB.

The pipeline must NOT (a) refuse to land 80% because it isn't 100%, or (b) silently land 80% with no record of what's left. Every partial migration produces a per-kernel block in the commit message / migration log:

```
kernel: ttnn/.../foo_compute.cpp
migrated: main loop, post-op chain
skipped:
  - prologue scaler init (raw)  — reason: in-DEST hold loop, see eltwise §3.7
  - mid-loop dtype swap         — reason: helper has no mid-chain reinit policy, GAP-12
```

The skipped list feeds the per-helper feature gap map directly. No skipped section = no signal that more work exists.

### Verifying the Test Exercises the Changed Kernel

Kernel filenames are NOT unique across the repo (e.g. `rmsnorm_post_allgather.cpp` exists in multiple directories). A passing test only validates the kernel if the test's program factory compiles THAT file.

Before trusting a passing test:

1. Grep for the **exact relative path** string in program factories:
   ```
   grep -rn "path/to/kernel.cpp" ttnn/cpp --include="*.cpp"
   ```
2. Trace the factory back to the `ttnn.*` op it creates.
3. Confirm the test file calls that op (grep for `ttnn.op_name` in the test).
4. Optional paranoid check: introduce a `static_assert(false, "sentinel")` in the kernel, run the test, confirm it FAILS to compile, then revert.


### Anti-patterns (do NOT do these during migration)

- **Hand-coding around a helper gap.** If an op, policy, or fusion point is missing, fix the helper (Helper Update / Helper Creation) — do not inline a workaround in the kernel.
- **Batching migrations of unrelated kernels in one commit.** One kernel per commit so failures bisect to a single change.
- **Silently dropping FP32_DEST_ACC-guarded reconfig calls.** Verify the helper emits an equivalent reconfig (or flag the gap and leave the path on raw LLK).
- **Marking a kernel NOT-MIGRATED when only some stages are blocked.** Log as PARTIAL, migrate the clean stages, record the specific blocker per blocked stage.
- **Assuming the first passing test validates your edit.** Verify the test exercises your exact kernel file — same-named files in other directories can silently shadow the one you changed.
- **Skipping the dtype matrix.** Re-run `fp32_dest_acc_en ∈ {False, True}` and any mixed-dtype combo the original kernel supported, even if a single bf16 run passes.

## Pipeline Self-Maintenance

Rules below govern the migration pipeline / orchestration layer itself. These are the easy-to-miss process failures that ship even when every per-kernel step passed.

### HQ doc carries helper-agnostic blockers only

This document must list only blockers that apply across **every** helper family — control-flow shape, CB lifecycle hand-off across helper boundaries, in-DEST hold loops, init-state clobbering, dtype dispatch matching. Helper-specific blockers (e.g. eltwise's `WaitAndPop + cb_tile_idx == 0` rule, `Auto + PerTile` deadlock at small CB capacity) live in the helper's own header doc-comment. When this doc accumulates eltwise-only or reduce-only items it sets a false ceiling on what other helper families think is migratable. Audit this doc on every cycle: if a blocker mentions a specific policy enum value or a specific op struct, it does not belong here — move it to the helper header.

### Catalog-coverage audit and remediation before close-out

Every pipeline run ends with a diff of the Phase 0 catalog against the final helper export list. The audit is **remediation-first, not log-first** — every catalogued LLK missing from the final API gets **added** (op struct via the appropriate CRTP base, ~4 lines) before close-out. Drop is an exception, not a co-equal option: a `dropped: <reason>` row in the gap map is only acceptable when the LLK is genuinely out of scope (e.g. macro-injection-only, hardware-not-yet-landed) and a justification is recorded. If the audit produces drops, the pipeline does NOT close out — it loops back into implementation to add the missing op structs first.

Silent omission has shipped helpers without coverage the catalog promised. Logging the gap without fixing it is the same failure with extra paperwork.

### Phase 2 has an explicit handoff step

The migration pipeline ends Phase 2 (helper-driven rewrite + validation) but then often stops mid-stride with no recorded next step. Define the post-Phase-2 handoff explicitly:

1. Update the per-helper `feature_gap_map` with new GAP entries discovered during this cycle.
2. Commit the partial-migration "skipped:" notes (see [Partial Migration](#partial-migration)).
3. Flag any newly-discovered helper-agnostic blocker for inclusion in this HQ doc.
4. Name the next migration target so the next pipeline run starts oriented.

Without this handoff, unmigrated kernels go undocumented and the next pipeline run rediscovers the same gaps.
