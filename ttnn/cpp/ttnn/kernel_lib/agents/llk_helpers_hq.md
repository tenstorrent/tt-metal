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

## Helper Design Principles (general)

Rules below apply across every helper family, not just eltwise. Per-helper specifics (policy enum values, op-struct catalog, batching pitfalls) live in the helper's own header doc-comment.

### Helper owns the CB lifecycle it can see

If the helper takes a CB id as a parameter, the helper itself does the `cb_wait_front` / `cb_pop_front` on that CB. Don't push wait/pop back to the caller "because the policy could be different" — the caller already told the helper what CB it is by passing it. The helper's policy enum picks *which* shape of wait/pop, not *whether* the helper does them. Caller-side wait/pop only survives where the helper genuinely cannot see the CB (sharded tensors, persistent prologues, fan-out across helper boundaries) — and that path is named explicitly (`NoWait*` / `*NoPop`), not the default.

A helper that asks the caller to do `cb_wait_front(cb_in0, 1)` *before* invoking it is leaking lifecycle state across the helper boundary, and any chain / composition trait checks the helper performs cannot reason about waits the helper never sees. Swallow the wait/pop or document the gap explicitly as a known-unsupported lifecycle in the helper's header.

### DEST capacity is compile-time, never literal

Half-sync fp16 has 8 DEST slots; full-sync has 16; fp32-DEST mode halves both. Use `DEST_AUTO_LIMIT` from `ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp` (constexpr, derived from `get_dest_limit()` / `DST_ACCUM_MODE`) anywhere a helper bounds DEST slot indices, batch sizes, or chain widths. A literal `8` in an enum or `static_assert` ships a helper that silently miscompiles the moment the kernel runs in a different DEST mode. `DEST_AUTO_LIMIT` is already used across `binary_op_helpers.inl`, `sfpu_helpers.hpp`, `untilize_helpers.inl` — match that convention.

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

Phase 2 (Validate) of the pipeline is responsible for adding these tests —
see [llk_validation_agent.md](llk_validation_agent.md) for the concrete
sub-stage 2c / 2d steps.

## Kernel Migration Steps

Migration is the FINAL step — it consumes helpers that already exist and are validated. If a missing op struct, missing enum value, or missing API surface is discovered mid-migration, stop and close that gap first (Helper Update / Helper Creation), then resume.

The steps below are helper-agnostic. Helper-specific guidance (policy enums, batching rules, op-struct catalog, fusion patterns) lives in the per-helper `.hpp` doc-comments and the helper's section of `llk_helpers_conventions.md` — read those for the helper you are migrating to before Step 1.

### Step 1 — Audit the target kernel

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

### Step 2 — Gate-check against the helper API

If the audit turns up ANY of these, return to a prior step before writing the migration:

- A needed op struct, API, or fusion point does not exist → Helper Update (conventions §5) or Helper Creation, then resume.
- A required policy / enum value is missing (e.g. a CB lifecycle the helper does not yet model, a dtype-reconfig mode it does not emit) → Helper Update via pipeline, then resume.
- The control-flow or LLK pattern is fundamentally unsupported (cumulative waits, deeply interleaved op classes with no fusion point, exotic pack patterns) → leave that block on raw LLK and move on.

### Step 3 — Write the migration

Keep the scope surgical:

- **Only replace the LLK-call block.** Do NOT refactor surrounding code, rename CBs, reorder operations, or touch unrelated paths in the same kernel. Other helper calls already in the file stay as-is.
- **Always fully qualify helper symbols** (`compute_kernel_lib::<Symbol>`). Do NOT introduce `using namespace compute_kernel_lib;` or namespace aliases (`namespace ckl = compute_kernel_lib;`). Namespace pollution — even block-scoped — leaks helper names into later additions to that scope and hides the dependency on the helper library at the call site. The verbosity of full qualification is the point: it is searchable, auditable, and prevents unrelated identifiers from silently binding to helper symbols.
- **Delete dead locals.** DEST-slot indices, scratch counters, and lifecycle bookkeeping that the helper now owns become unreachable. Remove them; do not keep them "for safety".
- **Preserve surrounding CB lifecycle the helper does not own.** If the helper's policy declares it does not wait/pop a given CB (caller-managed / persistent), the kernel's existing `cb_wait_front` / `cb_pop_front` at the surrounding scope MUST stay.
- **Trim includes only when fully unused.** Swap `api/compute/*.h` includes for `ttnn/cpp/ttnn/kernel_lib/*_helpers.hpp` only after every raw LLK call from that header is gone. Headers still needed by unmigrated paths in the same file stay.

### Step 4 — Verify on device

Build is necessary but **not** sufficient — kernels JIT at runtime, so a clean `./build_metal.sh` says nothing about correctness. Every migration MUST pass on real device.

- Build the host library: `./build_metal.sh`.
- Run the kernel_lib validation suite (covers the helper itself):
  `scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/kernel_lib/*.py`.
- Run the operation's pytest(s). Resolve the path via the [Pytest Manifest](#pytest-manifest) instead of re-searching the tree each session.
- Confirm the test exercises THIS kernel file — see [Verifying the Test Exercises the Changed Kernel](#verifying-the-test-exercises-the-changed-kernel). Same-named files in other directories silently shadow the one you changed.
- Cover the dtype matrix the kernel supported pre-migration. Read the **program-factory dispatch table** for the original kernel and mirror that exact list as pytest params — do NOT just run the dtype the migration author happened to test on. Common silent omissions: `float32` input, `int32` input, `bfloat8_b` packed format, mixed `bfloat16 × float32`. At minimum run `fp32_dest_acc_en` ∈ {False, True} when the op tests FP32 accumulation, and any mixed-dtype combos the original kernel handled. If a dtype is intentionally dropped, document why in the test file. "It worked on bf16" is not coverage.
- If any test fails: diagnose whether it's a helper gap (→ Step 2) or a migration mistake (→ Step 3). Do NOT relax the test or skip dtype combos to make it pass.

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
| Step 1 (audit) | If the target kernel is missing from the manifest, find its pytest (grep program factories → test imports), verify it runs, then append a row. |
| Step 4 (verify) | Run every pytest listed for the touched kernel(s). If the manifest row was stale (path moved, test renamed), fix the row before reporting the migration done. |
| Step 5 (record) | If the migration adds coverage on a *new* kernel that the test file didn't previously exercise, note that in the `*_analysis.md`; the manifest row needs no change (same pytest still covers it). |

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

### Step 5 — Record the migration

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

### Test changes require explicit user approval

Adding, removing, or skipping tests **requires explicit user approval** before the pipeline lands the change. "I added a test" is not approval; "I disabled the int32 dtype variant because it failed" is the failure mode that motivates this rule — disabled tests slip past review and the regression ships. The pipeline must surface each test add / skip / tolerance change as a separate approval gate, with the diff and the reason.

### Phase 2 has an explicit handoff step

The migration pipeline ends Phase 2 (helper-driven rewrite + validation) but then often stops mid-stride with no recorded next step. Define the post-Phase-2 handoff explicitly:

1. Update the per-helper `feature_gap_map` with new GAP entries discovered during this cycle.
2. Commit the partial-migration "skipped:" notes (see [Partial Migration](#partial-migration)).
3. Flag any newly-discovered helper-agnostic blocker for inclusion in this HQ doc.
4. Name the next migration target so the next pipeline run starts oriented.

Without this handoff, unmigrated kernels go undocumented and the next pipeline run rediscovers the same gaps.
