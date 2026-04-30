# Compute Kernel Library — Helper Pipeline

Sole entry for creating, extending, or migrating onto `compute_kernel_lib` helpers in `ttnn/cpp/ttnn/kernel_lib/`. One pipeline; the pipeline self-classifies into the right path based on content discovery, not on operator pre-knowledge.

---

## Invocation

```
inputs:
  category_name:  <string, mandatory>   # e.g. "Reduce", "Tilize / Untilize"
  learnings_doc:  <path, optional>      # distilled context from prior iterations
```

The pipeline derives every call site itself in Phase 0 — operator does NOT pass globs, file lists, or scope hints. Operator-supplied scope hides callers and hides bugs; if a scope-narrower is genuinely needed for an exploratory run, pass `--scope-override <glob>` explicitly and the report records the reduced scope.

**Architecture mandate:** every helper compiles and validates on **both Wormhole_b0 and Blackhole**. There is no per-helper arch toggle. The pipeline runs Phase 5 + Phase 7 on both archs and a helper that fails on one arch is BLOCKER, not partial-pass. The only exception is an LLK primitive that physically does not exist on one arch — in that case, Phase 4 proposal records `arch_only_llk: <list>` with citations and Phase 5 documents the asymmetry; the helper API still exposes the primitive on the unsupported arch as a `static_assert` failure, never as silent absence.

The pipeline always runs end-to-end from Phase 0. Branching happens inside Phase 1 — operator never picks "create vs extend." Migration (Phase 7) is opt-out via `--no-migration`, otherwise default-on.

---

## Slug Derivation

Three placeholders. Every artifact filename and every breadcrumb path uses these spellings, no variants.

```
category_slug = re.sub(r'_+', '_',
                  category_name.strip().lower()
                    .replace('/', '_')
                    .replace('-', '_')
                    .replace(' ', '_')
                ).strip('_')

helper_name   = decided in Phase 4 (Proposal), recorded in proposal frontmatter
group_slug    = derived from group label using the same rule as category_slug
```

Examples: `"Reduce"` → `reduce`. `"Tilize / Untilize"` → `tilize_untilize` (single underscore — the `_+` collapse runs after substitution). `"matmul-block"` → `matmul_block`.

A category that produces one helper has `helper_name == category_slug`. A category that produces N helpers records N `helper_name`s in the proposal frontmatter.

**Banned variants** (do not introduce in any agent prompt, doc, or commit): `{category}`, `{name}`, `${CATEGORY_SLUG}`, `{{CATEGORY_SLUG}}`, `{slug}`.

---

## Artifact Layout

Every phase output lives under one root: `agent_logs/{category_slug}/`. One canonical path per artifact class, used by both producer and resume-detection.

| Phase | Artifact path |
|---|---|
| 0 — Discovery | `agent_logs/{category_slug}/0_discovery.md` |
| 1 — Classify | `agent_logs/{category_slug}/1_classify.md` |
| 2 — Investigation (per group) | `agent_logs/{category_slug}/2_investigation_{group_slug}.md` |
| 3 — Verification (per group) | `agent_logs/{category_slug}/3_verification_{group_slug}.md` |
| 4 — Proposal | `agent_logs/{category_slug}/4_proposal.md` |
| 4 — Approval (gate) | `agent_logs/{category_slug}/4_proposal.approved` |
| 5 — Validation | `agent_logs/{category_slug}/5_validation.md` |
| 6 — Implementation | `agent_logs/{category_slug}/6_implementation.md` (notes; helpers land in `ttnn/cpp/ttnn/kernel_lib/`) |
| 7 — Migration | `agent_logs/{category_slug}/7_migration_{kernel_slug}.md` (one per migrated kernel) |
| 8 — Report | `agent_logs/{category_slug}/8_report.md` |
| Logs | `agent_logs/{category_slug}/breadcrumbs.jsonl` |

Every artifact begins with YAML frontmatter:

```yaml
---
pipeline_version: 1
phase: <0..8>
produced_at_commit: <git rev-parse HEAD>
inputs_hash: <sha256 of upstream artifact paths + their contents>
status: in_progress | done
---
```

Resume detects an artifact as **complete** iff `status == done` AND `inputs_hash` matches a fresh re-derivation. Anything else triggers re-run of that phase.

---

## Pipeline Overview

```
        ┌─ Phase 0: Discovery ──┐    content-based pattern search
        │                       │    against existing helpers + call sites
        └───────────┬───────────┘
                    ▼
        ┌─ Phase 1: Classify ───┐    NO_OP | EXTEND_EXISTING | CREATE_NEW
        └───────────┬───────────┘
            ┌───────┼───────────────────┐
        NO_OP    EXTEND               CREATE
            │       │                   │
            │   ┌───▼─── Phase 2 ───────▼───┐    Investigation (scoped vs full,
            │   │      (per group, parallel)│    branch decides scope)
            │   └─────────────┬─────────────┘
            │                 ▼
            │   ┌─────── Phase 3 ───────────┐    Verification (per group, parallel)
            │   └─────────────┬─────────────┘
            │                 ▼
            │   ┌─────── Phase 4 ───────────┐    Proposal (delta vs full per branch)
            │   │   + human approval gate   │
            │   └─────────────┬─────────────┘
            │                 ▼
            │   ┌─────── Phase 5 ───────────┐    Validation (5a raw / 5b param / 5c integration)
            │   └─────────────┬─────────────┘
            │                 ▼
            │   ┌─────── Phase 6 ───────────┐    Implementation
            │   └─────────────┬─────────────┘
            │                 ▼
            │   ┌─── Phase 7 (optional) ────┐    Migration of existing kernels
            │   └─────────────┬─────────────┘
            ▼                 ▼
        ┌─────────── Phase 8: Report ───────┐
        └───────────────────────────────────┘
```

---

## Phase 0 — Discovery

**Goal:** content-based pattern match. Decide whether the requested behavior already exists, partially exists, or is genuinely new.

**Agent:** `discovery_agent.md` (Explore)

**Steps:**

1. **Index existing helpers** in `ttnn/cpp/ttnn/kernel_lib/`:
   - Read each `*_helpers.hpp` doc-comment block.
   - Enumerate every op-struct declaration (parse `struct <Name> { ... };` blocks).
   - Enumerate every policy enum (`enum class <Name>Policy { ... };`).
   - Build `existing_helpers_index.md` (cached per pipeline run) with: helper_name → {covered LLK calls, op structs, policies, dtype matrix, owner}.

2. **Derive the LLK surface for the category.** No operator-supplied globs. Two complementary searches:

   a. **LLK-prefix bottom-up search.** From the category name, derive the candidate LLK function-name prefixes (e.g. `category_name = "Reduce"` → `llk_math_reduce_*`, `reduce_tile*`, `reduce_init*`; `"Matmul"` → `llk_math_matmul_*`, `matmul_init*`, `matmul_tile*`). Source the prefix list from `tt_metal/include/compute_kernel_api/` headers and the LLK directory (`tt_metal/third_party/tt_llk/`).

   b. **Helper-symbol top-down search.** From `existing_helpers_index.md`, list every public symbol in `compute_kernel_lib` that the category may already touch (e.g. `compute_kernel_lib::reduce_*`, `compute_kernel_lib::matmul_*`).

3. **Find every caller.** Grep the entire repo (not a glob handed by the operator):
   - `ttnn/cpp/ttnn/operations/**/*.cpp` and `**/*.hpp`
   - `ttnn/cpp/ttnn/kernel_lib/tests/**/*.cpp`
   - `tests/**/*kernel*.cpp`
   - `tests/tt_metal/**/*.cpp`

   For each match, record: `{file, line, llk_or_helper_symbol, surrounding_block_signature}`. Operator may pass `--scope-override <glob>` for exploratory runs; the override is recorded in `8_report.md` as a coverage caveat.

4. **Extract raw LLK call sequences per caller.** From each caller block, capture: `*_tile_init`, `*_tile`, `pack_tile`, `tile_regs_*`, `cb_*`, `reconfig_data_format*`, `reduce_*`, `matmul_*`, `unary_op_init_common`, `binary_dest_reuse_tiles_init`, etc. Group sequences by structural shape (init+exec pair, control-flow shape, dtype assumptions, CB lifecycle).

5. **Match call-site sequences against `existing_helpers_index.md` using content** (op-struct LLK signatures, policy semantics), NOT filenames. A call site whose filename mentions "reduce" but whose LLK sequence is actually a chained SFPU expression matches the SFPU helper, not the reduce helper. Filename = ranking hint only.

6. **Emit coverage table.** Per call-site group → one of:
   - `FULLY_COVERED by <helper_name>::<symbol>` (with op-struct evidence cited)
   - `PARTIALLY_COVERED by <helper_name>::<symbol>` (delta described: missing op struct / missing policy / missing dtype)
   - `NOT_COVERED` (raw LLK pattern recorded verbatim, with caller list)

**Output:** `agent_logs/{category_slug}/0_discovery.md` — coverage table + matched op-struct evidence + raw patterns with no match + the full caller list with file:line for every entry.

---

## Phase 1 — Classify

**Goal:** decide branch.

**Agent:** `classify_agent.md` (general-purpose, fast)

**Rules:**

| Discovery says | Branch |
|---|---|
| Every call-site group is fully covered by an existing helper | **NO_OP** — emit migration list, jump to Phase 7 |
| Some groups covered, some need new ops/policies on existing helper | **EXTEND_EXISTING** — Phase 2 scoped to delta |
| No existing helper covers, new family needed | **CREATE_NEW** — Phase 2 full investigation |

**Output:** `agent_logs/{category_slug}/1_classify.md` containing `branch: NO_OP | EXTEND_EXISTING | CREATE_NEW`, target_helper (when EXTEND), delta_summary (when EXTEND), migration_candidates list.

**Approval gate:** if branch == EXTEND_EXISTING, the target helper's owner (recorded in helper's `.hpp` doc-comment header) must approve before Phase 2. If unowned, the pipeline operator approves. Approval = `agent_logs/{category_slug}/1_classify.approved` containing the SHA of `1_classify.md` + the reviewer identifier.

---

## Phase 2 — Investigation

**Goal:** deep analysis of device behavior, host parameter flow, usage patterns, encapsulation requirements.

**Branch-scoped:**

- **CREATE_NEW**: full investigation. Functional groups are decided in Phase 1. One investigation agent per group, parallel.
- **EXTEND_EXISTING**: scoped to the delta. Investigation reads existing helper's `.hpp` + `.inl` as ground truth; only the new surface is investigated.

**Focus areas** (the agent prompt selects which apply per branch):

1. **Device** — wrapper signatures (init + exec), init state compatibility, DEST batching limits, FP32 accumulation requirements, disruptive inits.
2. **Host** — codegen table, program factory layout, parameter encoding (user API value → host transform → kernel receives).
3. **Usage** — call sites, init/exec pairing rules, init mutual exclusion, chaining patterns, parameter usage matrix.
4. **Encapsulation** — compile-time feature matrix, cross-iteration state, side-effect operations, parameter independence.
5. **CB Management** — every `cb_reserve_back`/`cb_push_back`/`cb_wait_front`/`cb_pop_front` in production kernels, with purpose annotation. Flag ordering constraints, shared-memory CBs, dangling reserves.
6. **Existing surface** — for EXTEND only: enumerate every op-struct + policy enum the new delta touches.

**Output:** `agent_logs/{category_slug}/2_investigation_{group_slug}.md` per group. No consolidation step — Phase 3 reads them per-group.

---

## Phase 3 — Verification

**Goal:** check investigation claims against actual code.

**Per group:** one verification agent, parallel.

**Per claim:** CONFIRMED / INCORRECT / UNVERIFIABLE.

**Conflict resolution:** when two groups verify the same shared claim with different verdicts, the verifier flags it as `CONFLICT` and the proposal agent (Phase 4) treats the claim as INCORRECT until re-checked.

**Output:** `agent_logs/{category_slug}/3_verification_{group_slug}.md` per group.

---

## Phase 4 — Proposal

**Goal:** design helper API, op structs, dispatch, migration tier list.

**Branch-shaped output:**

- **CREATE_NEW**: full API spec (signatures, enums, op structs, policy enums, CRTP bases, before/after examples, tier-1 / tier-2 / tier-3 migration list).
- **EXTEND_EXISTING**: delta spec — only the new op structs / policies / signatures, with `before/after` for the existing helper's section that changes.

**Frontmatter records:**

```yaml
helper_name: <one or more>
target_helper:    # only for EXTEND
  file: ttnn/cpp/ttnn/kernel_lib/<file>_helpers.hpp
  delta_kind: op_struct_add | policy_add | dispatch_add | reconfig_path_add

test_contract:
  golden_spec:    <pseudocode or torch expr; one per helper API entrypoint>
  tolerances:
    # Two complementary metrics. PCC catches systematic correlation regressions
    # (helper drift across many elements). ULP catches per-element precision
    # regressions (a 1-ULP shift signals lost mantissa bits). Both must pass.
    bf16:
      pcc:        0.9999     # min Pearson correlation against golden
      ulp_max:    8          # worst-element ULP distance against golden, cast to dtype
      ulp_p99:    2          # 99th-percentile ULP distance
    fp16_b:
      pcc:        0.9999
      ulp_max:    8
      ulp_p99:    2
    fp32:
      pcc:        0.99999
      ulp_max:    4
      ulp_p99:    1
    int32:
      mode:       exact      # bitwise equality; ULP/PCC do not apply
  dtype_matrix:   [[in, dest, out], ...]    # exact tuples sourced from program-factory dispatch table
  edge_axes:
    tile_counts:        [1, 8, 64]           # at minimum: single tile, fits-in-DEST, spans multiple DEST windows
    fp32_dest_acc_en:   [false, true]
    chains:             [single_op, two_op, fan_out]
  adversarial_inputs:
    - uniform(-1, 1)
    - normal(0, 1)
    - denormal_floats            # if dtype_matrix includes any float
    - boundary_tile_size:  [1, 32, 33]
    - extreme_values:      [inf, -inf, nan]   # for float helpers; document if helper does not handle
  arch_only_llk:   []            # populated only when an LLK primitive is single-arch; citations required
```

Note: helpers run on Wormhole_b0 AND Blackhole by default. `arch_only_llk` is the only legal asymmetry, and only when the underlying LLK primitive is single-arch.

`test_contract` is the **frozen semantic spec** that Phase 5c binds against. Approval of the proposal = approval of the contract. Phase 5c can freely **add** variants beyond the contract (more dtypes, more tile counts, more chain shapes — coverage growth is unconditional). Phase 5c cannot **remove**, **shrink**, **loosen** any contract entry without `5_test_change_{N}.approved`. See "Test approval discipline" below.

**Citation discipline:** every proposal claim cites the upstream evidence: `2_investigation_{group_slug}.md:<line>`, `3_verification_{group_slug}.md:<line>`, or `learnings_doc:<line>`. Proposal claims with no citation are flagged in the approval gate.

**Approval gate (HUMAN):** the proposal MUST be approved by a human reviewer before Phase 5 starts. Approval is a file: `agent_logs/{category_slug}/4_proposal.approved` containing:

```yaml
proposal_sha: <git sha-1 of 4_proposal.md content>
reviewer:     <name or identifier>
timestamp:    <ISO8601>
notes:        <optional>
```

The pipeline halts at Phase 4 until that file exists. On resume, if `proposal_sha` does not match the current proposal content, approval is invalidated and the gate re-blocks.

**Output:** `agent_logs/{category_slug}/4_proposal.md`.

---

## Phase 5 — Validation

**Goal:** prove the proposal is correct on device. Three sub-stages, each gates the next.

### 5a — Raw LLK validation

Generate test kernels using **raw LLK calls** (not the helper) that exercise the exact proposed LLK sequences. Run on device against torch goldens.

- Pass → 5b
- Fail/hang → BLOCKER, fix proposal, re-enter Phase 4 (counts toward iteration cap)

Test kernels live at `ttnn/cpp/ttnn/kernel_lib/tests/raw_5a/{category_slug}/`.

### 5b — Parameter coverage

For each proposed LLK call, sweep:

1. **Data formats**: at minimum `BFloat16`, `Float16_b`, `Float32`, plus mixed I/O if the proposal claims support.
2. **Template args**: every value of `Approx`, `Legacy`, `RoundingMode`, etc.
3. **Runtime args**: typical, edge, negative.

Record: observed_supported / observed_failing / unobserved_unsupported. Failing observed combo = BLOCKER. Failing unobserved combo = UNSUPPORTED, recorded in proposal frontmatter.

### 5c — Helper integration

Write test kernels using the helper API as proposed. Tests bind to the **`test_contract`** frozen in Phase 4 proposal frontmatter — every contract entry is exercised. Test boilerplate (parameterize, fixtures, file layout) is the agent's call; semantics are the contract's.

**Mandatory bindings from `test_contract`:**

1. **Golden** — every test asserts against `test_contract.golden_spec` exactly. No alternative goldens.
2. **Tolerances** — every test asserts BOTH metrics:
   - `comp_pcc(actual, expected) >= test_contract.tolerances[<dtype>].pcc`
   - `ulp_distance(actual, expected, dtype=<dtype>).max() <= test_contract.tolerances[<dtype>].ulp_max`
   - `ulp_distance(actual, expected, dtype=<dtype>).quantile(0.99) <= test_contract.tolerances[<dtype>].ulp_p99`
   For `int32` mode=`exact`, the assertion is `torch.equal(actual, expected)` and PCC/ULP do not apply.
3. **Dtype matrix** — every tuple in `test_contract.dtype_matrix` is run.
4. **Edge axes** — Cartesian product of `test_contract.edge_axes` is run, capped by the coverage budget rule below.
5. **Adversarial inputs** — every entry in `test_contract.adversarial_inputs` is exercised.
6. **Arch coverage** — every test runs on Wormhole_b0 AND Blackhole. Failure on either is BLOCKER. Tests covering an `arch_only_llk` primitive run on the supporting arch only and assert `static_assert` failure on the other.

**Append-only freedom:** the agent MAY add test variants beyond the contract — additional dtype tuples it discovers are supported in 5b, additional tile counts that surface a boundary case, additional chain shapes from composition discovery. Coverage growth is unconditional, no gate. Coverage shrinkage (any contract entry not exercised, or exercised with looser tolerance) requires `5_test_change_{N}.approved`.

**Coverage budget rule.** When the flag matrix has F flags, dtype matrix has D dtypes, tile-count axis T values, the maximum mandatory product is `min(F * D, 32) * T`. Beyond that, use a covering-array reduction. Operator can opt into full combinatorial via `--full-coverage` (records in report). Budget reduction must still cover every contract entry at least once — the budget caps the *combinatorial expansion*, not the contract surface.

- Helper fails but raw 5a passed → bug in `.hpp/.inl`, fix and re-run 5c only (no proposal change)
- Helper fails AND the contract is exercised correctly → contract is wrong; re-enter Phase 4 with a delta proposal

**Test change approval (gates contract shrinkage, not contract growth):** see "Test approval discipline" below.

**Output:** `agent_logs/{category_slug}/5_validation.md` with sub-stage results, parameter support matrix, generated-test paths.

---

## Phase 6 — Implementation

**Goal:** write the helper files and update existing-helpers-index.

**For CREATE_NEW:**

1. Write `ttnn/cpp/ttnn/kernel_lib/{helper_name}_helpers.hpp` (declarations, enums, op structs, doc-comment).
2. Write `ttnn/cpp/ttnn/kernel_lib/{helper_name}_helpers.inl` (implementation).
3. Doc-comment header MUST include: helper purpose, owned policy enums, op-struct catalog, supported dtype matrix, OWNER (single name).

**For EXTEND_EXISTING:**

1. Edit existing `*_helpers.hpp` / `*_helpers.inl`. Preserve line ranges for unchanged sections.
2. Append new op structs + enum values; do not reorder existing values.
3. Update doc-comment owner-history if owner changed.

**For both:**

- Re-run Phase 5c against the final `.hpp/.inl`. If anything regresses vs. validation kernels, the implementation has drifted from the proposal — revert and re-enter Phase 5.
- Update `existing_helpers_index.md` with the new surface.
- Commit. One commit per helper, message format: `kernel_lib({helper_name}): <create|extend> — <one-line summary>`.

**Output:** `agent_logs/{category_slug}/6_implementation.md` — notes + commit SHA.

---

## Phase 7 — Migration (optional, default-on)

**Goal:** walk the call sites the pipeline identified in Phase 0 (and any flagged in Phase 1's `migration_candidates` list), swap raw LLK → helper, gate each swap on the call site's existing pytest.

**Per-kernel loop:**

1. Audit the kernel against the new helper's API (existing op-struct catalog covers the LLK pattern? CB lifecycle policy matches?).
2. Gate-check: if the API does not cover, log a `helper_gap_*` entry and skip the kernel (do NOT hand-code around the gap).
3. Write the migration: replace LLK block with fully-qualified `compute_kernel_lib::<Symbol>` calls. Preserve surrounding CB lifecycle the helper does not own. Delete dead locals.
4. Verify on device: build + run the kernel's pytest manifest entry. Cover the dtype matrix the kernel supported pre-migration.
5. Confirm the test exercises this exact kernel file (paranoid sentinel check: temporarily plant `static_assert(false, "sentinel")`; expect compile fail; revert).

**Per-kernel artifact:** `agent_logs/{category_slug}/7_migration_{kernel_slug}.md` with audit table, helper-gap log, dtype-coverage record, commit SHA.

**Iteration cap:** if the helper-gap log grows during Phase 7, the pipeline stops migrating, reports the gaps, and exits to Phase 8. Operator decides whether to re-enter Phase 4 (proposal extension) or accept partial migration.

**Skip rules** (record in `7_migration_*.md`):

- Multi-chip ops untestable locally → `untestable_locally: <reason>` + named CI job that exercises the kernel.
- Arch-gated paths (e.g. Blackhole-only LLK) → `arch_skip: <arch>` + CI job.

---

## Phase 8 — Report

**Goal:** consumed by the next pipeline run and by humans.

**Schema** (YAML frontmatter + markdown body):

```yaml
---
pipeline_version: 1
category_name: <input>
category_slug: <derived>
branch: NO_OP | EXTEND_EXISTING | CREATE_NEW
helper_name: <list>
target_helper: <when EXTEND>
phases:
  - { phase: 0, status: done, artifact: ..., commit: ... }
  - ...
validation:
  parameter_support: <table>
  unsupported_combos: <list>
migration:
  total_candidates: N
  migrated: M
  helper_gaps: <list>
  skipped: <list with reasons>
open_items:
  - tier_2_call_sites: <list>
  - tier_3_call_sites: <list>
  - unsupported_combos: <list>
---
```

The next pipeline run reads `8_report.md` to seed `migration_candidates` and `unsupported_combos`. Human reads the markdown body for narrative.

---

## Resume Detection

Resume rule for every phase: read the artifact at the canonical path (table above), check frontmatter `status == done` AND `inputs_hash` matches re-derivation. Pass → skip phase. Fail → re-run.

Branch-aware: resume reads `1_classify.md` first to determine branch, then applies branch-specific resume rules (e.g. EXTEND skips a missing per-group investigation if the delta only touches one group).

Partial-write protection: every artifact writes to `<path>.tmp` and atomically renames on completion. Resume ignores `.tmp` files.

---

## Feedback Loops

| Failure | Action | Iteration cap |
|---|---|---|
| 5a fails (raw LLK invalid) | Re-enter Phase 4 (Proposal). | Max 3 attempts before BLOCKER + exit |
| 5b fails (observed param combo) | Re-enter Phase 4. | Max 3 attempts |
| 5c fails (helper bug, raw passed) | Re-run Phase 6 (fix `.hpp/.inl`). | Max 3 attempts |
| Phase 2 incomplete (discovered in Phase 4) | Re-enter Phase 2 for missing groups. | Max 2 attempts |
| Phase 7 helper-gap | Stop migration, exit to Phase 8. | N/A (single shot) |

Iteration counter persists across resume in the artifact frontmatter (`iteration: N`).

---

## Approval Gates (human-in-the-loop)

The pipeline has three explicit human-approval gates. Each is a file on disk; the pipeline halts at the gate until the file exists and matches the artifact's SHA.

| Gate | File | Required when |
|---|---|---|
| Phase 1 classify approval | `1_classify.approved` | branch == EXTEND_EXISTING (target-helper owner approves) |
| Phase 4 proposal approval | `4_proposal.approved` | always |
| Phase 5 test change approval | `5_test_change_{N}.approved` | per test add / remove / skip / tolerance change |

Gates record the artifact SHA at approval time. On resume, if the SHA no longer matches, approval is invalidated and the gate re-blocks. There is no implicit auto-approval.

See "Test approval discipline" for what counts as a test change requiring a gate.

---

## Test approval discipline

The **test contract** is frozen at Phase 4 proposal approval. The contract is **append-only after approval**: agents can grow coverage freely, but any shrinkage requires a separate gate.

**Gated changes** (require `5_test_change_{N}.approved`):

- Dropping any tuple from `test_contract.dtype_matrix`
- Removing any entry from `test_contract.adversarial_inputs`
- Removing any value from `test_contract.edge_axes.tile_counts` (or any other edge axis)
- Loosening any `test_contract.tolerances` entry — for floats this means *raising* `ulp_max` / `ulp_p99` or *lowering* `pcc`; for `int32` this means dropping `mode: exact`
- Replacing `test_contract.golden_spec` with a different reference
- Skipping a test on Wormhole_b0 or Blackhole for any reason other than the path being a documented `arch_only_llk` primitive
- Removing a test that already exercises a contract entry

**Free changes** (no gate, agent can do during 5c):

- Adding dtype tuples beyond `dtype_matrix` (e.g. 5b discovered support)
- Adding tile counts beyond `edge_axes.tile_counts`
- Adding chain shapes beyond `edge_axes.chains`
- Adding adversarial inputs beyond the contract list
- Adding tests for new helper API variants
- Renaming a test, splitting a test into two, merging two tests — as long as no contract entry loses coverage
- Tightening any tolerance — raising `pcc`, lowering `ulp_max` / `ulp_p99` (stricter is fine, no gate)

**Approval file format:**

```yaml
---
gate: 5_test_change
proposal_sha: <sha-1 of 4_proposal.md content at approval time>
contract_diff:
  removed: [<entries>]
  loosened:
    - { field: tolerances.bf16.pcc,     from: 0.9999, to: 0.999 }
    - { field: tolerances.bf16.ulp_max, from: 8,      to: 16 }
reviewer:  <name or identifier>
timestamp: <ISO8601>
notes:     <reason — required, not optional>
---
```

Reviewer continuity: the proposal-approval reviewer is the **default** test-change reviewer. A different reviewer is acceptable but must be recorded; pipeline does not enforce same-reviewer.

---

## Existing Helpers (location + reading order)

`ttnn/cpp/ttnn/kernel_lib/` holds `{helper_name}_helpers.{hpp,inl}` pairs. Each helper's `.hpp` carries a doc-comment block at the top:

```
// helper_name:        <slug>
// purpose:            <one paragraph>
// op_structs:         <list with one-line each>
// policies:           <enum names + their semantic meanings>
// dtype_matrix:       <supported (input, dest, output) tuples>
// owner:              <single name>
// last_changed:       <git SHA>
```

Phase 0 Discovery reads these doc-comments as ground truth. The doc-comments are the contract; no separate "conventions" doc.

---

## Design Constraints (the rules every proposal must respect)

These apply across every helper family. Helper-specific specifics live in the helper's own `.hpp` doc-comment.

### Helper owns the CB lifecycle it can see

If the helper takes a CB id as a parameter, the helper itself does the `cb_wait_front` / `cb_pop_front` on that CB. Don't push wait/pop back to the caller because "the policy could be different" — the caller already told the helper what CB it is. The helper's policy enum picks **which** shape of wait/pop, not **whether** the helper does them. Caller-side wait/pop only survives where the helper genuinely cannot see the CB (sharded tensors, persistent prologues, fan-out across helper boundaries) — and that path is named explicitly (`NoWait*` / `*NoPop`), not the default.

### DEST capacity is compile-time, never literal

Half-sync fp16 has 8 DEST slots; full-sync has 16; fp32-DEST mode halves both. Use `DEST_AUTO_LIMIT` from `dest_helpers.hpp` (constexpr, derived from `get_dest_limit()` / `DST_ACCUM_MODE`) anywhere a helper bounds DEST slot indices, batch sizes, or chain widths. A literal `8` in an enum or `static_assert` ships a helper that miscompiles silently the moment the kernel runs in a different DEST mode.

---

## CB Lifecycle Taxonomy

Every helper that wraps a CB-consuming or CB-producing block exposes a policy enum that selects the wait/pop or reserve/push pattern it emits. Names differ per helper; the underlying lifecycles are the same. Match raw-kernel patterns to this taxonomy, then look up the matching enum in the target helper's header.

**Input lifecycles**

| Raw pattern | Lifecycle |
|---|---|
| `cb_wait_front(A, 1); ... cb_pop_front(A, 1)` per tile | per-tile (helper waits + pops) |
| `cb_wait_front(A, N); ... cb_pop_front(A, N)` at end of block | upfront (helper waits + pops at end) |
| `cb_wait_front(A, N)` once before loop, popped once after, but **outside** the helper-replaced block | pre-waited (helper pops at end, caller waits) |
| `cb_wait_front(A, N)` once before loop, never popped in loop | persistent / caller-managed (helper neither waits nor pops) |
| Tiles already present (pushed by reader, no wait in compute) | streaming, pre-pushed (helper neither waits nor pops; caller pops downstream) |
| Cumulative wait (`cb_wait_front(A, base + i)` growing each iteration) | **unsupported** — leave on raw LLK |

**Output lifecycles**

| Raw pattern | Lifecycle |
|---|---|
| `cb_reserve_back(1); pack; cb_push_back(1)` per tile | per-tile |
| `cb_reserve_back(N)` upfront, pack sequential, `cb_push_back(N)` at end | bulk |
| `cb_reserve_back(chunk); pack chunk; cb_push_back(chunk)` repeated | per-chunk |
| CB pre-reserved by caller, helper packs sequentially, helper or caller pushes at end | bulk (a duplicate `cb_reserve_back` on already-reserved CB is a no-op — safe) |

---

## Anti-patterns

- **Hand-coding around a helper gap.** If an op, policy, or fusion point is missing, fix the helper (re-enter Phase 4 with extension proposal) — do not inline a workaround in the kernel.
- **Batching migrations of unrelated kernels in one commit.** One kernel per commit so failures bisect to a single change.
- **Silently dropping FP32_DEST_ACC-guarded reconfig calls.** Verify the helper emits an equivalent reconfig (or flag the gap and leave the path on raw LLK).
- **Marking a kernel NOT-MIGRATED when only some stages are blocked.** Log as PARTIAL (migrate clean stages, record specific blocker per blocked stage in `7_migration_{kernel}.md`).
- **Assuming the first passing test validates your edit.** Verify the test exercises your exact kernel file (sentinel `static_assert(false, "sentinel")` test, then revert).
- **Skipping the dtype matrix.** Re-run `fp32_dest_acc_en ∈ {False, True}` and any mixed-dtype combo the original kernel supported, even if a single bf16 run passes.
- **Using `using namespace compute_kernel_lib;` or `namespace ckl = compute_kernel_lib;`.** Always fully qualify helper symbols.

---

## Logging — Decision Log

Each pipeline run owns one rolling decision log: `agent_logs/{category_slug}/decisions.md`. Append-only, chronological, written by every agent at significant moments. **External observers track the run by reading this single file** — per-phase artifacts hold detail; the decision log is the spine.

**Sentinel:** `agent_logs/.active` (YAML, repo-wide, written by the pipeline orchestrator at run start, removed at run end):

```yaml
category_slug: <slug>
phase:         <0..8>
group_slug:    <optional, set during Phase 2 / 3 fan-out>
started_at:    <ISO8601>
```

The orchestrator updates `phase` (and optionally `group_slug`) at every phase transition. Agents read the sentinel implicitly via the helper script.

**Script:** `tt_metal/third_party/tt_ops_code_gen/scripts/logging/append_decision.sh`. Flock-protected atomic append. CLI:

```
append_decision.sh \
  --kind {decision|pain|phase_enter|phase_exit|gate_block|retry} \
  --phase <0..8> \
  --agent <agent_name> \
  [--group <group_slug>]            # optional; inherits from sentinel if absent
  [--category <category_slug>]      # optional; defaults to sentinel
  [--title <one-line title>] \
  ( --body-file <path> | --body-stdin )
```

Hand-rolled `echo` / direct Write to `decisions.md` is forbidden — the script handles frontmatter, timestamp, header on first write, and the flock. Without the lock, parallel Phase 2 / Phase 3 agents collide.

**Entry kinds:**

| Kind | When |
|---|---|
| `phase_enter` | First action of a phase. Frontmatter records inputs hash. |
| `phase_exit` | Last action of a phase. Records artifact path + status. |
| `decision` | Non-trivial choice with ≥ 2 alternatives (branch pick, golden choice, tolerance choice, op-struct catalog deviation). |
| `pain` | Friction: ambiguity, retry, missing surface, blocked path. Records `symptom`, `tried`, `resolved`, `cost`, `residual`. |
| `gate_block` | Pipeline halts at an approval gate (`1_classify.approved`, `4_proposal.approved`, `5_test_change_*.approved`). |
| `retry` | Feedback-loop retry. Records iteration counter against the cap. |

**Body schemas (markdown):**

```markdown
DECISION
**chose:**       <option>
**alternatives:**<other options>
**why:**         <reason>
**evidence:**    <artifact:line citations>
**impact:**      <downstream consequences>
```

```markdown
PAIN
**symptom:**     <what went wrong>
**tried:**       <ordered list of attempts>
**resolved:**    <final approach>
**cost:**        <attempts + wall-clock>
**residual:**    <follow-up flagged in artifact>
```

`phase_enter`/`phase_exit`/`gate_block`/`retry` have ad-hoc bodies; the frontmatter (`kind`, `phase`, `ts`) carries the structured signal.

**Emission rules per phase** (agent prompts enforce):

| Event | Required entry |
|---|---|
| Phase entry | `phase_enter` |
| Phase exit | `phase_exit` |
| Phase 1 branch pick (NO_OP / EXTEND / CREATE) | `decision` |
| Phase 3 conflict resolution (CONFIRMED vs INCORRECT shared claim) | `decision` |
| Phase 4 tolerance choice tighter or looser than dtype defaults | `decision` |
| Phase 4 golden_spec deviation from prior-iteration learnings | `decision` |
| Phase 5 raw vs helper test mismatch | `decision` or `pain` |
| Phase 5 retry on 5a/5b/5c failure | `retry` |
| Phase 7 helper-gap discovered mid-migration | `pain` + pipeline exit to Phase 8 |
| Any pipeline halt at an approval gate | `gate_block` |

**External tracking:**

- Single-run narrative: `cat agent_logs/{category_slug}/decisions.md`.
- Cross-run pain pattern: `grep -A5 "kind:    pain" agent_logs/*/decisions.md` → recurring frictions across categories feed helper-gap proposals.
- Iteration pressure: `grep "kind:    retry" agent_logs/*/decisions.md | wc -l` per phase per category.

A pipeline run with zero decisions emitted is a bug — the run either skipped phases without recording or the orchestrator never wrote the sentinel. Reviewers reject empty decision logs.

---

## Pytest Manifest

`agent_logs/{category_slug}/pytest_map.md` (per-pipeline-run, not repo-wide) records: kernel path → pytest path. Phase 0 seeds it from existing program-factory greps; Phase 7 reads + appends.

Why per-run: the previous repo-wide manifest had a parallel-write race across pipeline runs and no SHA pin for staleness. A per-run manifest is owned by exactly one pipeline run.

A second file `ttnn/cpp/ttnn/kernel_lib/pytest_map_shared.md` records the *shared-helper regression set* — pytests that must run after any change to a helper used across categories. Phase 6 appends to this file under a flock; Phase 5c reads it to choose which extra suites to run.

---

## Open questions deferred to follow-up

- Phase 5b coverage budget when F * D > 32: covering-array generation rule needs a concrete library or hand-rolled algorithm.
- Phase 7 untestable-locally CI handoff: the named CI job format (job name? test id?) needs alignment with the team's CI naming.
- Phase 8 report consumption by the next run: which fields auto-seed Phase 0 vs Phase 1 needs a worked example before locking the schema.
