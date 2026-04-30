# Pipeline Review — Cranky Edition (Round 3)

## Verdict

Marginal progress. The author finally added a `## Slug Derivation` section, a `## Pipeline Inputs` section, and an `## Existing Helpers` section in HQ; deleted Phase 4d cleanly; deleted the conventions doc cleanly; and converted the Phase-1/2 outputs from "consolidated" to per-group plurals. That is the most that has been fixed in any review cycle so far. But the deeper rot is mostly intact: `pytest_map.md` is still a vapor "single source of truth" (HQ:191), the Phase 2 → Phase 3 transition just relocated the orchestrator-gap one step right (PIPE:104 — "Investigation + verification outputs" plural, no synthesizer named, no consumer rule), the resume table still skips Phase 2 *and* now lies about Phase 1's path (PIPE:24), `llk_review_fix_agent.md` and `llk_device_validation_agent.md` are still orphans (HQ:68–69, never invoked in PIPE:123–166), Phase 3 checkpoint (PIPE:119) still has no actor, feedback loops (PIPE:230–235) still uncapped, anti-patterns still duplicated (HQ:285 vs HQ:302–315), the two contradictory Phase-2 definitions still ship (PIPE:82 = Verification; HQ:306 = "helper-driven rewrite + validation" handoff), and the Migration Steps vs Pipeline Phases dual-process collision is unchanged. The new slug derivation rule has a real bug — `"Tilize / Untilize"` → `tilize___untilize` (three underscores, HQ:27) — and the doc shrugs at it. New citation-discipline language ("agents must cite which claims they pulled from [the learnings doc]", HQ:44) is not enforced anywhere — the proposal agent does not implement it. Net: ~6 of 30 defects fixed or partially fixed; ~4 fresh defects introduced. Not greenlight. Request rewrite of the orchestration spine.

## Diff vs Round 2

Citations: HQ = `ttnn/cpp/ttnn/kernel_lib/agents/llk_helpers_hq.md`; PIPE = `ttnn/cpp/ttnn/kernel_lib/agents/llk_helpers_pipeline.md`.

| Round 2 defect | Round 3 status | Evidence |
|---|---|---|
| Fresh #1 — slug variables triple-named, zero defined | **PARTIAL** | `## Slug Derivation` exists at HQ:12–35. `{category_slug}` and `{helper_name}` and `{group_slug}` defined. PIPE no longer uses `{category}` or `{name}` (verified: only `{category_slug}` / `{helper_name}` / `{group_slug}` appear). But the example `tilize___untilize` (HQ:27) is shipped as a feature, not a bug — see new defect below. |
| Fresh #2 — breadcrumb path contract broken between PIPE and agent files | **UNCHANGED** | PIPE:34 says `agent_logs/{category_slug}/`. `llk_catalog_agent.md:24` writes `agent_logs/${CATEGORY_SLUG}_catalog_breadcrumbs.jsonl` (no subdir). Two layouts. |
| Fresh #3 — `tt_metal/third_party/tt-agents/scripts/logging/` referenced but unused | **UNCHANGED** | PIPE:34 still cites the directory; 9 scripts on disk, none invoked from any agent prompt. |
| Fresh #4 — anonymous orchestrator consolidates Phase 1 outputs | **PARTIAL/MOVED** | Phase 1 output is now per-group (PIPE:78). Phase 2 input is per-group (PIPE:88). Good. But Phase 3 input at PIPE:104 reads "Investigation + verification outputs, locator results" — plural, no synthesizer named, no rule for how the proposal agent reconciles N per-group files. The orchestrator gap moved from Phase 1→2 to Phase 2→3. See cross-cutting #1 below. |
| Fresh #5 — Phase 3 checkpoint hangs the pipeline | **UNCHANGED** | PIPE:119 still: "Checkpoint: Review proposal before proceeding." No actor, no artifact, no protocol. |
| Fresh #6 — feedback loops have no iteration cap | **UNCHANGED** | PIPE:228–235 unchanged. |
| Fresh #7 — typed feedback across markdown is unschemaed | **UNCHANGED** | PIPE:228–235 unchanged. |
| Fresh #8 — test-change approval gate vs Phase 4c emitting tests by design | **UNCHANGED** | HQ:302–304 still says "Adding ... tests requires explicit user approval"; PIPE:152–162 still mandates 7 test variants per Phase 4c. |
| Fresh #9 — Blackhole skip flag has no source | **UNCHANGED** | HQ:110 unchanged. |
| Fresh #10 — `tests/ttnn/unit_tests/kernel_lib/*.py` glob is a no-op | **UNCHANGED** | `ls`: only `__pycache__/`. HQ:114 + HQ:169 still cite the glob. |
| Fresh #11 — `pytest_map.md` does not exist; claimed "single source of truth" | **UNCHANGED** | `ls ttnn/cpp/ttnn/kernel_lib/pytest_map.md` → No such file. HQ:186–219 still calls it "single source of truth"; HQ:191 names the path. |
| Fresh #12 — Phase 0 catalog output location undefined | **UNCHANGED** | PIPE:50 says output is `{category_slug}_catalog.md` — no path prefix; PIPE:23 then mentions `agent_logs/{category_slug}/catalog_*.md` for resume. Two paths. |
| Fresh #13 — "Existing Helpers" focus is per-group cross-cutting grep | **UNCHANGED** | PIPE:74 unchanged. Each per-group investigation agent still re-greps the same `.inl` files in parallel. |
| Resume #1 — staleness, no version stamp, no input hash | **UNCHANGED** | PIPE:21–28 unchanged. |
| Resume #2 — partial-write detection absent | **UNCHANGED** | PIPE:21–28 unchanged. |
| Resume #3 — half-implemented `.hpp/.inl` satisfies "skip to Phase 4" | **UNCHANGED** | PIPE:26 unchanged. |
| Resume #4 — Phase 2 missing from resume detection table | **UNCHANGED** | PIPE:23–26 lists 0/1/3/4. Phase 2 still absent. Made worse: per-group Phase 1 now writes `agent_logs/{category_slug}/{group_slug}_investigation.md` (PIPE:78), but resume rule at PIPE:24 looks for `{category_slug}_investigation.md` (no group_slug, no path prefix). Resume detection cannot find Phase 1 output. |
| Resume #5 — `pytest_map.md` parallel-write race | **UNCHANGED** | HQ:186–219 unchanged. |
| Resume #6 — approval state is not persisted | **UNCHANGED** | PIPE:26 + PIPE:119 unchanged. |
| Resume #7 — feedback loops cross resume boundary with no replay | **UNCHANGED** | PIPE:228–235 unchanged. |
| Cross #1 — two contradictory "Phase 2" definitions | **UNCHANGED** | PIPE:82 = Verification; HQ:306 = "Phase 2 has an explicit handoff step" describing helper-driven rewrite + validation. Both still ship. |
| Cross #2 — Migration Steps vs Pipeline Phases parallel processes | **UNCHANGED** | HQ:121–292 vs PIPE entirety unchanged. |
| Cross #3 — `llk_review_fix_agent.md` / `llk_device_validation_agent.md` orphaned | **UNCHANGED** | HQ:68–69 still lists them. PIPE:123–166 (Phase 4 description) does not invoke either. The pipeline-level Agent Reference at PIPE:205–212 omits them entirely — table internally inconsistent with HQ:60–69. |
| Cross #4 — slug mining is not an algorithm | **FIXED** | HQ:12–35 defines the rule explicitly. `category_slug = strip().lower().replace(' ', '_').replace('/', '_').replace('-', '_')`. |
| Cross #5 — Anti-patterns appear twice | **UNCHANGED** | HQ:285–292 ("Anti-patterns (do NOT do these during migration)") + HQ:302–315 ("Pipeline Self-Maintenance") still overlap on test-add gate + helper-gap workaround. |
| Cross #6 — `DEST_AUTO_LIMIT` audit citation list | **FIXED** | HQ:92 now reads `tilize_helpers, untilize_helpers, reduce_helpers, matmul_helpers` — those are the `{helper_name}` slugs, not filenames. With the new slug-derivation rule the names are now consistent (slugs of helper families that exist, even though the on-disk filenames carry suffixes like `_compute`/`_dataflow`). Acceptable. |
| Cross #7 — Phase 6 report has no schema and no consumer | **UNCHANGED** | PIPE:188–199 unchanged. |
| Cross #8 — no commit-discipline contract | **UNCHANGED** | unchanged. |
| Cross #9 — no abort/cleanup contract | **UNCHANGED** | unchanged. |
| Confusing #1 — "improving an existing helper → Phase 4" needs a proposal | **UNCHANGED** | HQ:57 unchanged. |
| Confusing #2 — combinatorial coverage budget | **UNCHANGED** | PIPE:159–162 unchanged. |
| Confusing #3 — Step 1 audit artifact schema undefined | **UNCHANGED** | HQ:127–144 unchanged. |
| Confusing #4 — infrastructure-regression curation | **UNCHANGED** (still deleted, no replacement) | Removal-without-replacement remains lossy. |
| Confusing #5 — Phase 4a kernel location undefined | **UNCHANGED** | PIPE:131–137 unchanged. |
| Confusing #6 — manifest discovery has no fallback for kernels with no test | **UNCHANGED** | HQ:208–219 unchanged. |
| Confusing #7 — "Migration is the FINAL step" vs Phase 5 implementation also migrating | **UNCHANGED** | HQ:123 vs PIPE:171–184 unchanged. |
| Confusing #8 — Helper Integration 7 mandatory items, no priority | **UNCHANGED** | PIPE:152–162 unchanged. |
| Round-2 fresh — `matmul_helpers` / `reduce_helpers` cited as nonexistent files | **FIXED** (by reframing) | After the slug-derivation rule landed, those tokens read as `{helper_name}` slugs, not file basenames. Defect retired. |
| Round-2 fresh — conventions doc orphaned on disk | **FIXED** | `git status`: `D ttnn/cpp/ttnn/kernel_lib/agents/llk_helpers_conventions.md`. File deleted. |
| Round-2 fresh — "per-helper learnings doc" referenced 4×, no path/schema/example | **PARTIAL** | HQ:37–44 now defines learnings as **optional**, distilled, non-substitute for Phase 0/1/2. Better. Still: no path schema (where does it live? `agents/{helper_name}_learnings.md`? `kernel_lib/{helper_name}_learnings.md`?), no template, only one example exists on disk (`agents/eltwise_helper_lessons.md` — note the *different* slug `eltwise_helper_lessons` vs the rule's implied `eltwise_learnings`). The convention is undocumented. |

**Summary:** 4 fixed (slug rule, Phase 4d removal, conventions deletion, DEST_AUTO_LIMIT citation reframed), 3 partial (orchestrator gap moved not closed, learnings doc semi-defined, slug variables defined but resume path inconsistent), ~24 unchanged. Net forward motion: real but small.

## Current blockers (fresh-run)

1. **HQ:27 — `category_slug` rule produces `tilize___untilize` (triple underscore) for `"Tilize / Untilize"`.** The example is shipped as the canonical output, no `re.sub(r'_+', '_', ...)` collapse, no special-case for `' / '` → `'_'`. The slug appears in artifact filenames (`tilize___untilize_catalog.md`, `tilize___untilize_helper_proposal.md`, `agent_logs/tilize___untilize/...`). It is ugly and a near-certain typo magnet on resume runs (operator types two underscores, resume detection misses). Fix the rule: collapse runs of `_` and strip leading/trailing `_`.
2. **PIPE:104 — Phase 3 input is plural and the synthesizer is unnamed.** "Input: Investigation + verification outputs, locator results." Phase 1 emits N per-group `{group_slug}_investigation.md`. Phase 2 emits N per-group `{group_slug}_verification.md`. The proposal agent at PIPE:106–116 is asked to read all of them and emit a single `{category_slug}_helper_proposal.md`. There is no synthesis sub-step, no orchestrator that hands the proposal agent a flattened view, no rule for resolving conflicting verification verdicts across groups. The same orchestrator gap that hit Phase 1→2 in Round 2 has been reincarnated at Phase 2→3. Round-2 author claim "consolidation removed" is technically true and operationally meaningless — a downstream consumer that requires a synthesis must implement one, named or not.
3. **PIPE:24 vs PIPE:78 — resume detection cannot find Phase 1 output.** Phase 1 writes to `agent_logs/{category_slug}/{group_slug}_investigation.md` (per-group). Resume rule at PIPE:24 looks for `{category_slug}_investigation.md` (no group_slug, no path). The resume rule is for a file the pipeline never writes. Resume always re-runs Phase 1.
4. **PIPE:23–26 — Phase 2 still missing from resume table.** Verification can be deleted, resume cannot detect, Phase 3 proceeds against a pre-verification proposal. Same as Round 2.
5. **HQ:186–219 — `pytest_map.md` still vapor.** `ls`: file does not exist. "Single source of truth" in HQ:188 with zero existence. Same as Round 2.
6. **HQ:114, 169 — `tests/ttnn/unit_tests/kernel_lib/*.py` still empty.** `ls`: only `__pycache__/`. The glob is a no-op. Same as Round 2.
7. **HQ:68–69 vs PIPE:205–212 — `llk_review_fix_agent.md` and `llk_device_validation_agent.md` orphaned.** Files exist on disk. HQ table lists them as Phase 4. PIPE Phase 4 description (PIPE:123–166) does not invoke either. PIPE Agent Reference table (PIPE:205–212) omits both — internally inconsistent with HQ. Same as Round 2.
8. **PIPE:119 — Phase 3 checkpoint actor undefined.** "Checkpoint: Review proposal before proceeding." Who reviews? Where is approval recorded? Same as Round 2.
9. **PIPE:228–235 — feedback loops uncapped.** 4a fail → re-enter Phase 3, repeat. No iteration limit, no escalation rule. Same as Round 2.
10. **HQ:44 — citation-discipline rule unenforced.** "When [a learnings doc is] present, agents must cite which claims they pulled from it." `grep cite\|claim\|learnings` in `llk_helper_proposal_agent.md` finds only "cite file:line" for codebase examples and "cite cross-iteration state variables" — no instruction to attribute claims back to the learnings doc. The HQ rule is aspirational; no agent prompt implements it.
11. **HQ:48 — Phase 0 cross-reference rule unenforced.** "Phase 0 (Catalog) cross-references the existing surface to avoid duplicate proposals." `grep -i 'existing helpers\|cross-reference\|kernel_lib\|duplicate' llk_catalog_agent.md` returns nothing. The catalog agent does not enumerate existing `*_helpers.{hpp,inl}` files and does not check for overlap. Aspirational rule, no implementation.
12. **PIPE:34 — `tt-agents/scripts/logging/` cited but not invoked.** 9 scripts exist; no agent calls them. Two breadcrumb formats coexist.
13. **No perf gate anywhere in the pipeline.** Phase 4d (Performance) was deleted cleanly — `grep -in '4d\|performance\|benchmark\|threshold' pipeline.md` returns 0 hits. But the pipeline now has no overhead measurement, no perf regression gate, no acceptance criterion. Deletion was clean; replacement is absent. If the pipeline ships a helper with 3× overhead vs raw LLK, there is no signal.

## Current blockers (resume-run)

1. **PIPE:24 — resume rule looks for a file path Phase 1 never writes.** See fresh blocker #3. Phase 1 emits per-group; resume rule expects a single `{category_slug}_investigation.md`. Resume always re-runs Phase 1 unnecessarily.
2. **PIPE:23–26 — Phase 2 absent.** Verification artifact deletion not detected.
3. **PIPE:21–28 — no version stamp, no input hash, no `pipeline_version` frontmatter.** A 6-month-old artifact still satisfies "exists."
4. **PIPE:21–28 — no partial-write sentinel.** SIGINT mid-write leaves a half-file that resume treats as complete.
5. **PIPE:26 — half-built `.hpp/.inl` satisfies "skip to Phase 4."** No compile check, no pair-completeness check.
6. **HQ:186–219 — pytest_map.md parallel-write race.** Still nominal — the file doesn't exist, but the write protocol described would race if it did.
7. **PIPE:26 — approval state not persisted.** Phase 3 checkpoint output not pinned to a SHA, not rebound after resume.
8. **PIPE:228–235 — mid-loop kill leaves no replay state.** Resume detects later-phase artifact, skips earlier phase that was being re-run.

## Cross-cutting blockers

1. **The orchestrator gap moved from Phase 1→2 to Phase 2→3.** Round 2's "consolidation step" was the locus of the anonymous-actor defect. Removing the step did not remove the actor — it just moved the unspecified work into the proposal agent's read step (PIPE:104). Until a named synthesizer (or an explicit "proposal agent reads N files and resolves conflicts via rule X") lands, the gap persists.
2. **Two "Phase 2" definitions still ship.** PIPE:82 = Verification (input: investigation md). HQ:306 = "Phase 2 has an explicit handoff step" describing helper-driven rewrite + validation (i.e. PIPE Phases 4+5). Same as Round 2.
3. **HQ Migration Steps (HQ:121–292) vs PIPE Pipeline Phases (PIPE entirety) still parallel processes.** Step 4 (per-kernel verify) and Phase 4 (helper validation) collide on number and verb.
4. **Anti-patterns duplicated.** HQ:285–292 vs HQ:302–315 still overlap on test-add gate (HQ:285 anti-pattern: "Hand-coding around a helper gap" implicitly covers helper-creation skipping; HQ:304 has the explicit test-add approval rule). Round-2 unchanged.
5. **Phase 6 report (PIPE:188–199) has no schema and no consumer.** Same as Round 2.
6. **No commit-discipline contract.** Same as Round 2.
7. **No abort/cleanup contract.** Same as Round 2.

## New defects from this edit cycle

1. **HQ:27 — slug rule produces triple-underscore output and ships the bug as the example.** `"Tilize / Untilize"` → `tilize___untilize`. Author was warned in the Round-3 prompt; rule was committed unchanged. The rule needs a `re.sub(r'_+', '_', ...)` collapse and a leading/trailing strip. Fix is one line of regex.
2. **PIPE:24 — resume rule paths inconsistent with Phase 1 output paths.** Phase 1 now emits `agent_logs/{category_slug}/{group_slug}_investigation.md` (per-group, with path); resume looks for `{category_slug}_investigation.md` (no group, no path). Net effect: Phase 1 resume detection is broken. Same defect class as the Round-2 catalog mismatch (PIPE:23 vs PIPE:50), now reincarnated.
3. **HQ:44 — "agents must cite" without enforcement.** New rule. Zero implementation in `llk_helper_proposal_agent.md`. Aspirational.
4. **HQ:48 — Phase 0 "cross-references the existing surface" without enforcement.** New rule. Zero implementation in `llk_catalog_agent.md`. Aspirational.
5. **Phase 4d deletion left no perf gate behind.** The previous spec at minimum named a perf phase; deleting it without a replacement means the pipeline has no overhead criterion. Helpers that are slow vs raw LLK ship silently.
6. **PIPE:88 — Phase 2 verification runs in parallel per-group.** Same parallel-write hazard that hit `pytest_map.md` now applies to any output that aggregates across groups. Phase 2 itself writes per-group (safe), but Phase 3 reads N files; if any per-group verification disagrees with another (one says CONFIRMED, one says INCORRECT for a shared claim), there is no resolution rule. Phase 3 inherits the parallel-write hazard one step downstream.

## Confusing / underspecified

1. HQ:37–44 — learnings doc has no canonical path, no template, no naming convention (`eltwise_helper_lessons.md` does not match any obvious slug-derived name).
2. HQ:55 (When-to-Use-What table, "Adding ops to an existing helper") — instructs reading "the helper's `.hpp` doc-comment + its learnings doc" but learnings doc may not exist (HQ:42 says absence is normal). Operator-facing table assumes file presence the rule contradicts.
3. PIPE:50 (Phase 0 output path) vs PIPE:23 (resume detection path) — `{category_slug}_catalog.md` vs `agent_logs/{category_slug}/catalog_*.md`. Same path-mismatch class as new defect #2.
4. HQ:31 — "A category with one helper has `helper_name == category_slug`; a category that produces multiple helpers has one `helper_name` per helper file, all listed in the proposal." Where in the proposal? Schema undefined.
5. PIPE:159–162 — combinatorial budget still unbounded.
6. HQ:127–144 — Step 1 audit output schema still undefined.
7. PIPE:131–137 — Phase 4a raw-LLK kernel write location still undefined.
8. PIPE:152–162 — 7 mandatory Helper Integration items, no priority order.

## Things now genuinely correct

- HQ:12–35 `## Slug Derivation` — defines the rule explicitly and bans legacy variants. Modulo the triple-underscore bug, this is real progress.
- HQ:37–44 `## Pipeline Inputs` — the partition between mandatory `category_name` and optional learnings doc is sensible (the absence-is-normal clause prevents the operator from blocking on missing prior context).
- HQ:46–48 `## Existing Helpers` — pointing at `ttnn/cpp/ttnn/kernel_lib/` is the right move; the rule is just unenforced in the catalog agent.
- Phase 4d deletion is clean — no dangling `4d` references, Phase 5 step 4 (PIPE:181) reads "Run Phase 4c tests" not "4c/4d", dependency graph (PIPE:220) reads `4a->4b->4c`. Surgical.
- Conventions doc deletion is clean (`git status`: `D`).
- HQ:80–92 Helper Design Principles (CB-lifecycle ownership, no-literal-DEST-capacity rule). Substance unchanged.
- HQ:226–248 CB Lifecycle Taxonomy.
- HQ:270–283 Verifying-the-Test-Exercises-the-Changed-Kernel.
- HQ:250–268 Partial Migration commit-block schema.

## Recommended next step

Request rewrite of the orchestration spine, not another patch cycle. The author has demonstrated they can make surgical deletions (Phase 4d, conventions doc) and add structured rule sections (Slug Derivation, Pipeline Inputs, Existing Helpers) when prompted. The remaining defects are not surface-level — they are spec-shape. Specifically: (1) name an orchestrator and define the Phase 2 → Phase 3 synthesis explicitly (or push the synthesis into the proposal-agent prompt with concrete rules for resolving conflicting verification verdicts); (2) reconcile the resume-detection paths with the new per-group output paths (every artifact path appears in exactly two places — the producer and the resume rule — and they MUST match character-for-character); (3) collapse `_+` in the slug rule and re-derive every example in HQ; (4) wire `llk_review_fix_agent.md` and `llk_device_validation_agent.md` into Phase 4 with explicit invocation, or delete them from HQ; (5) seed `pytest_map.md` (even with one row) or delete the entire Pytest Manifest section; (6) seed at least one `*_learnings.md` per existing helper with the canonical filename and a 5-line stub, or strip the learnings doc references; (7) wire the citation-discipline rule into `llk_helper_proposal_agent.md` and the existing-surface rule into `llk_catalog_agent.md`, or delete both rules; (8) decide whether perf gating is required and either reinstate Phase 4d in some form or document the deliberate omission. Until those eight land together, every individual edit cycle will fix one defect and surface another. The pattern is structural — patch cycles will not converge.
