# Milestone A — remaining coverage gaps

Three handoff briefs, one per gap, each self-contained for a separate agent. They cover item 5 of the
Milestone A pending list: the coverage gaps that survive the 2026-08-25 RMSNorm2D and Attention2D
fixes.

| Brief | Gap | Host needed | Status |
| --- | --- | --- | --- |
| [gap1_sampling2d_stochastic_hardware.md](gap1_sampling2d_stochastic_hardware.md) | `Sampling2D` has one hardware test and it only exercises the greedy path | 6U WH Galaxy (exclusive) | **CLOSED** 2026-08-25 — found and fixed a real module defect (`ttnn.sampling` `temp` is `1/T`) |
| ↳ [gap1_completion_handoff.md](gap1_completion_handoff.md) | — | — | Done; see [`../tttv2_milestone_a_gap1_evidence/REPORT.md`](../tttv2_milestone_a_gap1_evidence/REPORT.md) |
| [gap2_prefetcher2d_galaxy_ccl_hardware.md](gap2_prefetcher2d_galaxy_ccl_hardware.md) | `Prefetcher2D` / Galaxy resource lifecycle is mock-only; no device test ever changes mode | 6U WH Galaxy (exclusive) | Suite written, 7/8 cases green; cut short by a spend limit |
| ↳ [gap2_completion_handoff.md](gap2_completion_handoff.md) | Finish gap 2: the untried attention case, 3× repeats, finding F1, `REPORT.md` | 6U WH Galaxy (exclusive) | **Next** |
| [gap3_batched_prefill_physical32_trace.md](gap3_batched_prefill_physical32_trace.md) | Batched-prefill policy delegation has no device evidence; physical-32 trace has no subject yet | N150 or T3K | Not started — partly deferrable, see the brief |

## Running the outstanding jobs

`../run_gap_jobs.sh` runs jobs sequentially in unattended `claude -p` sessions, with an auth check
before each and a device-free check between them. It defaults to `gap2-finish`:

```sh
screen -dmS ttgaps /proj_sw/user_dev/ctr-apbernal/tt-metal/run_gap_jobs.sh
screen -r ttgaps        # attach; ctrl-a d to detach
```

`--dry-run` runs every preflight and executes nothing. `--jobs a,b` picks jobs and their order; valid
names are `gap1-finish`, `gap2`, `gap2-finish`, `gap3`.

Both previous interruptions were credential/quota failures, not task failures — an expired login the
first time, an org monthly spend limit (HTTP 429) the second. The pre-job auth check exists to catch
exactly that before a 12 h device job starts on a dead token; when it trips mid-run, the driver stops
rather than burning the remaining jobs.

## Scheduling

**Gaps 1 and 2 both need exclusive use of the Galaxy and must be serialized against each other.**
Exactly one pytest process may hold the mesh at any moment; the work log records multiple invalidated
runs caused by two processes sharing it.

Gap 3's achievable half runs on N150/T3K, so it can proceed in parallel on a different host.

Suggested order if running sequentially on the Galaxy: gap 2 first (it perturbs shared test plumbing
that gap 1 does not depend on, and it re-runs the MLP/RMSNorm suites as its own regression gate),
then gap 1.

## Shared context every agent should read

1. `tttv2_2d_modules_plan.md` — "Milestone A exit gate", "Shared 2D Contracts", and the per-module
   contract for whichever module the brief targets.
2. `tttv2_2d_modules_work_log.md` — **the last two checkpoints in particular.** Both 2026-08-25 root
   causes were L1 address/ownership faults that presented as intermittent passes or hangs rather than
   clean failures. That is the failure mode to expect on this hardware.
3. `tttv2_milestone_a_device_evidence/REPORT.md` — the 2026-08-24 evidence run. Its header records
   which rows are superseded; the run procedure and triage rules in
   `tttv2_milestone_a_device_evidence_agent.md` are still the house standard.

## House rules, common to all three

- One pytest process on a device at a time. Never pipe pytest.
- Run every new device test at least three times in fresh processes before recording it as evidence.
  A case that flips across processes is a defect, not noise.
- A failing test is a result to report, not a bug to patch. Never relax a threshold, tolerance, or
  parametrization to turn a failure green.
- No `git commit` / `push` / `checkout` / `stash` / `reset`. Leave the tree dirty for review.
- Do not edit `models/common/modules/MILESTONE_A_STATUS.md`, `models/common/modules/README.md`, or
  `tttv2_2d_modules_plan.md` — all three are being revised as a separate task. Put proposed
  replacement text in your report instead.
- Do not modify any `models/common/modules/**/*_1d.py` implementation file. Sharing a *test* helper
  across the 1D and 2D suites is fine and has precedent
  (`models/common/tests/modules/_hf_reference.py`).
- Each brief ends with its own deliverables list: a new test file, an evidence directory with raw
  logs and a `REPORT.md`, and a terse checkpoint section appended to `tttv2_2d_modules_work_log.md`.

## Not covered by these briefs

The other Milestone A pending items are separate work:

1. re-running the full 21-case device matrix at the final commit;
2. rewriting `tttv2_milestone_a_device_evidence/REPORT.md` and `MILESTONE_A_STATUS.md`;
3. committing the uncommitted RMSNorm2D / Attention2D fixes;
4. re-running the 1D regression after the `_hf_reference.py` refactor;
5. the README 2D inventory update and the modularity scorecard audit;
6. filing the `models/demos/llama3_70b_galaxy` latent fused-stats exposure noted in the work log.
