# Job 4 — Milestone B exit-gate verdict and scorecard

**No device.** Host-only. This job reads evidence and writes documents; it produces no new
measurements and must not attempt any.

## Mission

Decide, on the recorded evidence, whether Milestone B passes its exit gate — and write that verdict
down in a form the next milestone can act on.

The single most important instruction in this brief: **this job is allowed to conclude that Milestone
B does not pass.** Milestone A declared its exit gate passed on 2026-08-19, was wrong, and the
independent re-run that disproved it found two real defects the "passing" evidence had been masking.
That correction is the most valuable artifact Milestone A produced. Do not repeat the original
mistake here.

## Inputs

- `tttv2_milestone_b_briefs/job3_completion_handoff.md` — start here. If it does not exist, report
  `BLOCKED (mb-coverage did not complete)` and write the status page from whatever evidence does
  exist, clearly marked as partial.
- `tttv2_milestone_b_evidence/{reconcile,llama,qwen,coverage}/REPORT.md` and their raw logs.
- `tttv2_2d_modules_milestone_b_work_log.md`, all checkpoints.
- `tttv2_2d_modules_plan.md` — "Milestone B exit gate" and "Modularity scorecard".
- `models/common/modules/MILESTONE_A_STATUS.md` — the model for how to write this, including its
  tone about its own errors.

## What to check, and how sceptically

For every exit-gate line, find the **raw log** that produced the claim. A number in a `REPORT.md` that
you cannot trace to a log file is not evidence; mark it `UNSUBSTANTIATED` and say which report asserted
it.

```text
Llama teacher-forced, batch 1, prefill 512 / decode 511    top-1 >= 91%   top-5 >= 99%
Qwen  teacher-forced, batch 1, sequence 512                top-1 >= 89%   top-5 >= 97%
Batch-32 direct demos valid, no cross-slot contamination
Batch-1 4K / 32K / 128K functional smokes pass
Prefix-cached output matches uncached execution
No dependency imports from an existing model-named implementation package
Zero changes to 1D module implementation files
Existing 1D model contract and demo-contract host tests green, expectations unchanged
```

Apply Milestone A's hard-won standards to each:

- **Was it run more than once, in fresh processes?** Three of Milestone A's four defects presented as
  intermittent passes. A single passing run is not a qualification; record it as
  `PASSED (single run — not qualified)`.
- **Was it measured at the final tree?** Evidence collected before a later shared-code change is
  provenance, not current evidence. `mb-coverage` was told to re-measure the accuracy gates for
  exactly this reason; check that it did.
- **Did anything pass because its coverage could not reach the defect?** That is how D4 and D5
  survived — greedy-only sampling could not reach a temperature bug, and uniform memory configs could
  not reach a swapped pair. For each passing line, ask what a defect would have to look like to slip
  through it.

The last three lines are mechanical. Verify them yourself rather than trusting a report:

```sh
git diff --name-only <milestone-a-final>..HEAD | grep '_1d\.py'      # must be empty
git diff --name-only <milestone-a-final>..HEAD | grep 'llm_runtime'  # must be empty
git grep -n "demos.llama3_70b_galaxy\|models.llama33_70b\b\|models.qwen3_32b\b" -- \
    models/common/models/galaxy models/common/models/*_galaxy
python -m pytest -q models/common/tests/modules models/common/tests/models \
                    models/common/tests/llm_runtime
```

## Deliverables

### 1. `models/common/models/MILESTONE_B_STATUS.md`

Modelled on `models/common/modules/MILESTONE_A_STATUS.md`, and honest in the same way. It needs:

- a **Current position** table: what is qualified, what is not, and the exit-gate verdict stated
  plainly in the first screen — not buried after the evidence;
- a **Verification status** table, one row per area (Llama block, Llama full model, Qwen block, Qwen
  full model, paged KV, concat-32, prefix cache, device sampling, long context, repeat/cleanup), each
  with its host evidence, its device evidence, and a status that distinguishes *qualified* from
  *passed once*;
- a **Defects found** table in the D1–D5 form — defect, how it hid, fix. "How it hid" is the column
  that teaches something; write it properly;
- **Known limitations, documented and accepted**, each with an anchor and a target milestone, in the
  L1/L2/L3 style;
- **Pending work**, split into blocking and deferrable, each deferrable item naming the milestone it
  belongs to;
- the **exit-gate result table**, requirement by requirement;
- the **modularity scorecard** from the plan: new files added; existing shared files changed and why
  config alone was insufficient; 1D module implementation files changed (required: zero); default
  runtime behaviours changed (required: zero); 1D regression suites run and their result; topology
  assumptions discovered in common code; whether the extension stayed inside module/config/model
  boundaries.

The plan is explicit that the scorecard is project evidence in its own right: *"Passing model tests
while violating these boundaries does not count as a successful TTTv2 extension."* If the boundaries
held, show it. If they did not, say so — that is a finding, not a failure of this job.

### 2. Documentation updates

- `models/common/modules/README.md` — the 2D inventory, the final module contracts as they now stand
  after job 0's amendments, and the Galaxy model packages. Remove any line still saying Milestone A is
  in progress if the Milestone A branch has since closed it; if it has not, leave it and note the
  dependency.
- `models/common/modules/MILESTONE_A_STATUS.md` — **only** the items Milestone B closed or changed:
  the L3 verdict from `mb-llama`, and any Milestone A limitation Milestone B resolved or proved
  worse. Surgical edits, not a rewrite — it is the signed-off Milestone A record, and you are
  appending to it rather than restructuring it.
- A final checkpoint in `tttv2_2d_modules_milestone_b_work_log.md`.

### 3. `tttv2_milestone_c_brief.md`

A short, honest handoff into Milestone C — executors, runtime integration, tracing and vLLM. It should
carry:

- what Milestone C inherits as working, with the commands that prove it;
- what it inherits as broken or unqualified, with the evidence;
- the items already routed to it by name: **L1** (`Prefetcher2D` global-CB ownership redesign),
  **D-A** (physical-32 real-device trace, which needs a model-owned executor and so genuinely could
  not be done before now), and the **Galaxy CCL / `tt_ccl.py` merge evaluation** the plan defers until
  both models pass;
- the performance-methodology requirements Milestone C will be measured against, so it can set up
  paired TTTv1/TTTv2 measurement from the start rather than retrofitting it: same host, same commit
  and firmware, same checkpoint, precision recipe, prompt corpus, batch, sequence, trace, sampling and
  KV setup; one unmeasured warmup; three measured runs; compare medians; retain profiler artifacts and
  exact commands.

## Finish condition

`MILESTONE_B_STATUS.md` exists and states a verdict, the documentation updates are made, the Milestone
C brief is written, and every claim in all three traces to a log. Print the absolute path of
`MILESTONE_B_STATUS.md` as your final line.

If the honest verdict is that Milestone B does not pass, say so in the first paragraph and list
exactly what remains. Do not begin Milestone C work; the plan gates it, and the gate is the point.
