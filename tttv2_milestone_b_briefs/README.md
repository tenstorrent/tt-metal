# Milestone B — reconciliation and completion briefs

Five unattended Claude Code jobs that reconcile the diverged Milestone A and Milestone B trees and
then finish Milestone B end to end. Driven by `run_milestone_b_jobs.sh`.

| Job | Brief | Device | Typical | What it delivers |
| --- | --- | --- | --- | --- |
| `bootstrap` | *(none — script only)* | no | minutes | Milestone B branch checked out, build + venv verified |
| `reconcile` | `job0_reconcile.md` | **no** | 4–6 h | Milestone B rebased on the final Milestone A tree, C1–C10 closed, host suites green |
| `mb-llama` | `job1_llama.md` | yes | 10–12 h | Plan steps 1–3: Llama adaptor, one-layer PCC, 80-layer model, teacher-forced accuracy |
| `mb-qwen` | `job2_qwen.md` | yes | 10–12 h | Plan steps 4–6: the same for Qwen3-32B, 64-head geometry |
| `mb-coverage` | `job3_coverage.md` | yes | 10–12 h | Plan step 7: paged KV, prefix cache, concat-32, device sampling, long context |
| `mb-signoff` | `job4_signoff.md` | no | 2–3 h | Exit-gate verdict, modularity scorecard, `MILESTONE_B_STATUS.md` |

The three `mb-*` jobs each need the Galaxy exclusively and are strictly sequential; one per night is
the realistic cadence. `reconcile` and `mb-signoff` need no device at all — do not take the mesh for
them.

## Where this work happens

**In place, in the working checkout:**

```text
/proj_sw/user_dev/ctr-apbernal/tt-metal     branch apbernal/tttv2_wh_glx_2d_modules_milestone_b
```

Milestone A is finished, so this tree is ours. `bootstrap` checks out the Milestone B branch here and
reuses the existing `build/` and `python_env/` rather than rebuilding; it refuses to switch branches
if there are uncommitted *tracked* changes, and records the previous branch in the run directory so
the switch is reversible by hand. Untracked files — evidence directories, these briefs, run logs —
survive the switch untouched.

Every later job runs on that branch. None of them may check out a different one.

## Shared context every agent must read

1. `tttv2_2d_modules_plan.md` — "Milestone B", "Milestone B tests", "Milestone B exit gate", the
   per-module contracts, and "Authoritative Design Constraints".
2. `tttv2_milestone_ab_reconciliation.md` — the static divergence analysis. Findings C1–C10 and the
   phased plan. **Its findings were derived by reading diffs, not by running anything.** Re-verify
   before acting on any one of them; the Milestone A branch has moved since it was written.
3. `models/common/modules/MILESTONE_A_STATUS.md` — defects D1–D4, limitations L1–L3, and the reason
   this project distrusts a single passing run.
4. `tttv2_2d_modules_milestone_b_work_log.md` and `tttv2_2d_modules_milestone_b_handoff.md` — what
   the Milestone B code already is, and its author's own ranked list of what is unproven.
5. `models/common/modules/README.md` and `models/common/llm_runtime/README.md` — the module and
   runtime contracts.

## House rules, common to all five

- **One pytest process on the device at a time.** Never pipe pytest; a pipeline can hand back control
  while the nested process still holds the mesh.
- **Three runs in fresh processes before any device claim.** Three of Milestone A's four defects
  presented as intermittent *passes*, not failures, because they read aliased or uninitialised L1. A
  case that flips across processes is a defect, not noise.
- **A failing test is a result, not a bug to patch.** Never relax a threshold, tolerance, or
  parametrization to turn a failure green. Never delete or `xfail` a test to get past it.
- **Never fabricate.** An honest `BLOCKED` with logs beats an invented pass. If you did not run it,
  say you did not run it.
- **Zero changes to `models/common/modules/**/*_1d.py`.** Sharing a *test* helper across the 1D and
  2D suites is fine and has precedent (`models/common/tests/modules/_hf_reference.py`).
- **Zero changes to `models/common/llm_runtime/**`.** Milestone B was verified to import none of it.
  If you believe you need a runtime change, stop and write the reduction the plan's "Extension
  discipline" section requires; do not just make the change.
- **No imports from an existing model-named package** — not `models/demos/llama3_70b_galaxy`, not
  `models/common/models/llama33_70b`, not `models/common/models/qwen3_32b`. They are behavioural
  references you may *read*, never dependencies you may import.
- **Git**: commit on `apbernal/tttv2_wh_glx_2d_modules_milestone_b`. Never `push`, never `checkout`
  another branch, never `reset --hard` or `stash` work you did not create in this session. The
  Milestone A branch (`gongyu/tttv2_wh_glx_2d_modules`) is finished and is a **read-only reference**:
  diff against it, cherry-pick from it, never write to it.
- **Do not edit `models/common/modules/MILESTONE_A_STATUS.md` or `tttv2_2d_modules_plan.md`** unless
  your brief names them explicitly. `job0` and `job4` do; the others do not — put proposed text in
  your report instead.
- Every job appends a terse checkpoint to `tttv2_2d_modules_milestone_b_work_log.md` and writes a
  completion handoff the next job reads.

## Do not kill your own process tree

You run inside `timeout … claude -p`, launched by `run_milestone_b_jobs.sh`. The driver's PID is in
`$MB_JOB_DRIVER_PID`. A previous run in this project read its own wrapper in `pgrep -af pytest`,
mistook it for a stuck test, and killed itself 16 minutes in.

Before signalling any PID:

```sh
ps -o pid=,ppid=,comm=,args= -p <pid>       # comm must be python/python3/pytest
```

Never signal a PID whose `comm` is `claude`, `timeout`, `bash`, or `screen`. Prefer targeting the
exact file: `pkill -f 'python.*pytest.*<the test file you launched>'`, confirmed with `pgrep` first.

## Device run procedure

```sh
ls /dev/tenstorrent | wc -l                 # must be 32
tt-smi -ls
pgrep -af 'pytest|ttnn' | grep -v grep      # must be empty (ignore the claude/timeout wrapper)

timeout --signal=TERM --kill-after=180 2700 \
  python -m pytest -v -rA --color=no -p no:cacheprovider <FILE-OR-NODEID> > "$LOG" 2>&1
echo "exit=$?" >> "$LOG"
```

The harness caps a foreground tool call at 600 s. Anything longer goes out as a tracked background
process that you block on before starting the next one. Re-check `pgrep` between runs.

On a hang or crash: kill the tree, confirm the device is free, `tt-smi -glx_reset`, confirm
`Re-initialized 32 boards`, retry. **Maximum 2 recovery attempts**, then record `BLOCKED (infra)`
with logs and move on. Before spending a recovery attempt on a hang, capture a traceback — a
repeating `faulthandler.dump_traceback_later` pytest plugin (diagnostic only, never committed) located
the D3 stall in two dumps 90 s apart. A hang here is most likely a subdevice/semaphore ownership
fault, exactly like D3.

A `TT_FATAL` abort inside a multi-subdevice program leaves the mesh un-drainable — teardown blocks in
`FDMeshCommandQueue::~FDMeshCommandQueue`. Budget a kill and a `tt-smi -glx_reset` after any such
abort.

Keep every log from every attempt. Never overwrite a log.
