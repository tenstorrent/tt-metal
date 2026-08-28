
---

# §A3 — attempt 3, the completing pass

Written 2026-08-28 by `mb-coverage` **attempt 3**, unattended, on
`apbernal/tttv2_wh_glx_2d_modules_milestone_b`. Run directory
`tttv2_milestone_b_runs/20260828T073724Z`.

**Attempt 3 ran as two agent invocations inside one driver run, and this section
is written by the second.** The first started at `07:37:58Z` at commit
`af589dff4d5`, committed `2061c126743` at `07:59Z`, and ended at `08:16:43Z`; the
driver relaunched immediately. The device queue it had started
(`cov_queue.sh`, PID 13308, reparented to init) **never stopped** — it dequeued
`a3_l_greedy` at `08:16:44`, one second after the relaunch — so the mesh was never
idle and no run was lost or repeated. The second invocation adopted that queue
rather than restarting it, re-prioritised `queue.txt` under it, and added commits
`6df3c4a14a3` and `152d4c49efb`. Every run below names the commit its log is
stamped with, and the two runs whose stamp does not match their source say so
explicitly.

Everything above this line is attempts 1 and 2 and is left untouched, except for
the `@@…@@` cells in §A2 that attempt 2 was cut off before it could fill: those
were resolved from the logs they were waiting on, and §A2 says so. **This section
is the final verdict and supersedes both where they disagree.**

## What attempt 3 inherited, and what it verified before planning

Attempt 2's handoff was a hand-written bridge, not the job's own account, so the
first thing this attempt did was check its claims against the tree.

| Inherited claim | Verified at `af589dff4d5` | How |
| --- | --- | --- |
| The mesh is alive | **True.** `ls /sys/class/tenstorrent \| wc -l` = 32, `/dev/tenstorrent` = 32, and `test_partition_wh_galaxy.py` opened a real 8×4 cluster: **5 passed in 13.66s** | `logs2/a3_00_mesh_health.log` |
| `HF_HOME` must be exported as `/localdev/ctr-apbernal/hf_data`; the inherited value is empty | **True.** `echo "[$HF_HOME]"` → `[]` in this job's own environment. Every harness script exports it | `cov_run3.sh`, `cov_device_run.sh` |
| 51 logs and 33 machine verdicts from attempt 2 are on disk | **True**, and re-derived rather than trusted: a watcher re-read every `logs2/*.log` and extracted its own pytest summary line. 38 rows agree with `RESULTS_A2.md` | `VERDICTS_A3.txt` |
| `queue.txt` holds 40 pending items, none of them run | **True** for 38 of them. **Two were already done**: `a2_L1_qwen_repeat_run3` (`1 passed in 458.26s`, `exit=0`) and `a2_L1_qwen_batch32_run3` (`1 passed in 175.50s`, `exit=0`) both completed at 03:38 and 03:42 and were never written down anywhere. Attempt 3 dropped them from its queue rather than pay for a fourth run | `logs2/a2_L1_qwen_*_run3.log` |
| `a2_L1_llama_repeat_run3` was in flight when the host went away | **True.** Its log stops inside `Loading weights: 100%` with no verdict and no `exit=` line | `logs2/a2_L1_llama_repeat_run3.log` |

**One inherited claim was wrong and it is worth naming**, because it is the kind
of error that costs a night: the bridge said the two Qwen run-3 verdicts did not
exist. They did. `RESULTS_A2.md` — written one row at a time precisely so it
would survive a kill — stops one row before the end, and the bridge was written
from `RESULTS_A2.md`. The logs are the record; the index of the logs is not.

## The one thing that makes attempt 2's numbers usable

The brief says: *re-measure the accuracy numbers at this tree, do not quote
them*, because "evidence collected at a tree that has since moved is not
evidence". That applies to attempt 2's own numbers as much as to `mb-llama`'s, so
attempt 3 established exactly how far the tree has moved:

```sh
git diff --stat 718997518ab..HEAD -- models/     # empty
git diff --stat 1451b192584..HEAD -- models/     # only the two test_step7_coverage_wh_galaxy.py files
```

* **every attempt-2 log stamped `718997518ab`** — which is all of `g2`…`g23`, the
  `L1_*` re-runs and the placement re-runs — was produced against source
  **byte-identical to `HEAD`** under `models/`. There is nothing to re-measure for
  those rows; they *are* measurements at this tree;
* the four logs stamped `1451b192584` (`a2_01`, `a2_01b`, `a2_02`, `a2_03`) sit two
  commits back, and both commits touched only
  `models/common/tests/models/{llama33_70b_galaxy,qwen3_32b_galaxy}/test_step7_coverage_wh_galaxy.py`
  — a file `test_full_model_wh_galaxy.py` neither imports nor shares a fixture
  with. The Llama accuracy figure in `a2_01` is therefore also unaffected, and
  attempt 3 re-ran it anyway.

This is the difference between *quoting* an earlier number and *inheriting a
measurement whose tree you have proved identical*. Every row in the gate table
below says which of the two it is.
