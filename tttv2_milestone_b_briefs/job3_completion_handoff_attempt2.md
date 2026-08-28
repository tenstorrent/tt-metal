# Job 3 (`mb-coverage`) attempt 2 → attempt 3: interrupted-run handoff

**Status: `INTERRUPTED`. Attempt 2 did not declare its finish condition. There is
no `state/mb-coverage.finished`, and the job wrote no handoff of its own.**

## What this document is, and who wrote it

This file was assembled **by hand on 2026-08-28 by an interactive session, not by
the `mb-coverage` job**. It exists for one reason: `latest_handoff` resolves to
the newest attempt on disk, and without this file the next attempt would inherit
`job3_completion_handoff.md` (attempt 1, 2026-08-27 03:31), which predates
everything below and still asserts a dead mesh. That document is wrong about the
mesh, wrong about Qwen's weights, and wrong about F-C1. Do not plan from it.

The job was killed mid-run when the host reservation expired at roughly 03:48
UTC. The kill took the whole process tree, so the driver never wrote its exit
line — `tttv2_milestone_b_runs/20260827T231746Z/driver.log` simply stops at
`23:18:17Z`. Nothing failed; the machine went away.

Everything under **"The job's own account"** below is verbatim from the fragments
the job wrote before it was cut off (`A2_HANDOFF_HEAD.md`, `A2_HANDOFF_BODY.md`,
`A2_HANDOFF_TAIL.md` in `tttv2_milestone_b_evidence/coverage/`). Only
`A2_HANDOFF_HEAD.md`'s own title line was dropped, to keep one `#` heading in
this file. No wording was changed. Read the four corrections in the next section
first, because the fragments contain forward references that were never
satisfied.

## Four corrections to the fragments below

1. **`REPORT.md` §A2 does not exist.** The fragments open with *"Full account:
   `tttv2_milestone_b_evidence/coverage/REPORT.md` §A2"*. That section was never
   appended: `cov_assemble_report.sh` never ran, and `REPORT.md` is still attempt
   1's file from 2026-08-27 03:29. Every fragment it would have been assembled
   from is present and committed (`A2_SECTION_HEAD`, `A2_GATE_TABLE`,
   `A2_METHOD`, `A2_AREA_MAP`, `A2_AREAS`, `A2_L1`, `A2_FINDINGS`,
   `A2_GATE_COMMANDS`, `A2_CLOSE`). **Read `RESULTS_A2.md` instead** — it is one
   row per run, written as each run finished, and it is the only complete account
   of attempt 2 that exists.

2. **The findings table in `A2_HANDOFF_TAIL.md` stops at D-C4 and is out of
   date.** Two further findings were measured afterwards and exist **only** as
   inline rows in `RESULTS_A2.md`, with no section in `A2_FINDINGS.md` and no row
   in that table:
   - **D-C5** (`a2_g23_qwen_demo_sampling.log`) — Qwen device sampling:
     `MatmulMultiCoreProgramConfig: Input B memory layout must be INTERLEAVED,
     got: TensorMemoryLayout::WIDTH_SHARDED`, at `collectives.py:445` in
     `GalaxyColumnUserSelector.__call__`, reached from `model.sample_decode`. The
     host-sampling half of the test ran first and passed.
   - **D-C6** (`a2_g22_qwen_demo_concat32.log`) — Qwen concat-32 prefill:
     `circular buffers on core range [0-0 - 2-3] grow to 1669312 B, beyond max L1
     size of 1499136 B`, from `validate_circular_buffer_region` while compiling
     the concatenated prefill program (`direct_runner.py:484`). An L1 **capacity**
     overflow, which is not the address clash Llama hits.

3. **`A2_FINDINGS.md` contains two unresolved placeholders.** Its L1 table has
   literal `@@QWEN_CONCAT@@` and `@@QWEN_SAMPLING@@` cells, left to be filled
   when `a2_g22` and `a2_g23` finished. They did finish — both **FAILED**, as
   D-C6 and D-C5 above. That matters for the section's thesis, *"L1's remaining
   half is Llama-specific at this tree, not universal"*, which was written at
   02:49 against the first two table rows only. Two of its four Qwen cells are
   now failures with different root causes than Llama's. **The claim needs
   re-reading against its own completed table before it goes to the scorecard** —
   it may narrow to "the *address clash* is Llama-specific, and Qwen fails the
   same two shapes for two unrelated reasons", which is a different finding.

4. **`mb-qwen`'s finish timestamp** is `2026-08-27T23:15:02Z` (`state/mb-qwen.finished`,
   commit `b1e824537a4`), not the `22:51 UTC` cited in the fragments — that was
   its last commit, not its exit.

## Where attempt 2 stopped

| | |
| --- | --- |
| launched | `23:17:46Z`, run dir `tttv2_milestone_b_runs/20260827T231746Z` |
| preflight | 32/32 boards on the bus, mesh probe OK, device free |
| agent started | `23:18:17Z`, 12 h timeout, so it was nowhere near it |
| last device verdict | `a2_L1_qwen_batch32_run3 rc=0` at `03:42:26Z` |
| interrupted during | `a2_L1_llama_repeat_run3`, dequeued `03:42:26Z` |
| killed | ~`03:48Z` — reservation expiry, ~4.5 h of a 12 h slot |

Evidence on disk: **51 logs** in `tttv2_milestone_b_evidence/coverage/logs2/`,
**38 rows** in `RESULTS_A2.md`. Through the queue harness alone: **33 verdicts —
24 `rc=0`, 8 `rc=1`, 1 deliberate stop** (`a2_03`, `rc=143`, stopped because
D-C4 had made the case a tautology). The eight failures are `a2_02`, `a2_g6`,
`a2_g7`, `a2_g10`, `a2_g11`, `a2_g22`, `a2_g23`, `a2_L1_llama_repeat_run2` — all
diagnosed in `RESULTS_A2.md`, none of them unexplained.

## How to resume without paying twice

**`queue.txt` is the resume point.** It holds **40 pending items** and is
consumed by `cov_queue.sh`; `a2_L1_llama_repeat_run3` was in flight when the host
went away and is not in it. Nothing in `queue.txt` has been run.

`RESULTS_A2.md` was deliberately written one row at a time, as each run finished,
*"so it survives a timeout"*. It did. **Re-running what it already records is the
one way to waste this slot** — those 33 runs cost 4.5 hours of Galaxy time.

Outstanding deliverables, in the order they unblock each other:

1. Fill the two `@@…@@` placeholders in `A2_FINDINGS.md` and add D-C5 and D-C6 as
   proper sections, then revisit the L1 thesis per correction 3 above.
2. Check every row of `A2_GATE_TABLE.md`: it is a `00:07Z` draft, written before
   most of the night's results existed. The brief requires **measured** values,
   not quoted ones.
3. Run `cov_assemble_report.sh` to append §A2 to `REPORT.md` (it is idempotent
   and refuses if the marker is already present).
4. Work through `queue.txt`.
5. Append the checkpoint to `tttv2_2d_modules_milestone_b_work_log.md`.
6. **Replace this file** with the job's own `job3_completion_handoff_attempt3.md`.
   This one is a hand-written bridge, not evidence.

Relaunch with:

```sh
./run_milestone_b_jobs.sh --jobs mb-coverage --attempts 2
```

Not `--resume --attempts 1`: `state/mb-coverage.done` is stale — it says
`20260826T161958Z`, the dead-mesh run — and under a single pass that marker would
skip the job entirely. Under `--attempts 2` a `.done` marker does not skip; only
`.finished` does, and mb-coverage has none.

---

# The job's own account

*Verbatim from `A2_HANDOFF_HEAD.md`, `A2_HANDOFF_BODY.md` and
`A2_HANDOFF_TAIL.md`. Corrected above, not edited here.*

Written 2026-08-27/28 by `mb-coverage` **attempt 2**, unattended, on
`apbernal/tttv2_wh_glx_2d_modules_milestone_b`.

Full account: `tttv2_milestone_b_evidence/coverage/REPORT.md` §A2.
Run-by-run index: `tttv2_milestone_b_evidence/coverage/RESULTS_A2.md`.
Machine and mesh facts, costs, the exact harness:
`tttv2_milestone_b_evidence/coverage/ENVIRONMENT.md`.

## Read this paragraph first

**Do not plan around `job3_completion_handoff.md` (attempt 1).** Its headline —
*"the mesh never came back … three consecutive device jobs have produced zero
numerical results from silicon, for either model"* — was true when written at
03:31 UTC and is false now. The mesh was repaired, `mb-qwen` attempt 2 then
qualified **both** models end to end on silicon (17:53–22:51 UTC), and attempt 2
of this job measured step 7 on a live 8×4 mesh. Attempt 1's host analysis is
still good and is still worth reading; its verdict is not.

Three of its statements are simply wrong at this tree, and one of them changes a
gate:

* the mesh is alive (`ls /sys/class/tenstorrent | wc -l` = 32, a real cluster
  opens in 12 s);
* Qwen's weights are on this machine, under
  `HF_HOME=/localdev/ctr-apbernal/hf_data`;
* **Llama does pad its vocabulary** — by 768 ids, `galaxy_padded_vocab_size(128256)
  = 129024`. Attempt 1's finding F-C1 says the opposite and calls Llama's
  padded-vocab gate vacuous. It is live, and attempt 2 added the device case.


## The environment, in four lines

```sh
export HF_HOME=/localdev/ctr-apbernal/hf_data     # reaches BOTH checkpoints; inherited value is empty
ls /sys/class/tenstorrent | wc -l                 # 32. NOT /dev/tenstorrent - those nodes persist after a board falls off
python -u -m pytest -v -rA --color=no -p no:cacheprovider --timeout=240 \
  models/common/tests/models/galaxy/test_partition_wh_galaxy.py    # 12 s, opens a real cluster
```

A `SKIPPED` in a run you meant to count is a failed run, not a green one:
`hf_config_or_skip` skips silently under a wrong `HF_HOME`.

## The cost model you need to schedule against

| Thing | Wall clock |
| --- | --- |
| mesh open, 32 devices | ~25 s |
| Llama 80-layer build, warm device weight cache | **~5.5 min** |
| Llama 80-layer build, *cold* weight set | **~26 min, and 138 GB of disk** |
| Llama teacher-forced 512/511 decode, after the build | ~18 min |

Every test in the step-7 files builds its own model and several build two, so a
17-node-id file is a three-hour run. That single fact is why attempt 2 ran
subsets and says, per row, how many fresh processes each claim got. Plan a night
around builds, not around tests.

## Two things not to spend a run on

* **`Prefetcher2DConfig.release_global_cb_on_prefill`** — `mb-llama` attempt 3
  implemented it and refuted it on hardware. Dropping the last Python reference
  to a `global_circular_buffer` does not return its L1; there is no `deallocate`
  on the type.
* **A `memory_config`-only fix for D-C1** — both the prefill and the decode page
  table are DRAM-*interleaved* on device (measured, `a2_01b`), so
  `is_sharded()` is false for both and cannot separate them by itself.


## Findings you need for the modularity scorecard

Attempt 1's seven stand except F-C1. Attempt 2 adds two and corrects one.

| ID | Severity | Where | What |
| --- | --- | --- | --- |
| **D-C1** | correctness | `attention_2d._validate_decode_page_table` | A prefill-shaped page table fed to decode is accepted. **Premise now confirmed on silicon**: the decode table's device-local view is 8 rows, the prefill table's is 32, `32 % 8 == 0`, and both are DRAM-interleaved. Unchanged verdict; the fix needs a 2D-module expectation changed, so it needs a decision |
| **D-C2** | contract conflict | `sampling_2d._seed_digest` | Moving a seeded request to another slot changes its stream. Product decision: is a seed per-request or per-(request, slot)? |
| **D-C3** | test-infra, expensive | `modules/lazy_weight.py` | The weight-cache fingerprint contains `MeshDevice.id()`, so every test after the first in one pytest process re-stages **every** weight: 965 tensors, 138 GB, 26 min for Llama. One node id per process is mandatory on this stack. One-line fix (fingerprint the mesh *shape*), outside this job's mandate |
| **D-C4** | contract gap | both `hf_adaptor.from_pretrained` | `paged_attention_config=None` installs the *default* pool, not a contiguous cache, so area 1's "PCC vs the contiguous path" gate is not expressible through the adaptor — and the committed test for it was comparing a pool against itself |
| **G-C1** | limitation | `direct_runner.prefill_batched` | Concat-32 needs all 32 slots active; it cannot combine with the `active_slots < 32` sink-block mechanism |
| **G-C2** | minor | `direct_runner.prefill_batched` | An empty row is rejected one call too late |
| **G-C3** | dead code | `attention_2d._validate_prefill` | An unreachable guard |
| **F-C1** | **superseded** | `recipes.galaxy_padded_vocab_size` | Attempt 1: "Llama has no vocabulary padding, its padded-vocab gate is vacuous". **False.** Llama pads 768 ids (129024), Qwen 1664 (153600). The gate is live for both and now has a device case for both |
| **F-C2** | test-infra | `tests/models/galaxy/test_plans.py` | Looks host-only, needs a cluster. On a live mesh its 13 failures should disappear — worth re-checking now that one exists |

## Suggested order for your night

1. Read `RESULTS_A2.md` first, not the report: it is one row per run with the log
   name, written as each finished, and it says how many fresh processes each
   claim got.
2. Take the exit-gate table from `REPORT.md` §A2. Every row has its command.
3. The verdict is **not** "infrastructure-blocked" any more. Say which lines pass,
   which fail, and which are *not expressible* — D-C4 makes one gate line
   unmeasurable at this API, which is a different thing from unmeasured.
4. D-C1, D-C2 and D-C4 are the three that should reach a human. D-C3 should reach
   whoever owns `lazy_weight.py`.
