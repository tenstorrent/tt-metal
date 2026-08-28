# `mb-coverage` — environment (attempt 2)

Recorded 2026-08-27/28, unattended, by the Milestone B step-7 coverage job,
**attempt 2**. Attempt 1's version of this file is preserved verbatim beside it
as `ENVIRONMENT_attempt1.md`; where the two disagree, this one is later and was
measured on a live mesh.

## The correction that matters

Attempt 1 recorded, as its headline, *"the mesh never came back … eleven boards
off the PCIe bus … `ttnn` cannot open a cluster at all"*. **That is not true of
this machine at 2026-08-27 23:21 UTC.** Established before any planning, by the
cheapest real check there is:

```sh
$ ls /sys/class/tenstorrent | wc -l
32
$ python -u -m pytest -v -rA --color=no -p no:cacheprovider --timeout=240 \
    models/common/tests/models/galaxy/test_partition_wh_galaxy.py
5 passed in 12.32s          # opens a real 8x4 cluster; "multidevice with 32 devices is created"
```

Log: `logs2/a2_00_mesh_health.log`. Every device number in this attempt's
`REPORT.md` was measured after that check.

The mesh was repaired between attempt 1 (03:31 UTC) and `mb-qwen` attempt 2
(which ran 17:53–22:51 UTC and qualified both models on silicon). Attempt 1 was
not wrong about what it saw; it was superseded.

## Tree

| | |
| --- | --- |
| Repository | `/proj_sw/user_dev/ctr-apbernal/tt-metal` |
| Branch | `apbernal/tttv2_wh_glx_2d_modules_milestone_b` |
| Commit at start | `b1e824537a4699003413dfa863db9fa3bb6253ad` (`Re-qualify the step-5 gate and the Q/K norm at this attempt's final commit`) |
| Milestone A base for the boundary greps | `bc6ad03bfc2` |
| Milestone A reference branch | `gongyu/tttv2_wh_glx_2d_modules` — read only, never written |

Attempt 1's two commits are both ancestors of this commit
(`git merge-base --is-ancestor 1cd451cd965 HEAD` → true), so every step-7 test
file it wrote is present in the tree this attempt measured.

## Host and mesh

| | |
| --- | --- |
| Kernel | `6.8.0-83-generic` |
| RAM | 566 GiB total, ~273 GiB available at start |
| `ls /sys/class/tenstorrent \| wc -l` | **32** (the health check; `/dev/tenstorrent` is not one) |
| Mesh | WH Galaxy, `(8, 4)`, exclusive to this job |
| Python env | pre-built `python_env/`, **not** recreated; no rebuild of tt-metal |

## The one environment variable that silently ruins a night

```sh
export HF_HOME=/localdev/ctr-apbernal/hf_data      # reaches BOTH checkpoints
```

Inherited value in this job's shell: **empty**. Under an empty or wrong value
`hf_config_or_skip` turns every real-checkpoint test into a `SKIPPED` and the run
looks green having measured nothing. Every script in this directory exports it.
Confirmed present under it: `models--meta-llama--Llama-3.3-70B-Instruct` and
`models--Qwen--Qwen3-32B`.

## Reference token files

`models/tt_transformers/tests/reference_outputs/{Llama-3.3-70B-Instruct,Qwen3-32B}.refpt`

Both hold **1024** tokens (`reference_tokens` shaped `(1, 1024)`, `top5_tokens`
`(1024, 5)` and `(1023, 5)`). This is a hard limit on two things measured here:
the 512/511 accuracy gate fits exactly, and `_distinct_rows(length, 32)` — which
needs `length + 32` tokens — cannot produce 32 distinct rows at length 1024 or
2048 from this file. See `REPORT.md` §A2 area 2.

## Run harness

`cov_run3.sh` → `cov_device_run.sh` → `cov_after_device_run.sh`, driven by
`cov_seq2.sh` over a pipe-delimited manifest. Copied from
`../qwen/`, with one deliberate change, recorded because it silently destroys
multi-test runs:

> the qwen `device_run.sh` arms its teardown-grace timer on the **first per-test
> verdict** in the log. In a one-node-id run that is the last thing before
> teardown; in a whole-file run it fires 90 s after test 1 passes and reaps a
> healthy process mid-test-2. `cov_device_run.sh` arms on the pytest **session
> summary** instead, and adds an idle-mtime trigger (no log write for
> `MB_IDLE`=900 s with a verdict already in hand) for the hang case attempt 3 of
> `mb-llama` described.

Never pipes pytest. Signals only a PID whose `comm` is python/python3/pytest.
Resets the mesh (`tt-smi -glx_reset`, 900 s cap) after any non-clean run.

## Costs measured here, for whoever schedules the next night

| Thing | Wall clock |
| --- | --- |
| mesh open, 32 devices | ~25 s |
| Llama 80-layer model build from the warm device weight cache | **~5.5 min** |
| Qwen 64-layer model build from the warm device weight cache | ~4 min |

Every test in these files builds its own model, and several build two. That
single number is what makes a 17-node-id file a three-hour run and is the reason
this attempt ran subsets rather than whole files where it had to choose.

---

# Attempt 3's additions

Recorded 2026-08-28 by **attempt 3**, unattended. Nothing above is edited; this
section adds what changed and what attempt 3 measured for itself.

## Tree

| | |
| --- | --- |
| Commit at start **and throughout** | `af589dff4d509b7afa3ea7b5ee41995c2e2761ad` |
| Run directory | `tttv2_milestone_b_runs/20260828T073724Z`, 12 h slot from `07:37:58Z` |
| Attempt 2's device commits | `1451b192584` (runs 01–g1) and `718997518ab` (runs g2 onward) |

**The fact that makes attempt 2's evidence usable at this tree**, established
before anything was scheduled:

```sh
git diff --stat 718997518ab..HEAD -- models/    # EMPTY
git diff --stat 1451b192584..HEAD -- models/    # only the two test_step7_coverage_wh_galaxy.py files
```

So attempt 2's 27 logs stamped `718997518ab` were produced against source
byte-identical to `HEAD` under `models/`, and its four earlier logs differ only in
a test file that `test_full_model_wh_galaxy.py` does not import. That is not the
same thing as quoting an earlier job's number — it is a measurement whose tree has
been proved identical — and every gate row in §A3 says which of the two it is.

## Mesh, at 07:38Z

```sh
$ ls /sys/class/tenstorrent | wc -l ; ls /dev/tenstorrent | wc -l
32
32
$ pgrep -af 'pytest|ttnn' | grep -v grep     # empty
$ python -u -m pytest ... models/common/tests/models/galaxy/test_partition_wh_galaxy.py
5 passed in 13.66s                            # logs2/a3_00_mesh_health.log
```

Attempt 2's mesh finding holds at this commit. `HF_HOME` was again inherited
**empty** and again exported to `/localdev/ctr-apbernal/hf_data` by every script.

## Disk

`/proj_sw` began this attempt with **1158 GiB free of 29803 (97% used)** — the
queue runner's own guard prunes only `model_cache/*.tensorbin` files this job
created, and halts rather than continue below 150 GiB. `/localdev` had 1206 GiB.
Both weight sets were already staged, so no cold 138 GB stage was paid.

## Harness

Attempt 2's `cov_run3.sh` → `cov_device_run.sh` → `cov_after_device_run.sh` chain
unchanged, driven by `cov_queue.sh` over `queue.txt` (one node id per line, one
process per line — see D-C3). Attempt 3 added one read-only script:

* `cov_watch3.sh` — polls `logs2/queue.out` and, for each finished item, re-reads
  the log itself and appends its own extracted pytest summary line to
  `VERDICTS_A3.txt`. It exists so that no verdict in this report comes from a
  human-written index: `RESULTS_A2.md` was one row short of the truth and the
  attempt-2 bridge inherited that gap. It never signals anything.

## One operational note, for the next unattended job on this box

`pkill -f <pattern>` matches **this agent's own tool-call wrapper shell**, whose
command line contains the pattern verbatim. Running `pkill -f cov_watch3.sh` to
restart a helper killed the calling shell mid-command (exit 144) rather than the
helper. The house rules already forbid killing your own tree; the specific trap is
that `-f` reads the wrapper's `eval '…'` string. Target a PID confirmed with
`ps -o comm=` instead, as `cov_ensure_mesh_free.sh` does.

## Attempt 3's second agent invocation, `08:16:43Z`

The first invocation ended at `08:16:43Z` and `run_milestone_b_jobs.sh` (PID
10669, `--no-screen --jobs mb-coverage --attempts 2`) relaunched immediately. The
second invocation re-verified the environment before touching anything, because
its brief says to:

```sh
$ ls /sys/class/tenstorrent | wc -l ; ls /dev/tenstorrent | wc -l
32
32
$ echo "[$HF_HOME]"
[]                                       # still inherited empty
$ df -h /proj_sw /localdev | tail -2
aus-wekacluster/proj_sw   30T   28T  1.2T  97% /proj_sw
/dev/md1                 7.0T  5.8T  1.2T  84% /localdev
```

**The device queue survived the relaunch and this matters more than the numbers
above.** `cov_queue.sh` had been started detached at `07:43:08Z` and was
reparented to init (PPID 1) when its parent agent went away. At `08:16:44Z` — one
second after the relaunch — it dequeued `a3_l_greedy` and ran it to completion.
So:

* the mesh was continuously occupied across an agent boundary, by design rather
  than by luck. Attempt 2 lost the tail of its night to a kill that took its whole
  process tree; a detached serial queue is what makes that survivable;
* the second invocation's **first** obligation was therefore *not* to start
  anything. `pgrep -af 'pytest|ttnn'` showed PID 28892 holding the mesh. It was
  confirmed with `ps -o pid=,ppid=,comm=,args=` (comm `python`, args the exact
  node id in `queue.out`), identified as the inherited queue's own work rather
  than a stuck process, and **left alone**. Killing it would have cost the Llama
  build already 16 minutes in — and it produced `a3_l_greedy`, the log that closed
  D-C5 for Llama.

Two mechanical consequences of the boundary, both recorded rather than tidied
away:

1. `cov_queue.sh`'s environment is fixed at its own launch, so an env var the
   second invocation exports cannot reach a queued run. Anything a new test needs
   must have a working default — which is why the cross-process pool artifacts
   default to `$TMPDIR/tttv2_step7_artifacts` and take `STEP7_ARTIFACT_DIR` only
   as an override;
2. `a3_q_pool_default` was dequeued at `08:33:24`, 90 s before its own source was
   committed as `6df3c4a14a3`. Its log is stamped `2061c126743`, a commit that does
   not contain the test it ran. The only difference between what ran and what was
   committed is `black` reformatting (the pre-commit run reports `black … Failed —
   files were modified`, every other hook `Passed`), and the run was repeated from
   the committed tree as `a3_q_pool_default_run2` rather than argued about.
