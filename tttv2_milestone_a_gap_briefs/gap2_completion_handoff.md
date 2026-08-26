# Handoff — Gap 2 completion (`Prefetcher2D` / Galaxy CCL hardware qualification)

The gap 2 job ran on the 6U WH Galaxy on 2026-08-25 for 2 h 52 m and was **killed by the org's
monthly spend limit (HTTP 429), not by a test or hardware failure**. The suite is written, the
shared refactor is validated, and most cases pass; what remains is one untried case, the repeat
runs, one open design decision, and the write-up.

Original brief: [gap2_prefetcher2d_galaxy_ccl_hardware.md](gap2_prefetcher2d_galaxy_ccl_hardware.md).
Read it for the contract and the prohibitions. **Do not restart its work from scratch** — read
"What is already done" first.

Prior sessions (context preserved, resumable by hand if you want the debugging trail):

- **Session 1**, 2026-08-25, `claude --resume 711beb89-db83-4238-b102-266cc969ff6f` — wrote the suite,
  validated the refactor, found F1. Killed by an org monthly spend limit (HTTP 429).
- **Session 2**, 2026-08-26, `claude --resume a6933b37-1765-4b23-bb17-2898d69b3921` — ran the attention
  case for the first time and diagnosed the teardown stall. Killed after 16 min **by itself**: it ran
  `pgrep -af pytest`, which matched its own `timeout … claude -p "<prompt>"` wrapper because the
  driver passed the prompt as an argv element and the prompt contained the word "pytest". It read that
  line as a stuck test process and ran `kill -TERM` on it. The driver now pipes the prompt on stdin so
  the text never reaches a command line, and warns about the pattern. **Nothing about session 2's
  technical findings is in doubt because of how it died** — the failure and the backtrace were both
  captured to disk before the kill.

## What is already done — verified against the tree and the logs

### New files (untracked)

| File | Lines | What it is |
| --- | --- | --- |
| [`models/common/tests/modules/_mlp_2d_galaxy.py`](../models/common/tests/modules/_mlp_2d_galaxy.py) | 468 | The qualified MLP2D geometry, factored out so the prefetcher suite reuses it verbatim |
| [`models/common/tests/modules/prefetcher/test_prefetcher_2d_wh_galaxy.py`](../models/common/tests/modules/prefetcher/test_prefetcher_2d_wh_galaxy.py) | 863 | The qualification suite — 7 tests, 8 cases |

### Shared files refactored (tracked, uncommitted)

- [`_wh_galaxy_hardware.py`](../models/common/tests/modules/_wh_galaxy_hardware.py) — `+71/-23`.
  `_create_hardware_prefetcher` was split: `GALAXY_PREFETCH_SENDER_COORDS`,
  `galaxy_prefetcher_sender_cores()` and `galaxy_prefetcher_config(..., global_cb_size=...)` are now
  public, so a test can build a deliberately undersized configuration. Behaviour-preserving —
  `_create_hardware_prefetcher` now just calls the new config factory with the same values.
- [`test_mlp_2d_wh_galaxy.py`](../models/common/tests/modules/mlp/test_mlp_2d_wh_galaxy.py) —
  `+22/-421`. Gutted into `_mlp_2d_galaxy.py`; it now imports the helpers it used to define.
  **This is a large change to a suite that is recorded Milestone A evidence**; its only validation is
  the `4 passed` device re-run below.

### Evidence in `tttv2_milestone_a_gap2_evidence/logs/`

| Log | Result |
| --- | --- |
| `host01_regression.log` | `78 passed` — prefetcher/galaxy/MLP host suites after the refactor |
| `host02_precommit.log`, `host03_precommit_rerun.log` | pre-commit clean |
| `dev01_mlp_regression.log` | **`4 passed`** — MLP2D device baseline, validates the `_mlp_2d_galaxy` extraction |
| `dev02_rmsnorm_regression.log` | **`8 passed`** — RMSNorm2D device baseline |
| `probe01_seams.log` | scratch probe; ends in `terminate called after throwing an instance of 'std::runtime_error'` (expected — it was probing failure seams) |
| `probe02_global_cb_lifetime.log` | `1 passed` — the probe that isolated finding F1 |
| `dev03_prefetcher_run01.log` | `2 failed, 5 passed, 1 deselected` |
| `dev04_prefetcher_run02.log` | `2 failed, 5 passed, 1 deselected` |
| `dev05_cleanup_nodeids.log` | **`2 passed`** — the two previously-failing cases, isolated, after the fix |
| `dev06_attention_with_prefetch_isolated.log` | **`FAILED`** — `TT_FATAL: Programs must be executed on a single sub-device` (session 2, 2026-08-26) |
| `dev06b_gdb_teardown_backtrace.log` | gdb backtrace of the post-abort teardown stall in `FDMeshCommandQueue::~FDMeshCommandQueue` |

**The two failures were the agent's own test bug**, not a product defect:
`UnboundLocalError: local variable 'prefetcher' referenced before assignment` — it referenced
`prefetcher` after `del prefetcher`. Fixed, and `dev05` shows both cases passing isolated.

### Case-by-case state

| # | Case | State |
| --- | --- | --- |
| 1 | `sealed_resources_are_real_on_device` | PASSED (dev03, dev04) |
| 2 | `mode_transition_matrix` | PASSED (dev03, dev04) |
| 3 | `failed_transition_rolls_back_on_device` | PASSED (dev03, dev04) |
| 4 | `cleanup_from_active_mode_frees_the_mesh[decode]` | test bug fixed → PASSED isolated (dev05) |
| 5 | `cleanup_from_active_mode_frees_the_mesh[prefill]` | test bug fixed → PASSED isolated (dev05) |
| 6 | `context_manager_cleanup_leaves_mesh_reusable` | PASSED (dev03, dev04) |
| 7 | `registration_and_sealing_rejections` | PASSED (dev03, dev04) |
| 8 | `attention_decode_with_active_prefetch` | **FAILED** on its first execution, 2026-08-26 — `TT_FATAL: Programs must be executed on a single sub-device`. See item 2 |

The transition matrix at
[`test_prefetcher_2d_wh_galaxy.py:70`](../models/common/tests/modules/prefetcher/test_prefetcher_2d_wh_galaxy.py#L70)
is `("decode","prefill","decode","prefill","decode") * 2 + ("prefill","prefill")`, which covers the
plan's full list including the repeated-decode seam between cycles and the repeated-prefill tail.

## Finding F1 — real, documented, and needs a decision you must make

`Prefetcher2D.cleanup()` **never frees the global circular buffer.** ttnn exposes no free for one, so
its L1 is released by RAII when the last `global_circular_buffer` handle dies — and every module
handed a `Prefetcher2DContext` holds one (`MLP2D` keeps it as `decode_prefetch_context`). A second
owner's `create_global_circular_buffer` on the same mesh therefore dies with:

```
TT_FATAL: Out of Memory: Not enough space to allocate 55444480 B L1 buffer across 70 banks,
where each bank needs to store 792064 B, but bank size is 1393472 B
(allocated: 792256 B, free: 601216 B, largest free block: 601216 B)
```

Isolated cleanly in `probe02`: holding **only** the global CB object reproduces it; holding only the
address metadata does not. This is exactly the ownership defect the gap existed to find, and it is
invisible to the mock suite.

The prior session worked around it **in the test** — the leak detector drops consumer handles and
`gc.collect()`s before building the second owner, with the reasoning recorded in a comment at
[`test_prefetcher_2d_wh_galaxy.py:559-569`](../models/common/tests/modules/prefetcher/test_prefetcher_2d_wh_galaxy.py#L559-L569).

**That workaround is defensible for the test but is not obviously the right end state, and nobody has
ruled on it.** Decide and argue it in the report:

- Is "cleanup does not release the global CB; consumers must drop their contexts first" an intended
  part of the `Prefetcher2D` ownership contract? If so it belongs in the `Prefetcher2D` docstring and
  in `modules/README.md`'s prefetcher-ownership paragraph, because a model-owned executor doing
  repeated startup/serving/cleanup cycles (a Milestone C gate) will hit it.
- Or should `cleanup()` drop the owner's own global CB reference and document that consumers must not
  outlive the owner — making the leak loud rather than latent?
- Either way: does the host suite need a contract test pinning the chosen behaviour, so the next
  person does not rediscover it on silicon?

Do **not** change `prefetcher_2d.py` on your own judgment. Write the analysis, state a
recommendation, and leave the module alone unless the analysis is conclusive — and if you do change
it, the host suite and both device baselines must be re-run.

## What remains

### 1. Run the full file green, end to end

**The whole file has never passed.** `dev03`/`dev04` predate the test-bug fix, and `dev05` only ran
the two repaired node IDs. First confirm what actually collects:

```sh
python -m pytest --collect-only -q \
  models/common/tests/modules/prefetcher/test_prefetcher_2d_wh_galaxy.py
```

Expect **8 cases**. Every previous run reported `1 deselected`, which is the attention case being
held back — make sure your final evidence run selects it (see item 2).

### 2. Attention-with-prefetcher — it HAS now run once, and it failed. Diagnose it.

> **Updated 2026-08-26.** A second unattended session ran this case for the first time before being
> killed 16 minutes in (see "Session 2" below). Two facts are now established and you should not
> re-derive them:
>
> - **`dev06_attention_with_prefetch_isolated.log`: `FAILED` with**
>   `TT_FATAL: Programs must be executed on a single sub-device (assert.hpp:104)`.
>   This is the predicted conflict, confirmed: `galaxy_prefetch_decode_mode_plan` splits the grid into
>   a sender subdevice (`x ∈ {0,4}`) and a worker subdevice (`x ∈ {1,2,3} ∪ {5,6}`), while
>   Attention2D's decode QKV/WO matmuls use a `(7,1)` grid spanning `x = 0..6`. The program straddles
>   both subdevices, and tt-metal rejects that outright. It is a clean, fast, deterministic failure —
>   not a hang, not a numerical error.
> - **The aborted program leaves the mesh un-drainable.** The process then sat in `mesh_device`
>   fixture teardown; a gdb backtrace (`dev06b_gdb_teardown_backtrace.log`) puts the main thread in
>   `FDMeshCommandQueue::~FDMeshCommandQueue → clear_expected_num_workers_completed →
>   wait_for_outstanding_reads → pthread_cond_wait`, under `MeshDevice::close()`. So a `TT_FATAL`
>   abort inside a multi-subdevice program leaves reads outstanding that the destructor waits on
>   forever. Budget a kill and a `tt-smi -glx_reset` after any run of this case that fails.
>
> Your job on this item is therefore **diagnosis and a recommendation, not a green run.** Decide and
> argue: can the case be made to work by narrowing the attention decode matmul grid to the worker
> subdevice (does Attention2D's config even allow that, and does the qualified geometry survive it)?
> Or is production prefetch fundamentally incompatible with the current attention decode grid, making
> this a Milestone B item where the production grids get chosen anyway? Record the answer either way;
> a well-argued "incompatible, deferred to B, here is why" closes this item honestly.
>
> If you do get it to pass, it still needs the three-fresh-process treatment like everything else.
>
> The teardown hang is worth a line in the report in its own right: it means **any** future
> multi-subdevice program abort on this stack costs a reset, which is useful for whoever sequences
> Milestone B device work.

The original framing follows, for the reasoning behind the case.

### 2b. Original brief text for this item

`test_prefetcher_2d_wh_galaxy_attention_decode_with_active_prefetch`
([:727](../models/common/tests/modules/prefetcher/test_prefetcher_2d_wh_galaxy.py#L727)) imports the
attention geometry unchanged from the qualified suite and swaps in
`galaxy_prefetch_decode_mode_plan`. **This is the brief's highest-risk item and the reason it was
sequenced last.** Run it isolated, by node ID, first.

Re-read the brief's Test 5 section before you do. The predicted conflict: with
`galaxy_prefetch_decode_mode_plan` the worker subdevice is `x ∈ {1,2,3} ∪ {5,6}` and `semaphore_cores`
matches it, but Attention2D's decode QKV/WO matmuls use a `(7,1)` grid spanning `x = 0..6`, which
overlaps the prefetch **sender** cores at `x ∈ {0,4}`.

If it hangs or fails, that is a **finding to write up**, not something to force. In particular:

- **Do not narrow `semaphore_cores` to make a hang go away.** That defect is documented at
  [`_wh_galaxy_hardware.py:316+`](../models/common/tests/modules/_wh_galaxy_hardware.py#L316) after it
  cost a full evidence run.
- If it hangs, capture a traceback before spending a recovery attempt — the 08-25 attention debugging
  used a repeating `faulthandler.dump_traceback_later` pytest plugin (diagnostic only, never
  committed) and located the stall in two dumps 90 s apart.
- Report precisely what conflicts with what, and recommend whether resolving it belongs in Milestone A
  or in Milestone B where the production grids get chosen anyway.

### 3. Three whole-file repeats in fresh processes

The brief's anti-aliasing requirement, and the one the two 2026-08-25 root causes justify: a case that
flips between processes is reading aliased or uninitialised L1. Never done — the job died just as it
started them. Three fresh processes, whole file, all green.

### 4. Confirm the attention device suite still passes

The shared-helper refactor touched `_wh_galaxy_hardware.py`, which
[`test_attention_2d_wh_galaxy.py`](../models/common/tests/modules/attention/test_attention_2d_wh_galaxy.py)
imports. The refactor is behaviour-preserving and attention only uses `galaxy_mode_plan` and
`require_galaxy_ccl_hardware_resources`, neither of which changed semantically — so this is low risk,
but it is unverified and cheap:

```sh
models/common/tests/modules/attention/test_attention_2d_wh_galaxy.py    # expect 2 passed
```

MLP (`4 passed`) and RMSNorm (`8 passed`) were already re-run post-refactor and are green; re-run them
only if you change shared plumbing again.

### 5. Write `tttv2_milestone_a_gap2_evidence/REPORT.md`

The directory exists with logs but no report. Sections, per the brief and the house format:

1. **Summary** — cases attempted/passed/failed/blocked, wall-clock, and the headline plainly. Say that
   the run spanned two sessions and why.
2. **Environment** — commit, branch, host, 32 devices, build type. Note the tree is dirty by design.
3. **What this suite closes** — the mode-transition matrix, the rollback paths, cleanup from an active
   mode, and the leak detector, none of which had ever run on silicon.
4. **Results table** — one row per node ID, every log cell pointing at a real file in `logs/`.
   Include `dev03`/`dev04` and the test bug honestly; a reader should be able to see the whole path.
5. **Finding F1** — the global CB lifetime, the probe that isolated it, the test-side workaround, and
   your recommendation (see above).
6. **Test 5 outcome** — attention with an active prefetcher: passed, or the precise conflict.
7. **Caveats and gaps** — at minimum: the payload is MLP2D geometry only, so contexts are qualified
   for that consumer shape; no trace/capture coverage; failure-injection covers the paths the host
   suite models and not arbitrary faults.

### 6. Append the work-log checkpoint

`## Hardware checkpoint: Prefetcher2D and Galaxy resource hardware qualification <ISO date>` in
`tttv2_2d_modules_work_log.md`, matching that file's terse bullet style. Cover the suite, the
transition matrix, F1 and your recommendation, the Test 5 outcome, and the repeat evidence.

### 7. Draft the status-page replacements

In the report, not applied: replacement text for the `Prefetcher2D` and `Galaxy CCL/resources` rows of
`MILESTONE_A_STATUS.md`, and for
[`modules/README.md:212`](../models/common/modules/README.md#L212) — "Integrated Prefetcher2D/Galaxy-resource
ownership has host coverage but is not yet qualified on hardware" — if that caveat can now be dropped.
**Do not edit those files.** They are being rewritten wholesale as a separate task.

### 8. Decide the fate of `tttv2_gap2_scratch/`

Two scratch probes (`test_probe.py`, `test_probe2.py`) plus a `__pycache__`. `probe02` is what isolated
F1, so it has evidentiary value. Either fold the essential part into the suite or the report, or keep
the directory and say in the report that it is diagnostic scratch, not a deliverable. Do not leave it
undescribed.

## Run procedure

Host: the reserved 6U WH Galaxy. **The reservation rolled over — you are on a different host than the
2026-08-25 run** (`...117439` at the time of writing, not `...116970`). Record the actual hostname in
the report; do not copy it from the earlier logs.

```sh
ls /dev/tenstorrent | wc -l      # must be 32
tt-smi -ls
pgrep -af 'pytest|ttnn' | grep -v grep     # must be empty
```

One pytest process on the device at a time. Never pipe pytest — redirect with `> LOG 2>&1`.

```sh
timeout --signal=TERM --kill-after=180 2700 \
  python -m pytest -v -rA --color=no -p no:cacheprovider <FILE-OR-NODEID> > "$LOG" 2>&1
echo "exit=$?" >> "$LOG"
```

The harness caps a foreground tool call at 600 s; issue longer runs as tracked background processes and
block on exit before the next one. Re-check `pgrep` between runs.

On a hang or crash: kill the process tree, confirm the device is free, `tt-smi -glx_reset`, confirm
`Re-initialized 32 boards`, retry. **Maximum 2 recovery attempts per case**, then `BLOCKED (infra)` with
logs. Keep every log; never overwrite one. Continue the existing `dev0N_` / `host0N_` numbering.

## Hard prohibitions

- Do not modify any `models/common/modules/**/*_1d.py` implementation file.
- Do not change `prefetcher_2d.py` on your own judgment — F1 is an analysis-and-recommend item.
- Do not narrow `semaphore_cores` for a generic async CCL to silence a hang.
- Do not relax a PCC threshold or a tolerance.
- Do not edit `MILESTONE_A_STATUS.md`, `modules/README.md`, or `tttv2_2d_modules_plan.md`.
- Do not `git commit`, `push`, `checkout`, `stash`, or `reset`. Leave the tree dirty for review.
- Do not rebuild tt-metal or recreate the venv.
- Do not run the full `models/common/tests` suite or any 1D hardware matrix.
- Do not claim a result you did not observe. `BLOCKED (infra)` with logs is honest; an invented pass is
  not. In particular, do not report case 8 as passing on the strength of the other seven.

## Finish condition

All 8 cases have a terminal state with at least one log. Case 8 is either passing, or carries a
written diagnosis and a recommendation on whether it belongs in Milestone A or Milestone B — with the
subdevice conflict and the post-abort teardown stall both documented. Cases 1-7 are green in three
fresh processes (run case 8 separately if it still aborts, so it does not poison the file runs); the attention device
suite still shows `2 passed`; F1 carries a written recommendation; `REPORT.md` and the work-log
checkpoint are written; the status-page replacements are drafted in the report but not applied; the
scratch directory's status is stated; the device is left clean with `tt-smi -ls` showing 32 boards.
Print the absolute path of `REPORT.md` as your last line.

If you finish early, do not invent extra work. Stop.
