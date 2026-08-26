# Agent Brief: WH Galaxy Device Accuracy Evidence for Milestone A 2D Modules

You are running **unsupervised and overnight** on a Wormhole Galaxy host. Nobody will answer
questions. Every decision below is already made for you. Follow it literally.

## Mission

Re-run, on real hardware, the device accuracy tests added by the last commit on branch
`gongyu/tttv2_wh_glx_2d_modules`, and produce a self-contained evidence package: one report plus one
raw log per case, where every claim in the report points at a log file on disk.

You are producing **evidence**, not fixing code. See "Hard prohibitions".

## Context to read first (read-only)

1. `tttv2_2d_modules_plan.md` — sections "Milestone A exit gate" and "Shared 2D Contracts". These
   define what the tests are supposed to prove: PCC >= 0.99 versus a PyTorch reference, KV-cache
   PCC >= 0.99 where applicable, repeat-invocation, and clean teardown.
2. `models/common/modules/MILESTONE_A_STATUS.md` — the previously *claimed* per-module hardware
   evidence. Your job is to independently reproduce it. Treat its table as the expectation to
   confirm or contradict, not as ground truth.
3. `tttv2_2d_modules_work_log.md` — read at least the last ~150 lines. It records the exact hardware
   failure modes, the required grids/geometries, and the reset procedures that were needed. This is
   the single most useful file for triaging a failure.

## Step 1 — Confirm the commit and the test selection

```sh
git rev-parse HEAD                       # expect de4c8f4e659... "add reusable WH Galaxy 2D modules"
git show --stat HEAD | grep -i test
```

Select **only** tests that execute on the device and check module numerical accuracy. The selection
criterion is mechanical: the test takes the `mesh_device` fixture parametrized indirectly with the
real `(8, 4)` mesh. In this commit that is exactly the seven `*_wh_galaxy.py` files:

```text
models/common/tests/modules/embedding/test_embedding_2d_wh_galaxy.py
models/common/tests/modules/rope/test_rope_2d_wh_galaxy.py
models/common/tests/modules/rmsnorm/test_rmsnorm_2d_wh_galaxy.py
models/common/tests/modules/mlp/test_mlp_2d_wh_galaxy.py
models/common/tests/modules/attention/test_attention_2d_wh_galaxy.py
models/common/tests/modules/lm_head/test_lm_head_2d_wh_galaxy.py
models/common/tests/modules/sampling/test_sampling_2d_wh_galaxy.py
```

**Explicitly out of scope** (they are host-only, using `_Mesh`/mock objects and no silicon; do not
run them and do not count them as device evidence):

- the sibling `test_*_2d.py` files without the `_wh_galaxy` suffix;
- `models/common/tests/models/galaxy/**`;
- `models/common/tests/modules/prefetcher/test_prefetcher_2d.py`;
- `models/common/tests/llm_runtime/**`;
- everything under `models/common/tests/modules/*/` not listed above.

Verify the selection instead of trusting this list. Run collection only (no device):

```sh
python -m pytest --collect-only -q \
  models/common/tests/modules/{embedding,rope,rmsnorm,mlp,attention,lm_head,sampling} \
  -k wh_galaxy > "$EVIDENCE/logs/00_collect.log" 2>&1
```

Expect **21 device cases**. If the count differs, record the actual collected node IDs in the report
and run what collection actually reports. Never invent a node ID; copy it from the collection log.
The expected matrix is:

| File | Test | Cases |
| --- | --- | --- |
| embedding | `test_embedding_2d_wh_galaxy_reference` | `llama`, `qwen` |
| rope | `test_rotary_setup_2d_wh_galaxy_reference` | `llama`, `qwen` |
| rmsnorm | `..._final_norm_decode_batch_32_fused_residual_repeat` | `llama-final-8192`, `qwen-final-5120` |
| rmsnorm | `..._final_norm_prefill_repeat` | 2 dims x `seq128`/`seq2048` |
| rmsnorm | `..._head_local_128_qk_decode_and_prefill_repeat` | `q_norm`, `k_norm` |
| mlp | `..._decode_batch_32_repeat` | `llama-8192x28672`, `qwen-5120x25600` |
| mlp | `..._prefill_128_then_2048_repeat` | same two |
| attention | `..._decode_and_prefill_repeat` | `llama-70b`, `qwen3-32b` |
| lm_head | `..._decode_reference` | `llama`, `qwen` |
| sampling | `..._exact_padded_vocab_exclusion` | single case |

## Step 2 — Record the environment baseline, once, before any device run

Write `$EVIDENCE/ENVIRONMENT.md` containing:

- `git rev-parse HEAD`, `git status --short`, and `git submodule status --recursive`;
- `tt-smi -ls` output (also save raw to `logs/01_tt_smi_before.log`);
- `ls /dev/tenstorrent` (expect 32 device nodes, `0`..`31`);
- the build type and commands the bootstrap script used, and the `python_env` Python version;
- the wall-clock start time.

If fewer than 32 device nodes are present, **stop**: write the report stating the host is not a
complete `(8, 4)` Galaxy and that no device evidence could be produced. Do not run a reduced mesh.

## Step 3 — Run the cases, one process at a time

### Serialization — non-negotiable

The work log records multiple invalidated runs caused by two pytest processes sharing the Galaxy.

- Exactly **one** pytest process may touch the device at any moment.
- Never background a pytest run. Never put pytest in a shell pipeline (`| tee`, `| grep`) — the
  pipeline can return control while the nested pytest is still holding the device. Redirect with
  `>` `2>&1` and read the file afterwards.
- Before each run, confirm no stale process holds the device:
  `pgrep -af 'pytest|ttnn' | grep -v grep`. If anything is found, kill that process tree and reset
  (below) before continuing.
- Do not launch subagents that run hardware. Host-only reading/analysis in parallel is fine.

### Grouping

Run **one file at a time**, in the order listed in Step 1 (cheapest and most-established first,
attention last since it is the largest and most fragile). Device parameters differ between files and
even between tests within `rmsnorm`, so pytest will re-open the mesh per parametrization; that is
expected and is itself part of the teardown evidence.

If a whole-file run fails or hangs, re-run its cases **individually by node ID** so a single bad case
cannot mask its siblings.

### The run command

For each group, with `LOG="$EVIDENCE/logs/<NN>_<module>.log"`:

```sh
timeout --signal=TERM --kill-after=180 2700 \
  python -m pytest -v -rA --color=no --showlocals -p no:cacheprovider \
  <FILE-OR-NODEID> > "$LOG" 2>&1
echo "exit=$?" >> "$LOG"
```

- The 2700 s (45 min) bound is per group and is generous on purpose: the first run of the session
  pays JIT kernel compilation. A group that hits the bound is a **hang**, not a slow pass.
- Always append the exit code to the log; you will cite it in the report.
- Record start and end timestamps for each group in the report.

### Triage and retry

Classify every non-pass into exactly one of:

1. **Genuine test failure** — pytest reported `FAILED` with an assertion, and the devices closed
   normally. Record it as a failure with the PCC/assertion text. **Retry at most once** to establish
   whether it is deterministic, then move on.
2. **Infrastructure fault** — timeout, segfault/core dump, hang during teardown, fabric or
   subdevice init error, `Input Tensor is not allocated` style runtime abort, or pytest producing no
   report at all. Recovery procedure:
   a. kill the whole process tree of the run (`pkill -TERM -f pytest`, wait, then `-KILL`);
   b. confirm nothing holds the device (`pgrep -af pytest`);
   c. `tt-smi -glx_reset > "$EVIDENCE/logs/reset_<NN>_<n>.log" 2>&1` and confirm all 32 boards
      complete post-reset reinitialization (fall back to `tt-smi -r` if `-glx_reset` is unavailable);
   d. re-run the same group. **Maximum 2 recovery attempts per group.** After that, mark the group
      `BLOCKED (infra)` and continue to the next group.
3. **Collection/import error** — the test never reached the device. No reset needed; record and
   continue.

Whenever pytest reports that devices "closed normally", note it — no reset is needed and resetting
needlessly costs ~minutes.

Keep every log from every attempt, including failed and reset ones. Name retries
`<NN>_<module>_attempt2.log`. Do not delete or overwrite a log.

## Step 4 — Extract the numbers

For each case that passed, pull the actual evidence out of its log rather than asserting success:

- the PCC values printed by `comp_pcc` (grep for `pcc`, `PCC`, `Max ATOL`);
- for attention, both output PCC **and** K and V cache PCC;
- the pytest summary line (`N passed ... in Xs`);
- confirmation of clean device close / teardown.

If a test passed but printed no PCC value, say so explicitly in the report — "passed; PCC asserted
internally at threshold 0.99, value not emitted to the log" — rather than inventing a number.

## Step 5 — Write the report

Write `$EVIDENCE/REPORT.md` with these sections:

1. **Summary** — one paragraph: total cases attempted, passed, failed, blocked, and wall-clock span.
   State the headline result plainly. If it is not a clean sweep, say that in the first sentence.
2. **Environment** — link to `ENVIRONMENT.md`; commit SHA, submodule SHAs, build type.
3. **Results table** — one row per case:

   | Module | Node ID | Result | PCC / assertion evidence | Log | Exit | Duration | Resets |

   Every row's Log cell must be a relative path to a real file in `logs/`.
4. **Comparison against `MILESTONE_A_STATUS.md`** — per module, does this run confirm, partially
   confirm, or contradict the recorded claim? Call out anything the status page claims that your run
   did **not** cover. Note explicitly that stochastic `Sampling2D` hardware coverage and real-device
   physical-32 trace runs are outside this test set, matching the status page's own caveats.
5. **Infrastructure events** — every timeout, crash, and reset, with its log path.
6. **Caveats and gaps** — what this evidence does *not* establish.

Also append a short chronological section to `tttv2_2d_modules_work_log.md` titled
`## Hardware checkpoint: unsupervised Milestone A device evidence re-run <ISO date>` summarizing the
outcome and pointing at `$EVIDENCE/REPORT.md`. Match the existing terse bullet style of that file.

## Hard prohibitions

These exist because you are unsupervised and the output is evidence.

- **Do not modify any non-test source file.** Not `models/common/modules/**`, not
  `models/common/models/**`, not `models/common/llm_runtime/**`, not anything under `tt_metal/` or
  `ttnn/`.
- **Do not modify any test file** — not thresholds, not parametrizations, not geometries, not
  `_wh_galaxy_hardware.py`. A test that fails is a result to report, not a bug to patch. Evidence
  produced from edited tests is worthless.
- **Do not `git commit`, `git push`, `git checkout`, `git stash`, or `git reset`.** Stay on the
  current commit. The only files you create are inside `$EVIDENCE/` plus the one appended section in
  `tttv2_2d_modules_work_log.md`.
- **Do not rebuild tt-metal or recreate the venv.** The bootstrap script did that. If the build is
  broken, report that and stop.
- **Do not relax, skip, or `-k`-filter away a failing case** to make the run look green.
- **Do not run the full `models/common/tests` suite** or any 1D hardware matrix; the work log records
  that expanding to affected-file selection ballooned to 516 hardware cases. Stay inside the 21.
- **Do not claim a result you did not observe.** If you ran out of attempts, the honest report is
  `BLOCKED (infra)` with the logs attached.

## Finish condition

You are done when every one of the collected device cases has a terminal state
(`PASSED` / `FAILED` / `BLOCKED (infra)`), each with at least one log file, and `REPORT.md` plus
`ENVIRONMENT.md` are written. Then run a final `tt-smi -ls > "$EVIDENCE/logs/99_tt_smi_after.log"`
and leave the device in a clean state. Print the absolute path of `REPORT.md` as your last line.

If you finish early, do not invent extra work. Stop.
