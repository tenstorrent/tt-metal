# Agent Brief — Gap 3: batched-prefill policy device evidence / physical-32 trace

You are closing one named Milestone A coverage gap. **Read the "Read this before you plan anything"
section first — this gap is not the same shape as the other two, and the obvious reading of it is
probably not achievable inside Milestone A.**

## Read this before you plan anything

Gaps 1 and 2 are missing tests for code that already ships. This one is different.

`MILESTONE_A_STATUS.md:33` records the batched-prefill policy as
"Host lifecycle qualified; **no real-device trace run recorded**", and the plan's
"Planning and trace coverage" section asks to
"Qualify physical-32 traces at sequence length 128 first. Then add each supported padded sequence
length through 2048."

**A physical-32 trace on WH Galaxy needs a model-owned executor with `TraceCompiler`/`TracedExecutor`
running a 2D model at batch 32 on `(8, 4)`. The executors are Milestone C deliverables and the 2D
models are Milestone B. There is nothing on the Galaxy to trace yet.**

Meanwhile Milestone A's own sequence, item 7, asks only for:

> 7. Add the generic batched-prefill policy and host/runtime tests.

which is done — 144 prefill-runtime tests, the 1259-test integrated gate.

So the literal item is blocked on work that is deliberately downstream. **But there is a real,
achievable, and currently-unmet Milestone A exit-gate obligation hiding underneath it**, and that is
your actual mission.

## The real gap: the runtime delegation has zero device evidence

The Milestone A exit gate requires:

> - preserve every pre-existing default-runtime test and expectation;
> - **keep any runtime execution-code change to tested, topology-neutral config delegation.**

And the plan's shared-code change budget says touching `prefill/plan.py` and `prefill/runtime.py`
"requires a focused test that fails without the delegation and passes with it."

The delegation exists:

| Location | What it does |
| --- | --- |
| [prefill/config.py:20](../models/common/llm_runtime/prefill/config.py#L20) | `BatchedPrefillPolicy` frozen dataclass |
| [prefill/config.py:~54-68](../models/common/llm_runtime/prefill/config.py#L54-L68) | `BatchedPrefillPolicy.default(...)` — the preserved default |
| [prefill/config.py:128-135, 212](../models/common/llm_runtime/prefill/config.py#L128-L135) | policy ↔ `max_prefill_batch_size` / `max_prefill_chunk_size` consistency |
| [prefill/plan.py:117, 175-183, 207](../models/common/llm_runtime/prefill/plan.py#L175-L183) | policy threaded into planning; `None` → `.default(...)` |
| [prefill/plan.py:457-460](../models/common/llm_runtime/prefill/plan.py#L457-L460) | physical batch selection reads `policy.physical_batch_sizes` / `maximum_physical_batch` |
| [prefill/runtime.py:198](../models/common/llm_runtime/prefill/runtime.py#L198) | runtime passes `self.config.batched_prefill_policy` through |

And its test coverage is **entirely host-side with mock meshes.** Verified:

- there is no `indirect=True` anywhere under `models/common/tests/llm_runtime/` — so no test in that
  directory ever parametrizes a real `mesh_device` fixture;
- [test_llama3_8b_integration.py:24-29](../models/common/tests/llm_runtime/test_llama3_8b_integration.py#L24-L29)
  defines `class _Mesh` with `shape = (1, 1)` and a stubbed `get_num_devices()`.

So the claim "the default-config path before and after this work is behaviorally identical for every
existing 1D model" — the plan's central modularity promise — has never been checked against silicon.
That is closable now, on hardware that exists, without a 2D model.

## Your mission, in priority order

1. **Do Task A.** Prove the batched-prefill delegation on real hardware with an existing 1D model.
   This is the achievable part and it is genuinely required.
2. **Write a recommendation on Task B vs Task C** for the physical-32 Galaxy trace. Do not silently
   pick one; produce the analysis and let a human decide.

## Task A — device evidence for the policy delegation (do this)

### The vehicle already exists

[models/common/tests/demos/llama3_8b/demo.py](../models/common/tests/demos/llama3_8b/demo.py) runs
the **real** `models/common/models/llama3_8b` executor — real `PrefillRuntime`, real
`ProgramCompiler`, real trace — on real hardware. Usage is in its own module docstring:

```sh
MESH_DEVICE=N150 HF_MODEL=meta-llama/Llama-3.1-8B-Instruct \
  python_env/bin/pytest models/common/tests/demos/llama3_8b/demo.py -k "batch-32" -v
```

It has `batch-32`, `batch-32-ci`, `batch-1`, `token-accuracy` and DP cases
([demo.py:164-206](../models/common/tests/demos/llama3_8b/demo.py#L164-L206)), and it already knows
how to drive batched prefill explicitly — `_run_seeded_cross_cardinality_batch` takes
`allow_batched_prefill` and asserts the runtime config actually followed it
([demo.py:523-547](../models/common/tests/demos/llama3_8b/demo.py#L523-L547)):

```python
assert executor.prefill_runtime.config.disable_batched_prefill is not allow_batched_prefill
```

That is the pattern to build on. Note `batch-32-ci` is skipped on N150
([demo.py:257-258](../models/common/tests/demos/llama3_8b/demo.py#L257-L258)); prefer T3K if you need
the larger capacity, and record which host you used.

### What to prove

Three properties, all on real silicon:

**A1 — the default is preserved exactly.** With no explicit policy (`batched_prefill_policy=None`,
so `plan.py:176` constructs `BatchedPrefillPolicy.default(...)`), a real prefill run produces
planning output identical to the pre-delegation behaviour. Capture and assert on, at minimum:

- the planned rank-2 `[physical_batch, sequence]` token tensor shapes;
- padded page tables — padding rows must carry zero tokens and `-1` page-table entries;
- source-row metadata and slot mapping order surviving assembly;
- the resolved physical batch chosen for a given active row count;
- the program/trace signature — it must key on **physical** geometry, not active row count
  (see `prefill/signatures.py`).

The cleanest form of this is a golden comparison: run the same prompts through the default path and
assert the planning artifacts against values recorded from the current `main` merge-base behaviour.
If reconstructing a pre-delegation baseline is impractical, say so and instead assert the artifacts
against explicitly written-out expected values, with the derivation shown.

**A2 — the delegation is load-bearing.** Construct an explicit **non-default** `BatchedPrefillPolicy`
and show the plan changes in exactly the predicted way and in no other way. The plan requires "a
focused test that fails without the delegation and passes with it" — on device, the equivalent is:
a policy that restricts `physical_batch_sizes` or lowers `maximum_physical_batch` must change the
selected physical batch at `plan.py:457-460`, and must leave source-row order, slot mapping and page
tables otherwise structurally identical.

Also cover the fail-closed edges on device: `maximum_sequence_length > max_prefill_chunk_size` must
raise ([plan.py:182-183](../models/common/llm_runtime/prefill/plan.py#L182-L183)), and a non-policy
object must raise `TypeError` ([plan.py:180-181](../models/common/llm_runtime/prefill/plan.py#L180-L181)).

**A3 — trace identity is stable across active row counts.** Two requests differing only in how many
rows are active must reuse the **same** trace, and the padded rows must not write KV or return logits
for inactive slots. This is the plan's "Concat-32 planning" risk, stated as:

> Risk: padding inactive rows writes KV or returns logits for inactive slots.
> Required response: inspect planned tokens/page tables/source rows and test KV/logit isolation for
> active batches 16, 31, and 32.

Do this at whatever physical batch the chosen host supports — the property is topology-neutral, which
is the whole point. If the host's max physical batch is below 32, run the analogous triple just below
its maximum (e.g. 8/15/16) and say so explicitly in the report.

### Where the tests should live

Prefer a new device-marked file rather than adding silicon dependencies to the existing host-only
`models/common/tests/llm_runtime/` suite — that directory's host-only character is a feature. Suggested:

```
models/common/tests/llm_runtime/hardware/test_batched_prefill_policy_device.py
```

with the real `mesh_device` fixture parametrized `indirect=True`, and IDs that state mesh, model,
mode, batch and sequence per the plan's test-organization rule. Keep the fast host tests untouched
and green.

## Task B — synthetic physical-32 Galaxy trace harness (analyse, do not build without approval)

Build a minimal stub model exposing the executor contract, whose prefill graph is one `MLP2D` or a
small 2D block, wired to `PrefillRuntime` + `TraceCompiler` with the Galaxy policy
(`physical_batch_sizes=(32,)`, `minimum_active_rows=16`, `maximum_physical_batch=32`,
`maximum_sequence_length=2048`, `allow_cached_prefix=False`), and capture/replay at sequence length
128.

Arguments against, which you should weigh honestly:

- it is scaffolding that Milestone B replaces wholesale;
- the plan explicitly warns against this shape of work — "Do not extract code from an existing model
  package merely to avoid writing the new package";
- a stub model's trace identity is not the product's trace identity, so a green result here is weak
  evidence for the thing the gate actually cares about.

Arguments for: it de-risks the Milestone C trace work early, and it would catch a policy/signature
mismatch at physical batch 32 before two models depend on it.

**Estimate the cost and state a recommendation. Do not build it in this task without explicit
approval.**

## Task C — formal deferral to Milestone C (analyse)

Record the physical-32 real-device trace as an explicit Milestone C gate item rather than an open
Milestone A gap, with a written justification: the subject of the test does not exist until B/C, and
Milestone A's own sequence item 7 scopes this area to "the policy and host/runtime tests".

If you recommend this, draft the exact text for:

- the `Batched-prefill policy` row of `models/common/modules/MILESTONE_A_STATUS.md`;
- a new explicit entry in the plan's Milestone C functional gate list.

Put the drafts in your report. Do not edit `MILESTONE_A_STATUS.md` or `tttv2_2d_modules_plan.md` —
both are being revised as separate tasks.

## Scheduling note

**Task A does not need the Galaxy.** It runs on N150 or T3K. Gaps 1 and 2 need exclusive use of the
6U Galaxy and must be serialized against each other. If all three agents are running, put Task A on
a different host so it does not contend — and confirm which host you are on before your first run,
because a Galaxy-contending run will corrupt someone else's evidence.

If you must share a host with another agent, coordinate explicitly: exactly one pytest process may
hold a given device at any moment.

## Run procedure

Confirm your host and that nothing else holds the device:

```sh
tt-smi -ls
ls /dev/tenstorrent
pgrep -af 'pytest|ttnn' | grep -v grep     # must be empty
```

Never put pytest in a shell pipeline (`| tee`, `| grep`) — the pipeline can return control while the
nested process still holds the device. Redirect with `> LOG 2>&1` and read the file afterwards.

```sh
timeout --signal=TERM --kill-after=180 2700 \
  python -m pytest -v -rA --color=no -p no:cacheprovider <FILE-OR-NODEID> > "$LOG" 2>&1
echo "exit=$?" >> "$LOG"
```

The agent harness caps a foreground tool call at 600 s. If a run needs longer, issue it as a tracked
background process and block on its exit before the next one. Re-check `pgrep` between runs.

On a hang or crash: kill the process tree, confirm the device is free, reset (`tt-smi -glx_reset` on
Galaxy, `tt-smi -r` otherwise), retry. **Maximum 2 recovery attempts**, then record `BLOCKED (infra)`
with logs and move on. Keep every log; never overwrite one.

The demo needs `HF_MODEL` and a downloaded checkpoint. If the checkpoint is not present on the host,
**stop and report** — do not download a 8B checkpoint unattended, and do not substitute a different
model to make the run go.

## Regression gates

The whole point of Task A is that default behaviour is unchanged, so the host suites must stay green:

```sh
python -m pytest models/common/tests/llm_runtime -q          # expect the full prefill/runtime set green
python -m pytest models/common/tests/models/llama3_8b -q
pre-commit run --files <every file you touched>
```

Note [test_demo_contract.py:59-60](../models/common/tests/models/llama3_8b/test_demo_contract.py#L59-L60)
asserts on the **source text** of `demo.py`. If you modify the demo, that contract test will fail —
update it deliberately and explain why, or better, avoid touching `demo.py` and drive the executor
from your new test file instead.

## Hard prohibitions

- Do not modify any `models/common/modules/**/*_1d.py` implementation file.
- Do not change default planner, warmup, trace, cache, or output semantics. If Task A reveals that
  the default is **not** preserved, that is a finding and a bug — report it, do not adjust the
  expected values to match.
- Do not add a Galaxy, Llama, Qwen, 2D-mesh, or `(8, 4)` branch to any runtime execution path. The
  plan forbids it and the modularity scorecard checks for it.
- Do not build Task B without explicit approval.
- Do not edit `MILESTONE_A_STATUS.md` or `tttv2_2d_modules_plan.md`.
- Do not `git commit`, `push`, `checkout`, `stash`, or `reset`. Leave the tree dirty for review.
- Do not rebuild tt-metal or recreate the venv.
- Do not download model checkpoints unattended.
- Do not claim a result you did not observe.

## Deliverables

1. `models/common/tests/llm_runtime/hardware/test_batched_prefill_policy_device.py` (or your chosen
   equivalent location), covering A1, A2 and A3.
2. An evidence directory `tttv2_milestone_a_gap3_evidence/` with one raw log per run plus `REPORT.md`
   containing: the host and mesh actually used, node IDs, results, the planning artifacts you
   captured and what you compared them against, the physical-batch triple you used for A3 and why,
   and anything you could not close.
3. **A written recommendation on Task B vs Task C**, with cost estimate, the argument each way, and
   the draft status-page / plan text if you recommend C. This is a required deliverable, not
   optional — it is the main decision this brief exists to inform.
4. A `## Checkpoint: batched-prefill policy device evidence <ISO date>` section appended to
   `tttv2_2d_modules_work_log.md` in that file's terse bullet style.

## Finish condition

Task A has recorded device evidence for A1, A2 and A3 or a documented reason each could not be
closed; the host regression suites are green; the Task B/C recommendation is written; `REPORT.md` and
the work-log section exist; the device is left clean. Print the absolute path of `REPORT.md` last.

If you finish early, do not invent extra work. Stop.
