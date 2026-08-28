# Job 3 — Milestone B step 7: paged KV, prefix cache, concat-32, sampling, long context

**Device job.** You have the Galaxy exclusively. One pytest process at a time.

## Mission

Plan step 7, for both models: "Add paged KV, prefix-cache, concat-32, device-sampling, and
long-context direct-demo coverage." This is the job that turns two working models into a Milestone B
that can actually pass its exit gate.

**Read `tttv2_milestone_b_briefs/job2_completion_handoff.md` first.** It gives you both models'
working entry points and the exact commands that produce a passing decode and prefill for each. If it
does not exist, stop and report `BLOCKED (mb-qwen did not complete)`. If it exists but records Qwen as
incomplete, do the Llama half of every item below, record the Qwen half as `BLOCKED (upstream)`, and
say so — do not quietly halve the scope.

## The five areas

Do them in this order. Each earlier one is a dependency of the ones after it.

### 1. Paged KV

Both models expose per-layer KV metadata and bind/unbind a paged cache transactionally through
`models/common/models/galaxy/kv_contract.py`. Prove:

- paged fill during prefill, then decode reading the same blocks, PCC ≥ 0.99 against the contiguous
  path;
- late capacity resolution — a cache bound after the model is constructed;
- transactional unbind, and that a failed bind leaves no partial state;
- **no cross-slot contamination**: a write for user *i* never appears in user *j*'s blocks.

Milestone B corrected a real page-table defect here: prefill's table carries one row per user because
`paged_fill_cache` indexes by `batch_idx`, while decode requires exactly `users_per_column` rows (or
that batch repeated once per core when L1-sharded). Those two layouts are the thing to test
adversarially — feed decode a prefill-shaped table and assert it is *rejected*, not silently accepted.

### 2. Concat-32 physical prefill

The plan's own risk section is explicit: padding inactive rows must not write KV or return logits for
inactive slots. Inspect the planned tokens, page tables and source rows directly, then test KV and
logit isolation for **active batches 16, 31 and 32** — the three cases the batched-prefill policy
distinguishes.

Qualify sequence length 128 first, then expand through 2048 in the padded lengths the policy supports.
Do not jump to 2048.

Note the boundary: the generic batched-prefill policy lives in `llm_runtime` and Milestone B imports
none of it. The Galaxy model reshapes and concatenates rows for its own device graph. **Nothing in
this job may add a Llama, Qwen, Galaxy or `(8, 4)` branch to runtime code**, and nothing may modify
`llm_runtime` at all. If you conclude otherwise, stop and write the reduction the plan requires.

### 3. Prefix-cached and chunked prefill

The gate is that **prefix-cached output matches uncached execution under the model's numerical
acceptance**. Cover the chunk-aligned SDPA path that reads the paged cache, and the single-row
page-table slicing that chunked SDPA needs (its leading dimension must equal Q's batch, which is 1 for
a single-row prefill — Milestone B slices the addressed user's row out for exactly this reason).

Test the interaction, not just the feature: a prefix-cached request followed by a normal one, and a
mix of both in the same batch.

### 4. Device sampling

Both greedy and stochastic, on device, through `Sampling2D` and the per-column user selection in
`GalaxyColumnUserSelector`. Prove:

- deterministic seeded requests stay **slot-stable** — the same seed in the same slot gives the same
  token across runs, and moving a request to a different slot does not change its stream;
- greedy matches the host argmax exactly;
- padded-vocabulary entries can never be sampled — Llama's 128256 and Qwen's 151936 both pad, and an
  invalid entry winning is a correctness bug, not a rounding issue;
- per-slot heterogeneous top-k / top-p / temperature, since serving mixes them.

Watch the temperature semantics: `ttnn.sampling`'s `temp` argument is the **reciprocal** temperature,
and `Sampling2D` writes `1/T`. That was defect D4. `direct_runner`'s host reference divides by `T`
while passing raw `T` to the module, which is correct against the fixed module — verify that pairing
on device rather than assuming it, because `T = 1.0` is its own reciprocal and hides the error.

`top_k > 32` is outside the config contract; do not test it, and do not extend the contract to allow
it.

### 5. Long-context direct demos

Batch-1 functional smokes at **4K, 32K and 128K**. Functional means it runs, produces coherent output
and tears down cleanly — not a PCC gate. Expect memory and paging behaviour to be the limiting factor
rather than numerics, and record where each one actually spends its capacity.

## Then: assemble the exit-gate evidence

Run the full Milestone B gate from the plan and record each line with the command that produced it:

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

The two accuracy numbers were measured by `mb-llama` and `mb-qwen`. **Re-measure them at this tree**,
not by quoting theirs — this job changes shared code, and Milestone A's central lesson is that
evidence collected at a tree that has since moved is not evidence. If a re-measurement differs from
the earlier one, that difference is the most important finding of the night.

## Repeat and cleanup

Cross-cutting, and easy to skip under time pressure — do not skip it:

- repeated requests against one live model, with deterministic results;
- repeated model construction and teardown in one process. This is where **L1** bites: `Prefetcher2D`
  cannot free its global circular buffer, so a second owner's `seal()` fails with an L1 OOM unless
  consumers are torn down before or with the owner. If you hit it, the fix is teardown ordering; if
  the ordering contract turns out to be unworkable at model scale, that is a real finding and belongs
  in the report as input to the Milestone B/C redesign, not a workaround.

## Regression gates

```sh
# host — everything, including the 1D suites the plan protects
python -m pytest -q models/common/tests/modules models/common/tests/models \
                    models/common/tests/llm_runtime

# boundaries
git diff --name-only <job0-base>..HEAD | grep '_1d\.py'      # must be empty
git diff --name-only <job0-base>..HEAD | grep 'llm_runtime'  # must be empty
```

Existing 1D model contract and demo-contract host tests must pass **without changed expectations**. If
one needs its expectation changed to accommodate this work, that is a boundary violation — report it
rather than editing the expectation.

## Deliverables

1. Committed coverage tests under `models/common/tests/models/{llama33_70b_galaxy,qwen3_32b_galaxy}/`
   and, where genuinely shared, `models/common/tests/models/galaxy/`.
2. `tttv2_milestone_b_evidence/coverage/` — raw logs from every attempt, plus `REPORT.md` with a row
   per area, the full exit-gate table with its commands and measured values, every defect with its
   root cause, and everything `BLOCKED` with logs.
3. `tttv2_milestone_b_evidence/coverage/ENVIRONMENT.md`.
4. A checkpoint appended to `tttv2_2d_modules_milestone_b_work_log.md`.
5. `tttv2_milestone_b_briefs/job3_completion_handoff.md` — for `mb-signoff`: which exit-gate lines
   pass, which fail with their measured values, which were not reached, and what Milestone C inherits
   (L1's ownership redesign and the CCL merge evaluation are both due there).

## Finish condition

All five areas attempted and recorded, the exit-gate table filled in with measured values rather than
quoted ones, repeat and cleanup exercised, and the handoff written. Print the absolute path of your
`REPORT.md` as your final line.

A `FAIL` with a diagnosis is a complete result for this job. Milestone B's gate is a fact to be
measured, not a target to be reached by adjusting the measurement.
