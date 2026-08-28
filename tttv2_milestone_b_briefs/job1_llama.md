# Job 1 — Milestone B steps 1–3: Llama-3.3-70B on WH Galaxy `(8, 4)`

**Device job.** You have the Galaxy exclusively. One pytest process at a time.

## Mission

Take `models/common/models/llama33_70b_galaxy` from "written but never executed" to
"prefill and decode correct on real silicon, at full 80-layer scale, with recorded teacher-forced
accuracy". Plan steps 1, 2 and 3.

Start by reading `tttv2_milestone_b_briefs/job0_completion_handoff.md`. It tells you which commit you
are on, what job 0 left `OPEN`, and which of the reconciliation findings turned out to be wrong. If it
does not exist, stop and report `BLOCKED (reconcile did not complete)` — the fused-norm stats defect
(C1) makes every decode fail until job 0 has fixed it, and rediscovering that costs a night.

## What is already true, and what is not

The model code exists for both models and the host suites should be green after job 0. **No line of it
has ever run on hardware.** The Milestone A modules underneath it are individually qualified on
`(8, 4)`; what has never been qualified is their composition.

The Milestone B author ranked the risks, and the order is worth trusting:

1. **RoPE composed with `Attention2D` is the expected first failure.** Milestone A qualified attention
   with an *identity* rotary and qualified `RotarySetup2D` standalone; the pairing has never run.
   `GalaxyAttentionCollectives.rotary` issues the production `rotary_embedding_llama` calls
   (non-fused decode by default, fused available through `use_qk_fused_rotary`). Decode-mode RoPE
   requires Q/K heads height-sharded with `cos.logical_shape()[1] == batch`; that is what
   `RotarySetup2D` produces for `users_per_column = 8`, but the pairing is unproven.
2. **L3 — attention decode on the prefetch subdevice partition.** Milestone A recorded this as
   terminal: the decode QKV `ttnn.linear` used a `(7,1)` grid that straddles the sender/worker
   subdevice split, and tt-metal rejects programs that straddle subdevices. The Milestone B recipes
   build the partition-compatible ring/`gather_in0` form instead. **You are the first job that can
   prove that.** If it works, say so precisely; it closes a Milestone A limitation.
3. **L1 — global-CB ownership.** `Prefetcher2D.cleanup()` cannot free the global circular buffer, so
   ~55 MB of L1 stays resident until every context handle dies. One prefetcher now feeds 80 layers.
   If a second model construction in the same process fails with an L1 OOM, this is why; the fix is
   teardown ordering, not more memory.
4. **Fused decode norm at real scale.** Job 0 fixed the placement defect on paper. This job runs it.

## Step 1 — provider adaptor and a one-layer model

Target exactly `meta-llama/Llama-3.3-70B-Instruct`.

- Confirm the adaptor's provider key/layout conversion and Llama 3 scaled-RoPE preparation against the
  Hugging Face checkpoint, on host, before you put anything on the mesh. A weight-layout error that
  reaches silicon costs an hour per iteration; on host it costs a minute.
- Build the one-layer model and get it onto the mesh. Prove construction, prefetcher sealing, CCL
  resource resolution, and clean teardown *before* you look at a single PCC number.
- If the checkpoint cannot be resolved, skip rather than invent: follow the
  `hf_config_or_skip` convention in `models/common/tests/models/galaxy/galaxy_hardware.py`. A skip is
  an honest result; a synthetic-weight "pass" recorded as model evidence is not.

## Step 2 — validate one Llama block, decode and prefill

The gate is **PCC ≥ 0.99 against an independent Hugging Face reference**, plus **KV-cache PCC ≥ 0.99**.
Use the same HF references the 1D suites use (`models/common/tests/modules/_hf_reference.py`) rather
than a hand-written re-implementation — Milestone A found that hand-written references hide errors on
both sides.

Cover, at minimum:

- decode at batch 32, the physical Galaxy batch, with users placed across columns;
- single-row prefill at 128, then 2048;
- the K and V cache contents after each, not just the block output.

Bisect by sub-module when a block fails. The individual modules are qualified, so a block failure is
almost certainly composition: placement between modules, a resource key that resolves to the wrong
plan, or the RoPE pairing above. Compare the residual stream at each boundary against the reference
before you suspect a module.

**Three runs in fresh processes** before you record any of this as evidence. If a case flips across
processes, you have found a defect of exactly the kind D1 and D3 were — chase it, do not average it.

## Step 3 — scale to 80 layers, and the direct demo

- Full-model prefill plus first decode token.
- Teacher-forced decode.
- Batch 1 and batch 32.
- The direct demo through `models/common/models/galaxy/direct_runner.py` and
  `llama33_70b_galaxy/demo.py`, producing real text.

Then the **Milestone B accuracy gate for Llama**, which is the number this whole job exists to
produce:

```text
teacher-forced, batch 1, prefill 512 / decode 511
  top-1 >= 91%
  top-5 >= 99%
```

Use the teacher-forcing convention already established in `galaxy_hardware.py` and the reference-token
files under `models/tt_transformers/tests/reference_outputs/`, so the number is comparable to the
existing product gates. Record the exact command, the reference file, and the raw counts — not just
the percentage.

If the gate misses, **do not tune to reach it**. Report the number you measured, then diagnose: the
usual causes are a precision recipe, a norm epsilon, a RoPE scaling parameter, or a KV-cache layout —
all of them legitimate to fix, all of them requiring a re-run of steps 2 and 3 afterwards. Weakening
the gate is never a fix.

## Out of scope for this job

Paged KV, prefix-cached/chunked prefill, concat-32 physical batching, device sampling, and the
long-context smokes are plan step 7 and belong to `mb-coverage`. Build nothing for them here. If you
find that step 3 cannot be reached without one of them, record the dependency in your handoff and say
so plainly rather than absorbing the extra scope silently.

Do not touch Qwen. `mb-qwen` runs next and will read your handoff; anything you learn that applies to
both belongs in that document, not in Qwen's code.

## Regression gates

Before recording your own evidence, and again before you finish:

```sh
# host
python -m pytest -q models/common/tests/modules models/common/tests/models/galaxy \
                    models/common/tests/models/llama33_70b_galaxy/test_model_host.py

# boundaries
git diff --name-only <job0-base>..HEAD | grep '_1d\.py'      # must be empty
git diff --name-only <job0-base>..HEAD | grep 'llm_runtime'  # must be empty
```

If you changed a shared 2D module to make the Llama model work, that is a significant event: name the
module, the change, and why config alone could not express it, in both your report and the handoff.
The plan's extension discipline requires that ordering — config first, frozen config value second,
mechanical delegation third, and a written reduction before anything larger.

## Deliverables

1. Working, committed Llama Galaxy model code on the Milestone B branch.
2. Device test files under `models/common/tests/models/llama33_70b_galaxy/` for everything you
   qualified, with mesh, checkpoint, mode, batch and sequence stated in their IDs or markers.
3. `tttv2_milestone_b_evidence/llama/` — raw logs from **every** attempt (never overwrite one), plus
   `REPORT.md` carrying: the results table, the accuracy numbers with their exact commands, every
   defect found with its root cause, the L3 verdict, and anything left `BLOCKED` with its logs.
4. `tttv2_milestone_b_evidence/llama/ENVIRONMENT.md` — commit, branch, firmware, `tt-smi -ls`, build
   flags, and the exact pytest invocations. `mb-coverage` needs this to make paired comparisons later.
5. A checkpoint appended to `tttv2_2d_modules_milestone_b_work_log.md`.
6. `tttv2_milestone_b_briefs/job1_completion_handoff.md` — written for `mb-qwen`: what worked, what
   broke and how you fixed it, which composition surprises to expect, the L3 verdict, and every shared
   change you made that Qwen will inherit.

## Finish condition

One Llama block qualified in decode and prefill at PCC ≥ 0.99, the 80-layer model producing coherent
demo output, the teacher-forced accuracy measured and recorded (whether or not it passes), and the
handoff written. Print the absolute path of your `REPORT.md` as your final line.

If you run out of time, stop cleanly: commit what is proven, write the handoff describing exactly
where you stopped and what the next session should do first. A truthful partial night is worth more
than a rushed claim — `mb-qwen` reads that handoff and will inherit whatever you assert.
