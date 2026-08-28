# Job 2 — Milestone B steps 4–6: Qwen3-32B on WH Galaxy `(8, 4)`

**Device job.** You have the Galaxy exclusively. One pytest process at a time.

## Mission

The same work `mb-llama` did, for `Qwen/Qwen3-32B`: provider adaptor and one-layer model, one block
validated in decode and prefill, then the full model and a direct demo with recorded teacher-forced
accuracy. Plan steps 4, 5 and 6.

**Read `tttv2_milestone_b_briefs/job1_llama_state_for_qwen.md` first, in full.** Llama went first
precisely so that you inherit its composition fixes instead of rediscovering them. That file is the
distilled state of the Llama job as of its third and final attempt: the numbers a passing Galaxy
model produces, the six edits your model file still needs, what you inherit for free, and the
harness rules that each cost a device run to learn. If it does not exist, stop and report
`BLOCKED (mb-llama did not complete)`.

**Do not read the `job1_completion_handoff*.md` files or `tttv2_milestone_b_evidence/llama/REPORT.md`
end to end.** `mb-llama` took three attempts and appended to the same package each time, so it is
~5,500 lines, and its earliest sections assert things that later
attempts disproved — that the mesh is dead, that no PCC number exists, that L3 is still live, that
the Qwen checkpoint is not on this host. The distilled file resolves every one of those and indexes
the sections worth opening for detail. Where the two disagree, **the distilled file is later**.

Everything the Llama job learned about module composition — residual placement between modules,
resource-key resolution, prefetch sealing order, the RoPE pairing, teardown ordering — applies here
unchanged. What differs is Qwen's architecture, and that difference is the whole risk of this job.

## Before your first device run: the checkpoint resolves under exactly one `HF_HOME`

Qwen3-32B **is** on this machine. A previous attempt of this job reported that it was not and spent
the night on host work; it searched one cache.

```sh
export HF_HOME=/localdev/ctr-apbernal/hf_data     # reaches Qwen3-32B (62 G, 17/17 shards) AND Llama
# NOT /proj_sw/user_dev/hf_data                   # Llama only - Qwen raises, and hf_config_or_skip skips
```

The Llama harness scripts you will copy (`tttv2_milestone_b_evidence/llama/run3.sh`,
`device_run.sh`, `run_sequence.sh`) all hardcode the `/proj_sw` value, and Llama's `ENVIRONMENT.md`
states that either path "reaches the same shards" — true for Llama, false for Qwen. **Change the
export in the first script you copy**, and treat a `skipped` in a run you meant to count as a
failure of the run, not a result.

## Also before your first device run: Llama's model-code fixes are not in your tree

Anything Llama fixed in a **shared module** you get by construction. Anything it fixed in
`llama33_70b_galaxy/model.py` or its adaptor, you do not — and three of those are verified present
in your tree today, each of which produces a **wrong number with no error of any kind**: the
prefetcher registering weights the ring never consumes, the rotary defaulting to the non-fused pair
(which writes an infinite K into the cache), and a decode LM head still wired the pre-ring way. Two
further warnings from Llama's earlier attempts are already discharged in your tree and must not be
re-spent.

All six are enumerated with file, line, and the symptom each one produced on silicon in
`job1_llama_state_for_qwen.md` §3. Apply them while you are on host, not after your first confusing
PCC.

## What makes Qwen different, and therefore risky

### 1. The 64-head decoupled geometry has no hardware evidence at all

This is the headline risk. Qwen3-32B has `hidden = 5120` but **64** attention heads of
`head_dim = 128`, so `attention_dim = n_heads * head_dim = 8192 ≠ dim = 5120`, and `wo` is
`(8192, 5120)` rather than square.

Milestone A's recorded "Qwen3-32B attention qualified, PCC ≥ 0.99" was measured against a **40-head**
fixture (`test_attention_2d_wh_galaxy.py:86`, `dim=5120, n_heads=40`), chosen so that
`n_heads * head_dim` happened to equal `dim`. The square case is the only one with silicon evidence.
Job 0 relaxed the `Attention2D` `wo` contract to express the real geometry and corrected the Milestone
A status page to say so.

So: treat Qwen attention as **unqualified**, not as a re-run. Give the decoupled path the same
scrutiny `mb-llama` gave the whole block — every placement between QKV, head creation, SDPA, concat
and WO is a place where a `dim`-vs-`attention_dim` confusion hides, and Llama cannot have caught any
of them because for Llama the two are equal.

Specifically check, before trusting a PCC:

- the WO input width is `attention_dim / GALAXY_ROWS`, not `dim / GALAXY_ROWS`;
- the head-concat output width matches what WO expects;
- the residual added after WO is `dim`-wide, not `attention_dim`-wide;
- `dram_sharded_weight_memory_config(mesh_device, geometry.local_attention_dim, geometry.local_dim)`
  is what `wo` actually gets — this is the pairing that D5 was silently breaking.

### 2. Per-head Q/K normalization

Qwen3 normalizes each `head_dim`-wide head independently: no column reduction, no collective, a plain
`rms_norm` over the created heads. `Attention2D` rejects any geometry other than
`RMSNorm2DGeometry.HEAD_LOCAL` for these, and rejects a weight whose width is not `head_dim`.

Milestone A's D2 fix changed how `HEAD_LOCAL` resolves — decode now defaults to interleaved DRAM like
prefill, and the sharded `decode_progcfg` / `decode_stats_memcfg` defaults are no longer emitted for
that geometry. Qwen's config passes interleaved DRAM explicitly and therefore agrees with the new
default, but **the composed path has never run**: D2's own defect was that head-local decode aborted
in op validation before producing any numerical result at all, so there is no prior Qwen Q/K norm
number anywhere to compare against.

Validate Q/K norm independently — its own geometry, its own PCC against the HF reference — *before*
enabling it inside the block. That ordering is what the plan's risk section asks for, and it is the
difference between a one-hour diagnosis and a night of bisection.

### 3. Decode ring widths

The scattered W1/W3 *placement* is padded to the 24-core ring (960 columns for both models, identical
to the qualified Llama recipe), while the resource *key* uses the logical width TTNN reports — 960 for
Llama but **800 for Qwen**. The Milestone B author traced this against the TTNN op source and
concluded it is correct, but it was never executed. If a Qwen decode all-gather cannot find its
resource, inspect this pair first; it is the one place where the two models' resource keys legitimately
differ.

### 4. No fused QKV bias

Neither target checkpoint has one, and `Attention2D` validates a bias against the projection's
DRAM-sharded weight placement, which a bias vector cannot satisfy. The pinned revision's own
`config.json` says `attention_bias: false` (checked on disk, revision `9216db5781bf`), so this
should not arise — but if the checkpoint you resolve turns out to carry QKV bias tensors, **stop and
report it**: supporting one needs a bias placement field on the module config, which is a contract
change, not a fix to make in passing.

## Sequence

Follow the same shape as `mb-llama`, and do not skip the early gates because Llama passed them.

1. **Adaptor and one-layer model.** Host-verify the provider key/layout conversion and the Qwen RoPE
   parameters before anything reaches the mesh. Qwen's adaptor is independent of Llama's by design —
   it must not import it, and the plan forbids extracting shared code from either merely to avoid
   writing the other. Apply the model-code deltas above in this step: they are host-visible edits
   and every one of them is cheaper here than on the mesh.
2. **Q/K norm alone**, per risk 2 above.
3. **One Qwen block, decode and prefill**, PCC ≥ 0.99 against an independent HF reference, KV-cache
   PCC ≥ 0.99. Decode batch 32; prefill 128 then 2048. Three runs in fresh processes before recording
   anything.
4. **Full model and direct demo.** Prefill plus first decode token, teacher-forced decode, batch 1 and
   batch 32, real text from `qwen3_32b_galaxy/demo.py`.

Then the **Milestone B accuracy gate for Qwen**:

```text
teacher-forced, batch 1, sequence length 512
  top-1 >= 89%
  top-5 >= 97%
```

Same conventions as Llama — the teacher-forcing helper in `galaxy_hardware.py`, the reference-token
files under `models/tt_transformers/tests/reference_outputs/`, exact commands and raw counts recorded,
not just percentages. If the gate misses, report the measured number and diagnose. Never tune the
gate.

## Out of scope

Plan step 7 — paged KV, prefix cache, concat-32, device sampling, long context — belongs to
`mb-coverage`. Do not build it here.

Do not modify the Llama package. If a shared change under `models/common/models/galaxy/` is needed for
Qwen and it alters Llama's behaviour, you must re-run Llama's block and full-model gates before
finishing, and say so in your report. A shared-code change that silently invalidates the previous
night's evidence is the failure mode to avoid.

## Regression gates

**Do not pass `models/common/tests/modules` as a directory** — it collects the 1D device suites and
takes the mesh for ten minutes. `mb-llama` lost a run to that. Use its selection, which is a script
in the tree, plus your own host suites:

```sh
# host — tttv2_milestone_b_evidence/llama/host_gate.sh is the Llama selection (565 passed, exit 0);
# it lists the 2D module host suites explicitly and passes --ignore-glob="*_wh_galaxy*.py".
# Run that script, then add Qwen's:
python -m pytest -q -rA --color=no -p no:cacheprovider --ignore-glob="*_wh_galaxy*.py" \
                    models/common/tests/models/qwen3_32b_galaxy/test_model_host.py \
                    models/common/tests/models/qwen3_32b_galaxy/test_hf_conversion_host.py

# boundaries
git diff --name-only <job0-base>..HEAD | grep '_1d\.py'      # must be empty
git diff --name-only <job0-base>..HEAD | grep 'llm_runtime'  # must be empty
git grep -n "models.common.models.llama33_70b_galaxy" -- models/common/models/qwen3_32b_galaxy
                                                              # must be empty
```

If you touched shared Galaxy code, re-run the Llama device gates before you finish and record both
results. The node ids, their wall-clock costs and the runner that drives them are in
`job1_llama_state_for_qwen.md` §10. They are cheap once the weight cache is warm; Llama's evidence
is three fresh processes per gate and a shared change that is not re-qualified invalidates all of
it.

## Deliverables

1. Working, committed Qwen Galaxy model code on the Milestone B branch.
2. Device test files under `models/common/tests/models/qwen3_32b_galaxy/`, with mesh, checkpoint,
   mode, batch and sequence in their IDs or markers.
3. `tttv2_milestone_b_evidence/qwen/` — raw logs from every attempt, plus `REPORT.md` with the results
   table, the accuracy numbers and their commands, the **64-head geometry verdict stated explicitly**,
   the Q/K norm result, the ring-width finding, and anything `BLOCKED`.
4. `tttv2_milestone_b_evidence/qwen/ENVIRONMENT.md`, same form as Llama's.
5. A checkpoint appended to `tttv2_2d_modules_milestone_b_work_log.md`.
6. `tttv2_milestone_b_briefs/job2_completion_handoff.md` — written for `mb-coverage`: both models'
   working entry points, the exact commands that produce a passing decode and prefill for each, any
   shared change that affected Llama, and every open item.

## Finish condition

One Qwen block qualified in decode and prefill at PCC ≥ 0.99 with the **real 64-head geometry**, the
full model producing coherent demo output, the teacher-forced accuracy measured and recorded, Llama's
gates still green if you touched shared code, and the handoff written. Print the absolute path of your
`REPORT.md` as your final line.

If you run out of time, commit what is proven and write the handoff describing exactly where you
stopped. `mb-coverage` needs both models working; if Qwen is not there yet, saying so is what lets the
next job plan around it.
