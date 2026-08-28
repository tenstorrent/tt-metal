# Job 0 — Reconcile the Milestone A and Milestone B trees

**No device.** This job must not run a single test on the Galaxy. Everything it needs is host-only,
and the mesh time is worth more to the jobs that follow.

## Mission

Produce one tree in which the Milestone B models sit on top of the *final* Milestone A modules, with
every divergence in `tttv2_milestone_ab_reconciliation.md` either closed or explicitly recorded as
still open. When you finish, `mb-llama` must be able to start against real hardware without first
having to think about the merge.

Read `tttv2_milestone_ab_reconciliation.md` in full before touching anything. It is the input to this
job, not the specification of it — **it was produced by static diff reading and nothing in it was
executed.** The Milestone A branch has moved since. Re-derive every finding you act on.

## Step 1 — establish the real state, do not assume it

You are in `/proj_sw/user_dev/ctr-apbernal/tt-metal`, already on
`apbernal/tttv2_wh_glx_2d_modules_milestone_b`.

```sh
cd /proj_sw/user_dev/ctr-apbernal/tt-metal
git fetch origin
A=gongyu/tttv2_wh_glx_2d_modules             # the finished Milestone A branch, local tip
B=apbernal/tttv2_wh_glx_2d_modules_milestone_b
git log --oneline $(git merge-base $A $B)..$A
git log --oneline $(git merge-base $A $B)..$B
comm -12 <(git diff --name-only $(git merge-base $A $B) $A | sort) \
         <(git diff --name-only $(git merge-base $A $B) $B | sort)
```

Record the exact `A` SHA you reconcile against, in your report and in the work-log checkpoint. **Do
not hardcode a Milestone A SHA from any document, including the reconciliation report** — that branch
was rebased at least once during the analysis, so its commit hashes moved while the merge base
(`de4c8f4e659`) and the substance did not. Re-derive the tip every time.

The Milestone A branch is **finished and read-only** for this job: diff against it and cherry-pick
from it, never write to it, never check it out.

Then re-check each finding against the tree as it actually is now. For each of C1–C10 record one of:
`STILL PRESENT`, `ALREADY FIXED BY A`, or `WAS WRONG` (with the evidence that shows it was wrong).
Finding C3 in particular may already have landed on the A branch as D5.

## Step 2 — the rebase

```sh
git rebase --onto $A $(git merge-base $A $B) $B
```

Expect zero code conflicts: the two commit-level file sets do not intersect. The one document that
collides is `models/common/modules/MILESTONE_A_STATUS.md` — Milestone B adds a "Post-Record Module
Corrections" section to a version of that file the Milestone A branch has since rewritten.

Resolution: **drop Milestone B's section entirely.** Its three facts belong in the rewritten Milestone
A document, and step 3 puts them there. Take the Milestone A side of that file wholesale.

If the rebase produces any *code* conflict, that means the two branches moved under each other in a
way the analysis did not predict. Stop, `git rebase --abort`, and report it — do not resolve a
surprise conflict by guessing which side wins.

## Step 3 — the Milestone A corrections that are currently stowaways in the B commit

Three module-level changes live in the Milestone B commit but amend contracts Milestone A owns and
audits. They must end up visible to the Milestone A audit, not buried in a model commit.

| | Change | File |
| --- | --- | --- |
| **D5** | `wqkv` / `wo` lazy-weight resolution no longer swaps `weight_memory_config` and `wo_weight_memory_config` | `modules/attention/attention_2d.py` ~:517/:525 |
| **C4** | `wo` source shape is `(n_heads * head_dim, dim)`, not `(dim, dim)` | `modules/attention/attention_2d.py` ~:440-455 |
| **C5** | `LMHead2D` accepts a column-local activation width (`dim / 4`) as well as full `dim` | `modules/lm_head/lm_head_2d.py` ~:422 |

For each: if the Milestone A branch already carries it, drop Milestone B's copy. If it does not,
**keep it and isolate it into its own commit** at the base of the Milestone B stack, titled so an
auditor reading the Milestone A diff can find it:

```text
Fix three WH Galaxy 2D module contract defects found during Milestone B
```

Each of the three needs a host regression test that **fails without the change**:

- **D5** — build an `Attention2DConfig` with `weight_memory_config` and `wo_weight_memory_config` set
  to two *different* values and assert each materialized weight carries its own. This is why D5
  survived: every existing test leaves both at `ttnn.DRAM_MEMORY_CONFIG`, so the swap is invisible.
  Write the test so that identical configs could not make it pass.
- **C4** — pin both the decoupled case (`n_heads * head_dim = 8192`, `dim = 5120`) and the square case
  (Llama, `8192 == 8192`), and assert the rejection message for a genuinely wrong shape.
- **C5** — pin both accepted widths and the rejection of a third.

Then append to `models/common/modules/MILESTONE_A_STATUS.md`:

- **D5** as a row in the "Defects found after the premature sign-off" table, in the same four-column
  form as D1–D4 (`Defect` / `How it hid` / `Fix`). Its "how it hid" is the D4 pattern: the only
  hardware test that sets both configs builds both `LazyWeight`s through a helper whose
  `memory_config` parameter defaults to `ttnn.DRAM_MEMORY_CONFIG`, so the two values are equal and the
  swap is a no-op.
- A correction to the `Attention2D` evidence row: the recorded Qwen qualification used a **40-head**
  fixture (`test_attention_2d_wh_galaxy.py:86`, `dim=5120, n_heads=40`), chosen so
  `n_heads * head_dim == dim`. Real Qwen3-32B has **64** heads and `attention_dim = 8192 ≠ dim`. State
  plainly that the decoupled-head-dim path has no hardware evidence and that `mb-qwen` is where it
  gets some.
- A note that the C5 width relaxation is a post-record contract amendment, host-tested only.
- An update to **L3**: point it at the ring/`gather_in0` decode matmul that the Milestone B recipes
  now build (`models/common/models/galaxy/recipes.py`, the 24-core `gather_in0=True` config with
  `hop_cores`), rather than leaving it deferred against an implementation that did not exist when it
  was written. Say clearly that it is wired but unqualified — `mb-llama` is the first job that can
  prove it.

These edits to `MILESTONE_A_STATUS.md` are **explicitly in scope for this job**, overriding the
house rule in the briefs README. Keep them surgical: rows and paragraphs, not a rewrite. That document
is the signed-off Milestone A record and its structure is deliberate — you are appending corrections
to it, not restructuring it.

## Step 4 — C1, the one guaranteed device failure

`models/common/models/galaxy/recipes.py` pins the fused-norm statistics buffer to `CoreCoord(1, 0)`,
which is the pre-D1 default. Milestone A's D1 fix moved the distributed decode norm origin to
`CoreCoord(2, 0)`, made the stats default follow it, and added `_require_fused_stats_placement`, which
raises `ValueError` when the stats shard origin is not the decode-input shard origin.

Both models pass the stale value explicitly (`llama33_70b_galaxy/model.py` and
`qwen3_32b_galaxy/model.py`, `decode_stats_memcfg=decode_placements.norm_stats_memcfg`).

**Preferred fix — remove the coupling, do not re-point it:**

1. delete `distributed_norm_stats_memory_config()`;
2. remove `norm_stats_memcfg` from `GalaxyDecodePlacements`;
3. remove `decode_stats_memcfg=` from both models' distributed norm configs, letting `RMSNorm2D`
   resolve its own stats placement.

The module then owns the invariant its own validator enforces, and the two can never disagree again.
Re-pinning to `(2, 0)` also works and is the fallback if removal turns out to break a construction the
models genuinely need — but it re-creates exactly the coupling that produced D1, so take it only with
a recorded reason.

Also correct, in the same commit:

- the `distributed_norm_decode_memory_config` docstring, which still asserts "`x=1` owns the fused
  distributed-stats circular buffer";
- `tttv2_2d_modules_milestone_b_work_log.md` Checkpoint 5, item 5, which encodes the same dead
  premise;
- add a host test that constructs both models' distributed norm configs and asserts the resolved
  stats origin equals the resolved decode-input origin. That test is the guard that would have caught
  this at merge time.

Then check the neighbours: `_subgrid_cores` anchors the attention decode core sets at
`CoreCoord(1, 0)`, and the norm origin has moved to `(2, 0)`. Confirm on paper that no decode core set
now overlaps the norm shard grid in a way that matters, and write down what you checked. Both are
inside the worker subdevice (`((1,0,3,9), (5,0,6,9))`), so this is an allocator question, not a
partition one.

## Step 5 — the rest, in descending value

- **C6** — promote D3's `semaphore_cores` invariant out of a test-helper docstring
  (`tests/modules/_wh_galaxy_hardware.py::galaxy_mode_plan`) into `GalaxyModePlan` validation in
  `models/common/models/galaxy/resources.py`. Today production only checks the field is not `None`.
  Milestone B's plans already comply; the point is that nothing *makes* them. Its failure mode is an
  indefinite hang, so a fail-closed check is worth more than a comment. If the invariant cannot be
  expressed as a cheap structural check, record why and leave the comment — do not invent a check that
  passes everything.
- **C9** — de-duplicate the Galaxy prefetch geometry. The 12 sender coordinates, the
  `728 * 1088` global-CB size, and the dummy sender/receiver mapping now exist in both
  `tests/modules/_wh_galaxy_hardware.py` and `models/common/models/galaxy/{recipes,prefetch}.py`.
  Make the test helper import from the production module. Do not move production code into tests.
- **C2** — verify, do not change: Qwen's head-local Q/K norm config passes interleaved DRAM explicitly
  and now agrees with D2's new `HEAD_LOCAL` default. Add a host assertion pinning the resolved
  head-local decode placement so the agreement is not accidental.
- **C8 / L1** — do not redesign the global-CB ownership. Record in your report that Milestone B
  creates one prefetcher per model and hands contexts to every layer, so L1's teardown ordering now
  governs whole-model cleanup, and flag it as the first thing to suspect if `mb-llama` sees an L1 OOM
  on a second model construction in one process.

## Regression gates

Everything here is host-only. All of it must be green before you finish.

```sh
python -m pytest -q \
  models/common/tests/modules/attention/test_attention_2d.py \
  models/common/tests/modules/lm_head/test_lm_head_2d.py \
  models/common/tests/modules/rmsnorm/test_rmsnorm_2d.py \
  models/common/tests/modules/mlp/test_mlp_2d.py \
  models/common/tests/modules/sampling/test_sampling_2d.py \
  models/common/tests/modules/prefetcher/test_prefetcher_2d.py \
  models/common/tests/models/galaxy

python -m pytest -q \
  models/common/tests/models/llama33_70b_galaxy/test_model_host.py \
  models/common/tests/models/qwen3_32b_galaxy/test_model_host.py

python -m pytest -q models/common/tests/llm_runtime
python -m pytest -q models/common/tests/modules       # full module set, 1D included
```

**The four new Milestone B host suites and the two updated module suites have never been executed at
all** — that is the Milestone B author's own recorded risk #1, and this job is the first opportunity.
Expect real failures here that have nothing to do with the merge. Fix them; that is the work. If one
is a genuine design problem rather than a slip, record it as `OPEN` with a diagnosis rather than
forcing it green.

Finally, prove the boundaries mechanically and paste the output into your report:

```sh
git diff --name-only $A..HEAD | grep '_1d\.py'                    # must be empty
git diff --name-only $A..HEAD | grep 'llm_runtime'                # must be empty
git grep -n "demos.llama3_70b_galaxy\|models.llama33_70b\b\|models.qwen3_32b\b" -- \
    models/common/models/galaxy models/common/models/*_galaxy      # must be empty
```

## Deliverables

1. A rebased, committed `apbernal/tttv2_wh_glx_2d_modules_milestone_b`, with the contract-defect
   commit isolated at its base.
2. `tttv2_milestone_b_evidence/reconcile/REPORT.md` — the C1–C10 disposition table (one row per
   finding, with `STILL PRESENT` / `ALREADY FIXED` / `WAS WRONG` and what you did), the Milestone A
   SHA you rebased onto, every host suite result verbatim, and the boundary-check output.
3. The surgical `MILESTONE_A_STATUS.md` edits from step 3.
4. A checkpoint appended to `tttv2_2d_modules_milestone_b_work_log.md`.
5. `tttv2_milestone_b_briefs/job0_completion_handoff.md` — for `mb-llama`: the exact commit to start
   from, anything left `OPEN`, and every assumption you had to make.

## Finish condition

The rebase is committed, every host suite listed above is green or has a recorded `OPEN` diagnosis,
the three boundary greps are empty, and the handoff exists. Print the absolute path of your
`REPORT.md` as your final line.

Do not start any Milestone B hardware work. That is `mb-llama`'s job, and it needs the mesh
exclusively.
