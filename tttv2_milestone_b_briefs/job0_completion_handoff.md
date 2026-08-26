# Job 0 → `mb-llama`: completion handoff

Written 2026-08-26 by `job0/reconcile`, unattended, host-only.
Full account: `tttv2_milestone_b_evidence/reconcile/REPORT.md`.

## Start here

```sh
cd /proj_sw/user_dev/ctr-apbernal/tt-metal
git rev-parse --abbrev-ref HEAD     # apbernal/tttv2_wh_glx_2d_modules_milestone_b
git log --oneline -6
```

**Start from the branch tip of `apbernal/tttv2_wh_glx_2d_modules_milestone_b`.** Already checked
out, working tree clean of tracked changes. Do not rebase again.

The last **code-bearing** commit is `52def65194c3938ed6e5cb6f52661ec3a3a15547`; every commit after it
on this branch is documentation only (the work-log checkpoint and this evidence package), so the tree
you build and run is that one.

Rebased onto Milestone A tip **`bc6ad03bfc2`**. Stack, contract-defect commit at the base:

```text
(docs)       work log + reconcile evidence
52def65194c  Enforce the D3 semaphore invariant, de-duplicate the prefetch geometry, pin C2   <- last code commit
c8c96558ad2  Derive the fused-norm statistics placement instead of pinning CoreCoord(1, 0)
35fe6f34115  TTTv2 Milestone B: Llama-3.3-70B and Qwen3-32B Galaxy 2D models
a38cc7bf506  Fix three WH Galaxy 2D module contract defects found during Milestone B
bc6ad03bfc2  (Milestone A tip)
```

The pre-rebase Milestone B commit is kept as the local tag `mb-prerebase-backup` (`7c2eaeb4c60`).

## The mesh is clean and free

`tt-smi -glx_reset` was run at 2026-08-26T17:0x and reported
`Re-initialized 32 boards after reset`. 32 devices present, no holders. Nothing of Milestone B has
been run on hardware — that is entirely your job.

## Read this before you select any test

Two of the brief's own gate commands take the mesh. Both bit this job; both are in the report §7.

- **`models/common/tests/models/galaxy` as a directory collects `test_column_user_selector_wh_galaxy.py`.**
  Always pass `--ignore-glob="*_wh_galaxy*.py"` for a host selection. This is the same trap Milestone
  A recorded at P3.
- **`models/common/tests/modules` as a directory collects the 1D hardware matrix** — it will start
  running `test_attention_1d.py[...Mistral-7B-1x2]` on the Galaxy. Milestone A P4 routes that matrix
  to separate hardware deliberately. For a host run, list the files explicitly (the set is in report
  §8).
- **`models/common/tests/models/galaxy/test_plans.py` opens the UMD driver when run as a whole file** —
  topology discovery plus all 32 local chips, no program executed, not reproducible from any single
  node. Harmless alone, but do not run B's host suites while you hold a device session.

## What changed under you, that your bring-up order depends on

1. **The fused-norm stats placement moved, and neither model pins it any more.** `RMSNorm2D` resolves
   its own `decode_stats_memcfg` from `decode_input_memcfg`; `plans.py` derives the persistent
   all-gather buffer from the same residual placement via
   `distributed_norm_stats_memory_config(placements.residual_memcfg)`. If you see
   `ValueError: fused decode stats buffer must be L1-sharded on the first core of the norm input
   shard grid`, something re-introduced an independent stats placement — that is C1/D1 coming back,
   not a new bug.
2. **`GalaxyModePlan` now rejects a `semaphore_cores` narrower than `worker_cores`** unless
   `allow_narrow_semaphore_cores=True`. If a plan you build raises
   `semaphore_cores must cover the worker subdevice`, that is Milestone A defect D3 being caught at
   construction instead of hanging for 2700 s. Set the flag **only** for a collective that binds its
   semaphore to a grid it owns (the fused RMS all-gather).
3. **`Attention2D`'s `wo` contract is `(n_heads * head_dim, dim)`.** A `(dim, dim)` weight is now
   rejected unless the two coincide.
4. **`_wh_galaxy_hardware.py` imports its prefetch geometry from
   `models/common/models/galaxy/prefetch.py`.** Proven to build a field-identical
   `Prefetcher2DConfig`, but it is now one definition, so a production change moves the tests too.

## Suggested first hour, and why

Work-log Checkpoint 12's order still holds, with C1 now closed on paper but unproven on silicon:

1. **The fused decode norm inside a real model layer** — the C1 site. It is the one place where a
   wrong answer here is a hard `ValueError`, not silent corruption, so it fails fast and cheap.
2. **RoPE composed with `Attention2D`** — Checkpoint 5 risk #2, and the Milestone B author's own
   prediction for the first hardware failure. Milestone A qualified attention with an identity rotary.
3. **The ring/`gather_in0` decode matmul on the prefetch subdevice partition** — L3/C7. It is wired
   (`recipes.py::ring_matmul_program_config`, 24 cores, `hop_cores`, `gather_in0=True`) and
   `ring_cores()` was verified to lie entirely inside the worker subdevice, 24/24. **You are the
   first job that can prove it.** Milestone A's only device evidence here is the terminal-FAILED
   `attention_decode_with_active_prefetch`, recorded against the old `(7,1)` grid.
4. **Repeated model construction and teardown** — L1/C8, below.

**Apply the three-runs-in-fresh-processes rule.** Three of Milestone A's four defects presented as
intermittent *passes*.

## Open items you inherit

| | Item | What to do with it |
| --- | --- | --- |
| **O1** | The 1D device regression matrix was not run | Not yours. Milestone A P4, on separate hardware. No `*_1d.py` implementation file changed — boundary grep empty |
| **O2** | Five pre-existing host failures in `deepseek_r1_distill_qwen_14b`, `llama32_3b`, `llama33_70b`, `qwen25_7b`, `qwen2_7b` | Not yours. Proven independent of this job: those packages are untouched by the diff and import nothing it changed. If you run all of `tests/models`, expect them |
| **O3** | D5 is latent, not live | The `wqkv`/`wo` memory-config swap is real code but unreachable behind `_require_exact_weight_policy`. No test can fail without the fix, and none pretends to |
| **O4** | **The decoupled 64-head Qwen geometry has zero hardware evidence** | Not yours either — it is `mb-qwen`'s. Flagged because the recorded Milestone A "Qwen3-32B PCC ≥ 0.99" row used a 40-head fixture chosen so `n_heads * head_dim == dim`. The status page now says so |
| **O5** | An L1 OOM on a *second* model construction in one process | **Suspect L1/C8 first.** Each model creates one prefetcher and hands `Prefetcher2DContext` handles to every module config of all 80 layers; the global CB's L1 is reclaimed only when the last handle dies. Teardown order in `close()` is correct, but ~55 MB can stay resident after a truthful `owned_resources == ()`. One model per process sidesteps it |

## Assumptions made, that you may need to revisit

1. **C1 was fixed by derivation, not deletion.** The brief asked for
   `distributed_norm_stats_memory_config()` to be deleted outright. It could not be: `plans.py:165`
   needs a memory config to size the persistent buffer that `_require_fused_stats_placement`
   inspects, and neither the brief nor the reconciliation report mentions that call site. The
   function now derives the origin from the residual placement it is handed. Same guarantee, nothing
   re-pinned to `(2, 0)`. **If a model genuinely needs an independent stats placement, this is the
   decision to revisit.**
2. **The `README.md` rebase conflict** (unpredicted) was resolved by keeping Milestone A's closing
   sentence and Milestone B's two additive paragraphs. A judgement call on a document.
3. **`SKIP=prefer-expect-error` on every commit.** That hook flags ~35 pre-existing `pytest.raises`
   blocks in two suites Milestone A never touched. Converting them all is a large unrelated
   refactor. Noted in each commit message; every other hook ran. `black` and `isort` reformatted
   several Milestone B files, which had never been linted — so the Milestone B commit's content
   differs from `mb-prerebase-backup` by formatting in six files.
4. **Gate 4 was narrowed** to files with no `indirect=True` device parametrization, minus
   `moe/test_generalized_moe_gate.py`, which opens a device through a plain fixture.

## Do not

- Do not rebase or re-order this stack; the contract-defect commit sits at the base on purpose, so
  the Milestone A audit can see it.
- Do not edit `models/common/modules/MILESTONE_A_STATUS.md` — job0 and job4 own it, `mb-llama` does
  not. Put proposed text in your report.
- Do not touch `models/common/modules/**/*_1d.py` or `models/common/llm_runtime/**`.
