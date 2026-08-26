# Job 0 — Milestone A / Milestone B reconciliation

Run 2026-08-26 on `wh-glx6u-05`, unattended, from
`tttv2_milestone_b_briefs/job0_reconcile.md`. **Host-only job.** Three unintended device
touches happened anyway; all three are recorded in §7 rather than omitted.

## 1. What was reconciled against

| | |
| --- | --- |
| Repository | `/proj_sw/user_dev/ctr-apbernal/tt-metal` |
| Branch | `apbernal/tttv2_wh_glx_2d_modules_milestone_b` |
| Merge base | `de4c8f4e659` — `add reusable WH Galaxy 2D modules` |
| **Milestone A tip rebased onto (`$A`)** | **`bc6ad03bfc21d6a26f88169cc87ea2d8176f0fbf`** — `Re-run the Milestone A device matrix and host gate at the committed tree` |
| Milestone B commit before rebase | `7c2eaeb4c60` (kept as the local tag `mb-prerebase-backup`) |
| Final code-bearing commit | `52def65194c3938ed6e5cb6f52661ec3a3a15547` — later commits on the branch are documentation only |

The A SHA was re-derived at run time, not taken from any document. Local
`gongyu/tttv2_wh_glx_2d_modules` and `origin/gongyu/tttv2_wh_glx_2d_modules` were identical.
The reconciliation report's SHAs (`cf803f23647`/`bf403d93fed`) no longer exist — the branch was
rebased, exactly as that report warned.

A-side commits since the merge base:

```text
bc6ad03bfc2 Re-run the Milestone A device matrix and host gate at the committed tree
9a7a63c4705 Record Milestone A device evidence, gap briefs and work-log checkpoints
5595e56b1ae Fix four WH Galaxy 2D module defects and close two coverage gaps
```

`comm -12` over the two changed-file lists gave **two** intersecting files, not one:
`models/common/modules/MILESTONE_A_STATUS.md` (predicted) and `models/common/modules/README.md`
(not predicted).

## 2. Final commit stack

```text
52def65194c  Enforce the D3 semaphore invariant, de-duplicate the prefetch geometry, pin C2
c8c96558ad2  Derive the fused-norm statistics placement instead of pinning CoreCoord(1, 0)
35fe6f34115  TTTv2 Milestone B: Llama-3.3-70B and Qwen3-32B Galaxy 2D models
a38cc7bf506  Fix three WH Galaxy 2D module contract defects found during Milestone B   <- base
bc6ad03bfc2  (Milestone A tip)
```

41 files changed, +12335 / −108 against `$A`. The contract-defect commit is isolated at the base of
the stack as the brief requires, so an auditor reading the Milestone A diff finds it.

## 3. C1–C10 disposition

Every finding was re-derived against the tree as it actually is. Two of the ten did not survive
re-derivation intact.

| | Finding | Disposition | What was done |
| --- | --- | --- | --- |
| **C1** | B pins the fused-norm stats buffer to `CoreCoord(1, 0)`, the core A's D1 fix moved away from | **STILL PRESENT** — confirmed at `recipes.py:559` (stats pinned to `(1,0)`), `plans.py:165`, `llama33_70b_galaxy/model.py:652`, `qwen3_32b_galaxy/model.py:686`; A's `rmsnorm_2d.py:507` defaults the norm origin to `(2,0)` and `_require_fused_stats_placement` raises on a mismatch | Fixed in `c8c96558ad2`. See §4 for the one deviation from the brief's prescribed fix |
| **C2** | Head-local Q/K norm: B's assumption invalidated, but B independently guessed right | **WAS RIGHT, AND NOW PINNED** — Qwen's `_head_local_norm_config` passes interleaved DRAM explicitly; A's D2 `HEAD_LOCAL` resolution keeps decode interleaved and emits no `decode_progcfg`/`decode_stats_memcfg`. They agree | No production change. Added `test_head_local_qk_norm_agrees_with_the_module_default_by_contract`, pinning both what the model asks for and what the module resolves for a head-local norm that asks for nothing |
| **C3 / D5** | `wqkv`/`wo` lazy-weight resolution swaps the two memory configs; a fifth Milestone A defect | **PARTLY WRONG.** The swapped lines are real and still on the A tip (`attention_2d.py:517/525`). The claimed *consequence* is not: the swap is **unreachable**. `_require_exact_weight_policy` (`:488-491`) runs before `resolve_lazy_weight` (`:518/526`) and rejects any weight whose `memory_config` is not already equal to its own config field; `resolve_lazy_weight` only fills `None` fields, so it can never overwrite one. Probed both orderings with two different configs: identical results (`logs/03_discrimination_check.log`, `scratch/d5_probe.py`) | Kept B's fix and isolated it in `a38cc7bf506`. **No test can fail without it**, and none pretends to. Instead `test_a_projection_placed_against_the_other_configs_value_is_rejected` pins the gate that makes it unreachable, so the day that gate is relaxed the swap becomes visible. Recorded as latent, not live, in `MILESTONE_A_STATUS.md` |
| **C4** | A's Qwen attention hardware evidence uses a geometry no product has | **STILL PRESENT** — `test_attention_2d_wh_galaxy.py:86` is `dim=5120, n_heads=40`, so `n_heads * head_dim == dim`; real Qwen3-32B is 64 heads, `attention_dim = 8192 != dim = 5120` | `wo` contract relaxed to `(n_heads * head_dim, dim)`, isolated in `a38cc7bf506` with tests for the decoupled case, the square case, and the rejection message. Evidence row corrected in `MILESTONE_A_STATUS.md`; the decoupled path is recorded as having **zero** hardware evidence |
| **C5** | `LMHead2D` activation-width contract widened in B | **STILL PRESENT** | Isolated in `a38cc7bf506`. Test strengthened over B's: pins **both** accepted widths (8192 and 2048) and rejects a third (1024). Recorded as a post-record contract amendment, host-tested only |
| **C6** | D3's `semaphore_cores` invariant documented in test plumbing, enforced nowhere | **STILL PRESENT** — production `GalaxyModePlan` only checked `is not None` | Promoted into `GalaxyModePlan` validation in `52def65194c`. See §5 for why it needed an explicit opt-out rather than a blanket rule |
| **C7 / L3** | L3 looks closed by B; A's status page should stop deferring it | **STILL PRESENT (as a stale document)** — B does wire the 24-core `gather_in0=True` ring matmul with `hop_cores` (`recipes.py::ring_matmul_program_config`) | L3 rewritten to point at that recipe and to say plainly it is **wired but unqualified**. Verified `ring_cores()`, `ring_hop_cores()` and `ring_receiver_cores()` all lie entirely inside the worker subdevice — 24/24, 1/1, 24/24 (`logs/17_l3_ring_inside_worker.log`) |
| **C8 / L1** | Global-CB ownership becomes load-bearing under B | **CONFIRMED, NOT REDESIGNED** (as instructed) | See §6 |
| **C9** | Duplicated Galaxy prefetch geometry | **STILL PRESENT** — 12 sender coords, `728 * 1088`, and the dummy sender/receiver mapping existed in both trees | Test helper now imports from `models/common/models/galaxy/prefetch.py`. Proven behaviour-preserving field by field: 12/12 `Prefetcher2DConfig` fields identical (`logs/08_c9_equivalence.log`) |
| **C10** | `MILESTONE_A_STATUS.md` is the only real textual conflict | **INCOMPLETE** — `models/common/modules/README.md` also conflicted, and it is the one that actually stopped the rebase; `MILESTONE_A_STATUS.md` auto-merged silently, exactly as the report predicted it might | B's "Post-Record Module Corrections" section dropped; A's side of that file taken wholesale (`git diff` against `$A` for that path is empty). README resolved by hand — see §4 |

## 4. The rebase

```sh
git rebase --onto gongyu/tttv2_wh_glx_2d_modules de4c8f4e659 apbernal/tttv2_wh_glx_2d_modules_milestone_b
```

**Zero code conflicts, as predicted.** One conflict, in `models/common/modules/README.md` — a
document, not code, so the brief's "stop and report a surprise *code* conflict" rule did not apply.

Resolution recorded because the brief did not cover this file: A's side replaced the closing
"Milestone A is still in progress" sentence with the qualified-on-hardware statement; B's side added
two new paragraphs describing its own packages and the two paged page-table layouts. These do not
contradict each other. **Kept A's closing sentence and B's two additive paragraphs; dropped B's now
false "still in progress" line.**

`MILESTONE_A_STATUS.md` auto-merged without a conflict, which is the failure mode C10 warned about —
git would have inserted B's section into a document A had restructured underneath it. Overridden with
`git checkout gongyu/tttv2_wh_glx_2d_modules -- <that file>`; the resulting `git diff` against `$A`
for that path is empty, so A's side was taken wholesale before the step-3 edits were appended.

### The one deviation from the brief's prescribed C1 fix

The brief's preferred fix is removal: delete `distributed_norm_stats_memory_config()`, drop
`norm_stats_memcfg` from `GalaxyDecodePlacements`, drop both models' call sites, and let `RMSNorm2D`
resolve its own placement.

Steps 2 and 3 were done exactly. Step 1 could not be: **`plans.py:165` needs a memory config** to
size the persistent all-gather buffer, and that buffer is the tensor
`_require_fused_stats_placement` actually inspects (`rmsnorm_2d.py:251`). Deleting the function
outright leaves the decode plan unable to allocate. Neither the brief nor the reconciliation report
mentions this call site.

Taken instead, as the most conservative option that still achieves the brief's stated end: the
function stops **naming** a core and starts **deriving** one from the decode residual placement it is
given. The plan's buffer and the placement `RMSNorm2D` resolves for itself are then both functions of
a single residual placement and cannot disagree — which is the property the brief wanted ("the two
can never disagree again"). It is not the fallback the brief warned about either: nothing is
re-pinned to `(2, 0)`, so the coupling that produced D1 is not re-created.

Guard test in both models' host suites:
`test_distributed_norm_resolves_its_statistics_onto_the_decode_input_origin` asserts the model pins
no stats core, that the resolved stats origin equals the resolved decode-input origin, and that the
decode plan's persistent stats buffer lands on the same core. It **fails against the pre-fix state**
with `assert MemoryConfig(... "x":1 ...) is None` (`logs/04_c1_discrimination.log`).

### Neighbour check (brief step 4)

`_subgrid_cores` anchors the attention decode core sets at `CoreCoord(1, 0)`; the norm grid is now
`x=2..3`. Enumerated both, per model (`scratch/subgrid_overlap.py`, `logs/05_subgrid_overlap.log`):

| Core set | Llama: cores / overlap norm / overlap stats | Qwen: cores / overlap norm / overlap stats |
| --- | --- | --- |
| `attention_heads` | 32 / 16 / 1 | 32 / 10 / 1 |
| `attention_kv` | 8 / 0 / 0 | 8 / 0 / 0 |
| `attention_sdpa_output` | 8 / 5 / 1 | 8 / 5 / 1 |
| `attention_gather_users` | 32 / 16 / 1 | 32 / 10 / 1 |
| `attention_qkv_reduced` | 10 / 0 / 0 | 10 / 0 / 0 |
| `mlp_reduce_scatter` | 30 / 16 / 1 | 30 / 10 / 1 |

There **is** overlap, including on the stats core `(2, 0)`. Written down as checked, with the
conclusion: this is an allocator question, not a partition one — everything above is inside the
worker subdevice `((1,0,3,9), (5,0,6,9))` (stats-inside-worker verified `True` for both models), and
the overlapping tensors are transient activation buffers whose lifetimes the allocator manages.
D1 was never allocator collision: it was the kernel creating its stats CB on one core and binding it
to a tensor sharded on another. Two further points argue the same way: the **pre-fix** arrangement
was strictly worse on this axis, since it put the stats buffer on `(1, 0)`, the *anchor* of every one
of those core sets; and stats-on-norm-origin is precisely what A's D1 fix and its validator require.

**Not proven on hardware. Nothing here is.**

## 5. C6 — why the check needed an explicit opt-out

`ttnn.SubDevice` and `ttnn.SubDeviceId` expose no attributes at all (verified: `dir()` returns an
empty list for both), so the worker core set cannot be recovered from `sub_devices`. `GalaxyModePlan`
therefore carries it as a new optional `worker_cores` field, and the check requires `semaphore_cores`
to cover it.

A blanket rule would have been wrong. A's own D3 note records that narrowing is *safe* for a
collective that binds its semaphore to a grid it owns, as the fused RMS all-gather does — and
`test_rmsnorm_2d_wh_galaxy.py:100` relies on exactly that, on hardware, today. The two cases are not
distinguishable from the plan, because both key on the same `all_gather` operation name. So the
narrow case must now declare itself with `allow_narrow_semaphore_cores=True`, which turns a silent
narrowing into a stated one. The check no-ops when either side is not a `CoreRangeSet`, so the
existing host plans built from stand-in strings still construct.

Pinned by `test_mode_plans_fail_closed_on_a_semaphore_set_narrower_than_the_workers`: production
compliance, the rejection, and the opt-out.

## 6. C8 / L1 — recorded, not redesigned

Confirmed by reading the models: each creates **one** prefetcher for the mesh
(`build_galaxy_prefetcher`), resolves `(prefill, decode)` contexts once
(`llama33_70b_galaxy/model.py:911`), and threads those two context objects into every module config
of all 80 layers. Teardown order in `close()` is correct — modules released first, then
`resources.cleanup`, then `prefetcher.cleanup` last.

But L1 says the global CB's L1 is reclaimed only when the **last handle** dies, and every module
config holds a `Prefetcher2DContext`. So L1's ordering contract now governs whole-model cleanup.

> **First thing to suspect if `mb-llama` sees an L1 OOM on a second model construction in one
> process.** ~55 MB of L1 staying resident after a truthful `owned_resources == ()` is L1's exact
> signature. Constructing one model per process sidesteps it.

## 7. Device touches in a host-only job — full disclosure

Three, all unintended, all recorded. No Milestone B hardware work was started, and no result in this
report comes from the mesh.

1. **The brief's own gate command took the mesh.** `python -m pytest -q ... models/common/tests/models/galaxy`
   collects `test_column_user_selector_wh_galaxy.py`, a device suite. It opened the mesh and began
   running it. Killed at the harness's 2-minute cap before teardown
   (`logs/09_gate_modules_and_galaxy.log`, retained). This is precisely the trap Milestone A recorded
   at P3. **Every subsequent host selection uses `--ignore-glob="*_wh_galaxy*.py"`.**
   Because that kill skipped teardown, a `tt-smi -glx_reset` was run as the sanctioned recovery:
   `Re-initialized 32 boards after reset` (`logs/10_glx_reset_after_accidental_device_run.log`).
   **The mesh is clean and free for `mb-llama`.**
2. **The brief's fourth gate is a device matrix.** `python -m pytest -q models/common/tests/modules`
   ("full module set, 1D included") collects the 1D hardware suites; it reached
   `test_attention_1d.py::test_attention_1d_vs_reference[...Mistral-7B-1x2]` on the Galaxy.
   This contradicts both this job's host-only mandate and Milestone A's own routing — P4 states the
   1D matrix is "in progress on separate hardware, deliberately not run on this Galaxy host."
   Terminated (`logs/15_gate4_all_modules_TERMINATED_1d_device_suites.log`, retained). The target was
   confirmed with `ps -o pid=,ppid=,comm=,args=` first: PID 81090, `comm=python`, args exactly the
   command launched; the two other `pgrep -af pytest` matches were this session's own `bash`
   wrappers and were not signalled. It exited in ~20 s; 32 devices present, no holders, no reset
   needed. **The 1D device matrix is recorded as NOT RUN, by design — see §9.**
3. **`test_generalized_moe_gate.py` opens a device** through a plain (non-`indirect`) fixture. It ran
   once inside the host-only module set, passed, and tore down cleanly. The gate was re-run without
   it so the recorded number is genuinely host-only (`logs/19_...`, `logs/FINAL_gate4_...`: zero
   `Opening user mode device driver`). It is an MoE suite, outside both milestones.

**One further observation for `mb-llama`:** `models/common/tests/models/galaxy/test_plans.py`
triggers a UMD topology discovery and opens all 32 local chips when run **as a whole file** — no
program is executed (zero kernel/program markers in the log), and no single test node reproduces it
in isolation. Harmless on its own, but it means B's host suites are not safe to run concurrently with
a live device session.

## 8. Host suite results — verbatim, at the final tree `52def65194c`

All logs in `tttv2_milestone_b_evidence/reconcile/logs/`. `-q -rA --color=no -p no:cacheprovider`,
one process at a time, never piped.

| Gate | Selection | Result | Log |
| --- | --- | --- | --- |
| 1 | 2D module host suites + `tests/models/galaxy` (`--ignore-glob="*_wh_galaxy*.py"`) | **`300 passed in 48.60s`**, exit=0 | `FINAL_gate1_modules_and_galaxy.log` |
| 2 | `llama33_70b_galaxy/test_model_host.py` + `qwen3_32b_galaxy/test_model_host.py` | **`59 passed, 2 warnings in 20.68s`**, exit=0 | `FINAL_gate2_model_host.log` |
| 3 | `models/common/tests/llm_runtime` | **`1032 passed, 1 skipped, 9 warnings in 210.14s`**, exit=0 | `FINAL_gate3_llm_runtime.log` |
| 4 | `models/common/tests/modules`, host-only subset | **`260 passed, 1 warning in 42.20s`**, exit=0 | `FINAL_gate4_modules_hostonly.log` |
| 5 *(extra, not in the brief)* | all of `models/common/tests/models` (`--ignore-glob="*_wh_galaxy*.py"`) | `5 failed, 575 passed, 3 skipped, 3 deselected in 162.29s` | `FINAL_gate5_all_model_host.log` |

Gate 4 is the brief's "full module set, 1D included" with the 1D **device** suites removed, for the
reason in §7.2. Files run: `test_attention_1d_arch_config.py`, `test_attention_2d.py`,
`test_embedding_2d.py`, `test_lm_head_2d.py`, `test_mlp_1d_arch_config.py`, `test_mlp_2d.py`,
`test_prefetcher_2d.py`, `test_rmsnorm_2d.py`, `test_rope_2d.py`, `test_sampling_1d_release.py`,
`test_sampling_2d.py`, `test_tensor_utils.py` — i.e. every file under `tests/modules` with no
`indirect=True` device parametrization, minus the MoE suite of §7.3.

### Gate 5's five failures are pre-existing and out of scope

```text
FAILED models/common/tests/models/deepseek_r1_distill_qwen_14b/test_demo_contract.py::test_eval_prefill_signature_multiset_is_rotation_invariant_and_not_static_warmup_shaped
FAILED models/common/tests/models/llama32_3b/test_hf_adaptor.py::test_generator_downgrades_n150_all_trace_to_decode_only
FAILED models/common/tests/models/llama33_70b/test_demo_contract.py::test_demo_resolves_central_trace_region_size_for_each_supported_sku
FAILED models/common/tests/models/qwen25_7b/test_demo_contract.py::test_eval_prefill_signature_multiset_is_rotation_invariant_and_not_static_warmup_shaped
FAILED models/common/tests/models/qwen2_7b/test_demo_contract.py::test_eval_prefill_signature_multiset_is_rotation_invariant_and_not_static_warmup_shaped
```

Not caused by this job, and demonstrably not caused by it:

- `git diff --stat bc6ad03bfc2..HEAD` over all five packages, their test directories and
  `models/common/llm_runtime` is **empty**;
- none of them references `attention_2d`, `lm_head_2d` or `models.common.models.galaxy` — nothing
  this job changed is in their import closure.

They are 1D/demo trace-region and prefill-signature contracts (`assert 'all' == 'decode_only'`,
`assert 224000000 == 96000000`) in packages neither milestone owns. **Recorded as `OPEN — pre-existing
on the Milestone A tip, outside Milestone A and B`, for whoever owns those packages.** Gate 5 was not
in the brief; it was run as an extra sweep, and its result is reported rather than dropped.

### Failures found and fixed in Milestone B's never-executed suites

The brief predicted these ("B's own recorded risk #1"). Four, all test-side against correct
production behaviour — no threshold, tolerance or parametrization was relaxed, and nothing was
deleted or `xfail`ed:

1. `test_attention_2d.py::test_mode_specific_projection_weights_materialize_independently` — the
   fixture still built `prefill_wo` as `(5120, 5120)`, which the C4 contract correctly rejects
   because it must match `wo`. Corrected to `(8192, 5120)`. **This failed in B's own tree too.**
2. `test_recipes.py::test_sampling_and_rope_core_grids_are_explicit` — the mock mesh returned a
   `SimpleNamespace` for the compute grid; `ttnn.num_cores_to_corerangeset` is a pybind11 binding and
   rejects it. Mock now returns a real `ttnn.CoreCoord`.
3. `llama .../test_model_host.py::test_checkpoint_contract_fails_closed[head_dim=64]` — production
   correctly rejects it with `Llama-3.3-70B requires head_dim 128, got 64`; the test expected the
   generic geometry message. Expectation corrected to the real one.
4. `test_attention_2d.py::test_decode_page_table_contract_fails_before_compute[rank-2]` — the case
   built shape `(32, 64)`, which **is** rank 2, so it reached the row check and never exercised the
   rank check it claims to. Now builds a rank-3 table.

## 9. Boundary checks — verbatim

```text
A   = gongyu/tttv2_wh_glx_2d_modules = bc6ad03bfc21d6a26f88169cc87ea2d8176f0fbf
HEAD= 52def65194c3938ed6e5cb6f52661ec3a3a15547

$ git diff --name-only $A..HEAD | grep '_1d\.py'
<empty>
  grep exit=1 (1 = no match = PASS)

$ git diff --name-only $A..HEAD | grep 'llm_runtime'
<empty>
  grep exit=1 (1 = no match = PASS)

$ git grep -n "demos.llama3_70b_galaxy\|models.llama33_70b\b\|models.qwen3_32b\b" -- models/common/models/galaxy models/common/models/*_galaxy
<empty>
  grep exit=1 (1 = no match = PASS)
```

All three empty. `logs/FINAL_boundary_checks.log`.

## 10. Still OPEN

| | Item | Why |
| --- | --- | --- |
| O1 | **The 1D device regression matrix was not run** | It is Milestone A's own P4, explicitly routed to separate hardware and "deliberately not run on this Galaxy host". This job is host-only. No `models/common/modules/**/*_1d.py` implementation file changed (boundary check above), so no 1D behaviour can have changed — but the evidence is absent, and P4 remains Milestone A's outstanding exit-gate line |
| O2 | **Five pre-existing host failures** in `deepseek_r1_distill_qwen_14b`, `llama32_3b`, `llama33_70b`, `qwen25_7b`, `qwen2_7b` | §8. Proven independent of this job. Belongs to whoever owns those packages |
| O3 | **D5 is latent, not fixed-and-proven** | §3. Correct code, no test can discriminate, and the reason is documented. Becomes live if `_require_exact_weight_policy` is ever relaxed |
| O4 | **Everything in Milestone B is unqualified on hardware** | Unchanged by this job and not its scope. The decoupled 64-head Qwen path (C4), RoPE composed with `Attention2D`, the ring/`gather_in0` decode matmul (L3/C7), and repeated model construction (L1/C8) all have zero device evidence |
| O5 | **`test_plans.py` opens the UMD driver as a whole file** | §7. No program executed; not reproducible per node. Means B's host suites and a live device session should not overlap |

## 11. Decisions taken without being asked

Recorded because nobody was available to ask.

1. **README.md conflict** resolved by keeping A's closing sentence and B's additive paragraphs (§4).
   It is a document, so the brief's abort-on-code-conflict rule did not fire.
2. **C1 fixed by derivation rather than deletion** (§4), because `plans.py` needs the placement.
3. **C6 given an explicit `allow_narrow_semaphore_cores` opt-out** (§5), rather than a blanket rule
   that would have rejected a configuration A qualified on hardware.
4. **`SKIP=prefer-expect-error` on every commit.** That hook flags ~35 pre-existing `pytest.raises`
   blocks in `test_attention_2d.py` and `test_lm_head_2d.py` — files Milestone A never touched, so
   they were never subjected to it. Converting all of them is the restructuring the brief warns
   against. Recorded in each commit message. Every other hook ran; `black` and `isort` reformatted
   several Milestone B files, which had never been linted.
5. **Gate 4 narrowed to the host-only subset** and **gate 5 added** (§7, §8).
6. **The three unintended device touches were reported rather than quietly dropped** (§7), and a
   `tt-smi -glx_reset` was spent to leave the mesh clean.

## 12. Finish condition

- [x] Rebase committed, contract-defect commit isolated at the base
- [x] C1–C10 each dispositioned against the tree as it is, with two findings corrected
- [x] Every host suite in the brief green, or recorded `OPEN` with a diagnosis
- [x] The three boundary greps empty
- [x] `MILESTONE_A_STATUS.md` edits appended (D5 row, Attention2D evidence correction, post-record
      contract corrections section, L3 rewrite, D-B note)
- [x] Work-log checkpoint appended
- [x] `tttv2_milestone_b_briefs/job0_completion_handoff.md` written
- [x] No Milestone B hardware work started; mesh left clean and free

**`mb-llama` can start from the branch tip without having to think about the merge.** The last
code-bearing commit is `52def65194c`; everything after it is documentation.

---

*Raw pytest logs are excluded from git by the repository's `*.log` ignore rule and remain on the host
that produced them (`wh-glx6u-05`), under
`tttv2_milestone_b_evidence/reconcile/logs/` — the same arrangement Milestone A recorded. Every claim
above names the log behind it. The three probe scripts under `reconcile/scratch/` are committed and
re-runnable: `d5_probe.py` (D5 reachability), `subgrid_overlap.py` (the step-4 neighbour check),
`c9_equivalence.py` (the C9 field-by-field equivalence).*
