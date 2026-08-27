# Job 3 (`mb-coverage`) → `mb-signoff`: completion handoff

Written 2026-08-27 by `mb-coverage`, unattended.
Full account: `tttv2_milestone_b_evidence/coverage/REPORT.md`.
Environment and mesh facts: `tttv2_milestone_b_evidence/coverage/ENVIRONMENT.md`.

## Read this paragraph first

**The mesh never came back.** Same eleven boards off the PCIe bus as `mb-qwen`
found (`0 1 2 3 4 5 6 7 10 11 14`), same `Read 0xffffffff over PCIe ID 17`,
`ttnn` cannot open a cluster at all. **No recovery attempt was spent** — two
jobs already burned all four permitted attempts on this exact fault and proved
that neither reset path can bring back a board that is not on the bus. This
needs an IPMI power cycle or a host reboot.

You need no device, so this does not block you. But it decides your verdict:

> **Three consecutive device jobs have produced zero numerical results from
> silicon, for either model.** No PCC, no accuracy number, no demo output, no
> functional smoke. Milestone B's exit gate cannot be signed off as passing, and
> the reason is infrastructure, not the code.

**Qwen has a second, independent blocker**: its weights are not on this machine
(`config.json` only, ~65 GB to fetch into `/proj_sw/user_dev/hf_data`). Even a
healthy mesh does not unblock the Qwen accuracy gate.

**Export `HF_HOME=/proj_sw/user_dev/hf_data`.** It is unset in the inherited
environment and the failure mode is a silent *skip*, not a failure.

## The exit gate, as measured at this tree

Commit measured: `0c1ccd8557c7cb25cd1ca300d522eab1ed5db733` (this job's own
commit adds tests only and changes no implementation file).

| Gate line | Verdict |
| --- | --- |
| Llama teacher-forced, batch 1, 512/511, top-1 ≥ 91% / top-5 ≥ 99% | **NOT REACHED** — no mesh. Never measured by anyone, at any tree. |
| Qwen teacher-forced, batch 1, 512, top-1 ≥ 89% / top-5 ≥ 97% | **BLOCKED (upstream)** — weights absent — **and** NOT REACHED. |
| Batch-32 direct demos valid, no cross-slot contamination | **PARTIAL** — the block-ownership mechanism is proved on host at active batch 1/8/16/31/32; no device demo has ever produced output. |
| Batch-1 4K / 32K / 128K functional smokes | **NOT REACHED** — capacity accounting produced instead (table in REPORT.md area 5). |
| Prefix-cached output matches uncached execution | **NOT REACHED** — the addressing is proved on host; there is no PCC. |
| No dependency imports from a model-named implementation package | **PASS** — 0 matches. |
| Zero changes to 1D module implementation files | **PASS** — 0 files over all 190 changed paths since `bc6ad03bfc2`. |
| Zero changes to `llm_runtime` | **PASS** — 0 files. |
| Existing 1D model contract and demo-contract host tests green | **FAIL** — 5 failures, re-measured here, identical to `reconcile`'s finding O2 and **proved not caused by Milestone B**. |

### On the "re-measure, do not quote" instruction

Your brief and mine both assume `mb-llama` and `mb-qwen` produced accuracy
numbers to re-measure. **They did not.** Both recorded `BLOCKED (infra)` with no
number. There is nothing to compare against; the honest line is "never
measured".

### On the 1D demo-contract FAIL

Do not write this up as a Milestone B regression. The five failures
(`deepseek_r1_distill_qwen_14b`, `qwen2_7b`, `qwen25_7b` demo contracts;
`llama33_70b` demo contract; `llama32_3b` hf_adaptor) live in packages Milestone
B never touched, and three of them fail inside `llm_runtime`'s
`_plan_prefill_requests`, which is byte-identical to Milestone A's. Proof is
mechanical and in REPORT.md:

```sh
git diff --name-only bc6ad03bfc2..HEAD | grep -v '^models/common/\(models\|modules\|tests\)/' | grep -v '^tttv2'   # empty
```

The gate line is FAIL as written. The owner is not Milestone B.

## Findings you need for the scorecard

Seven, none of them fixed here. This job changed **no implementation file** —
deliberately: every finding either needs a mesh to validate a fix, or needs a
product decision first.

| ID | Severity | Where | What |
| --- | --- | --- | --- |
| **D-C1** | correctness | `attention_2d.py::_validate_decode_page_table` | A prefill-shaped page table fed to decode is **accepted**. The step-7 gate asks for it to be rejected; the current contract cannot do that. |
| **D-C2** | contract conflict | `sampling_2d.py::_seed_digest` | Moving a seeded request to another slot **changes its stream**. Contradicts the step-7 slot-stability gate. Needs a product decision. |
| **G-C1** | limitation | `direct_runner.prefill_batched` | Concat-32 requires all 32 slots active; it cannot be combined with the `active_slots < 32` sink-block mechanism. |
| **G-C2** | minor | `direct_runner.prefill_batched` | An empty row is rejected one call too late, after the whole concatenated graph has run. |
| **G-C3** | dead code | `attention_2d._validate_prefill` | `"chunk_page_table requires a prefix/chunked recipe"` is unreachable. |
| **F-C1** | premise correction | `recipes.galaxy_padded_vocab_size` | **Llama has no vocabulary padding.** 128256 is already a multiple of `8 * 32`. Its padded-vocab gate is vacuous; only Qwen pads (128 ids). |
| **F-C2** | test-infra | `tests/models/galaxy/test_plans.py` | Looks host-only, needs a cluster (`ttnn.SubDevice` constructs the `MetalContext`). 13 of the 18 baseline host failures are this, not defects. |

### D-C1 in the detail you will need

`_validate_decode_page_table` discriminates on row count alone and accepts any
positive multiple of `users_per_column`, because an L1-sharded table
legitimately repeats the device-local batch once per core. The replicated
prefill table's device-local view is 32 rows, and `32 == 4 * 8`, so it passes.
The width check passes too (the prefill table is stick-aligned and therefore
*wider*). The dtype matches. It reaches `paged_update_cache`.

Shape cannot separate the two cases. The distinguishing fact is placement:
interleaved-and-replicated versus L1 height-sharded over
`rows / users_per_column` cores. The validator never reads `memory_config()`.

**Why it was not fixed here.** An existing 2D module test —
`test_attention_2d.py::test_decode_page_table_accepts_the_device_local_batch_and_its_core_repeats[32]`
— asserts that the 32-row table *is* accepted. Fixing decode means changing that
expectation, which both our briefs call a boundary violation to report rather
than commit. The proposed fix is written out in REPORT.md area 1.

### D-C2 in the detail you will need

`_device_seed`/`_host_seed` are `blake2b("sampling2d:{seed}:{slot}")`. That is
deliberate: it stops 32 slots given one seed by a serving front end from all
emitting the same token, and that protection is itself proved by a test. But it
means a request that migrates between slots does not keep its stream, which is
the opposite of what the step-7 gate asks. **This is a decision about the
serving contract — is a seed per-request or per-(request, slot)? — not a bug.**
Put it in front of whoever owns that contract rather than filing it as a defect.

## What you inherit in the tree

One commit, tests and evidence only:

```text
Add Milestone B step-7 coverage: 162 host tests, and the device tests we cannot run
```

```text
models/common/tests/models/galaxy/step7_harness.py                     helper, not collected
models/common/tests/models/galaxy/test_step7_paged_kv.py               39
models/common/tests/models/galaxy/test_step7_concat32.py               34
models/common/tests/models/galaxy/test_step7_prefix_cache.py           19
models/common/tests/models/galaxy/test_step7_sampling.py               26
models/common/tests/models/galaxy/test_step7_long_context.py           32
models/common/tests/models/galaxy/test_step7_repeat_and_cleanup.py     12
                                                                     ---- 162, passing in three fresh processes

models/common/tests/models/llama33_70b_galaxy/test_step7_coverage_wh_galaxy.py   17  NEVER EXECUTED
models/common/tests/models/qwen3_32b_galaxy/test_step7_coverage_wh_galaxy.py     16  NEVER EXECUTED
```

**No implementation file changed.** Both boundary greps stay empty.

`mb-qwen` argued against committing device tests that have never run, and the
argument is sound. This job took the other side because the step-7 gaps are now
*specific* and leaving them as prose means re-deriving them under time pressure
next time. The mitigation is loudness, not abstention: both files say "This file
has never been executed" in their module docstring, with the date and the
reason. Both were verified to collect (17 and 16 node ids) and nothing more.

## The one host assumption a mesh must check

`step7_harness.py` models a non-obvious `ttnn` fact: a distributed tensor's
`.shape` is the **shard** shape, not the global one. That was read out of
`ttnn/core/distributed/distributed_tensor.cpp` — `TensorToMesh::Impl::create_tensor`
builds the output `Tensor` from `compute_tensor_spec_for_shards` — and **not
measured on silicon**. D-C1 rests on it.

One line settles it:

```python
t = ttnn.from_torch(torch.zeros(32, 64, dtype=torch.int32), device=mesh,
                    mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(None, 0), mesh_shape=(8, 4)),
                    dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
assert tuple(t.shape) == (8, 64)
```

If it is `(32, 64)` instead, D-C1 is worse than described: the "device-local
rows" branch would be unreachable for a correctly-mapped table, so decode's page
table would have no effective validation at all.

## What Milestone C inherits

* **L1 — `Prefetcher2D` global-CB ownership redesign.** Confirmed on the host
  here: `cleanup()` clears `self._global_cb` without ever handing it to
  `deallocate`, so the owner truthfully reports `owned_resources == ()` while the
  CB is still resident. Two owners in one process allocate two CBs and free
  neither. The **OOM** needs real L1 and was not reproduced. Whether the
  teardown-ordering contract is workable at model scale is still unknown — the
  80-layer model has never been built, and `test_two_models_in_one_process` has
  never run. `reconcile`'s O5 stands unchanged.
* **The CCL merge evaluation** — untouched, as expected.
* **D-C1 and D-C2 above**, both of which are contract decisions rather than
  local fixes.
* **L3** (attention decode matmuls on `dense_matmul_program_config`) and
  **D-B9** (attention decode CB/L1 clash, ~20 kB) — inherited from `mb-llama`
  and `mb-qwen`, untouched here, still device-unverified.

## Suggested order for your night

1. Read REPORT.md's exit-gate table. Every row has the command that produced it;
   none is quoted from an earlier job.
2. Record the verdict as **infrastructure-blocked**, not as a code failure. The
   distinction matters for whoever schedules the mesh repair.
3. Use the seven findings for the modularity scorecard. D-C1 and D-C2 are the
   two that should reach a human.
4. When you write `MILESTONE_B_STATUS.md`, say plainly that the two accuracy
   gates have never been measured. Three jobs have now had the chance.

## Do not

* Do not trust `ls /dev/tenstorrent | wc -l` — it returned 32 all night while
  only 21 boards existed. Use `ls /sys/class/tenstorrent | wc -l`.
* Do not read any device test in this tree as evidence. None of them has run.
* Do not treat the five 1D demo-contract failures as a Milestone B regression.
* Do not treat the 13 `test_plans.py` failures as defects — they are F-C2.
