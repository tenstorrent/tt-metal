
## The Milestone B exit gate — final table, measured

**This table supersedes §A2's.** Every number in it was produced by a command in
"Gate commands, §A3" below, on this machine, against source identical to `HEAD`
under `models/`. Nothing is quoted from `mb-llama`, `mb-qwen` or attempt 1.

### What "measured at this tree" means for a log stamped with an older commit

The brief's instruction is *re-measure at this tree, do not quote*, and its reason
is Milestone A's lesson that evidence from a tree that has moved is not evidence.
Attempt 3 discharged that instruction by proving the tree has **not** moved under
the code any of these gates exercise:

```sh
git diff --name-only 718997518ab..HEAD -- models/
# models/common/tests/models/qwen3_32b_galaxy/test_step7_coverage_wh_galaxy.py
git diff --name-only 1451b192584..HEAD -- models/
# models/common/tests/models/qwen3_32b_galaxy/test_step7_coverage_wh_galaxy.py
# models/common/tests/models/llama33_70b_galaxy/test_step7_coverage_wh_galaxy.py
```

`models/` has exactly **one changed file** between the commit every gate log was
produced at and `HEAD` — and it is a *step-7 test file*, which
`test_full_model_wh_galaxy.py`, `demo.py` and every module under
`models/common/{models,modules}` neither import nor share a fixture with. So a
`718997518ab` gate log is not an older measurement of a changed thing; it is a
measurement of a byte-identical thing. That is the distinction the instruction
turns on, and every row below states which commit produced its number.

### The nine lines

| # | Gate line | Verdict | Measured value, and the log |
| --- | --- | --- | --- |
| 1 | Llama teacher-forced, batch 1, prefill 512 / decode 511 — top-1 ≥ 91%, top-5 ≥ 99% | **PASS** | top-1 **501/511 = 98.04%**, top-5 **511/511 = 100.00%**. `1 passed in 1029.52s (0:17:09)`. `logs2/a2_g1_llama_tf.log`, commit `1451b192584`; character-identical to `logs2/a2_01_llama_full_model_file.log`. 2 fresh processes |
| 2 | Qwen teacher-forced, batch 1, sequence 512 — top-1 ≥ 89%, top-5 ≥ 97% | **PASS** | top-1 **498/511 = 97.46%**, top-5 **511/511 = 100.00%**. `1 passed in 915.10s (0:15:15)`. `logs2/a2_g12_qwen_tf.log`, commit `718997518ab`. 1 fresh process |
| 3 | Batch-32 direct demos valid, no cross-slot contamination | **PASS** | Llama `logs2/a2_g9_llama_demo_batch32.log`, `1 passed in 277.69s`; Qwen `logs2/a2_g21_qwen_demo_batch32.log`, `1 passed in 153.47s`. 32 slots, each answering its own prompt, slot texts printed per slot; Llama slot 0 character-identical to the batch-1 demo (`a2_g8`). Commit `718997518ab` |
| 4 | Batch-1 4K / 32K / 128K functional smokes pass | **PASS** | Llama `a2_g3` 4K `357.81s`, `a2_g4` 32K `641.17s`, `a2_g5` 128K `721.70s`; Qwen `a2_g14` 4K `117.91s`, `a2_g15` 32K `136.29s`, `a2_g16` 128K `245.76s`. All `1 passed`, commit `718997518ab`, 1 run per geometry per model |
| 5 | Prefix-cached output matches uncached execution | **PASS** | Llama `a2_g2_llama_prefix.log` `1 passed in 424.35s`, Qwen `a2_g13_qwen_prefix.log` `1 passed in 158.58s` — two 128-token chunks against one 256-token prefill, same argmax and PCC ≥ 0.99. Commit `718997518ab` |
| 6 | No dependency imports from an existing model-named implementation package | **PASS** for Milestone B, with one pre-existing exception named below | **0 matches** at `HEAD` for `models\.common\.models\.(llama33_70b\|qwen3_32b)` and **0** for `models\.common\.llm_runtime`, over all eight of Milestone B's own source and test directories. `models\.demos\.` matches **once**, and attempt 3 widened §A2's grep to find it: `models/common/tests/modules/moe/test_tt_moe_decode.py:33` imports three helpers from `models.demos.deepseek_v3`. It **exists unchanged at the job-0 base** `bc6ad03bfc2` (added upstream by `b705bc150e5`, "MoE: (towards) a configurable e2e decode module (#45041)"), is a *test*, and is nowhere on any Galaxy import path. Milestone B did not introduce it and does not depend on it — but the gate as written is not literally 0 over `models/common`, and `mb-signoff` should say so rather than assert a clean zero. Finding **F-C3** |
| 7 | Zero changes to 1D module implementation files | **PASS** | `git diff --name-only bc6ad03bfc2..HEAD \| grep '_1d\.py'` → **0** of **384** changed paths, at `HEAD` |
| 8 | Zero changes to `llm_runtime` | **PASS** | same diff, `grep llm_runtime` → **0** of **384**, at `HEAD` |
| 9 | Existing 1D model contract and demo-contract host tests green, expectations unchanged | **FAIL**, and demonstrably not owned by Milestone B | **5 failed, 296 passed in 108.67s** (`logs3/a3_h1_1d_contract_gate.log`, commit `af589dff4d5`). The same five node ids attempt 1 and attempt 2 recorded, now at three different commits. **No expectation was edited.** Attribution: none of the five packages (`deepseek_r1_distill_qwen_14b`, `llama32_3b`, `llama33_70b`, `qwen25_7b`, `qwen2_7b`) appears anywhere in `bc6ad03bfc2..HEAD`, so Milestone B cannot be their cause |

Plus the host regression gate the brief's "Regression gates" section names, which
is not one of the nine but is the thing the nine sit on:

| Gate | Verdict | Measured |
| --- | --- | --- |
| `models/common/tests/{modules,models,llm_runtime}` host suites | **PASS** | **553 passed, 0 failed in 139.00s** (`logs3/a3_h2_host_gate.log`, commit `af589dff4d5`), with `models/common/tests/models/galaxy/test_plans.py` excluded — finding **F-C2**, it needs a live cluster and this job holds it |
