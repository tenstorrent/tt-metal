
## The Milestone B exit gate, measured at this tree

Commit measured: `b1e824537a4` plus attempt 2's own test-only commit. Every value
below was produced by a command in this section — none is quoted from `mb-llama`,
`mb-qwen` or attempt 1. Where a number *does* agree with an earlier job's, that
agreement is stated as a result of re-measurement, which is what the brief asked
for.

| Gate line | Verdict | Measured |
| --- | --- | --- |
| Llama teacher-forced, batch 1, 512/511, top-1 ≥ 91% / top-5 ≥ 99% | @@V_LTF@@ | @@M_LTF@@ |
| Qwen teacher-forced, batch 1, 512, top-1 ≥ 89% / top-5 ≥ 97% | @@V_QTF@@ | @@M_QTF@@ |
| Batch-32 direct demos valid, no cross-slot contamination | @@V_B32@@ | @@M_B32@@ |
| Batch-1 4K / 32K / 128K functional smokes | @@V_LC@@ | @@M_LC@@ |
| Prefix-cached output matches uncached execution | @@V_PC@@ | @@M_PC@@ |
| No dependency imports from a model-named implementation package | **PASS** | 0 matches, over `models/common/{models/galaxy,modules,models/llama33_70b_galaxy,models/qwen3_32b_galaxy}` |
| Zero changes to 1D module implementation files | **PASS** | `git diff --name-only bc6ad03bfc2..HEAD \| grep '_1d\.py'` → 0 of 338 changed paths |
| Zero changes to `llm_runtime` | **PASS** | same diff, `grep llm_runtime` → 0 |
| Existing 1D model contract and demo-contract host tests green, expectations unchanged | **FAIL**, and not owned by Milestone B | **5 failed, 296 passed** (`a2_h1_1d_contract_gate.log`). The same five ids attempt 1 recorded. None of the five packages appears in `bc6ad03bfc2..HEAD` at all, so Milestone B cannot be their cause. Expectations unchanged — nothing was edited to accommodate this work |
