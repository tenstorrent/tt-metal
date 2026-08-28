
## The Milestone B exit gate, measured at this tree

Commit measured: `1451b192584` for runs 01/01b/02/03/g1 and `718997518ab` for every run from `g2` onward. `mb-coverage` attempt 3 established that `git diff 718997518ab..HEAD -- models/` is **empty**, so every `718997518ab` row below was produced against source identical to `af589dff4d5`; the two commits between `1451b192584` and `718997518ab` touched only the two `test_step7_coverage_wh_galaxy.py` files, which `test_full_model_wh_galaxy.py` does not import. See §A3 for the final table, which supersedes this one.

Every value below was produced by a command in this section — none is quoted from `mb-llama`,
`mb-qwen` or attempt 1. Where a number *does* agree with an earlier job's, that
agreement is stated as a result of re-measurement, which is what the brief asked
for.

| Gate line | Verdict | Measured |
| --- | --- | --- |
| Llama teacher-forced, batch 1, 512/511, top-1 ≥ 91% / top-5 ≥ 99% | **PASS**, 2 runs | top-1 **501/511 = 98.04%** (gate ≥ 91%), top-5 **511/511 = 100.00%** (gate ≥ 99%). `a2_01_llama_full_model_file.log` and `a2_g1_llama_tf.log`, character-identical |
| Qwen teacher-forced, batch 1, 512, top-1 ≥ 89% / top-5 ≥ 97% | **PASS**, 1 run | top-1 **498/511 = 97.46%** (gate ≥ 89%), top-5 **511/511 = 100.00%** (gate ≥ 97%). `a2_g12_qwen_tf.log` |
| Batch-32 direct demos valid, no cross-slot contamination | **PASS**, 1 run per model | Llama `a2_g9`, Qwen `a2_g21`: 32 slots, each answering its own prompt; Llama slot 0 character-identical to the batch-1 demo. The *test* `*_batch32_slots_are_isolated` is a different shape and FAILED for Llama on L1 (`a2_g7`), PASSED for Qwen 3/3 |
| Batch-1 4K / 32K / 128K functional smokes | **PASS**, 1 run per geometry per model | Llama 4K/32K/128K `a2_g3`/`a2_g4`/`a2_g5` (7/11/13 min); Qwen `a2_g14`/`a2_g15`/`a2_g16` (3/3/5 min). Qwen 128K exceeds its own `max_position_embeddings` (40960) and nothing enforces it: a capacity-and-plumbing result, not a quality one |
| Prefix-cached output matches uncached execution | **PASS**, 1 run per model | Llama `a2_g2`, Qwen `a2_g13`: two 128-token chunks against one 256-token prefill, same argmax and PCC ≥ 0.99 |
| No dependency imports from a model-named implementation package | **PASS** | 0 matches, over `models/common/{models/galaxy,modules,models/llama33_70b_galaxy,models/qwen3_32b_galaxy}` |
| Zero changes to 1D module implementation files | **PASS** | `git diff --name-only bc6ad03bfc2..HEAD \| grep '_1d\.py'` → 0 of 338 changed paths |
| Zero changes to `llm_runtime` | **PASS** | same diff, `grep llm_runtime` → 0 |
| Existing 1D model contract and demo-contract host tests green, expectations unchanged | **FAIL**, and not owned by Milestone B | **5 failed, 296 passed** (`a2_h1_1d_contract_gate.log`). The same five ids attempt 1 recorded. None of the five packages appears in `bc6ad03bfc2..HEAD` at all, so Milestone B cannot be their cause. Expectations unchanged — nothing was edited to accommodate this work |
