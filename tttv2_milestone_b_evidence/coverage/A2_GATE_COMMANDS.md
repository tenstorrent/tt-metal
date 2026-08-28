### The command behind each exit-gate line

All of them under `HF_HOME=/localdev/ctr-apbernal/hf_data`, one pytest process at
a time, through `cov_run3.sh`:

```sh
L=models/common/tests/models/llama33_70b_galaxy
Q=models/common/tests/models/qwen3_32b_galaxy

# Llama teacher-forced, batch 1, prefill 512 / decode 511
$L/test_full_model_wh_galaxy.py::test_llama33_70b_galaxy_teacher_forced_accuracy_batch1
# Qwen teacher-forced, batch 1, 512
$Q/test_full_model_wh_galaxy.py::test_qwen3_32b_galaxy_teacher_forced_accuracy_batch1
# batch-32 direct demos, no cross-slot contamination
models/common/models/llama33_70b_galaxy/demo.py::test_llama33_70b_galaxy_direct_demo_batch32_has_no_cross_slot_contamination
models/common/models/qwen3_32b_galaxy/demo.py::test_qwen3_32b_galaxy_direct_demo_batch32_has_no_cross_slot_contamination
# batch-1 4K / 32K / 128K functional smokes
$L/test_full_model_wh_galaxy.py::test_llama33_70b_galaxy_long_context_smoke   # 4k, 32k, 128k
$Q/test_full_model_wh_galaxy.py::test_qwen3_32b_galaxy_long_context_smoke     # 4k, 32k, 128k
# prefix-cached output matches uncached execution
$L/test_full_model_wh_galaxy.py::test_llama33_70b_galaxy_prefix_cached_prefill_matches_uncached
$Q/test_full_model_wh_galaxy.py::test_qwen3_32b_galaxy_prefix_cached_prefill_matches_uncached
$L/test_step7_coverage_wh_galaxy.py -k chunked_prefill_matches      # and the decode after it
$Q/test_step7_coverage_wh_galaxy.py -k chunked_prefill_matches
```

Host, device-free:

```sh
# no dependency imports from a model-named implementation package
grep -rnE '^\s*(from|import)\s+models\.(demos\.llama3_70b_galaxy|common\.models\.(llama33_70b|qwen3_32b)([^_]|$))' \
    models/common/models/galaxy models/common/modules models/common/models/*_galaxy
# zero changes to 1D module implementation files, and to llm_runtime
git diff --name-only bc6ad03bfc2..HEAD | grep '_1d\.py'
git diff --name-only bc6ad03bfc2..HEAD | grep 'llm_runtime'
# existing 1D model contract and demo-contract host tests
bash tttv2_milestone_b_evidence/coverage/cov_1d_contract_gate.sh <log>
```
