
### Gate commands, §A3

Every device row above was produced by one line of `queue.txt` through
`cov_queue.sh` → `cov_run3.sh` → `cov_device_run.sh`, which is

```sh
export HF_HOME=/localdev/ctr-apbernal/hf_data
timeout --signal=TERM --kill-after=180 "$MB_DEADLINE" \
  python -u -m pytest -v -rA --color=no -p no:cacheprovider \
    --timeout="$MB_PYTEST_TIMEOUT" "<one node id>" -o faulthandler_timeout=900 > "$LOG" 2>&1
```

never piped, one process at a time, with `cov_ensure_mesh_free.sh` before and
`cov_after_device_run.sh` (which runs `tt-smi -glx_reset` after any non-clean
exit) behind. The node ids, per gate line:

```sh
L=models/common/tests/models/llama33_70b_galaxy
Q=models/common/tests/models/qwen3_32b_galaxy

# 1  Llama teacher-forced, batch 1, prefill 512 / decode 511
$L/test_full_model_wh_galaxy.py::test_llama33_70b_galaxy_teacher_forced_accuracy_batch1
# 2  Qwen teacher-forced, batch 1, 512
$Q/test_full_model_wh_galaxy.py::test_qwen3_32b_galaxy_teacher_forced_accuracy_batch1
# 3  batch-32 direct demos, no cross-slot contamination
models/common/models/llama33_70b_galaxy/demo.py::test_llama33_70b_galaxy_direct_demo_batch32_has_no_cross_slot_contamination
models/common/models/qwen3_32b_galaxy/demo.py::test_qwen3_32b_galaxy_direct_demo_batch32_has_no_cross_slot_contamination
# 4  batch-1 4K / 32K / 128K functional smokes
$L/test_full_model_wh_galaxy.py::test_llama33_70b_galaxy_long_context_smoke[4k|32k|128k]
$Q/test_full_model_wh_galaxy.py::test_qwen3_32b_galaxy_long_context_smoke[4k|32k|128k]
# 5  prefix-cached output matches uncached execution
$L/test_full_model_wh_galaxy.py::test_llama33_70b_galaxy_prefix_cached_prefill_matches_uncached
$Q/test_full_model_wh_galaxy.py::test_qwen3_32b_galaxy_prefix_cached_prefill_matches_uncached
$L/test_step7_coverage_wh_galaxy.py::test_llama_chunked_prefill_matches_a_single_uncached_prefill
$Q/test_step7_coverage_wh_galaxy.py::test_qwen_chunked_prefill_matches_a_single_uncached_prefill
```

Host, device-free, and re-run at `HEAD` by attempt 3's second invocation:

```sh
# 6  no dependency imports from a model-named implementation package.
#    NOTE the directory list: it is wider than §A2's, which is how F-C3 was found.
DIRS="models/common/models/galaxy models/common/modules \
      models/common/models/llama33_70b_galaxy models/common/models/qwen3_32b_galaxy \
      models/common/tests/models/galaxy models/common/tests/models/llama33_70b_galaxy \
      models/common/tests/models/qwen3_32b_galaxy models/common/tests/modules"
grep -rnE '^\s*(from|import)\s+(models\.demos|models\.common\.models\.(llama33_70b|qwen3_32b)([^_]|$)|models\.common\.llm_runtime)' $DIRS
#   -> 1 match, models/common/tests/modules/moe/test_tt_moe_decode.py:33, pre-existing

# 7, 8  boundaries
git diff --name-only bc6ad03bfc2..HEAD | grep '_1d\.py'      # 0 of 384
git diff --name-only bc6ad03bfc2..HEAD | grep 'llm_runtime'  # 0 of 384

# 9  existing 1D model contract and demo-contract host tests
bash tttv2_milestone_b_evidence/coverage/cov_1d_contract_gate.sh logs3/a3_h1_1d_contract_gate.log
#   and, for "expectations unchanged", the check that matters more than the run:
git diff --name-only bc6ad03bfc2..HEAD -- models/common/tests/models/ | grep -v galaxy   # empty

# the host regression gate the brief's "Regression gates" section names
python -m pytest -q models/common/tests/modules models/common/tests/models \
                   models/common/tests/llm_runtime \
                   --ignore=models/common/tests/models/galaxy/test_plans.py   # F-C2

# area 1's cross-process comparison, host only
python -m pytest -v $Q/test_step7_coverage_wh_galaxy.py::test_qwen_two_paged_pools_agree_across_processes
```
