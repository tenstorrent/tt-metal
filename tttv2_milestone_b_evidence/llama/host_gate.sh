#!/bin/bash
# Job1 host regression gate. Host-only selection: the 2D module host suites, the
# galaxy support suites and the Llama host suite, with every *_wh_galaxy*.py device
# file excluded (job0 report S7.1/S7.2 -- those globs take the mesh).
LOG="$1"; shift
timeout --signal=TERM --kill-after=60 1800 python -m pytest -q -rA --color=no -p no:cacheprovider \
  --ignore-glob="*_wh_galaxy*.py" \
  models/common/tests/modules/attention/test_attention_1d_arch_config.py \
  models/common/tests/modules/attention/test_attention_2d.py \
  models/common/tests/modules/embedding/test_embedding_2d.py \
  models/common/tests/modules/lm_head/test_lm_head_2d.py \
  models/common/tests/modules/mlp/test_mlp_1d_arch_config.py \
  models/common/tests/modules/mlp/test_mlp_2d.py \
  models/common/tests/modules/prefetcher/test_prefetcher_2d.py \
  models/common/tests/modules/rmsnorm/test_rmsnorm_2d.py \
  models/common/tests/modules/rope/test_rope_2d.py \
  models/common/tests/modules/sampling/test_sampling_1d_release.py \
  models/common/tests/modules/sampling/test_sampling_2d.py \
  models/common/tests/modules/test_tensor_utils.py \
  models/common/tests/models/galaxy \
  models/common/tests/models/llama33_70b_galaxy/test_model_host.py "$@" > "$LOG" 2>&1
echo "exit=$?" >> "$LOG"
