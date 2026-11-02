# SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""The op's unit suite (test_qkv_proj_heads_rope.py) re-run against the bias_seed fork. BS_MODE / BS_KB as in
test_bias_seed.py. The two rejection tests (need the tests-tree expect_error fixture) are not re-run."""

import os

from test_bias_seed import fork, pd, ref

pd.BIAS_MODE = os.environ.get("BS_MODE", "ktile")
pd.K_BLOCK_SEEDED = int(os.environ.get("BS_KB", "4"))
ref.qkv_proj_heads_rope = fork.qkv_proj_heads_rope

test_qkv_proj_heads_rope = ref.test_qkv_proj_heads_rope
test_default_config_and_inputs_unchanged = ref.test_default_config_and_inputs_unchanged
test_hifi4_explicit_memory_config = ref.test_hifi4_explicit_memory_config
test_x_in_dram = ref.test_x_in_dram
test_non_aligned_padding_zero = ref.test_non_aligned_padding_zero
test_dram_outputs_and_tables = ref.test_dram_outputs_and_tables
