# Pytest Manifest

Kernel → exercising pytest(s), maintained by the migration pipeline. See
`agents/llk_helpers_hq.md` → "Pytest Manifest" for format and ownership.

Format: `<kernel path> :: <pytest path>[; <pytest path> ...]`

Rows are appended as kernels are first touched by a pipeline run; stale rows
are fixed by Step 4 of the same run that hits the staleness.

## Infrastructure regression set

Pytests that must pass after any change to the shared helpers
(`binary_op_helpers`, `sfpu_chain`, `reduce_helpers_compute`). Order is
irrelevant — run back-to-back.

```
tests/ttnn/unit_tests/kernel_lib/*.py
tests/ttnn/nightly/unit_tests/operations/fused/test_layernorm.py
tests/ttnn/nightly/unit_tests/operations/fused/test_distributed_layernorm_pre_allgather.py
tests/ttnn/nightly/unit_tests/operations/fused/test_distributed_layernorm_post_allgather.py
tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_layer_norm.py
tests/ttnn/nightly/unit_tests/operations/experimental/test_rotary_embedding_llama.py
```

## Kernel → pytest

ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/compute/layernorm_pre_allgather.cpp :: tests/ttnn/nightly/unit_tests/operations/fused/test_distributed_layernorm_pre_allgather.py
ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/compute/layernorm_pre_allgather_2d.cpp :: tests/ttnn/nightly/unit_tests/operations/fused/test_distributed_layernorm_pre_allgather.py
ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/compute/layernorm_post_allgather.cpp :: tests/ttnn/nightly/unit_tests/operations/fused/test_distributed_layernorm_post_allgather.py
ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device/kernels/compute/layernorm_post_allgather_welford.cpp :: tests/ttnn/nightly/unit_tests/operations/fused/test_distributed_layernorm_post_allgather.py
ttnn/cpp/ttnn/operations/normalization/rmsnorm_distributed/device/kernels/compute/rmsnorm_pre_allgather.cpp :: tests/ttnn/nightly/unit_tests/operations/fused/test_distributed_layernorm_pre_allgather.py
ttnn/cpp/ttnn/operations/normalization/rmsnorm_distributed/device/kernels/compute/rmsnorm_pre_allgather_2d.cpp :: tests/ttnn/nightly/unit_tests/operations/fused/test_distributed_layernorm_pre_allgather.py
ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm/device/kernels/moreh_layer_norm_small_kernel.cpp :: tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_layer_norm.py
ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm/device/kernels/moreh_layer_norm_large_kernel.cpp :: tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_layer_norm.py
ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/device/kernels/compute/rotary_embedding_llama.cpp :: tests/ttnn/nightly/unit_tests/operations/experimental/test_rotary_embedding_llama.py
ttnn/cpp/ttnn/operations/experimental/transformer/dit_layernorm_post_all_gather/device/kernels/compute/layernorm_post_allgather_welford.cpp :: tests/ttnn/nightly/unit_tests/operations/transformers/test_distributed_dit_layernorm.py
