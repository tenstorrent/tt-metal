# Source files for ttnn_op_experimental_deepseek_prefill_zero_padded_kv_cache.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_ZERO_PADDED_KV_CACHE_API_HEADERS zero_padded_kv_cache.hpp)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/zero_padded_kv_cache/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_ZERO_PADDED_KV_CACHE_NANOBIND_SRCS zero_padded_kv_cache_nanobind.cpp)
