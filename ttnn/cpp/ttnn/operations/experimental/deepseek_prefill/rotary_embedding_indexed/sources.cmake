# Source files for ttnn_op_experimental_deepseek_prefill_rotary_embedding_indexed.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_ROTARY_EMBEDDING_INDEXED_API_HEADERS rotary_embedding_indexed.hpp)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/rotary_embedding_indexed/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_PREFILL_ROTARY_EMBEDDING_INDEXED_NANOBIND_SRCS rotary_embedding_indexed_nanobind.cpp)
