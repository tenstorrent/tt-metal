# Source files for ttnn_op_experimental_indexer_score.
# Module owners should update this file when adding/removing/renaming source files.

# Install the full public-header chain (indexer_score.hpp pulls in the device headers transitively),
# mirroring topk_large_indices, so external ttnn-dev consumers can include the op.
set(TTNN_OP_EXPERIMENTAL_INDEXER_SCORE_API_HEADERS
    indexer_score.hpp
    device/indexer_score_device_operation.hpp
    device/indexer_score_device_operation_types.hpp
    device/indexer_score_program_factory.hpp
)

set(TTNN_OP_EXPERIMENTAL_INDEXER_SCORE_SRCS
    device/indexer_score_device_operation.cpp
    device/indexer_score_program_factory.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/indexer_score/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_EXPERIMENTAL_INDEXER_SCORE_NANOBIND_SRCS indexer_score_nanobind.cpp)
