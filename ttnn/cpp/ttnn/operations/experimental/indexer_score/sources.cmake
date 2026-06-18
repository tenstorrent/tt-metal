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
