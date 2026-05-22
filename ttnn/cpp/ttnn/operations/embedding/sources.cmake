# Source files for ttnn_op_embedding.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EMBEDDING_SRCS
    device/embedding_device_operation.cpp
    device/embeddings_fused_program_factory.cpp
    device/embeddings_rm_program_factory.cpp
    device/embeddings_tilized_indices_program_factory.cpp
    device/embedding_program_factory_common.cpp
    embedding.cpp
)

set(TTNN_OP_EMBEDDING_API_HEADERS
    embedding.hpp
    device/embedding_device_operation.hpp
    device/embedding_device_operation_types.hpp
    device/embeddings_fused_program_factory.hpp
    device/embeddings_rm_program_factory.hpp
    device/embeddings_tilized_indices_program_factory.hpp
    device/embedding_program_factory_common.hpp
)
