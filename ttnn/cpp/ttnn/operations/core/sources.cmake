# Source files for ttnn_op_core.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_CORE_SRCS
    compute_kernel/compute_kernel_config.cpp
    core.cpp
    to_layout/to_layout_op.cpp
    to_dtype/to_dtype_op.cpp
    to_memory_config/to_memory_config_op.cpp
)

set(TTNN_OP_CORE_API_HEADERS
    core.hpp
    to_dtype/to_dtype_op.hpp
    to_layout/to_layout_op.hpp
    compute_kernel/compute_kernel_config.hpp
    to_memory_config/to_memory_config_op.hpp
)
