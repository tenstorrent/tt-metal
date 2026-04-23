# Source files for ttnn_op_eltwise_binary.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_ELTWISE_BINARY_SRCS
    binary.cpp
    common/binary_op_utils.cpp
    device/binary_composite_op.cpp
    device/binary_device_operation.cpp
    device/broadcast_height_and_width_multi_core_program_factory.cpp
    device/broadcast_height_multi_core_program_factory.cpp
    device/broadcast_width_multi_core_program_factory.cpp
    device/element_wise_multi_core_program_factory.cpp
    device/element_wise_multi_core_sfpu_pgm_factory.cpp
    device/broadcast_height_multi_core_sharded_optimized_program_factory.cpp
    device/broadcast_height_multi_core_sharded_program_factory.cpp
)

set(TTNN_OP_ELTWISE_BINARY_API_HEADERS
    binary.hpp
    binary_composite.hpp
    common/binary_op_types.hpp
    common/binary_op_utils.hpp
    device/binary_device_operation.hpp
)
