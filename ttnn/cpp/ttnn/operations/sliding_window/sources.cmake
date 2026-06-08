# Source files for ttnn_op_sliding_window.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_SLIDING_WINDOW_SRCS
    halo/device/halo_device_operation.cpp
    halo/device/untilize_with_halo_program_factory.cpp
    halo/halo.cpp
    sliding_window.cpp
    op_slicing/op_slicing.cpp
)

set(TTNN_OP_SLIDING_WINDOW_API_HEADERS
    sliding_window.hpp
    op_slicing/op_slicing.hpp
)
