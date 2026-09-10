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

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/sliding_window/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_SLIDING_WINDOW_NANOBIND_SRCS sliding_window_nanobind.cpp)
