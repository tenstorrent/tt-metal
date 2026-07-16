# Source files for ttnn_op_eltwise_ternary.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_ELTWISE_TERNARY_SRCS
    ternary_composite_op.cpp
    ternary.cpp
    device/ternary_device_operation.cpp
    device/ternary_program_factory.cpp
    device/ternary_op_utils.cpp
)

set(TTNN_OP_ELTWISE_TERNARY_API_HEADERS
    common/ternary_op_types.hpp
    ternary.hpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/eltwise/ternary/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_ELTWISE_TERNARY_NANOBIND_SRCS ternary_nanobind.cpp)
