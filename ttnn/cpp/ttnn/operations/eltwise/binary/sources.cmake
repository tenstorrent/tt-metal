# Source files for ttnn_op_eltwise_binary.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_ELTWISE_BINARY_SRCS
    binary.cpp
    common/binary_op_dtype_policy.cpp
    common/binary_op_utils.cpp
    device/binary_composite_op.cpp
)

set(TTNN_OP_ELTWISE_BINARY_API_HEADERS
    binary.hpp
    binary_composite.hpp
    common/binary_op_dtype_policy.hpp
    common/binary_op_types.hpp
    common/binary_op_utils.hpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/eltwise/binary/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
# Listed here rather than inline in CMakeLists.txt so that
# add/remove/rename doesn't touch a file with metalium-developers-infra
# as a required co-owner.
set(TTNN_OP_ELTWISE_BINARY_NANOBIND_SRCS binary_nanobind.cpp)
