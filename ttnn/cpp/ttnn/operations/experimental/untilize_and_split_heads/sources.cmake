# Source files for ttnn_op_experimental_untilize_and_split_heads.

set(TTNN_OP_EXPERIMENTAL_UNTILIZE_AND_SPLIT_HEADS_API_HEADERS untilize_and_split_heads.hpp)

set(TTNN_OP_EXPERIMENTAL_UNTILIZE_AND_SPLIT_HEADS_SRCS
    untilize_and_split_heads.cpp
    device/untilize_and_split_heads_device_operation.cpp
    device/untilize_and_split_heads_program_factory.cpp
)

# Registered on the shared `ttnn` Python module target from CMakeLists.txt (see
# the `if(TARGET ttnn)` block there).
set(TTNN_OP_EXPERIMENTAL_UNTILIZE_AND_SPLIT_HEADS_NANOBIND_SRCS untilize_and_split_heads_nanobind.cpp)
