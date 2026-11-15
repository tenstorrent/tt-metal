# Source files for ttnn_op_experimental_deepseek_step_context.
# Module owners should update this file when adding/removing/renaming source files.

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_STEP_CONTEXT_API_HEADERS step_context.hpp)

set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_STEP_CONTEXT_SRCS
    step_context.cpp
    device/step_context_device_operation.cpp
    device/step_context_program_factory.cpp
)

# Registered on the shared `ttnn` Python module target from
# ttnn/cpp/ttnn/operations/experimental/deepseek/step_context/CMakeLists.txt (see the `if(TARGET ttnn)` block there).
set(TTNN_OP_EXPERIMENTAL_DEEPSEEK_STEP_CONTEXT_NANOBIND_SRCS step_context_nanobind.cpp)
